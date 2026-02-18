# guitaRL — Reinforcement Learning for Guitar Harmonics

Real-robot RL system that trains a continuous-action policy to produce natural
harmonics on [GuitarBot](../GuitarBot), a mechatronic guitar-playing robot.  A pretrained CNN
harmonic classifier ([HarmonicsClassifier](../HarmonicsClassifier)) provides the audio reward signal.
The algorithm is **SAC** (Soft Actor-Critic) implemented via Stable-Baselines3.

> **Terminology note**: The "torque" dimension in the action space controls the
> *presser* encoder target — it is functionally a position setpoint for the
> fretting mechanism, not directly a torque command.  The name is preserved
> throughout the codebase for consistency with the GuitarBot firmware.

---

## Setup

1. **Install dependencies**:
```bash
conda env create -f environment.yml
conda activate guitaRL
```

2. **Audio routing**:
   - Connect guitar audio output to an interface (default: Scarlett).
   - Pass `--audio-device <substring>` if your device name differs.

3. **GuitarBot middleware**:
   - Start `arm_list_recieverNN.py` (port 12000) on the GuitarBot machine before training.
   - All robot calls are serialised through a `threading.Lock()` — `/Reset` safely
     queues behind any active trajectory.

4. **Classifier model**:
   - Ensure a trained model exists at `../HarmonicsClassifier/models/best_model.pt`
     (or pass an explicit `--model-path`).

---

## Project Structure

```
guitaRL/
├── env/
│   ├── __init__.py
│   ├── action_space.py     # RLFretAction dataclass, fret↔mm conversion, physical limits
│   ├── osc_client.py       # OSC communication with GuitarBot (/RLFret, /Reset)
│   └── harmonic_env.py     # Gymnasium environment (HarmonicEnv)
├── utils/
│   ├── __init__.py
│   ├── reward.py           # All reward logic — single source of truth
│   └── audio_reward.py     # Audio capture + classifier dispatch → reward.py
├── scripts/
│   ├── ablation_no_filtration.sh   # Ablation: bypass physics gate
│   └── ablation_no_audio.sh        # Ablation: no CNN, fret+torque shaping only
├── train.py                # SAC training script (resume supported)
├── test_rl_loop.py         # Diagnostic: manual action → OSC → audio → classify → reward
├── evaluate.py             # Evaluation script
├── environment.yml         # Conda dependencies (env name: guitaRL)
└── README.md
```

---

## Environment Details

### Observation Space (14-dim)

| Slot | Dim | Encoding | Notes |
|------|-----|----------|-------|
| `target_fret_one_hot` | 3 | One-hot over {4, 5, 7} | Which harmonic node to play |
| `string_one_hot` | 3 | One-hot over {0, 2, 4} | Which string is active this episode |
| `current_fret` | 1 | Fractional fret (0.0 – 9.0) | Last commanded fret position |
| `current_torque_norm` | 1 | torque / 650 | Last commanded presser value (normalised) |
| `fret_history` | 3 | Fractional frets, oldest first | Last 3 fret commands (zero-padded) |
| `torque_history` | 3 | Normalised, oldest first | Last 3 presser commands (zero-padded) |

The string one-hot allows a **single shared policy** to learn string-specific
fret/torque mappings, enabling multi-string training without separate models.

### Action Space (5-dim, continuous, normalised to [−1, 1])

| Dim | Maps to | Physical range |
|-----|---------|----------------|
| 0–2 | String selection (argmax → string index) | Strings 0, 2, 4 (pluckers only) |
| 3 | Fret position | 0.0 – 9.0 fractional frets |
| 4 | Presser torque | 16 – 650 encoder units |

`always_press=True` (default) removes the explicit press/unpress dimension,
avoiding the microcontroller's position-mode override below 15 encoder ticks.
The chosen string is **always overridden** by the episode's active string — the
string logits exist so the policy learns string-conditioned behaviour through
gradient flow, but cannot accidentally command the wrong actuator.

### Target Harmonics

| Fret | Position (mm) | Interval above open |
|------|---------------|---------------------|
| 4    | 112.0         | Major 17th (5th harmonic) |
| 5    | 139.0         | Major 14th (4th harmonic) |
| 7    | 187.0         | Perfect 12th (3rd harmonic) |
| 9    | 211.0         | Major 10th |

---

## Reward Function — Two-Layer Architecture

Defined exclusively in `utils/reward.py`.  Both the training env
(`audio_reward.py` → `harmonic_env.py`) and the diagnostic test loop
(`test_rl_loop.py`) import from there — changing a threshold or weight
propagates everywhere automatically.

### Layer 1: Filtration (physics gate)

A deterministic physics check runs before the CNN classifier.  If the action
is mechanically implausible, the agent receives a flat penalty and the
classifier is never invoked — saving inference time and giving a hard,
unambiguous signal to avoid degenerate regions of the action space.

| Check | Threshold | Rationale |
|-------|-----------|-----------|
| Torque too high | > 350 | Harmonics require a light touch; heavy fretting damps the partial |
| Torque too low | < 15 | Presser below the microcontroller's safety dead-zone |
| Fret too far | > 3 frets from nearest harmonic node | Not near any harmonic position |

**Filtration penalty**: −1.0 (constant, independent of how far outside limits the action is).

> **Note**: Silence detection via audio RMS was trialled but removed — onset
> timing relative to capture windows was insufficiently reliable.

### Layer 2: Audio (classifier reward)

Reached only when Layer 1 passes.

$$r = 0.6 \cdot r_\text{audio} + 0.2 \cdot r_\text{fret} + 0.2 \cdot r_\text{torque}$$

| Component | Formula | Range | Purpose |
|-----------|---------|-------|---------|
| $r_\text{audio}$ | `harmonic_prob` from CNN | [0, 1] | Does it *sound* like a harmonic? |
| $r_\text{fret}$ | $\exp\!\left(-\dfrac{d_\text{nearest}^2}{2 \cdot 0.35^2}\right)$ | [0, 1] | Gaussian centred on the **nearest harmonic node** (σ = 0.35) |
| $r_\text{torque}$ | $2\exp\!\left(-\dfrac{e_\tau^2}{2 \cdot 75^2}\right) - 1$ | [−1, 1] | Shifted Gaussian at optimal presser value 30 (σ = 75) |

where $d_\text{nearest} = \min_{h \in \{4,5,7,9\}} |f - h|$.
Being on fret 4 scores full fret reward regardless of the episode target —
all harmonic nodes are physically valid positions.  The target fret guides the
agent through the observation one-hot and curriculum, not through per-step penalisation.

- Optimal presser target: **30** encoder units (light touch for harmonics)
- **Success bonus**: +1.0 added when `harmonic_prob > 0.8`; episode terminates early

---

## Multi-String Training & Motor Wear Distribution

Training on a single string concentrates all motor cycles on one actuator.
`--string-indices` enables **episode-level string rotation**: at each `reset()`,
the active string is sampled uniformly from the provided pool.

When the active string changes between episodes, `HarmonicEnv` automatically:
1. Sends `/Reset` to home all arms
2. Waits **10 seconds** for the mechanism to settle before the next episode begins

The string one-hot in the observation ensures the policy receives explicit
context about which string is active, allowing it to learn string-specific
fret/torque adjustments within a single set of network weights.

---

## Ablation Studies

Three reward modes are available via `--reward-mode`:

| Mode | Layer 1 | Layer 2 (CNN) | Audio captured | Purpose |
|------|---------|---------------|----------------|---------|
| `full` (default) | ✅ | ✅ | ✅ | Full two-layer reward |
| `no_filtration` | ❌ bypassed | ✅ | ✅ | Isolate contribution of physics gate |
| `no_audio` | ✅ | ❌ | ❌ | Isolate contribution of CNN; faster steps |

In `no_audio` mode, fret and torque weights are rebalanced to 0.5 / 0.5.
Ready-to-run scripts are in `scripts/`:

```bash
./scripts/ablation_no_filtration.sh
./scripts/ablation_no_audio.sh

# Override defaults
TOTAL_TIMESTEPS=100000 ./scripts/ablation_no_audio.sh --curriculum random
```

---

## Usage

### Diagnostic Test Loop

Runs the full action → OSC → audio → classify → reward pipeline manually,
one note at a time, with per-note plots:

```bash
conda run -n guitaRL python test_rl_loop.py \
    --model ../HarmonicsClassifier/models/best_model.pt \
    --num-tests 5 \
    --target-harmonic
```

Pass `--no-plot` to suppress figures.

### Training

```bash
# Single string
python train.py \
    --model-path ../HarmonicsClassifier/models/best_model.pt \
    --string-index 2

# Multi-string rotation (distributes motor wear across strings 0, 2, 4)
python train.py \
    --model-path ../HarmonicsClassifier/models/best_model.pt \
    --string-indices 0 2 4 \
    --curriculum easy_to_hard \
    --total-timesteps 50000
```

**Curriculum modes**:
- `easy_to_hard` — fret 7 for first 100 episodes, then fret 5, then fret 4
- `random` — uniform random fret each episode
- `fixed_fret` — always fret 7

### Resuming an Interrupted Run

On `KeyboardInterrupt` **or** an unhandled exception (e.g. a robot error),
`train.py` saves `interrupted_model.zip` and `interrupted_model_replay_buffer.pkl`
directly into the run directory root — **not** in `checkpoints/`.
Periodic checkpoints save to `checkpoints/` only when `--checkpoint-freq` steps
are reached (default 5000); short runs may have nothing there.

To resume, point `--resume` at the run directory:

```bash
# Automatic: finds interrupted_model.zip, or the latest checkpoint
python train.py \
    --model-path ../HarmonicsClassifier/models/best_model.pt \
    --resume ./runs/harmonic_sac_20260218_150711
```

`--resume-checkpoint` is only needed when you want a specific checkpoint file
(without `.zip` extension), overriding the automatic selection:

```bash
python train.py \
    --model-path ../HarmonicsClassifier/models/best_model.pt \
    --resume ./runs/harmonic_sac_20260218_150711 \
    --resume-checkpoint ./runs/harmonic_sac_20260218_150711/checkpoints/harmonic_sac_5000_steps
```

`reset_num_timesteps=False` is passed to SB3 so the internal timestep counter
continues from where it left off and `--total-timesteps` is treated as an
absolute ceiling rather than additional steps.

### Evaluation

```bash
python evaluate.py \
    --model runs/harmonic_sac_TIMESTAMP/best_model/best_model.zip \
    --episodes 20 \
    --visualize \
    --deterministic

# Test a specific fret
python evaluate.py \
    --model runs/harmonic_sac_TIMESTAMP/best_model/best_model.zip \
    --target-fret 7 \
    --episodes 10
```

---

## Robot Safety

- **`/Reset` on exit is on by default.** Pass `--reset-on-exit` to send it.
- On the GuitarBot side, `arm_list_recieverNN.py` serialises all
  `RobotController.main()` calls through a `threading.Lock`, so a `/Reset`
  arriving during an active trajectory will block until the trajectory
  completes rather than corrupting the UDP stream.
- String-switch resets (automatic, between episodes) send `/Reset` and wait
  10 s before the next episode.  Single-string runs never send `/Reset`
  during training.

---

## Monitoring

```bash
tensorboard --logdir runs/harmonic_sac_TIMESTAMP/logs
```

Per-step console output shows string index, target fret, commanded fret,
presser value, classifier probabilities (H / D / G), reward decomposition,
and filtration status when Layer 1 rejects an action.