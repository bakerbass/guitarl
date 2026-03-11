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

## Table of Contents

- [Setup](#setup)
- [Project Structure](#project-structure)
- [Environment Details](#environment-details)
  - [Observation Space](#observation-space-14-dim)
  - [Action Space](#action-space-5-dim-continuous-normalised-to-1-1)
  - [Target Harmonics](#target-harmonics)
- [Reward Function — Two-Layer Architecture](#reward-function--two-layer-architecture)
  - [Layer 1: Filtration](#layer-1-filtration-physics-gate)
  - [Layer 2: Audio](#layer-2-audio-classifier-reward)
- [Multi-String Training & Motor Wear Distribution](#multi-string-training--motor-wear-distribution)
- [Ablation Studies](#ablation-studies)
- [Usage](#usage)
  - [Diagnostic Test Loop](#diagnostic-test-loop)
  - [Training](#training)
  - [Recording Successful Harmonics](#recording-successful-harmonics)
  - [Resuming an Interrupted Run](#resuming-an-interrupted-run)
  - [Evaluation](#evaluation)
  - [Reviewing & Relabeling Clips](#reviewing--relabeling-clips)
  - [Exporting an Augmented Dataset](#exporting-an-augmented-dataset)
- [Robot Safety](#robot-safety)
- [Monitoring](#monitoring)
- [Offline Pre-Training vs. Robot Training](#offline-pre-training-vs-robot-training)
  - [Step 1 — Offline pre-training](#step-1--offline-pre-training-no-robot-no-audio)
  - [Step 2 — Inspect the pre-trained policy](#step-2--inspect-the-pre-trained-policy)
  - [Step 3 — Transition to robot training](#step-3--transition-to-robot-training)
  - [Summary: recommended workflow](#summary-recommended-workflow)

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
   - Start `OSC_Message_Receiver.py` (port 12000) on the GuitarBot machine before training.
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
│   ├── ablation_no_filtration.sh       # Ablation: bypass physics gate
│   ├── ablation_no_audio.sh            # Ablation: no CNN, fret+torque shaping only
│   ├── review_successes.py             # Interactive clip review / relabeling tool
│   └── export_augmented_dataset.py    # Package augmented dataset zip for retraining
├── utils/
│   ├── __init__.py
│   ├── reward.py           # All reward logic — single source of truth
│   ├── audio_reward.py     # Audio capture + classifier dispatch → reward.py
│   └── success_recorder.py # Async background writer for successful harmonic clips
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
| Fret too far | > 3 frets from target | Outside any plausible harmonic neighbourhood |

**Filtration penalty**: −1.0 (constant, independent of how far outside limits the action is).

> **Note**: Silence detection via audio RMS was trialled but removed — onset
> timing relative to capture windows was insufficiently reliable.

### Layer 2: Audio (classifier reward)

Reached only when Layer 1 passes.

$$r = 0.6 \cdot r_\text{audio} + 0.2 \cdot r_\text{fret} + 0.2 \cdot r_\text{torque}$$

| Component | Formula | Range | Purpose |
|-----------|---------|-------|---------|
| $r_\text{audio}$ | `harmonic_prob` from CNN | [0, 1] | Does it *sound* like a harmonic? |
| $r_\text{fret}$ | $\exp\!\left(-\dfrac{e_f^2}{2 \cdot 0.35^2}\right)$ | [0, 1] | Gaussian centred on target fret (σ = 0.35) |
| $r_\text{torque}$ | $2\exp\!\left(-\dfrac{e_\tau^2}{2 \cdot 75^2}\right) - 1$ | [−1, 1] | Shifted Gaussian at optimal presser value 30 (σ = 75) |

- Optimal presser target: **70** encoder units (light touch for harmonics)
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

### Recording Successful Harmonics

Pass `--record-successes` to save every success (classification `harmonic_prob > 0.8`) to
disk as a WAV + JSON pair.  Clips are written asynchronously in a background
thread — no impact on step latency.

```bash
python train.py \
    --model-path ../HarmonicsClassifier/models/best_model.pt \
    --record-successes
```

Clips are saved under the run's `successes/` subdirectory:

```
runs/harmonic_sac_TIMESTAMP/
└── successes/
    ├── 000001_20260218_143201_str2_fret7.00_torque68.wav
    ├── 000001_20260218_143201_str2_fret7.00_torque68.json
    ├── 000002_...
    └── ...
```

Each JSON sidecar contains the full metadata for the episode:

```json
{
  "suggested_label": "harmonic",
  "reviewed": false,
  "string_idx": 2,
  "target_fret": 7,
  "fret": 7.03,
  "torque": 68,
  "harmonic_prob": 0.93,
  "dead_prob": 0.04,
  "general_prob": 0.03,
  "reward": 1.56,
  "device_sr": 44100,
  "timestamp": "2026-02-18T14:32:01.412"
}
```

These clips feed directly into the HarmonicsClassifier retraining pipeline after
review with `scripts/review_successes.py`.

---

### Resuming an Interrupted Run

The `KeyboardInterrupt` handler saves both the model weights and the SAC
replay buffer.  Periodic checkpoints also include the buffer
(`save_replay_buffer=True`).

Every run — including resumes — creates a **fresh timestamped directory**.
The source run is only used to locate the checkpoint and replay buffer to
load from.  A `resumed_from.txt` file in the new directory records the
lineage so the chain of runs is traceable.

To resume:

```bash
# Auto-selects interrupted_model, or latest checkpoint in the source run
python train.py \
    --model-path ../HarmonicsClassifier/models/best_model.pt \
    --resume ./runs/harmonic_sac_20260218_134500

# Resume from a specific checkpoint
python train.py \
    --model-path ../HarmonicsClassifier/models/best_model.pt \
    --resume ./runs/harmonic_sac_20260218_134500 \
    --resume-checkpoint ./runs/harmonic_sac_20260218_134500/checkpoints/harmonic_sac_5000_steps
```

### CLI Reference — train.py

#### Environment

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path PATH` | *(required online)* | Path to the trained HarmonicsClassifier `.pt` file. Not required when `--pretrain` is set. |
| `--string-index N` | `2` | Single string to train on. Must be `0`, `2`, or `4` (strings with pluckers). Ignored when `--string-indices` is provided. |
| `--string-indices N …` | *(unset)* | One or more strings to rotate across episodes (e.g. `--string-indices 0 2 4`). At each `reset()` the active string is sampled uniformly — distributes motor wear while keeping the policy string-aware via a one-hot in the observation. Overrides `--string-index`. |
| `--curriculum MODE` | `easy_to_hard` | Target fret schedule: `easy_to_hard` (fret 7 → 5 → 4, 100 episodes each), `random` (uniform), `fixed_fret` (always fret 7). |
| `--osc-port PORT` | `12000` | UDP port of the GuitarBot middleware. Use `8000` for StringSim. |
| `--audio-device STR` | `Scarlett` | Substring matched against available audio device names to select the recording input. |

#### SAC Hyperparameters

| Flag | Default | Description |
|------|---------|-------------|
| `--total-timesteps N` | `2000` | Training budget. When resuming, treated as an **absolute ceiling** (not additional steps) because `reset_num_timesteps=False` is passed to SB3. |
| `--learning-rate LR` | `3e-4` | Adam learning rate for all SAC networks. |
| `--buffer-size N` | `100000` | Maximum SAC replay buffer capacity (transitions). |
| `--learning-starts N` | `100` | Number of **real robot actions** (OSC sent, audio captured) before SAC gradient updates begin. Filtered steps — which return instantly without touching the robot — are excluded from this count, so the value maps directly to meaningful transitions in the buffer. In `--pretrain` mode reverts to the standard SB3 total-timestep threshold since every step is real. |
| `--batch-size N` | `256` | Mini-batch size drawn from the replay buffer each gradient step. |
| `--tau τ` | `0.005` | Polyak averaging rate for the target network soft update. |
| `--gamma γ` | `0.99` | Discount factor for future rewards. |
| `--ent-coef VALUE` | `auto` | SAC entropy coefficient. `auto` lets SB3 tune it to match a target entropy. Set a fixed float (e.g. `0.1`–`0.3`) for `--pretrain` runs — the auto-tuner can collapse policy entropy to near-zero when steps are instant, leaving the agent stuck at a point estimate. |
| `--device DEVICE` | `auto` | Torch device: `auto` (GPU if available), `cuda`, or `cpu`. |

#### Checkpointing & Logging

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir DIR` | `./runs` | Root directory for run output. Each run creates a timestamped subdirectory `harmonic_sac_YYYYMMDD_HHMMSS/` containing logs, checkpoints, best model, and eval logs. |
| `--checkpoint-freq N` | `5000` | Save a checkpoint (weights + replay buffer) every N timesteps. |
| `--eval-freq N` | `2000` | Run `EvalCallback` every N timesteps to update `best_model/`. |

#### Resuming

| Flag | Default | Description |
|------|---------|-------------|
| `--resume DIR` | *(unset)* | Path to an existing run directory to resume from. Weights and replay buffer are loaded from there, but all new output (logs, checkpoints, best model) goes into a **fresh timestamped directory**. A `resumed_from.txt` file in the new directory records the source path for lineage tracking. Automatically picks `interrupted_model.zip` if present, otherwise the highest-numbered checkpoint. |
| `--resume-checkpoint PATH` | *(unset)* | Explicit checkpoint to load (without `.zip`). Overrides the automatic search within `--resume`. |
| `--clear-buffer` | off | When resuming, discard the saved replay buffer. **Required once** when transitioning from `--pretrain` to the first robot run — offline transitions carry no audio reward and will bias the critic if kept. Do **not** pass this flag when resuming an interrupted robot run, or you will throw away valuable online transitions. |

#### Offline Pre-Training

| Flag | Default | Description |
|------|---------|-------------|
| `--pretrain` | off | Offline mode: no robot, no audio. Reward is computed from the filtration layer only (fret + torque shaping) with a wider fret Gaussian (σ = 1.5 frets). Steps are instant — run hundreds of thousands of steps before touching hardware. Resume on the robot with `--resume <dir> --clear-buffer`. |

#### Robot Safety & Debugging

| Flag | Default | Description |
|------|---------|-------------|
| `--reset-on-exit` | on | Send `/Reset` to GuitarBot when training exits (normally or via Ctrl+C). The middleware serialises the reset behind any active trajectory, but avoid triggering this mid-motion on fragile mechanisms. |
| `--reward-mode MODE` | `full` | Reward variant for ablation studies. `full`: both layers (default). `no_filtration`: bypass the physics gate — all actions reach the CNN. `no_audio`: skip CNN entirely, use fret + torque shaping only (weights 0.5 / 0.5). |
| `--slow` | off | After every **episode**, pause training and display a waveform + mel spectrogram plot with classification results and reward breakdown. Close the window to continue. Use this to visually verify the classifier is hearing the note before committing to a long run. No-op with `--pretrain`. |
| `--record-successes` | off | Save every successful harmonic (`harmonic_prob > 0.8`) to `<run-dir>/successes/` as a float32 WAV + JSON sidecar. Written asynchronously — no step latency penalty. Clips can be reviewed and relabeled with `scripts/review_successes.py` for HarmonicsClassifier retraining. No-op with `--pretrain`. |
| `--verbose` | off | Enable per-step `DEBUG` logging. Normally suppressed to keep the terminal readable, especially during `--pretrain` where steps are instant. |

---

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

### Reviewing & Relabeling Clips

`scripts/review_successes.py` is an interactive terminal tool for a researcher
to listen to each saved clip and confirm or correct its label before
retraining the classifier.

```bash
# Review all clips in a run's successes directory
python scripts/review_successes.py runs/harmonic_sac_TIMESTAMP/successes/

# Review every clip across ALL runs at once
python scripts/review_successes.py --all-runs
python scripts/review_successes.py --all-runs ./runs   # explicit runs root

# Only show clips not yet reviewed
python scripts/review_successes.py runs/harmonic_sac_TIMESTAMP/successes/ --unreviewed-only
python scripts/review_successes.py --all-runs --unreviewed-only

# Skip audio playback (e.g. no speaker connected)
python scripts/review_successes.py runs/harmonic_sac_TIMESTAMP/successes/ --no-audio
```

When audio is enabled the script always prompts you to choose an output device before
starting, so the right speaker/interface is used regardless of the system default.
When `--all-runs` is used, a per-run clip count and total is shown and you must
confirm before review begins.

**Keypress actions during review**

| Key | Action |
|-----|--------|
| `h` | Label as `harmonic` |
| `d` | Label as `dead_note` |
| `g` | Label as `general_note` |
| `r` | Replay audio |
| `s` | Skip (leave label unchanged) |
| `q` | Quit and print session summary |

Reviewing a clip sets `reviewed: true` in the JSON sidecar and updates
`suggested_label` in-place — the WAV file is never modified.  A session
summary with per-label counts is printed on exit.

Reviewed clips (with accurate labels) can then be imported into the
HarmonicsClassifier dataset for retraining:

```bash
# Copy reviewed clips into the HarmonicsClassifier data directory
python ../HarmonicsClassifier/copy_harmonics_to_clips.py \
    runs/harmonic_sac_TIMESTAMP/successes/ \
    --reviewed-only
```

---

### Exporting an Augmented Dataset

`scripts/export_augmented_dataset.py` bundles the original HarmonicsClassifier
`note_clips/` together with all reviewed RL clips into a single zip, ready to
drop on a flash drive and retrain on another machine.

```bash
# Default: uses ./runs and ../HarmonicsClassifier, writes zip to current directory
python scripts/export_augmented_dataset.py

# Explicit paths and custom output location
python scripts/export_augmented_dataset.py \
    --runs-dir ./runs \
    --classifier-dir ../HarmonicsClassifier \
    --output-dir ~/Desktop

# Include clips not yet reviewed (uses suggested_label as-is — use with caution)
python scripts/export_augmented_dataset.py --include-unreviewed

# Export only the new RL clips, without the original dataset
python scripts/export_augmented_dataset.py --no-original
```

The zip contains:
- `note_clips/{harmonic,dead_note,general_note}/` — originals + new `RL_*` clips
- `manifest.json` — per-run provenance, label counts, fret/torque/prob metadata

**On the target machine:**
```bash
unzip harmonics_dataset_augmented_TIMESTAMP.zip
# Move note_clips/ into HarmonicsClassifier/, then:
python run_build_dataset.py
python train_cnn.py
```

---

## Robot Safety

- **`/Reset` on exit is on by default.** Pass `--reset-on-exit` to send it.
- On the GuitarBot side, `OSC_Message_Receiver.py` serialises all
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

---

## Offline Pre-Training vs. Robot Training

### Why pre-train offline?

The physical robot is slow: each step takes ~2–3 s of audio capture and OSC
round-trips.  Offline pre-training skips the robot entirely — steps are
instant — so the policy can learn **which fret positions and presser values
are physically plausible** before the first real string is touched.

Offline reward uses only the **physics filtration layer** (Layer 1) with a
wider fret Gaussian (σ = 1.5 frets instead of 0.35) so the agent receives
a useful gradient signal across the whole fret range, not just within ±0.35
of the target node.

---

### Step 1 — Offline pre-training (no robot, no audio)

```bash
# Recommended starting point
python train.py \
    --pretrain \
    --ent-coef 0.1 \
    --curriculum easy_to_hard \
    --total-timesteps 200000

# Multi-string, more exploration, verbose step logs
python train.py \
    --pretrain \
    --ent-coef 0.3 \
    --curriculum random \
    --string-indices 0 2 4 \
    --total-timesteps 300000 \
    --verbose
```

**Key flags for pre-training**

| Flag | Recommended value | Why |
|------|-------------------|-----|
| `--pretrain` | — | Disables OSC + audio; instant steps |
| `--ent-coef` | `0.1` – `0.3` | Prevents SAC entropy collapse during fast offline steps; `auto` can drive policy std to ~0 |
| `--curriculum` | `easy_to_hard` | Starts on fret 7 (strongest harmonic), then progressively harder targets |
| `--total-timesteps` | `100k` – `300k` | Offline steps are free; run until fret and torque distributions converge |

> **Watch out for policy collapse**: if `--ent-coef auto` is used during
> pre-training, the very fast step rate can cause the SAC entropy tuner to
> over-shrink policy standard deviation.  Use a fixed value like `0.1` and
> increase to `0.3` if exploration is insufficient.

---

### Step 2 — Inspect the pre-trained policy

Use `query.py` to verify the policy is exploring a reasonable range before
deploying on hardware.  With a healthy pre-trained model, stochastic samples
should vary across the expected fret neighbourhood.

```bash
# Single deterministic action
python query.py \
    --model runs/harmonic_sac_TIMESTAMP/best_model/best_model.zip \
    --target-fret 7 --string 2

# 10 stochastic samples — check variance across fret and torque
python query.py \
    --model runs/harmonic_sac_TIMESTAMP/best_model/best_model.zip \
    --target-fret 7 --string 2 \
    --num-actions 10 --stochastic
```

Expected healthy output (varied fret, torque near 70):

```
  #    fret  torque      mm  harmonic
  ---  ------  ------  ------  --------
    1   6.821      64   179.4  -
    2   7.134      71   188.6  YES
    3   7.058      68   186.8  YES
    4   6.654      58   175.2  -
    5   7.203      77   190.2  YES
```

If all rows are identical (policy collapse), re-run pre-training with a
higher `--ent-coef`.

---

### Step 3 — Transition to robot training

Resume from the pre-trained checkpoint and pass `--clear-buffer` to discard
the offline replay buffer.  Pre-train transitions carry **no audio signal**
and will bias the critic if kept.

```bash
python train.py \
    --model-path ../HarmonicsClassifier/models/best_model.pt \
    --resume ./runs/harmonic_sac_TIMESTAMP \
    --clear-buffer \
    --ent-coef auto \
    --curriculum easy_to_hard \
    --total-timesteps 50000
```

| Flag | Why |
|------|-----|
| `--resume` | Loads pre-trained weights + (discarded) buffer |
| `--clear-buffer` | Prevents offline (no-audio) transitions from biasing the online critic |
| `--ent-coef auto` | Let SAC self-tune entropy once audio reward provides richer signal |
| `--total-timesteps` | Real-robot budget is scarcer; start small and extend if needed |

> **`--clear-buffer` is a one-time flag for the offline → online transition only.**
> Without it, the replay buffer is filled with transitions that have `audio_reward = 0`
> by construction, which will suppress the agent's weight on audio even once the real
> classifier starts providing signal.
> Once you are running on the robot, do **not** pass `--clear-buffer` on subsequent
> resumes — those interruptions will have saved a buffer of real audio transitions
> that are worth keeping.

---

### Summary: recommended workflow

```
1. Offline pre-train (fast, no hardware)
   python train.py --pretrain --ent-coef 0.1 --curriculum easy_to_hard --total-timesteps 200000

2. Verify policy quality
   python query.py --model runs/.../best_model/best_model.zip --num-actions 10 --stochastic

3. Deploy on robot — clear the offline buffer (one time only)
   python train.py --model-path ../HarmonicsClassifier/models/best_model.pt \
       --resume runs/harmonic_sac_PRETRAIN_TIMESTAMP --clear-buffer --ent-coef auto

4. Resume an interrupted robot run — do NOT clear the buffer
   python train.py --model-path ../HarmonicsClassifier/models/best_model.pt \
       --resume runs/harmonic_sac_ROBOT_TIMESTAMP --ent-coef auto
```