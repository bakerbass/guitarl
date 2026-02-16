# guitaRL - Reinforcement Learning for Guitar Harmonics

RL environment for learning to play natural harmonics on the StringSim guitar simulator.

## Setup

1. **Install dependencies**:
```bash
conda env create -f environment.yml
conda activate guitaRL
```

2. **Setup VB-CABLE**:
   - Install VB-CABLE virtual audio device
   - Route StringSim audio output to VB-Cable Input

3. **Verify HarmonicsClassifier model**:
   - Ensure trained model exists at `../HarmonicsClassifier/models/best_model.pt`

## Project Structure

```
guitaRL/
├── env/
│   ├── __init__.py
│   ├── action_space.py          # RLFretAction, PresserAction, GuitarBotActionSpace
│   ├── osc_client.py            # OSC communication with GuitarBot
│   └── harmonic_env.py          # Gymnasium environment (HarmonicEnv)
├── utils/
│   ├── __init__.py
│   ├── reward.py                # Shared reward constants & function (single source of truth)
│   └── audio_reward.py          # Audio capture + classifier → calls reward.py
├── train.py                      # SAC training script
├── test_rl_loop.py               # Diagnostic: action → OSC → audio → classify → reward
├── evaluate.py                   # Evaluation script
├── environment.yml               # Conda dependencies (env name: guitaRL)
└── README.md
```

## Usage
### Test Loop

Diagnostic script that runs the full action → OSC → audio → classify → reward
pipeline one note at a time, plotting each note with classification and reward:

```bash
conda run -n guitaRL python test_rl_loop.py \
    --model ../HarmonicsClassifier/models/best_model.pt \
    --num-tests 5 \
    --target-harmonic
```

Plots are shown by default; pass `--no-plot` to disable.

### Training

Train agent to learn natural harmonics at frets 4, 5, and 7:

```bash
# Basic training (uses guitaRL conda env)
conda activate guitaRL
python train.py \
    --model-path ../HarmonicsClassifier/models/best_model.pt

# Custom training parameters
conda run -n guitaRL python train.py \
    --model-path ../HarmonicsClassifier/models/best_model.pt \
    --curriculum easy_to_hard \
    --total-timesteps 50000 \
    --learning-rate 3e-4 \
    --string-index 3
```

**Curriculum modes**:
- `easy_to_hard`: Start with fret 7, progress to 5, then 4
- `random`: Random fret each episode
- `fixed_fret`: Always train on fret 7



### Evaluation

Evaluate trained policy:

```bash
# Basic evaluation
python evaluate.py --model runs/harmonic_sac_TIMESTAMP/best_model/best_model.zip --episodes 20

# With visualization
python evaluate.py \
    --model runs/harmonic_sac_TIMESTAMP/best_model/best_model.zip \
    --episodes 20 \
    --visualize \
    --deterministic

# Test specific fret
python evaluate.py \
    --model runs/harmonic_sac_TIMESTAMP/best_model/best_model.zip \
    --target-fret 7 \
    --episodes 10 \
    --visualize
```

## Environment Details

### Observation Space
- Target fret (one-hot: 3 values)
- Current position (mm)
- Current force (0-1)
- Position history (last 3)
- Force history (last 3)

**Total: 11 dimensions**

### Action Space
- Fret position (continuous, fractional frets 0.0 – 9.0)
- Torque (continuous, 0 – 650)

### Reward Function — Two-Layer Architecture

Defined once in `utils/reward.py` and imported by both the training env and the test loop.

#### Layer 1: Filtration (physics gate)

Before running the CNN classifier, a fast physics check rejects obviously
bad actions with a flat **−1.0 penalty**.  The classifier is never called,
saving inference time and giving the agent an immediate, unambiguous signal.

| Check | Threshold | Rationale |
|-------|-----------|-----------|
| Torque too high | > 500 | Harmonics are physically impossible with heavy fretting |
| Torque too low | < 10 | Presser barely touching the string — no useful sound |
| Fret too far | > 3 frets from target | Not even in the right neighbourhood |
| Silence | RMS < 0.005 | No audible onset detected — nothing to classify |

#### Layer 2: Audio (classifier reward)

Only reached when the filtration layer passes.  The CNN harmonic
probability is the primary reward signal, with small shaping bonuses
for fret accuracy and torque optimality.

$$r = 0.6 \cdot r_{\text{audio}} + 0.2 \cdot r_{\text{fret}} + 0.2 \cdot r_{\text{torque}}$$

| Component | Formula | Range | Purpose |
|-----------|---------|-------|---------|
| Audio     | `harmonic_prob` from CNN classifier | [0, 1] | Does it *sound* like a harmonic? |
| Fret      | $\exp\!\left(-\frac{e_f^2}{2 \cdot 0.3^2}\right)$ | [0, 1] | Gaussian at target fret (σ = 0.3) |
| Torque    | $2\exp\!\left(-\frac{e_\tau^2}{2 \cdot 75^2}\right) - 1$ | [−1, 1] | Shifted Gaussian at optimal torque 30 |

- Optimal torque: **30** (light touch for harmonics)
- Torque tolerance (σ): **75**
- **Success bonus**: +1.0 when `harmonic_prob > 0.8`

### Target Harmonics
| Fret | Position (mm) | Harmonic |
|------|---------------|----------|
| 4    | 112.0         | Major 17th |
| 5    | 139.0         | Major 14th |
| 7    | 187.0         | Perfect 12th |

## Training Tips

1. **Start with curriculum learning**: `--curriculum easy_to_hard` helps agent learn progressively
2. **Monitor tensorboard**: `tensorboard --logdir runs/harmonic_sac_TIMESTAMP/logs`
3. **Adjust episode length**: Default 10 steps, increase for exploration
4. **Torque range**: Harmonics require light touch; optimal torque ≈ 30, max safe 650
5. **Tune reward weights**: Edit `utils/reward.py` — both training and test scripts pick up changes automatically

## Todo:
TODO:

filtration layer: actively penalize totally wrong choices (high torque, 0 torque) or nonsensicle audio output (no onset detected). This layer can allow for a more physics based reward, while the main layer prioritizes the audio reward

(low priority, mainly for paper clarity)
refactor for clarity: we are actually changing the presser position, not the torque