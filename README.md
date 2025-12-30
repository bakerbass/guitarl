# guitaRL - Reinforcement Learning for Guitar Harmonics

RL environment for learning to play natural harmonics on the StringSim guitar simulator.

## Setup

1. **Install dependencies**:
```bash
conda env update -f environment.yml -n gen-audio-bench
conda activate gen-audio-bench
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
│   ├── osc_client.py           # OSC communication with StringSim
│   └── stringsim_env.py        # Gymnasium environment
├── utils/
│   ├── __init__.py
│   └── audio_reward.py         # Audio-based reward using HarmonicsClassifier
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── environment.yml              # Conda dependencies
└── README.md
```

## Usage

### Training

Train agent to learn natural harmonics at frets 4, 5, and 7:

```bash
# Basic training with curriculum learning (easy to hard)
python train.py --model-path ../HarmonicsClassifier/models/best_model.pt

# Custom training parameters
python train.py \
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
- Position (continuous, 0-234 mm)
- Force (continuous, 0-1)

### Reward Function
- **Audio reward (50%)**: Harmonic classifier confidence
- **Position reward (30%)**: Gaussian centered at target harmonic position
- **Force reward (20%)**: Gaussian centered at optimal force (0.3)

**Success bonus**: +1.0 when harmonic_prob > 0.8

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
4. **Force range**: Harmonics require light touch (0.15-0.5), optimal ~0.3

## Troubleshooting

- **No audio captured**: Check VB-CABLE routing in Windows Sound settings
- **Low success rate**: Try `--curriculum fixed_fret` to master one position first
- **Slow training**: Reduce `--capture-duration` in audio reward (default 0.8s)
- **OSC connection failed**: Ensure StringSim is running and listening on port 8000
