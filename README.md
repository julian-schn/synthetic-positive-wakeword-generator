# Synthetic Positive Wakeword Generator

Generate synthetic positive wakeword samples for training [openWakeWord](https://github.com/dscripka/openWakeWord) models using Kokoro TTS.

## Overview

This tool creates reproducible, high-coverage datasets of positive wakeword utterances through TTS synthesis and realistic audio augmentation. The pipeline applies physically plausible transformations (gain, reverb, noise) to prevent models from overfitting to clean TTS artifacts.

## Requirements

- Python 3.10+
- [Kokoro](https://github.com/hexgrad/kokoro) v0.19+
- Room Impulse Responses (RIRs) and background noise samples (optional)

## Setup

```bash
# Create conda environment
conda create -n wakeword python=3.10
conda activate wakeword

# Install dependencies
pip install -r requirements.txt

# Register kernel for Jupyter
python -m ipykernel install --user --name wakeword
```

## Usage

### Jupyter Notebook (recommended)

Open `generate.ipynb` for interactive experimentation:

```bash
jupyter notebook generate.ipynb
```

The notebook includes demos for:
- Voice blending
- Speed variation
- Gain adjustment
- Low-pass filter (distance simulation)
- Room reverb (RIR convolution)
- Background noise mixing
- Full augmentation chain

### CLI

```bash
python generate.py --wakeword "Hey Jarvis" --count 100
```

| Flag | Description |
|------|-------------|
| `--wakeword, -w` | Target phrase (required) |
| `--count, -n` | Number of samples (default: 100) |
| `--output, -o` | Output directory (default: `output/`) |
| `--config, -c` | Augmentation config YAML |
| `--rir-dir` | Room impulse response directory |
| `--noise-dir` | Background noise directory |
| `--seed, -s` | Master random seed (default: 42) |

## Output

All samples are output as **16 kHz, mono, 16-bit PCM** WAV files:

```
output/
├── manifest.jsonl    # Provenance: seed, voice, augmentation params
└── positives/
    └── *.wav
```

## License

MIT
