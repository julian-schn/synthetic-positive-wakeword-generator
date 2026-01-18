# Synthetic Positive Wakeword Generator

Generate synthetic positive wakeword samples for training [openWakeWord](https://github.com/dscripka/openWakeWord) models using Kokoro TTS with realistic audio augmentation.

## Overview

This tool creates reproducible, high-coverage datasets of positive wakeword utterances through TTS synthesis and realistic audio augmentation. The pipeline applies physically plausible transformations (gain, reverb, noise, filtering) to prevent models from overfitting to clean TTS artifacts.

**Key Features:**
- 🎤 High-quality TTS synthesis using [Kokoro](https://github.com/hexgrad/kokoro)
- 🔊 Realistic audio augmentation with MIT Room Impulse Responses
- 🎛️ 10+ built-in voices with voice blending support
- 📊 Interactive Jupyter notebook with audio previews
- 🔄 Fully reproducible with seed-based generation
- 📝 Provenance tracking via manifest files

## Requirements

- Python 3.11+ (tested with 3.11)
- [Kokoro](https://github.com/hexgrad/kokoro) v0.9+
- FFmpeg (for audio decoding)
- ~500MB disk space for MIT Room Impulse Responses

## Installation

### 1. Create Conda Environment

```bash
# Create and activate environment
conda create -n wakeword python=3.11
conda activate wakeword
```

### 2. Install FFmpeg

FFmpeg is required for audio file decoding:

```bash
# macOS
conda install -c conda-forge ffmpeg

# Linux
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### 3. Install Python Dependencies

```bash
# Install all required packages
python -m pip install -r requirements.txt
```

### 4. Download MIT Room Impulse Responses

Real-world room impulse responses are **required** for realistic reverb augmentation:

```bash
# Run the download script (takes a few minutes, ~500MB)
python download_mit_rirs.py
```

This script will:
- Automatically fix package version compatibility issues
- Download 270 real room impulse response recordings from MIT
- Save files to `mit_rirs/` directory

**Note:** The script handles complex dependency conflicts (numpy, pyarrow, datasets) automatically. If you encounter issues, the script will display helpful error messages.

### 5. Set Up Jupyter Kernel (Optional)

If you want to use the Jupyter notebook:

```bash
# Install and register the kernel
python -m pip install jupyter ipykernel
python -m ipykernel install --user --name wakeword --display-name "Python 3 (wakeword)"
```

## Usage

### Jupyter Notebook (Recommended)

The notebook provides an interactive interface with audio previews:

```bash
jupyter notebook generate.ipynb
```

**Features:**
1. **Audio Manipulation Demos** - Interactive examples of each augmentation type:
   - Basic TTS synthesis
   - Voice blending
   - Speed variation
   - Gain adjustment
   - Low-pass filtering (distance simulation)
   - High-pass filtering
   - Band-pass filtering (telephone/radio effect)
   - Pitch shifting
   - Time stretching
   - Room impulse response (reverb)
   - Background noise mixing
   - Combined augmentation chain

2. **Batch Generation** - Generate hundreds of augmented samples with:
   - Configurable wakeword phrase
   - Adjustable sample count
   - Customizable augmentation parameters
   - Automatic provenance tracking

### Configuration

Edit the configuration cells in the notebook to customize generation:

```python
# Basic settings
WAKEWORD = "How do you want to do this"  # Your target phrase
NUM_SAMPLES = 100                         # Number of samples to generate
MASTER_SEED = 42                          # For reproducibility

# Advanced augmentation settings
AUG_CONFIG = {
    "gain": {"db_range": [-6, 3]},                    # Volume adjustment
    "speed": {"range": [0.9, 1.1]},                   # Speech tempo
    "pitch": {"probability": 0.5, "semitone_range": [-2, 2]},  # Pitch shift
    "reverb": {"probability": 0.7},                   # Room acoustics
    "noise": {"probability": 0.8, "snr_db_range": [15, 30]},   # Background noise
    "lowpass": {"probability": 0.3, "cutoff_range": [3000, 6000]},  # Distance sim
    # ... see notebook for full configuration options
}
```

## Output

All samples are output as **16 kHz, mono, 16-bit PCM** WAV files (openWakeWord format):

```
output/
├── manifest.jsonl           # Provenance: seed, voice, augmentation params
└── positives/
    ├── how_do_you_want_to_do_this_af_bella_s091_seed184566854.wav
    ├── how_do_you_want_to_do_this_mix_af_am_s108_seed882294417.wav
    └── ...
```

The `manifest.jsonl` file contains full provenance information for reproducibility:
```json
{
  "filename": "how_do_you_want_to_do_this_af_bella_s091_seed184566854.wav",
  "seed": 184566854,
  "wakeword": "How do you want to do this",
  "voice_config": {"voice_type": "single", "voice": "af_bella"},
  "speed": 0.91,
  "augmentations": {"gain_db": -2.05, "reverb_applied": true, ...}
}
```

## Troubleshooting

### Package Dependency Issues

If you encounter `AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'`:

1. **Restart your Jupyter kernel** (Kernel → Restart Kernel)
2. Run the download script again: `python download_mit_rirs.py`
3. The script automatically fixes version conflicts

### FFmpeg Not Found

If you see errors about missing FFmpeg or `torchcodec`:

```bash
# Install FFmpeg in your conda environment
conda install -c conda-forge ffmpeg
```

### NumPy Version Conflicts

If you see `numpy.core.multiarray failed to import`:

```bash
# The download script handles this automatically, but you can also:
pip install "numpy<2.0"
```

## Project Structure

```
.
├── generate.ipynb              # Main Jupyter notebook
├── download_mit_rirs.py       # MIT RIR download script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── SPEC.md                    # Technical specification
├── src/
│   ├── tts.py                # Kokoro TTS wrapper
│   ├── augment.py            # Augmentation pipeline
│   ├── audio_utils.py        # Audio processing utilities
│   └── manifest.py           # Manifest generation
└── mit_rirs/                 # Downloaded room impulse responses (270 files)
```

## Citation

If you use this tool in your research, please cite:

- **Kokoro TTS**: [hexgrad/Kokoro](https://github.com/hexgrad/kokoro)
- **MIT Room Impulse Responses**: [McDermott Lab](https://mcdermottlab.mit.edu/Reverb/IR_Survey.html)
- **openWakeWord**: [dscripka/openWakeWord](https://github.com/dscripka/openWakeWord)

## License

MIT
