# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- `download_mit_rirs.py` - Standalone script for downloading MIT Room Impulse Responses
  - Automatically handles complex package dependency conflicts
  - Downloads 270 real-world RIR recordings (~500MB)
  - Fixes numpy/pyarrow/datasets version compatibility issues
- Comprehensive troubleshooting section in README
- Project structure documentation
- Citation section for academic use
- Enhanced .gitignore with Jupyter and additional patterns

### Changed
- Updated README.md with detailed installation instructions
- Improved setup process with step-by-step guide
- Updated notebook Cell 5 to check for MIT RIRs and provide clear instructions
- Corrected Python version requirement to 3.11+ (tested version)
- Removed non-existent CLI documentation (generate.py doesn't exist)
- Updated requirements.txt to use flexible version constraints

### Fixed
- **Critical:** Package dependency conflicts between numpy 2.x, pyarrow, and datasets
  - Solution: Downgrade numpy to 1.26.4, use pyarrow 12.0.0 with datasets 2.18.0
- **Critical:** Missing FFmpeg causing torchcodec failures
  - Solution: Added FFmpeg installation instructions for all platforms
- **Critical:** `AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'`
  - Solution: Use compatible pyarrow version (12.0.0) that has this attribute
- Missing librosa and soundfile packages for audio decoding
- Invalid glob pattern error in datasets library (fixed by upgrading to 2.18.0)

### Technical Details

The following dependency stack is known to work:
```
numpy==1.26.4          # pyarrow 12.0.0 requires numpy<2
pyarrow==12.0.0        # Has PyExtensionType for datasets compatibility
datasets==2.18.0       # Compatible with pyarrow 12.0.0 and modern fsspec
librosa                # Required for audio decoding
soundfile              # Required for audio I/O
```

## [2026-01-13] - Initial Implementation

### Added
- Jupyter notebook with interactive audio augmentation demos
- TTS synthesis using Kokoro
- Voice blending support (10+ voices)
- Comprehensive augmentation pipeline:
  - Gain adjustment
  - Speed variation
  - Pitch shifting
  - Time stretching
  - Low-pass/high-pass/band-pass filtering
  - Room impulse response (reverb)
  - Background noise mixing
- Batch generation with provenance tracking
- Manifest file generation (JSONL format)
- Modular source code structure:
  - `src/tts.py` - TTS engine wrapper
  - `src/augment.py` - Augmentation pipeline
  - `src/audio_utils.py` - Audio processing utilities
  - `src/manifest.py` - Manifest generation
- Technical specification (SPEC.md)
- MIT License

### Features
- Fully reproducible with seed-based generation
- 16 kHz, mono, 16-bit PCM output (openWakeWord format)
- Physically plausible augmentation chain
- Configurable augmentation parameters
- Audio preview in Jupyter notebook
