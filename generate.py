#!/usr/bin/env python3
"""CLI entry point for synthetic wakeword generation."""

import argparse
import re
from pathlib import Path


TARGET_SAMPLE_RATE = 16000


def slugify(text: str) -> str:
    """Convert text to filename-safe slug."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '_', text)
    return text


def generate_filename(
    wakeword: str,
    voice_config: dict,
    speed: float,
    seed: int
) -> str:
    """Generate descriptive filename for a sample."""
    slug = slugify(wakeword)
    
    if voice_config["voice_type"] == "blend":
        voice_part = f"mix_{voice_config['voice_a'][:2]}_{voice_config['voice_b'][:2]}"
    else:
        voice_part = voice_config["voice"]
    
    speed_part = f"s{int(speed * 100):03d}"
    
    return f"{slug}_{voice_part}_{speed_part}_seed{seed}.wav"


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic positive wakeword samples"
    )
    parser.add_argument(
        "--wakeword", "-w",
        required=True,
        help="Target wakeword phrase (e.g., 'Hey Jarvis')"
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=100,
        help="Number of samples to generate (default: 100)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output/)"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path("augment_config.yaml"),
        help="Augmentation config file (default: augment_config.yaml)"
    )
    parser.add_argument(
        "--rir-dir",
        type=Path,
        help="Directory containing room impulse response files"
    )
    parser.add_argument(
        "--noise-dir",
        type=Path,
        help="Directory containing background noise files"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Master random seed (default: 42)"
    )
    parser.add_argument(
        "--voices",
        nargs="+",
        default=None,
        help="Voice pool (default: af_bella, af_nicole, af_sarah, ...)"
    )
    
    args = parser.parse_args()
    
    # Lazy imports for dependencies
    import numpy as np
    import soundfile as sf
    import yaml
    from tqdm import tqdm

    from src.tts import TTSEngine, KOKORO_SAMPLE_RATE, DEFAULT_VOICES
    from src.augment import AugmentationChain
    from src.audio_utils import resample, to_pcm16, add_silence_padding
    from src.manifest import ManifestEntry, ManifestWriter
    
    # Load augmentation config
    if args.config.exists():
        with open(args.config) as f:
            aug_config = yaml.safe_load(f)
    else:
        print(f"Warning: Config file {args.config} not found, using defaults")
        aug_config = {}
    
    # Setup output directories
    positives_dir = args.output / "positives"
    positives_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    print("Initializing TTS engine...")
    tts = TTSEngine()
    
    augmenter = AugmentationChain(
        config=aug_config,
        rir_dir=args.rir_dir,
        noise_dir=args.noise_dir
    )
    
    manifest = ManifestWriter(args.output / "manifest.jsonl")
    
    # Master RNG for reproducibility (NFR-1)
    master_rng = np.random.default_rng(args.seed)
    
    voice_pool = args.voices or DEFAULT_VOICES
    speed_range = aug_config.get("speed", {}).get("range", [0.9, 1.1])
    silence_cfg = aug_config.get("silence", {})
    leading_range = silence_cfg.get("leading_range", [0.1, 0.5])
    trailing_range = silence_cfg.get("trailing_range", [0.1, 0.5])
    
    print(f"Generating {args.count} samples for '{args.wakeword}'...")
    
    for i in tqdm(range(args.count), desc="Generating"):
        # Per-sample seed for reproducibility
        sample_seed = int(master_rng.integers(0, 2**31))
        rng = np.random.default_rng(sample_seed)
        
        # Get random voice configuration
        voice, voice_config = tts.get_random_voice_config(
            rng, voice_pool=voice_pool
        )
        
        # Random speed jitter
        speed = rng.uniform(*speed_range)
        
        # Synthesize speech (24kHz)
        try:
            audio = tts.synthesize(args.wakeword, voice, speed)
        except Exception as e:
            print(f"Warning: TTS failed for sample {i}: {e}")
            continue
        
        # Apply augmentations
        audio, aug_metadata = augmenter.apply(audio, KOKORO_SAMPLE_RATE, rng)
        
        # Add silence padding (still at 24kHz)
        audio = add_silence_padding(
            audio, KOKORO_SAMPLE_RATE,
            leading_range, trailing_range, rng
        )
        
        # Resample to 16kHz (FR-1)
        audio = resample(audio, KOKORO_SAMPLE_RATE, TARGET_SAMPLE_RATE)
        
        # Convert to 16-bit PCM
        audio_pcm = to_pcm16(audio)
        
        # Generate filename and save
        filename = generate_filename(
            args.wakeword, voice_config, speed, sample_seed
        )
        output_path = positives_dir / filename
        sf.write(output_path, audio_pcm, TARGET_SAMPLE_RATE, subtype='PCM_16')
        
        # Write manifest entry
        entry = ManifestEntry(
            filename=filename,
            seed=sample_seed,
            wakeword=args.wakeword,
            voice_config=voice_config,
            speed=round(speed, 3),
            augmentations=aug_metadata
        )
        manifest.write_entry(entry)
    
    print(f"Done! Output saved to {args.output}/")


if __name__ == "__main__":
    main()
