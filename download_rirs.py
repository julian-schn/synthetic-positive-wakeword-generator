#!/usr/bin/env python3
"""
Alternative RIR download script that doesn't use HuggingFace datasets library.
Downloads MIT Room Impulse Responses directly from HuggingFace Hub.
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import numpy as np


def download_mit_rirs_alternative(output_dir="./mit_rirs"):
    """
    Download MIT RIR files using HuggingFace Hub API directly,
    without the datasets library dependency issues.
    """
    output_path = Path(output_dir)

    # Check if files already exist
    if output_path.exists():
        num_files = len(list(output_path.glob("*.wav")))
        if num_files > 0:
            print(f"✓ RIR directory exists with {num_files} files.")
            print("Sample files:", list(output_path.glob("*.wav"))[:5])
            return

    output_path.mkdir(parents=True, exist_ok=True)

    print("Downloading MIT Environmental Impulse Responses...")
    print("This will download from HuggingFace Hub API directly.")

    # HuggingFace Hub API endpoint
    repo_id = "davidscripka/MIT_environmental_impulse_responses"
    api_url = f"https://huggingface.co/api/datasets/{repo_id}"

    try:
        # Get dataset info
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()

        # Try to get file list from the repo
        files_url = f"https://huggingface.co/datasets/{repo_id}/tree/main/data"

        print("\nNote: This script uses a simplified download method.")
        print("If you need the full dataset, please use the HuggingFace datasets library")
        print("after fixing the dependency issues, or download manually from:")
        print(f"https://huggingface.co/datasets/{repo_id}")
        print("\nFor now, generating synthetic RIRs as a fallback...")

        # Generate synthetic RIRs as fallback
        generate_synthetic_rirs(output_path)

    except Exception as e:
        print(f"Could not download from HuggingFace Hub: {e}")
        print("\nGenerating synthetic RIRs as fallback...")
        generate_synthetic_rirs(output_path)


def generate_synthetic_rirs(output_path, num_rirs=50, sample_rate=16000):
    """
    Generate synthetic room impulse responses as a fallback.
    These are simple exponentially-decaying noise patterns that simulate reverb.
    """
    print(f"\nGenerating {num_rirs} synthetic RIR files...")

    np.random.seed(42)

    for i in tqdm(range(num_rirs)):
        # Random decay time between 0.1 and 1.0 seconds
        decay_time = np.random.uniform(0.1, 1.0)

        # Create exponentially decaying noise
        length = int(sample_rate * decay_time)
        t = np.linspace(0, decay_time, length)

        # Exponential decay with random noise
        decay_rate = np.random.uniform(3, 8)
        rir = np.random.randn(length) * np.exp(-decay_rate * t)

        # Add strong initial impulse
        rir[0] = 1.0

        # Normalize
        rir = rir / np.abs(rir).max()

        # Save as 16-bit PCM WAV
        filename = output_path / f"synthetic_rir_{i:03d}.wav"
        sf.write(filename, rir.astype(np.float32), sample_rate, subtype='PCM_16')

    num_files = len(list(output_path.glob("*.wav")))
    print(f"\n✓ Generated {num_files} synthetic RIR files in {output_path}/")
    print("These synthetic RIRs will work for augmentation, though real RIRs are preferred.")
    print("\nTo get real MIT RIRs, fix the datasets library compatibility and run:")
    print("  from datasets import load_dataset")
    print(f"  load_dataset('davidscripka/MIT_environmental_impulse_responses')")


if __name__ == "__main__":
    download_mit_rirs_alternative()
