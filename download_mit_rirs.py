#!/usr/bin/env python3
"""
MIT Room Impulse Response Downloader

Downloads 270 real-world room impulse response recordings from the MIT McDermott Lab
for use in realistic audio augmentation. These RIRs capture actual room acoustics from
various environments (small rooms, large halls, etc.) and are essential for training
robust wakeword models.

This script automatically handles complex package dependency conflicts:
- Downgrades NumPy to 1.26.4 (required for pyarrow 12.0.0)
- Installs pyarrow 12.0.0 (has PyExtensionType for datasets compatibility)
- Installs datasets 2.18.0 (for downloading from HuggingFace Hub)
- Installs librosa and soundfile (for audio decoding)

Dataset: https://mcdermottlab.mit.edu/Reverb/IR_Survey.html
HuggingFace: davidscripka/MIT_environmental_impulse_responses

Usage:
    python download_mit_rirs.py

Output:
    ./mit_rirs/*.wav (270 files, ~500MB total)
"""

import sys
import subprocess
import os
from pathlib import Path

def main():
    output_dir = "./mit_rirs"

    # Check if files already exist
    if Path(output_dir).exists():
        num_files = len(list(Path(output_dir).glob("*.wav")))
        if num_files > 0:
            print(f"✓ RIR directory already exists with {num_files} files.")
            print("Sample files:", list(Path(output_dir).glob("*.wav"))[:5])
            return

    print("=" * 60)
    print("MIT Room Impulse Response Downloader")
    print("=" * 60)
    print(f"Python: {sys.executable}\n")

    # Step 1: Uninstall incompatible versions
    print("Step 1: Removing incompatible packages...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "datasets", "pyarrow"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("  ✓ Removed\n")

    # Step 2: Install compatible numpy (pyarrow 12.0.0 needs numpy<2)
    print("Step 2: Installing numpy 1.26.4 (pyarrow 12.0.0 needs numpy<2)...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "numpy==1.26.4", "-q"],
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ✗ ERROR: {result.stderr}")
        sys.exit(1)
    print("  ✓ Installed\n")

    # Step 3: Install compatible pyarrow
    print("Step 3: Installing pyarrow 12.0.0...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "pyarrow==12.0.0", "-q"],
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ✗ ERROR: {result.stderr}")
        sys.exit(1)
    print("  ✓ Installed\n")

    # Step 4: Install audio libraries
    print("Step 4: Installing librosa and soundfile...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "librosa", "soundfile", "-q"],
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ✗ ERROR: {result.stderr}")
        sys.exit(1)
    print("  ✓ Installed\n")

    # Step 5: Install compatible datasets (upgrade to 2.18 for fsspec compatibility)
    print("Step 5: Installing datasets 2.18.0...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "datasets==2.18.0", "-q"],
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ✗ ERROR: {result.stderr}")
        sys.exit(1)
    print("  ✓ Installed\n")

    # Step 6: Import and download
    print("Step 6: Importing libraries...")
    try:
        from datasets import load_dataset
        from tqdm import tqdm
        import scipy.io
        import numpy as np
        import pyarrow as pa

        print(f"  ✓ Using pyarrow {pa.__version__}")
        print(f"  ✓ PyExtensionType available: {hasattr(pa, 'PyExtensionType')}\n")
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        sys.exit(1)

    # Step 7: Download
    print("=" * 60)
    print("Downloading MIT Environmental Impulse Responses...")
    print("This will take a few minutes (~500MB)")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    try:
        rir_dataset = load_dataset("davidscripka/MIT_environmental_impulse_responses",
                                    split="train", streaming=True)

        processed = 0
        for row in tqdm(rir_dataset, desc="Downloading"):
            name = row['audio']['path'].split('/')[-1]
            scipy.io.wavfile.write(
                os.path.join(output_dir, name),
                16000,
                (row['audio']['array'] * 32767).astype(np.int16)
            )
            processed += 1

        num_files = len(list(Path(output_dir).glob("*.wav")))

        print("\n" + "=" * 60)
        print(f"✓ SUCCESS!")
        print(f"  Downloaded: {num_files} RIR files")
        print(f"  Location: {os.path.abspath(output_dir)}/")
        print("=" * 60)
        print("\nYou can now run your Jupyter notebook!")

    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
