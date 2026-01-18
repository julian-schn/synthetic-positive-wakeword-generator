#!/usr/bin/env python3
"""
Test script to verify the RIR download will work.
Run this with: python test_download.py
"""

import sys
import subprocess
import os
from pathlib import Path

def test_download():
    output_dir = './mit_rirs_test'

    print("=" * 60)
    print("TESTING RIR DOWNLOAD LOGIC")
    print("=" * 60)
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print()

    # Step 1: Check current versions
    print("Step 1: Checking current package versions...")
    try:
        import pyarrow as pa
        print(f"  ✓ pyarrow {pa.__version__} is installed")
        print(f"    - Has PyExtensionType: {hasattr(pa, 'PyExtensionType')}")
        print(f"    - Has ExtensionType: {hasattr(pa, 'ExtensionType')}")
    except ImportError:
        print("  ✗ pyarrow not installed")

    try:
        import datasets
        print(f"  ✓ datasets {datasets.__version__} is installed")
    except ImportError:
        print("  ✗ datasets not installed")
    print()

    # Step 2: Install compatible versions
    print("Step 2: Installing compatible versions...")
    print("  Uninstalling existing packages...")
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'datasets', 'pyarrow'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("  Installing pyarrow 12.0.0...")
    result = subprocess.run([sys.executable, '-m', 'pip', 'install', 'pyarrow==12.0.0'],
                           capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ✗ Failed to install pyarrow 12.0.0")
        print(f"    Error: {result.stderr}")
        return False

    print("  Installing datasets 2.14.0...")
    result = subprocess.run([sys.executable, '-m', 'pip', 'install', 'datasets==2.14.0'],
                           capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ✗ Failed to install datasets 2.14.0")
        print(f"    Error: {result.stderr}")
        return False

    print("  ✓ Packages installed successfully")
    print()

    # Step 3: Clear module cache and reimport
    print("Step 3: Importing fresh modules...")
    if 'datasets' in sys.modules:
        del sys.modules['datasets']
    if 'pyarrow' in sys.modules:
        del sys.modules['pyarrow']

    try:
        import pyarrow as pa
        import datasets
        print(f"  ✓ pyarrow {pa.__version__}")
        print(f"  ✓ datasets {datasets.__version__}")
        print(f"    - PyExtensionType available: {hasattr(pa, 'PyExtensionType')}")
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False
    print()

    # Step 4: Test download
    print("Step 4: Testing download (3 files)...")
    try:
        from datasets import load_dataset
        import scipy.io
        import numpy as np

        os.makedirs(output_dir, exist_ok=True)

        rir_dataset = load_dataset('davidscripka/MIT_environmental_impulse_responses',
                                    split='train', streaming=True)

        processed = 0
        for row in rir_dataset:
            name = row['audio']['path'].split('/')[-1]
            print(f"  Processing: {name}")
            scipy.io.wavfile.write(
                os.path.join(output_dir, name),
                16000,
                (row['audio']['array'] * 32767).astype(np.int16)
            )
            processed += 1
            if processed >= 3:
                break

        num_files = len(list(Path(output_dir).glob('*.wav')))
        print(f"  ✓ Downloaded {num_files} files to {output_dir}/")
        print()

        # Cleanup test directory
        import shutil
        shutil.rmtree(output_dir)
        print("  ✓ Test files cleaned up")

        return True

    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_download()
    print()
    print("=" * 60)
    if success:
        print("✓ TEST PASSED - The notebook cell should work!")
    else:
        print("✗ TEST FAILED - There are still issues to resolve")
    print("=" * 60)
