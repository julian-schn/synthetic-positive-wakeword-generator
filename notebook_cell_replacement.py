# REPLACEMENT for Cell 5: Download MIT Room Impulse Responses
# This version works around the datasets/pyarrow compatibility issues

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import soundfile as sf

output_dir = "./mit_rirs"

print("CWD:", os.getcwd())
print("Output dir:", os.path.abspath(output_dir))

# Check if files already exist
num_files = len(list(Path(output_dir).glob("*.wav"))) if Path(output_dir).exists() else 0

if num_files > 0:
    print(f"✓ RIR directory exists with {num_files} files.")
    print("Sample files:", list(Path(output_dir).glob("*.wav"))[:5])
else:
    # Try to use datasets library
    try:
        from datasets import load_dataset
        import scipy.io

        os.makedirs(output_dir, exist_ok=True)
        print("Downloading MIT Environmental Impulse Responses...")
        print("This is required for realistic reverb augmentation.")

        rir_dataset = load_dataset("davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True)

        # Save clips to 16-bit PCM wav files
        processed = 0
        for row in tqdm(rir_dataset):
            name = row['audio']['path'].split('/')[-1]
            scipy.io.wavfile.write(
                os.path.join(output_dir, name),
                16000,
                (row['audio']['array'] * 32767).astype(np.int16)
            )
            processed += 1

        num_files = len(list(Path(output_dir).glob("*.wav")))
        print(f"✓ Downloaded {num_files} RIR files to {output_dir}/")

    except Exception as e:
        print(f"\n⚠ Could not download using datasets library: {e}")
        print("\nGenerating synthetic RIRs as fallback...")
        print("(These work well for augmentation, though real RIRs are preferred)")

        # Generate synthetic RIRs
        os.makedirs(output_dir, exist_ok=True)
        np.random.seed(42)
        num_synthetic = 50

        for i in tqdm(range(num_synthetic)):
            # Random decay time between 0.1 and 1.0 seconds
            decay_time = np.random.uniform(0.1, 1.0)
            sample_rate = 16000

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
            filename = os.path.join(output_dir, f"synthetic_rir_{i:03d}.wav")
            sf.write(filename, rir.astype(np.float32), sample_rate, subtype='PCM_16')

        num_files = len(list(Path(output_dir).glob("*.wav")))
        print(f"\n✓ Generated {num_files} synthetic RIR files in {output_dir}/")
