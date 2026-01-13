"""Audio augmentation chain following SPEC.md FR-4."""

import os
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from .audio_utils import apply_gain, apply_lowpass, apply_bandpass, pitch_shift


class AugmentationChain:
    """Applies augmentations in physically plausible order."""
    
    def __init__(
        self,
        config: dict,
        rir_dir: Path | None = None,
        noise_dir: Path | None = None
    ):
        """Initialize augmentation chain.
        
        Args:
            config: Augmentation configuration dict
            rir_dir: Directory containing room impulse responses
            noise_dir: Directory containing background noise files
        """
        self.config = config
        self.rir_files = self._load_file_list(rir_dir, [".wav", ".flac"])
        self.noise_files = self._load_file_list(noise_dir, [".wav", ".flac"])
    
    def _load_file_list(
        self,
        directory: Path | None,
        extensions: list[str]
    ) -> list[Path]:
        """Load list of audio files from directory."""
        if directory is None or not directory.exists():
            return []
        
        files = []
        for ext in extensions:
            files.extend(directory.glob(f"*{ext}"))
        return sorted(files)
    
    def apply(
        self,
        audio: np.ndarray,
        sr: int,
        rng: np.random.Generator
    ) -> tuple[np.ndarray, dict]:
        """Apply augmentation chain to audio.
        
        Order (FR-4):
        1. Gain / Distance (Low Pass)
        2. Room Impulse Response (Reverb)
        3. Noise Mix
        
        Args:
            audio: Input audio array
            sr: Sample rate
            rng: Random number generator
        
        Returns:
            Tuple of (augmented audio, metadata dict)
        """
        metadata = {}
        
        # 0. Pitch shift (applied first, before other processing)
        pitch_cfg = self.config.get("pitch", {})
        if rng.random() < pitch_cfg.get("probability", 0.3):
            semitone_range = pitch_cfg.get("semitone_range", [-2, 2])
            semitones = rng.uniform(*semitone_range)
            audio = pitch_shift(audio, sr, semitones)
            metadata["pitch_semitones"] = round(semitones, 2)
        
        # 1. Gain adjustment
        gain_cfg = self.config.get("gain", {})
        db_range = gain_cfg.get("db_range", [-6, 3])
        gain_db = rng.uniform(*db_range)
        audio = apply_gain(audio, gain_db)
        metadata["gain_db"] = round(gain_db, 2)
        
        # 1b. Low-pass filter (distance simulation)
        lowpass_cfg = self.config.get("lowpass", {})
        if rng.random() < lowpass_cfg.get("probability", 0.3):
            cutoff_range = lowpass_cfg.get("cutoff_range", [3000, 6000])
            cutoff = rng.uniform(*cutoff_range)
            audio = apply_lowpass(audio, sr, cutoff)
            metadata["lowpass_cutoff_hz"] = round(cutoff, 0)
        
        # 1c. Band-pass filter (telephone/radio simulation)
        bandpass_cfg = self.config.get("bandpass", {})
        if rng.random() < bandpass_cfg.get("probability", 0.1):
            low_hz = bandpass_cfg.get("low_hz", 300)
            high_hz = bandpass_cfg.get("high_hz", 3400)
            audio = apply_bandpass(audio, sr, low_hz, high_hz)
            metadata["bandpass"] = f"{low_hz}-{high_hz}Hz"
        
        # 2. Room reverb (RIR convolution)
        reverb_cfg = self.config.get("reverb", {})
        if rng.random() < reverb_cfg.get("probability", 0.7):
            if self.rir_files:
                # Use real RIR file
                rir_file = rng.choice(self.rir_files)
                audio, rir_meta = self._apply_rir(audio, sr, rir_file)
                metadata.update(rir_meta)
            else:
                # Generate synthetic RIR (exponential decay)
                decay_time = rng.uniform(0.1, 0.5)
                rir = self._create_synthetic_rir(sr, decay_time, rng)
                audio = signal.fftconvolve(audio, rir, mode='full')[:len(audio)]
                metadata["reverb"] = f"synthetic_decay_{decay_time:.2f}s"
        
        # 3. Background noise mix
        noise_cfg = self.config.get("noise", {})
        if rng.random() < noise_cfg.get("probability", 0.8):
            snr_range = noise_cfg.get("snr_db_range", [15, 30])
            snr_db = rng.uniform(*snr_range)
            
            if self.noise_files:
                # Use real noise file
                noise_file = rng.choice(self.noise_files)
                audio, noise_meta = self._mix_noise(audio, sr, noise_file, snr_db, rng)
                metadata.update(noise_meta)
            else:
                # Generate synthetic noise (white noise)
                audio = self._mix_synthetic_noise(audio, snr_db, rng)
                metadata["noise"] = f"synthetic_white_snr_{snr_db:.1f}dB"
        
        return audio, metadata
    
    def _apply_rir(
        self,
        audio: np.ndarray,
        sr: int,
        rir_file: Path
    ) -> tuple[np.ndarray, dict]:
        """Apply room impulse response convolution."""
        rir, rir_sr = sf.read(rir_file)
        
        # Convert to mono if stereo
        if rir.ndim > 1:
            rir = rir.mean(axis=1)
        
        # Resample RIR if needed
        if rir_sr != sr:
            num_samples = int(len(rir) * sr / rir_sr)
            rir = signal.resample(rir, num_samples)
        
        # Normalize RIR
        rir = rir / np.abs(rir).max()
        
        # Convolve
        audio = signal.fftconvolve(audio, rir, mode='full')[:len(audio)]
        
        return audio, {"rir_file": rir_file.name}
    
    def _mix_noise(
        self,
        audio: np.ndarray,
        sr: int,
        noise_file: Path,
        snr_db: float,
        rng: np.random.Generator
    ) -> tuple[np.ndarray, dict]:
        """Mix background noise at specified SNR."""
        noise, noise_sr = sf.read(noise_file)
        
        # Convert to mono if stereo
        if noise.ndim > 1:
            noise = noise.mean(axis=1)
        
        # Resample noise if needed
        if noise_sr != sr:
            num_samples = int(len(noise) * sr / noise_sr)
            noise = signal.resample(noise, num_samples)
        
        # Loop or trim noise to match audio length
        if len(noise) < len(audio):
            repeats = int(np.ceil(len(audio) / len(noise)))
            noise = np.tile(noise, repeats)
        
        # Random offset into noise
        max_offset = len(noise) - len(audio)
        if max_offset > 0:
            offset = rng.integers(0, max_offset)
            noise = noise[offset:offset + len(audio)]
        else:
            noise = noise[:len(audio)]
        
        # Calculate scaling for target SNR
        audio_power = np.mean(audio ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power > 0:
            target_noise_power = audio_power / (10 ** (snr_db / 10))
            scale = np.sqrt(target_noise_power / noise_power)
            noise = noise * scale
        
        return audio + noise, {
            "noise_file": noise_file.name,
            "snr_db": round(snr_db, 1)
        }
    
    def _create_synthetic_rir(self, sr: int, decay_time: float, rng: np.random.Generator) -> np.ndarray:
        """Create synthetic room impulse response with exponential decay."""
        length = int(sr * decay_time)
        t = np.linspace(0, decay_time, length)
        # Exponential decay with some randomness
        rir = rng.standard_normal(length) * np.exp(-5 * t)
        rir[0] = 1.0  # Direct sound
        # Normalize
        rir = rir / np.abs(rir).max()
        return rir
    
    def _mix_synthetic_noise(
        self,
        audio: np.ndarray,
        snr_db: float,
        rng: np.random.Generator
    ) -> np.ndarray:
        """Mix synthetic white noise at specified SNR."""
        noise = rng.standard_normal(len(audio))
        
        # Calculate scaling for target SNR
        audio_power = np.mean(audio ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power > 0:
            target_noise_power = audio_power / (10 ** (snr_db / 10))
            scale = np.sqrt(target_noise_power / noise_power)
            noise = noise * scale
        
        return audio + noise