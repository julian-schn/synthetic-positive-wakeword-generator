"""Audio utility functions for resampling, format conversion, and padding."""

import numpy as np
from scipy import signal


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate.
    
    Args:
        audio: Input audio array
        orig_sr: Original sample rate (e.g., 24000 for Kokoro)
        target_sr: Target sample rate (e.g., 16000 for openWakeWord)
    
    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio
    
    num_samples = int(len(audio) * target_sr / orig_sr)
    return signal.resample(audio, num_samples)


def to_pcm16(audio: np.ndarray) -> np.ndarray:
    """Convert float audio to 16-bit PCM.
    
    Args:
        audio: Float audio array (expected range [-1, 1])
    
    Returns:
        16-bit integer audio array
    """
    # Normalize to prevent clipping
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.95
    
    # Convert to 16-bit
    return (audio * 32767).astype(np.int16)


def add_silence_padding(
    audio: np.ndarray,
    sr: int,
    leading_range: tuple[float, float],
    trailing_range: tuple[float, float],
    rng: np.random.Generator
) -> np.ndarray:
    """Add random leading and trailing silence.
    
    Args:
        audio: Input audio array
        sr: Sample rate
        leading_range: (min, max) seconds for leading silence
        trailing_range: (min, max) seconds for trailing silence
        rng: Random number generator for reproducibility
    
    Returns:
        Padded audio array
    """
    leading_s = rng.uniform(*leading_range)
    trailing_s = rng.uniform(*trailing_range)
    
    leading_samples = int(leading_s * sr)
    trailing_samples = int(trailing_s * sr)
    
    return np.concatenate([
        np.zeros(leading_samples, dtype=audio.dtype),
        audio,
        np.zeros(trailing_samples, dtype=audio.dtype)
    ])


def apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    """Apply gain in decibels.
    
    Args:
        audio: Input audio array
        gain_db: Gain to apply in dB
    
    Returns:
        Audio with gain applied
    
    Note: IPython Audio widget normalizes playback, so gain differences
    may not be audible when listening, but the actual audio data is modified.
    """
    gain_linear = 10 ** (gain_db / 20)
    return audio * gain_linear


def apply_lowpass(
    audio: np.ndarray,
    sr: int,
    cutoff_hz: float,
    order: int = 4
) -> np.ndarray:
    """Apply low-pass filter to simulate distance.
    
    Args:
        audio: Input audio array
        sr: Sample rate
        cutoff_hz: Cutoff frequency in Hz
        order: Filter order
    
    Returns:
        Filtered audio
    """
    nyquist = sr / 2
    normalized_cutoff = cutoff_hz / nyquist
    b, a = signal.butter(order, normalized_cutoff, btype='low')
    return signal.filtfilt(b, a, audio)


def pitch_shift(
    audio: np.ndarray,
    sr: int,
    semitones: float
) -> np.ndarray:
    """Shift pitch using phase vocoder (preserves duration).
    
    Args:
        audio: Input audio array
        sr: Sample rate
        semitones: Pitch shift in semitones (positive = higher, negative = lower)
    
    Returns:
        Pitch-shifted audio (same length as input)
    """
    try:
        import librosa
        # Use librosa's pitch_shift which properly preserves duration
        return librosa.effects.pitch_shift(
            y=audio,
            sr=sr,
            n_steps=semitones,
            bins_per_octave=12
        )
    except ImportError:
        # Fallback: simple resampling approach (less accurate)
        # This is a workaround - librosa should be preferred
        factor = 2 ** (semitones / 12)
        # Resample to shift pitch (changes duration)
        shifted_length = int(len(audio) / factor)
        shifted = signal.resample(audio, shifted_length)
        # Time-stretch back to original length using linear interpolation
        indices = np.linspace(0, len(shifted) - 1, len(audio))
        return np.interp(indices, np.arange(len(shifted)), shifted)


def time_stretch(
    audio: np.ndarray,
    factor: float
) -> np.ndarray:
    """Time-stretch audio by resampling.
    
    Args:
        audio: Input audio array
        factor: Stretch factor (>1 = slower/longer, <1 = faster/shorter)
    
    Returns:
        Time-stretched audio
    """
    new_length = int(len(audio) * factor)
    return signal.resample(audio, new_length)


def apply_highpass(
    audio: np.ndarray,
    sr: int,
    cutoff_hz: float,
    order: int = 4
) -> np.ndarray:
    """Apply high-pass filter to remove low frequencies.
    
    Args:
        audio: Input audio array
        sr: Sample rate
        cutoff_hz: Cutoff frequency in Hz
        order: Filter order
    
    Returns:
        Filtered audio
    """
    nyquist = sr / 2
    normalized_cutoff = cutoff_hz / nyquist
    b, a = signal.butter(order, normalized_cutoff, btype='high')
    return signal.filtfilt(b, a, audio)


def apply_bandpass(
    audio: np.ndarray,
    sr: int,
    low_hz: float,
    high_hz: float,
    order: int = 4
) -> np.ndarray:
    """Apply band-pass filter (simulates telephone/radio).
    
    Args:
        audio: Input audio array
        sr: Sample rate
        low_hz: Low cutoff frequency in Hz
        high_hz: High cutoff frequency in Hz
        order: Filter order
    
    Returns:
        Filtered audio
    """
    nyquist = sr / 2
    low = low_hz / nyquist
    high = high_hz / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, audio)
