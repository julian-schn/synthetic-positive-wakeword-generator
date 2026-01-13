"""Kokoro TTS wrapper with voice blending support."""

import numpy as np
from kokoro import KPipeline


# Default voice pool for variability (FR-2)
DEFAULT_VOICES = [
    "af_bella", "af_nicole", "af_sarah", "af_sky",
    "am_adam", "am_michael", "bf_emma", "bf_isabella",
    "bm_george", "bm_lewis"
]

KOKORO_SAMPLE_RATE = 24000


class TTSEngine:
    """Kokoro TTS engine with voice blending."""
    
    def __init__(self, lang_code: str = "a"):
        """Initialize the Kokoro pipeline.
        
        Args:
            lang_code: Language code ('a' for American English)
        """
        self.pipeline = KPipeline(lang_code=lang_code)
        self._voice_cache: dict[str, np.ndarray] = {}
    
    def _get_voice(self, voice_name: str) -> np.ndarray:
        """Get or cache a voice vector."""
        if voice_name not in self._voice_cache:
            self._voice_cache[voice_name] = self.pipeline.load_voice(voice_name)
        return self._voice_cache[voice_name]
    
    def blend_voices(
        self,
        voice_a: str,
        voice_b: str,
        ratio: float = 0.5
    ) -> np.ndarray:
        """Blend two voice vectors.
        
        Args:
            voice_a: First voice name
            voice_b: Second voice name
            ratio: Blend ratio (0.0 = all A, 1.0 = all B)
        
        Returns:
            Blended voice vector
        """
        vec_a = self._get_voice(voice_a)
        vec_b = self._get_voice(voice_b)
        return vec_a * (1 - ratio) + vec_b * ratio
    
    def synthesize(
        self,
        text: str,
        voice: str | np.ndarray,
        speed: float = 1.0
    ) -> np.ndarray:
        """Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            voice: Voice name or blended voice vector
            speed: Speech speed multiplier
        
        Returns:
            Audio array at 24kHz
        """
        if isinstance(voice, str):
            voice = self._get_voice(voice)
        
        # Generate audio using Kokoro pipeline
        audio_segments = []
        for _, _, audio in self.pipeline(text, voice=voice, speed=speed):
            audio_segments.append(audio)
        
        if not audio_segments:
            raise ValueError(f"No audio generated for text: {text}")
        
        return np.concatenate(audio_segments)
    
    def get_random_voice_config(
        self,
        rng: np.random.Generator,
        voice_pool: list[str] | None = None,
        blend_probability: float = 0.5
    ) -> tuple[str | np.ndarray, dict]:
        """Get a random voice configuration.
        
        Args:
            rng: Random number generator
            voice_pool: List of voice names (defaults to DEFAULT_VOICES)
            blend_probability: Probability of blending two voices
        
        Returns:
            Tuple of (voice, metadata dict)
        """
        if voice_pool is None:
            voice_pool = DEFAULT_VOICES
        
        if rng.random() < blend_probability and len(voice_pool) >= 2:
            # Blend two random voices
            voices = rng.choice(voice_pool, size=2, replace=False)
            ratio = rng.uniform(0.3, 0.7)
            voice = self.blend_voices(voices[0], voices[1], ratio)
            metadata = {
                "voice_type": "blend",
                "voice_a": voices[0],
                "voice_b": voices[1],
                "blend_ratio": round(ratio, 3)
            }
        else:
            # Single voice
            voice_name = rng.choice(voice_pool)
            voice = voice_name
            metadata = {
                "voice_type": "single",
                "voice": voice_name
            }
        
        return voice, metadata
