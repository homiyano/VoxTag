"""Small WAV metadata and speech-readiness analysis."""

from .core import SpeechMetrics, VoiceSegment, VoxTag, VoxTagError, WavInfo

__all__ = [
    "SpeechMetrics",
    "VoiceSegment",
    "VoxTag",
    "VoxTagError",
    "WavInfo",
]

__version__ = "0.1.0"
