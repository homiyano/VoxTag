# VoxTag

VoxTag is a small Python library for WAV metadata and speech-readiness analysis. It borrows the simple `TinyTag.get(...)` shape, but focuses on speech pipelines: clean duration and PCM properties, RIFF INFO metadata, loudness, peak, DC offset, zero-crossing rate, clipping, silence ratio, and simple energy-based voice segments.

The first version is intentionally zero-dependency and WAV-first. That keeps it easy to embed in transcription, data cleanup, call QA, and dataset triage scripts.

## Install

```bash
python3 -m pip install -e .
```

## Python Usage

```python
from voxtag import VoxTag

tag = VoxTag.get("sample.wav", analyze=True)

print(tag.info.duration)
print(tag.info.samplerate)
print(tag.metrics.speech_ratio)
print(tag.metrics.voice_segments)
```

## CLI

```bash
voxtag sample.wav
voxtag sample.wav --analyze
voxtag sample.wav --analyze --speech-threshold-db -40 --frame-ms 20
```

## Current Options

- TinyTag-like `VoxTag.get(path)` API
- File-like object support
- JSON output through `as_dict()` and `to_json()`
- WAV metadata: filename, filesize, duration, channels, sample rate, sample width, bitrate, compression type, RIFF chunks, and INFO tags
- Speech metrics: RMS, RMS dBFS, peak, peak dBFS, DC offset, zero-crossing rate, clipping ratio, silence ratio, speech ratio, and voice segments
- Tunable voice activity: `frame_ms`, `speech_threshold_db`, and `min_voice_ms`

## Roadmap

- RF64/WAVE64 support for long recordings
- Broadcast WAV metadata
- Optional NumPy backend for faster large-file analysis
- Windowed features for ASR dataset prep: SNR estimates, spectral centroid, rolloff, pauses, and speaking-rate helpers
- Optional integrations for transcript alignment and diarization metadata
- MP3, FLAC, OGG, and M4A metadata readers with the same API

## Why Not Just `wave`?

Python's `wave` module is excellent for reading and writing WAV files, but it stops at format parameters and frames. VoxTag uses it as the safe baseline, then adds metadata flattening and speech-oriented signal summaries for automation.
