from __future__ import annotations

import dataclasses
import io
import json
import math
import os
import struct
import wave
from pathlib import Path
from typing import BinaryIO, Iterable


class VoxTagError(Exception):
    """Raised when audio cannot be inspected by VoxTag."""


@dataclasses.dataclass(frozen=True)
class WavInfo:
    filename: str | None
    filesize: int | None
    duration: float
    channels: int
    samplerate: int
    sample_width: int
    frames: int
    comptype: str
    compname: str
    bitrate: int
    chunks: tuple[str, ...]
    metadata: dict[str, str]

    @property
    def bit_depth(self) -> int:
        return self.sample_width * 8


@dataclasses.dataclass(frozen=True)
class VoiceSegment:
    start: float
    end: float
    duration: float


@dataclasses.dataclass(frozen=True)
class SpeechMetrics:
    rms: float
    rms_dbfs: float | None
    peak: float
    peak_dbfs: float | None
    dc_offset: float
    zero_crossing_rate: float
    clipping_ratio: float
    silence_ratio: float
    speech_ratio: float
    voice_segments: tuple[VoiceSegment, ...]


class VoxTag:
    """TinyTag-style entry point for WAV metadata plus speech-oriented metrics."""

    def __init__(self, info: WavInfo, metrics: SpeechMetrics | None = None):
        self.info = info
        self.metrics = metrics

    @classmethod
    def get(
        cls,
        file: str | os.PathLike[str] | BinaryIO,
        *,
        analyze: bool = False,
        frame_ms: int = 30,
        speech_threshold_db: float = -35.0,
        min_voice_ms: int = 120,
    ) -> "VoxTag":
        """Read WAV metadata and, optionally, compute speech-readiness metrics."""

        source = _Source.from_file(file)
        try:
            info, pcm = _read_wav(source, keep_pcm=analyze)
            metrics = None
            if analyze:
                metrics = _analyze_pcm(
                    pcm,
                    channels=info.channels,
                    sample_width=info.sample_width,
                    samplerate=info.samplerate,
                    frame_ms=frame_ms,
                    speech_threshold_db=speech_threshold_db,
                    min_voice_ms=min_voice_ms,
                )
            return cls(info, metrics)
        finally:
            source.close_if_owned()

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = dataclasses.asdict(self.info)
        if self.metrics is not None:
            payload["metrics"] = dataclasses.asdict(self.metrics)
        return payload

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, sort_keys=True)


@dataclasses.dataclass
class _Source:
    fp: BinaryIO
    filename: str | None
    filesize: int | None
    owned: bool = False

    @classmethod
    def from_file(cls, file: str | os.PathLike[str] | BinaryIO) -> "_Source":
        if isinstance(file, (str, os.PathLike)):
            path = Path(file)
            return cls(path.open("rb"), str(path), path.stat().st_size, True)

        filename = getattr(file, "name", None)
        filesize = None
        if isinstance(filename, str) and os.path.exists(filename):
            filesize = os.path.getsize(filename)
        return cls(file, filename, filesize, False)

    def close_if_owned(self) -> None:
        if self.owned:
            self.fp.close()


def _read_wav(source: _Source, *, keep_pcm: bool) -> tuple[WavInfo, bytes]:
    try:
        source.fp.seek(0)
        chunks, metadata = _scan_riff(source.fp)
        source.fp.seek(0)
        with wave.open(source.fp, "rb") as reader:
            channels = reader.getnchannels()
            sample_width = reader.getsampwidth()
            samplerate = reader.getframerate()
            frames = reader.getnframes()
            comptype = reader.getcomptype()
            compname = reader.getcompname()
            pcm = reader.readframes(frames) if keep_pcm else b""
    except (EOFError, OSError, wave.Error, struct.error) as exc:
        raise VoxTagError(f"Could not read WAV file: {exc}") from exc

    duration = frames / samplerate if samplerate else 0.0
    bitrate = samplerate * channels * sample_width * 8
    info = WavInfo(
        filename=source.filename,
        filesize=source.filesize,
        duration=duration,
        channels=channels,
        samplerate=samplerate,
        sample_width=sample_width,
        frames=frames,
        comptype=comptype,
        compname=compname,
        bitrate=bitrate,
        chunks=tuple(chunks),
        metadata=metadata,
    )
    return info, pcm


def _scan_riff(fp: BinaryIO) -> tuple[list[str], dict[str, str]]:
    fp.seek(0)
    header = fp.read(12)
    if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise VoxTagError("Not a RIFF/WAVE file")

    chunks: list[str] = []
    metadata: dict[str, str] = {}
    while True:
        chunk_header = fp.read(8)
        if len(chunk_header) == 0:
            break
        if len(chunk_header) < 8:
            raise VoxTagError("Truncated WAV chunk header")

        raw_id, size = struct.unpack("<4sI", chunk_header)
        chunk_id = raw_id.decode("ascii", errors="replace")
        chunks.append(chunk_id)
        data = fp.read(size)
        if len(data) < size:
            raise VoxTagError(f"Truncated WAV chunk {chunk_id!r}")
        if size % 2:
            fp.seek(1, io.SEEK_CUR)

        if chunk_id == "LIST" and data[:4] == b"INFO":
            metadata.update(_parse_info_list(data[4:]))

    return chunks, metadata


def _parse_info_list(data: bytes) -> dict[str, str]:
    labels = {
        "IART": "artist",
        "ICMT": "comment",
        "ICOP": "copyright",
        "ICRD": "date",
        "IGNR": "genre",
        "INAM": "title",
        "IPRD": "product",
        "ISFT": "software",
        "ISRC": "source",
        "ITCH": "technician",
    }
    pos = 0
    metadata: dict[str, str] = {}
    while pos + 8 <= len(data):
        raw_id, size = struct.unpack("<4sI", data[pos : pos + 8])
        pos += 8
        value = data[pos : pos + size]
        pos += size + (size % 2)
        key = raw_id.decode("ascii", errors="replace")
        text = value.rstrip(b"\x00").decode("utf-8", errors="replace").strip()
        if text:
            metadata[labels.get(key, key)] = text
    return metadata


def _analyze_pcm(
    pcm: bytes,
    *,
    channels: int,
    sample_width: int,
    samplerate: int,
    frame_ms: int,
    speech_threshold_db: float,
    min_voice_ms: int,
) -> SpeechMetrics:
    if sample_width not in (1, 2, 3, 4):
        raise VoxTagError(f"Unsupported sample width: {sample_width}")
    if not pcm or not samplerate or not channels:
        return SpeechMetrics(0.0, None, 0.0, None, 0.0, 0.0, 0.0, 1.0, 0.0, ())

    mono_samples = _to_mono_samples(pcm, sample_width, channels)
    max_amp = float(1 << (sample_width * 8 - 1))
    sample_count = max(1, len(mono_samples))
    square_sum = sum(sample * sample for sample in mono_samples)
    rms_raw = math.sqrt(square_sum / sample_count)
    peak_raw = max(abs(sample) for sample in mono_samples)
    avg_raw = sum(mono_samples) / sample_count
    crossings = _zero_crossings(mono_samples)

    rms = rms_raw / max_amp
    peak = peak_raw / max_amp
    dc_offset = avg_raw / max_amp
    zcr = crossings / sample_count
    clipping_ratio = _clipping_ratio(mono_samples, sample_width, max_amp)
    rms_dbfs = _dbfs(rms)
    peak_dbfs = _dbfs(peak)
    silence_ratio, voice_segments = _voice_activity(
        mono_samples,
        samplerate=samplerate,
        max_amp=max_amp,
        frame_ms=frame_ms,
        threshold_db=speech_threshold_db,
        min_voice_ms=min_voice_ms,
    )
    speech_ratio = 1.0 - silence_ratio

    return SpeechMetrics(
        rms=rms,
        rms_dbfs=rms_dbfs,
        peak=peak,
        peak_dbfs=peak_dbfs,
        dc_offset=dc_offset,
        zero_crossing_rate=zcr,
        clipping_ratio=clipping_ratio,
        silence_ratio=silence_ratio,
        speech_ratio=speech_ratio,
        voice_segments=tuple(voice_segments),
    )


def _dbfs(value: float) -> float | None:
    if value <= 0:
        return None
    return 20.0 * math.log10(value)


def _clipping_ratio(samples: list[int], _sample_width: int, max_amp: float) -> float:
    threshold = max_amp * 0.999
    clipped = sum(1 for sample in samples if abs(sample) >= threshold)
    total = len(samples)
    return clipped / total if total else 0.0


def _voice_activity(
    samples: list[int],
    *,
    samplerate: int,
    max_amp: float,
    frame_ms: int,
    threshold_db: float,
    min_voice_ms: int,
) -> tuple[float, list[VoiceSegment]]:
    frame_samples = max(1, samplerate * frame_ms // 1000)
    min_frames = max(1, math.ceil(min_voice_ms / frame_ms))
    frame_count = math.ceil(len(samples) / frame_samples)
    if frame_count <= 0:
        return 1.0, []

    voiced: list[bool] = []
    for offset in range(0, len(samples), frame_samples):
        frame = samples[offset : offset + frame_samples]
        square_sum = sum(sample * sample for sample in frame)
        rms = math.sqrt(square_sum / len(frame)) / max_amp if frame else 0.0
        db = _dbfs(rms)
        voiced.append(db is not None and db >= threshold_db)

    segments: list[VoiceSegment] = []
    start: int | None = None
    for index, is_voiced in enumerate([*voiced, False]):
        if is_voiced and start is None:
            start = index
        elif not is_voiced and start is not None:
            if index - start >= min_frames:
                start_time = start * frame_ms / 1000
                end_time = min(index * frame_ms / 1000, len(samples) / samplerate)
                segments.append(VoiceSegment(start_time, end_time, end_time - start_time))
            start = None

    voiced_frames = sum(voiced)
    silence_ratio = 1.0 - (voiced_frames / len(voiced))
    return silence_ratio, segments


def _to_mono_samples(data: bytes, sample_width: int, channels: int) -> list[int]:
    samples = list(_iter_samples(data, sample_width))
    if channels <= 1:
        return samples

    mono: list[int] = []
    for index in range(0, len(samples) - channels + 1, channels):
        mono.append(round(sum(samples[index : index + channels]) / channels))
    return mono


def _zero_crossings(samples: list[int]) -> int:
    crossings = 0
    previous = 0
    for sample in samples:
        sign = 1 if sample > 0 else -1 if sample < 0 else previous
        if previous and sign and sign != previous:
            crossings += 1
        previous = sign
    return crossings


def _iter_samples(data: bytes, sample_width: int) -> Iterable[int]:
    if sample_width == 1:
        for byte in data:
            yield byte - 128
    elif sample_width == 2:
        for index in range(0, len(data) - 1, 2):
            yield int.from_bytes(data[index : index + 2], "little", signed=True)
    elif sample_width == 3:
        for index in range(0, len(data) - 2, 3):
            chunk = data[index : index + 3]
            yield int.from_bytes(chunk + (b"\xff" if chunk[2] & 0x80 else b"\x00"), "little", signed=True)
    elif sample_width == 4:
        for index in range(0, len(data) - 3, 4):
            yield int.from_bytes(data[index : index + 4], "little", signed=True)
