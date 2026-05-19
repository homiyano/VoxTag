from __future__ import annotations

import argparse
import math
import random
import struct
import wave
from pathlib import Path


SAMPLE_RATE = 16_000
MAX_INT16 = 32_767


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic WAV files for VoxTag analysis.")
    parser.add_argument(
        "--out",
        default="examples/audio",
        help="Directory where sample WAV files will be written.",
    )
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = {
        "clean_speech_like.wav": clean_speech_like(),
        "noisy_room.wav": with_white_noise(clean_speech_like(), amplitude=2_300, seed=7),
        "hum_50hz.wav": with_hum(clean_speech_like(), frequency=50.0, amplitude=2_800),
        "clipped_loud.wav": clipped_loud(),
        "mostly_silence.wav": mostly_silence(),
    }

    for filename, pcm in samples.items():
        write_wav(output_dir / filename, pcm)
        print(output_dir / filename)

    return 0


def clean_speech_like() -> list[int]:
    samples: list[int] = []
    samples += silence(0.35)
    samples += phrase(0.55, base_hz=145.0, amplitude=8_800)
    samples += silence(0.18)
    samples += phrase(0.45, base_hz=190.0, amplitude=7_400)
    samples += silence(0.28)
    samples += phrase(0.65, base_hz=125.0, amplitude=9_200)
    samples += silence(0.5)
    return samples


def clipped_loud() -> list[int]:
    clean = clean_speech_like()
    return [clip(round(sample * 4.8)) for sample in clean]


def mostly_silence() -> list[int]:
    samples: list[int] = []
    samples += silence(1.0)
    samples += phrase(0.18, base_hz=170.0, amplitude=6_000)
    samples += silence(1.2)
    return samples


def silence(seconds: float) -> list[int]:
    return [0] * int(SAMPLE_RATE * seconds)


def phrase(seconds: float, *, base_hz: float, amplitude: int) -> list[int]:
    total = int(SAMPLE_RATE * seconds)
    samples: list[int] = []
    for index in range(total):
        t = index / SAMPLE_RATE
        envelope = _speech_envelope(index, total)
        vibrato = 1.0 + 0.025 * math.sin(2.0 * math.pi * 5.2 * t)
        f0 = base_hz * vibrato
        sample = (
            math.sin(2.0 * math.pi * f0 * t)
            + 0.45 * math.sin(2.0 * math.pi * f0 * 2.0 * t)
            + 0.2 * math.sin(2.0 * math.pi * f0 * 3.0 * t)
        )
        samples.append(clip(round(sample * amplitude * envelope)))
    return samples


def with_white_noise(samples: list[int], *, amplitude: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [clip(sample + rng.randint(-amplitude, amplitude)) for sample in samples]


def with_hum(samples: list[int], *, frequency: float, amplitude: int) -> list[int]:
    output: list[int] = []
    for index, sample in enumerate(samples):
        t = index / SAMPLE_RATE
        hum = math.sin(2.0 * math.pi * frequency * t) * amplitude
        output.append(clip(round(sample + hum)))
    return output


def write_wav(path: Path, samples: list[int]) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


def clip(sample: int) -> int:
    return max(-MAX_INT16, min(MAX_INT16, sample))


def _speech_envelope(index: int, total: int) -> float:
    attack = max(1, int(total * 0.08))
    release = max(1, int(total * 0.12))
    if index < attack:
        return index / attack
    if index > total - release:
        return max(0.0, (total - index) / release)
    return 0.65 + 0.35 * math.sin(2.0 * math.pi * 7.0 * index / SAMPLE_RATE) ** 2


if __name__ == "__main__":
    raise SystemExit(main())
