from __future__ import annotations

import math
import struct
import tempfile
import unittest
import wave
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from voxtag import VoxTag
from voxtag.cli import main


def write_wav(path: Path, samples: list[int], *, samplerate: int = 8000) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(samplerate)
        writer.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


class VoxTagTests(unittest.TestCase):
    def test_reads_wav_metadata_without_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tone.wav"
            write_wav(path, [0] * 8000)

            tag = VoxTag.get(path)

        self.assertEqual(tag.info.channels, 1)
        self.assertEqual(tag.info.samplerate, 8000)
        self.assertEqual(tag.info.sample_width, 2)
        self.assertAlmostEqual(tag.info.duration, 1.0)
        self.assertIn("fmt ", tag.info.chunks)
        self.assertIsNone(tag.metrics)

    def test_computes_speech_metrics(self) -> None:
        samplerate = 8000
        silence = [0] * int(samplerate * 0.2)
        voiced = [
            int(10000 * math.sin(2 * math.pi * 220 * index / samplerate))
            for index in range(int(samplerate * 0.4))
        ]
        samples = silence + voiced + silence
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "speech.wav"
            write_wav(path, samples, samplerate=samplerate)

            tag = VoxTag.get(path, analyze=True, speech_threshold_db=-35, min_voice_ms=60)

        self.assertIsNotNone(tag.metrics)
        assert tag.metrics is not None
        self.assertGreater(tag.metrics.rms, 0)
        self.assertGreater(tag.metrics.peak, 0.2)
        self.assertGreater(tag.metrics.speech_ratio, 0.3)
        self.assertLess(tag.metrics.speech_ratio, 0.7)
        self.assertEqual(len(tag.metrics.voice_segments), 1)

    def test_cli_returns_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tone.wav"
            write_wav(path, [0] * 800)

            stdout = StringIO()
            with redirect_stdout(stdout):
                code = main([str(path), "--compact"])

        self.assertEqual(code, 0)
        self.assertIn('"duration": 0.1', stdout.getvalue())

if __name__ == "__main__":
    unittest.main()
