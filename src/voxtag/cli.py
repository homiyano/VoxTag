from __future__ import annotations

import argparse
import sys

from .core import VoxTag, VoxTagError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect WAV metadata and speech-readiness metrics.")
    parser.add_argument("file", help="Path to a WAV file")
    parser.add_argument("--analyze", action="store_true", help="Compute speech-oriented signal metrics")
    parser.add_argument("--frame-ms", type=int, default=30, help="Frame size for voice activity analysis")
    parser.add_argument(
        "--speech-threshold-db",
        type=float,
        default=-35.0,
        help="Frame RMS dBFS threshold used for energy-based voice activity",
    )
    parser.add_argument("--min-voice-ms", type=int, default=120, help="Minimum voiced segment length")
    parser.add_argument("--compact", action="store_true", help="Emit compact JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        tag = VoxTag.get(
            args.file,
            analyze=args.analyze,
            frame_ms=args.frame_ms,
            speech_threshold_db=args.speech_threshold_db,
            min_voice_ms=args.min_voice_ms,
        )
    except VoxTagError as exc:
        print(f"voxtag: {exc}", file=sys.stderr)
        return 1

    print(tag.to_json(indent=None if args.compact else 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
