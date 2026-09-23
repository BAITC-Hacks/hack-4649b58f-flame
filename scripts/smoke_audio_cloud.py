#!/usr/bin/env python3
"""Explicitly upload an MP3/WAV to OpenAI for an optional audio MVP check."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.audio_cloud import process_audio_cloud


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path", type=Path)
    parser.add_argument(
        "--allow-cloud-upload",
        action="store_true",
        help="Explicitly send the supplied audio to OpenAI",
    )
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--output", type=Path, help="Optional JSON transcript path outside Git")
    args = parser.parse_args()
    if not args.allow_cloud_upload:
        parser.error("pass --allow-cloud-upload to send the audio to OpenAI")

    started = time.perf_counter()
    segments = process_audio_cloud(str(args.audio_path), allow_upload=True)
    elapsed = time.perf_counter() - started
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps([item.model_dump() for item in segments], ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "provider": "openai",
                "elapsed_seconds": round(elapsed, 2),
                "segment_count": len(segments),
                "speakers": sorted({item.speaker_id for item in segments}),
            },
            ensure_ascii=False,
        )
    )
    for item in segments if args.limit == 0 else segments[: args.limit]:
        print(json.dumps(item.model_dump(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
