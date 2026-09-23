#!/usr/bin/env python3
"""Run the real local audio pipeline and print inspectable segments."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.audio import process_audio


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path", type=Path, help="Path to an MP3 or WAV file")
    parser.add_argument("--limit", type=int, default=8, help="Segments to print (0 = all)")
    parser.add_argument("--output", type=Path, help="Optional JSON transcript path outside Git")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    started = time.perf_counter()
    segments = process_audio(str(args.audio_path))
    elapsed = time.perf_counter() - started

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps([segment.model_dump() for segment in segments], ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )

    speakers = sorted({segment.speaker_id for segment in segments})
    print(
        json.dumps(
            {
                "audio": str(args.audio_path.resolve()),
                "elapsed_seconds": round(elapsed, 2),
                "segment_count": len(segments),
                "speakers": speakers,
            },
            ensure_ascii=False,
        )
    )
    selected = segments if args.limit == 0 else segments[: args.limit]
    for segment in selected:
        print(json.dumps(segment.model_dump(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
