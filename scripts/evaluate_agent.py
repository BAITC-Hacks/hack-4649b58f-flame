#!/usr/bin/env python3
"""Run the local meeting agent and compare it with a case expectation file."""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents import MeetingProtocolAgent
from app.models.schemas import TranscriptSegment


def _normalise(value: str) -> str:
    return " ".join(re.findall(r"[a-zа-яё0-9]+", value.lower()))


def _keyword_matches(keyword: str, task: str) -> bool:
    task_tokens = _normalise(task).split()
    keyword_tokens = _normalise(keyword).split()
    return all(
        any(
            expected == actual
            or (
                len(expected) >= 4
                and expected[: min(5, len(expected))]
                == actual[: min(5, len(expected))]
            )
            for actual in task_tokens
        )
        for expected in keyword_tokens
    )


def _matches(expected: dict, actual: dict) -> bool:
    task = _normalise(actual["task"])
    assignee = _normalise(actual.get("assignee") or "")
    deadline = _normalise(actual.get("deadline_text") or "")
    task_ok = all(_keyword_matches(keyword, task) for keyword in expected["task_keywords"])
    expected_assignee_tokens = [
        token for token in _normalise(expected["assignee"]).split() if len(token) >= 4
    ]
    actual_assignee_tokens = [token for token in assignee.split() if len(token) >= 4]
    assignee_ok = any(
        expected_token in actual_token
        or actual_token in expected_token
        or SequenceMatcher(None, expected_token, actual_token).ratio() >= 0.82
        for expected_token in expected_assignee_tokens
        for actual_token in actual_assignee_tokens
    )
    deadline_keywords = expected.get("deadline_keywords") or []
    deadline_ok = not deadline_keywords or any(
        _normalise(keyword) in deadline for keyword in deadline_keywords
    )
    return task_ok and assignee_ok and deadline_ok


def _assignee_matches(expected: dict, actual: dict) -> bool:
    probe = {
        "task": " ".join(expected["task_keywords"]),
        "assignee": actual.get("assignee"),
        "deadline_text": " ".join(expected.get("deadline_keywords") or []),
    }
    return _matches(expected, probe)


def _matches_one_or_group(expected: dict, actual: list[dict]) -> bool:
    if any(_matches(expected, candidate) for candidate in actual):
        return True
    same_assignee = [
        candidate for candidate in actual if _assignee_matches(expected, candidate)
    ]
    if len(same_assignee) < 2:
        return False
    combined = {
        "task": " ".join(item["task"] for item in same_assignee),
        "assignee": expected["assignee"],
        "deadline_text": " ".join(
            item.get("deadline_text") or "" for item in same_assignee
        ),
    }
    return _matches(expected, combined)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", type=Path)
    parser.add_argument("expected", type=Path)
    parser.add_argument("--meeting-date", type=date.fromisoformat, default=None)
    parser.add_argument("--title", default="Совещание")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300")),
        help="Local Ollama request timeout in seconds (default: 300).",
    )
    args = parser.parse_args()

    transcript = [
        TranscriptSegment.model_validate(item)
        for item in json.loads(args.transcript.read_text(encoding="utf-8"))
    ]
    expectation = json.loads(args.expected.read_text(encoding="utf-8"))
    result = MeetingProtocolAgent(timeout=args.timeout).run(
        transcript, args.meeting_date, args.title
    )
    actual = [item.model_dump(mode="json") for item in result.action_items]

    checks = []
    for expected in expectation["expected_action_items"]:
        matched = _matches_one_or_group(expected, actual)
        checks.append(
            {
                "task_keywords": expected["task_keywords"],
                "assignee": expected["assignee"],
                "optional": bool(expected.get("optional", False)),
                "matched": matched,
            }
        )
    required = [check for check in checks if not check["optional"]]
    matched_required = sum(check["matched"] for check in required)
    metrics = {
        "required_expected": len(required),
        "required_matched": matched_required,
        "required_recall": round(matched_required / len(required), 3)
        if required
        else 1.0,
        "actual_action_items": len(actual),
    }
    report = {
        "metrics": metrics,
        "checks": checks,
        "result": result.model_dump(mode="json"),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0 if matched_required == len(required) else 1


if __name__ == "__main__":
    raise SystemExit(main())
