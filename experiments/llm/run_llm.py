"""Task 16 entry point; only deterministic dry-run is enabled in checkpoint 16-1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.llm.prepare_inputs import prepare_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        raise SystemExit("Task 16-1 permits --dry-run only; provider execution belongs to 16-2/17")
    result = prepare_inputs()
    print(json.dumps(result, sort_keys=True))
    print("network_access=false")
    print("formal_responses_created=0")


if __name__ == "__main__":
    main()
