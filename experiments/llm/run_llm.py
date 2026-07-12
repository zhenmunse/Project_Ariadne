"""Stateless LLM experiment CLI with dry-run, filters, resume and mock mode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.llm.prepare_inputs import prepare_inputs
from experiments.llm.artifacts import load_json
from experiments.llm.harness import execute_run, select_runs
from experiments.llm.providers.closed_frontier import ClosedFrontierProvider
from experiments.llm.providers.mock import MockProvider
from experiments.llm.providers.open_weight import OpenWeightProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--provider", choices=("mock", "closed_frontier", "open_weight"))
    parser.add_argument("--only", choices=("pending",), default="pending")
    parser.add_argument("--model", choices=("closed_frontier", "open_weight"))
    parser.add_argument("--condition", choices=("zero", "full"))
    parser.add_argument("--target", type=int)
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--mock-fixture", default="valid")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "llm")
    return parser.parse_args()


def build_provider(name: str, config: dict, fixture: str):
    if name == "mock":
        return MockProvider(fixture=fixture)
    model = config["models"][name]
    cls = ClosedFrontierProvider if name == "closed_frontier" else OpenWeightProvider
    provider = cls(
        endpoint=model["endpoint"],
        requested_model_id=model["requested_model_id"],
        reasoning=model["reasoning"],
        api_key_env=model["api_key_env"],
    )
    provider.require_ready()
    return provider


def main() -> None:
    args = parse_args()
    if args.dry_run:
        result = prepare_inputs()
        print(json.dumps(result, sort_keys=True))
        print("network_access=false")
        print("formal_responses_created=0")
        return
    if args.provider is None:
        raise SystemExit("--provider is required unless --dry-run is used")
    config = load_json(ROOT / "experiments/llm/run_config.json")
    run_manifest = load_json(ROOT / "experiments/llm/generated/run_manifest.json")
    model_filter = args.model
    if args.provider in {"closed_frontier", "open_weight"}:
        if model_filter is not None and model_filter != args.provider:
            raise SystemExit("Formal provider and --model must match")
        model_filter = args.provider
    runs = select_runs(
        run_manifest["runs"],
        model=model_filter,
        condition=args.condition,
        target=args.target,
        run_id=args.run_id,
    )
    provider = build_provider(args.provider, config, args.mock_fixture)
    counts: dict[str, int] = {}
    max_attempts = int(config["request_policy"]["transport_retry"]["max_attempts"])
    for run in runs:
        result = execute_run(
            run,
            provider,
            output_root=args.output_root,
            force_rerun=args.force_rerun,
            max_transport_attempts=max_attempts,
        )
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    print(json.dumps({"selected_runs": len(runs), "statuses": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
