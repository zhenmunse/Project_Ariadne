"""Stateless LLM experiment CLI with dry-run, filters, resume and mock mode."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.llm.artifacts import load_json
from experiments.llm.harness import execute_run, select_runs
from experiments.llm.providers.factory import build_provider


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--provider", choices=("mock", "closed_frontier", "open_weight"))
    parser.add_argument("--only", choices=("pending",), default="pending")
    parser.add_argument("--model", choices=("closed_frontier", "open_weight"))
    parser.add_argument("--condition", choices=("zero", "full"))
    parser.add_argument("--target", type=int)
    parser.add_argument("--run-id", type=int)
    parser.add_argument(
        "--single-run",
        action="store_true",
        help=(
            "Safety mode for exactly one logical experiment; requires --provider, "
            "--model, --condition, --target and --run-id"
        ),
    )
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent provider calls for distinct logical runs (default: 1)",
    )
    parser.add_argument("--mock-fixture", default="valid")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "llm")
    return parser.parse_args(argv)


def validate_single_run_args(args: argparse.Namespace) -> None:
    """Reject incomplete single-run selectors before any provider is built."""
    if not args.single_run:
        return
    missing = [
        flag
        for flag, value in (
            ("--provider", args.provider),
            ("--model", args.model),
            ("--condition", args.condition),
            ("--target", args.target),
            ("--run-id", args.run_id),
        )
        if value is None
    ]
    if missing:
        raise SystemExit(
            "--single-run requires explicit " + ", ".join(missing)
        )


def validate_workers(workers: int) -> None:
    if workers < 1:
        raise SystemExit("--workers must be at least 1")


def execute_selected_runs(
    runs: list[dict[str, Any]],
    provider,
    *,
    output_root: Path,
    force_rerun: bool,
    max_transport_attempts: int,
    workers: int,
) -> dict[str, int]:
    """Execute distinct logical runs, optionally overlapping network waits."""
    identities = [run["logical_run_key"] for run in runs]
    if len(identities) != len(set(identities)):
        raise ValueError("Selected run list contains duplicate logical identities")

    def execute(run: dict[str, Any]) -> dict[str, Any]:
        return execute_run(
            run,
            provider,
            output_root=output_root,
            force_rerun=force_rerun,
            max_transport_attempts=max_transport_attempts,
        )

    results: list[dict[str, Any]]
    if workers == 1 or len(runs) <= 1:
        results = [execute(run) for run in runs]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="llm-run") as pool:
            futures = [pool.submit(execute, run) for run in runs]
            for future in as_completed(futures):
                results.append(future.result())

    counts: dict[str, int] = {}
    for result in results:
        status = result["status"]
        counts[status] = counts.get(status, 0) + 1
    return counts


def main() -> None:
    args = parse_args()
    validate_single_run_args(args)
    validate_workers(args.workers)
    if args.dry_run:
        # Input regeneration requires the data-preparation environment (pandas,
        # parquet support). Normal execution consumes the frozen JSON artifacts
        # and deliberately does not import that heavier dependency stack.
        from experiments.llm.prepare_inputs import prepare_inputs

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
    if args.single_run and len(runs) != 1:
        raise SystemExit(
            f"--single-run must select exactly one logical run, selected {len(runs)}"
        )
    provider = build_provider(args.provider, config, args.mock_fixture)
    max_attempts = int(config["request_policy"]["transport_retry"]["max_attempts"])
    counts = execute_selected_runs(
        runs,
        provider,
        output_root=args.output_root,
        force_rerun=args.force_rerun,
        max_transport_attempts=max_attempts,
        workers=args.workers,
    )
    print(json.dumps({"selected_runs": len(runs), "statuses": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
