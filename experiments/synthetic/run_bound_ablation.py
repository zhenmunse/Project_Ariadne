"""Certified heuristic-bound ablation runner (Task S9 implementation point)."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.synthetic.config import BoundAblationConfig, initialize_from_cli, scaffold_cli
else:
    from .config import BoundAblationConfig, initialize_from_cli, scaffold_cli


def main() -> None:
    parser = scaffold_cli("Run the synthetic certified-bound ablation")
    initialize_from_cli(parser.parse_args(), config_type=BoundAblationConfig)


if __name__ == "__main__":
    main()
