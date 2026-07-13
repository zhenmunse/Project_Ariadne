"""Monotone synthetic oracle (Task S3 implementation point)."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.synthetic.config import OracleParameterConfig, initialize_from_cli, scaffold_cli
else:
    from .config import OracleParameterConfig, initialize_from_cli, scaffold_cli

__all__ = ["OracleParameterConfig"]


def main() -> None:
    parser = scaffold_cli("Initialize the monotone synthetic oracle")
    initialize_from_cli(parser.parse_args())


if __name__ == "__main__":
    main()
