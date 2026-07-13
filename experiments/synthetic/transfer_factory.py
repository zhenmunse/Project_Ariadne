"""Soft-transfer graph factories (Task S2/S5 implementation point)."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.synthetic.config import TransferArtifactConfig, initialize_from_cli, scaffold_cli
else:
    from .config import TransferArtifactConfig, initialize_from_cli, scaffold_cli

__all__ = ["TransferArtifactConfig"]


def main() -> None:
    parser = scaffold_cli("Initialize synthetic soft-transfer graph generation")
    initialize_from_cli(parser.parse_args())


if __name__ == "__main__":
    main()
