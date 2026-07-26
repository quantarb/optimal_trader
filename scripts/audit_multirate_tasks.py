"""Persist the resolved current MTL task inventory."""

from __future__ import annotations

import argparse
from pathlib import Path

from multirate_transformer.task_registry import task_inventory_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = task_inventory_frame()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()

