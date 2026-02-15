"""Deprecated entrypoint.

The old interactive Godot/Redot play path was removed in the cpp-only cutover.
Use `training.tools.cpp_visualizer` for visualization.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "play_checkpoint.py is deprecated after cpp-only cutover. "
        "Use `python -m training.tools.cpp_visualizer` instead."
    )


if __name__ == "__main__":
    main()
