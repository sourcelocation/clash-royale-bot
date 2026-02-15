"""Regenerate training/custom/train_ppo.py from upstream source.

Current workflow keeps upstream source pinned for easy diffing.
If patching is automated later, this script should apply patch hunks and rewrite
training/custom/train_ppo.py. For now it prints guidance and exits.
"""

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    upstream = root / "training" / "custom" / "cleanrl_upstream" / "ppo.py"
    target = root / "training" / "custom" / "train_ppo.py"
    print(f"upstream={upstream}")
    print(f"target={target}")
    print("Manual sync currently required for project-specific env/model integration.")


if __name__ == "__main__":
    main()
