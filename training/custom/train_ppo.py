"""Thin CLI entrypoint for Knockoff PPO training.

This module intentionally stays small. Runtime orchestration lives in
`training.custom.knockoff_ppo_runner`, while upstream CleanRL reference code is
pinned in `training/custom/cleanrl_upstream/ppo.py`.
"""

import tyro

from .knockoff_ppo_runner import Args, run_training


def main() -> None:
    run_training(tyro.cli(Args))


if __name__ == "__main__":
    main()
