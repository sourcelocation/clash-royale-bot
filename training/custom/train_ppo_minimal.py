"""Entry point for the minimal PPO rewrite."""

import tyro

from .minimal_ppo_args import Args
from .minimal_ppo_training import run_training


def main() -> None:
    run_training(tyro.cli(Args))


if __name__ == "__main__":
    main()

