"""Compatibility shim.

Historically `train_ppo.py` imported Args/run_training from this module.
The implementation now lives in smaller modules:
- knockoff_ppo_args.py
- knockoff_ppo_training.py
"""

from .knockoff_ppo_args import Args
from .knockoff_ppo_training import run_training

__all__ = ["Args", "run_training"]
