import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch


@dataclass
class CheckpointManagerConfig:
    run_dir: str
    keep_latest: int = 20


class CheckpointManager:
    def __init__(self, config: CheckpointManagerConfig):
        self.config = config
        self.run_dir = Path(config.run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.ckpt_dir / "index.jsonl"

    def save(self, step: int, train_state: Dict[str, Any], policy_state: Dict[str, Any], metadata: Dict[str, Any]) -> Path:
        step_dir = self.ckpt_dir / f"step_{step:09d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        train_state_path = step_dir / "train_state.pt"
        policy_state_path = step_dir / "policy_only.pt"
        torch.save(train_state, train_state_path)
        torch.save(policy_state, policy_state_path)

        row = {
            "step": step,
            "train_state": str(train_state_path),
            "policy_only": str(policy_state_path),
            "metadata": metadata,
        }
        with self.index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        self._prune_old()
        return step_dir

    def _prune_old(self) -> None:
        step_dirs = sorted([p for p in self.ckpt_dir.iterdir() if p.is_dir()])
        if len(step_dirs) <= self.config.keep_latest:
            return
        for stale_dir in step_dirs[: len(step_dirs) - self.config.keep_latest]:
            for child in stale_dir.iterdir():
                child.unlink(missing_ok=True)
            stale_dir.rmdir()
