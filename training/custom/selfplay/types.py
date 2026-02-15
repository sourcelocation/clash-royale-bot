"""Data models for self-play opponent pool state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OpponentCheckpoint:
    step: int
    policy_path: Path


@dataclass(frozen=True)
class OpponentSelection:
    category: str
    checkpoint: OpponentCheckpoint | None

    @property
    def policy_path(self) -> Path | None:
        if self.checkpoint is None:
            return None
        return self.checkpoint.policy_path
