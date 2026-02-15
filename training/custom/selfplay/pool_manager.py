"""Recent+anchor self-play pool manager.

Stores a compact pool index and archives policy checkpoints so the pool remains
stable even if the trainer prunes regular checkpoints.
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from .types import OpponentCheckpoint, OpponentSelection


@dataclass
class SelfPlayPoolConfig:
    root_dir: str
    recent_capacity: int = 32
    anchor_capacity: int = 8
    anchor_every: int = 4
    latest_prob: float = 0.4
    recent_prob: float = 0.4
    anchor_prob: float = 0.2


class SelfPlayPoolManager:
    def __init__(self, config: SelfPlayPoolConfig):
        self.config = config
        self.root_dir = Path(config.root_dir)
        self.policies_dir = self.root_dir / "policies"
        self.index_path = self.root_dir / "pool.json"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.policies_dir.mkdir(parents=True, exist_ok=True)

        self._recent: list[OpponentCheckpoint] = []
        self._anchors: list[OpponentCheckpoint] = []
        self._registered_count: int = 0
        self._load()

    @property
    def has_any(self) -> bool:
        return bool(self._recent or self._anchors)

    def register_checkpoint(self, *, step: int, policy_source_path: Path) -> OpponentCheckpoint:
        if not policy_source_path.exists():
            raise FileNotFoundError(f"Policy checkpoint not found: {policy_source_path}")

        archived = self.policies_dir / f"policy_step_{int(step):09d}.pt"
        if not archived.exists():
            shutil.copy2(policy_source_path, archived)

        entry = OpponentCheckpoint(step=int(step), policy_path=archived)
        self._registered_count += 1

        self._recent = [x for x in self._recent if x.step != entry.step]
        self._recent.append(entry)
        self._recent.sort(key=lambda x: x.step)
        if len(self._recent) > int(self.config.recent_capacity):
            self._recent = self._recent[-int(self.config.recent_capacity) :]

        should_anchor = int(self.config.anchor_every) > 0 and (self._registered_count % int(self.config.anchor_every) == 0)
        if should_anchor:
            self._anchors = [x for x in self._anchors if x.step != entry.step]
            self._anchors.append(entry)
            self._anchors.sort(key=lambda x: x.step)
            if len(self._anchors) > int(self.config.anchor_capacity):
                self._anchors = self._anchors[-int(self.config.anchor_capacity) :]

        self._save()
        return entry

    def sample(self) -> OpponentSelection:
        latest = self._latest()
        if latest is None:
            return OpponentSelection(category="none", checkpoint=None)

        categories = self._available_categories()
        if not categories:
            return OpponentSelection(category="latest", checkpoint=latest)

        category = self._sample_category(categories)
        if category == "latest":
            return OpponentSelection(category="latest", checkpoint=latest)
        if category == "recent":
            recent = self._random_recent_excluding_latest()
            if recent is None:
                return OpponentSelection(category="latest", checkpoint=latest)
            return OpponentSelection(category="recent", checkpoint=recent)
        if category == "anchor":
            anchor = self._random_anchor()
            if anchor is None:
                return OpponentSelection(category="latest", checkpoint=latest)
            return OpponentSelection(category="anchor", checkpoint=anchor)
        return OpponentSelection(category="latest", checkpoint=latest)

    def _latest(self) -> OpponentCheckpoint | None:
        if not self._recent:
            return None
        return self._recent[-1]

    def summary(self) -> dict[str, int | None]:
        latest = self._latest()
        return {
            "recent_count": int(len(self._recent)),
            "anchor_count": int(len(self._anchors)),
            "registered_count": int(self._registered_count),
            "latest_step": (int(latest.step) if latest is not None else None),
        }

    def _random_recent_excluding_latest(self) -> OpponentCheckpoint | None:
        if len(self._recent) <= 1:
            return None
        return random.choice(self._recent[:-1])

    def _random_anchor(self) -> OpponentCheckpoint | None:
        if not self._anchors:
            return None
        return random.choice(self._anchors)

    def _available_categories(self) -> list[str]:
        out = ["latest"]
        if len(self._recent) > 1:
            out.append("recent")
        if self._anchors:
            out.append("anchor")
        return out

    def _sample_category(self, categories: list[str]) -> str:
        weights = []
        for category in categories:
            if category == "latest":
                weights.append(max(0.0, float(self.config.latest_prob)))
            elif category == "recent":
                weights.append(max(0.0, float(self.config.recent_prob)))
            elif category == "anchor":
                weights.append(max(0.0, float(self.config.anchor_prob)))
            else:
                weights.append(0.0)

        if sum(weights) <= 0.0:
            return categories[0]
        return random.choices(categories, weights=weights, k=1)[0]

    def _to_entry_dict(self, item: OpponentCheckpoint) -> dict[str, str | int]:
        return {"step": int(item.step), "policy_path": str(item.policy_path)}

    def _from_entry_dict(self, payload: dict[str, object]) -> OpponentCheckpoint | None:
        step = payload.get("step")
        policy_path = payload.get("policy_path")
        if not isinstance(step, int) or not isinstance(policy_path, str):
            return None
        path = Path(policy_path)
        if not path.exists():
            return None
        return OpponentCheckpoint(step=int(step), policy_path=path)

    def _save(self) -> None:
        data = {
            "version": 1,
            "registered_count": int(self._registered_count),
            "recent": [self._to_entry_dict(x) for x in self._recent],
            "anchors": [self._to_entry_dict(x) for x in self._anchors],
        }
        self.index_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _load(self) -> None:
        if not self.index_path.exists():
            return
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return

        loaded_recent: list[OpponentCheckpoint] = []
        for item in payload.get("recent", []):
            if isinstance(item, dict):
                parsed = self._from_entry_dict(item)
                if parsed is not None:
                    loaded_recent.append(parsed)

        loaded_anchors: list[OpponentCheckpoint] = []
        for item in payload.get("anchors", []):
            if isinstance(item, dict):
                parsed = self._from_entry_dict(item)
                if parsed is not None:
                    loaded_anchors.append(parsed)

        self._recent = sorted(loaded_recent, key=lambda x: x.step)[-int(self.config.recent_capacity) :]
        self._anchors = sorted(loaded_anchors, key=lambda x: x.step)[-int(self.config.anchor_capacity) :]
        self._registered_count = int(payload.get("registered_count", len(self._recent)))
