"""Persistent Elo tracking for self-play checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EloRatingTrackerConfig:
    root_dir: str
    initial_rating: float = 1000.0
    k_factor: float = 16.0


class EloRatingTracker:
    def __init__(self, config: EloRatingTrackerConfig):
        self.config = config
        self.root_dir = Path(config.root_dir)
        self.path = self.root_dir / "ratings.json"
        self.root_dir.mkdir(parents=True, exist_ok=True)

        self.current_rating: float = float(config.initial_rating)
        self.total_games: int = 0
        self.total_wins: int = 0
        self.total_draws: int = 0
        self.total_losses: int = 0
        self.checkpoints: dict[int, dict[str, float | int]] = {}
        self._dirty: bool = False
        self._load()

    def ensure_checkpoint(self, step: int) -> None:
        key = int(step)
        if key in self.checkpoints:
            return
        self.checkpoints[key] = {
            "rating": float(self.current_rating),
            "games": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
        }
        self._dirty = True

    def get_checkpoint_rating(self, step: int) -> float:
        key = int(step)
        self.ensure_checkpoint(key)
        row = self.checkpoints[key]
        return float(row.get("rating", self.config.initial_rating))

    def record_match_vs_checkpoint(self, checkpoint_step: int, score_current: float) -> tuple[float, float]:
        score = max(0.0, min(1.0, float(score_current)))
        self.ensure_checkpoint(int(checkpoint_step))

        checkpoint_row = self.checkpoints[int(checkpoint_step)]
        rating_current = float(self.current_rating)
        rating_opponent = float(checkpoint_row.get("rating", self.config.initial_rating))

        expected_current = 1.0 / (1.0 + 10.0 ** ((rating_opponent - rating_current) / 400.0))
        expected_opponent = 1.0 - expected_current

        k = float(self.config.k_factor)
        new_current = rating_current + k * (score - expected_current)
        score_opponent = 1.0 - score
        new_opponent = rating_opponent + k * (score_opponent - expected_opponent)

        self.current_rating = float(new_current)
        checkpoint_row["rating"] = float(new_opponent)

        self.total_games += 1
        checkpoint_row["games"] = int(checkpoint_row.get("games", 0)) + 1
        if score >= 0.999:
            self.total_wins += 1
            checkpoint_row["losses"] = int(checkpoint_row.get("losses", 0)) + 1
        elif score <= 0.001:
            self.total_losses += 1
            checkpoint_row["wins"] = int(checkpoint_row.get("wins", 0)) + 1
        else:
            self.total_draws += 1
            checkpoint_row["draws"] = int(checkpoint_row.get("draws", 0)) + 1

        self._dirty = True
        return self.current_rating, float(checkpoint_row["rating"])

    def flush(self, *, force: bool = False) -> None:
        if not self._dirty and not force:
            return
        data = {
            "version": 1,
            "initial_rating": float(self.config.initial_rating),
            "k_factor": float(self.config.k_factor),
            "current": {
                "rating": float(self.current_rating),
                "games": int(self.total_games),
                "wins": int(self.total_wins),
                "draws": int(self.total_draws),
                "losses": int(self.total_losses),
            },
            "checkpoints": {
                str(step): {
                    "rating": float(row.get("rating", self.config.initial_rating)),
                    "games": int(row.get("games", 0)),
                    "wins": int(row.get("wins", 0)),
                    "draws": int(row.get("draws", 0)),
                    "losses": int(row.get("losses", 0)),
                }
                for step, row in sorted(self.checkpoints.items(), key=lambda kv: kv[0])
            },
        }
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._dirty = False

    def _load(self) -> None:
        if not self.path.exists():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return

        current = payload.get("current", {})
        if isinstance(current, dict):
            self.current_rating = float(current.get("rating", self.current_rating))
            self.total_games = int(current.get("games", self.total_games))
            self.total_wins = int(current.get("wins", self.total_wins))
            self.total_draws = int(current.get("draws", self.total_draws))
            self.total_losses = int(current.get("losses", self.total_losses))

        rows = payload.get("checkpoints", {})
        if isinstance(rows, dict):
            parsed: dict[int, dict[str, float | int]] = {}
            for raw_step, row in rows.items():
                if not isinstance(raw_step, str) or not raw_step.isdigit() or not isinstance(row, dict):
                    continue
                step = int(raw_step)
                parsed[step] = {
                    "rating": float(row.get("rating", self.config.initial_rating)),
                    "games": int(row.get("games", 0)),
                    "wins": int(row.get("wins", 0)),
                    "draws": int(row.get("draws", 0)),
                    "losses": int(row.get("losses", 0)),
                }
            self.checkpoints = parsed

        self._dirty = False
