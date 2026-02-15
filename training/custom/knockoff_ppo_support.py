"""Shared helpers/constants for Knockoff PPO runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

POLICY_CHECKPOINT_FORMAT = "clash_policy_v1"
PATCH_ID = "knockoff_cleanrl_patch_v3"


def resolve_torch_device(use_cuda: bool, use_mps: bool) -> torch.device:
    """Select torch device with priority: CUDA -> MPS -> CPU."""
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if use_mps and mps_backend is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_upstream_lock() -> dict[str, str]:
    lock_path = Path(__file__).resolve().parent / "cleanrl_upstream" / "UPSTREAM_LOCK.json"
    upstream_path = Path(__file__).resolve().parent / "cleanrl_upstream" / "ppo.py"
    if not lock_path.exists():
        raise FileNotFoundError(f"Missing upstream lock file: {lock_path}")
    if not upstream_path.exists():
        raise FileNotFoundError(f"Missing pinned upstream file: {upstream_path}")

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    expected_hash = str(lock.get("sha256", "")).strip().lower()
    actual_hash = hashlib.sha256(upstream_path.read_bytes()).hexdigest().lower()
    if expected_hash and expected_hash != actual_hash:
        raise RuntimeError(
            f"Pinned upstream hash mismatch for {upstream_path}. expected={expected_hash} actual={actual_hash}"
        )
    return {
        "repo": str(lock.get("repo", "")),
        "path": str(lock.get("path", "")),
        "commit": str(lock.get("commit", "")),
        "sha256": actual_hash,
    }


def next_experiment_dir(run_root: Path) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    max_index = 0
    for child in run_root.iterdir():
        if child.is_dir() and child.name.startswith("exp_") and child.name[4:].isdigit():
            max_index = max(max_index, int(child.name[4:]))
    return run_root / f"exp_{max_index + 1:04d}"


def find_latest_checkpoint_row(run_root: Path) -> tuple[Path, dict[str, Any]]:
    if not run_root.exists():
        raise FileNotFoundError(f"Checkpoint run root does not exist: {run_root}")
    exp_dirs = sorted([p for p in run_root.iterdir() if p.is_dir() and p.name.startswith("exp_") and p.name[4:].isdigit()])
    if not exp_dirs:
        raise FileNotFoundError(f"No experiment dirs found under: {run_root}")
    for exp_dir in reversed(exp_dirs):
        index_path = exp_dir / "checkpoints" / "index.jsonl"
        if not index_path.exists():
            continue
        lines = [ln.strip() for ln in index_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for line in reversed(lines):
            row = json.loads(line)
            train_state = Path(row.get("train_state", ""))
            if train_state.exists():
                return exp_dir, row
    raise FileNotFoundError(f"No resumeable checkpoints found under: {run_root}")


def wait_action(action_order: list[str]) -> dict[str, int]:
    action = {key: 0 for key in action_order}
    if "wait" in action:
        action["wait"] = 1
    return action


def torch_load_dict(path: Path, *, label: str, map_location: str | torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported {label} payload type: {type(payload)}")
    return payload


def build_policy_payload(model: torch.nn.Module, upstream: dict[str, str]) -> dict[str, Any]:
    model_cfg = getattr(model, "config", None)
    return {
        "format": POLICY_CHECKPOINT_FORMAT,
        "model_state_dict": model.state_dict(),
        "model_config": {
            "hidden_sizes": list(getattr(model_cfg, "hidden_sizes", [])),
            "activation": str(getattr(model_cfg, "activation", "tanh")),
        },
        "upstream_repo": upstream["repo"],
        "upstream_path": upstream["path"],
        "upstream_commit": upstream["commit"],
        "upstream_sha256": upstream["sha256"],
        "patch_id": PATCH_ID,
    }
