"""Frozen opponent policy loading and inference cache for self-play."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..knockoff_ppo_support import torch_load_dict
from ..env_layout import as_card_position_masks
from ..masked_policy_adapter import MaskedPPOAgent, MaskedPolicyConfig, PlacementConfig


def _normalize_agent_state_dict_keys(raw_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keys = list(raw_state.keys())
    if not keys:
        return raw_state
    if all(k.startswith("model.") for k in keys):
        return raw_state
    return {f"model.{k}": v for k, v in raw_state.items()}


def _parse_hidden_sizes(raw: str) -> tuple[int, ...]:
    values = tuple(int(x.strip()) for x in str(raw).split(",") if x.strip())
    if not values:
        raise ValueError("hidden_sizes must not be empty")
    return values


def load_checkpoint_model_state(ckpt_path: Path) -> tuple[dict[str, torch.Tensor], tuple[int, ...]]:
    payload = torch_load_dict(ckpt_path, label="checkpoint", map_location="cpu")

    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError(f"Unsupported checkpoint format (missing model_state_dict): {ckpt_path}")

    hidden_sizes: tuple[int, ...] | None = None
    model_cfg = payload.get("model_config")
    if isinstance(model_cfg, dict) and isinstance(model_cfg.get("hidden_sizes"), list):
        hidden_sizes = tuple(int(x) for x in model_cfg["hidden_sizes"])

    if hidden_sizes is None:
        raw_args = payload.get("args")
        if isinstance(raw_args, dict) and isinstance(raw_args.get("hidden_sizes"), str):
            hidden_sizes = _parse_hidden_sizes(raw_args["hidden_sizes"])

    if hidden_sizes is None or len(hidden_sizes) == 0:
        hidden_sizes = (256, 256)

    return _normalize_agent_state_dict_keys(model_state), hidden_sizes


@dataclass
class OpponentModelShape:
    obs_dim: int
    action_branch_sizes: tuple[int, ...]
    action_order: tuple[str, ...]
    placement: PlacementConfig

    @property
    def position_count(self) -> int:
        return int(self.placement.grid_width) * int(self.placement.grid_height)


class FrozenOpponentPolicy:
    def __init__(
        self,
        *,
        checkpoint_path: Path,
        shape: OpponentModelShape,
        device: torch.device,
        deterministic: bool,
    ):
        model_state, hidden_sizes = load_checkpoint_model_state(checkpoint_path)
        self.shape = shape
        self.device = device
        self.deterministic = bool(deterministic)
        self.agent = MaskedPPOAgent(
            MaskedPolicyConfig(
                obs_dim=int(shape.obs_dim),
                action_branch_sizes=list(shape.action_branch_sizes),
                action_order=shape.action_order,
                hidden_sizes=hidden_sizes,
                placement=shape.placement,
            )
        ).to(device)
        self.agent.load_state_dict(model_state)
        self.agent.eval()

    def act(self, obs_entry: dict[str, np.ndarray]) -> dict[str, int]:
        vec = np.asarray(obs_entry["vector"], dtype=np.float32).reshape(1, -1)
        mask = np.asarray(obs_entry["action_mask"], dtype=np.float32).reshape(1, -1)
        card_masks = as_card_position_masks(
            obs_entry,
            expected_position_count=int(self.shape.position_count),
        ).reshape(1, -1, int(self.shape.position_count))
        return self.act_arrays(vec=vec, mask=mask, card_masks=card_masks)

    def act_arrays(self, *, vec: np.ndarray, mask: np.ndarray, card_masks: np.ndarray) -> dict[str, int]:
        vec_np = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        mask_np = np.asarray(mask, dtype=np.float32).reshape(1, -1)
        card_np = np.asarray(card_masks, dtype=np.float32).reshape(1, -1, int(self.shape.position_count))

        with torch.no_grad():
            x = torch.from_numpy(vec_np).to(self.device)
            x_mask = torch.from_numpy(mask_np).to(self.device)
            x_card_masks = torch.from_numpy(card_np).to(self.device)
            action, _, _, _, _, _ = self.agent.get_action_and_value(
                x,
                x_mask,
                x_card_masks,
                deterministic=self.deterministic,
            )

        branches = action[0].detach().cpu().tolist()
        return {key: int(branches[i]) for i, key in enumerate(self.shape.action_order)}


class OpponentPolicyCache:
    def __init__(
        self,
        *,
        shape: OpponentModelShape,
        device: torch.device,
        deterministic: bool,
        max_cached: int = 6,
    ):
        self.shape = shape
        self.device = device
        self.deterministic = bool(deterministic)
        self.max_cached = max(1, int(max_cached))
        self._cache: OrderedDict[str, FrozenOpponentPolicy] = OrderedDict()

    def get(self, checkpoint_path: Path) -> FrozenOpponentPolicy:
        key = str(checkpoint_path)
        if key in self._cache:
            policy = self._cache.pop(key)
            self._cache[key] = policy
            return policy

        policy = FrozenOpponentPolicy(
            checkpoint_path=checkpoint_path,
            shape=self.shape,
            device=self.device,
            deterministic=self.deterministic,
        )
        self._cache[key] = policy
        while len(self._cache) > self.max_cached:
            self._cache.popitem(last=False)
        return policy
