from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical


def branch_sizes_from_action_space(action_space: Dict[str, Any], action_order: Sequence[str]) -> List[int]:
    """Build per-branch sizes from env spec action space + action order."""
    sizes: List[int] = []
    for key in action_order:
        if key not in action_space:
            raise KeyError(f"Missing action key in action_space: {key}")
        raw = action_space[key]
        if isinstance(raw, dict):
            size = int(raw.get("size", 0))
        else:
            size = int(getattr(raw, "n", 0))
        if size <= 0:
            raise ValueError(f"Invalid branch size for key={key}: {size}")
        sizes.append(size)
    return sizes


@dataclass(frozen=True)
class ClashPPOModelConfig:
    obs_dim: int
    action_branch_sizes: Sequence[int]
    hidden_sizes: Sequence[int] = (256, 256)
    activation: str = "tanh"
    ortho_init: bool = True
    policy_init_gain: float = 0.01
    value_init_gain: float = 1.0


class MaskedMultiDiscreteDistribution:
    """Categorical distribution over multiple branches with per-branch masks."""

    def __init__(self, logits_by_branch: Sequence[Tensor], masks_by_branch: Sequence[Tensor]):
        if len(logits_by_branch) != len(masks_by_branch):
            raise ValueError("logits_by_branch and masks_by_branch must have same length")
        self._dists: List[Categorical] = []
        for logits, mask in zip(logits_by_branch, masks_by_branch):
            if logits.shape != mask.shape:
                raise ValueError(
                    f"logits/mask shape mismatch: logits={tuple(logits.shape)} mask={tuple(mask.shape)}"
                )
            masked_logits = _apply_action_mask(logits, mask)
            self._dists.append(Categorical(logits=masked_logits))

    def sample(self) -> Tensor:
        return torch.stack([d.sample() for d in self._dists], dim=-1)

    def mode(self) -> Tensor:
        return torch.stack([torch.argmax(d.logits, dim=-1) for d in self._dists], dim=-1)

    def log_prob(self, actions: Tensor) -> Tensor:
        if actions.dim() != 2 or actions.shape[-1] != len(self._dists):
            raise ValueError(
                f"Expected actions shape [batch, {len(self._dists)}], got {tuple(actions.shape)}"
            )
        per_branch = [dist.log_prob(actions[:, i]) for i, dist in enumerate(self._dists)]
        return torch.stack(per_branch, dim=-1).sum(dim=-1)

    def entropy(self) -> Tensor:
        per_branch = [dist.entropy() for dist in self._dists]
        return torch.stack(per_branch, dim=-1).sum(dim=-1)


class ClashPPOModel(nn.Module):
    """Shared encoder + masked multi-branch policy/value heads for PPO."""

    def __init__(self, config: ClashPPOModelConfig):
        super().__init__()
        if config.obs_dim <= 0:
            raise ValueError(f"obs_dim must be > 0, got {config.obs_dim}")
        if len(config.action_branch_sizes) == 0:
            raise ValueError("action_branch_sizes must be non-empty")
        for i, size in enumerate(config.action_branch_sizes):
            if int(size) <= 0:
                raise ValueError(f"Invalid branch size at index {i}: {size}")

        self.config = config
        self.obs_dim = int(config.obs_dim)
        self.action_branch_sizes = [int(s) for s in config.action_branch_sizes]
        self.num_branches = len(self.action_branch_sizes)
        self.action_mask_dim = int(sum(self.action_branch_sizes))

        activation = _build_activation(config.activation)
        trunk_layers: List[nn.Module] = []
        in_dim = self.obs_dim
        for hidden_dim in config.hidden_sizes:
            hidden_dim = int(hidden_dim)
            if hidden_dim <= 0:
                raise ValueError(f"hidden_sizes must be > 0, got {hidden_dim}")
            trunk_layers.append(nn.Linear(in_dim, hidden_dim))
            trunk_layers.append(activation())
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*trunk_layers)

        self.policy_heads = nn.ModuleList([nn.Linear(in_dim, size) for size in self.action_branch_sizes])
        self.value_head = nn.Linear(in_dim, 1)

        if config.ortho_init:
            self._apply_orthogonal_init(
                policy_gain=float(config.policy_init_gain),
                value_gain=float(config.value_init_gain),
            )

    def split_action_mask(self, action_mask: Tensor) -> List[Tensor]:
        if action_mask.dim() != 2:
            raise ValueError(f"Expected action_mask shape [batch, mask_dim], got {tuple(action_mask.shape)}")
        if action_mask.shape[-1] != self.action_mask_dim:
            raise ValueError(
                f"Expected action_mask dim={self.action_mask_dim}, got {int(action_mask.shape[-1])}"
            )

        masks: List[Tensor] = []
        offset = 0
        for size in self.action_branch_sizes:
            masks.append(action_mask[:, offset : offset + size])
            offset += size
        return masks

    def encode(self, obs_vector: Tensor) -> Tensor:
        if obs_vector.dim() != 2 or obs_vector.shape[-1] != self.obs_dim:
            raise ValueError(f"Expected obs_vector shape [batch, {self.obs_dim}], got {tuple(obs_vector.shape)}")
        return self.encoder(obs_vector)

    def distribution(self, obs_vector: Tensor, action_mask: Tensor) -> MaskedMultiDiscreteDistribution:
        features = self.encode(obs_vector)
        logits_by_branch = [head(features) for head in self.policy_heads]
        masks_by_branch = self.split_action_mask(action_mask)
        return MaskedMultiDiscreteDistribution(logits_by_branch, masks_by_branch)

    def value(self, obs_vector: Tensor) -> Tensor:
        features = self.encode(obs_vector)
        return self.value_head(features).squeeze(-1)

    def act(
        self,
        obs_vector: Tensor,
        action_mask: Tensor,
        deterministic: bool = False,
    ) -> Dict[str, Tensor]:
        dist = self.distribution(obs_vector, action_mask)
        actions = dist.mode() if deterministic else dist.sample()
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.value(obs_vector)
        return {
            "actions": actions,
            "log_prob": log_prob,
            "entropy": entropy,
            "value": value,
        }

    def evaluate_actions(
        self,
        obs_vector: Tensor,
        action_mask: Tensor,
        actions: Tensor,
    ) -> Dict[str, Tensor]:
        dist = self.distribution(obs_vector, action_mask)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.value(obs_vector)
        return {
            "log_prob": log_prob,
            "entropy": entropy,
            "value": value,
        }

    def _apply_orthogonal_init(self, policy_gain: float, value_gain: float) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("tanh"))
                nn.init.constant_(module.bias, 0.0)
        for head in self.policy_heads:
            nn.init.orthogonal_(head.weight, gain=policy_gain)
            nn.init.constant_(head.bias, 0.0)
        nn.init.orthogonal_(self.value_head.weight, gain=value_gain)
        nn.init.constant_(self.value_head.bias, 0.0)


def _build_activation(name: str) -> type[nn.Module]:
    normalized = str(name).strip().lower()
    if normalized == "tanh":
        return nn.Tanh
    if normalized == "relu":
        return nn.ReLU
    if normalized == "silu":
        return nn.SiLU
    raise ValueError(f"Unsupported activation: {name}")


def _apply_action_mask(logits: Tensor, mask: Tensor) -> Tensor:
    """Mask invalid logits. If a row has no valid actions, force index 0 valid."""
    if logits.dim() != 2:
        raise ValueError(f"Expected 2D logits [batch, branch_size], got {tuple(logits.shape)}")
    if mask.dim() != 2:
        raise ValueError(f"Expected 2D mask [batch, branch_size], got {tuple(mask.shape)}")

    valid = mask > 0.5
    any_valid = valid.any(dim=-1, keepdim=True)

    safe_valid = valid.clone()
    safe_valid[~any_valid.expand_as(safe_valid)] = False
    safe_valid[~any_valid.squeeze(-1), 0] = True

    masked_logits = logits.masked_fill(~safe_valid, -1e9)
    return masked_logits

