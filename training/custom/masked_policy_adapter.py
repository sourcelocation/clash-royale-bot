"""Mask-aware PPO policy adapter for Knockoff multi-discrete action spaces."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .ppo import ClashPPOModel, ClashPPOModelConfig


@dataclass(frozen=True)
class PlacementConfig:
    grid_width: int
    grid_height: int
    region_cols: int
    region_rows: int
    region_cell_count: int


@dataclass(frozen=True)
class MaskedPolicyConfig:
    obs_dim: int
    action_branch_sizes: list[int]
    action_order: tuple[str, ...]
    hidden_sizes: tuple[int, ...]
    placement: PlacementConfig


class MaskedPPOAgent(nn.Module):
    """Thin adapter exposing CleanRL-like methods on top of ClashPPOModel."""

    def __init__(self, config: MaskedPolicyConfig):
        super().__init__()
        self.model = ClashPPOModel(
            ClashPPOModelConfig(
                obs_dim=config.obs_dim,
                action_branch_sizes=config.action_branch_sizes,
                hidden_sizes=config.hidden_sizes,
            )
        )
        self.action_order = tuple(config.action_order)
        self.branch_index = {key: i for i, key in enumerate(self.action_order)}
        required = ("wait", "card_selection", "position_region", "position_cell")
        for key in required:
            if key not in self.branch_index:
                raise ValueError(f"Missing required action branch: {key}")

        self.wait_idx = self.branch_index["wait"]
        self.card_idx = self.branch_index["card_selection"]
        self.region_idx = self.branch_index["position_region"]
        self.cell_idx = self.branch_index["position_cell"]

        self.placement = config.placement
        region_positions, region_positions_valid, region_cell_to_position = self._build_position_lookup()
        self.register_buffer("_region_positions", region_positions, persistent=False)
        self.register_buffer("_region_positions_valid", region_positions_valid, persistent=False)
        self.register_buffer("_region_cell_to_position", region_cell_to_position, persistent=False)

    def _build_position_lookup(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        width = int(self.placement.grid_width)
        height = int(self.placement.grid_height)
        region_cols = max(1, int(self.placement.region_cols))
        region_rows = max(1, int(self.placement.region_rows))
        cell_count = max(1, int(self.placement.region_cell_count))
        region_w = int((width + region_cols - 1) // region_cols)
        region_h = int((height + region_rows - 1) // region_rows)
        region_count = region_cols * region_rows

        region_to_positions: list[list[int]] = [[] for _ in range(region_count)]
        region_cell_to_position = torch.full((region_count, cell_count), -1, dtype=torch.long)

        for y in range(height):
            for x in range(width):
                pos_index = y * width + x
                region_x = min(region_cols - 1, x // max(1, region_w))
                region_y = min(region_rows - 1, y // max(1, region_h))
                region = region_y * region_cols + region_x
                local_x = x - region_x * region_w
                local_y = y - region_y * region_h
                cell = local_y * region_w + local_x
                if 0 <= cell < cell_count:
                    region_cell_to_position[region, cell] = pos_index
                region_to_positions[region].append(pos_index)

        max_positions = max((len(v) for v in region_to_positions), default=0)
        if max_positions <= 0:
            max_positions = 1
        region_positions = torch.full((region_count, max_positions), -1, dtype=torch.long)
        region_positions_valid = torch.zeros((region_count, max_positions), dtype=torch.bool)
        for region, positions in enumerate(region_to_positions):
            if not positions:
                continue
            count = len(positions)
            region_positions[region, :count] = torch.tensor(positions, dtype=torch.long)
            region_positions_valid[region, :count] = True
        return region_positions, region_positions_valid, region_cell_to_position

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.value(x).unsqueeze(-1)

    def _split_action_mask(self, mask: torch.Tensor) -> list[torch.Tensor]:
        return self.model.split_action_mask(mask)

    def _masked_dist(self, logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
        valid = mask > 0.5
        any_valid = valid.any(dim=-1, keepdim=True)
        safe_valid = valid.clone()
        safe_valid[~any_valid.expand_as(safe_valid)] = False
        safe_valid[~any_valid.squeeze(-1), 0] = True
        masked_logits = logits.masked_fill(~safe_valid, -1e9)
        return Categorical(logits=masked_logits)

    def _select_card_position_mask(
        self, card_position_masks: torch.Tensor, selected_cards: torch.Tensor
    ) -> torch.Tensor:
        # card_position_masks: [B, cards, positions]
        batch, cards, positions = card_position_masks.shape
        clamped = selected_cards.clamp(min=0, max=max(0, cards - 1))
        gathered = card_position_masks.gather(
            dim=1, index=clamped.view(batch, 1, 1).expand(batch, 1, positions)
        ).squeeze(1)
        valid = (selected_cards >= 0) & (selected_cards < cards)
        return gathered * valid.float().unsqueeze(-1)

    def _build_region_mask(self, selected_card_position_mask: torch.Tensor) -> torch.Tensor:
        # selected_card_position_mask: [B, positions]
        batch = selected_card_position_mask.shape[0]
        dtype = selected_card_position_mask.dtype
        region_positions = self._region_positions
        region_positions_valid = self._region_positions_valid
        region_count, positions_per_region = region_positions.shape

        gather_index = region_positions.clamp(min=0).unsqueeze(0).expand(batch, region_count, positions_per_region)
        gathered = selected_card_position_mask.gather(dim=1, index=gather_index.reshape(batch, -1)).reshape(
            batch, region_count, positions_per_region
        )
        valid = region_positions_valid.to(dtype=dtype).unsqueeze(0)
        return (gathered * valid).amax(dim=2)

    def _build_cell_mask(
        self, selected_card_position_mask: torch.Tensor, selected_regions: torch.Tensor
    ) -> torch.Tensor:
        _batch, _positions = selected_card_position_mask.shape
        dtype = selected_card_position_mask.dtype
        region_count, cell_count = self._region_cell_to_position.shape
        clamped_regions = selected_regions.clamp(min=0, max=max(0, region_count - 1))
        region_ok = ((selected_regions >= 0) & (selected_regions < region_count)).to(dtype=dtype).unsqueeze(-1)

        pos_idx = self._region_cell_to_position[clamped_regions]
        valid_cells = pos_idx >= 0
        selected = selected_card_position_mask.gather(dim=1, index=pos_idx.clamp(min=0))
        cell_masks = selected * valid_cells.to(dtype=dtype)
        return cell_masks * region_ok

    def get_action_and_value(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        card_position_masks: torch.Tensor,
        action: torch.Tensor | None = None,
        region_mask: torch.Tensor | None = None,
        cell_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.model.encode(x)
        logits_by_branch = [head(features) for head in self.model.policy_heads]
        masks_by_branch = self._split_action_mask(mask)

        wait_dist = self._masked_dist(logits_by_branch[self.wait_idx], masks_by_branch[self.wait_idx])
        wait_action = (
            torch.argmax(wait_dist.logits, dim=-1) if (action is None and deterministic) else
            (wait_dist.sample() if action is None else action[:, self.wait_idx])
        )

        card_dist = self._masked_dist(logits_by_branch[self.card_idx], masks_by_branch[self.card_idx])
        card_action = (
            torch.argmax(card_dist.logits, dim=-1) if (action is None and deterministic) else
            (card_dist.sample() if action is None else action[:, self.card_idx])
        )

        selected_card_position_mask = self._select_card_position_mask(card_position_masks, card_action)

        if action is None:
            region_mask_used = self._build_region_mask(selected_card_position_mask)
            region_dist = self._masked_dist(logits_by_branch[self.region_idx], region_mask_used)
            region_action = torch.argmax(region_dist.logits, dim=-1) if deterministic else region_dist.sample()
            cell_mask_used = self._build_cell_mask(selected_card_position_mask, region_action)
            cell_dist = self._masked_dist(logits_by_branch[self.cell_idx], cell_mask_used)
            cell_action = torch.argmax(cell_dist.logits, dim=-1) if deterministic else cell_dist.sample()
        else:
            region_action = action[:, self.region_idx]
            region_mask_used = region_mask if region_mask is not None else self._build_region_mask(
                selected_card_position_mask
            )
            region_dist = self._masked_dist(logits_by_branch[self.region_idx], region_mask_used)
            cell_mask_used = cell_mask if cell_mask is not None else self._build_cell_mask(
                selected_card_position_mask, region_action
            )
            cell_dist = self._masked_dist(logits_by_branch[self.cell_idx], cell_mask_used)
            cell_action = action[:, self.cell_idx]

        not_wait = (wait_action == 0).float()
        wait_logprob = wait_dist.log_prob(wait_action)
        card_logprob = card_dist.log_prob(card_action)
        region_logprob = region_dist.log_prob(region_action)
        cell_logprob = cell_dist.log_prob(cell_action)
        logprob = wait_logprob + not_wait * (card_logprob + region_logprob + cell_logprob)

        wait_entropy = wait_dist.entropy()
        card_entropy = card_dist.entropy()
        region_entropy = region_dist.entropy()
        cell_entropy = cell_dist.entropy()
        entropy = wait_entropy + not_wait * (card_entropy + region_entropy + cell_entropy)

        if action is None:
            sampled = torch.zeros(
                (x.shape[0], len(self.action_order)), dtype=torch.long, device=x.device
            )
            sampled[:, self.wait_idx] = wait_action
            sampled[:, self.card_idx] = torch.where(wait_action == 1, torch.zeros_like(card_action), card_action)
            sampled[:, self.region_idx] = torch.where(
                wait_action == 1, torch.zeros_like(region_action), region_action
            )
            sampled[:, self.cell_idx] = torch.where(wait_action == 1, torch.zeros_like(cell_action), cell_action)
            out_action = sampled
        else:
            out_action = action

        value = self.model.value_head(features)
        return out_action, logprob, entropy, value, region_mask_used, cell_mask_used
