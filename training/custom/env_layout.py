"""Shared env layout/observation helpers for training and play runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .masked_policy_adapter import PlacementConfig


@dataclass(frozen=True)
class PlacementRuntime:
    config: PlacementConfig
    position_count: int


def parse_placement_runtime(spec_data: dict[str, Any]) -> PlacementRuntime:
    placement = spec_data.get("placement_schema", {})
    config = PlacementConfig(
        grid_width=int(placement.get("grid_width", 0)),
        grid_height=int(placement.get("grid_height", 0)),
        region_cols=int(placement.get("region_cols", 0)),
        region_rows=int(placement.get("region_rows", 0)),
        region_cell_count=int(placement.get("region_cell_count", 0)),
    )
    position_count = int(placement.get("position_count", 0))
    if position_count <= 0:
        raise ValueError(f"Invalid placement_schema.position_count={position_count}")
    return PlacementRuntime(config=config, position_count=position_count)


def as_card_position_masks(obs_entry: dict[str, Any], expected_position_count: int) -> np.ndarray:
    masks = np.asarray(obs_entry["position_masks_for_all_cards"], dtype=np.float32)
    if masks.ndim != 2:
        raise ValueError(f"Expected position_masks_for_all_cards 2D, got shape={masks.shape}")
    if expected_position_count > 0 and int(masks.shape[1]) != expected_position_count:
        raise ValueError(
            f"position_masks width mismatch expected={expected_position_count} got={int(masks.shape[1])}"
        )
    return masks

