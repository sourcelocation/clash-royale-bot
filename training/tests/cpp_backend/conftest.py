from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_PROTOCOL_VERSION = "knockoff_env_v1"
EXPECTED_OBS_VERSION = "v1"
EXPECTED_SCHEMA_VERSION = "knockoff_cr_env_v2"
EXPECTED_DEBUG_PROTOCOL_VERSION = "knockoff_env_debug_v1"

GRID_W = 18
GRID_H = 32
HALF_W = GRID_W / 2.0
HALF_H = GRID_H / 2.0


def _import_module():
    return pytest.importorskip("knockoff_cr_cpp")


@pytest.fixture
def cpp_module():
    return _import_module()


@pytest.fixture
def env_factory(cpp_module):
    def _make(*, tick_hz: int = 10, max_sim_seconds: float = 60.0, seed: int = 1):
        return cpp_module.ClashEnv(tick_hz=tick_hz, max_sim_seconds=max_sim_seconds, seed=seed)

    return _make


@pytest.fixture
def batch_env_factory(cpp_module):
    def _make(*, num_envs: int = 4, tick_hz: int = 10, max_sim_seconds: float = 60.0, seed: int = 1, num_threads: int = 0):
        return cpp_module.ClashEnvBatch(
            num_envs=num_envs,
            tick_hz=tick_hz,
            max_sim_seconds=max_sim_seconds,
            seed=seed,
            num_threads=num_threads,
        )

    return _make


@pytest.fixture
def env(env_factory):
    instance = env_factory()
    try:
        yield instance
    finally:
        del instance


def branch_sizes(spec: Dict[str, Any]) -> List[Tuple[str, int]]:
    order = list(spec["action_order"])
    space = spec["action_space"]
    return [(key, int(space[key]["size"])) for key in order]


def split_mask(mask: Iterable[float], sizes: List[Tuple[str, int]]) -> Dict[str, List[float]]:
    vals = list(mask)
    out: Dict[str, List[float]] = {}
    offset = 0
    for key, size in sizes:
        out[key] = vals[offset : offset + size]
        offset += size
    return out


def sample_action_from_mask(
    mask: Iterable[float],
    sizes: List[Tuple[str, int]],
    rng: random.Random,
) -> Dict[str, int]:
    parts = split_mask(mask, sizes)
    action: Dict[str, int] = {}
    for key, branch in parts.items():
        valid = [i for i, v in enumerate(branch) if float(v) > 0.0]
        action[key] = rng.choice(valid) if valid else 0
    return action


def assert_finite_nested(values: Any) -> None:
    if isinstance(values, (list, tuple)):
        for v in values:
            assert_finite_nested(v)
        return
    assert math.isfinite(float(values))


def assert_debug_state_invariants(state: Dict[str, Any]) -> None:
    assert state["protocol_version"] == EXPECTED_DEBUG_PROTOCOL_VERSION
    assert isinstance(state["entities"], list)
    assert isinstance(state["pending_spawns"], list)
    assert isinstance(state["fireballs"], list)
    assert isinstance(state["logs"], list)

    elixir = list(state["elixir"])
    assert len(elixir) == 2
    for v in elixir:
        assert -1e-6 <= float(v) <= 10.0 + 1e-6

    seen_ids = set()
    for e in state["entities"]:
        eid = int(e["id"])
        assert eid not in seen_ids
        seen_ids.add(eid)

        assert e["kind"] in {"troop", "building", "tower"}
        assert int(e["team"]) in (0, 1)
        assert bool(e["alive"]) in (True, False)

        x = float(e["x"])
        y = float(e["y"])
        hp = float(e["hp"])
        max_hp = float(e["max_hp"])
        radius = float(e["radius"])

        assert -HALF_W - 1e-6 <= x <= HALF_W + 1e-6
        assert -HALF_H - 1e-6 <= y <= HALF_H + 1e-6
        assert radius > 0.0
        assert max_hp >= 0.0
        assert hp <= max_hp + 1e-6

        assert math.isfinite(float(e["stun_rem"]))
        assert math.isfinite(float(e["deployment_lock_rem"]))

    for p in state["pending_spawns"]:
        assert int(p["team"]) in (0, 1)
        assert isinstance(p["entity_ids"], list)
        assert isinstance(p["state"], str)
        assert math.isfinite(float(p["spawn_at_s"]))

    for f in state["fireballs"]:
        assert int(f["team"]) in (0, 1)
        assert math.isfinite(float(f["detonate_at_s"]))

    for s in state["logs"]:
        assert int(s["team"]) in (0, 1)
        assert math.isfinite(float(s["time_left_s"]))


def rollout_fingerprint(
    env,
    *,
    seed: int,
    steps: int,
) -> List[Tuple[Any, ...]]:
    rng = random.Random(seed)
    spec = env.spec()
    sizes = branch_sizes(spec)
    transition = env.reset(seed=seed, options={})
    trace: List[Tuple[Any, ...]] = []

    for _ in range(steps):
        actions = []
        for team in range(2):
            action = sample_action_from_mask(transition["action_mask"][team], sizes, rng)
            actions.append(action)

        transition = env.step(actions)
        state = env.debug_state()

        rewards = tuple(round(float(r), 6) for r in transition["reward"])
        team_counts = (
            sum(1 for e in state["entities"] if int(e["team"]) == 0 and bool(e["alive"])),
            sum(1 for e in state["entities"] if int(e["team"]) == 1 and bool(e["alive"])),
        )
        hp_totals = (
            round(sum(float(e["hp"]) for e in state["entities"] if int(e["team"]) == 0 and bool(e["alive"])), 3),
            round(sum(float(e["hp"]) for e in state["entities"] if int(e["team"]) == 1 and bool(e["alive"])), 3),
        )

        trace.append(
            (
                round(float(state["sim_time_s"]), 4),
                rewards,
                bool(transition["done"]),
                bool(transition["truncation"]),
                team_counts,
                hp_totals,
                len(state["pending_spawns"]),
                len(state["fireballs"]),
                len(state["logs"]),
            )
        )

        if transition["done"] or transition["truncation"]:
            break

    return trace


def assert_transition_contract(transition: Dict[str, Any], spec: Dict[str, Any]) -> None:
    assert set(transition.keys()) == {"obs", "action_mask", "reward", "done", "truncation", "info"}
    assert len(transition["obs"]) == 2
    assert len(transition["action_mask"]) == 2
    assert len(transition["reward"]) == 2

    obs_schema = spec["obs_schema"]
    expected_vec = int(obs_schema["vector_size"])
    expected_cards = int(obs_schema["position_masks_cards"])
    expected_positions = int(obs_schema["position_masks_per_card"])
    expected_mask = int(spec["action_mask_size"])

    for i in range(2):
        obs_i = transition["obs"][i]
        assert len(obs_i["vector"]) == expected_vec
        assert len(obs_i["position_masks_for_all_cards"]) == expected_cards
        for card_mask in obs_i["position_masks_for_all_cards"]:
            assert len(card_mask) == expected_positions
        assert len(transition["action_mask"][i]) == expected_mask

        assert_finite_nested(obs_i["vector"])
        assert_finite_nested(obs_i["position_masks_for_all_cards"])
        assert_finite_nested(transition["action_mask"][i])
        assert math.isfinite(float(transition["reward"][i]))
