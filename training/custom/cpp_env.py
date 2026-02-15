from __future__ import annotations

import importlib
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from gymnasium import spaces


class CppClashEnvBatch:
    EXPECTED_PROTOCOL_VERSION = "knockoff_env_v1"
    EXPECTED_OBS_VERSION = "v1"
    EXPECTED_SCHEMA_VERSION = "knockoff_cr_env_v2"

    def __init__(
        self,
        num_envs: int,
        tick_hz: int = 10,
        max_sim_seconds: float = 120.0,
        seed: int = 1,
        num_threads: int = 0,
    ):
        self.num_envs = max(1, int(num_envs))
        self.tick_hz = max(1, int(tick_hz))
        self.max_sim_seconds = float(max_sim_seconds)

        mod = importlib.import_module("knockoff_cr_cpp")
        self.core = mod.ClashEnvBatch(
            num_envs=self.num_envs,
            tick_hz=self.tick_hz,
            max_sim_seconds=self.max_sim_seconds,
            seed=int(seed),
            num_threads=int(num_threads),
        )

        self.spec_data: Dict[str, Any] = self.core.spec()
        self._validate_spec(self.spec_data)
        self.action_order = self.spec_data.get("action_order", [])
        self.agent_count = int(self.spec_data.get("n_agents", 1))
        self.agent_keys = [f"agent_{i}" for i in range(self.agent_count)]
        self.vector_size = int(self.spec_data.get("obs_schema", {}).get("vector_size", 0))
        self.action_mask_size = int(self.spec_data.get("action_mask_size", 0))
        self.position_masks_cards = int(self.spec_data.get("obs_schema", {}).get("position_masks_cards", 0))
        self.position_masks_per_card = int(self.spec_data.get("obs_schema", {}).get("position_masks_per_card", 0))

        action_space_spec = self.spec_data["action_space"]
        per_agent_action_space = spaces.Dict(
            {key: spaces.Discrete(int(action_space_spec[key]["size"])) for key in self.action_order}
        )
        self.action_space = spaces.Dict({agent_key: per_agent_action_space for agent_key in self.agent_keys})

    def reset_many(
        self,
        *,
        seeds: List[int | None] | None = None,
        options_per_env: List[Dict[str, Any] | None] | None = None,
    ) -> Tuple[List[Dict[str, Dict[str, np.ndarray]] | None], List[Dict[str, Any]]]:
        if seeds is not None and len(seeds) != self.num_envs:
            raise ValueError(f"seeds length mismatch expected={self.num_envs} got={len(seeds)}")
        if options_per_env is not None and len(options_per_env) != self.num_envs:
            raise ValueError(f"options_per_env length mismatch expected={self.num_envs} got={len(options_per_env)}")

        payloads = self.core.reset_many(
            seeds=seeds if seeds is not None else None,
            options_per_env=options_per_env if options_per_env is not None else None,
        )
        if not isinstance(payloads, list) or len(payloads) != self.num_envs:
            raise ValueError("reset_many payload count mismatch")

        obs_out: List[Dict[str, Dict[str, np.ndarray]] | None] = [None for _ in range(self.num_envs)]
        info_out: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]
        for env_idx, data in enumerate(payloads):
            if data is None:
                continue
            self._validate_transition_payload(data)
            obs_out[env_idx] = self._to_obs(data)
            info_out[env_idx] = dict(data.get("info", {}))
        return obs_out, info_out

    def step_many_packed(
        self,
        actions_per_env: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        arr = np.asarray(actions_per_env, dtype=np.int32)
        expected_shape = (self.num_envs, self.agent_count, len(self.action_order))
        if arr.shape != expected_shape:
            raise ValueError(f"actions_per_env shape mismatch expected={expected_shape} got={arr.shape}")

        t0 = time.perf_counter()
        payload = self.core.step_many_packed(actions_per_env=arr)
        roundtrip_s = max(0.0, time.perf_counter() - t0)
        if not isinstance(payload, dict):
            raise ValueError("step_many_packed returned non-dict payload")

        obs = np.asarray(payload.get("obs"), dtype=np.float32)
        action_mask = np.asarray(payload.get("action_mask"), dtype=np.float32)
        card_position_masks = np.asarray(payload.get("card_position_masks"), dtype=np.float32)
        reward = np.asarray(payload.get("reward"), dtype=np.float32)
        done = np.asarray(payload.get("done"), dtype=np.uint8).astype(np.bool_, copy=False)
        truncation = np.asarray(payload.get("truncation"), dtype=np.uint8).astype(np.bool_, copy=False)
        winner = np.asarray(payload.get("winner"), dtype=np.int8)
        self._validate_packed_payload_arrays(
            obs=obs,
            action_mask=action_mask,
            card_position_masks=card_position_masks,
            reward=reward,
            done=done,
            truncation=truncation,
            winner=winner,
        )
        return obs, action_mask, card_position_masks, reward, done, truncation, winner, float(roundtrip_s)

    def close(self) -> None:
        self.core = None

    def debug_state_many(self) -> List[Dict[str, Any]]:
        if self.core is None:
            raise RuntimeError("CppClashEnvBatch is closed")
        states = self.core.debug_state_many()
        if not isinstance(states, list):
            raise ValueError("debug_state_many returned non-list")
        return states

    def _to_obs(self, data: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
        result: Dict[str, Dict[str, np.ndarray]] = {}
        for i, agent_key in enumerate(self.agent_keys):
            raw_obs = data["obs"][i]
            vector = np.asarray(raw_obs.get("vector", []), dtype=np.float32)
            action_mask = np.asarray(data["action_mask"][i], dtype=np.float32)
            pos_masks = np.asarray(raw_obs.get("position_masks_for_all_cards", []), dtype=np.float32)
            result[agent_key] = {
                "vector": vector,
                "action_mask": action_mask,
                "position_masks_for_all_cards": pos_masks,
            }
        return result

    def _validate_spec(self, spec: Dict[str, Any]) -> None:
        if str(spec.get("protocol_version", "")) != self.EXPECTED_PROTOCOL_VERSION:
            raise ValueError(
                f"Protocol mismatch: expected={self.EXPECTED_PROTOCOL_VERSION} got={spec.get('protocol_version')}"
            )
        if str(spec.get("obs_version", "")) != self.EXPECTED_OBS_VERSION:
            raise ValueError(f"Obs version mismatch: expected={self.EXPECTED_OBS_VERSION} got={spec.get('obs_version')}")
        if str(spec.get("schema_version", "")) != self.EXPECTED_SCHEMA_VERSION:
            raise ValueError(
                f"Schema version mismatch: expected={self.EXPECTED_SCHEMA_VERSION} got={spec.get('schema_version')}"
            )
        action_order = spec.get("action_order", [])
        action_space = spec.get("action_space", {})
        if not isinstance(action_order, list) or len(action_order) == 0:
            raise ValueError("Spec missing non-empty action_order")
        for key in action_order:
            if key not in action_space:
                raise ValueError(f"Spec action_space missing key={key}")
            size = int(action_space[key].get("size", 0))
            if size <= 0:
                raise ValueError(f"Invalid action size for key={key}: {size}")
        vector_size = int(spec.get("obs_schema", {}).get("vector_size", 0))
        cards_count = int(spec.get("obs_schema", {}).get("position_masks_cards", 0))
        per_card_positions = int(spec.get("obs_schema", {}).get("position_masks_per_card", 0))
        mask_size = int(spec.get("action_mask_size", 0))
        if vector_size <= 0 or mask_size <= 0:
            raise ValueError(f"Invalid obs/mask sizes vector={vector_size} mask={mask_size}")
        if cards_count <= 0 or per_card_positions <= 0:
            raise ValueError(
                f"Invalid position mask schema cards={cards_count} per_card_positions={per_card_positions}"
            )

    def _validate_transition_payload(self, data: Dict[str, Any]) -> None:
        if "obs" not in data or "action_mask" not in data or "reward" not in data:
            raise ValueError("Transition payload missing required keys")
        obs = data["obs"]
        action_mask = data["action_mask"]
        reward = data["reward"]
        if not isinstance(obs, list) or len(obs) != self.agent_count:
            raise ValueError(f"obs agent count mismatch expected={self.agent_count}")
        if not isinstance(action_mask, list) or len(action_mask) != self.agent_count:
            raise ValueError(f"action_mask agent count mismatch expected={self.agent_count}")
        if not isinstance(reward, list) or len(reward) != self.agent_count:
            raise ValueError(f"reward agent count mismatch expected={self.agent_count}")

    def _validate_packed_payload_arrays(
        self,
        *,
        obs: np.ndarray,
        action_mask: np.ndarray,
        card_position_masks: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        truncation: np.ndarray,
        winner: np.ndarray,
    ) -> None:
        expected_obs = (self.num_envs, self.agent_count, self.vector_size)
        expected_action_mask = (self.num_envs, self.agent_count, self.action_mask_size)
        expected_card_masks = (
            self.num_envs,
            self.agent_count,
            self.position_masks_cards,
            self.position_masks_per_card,
        )
        expected_reward = (self.num_envs, self.agent_count)
        expected_flags = (self.num_envs,)

        if obs.shape != expected_obs:
            raise ValueError(f"packed obs shape mismatch expected={expected_obs} got={obs.shape}")
        if action_mask.shape != expected_action_mask:
            raise ValueError(
                f"packed action_mask shape mismatch expected={expected_action_mask} got={action_mask.shape}"
            )
        if card_position_masks.shape != expected_card_masks:
            raise ValueError(
                f"packed card_position_masks shape mismatch expected={expected_card_masks} got={card_position_masks.shape}"
            )
        if reward.shape != expected_reward:
            raise ValueError(f"packed reward shape mismatch expected={expected_reward} got={reward.shape}")
        if done.shape != expected_flags:
            raise ValueError(f"packed done shape mismatch expected={expected_flags} got={done.shape}")
        if truncation.shape != expected_flags:
            raise ValueError(f"packed truncation shape mismatch expected={expected_flags} got={truncation.shape}")
        if winner.shape != expected_flags:
            raise ValueError(f"packed winner shape mismatch expected={expected_flags} got={winner.shape}")
