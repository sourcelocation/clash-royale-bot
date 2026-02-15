"""Deprecated Godot socket env.

Training has hard-cut over to cpp batch backend (`CppClashEnvBatch`).
This module is retained only for short-term rollback/debug.
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .env_client import EnvClient, EnvClientConfig


class GodotClashEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}
    EXPECTED_PROTOCOL_VERSION = "knockoff_env_v1"
    EXPECTED_OBS_VERSION = "v1"
    EXPECTED_SCHEMA_VERSION = "knockoff_cr_env_v2"

    def __init__(
        self,
        godot_path: str,
        project_path: str,
        env_id: int,
        port: int,
        visible: bool = False,
        engine_time_scale: float = 1.0,
        physics_tps: int = 60,
        logs_dir: str = "training/logs/custom",
        connect_timeout_s: float = 20.0,
    ):
        super().__init__()
        self.godot_path = self._resolve_godot_binary(godot_path)
        self.project_path = project_path
        self.env_id = env_id
        self.port = port
        self.visible = visible
        self.engine_time_scale = float(engine_time_scale)
        self.physics_tps = int(physics_tps)
        self.logs_dir = Path(logs_dir)
        self.connect_timeout_s = connect_timeout_s

        self.process: subprocess.Popen | None = None
        self.log_handle = None
        self.client: EnvClient | None = None

        self.spec_data: Dict[str, Any] = {}
        self.action_order: list[str] = []
        self.agent_count: int = 1
        self.agent_keys: list[str] = []

        self.observation_space: spaces.Space
        self.action_space: spaces.Space

        self._start_worker()
        self._connect_client()
        self._load_spec_and_spaces()

    def _resolve_godot_binary(self, configured_path: str) -> str:
        path = Path(configured_path)
        if path.suffix != ".app":
            return str(path)

        candidates = [
            path / "Contents" / "MacOS" / "Redot",
            path / "Contents" / "MacOS" / "Godot",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        raise FileNotFoundError(
            f"Could not resolve engine binary inside app bundle: {configured_path}"
        )

    def _start_worker(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.logs_dir / f"env_{self.env_id:02d}.log"
        self.log_handle = open(log_path, "w", encoding="utf-8")

        cmd = [
            self.godot_path,
            "--path",
            self.project_path,
            "--",
            "--n=1",
            f"--port={self.port}",
            f"--env_id={self.env_id}",
            f"--engine_time_scale={self.engine_time_scale}",
            f"--physics_tps={self.physics_tps}",
        ]
        if not self.visible:
            cmd.insert(1, "--headless")

        self.process = subprocess.Popen(
            cmd,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            cwd=self.project_path,
            env=os.environ.copy(),
        )
        print(
            f"[GymEnv] Spawned env_id={self.env_id} port={self.port} pid={self.process.pid} "
            f"visible={self.visible}"
        )

    def _connect_client(self) -> None:
        deadline = time.time() + self.connect_timeout_s
        # Step roundtrips can occasionally exceed a couple seconds during
        # transient scheduler stalls; keep the socket timeout less fragile.
        self.client = EnvClient(EnvClientConfig(port=self.port, timeout_s=10.0))

        while True:
            try:
                self.client.connect()
                hello = self.client.hello()
                print(
                    f"[GymEnv] Connected env_id={self.env_id} protocol={hello.get('protocol_version', 'unknown')}"
                )
                return
            except Exception:
                if time.time() > deadline:
                    raise TimeoutError(
                        f"Timed out connecting to env_id={self.env_id} port={self.port}"
                    )
                time.sleep(0.25)

    def _load_spec_and_spaces(self) -> None:
        assert self.client is not None
        spec_msg = self.client.spec()
        self.spec_data = spec_msg["spec"]
        self._validate_spec(self.spec_data)
        self.action_order = self.spec_data.get("action_order", [])
        self.agent_count = int(self.spec_data.get("n_agents", 1))
        self.agent_keys = [f"agent_{i}" for i in range(self.agent_count)]

        action_space_spec = self.spec_data["action_space"]
        per_agent_action_space = spaces.Dict(
            {key: spaces.Discrete(int(action_space_spec[key]["size"])) for key in self.action_order}
        )
        self.action_space = spaces.Dict(
            {agent_key: per_agent_action_space for agent_key in self.agent_keys}
        )

        vector_size = int(self.spec_data.get("obs_schema", {}).get("vector_size", 0))
        action_mask_size = int(self.spec_data.get("action_mask_size", 0))
        per_agent_obs_space = spaces.Dict(
            {
                "vector": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(vector_size,),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(action_mask_size,),
                    dtype=np.float32,
                ),
                "position_masks_for_all_cards": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(
                        int(self.spec_data.get("obs_schema", {}).get("position_masks_cards", 0)),
                        int(self.spec_data.get("obs_schema", {}).get("position_masks_per_card", 0)),
                    ),
                    dtype=np.float32,
                ),
            }
        )
        self.observation_space = spaces.Dict(
            {agent_key: per_agent_obs_space for agent_key in self.agent_keys}
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Any]]:
        super().reset(seed=seed)
        assert self.client is not None
        msg = self.client.reset(seed=-1 if seed is None else int(seed), options=options)
        data = msg["data"]
        self._validate_transition_payload(data)
        obs = self._to_obs(data)
        info = data.get("info", {})
        return obs, info

    def step(self, action: Dict[str, Any]):
        assert self.client is not None
        godot_actions = []
        for agent_key in self.agent_keys:
            agent_action = action[agent_key]
            godot_actions.append({key: int(agent_action[key]) for key in self.action_order})

        t0 = time.perf_counter()
        msg = self.client.step(actions=godot_actions)
        roundtrip_s = max(0.0, time.perf_counter() - t0)
        data = msg["data"]
        self._validate_transition_payload(data)

        obs = self._to_obs(data)
        reward = np.asarray(data["reward"], dtype=np.float32)
        terminated = bool(data["done"])
        truncated = bool(data["truncation"])
        info = dict(data.get("info", {}))
        info["py_roundtrip_s"] = float(roundtrip_s)
        ticks_requested = int(info.get("ticks_requested", 0))
        if ticks_requested > 0:
            budget_wall_s = float(ticks_requested) / (
                max(1, int(self.physics_tps)) * max(0.01, float(self.engine_time_scale))
            )
            info["py_budget_wall_s"] = float(budget_wall_s)
            info["py_late"] = bool(roundtrip_s > budget_wall_s)
            info["py_late_ratio"] = float(roundtrip_s / max(1e-9, budget_wall_s))
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None

        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None

        if self.log_handle:
            self.log_handle.close()
            self.log_handle = None

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
            raise ValueError(
                f"Obs version mismatch: expected={self.EXPECTED_OBS_VERSION} got={spec.get('obs_version')}"
            )
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

        expected_vec = int(self.spec_data.get("obs_schema", {}).get("vector_size", 0))
        expected_cards = int(self.spec_data.get("obs_schema", {}).get("position_masks_cards", 0))
        expected_per_card = int(self.spec_data.get("obs_schema", {}).get("position_masks_per_card", 0))
        expected_mask = int(self.spec_data.get("action_mask_size", 0))
        for i in range(self.agent_count):
            agent_obs = obs[i]
            if not isinstance(agent_obs, dict):
                raise ValueError(f"obs[{i}] must be a dictionary")
            vec = agent_obs.get("vector", [])
            if not isinstance(vec, list):
                raise ValueError(f"obs[{i}].vector must be a list")
            if expected_vec > 0 and len(vec) != expected_vec:
                raise ValueError(
                    f"obs[{i}].vector size mismatch expected={expected_vec} got={len(vec)}"
                )
            pos_masks = agent_obs.get("position_masks_for_all_cards", [])
            if not isinstance(pos_masks, list):
                raise ValueError(f"obs[{i}].position_masks_for_all_cards must be a list")
            if expected_cards > 0 and len(pos_masks) != expected_cards:
                raise ValueError(
                    f"obs[{i}].position_masks_for_all_cards card count mismatch "
                    f"expected={expected_cards} got={len(pos_masks)}"
                )
            for c_idx, mask_vec in enumerate(pos_masks):
                if not isinstance(mask_vec, list):
                    raise ValueError(
                        f"obs[{i}].position_masks_for_all_cards[{c_idx}] must be a list"
                    )
                if expected_per_card > 0 and len(mask_vec) != expected_per_card:
                    raise ValueError(
                        f"obs[{i}].position_masks_for_all_cards[{c_idx}] size mismatch "
                        f"expected={expected_per_card} got={len(mask_vec)}"
                    )
            mask = action_mask[i]
            if not isinstance(mask, list):
                raise ValueError(f"action_mask[{i}] must be a list")
            if expected_mask > 0 and len(mask) != expected_mask:
                raise ValueError(
                    f"action_mask[{i}] size mismatch expected={expected_mask} got={len(mask)}"
                )
