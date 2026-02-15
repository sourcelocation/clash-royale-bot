"""Deprecated Godot socket transport client.

Training now uses the in-process cpp backend batch API and does not use this transport.
"""

import json
import socket
import time
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class EnvClientConfig:
    host: str = "127.0.0.1"
    port: int = 12000
    timeout_s: float = 5.0


class EnvClient:
    def __init__(self, config: EnvClientConfig):
        self.config = config
        self._sock: socket.socket | None = None
        self._seq = 0
        self._buffer = ""

    def connect(self) -> None:
        self._sock = socket.create_connection((self.config.host, self.config.port), timeout=self.config.timeout_s)
        self._sock.settimeout(self.config.timeout_s)

    def close(self) -> None:
        if not self._sock:
            return
        try:
            self._send({"type": "close", "seq": self._next_seq()})
        except OSError:
            pass
        try:
            self._sock.close()
        finally:
            self._sock = None

    def hello(self) -> Dict[str, Any]:
        return self._roundtrip({"type": "hello", "seq": self._next_seq()})

    def spec(self) -> Dict[str, Any]:
        return self._roundtrip({"type": "spec", "seq": self._next_seq()})

    def reset(self, seed: int = -1, options: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return self._roundtrip(
            {
                "type": "reset",
                "seq": self._next_seq(),
                "seed": seed,
                "options": options or {},
            }
        )

    def step(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self._roundtrip(
            {
                "type": "step",
                "seq": self._next_seq(),
                "actions": actions,
            }
        )

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _roundtrip(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._send(payload)
        return self._recv_response(expected_seq=payload["seq"])

    def _send(self, payload: Dict[str, Any]) -> None:
        if not self._sock:
            raise RuntimeError("EnvClient is not connected")
        wire = json.dumps(payload, separators=(",", ":")) + "\n"
        self._sock.sendall(wire.encode("utf-8"))

    def _recv_response(self, expected_seq: int) -> Dict[str, Any]:
        deadline = time.time() + self.config.timeout_s
        while time.time() < deadline:
            line = self._recv_line()
            if not line:
                continue
            msg = json.loads(line)
            msg_seq = int(msg.get("seq", -1))
            if msg_seq != expected_seq:
                continue
            if msg.get("type") == "error":
                raise RuntimeError(f"Env returned error: {msg}")
            return msg
        raise TimeoutError(f"Timed out waiting for response seq={expected_seq}")

    def _recv_line(self) -> str:
        if not self._sock:
            raise RuntimeError("EnvClient is not connected")

        while "\n" not in self._buffer:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise ConnectionError("Connection closed by environment")
            self._buffer += chunk.decode("utf-8")

        line, self._buffer = self._buffer.split("\n", 1)
        return line.strip()
