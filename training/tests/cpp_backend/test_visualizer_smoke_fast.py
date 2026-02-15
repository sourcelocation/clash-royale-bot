from __future__ import annotations

import pytest


def test_visualizer_headless_smoke(monkeypatch):
    pytest.importorskip("pygame")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")

    from training.tools.cpp_visualizer import run_visualizer

    run_visualizer(
        max_sim_seconds=5.0,
        seed=123,
        fps=10,
        decision_hz=1.0,
        max_frames=8,
        headless=True,
    )
