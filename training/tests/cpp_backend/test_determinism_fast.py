from __future__ import annotations

from .conftest import rollout_fingerprint


def test_same_seed_same_trace(env_factory):
    env_a = env_factory(seed=999)
    env_b = env_factory(seed=999)
    try:
        trace_a = rollout_fingerprint(env_a, seed=2024, steps=180)
        trace_b = rollout_fingerprint(env_b, seed=2024, steps=180)
        assert trace_a == trace_b
    finally:
        del env_a
        del env_b


def test_different_seed_diverges(env_factory):
    env_a = env_factory(seed=7)
    env_b = env_factory(seed=8)
    try:
        trace_a = rollout_fingerprint(env_a, seed=100, steps=140)
        trace_b = rollout_fingerprint(env_b, seed=101, steps=140)
        assert trace_a != trace_b
    finally:
        del env_a
        del env_b
