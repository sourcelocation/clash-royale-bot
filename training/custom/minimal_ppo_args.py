"""Minimal PPO config for the rewritten training pipeline."""

from dataclasses import dataclass


@dataclass
class Args:
    exp_name: str = "train_ppo_minimal"
    seed: int = 1
    cuda: bool = True
    mps: bool = True
    torch_deterministic: bool = False

    total_timesteps: int = 1_000_000
    learning_rate: float = 2.5e-4
    num_envs: int = 32
    num_steps: int = 48
    num_minibatches: int = 8
    update_epochs: int = 4
    anneal_lr: bool = True
    gamma: float = 0.995
    gae_lambda: float = 0.95
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    hidden_sizes: str = "256,256"
    cpp_tick_hz: int = 10
    ticks_per_decision: int = 5
    cpp_max_sim_seconds: float = 120.0
    cpp_num_threads: int = 0
    run_dir: str = "training/logs/custom/minimal"
    tb_dir: str = ""
    log_every: int = 10
    ckpt_every: int = 100
    visualize: bool = False
    visualize_fps: int = 60
    sim_speed_mode: str = "max"  # one of: max, 1x

    pool_enabled: bool = True
    pool_recent_capacity: int = 32
    pool_anchor_capacity: int = 8
    pool_active_recent_size: int = 5
    pool_active_anchor_size: int = 2
    pool_promote_count: int = 32
    pool_refresh_count: int = 100
    pool_anchor_every: int = 4
    pool_latest_latest_prob: float = 0.3
    pool_latest_recent_prob: float = 0.55
    pool_latest_anchor_prob: float = 0.15
    pool_deterministic: bool = True

    benchmark_enabled: bool = True
    benchmark_initial: float = 1000.0
    benchmark_k: float = 8.0
    benchmark_ema_alpha: float = 0.1
