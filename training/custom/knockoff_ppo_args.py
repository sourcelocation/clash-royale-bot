"""CLI/config dataclass for Knockoff PPO training."""

from dataclasses import dataclass


@dataclass
class Args:
    exp_name: str = "train_ppo"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=True`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    mps: bool = True
    """if toggled, Apple Metal (MPS) will be enabled when CUDA is unavailable"""
    prefer_cpu_for_small_batches: bool = True
    """fallback to CPU when MPS would be used with very small PPO batches"""
    small_batch_threshold: int = 1024
    """maximum PPO batch_size considered "small" for MPS fallback"""

    # CleanRL PPO core arguments
    total_timesteps: int = 100000
    learning_rate: float = 2.5e-4
    num_envs: int = 8
    num_steps: int = 16
    anneal_lr: bool = True
    gamma: float = 0.995
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    # Knockoff runtime/env arguments
    cpp_tick_hz: int = 10
    cpp_max_sim_seconds: float = 120.0
    cpp_num_threads: int = 0
    run_dir: str = "training/logs/custom/run"
    tb_dir: str = ""
    log_every: int = 1000
    flush_every_logs: int = 10
    ckpt_every: int = 5000
    resume_latest: bool = False
    team0_controller: str = "external"
    team1_controller: str = "selfplay_pool"
    training_mode: bool = True
    hidden_sizes: str = "256,256"
    selfplay_enabled: bool = True
    selfplay_recent_capacity: int = 32
    selfplay_anchor_capacity: int = 8
    selfplay_anchor_every: int = 4
    selfplay_latest_prob: float = 0.4
    selfplay_recent_prob: float = 0.4
    selfplay_anchor_prob: float = 0.2
    selfplay_deterministic: bool = True
    selfplay_max_cached_policies: int = 6
    selfplay_elo_enabled: bool = True
    selfplay_elo_initial: float = 1000.0
    selfplay_elo_k: float = 16.0
    visualize: bool = False
    visualize_fps: int = 60
