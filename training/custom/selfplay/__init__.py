"""Self-play opponent pool components."""

from .opponent_policy import OpponentModelShape, OpponentPolicyCache
from .pool_manager import SelfPlayPoolConfig, SelfPlayPoolManager
from .rating_tracker import EloRatingTracker, EloRatingTrackerConfig
from .types import OpponentCheckpoint, OpponentSelection

__all__ = [
    "EloRatingTracker",
    "EloRatingTrackerConfig",
    "OpponentCheckpoint",
    "OpponentModelShape",
    "OpponentPolicyCache",
    "OpponentSelection",
    "SelfPlayPoolConfig",
    "SelfPlayPoolManager",
]
