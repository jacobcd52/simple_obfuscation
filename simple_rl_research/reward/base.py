"""RewardFunction base class used by all rewards / penalties."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any


class RewardFunction(ABC):
    """Abstract interface for computing (scalar) rewards from a rollout."""

    #: key under which the concrete reward logs its value (e.g. wandb)
    name: str = "reward"

    def __init__(self, coefficient: float = 1.0, max_clip: float | None = None):
        self.coefficient = coefficient
        self.max_clip = max_clip

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @abstractmethod
    def __call__(self, rollout: Dict) -> Any:  # noqa: D401 – imperative style
        """Compute reward value for *single* rollout.
        
        Can be implemented as either sync (returning float) or async (returning asyncio.Future[float]).
        """

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _post_process(self, value: float) -> float:
        value *= self.coefficient
        if self.max_clip is not None:
            value = max(min(value, self.max_clip), -self.max_clip)
        return value
