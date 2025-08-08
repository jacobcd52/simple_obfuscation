"""RewardFunction base class used by all rewards / penalties."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any


class RewardFunction(ABC):
    """Abstract interface for computing (scalar) rewards from a rollout."""

    #: key under which the concrete reward logs its value (e.g. wandb)
    name: str = "reward"

    def __init__(
        self,
        *,
        coefficient: float = 1.0,
        max_clip: float | None = None,
        log_thinking: bool = False,
        log_only: bool = False,
    ) -> None:
        """Create a new reward instance.

        Parameters
        ----------
        coefficient
            Scalar applied to the raw reward.  Use a negative value to turn a
            *reward* into a *penalty* via sign flip.
        max_clip
            If given, the reward's absolute value is clipped to this bound *after*
            applying ``coefficient``.
        log_thinking
            If ``True``, the reward will *also* be computed on the model's
            <think>…</think> content for logging.  The logged value is stored in
            ``rollout['reward_breakdown']`` under the key ``f"{self.name}_thinking"``.
        log_only
            If ``True``, the reward will be computed and logged but its return
            value will be zeroed by subclasses so it does not contribute to the
            optimisation objective.
        """
        self.coefficient = coefficient
        self.max_clip = max_clip
        self.log_thinking = log_thinking
        self.log_only = log_only

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
