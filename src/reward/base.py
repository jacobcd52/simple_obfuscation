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

    def _log_reward_values(self, rollout: Dict, raw_value: float, *, on_thinking: bool = False) -> float:
        """Apply post-processing, log both raw and processed, and return final value.

        This centralises consistent logging so that both raw and post-processed
        rewards are always available to the training loop (and therefore wandb).

        Parameters
        ----------
        rollout
            The rollout dictionary which will receive entries in
            ``rollout['reward_breakdown']``.
        raw_value
            The unscaled, unclipped reward value computed by the concrete
            subclass.
        on_thinking
            When True, logs under the ``*_thinking`` key variant.

        Returns
        -------
        float
            The value to contribute to optimisation (0.0 when ``log_only`` is True).
        """
        processed = self._post_process(raw_value)
        key = self.name if not on_thinking else f"{self.name}_thinking"

        breakdown = rollout.setdefault("reward_breakdown", {})
        breakdown[key] = processed
        breakdown[f"{key}_raw"] = raw_value

        return 0.0 if self.log_only else processed
