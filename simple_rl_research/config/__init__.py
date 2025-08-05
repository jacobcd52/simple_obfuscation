"""Configuration objects used across the code-base.

We rely on *OmegaConf* under the hood but expose strongly-typed
``dataclass`` wrappers so that downstream code enjoys static checking and
IDE completion.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Type

__all__ = [
    "TrainConfig",
    "RewardConfig",
]


@dataclass
class RewardConfig:
    """Configuration for a single reward (or penalty) component."""

    # dotted import path of the concrete reward class, e.g.
    # ``simple_rl_research.reward.regex_penalty.RegexPenalty``
    cls: str
    # free-form kwargs passed directly to the constructor
    params: dict = field(default_factory=dict)


@dataclass
class TrainConfig:
    """Top-level training configuration."""

    # ----------------------------------------------------------------------------
    # model / tokeniser
    # ----------------------------------------------------------------------------
    model_name: str = "Qwen/Qwen1_5-4B"

    # Padding must be *left* for causal models when using generate with max_length
    padding_side: str = "left"
    enable_thinking: bool = True  # pass to tokenizer

    # ----------------------------------------------------------------------------
    # prompt construction
    # ----------------------------------------------------------------------------
    prompt_builder_cls: str = "simple_rl_research.generation.prompt_builder.JsonlPromptBuilder"
    prompt_builder_params: dict = field(default_factory=dict)

    # ----------------------------------------------------------------------------
    # RL specific
    # ----------------------------------------------------------------------------
    batch_size: int = 2
    grad_accum_steps: int = 1
    epochs: int = 1
    learning_rate: float = 5e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    # thinking / output budgets
    thinking_max_tokens: Optional[int] = None
    thinking_min_tokens: int = 0
    output_max_tokens: Optional[int] = None  # enforced via generate(max_new_tokens)

    # reward setup – list allows arbitrarily many (task reward + penalties)
    rewards: List[RewardConfig] = field(default_factory=list)

    # ----------------------------------------------------------------------------
    # logging / checkpointing
    # ----------------------------------------------------------------------------
    wandb_project: str = "simple-rl-research"
    wandb_run_name: Optional[str] = None

    save_every_steps: Optional[int] = None  # disabled by default
    save_to_hf_hub: bool = False
    hf_repo: Optional[str] = None  # organisation/repo name when uploading

    # ----------------------------------------------------------------------------
    # multi-GPU
    # ----------------------------------------------------------------------------
    multi_gpu: str = "none"  # "none" | "ddp" | "fsdp"
