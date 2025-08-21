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
    """Configuration for a single reward component."""

    # dotted import path of the concrete reward class
    cls: str = "boxed_answer"
    # free-form kwargs passed directly to the constructor
    params: dict = field(default_factory=dict)
    # judge prompt template to use (only relevant for judge-based rewards)
    judge_prompt_template: Optional[str] = None

    # When True (default) the reward term contributes to optimisation via both
    # the hidden chain-of-thought and the visible answer.  When False, the
    # reward only influences gradients through the answer (output) tokens.
    apply_to_thinking: bool = True


@dataclass
class TrainConfig:
    """Top-level training configuration."""

    # model / tokenizer
    model_name: str = "Qwen/Qwen3-4B"
    enable_thinking: bool = True  # pass to tokenizer

    # prompt construction
    prompt_builder_cls: str = "src.prompt_builders.jsonl_prompt_builder.JsonlPromptBuilder"
    prompt_builder_params: dict = field(default_factory=dict)

    # RL specific
    batch_size: int = 2
    grad_accum_steps: int = 1
    # total number of RL optimisation steps (batches)
    num_episodes: int = 100
    learning_rate: float = 5e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    # thinking / output budgets
    thinking_max_tokens: Optional[int] = None
    thinking_min_tokens: int = 0
    output_max_tokens: Optional[int] = None  # enforced via generate(max_new_tokens)

    # reward setup – list allows arbitrarily many (task reward + penalties)
    rewards: List[RewardConfig] = field(default_factory=list)

    # logging / checkpointing
    wandb_project: str = "simple-rl-research"
    wandb_run_name: Optional[str] = None
    # When True, upload collected rollouts as a W&B artifact at the end of training.
    save_rollouts_to_wandb: bool = False

    # TODO: Future config options to consider adding:
    # - save_every_steps: Optional[int] = None  # for model checkpointing
    # - save_to_hf_hub: bool = False  # for uploading to Hugging Face Hub
    # - hf_repo: Optional[str] = None  # organisation/repo name when uploading
    # - max_grad_norm: float = 1.0  # for gradient clipping

    # multi-GPU
    multi_gpu: str = "none"  # "none" | "ddp" | "fsdp"

    # ------------------------------------------------------------------
    # mind / face dual-model training
    # ------------------------------------------------------------------
    # When True, *ReinforceTrainer* will load the *MindFace* wrapper (see
    # ``src.models.mind_face``) instead of a single language model.  Both the
    # mask and the face sub-models are included in the optimiser so their
    # parameters are updated jointly.
    use_mind_face: bool = False
    """When true, enables two-model 'mind-face' generation."""
    mind_model_name: Optional[str] = None
    """Name of the 'mind' model (HF hub or local path)."""
    face_model_name: Optional[str] = None
    """Name of the 'face' model (HF hub or local path)."""

    def __post_init__(self):
        # No additional post-initialisation logic required at present.
        pass
