"""Registry look-up tests."""

from simple_rl_research.reward.registry import get_reward_class
from simple_rl_research.reward.regex_penalty import RegexPenalty
from simple_rl_research.reward.judge_penalty import JudgePenalty


def test_registry_returns_correct_class():
    assert get_reward_class("regex_penalty") is RegexPenalty
    assert get_reward_class("judge_penalty") is JudgePenalty
