"""Registry look-up tests."""

from src.reward.registry import get_reward_class
from src.reward.regex_penalty import RegexReward
from src.reward.judge_penalty import JudgeReward


def test_registry_returns_correct_class():
    assert get_reward_class("regex_reward") is RegexReward
    assert get_reward_class("judge_reward") is JudgeReward
