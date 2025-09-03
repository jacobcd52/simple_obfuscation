from __future__ import annotations

import pytest

from src.reward.factored_polynomial_reward import (
    FactoredPolynomialReward,
)


def make_rollout(answer: str, target: str):
    return {
        "prompt": "",
        "response": f"### Answer\n\\boxed{{{answer}}}\n\n### Explanation\n...",
        "target": target,
    }


@pytest.mark.parametrize(
    "answer,target,expect",
    [
        ("(x-1)(x+2)", "1*(x-1)*(x+2)", 1.0),
        ("(x - 1)(x + 2)", "1*(x-1)*(x+2)", 1.0),
        ("(x-1)*(x+2)", "1*(x-1)*(x+2)", 1.0),
        ("x(x-5)", "1*(x)*(x-5)", 1.0),
        ("-1*(2-x)", "1*(x-2)", 1.0),  # -1*(2-x) has roots {2}
        ("(3-x)", "1*(x-3)", 1.0),     # (3-x) has root {3}
        ("(x-4)(x+1)(x-3)(4x+1)", "1*(x-4)*(x+1)*(x-3)*(4x+1)", 1.0),
        ("(x-4)(x+1)(x-3)", "1*(x-4)*(x+1)*(x-3)*(4x+1)", 0.0),  # missing factor
    ],
)
def test_factored_polynomial_reward_matches(answer, target, expect):
    r = FactoredPolynomialReward()
    rollout = make_rollout(answer, target)
    val = r(rollout)
    assert rollout["reward_breakdown"]["factored_polynomial_reward"] == expect
    assert val == expect


