from __future__ import annotations

import pytest
from fractions import Fraction

from src.reward.expanded_polynomial_reward import ExpandedPolynomialReward


def make_rollout(k: int, poly_coeffs: list[str], final: str):
    return {
        "k": k,
        "poly_coeffs": poly_coeffs,
        "response": final,
    }


def test_expanded_polynomial_reward_simple_match():
    # f(x) = (1/2)x^2 -> k=1 derivative = x
    rollout = make_rollout(1, ["0", "0", "1/2"], "Answer: x")
    r = ExpandedPolynomialReward()
    val = r(rollout)
    assert val >= 1.0
    assert rollout["reward_breakdown"]["expanded_polynomial_reward"] >= 1.0


@pytest.mark.parametrize(
    "coeffs,k,expect_strs",
    [
        (["0", "0", "3/2"], 1, ["3x", "3*x", r"3 \\cdot x", "3 x"]),  # f = 3/2 x^2 -> f' = 3x
        (["0", "0", "1"], 2, ["2", "$ 2 $", "(2)"]),             # f = x^2 -> f'' = 2
        (["0", "1", "0", "0"], 1, ["1", "(1)", "$1$"]),        # f = x -> f' = 1
    ],
)
def test_expanded_polynomial_reward_variants(coeffs, k, expect_strs):
    r = ExpandedPolynomialReward()
    for s in expect_strs:
        rollout = make_rollout(k, coeffs, f"Here is the result: {s}.")
        val = r(rollout)
        assert val >= 1.0


