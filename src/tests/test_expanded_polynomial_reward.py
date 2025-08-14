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


def test_expanded_polynomial_reordered_terms_detected():
    # f(x) = (1/2)x^3 - (3/2)x^2 + 4x - 5 -> f'(x) = (3/2)x^2 - 3x + 4
    coeffs = ["-5", "4", "-3/2", "1/2"]
    r = ExpandedPolynomialReward()
    candidates = [
        "(3/2)x^2 - 3x + 4",
        "4 - 3x + (3/2)x^2",
        "- 3x + (3/2)x^2 + 4",
        "$ (3/2)x^{2} - 3x + 4 $",
        r"(3/2) x^{2} - 3 \cdot x + 4",
    ]
    for s in candidates:
        rollout = make_rollout(1, coeffs, f"Answer: {s}")
        val = r(rollout)
        assert val == 1.0


def test_expanded_polynomial_constant_zero_derivative():
    # f(x) = 7 -> f'(x) = 0
    r = ExpandedPolynomialReward()
    coeffs = ["7"]
    for s in ["0", "$0$", "(0)"]:
        rollout = make_rollout(1, coeffs, f"Result: {s}")
        assert r(rollout) == 1.0


def test_expanded_polynomial_star_and_brace_power():
    # f(x) = x^3 -> f'(x) = 3x^2
    r = ExpandedPolynomialReward()
    coeffs = ["0", "0", "0", "1"]
    for s in ["3*x^2", "3 * x^{2}", r"3 \cdot x^{2}"]:
        rollout = make_rollout(1, coeffs, f"Result: {s}")
        assert r(rollout) == 1.0


def test_expanded_polynomial_no_match_returns_zero():
    r = ExpandedPolynomialReward()
    rollout = make_rollout(1, ["0", "1"], "No math here.")
    assert r(rollout) == 0.0


def test_expanded_polynomial_detects_in_narrative_paragraph():
    # f(x) = -3/5 x^5 - 15/4 x^4 -> f'(x) = -3x^4 - 15x^3
    coeffs = ["0", "0", "0", "0", "-15/4", "-3/5"]
    paragraph = (
        "The derivative of -3/5x^5 is -3/5 * 5x^4, which simplifies to -3x^4. "
        "Then the derivative of -15/4x^4 is -15/4 * 4x^3, which simplifies to -15x^3. "
        "So combining those, the first derivative is -3x^4 -15x^3."
    )
    r = ExpandedPolynomialReward()
    rollout = make_rollout(1, coeffs, paragraph)
    assert r(rollout) == 1.0


