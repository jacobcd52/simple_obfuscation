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


# NEW TEST: detection of -x^3 + 8x^2 - 16x within explanatory text

def test_expanded_polynomial_detects_negative_cubic_derivative_in_narrative():
    """Verify that the reward detects the derivative -x^3 + 8x^2 - 16x when it is embedded
    in a long explanatory paragraph.
    The source polynomial is -1/4 x^4 + 8/3 x^3 - 8 x^2, whose first derivative is
    -x^3 + 8x^2 - 16x.
    """
    coeffs = ["0", "0", "-8", "8/3", "-1/4"]  # ascending order: constant, x, x^2, x^3, x^4
    narrative = NARRATIVE
    r = ExpandedPolynomialReward()
    rollout = make_rollout(1, coeffs, narrative)
    assert r(rollout) == 1.0


NARRATIVE = """Okay, let's see. I need to find the factored form of the first derivative of the polynomial -1/4x^4 + 8/3x^3 - 8x^2. Hmm. Alright, first step is probably to take the derivative. Let me remember how to do that. The derivative of x^n is n*x^(n-1), right? So for each term, I can apply that rule.

Starting with the first term: -1/4x^4. The derivative would be -1/4 * 4x^3. The 4 and the 1/4 cancel out, so that becomes -x^3.

Next term: 8/3x^3. The derivative is 8/3 * 3x^2. Again, the 3 cancels with the denominator, so that's 8x^2.

Third term: -8x^2. The derivative is -8 * 2x, which is -16x.

So putting it all together, the first derivative is -x^3 + 8x^2 -16x. Now I need to factor this expression. Let me write it as -x^3 + 8x^2 -16x. Maybe I can factor out a common term first. All the terms have an x, so factoring out an x gives x(-x^2 + 8x -16). But the coefficient of x^2 inside the parentheses is negative. Maybe I can factor out a negative sign as well. Let me see:

x(-x^2 +8x -16) = x*(-1)(x^2 -8x +16) = -x(x^2 -8x +16). Now, looking at the quadratic x^2 -8x +16. Is that factorable? Let's check the discriminant: b²-4ac = 64 - 64 = 0. Oh, so it's a perfect square. Therefore, x^2 -8x +16 factors into (x -4)^2. So substituting back, we have -x*(x -4)^2. But the question says to present the factored form. However, sometimes people prefer not to have a negative sign in front. Let me check if I can adjust that.

Alternatively, maybe I made a mistake in the sign when factoring. Let me go back. Original derivative was -x^3 +8x^2 -16x. Factoring out an x gives x(-x^2 +8x -16). Alternatively, I could factor out a -x instead. Let's try that: -x(x^2 -8x +16). Which is the same as before. So that gives -x(x -4)^2. But the original problem might expect a certain format. Let me check if there's another way. Alternatively, if I factor out a negative sign from the entire expression. The original derivative is -x^3 +8x^2 -16x. If I factor out a negative sign, it's - (x^3 -8x^2 +16x). Then factor out an x: -x(x^2 -8x +16) = -x(x -4)^2. Same result. 

But maybe the answer expects it without the negative sign. Let me check if there's a different approach. Wait, when I took the derivative, maybe I made a mistake? Let me double-check the derivative. Original polynomial: -1/4x^4 +8/3x^3 -8x^2. Derivative term by term:

First term: d/dx (-1/4 x^4) = -1/4 *4x^3 ="""