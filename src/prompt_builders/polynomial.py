"""PromptBuilder for polynomial k-th derivative tasks with factored targets.

Task per PDF spec (Sec. 7 + Appendix):
- Start from a factored expression representing the k-th derivative f^{(k)}(x)
  as a product of linear factors with an overall coefficient.
- Integrate this expression k times to obtain f(x) which is presented to the
  assistant in expanded form. The assistant must output the k-th derivative in
  canonical factored form inside a LaTeX box.

Yielded fields:
- ``prompt``: problem statement with f(x)
- ``k``: order of derivative required
- ``target``: canonical factored form string for f^{(k)}(x) (without LaTeX)
- ``poly_coeffs``: coefficients of f(x) (ascending) for reference/analysis
"""

from __future__ import annotations

import random
from typing import Dict, Iterator, List, Tuple

from .base import PromptBuilder
from ..utils.polynomial import (
    integrate_coefficients,
    format_polynomial,
    poly_from_linear_factors,
    format_factored,
)

__all__ = ["PolynomialDerivativePromptBuilder"]


class PolynomialDerivativePromptBuilder(PromptBuilder):
    """Generate k-th derivative tasks from factored base derivative.

    Sampling process:
      1) Sample degree deg (>= k+1). Choose roots for a polynomial of degree
         deg - k representing f^{(k)}(x).
      2) Build factored canonical target for f^{(k)}(x).
      3) Integrate coefficients k times to produce f(x); present f(x) expanded.
    """

    def __init__(
        self,
        *,
        max_degree: int = 6,
        min_degree: int | None = None,
        k_min: int = 1,
        k_max: int = 3,
        root_range: int = 5,
        coef_range: int = 5,
        multiplicity_max: int = 2,
        num_nonzero_roots: int | None = None,
        seed: int | None = 0,
    ) -> None:
        super().__init__()
        if max_degree < 2:
            raise ValueError("max_degree must be >= 2")
        if min_degree is not None and min_degree < 2:
            raise ValueError("min_degree must be >= 2 when provided")
        if min_degree is not None and min_degree > max_degree:
            raise ValueError("min_degree cannot exceed max_degree")
        self.max_degree = max_degree
        self.min_degree = min_degree
        self.k_min = max(1, k_min)
        self.k_max = max(self.k_min, k_max)
        self.root_range = max(1, root_range)
        self.coef_range = max(1, coef_range)
        self.multiplicity_max = max(1, multiplicity_max)
        if num_nonzero_roots is not None and num_nonzero_roots < 0:
            raise ValueError("num_nonzero_roots must be >= 0 when provided")
        self.num_nonzero_roots = num_nonzero_roots
        if seed is not None:
            random.seed(seed)

    def _sample_k_and_roots(self) -> tuple[int, List[int], float]:
        k = random.randint(self.k_min, self.k_max)
        # choose total degree >= k+1 to ensure non-trivial derivative
        if self.min_degree is None:
            deg = random.randint(k + 1, self.max_degree)
        else:
            deg = random.randint(max(k + 1, self.min_degree), self.max_degree)
        # f^{(k)} degree is deg - k
        target_deg = deg - k
        roots: List[int] = []
        while len(roots) < target_deg:
            r = random.randint(-self.root_range, self.root_range)
            if r == 0 and random.random() < 0.2:
                # allow zero roots but not too often
                pass
            # sample multiplicity but cap remaining degree budget
            mult = random.randint(1, self.multiplicity_max)
            mult = min(mult, target_deg - len(roots))
            roots.extend([r] * mult)
        # overall coefficient for f^{(k)}(x) non-zero
        coef = 0
        while coef == 0:
            coef = random.randint(-self.coef_range, self.coef_range)
        return k, roots, float(coef)

    def reformat_prompt(self, prompt: str) -> str:
        instruction = """You will be given a polynomial and asked to find the factored form of its k-th derivative, for some k.
In your output, immediately write your answer as a LaTeX boxed expression.
Your answer must be in this format so that the verifier can check your answer:
### Answer
\\boxed{{your_factored_derivative_here}}

"""
        return instruction + prompt + HINT

    def generate(self) -> Iterator[Dict]:
        while True:
            k, roots, coef = self._sample_k_and_roots()
            # If num_nonzero_roots is specified, enforce exactly that many non-zero roots in f^{(k)}(x).
            # Otherwise, keep the existing behavior of a single repeated root.
            if self.num_nonzero_roots is None:
                # Existing behavior: monomial factor form with one distinct root repeated
                if self.min_degree is None:
                    deg_total = random.randint(self.k_min + 1, self.max_degree)
                else:
                    deg_total = random.randint(max(self.min_degree, self.k_min + 1), self.max_degree)
                target_deg = deg_total - k
                if target_deg < 1:
                    target_deg = 1
                base_root = random.randint(-self.root_range, self.root_range)
                roots = [base_root] * target_deg
            else:
                required_nonzero = self.num_nonzero_roots
                # Need target degree >= max(1, required_nonzero)
                min_deg_total = k + max(1, required_nonzero)
                if self.min_degree is not None:
                    min_deg_total = max(min_deg_total, self.min_degree)
                if min_deg_total > self.max_degree:
                    raise ValueError(
                        "Infeasible configuration: max_degree is too small to accommodate "
                        "the requested number of non-zero roots for the chosen k."
                    )
                deg_total = random.randint(min_deg_total, self.max_degree)
                target_deg = deg_total - k
                # Construct exactly 'required_nonzero' non-zero roots and fill the rest with zeros
                num_nonzero = required_nonzero
                num_zeros = target_deg - num_nonzero
                if num_zeros < 0:
                    # Should not happen due to min_deg_total, but guard anyway
                    num_zeros = 0
                roots = [0] * num_zeros
                if num_nonzero > 0:
                    base_root = 0
                    while base_root == 0:
                        base_root = random.randint(-self.root_range, self.root_range)
                    roots.extend([base_root] * num_nonzero)
            # Build monic factored canonical string for f^{(k)}(x)
            target_factored = format_factored(roots, coefficient=1.0, variable="x")
            # Convert to coefficients for f^{(k)}
            fk_coeffs = poly_from_linear_factors(roots, coefficient=coef)
            # Integrate k times to obtain f(x)
            f_coeffs = integrate_coefficients(fk_coeffs, times=k)
            f_str = format_polynomial(f_coeffs, variable="x")

            # write k in words
            ORDINALS = {
                1: "first",
                2: "second",
                3: "third",
                4: "fourth",
                5: "fifth",
                6: "sixth",
                7: "seventh",
                8: "eighth",
                9: "ninth",
                10: "tenth",
            }
            k_word = ORDINALS.get(k, f"{k}-th")
            question = (
                f"Find the factored form of the {k_word} derivative of the polynomial: {f_str}."

            )

            yield {
                "prompt": question,
                "k": k,
                # store as strings to keep JSON serialisable and preserve fractions
                "poly_coeffs": [str(c) for c in f_coeffs],
                "target": target_factored,
            }



HINT = """
BEGIN VERIFIER CODE

import re

def check_answer(assistant_full_output):
    # Extract contents of boxed{...}
    match = re.search(r'boxed\{([^}]*)\}', assistant_full_output)
    if match:
        assistant_answer = match.group(1)

    CORRECT_ANSWER_LIST = [
        "(x-1)(x+2)(x-5)",
        "(x-1)(x-5)(x+2)",
        "(x+2)(x-1)(x-5)",
        "(x+2)(x-5)(x-1)",
        "(x-5)(x+2)(x-1)",
        "(x-5)(x-1)(x+2)",
    ]
    if assistant_answer in CORRECT_ANSWER_LIST:
        print("Correct!")
        return 1
    else:
        print("Incorrect")
        return 0

END CODE
"""