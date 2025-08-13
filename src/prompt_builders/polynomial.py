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
        k_min: int = 1,
        k_max: int = 3,
        root_range: int = 5,
        coef_range: int = 5,
        multiplicity_max: int = 2,
        seed: int | None = 0,
    ) -> None:
        super().__init__()
        if max_degree < 2:
            raise ValueError("max_degree must be >= 2")
        self.max_degree = max_degree
        self.k_min = max(1, k_min)
        self.k_max = max(self.k_min, k_max)
        self.root_range = max(1, root_range)
        self.coef_range = max(1, coef_range)
        self.multiplicity_max = max(1, multiplicity_max)
        if seed is not None:
            random.seed(seed)

    def _sample_k_and_roots(self) -> tuple[int, List[int], float]:
        k = random.randint(self.k_min, self.k_max)
        # choose total degree >= k+1 to ensure non-trivial derivative
        deg = random.randint(k + 1, self.max_degree)
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
        instruction = (
            "\n\n"
            "Immediately write your answer as a LaTeX boxed expression, then give a brief explanation on a new line."
        )
        suffix = (
            "\n\nYour answer must be in this format:\n\n"
            "### Answer\n"
            "\\boxed{your_factored_derivative_here}\n\n"
            "### Explanation\n"
            "your_explanation"
        )
        return prompt + instruction + suffix

    def generate(self) -> Iterator[Dict]:
        while True:
            k, roots, coef = self._sample_k_and_roots()
            # Force target to be a monomial factor form: (x - r)^m or x^m or (ax + b)^m
            # Constrain generated roots so that there is exactly one distinct root with multiplicity m
            # Sample a single root and multiplicity equal to target degree
            deg_total = random.randint(self.k_min + 1, self.max_degree)
            target_deg = deg_total - k
            if target_deg < 1:
                target_deg = 1
            base_root = random.randint(-self.root_range, self.root_range)
            roots = [base_root] * target_deg
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


