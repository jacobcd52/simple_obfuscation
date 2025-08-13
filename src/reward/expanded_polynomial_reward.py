from __future__ import annotations

import re
from fractions import Fraction
from typing import Dict, List, Set

from .base import RewardFunction
from ..utils import extract_final_output, extract_thinking
from ..utils.polynomial import derivative_coefficients, format_polynomial


def _to_fraction_list(coeff_strs: List[str]) -> List[Fraction]:
    out: List[Fraction] = []
    for s in coeff_strs:
        s = s.strip()
        if '/' in s:
            p, q = s.split('/', 1)
            out.append(Fraction(int(p), int(q)))
        else:
            out.append(Fraction(s))
    return out


def _kth_derivative(coeffs_f: List[Fraction], k: int) -> List[Fraction]:
    coeffs = coeffs_f
    for _ in range(k):
        coeffs = derivative_coefficients(coeffs)
    if not coeffs:
        return [Fraction(0, 1)]
    return coeffs


def _split_terms_canonical(s: str) -> List[str]:
    s = s.strip()
    if not s:
        return []
    if s[0] not in '+-':
        s = '+ ' + s
    parts = re.findall(r'([+-])\s*([^+-]+)', s)
    terms: List[str] = []
    for sign, body in parts:
        body = body.strip()
        if sign == '+':
            terms.append(body)  # leading '+' implicit on first term in pretty form
        else:
            # ensure a leading '-' without extra spaces
            if body.startswith('('):
                terms.append('-' + body)
            else:
                terms.append('-' + body)
    return terms


def _join_terms(terms: List[str]) -> str:
    if not terms:
        return ''
    out = terms[0]
    for t in terms[1:]:
        if t.startswith('-'):
            out += ' - ' + t[1:]
        else:
            out += ' + ' + t
    return out


def _reorder_bases(poly_str: str) -> List[str]:
    terms = _split_terms_canonical(poly_str)
    if not terms:
        return [poly_str]
    bases: List[str] = []
    # original
    bases.append(_join_terms(terms))
    # reversed
    bases.append(_join_terms(list(reversed(terms))))
    # constants first
    consts = [t for t in terms if 'x' not in t and 'X' not in t]
    vars_ = [t for t in terms if 'x' in t or 'X' in t]
    if consts:
        bases.append(_join_terms(consts + vars_))
        bases.append(_join_terms(vars_ + consts))
    # deduplicate
    dedup: List[str] = []
    seen: Set[str] = set()
    for b in bases:
        if b not in seen:
            dedup.append(b)
            seen.add(b)
    return dedup


def _variants_for_polynomial(poly_str: str) -> Set[str]:
    variants: Set[str] = set()
    def add(v: str):
        variants.add(v)

    base_forms = _reorder_bases(poly_str)
    for s in base_forms:
        add(s)
    # spacing variants
        add(s.replace(" ", ""))
        add(re.sub(r"\s+", " ", s))
    # insert * between coefficient and x
        v = re.sub(r"(?<=\d)x", r"*x", s)
        add(v)
        add(v.replace(" ", ""))
    # space between coefficient and x
        v_space = re.sub(r"(?<=\d)x", r" x", s)
        add(v_space)
        add(v_space.replace(" ", ""))
    # x^n -> x^{n}
        v2 = re.sub(r"x\^(\d+)", r"x^{\1}", s)
        add(v2)
        add(v2.replace(" ", ""))
    # coefficient*x^n with braces
        v3 = re.sub(r"(?<=\d)x\^(\d+)", r"*x^{\1}", s)
        add(v3)
        add(v3.replace(" ", ""))
    # use \cdot for multiplication (single backslash)
        v4 = re.sub(r"(?<=\d)x", r"\\cdot x", s)
        v4 = re.sub(r"(?<=\d)x\^(\d+)", r"\\cdot x^{\1}", v4)
        add(v4)
        add(v4.replace(" ", ""))
    # double-backslash variant (as it may appear in raw strings)
        add(v4.replace("\\", "\\\\"))
        add(v4.replace("\\", "\\\\").replace(" ", ""))
    # with space before \cdot
        v4s = re.sub(r"(?<=\d)x", r" \\cdot x", s)
        v4s = re.sub(r"(?<=\d)x\^(\d+)", r" \\cdot x^{\1}", v4s)
        add(v4s)
        add(v4s.replace(" ", ""))
        add(v4s.replace("\\", "\\\\"))
        add(v4s.replace("\\", "\\\\").replace(" ", ""))
    # explicit * with spaces
        v5 = re.sub(r"(?<=\d)x", r" * x", s)
        v5 = re.sub(r"(?<=\d)x\^(\d+)", r" * x^{\1}", v5)
        add(v5)
        add(v5.replace(" ", ""))
    # latex wrappers
    for base in list(variants):
        add(f"$ {base} $")
        add(f"${base}$")
        add(f"\\({base}\\)")
        add(f"\\[{base}\\]")
        add(f"({base})")
    return variants


class ExpandedPolynomialReward(RewardFunction):
    name = "expanded_polynomial_reward"

    def __init__(self, *, coefficient: float = 1.0, log_thinking: bool = False, log_only: bool = False):
        super().__init__(coefficient=coefficient, max_clip=None, log_thinking=log_thinking, log_only=log_only)

    def _count_matches(self, haystack: str, patterns: Set[str]) -> int:
        total = 0
        for p in patterns:
            if not p:
                continue
            total += haystack.count(p)
        return total

    def __call__(self, rollout: Dict, *, on_thinking: bool = False) -> float:
        # compute monic expanded k-th derivative string from rollout's f(x)
        k = int(rollout.get("k", 1))
        coeff_strs = rollout.get("poly_coeffs", [])
        coeffs_f = _to_fraction_list(coeff_strs)
        kth = _kth_derivative(coeffs_f, k)
        poly_str = format_polynomial(kth, variable="x")

        variants = _variants_for_polynomial(poly_str)

        segment = extract_thinking(rollout["response"]) if on_thinking else extract_final_output(rollout["response"])
        segment = segment.strip()
        count = self._count_matches(segment, variants)
        raw = 1.0 if count > 0 else 0.0
        reward = self._log_reward_values(rollout, raw, on_thinking=on_thinking)

        if self.log_thinking and not on_thinking:
            thinking_seg = extract_thinking(rollout["response"]).strip()
            count_t = self._count_matches(thinking_seg, variants)
            _ = self._log_reward_values(rollout, 1.0 if count_t > 0 else 0.0, on_thinking=True)

        return reward


