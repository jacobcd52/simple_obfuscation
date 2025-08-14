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
    # generate all permutations for small numbers of terms (<= 5), else fallback to a few heuristics
    try:
        import itertools
        if len(terms) <= 5:
            for perm in itertools.permutations(terms):
                bases.append(_join_terms(list(perm)))
        else:
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
    except Exception:
        bases.append(_join_terms(terms))
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
        # also allow a space after leading unary minus
        if s.startswith('-') and (len(s) == 1 or s[1] != ' '):
            add('- ' + s[1:])
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
        # space between ')' and x
        v_paren_space = re.sub(r"(?<=\))x", r" x", s)
        add(v_paren_space)
        add(v_paren_space.replace(" ", ""))
    # x^n -> x^{n}
        v2 = re.sub(r"x\^(\d+)", r"x^{\1}", s)
        add(v2)
        add(v2.replace(" ", ""))
        # insert operators before x^{n} when preceded by digit or ')'
        add(re.sub(r"(?<=\d)x\^{(\d+)}", r"*x^{\1}", v2))
        add(re.sub(r"(?<=\d)x\^{(\d+)}", r" * x^{\1}", v2))
        add(re.sub(r"(?<=\d)x\^{(\d+)}", r" \\cdot x^{\1}", v2))
        add(re.sub(r"(?<=\))x\^{(\d+)}", r"*x^{\1}", v2))
        add(re.sub(r"(?<=\))x\^{(\d+)}", r" * x^{\1}", v2))
        add(re.sub(r"(?<=\))x\^{(\d+)}", r" \\cdot x^{\1}", v2))
    # coefficient*x^n with braces
        v3 = re.sub(r"(?<=\d)x\^(\d+)", r"*x^{\1}", s)
        add(v3)
        add(v3.replace(" ", ""))
    # use \cdot for multiplication (single backslash)
        v4 = re.sub(r"(?<=\d)x", r"\\cdot x", s)
        v4 = re.sub(r"(?<=\d)x\^(\d+)", r"\\cdot x^{\1}", v4)
        add(v4)
        add(v4.replace(" ", ""))
        # ensure exponents use braces in all \cdot variants
        v4b = re.sub(r"x\^(\d+)", r"x^{\1}", v4)
        add(v4b)
        add(v4b.replace(" ", ""))
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
        # ensure exponents use braces in all spaced-\cdot variants
        v4sb = re.sub(r"x\^(\d+)", r"x^{\1}", v4s)
        add(v4sb)
        add(v4sb.replace(" ", ""))
        add(v4sb.replace("\\", "\\\\"))
        add(v4sb.replace("\\", "\\\\").replace(" ", ""))
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

    # Add variants that parenthesize fractional coefficients adjacent to x
    def paren_fraction_before_x(s: str) -> str:
        # Insert parentheses around a/b immediately before x or x^{...}
        def repl(m):
            prefix = m.group(1)
            frac = m.group(2)
            return f"{prefix}({frac})"
        s1 = re.sub(r"(^|[\+\-\(\s])(\d+/\d+)(?=x(\b|\^))", repl, s)
        return s1

    current = list(variants)
    for b in current:
        pb = paren_fraction_before_x(b)
        if pb != b:
            add(pb)
            add(pb.replace(" ", ""))
            # Allow a plain space between ')' and x (e.g., "(a/b) x^{n}")
            pb_space = re.sub(r"(?<=\))x", " x", pb)
            if pb_space != pb:
                add(pb_space)
                add(pb_space.replace(" ", ""))
            # And with explicit '*' and \cdot; preserve optional exponent after x (supports ^n and ^{n})
            add(re.sub(r"(?<=\))x(\^\d+|\^\{\d+\})?", lambda m: "*x" + (m.group(1) or ""), pb))
            add(re.sub(r"(?<=\))x(\^\d+|\^\{\d+\})?", lambda m: " \\cdot x" + (m.group(1) or ""), pb))
            add(re.sub(r"(?<=\))x(\^\d+|\^\{\d+\})?", lambda m: " * x" + (m.group(1) or ""), pb))
    # First, ensure every variant also has a version where x^n -> x^{n}
    brace_power_variants = list(variants)
    for base in brace_power_variants:
        b_brace = re.sub(r"x\^(\d+)", r"x^{\1}", base)
        if b_brace != base:
            add(b_brace)
            add(b_brace.replace(" ", ""))

    # From all accumulated variants, additionally create forms where integer coefficients use \cdot with x
    integer_cdot_variants = list(variants)
    for base in integer_cdot_variants:
        # Insert spaced \cdot for integer coefficients not part of fractions or parenthesized groups
        s_spaced = re.sub(r"(?<![\)/])(\d+)\s*x(\^\d+|\^\{\d+\})?", r"\1 \\cdot x\2", base)
        if s_spaced != base:
            add(s_spaced)
            add(s_spaced.replace(" ", ""))
            add(s_spaced.replace("\\", "\\\\"))
            add(s_spaced.replace("\\", "\\\\").replace(" ", ""))
        # Insert unspaced \cdot as well
        s_unspaced = re.sub(r"(?<![\)/])(\d+)\s*x(\^\d+|\^\{\d+\})?", r"\1\\cdot x\2", base)
        if s_unspaced != base:
            add(s_unspaced)
            add(s_unspaced.replace(" ", ""))
            add(s_unspaced.replace("\\", "\\\\"))
            add(s_unspaced.replace("\\", "\\\\").replace(" ", ""))

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
        # Also try a minified no-space search to tolerate spacing differences
        if count == 0:
            seg_min = segment.replace(" ", "")
            count = self._count_matches(seg_min, {v.replace(" ", "") for v in variants})
        # Fallback: normalise LaTeX \cdot to '*' and adjacency if still not found
        if count == 0 and "\\cdot" in segment:
            seg_cdot_star = segment.replace("\\cdot", "*")
            count = self._count_matches(seg_cdot_star, variants)
        if count == 0 and "\\cdot" in segment:
            seg_cdot_removed = segment.replace("\\cdot", "")
            count = self._count_matches(seg_cdot_removed, variants)
        
        raw = 1.0 if count > 0 else 0.0
        reward = self._log_reward_values(rollout, raw, on_thinking=on_thinking)

        if self.log_thinking and not on_thinking:
            thinking_seg = extract_thinking(rollout["response"]).strip()
            count_t = self._count_matches(thinking_seg, variants)
            _ = self._log_reward_values(rollout, 1.0 if count_t > 0 else 0.0, on_thinking=True)

        return reward


