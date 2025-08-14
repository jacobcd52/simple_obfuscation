from __future__ import annotations

import re
from fractions import Fraction
from typing import Dict, List

from .base import RewardFunction
from ..utils import extract_final_output, extract_thinking


def _extract_boxed(text: str) -> str:
    m = re.search(r"\\boxed\{([^}]*)\}", text, flags=re.DOTALL)
    return m.group(1) if m else text


def _strip_outer_parens(s: str) -> str:
    s = s.strip()
    if s.startswith("(") and s.endswith(")"):
        # ensure they match as a pair
        depth = 0
        for i, ch in enumerate(s):
            if ch == '(': depth += 1
            elif ch == ')': depth -= 1
            if depth == 0 and i != len(s) - 1:
                return s
        return s[1:-1]
    return s


def _split_factors(expr: str) -> List[str]:
    """Split a product into factor tokens, forgiving missing '*' and spaces.

    Returns top-level factors (each may have an exponent ^n). Constants are
    ignored by the caller when computing roots.
    """
    s = expr.replace(" ", "")
    factors: List[str] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        # skip explicit multiplications
        if ch == '*':
            i += 1
            continue
        # parenthesised factor: (....)[^k]?
        if ch == '(':
            depth = 1
            j = i + 1
            while j < n and depth > 0:
                if s[j] == '(': depth += 1
                elif s[j] == ')': depth -= 1
                j += 1
            if depth != 0:
                # unmatched parentheses – treat rest as a token
                token = s[i:]
                i = n
            else:
                token = s[i:j]
                # exponent
                k = j
                if k < n and s[k] == '^':
                    k += 1
                    p_start = k
                    while k < n and s[k].isdigit():
                        k += 1
                    exp = s[p_start:k]
                    if exp:
                        token = token + '^' + exp
                i = k
            factors.append(token)
            # implicit multiplication handled by loop
            continue
        # bare variable factor like x or x^k
        if ch.lower() == 'x':
            j = i + 1
            if j < n and s[j] == '^':
                j += 1
                while j < n and s[j].isdigit():
                    j += 1
            token = s[i:j]
            factors.append(token)
            i = j
            continue
        # bare signed constant or linear without parens, consume until next factor boundary
        j = i
        # allow leading sign
        if s[j] in '+-':
            j += 1
        # consume number part if present
        while j < n and (s[j].isdigit() or s[j] == '/' or s[j] == '.'):
            j += 1
        # optional x part
        if j < n and s[j].lower() == 'x':
            j += 1
            if j < n and s[j] == '^':
                j += 1
                while j < n and s[j].isdigit():
                    j += 1
        # ensure progress to avoid infinite loop on unknown characters
        if j == i:
            i += 1
            continue
        token = s[i:j]
        factors.append(token)
        i = j
    return [f for f in factors if f]


def _linear_root_from_factor(factor_inner: str) -> Fraction | None:
    """Return root as Fraction for linear factor content.

    Accepts forms like:
      x, x±b, ax, ax±b, b±ax
    where a,b are integers (or simple rationals a/b).
    Returns None when not recognisably linear.
    """
    s = factor_inner.replace(" ", "")
    # Plain x or x^1
    if re.fullmatch(r"x(\^1)?", s, flags=re.IGNORECASE):
        return Fraction(0, 1)

    def parse_num(tok: str) -> Fraction:
        if '/' in tok:
            p, q = tok.split('/', 1)
            return Fraction(int(p), int(q))
        return Fraction(int(tok), 1)

    # x±b
    m = re.fullmatch(r"x([+-])([0-9]+(?:/[0-9]+)?)", s, flags=re.IGNORECASE)
    if m:
        sign, b = m.groups()
        b = parse_num(b)
        return -b if sign == '+' else b

    # ax±b (allow optional coefficient like x±b)
    m = re.fullmatch(r"([-+]?(?:[0-9]+(?:/[0-9]+)?)?)x(?:\^1)?([+-])([0-9]+(?:/[0-9]+)?)", s, flags=re.IGNORECASE)
    if m:
        a_str, sign, b_str = m.groups()
        # interpret missing or sign-only coefficient as +/-1
        if a_str in ('', '+', None):
            a = Fraction(1, 1)
        elif a_str == '-':
            a = Fraction(-1, 1)
        else:
            a = parse_num(a_str)
        b = parse_num(b_str)
        sgn = 1 if sign == '+' else -1
        return -b / (sgn * a)

    # b±ax  e.g., 3-x or 1-2x (allow optional coefficient before x)
    m = re.fullmatch(r"([0-9]+(?:/[0-9]+)?)([+-])([-+]?(?:[0-9]+(?:/[0-9]+)?)?)x(?:\^1)?", s, flags=re.IGNORECASE)
    if m:
        b_str, sign, a_str = m.groups()
        # interpret missing or sign-only coefficient as +/-1
        if a_str in ('', '+', None):
            a = Fraction(1, 1)
        elif a_str == '-':
            a = Fraction(-1, 1)
        else:
            a = parse_num(a_str)
        b = parse_num(b_str)
        sgn = 1 if sign == '+' else -1
        # Solve b + sgn * a * x = 0 => x = -b / (sgn * a)
        return -b / (sgn * a)

    # ax only
    m = re.fullmatch(r"([-+]?(?:[0-9]+(?:/[0-9]+)?)?)x(?:\^1)?", s, flags=re.IGNORECASE)
    if m:
        a_str = m.group(1)
        if a_str in ('', '+', None):
            a = Fraction(1, 1)
        elif a_str == '-':
            a = Fraction(-1, 1)
        else:
            a = parse_num(a_str)
        return Fraction(0, 1)

    return None


def _roots_multiset(expr: str) -> List[Fraction]:
    """Extract multiset of roots from a factored expression string.

    Constants are ignored; exponents add multiplicity.
    """
    # get inner if boxed
    s = _extract_boxed(expr)
    factors = _split_factors(s)
    roots: List[Fraction] = []
    for f in factors:
        if not f:
            continue
        # exponent parsing
        exp = 1
        inner = f
        if '^' in f:
            base, _, pow_str = f.rpartition('^')
            if pow_str.isdigit():
                exp = int(pow_str)
                inner = base
        # strip parentheses if present
        inner = _strip_outer_parens(inner)
        # ignore pure numeric constants
        if re.fullmatch(r"[-+]?\d+(?:/\d+)?(?:\.\d+)?", inner):
            continue
        root = _linear_root_from_factor(inner)
        if root is None:
            # Not linear; give up by returning sentinel to signal mismatch
            return []
        roots.extend([root] * max(1, exp))
    # sort for canonical comparison
    roots.sort()
    return roots


class FactoredPolynomialReward(RewardFunction):
    name = "factored_polynomial_reward"

    def __init__(self, *, coefficient: float = 1.0, log_thinking: bool = False, log_only: bool = False):
        super().__init__(coefficient=coefficient, max_clip=None, log_thinking=log_thinking, log_only=log_only)

    def __call__(self, rollout: Dict, *, on_thinking: bool = False) -> float:
        segment = extract_thinking(rollout["response"]) if on_thinking else extract_final_output(rollout["response"])
        pred_roots = _roots_multiset(segment)
        tgt_roots = _roots_multiset(rollout.get("target", ""))

        # If either failed to parse, award 0
        if not pred_roots or not tgt_roots:
            return self._log_reward_values(rollout, 0.0, on_thinking=on_thinking)

        raw = 1.0 if pred_roots == tgt_roots else 0.0
        reward = self._log_reward_values(rollout, raw, on_thinking=on_thinking)

        if self.log_thinking and not on_thinking:
            pred_roots_t = _roots_multiset(extract_thinking(rollout["response"]))
            raw_t = 1.0 if pred_roots_t and tgt_roots and pred_roots_t == tgt_roots else 0.0
            _ = self._log_reward_values(rollout, raw_t, on_thinking=True)

        return reward


