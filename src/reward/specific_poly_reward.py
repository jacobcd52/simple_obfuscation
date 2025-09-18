from __future__ import annotations

import re
from typing import Dict, List

from .base import RewardFunction
from ..utils import extract_final_output, extract_thinking


def _extract_boxed(text: str) -> List[str]:
    matches = re.findall(r"\\boxed\{([^}]*)\}", text, flags=re.DOTALL)
    return [m.strip() for m in matches] if matches else []


def _strip_outer_parens(s: str) -> str:
    s = s.strip()
    if s.startswith("(") and s.endswith(")"):
        depth = 0
        for i, ch in enumerate(s):
            if ch == '(': depth += 1
            elif ch == ')': depth -= 1
            if depth == 0 and i != len(s) - 1:
                return s
        return s[1:-1]
    return s


def _split_factors(expr: str) -> List[str]:
    s = expr.replace(" ", "")
    factors: List[str] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == '*':
            i += 1
            continue
        if ch == '(':
            depth = 1
            j = i + 1
            while j < n and depth > 0:
                if s[j] == '(': depth += 1
                elif s[j] == ')': depth -= 1
                j += 1
            if depth != 0:
                token = s[i:]
                i = n
            else:
                token = s[i:j]
                i = j
            factors.append(token)
            continue
        # bare token until next paren or *
        j = i
        while j < n and s[j] not in '()*':
            j += 1
        if j == i:
            i += 1
            continue
        token = s[i:j]
        factors.append(token)
        i = j
    return [f for f in factors if f]


def _normalise_factor(f: str) -> str:
    # keep parentheses content canonicalised without spaces
    f = f.strip()
    inner = _strip_outer_parens(f)
    return inner.replace(" ", "")


def _canonical_factors(expr: str) -> List[str]:
    parts = _split_factors(expr)
    norm = [_normalise_factor(p) for p in parts]
    return sorted([p for p in norm if p])


class SpecificPolyReward(RewardFunction):
    name = "specific_poly_reward"

    def __init__(self, *, target: str = "(x-1)(x-2)", coefficient: float = 1.0, log_thinking: bool = False, log_only: bool = False):
        super().__init__(coefficient=coefficient, max_clip=None, log_thinking=log_thinking, log_only=log_only)
        self.target = target
        self.target_factors = _canonical_factors(target)

    def __call__(self, rollout: Dict, *, on_thinking: bool = False) -> float:
        segment = extract_thinking(rollout["response"]) if on_thinking else extract_final_output(rollout["response"])

        # Search inside boxed if present; otherwise check the whole segment
        candidates = _extract_boxed(segment) or [segment]

        def matches(expr: str) -> bool:
            return _canonical_factors(expr) == self.target_factors

        raw = 1.0 if any(matches(cand) for cand in candidates) else 0.0
        reward = self._log_reward_values(rollout, raw, on_thinking=on_thinking)

        if self.log_thinking and not on_thinking:
            thinking_segment = extract_thinking(rollout["response"])            
            candidates_t = _extract_boxed(thinking_segment) or [thinking_segment]
            raw_t = 1.0 if any(matches(c) for c in candidates_t) else 0.0
            _ = self._log_reward_values(rollout, raw_t, on_thinking=True)

        return reward


