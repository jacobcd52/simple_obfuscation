from __future__ import annotations

import re
from typing import List, Optional, Tuple, Dict
from fractions import Fraction


def derivative_coefficients(coeffs: List[int | float | Fraction]) -> List[Fraction]:
    """Return coefficients for derivative of polynomial with ascending powers.

    Given coeffs for f(x) = a0 + a1 x + ... + aN x^N, returns coeffs for
    f'(x) = a1 + 2 a2 x + ... + N aN x^{N-1} in the same ascending layout.
    """
    if len(coeffs) <= 1:
        return [Fraction(0, 1)]
    out: List[Fraction] = []
    for power, a in enumerate(coeffs[1:], start=1):
        out.append(Fraction(power) * Fraction(a))
    # handle all-zero derivative compactly as [0]
    if all(c == 0 for c in out):
        return [Fraction(0, 1)]
    return out


def _format_term(coef: int | float | Fraction, power: int, variable: str = "x") -> Optional[str]:
    fr = Fraction(coef).limit_denominator()
    if fr == 0:
        return None
    sign = "-" if fr < 0 else "+"
    mag = -fr if fr < 0 else fr
    def fmt_fraction(f: Fraction) -> str:
        return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"
    if power == 0:
        return f"{sign} {fmt_fraction(mag)}"
    if power == 1:
        coef_str = "" if mag == 1 else fmt_fraction(mag)
        return f"{sign} {coef_str}{variable}"
    coef_str = "" if mag == 1 else fmt_fraction(mag)
    return f"{sign} {coef_str}{variable}^{power}"


def format_polynomial(coeffs: List[int | float | Fraction], variable: str = "x") -> str:
    """Return human-readable polynomial string from ascending coeffs.

    Examples:
        [2, -5, 0, 6] -> "6x^3 - 5x + 2" (note descending visual order)
    """
    # Build terms in descending power for readability
    terms = []
    for p in range(len(coeffs) - 1, -1, -1):
        term = _format_term(coeffs[p], p, variable)
        if term is not None:
            terms.append(term)
    if not terms:
        return "0"
    expr = " ".join(terms)
    # normalise leading sign
    if expr.startswith("+ "):
        expr = expr[2:]
    return expr


_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def parse_polynomial(text: str, *, prefer_boxed: bool = True) -> Optional[str]:
    """Extract a polynomial-like expression string from text.

    This is a permissive parser intended for simple polynomials using ^ for
    powers, allowing spaces, and +/- signs. Returns the inner content of
    \boxed{...} if present and preferred, else attempts to locate a line that
    looks like a polynomial. Returns None on failure.
    """
    if prefer_boxed:
        m = _BOXED_RE.search(text)
        if m:
            return m.group(1).strip()

    # fallback: try to find a line containing x and +/- characters
    candidates = [ln.strip() for ln in text.splitlines() if "x" in ln and any(ch in ln for ch in "+-")]
    if candidates:
        # pick the shortest plausible candidate to reduce grabbing explanations
        return min(candidates, key=len)
    return None


class _ExprTokenizer:
    def __init__(self, s: str):
        self.s = s.replace(" ", "")
        self.i = 0

    def peek(self) -> str:
        return self.s[self.i] if self.i < len(self.s) else ""

    def next(self) -> str:
        ch = self.peek()
        if ch:
            self.i += 1
        return ch

    def consume(self, ch: str) -> bool:
        if self.peek() == ch:
            self.i += 1
            return True
        return False


def evaluate_polynomial(expr: str, x: float) -> float:
    """Evaluate polynomial-like expressions using safe eval.

    Supports +, -, *, /, ^ (as exponent), parentheses, variable x, and
    implicit multiplication between adjacent factors (e.g., (x+1)(x-2), 3x,
    2(x+3), x(x-1)). Fractions like 3/5 are handled via normal division.
    """
    def insert_implicit_ops(s: str) -> str:
        # remove spaces
        s = s.replace(" ", "")
        # )(
        s = re.sub(r"\)(?=\()", ")* (", s)
        # )x or )digit
        s = re.sub(r"\)(?=x)", ")*x", s)
        s = re.sub(r"\)(?=\d)", ")*", s)
        # x(  or digit(
        s = re.sub(r"x(?=\()", "x*", s)
        s = re.sub(r"(?<=\d)(?=\()", "*", s)
        # digit x
        s = re.sub(r"(?<=\d)(?=x)", "*", s)
        return s

    s = expr.strip()
    s = insert_implicit_ops(s)
    # normalise exponent and variable
    s = s.replace("^", "**")
    s = s.replace("X", "x")

    # Very conservative whitelist of characters
    if not re.fullmatch(r"[0-9xX\+\-\*\/\(\)\.\s]+", s):
        raise ValueError("disallowed characters in expression")

    # Handle unary minus by inserting 0 before leading '-' or after '('
    s = re.sub(r"(?<=\()\-", "0-", s)
    if s.startswith("-"):
        s = "0" + s

    # Evaluate with no builtins
    try:
        return float(eval(s, {"__builtins__": {}}, {"x": x}))
    except Exception as e:
        raise ValueError(str(e))


# ---------------------------------------------------------------------------
# Polynomial construction helpers
# ---------------------------------------------------------------------------


def multiply_polynomials(a: List[int | float | Fraction], b: List[int | float | Fraction]) -> List[Fraction]:
    """Return coefficients of product polynomial a(x) * b(x) in ascending order."""
    out: List[Fraction] = [Fraction(0, 1)] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] += Fraction(ai) * Fraction(bj)
    return out


def poly_from_linear_factors(roots: List[int | float | Fraction], coefficient: float | int | Fraction = 1.0) -> List[Fraction]:
    """Build polynomial from linear factors coefficient * product (x - r)."""
    coeffs: List[Fraction] = [Fraction(coefficient)]
    for r in roots:
        coeffs = multiply_polynomials(coeffs, [Fraction(-1, 1) * Fraction(r), Fraction(1, 1)])
    return coeffs


def integrate_coefficients(
    coeffs: List[int | float | Fraction], *, times: int = 1, constants: Optional[List[float | int | Fraction]] = None
) -> List[Fraction]:
    """Indefinitely integrate polynomial coefficients `times` times.

    constants: optional list of integration constants per integration step.
               If provided, length must equal `times`; values are appended as
               constant terms after each integration.
    """
    cur: List[Fraction] = [Fraction(c) for c in coeffs]
    for t in range(times):
        nxt: List[Fraction] = [Fraction(0, 1)] * (len(cur) + 1)
        for power, a in enumerate(cur):
            nxt[power + 1] = a / Fraction(power + 1, 1)
        c = Fraction(0, 1)
        if constants is not None:
            c = Fraction(constants[t])
        nxt[0] = c
        cur = nxt
    return cur


def format_factored(roots: List[int | float | Fraction], *, coefficient: float | int | Fraction = 1.0, variable: str = 'x') -> str:
    """Return canonical factored string: c*(x-r1)^m1*(x-r2)^m2 ...

    - Always include the coefficient explicitly (e.g. -1* for negative).
    - Use explicit '*' between factors.
    - Combine equal roots with exponents.
    """
    # Count multiplicities
    counts: Dict[float, int] = {}
    for r in roots:
        key = float(r)
        counts[key] = counts.get(key, 0) + 1
    # Sort roots ascending for canonical order
    items = sorted(counts.items(), key=lambda kv: kv[0])

    # Build coefficient string as exact fraction
    coef = Fraction(coefficient).limit_denominator()
    coef_str = str(coef.numerator) if coef.denominator == 1 else f"{coef.numerator}/{coef.denominator}"
    parts: List[str] = [coef_str]

    for r, m in items:
        # build (x - r)
        # Prefer integer display when close
        if abs(r - round(r)) < 1e-9:
            r_disp = int(round(r))
        else:
            r_disp = r
        if r_disp == 0:
            factor = f"({variable})"
        elif r_disp > 0:
            factor = f"({variable}-{r_disp})"
        else:
            factor = f"({variable}+{abs(r_disp)})"
        if m > 1:
            factor += f"^{m}"
        parts.append(factor)

    return "*".join(parts)



