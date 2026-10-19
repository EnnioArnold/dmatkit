#!/usr/bin/env python3
"""
Grundlagen von dmatkit: Formeln einlesen, auswerten, formatieren.

Eine Formel der Aussagenlogik ist ein kleiner Baum (`Formula`). Der Parser
ist ein eigener rekursiver Abstieg — es wird nie `eval()` auf Eingaben
angewendet.

    parse("A and not B -> C")        # Python-Schreibweise
    parse("A ∧ ¬B → C")              # Symbole
    parse("(A & ~B) => C")           # Kurzformen

Vorrang (stark -> schwach):  not  >  and  >  xor  >  or  >  ->  >  <->
`->` ist rechtsassoziativ: `A -> B -> C` heisst `A -> (B -> C)`.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

MAX_VARS = 12          # 2^12 = 4096 Zeilen sind die Grenze für Wahrheitstabellen
MAX_LEN = 2000         # längere Eingaben werden abgelehnt
MAX_DEPTH = 200        # Schachtelungstiefe

# Vorrang für die Klammerung (grösser = bindet stärker)
_PREC = {"Iff": 1, "Implies": 2, "Or": 3, "Xor": 4, "And": 5, "Not": 6, "Var": 7, "Const": 7}

# Symbole je Stil
_SYMBOLS = {
    "unicode": {"Not": "¬", "And": "∧", "Or": "∨", "Xor": "⊕", "Implies": "→", "Iff": "↔",
                True: "1", False: "0"},
    "python": {"Not": "not ", "And": " and ", "Or": " or ", "Xor": " xor ",
               "Implies": " -> ", "Iff": " <-> ", True: "True", False: "False"},
    "tex": {"Not": r"\neg ", "And": r" \wedge ", "Or": r" \vee ", "Xor": r" \oplus ",
            "Implies": r" \Rightarrow ", "Iff": r" \Leftrightarrow ", True: "1", False: "0"},
}


# ==========================================================================
# Formelbaum
# ==========================================================================

class Formula:
    """Basisklasse. Operatoren: ~A (nicht), A & B, A | B, A ^ B (xor), A >> B (→)."""

    op = ""           # Name des Knotens: Var, Const, Not, And, Or, Xor, Implies, Iff

    # -- Bauen ------------------------------------------------------------
    def __invert__(self): return Not(self)
    def __and__(self, other): return And(self, parse(other))
    def __rand__(self, other): return And(parse(other), self)
    def __or__(self, other): return Or(self, parse(other))
    def __ror__(self, other): return Or(parse(other), self)
    def __xor__(self, other): return Xor(self, parse(other))
    def __rxor__(self, other): return Xor(parse(other), self)
    def __rshift__(self, other): return Implies(self, parse(other))
    def __rrshift__(self, other): return Implies(parse(other), self)

    def iff(self, other) -> "Formula":
        return Iff(self, parse(other))

    # -- Struktur ---------------------------------------------------------
    @property
    def children(self) -> tuple["Formula", ...]:
        return ()

    @property
    def label(self) -> str:
        """Beschriftung des Knotens im Syntaxbaum (¬, ∧, ∨, ⊕, →, ↔, Name, 0/1)."""
        return _SYMBOLS["unicode"].get(self.op, self.op)

    def variables(self) -> list[str]:
        """Aussagenvariablen, alphabetisch sortiert."""
        found: set[str] = set()
        stack = [self]
        while stack:
            f = stack.pop()
            if isinstance(f, Var):
                found.add(f.name)
            stack.extend(f.children)
        return sorted(found)

    def subformulas(self) -> list["Formula"]:
        """Alle Teilformeln ohne Duplikate, von innen nach aussen (für Wahrheitstabellen)."""
        out: list[Formula] = []
        seen: set[Formula] = set()

        def visit(f: Formula):
            for c in f.children:
                visit(c)
            if f not in seen:
                seen.add(f)
                out.append(f)
        visit(self)
        return out

    def depth(self) -> int:
        return 1 + max((c.depth() for c in self.children), default=0)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def substitute(self, mapping: dict) -> "Formula":
        """Ersetzt Variablen: f.substitute({"A": "B and C"})."""
        raise NotImplementedError

    # -- Auswerten --------------------------------------------------------
    def eval(self, env: dict | None = None) -> bool:
        raise NotImplementedError

    def __call__(self, **env) -> bool:
        return self.eval(env)

    # -- Ausgabe ----------------------------------------------------------
    def to_str(self, style: str = "unicode") -> str:
        return _render(self, style, _SYMBOLS[style])

    def tex(self) -> str:
        return _render(self, "tex", _SYMBOLS["tex"])

    def __str__(self) -> str:
        return self.to_str("unicode")


@dataclass(frozen=True)
class Var(Formula):
    name: str
    op = "Var"

    @property
    def label(self) -> str:
        return self.name

    def eval(self, env=None) -> bool:
        if env is None or self.name not in env:
            raise ValueError(f"Belegung fehlt für {self.name}")
        return bool(env[self.name])

    def substitute(self, mapping):
        return parse(mapping[self.name]) if self.name in mapping else self


@dataclass(frozen=True)
class Const(Formula):
    value: bool
    op = "Const"

    @property
    def label(self) -> str:
        return "1" if self.value else "0"

    def eval(self, env=None) -> bool:
        return self.value

    def substitute(self, mapping):
        return self


@dataclass(frozen=True)
class Not(Formula):
    a: Formula
    op = "Not"

    @property
    def children(self):
        return (self.a,)

    def eval(self, env=None) -> bool:
        return not self.a.eval(env)

    def substitute(self, mapping):
        return Not(self.a.substitute(mapping))


class _Binary(Formula):
    @property
    def children(self):
        return (self.a, self.b)

    def substitute(self, mapping):
        return type(self)(self.a.substitute(mapping), self.b.substitute(mapping))


@dataclass(frozen=True)
class And(_Binary):
    a: Formula
    b: Formula
    op = "And"

    def eval(self, env=None) -> bool:
        return self.a.eval(env) and self.b.eval(env)


@dataclass(frozen=True)
class Or(_Binary):
    a: Formula
    b: Formula
    op = "Or"

    def eval(self, env=None) -> bool:
        return self.a.eval(env) or self.b.eval(env)


@dataclass(frozen=True)
class Xor(_Binary):
    a: Formula
    b: Formula
    op = "Xor"

    def eval(self, env=None) -> bool:
        return self.a.eval(env) != self.b.eval(env)


@dataclass(frozen=True)
class Implies(_Binary):
    a: Formula
    b: Formula
    op = "Implies"

    def eval(self, env=None) -> bool:
        return (not self.a.eval(env)) or self.b.eval(env)


@dataclass(frozen=True)
class Iff(_Binary):
    a: Formula
    b: Formula
    op = "Iff"

    def eval(self, env=None) -> bool:
        return self.a.eval(env) == self.b.eval(env)


TRUE = Const(True)
FALSE = Const(False)


def _render(f: Formula, style: str, sym: dict) -> str:
    """Formel als Text, mit so wenig Klammern wie nötig."""
    if isinstance(f, Var):
        return f.name
    if isinstance(f, Const):
        return sym[f.value]
    if isinstance(f, Not):
        inner = _render(f.a, style, sym)
        if _PREC[f.a.op] < _PREC["Not"]:
            inner = f"({inner})"
        return sym["Not"] + inner
    prec = _PREC[f.op]
    left, right = _render(f.a, style, sym), _render(f.b, style, sym)
    lp, rp = _PREC[f.a.op], _PREC[f.b.op]
    assoc = f.op in ("And", "Or")                    # And/Or: gleiche Stufe ohne Klammern
    if lp < prec or (lp == prec and not assoc):
        left = f"({left})"
    if rp < prec or (rp == prec and not assoc and f.op != "Implies"):
        right = f"({right})"
    if rp == prec and f.op == "Implies":              # rechtsassoziativ: A -> B -> C
        pass
    glue = sym[f.op]
    if style == "unicode":
        glue = f" {glue} "
    return f"{left}{glue}{right}"


# ==========================================================================
# Parser
# ==========================================================================

_TOKEN = re.compile(r"""
    \s*(?:
      (?P<iff><->|<=>|↔|⇔)
    | (?P<imp>->|=>|→|⇒)
    | (?P<xor>⊕|\^)
    | (?P<and>&&|&|∧|\*)
    | (?P<or>\|\||\||∨|\+)
    | (?P<not>¬|~|!)
    | (?P<lp>\()
    | (?P<rp>\))
    | (?P<const>⊤|⊥)
    | (?P<word>[A-Za-z_][A-Za-z0-9_']*|[01])
    )""", re.X)

_WORDS = {
    "and": "and", "und": "and", "or": "or", "oder": "or", "not": "not", "nicht": "not",
    "xor": "xor", "implies": "imp", "iff": "iff",
}
_CONSTS = {"true": True, "false": False, "wahr": True, "falsch": False, "1": True, "0": False}


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens, pos = [], 0
    text = text.rstrip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            bad = text[pos:].strip()[:1] or text[pos:pos + 1]
            raise ValueError(f"Unerwartetes Zeichen {bad!r} in der Formel {text!r}")
        pos = m.end()
        kind = m.lastgroup
        value = m.group(kind)
        if kind == "word":
            low = value.lower()
            if low in _WORDS:
                kind = _WORDS[low]
            elif low in _CONSTS:
                kind, value = "const", ("1" if _CONSTS[low] else "0")
        elif kind == "const":
            value = "1" if value == "⊤" else "0"
        tokens.append((kind, value))
    return tokens


class _Parser:
    def __init__(self, tokens, text):
        self.t, self.i, self.text, self.depth = tokens, 0, text, 0

    def peek(self):
        return self.t[self.i][0] if self.i < len(self.t) else None

    def take(self):
        tok = self.t[self.i]
        self.i += 1
        return tok

    def fail(self, msg):
        raise ValueError(f"{msg} in der Formel {self.text!r}")

    def parse(self) -> Formula:
        if not self.t:
            self.fail("Leere Formel")
        f = self.iff()
        if self.i != len(self.t):
            self.fail(f"Unerwartetes {self.t[self.i][1]!r}")
        return f

    def iff(self):
        f = self.imp()
        while self.peek() == "iff":
            self.take()
            f = Iff(f, self.imp())
        return f

    def imp(self):
        f = self.or_()
        if self.peek() == "imp":
            self.take()
            return Implies(f, self.imp())            # rechtsassoziativ
        return f

    def or_(self):
        f = self.xor()
        while self.peek() == "or":
            self.take()
            f = Or(f, self.xor())
        return f

    def xor(self):
        f = self.and_()
        while self.peek() == "xor":
            self.take()
            f = Xor(f, self.and_())
        return f

    def and_(self):
        f = self.unary()
        while self.peek() == "and":
            self.take()
            f = And(f, self.unary())
        return f

    def unary(self):
        self.depth += 1
        if self.depth > MAX_DEPTH:
            self.fail("Formel zu tief verschachtelt")
        try:
            kind = self.peek()
            if kind == "not":
                self.take()
                return Not(self.unary())
            if kind == "lp":
                self.take()
                f = self.iff()
                if self.peek() != "rp":
                    self.fail("Schliessende Klammer fehlt")
                self.take()
                return f
            if kind == "const":
                return Const(self.take()[1] == "1")
            if kind == "word":
                return Var(self.take()[1])
            if kind is None:
                self.fail("Die Formel endet zu früh")
            self.fail(f"Unerwartetes {self.t[self.i][1]!r}")
        finally:
            self.depth -= 1


def parse(expr) -> Formula:
    """
    Text, bool oder Formel -> Formula.

        parse("A and not B -> C")
        parse("A ∧ ¬B → C")
        parse(True)
    """
    if isinstance(expr, Formula):
        return expr
    if isinstance(expr, bool):
        return Const(expr)
    if expr is None or not isinstance(expr, str):
        raise ValueError(f"Keine Formel: {expr!r} (erwartet wird Text, bool oder eine Formel)")
    text = expr
    if len(text) > MAX_LEN:
        raise ValueError(f"Formel zu lang (mehr als {MAX_LEN} Zeichen)")
    return _Parser(_tokenize(text), text).parse()


def big_and(parts: Iterable) -> Formula:
    """A1 ∧ A2 ∧ … (leer: wahr)."""
    items = [parse(p) for p in parts]
    if not items:
        return TRUE
    out = items[0]
    for p in items[1:]:
        out = And(out, p)
    return out


def big_or(parts: Iterable) -> Formula:
    """A1 ∨ A2 ∨ … (leer: falsch)."""
    items = [parse(p) for p in parts]
    if not items:
        return FALSE
    out = items[0]
    for p in items[1:]:
        out = Or(out, p)
    return out


def flatten(f: Formula, op: str) -> list[Formula]:
    """Kette gleicher Operatoren als Liste: flatten(A∧(B∧C), 'And') -> [A, B, C]."""
    if f.op == op and isinstance(f, _Binary):
        return flatten(f.a, op) + flatten(f.b, op)
    return [f]


# ==========================================================================
# Belegungen, Wahrheitswerte
# ==========================================================================

def bit(value) -> str:
    """True -> '1', False -> '0'."""
    return "1" if value else "0"


def assignments(names: Sequence[str], true_first: bool = False) -> list[dict]:
    """
    Alle Belegungen in Tabellenreihenfolge.
    Standard: 0 0 / 0 1 / 1 0 / 1 1 (wie in der Vorlesung). `true_first=True`
    beginnt mit 1 1.
    """
    names = list(names)
    if len(names) > MAX_VARS:
        raise ValueError(f"Zu viele Variablen ({len(names)}); erlaubt sind höchstens {MAX_VARS}")
    combos = itertools.product([True, False] if true_first else [False, True], repeat=len(names))
    return [dict(zip(names, c)) for c in combos]


def truth_vector(f, names: Sequence[str] | None = None) -> list[bool]:
    """Wahrheitswerte von f in Tabellenreihenfolge."""
    f = parse(f)
    names = list(names) if names is not None else f.variables()
    return [f.eval(env) for env in assignments(names)]


def models(f, names: Sequence[str] | None = None) -> list[dict]:
    """Alle Belegungen, die f wahr machen."""
    f = parse(f)
    names = list(names) if names is not None else f.variables()
    return [env for env in assignments(names) if f.eval(env)]


def _joint_vars(*fs) -> list[str]:
    return sorted(set().union(*(parse(f).variables() for f in fs)))


def is_tautology(f) -> bool:
    f = parse(f)
    return all(f.eval(env) for env in assignments(f.variables()))


def is_contradiction(f) -> bool:
    f = parse(f)
    return not any(f.eval(env) for env in assignments(f.variables()))


def is_satisfiable(f) -> bool:
    return not is_contradiction(f)


def equivalent(f, g) -> bool:
    """f ≡ g: gleiche Wahrheitswerte bei jeder Belegung."""
    f, g = parse(f), parse(g)
    return all(f.eval(env) == g.eval(env) for env in assignments(_joint_vars(f, g)))


def entails(wb, q) -> bool:
    """WB ⊨ Q: überall, wo WB wahr ist, ist auch Q wahr."""
    wb, q = parse(wb), parse(q)
    return all(q.eval(env) for env in assignments(_joint_vars(wb, q)) if wb.eval(env))


# ==========================================================================
# Formatierung beliebiger Grössen
# ==========================================================================

def fmt(e, decimal: bool = False, digits: int = 4) -> str:
    """Lesbarer Text für die Konsole: Formeln, Zahlen, Wahrheitswerte, Listen, Mengen."""
    if isinstance(e, Formula):
        return e.to_str("unicode")
    if isinstance(e, bool):
        return "wahr" if e else "falsch"
    if isinstance(e, (list, tuple)):
        return ", ".join(fmt(v, decimal, digits) for v in e)
    if isinstance(e, (set, frozenset)):
        return "{" + ", ".join(sorted(fmt(v, decimal, digits) for v in e)) + "}"
    if isinstance(e, float):
        return f"{e:.{digits}f}".rstrip("0").rstrip(".") if decimal else str(e)
    if isinstance(e, str):
        return e
    try:
        import sympy as sp
        if isinstance(e, sp.Basic):
            s = sp.sstr(e).replace("**", "^").replace("*", "·").replace("oo", "∞")
            return s.replace("log(", "ln(")
    except ImportError:
        pass
    return str(e)


def tex(e, decimal: bool = False, digits: int = 4) -> str:
    """LaTeX für Formeln, Zahlen, SymPy-Terme, Wahrheitswerte, Listen oder Text."""
    if isinstance(e, Formula):
        return e.tex()
    if isinstance(e, str):
        return e if "\\" in e or "$" in e else r"\text{" + e + "}"
    if isinstance(e, bool):
        return r"\text{" + ("wahr" if e else "falsch") + "}"
    if isinstance(e, (list, tuple)):
        return r",\; ".join(tex(v, decimal, digits) for v in e)
    if isinstance(e, (set, frozenset)):
        return r"\{" + r",\; ".join(sorted(tex(v, decimal, digits) for v in e)) + r"\}"
    if isinstance(e, int):
        return str(e)
    if isinstance(e, float):
        return f"{e:.{digits}f}".rstrip("0").rstrip(".") if decimal else str(e)
    try:
        import sympy as sp
        if isinstance(e, sp.Basic):
            return sp.latex(e, ln_notation=True, mul_symbol="dot")
    except ImportError:
        pass
    return r"\text{" + str(e) + "}"
