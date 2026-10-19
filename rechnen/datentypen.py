#!/usr/bin/env python3
"""
Semesterwoche 1: Elementare Datentypen.

boolean · integer · float · set · list · matrix · function — mit dem Rechenweg,
wie er in der Prüfung auf dem Papier stünde.

    from dmatkit.rechnen.datentypen import eval_bool, int_op, to_binary, frac_to_binary
    eval_bool("False or True and False and True or False")   # Serie 1, Aufgabe 1a
    int_op(10, "%", 3)                                        # 10 % 3 = 1, weil 3·3 + 1 = 10
    to_binary(409)                                            # (409)₂ = 110011001
    frac_to_binary("0.3")                                     # (0.3)₂ = 0.0(1001)
    repeating_to_fraction("2.10(365)")                        # 42031/19980
    ieee754("0.00(10)", bits=32, base=2)                      # Aufgabe 5a
    twos_complement(-10, 8)                                   # 11110110

Schreibweise für Perioden: die Periode steht in runden Klammern, `0.0(1001)`
heisst 0.0 1001 1001 1001 … (in der Vorlesung: Überstrich).

Korrekturen gegenüber den Unterlagen (richtig gerechnet):
* Serie 1.md (Mitschrift), Binärdarstellung von 409: »204 = 102 * 1 + 0« muss
  »204 = 102 · 2 + 0« heissen (Tippfehler, das Ergebnis 110011001 stimmt).
* Serie 1.md, Aufgabe 1b: Die Mitschrift schreibt »False and True and not True
  or False«, im Übungsblatt steht »False and True or False and not True or
  False«. Beide Fassungen werden ausgewertet; beide ergeben False.
* Serie 1.md, Aufgabe 1c ist unvollständig (hört nach der Abkürzung auf): richtig
  zu Ende gerechnet ergibt sie True.
* Der Python-Algorithmus `decToBin` der Folie liefert für x = 0 die leere
  Zeichenkette; `dec_to_bin(0)` liefert hier '0'.
* Übungsblatt: »(0.3)₂ = 0.01001« ist die Periode-Schreibweise mit Überstrich
  (0.0 1001 …); hier `0.0(1001)`.
"""

from __future__ import annotations

import itertools
import math
import re
from decimal import Decimal
from fractions import Fraction

import sympy as sp

from .core import parse
from .render import Result, Step, Table

__all__ = [
    # Grenzen
    "MAX_NUMBER", "MAX_BITS", "MAX_EXPR",
    # Ergebnisklassen
    "BoolEvalResult", "IntEvalResult", "ArrayEvalResult", "IntOpResult",
    "BinaryResult", "FractionResult", "FracBinaryResult", "SqrtBinaryResult",
    "FloatBitsResult", "FloatDecodeResult", "FloatCompareResult",
    "TwosComplementResult", "RangeResult", "OverflowResult",
    "SetOpResult", "SetRelationResult", "PowerSetResult", "ProductResult",
    "PartitionResult", "BitstringResult", "IndexResult", "DimensionResult",
    "FunctionCheckResult",
    # Ausdrücke
    "eval_bool", "eval_int", "broadcast_calc",
    # Ganzzahlen
    "int_op", "divmod_steps",
    # Zahlsysteme
    "to_base", "to_binary", "from_base", "from_binary", "dec_to_bin",
    "repeating_to_fraction", "frac_to_binary", "sqrt_binary",
    # Gleitkomma
    "ieee754", "ieee754_decode", "float_compare",
    # Zweierkomplement
    "twos_complement", "from_twos_complement", "int_range", "overflow_check",
    # Mengen, Listen, Matrizen, Funktionen
    "set_op", "set_relation", "power_set", "cartesian_product", "is_partition",
    "set_bitstring", "list_index", "matrix_index", "matrix_dim", "check_function",
]

# ==========================================================================
# Grenzen
# ==========================================================================

MAX_NUMBER = 10 ** 18        # grösste erlaubte Eingabezahl (Betrag)
MAX_BITS = 64                # grösste Bitbreite
MAX_EXPR = 300               # Zeichen pro Ausdruck
MAX_ARRAY = 400              # Elemente pro Liste/Matrix
MAX_POWERSET = 10            # Elemente einer Menge für die Potenzmenge (2^10 = 1024 Zeilen)
MAX_SET = 200                # Elemente einer Menge
MAX_FRAC_BITS = 2000         # Nachkommabits
MAX_SQRT_DIGITS = 1000       # Nachkommastellen bei sqrt_binary
_MAX_INTER = 10 ** 120       # Zwischenergebnisse in Ausdrücken
_MAX_REP_DIGITS = 40         # Ziffern vor/nach dem Komma bzw. in der Periode

_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
_DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _sub(n) -> str:
    return str(n).translate(_SUB)


def _sup(n) -> str:
    return str(n).translate(_SUP)


def _pv(v) -> str:
    """Zahl für den Text, negative Zahlen in Klammern."""
    s = _vtext(v, False)
    return f"({s})" if s.startswith("-") else s


def _check_int(x, name: str = "Zahl", limit: int = MAX_NUMBER) -> int:
    if isinstance(x, bool):
        raise ValueError(f"{name} muss eine ganze Zahl sein, nicht {x!r}")
    if isinstance(x, sp.Integer):
        x = int(x)
    if isinstance(x, str):
        s = x.strip()
        if len(s) > 40 or not re.fullmatch(r"[-+]?\d+", s):
            raise ValueError(f"{name} muss eine ganze Zahl sein, nicht {x!r}")
        x = int(s)
    if not isinstance(x, int):
        raise ValueError(f"{name} muss eine ganze Zahl (integer) sein, nicht {x!r}")
    if abs(x) > limit:
        raise ValueError(f"{name} ist zu gross: |{name}| darf höchstens {limit} sein")
    return x


def _check_bits(bits, lo: int = 2) -> int:
    b = _check_int(bits, "Bitbreite", MAX_BITS)
    if b < lo:
        raise ValueError(f"Die Bitbreite muss mindestens {lo} sein (höchstens {MAX_BITS})")
    return b


def _R(f: Fraction) -> sp.Rational:
    return sp.Rational(f.numerator, f.denominator)


def _ftext(f: Fraction) -> str:
    """Bruch als Text: Ganzzahl, endliche Dezimalzahl oder p/q."""
    if f.denominator == 1:
        return str(f.numerator)
    d = f.denominator
    a = b = 0
    while d % 2 == 0:
        d //= 2
        a += 1
    while d % 5 == 0:
        d //= 5
        b += 1
    if d == 1:
        k = max(a, b)
        n = abs(f.numerator) * 10 ** k // f.denominator
        s = str(n).rjust(k + 1, "0")
        s = s[:-k] + "." + s[-k:]
        return ("-" if f < 0 else "") + s
    return f"{f.numerator}/{f.denominator}"


def _group(bits: str, n: int = 4) -> str:
    """Bits in Vierergruppen (von rechts) für die Lesbarkeit."""
    rev = bits[::-1]
    return " ".join(rev[i:i + n] for i in range(0, len(rev), n))[::-1]


# ==========================================================================
# Ausdrücke: Einlesen (Baum mit Klammern) und Schritt-für-Schritt-Auswertung
# ==========================================================================
#
# core.parse liest boolesche Ausdrücke ein und liefert den Vergleichswert.
# Für die Anzeige brauchen wir aber die Klammern des Originals und eigene
# Reduktionsschritte — deshalb ein eigener kleiner Baum (_N), der Klammern behält.

_TOK = re.compile(r"\s*(?:(?P<num>\d+(?:\.\d+)?)|(?P<op>//|&&|\|\||[-+*%/()\[\],&|!~¬∧∨])"
                  r"|(?P<word>[A-Za-z_][A-Za-z0-9_']*))")
_AND_W, _OR_W, _NOT_W = {"and", "und", "&&", "&", "∧"}, {"or", "oder", "||", "|", "∨"}, \
    {"not", "nicht", "!", "~", "¬"}
_BOOL_W = {"true": True, "wahr": True, "false": False, "falsch": False}
_LOGIC_ONLY = {"xor", "implies", "iff"}

_LEVEL = {"or": 1, "and": 2, "not": 3, "+": 4, "-": 4, "*": 5, "//": 5, "%": 5, "/": 5, "neg": 6}
_MODE_OPS = {
    "bool": {"and", "or", "not"},
    "int": {"+", "-", "*", "//", "%", "neg"},
    "array": {"and", "or", "not", "+", "-", "*", "//", "%", "neg"},
    "float": {"+", "-", "*", "/", "neg"},
}
_OP_NAME = {"+": "Addition", "-": "Subtraktion", "*": "Multiplikation",
            "//": "Ganzzahldivision //", "%": "Modulo %", "/": "Division /",
            "neg": "Vorzeichen (Negation)", "not": "not auswerten",
            "and": "and auswerten", "or": "or auswerten"}


class _Arr:
    """Liste (1×n) oder Matrix (m×n) mit Zeilen als Python-Listen."""

    def __init__(self, rows, kind: str):
        self.rows, self.kind = rows, kind

    @property
    def shape(self):
        return (len(self.rows), len(self.rows[0]))

    def flat(self):
        return [v for r in self.rows for v in r]

    def py(self):
        return list(self.rows[0]) if self.kind == "list" else [list(r) for r in self.rows]

    def text(self, abbrev=False) -> str:
        if self.kind == "list":
            return "[" + ", ".join(_vtext(v, abbrev) for v in self.rows[0]) + "]"
        return "[" + ", ".join("[" + ", ".join(_vtext(v, abbrev) for v in r) + "]"
                               for r in self.rows) + "]"

    def __eq__(self, o):
        return isinstance(o, _Arr) and self.kind == o.kind and self.rows == o.rows


def _vtext(v, abbrev: bool = True) -> str:
    if isinstance(v, bool):
        return ("T" if v else "F") if abbrev else ("True" if v else "False")
    if isinstance(v, _Arr):
        return v.text(abbrev)
    if isinstance(v, Fraction):
        return _ftext(v)
    return str(v)


def _num(x, name):
    """int, Fraction oder float (als Dezimalzahl gelesen) → int oder Fraction."""
    if isinstance(x, bool):
        raise ValueError(f"{name}: Zahl erwartet")
    if isinstance(x, int):
        return _check_int(x, name)
    if isinstance(x, float):
        if not math.isfinite(x):
            raise ValueError(f"{name}: keine endliche Zahl")
        x = Fraction(repr(x))
    if isinstance(x, Fraction):
        if abs(x.numerator) > MAX_NUMBER * 10 ** 6 or x.denominator > MAX_NUMBER:
            raise ValueError(f"{name}: Zahl zu gross")
        return int(x) if x.denominator == 1 else x
    raise ValueError(f"{name}: Zahl erwartet, nicht {x!r}")


def _conv_value(name: str, v, mode: str):
    """Wert aus den Schlüsselwort-Argumenten (bool, int, Liste, Matrix) prüfen."""
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return _check_int(v, name)
    if isinstance(v, (float, Fraction)) and mode == "array":
        return _num(v, name)
    if isinstance(v, (list, tuple)):
        if mode != "array":
            raise ValueError(f"{name}: Listen und Matrizen gehören zu broadcast_calc")
        if not v:
            raise ValueError(f"{name}: leere Liste")
        if all(isinstance(r, (list, tuple)) for r in v):
            rows = [list(r) for r in v]
            kind = "matrix"
        elif any(isinstance(r, (list, tuple)) for r in v):
            raise ValueError(f"{name}: gemischte Zeilen und Zahlen")
        else:
            rows, kind = [list(v)], "list"
        n = len(rows[0])
        if n == 0 or any(len(r) != n for r in rows):
            raise ValueError(f"{name}: eine Matrix muss rechteckig sein (alle Zeilen gleich lang)")
        if len(rows) * n > MAX_ARRAY:
            raise ValueError(f"{name}: zu viele Einträge (höchstens {MAX_ARRAY})")
        flat = [x for r in rows for x in r]
        if all(isinstance(x, bool) for x in flat):
            pass
        elif any(isinstance(x, bool) for x in flat):
            raise ValueError(f"{name}: Wahrheitswerte und Zahlen lassen sich nicht mischen")
        else:
            rows = [[_num(x, f"{name}-Eintrag") for x in r] for r in rows]
        return _Arr(rows, kind)
    raise ValueError(f"{name}: erwartet bool, integer, Liste oder Matrix, nicht {v!r}")


class _N:
    """Knoten: val (Wert), neg/not (einstellig, Kind a), bin (a op b)."""
    __slots__ = ("kind", "op", "a", "b", "val", "label", "paren", "start", "pos")

    def __init__(self, kind, op=None, a=None, b=None, val=None, label=None, start=0, pos=0):
        self.kind, self.op, self.a, self.b = kind, op, a, b
        self.val, self.label, self.paren, self.start, self.pos = val, label, 0, start, pos

    def children(self):
        return [c for c in (self.a, self.b) if c is not None]


def _lex(text: str, env: dict, mode: str):
    toks, pos = [], 0
    while pos < len(text):
        m = _TOK.match(text, pos)
        if not m or m.end() == pos:
            if not text[pos:].strip():
                break
            bad = text[pos:].strip()[:1]
            raise ValueError(f"Unerwartetes Zeichen {bad!r} im Ausdruck {text!r}")
        idx = len(toks)
        pos = m.end()
        if m.lastgroup == "num":
            toks.append(("num", m.group("num"), idx))
        elif m.lastgroup == "op":
            v = m.group("op")
            kind = "and" if v in _AND_W else "or" if v in _OR_W else "not" if v in _NOT_W else "op"
            toks.append((kind, v, idx))
        else:
            w = m.group("word")
            low = w.lower()
            if w in env:
                toks.append(("name", w, idx))
            elif low in _AND_W:
                toks.append(("and", w, idx))
            elif low in _OR_W:
                toks.append(("or", w, idx))
            elif low in _NOT_W:
                toks.append(("not", w, idx))
            elif low in _BOOL_W:
                toks.append(("bool", _BOOL_W[low], idx))
            elif w in ("T", "F") and mode in ("bool", "array"):
                toks.append(("bool", w == "T", idx))
            elif low in _LOGIC_ONLY:
                raise ValueError(f"»{w}« gehört zur Aussagenlogik (Serie 2); hier gibt es nur "
                                 f"not, and, or. Ausdruck: {text!r}")
            else:
                raise ValueError(f"Unbekannter Name {w!r} im Ausdruck {text!r}. Werte für "
                                 f"Variablen als Schlüsselwort übergeben, z. B. A=True.")
    return toks


class _Parser:
    MAX_DEPTH = 60

    def __init__(self, toks, text, env, mode):
        self.t, self.i, self.text, self.env, self.mode, self.depth = toks, 0, text, env, mode, 0

    def kind(self):
        return self.t[self.i][0] if self.i < len(self.t) else None

    def val(self):
        return self.t[self.i][1]

    def idx(self):
        return self.t[self.i][2]

    def at_op(self, *ops):
        return self.kind() == "op" and self.val() in ops

    def fail(self, msg):
        raise ValueError(f"{msg} im Ausdruck {self.text!r}")

    def need(self, op):
        if op in _MODE_OPS[self.mode]:
            return
        if self.mode == "bool":
            self.fail(f"In einem booleschen Ausdruck sind nur True, False, not, and, or und "
                      f"Klammern erlaubt, nicht »{op}«")
        if self.mode == "int" and op == "/":
            self.fail("»/« liefert eine Kommazahl; für integer gilt die Ganzzahldivision »//«")
        if self.mode == "int":
            self.fail(f"Für integer sind +, -, *, //, % erlaubt, nicht »{op}«")
        self.fail(f"Der Operator »{op}« ist hier nicht erlaubt")

    def parse(self):
        if not self.t:
            raise ValueError("Leerer Ausdruck")
        n = self.or_()
        if self.i != len(self.t):
            self.fail(f"Unerwartetes {self.val()!r}")
        return n

    def or_(self):
        n = self.and_()
        while self.kind() == "or":
            i = self.idx()
            self.i += 1
            self.need("or")
            n = _N("bin", "or", n, self.and_(), start=n.start, pos=i)
        return n

    def and_(self):
        n = self.not_()
        while self.kind() == "and":
            i = self.idx()
            self.i += 1
            self.need("and")
            n = _N("bin", "and", n, self.not_(), start=n.start, pos=i)
        return n

    def not_(self):
        if self.kind() == "not":
            i = self.idx()
            self.i += 1
            self.need("not")
            self.depth += 1
            if self.depth > self.MAX_DEPTH:
                self.fail("Ausdruck zu tief verschachtelt")
            c = self.not_()
            self.depth -= 1
            return _N("not", "not", c, start=i, pos=i)
        return self.arith()

    def arith(self):
        n = self.term()
        while self.at_op("+", "-"):
            op, i = self.val(), self.idx()
            self.i += 1
            self.need(op)
            n = _N("bin", op, n, self.term(), start=n.start, pos=i)
        return n

    def term(self):
        n = self.unary()
        while self.at_op("*", "//", "%", "/"):
            op, i = self.val(), self.idx()
            self.i += 1
            self.need(op)
            n = _N("bin", op, n, self.unary(), start=n.start, pos=i)
        return n

    def number(self, negative=False):
        s = self.val()
        self.i += 1
        if "." in s:
            if self.mode not in ("float", "array"):
                self.fail(f"{s} ist kein integer (Kommazahlen: float_compare, frac_to_binary)")
            v = Fraction(s)
        else:
            if self.mode == "bool":
                self.fail("In einem booleschen Ausdruck sind keine Zahlen erlaubt")
            v = int(s)
        if len(s) > 19 or abs(v) > MAX_NUMBER:
            self.fail(f"Zahl {s} zu gross (höchstens {MAX_NUMBER})")
        return -v if negative else v

    def unary(self):
        if self.at_op("-"):
            i = self.idx()
            self.i += 1
            self.need("neg")
            if self.kind() == "num":
                return _N("val", val=self.number(True), start=i, pos=i)
            self.depth += 1
            if self.depth > self.MAX_DEPTH:
                self.fail("Ausdruck zu tief verschachtelt")
            c = self.unary()
            self.depth -= 1
            return _N("neg", "neg", c, start=i, pos=i)
        if self.at_op("+"):
            self.fail("Ein Plus als Vorzeichen wird nicht benötigt")
        return self.atom()

    def atom(self):
        k = self.kind()
        if k == "num":
            i = self.idx()
            return _N("val", val=self.number(), start=i, pos=i)
        if k == "bool":
            i, v = self.idx(), self.val()
            self.i += 1
            if self.mode in ("int", "float"):
                self.fail("Wahrheitswerte sind hier nicht erlaubt")
            return _N("val", val=v, start=i, pos=i)
        if k == "name":
            i, name = self.idx(), self.val()
            self.i += 1
            v = _conv_value(name, self.env[name], self.mode)
            if isinstance(v, bool) and self.mode in ("int", "float"):
                self.fail(f"{name} ist ein Wahrheitswert; hier sind Zahlen gefragt")
            if isinstance(v, int) and not isinstance(v, bool) and self.mode == "bool":
                self.fail(f"{name} ist eine Zahl; hier sind Wahrheitswerte gefragt")
            return _N("val", val=v, label=name, start=i, pos=i)
        if self.at_op("("):
            i = self.idx()
            self.i += 1
            self.depth += 1
            if self.depth > self.MAX_DEPTH:
                self.fail("Ausdruck zu tief verschachtelt")
            n = self.or_()
            if not self.at_op(")"):
                self.fail("Schliessende Klammer fehlt")
            self.i += 1
            self.depth -= 1
            n.paren += 1
            n.start = min(n.start, i)
            return n
        if self.at_op("[") and self.mode == "array":
            return self.array()
        if k is None:
            self.fail("Der Ausdruck endet zu früh")
        self.fail(f"Unerwartetes {self.val()!r}")

    def lit(self):
        neg = False
        if self.at_op("-"):
            neg = True
            self.i += 1
        if self.kind() == "num":
            return self.number(neg)
        if self.kind() == "bool" and not neg:
            v = self.val()
            self.i += 1
            return v
        self.fail("In einer Liste stehen nur Zahlen oder True/False")

    def array(self):
        i0 = self.idx()
        self.i += 1                         # [

        def items():
            out = []
            while True:
                if self.at_op("["):
                    self.i += 1
                    out.append(items())
                else:
                    out.append(self.lit())
                if self.at_op(","):
                    self.i += 1
                    continue
                if self.at_op("]"):
                    self.i += 1
                    return out
                self.fail("In einer Liste fehlt ein Komma oder die Klammer »]«")

        v = _conv_value("Liste", items(), "array")
        return _N("val", val=v, start=i0, pos=i0)


def _check_expr(expr) -> str:
    if not isinstance(expr, str):
        raise ValueError(f"Der Ausdruck muss ein Text sein, nicht {expr!r}")
    if not expr.strip():
        raise ValueError("Leerer Ausdruck")
    if len(expr) > MAX_EXPR:
        raise ValueError(f"Ausdruck zu lang (mehr als {MAX_EXPR} Zeichen)")
    return expr.strip()


def _build(expr, env, mode):
    text = _check_expr(expr)
    toks = _lex(text, env, mode)
    return text, toks, _Parser(toks, text, env, mode).parse()


# -- Anzeige ---------------------------------------------------------------

def _render(n, abbrev, mark=None, lead=True, strip=False):
    if n.kind == "val":
        s = n.label if n.label else _vtext(n.val, abbrev)
        if not n.label and isinstance(n.val, (int, Fraction)) and not isinstance(n.val, bool) \
                and n.val < 0 and not lead:
            s = f"({s})"
    elif n.kind == "neg":
        s = "-" + _render(n.a, abbrev, mark, False)
    elif n.kind == "not":
        c = _render(n.a, abbrev, mark, False)
        s = "not" + ("" if n.a.paren else " ") + c
    else:
        inner_lead = lead or n.paren > 0
        s = (f"{_render(n.a, abbrev, mark, inner_lead)} {n.op} "
             f"{_render(n.b, abbrev, mark, False)}")
    if n is mark:
        s = "==" + s + "=="
    p = 0 if strip else n.paren
    return "(" * p + s + ")" * p


# -- Rechnen ---------------------------------------------------------------

def _bin_scalar(op, x, y):
    if op in ("and", "or"):
        if not (isinstance(x, bool) and isinstance(y, bool)):
            raise ValueError(f"»{op}« verknüpft Wahrheitswerte, nicht {_vtext(x, False)} "
                             f"und {_vtext(y, False)}")
        return (x and y) if op == "and" else (x or y)
    if isinstance(x, bool) or isinstance(y, bool):
        raise ValueError(f"Wahrheitswerte und Zahlen lassen sich mit »{op}« nicht mischen")
    if op == "+":
        r = x + y
    elif op == "-":
        r = x - y
    elif op == "*":
        r = x * y
    elif op == "//":
        if y == 0:
            raise ValueError(f"Division durch 0: {x} // 0 ist nicht definiert")
        r = x // y
    elif op == "%":
        if y == 0:
            raise ValueError(f"Modulo 0: {x} % 0 ist nicht definiert (Rest einer Division durch 0)")
        r = x % y
    else:
        raise ValueError(f"Unbekannter Operator {op}")
    if isinstance(r, Fraction) and r.denominator == 1:
        r = int(r)
    if abs(r) > _MAX_INTER:
        raise ValueError("Zwischenergebnis zu gross")
    return r


def _un_scalar(op, x):
    if op == "not":
        if not isinstance(x, bool):
            raise ValueError(f"»not« erwartet einen Wahrheitswert, nicht {_vtext(x, False)}")
        return not x
    if isinstance(x, bool):
        raise ValueError("Das Vorzeichen »-« erwartet eine Zahl, keinen Wahrheitswert")
    return -x


def _expand(v, shape, kind):
    return _Arr([[v] * shape[1] for _ in range(shape[0])], kind)


def _broadcast(x, y):  # ponytail: nur die drei Fälle der Serie (1×1, m×1, 1×n), kein volles NumPy-Broadcasting
    """Gibt (X, Y, Notizen) zurück, beide auf gleiche Grösse gebracht (Regeln der Serie 1)."""
    notes = []
    if not isinstance(x, _Arr):
        X = _expand(x, y.shape, y.kind)
        notes.append(f"{_vtext(x, False)} → {X.text()}")
        return X, y, notes
    if not isinstance(y, _Arr):
        Y = _expand(y, x.shape, x.kind)
        notes.append(f"{_vtext(y, False)} → {Y.text()}")
        return x, Y, notes
    if x.kind != y.kind:
        raise ValueError("Eine Liste lässt sich nur mit einer Liste (oder einem Einzelwert) "
                         "verrechnen, eine Matrix nur mit einer Matrix")
    if x.shape == y.shape:
        return x, y, notes
    if x.kind == "list":
        raise ValueError(f"Listen müssen gleich lang sein ({x.shape[1]} ≠ {y.shape[1]}); "
                         "nur ein Einzelwert wird auf die Listenlänge erweitert (Broadcasting)")
    big, small = (x, y) if x.shape[0] * x.shape[1] >= y.shape[0] * y.shape[1] else (y, x)
    m, n = big.shape
    sm, sn = small.shape
    if not ((sm, sn) == (1, 1) or (sm, sn) == (m, 1) or (sm, sn) == (1, n)):
        raise ValueError(f"Broadcasting nicht möglich: Eine {m}×{n} Matrix lässt sich nur mit einer "
                         f"1×1, {m}×1 oder 1×{n} Matrix verrechnen, nicht mit {sm}×{sn}")
    ex = _Arr([[small.rows[0 if sm == 1 else i][0 if sn == 1 else j] for j in range(n)]
               for i in range(m)], "matrix")
    notes.append(f"{small.text()} ({sm}×{sn}) → {ex.text()} ({m}×{n})")
    return (x, ex, notes) if big is x else (ex, y, notes)


def _apply_bin(op, x, y):
    if not isinstance(x, _Arr) and not isinstance(y, _Arr):
        return _bin_scalar(op, x, y), []
    X, Y, notes = _broadcast(x, y)
    rows = [[_bin_scalar(op, a, b) for a, b in zip(r1, r2)] for r1, r2 in zip(X.rows, Y.rows)]
    return _Arr(rows, X.kind), notes


def _apply_un(op, x):
    if isinstance(x, _Arr):
        return _Arr([[_un_scalar(op, a) for a in r] for r in x.rows], x.kind)
    return _un_scalar(op, x)


def _eval_tree(n):
    """Unabhängige Kontrollrechnung (rekursiv, ohne Schritte)."""
    if n.kind == "val":
        return n.val
    if n.kind in ("neg", "not"):
        return _apply_un(n.op, _eval_tree(n.a))
    return _apply_bin(n.op, _eval_tree(n.a), _eval_tree(n.b))[0]


def _div_text(a, b) -> str:
    q, r = a // b, a % b
    return f"{_pv(a)} = {_pv(q)} · {_pv(b)} + {_pv(r)}"


def _describe(op, x, y, res, notes, abbrev=False):
    """(detail, items) für einen Reduktionsschritt."""
    items = [("Broadcasting", n) for n in notes]
    if op in ("not", "neg"):
        sym = "not " if op == "not" else "-"
        return f"{sym}{_pv(x) if op == 'neg' else _vtext(x, abbrev)} = {_vtext(res, abbrev)}", items
    if isinstance(x, _Arr) or isinstance(y, _Arr):
        return (f"{_vtext(x, False)} {op} {_vtext(y, False)} = {_vtext(res, False)}", items)
    if op in ("and", "or"):
        return f"{_vtext(x, abbrev)} {op} {_vtext(y, abbrev)} = {_vtext(res, abbrev)}", items
    d = f"{_pv(x)} {op} {_pv(y)} = {_pv(res)}"
    if op == "//":
        d += f", denn {_div_text(x, y)}"
    elif op == "%":
        d += f", weil {_pv(x // y)} · {_pv(y)} + {_pv(res)} = {_pv(x)}"
    return d, items


def _rule(mode_bool, op, in_paren, has_same):
    parts = []
    if in_paren:
        parts.append("Klammer zuerst")
    if mode_bool:
        parts.append({"not": "not vor and vor or", "and": "and vor or",
                      "or": "or zuletzt"}[op])
    else:
        parts.append({"neg": "Vorzeichen zuerst", "*": "Punkt vor Strich", "//": "Punkt vor Strich",
                      "%": "Punkt vor Strich", "/": "Punkt vor Strich", "+": "Strich nach Punkt",
                      "-": "Strich nach Punkt"}[op])
    if has_same:
        parts.append("von links nach rechts")
    return " · ".join(parts)


def _paren_count(n):
    return (1 if n.paren > 0 else 0) + sum(_paren_count(c) for c in n.children())


def _collect(n, acc):
    acc.append(n)
    for c in n.children():
        _collect(c, acc)


def _reduce(root, abbrev: bool):
    """
    Wertet den Baum Schritt für Schritt aus: innerste linke Klammer zuerst, darin
    Vorrang (not, dann and / Punkt, dann or / Strich), gleiche Stufe von links.
    Gibt die Liste der Schritte (Dicts) zurück; `root` wird dabei verändert.
    """
    recs = []
    limit = 4 * len(_nodes_of(root)) + 10   # ponytail: einfache Schranke gegen Endlosschleifen statt Beweis
    while not (root.kind == "val" and root.paren == 0):
        limit -= 1
        if limit < 0:
            raise RuntimeError("Interner Fehler: Auswertung endet nicht")
        allnodes = []
        _collect(root, allnodes)
        inner = [p for p in allnodes if p.paren > 0
                 and not any(_paren_count(c) for c in p.children())]
        group = min(inner, key=lambda p: p.start) if inner else root
        if group.kind == "val":                      # »(True)« → True
            line = _render(root, abbrev, mark=group)
            group.paren -= 1
            recs.append(dict(op="paren", title="Klammer auflösen", rule="Klammer zuerst",
                             detail=f"({_vtext(group.val, abbrev)}) = {_vtext(group.val, abbrev)}",
                             items=[], line=line, sub=_vtext(group.val, abbrev),
                             result=group.val))
            continue
        gnodes = []
        _collect(group, gnodes)
        ops = [n for n in gnodes if n.kind != "val"]
        red = [n for n in ops if all(c.kind == "val" for c in n.children())]
        # ponytail: genau eine Teilrechnung pro Schritt (die Serie wertet unabhängige Teile oft gleichzeitig)
        node = max(red, key=lambda n: (_LEVEL[n.op], -n.pos))
        has_same = any(o is not node and _LEVEL[o.op] == _LEVEL[node.op] for o in ops)
        x = node.a.val
        y = node.b.val if node.b is not None else None
        res, notes = (_apply_un(node.op, x), []) if node.b is None else _apply_bin(node.op, x, y)
        line = _render(root, abbrev, mark=node)
        sub = _render(node, abbrev, strip=True)
        detail, items = _describe(node.op, x, y, res, notes, abbrev)
        recs.append(dict(op=node.op, title=_OP_NAME[node.op],
                         rule=_rule(abbrev_bool(node.op), node.op, group.paren > 0, has_same),
                         detail=detail, items=items, line=line, sub=sub, result=res))
        node.kind, node.val, node.label = "val", res, None
        node.a = node.b = None
        node.paren = max(node.paren - 1, 0)
    return recs


def abbrev_bool(op) -> bool:
    return op in ("not", "and", "or")


def _nodes_of(root):
    acc = []
    _collect(root, acc)
    return acc


def _steps_for(task, text, root0_line, recs, final, env_items, abbrev, intro):
    steps = [Step("Aufgabe", intro, [("", root0_line)])]
    if env_items:
        steps.append(Step("Werte einsetzen", "", env_items))
    for r in recs:
        steps.append(Step(r["title"], r["detail"], [("", r["line"])] + r["items"], rule=r["rule"]))
    steps.append(Step("Ergebnis", "Es bleibt ein einzelner Wert übrig.",
                      [("", _vtext(final, False))]))
    return steps


def _env_items(root):
    seen, out = set(), []
    for n in _nodes_of(root):
        if n.kind == "val" and n.label and n.label not in seen:
            seen.add(n.label)
            out.append((n.label, _vtext(n.val, False)))
    return out


# ==========================================================================
# Boolesche Ausdrücke
# ==========================================================================

class BoolEvalResult(Result):
    """
    Wert eines booleschen Ausdrucks, Schritt für Schritt.

    Attribute: `expr` (Eingabe), `value` (bool), `lines` (Zeilen wie in der Mitschrift:
    Originalausdruck, dann je Schritt der Ausdruck mit ==markiertem== nächstem Teil
    in T/F-Kurzschreibweise, zuletzt der Wert), `reductions` (Liste von Dicts mit
    `op`, `sub`, `result`, `line`), `checked` (True: stimmt mit `core.Formula.eval` überein).
    """

    title = "Boolescher Ausdruck"

    def __init__(self, expr, value, lines, reductions, checked, formula):
        super().__init__()
        self.expr, self.value, self.lines = expr, value, lines
        self.reductions, self.checked, self.formula = reductions, checked, formula
        self.summary = f"{expr} = {_vtext(value, False)}"
        self.highlights = [("Ergebnis", _vtext(value, False))]

    @property
    def n_steps(self):
        return len(self.reductions)


def eval_bool(expr, short: bool = True, **env) -> BoolEvalResult:
    """
    Boolescher Ausdruck in Python-Schreibweise, Schritt für Schritt (Serie 1, Aufgabe 1).

    Vorrang `not` > `and` > `or`, gleiche Operatoren von links nach rechts, Klammern
    zuerst (innerste linke zuerst). In jedem Schritt steht der als Nächstes
    ausgewertete Teil zwischen `==`; Wahrheitswerte sind mit T/F abgekürzt
    (`short=False`: True/False). Variablen als Schlüsselwort: `eval_bool("A and not B", A=True, B=False)`.

        r = eval_bool("False or True and False and True or False")
        r.value            # False
        r.lines            # ['False or True and False and True or False',
                           #  'F or ==T and F== and T or F', ...]
        eval_bool("not(not True or not(False and not True)) or False")   # Aufgabe 1d
    """
    text, toks, root = _build(expr, env, "bool")
    # core.parse liest denselben Ausdruck (Prüfung der Schreibweise und Vergleichswert)
    norm = " ".join(
        ("(" if v == "(" else ")" if v == ")" else str(v)) if k == "op"
        else (str(env[v]) if k == "name" else str(v)) for k, v, _ in toks)
    f = parse(norm.replace("( ", "(").replace(" )", ")"))
    if f.variables():
        raise ValueError(f"Variablen ohne Wert: {', '.join(f.variables())}")
    expected = f.eval({})
    tree_value = _eval_tree(root)
    abbrev = short
    line0 = _render(root, False)
    env_items = _env_items(root)
    recs = _reduce(root, abbrev)
    value = root.val
    if value != expected or value != tree_value:
        raise RuntimeError("Interner Fehler: die Schritte stimmen nicht mit core.Formula.eval überein")
    lines = [line0] + [r["line"] for r in recs] + [_vtext(value, abbrev)]
    res = BoolEvalResult(text, value, lines, recs, True, f)
    res.steps = _steps_for("bool", text, line0, recs, value, env_items, abbrev,
                           "Vorrang: not vor and vor or; gleiche Operatoren von links nach rechts; "
                           "Klammern zuerst. Der nächste Teil steht zwischen ==…==.")
    if short:
        res.steps.insert(1 + (1 if env_items else 0), Step(
            "Abkürzen", "True = T, False = F (nur zum schnelleren Schreiben).",
            [("", _render(_build(text, env, "bool")[2], True))]))
    return res


# ==========================================================================
# Ganzzahl-Ausdrücke
# ==========================================================================

class IntEvalResult(Result):
    """
    Wert eines integer-Ausdrucks mit +, -, *, //, %, Klammern.
    Attribute: `expr`, `value` (int), `lines`, `reductions`.
    """

    title = "Ganzzahl-Ausdruck"

    def __init__(self, expr, value, lines, reductions):
        super().__init__()
        self.expr, self.value, self.lines, self.reductions = expr, value, lines, reductions
        self.summary = f"{expr} = {value}"
        self.highlights = [("Ergebnis", value)]


def eval_int(expr, **env) -> IntEvalResult:
    """
    integer-Ausdruck Schritt für Schritt: Klammern zuerst, dann Vorzeichen, dann
    `*` `//` `%` (von links), dann `+` `-` (von links). Python-Regel für negative Zahlen.

        eval_int("10 // 3 + 10 % 3 * 2")     # 3 + 1 * 2 = 5
        eval_int("(-7) // 2")                # -4
    """
    text, toks, root = _build(expr, env, "int")
    expected = _eval_tree(root)
    line0 = _render(root, False)
    env_items = _env_items(root)
    recs = _reduce(root, False)
    if root.val != expected:
        raise RuntimeError("Interner Fehler: Schritte und Kontrollrechnung stimmen nicht überein")
    lines = [line0] + [r["line"] for r in recs] + [str(root.val)]
    res = IntEvalResult(text, root.val, lines, recs)
    res.steps = _steps_for("int", text, line0, recs, root.val, env_items, False,
                           "Klammern zuerst; * // % vor + -; gleiche Stufe von links nach rechts.")
    return res


# ==========================================================================
# Listen und Matrizen mit Broadcasting
# ==========================================================================

class ArrayEvalResult(Result):
    """
    Komponentenweise Rechnung mit Listen/Matrizen und Broadcasting.
    Attribute: `value` (Zahl, bool, Liste oder Liste von Zeilen), `kind`
    ('scalar' | 'list' | 'matrix'), `shape` ((m, n) oder None), `lines`, `reductions`.
    """

    title = "Rechnen mit Listen und Matrizen (komponentenweise, Broadcasting)"

    def __init__(self, expr, value, kind, shape, lines, reductions):
        super().__init__()
        self.expr, self.value, self.kind, self.shape = expr, value, kind, shape
        self.lines, self.reductions = lines, reductions
        shown = _vtext(_wrap(value) if isinstance(value, list) else value, False)
        self.summary = f"{expr} = {shown}"
        self.highlights = [("Ergebnis", shown)]


def _wrap(py):
    return _Arr([list(py)], "list") if py and not isinstance(py[0], list) else _Arr(py, "matrix")


def broadcast_calc(expr, **env) -> ArrayEvalResult:
    """
    Rechnen mit Listen (`[1, 2, 3]`) und Matrizen (`[[1, 2], [3, 4]]`): Operatoren
    wirken komponentenweise; Einzelwerte, 1×1-, m×1- und 1×n-Matrizen werden
    auf die Grösse der m×n-Matrix erweitert (Broadcasting). Namen als Schlüsselwort.

        broadcast_calc("3 * [2, 5, 4, -2]")                         # [6, 15, 12, -6]
        broadcast_calc("not [True, False, False, True, False]")
        broadcast_calc("[[1,1,2,2],[2,2,0,0]] + 1 - [[2],[1]]")      # Lerngerüst Serie 1 (f)
        broadcast_calc("A + B", A=[[1, 2], [3, 4]], B=[[0, 4], [1, 1]])
    """
    text, toks, root = _build(expr, env, "array")
    expected = _eval_tree(root)
    line0 = _render(root, False)
    env_items = _env_items(root)
    recs = _reduce(root, False)
    v = root.val
    if v != expected:
        raise RuntimeError("Interner Fehler: Schritte und Kontrollrechnung stimmen nicht überein")
    if isinstance(v, _Arr):
        kind, shape, py = v.kind, v.shape, v.py()
    else:
        kind, shape, py = "scalar", None, v
    lines = [line0] + [r["line"] for r in recs] + [_vtext(v, False)]
    res = ArrayEvalResult(text, py, kind, shape, lines, recs)
    res.steps = _steps_for("array", text, line0, recs, v, env_items, False,
                           "Operatoren wirken komponentenweise; ein Einzelwert wird zu einer Liste "
                           "gleicher Grösse erweitert (Broadcasting).")
    return res


# ==========================================================================
# Ganzzahlen: + - * // %
# ==========================================================================

class IntOpResult(Result):
    """
    Eine Rechnung mit zwei ganzen Zahlen.

    Attribute: `a`, `b`, `op`, `value`; bei `//`, `%` und `divmod` zusätzlich
    `quotient`, `remainder` und `equation` (»a = q·b + r«), `check` (True, wenn a = q·b + r
    und der Rest das Vorzeichen des Teilers hat bzw. 0 ist).
    """

    title = "Rechnen mit ganzen Zahlen"

    def __init__(self, a, op, b, value, quotient=None, remainder=None):
        super().__init__()
        self.a, self.b, self.op, self.value = a, b, op, value
        self.quotient, self.remainder = quotient, remainder
        self.equation = None
        self.check = True
        if quotient is not None:
            self.equation = _div_text(a, b)
            self.check = (a == quotient * b + remainder
                          and (remainder == 0 or (remainder > 0) == (b > 0))
                          and abs(remainder) < abs(b))
        shown = (f"({quotient}, {remainder})" if op == "divmod" else str(value))
        name = "divmod" if op == "divmod" else op
        self.summary = (f"divmod({a}, {b}) = {shown}" if op == "divmod"
                        else f"{a} {name} {b} = {shown}")
        if op == "divmod":
            self.highlights = [("Quotient q", quotient), ("Rest r", remainder)]
        else:
            self.highlights = [(f"{a} {op} {b}", value)]


def _div_steps(a, b, want):
    q, r = a // b, a % b
    steps = []
    frac = Fraction(a, b)
    exact = _ftext(frac) if frac.denominator != 1 else str(frac.numerator)
    steps.append(Step(
        "Ganzzahldivision: Quotient q",
        f"{a} : {b} = {exact}" + ("" if frac.denominator == 1 else
                                   f"; q ist die grösste ganze Zahl ≤ {exact}, "
                                   f"also {q} (nach unten gerundet, nicht abgeschnitten)"),
        [("q", q)], rule="Ganzzahldivision //"))
    steps.append(Step(
        "Rest r",
        f"r = a − q · b = {a} − {_pv(q)} · {_pv(b)} = {r}",
        [("r", r)], rule="Rest %",
        note=("Der Rest liegt immer zwischen 0 und dem Teiler (bei negativem Teiler: zwischen "
              "dem Teiler und 0)." )))
    steps.append(Step(
        "Probe",
        f"{_div_text(a, b)}" + (f";  0 ≤ {r} < {b}" if b > 0 else f";  {b} < {r} ≤ 0"
                                if r != 0 else f";  r = 0"),
        [("a", a), ("q·b + r", q * b + r)], rule="a = q · b + r"))
    if a < 0 or b < 0:
        qa = abs(a) // abs(b)
        qj = -qa if (a < 0) != (b < 0) else qa
        steps[-1].note = (f"Python rundet nach unten (Richtung −∞). Java und C schneiden dagegen "
                          f"ab: {a} / {b} = {qj} mit Rest {a - qj * b}. Auf dem Papier gilt hier "
                          f"die Python-Regel, wenn nichts anderes verlangt ist.")
    return steps


def int_op(a, op, b) -> IntOpResult:
    """
    a op b für ganze Zahlen mit op in `+ - * // %` (`divmod`: beides) und Probe a = q·b + r.

        int_op(10, "//", 3)    # 3
        int_op(10, "%", 3)     # 1, weil 3·3 + 1 = 10
        int_op(-7, "//", 2)    # -4, Rest 1  (-7 = -4·2 + 1)
        int_op(5, "%", 0)      # ValueError: Modulo 0 ist nicht definiert
    """
    a, b = _check_int(a, "a"), _check_int(b, "b")
    op = {"÷": "//", "mod": "%", "·": "*", "×": "*", "−": "-"}.get(op, op)
    if op == "divmod":
        return divmod_steps(a, b)
    if op not in ("+", "-", "*", "//", "%"):
        raise ValueError(f"Unbekannter Operator {op!r}; erlaubt sind + - * // % (und divmod)")
    if op in ("//", "%") and b == 0:
        raise ValueError(f"Division durch 0: {a} {op} 0 ist nicht definiert "
                         f"(es gibt keine Zahl q mit 0 · q + r = {a} und 0 ≤ r < 0)")
    value = {"+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b,
             "//": lambda: a // b, "%": lambda: a % b}[op]()
    res = IntOpResult(a, op, b, value,
                      *((a // b, a % b) if op in ("//", "%") else (None, None)))
    res.steps = [Step("Aufgabe", f"{a} {op} {b}", [("", f"{_pv(a)} {op} {_pv(b)}")])]
    if op in ("+", "-", "*"):
        names = {"+": "Addition", "-": "Subtraktion", "*": "Multiplikation"}
        res.steps.append(Step(names[op], f"{_pv(a)} {op} {_pv(b)} = {value}", [("", value)],
                              rule=names[op]))
    else:
        res.steps += _div_steps(a, b, op)
        res.steps.append(Step("Ergebnis",
                              f"{a} {op} {b} = {value}" +
                              (f", weil {_pv(a // b)} · {_pv(b)} + {_pv(a % b)} = {_pv(a)}"
                               if op == "%" else f", denn {_div_text(a, b)}"),
                              [("", value)]))
    return res


def divmod_steps(a, b) -> IntOpResult:
    """
    divmod(a, b) = (q, r) mit a = q·b + r — Quotient und Rest in einem Rechenweg.

        divmod_steps(25, 2)    # q = 12, r = 1:  25 = 12·2 + 1
    """
    a, b = _check_int(a, "a"), _check_int(b, "b")
    if b == 0:
        raise ValueError(f"Division durch 0: divmod({a}, 0) ist nicht definiert")
    q, r = divmod(a, b)
    res = IntOpResult(a, "divmod", b, (q, r), q, r)
    res.steps = [Step("Aufgabe", f"divmod({a}, {b}): finde q und r mit a = q · b + r",
                      [("", f"divmod({a}, {b})")])] + _div_steps(a, b, "divmod")
    return res


# ==========================================================================
# Zahlsysteme: integer <-> Binär (und andere Basen)
# ==========================================================================

class BinaryResult(Result):
    """
    Ganze Zahl in einer Basis (meist 2).

    Attribute: `number` (int), `base`, `digits` (Liste, höchste Stelle zuerst),
    `text` (z. B. '110011001'), `weights` (Stellenwerte, höchster zuerst),
    `rows` (Divisionsschritte als Tupel (zahl, quotient, rest)), `width`.
    """

    title = "Zahl in Binärdarstellung"

    def __init__(self, number, base, digits, rows, width=None):
        super().__init__()
        self.number, self.base, self.digits, self.rows = number, base, digits, rows
        self.text = "".join(_DIGITS[d] for d in digits)
        self.width = width
        self.weights = [base ** k for k in range(len(digits) - 1, -1, -1)]
        self.bits = self.text                     # Alias für die Basis 2
        if base != 2:
            self.title = f"Zahl zur Basis {base}"
        self.summary = f"{number} = ({self.text}){_sub(base)}"
        self.highlights = [(f"({number}){_sub(10)}", f"({self.text}){_sub(base)}")]

    @property
    def value(self):
        return self.number


def _div_rows(n: int, base: int):
    rows, q = [], n
    while True:
        q2, r = divmod(q, base)
        rows.append((q, q2, r))
        q = q2
        if q == 0:
            return rows


def _terms(digits, base):
    n = len(digits)
    out = []
    for i, d in enumerate(digits):
        if d == 0:
            continue
        k = n - 1 - i
        w = base ** k
        out.append(str(w) if d == 1 else f"{d}·{w}")
    return " + ".join(out) if out else "0"


def _place_table(digits, base, number, caption=""):
    n = len(digits)
    header = [str(base ** k) for k in range(n - 1, -1, -1)] + ["Zahl"]
    return Table(header, [[_DIGITS[d] for d in digits] + [str(number)]], caption=caption)


def to_base(n, base: int = 2, method: str = "both", width: int | None = None) -> BinaryResult:
    """
    Natürliche Zahl n in der Basis `base` (2–36).

    `method`: "division" (wiederholt durch die Basis teilen, Reste von unten nach oben
    lesen), "table" (Stellenwerttabelle 16 8 4 2 1) oder "both". `width`: mit führenden
    Nullen auf diese Stellenzahl auffüllen.

        to_base(25, 2).text      # '11001'
        to_base(255, 16).text    # 'FF'
    """
    n = _check_int(n, "n")
    base = _check_int(base, "Basis", 36)
    if base < 2:
        raise ValueError("Die Basis muss mindestens 2 sein")
    if n < 0:
        raise ValueError(f"{n} ist negativ: negative Zahlen stellt man im Zweierkomplement "
                         f"dar (twos_complement({n}, bits))")
    if method not in ("both", "division", "table"):
        raise ValueError("method muss 'both', 'division' oder 'table' sein")
    rows = _div_rows(n, base)
    digits = [r for _, _, r in reversed(rows)]
    if n == 0:
        digits = [0]
    if width is not None:
        width = _check_int(width, "Breite", MAX_BITS)
        if width < len(digits):
            raise ValueError(f"{n} braucht {len(digits)} Stellen, die Breite {width} reicht nicht")
        digits = [0] * (width - len(digits)) + digits
    res = BinaryResult(n, base, digits, rows, width)
    res.steps.append(Step("Aufgabe", f"Zahl {n} zur Basis {base} schreiben", [("", n)]))
    if method in ("both", "division"):
        trows = [[f"{q} = {q2} · {base} + {r}", _DIGITS[r]] for q, q2, r in rows]
        res.steps.append(Step(
            f"Methode: fortlaufend durch {base} teilen",
            f"Immer mit Rest durch {base} teilen, bis der Quotient 0 ist.",
            table=Table(["Division", "Rest"], trows), rule="Division mit Rest"))
        res.steps.append(Step(
            "Reste von unten nach oben lesen",
            "Der letzte Rest ist die höchste Stelle. Die Reste in der Reihenfolge des "
            "Aufschreibens ergeben die Zahl rückwärts.",
            [("Reste von oben nach unten", " ".join(_DIGITS[r] for _, _, r in rows)),
             ("umgekehrt", "".join(_DIGITS[d] for d in digits))]))
    if method in ("both", "table"):
        core_digits = digits if width is None else digits[width - len(rows):] if n else [0]
        core_digits = [d for d in core_digits]
        res.steps.append(Step(
            "Methode: Stellenwerttabelle",
            f"Über jeder Stelle steht ihr Wert (Potenzen von {base}); die Summe der Stellen mit "
            f"Wert × Ziffer ist die Zahl.",
            [("Zerlegung", f"{n} = {_terms(core_digits, base)}")],
            table=_place_table(digits, base, n), rule="Stellenwerttabelle"))
    back = sum(d * base ** k for k, d in enumerate(reversed(digits)))
    res.steps.append(Step(
        "Gegenprobe", f"Stellenwerte addieren: {_terms(digits, base)} = {back}",
        [("", back)], note="✓" if back == n else "✗"))
    assert back == n
    return res


def to_binary(n, width: int | None = None, method: str = "both") -> BinaryResult:
    """
    Binärdarstellung einer natürlichen Zahl (Serie 1, Aufgabe 2: 409 → 110011001).

        to_binary(25).text            # '11001'
        to_binary(25, width=8).text   # '00011001'
        to_binary(123), to_binary(741)   # Lerngerüst (b)
    """
    return to_base(n, 2, method=method, width=width)


def dec_to_bin(x) -> str:
    """
    Die Funktion `decToBin : N → W` der Vorlesung (Folie, Frage 5) als einfache Funktion.
    Anders als die Folie liefert sie für 0 die Zeichenkette '0' (die Folie die leere).

        dec_to_bin(25)   # '11001'
    """
    return to_base(x, 2, method="division").text


def from_base(text, base: int = 2) -> BinaryResult:
    """
    Zahl aus einer Ziffernfolge zur Basis `base` (2–36) in eine Dezimalzahl.
    Stellenwerttabelle und Rückwärts-Einsetzen (Horner-Schema).

        from_base("11001", 2).number    # 25
        from_base("FF", 16).number      # 255
    """
    base = _check_int(base, "Basis", 36)
    if base < 2:
        raise ValueError("Die Basis muss mindestens 2 sein")
    s = str(text).strip().replace(" ", "").replace("_", "")
    s = re.sub(r"^\(?(.*?)\)?[₀-₉]*$", r"\1", s)
    if base == 2 and s.lower().startswith("0b"):
        s = s[2:]
    if not s:
        raise ValueError("Leere Ziffernfolge")
    if len(s) > (MAX_BITS if base == 2 else 40):
        raise ValueError(f"Ziffernfolge zu lang (höchstens {MAX_BITS if base == 2 else 40} Stellen)")
    digits = []
    for ch in s.upper():
        d = _DIGITS.find(ch)
        if d < 0 or d >= base:
            raise ValueError(f"»{ch}« ist keine Ziffer zur Basis {base}")
        digits.append(d)
    n = 0
    for d in digits:
        n = n * base + d
    res = BinaryResult(n, base, digits, [])
    res.title = f"Ziffernfolge zur Basis {base} in Dezimalzahl" if base != 2 else \
        "Binärzahl in Dezimalzahl"
    res.summary = f"({s.upper()}){_sub(base)} = {n}"
    res.highlights = [(f"({s.upper()}){_sub(base)}", n)]
    res.steps.append(Step("Aufgabe", f"({s.upper()}){_sub(base)} als Dezimalzahl",
                          [("", s.upper())]))
    res.steps.append(Step(
        "Stellenwerte", f"Jede Ziffer mal ihren Stellenwert ({base}^k), alles addieren.",
        [("Summe", f"{_terms(digits, base)} = {n}")], table=_place_table(digits, base, n),
        rule="Stellenwerttabelle"))
    horner, acc = [], 0
    for d in digits:
        horner.append([f"{acc} · {base} + {d}", str(acc * base + d)])
        acc = acc * base + d
    res.steps.append(Step(
        "Rückwärts einsetzen (Horner)", f"Von links: Wert · {base} + nächste Ziffer.",
        table=Table(["Rechnung", "Wert"], horner), rule="Horner-Schema"))
    return res


def from_binary(text):
    """
    Binärzahl → Dezimalzahl (`from_base(text, 2)`). Enthält der Text ein Komma oder eine
    Periode (`101.011`, `0.0(1001)`), wird daraus der Bruch (`repeating_to_fraction(text, 2)`).

        from_binary("11001").number      # 25
        from_binary("0.0(1001)").value   # 3/10
    """
    s = str(text)
    if "." in s or "," in s or "(" in s and "." in s:
        return repeating_to_fraction(text, 2)
    return from_base(text, 2)


# ==========================================================================
# Periodische Zahlen <-> Bruch
# ==========================================================================

class FractionResult(Result):
    """
    Bruch zu einer (periodischen) Kommazahl.

    Attribute: `value` (sympy Rational, gekürzt), `numerator`, `denominator`,
    `unreduced` ((Zähler, Nenner) vor dem Kürzen), `integer`, `pre` (Ziffern vor der
    Periode), `period`, `base`, `text` (Eingabe in der Form `2.10(365)`).
    """

    title = "Kommazahl als Bruch"

    def __init__(self, value, unreduced, integer, pre, period, base, text):
        super().__init__()
        self.value = _R(value)
        self.numerator, self.denominator = value.numerator, value.denominator
        self.unreduced, self.integer, self.pre, self.period = unreduced, integer, pre, period
        self.base, self.text = base, text
        sb = "" if base == 10 else _sub(base)
        shown = f"{self.numerator}/{self.denominator}" if self.denominator != 1 else str(self.numerator)
        self.summary = f"{text}{sb} = {shown}"
        self.highlights = [(f"{text}{sb}", self.value)]


_REP = re.compile(r"([-+]?)\s*([0-9A-Za-z]*)(?:\.([0-9A-Za-z]*)(?:\(([0-9A-Za-z]+)\))?)?")


def _parse_rep(text: str, base: int):
    """'−2.10(365)' → (sign, int_str, pre_str, period_str) mit Prüfung der Ziffern."""
    s = str(text).strip().replace(" ", "").replace(",", ".")
    s = s.replace("−", "-")
    m = _REP.fullmatch(s)
    if not m:
        raise ValueError(f"Keine Kommazahl mit Periode: {text!r} (Beispiel: 2.10(365) für 2.10365365…)")
    sign, ip, pre, per = m.group(1), m.group(2), m.group(3) or "", m.group(4) or ""
    if not ip:
        ip = "0"
    if "." not in s and "(" in s:
        raise ValueError(f"Die Periode braucht ein Komma: {text!r}")
    for part in (ip, pre, per):
        cap = MAX_FRAC_BITS if base == 2 else _MAX_REP_DIGITS
        if len(part) > cap:
            raise ValueError(f"Zu viele Ziffern (höchstens {cap} je Teil)")
        for ch in part.upper():
            d = _DIGITS.find(ch)
            if d < 0 or d >= base:
                raise ValueError(f"»{ch}« ist keine Ziffer zur Basis {base}: {text!r}")
    return sign, ip, pre, per


def _digits_val(s: str, base: int) -> int:
    v = 0
    for ch in s.upper():
        v = v * base + _DIGITS.index(ch)
    return v


def _rep_value(sign, ip, pre, per, base):
    k, p = len(pre), len(per)
    n_all = _digits_val(ip + pre + per, base)
    n_pre = _digits_val(ip + pre, base)
    if p:
        v = Fraction(n_all - n_pre, base ** (k + p) - base ** k)
    else:
        v = Fraction(n_pre, base ** k)
    return -v if sign == "-" else v


def repeating_to_fraction(text, base: int = 10) -> FractionResult:
    """
    (Periodische) Kommazahl → Bruch, mit dem Trick »zwei Zehnerpotenzen, subtrahieren«
    (Serie 1, Aufgabe 3). Die Periode steht in Klammern.

        repeating_to_fraction("2.10(365)")   # 42031/19980  (= 210155/99900)
        repeating_to_fraction("14.2(51)")    # 14109/990 = 4703/330
        repeating_to_fraction("14.05")       # 281/20      (= 1405/100)
        repeating_to_fraction("20.1(445)"), repeating_to_fraction("0.173(15)")   # Lerngerüst (c)
        repeating_to_fraction("0.0(1001)", base=2)   # 3/10  (Binärzahl)
    """
    base = _check_int(base, "Basis", 36)
    if base < 2:
        raise ValueError("Die Basis muss mindestens 2 sein")
    sign, ip, pre, per = _parse_rep(text, base)
    k, p = len(pre), len(per)
    value = _rep_value(sign, ip, pre, per, base)
    shown = f"{sign}{ip}" + (f".{pre}" + (f"({per})" if per else "") if (pre or per) else "")
    n_all = _digits_val(ip + pre + per, base)
    n_pre = _digits_val(ip + pre, base)
    B = lambda e: f"{base}^{e}" if base != 10 else f"10^{e}"       # noqa: E731
    pw = lambda e: base ** e                                       # noqa: E731
    sgn = "-" if sign == "-" else ""
    if p:
        unred = (n_all - n_pre, pw(k + p) - pw(k))
    else:
        unred = (n_pre, pw(k))
    res = FractionResult(value, unred, ip, pre, per, base, shown)
    res.steps.append(Step("Aufgabe", f"Bruch zu x = {shown}" +
                          (f" (Basis {base})" if base != 10 else ""), [("x", shown)]))
    if p:
        dot = f"{ip}{pre}{per}"
        res.steps.append(Step(
            "Zwei Vielfache bilden",
            f"Mit {B(k + p)} und {B(k)} multiplizieren: beide Zahlen haben danach dieselbe "
            f"Periode direkt hinter dem Komma.",
            [(f"{pw(k + p)} · x", f"{dot}.({per})"),
             (f"{pw(k)} · x", f"{ip}{pre}.({per})")],
            rule="Periode abschneiden"))
        res.steps.append(Step(
            "Subtrahieren", "Die Periode fällt weg.",
            [(f"({pw(k + p)} − {pw(k)}) · x", f"{n_all} − {n_pre} = {n_all - n_pre}")],
            rule="Differenz"))
        res.steps.append(Step(
            "Nach x auflösen",
            f"x = {n_all - n_pre} / {pw(k + p) - pw(k)}",
            [("x", f"{sgn}{n_all - n_pre}/{pw(k + p) - pw(k)}")]))
    elif k:
        res.steps.append(Step(
            "Endliche Kommazahl", f"Ohne Periode: Komma um {k} Stellen verschieben, durch {B(k)} teilen.",
            [("x", f"{sgn}{n_pre}/{pw(k)}")], rule="Nachkommastellen als Nenner"))
    else:
        res.steps.append(Step("Ganze Zahl", "Keine Nachkommastellen: x ist schon ein Bruch n/1.",
                              [("x", f"{sgn}{n_pre}")]))
    g = math.gcd(unred[0], unred[1])
    if value.denominator != 1 or unred[1] != 1:
        note = "(schon gekürzt)" if g == 1 else f"(mit {g} gekürzt)"
        res.steps.append(Step(
            "Kürzen", f"ggT({unred[0]}, {unred[1]}) = {g} {note}",
            [("x", f"{sgn}{value.numerator}/{value.denominator}" if value.denominator != 1
              else f"{sgn}{value.numerator}")], rule="Kürzen mit dem ggT"))
    if per and set(per) == {_DIGITS[base - 1]}:
        res.warnings.append(f"Periode aus lauter {_DIGITS[base - 1]}: {shown} ist dieselbe Zahl "
                            f"wie ein endlicher Wert (0.(9) = 1).")
    return res


# ==========================================================================
# Kommazahlen -> Binär
# ==========================================================================

def _to_fraction(x) -> Fraction:
    """Zahl, Bruch oder Text ('0.3', '1/6', '2.10(365)') → exakter Bruch."""
    if isinstance(x, bool):
        raise ValueError(f"Zahl erwartet, nicht {x!r}")
    if isinstance(x, int):
        f = Fraction(_check_int(x))
    elif isinstance(x, Fraction):
        f = x
    elif isinstance(x, sp.Rational):
        f = Fraction(int(x.p), int(x.q))
    elif isinstance(x, float):
        if not math.isfinite(x):
            raise ValueError(f"{x} ist keine endliche Zahl")
        f = Fraction(repr(x))        # ponytail: Dezimalschreibweise lesen (0.1 → 1/10), nicht den Binärwert des floats
    elif isinstance(x, str):
        s = x.strip().replace(",", ".")
        if "/" in s:
            a, _, b = s.partition("/")
            num, den = _check_int(a, "Zähler"), _check_int(b, "Nenner")
            if den == 0:
                raise ValueError(f"Der Nenner darf nicht 0 sein: {s}")
            f = Fraction(num, den)
        elif re.fullmatch(r"[-+]?(\d+\.?\d*|\.\d+)[eE][-+]?\d{1,3}", s):
            f = Fraction(s)               # Schreibweise 1e-10 wie in Python
        else:
            sign, ip, pre, per = _parse_rep(s, 10)
            f = _rep_value(sign, ip, pre, per, 10)
    else:
        raise ValueError(f"Keine Zahl: {x!r}")
    if abs(f.numerator) > 10 ** 400 or f.denominator > 10 ** 400:
        raise ValueError("Zähler oder Nenner zu gross")
    return f


class FracBinaryResult(Result):
    """
    Binärdarstellung einer rationalen Zahl.

    Attribute: `value` (sympy Rational), `sign` (1/-1), `int_bits` (Ganzzahlanteil),
    `pre`, `period` (Bits; `period` leer: endlich), `complete` (False: nach `max_bits`
    abgebrochen, ohne Wiederholung), `text` (z. B. '0.0(1001)'); `expand(n)` liefert die
    ersten n Nachkommabits, `rows` die Rechenzeilen (rest, bit, neuer_rest).
    """

    title = "Binärdarstellung einer Kommazahl"

    def __init__(self, value, sign, int_bits, pre, period, complete, rows, source):
        super().__init__()
        self.value = _R(value)
        self.sign, self.int_bits, self.pre, self.period = sign, int_bits, pre, period
        self.complete, self.rows = complete, rows
        frac = pre + (f"({period})" if period else "") + ("…" if not complete else "")
        self.text = ("-" if sign < 0 else "") + int_bits + ("." + frac if frac else "")
        self.source = source
        self.summary = f"{source} = ({self.text}){_sub(2)}"
        self.highlights = [(f"({source}){_sub(10)}", f"({self.text}){_sub(2)}")]

    def expand(self, n: int) -> str:
        """Die ersten n Nachkommabits (die Periode wird wiederholt)."""
        n = _check_int(n, "n", MAX_FRAC_BITS)
        if self.period:
            s = self.pre + self.period * (n // len(self.period) + 1)
        else:
            s = self.pre
        return (s + "0" * n)[:n]


def frac_to_binary(x, max_bits: int = 64) -> FracBinaryResult:
    """
    Dezimalzahl/Bruch → Binärdarstellung: Ganzzahlanteil durch 2 teilen, Nachkommaanteil
    fortlaufend mal 2 nehmen (Bits von oben nach unten). Eine wiederkehrende Rechnung wird
    erkannt (Periode); `max_bits` begrenzt die Nachkommabits, falls die Periode sehr lang ist.

        frac_to_binary("0.625").text     # '0.101'
        frac_to_binary("0.3").text       # '0.0(1001)'
        frac_to_binary("1/6").text       # '0.0(01)'
        frac_to_binary("0.1").text       # '0.0(0011)'       Lerngerüst (d)
        frac_to_binary("35.2").text      # '100011.(0011)'
    """
    mb = _check_int(max_bits, "max_bits", MAX_FRAC_BITS)
    if mb < 1:
        raise ValueError("max_bits muss mindestens 1 sein")
    f = _to_fraction(x)
    sign = -1 if f < 0 else 1
    a = abs(f)
    n = a.numerator // a.denominator
    r = a - n
    source = x if isinstance(x, str) else _ftext(f)
    src = str(source).strip()
    int_rows = _div_rows(n, 2)
    int_bits = "".join(str(rr) for _, _, rr in reversed(int_rows)) if n else "0"
    rows, seen, start = [], {}, None
    while r != 0 and len(rows) < mb:
        if r in seen:
            start = seen[r]
            break
        seen[r] = len(rows)
        t = r * 2
        bit = int(t)
        rows.append((r, bit, t - bit))
        r = t - bit
    else:
        if r != 0 and r in seen:
            start = seen[r]
    bits = "".join(str(b) for _, b, _ in rows)
    if r == 0:
        pre, period, complete = bits, "", True
    elif start is not None:
        pre, period, complete = bits[:start], bits[start:], True
    else:
        pre, period, complete = bits, "", False
    res = FracBinaryResult(f, sign, int_bits, pre, period, complete, rows, src)
    res.steps.append(Step("Aufgabe", f"({src}){_sub(10)} in Binärdarstellung", [("", src)]))
    if sign < 0:
        res.steps.append(Step("Vorzeichen", "Negativ: erst den Betrag umrechnen, dann − davor.",
                              [("|x|", _ftext(a))]))
    if n:
        irows = [[f"{q} = {q2} · 2 + {rr}", str(rr)] for q, q2, rr in int_rows]
        res.steps.append(Step(
            "Ganzzahlanteil: durch 2 teilen",
            f"Der Ganzzahlanteil {n} wird wie bei den integer-Zahlen umgerechnet (Reste von unten nach oben).",
            [("Ganzzahlanteil", int_bits)], table=Table(["Division", "Rest"], irows),
            rule="Division mit Rest"))
    if rows:
        trows = []
        for i, (rr, bit, rn) in enumerate(rows):
            mark = ""
            if start is not None and i == start:
                mark = "← ab hier wiederholt sich die Rechnung"
            if start is not None and i == len(rows) - 1:
                mark = f"← {_ftext(rn)} ist wieder da"
            trows.append([f"{_ftext(rr)} · 2 = {bit} + {_ftext(rn)}", str(bit), mark])
        res.steps.append(Step(
            "Nachkommaanteil: fortlaufend mal 2",
            "Der ganzzahlige Teil des Produkts ist das nächste Bit; mit dem Rest weiterrechnen, "
            "bis der Rest 0 ist (endlich) oder sich ein Rest wiederholt (Periode).",
            table=Table(["Rechnung", "Bit", ""], trows), rule="Multiplikation mit 2"))
        res.steps.append(Step(
            "Bits von oben nach unten lesen",
            "Anders als bei den integer-Zahlen: nicht von unten nach oben.",
            [("Nachkommabits", pre + (f"({period})" if period else "") + ("…" if not complete else ""))]))
    if not complete:
        res.warnings.append(f"Die Rechnung wurde nach {mb} Bits abgebrochen: keine Wiederholung "
                            f"gefunden (Periode länger als {mb}).")
    res.steps.append(Step("Ergebnis", "", [(f"({src}){_sub(10)}", f"({res.text}){_sub(2)}")]))
    if complete:
        back = _rep_value("-" if sign < 0 else "", int_bits, pre, period, 2)
        res.steps.append(Step(
            "Gegenprobe", "Zurückrechnen mit den Stellenwerten 2^-1, 2^-2, …" +
            (" (Periode: Differenztrick wie bei den Dezimalzahlen)" if period else ""),
            [("", _ftext(back))], note="✓" if back == f else "✗"))
        assert back == f
    return res


class SqrtBinaryResult(Result):
    """
    Quadratwurzel in Binärdarstellung auf n Nachkommastellen (abgeschnitten).
    Attribute: `text` ('1.0110101000…'), `digits`, `int_bits`, `frac_bits`, `floor_value`
    (Bruch, mit dem das Ergebnis gerechnet wird), `exact` (True, wenn die Wurzel genau
    so viele Stellen hat).
    """

    title = "Quadratwurzel in Binärdarstellung"

    def __init__(self, x, digits, int_bits, frac_bits, floor_value, exact):
        super().__init__()
        self.x, self.digits, self.int_bits, self.frac_bits = x, digits, int_bits, frac_bits
        self.floor_value, self.exact = floor_value, exact
        self.text = int_bits + ("." + frac_bits if frac_bits else "")
        self.value = _R(floor_value)
        self.summary = f"√{_ftext(x)} ≈ ({self.text}){_sub(2)} (auf {digits} Nachkommastellen abgeschnitten)"
        self.highlights = [(f"√{_ftext(x)}", f"({self.text}){_sub(2)}")]


def sqrt_binary(x, digits: int = 20) -> SqrtBinaryResult:
    """
    √x in Binärdarstellung mit `digits` Nachkommastellen (abgeschnitten, nicht gerundet)
    — Serie 1, Aufgabe 7c. Exakt mit Ganzzahlen: ⌊√x · 2^n⌋ = ⌊√(⌊x · 4^n⌋)⌋.

        sqrt_binary(2, 10).text      # '1.0110101000'
        sqrt_binary("0.25", 4).text  # '0.1000'
    """
    n = _check_int(digits, "Stellenzahl", MAX_SQRT_DIGITS)
    if n < 0:
        raise ValueError("Die Stellenzahl darf nicht negativ sein")
    f = _to_fraction(x)
    if f < 0:
        raise ValueError(f"Aus {_ftext(f)} gibt es keine reelle Wurzel (negativer Radikand)")
    # ponytail: Ganzzahl-Wurzel (math.isqrt) statt Bit-für-Bit-Verfahren
    big = (f.numerator * 4 ** n) // f.denominator
    m = math.isqrt(big)
    fv = Fraction(m, 2 ** n)
    exact = fv * fv == f
    s = bin(m)[2:].rjust(n + 1, "0")
    int_bits, frac_bits = (s[:-n], s[-n:]) if n else (s, "")
    res = SqrtBinaryResult(f, n, int_bits, frac_bits, fv, exact)
    res.steps = [
        Step("Aufgabe", f"√{_ftext(f)} mit {n} Nachkommabits", [("x", _ftext(f))]),
        Step("Verschieben", f"Das Komma um {n} Bits nach rechts schieben heisst mit 2^{n} multiplizieren; "
             f"unter der Wurzel ist das 4^{n} = {4 ** n}.",
             [(f"⌊x · 4^{n}⌋", big)], rule="⌊√a⌋ mit ganzzahliger Wurzel"),
        Step("Ganzzahlige Wurzel", f"m = ⌊√{big}⌋ (grösste ganze Zahl mit m² ≤ {big})",
             [("m", m), ("m²", m * m), ("(m+1)²", (m + 1) ** 2)]),
        Step("Binär schreiben", f"m = ({bin(m)[2:]}){_sub(2)}; das Komma steht {n} Stellen von rechts.",
             [("", res.text)]),
        Step("Gegenprobe", f"Es gilt ({_ftext(fv)})² ≤ x < ({_ftext(fv + Fraction(1, 2 ** n))})²",
             [("", fv * fv <= f < (fv + Fraction(1, 2 ** n)) ** 2)], note="✓"),
    ]
    assert fv * fv <= f < (fv + Fraction(1, 2 ** n)) ** 2
    if not exact:
        res.steps[-1].detail += f"; der Fehler ist kleiner als 2^-{n}."
    return res


# ==========================================================================
# float: IEEE 754
# ==========================================================================

_FORMATS = {32: (8, 23, 127), 64: (11, 52, 1023)}     # ponytail: nur float/double wie in der Serie, kein half/quad
# Bits: (Exponent, Mantisse, Bias)


def _round_half_even(f: Fraction) -> int:
    fl = f.numerator // f.denominator
    rem = f - fl
    if rem > Fraction(1, 2):
        return fl + 1
    if rem == Fraction(1, 2):
        return fl + (fl & 1)
    return fl


def _frac_bits(f: Fraction, n: int) -> str:
    """Die ersten n Bits des Nachkommaanteils f (0 ≤ f < 1), abgeschnitten."""
    return bin((f.numerator << n) // f.denominator)[2:].rjust(n, "0") if n else ""


def _binary_text(a: Fraction, max_bits: int = 120) -> str:
    """Betrag a als Binärtext mit Periode, z. B. '100011.(0011)'."""
    n = a.numerator // a.denominator
    r = a - n
    ib = bin(n)[2:]
    seen, bits, start = {}, [], None
    while r != 0 and len(bits) < max_bits:
        if r in seen:
            start = seen[r]
            break
        seen[r] = len(bits)
        t = r * 2
        b = int(t)
        bits.append(str(b))
        r = t - b
    s = "".join(bits)
    if r == 0:
        return ib + ("." + s if s else "")
    if start is not None:
        return ib + "." + s[:start] + "(" + s[start:] + ")"
    return ib + "." + s + "…"


class FloatBitsResult(Result):
    """
    Darstellung als Fliesskommazahl nach IEEE 754.

    Attribute: `bits` (32 oder 64), `sign`, `exponent_field` (int), `exponent_bits`,
    `mantissa_bits`, `pattern` (alle Bits ohne Leerzeichen), `hex`, `exponent`
    (tatsächlicher Exponent e in 1.x · 2^e), `bias`, `kind` ('zero' | 'subnormal' |
    'normal' | 'inf'), `exact` (Eingabewert, Rational), `stored` (tatsächlich gespeicherter
    Wert, Rational), `error` (stored − exact), `rounded` (True, wenn gerundet wurde).
    """

    title = "Fliesskommazahl nach IEEE 754"

    def __init__(self, bits, sign, exp_field, exp_bits, man_bits, exponent, bias, kind,
                 exact, stored, rounded):
        super().__init__()
        self.bits, self.sign, self.exponent_field = bits, sign, exp_field
        self.exponent_bits, self.mantissa_bits = exp_bits, man_bits
        self.exponent, self.bias, self.kind = exponent, bias, kind
        self.exact, self.stored, self.rounded = _R(exact), (_R(stored) if stored is not None else None), rounded
        self.error = (self.stored - self.exact) if stored is not None else None
        self.pattern = f"{sign}{exp_bits}{man_bits}"
        self.hex = format(int(self.pattern, 2), f"0{bits // 4}X")
        self.title = f"{bits}-Bit-Fliesskommazahl nach IEEE 754 ({'float' if bits == 32 else 'double'})"
        self.summary = f"{sign} | {exp_bits} | {man_bits}"
        self.highlights = [("Vorzeichen", sign), ("Exponent", exp_bits), ("Mantisse", man_bits),
                           ("Hex", self.hex)]


def ieee754(x, bits: int = 32, base: int = 10) -> FloatBitsResult:
    """
    Zahl als Fliesskommazahl nach IEEE 754, 32 Bit (float) oder 64 Bit (double)
    — Serie 1, Aufgabe 5. Normalisieren auf 1.xxx · 2^e, Exponent + Bias, Mantisse (ohne
    führende 1) mit Rundung zur nächsten Zahl (bei Gleichstand zur geraden).

        ieee754("0.00(10)", 32, base=2)   # Aufgabe 5a: 0 | 01111100 | 01010101010101010101011
        ieee754("0.00(10)", 64, base=2)   # Exponent 01111111100
        ieee754("35.2", 32)               # Aufgabe 5b: 0 | 10000100 | 00011001100110011001101
        ieee754("0.0(110)", 32, base=2)   # Lerngerüst (e)

    `base=2`: x ist eine Binärzahl als Text (mit Komma und Periode in Klammern).
    """
    bits = _check_int(bits, "Bits", 64)
    if bits not in _FORMATS:
        raise ValueError("Es gibt hier nur 32 Bit (float) und 64 Bit (double)")
    eb, mb, bias = _FORMATS[bits]
    base = _check_int(base, "Basis", 36)
    if base == 2:
        if not isinstance(x, str):
            raise ValueError("Mit base=2 muss die Binärzahl als Text übergeben werden, z. B. '0.00(10)'")
        sign_s, ip, pre, per = _parse_rep(x, 2)
        v = _rep_value(sign_s, ip, pre, per, 2)
        src = x.strip()
    else:
        v = _to_fraction(x)
        src = x.strip() if isinstance(x, str) else _ftext(v)
    s = 1 if v < 0 else 0
    a = abs(v)
    e_min = 1 - bias
    notes = []
    if a == 0:
        kind, e, ef, m, rounded, stored = "zero", 0, 0, 0, False, Fraction(0)
    else:
        e = a.numerator.bit_length() - a.denominator.bit_length()
        while Fraction(2) ** e > a:
            e -= 1
        while Fraction(2) ** (e + 1) <= a:
            e += 1
        if e < e_min:
            kind = "subnormal"
            q = a / Fraction(2) ** (e_min - mb)
            m = _round_half_even(q)
            rounded = q != m
            ef = 0
            if m == 2 ** mb:
                kind, ef, m = "normal", 1, 0
            stored = (Fraction(2 ** mb + m, 2 ** mb) if ef else Fraction(m, 2 ** mb)) \
                * Fraction(2) ** e_min
            e = e_min
        else:
            kind = "normal"
            sig = a / Fraction(2) ** e
            q = (sig - 1) * 2 ** mb
            m = _round_half_even(q)
            rounded = q != m
            if m == 2 ** mb:
                m, e = 0, e + 1
            if e > bias:
                kind, ef, m, stored, rounded = "inf", 2 ** eb - 1, 0, None, True
            else:
                ef = e + bias
                stored = (1 + Fraction(m, 2 ** mb)) * Fraction(2) ** e
    exp_bits = format(ef, f"0{eb}b")
    man_bits = format(m, f"0{mb}b")
    if stored is not None and s:
        stored = -stored
    res = FloatBitsResult(bits, str(s), ef, exp_bits, man_bits, e, bias, kind, v, stored, rounded)
    st = res.steps
    st.append(Step("Aufgabe", f"{src}" + (" (Binärzahl)" if base == 2 else "") +
                   f" als {bits}-Bit-Fliesskommazahl ({eb} Bit Exponent, {mb} Bit Mantisse, Bias {bias})",
                   [("", src)]))
    if v == 0:
        st.append(Step("Null", "Null hat lauter Nullen: Vorzeichen, Exponent und Mantisse sind 0.",
                       [("", res.pattern)]))
        return res
    bt = _binary_text(a) if base != 2 else src.lstrip("+-")
    if base != 2:
        st.append(Step("In Binär umrechnen", "Ganzzahlanteil durch 2 teilen, Nachkommaanteil mal 2 "
                       "(frac_to_binary).", [(f"|{src}|", bt)], rule="Zahl in Binärdarstellung"))
    sign_txt = "negativ" if s else "positiv"
    st.append(Step("Vorzeichen", f"Die Zahl ist {sign_txt}.", [("Vorzeichenbit", s)]))
    if kind == "subnormal":
        res.warnings.append(f"Der Betrag ist kleiner als 2^{e_min}: subnormale Zahl "
                            f"(Exponentenfeld 0, keine führende 1).")
    if kind in ("normal", "inf"):
        shift = (f"{abs(e)} Stellen nach {'links' if e > 0 else 'rechts'}" if e != 0
                 else "keine Verschiebung")
        sig_full = a / Fraction(2) ** e
        bits_after = _frac_bits(sig_full - 1 if kind == "normal" else Fraction(0), mb + 6)
        st.append(Step(
            "Normalisieren: 1.xxx · 2^e",
            f"Das Komma so verschieben, dass genau eine 1 davor steht ({shift}); der Exponent "
            f"ist e = {e}. Die führende 1 wird nicht gespeichert.",
            [("normalisiert", f"1.{bits_after}… · 2^{e}"), ("e", e)], rule="Normalisieren"))
        if kind == "normal":
            st.append(Step(
                "Exponent + Bias",
                f"Exponentenfeld = e + {bias} = {e} + {bias} = {ef} = ({exp_bits}){_sub(2)} "
                f"(auf {eb} Bit mit Nullen aufgefüllt).",
                [("Exponentenfeld", exp_bits)], rule="Exponent + Bias"))
            first, rest = bits_after[:mb], bits_after[mb:]
            if rounded:
                r_exact = ((sig_full - 1) * 2 ** mb) % 1
                how = ("grösser als ½ → aufrunden" if r_exact > Fraction(1, 2)
                       else "genau ½ → zur geraden Mantisse runden" if r_exact == Fraction(1, 2)
                       else "kleiner als ½ → abrunden")
                st.append(Step(
                    "Mantisse runden",
                    f"Nach den ersten {mb} Bits folgen noch {rest}…: der Rest ist {how}.",
                    [("erste Bits", first), ("gerundet", man_bits)], rule="Runden"))
            else:
                st.append(Step("Mantisse", f"Die Nachkommabits von 1.xxx, auf {mb} Bit mit Nullen "
                                f"aufgefüllt (es musste nicht gerundet werden).",
                               [("Mantisse", man_bits)]))
        else:
            res.warnings.append(f"Der Betrag ist zu gross für {bits} Bit (Exponent > {bias}): "
                                f"Überlauf, gespeichert wird ∞.")
    st.append(Step(
        "Zusammensetzen", "Vorzeichen | Exponent | Mantisse",
        [("", f"{res.sign} | {exp_bits} | {man_bits}"), ("Hex", res.hex)],
        table=Table(["Vorzeichen (1)", f"Exponent ({eb})", f"Mantisse ({mb})"],
                    [[res.sign, exp_bits, man_bits]])))
    if stored is not None and rounded:
        st.append(Step("Genauigkeit",
                       "Wegen der Rundung wird nicht exakt der Eingabewert gespeichert.",
                       [("gespeichert", float(stored)), ("Fehler", float(res.error))]))
    return res


class FloatDecodeResult(Result):
    """
    Wert einer IEEE-754-Bitfolge. Attribute: `bits`, `sign`, `exponent_field`, `exponent`,
    `mantissa_bits`, `kind` ('zero' | 'subnormal' | 'normal' | 'inf' | 'nan'), `value`
    (sympy Rational, `oo`, `-oo` oder `nan`).
    """

    title = "IEEE-754-Bitfolge lesen"

    def __init__(self, bits, sign, ef, exponent, man, kind, value):
        super().__init__()
        self.bits, self.sign, self.exponent_field, self.exponent = bits, sign, ef, exponent
        self.mantissa_bits, self.kind, self.value = man, kind, value
        shown = (str(value) if not value.is_Rational or value.q == 1
                 else f"{value} ≈ {float(value)!r}")
        self.summary = f"Wert = {shown}"
        self.highlights = [("Wert", shown)]


def ieee754_decode(pattern, bits: int | None = None) -> FloatDecodeResult:
    """
    Bitfolge (32 oder 64 Zeichen 0/1, Leerzeichen erlaubt) → Zahl.

        ieee754_decode("0 01111100 01010101010101010101011").value   # 11184811/67108864
    """
    s = str(pattern).replace(" ", "").replace("_", "")
    if not re.fullmatch(r"[01]+", s):
        raise ValueError("Die Bitfolge darf nur aus 0 und 1 bestehen")
    n = len(s) if bits is None else _check_int(bits, "Bits", 64)
    if n not in _FORMATS or len(s) != n:
        raise ValueError(f"Erwartet werden genau 32 oder 64 Bit, nicht {len(s)}")
    eb, mb, bias = _FORMATS[n]
    sign, eps, man = int(s[0]), s[1:1 + eb], s[1 + eb:]
    ef, m = int(eps, 2), int(man, 2)
    sg = -1 if sign else 1
    if ef == 2 ** eb - 1:
        kind = "nan" if m else "inf"
        value = sp.nan if m else (sp.oo if sg > 0 else -sp.oo)
        e = ef - bias
    elif ef == 0:
        e = 1 - bias
        kind = "zero" if m == 0 else "subnormal"
        value = _R(sg * Fraction(m, 2 ** mb) * Fraction(2) ** e)
    else:
        e = ef - bias
        kind = "normal"
        value = _R(sg * (1 + Fraction(m, 2 ** mb)) * Fraction(2) ** e)
    res = FloatDecodeResult(n, str(sign), ef, e, man, kind, value)
    res.steps.append(Step("Felder", f"{n} Bit = 1 Vorzeichen | {eb} Exponent | {mb} Mantisse",
                          [("Vorzeichen", s[0]), ("Exponent", eps), ("Mantisse", man)],
                          table=Table(["Vorzeichen", "Exponent", "Mantisse"], [[s[0], eps, man]])))
    if kind in ("inf", "nan"):
        res.steps.append(Step("Sonderwert", "Exponentenfeld nur aus Einsen: "
                              + ("Mantisse 0 → unendlich" if kind == "inf" else "Mantisse ≠ 0 → NaN (keine Zahl)"),
                              [("", str(value))]))
        return res
    res.steps.append(Step("Vorzeichen", f"Vorzeichenbit {sign} → {'negativ' if sign else 'positiv'}",
                          [("(-1)^s", sg)]))
    if kind == "normal":
        res.steps.append(Step("Exponent", f"e = Exponentenfeld − Bias = {ef} − {bias} = {e}",
                              [("e", e)], rule="Exponent − Bias"))
        res.steps.append(Step("Mantisse", "Vorne die nicht gespeicherte 1 ergänzen: 1.xxx",
                              [("1.m", f"1.{man}")]))
    else:
        res.steps.append(Step("Subnormale Zahl / Null", f"Exponentenfeld 0: keine führende 1, "
                              f"Exponent e = 1 − {bias} = {e}.", [("0.m", f"0.{man}")]))
    res.steps.append(Step("Wert", "(−1)^s · 1.m · 2^e", [("Wert", value),
                                                          ("≈", float(value))],
                          rule="IEEE 754"))
    return res


# ==========================================================================
# float: Vergleich und Genauigkeit
# ==========================================================================

class FloatCompareResult(Result):
    """
    Auswertung einer Rechnung mit float (Python) und exakt (Brüche).

    Attribute: `left`/`right` (Python-float), `left_exact`/`right_exact` (Rational),
    `python_equal` (Ergebnis von `==` in Python), `exact_equal` (mathematisch gleich),
    `difference` (float), `compared` (False: nur ein Ausdruck ohne `==`).
    """

    title = "float-Rechnung und Vergleich"


def _fl_eval(n, conv, div):
    if n.kind == "val":
        return conv(n.val)
    if n.kind == "neg":
        return -_fl_eval(n.a, conv, div)
    x, y = _fl_eval(n.a, conv, div), _fl_eval(n.b, conv, div)
    if n.op == "+":
        return x + y
    if n.op == "-":
        return x - y
    if n.op == "*":
        return x * y
    if y == 0:
        raise ValueError("Division durch 0")
    return div(x, y)


def float_compare(expr) -> FloatCompareResult:
    """
    Rechnet einen Ausdruck mit Dezimalzahlen als Python-`float` und exakt mit Brüchen und
    vergleicht (Serie 1: `0.7 + 0.1 == 0.9 - 0.1` ist in Python `False`).

        r = float_compare("0.7 + 0.1 == 0.9 - 0.1")
        r.python_equal      # False
        r.exact_equal       # True
        r.left, r.right     # 0.7999999999999999, 0.8
    """
    text = _check_expr(expr)
    parts = text.split("==")
    if len(parts) > 2:
        raise ValueError("Nur ein »==« erlaubt")
    trees = []
    for p in parts:
        t, _, root = _build(p, {}, "float")
        trees.append(root)
    fl = [_fl_eval(t, float, lambda a, b: a / b) for t in trees]
    ex = [_fl_eval(t, Fraction, lambda a, b: Fraction(a) / Fraction(b)) for t in trees]
    res = FloatCompareResult()
    res.expr = text
    res.compared = len(trees) == 2
    res.left, res.left_exact = fl[0], _R(Fraction(ex[0]))
    if res.compared:
        res.right, res.right_exact = fl[1], _R(Fraction(ex[1]))
        res.python_equal = fl[0] == fl[1]
        res.exact_equal = ex[0] == ex[1]
        res.difference = abs(fl[0] - fl[1])
        res.value = res.python_equal
        res.summary = (f"Python: {text} ist {res.python_equal}; "
                       f"in der Mathematik ist die Aussage {'wahr' if res.exact_equal else 'falsch'}.")
        res.highlights = [("Python ==", str(res.python_equal)), ("exakt ==", str(res.exact_equal))]
    else:
        res.right = res.right_exact = None
        res.python_equal = res.exact_equal = None
        res.value = fl[0]
        res.difference = None
        res.summary = f"{text} = {fl[0]!r} (als float)"
        res.highlights = [("float", repr(fl[0])), ("exakt", res.left_exact)]
    # Dezimalzahlen im Ausdruck: gespeicherter Wert
    leaves, seen = [], set()
    for root in trees:
        for n in _nodes_of(root):
            if n.kind == "val" and n.val not in seen:
                seen.add(n.val)
                leaves.append(n.val)
    rows = []
    for v in leaves:
        fv = float(v)
        stored = Decimal(fv)
        exact_flag = Fraction(fv) == Fraction(v)
        rows.append([_ftext(Fraction(v)), "genau" if exact_flag else str(stored)[:40] + "…",
                     "ja" if exact_flag else "nein"])
    res.steps.append(Step("Aufgabe", text, [("", text)]))
    res.steps.append(Step(
        "Zahlen im float-Format",
        "Ein float hat endlich viele Binärstellen. Viele Dezimalzahlen (0.1, 0.7, …) haben in "
        "Binär eine unendliche Periode und werden gerundet gespeichert.",
        table=Table(["Zahl", "tatsächlich gespeichert", "exakt?"], rows), rule="IEEE 754 (Rundung)"))
    if res.compared:
        res.steps.append(Step(
            "Mit float rechnen",
            "Jede Rechnung rundet; die Rundungsfehler heben sich nicht auf.",
            [("links", repr(fl[0])), ("rechts", repr(fl[1])), ("links == rechts", str(res.python_equal))]))
        res.steps.append(Step(
            "Mathematisch (exakt)", "Mit Brüchen gerechnet, ohne Rundung:",
            [("links", res.left_exact), ("rechts", res.right_exact),
             ("links == rechts", str(res.exact_equal))]))
        if res.python_equal != res.exact_equal:
            res.steps.append(Step(
                "Folgerung", "Floats nie mit == vergleichen, sondern mit einer Toleranz: "
                "abs(a − b) < ε (z. B. math.isclose).",
                [("|a − b|", res.difference), ("math.isclose", str(math.isclose(fl[0], fl[1])))]))
    else:
        res.steps.append(Step("Mit float", "", [("float", repr(fl[0])), ("exakt", res.left_exact)]))
    return res


# ==========================================================================
# Zweierkomplement, Wertebereich, Überlauf
# ==========================================================================

class TwosComplementResult(Result):
    """
    Zahl im Zweierkomplement mit fester Bitbreite.

    Attribute: `value` (int), `bits` (Breite), `pattern` (Bitfolge), `ones` (Einerkomplement,
    nur bei negativen Zahlen), `min`, `max` (Wertebereich).
    """

    title = "Zweierkomplement"

    def __init__(self, value, bits, pattern, ones=None):
        super().__init__()
        self.value, self.bits, self.pattern, self.ones = value, bits, pattern, ones
        self.min, self.max = -2 ** (bits - 1), 2 ** (bits - 1) - 1
        self.summary = f"{value} = ({pattern}){_sub(2)} im Zweierkomplement mit {bits} Bit"
        self.highlights = [(f"{value} ({bits} Bit)", pattern)]


class RangeResult(Result):
    """Wertebereich einer n-Bit-Zahl. Attribute: `bits`, `signed`, `min`, `max`, `count`."""

    title = "Wertebereich einer n-Bit-Zahl"

    def __init__(self, bits, signed):
        super().__init__()
        self.bits, self.signed = bits, signed
        self.count = 2 ** bits
        self.min = -2 ** (bits - 1) if signed else 0
        self.max = 2 ** (bits - 1) - 1 if signed else 2 ** bits - 1
        kind = "mit Vorzeichen (Zweierkomplement)" if signed else "ohne Vorzeichen"
        self.summary = f"{bits} Bit {kind}: {self.min} … {self.max}"
        self.highlights = [("kleinste Zahl", self.min), ("grösste Zahl", self.max),
                           ("Anzahl Zahlen", self.count)]


class OverflowResult(Result):
    """
    Rechnung mit fester Bitbreite. Attribute: `a`, `b`, `op`, `bits`, `signed`, `exact`
    (mathematisches Ergebnis), `wrapped` (was die n-Bit-Rechnung liefert), `overflow` (bool),
    `pattern` (Bitfolge des Ergebnisses), `min`, `max`.
    """

    title = "Überlauf"

    def __init__(self, a, op, b, bits, signed, exact, wrapped, overflow, pattern):
        super().__init__()
        self.a, self.op, self.b, self.bits, self.signed = a, op, b, bits, signed
        self.exact, self.wrapped, self.overflow, self.pattern = exact, wrapped, overflow, pattern
        self.min = -2 ** (bits - 1) if signed else 0
        self.max = 2 ** (bits - 1) - 1 if signed else 2 ** bits - 1
        self.value = wrapped
        self.summary = (f"{a} {op} {b} = {exact}, passt nicht in {bits} Bit → Überlauf, "
                        f"gespeichert wird {wrapped}" if overflow
                        else f"{a} {op} {b} = {exact} passt in {bits} Bit")
        self.highlights = [("exakt", exact), (f"{bits}-Bit-Ergebnis", wrapped),
                           ("Überlauf", "ja" if overflow else "nein")]


def _wrap_signed(v: int, bits: int, signed: bool) -> int:
    m = v % 2 ** bits
    return m - 2 ** bits if signed and m >= 2 ** (bits - 1) else m


def int_range(bits, signed: bool = True) -> RangeResult:
    """
    Wertebereich einer n-Bit-Zahl: 2^n Werte; mit Vorzeichen −2^(n−1) … 2^(n−1) − 1,
    ohne Vorzeichen 0 … 2^n − 1.

        int_range(8)            # -128 … 127
        int_range(8, False)     # 0 … 255
        int_range(32).max       # 2147483647
    """
    b = _check_bits(bits, 1)
    res = RangeResult(b, bool(signed))
    res.steps.append(Step("Anzahl Werte", f"Jedes der {b} Bits hat 2 Möglichkeiten: 2^{b} = {2 ** b} Werte.",
                          [("2^n", 2 ** b)], rule="Produktregel"))
    if signed:
        res.steps.append(Step(
            "Mit Vorzeichen (Zweierkomplement)",
            f"Das höchste Bit hat den Stellenwert −2^{b - 1}. Negative Zahlen: −2^{b - 1} … −1 "
            f"({2 ** (b - 1)} Werte), 0, positive: 1 … 2^{b - 1} − 1.",
            [("kleinste", f"−2^{b - 1} = {res.min}"), ("grösste", f"2^{b - 1} − 1 = {res.max}")],
            rule="Wertebereich"))
    else:
        res.steps.append(Step("Ohne Vorzeichen", f"Alle Bits sind Stellen: 0 … 2^{b} − 1.",
                              [("kleinste", 0), ("grösste", f"2^{b} − 1 = {res.max}")],
                              rule="Wertebereich"))
    return res


def twos_complement(n, bits: int = 8) -> TwosComplementResult:
    """
    Ganze Zahl im Zweierkomplement mit `bits` Bit (2–64): positive Zahlen wie gewohnt mit
    führenden Nullen; negative Zahlen: Betrag in Binär, alle Bits kippen, 1 addieren
    (oder 2^n + z binär schreiben).

        twos_complement(-10, 8).pattern    # '11110110'
        twos_complement(5, 8).pattern      # '00000101'
        twos_complement(-200, 8)           # ValueError: ausserhalb von -128 … 127
    """
    b = _check_bits(bits)
    n = _check_int(n, "n", 2 ** 63)
    lo, hi = -2 ** (b - 1), 2 ** (b - 1) - 1
    if not lo <= n <= hi:
        raise ValueError(f"{n} liegt ausserhalb des Wertebereichs {lo} … {hi} von {b} Bit "
                         f"(Überlauf: overflow_check)")
    pat = format(n % 2 ** b, f"0{b}b")
    if n >= 0:
        res = TwosComplementResult(n, b, pat)
        res.steps = [
            Step("Wertebereich", f"{b} Bit mit Vorzeichen: {lo} … {hi}; {n} passt.", [("", f"{lo} … {hi}")]),
            Step("Positive Zahl", f"Wie gewohnt in Binär schreiben und links mit Nullen auf {b} Bit auffüllen "
                 f"(das höchste Bit ist 0).", [("", f"({format(n, 'b')}){_sub(2)} → {pat}")],
                 rule="Binärdarstellung")]
    else:
        mag = format(-n, f"0{b}b")
        ones = "".join("1" if c == "0" else "0" for c in mag)
        res = TwosComplementResult(n, b, pat, ones)
        res.steps = [
            Step("Wertebereich", f"{b} Bit mit Vorzeichen: {lo} … {hi}; {n} passt.", [("", f"{lo} … {hi}")]),
            Step("Betrag binär", f"|{n}| = {-n} in Binär, auf {b} Bit aufgefüllt.",
                 [("", mag)], rule="Binärdarstellung"),
            Step("Alle Bits kippen", "0 ↔ 1 (Einerkomplement).", [("", ones)], rule="Einerkomplement"),
            Step("1 addieren", f"{ones} + 1", [("", pat)], rule="Zweierkomplement"),
            Step("Zweite Methode", f"2^{b} + ({n}) = {2 ** b + n} = ({pat}){_sub(2)}",
                 [("", pat)], note="führt zum selben Ergebnis")]
    weights = [str(-2 ** (b - 1))] + [str(2 ** k) for k in range(b - 2, -1, -1)]
    res.steps.append(Step(
        "Gegenprobe mit Stellenwerten",
        f"Das höchste Bit zählt −2^{b - 1} = {-2 ** (b - 1)}, alle anderen wie üblich: "
        f"{' '.join(pat)} → {n}.",
        [("", n)], table=Table(weights + ["Zahl"], [list(pat) + [str(n)]]), rule="Stellenwerte im Zweierkomplement"))
    assert _from_pat(pat) == n
    return res


def _from_pat(pat: str) -> int:
    v = int(pat, 2)
    return v - 2 ** len(pat) if pat[0] == "1" else v


def from_twos_complement(pattern) -> TwosComplementResult:
    """
    Bitfolge → Zahl im Zweierkomplement (das höchste Bit hat den Wert −2^(n−1)).

        from_twos_complement("11110110").value    # -10
    """
    s = str(pattern).replace(" ", "").replace("_", "")
    if not re.fullmatch(r"[01]+", s):
        raise ValueError("Die Bitfolge darf nur aus 0 und 1 bestehen")
    b = len(s)
    if b < 2 or b > MAX_BITS:
        raise ValueError(f"Die Bitbreite muss zwischen 2 und {MAX_BITS} liegen")
    v = _from_pat(s)
    ones = None
    res = TwosComplementResult(v, b, s, ones)
    weights = [str(-2 ** (b - 1))] + [str(2 ** k) for k in range(b - 2, -1, -1)]
    terms = " + ".join(f"({-2 ** (b - 1)})" if i == 0 else str(2 ** (b - 1 - i))
                       for i, c in enumerate(s) if c == "1") or "0"
    res.steps.append(Step("Aufgabe", f"({s}){_sub(2)} im Zweierkomplement mit {b} Bit", [("", s)]))
    res.steps.append(Step("Stellenwerte",
                          f"Das höchste Bit zählt −2^{b - 1}; die Summe aller Stellen mit Bit 1 ist die Zahl.",
                          [("Summe", f"{terms} = {v}")],
                          table=Table(weights + ["Zahl"], [list(s) + [str(v)]]),
                          rule="Stellenwerte im Zweierkomplement"))
    if s[0] == "1":
        inv = "".join("1" if c == "0" else "0" for c in s)
        mag = int(inv, 2) + 1
        res.steps.append(Step("Zweite Methode", f"Negativ (höchstes Bit 1): alle Bits kippen und 1 addieren "
                              f"ergibt den Betrag: {inv} + 1 = {mag} → −{mag}.", [("", v)],
                              rule="Betrag = Zweierkomplement"))
    return res


def overflow_check(a, op, b, bits: int = 8, signed: bool = True) -> OverflowResult:
    """
    Rechnet a op b (op: `+ - *`) mit n-Bit-Zahlen: passt das Ergebnis in den Wertebereich? Sonst
    Überlauf — gespeichert wird das Ergebnis modulo 2^n.

        overflow_check(100, "+", 100, 8)       # exakt 200, gespeichert -56 (Überlauf)
        overflow_check(255, "+", 1, 8, False)  # exakt 256, gespeichert 0
        overflow_check(50, "+", 50, 8)         # 100, kein Überlauf
    """
    bt = _check_bits(bits, 1)
    a, b = _check_int(a, "a", 2 ** 64), _check_int(b, "b", 2 ** 64)
    if op not in ("+", "-", "*"):
        raise ValueError("Erlaubt sind + - * (op)")
    lo = -2 ** (bt - 1) if signed else 0
    hi = 2 ** (bt - 1) - 1 if signed else 2 ** bt - 1
    for name, v in (("a", a), ("b", b)):
        if not lo <= v <= hi:
            raise ValueError(f"{name} = {v} liegt selbst nicht im Wertebereich {lo} … {hi} von {bt} Bit")
    exact = a + b if op == "+" else a - b if op == "-" else a * b
    wrapped = _wrap_signed(exact, bt, signed)
    over = exact != wrapped
    pat = format(exact % 2 ** bt, f"0{bt}b")
    res = OverflowResult(a, op, b, bt, bool(signed), exact, wrapped, over, pat)
    res.steps = [
        Step("Wertebereich", f"{bt} Bit {'mit' if signed else 'ohne'} Vorzeichen: {lo} … {hi}",
             [("", f"{lo} … {hi}")], rule="Wertebereich"),
        Step("Exakt rechnen", f"{a} {op} {b} = {exact}", [("", exact)]),
        Step("Passt es?", f"{lo} ≤ {exact} ≤ {hi}  →  {'ja' if not over else 'nein: Überlauf'}",
             [("", "ja" if not over else "nein")]),
    ]
    if over:
        res.steps.append(Step(
            "Modulo 2^n",
            f"Es bleiben nur die untersten {bt} Bit: {exact} mod 2^{bt} = {exact % 2 ** bt} = "
            f"({pat}){_sub(2)}."
            + (f" Als Zweierkomplement gelesen ist das {wrapped}." if signed else ""),
            [("gespeichert", wrapped), ("Bitfolge", pat)], rule="Überlauf"))
        res.warnings.append(f"Überlauf: {exact} ist ausserhalb von {lo} … {hi}.")
    return res


# ==========================================================================
# Mengen
# ==========================================================================

def _skey(x):
    return (0, x, "") if isinstance(x, (int, float)) and not isinstance(x, bool) else (1, 0, str(x))


def _stext(s) -> str:
    """Menge als Text: {a, b, c}, die leere Menge als ∅."""
    items = sorted(s, key=_skey) if all(not isinstance(x, (set, frozenset)) for x in s) \
        else sorted(s, key=lambda t: str(sorted(map(str, t)) if isinstance(t, (set, frozenset)) else t))
    if not items:
        return "∅"
    return "{" + ", ".join(_stext(x) if isinstance(x, (set, frozenset)) else str(x)
                           for x in items) + "}"


def _as_set(x, name: str = "Menge") -> frozenset:
    if isinstance(x, (set, frozenset)):
        s = frozenset(x)
    elif isinstance(x, (list, tuple, range)):
        s = frozenset(x)
    elif isinstance(x, str):
        t = x.strip()
        if t.startswith("{") and t.endswith("}"):
            t = t[1:-1]
        t = t.strip()
        if "," in t:
            items = [i.strip() for i in t.split(",")]
        elif " " in t:
            items = t.split()
        else:
            items = list(t)               # ponytail: »abc« = {a, b, c}, einzelne Zeichen
        s = frozenset(int(i) if re.fullmatch(r"-?\d+", i) else i for i in items if i != "")
    else:
        raise ValueError(f"{name}: erwartet eine Menge, Liste oder einen Text wie '{{a, b, c}}', nicht {x!r}")
    if len(s) > MAX_SET:
        raise ValueError(f"{name} hat zu viele Elemente (höchstens {MAX_SET})")
    return s


_SETOPS = {
    "∩": "∩", "&": "∩", "cap": "∩", "durchschnitt": "∩", "intersection": "∩",
    "∪": "∪", "|": "∪", "cup": "∪", "vereinigung": "∪", "union": "∪",
    "\\": "\\", "-": "\\", "minus": "\\", "differenz": "\\", "difference": "\\",
    "c": "c", "komplement": "c", "complement": "c", "¯": "c",
}


class SetOpResult(Result):
    """
    Mengenoperation. Attribute: `result` (frozenset), `op` ('∩' '∪' '\\' 'c'), `A`, `B`,
    `universe` (nur beim Komplement); `rows` (Element → Zugehörigkeit).
    """

    title = "Mengenoperation"

    def __init__(self, A, op, B, universe, result):
        super().__init__()
        self.A, self.op, self.B, self.universe, self.result = A, op, B, universe, result
        self.value = result
        if op == "c":
            self.summary = f"Komplement von {_stext(A)} in {_stext(universe)} = {_stext(result)}"
        else:
            self.summary = f"{_stext(A)} {op} {_stext(B)} = {_stext(result)}"
        self.highlights = [("Ergebnis", _stext(result))]


def set_op(A, op, B=None, universe=None) -> SetOpResult:
    """
    Durchschnitt `∩` (cap), Vereinigung `∪` (cup), Differenz `\\` (minus) und Komplement
    `c` (mit Grundmenge `universe`) von Mengen — mit Tabelle »Element ∈ A?, ∈ B?«.

        set_op("{a,b,c}", "∩", "{b,c,d}")                 # {b, c}
        set_op("{a,b,c}", "∪", "{b,c,d}")                 # {a, b, c, d}
        set_op("{a,b,c}", "\\\\", "{b,c,d}")                 # {a}
        set_op("{a,b}", "komplement", universe="{a,b,c,d,e}")   # {c, d, e}
    """
    key = str(op).strip().lower() if str(op).strip() not in ("\\",) else "\\"
    sym = _SETOPS.get(key)
    if sym is None:
        raise ValueError(f"Unbekannte Mengenoperation {op!r}: ∩, ∪, \\ (Differenz) oder Komplement")
    A = _as_set(A, "A")
    if sym == "c":
        if universe is None:
            raise ValueError("Das Komplement braucht eine Grundmenge: universe=…")
        U = _as_set(universe, "Grundmenge")
        if not A <= U:
            raise ValueError(f"A ist keine Teilmenge der Grundmenge (zu viel: {_stext(A - U)})")
        result = U - A
        res = SetOpResult(A, "c", None, U, result)
        rows = [[str(x), "ja" if x in A else "nein", "nein" if x in A else "ja"]
                for x in sorted(U, key=_skey)]
        res.steps = [
            Step("Aufgabe", "Komplement: alles aus der Grundmenge, was nicht in A ist",
                 [("A", _stext(A)), ("Grundmenge", _stext(U))]),
            Step("Element für Element", "Jedes Element der Grundmenge prüfen.",
                 table=Table(["x", "x ∈ A", "x ∈ Komplement"], rows,
                             marks={i: "hi" for i, r in enumerate(rows) if r[2] == "ja"}),
                 rule="Komplement"),
            Step("Ergebnis", "", [("Komplement", _stext(result))])]
        return res
    if B is None:
        raise ValueError(f"Die Operation {sym} braucht zwei Mengen")
    B = _as_set(B, "B")
    result = {"∩": A & B, "∪": A | B, "\\": A - B}[sym]
    result = frozenset(result)
    res = SetOpResult(A, sym, B, None, result)
    names = {"∩": "Durchschnitt: Elemente, die in A und in B sind",
             "∪": "Vereinigung: alles zusammen, ohne Duplikate",
             "\\": "Differenz: Elemente aus A, die nicht in B sind"}
    cond = {"∩": lambda a, b: a and b, "∪": lambda a, b: a or b, "\\": lambda a, b: a and not b}[sym]
    elems = sorted(A | B, key=_skey)
    rows = [[str(x), "ja" if x in A else "nein", "ja" if x in B else "nein",
             "ja" if cond(x in A, x in B) else "nein"] for x in elems]
    res.steps = [
        Step("Aufgabe", f"A {sym} B", [("A", _stext(A)), ("B", _stext(B))]),
        Step("Definition", names[sym], rule={"∩": "Durchschnitt", "∪": "Vereinigung",
                                             "\\": "Differenz"}[sym]),
        Step("Element für Element", "Jedes Element aus A oder B prüfen.",
             table=Table(["x", "x ∈ A", "x ∈ B", f"x ∈ A {sym} B"], rows,
                         marks={i: "hi" for i, r in enumerate(rows) if r[3] == "ja"})),
        Step("Ergebnis", "", [(f"A {sym} B", _stext(result))]),
    ]
    if sym == "∩" and not result:
        res.steps[-1].note = "Kein gemeinsames Element: der Durchschnitt ist die leere Menge ∅."
    if sym == "\\":
        res.steps[-1].note = "Es zählt nur, was von links kommt: B \\ A wäre eine andere Menge."
    return res


class SetRelationResult(Result):
    """Relation (Antwort wahr/falsch). Attribute: `value` (bool), `rel`, `left`, `right`, `witness`."""

    title = "Relation"

    def __init__(self, left, rel, right, value, witness=None):
        super().__init__()
        self.left, self.rel, self.right, self.value, self.witness = left, rel, right, value, witness
        lt = _stext(left) if isinstance(left, (set, frozenset)) else str(left)
        self.summary = f"{lt} {rel} {_stext(right)} ist {'wahr' if value else 'falsch'}"
        self.highlights = [("Ergebnis", "True" if value else "False")]


def set_relation(left, rel, right) -> SetRelationResult:
    """
    Relationen auf Mengen: `=` (Gleichheit), `⊂`/`⊆` (Teilmenge: ist alles aus links auch
    rechts?), `⊊` (echte Teilmenge), `∈` (Element) und `∉`. Das Ergebnis ist ein Wahrheitswert.

        set_relation("{a,b}", "⊂", "{a,b,c}")    # True
        set_relation("b", "∈", "{a,b,c}")        # True  (Objekt ∈ Menge)
        set_relation("{a,b}", "=", "{b,a,a}")    # True  (Reihenfolge und Wiederholung egal)
    """
    r = {"subset": "⊆", "⊂": "⊆", "⊆": "⊆", "<=": "⊆", "proper": "⊊", "⊊": "⊊", "<": "⊊",
         "=": "=", "==": "=", "in": "∈", "∈": "∈", "notin": "∉", "∉": "∉"}.get(str(rel).strip())
    if r is None:
        raise ValueError(f"Unbekannte Relation {rel!r}: =, ⊂ (⊆), ⊊, ∈, ∉")
    if r in ("∈", "∉"):
        M = _as_set(right, "Menge")
        x = left
        if isinstance(x, str):
            x = int(x) if re.fullmatch(r"-?\d+", x.strip()) else x.strip()
        val = (x in M) if r == "∈" else (x not in M)
        res = SetRelationResult(x, r, M, val)
        res.steps = [
            Step("Aufgabe", f"{x} {r} {_stext(M)}?", [("Objekt", x), ("Menge", _stext(M))]),
            Step("Element sein", "∈ vergleicht ein Objekt mit einer Menge: kommt es als Element vor?",
                 [("", "True" if val else "False")], rule="Element"),
        ]
        return res
    A, B = _as_set(left, "A"), _as_set(right, "B")
    if r == "=":
        val = A == B
        res = SetRelationResult(A, "=", B, val)
        res.steps = [Step("Aufgabe", "Gleichheit: dieselben Elemente (Reihenfolge und Wiederholung zählen nicht)",
                          [("A", _stext(A)), ("B", _stext(B))]),
                     Step("Prüfen", "A = B genau dann, wenn A ⊂ B und B ⊂ A.",
                          [("A ⊂ B", str(A <= B)), ("B ⊂ A", str(B <= A))], rule="Gleichheit"),
                     Step("Ergebnis", "", [("", "True" if val else "False")])]
        if not val:
            res.witness = sorted((A ^ B), key=_skey)[0]
        return res
    val = (A <= B) if r == "⊆" else (A < B)
    miss = sorted(A - B, key=_skey)
    res = SetRelationResult(A, r, B, val, miss[0] if miss else None)
    rows = [[str(x), "ja" if x in B else "nein"] for x in sorted(A, key=_skey)]
    res.steps = [
        Step("Aufgabe", f"Ist A {r} B?", [("A", _stext(A)), ("B", _stext(B))]),
        Step("Teilmenge prüfen", "Von links nach rechts: ist jedes Element aus A auch in B?",
             table=Table(["x ∈ A", "x ∈ B"], rows, marks={i: ("ok" if r_[1] == "ja" else "bad")
                                                          for i, r_ in enumerate(rows)}),
             rule="Teilmenge"),
    ]
    if r == "⊊" and A <= B:
        res.steps.append(Step("Echt?", "Echte Teilmenge: zusätzlich A ≠ B.", [("A ≠ B", str(A != B))]))
    res.steps.append(Step(
        "Ergebnis", "" if val else f"Gegenbeispiel: {miss[0]} ∈ A, aber {miss[0]} ∉ B." if miss
        else "A = B, also keine echte Teilmenge.",
        [("", "True" if val else "False")],
        note="Teilmenge vergleicht zwei Mengen, Element sein ein Objekt mit einer Menge."))
    return res


class PowerSetResult(Result):
    """
    Alle Teilmengen. Attribute: `subsets` (Liste von frozensets, nach Grösse sortiert),
    `bitstrings` (je Teilmenge, ein Bit pro Element in der Reihenfolge `elements`), `elements`,
    `count` (= 2^|S|).
    """

    title = "Potenzmenge (alle Teilmengen)"

    def __init__(self, elements, subsets, bitstrings):
        super().__init__()
        self.elements, self.subsets, self.bitstrings = elements, subsets, bitstrings
        self.count = len(subsets)
        self.value = subsets
        self.summary = f"Die Menge {_stext(elements)} hat 2^{len(elements)} = {self.count} Teilmengen."
        self.highlights = [("|S|", len(elements)), ("Anzahl Teilmengen", self.count)]


def power_set(S) -> PowerSetResult:
    """
    Alle Teilmengen von S (Serie 1, Aufgabe 6: {a, b, c, d} hat 2^4 = 16). Mit Bitstring je
    Teilmenge: eine 1 heisst, das Element ist dabei.

        power_set("{a,b,c,d}").count     # 16
    """
    S = _as_set(S, "S")
    # ponytail: harte Grenze 10 Elemente statt Stückweise-Ausgabe
    if len(S) > MAX_POWERSET:
        raise ValueError(f"Die Potenzmenge von {len(S)} Elementen hat {2 ** len(S)} Teilmengen; "
                         f"erlaubt sind höchstens {MAX_POWERSET} Elemente")
    el = sorted(S, key=_skey)
    subs, bits = [], []
    for k in range(len(el) + 1):
        for c in itertools.combinations(range(len(el)), k):
            subs.append(frozenset(el[i] for i in c))
            bits.append("".join("1" if i in c else "0" for i in range(len(el))))
    res = PowerSetResult(el, subs, bits)
    rows = [[b, _stext(s), str(len(s))] for b, s in zip(bits, subs)]
    res.steps = [
        Step("Aufgabe", "Alle Teilmengen aufschreiben (inklusive ∅ und der Menge selbst).",
             [("S", _stext(S)), ("|S|", len(S))]),
        Step("Anzahl", f"Jedes der {len(S)} Elemente ist dabei oder nicht: 2 Möglichkeiten je Element "
             f"→ 2^{len(S)} = {2 ** len(S)} Teilmengen.", [("|2^S|", 2 ** len(S))],
             rule="Produktregel / Potenzmenge"),
        Step("Alle Teilmengen", "Der Bitstring hat ein Bit je Element "
             f"(Reihenfolge {', '.join(map(str, el))}): 1 = Element ist dabei.",
             table=Table(["Bitstring", "Teilmenge", "Anzahl"], rows), rule="Bitstring-Darstellung"),
    ]
    return res


class ProductResult(Result):
    """Kartesisches Produkt A × B: `pairs` (Liste von Tupeln), `size` (= |A|·|B|)."""

    title = "Kartesisches Produkt"

    def __init__(self, A, B, pairs):
        super().__init__()
        self.A, self.B, self.pairs, self.size = A, B, pairs, len(pairs)
        self.value = pairs
        self.summary = f"|A × B| = {len(A)} · {len(B)} = {len(pairs)}"
        self.highlights = [("|A × B|", len(pairs))]


def cartesian_product(A, B) -> ProductResult:
    """
    A × B = {(a, b) : a ∈ A, b ∈ B}; die Reihenfolge im Paar zählt (A × B ≠ B × A).

        cartesian_product("{1,2,3}", "{a,b}").size    # 6
    """
    A, B = _as_set(A, "A"), _as_set(B, "B")
    if len(A) * len(B) > 2000:
        raise ValueError("Das Produkt hat zu viele Paare (höchstens 2000)")
    a, b = sorted(A, key=_skey), sorted(B, key=_skey)
    pairs = [(x, y) for y in b for x in a]
    res = ProductResult(A, B, pairs)
    res.steps = [
        Step("Aufgabe", "Alle Paare (a, b) mit a ∈ A und b ∈ B", [("A", _stext(A)), ("B", _stext(B))]),
        Step("Paare", "Reihenfolge im Paar zählt: (1, a) ≠ (a, 1).",
             [("A × B", "{" + ", ".join(f"({x}, {y})" for x, y in pairs) + "}")], rule="Kartesisches Produkt"),
        Step("Anzahl", f"|A × B| = |A| · |B| = {len(A)} · {len(B)} = {len(pairs)}", [("", len(pairs))],
             rule="Produktregel"),
    ]
    return res


class PartitionResult(Result):
    """Prüfung einer Zerlegung. Attribute: `value` (bool), `reason` (Text), `parts`, `universe`."""

    title = "Partition"

    def __init__(self, parts, universe, value, reason):
        super().__init__()
        self.parts, self.universe, self.value, self.reason = parts, universe, value, reason
        self.summary = f"{'Ist' if value else 'Ist keine'} Partition von {_stext(universe)}: {reason}"
        self.highlights = [("Partition", "True" if value else "False")]


def is_partition(parts, universe) -> PartitionResult:
    """
    Zerlegen die Teile die Grundmenge lückenlos und ohne Überlappung (z. B. A und das Komplement)?

        is_partition(["{a,b}", "{c,d,e}"], "{a,b,c,d,e}")   # True
    """
    U = _as_set(universe, "Grundmenge")
    P = [_as_set(p, f"Teil {i + 1}") for i, p in enumerate(parts)]
    if not P:
        raise ValueError("Keine Teile angegeben")
    union = frozenset().union(*P)
    overlap = sorted({x for i, a in enumerate(P) for b in P[i + 1:] for x in a & b}, key=_skey)
    empty = any(not p for p in P)
    ok = union == U and not overlap and not empty
    if union != U:
        reason = f"die Vereinigung {_stext(union)} ist nicht die Grundmenge"
    elif overlap:
        reason = f"die Teile überlappen sich in {_stext(set(overlap))}"
    elif empty:
        reason = "ein Teil ist leer"
    else:
        reason = "Vereinigung = Grundmenge, Teile paarweise ohne gemeinsames Element"
    res = PartitionResult(P, U, ok, reason)
    res.steps = [
        Step("Aufgabe", "Partition: Teile decken die Grundmenge ab und überlappen nicht.",
             [("Teile", ", ".join(_stext(p) for p in P)), ("Grundmenge", _stext(U))]),
        Step("Vereinigung", "", [("∪ Teile", _stext(union)), ("= Grundmenge", str(union == U))]),
        Step("Überlappung", "", [("Durchschnitte", _stext(set(overlap)) if overlap else "alle ∅")]),
        Step("Ergebnis", reason, [("", "True" if ok else "False")]),
    ]
    return res


class BitstringResult(Result):
    """Teilmenge als Bitstring. Attribute: `bits`, `elements` (Reihenfolge der Bits), `subset`."""

    title = "Menge als Bitstring"

    def __init__(self, elements, subset, bits):
        super().__init__()
        self.elements, self.subset, self.bits = elements, subset, bits
        self.value = bits
        self.summary = f"{_stext(subset)} = {bits}"
        self.highlights = [("Bitstring", bits)]


def set_bitstring(subset, universe) -> BitstringResult:
    """
    Teilmenge als Bitstring über der Grundmenge: ein Bit je Element (1 = dabei).

        set_bitstring("{2,*}", "{1,2,a,*}")    # '0101'   (Reihenfolge der Elemente: sortiert)
    """
    U = _as_set(universe, "Grundmenge")
    A = _as_set(subset, "Teilmenge")
    if not A <= U:
        raise ValueError(f"{_stext(A - U)} gehört nicht zur Grundmenge")
    el = sorted(U, key=_skey)
    bits = "".join("1" if x in A else "0" for x in el)
    res = BitstringResult(el, A, bits)
    res.steps = [
        Step("Reihenfolge", "Jedem Element der Grundmenge ein Bit zuordnen.",
             table=Table([str(x) for x in el], [list(bits)])),
        Step("Ergebnis", "1 = Element ist in der Teilmenge, 0 = fehlt.", [(_stext(A), bits)]),
    ]
    return res


# ==========================================================================
# Listen und Matrizen: Index, Dimension
# ==========================================================================

class IndexResult(Result):
    """Zugriff auf eine Komponente. Attribute: `value`, `index` (so wie angegeben), `python_index`."""

    title = "Index"

    def __init__(self, value, index, python_index):
        super().__init__()
        self.value, self.index, self.python_index = value, index, python_index
        self.summary = f"Komponente {index} = {_vtext(value, False)}"
        self.highlights = [("Wert", _vtext(value, False))]


def list_index(seq, i: int, start: int = 1) -> IndexResult:
    """
    Komponente a_i einer Liste. Die Mathematik nummeriert ab 1, Python ab 0:
    a_2 = a[2] (Index ab 1) = a[1] in Python.

        list_index([5, 3, 8, 8, 1], 2)            # 3
        list_index([5, 3, 8, 8, 1], 2, start=0)   # 8
    """
    if not isinstance(seq, (list, tuple)) or not seq or isinstance(seq, str):
        raise ValueError("Erwartet wird eine nichtleere Liste")
    if start not in (0, 1):
        raise ValueError("Der Index beginnt bei 0 oder 1")
    i = _check_int(i, "Index")
    py = i - start
    if not 0 <= py < len(seq):
        raise ValueError(f"Index {i} ausserhalb der Liste: erlaubt sind {start} … {len(seq) - 1 + start}")
    res = IndexResult(seq[py], i, py)
    res.steps = [
        Step("Aufgabe", f"a = ({', '.join(map(str, seq))}), gesucht a_{i} bei Index ab {start}",
             [("Länge", len(seq))]),
        Step("Index umrechnen", f"Python zählt ab 0: Python-Index = {i} − {start} = {py}.",
             [("a[%d]" % py, seq[py])], rule="Mathematiker ab 1, Informatiker ab 0"),
    ]
    return res


class DimensionResult(Result):
    """Dimension m × n. Attribute: `rows`, `cols`, `count`, `value` = (m, n)."""

    title = "Dimension einer Matrix"

    def __init__(self, rows, cols):
        super().__init__()
        self.rows, self.cols, self.count, self.value = rows, cols, rows * cols, (rows, cols)
        self.summary = f"{rows} × {cols} Matrix mit {rows * cols} Einträgen"
        self.highlights = [("Dimension", f"{rows} × {cols}"), ("Einträge", rows * cols)]


def _as_matrix(M):
    if not isinstance(M, (list, tuple)) or not M or not all(isinstance(r, (list, tuple)) and r for r in M):
        raise ValueError("Eine Matrix ist eine nichtleere Liste von Zeilen, z. B. [[5, 3], [1, 0]]")
    n = len(M[0])
    if any(len(r) != n for r in M):
        raise ValueError("Eine Matrix muss rechteckig sein (alle Zeilen gleich lang)")
    if len(M) * n > MAX_ARRAY:
        raise ValueError(f"Matrix zu gross (höchstens {MAX_ARRAY} Einträge)")
    return [list(r) for r in M]


def matrix_dim(M) -> DimensionResult:
    """
    Dimension m × n einer Matrix: m Zeilen (Höhe), n Spalten (Breite).

        matrix_dim([[5, 3, 8, 8, 1], [1, 0, 0, 9, 7]])    # 2 × 5, 10 Einträge
    """
    M = _as_matrix(M)
    res = DimensionResult(len(M), len(M[0]))
    res.steps = [Step("Zählen", "Links steht die Zahl der Zeilen, rechts die der Spalten.",
                      [("m (Zeilen)", len(M)), ("n (Spalten)", len(M[0])), ("m · n", len(M) * len(M[0]))],
                      rule="m × n")]
    return res


def matrix_index(M, i: int, j: int, start: int = 1) -> IndexResult:
    """
    Komponente a_ij: Zeile i, Spalte j (Index ab 1 wie in der Mathematik, `start=0` für Python).

        matrix_index([[5, 3, 8, 8, 1], [1, 0, 0, 9, 7]], 2, 4)    # 9  (= A[2][4], Python: A[1][3])
    """
    M = _as_matrix(M)
    if start not in (0, 1):
        raise ValueError("Der Index beginnt bei 0 oder 1")
    i, j = _check_int(i, "i"), _check_int(j, "j")
    pi, pj = i - start, j - start
    if not (0 <= pi < len(M) and 0 <= pj < len(M[0])):
        raise ValueError(f"Index ({i}, {j}) ausserhalb der {len(M)}×{len(M[0])}-Matrix: Zeile "
                         f"{start}…{len(M) - 1 + start}, Spalte {start}…{len(M[0]) - 1 + start}")
    res = IndexResult(M[pi][pj], (i, j), (pi, pj))
    res.steps = [
        Step("Dimension", f"{len(M)} × {len(M[0])}: erste Zahl Zeilen, zweite Spalten.",
             [("", f"{len(M)} × {len(M[0])}")]),
        Step("Doppelindex", f"a_{i}{j} = Zeile {i}, Spalte {j}. Python zählt ab 0: A[{pi}][{pj}].",
             [(f"a_{i}{j}", M[pi][pj])], rule="Mathematiker ab 1, Informatiker ab 0"),
    ]
    return res


# ==========================================================================
# Funktionen
# ==========================================================================

class FunctionCheckResult(Result):
    """
    Prüfung, ob eine Zuordnung D → W eine Funktion ist.

    Attribute: `is_function`, `problems` (Liste von Texten), `injective`, `surjective`,
    `bijective` (None, wenn keine Funktion), `pairs` (Liste der Paare).
    """

    title = "Funktion f : D → W"

    def __init__(self, D, W, pairs, is_function, problems, inj, surj):
        super().__init__()
        self.D, self.W, self.pairs, self.is_function, self.problems = D, W, pairs, is_function, problems
        self.injective, self.surjective = inj, surj
        self.bijective = (inj and surj) if is_function else None
        self.value = is_function
        self.summary = ("Ist eine Funktion" if is_function else "Ist keine Funktion: " + "; ".join(problems))
        self.highlights = [("Funktion", "True" if is_function else "False")]
        if is_function:
            self.highlights += [("injektiv", "True" if inj else "False"),
                                ("surjektiv", "True" if surj else "False")]


def check_function(D, W, mapping) -> FunctionCheckResult:
    """
    Ist die Zuordnung eine Funktion f : D → W? Jedes Element von D muss genau ein Bild in W
    haben. `mapping`: dict {x: y}, Liste von Paaren [(x, y), …] oder ein Python-Aufruf f(x).
    Bei einer Funktion zusätzlich injektiv / surjektiv / bijektiv.

        check_function("{1,2,3}", "{a,b}", {1: "a", 2: "b", 3: "a"})    # Funktion, surjektiv, nicht injektiv
        check_function("{1,2}", "{a,b}", [(1, "a"), (1, "b"), (2, "a")])  # keine Funktion
        check_function(range(5), range(10), lambda x: 2 * x)             # injektiv, nicht surjektiv
    """
    Ds, Ws = _as_set(D, "D"), _as_set(W, "W")
    if callable(mapping):
        pairs = [(x, mapping(x)) for x in sorted(Ds, key=_skey)]
    elif isinstance(mapping, dict):
        pairs = list(mapping.items())
    else:
        pairs = [tuple(p) for p in mapping]
        if any(len(p) != 2 for p in pairs):
            raise ValueError("Paare müssen die Form (x, y) haben")
    if len(pairs) > 2 * MAX_SET:
        raise ValueError("Zu viele Zuordnungen")
    images: dict = {}
    for x, y in pairs:
        images.setdefault(x, []).append(y)
    problems = []
    for x in sorted(Ds, key=_skey):
        ys = images.get(x, [])
        if not ys:
            problems.append(f"{x} hat kein Bild")
        elif len(set(ys)) > 1:
            problems.append(f"{x} hat mehrere Bilder ({', '.join(map(str, ys))})")
    for x in images:
        if x not in Ds:
            problems.append(f"{x} gehört nicht zu D")
    for x, ys in images.items():
        for y in ys:
            if y not in Ws:
                problems.append(f"Das Bild {y} von {x} liegt nicht in W")
    ok = not problems
    imgs = {x: ys[0] for x, ys in images.items()}
    inj = surj = None
    if ok:
        vals = [imgs[x] for x in Ds]
        inj = len(set(vals)) == len(vals)
        surj = set(vals) == set(Ws)
    res = FunctionCheckResult(Ds, Ws, pairs, ok, problems, inj, surj)
    rows = []
    for x in sorted(Ds, key=_skey):
        ys = images.get(x, [])
        rows.append([str(x), ", ".join(map(str, ys)) or "—",
                     "ja" if len(set(ys)) == 1 and ys[0] in Ws else "nein"])
    res.steps = [
        Step("Aufgabe", "Eine Funktion ordnet jedem Element von D genau ein Element von W zu.",
             [("D", _stext(Ds)), ("W", _stext(Ws))], rule="Definition 7"),
        Step("Jedes x ∈ D prüfen", "Genau ein Bild, und das Bild liegt in W?",
             table=Table(["x", "f(x)", "genau ein Bild in W"], rows,
                         marks={i: ("ok" if r[2] == "ja" else "bad") for i, r in enumerate(rows)})),
    ]
    if ok:
        res.steps.append(Step(
            "Eigenschaften",
            "injektiv: verschiedene x haben verschiedene Bilder; surjektiv: jedes Element von W "
            "wird getroffen; bijektiv: beides.",
            [("injektiv", str(inj)), ("surjektiv", str(surj)), ("bijektiv", str(inj and surj))]))
    else:
        res.steps.append(Step("Ergebnis", "; ".join(problems), [("Funktion", "False")]))
    return res
