#!/usr/bin/env python3
"""
Prädikatenlogik (Semesterwoche 3, Serie 3).

Ein **Prädikat** ist eine Aussage mit Lücken (Definition 1): erst wenn alle Lücken durch
Werte aus dem Universum ersetzt sind, entsteht eine Aussage mit Wahrheitswert.
**Quantoren** machen aus einem Prädikat direkt eine Aussage (Definition 2): der Allquantor
∀ (Generalisierung) ist wahr, wenn P(x) für *alle* x gilt; der Existenzquantor ∃
(Spezialisierung) ist wahr, wenn P(x) für *mindestens ein* x gilt.

Prädikate
    Predicate(name, params, func=…) / Predicate.from_expr("istgerade", "x % 2 == 0")
    p.holds(x=3) / p(3) / p.evaluate(3)   (evaluate zeigt den Rechenweg)
    p & q, p | q, ~p, p >> q, p.iff(q)    (Verknüpfung von Prädikaten)
    SUM_EQUAL_TO_FIVE, PAL, FERMAT        (Prädikate der Serie)

Sichere Ausdrücke: Prädikate aus Text werden nie mit eval/exec ausgeführt, sondern von einem
eigenen Auswerter über dem Python-Syntaxbaum mit **Positivliste** (siehe `safe_eval`).

Quantoren über endlichen Universen
    forall(universe, pred, **fixed)       ∀   (Allquantor)
    exists(universe, pred, **fixed)       ∃   (Existenzquantor)
    quantify("∀i ∃j", pred, universes={...}, **fixed)   beliebig verschachtelt
    unroll(kind, universe, pred)          Aufgabe 2: ohne Quantoren ausschreiben + Algorithmus
    python_code("∀i ∃j")                  Schleifen-Code zu einer Quantorenfolge
    naturals / integers / reals / index_set / chars   (Universen, siehe `Universe`)

Regeln
    negate_quantifier(spec, pred, …)      ¬∀ ≡ ∃¬ und ¬∃ ≡ ∀¬ mit Gegenprobe
    negate_formula("¬∃y ∈ R: ∀x ∈ R: x < y")   Negation nach innen schieben (Aufgabe 6)
    swap_order("∀x ∃y", pred, …)          Vertauschen von Quantoren (Aufgabe 4)
    everybody_loves_somebody()            Aufgabe 4 mit Beispieldaten

Aufgaben der Serie
    find_examples(FERMAT, …)              Aufgabe 1: wahre und falsche Aussagen suchen
    has_duplicate(a), all_distinct(a), occurs_multiple(a)    Aufgabe 3f (Listen)
    matrix_quantors(M)                    Aufgabe 5: acht Quantorenfolgen für Matrizen
    to_formula("Für alle x aus U gilt P(x)")    Übersetzen (Aufgabe 7), describe_formula
    parse_formula / evaluate_formula      Quantoren-Formeln als Text

Bekannte Abweichung zur Vorlage: Die Serie zählt Listenpositionen ab 1 (a[1] ist das erste
Element). Die Funktionen für Listen und Matrizen tun das auch (`Seq1`); Python selbst zählt
ab 0. Die Serie schreibt N = {0, 1, 2, …} (Zusammenfassung).

Wichtig bei unendlichen Universen (N, Z, R): Das Programm kann nur einen endlichen
Ausschnitt prüfen. Ein Zeuge (∃) oder Gegenbeispiel (∀) ist trotzdem ein echter Beweis; dass
"nichts gefunden" wurde, beweist nichts. Das Ergebnis meldet das in `certain`/`warnings`.
"""

from __future__ import annotations

import ast
import copy
import inspect
import itertools
import math
import numbers
import re
import warnings
from fractions import Fraction

from .core import bit
from .render import Result, Step, Table

__all__ = [
    # Ausdrücke
    "safe_eval", "free_names", "ExprError", "Seq1", "one_based",
    # Prädikate
    "Predicate", "StatementResult", "SUM_EQUAL_TO_FIVE", "PAL", "FERMAT",
    "relation_predicate",
    # Universen
    "Universe", "naturals", "integers", "reals", "rationals", "index_set", "chars",
    "as_universe", "named_universe",
    # Quantoren
    "QuantResult", "forall", "exists", "quantify", "allquantor", "existenzquantor",
    "unroll", "UnrollResult", "python_code",
    # Regeln
    "negate_quantifier", "NegationResult", "negate_formula", "push_negation", "FormulaNegation",
    "swap_order", "OrderResult", "everybody_loves_somebody",
    # Aufgaben
    "find_examples", "ExamplesResult", "has_duplicate", "all_distinct", "occurs_multiple",
    "ListQuantResult", "matrix_quantors", "MatrixResult",
    # Formeln als Text
    "QAtom", "QNot", "QBin", "QQuant", "parse_formula", "evaluate_formula",
    "FormulaEvaluation", "to_formula", "TranslationResult", "describe_formula", "push_negation",
    # Grenzen
    "MAX_UNIVERSE", "MAX_COMBINATIONS", "MAX_EXPR_LEN",
]

# ==========================================================================
# Grenzen
# ==========================================================================

MAX_EXPR_LEN = 500          # Zeichen pro Ausdruck
MAX_UNIVERSE = 10_000       # Elemente pro Universum
MAX_COMBINATIONS = 50_000   # Auswertungen bei verschachtelten Quantoren (Produkt der Universen)
MAX_NODES = 20_000          # Knoten pro Auswertung eines Ausdrucks (Schutz vor Endlosrechnung)
MAX_RANGE = 100_000         # grösstes range(...) im Ausdruck
MAX_SEQ = 100_000           # grösste Liste/Text, die ein Ausdruck erzeugen darf
MAX_INT_BITS = 4096         # grösste ganze Zahl im Ausdruck (Bits)
MAX_POW_EXP = 1000          # grösster Exponent bei a**n
MAX_PRIME = 10 ** 12        # grösste Zahl für is_prime
MAX_TABLE_ROWS = 80         # Zeilen der Tabelle im Rechenweg


class ExprError(ValueError):
    """Ein Ausdruck ist nicht erlaubt, nicht lesbar oder nicht auswertbar."""


# ==========================================================================
# Hilfen: Text
# ==========================================================================

_STR_RE = re.compile(r"""('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")""")


def _map_outside_strings(text: str, fn) -> str:
    """Wendet fn auf alle Teile ausserhalb von Zeichenketten-Literalen an."""
    parts = _STR_RE.split(text)
    return "".join(fn(p) if i % 2 == 0 else p for i, p in enumerate(parts))


def _pretty(text: str) -> str:
    """Python-Ausdruck in Schreibweise der Vorlesung: != → ≠, == → =, and → ∧ …"""
    def fix(s: str) -> str:
        s = s.replace("**", "^").replace("!=", "≠").replace("==", "=")
        s = s.replace("<=", "≤").replace(">=", "≥")
        s = re.sub(r"\bnot\s+in\b", "∉", s)
        s = re.sub(r"\bin\b", "∈", s)
        s = re.sub(r"\band\b", "∧", s)
        s = re.sub(r"\bor\b", "∨", s)
        s = re.sub(r"\bnot\b\s*", "¬", s)
        return s
    return _map_outside_strings(text, fix)


def _vs(v) -> str:
    """Wert als kurzer Text (Zahlen, Texte in Anführungszeichen, Listen gekürzt)."""
    if isinstance(v, bool):
        return "wahr" if v else "falsch"
    if isinstance(v, Seq1):
        return _vs(list(v))
    if hasattr(v, "tolist") and not isinstance(v, (str, bytes)):
        try:
            return _vs(v.tolist())
        except Exception:                                       # pragma: no cover
            return str(v)
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, (list, tuple)):
        items = [_vs(x) for x in list(v)[:12]]
        more = ", …" if len(v) > 12 else ""
        inner = ", ".join(items) + more
        return f"({inner})" if isinstance(v, tuple) else f"[{inner}]"
    if isinstance(v, (set, frozenset)):
        return "{" + ", ".join(sorted(_vs(x) for x in v)) + "}"
    if isinstance(v, range):
        return f"range({v.start}, {v.stop})" if v.step == 1 else repr(v)
    if isinstance(v, Fraction):
        return str(v)
    return str(v)


def _bind_text(binding: dict) -> str:
    return ", ".join(f"{k} = {_vs(v)}" for k, v in binding.items())


def _norm_number(r):
    """Brüche mit Nenner 1 werden zu int; zu grosse ganze Zahlen werden abgelehnt."""
    if isinstance(r, Fraction) and r.denominator == 1:
        r = int(r.numerator)
    if isinstance(r, int) and not isinstance(r, bool) and r.bit_length() > MAX_INT_BITS:
        raise ExprError(f"Die Zahl wird zu gross (mehr als {MAX_INT_BITS} Bits).")
    return r


# ==========================================================================
# Listen mit Index ab 1
# ==========================================================================

class Seq1:
    """
    Liste mit Positionen ab 1 — so zählt die Serie 3 (a[1] ist das erste Element).

        a = Seq1([3, 2, 1, 1, 4, 4, 2, 3, 1]);  a[1] -> 3;  a[9] -> 1;  len(a) -> 9
    """

    __slots__ = ("_items",)

    def __init__(self, items):
        self._items = list(items)

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, i):
        if isinstance(i, slice):
            raise ExprError("Teilstücke (Slicing) gibt es bei Listen mit Index ab 1 nicht; "
                            "benutze einzelne Positionen a[i].")
        if isinstance(i, bool) or not isinstance(i, numbers.Integral):
            raise ExprError(f"Position muss eine ganze Zahl sein, nicht {_vs(i)}.")
        if i == 0:
            raise ExprError("Position 0 gibt es nicht: Listen zählen hier ab 1 (a[1] ist das erste Element).")
        if not 1 <= i <= len(self._items):
            raise ExprError(f"Position {i} liegt ausserhalb der Liste (erlaubt: 1 bis {len(self._items)}).")
        return self._items[int(i) - 1]

    def __eq__(self, other):
        if isinstance(other, Seq1):
            return self._items == other._items
        return NotImplemented

    def __hash__(self):
        return hash(tuple(map(repr, self._items)))

    def tolist(self):
        return [x.tolist() if isinstance(x, Seq1) else x for x in self._items]

    def __repr__(self):
        return f"Seq1({self.tolist()!r})"


def one_based(obj):
    """Liste (auch verschachtelt, auch numpy-Feld) → `Seq1`: Positionen zählen ab 1."""
    if isinstance(obj, Seq1):
        return obj
    if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes)):
        obj = obj.tolist()
    if isinstance(obj, (list, tuple)):
        return Seq1(one_based(x) if isinstance(x, (list, tuple)) or hasattr(x, "tolist") else x
                    for x in obj)
    return obj


# ==========================================================================
# Sicherer Ausdrucks-Auswerter (Positivliste, kein eval/exec)
# ==========================================================================

def _is_int(x) -> bool:
    return isinstance(x, numbers.Integral) and not isinstance(x, bool)


def _is_num(x) -> bool:
    return isinstance(x, numbers.Real)


def _is_ndarray(x) -> bool:
    return type(x).__name__ == "ndarray"


def _is_seq(x) -> bool:
    return isinstance(x, (list, tuple, str, range, Seq1, dict, set, frozenset)) or _is_ndarray(x)


def _need_seq(x, fname):
    if not _is_seq(x):
        raise ExprError(f"{fname}() erwartet eine Liste, einen Text oder eine Menge, nicht {_vs(x)}.")
    return x


def _need_ints(fname, *xs):
    for x in xs:
        if not _is_int(x):
            raise ExprError(f"{fname}() erwartet ganze Zahlen, nicht {_vs(x)}.")


def _fn_len(x):
    return len(_need_seq(x, "len"))


def _fn_abs(x):
    if not _is_num(x):
        raise ExprError(f"abs() erwartet eine Zahl, nicht {_vs(x)}.")
    return abs(x)


def _fn_min(*a):
    vals = list(_need_seq(a[0], "min")) if len(a) == 1 else list(a)
    if not vals:
        raise ExprError("min() von nichts: die Menge ist leer.")
    return min(vals)


def _fn_max(*a):
    vals = list(_need_seq(a[0], "max")) if len(a) == 1 else list(a)
    if not vals:
        raise ExprError("max() von nichts: die Menge ist leer.")
    return max(vals)


def _fn_sum(xs, start=0):
    vals = list(_need_seq(xs, "sum"))
    if not all(_is_num(v) for v in vals) or not _is_num(start):
        raise ExprError("sum() erwartet eine Liste von Zahlen.")
    return _norm_number(sum(vals, start))


def _fn_range(*a):
    _need_ints("range", *a)
    if not 1 <= len(a) <= 3:
        raise ExprError("range() braucht 1 bis 3 Zahlen.")
    if len(a) == 3 and a[2] == 0:
        raise ExprError("range(): die Schrittweite darf nicht 0 sein.")
    r = range(*[int(v) for v in a])
    if len(r) > MAX_RANGE:
        raise ExprError(f"range() ist zu gross (höchstens {MAX_RANGE} Elemente).")
    return r


def _fn_sorted(xs):
    return sorted(_need_seq(xs, "sorted"))


def _fn_set(xs):
    try:
        return set(_need_seq(xs, "set"))
    except TypeError:
        raise ExprError("set(): die Elemente müssen vergleichbar/hashbar sein.") from None


def _fn_list(xs):
    return list(_need_seq(xs, "list"))


def _fn_tuple(xs):
    return tuple(_need_seq(xs, "tuple"))


def _fn_int(x):
    if isinstance(x, str):
        if not re.fullmatch(r"\s*[+-]?\d{1,1000}\s*", x):
            raise ExprError(f"int(): '{x}' ist keine ganze Zahl.")
        return int(x)
    if _is_num(x):
        if isinstance(x, float) and not math.isfinite(x):
            raise ExprError("int(): unendlich/NaN ist keine Zahl.")
        return int(x)
    raise ExprError(f"int() erwartet eine Zahl oder einen Text, nicht {_vs(x)}.")


def _fn_str(x):
    if isinstance(x, (str, numbers.Real)):
        return str(x)
    raise ExprError(f"str() erwartet eine Zahl oder einen Text, nicht {_vs(x)}.")


def _fn_all(xs):
    return all(bool(v) for v in _need_seq(xs, "all"))


def _fn_any(xs):
    return any(bool(v) for v in _need_seq(xs, "any"))


def _fn_is_prime(n):
    _need_ints("is_prime", n)
    if n > MAX_PRIME:
        raise ExprError(f"is_prime() erwartet n ≤ {MAX_PRIME}.")
    n = int(n)
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def _fn_gcd(a, b):
    _need_ints("gcd", a, b)
    return math.gcd(int(a), int(b))


def _fn_divides(a, b):
    """divides(a, b): a teilt b (b ist ein Vielfaches von a)."""
    _need_ints("divides", a, b)
    if a == 0:
        return b == 0
    return int(b) % int(a) == 0


def _fn_is_even(n):
    _need_ints("is_even", n)
    return int(n) % 2 == 0


def _fn_is_odd(n):
    _need_ints("is_odd", n)
    return int(n) % 2 == 1


def _fn_isqrt(n):
    _need_ints("isqrt", n)
    if n < 0:
        raise ExprError("isqrt() erwartet n ≥ 0.")
    return math.isqrt(int(n))


def _fn_implies(a, b):
    return (not bool(a)) or bool(b)


def _fn_iff(a, b):
    return bool(a) == bool(b)


FUNCS = {
    "len": _fn_len, "abs": _fn_abs, "min": _fn_min, "max": _fn_max, "sum": _fn_sum,
    "range": _fn_range, "sorted": _fn_sorted, "set": _fn_set, "list": _fn_list,
    "tuple": _fn_tuple, "int": _fn_int, "str": _fn_str, "all": _fn_all, "any": _fn_any,
    "is_prime": _fn_is_prime, "gcd": _fn_gcd, "divides": _fn_divides,
    "is_even": _fn_is_even, "is_odd": _fn_is_odd, "isqrt": _fn_isqrt,
    "implies": _fn_implies, "iff": _fn_iff,
}

# Methoden (nur als Aufruf obj.methode(...); Namen mit _ sind nie erlaubt)
_STR_METHODS = {"lower", "upper", "strip", "count", "startswith", "endswith", "isdigit",
                "isalpha", "isalnum", "isupper", "islower", "replace", "find", "casefold", "title"}
_LIST_METHODS = {"count", "index"}
_METHODS = _STR_METHODS | _LIST_METHODS

_OK_NODES = {
    ast.Expression, ast.Constant, ast.Name, ast.Load, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp,
    ast.Not, ast.USub, ast.UAdd, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
    ast.Mod, ast.Pow, ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In,
    ast.NotIn, ast.Subscript, ast.Slice, ast.Call, ast.Attribute, ast.List, ast.Tuple, ast.Set,
    ast.IfExp,
}

_DENY_TEXT = {
    "Lambda": "Lambda-Ausdrücke", "ListComp": "Schleifen/Comprehensions", "SetComp": "Schleifen/Comprehensions",
    "DictComp": "Schleifen/Comprehensions", "GeneratorExp": "Schleifen/Comprehensions",
    "JoinedStr": "f-Strings", "FormattedValue": "f-Strings", "NamedExpr": "Zuweisungen (:=)",
    "Await": "await", "Yield": "yield", "YieldFrom": "yield", "Starred": "Sternchen-Argumente",
    "Dict": "Wörterbuch-Literale", "Is": "'is'", "IsNot": "'is not'", "BitAnd": "Bit-Operatoren",
    "BitOr": "Bit-Operatoren", "BitXor": "Bit-Operatoren", "LShift": "Schiebe-Operatoren",
    "RShift": "Schiebe-Operatoren", "Invert": "Bit-Operatoren", "MatMult": "Matrix-Multiplikation (@)",
}


def _deny(node) -> ExprError:
    name = type(node).__name__
    what = _DENY_TEXT.get(name, f"Konstrukt '{name}'")
    return ExprError(f"Nicht erlaubt in Prädikat-Ausdrücken: {what}. Erlaubt sind Zahlen, "
                     "Texte, Namen, Vergleiche, and/or/not, + - * / // % **, a[i], Teilstücke s[::-1], "
                     "Listen/Tupel/Mengen, 'in' und die Funktionen: " + ", ".join(FUNCS) + ".")


def _validate(node, extra: set) -> None:
    """Prüft den Syntaxbaum vorab gegen die Positivliste."""
    t = type(node)
    if t not in _OK_NODES:
        raise _deny(node)
    if t is ast.Constant:
        if not isinstance(node.value, (int, float, str, bool, type(None))):
            raise ExprError("Dieser Wert (Bytes, komplexe Zahl …) ist nicht erlaubt.")
        if isinstance(node.value, int) and node.value.bit_length() > MAX_INT_BITS:
            raise ExprError("Die Zahl ist zu gross.")
        return
    if t is ast.Name:
        if node.id.startswith("_"):
            raise ExprError(f"Der Name '{node.id}' ist nicht erlaubt (Namen mit Unterstrich am Anfang sind gesperrt).")
        return
    if t is ast.Attribute:
        raise ExprError(f"Attributzugriff '.{node.attr}' ist nicht erlaubt "
                        "(erlaubt sind nur Aufrufe wie s.lower() oder a.count(x)).")
    if t is ast.Call:
        f = node.func
        if node.keywords:
            raise ExprError("Schlüsselwort-Argumente (f(a=1)) sind nicht erlaubt.")
        if isinstance(f, ast.Name):
            if f.id not in FUNCS and f.id not in extra:
                raise ExprError(f"Funktion '{f.id}' ist nicht erlaubt. Erlaubt: "
                                + ", ".join(sorted(set(FUNCS) | set(extra))) + ".")
        elif isinstance(f, ast.Attribute):
            if f.attr.startswith("_") or f.attr not in _METHODS:
                raise ExprError(f"Methode '.{f.attr}()' ist nicht erlaubt. Erlaubt: "
                                + ", ".join(sorted(_METHODS)) + ".")
            _validate(f.value, extra)
        else:
            raise ExprError("Nur Aufrufe von Funktionen mit festem Namen sind erlaubt.")
        for a in node.args:
            _validate(a, extra)
        return
    for child in ast.iter_child_nodes(node):
        _validate(child, extra)


def _normalize_expr(text: str) -> str:
    """Mathematische Zeichen und einfaches '=' in Python-Schreibweise."""
    def fix(s: str) -> str:
        for a, b in (("≤", "<="), ("≥", ">="), ("≠", "!="), ("∧", " and "), ("∨", " or "),
                     ("¬", " not "), ("∉", " not in "), ("∈", " in "), ("·", "*"), ("⋅", "*"),
                     ("×", "*"), ("−", "-"), ("^", "**")):
            s = s.replace(a, b)
        return re.sub(r"(?<![<>=!])=(?!=)", "==", s)
    return _map_outside_strings(text, fix)


def _parse_expr(expr, extra=()) -> ast.Expression:
    if not isinstance(expr, str):
        raise ExprError("Ein Ausdruck muss ein Text sein.")
    text = expr.strip()
    if not text:
        raise ExprError("Der Ausdruck ist leer.")
    if len(text) > MAX_EXPR_LEN:
        raise ExprError(f"Der Ausdruck ist zu lang (höchstens {MAX_EXPR_LEN} Zeichen).")
    py = _normalize_expr(text)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(py, mode="eval")
    except SyntaxError as e:
        raise ExprError(f"Ausdruck nicht lesbar: '{text}' ({e.msg}).") from None
    except (RecursionError, MemoryError, ValueError, OverflowError):
        raise ExprError("Der Ausdruck ist zu stark verschachtelt oder nicht lesbar.") from None
    _validate(tree, set(extra))
    return tree


def free_names(expr, extra_funcs=()) -> list[str]:
    """
    Namen (Variablen) eines Ausdrucks in der Reihenfolge des ersten Auftretens.

        free_names("x + y == 5")           # ['x', 'y']
        free_names("a[i] == a[j]")         # ['a', 'i', 'j']
    """
    tree = _parse_expr(expr, extra_funcs)
    seen: list[str] = []

    def walk(n):
        if isinstance(n, ast.Name):
            if n.id not in seen:
                seen.append(n.id)
            return
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Attribute):
                walk(n.func.value)
            for a in n.args:
                walk(a)
            return
        for c in ast.iter_child_nodes(n):
            walk(c)
    walk(tree.body)
    return seen


class _Eval:
    """Wertet einen geprüften Syntaxbaum aus. Jeder Knoten zählt gegen MAX_NODES."""

    def __init__(self, env: dict, funcs: dict | None = None):
        self.env = env
        self.funcs = funcs or {}
        self.ticks = 0

    def run(self, tree):
        try:
            return self.v(tree.body)
        except ExprError:
            raise
        except ZeroDivisionError:
            raise ExprError("Division durch 0.") from None
        except RecursionError:
            raise ExprError("Der Ausdruck ist zu stark verschachtelt.") from None
        except OverflowError:
            raise ExprError("Die Zahl wird zu gross.") from None
        except MemoryError:
            raise ExprError("Zu wenig Speicher für diese Rechnung.") from None
        except (TypeError, ValueError, IndexError, KeyError, AttributeError) as e:
            raise ExprError(f"Auswertung nicht möglich: {e}") from None

    # -- Knoten -------------------------------------------------------------
    def v(self, n):
        self.ticks += 1
        if self.ticks > MAX_NODES:
            raise ExprError("Die Auswertung ist zu aufwendig (zu viele Rechenschritte).")
        t = type(n)
        if t is ast.Constant:
            return n.value
        if t is ast.Name:
            if n.id in self.env:
                return self.env[n.id]
            if n.id in FUNCS or n.id in self.funcs:
                raise ExprError(f"'{n.id}' ist eine Funktion und muss mit Klammern aufgerufen werden.")
            raise ExprError(f"Name '{n.id}' ist nicht belegt (Lücke nicht gefüllt?).")
        if t is ast.BoolOp:
            if isinstance(n.op, ast.And):
                r = True
                for x in n.values:
                    r = self.v(x)
                    if not r:
                        return r
                return r
            r = False
            for x in n.values:
                r = self.v(x)
                if r:
                    return r
            return r
        if t is ast.UnaryOp:
            a = self.v(n.operand)
            if isinstance(n.op, ast.Not):
                return not a
            if not _is_num(a):
                raise ExprError(f"Vorzeichen vor {_vs(a)} nicht möglich.")
            return -a if isinstance(n.op, ast.USub) else +a
        if t is ast.BinOp:
            return self._binop(n.op, self.v(n.left), self.v(n.right))
        if t is ast.Compare:
            left = self.v(n.left)
            for op, comp in zip(n.ops, n.comparators):
                right = self.v(comp)
                if not self._cmp(op, left, right):
                    return False
                left = right
            return True
        if t is ast.IfExp:
            return self.v(n.body) if self.v(n.test) else self.v(n.orelse)
        if t is ast.Subscript:
            return self._subscript(n)
        if t is ast.Call:
            return self._call(n)
        if t in (ast.List, ast.Tuple, ast.Set):
            items = [self.v(x) for x in n.elts]
            if len(items) > MAX_SEQ:
                raise ExprError("Liste zu gross.")
            if t is ast.List:
                return items
            if t is ast.Tuple:
                return tuple(items)
            try:
                return set(items)
            except TypeError:
                raise ExprError("Mengenelemente müssen hashbar sein.") from None
        raise _deny(n)                                                  # pragma: no cover

    # -- Operationen ----------------------------------------------------------
    def _binop(self, op, a, b):
        if isinstance(op, ast.Add):
            if _is_num(a) and _is_num(b):
                return _norm_number(a + b)
            if type(a) is type(b) and isinstance(a, (str, list, tuple)):
                if len(a) + len(b) > MAX_SEQ:
                    raise ExprError("Ergebnis zu gross.")
                return a + b
            raise ExprError(f"{_vs(a)} + {_vs(b)} ist nicht möglich.")
        if isinstance(op, ast.Mult):
            if _is_num(a) and _is_num(b):
                return _norm_number(a * b)
            for s, k in ((a, b), (b, a)):
                if isinstance(s, (str, list, tuple)) and _is_int(k):
                    if len(s) * max(int(k), 0) > MAX_SEQ:
                        raise ExprError("Ergebnis zu gross.")
                    return s * int(k)
            raise ExprError(f"{_vs(a)} * {_vs(b)} ist nicht möglich.")
        if not (_is_num(a) and _is_num(b)):
            raise ExprError(f"Rechnung mit {_vs(a)} und {_vs(b)} ist nicht möglich (Zahlen erwartet).")
        if isinstance(op, ast.Sub):
            return _norm_number(a - b)
        if isinstance(op, ast.Div):
            if b == 0:
                raise ExprError("Division durch 0.")
            if isinstance(a, numbers.Rational) and isinstance(b, numbers.Rational):
                return _norm_number(Fraction(a) / Fraction(b))
            return a / b
        if isinstance(op, ast.FloorDiv):
            if b == 0:
                raise ExprError("Division durch 0.")
            return _norm_number(a // b)
        if isinstance(op, ast.Mod):
            if b == 0:
                raise ExprError("Rest einer Division durch 0.")
            return _norm_number(a % b)
        if isinstance(op, ast.Pow):
            return self._pow(a, b)
        raise _deny(op)                                                 # pragma: no cover

    def _pow(self, a, b):
        exact = isinstance(a, numbers.Rational) and isinstance(b, numbers.Integral)
        if exact:
            b = int(b)
            if abs(a) > 1 and abs(b) > MAX_POW_EXP:
                raise ExprError(f"Exponent zu gross (höchstens {MAX_POW_EXP}).")
            base = Fraction(a)
            bits = max(base.numerator.bit_length(), base.denominator.bit_length())
            if bits * abs(b) > MAX_INT_BITS:
                raise ExprError("Die Potenz wird zu gross.")
            if base == 0 and b < 0:
                raise ExprError("0 hoch negative Zahl: Division durch 0.")
            return _norm_number(base ** b)
        fa, fb = float(a), float(b)
        if fa < 0 and fb != int(fb):
            raise ExprError("Negative Zahl hoch Bruch ist in den reellen Zahlen nicht definiert.")
        return fa ** fb

    def _cmp(self, op, a, b) -> bool:
        if isinstance(op, ast.Eq):
            return bool(a == b)
        if isinstance(op, ast.NotEq):
            return bool(a != b)
        if isinstance(op, (ast.In, ast.NotIn)):
            if not _is_seq(b):
                raise ExprError(f"'in' braucht rechts eine Liste, einen Text oder eine Menge, nicht {_vs(b)}.")
            if isinstance(b, str) and not isinstance(a, str):
                raise ExprError("In einem Text kann man nur nach Text suchen.")
            try:
                found = a in b
            except TypeError:
                raise ExprError(f"{_vs(a)} kann hier nicht gesucht werden.") from None
            return found if isinstance(op, ast.In) else not found
        ordered = (_is_num(a) and _is_num(b)) or (isinstance(a, str) and isinstance(b, str))
        if not ordered:
            raise ExprError(f"{_vs(a)} und {_vs(b)} lassen sich nicht der Grösse nach vergleichen.")
        if isinstance(op, ast.Lt):
            return bool(a < b)
        if isinstance(op, ast.LtE):
            return bool(a <= b)
        if isinstance(op, ast.Gt):
            return bool(a > b)
        if isinstance(op, ast.GtE):
            return bool(a >= b)
        raise _deny(op)                                                 # pragma: no cover

    def _subscript(self, n):
        obj = self.v(n.value)
        sl = n.slice
        if isinstance(sl, ast.Slice):
            if not (isinstance(obj, (list, tuple, str, range)) or _is_ndarray(obj)):
                raise ExprError("Teilstücke gibt es nur bei Listen und Texten.")
            parts = [None if x is None else self.v(x) for x in (sl.lower, sl.upper, sl.step)]
            if any(p is not None and not _is_int(p) for p in parts):
                raise ExprError("Teilstück-Grenzen müssen ganze Zahlen sein.")
            if parts[2] == 0:
                raise ExprError("Die Schrittweite eines Teilstücks darf nicht 0 sein.")
            return obj[slice(*[None if p is None else int(p) for p in parts])]
        idx = self.v(sl)
        if isinstance(obj, Seq1):
            return obj[idx]
        if isinstance(obj, dict):
            try:
                return obj[idx]
            except (KeyError, TypeError):
                raise ExprError(f"Schlüssel {_vs(idx)} fehlt.") from None
        if isinstance(obj, (list, tuple, str, range)) or _is_ndarray(obj):
            if not _is_int(idx):
                raise ExprError(f"Position muss eine ganze Zahl sein, nicht {_vs(idx)}.")
            try:
                return obj[int(idx)]
            except IndexError:
                raise ExprError(f"Position {idx} liegt ausserhalb (Länge {len(obj)}; Python zählt ab 0, "
                                "für Zählung ab 1 die Liste mit one_based() umwandeln).") from None
        raise ExprError(f"{_vs(obj)} kann man nicht mit [..] ansprechen.")

    def _call(self, n):
        f = n.func
        args = [self.v(a) for a in n.args]
        if isinstance(f, ast.Name):
            if f.id in FUNCS:
                return FUNCS[f.id](*args)
            fn = self.funcs.get(f.id)
            if fn is None:
                raise ExprError(f"Funktion '{f.id}' ist nicht erlaubt.")
            res = fn.holds(*args) if isinstance(fn, Predicate) else fn(*args)
            return _as_bool(res, f.id)
        obj = self.v(f.value)
        name = f.attr
        if isinstance(obj, str) and name in _STR_METHODS:
            pass
        elif isinstance(obj, (list, tuple)) and name in _LIST_METHODS:
            pass
        else:
            raise ExprError(f"Methode '.{name}()' ist für {_vs(obj)} nicht erlaubt.")
        for a in args:
            if not isinstance(a, (str, numbers.Integral)) and not (name in _LIST_METHODS):
                raise ExprError("Methoden-Argumente müssen Zahlen oder Texte sein.")
        if name == "replace" and len(obj) * max(1, len(args[1]) if len(args) > 1 and isinstance(args[1], str) else 1) > MAX_SEQ:
            raise ExprError("Ergebnis zu gross.")
        try:
            res = getattr(obj, name)(*args)
        except ValueError:
            raise ExprError(f"{name}(): Element nicht gefunden.") from None
        if isinstance(res, (str, list)) and len(res) > MAX_SEQ:
            raise ExprError("Ergebnis zu gross.")
        return res


def safe_eval(expr: str, env: dict | None = None, *, functions: dict | None = None):
    """
    Wertet einen mathematischen Ausdruck sicher aus — **ohne** eval/exec.

    Der Text wird mit `ast.parse` nur *gelesen* (nichts wird ausgeführt) und dann von einem
    eigenen Auswerter über einer Positivliste berechnet: Zahlen, Texte, Namen aus `env`,
    Vergleiche (auch `a < b < c`, `in`), and/or/not, + - * / // % ** (exakt: `/` gibt Brüche),
    Indexzugriff `a[i]`, Teilstücke `s[::-1]`, Listen/Tupel/Mengen-Literale und die Funktionen
    len, abs, min, max, sum, range, sorted, set, list, tuple, int, str, all, any, is_prime,
    gcd, divides(a, b), is_even, is_odd, isqrt, implies, iff sowie wenige Text-Methoden
    (lower, upper, count, …). Alles andere (`__import__`, `x.__class__`, `open`, Lambda,
    Comprehensions, f-Strings …) wird mit einer klaren Meldung abgelehnt (`ExprError`).

        safe_eval("x % 2 == 0", {"x": 6})            # True
        safe_eval("s == s[::-1]", {"s": "REITTIER"}) # True
        safe_eval("a[1] == a[8]", {"a": one_based([3,2,1,1,4,4,2,3,1])})   # True
    """
    tree = _parse_expr(expr, (functions or {}).keys())
    return _Eval(dict(env or {}), functions).run(tree)


def _as_bool(r, who="Das Prädikat") -> bool:
    if isinstance(r, bool):
        return r
    if type(r).__name__ in ("bool_", "bool"):                           # numpy-Wahrheitswerte
        return bool(r)
    raise ValueError(f"{who} lieferte {_vs(r)}, aber ein Prädikat muss wahr oder falsch liefern "
                     "(Definition 1: es muss eine Aussage entstehen). Tipp: mit einem Vergleich "
                     "abschliessen, z. B. '== 1'.")


# ==========================================================================
# Prädikate
# ==========================================================================

class _Subst(ast.NodeTransformer):
    """Setzt Werte für Namen ein (nur Zahlen, Texte, Wahrheitswerte), um den Rechenweg zu zeigen."""

    def __init__(self, env):
        self.env = env

    def visit_Name(self, node):
        if node.id in self.env:
            v = self.env[node.id]
            if isinstance(v, bool):
                return ast.Constant(v)
            if isinstance(v, str):
                return ast.Constant(v)
            if isinstance(v, numbers.Real) and not isinstance(v, float):
                s = str(Fraction(v)) if isinstance(v, Fraction) else str(int(v))
                return ast.Name(id=f"({s})" if s.startswith("-") or "/" in s else s, ctx=ast.Load())
            if isinstance(v, float):
                return ast.Name(id=repr(v), ctx=ast.Load())
        return node


_COMBO_SYM = {"and": "∧", "or": "∨", "imp": "⇒", "iff": "⇔"}


class Predicate:
    """
    Ein Prädikat: Aussage mit Lücken (Definition 1).

        Predicate("sumequaltofive", ("x", "y"), func=lambda x, y: x + y == 5)
        Predicate.from_expr("istgerade", "x % 2 == 0")
        Predicate.from_expr("Pal(s)", "s == s[::-1]")
        p.holds(x=3, y=2)   # True   (alle Lücken gefüllt → Aussage)
        p(3, 2)             # dasselbe, positionell
        p.evaluate(1, 3)    # StatementResult mit Rechenweg

    `from_expr` wertet den Text mit dem sicheren Auswerter (`safe_eval`) aus, `func` darf eine
    beliebige Python-Funktion sein, die wahr/falsch liefert. `name` darf eine Kopfzeile
    "Q(i, j)" sein: dann sind i, j die *Lücken*, weitere Namen im Ausdruck (z. B. die Liste a)
    sind feste Werte, die beim Aufruf mitgegeben werden.

    Verknüpfung: `p & q`, `p | q`, `~p`, `p >> q` (⇒), `p.iff(q)`; `p.fill(x=3)` füllt einzelne
    Lücken (Teilauswertung).
    """

    def __init__(self, name: str = "", params=None, func=None, expr=None, *, gaps=None,
                 bound: dict | None = None, funcs: dict | None = None, label_fn=None,
                 _combo=None):
        if (func is None) == (expr is None) and _combo is None:
            raise ValueError("Ein Prädikat braucht genau eine Definition: func=… (Funktion) oder expr=… (Ausdruck).")
        name = (name or "").strip()
        header = re.fullmatch(r"([^()\s]*)\s*\(([^()]*)\)", name)
        if header:
            name = header.group(1)
            hp = [p.strip() for p in header.group(2).split(",") if p.strip()]
            gaps = hp if gaps is None else gaps
        if len(name) > 60:
            raise ValueError("Der Name des Prädikats ist zu lang.")
        self.name = name
        self.bound = dict(bound or {})
        self.funcs = dict(funcs or {})
        self.expr = expr
        self.func = func
        self._combo = _combo
        self._label_fn = label_fn
        self._tree = None
        if _combo is not None:
            self.params = tuple(params)
        elif expr is not None:
            self._tree = _parse_expr(expr, self.funcs.keys())
            free = [n for n in free_names(expr, self.funcs.keys())]
            if params is None:
                lead = list(gaps or [])
                params = lead + [n for n in free if n not in lead]
            params = tuple(params)
            unknown = [n for n in free if n not in params and n not in self.bound]
            if unknown:
                raise ValueError(f"Im Ausdruck kommen Namen vor, die keine Lücke sind: {', '.join(unknown)}. "
                                 f"Lücken: {', '.join(params) or '(keine)'}.")
            self.params = params
        else:
            if not callable(func):
                raise ValueError("func muss aufrufbar sein.")
            if params is None:
                try:
                    sig = inspect.signature(func)
                except (TypeError, ValueError):
                    raise ValueError("Die Lücken der Funktion sind nicht erkennbar; gib params=(…) an.") from None
                ok = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                if any(p.kind not in ok for p in sig.parameters.values()):
                    raise ValueError("Die Funktion darf nur normale Parameter haben (kein *args/**kwargs).")
                params = tuple(sig.parameters)
            self.params = tuple(params)
            if not self.name:
                fname = getattr(func, "__name__", "")
                self.name = "P" if fname in ("", "<lambda>") else fname
        if not self.params:
            raise ValueError("Ein Prädikat braucht mindestens eine Lücke (sonst ist es schon eine Aussage).")
        if len(set(self.params)) != len(self.params) or not all(
                isinstance(p, str) and p.isidentifier() for p in self.params):
            raise ValueError("Die Lücken müssen verschiedene Namen (Bezeichner) haben.")
        self.gaps = tuple(gaps) if gaps is not None else tuple(p for p in self.params if p not in self.bound)
        if any(g not in self.params for g in self.gaps):
            raise ValueError("gaps müssen Parameter des Prädikats sein.")

    # -- Bauen --------------------------------------------------------------
    @classmethod
    def from_expr(cls, name: str, expr: str, params=None, **kw) -> "Predicate":
        """Prädikat aus einem Ausdruck-Text: Predicate.from_expr("istgerade", "x % 2 == 0")."""
        return cls(name, params, expr=expr, **kw)

    @classmethod
    def from_func(cls, func, name: str = "", params=None, **kw) -> "Predicate":
        """Prädikat aus einer Python-Funktion; die Lücken sind ihre Parameter."""
        return cls(name, params, func=func, **kw)

    @classmethod
    def _make_combo(cls, op, a, b=None):
        parts = [a] if b is None else [a, b]
        params = []
        for p in parts:
            params += [q for q in p.open_params if q not in params]
        obj = cls("", params, _combo=(op, a, b))
        obj.gaps = tuple(params)
        return obj

    @property
    def open_params(self) -> tuple:
        """Die noch ungefüllten Lücken in Reihenfolge."""
        return tuple(p for p in self.params if p not in self.bound)

    def fill(self, *args, **kwargs) -> "Predicate":
        """Teilweise einsetzen: P(x, y).fill(y=3) ist ein Prädikat nur in x."""
        env = self._bind(args, kwargs, partial=True)
        new = copy.copy(self)
        new.bound = {**self.bound, **env}
        new.gaps = tuple(g for g in self.gaps if g not in new.bound)
        if not new.open_params:
            raise ValueError("Alle Lücken sind gefüllt: benutze holds(...) für die Aussage.")
        return new

    # -- Verknüpfung ----------------------------------------------------------
    @staticmethod
    def _coerce(o):
        if isinstance(o, Predicate):
            return o
        raise TypeError("Prädikate lassen sich nur mit Prädikaten verknüpfen.")

    def __and__(self, o): return Predicate._make_combo("and", self, self._coerce(o))
    def __or__(self, o): return Predicate._make_combo("or", self, self._coerce(o))
    def __invert__(self): return Predicate._make_combo("not", self)
    def __rshift__(self, o): return Predicate._make_combo("imp", self, self._coerce(o))
    def iff(self, o): return Predicate._make_combo("iff", self, self._coerce(o))

    # -- Auswerten ------------------------------------------------------------
    def _bind(self, args, kwargs, partial=False) -> dict:
        opn = self.open_params
        if len(args) > len(opn):
            raise ValueError(f"{self.signature()} hat {len(opn)} Lücke(n), aber {len(args)} Werte wurden übergeben.")
        env = dict(zip(opn, args))
        for k, v in kwargs.items():
            if k not in opn:
                raise ValueError(f"'{k}' ist keine (offene) Lücke von {self.signature()}; Lücken: {', '.join(opn)}.")
            if k in env:
                raise ValueError(f"Lücke '{k}' wurde doppelt gefüllt.")
            env[k] = v
        missing = [p for p in opn if p not in env]
        if missing and not partial:
            raise ValueError(f"Lücke(n) nicht gefüllt: {', '.join(missing)}. Erst wenn alle Lücken durch "
                             "Werte ersetzt sind, entsteht eine Aussage (Definition 1).")
        return env

    def _apply(self, env: dict) -> bool:
        """Auswertung mit vollständiger Belegung (intern; env darf mehr Namen enthalten)."""
        if self._combo is not None:
            op, a, b = self._combo
            env = {**env, **self.bound}
            va = a._apply(env)
            if op == "not":
                return not va
            if op == "and":
                return va and b._apply(env)
            if op == "or":
                return va or b._apply(env)
            if op == "imp":
                return (not va) or b._apply(env)
            return va == b._apply(env)
        full = {**env, **self.bound}
        if self._tree is not None:
            r = _Eval(full, self.funcs).run(self._tree)
            return _as_bool(r, self.signature())
        missing = [p for p in self.params if p not in full]
        if missing:
            raise ValueError(f"Wert für {', '.join(missing)} fehlt.")
        try:
            r = self.func(*[full[p] for p in self.params])
        except ValueError:
            raise
        except Exception as e:                                          # Fehler der eigenen Funktion
            raise ValueError(f"{self.signature()} liess sich nicht auswerten: {type(e).__name__}: {e}") from e
        return _as_bool(r, self.signature())

    def holds(self, *args, **kwargs) -> bool:
        """Setzt Werte für alle Lücken ein; Ergebnis ist der Wahrheitswert der Aussage."""
        return self._apply(self._bind(args, kwargs))

    __call__ = holds

    def evaluate(self, *args, **kwargs) -> "StatementResult":
        """Wie holds, aber mit Rechenweg (Lücken, Einsetzen, Teilaussagen, Wahrheitswert)."""
        return StatementResult(self, self._bind(args, kwargs))

    # -- Darstellung ------------------------------------------------------------
    def signature(self) -> str:
        """sumequaltofive(x, y) — oder der Ausdruck selbst bei namenlosen Prädikaten."""
        return self.label()

    def label(self, values: dict | None = None) -> str:
        """Aussage-Text, optional mit eingesetzten Werten: label({'x': 1, 'y': 3}) → sumequaltofive(1, 3)."""
        if self._label_fn is not None:
            return self._label_fn(values or {})
        if self._combo is not None:
            op, a, b = self._combo
            la = a.label(values)
            if op == "not":
                return "¬" + (la if re.fullmatch(r"[\w¬]+(\([^()]*\))?", la) else f"({la})")
            return f"({la} {_COMBO_SYM[op]} {b.label(values)})"
        if self.name:
            shown = []
            for g in (self.gaps or self.open_params):
                shown.append(_vs(values[g]) if values and g in values else g)
            return f"{self.name}({', '.join(shown)})"
        text = _pretty(ast.unparse(self._tree.body))
        if values:
            text = _pretty(ast.unparse(_Subst(values).visit(copy.deepcopy(self._tree.body))))
        return text

    def definition(self) -> str:
        """Bedeutung des Prädikats (Ausdruck in Schreibweise der Vorlesung) oder ''."""
        if self._tree is not None:
            return _pretty(ast.unparse(self._tree.body))
        if self._combo is not None:
            return self.label()
        return ""

    def __repr__(self):
        d = self.definition()
        return f"Predicate({self.signature()}" + (f" :⇔ {d}" if d and self.name else "") + ")"

    # -- Rechenweg ----------------------------------------------------------------
    def _rechenweg(self, env: dict):
        """(eingesetzter Text, Teilaussagen [(text, wert|None)], Nachschlagen [text], Wahrheitswert)."""
        value = self._apply(env)
        if self._tree is None:
            return self.label(env), [], [], value
        full = {**env, **self.bound}
        ev = _Eval(full, self.funcs)
        sub = _Subst(full)

        def text(n):
            return _pretty(ast.unparse(sub.visit(copy.deepcopy(n))))

        leaves: list = []

        def walk(n):
            if isinstance(n, ast.BoolOp):
                is_and = isinstance(n.op, ast.And)
                res = is_and
                decided = False
                for c in n.values:
                    if decided:
                        leaves.append((text(c), None))
                        continue
                    r = bool(walk(c))
                    res = r
                    if (is_and and not r) or (not is_and and r):
                        decided = True
                return res
            try:
                r = bool(ev.v(n))
            except ExprError:
                leaves.append((text(n), None))
                return False
            leaves.append((text(n), r))
            return r

        walk(self._tree.body)
        lookups: list[str] = []
        for node in ast.walk(self._tree.body):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) \
                    and not isinstance(node.slice, ast.Slice):
                try:
                    val = ev.v(node)
                except ExprError:
                    continue
                s = f"{text(node)} = {_vs(val)}"
                if s not in lookups:
                    lookups.append(s)
        return text(self._tree.body), leaves, lookups[:8], value


class StatementResult(Result):
    """Aus einem Prädikat und Werten wird eine Aussage (Definition 1)."""

    def __init__(self, pred: Predicate, env: dict):
        super().__init__()
        self.predicate = pred
        self.values = dict(env)
        self.text = pred.label(env)
        text, leaves, lookups, value = pred._rechenweg(env)
        self.value = value
        self.title = f"Aussage aus dem Prädikat {pred.signature()}"
        self.summary = f"{self.text} ist {'wahr' if value else 'falsch'} (Wahrheitswert {bit(value)})."
        d = pred.definition()
        items = [("Prädikat", pred.signature())]
        if d and pred.name:
            items.append(("Bedeutung", d))
        self.steps.append(Step(
            "Prädikat und Lücken", "Ein Prädikat ist ein Satz mit Lücken. Erst wenn jede Lücke durch einen Wert "
            "aus dem Universum ersetzt ist, entsteht eine Aussage mit Wahrheitswert.",
            items=items, rule="Definition 1"))
        self.steps.append(Step("Werte einsetzen", items=[(k, v) for k, v in env.items()]
                               + [("Aussage", self.text)]))
        if pred._tree is not None:
            its = [("eingesetzt", text)]
            if len(leaves) > 1:
                for k, (t, r) in enumerate(leaves, 1):
                    its.append((f"Teil {k}", f"{t}  →  " + ("nicht nötig" if r is None else bit(r))))
            its += [("nachgeschlagen", s) for s in lookups]
            self.steps.append(Step("Auswerten", "Die eingesetzten Werte werden ausgerechnet.", items=its,
                                   rule="Aussagenlogik (∧ ∨ ¬ ⇒ ⇔)" if len(leaves) > 1 else ""))
        else:
            self.steps.append(Step("Auswerten", "Die Funktion wird mit den Werten aufgerufen."))
        self.steps.append(Step("Wahrheitswert", items=[(self.text, value)],
                               note="Eine Aussage ist wahrheitsdefinit: entweder wahr (1) oder falsch (0)."))
        self.highlights = [("Aussage", self.text), ("Wahrheitswert", value)]

    def __bool__(self):
        return self.value


def relation_predicate(pairs, name: str = "R", params=("x", "y")) -> Predicate:
    """
    Zweistelliges Prädikat aus einer Menge von Paaren (oder einem Wörterbuch Person → Menge).

        mag = relation_predicate({("Anna", "Ben"), ("Ben", "Cleo")}, "mag")
        mag.holds("Anna", "Ben")      # True
        relation_predicate({"Anna": {"Ben"}, "Ben": {"Cleo"}}, "mag")
    """
    if isinstance(pairs, dict):
        pairs = {(a, b) for a, targets in pairs.items() for b in targets}
    pairs = {tuple(p) for p in pairs}
    if any(len(p) != 2 for p in pairs):
        raise ValueError("Eine Relation besteht aus Paaren (x, y).")
    pred = Predicate(name, params, func=lambda x, y: (x, y) in pairs)
    pred.pairs = frozenset(pairs)
    return pred


# Prädikate der Serie
SUM_EQUAL_TO_FIVE = Predicate.from_expr("sumequaltofive(x, y)", "x + y == 5")
PAL = Predicate.from_expr("Pal(s)", "s == s[::-1]")
FERMAT = Predicate.from_expr("Fermat(a, b, c, n)", "a**n + b**n == c**n")


# ==========================================================================
# Universen
# ==========================================================================

class Universe:
    """
    Ein endliches Universum: die Menge der Werte, über die ein Quantor läuft.

        Universe([1, 2, 3], name="U");   len(u);   list(u)
        naturals(100) · integers(50) · reals(2, 4) · index_set(9) · chars("abc")

    `finite=False` heisst: nur ein *Ausschnitt* eines unendlichen Universums (N, Z, R). Dann
    ist ein Zeuge/Gegenbeispiel weiterhin ein echter Beweis, "nichts gefunden" aber nicht.
    """

    def __init__(self, items, name: str = "", finite: bool = True, note: str = ""):
        items = tuple(items)
        if len(items) > MAX_UNIVERSE:
            raise ValueError(f"Das Universum ist zu gross ({len(items)} Elemente; höchstens {MAX_UNIVERSE}). "
                             "Wähle einen kleineren Ausschnitt.")
        self.items = items
        self.name = name
        self.finite = finite
        self.note = note

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def label(self) -> str:
        """Name des Universums (U, N, …) oder, falls namenlos, die Menge selbst."""
        return self.name or self.show()

    def show(self, maxn: int = 12) -> str:
        n = len(self.items)
        if n == 0:
            return "∅"
        if n <= maxn:
            body = ", ".join(_vs(x) for x in self.items)
        else:
            body = ", ".join(_vs(x) for x in self.items[:4]) + ", …, " + _vs(self.items[-1])
        return "{" + body + (", …" if not self.finite else "") + "}"

    def __repr__(self):
        return f"Universe({self.label()} = {self.show()})"


def naturals(bound: int = 100, start: int = 0) -> Universe:
    """Ausschnitt von ℕ = {0, 1, 2, …} (Zusammenfassung): start … bound. `start=1` für ℕ ohne 0."""
    _check_bound(bound)
    return Universe(range(start, bound + 1), "N", finite=False, note=f"Ausschnitt {start}…{bound} von ℕ")


def integers(bound: int = 100) -> Universe:
    """Ausschnitt von ℤ: −bound … bound."""
    _check_bound(bound)
    return Universe(range(-bound, bound + 1), "Z", finite=False, note=f"Ausschnitt −{bound}…{bound} von ℤ")


def reals(bound: int = 2, denominator: int = 4, name: str = "R") -> Universe:
    """
    Ausschnitt von ℝ als exakte Brüche p/q mit q ≤ denominator im Bereich [−bound, bound]
    (ponytail: ein Gitter von Brüchen statt echter reeller Zahlen; ein Zeuge ist trotzdem echt).
    """
    _check_bound(bound)
    if not isinstance(denominator, int) or not 1 <= denominator <= 20:
        raise ValueError("denominator muss eine ganze Zahl von 1 bis 20 sein.")
    vals = sorted({Fraction(p, q) for q in range(1, denominator + 1)
                   for p in range(-bound * q, bound * q + 1)})
    items = [int(v) if v.denominator == 1 else v for v in vals]
    return Universe(items, name, finite=False, note=f"Gitter p/q (q ≤ {denominator}) in [−{bound}, {bound}]")


def rationals(bound: int = 2, denominator: int = 4) -> Universe:
    """Ausschnitt von ℚ (wie `reals`, Name Q)."""
    return reals(bound, denominator, name="Q")


def index_set(n: int, start: int = 1) -> Universe:
    """Indexmenge {start, …, start+n−1}, Standard {1, 2, …, n} wie in der Serie."""
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError("n muss eine ganze Zahl ≥ 0 sein.")
    return Universe(range(start, start + n))


def chars(text: str) -> Universe:
    """Die Zeichen eines Textes als Universum."""
    if not isinstance(text, str):
        raise ValueError("chars() erwartet einen Text.")
    return Universe(tuple(text))


def _check_bound(bound):
    if not isinstance(bound, int) or isinstance(bound, bool) or bound < 0:
        raise ValueError("Die Schranke muss eine ganze Zahl ≥ 0 sein.")
    if bound > MAX_UNIVERSE:
        raise ValueError(f"Die Schranke ist zu gross (höchstens {MAX_UNIVERSE}).")


_ALIASES = {"N": "N", "ℕ": "N", "N0": "N", "ℕ0": "N", "N*": "N+", "ℕ*": "N+", "N+": "N+",
            "Z": "Z", "ℤ": "Z", "R": "R", "ℝ": "R", "Q": "Q", "ℚ": "Q"}


def named_universe(name: str, bound: int | None = None) -> Universe:
    """Ausschnitt eines Standardmengen-Namens: N/ℕ, Z/ℤ, R/ℝ, Q/ℚ, N* (ohne 0)."""
    key = _ALIASES.get(str(name).strip())
    if key is None:
        raise ValueError(f"Unbekannte Standardmenge '{name}' (bekannt: N, Z, R, Q, N*).")
    if key == "N":
        return naturals(bound or 100)
    if key == "N+":
        u = naturals(bound or 100, start=1)
        u.name = "N*"
        return u
    if key == "Z":
        return integers(bound or 100)
    if key == "R":
        return reals(bound or 2, 4)
    return rationals(bound or 2, 4)


def as_universe(u, name: str = "") -> Universe:
    """Macht aus Liste/Tupel/Menge/range/Text/Dictionary(Schlüssel)/Seq1 ein `Universe`."""
    if isinstance(u, Universe):
        return u
    if isinstance(u, (bool, numbers.Number)):
        raise ValueError("Ein Universum ist eine Menge oder Liste von Werten, keine einzelne Zahl "
                         "(meinst du range(n) oder index_set(n)?).")
    if isinstance(u, range):
        if len(u) > MAX_UNIVERSE:
            raise ValueError(f"Das Universum ist zu gross ({len(u)} Elemente; höchstens {MAX_UNIVERSE}).")
        return Universe(tuple(u), name)
    if isinstance(u, str):
        return Universe(tuple(u), name)
    if isinstance(u, (set, frozenset)):
        try:
            return Universe(sorted(u), name)
        except TypeError:
            return Universe(tuple(u), name)
    if isinstance(u, dict):
        return Universe(tuple(u), name)
    if hasattr(u, "tolist") and not isinstance(u, (list, tuple)):
        u = u.tolist()
    if isinstance(u, (list, tuple, Seq1)):
        return Universe(tuple(u), name)
    try:
        it = iter(u)
    except TypeError:
        raise ValueError(f"Kein Universum: {u!r} lässt sich nicht durchlaufen.") from None
    items = tuple(itertools.islice(it, MAX_UNIVERSE + 1))
    return Universe(items, name)


# ==========================================================================
# Quantoren
# ==========================================================================

_SYM = {"forall": "∀", "exists": "∃"}


def _parse_prefix(spec, universes, need_universe=True):
    """Quantorenfolge → [(kind, var, Universe|None)]. Text "∀i ∃j", "forall i exists j" oder Liste."""
    universes = dict(universes or {})
    out: list = []

    def universe_for(var, uname):
        key = uname or var
        if key in universes:
            v = universes[key]
            if isinstance(v, Universe) and uname and not v.name:
                return Universe(v.items, uname, v.finite, v.note)
            return as_universe(v, name=uname or "")
        if uname and _ALIASES.get(uname):
            return named_universe(uname)
        if need_universe:
            raise ValueError(f"Kein Universum für '{key}': übergib universes={{'{key}': …}}.")
        return None

    if isinstance(spec, str):
        pos = 0
        for m in _PREFIX_RE.finditer(spec):
            if spec[pos:m.start()].strip(" \t:.,;"):
                raise ValueError(f"Quantorenfolge nicht lesbar bei '{spec[pos:m.start()].strip()}'.")
            kind = "forall" if "all" in m.group(1) or m.group(1) == "∀" else "exists"
            for var in re.split(r"\s*,\s*", m.group(2).strip()):
                out.append((kind, var, universe_for(var, m.group(3))))
            pos = m.end()
        if spec[pos:].strip(" \t:.,;"):
            raise ValueError(f"Quantorenfolge nicht lesbar bei '{spec[pos:].strip()}'.")
    else:
        for entry in spec:
            if not isinstance(entry, (tuple, list)) or len(entry) not in (2, 3):
                raise ValueError("Jeder Eintrag der Quantorenfolge ist (Quantor, Variable[, Universum]).")
            kind = _norm_kind(entry[0])
            var = entry[1]
            uni = as_universe(entry[2]) if len(entry) == 3 else universe_for(var, None)
            out.append((kind, var, uni))
    if not out:
        raise ValueError("Keine Quantoren gefunden (erwartet z. B. '∀i ∃j').")
    names = [v for _, v, _ in out]
    if len(set(names)) != len(names):
        raise ValueError("Jede Variable darf nur einmal quantifiziert werden.")
    return out


_PREFIX_RE = re.compile(r"(∀|∃|\bfor\s*all\b|\bforall\b|\bexists\b)\s*([^\W\d]\w*(?:\s*,\s*[^\W\d]\w*)*)"
                        r"(?:\s*(?:∈|\bin\b)\s*([^\W\d]\w*))?")


def _norm_kind(k) -> str:
    k = str(k).strip().lower()
    if k in ("forall", "∀", "all", "alle", "a"):
        return "forall"
    if k in ("exists", "∃", "exist", "ex", "e"):
        return "exists"
    raise ValueError(f"Unbekannter Quantor '{k}' (erwartet 'forall'/'∀' oder 'exists'/'∃').")


def _as_predicate(pred) -> Predicate:
    if isinstance(pred, Predicate):
        return pred
    if isinstance(pred, str):
        return Predicate("", expr=pred)
    if callable(pred):
        return Predicate.from_func(pred)
    raise ValueError("Das Prädikat muss ein Predicate, ein Ausdruck-Text oder eine Funktion sein.")


def _formula_str(qs, body: str) -> str:
    head = "".join(f"{_SYM[k]}{v} ∈ {u.label()}: " if u is not None else f"{_SYM[k]}{v}: " for k, v, u in qs)
    return head + body


def _words(qs, body: str) -> str:
    if not qs:
        return body
    k, v, u = qs[0]
    rest = _words(qs[1:], body)
    where = f" ∈ {u.label()}" if u is not None else ""
    if k == "forall":
        return f"für alle {v}{where} gilt: {rest}"
    return f"es gibt ein {v}{where}, sodass {rest}"


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def python_code(spec, universes=None) -> str:
    """
    Schleifen-Code, der die Aussage zur Quantorenfolge ausrechnet (wie die Funktionen im Notebook).

        print(python_code("∀i ∃j"))
    """
    qs = _parse_prefix(spec, universes, need_universe=False)
    return _code(qs)


def _code(qs) -> str:
    n = len(qs)
    names = [v for _, v, _ in qs]
    lines = [f"def aussage(P, {', '.join('U_' + v for v in names)}):"]

    def emit(k, ind):
        if k == n:
            return [f"{ind}r{k} = P({', '.join(names)})"]
        kind, v, _ = qs[k]
        out = [f"{ind}r{k} = {'True' if kind == 'forall' else 'False'}",
               f"{ind}for {v} in U_{v}:   # {_SYM[kind]}{v}"]
        out += emit(k + 1, ind + "    ")
        if kind == "forall":
            out += [f"{ind}    if not r{k + 1}:", f"{ind}        r{k} = False", f"{ind}        break"]
        else:
            out += [f"{ind}    if r{k + 1}:", f"{ind}        r{k} = True", f"{ind}        break"]
        return out

    lines += emit(0, "    ")
    lines.append("    return r0")
    return "\n".join(lines)


class _Node:
    __slots__ = ("kind", "var", "value", "certain", "children")

    def __init__(self, kind, var, value, certain, children):
        self.kind, self.var, self.value, self.certain, self.children = kind, var, value, certain, children


def _solve(qs, k, leaf) -> _Node:
    if k == len(qs):
        return _Node(None, None, leaf(), True, [])
    kind, var, U = qs[k]
    kids = []
    for item in U.items:
        leaf.binding[var] = item
        kids.append((item, _solve(qs, k + 1, leaf)))
    leaf.binding.pop(var, None)
    if kind == "forall":
        value = all(n.value for _, n in kids)
        certain = (U.finite and all(n.certain for _, n in kids)) if value \
            else any((not n.value) and n.certain for _, n in kids)
    else:
        value = any(n.value for _, n in kids)
        certain = any(n.value and n.certain for _, n in kids) if value \
            else (U.finite and all(n.certain for _, n in kids))
    return _Node(kind, var, value, certain, kids)


def _evidence(node: _Node):
    """Zeuge/Gegenbeispiel entlang der äussersten Quantoren; None, wenn es keinen einzelnen gibt."""
    if node.kind is None:
        return None
    target = node.kind == "exists"
    if node.value != target:
        return None
    best = None
    for item, ch in node.children:
        if ch.value == target:
            if ch.certain:
                best = (item, ch)
                break
            if best is None:
                best = (item, ch)
    if best is None:
        return None
    d = {node.var: best[0]}
    sub = _evidence(best[1])
    if sub:
        d.update(sub)
    return d


class QuantResult(Result):
    """
    Ergebnis einer quantifizierten Aussage (Definition 2).

    Attribute: `value` (Wahrheitswert), `certain` (False, wenn nur ein Ausschnitt eines
    unendlichen Universums geprüft wurde und das Ergebnis dort nichts beweist), `witness`
    (Zeuge bei äusserem ∃ und wahr), `counterexample` (Gegenbeispiel bei äusserem ∀ und falsch),
    `formula`, `meaning` (Deutung in Worten), `table` (alle Auswertungen), `evaluations`
    (Liste (Belegung, Wahrheitswert)), `vacuous` (leeres Universum), `code` (Schleifen-Code).
    """

    def __init__(self, qs, pred: Predicate, fixed: dict, root: _Node, evals: list):
        super().__init__()
        self.quantifiers = qs
        self.predicate = pred
        self.fixed = dict(fixed)
        self.root = root
        self.evaluations = evals
        self.value = root.value
        self.certain = root.certain
        self.vars = [v for _, v, _ in qs]
        body = pred.label()
        self.formula = _formula_str(qs, body)
        self.meaning = _cap(_words(qs, body)) + "."
        self.vacuous = any(len(u) == 0 for _, _, u in qs)
        self.code = _code(qs)
        ev = _evidence(root)
        self._ev = ev
        outer = qs[0][0]
        self.witness = ev if (outer == "exists" and self.value and ev) else None
        self.counterexample = ev if (outer == "forall" and not self.value and ev) else None
        self.table: Table | None = None
        self._build()

    def __bool__(self):
        return self.value

    # -- Aufbau des Rechenwegs ---------------------------------------------
    def _matches(self, binding: tuple) -> bool:
        if not self._ev:
            return False
        pos = {v: i for i, v in enumerate(self.vars)}
        return all(binding[pos[v]] == x for v, x in self._ev.items())

    def _make_table(self) -> Table:
        evals = self.evaluations
        idxs = list(range(len(evals)))
        caption = "grün = wahr, rot = falsch, blau = Zeuge/Gegenbeispiel"
        if len(evals) > MAX_TABLE_ROWS:
            # ponytail: lange Tabellen werden gekürzt (Zeuge/Gegenbeispiel bleibt sichtbar)
            deciding = [k for k, (b, _) in enumerate(evals) if self._matches(b)][:MAX_TABLE_ROWS // 2]
            keep = set(deciding)
            for k in idxs:
                if len(keep) >= MAX_TABLE_ROWS:
                    break
                keep.add(k)
            idxs = sorted(keep)
            caption += f" · gekürzt: {len(idxs)} von {len(evals)} Zeilen"
        header = list(self.vars) + [self.predicate.label()]
        rows, marks = [], {}
        for pos, k in enumerate(idxs):
            b, v = evals[k]
            rows.append([_vs(x) for x in b] + [bit(v)])
            marks[pos] = "hi" if self._matches(b) else ("ok" if v else "bad")
        return Table(header, rows, marks, caption)

    def _build(self):
        qs, pred = self.quantifiers, self.predicate
        kind0, var0, U0 = qs[0]
        self.title = ("Allquantor " if kind0 == "forall" else "Existenzquantor ") \
            if len(qs) == 1 else "Verschachtelte Quantoren "
        self.title += self.formula
        self.summary = f"{self.meaning} Die Aussage ist {'wahr' if self.value else 'falsch'}" \
                       + ("" if self.certain else " (im geprüften Ausschnitt)") + "."
        # 1. Lesen
        items = [("Formel", self.formula)]
        items += [(f"Universum {v}", f"{u.label()} = {u.show()}" if u.name else u.show()) for _, v, u in qs]
        if pred.name and pred.definition():
            items.append(("Prädikat", f"{pred.signature()} :⇔ {pred.definition()}"))
        for k, v in self.fixed.items():
            items.append((k, _vs(v)))
        note = "Positionen in Listen zählen hier ab 1." if any(isinstance(v, Seq1) for v in self.fixed.values()) else ""
        self.steps.append(Step(
            "Aussage lesen", f"In Worten: {self.meaning}", items=items, note=note,
            rule="Definition 2 (∀ Generalisierung, ∃ Spezialisierung)"))
        # leeres Universum
        empties = [v for _, v, u in qs if len(u) == 0]
        if empties:
            self.steps.append(Step(
                "Leeres Universum",
                f"Das Universum von {', '.join(empties)} hat kein Element. Dann gibt es nichts zu widerlegen: "
                "ein Allquantor ist wahr (leere Konjunktion), und es gibt keinen Zeugen: ein Existenzquantor "
                "ist falsch (leere Disjunktion).", rule="Leeres Universum"))
        # 2. ausschreiben (Aufgabe 2)
        if len(qs) == 1 and 0 < len(U0) <= MAX_TABLE_ROWS:
            vals = [(item, v) for (item,), v in self.evaluations]
            sym = " ∧ " if kind0 == "forall" else " ∨ "
            if len(vals) > 6:
                shown = vals[:3] + [None] + vals[-1:]
            else:
                shown = vals
            terms = [pred.label({var0: it}) if s else "…" for s in shown for it in [s[0] if s else None]]
            bits_ = [bit(s[1]) if s else "…" for s in shown]
            self.steps.append(Step(
                "Ohne Quantor ausschreiben",
                ("∀ bedeutet: P(x₁) ∧ P(x₂) ∧ … ∧ P(xₙ). " if kind0 == "forall"
                 else "∃ bedeutet: P(x₁) ∨ P(x₂) ∨ … ∨ P(xₙ). ") + "Jeder Wert des Universums wird eingesetzt.",
                items=[("ausgeschrieben", sym.join(terms)), ("Wahrheitswerte", sym.join(bits_))],
                rule="Allquantor als Konjunktion" if kind0 == "forall" else "Existenzquantor als Disjunktion"))
        # 3. Tabelle
        if self.evaluations:
            self.table = self._make_table()
            self.steps.append(Step(
                "Alle Werte einsetzen",
                f"{len(self.evaluations)} Auswertung(en) des Prädikats" + (
                    "; die innersten Quantoren laufen am schnellsten." if len(qs) > 1 else "."),
                table=self.table))
        # 4. innere Aussagen je äusserem Wert
        if len(qs) >= 2 and self.root.children:
            rows, marks = [], {}
            for pos, (item, ch) in enumerate(self.root.children[:MAX_TABLE_ROWS]):
                inner = _formula_str(qs[1:], pred.label({var0: item}))
                evid = _evidence(ch)
                rows.append([_vs(item), inner, bit(ch.value), _bind_text(evid) if evid else ""])
                marks[pos] = "ok" if ch.value else "bad"
            self.steps.append(Step(
                f"Innere Aussage für jedes {var0}",
                f"Von innen nach aussen: zuerst wird die Aussage ohne {_SYM[kind0]}{var0} für jeden Wert von "
                f"{var0} bestimmt.", rule="Verschachtelte Quantoren",
                table=Table([var0, "innere Aussage", "Wahrheitswert", "Zeuge/Gegenbeispiel"], rows, marks)))
        # 5. Schluss
        sym0 = _SYM[kind0]
        ev = self._ev
        if len(U0) == 0:
            text = f"{sym0}{var0} über ein leeres Universum ist " + ("wahr." if kind0 == "forall" else "falsch.")
            rule = "Leeres Universum"
        elif kind0 == "exists" and self.value:
            text = f"Zeuge gefunden: {_bind_text(ev) if ev else ''}. Ein einziger Zeuge genügt, damit ∃ wahr ist."
            rule = "Existenzquantor: ein Zeuge genügt"
        elif kind0 == "forall" and not self.value:
            text = (f"Gegenbeispiel gefunden: {_bind_text(ev) if ev else ''}. Ein einziges Gegenbeispiel macht "
                    "∀ falsch.")
            rule = "Allquantor: ein Gegenbeispiel genügt"
        elif kind0 == "forall":
            text = ("Kein Gegenbeispiel: " + ("das Prädikat gilt für jeden Wert." if len(qs) == 1 else
                    f"die innere Aussage ist für jeden Wert von {var0} wahr."))
            rule = "Allquantor: alle Werte müssen gelten"
        else:
            text = ("Kein Zeuge: " + ("das Prädikat gilt für keinen Wert." if len(qs) == 1 else
                    f"die innere Aussage ist für keinen Wert von {var0} wahr."))
            rule = "Existenzquantor: alle Werte müssen scheitern"
        self.steps.append(Step("Schluss", text, rule=rule))
        if not self.certain:
            names = ", ".join(f"{u.label()}" for _, _, u in qs if not u.finite)
            self.warnings.append(
                f"Nur ein endlicher Ausschnitt ({names}) wurde geprüft. Ein Zeuge/Gegenbeispiel wäre ein echter "
                "Beweis, 'nichts gefunden' beweist für das ganze Universum nichts — dort muss man argumentieren.")
        self.highlights = [("Aussage", self.formula), ("Wahrheitswert", self.value)]
        if self.witness:
            self.highlights.append(("Zeuge", _bind_text(self.witness)))
        if self.counterexample:
            self.highlights.append(("Gegenbeispiel", _bind_text(self.counterexample)))


def _check_fixed_and_build(qs, pred: Predicate, fixed: dict, cls=QuantResult):
    vars_ = [v for _, v, _ in qs]
    if len(set(vars_)) != len(vars_):
        raise ValueError("Jede Variable darf nur einmal quantifiziert werden.")
    for v in vars_:
        if v not in pred.params:
            raise ValueError(f"Die Variable '{v}' ist keine Lücke von {pred.signature()} "
                             f"(Lücken: {', '.join(pred.params)}).")
        if v in pred.bound:
            raise ValueError(f"Die Lücke '{v}' ist schon gefüllt.")
    for k in fixed:
        if k not in pred.params:
            raise ValueError(f"'{k}' ist keine Lücke von {pred.signature()} (Lücken: {', '.join(pred.params)}).")
        if k in vars_:
            raise ValueError(f"'{k}' ist gleichzeitig quantifiziert und fest vorgegeben.")
    uncovered = [p for p in pred.open_params if p not in vars_ and p not in fixed]
    if uncovered:
        raise ValueError(f"Freie Variable(n): {', '.join(uncovered)}. Binde sie mit einem Quantor "
                         "oder gib einen festen Wert als Schlüsselwort mit (z. B. a=[…]).")
    total = 1
    for _, _, u in qs:
        total *= len(u)
        if total > MAX_COMBINATIONS:
            raise ValueError(f"Zu viele Kombinationen (mehr als {MAX_COMBINATIONS}). Verkleinere die Universen.")
    evals: list = []
    binding: dict = {}

    def leaf():
        env = {**fixed, **binding}
        try:
            v = pred._apply(env)
        except ValueError as e:
            raise ValueError(f"Auswertung bei {_bind_text(binding)} nicht möglich: {e}") from None
        evals.append((tuple(binding[x] for x in vars_), v))
        return v

    leaf.binding = binding
    root = _solve(qs, 0, leaf)
    return cls(qs, pred, fixed, root, evals)


def quantify(spec, pred, universes=None, **fixed) -> QuantResult:
    """
    Beliebige Quantorenfolge auswerten, z. B. die Notebook-Funktionen forAll_i_itEx_j.

        quantify("∀i ∃j", "M[i][j] == 1", {"i": range(3), "j": range(3)}, M=M)
        quantify("∃i ∀j", pred, universes={"i": index_set(6), "j": index_set(6)}, M=M)
        quantify([("forall", "x", U), ("exists", "y", V)], pred)

    spec: Text ("∀i ∃j", "forall i exists j", "∀x ∈ U ∃y ∈ V") oder Liste von (Quantor, Variable[, Universum]).
    pred: `Predicate`, Ausdruck-Text oder Funktion. fixed: feste Werte für weitere Namen (z. B. Listen).
    """
    qs = _parse_prefix(spec, universes)
    return _check_fixed_and_build(qs, _as_predicate(pred), fixed)


def _simple(kind, universe, pred, fixed) -> QuantResult:
    p = _as_predicate(pred)
    if isinstance(universe, dict):
        pairs = list(universe.items())
    else:
        cands = [x for x in p.open_params if x not in fixed]
        if len(cands) != 1:
            raise ValueError(f"Das Prädikat {p.signature()} hat mehrere offene Lücken ({', '.join(cands) or 'keine'}). "
                             "Gib das Universum je Variable an: forall({'i': …, 'j': …}, pred).")
        pairs = [(cands[0], universe)]
    qs = [(kind, v, as_universe(u)) for v, u in pairs]
    return _check_fixed_and_build(qs, p, fixed)


def forall(universe, pred, /, **fixed) -> QuantResult:
    """
    Allquantor ∀ (Generalisierung): wahr, wenn P(x) für *alle* x des Universums gilt.
    Leeres Universum → wahr. Ein Gegenbeispiel genügt für falsch.

        forall(range(10), "x*x >= x")                      # ∀x ∈ {0..9}: x² ≥ x
        forall({"i": range(3), "j": range(3)}, pred)       # ∀i ∀j
        forall(index_set(9), "a[i] > 0", a=one_based(a))   # feste Werte als Schlüsselwort
    """
    return _simple("forall", universe, pred, fixed)


def exists(universe, pred, /, **fixed) -> QuantResult:
    """
    Existenzquantor ∃ (Spezialisierung): wahr, wenn P(x) für *mindestens ein* x gilt.
    Leeres Universum → falsch. Ein Zeuge genügt für wahr.

        exists(range(100), "is_prime(x) and x > 50 and x % 2 == 0")   # falsch
    """
    return _simple("exists", universe, pred, fixed)


allquantor = forall            # Vorlesung: Allquantor
existenzquantor = exists       # Vorlesung: Existenzquantor


# --------------------------------------------------------------------------
# Aufgabe 2: ohne Quantoren ausschreiben + Algorithmus
# --------------------------------------------------------------------------

class UnrollResult(Result):
    """Aufgabe 2: ∀ als Konjunktion bzw. ∃ als Disjunktion, mit Algorithmus (`algorithm`)."""

    def __init__(self, kind, universe: Universe, pred, var, fixed):
        super().__init__()
        self.kind = kind
        self.universe = universe
        sym = _SYM[kind]
        op = "∧" if kind == "forall" else "∨"
        n = len(universe)
        generic = f"P(x₁) {op} P(x₂) {op} … {op} P(xₙ)"
        self.expression = generic
        self.algorithm = (
            "def alle(U, P):\n    for x in U:\n        if not P(x):\n            return False   # Gegenbeispiel\n"
            "    return True            # kein Gegenbeispiel (auch bei leerem U)" if kind == "forall" else
            "def mindestens_eines(U, P):\n    for x in U:\n        if P(x):\n            return True    # Zeuge\n"
            "    return False           # kein Zeuge (auch bei leerem U)")
        self.title = f"{sym}x ∈ U: P(x) ohne Quantor"
        self.summary = f"{sym}x ∈ U: P(x) ≡ {generic}"
        self.steps.append(Step(
            "Ausschreiben", f"Das Universum sei U = x₁, x₂, …, xₙ. "
            + ("Alle Werte müssen das Prädikat erfüllen: Konjunktion." if kind == "forall"
               else "Mindestens ein Wert muss das Prädikat erfüllen: Disjunktion."),
            items=[("Aussage", f"{sym}x ∈ U: P(x)"), ("ohne Quantor", generic)],
            rule="Allquantor als Konjunktion" if kind == "forall" else "Existenzquantor als Disjunktion"))
        self.steps.append(Step(
            "Algorithmus",
            "Schleife über das Universum" + (" mit Abbruch beim ersten Gegenbeispiel." if kind == "forall"
                                              else " mit Abbruch beim ersten Zeugen."),
            items=[("", line.replace("    ", "\u00a0\u00a0\u00a0\u00a0")) for line in self.algorithm.split("\n")]))
        self.value = None
        if pred is not None:
            p = _as_predicate(pred)
            res = _simple(kind, universe, p, fixed) if var is None else \
                _check_fixed_and_build([(kind, var, universe)], p, fixed)
            self.value = res.value
            self.result = res
            terms = [p.label({res.vars[0]: x}) for x in universe.items[:6]]
            self.expression = f" {op} ".join(terms) + (f" {op} …" if n > 6 else "") if terms else \
                ("wahr (leere Konjunktion)" if kind == "forall" else "falsch (leere Disjunktion)")
            self.steps.append(Step(
                "Beispiel mit konkretem Universum", items=[("Universum", universe.show()),
                                                          ("ausgeschrieben", self.expression),
                                                          ("Wahrheitswert", res.value)]))
            self.highlights = [("Wahrheitswert", res.value)]
        else:
            self.highlights = [("ohne Quantor", generic)]


def unroll(kind, universe, pred=None, /, **fixed) -> UnrollResult:
    """
    Aufgabe 2: Quantor ohne Quantoren schreiben und einen Algorithmus angeben.

        unroll("forall", [1, 2, 3])                 # allgemein: P(x₁) ∧ … ∧ P(xₙ), Algorithmus
        unroll("exists", range(5), "x*x == 9")      # mit Beispiel: P(0) ∨ P(1) ∨ … , Wahrheitswert
    """
    return UnrollResult(_norm_kind(kind), as_universe(universe), pred, None, fixed)


# ==========================================================================
# Quantoren-Formeln als Text: lesen, ausgeben, Negation nach innen schieben
# ==========================================================================

class QFormula:
    """Basisklasse für Formeln mit Quantoren (Text → Baum). `str(f)` gibt ∀x ∈ U: … aus."""

    def to_str(self) -> str:
        return _fstr(self, False)

    def tex(self) -> str:
        return _fstr(self, True)

    def __str__(self):
        return self.to_str()

    def negate(self) -> "QFormula":
        return QNot(self)


class QAtom(QFormula):
    """Eine Teilaussage ohne Quantor und ohne Verknüpfung: P(x, y), x < y, a[i] = a[j]."""

    def __init__(self, text: str):
        self.text = text

    def __eq__(self, o):
        return isinstance(o, QAtom) and o.text == self.text

    def __hash__(self):
        return hash(("atom", self.text))

    def __repr__(self):
        return f"QAtom({self.text!r})"


class QNot(QFormula):
    def __init__(self, arg: QFormula):
        self.arg = arg

    def __eq__(self, o):
        return isinstance(o, QNot) and o.arg == self.arg

    def __hash__(self):
        return hash(("not", self.arg))

    def __repr__(self):
        return f"QNot({self.arg!r})"


class QBin(QFormula):
    """op ∈ 'and', 'or', 'imp', 'iff'."""

    def __init__(self, op: str, a: QFormula, b: QFormula):
        self.op, self.a, self.b = op, a, b

    def __eq__(self, o):
        return isinstance(o, QBin) and (o.op, o.a, o.b) == (self.op, self.a, self.b)

    def __hash__(self):
        return hash((self.op, self.a, self.b))

    def __repr__(self):
        return f"QBin({self.op!r}, {self.a!r}, {self.b!r})"


class QQuant(QFormula):
    """kind ∈ 'forall', 'exists'; universe ist der Text hinter ∈ (oder None)."""

    def __init__(self, kind: str, var: str, universe: str | None, body: QFormula):
        self.kind, self.var, self.universe, self.body = kind, var, universe, body

    def __eq__(self, o):
        return isinstance(o, QQuant) and (o.kind, o.var, o.universe, o.body) == \
            (self.kind, self.var, self.universe, self.body)

    def __hash__(self):
        return hash((self.kind, self.var, self.universe, self.body))

    def __repr__(self):
        return f"QQuant({self.kind!r}, {self.var!r}, {self.universe!r}, {self.body!r})"


_PREC = {"iff": 1, "imp": 2, "or": 3, "and": 4}
_TEXSYM = {"and": r" \wedge ", "or": r" \vee ", "imp": r" \Rightarrow ", "iff": r" \Leftrightarrow "}
_UNISYM = {"and": " ∧ ", "or": " ∨ ", "imp": " ⇒ ", "iff": " ⇔ "}


def _atom_tex(t: str) -> str:
    for a, b in (("≤", r" \leq "), ("≥", r" \geq "), ("≠", r" \neq "), ("∉", r" \notin "), ("∈", r" \in "),
                 ("·", r" \cdot ")):
        t = t.replace(a, b)
    return t.replace("_", r"\_").replace("%", r"\%")


def _simple_atom(t: str) -> bool:
    return bool(re.fullmatch(r"[\w']+(\([^()]*\))?", t))


def _fstr(f: QFormula, tex: bool) -> str:
    if isinstance(f, QAtom):
        return _atom_tex(f.text) if tex else f.text
    if isinstance(f, QNot):
        a = f.arg
        inner = _fstr(a, tex)
        neg = r"\neg " if tex else "¬"
        if isinstance(a, QBin) or (isinstance(a, QAtom) and not _simple_atom(a.text)):
            return f"{neg}({inner})"
        return neg + inner
    if isinstance(f, QBin):
        def side(c, left):
            s = _fstr(c, tex)
            inner = c
            while isinstance(inner, QNot):                  # ¬∃x … würde sonst alles rechts davon verschlucken
                inner = inner.arg
            if isinstance(inner, QQuant):
                return f"({s})"
            if isinstance(c, QBin):
                pc, pp = _PREC[c.op], _PREC[f.op]
                mixed = c.op != f.op and {c.op, f.op} == {"and", "or"}
                if pc < pp or mixed or (pc == pp and (f.op == "iff" or (f.op == "imp" and left))):
                    return f"({s})"
            return s
        sym = (_TEXSYM if tex else _UNISYM)[f.op]
        return side(f.a, True) + sym + side(f.b, False)
    if isinstance(f, QQuant):
        if tex:
            sym = r"\forall " if f.kind == "forall" else r"\exists "
            head = sym + f.var + (r" \in " + f.universe if f.universe else "") + r":\; "
        else:
            head = _SYM[f.kind] + f.var + (f" ∈ {f.universe}" if f.universe else "") + ": "
        body = _fstr(f.body, tex)
        if isinstance(f.body, QBin):
            body = f"({body})"
        return head + body
    raise TypeError(f"Unbekannter Formelknoten {f!r}")


# -- Lesen ------------------------------------------------------------------------

_SPECIAL = re.compile(r"""
    (?P<IFF><->|<=>|↔|⇔)
   |(?P<IMP>->|=>|→|⇒)
   |(?P<AND>∧|&&|&|\band\b|\bund\b)
   |(?P<OR>∨|\|\||\||\boder\b|\bor\b)
   |(?P<NOT>¬|~|!(?!=)|\bnot\b(?!\s+in\b)|\bnicht\b)
   |(?P<Q>∀|∃|\bforall\b|\bexists\b)
   |(?P<LP>\()
   |(?P<RP>\))
""", re.X)

_HEAD = re.compile(r"""\s*(?P<vars>[^\W\d]\w*(?:\s*,\s*[^\W\d]\w*)*)
    (?:\s*(?:∈|\bin\b)\s*(?P<u>\{[^}]*\}|range\([^)]*\)|\[[^\]]*\]|∅|[^\W\d]\w*))?
    \s*[:.]?""", re.X)

_CONNECTIVES = {"IFF", "IMP", "AND", "OR", "NOT", "Q"}


def _tokenize(text: str):
    toks = []
    pos = 0
    for m in _SPECIAL.finditer(text):
        if m.start() < pos:
            continue
        if text[pos:m.start()].strip():
            toks.append(("TEXT", pos, m.start(), None))
        kind = m.lastgroup
        if kind == "Q":
            h = _HEAD.match(text, m.end())
            if not h:
                raise ValueError(f"Nach dem Quantor {m.group(0)} fehlt eine Variable (Position {m.end()}).")
            qk = "forall" if m.group(0) in ("∀", "forall") else "exists"
            names = re.split(r"\s*,\s*", h.group("vars").strip())
            toks.append(("Q", m.start(), h.end(), (qk, names, h.group("u"))))
            pos = h.end()
        else:
            toks.append((kind, m.start(), m.end(), None))
            pos = m.end()
    if text[pos:].strip():
        toks.append(("TEXT", pos, len(text), None))
    return toks


def _outer_wrapped(t: str) -> bool:
    if not (t.startswith("(") and t.endswith(")")):
        return False
    depth = 0
    for i, ch in enumerate(t):
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0 and i < len(t) - 1:
            return False
    return True


def _atom_display(t: str) -> str:
    t = " ".join(t.split())
    while _outer_wrapped(t):
        t = t[1:-1].strip()
    return _map_outside_strings(t, lambda s: re.sub(r",(?=\S)", ", ", s).replace("<=", "≤").replace(">=", "≥")
                                .replace("!=", "≠").replace("==", "="))


def _mask(text: str):
    """Pro Zeichen: Klammertiefe und 'in Zeichenkette'."""
    depth, quote, out = 0, "", []
    for ch in text:
        if quote:
            out.append((depth, True))
            if ch == quote:
                quote = ""
            continue
        if ch in "'\"":
            quote = ch
            out.append((depth, True))
            continue
        if ch in "([{":
            out.append((depth, False))
            depth += 1
            continue
        if ch in ")]}":
            depth -= 1
        out.append((depth, False))
    return out


_CMP_RE = re.compile(r"≤|≥|≠|∉|∈|<|>|=|\bnot\s+in\b|\bin\b")
_FLIP = {"<": "≥", ">": "≤", "≤": ">", "≥": "<", "=": "≠", "≠": "=", "∈": "∉", "∉": "∈",
         "in": "∉", "not in": "∈"}


def _negate_atom(text: str):
    """Vergleich ohne Negationszeichen umkehren (x < y → x ≥ y); None, wenn kein einzelner Vergleich."""
    mask = _mask(text)
    ops = [m for m in _CMP_RE.finditer(text) if mask[m.start()] == (0, False)]
    if len(ops) != 1:
        return None
    m = ops[0]
    op = re.sub(r"\s+", " ", m.group(0))
    left, right = text[:m.start()].rstrip(), text[m.end():].lstrip()
    if not left or not right:
        return None
    return f"{left} {_FLIP[op]} {right}"


class _FParser:
    def __init__(self, text: str):
        self.text = text
        self.toks = _tokenize(text)
        self.i = 0

    def peek(self):
        return self.toks[self.i][0] if self.i < len(self.toks) else None

    def fail(self, msg):
        raise ValueError(f"Formel nicht lesbar: {msg} (in '{self.text.strip()}').")

    def parse(self) -> QFormula:
        if not self.toks:
            self.fail("die Formel ist leer")
        f = self.iff()
        if self.i < len(self.toks):
            self.fail("unerwartetes Zeichen ')' oder Rest nach der Formel")
        return f

    def iff(self):
        left = self.imp()
        while self.peek() == "IFF":
            self.i += 1
            left = QBin("iff", left, self.imp())
        return left

    def imp(self):
        left = self.or_()
        if self.peek() == "IMP":
            self.i += 1
            return QBin("imp", left, self.imp())
        return left

    def or_(self):
        left = self.and_()
        while self.peek() == "OR":
            self.i += 1
            left = QBin("or", left, self.and_())
        return left

    def and_(self):
        left = self.not_()
        while self.peek() == "AND":
            self.i += 1
            left = QBin("and", left, self.not_())
        return left

    def not_(self):
        if self.peek() == "NOT":
            self.i += 1
            return QNot(self.not_())
        return self.primary()

    def match_paren(self, i):
        depth = 0
        for k in range(i, len(self.toks)):
            if self.toks[k][0] == "LP":
                depth += 1
            elif self.toks[k][0] == "RP":
                depth -= 1
                if depth == 0:
                    return k
        self.fail("Klammer nicht geschlossen")

    def primary(self):
        kind = self.peek()
        if kind is None:
            self.fail("die Formel ist unvollständig")
        if kind == "Q":
            _, _, _, (qk, names, uni) = self.toks[self.i]
            self.i += 1
            body = self.iff()
            for v in reversed(names):
                body = QQuant(qk, v, uni, body)
            return body
        if kind == "LP":
            k = self.match_paren(self.i)
            depth = 0
            group = False
            for t in self.toks[self.i + 1:k]:
                depth += t[0] == "LP"
                depth -= t[0] == "RP"
                if depth == 0 and t[0] in _CONNECTIVES:
                    group = True
                    break
            if group:
                self.i += 1
                f = self.iff()
                if self.peek() != "RP":
                    self.fail("')' erwartet")
                self.i += 1
                return f
        return self.atom()

    def atom(self):
        start = self.i
        depth = 0
        while self.i < len(self.toks):
            k = self.toks[self.i][0]
            if k == "LP":
                depth += 1
            elif k == "RP":
                if depth == 0:
                    break
                depth -= 1
            elif k in _CONNECTIVES and depth == 0:
                break
            self.i += 1
        if self.i == start or depth != 0:
            self.fail("Teilaussage fehlt oder Klammer nicht geschlossen")
        a, b = self.toks[start][1], self.toks[self.i - 1][2]
        text = _atom_display(self.text[a:b])
        if not text:
            self.fail("leere Teilaussage")
        return QAtom(text)


def parse_formula(text) -> QFormula:
    """
    Formel mit Quantoren aus Text lesen (kein eval).

        parse_formula("∀x ∈ N: ∃y ∈ N: x > y")
        parse_formula("forall x in U: exists y in U: mag(x, y)")
        parse_formula("¬∃y ∈ R: ∀x ∈ R: x < y")
        parse_formula("∀x P(x) -> ∃y Q(x, y)")        # Quantor reicht so weit nach rechts wie möglich

    Verknüpfungen: ¬ ~ ! not · ∧ & and · ∨ | or · ⇒ → -> => · ⇔ ↔ <-> <=>; Vorrang ¬ > ∧ > ∨ > ⇒ > ⇔.
    Quantor: ∀/forall bzw. ∃/exists, Variable(n) (auch x, y), optional ∈ Universum (Name, {…}, range(…)),
    optional ':'. Teilaussagen (Atome) sind beliebiger Text wie P(x, y) oder a[i] = a[j].
    Grenze: Verknüpfungszeichen in Zeichenketten und '|' als "teilt" werden als Verknüpfung gelesen.
    """
    if isinstance(text, QFormula):
        return text
    if not isinstance(text, str):
        raise ValueError("Die Formel muss ein Text sein.")
    if len(text) > 2000:
        raise ValueError("Die Formel ist zu lang (höchstens 2000 Zeichen).")
    return _FParser(text).parse()


# -- Negation nach innen schieben ------------------------------------------------

def _rw(f: QFormula, simp: bool):
    """Ein Umformungsschritt am ersten passenden Ort: (neue Formel, Regel, Gleichung) oder None."""
    if isinstance(f, QNot):
        a = f.arg
        if isinstance(a, QNot):
            return a.arg, "Doppelte Negation", "¬¬A ≡ A"
        if isinstance(a, QQuant):
            if a.kind == "forall":
                return (QQuant("exists", a.var, a.universe, QNot(a.body)),
                        "Negation des Allquantors", "¬∀x P(x) ≡ ∃x ¬P(x)")
            return (QQuant("forall", a.var, a.universe, QNot(a.body)),
                    "Negation des Existenzquantors", "¬∃x P(x) ≡ ∀x ¬P(x)")
        if isinstance(a, QBin):
            if a.op == "and":
                return QBin("or", QNot(a.a), QNot(a.b)), "De Morgan", "¬(A ∧ B) ≡ ¬A ∨ ¬B"
            if a.op == "or":
                return QBin("and", QNot(a.a), QNot(a.b)), "De Morgan", "¬(A ∨ B) ≡ ¬A ∧ ¬B"
            if a.op == "imp":
                return QBin("and", a.a, QNot(a.b)), "Negation der Implikation", "¬(A ⇒ B) ≡ A ∧ ¬B"
            return QBin("iff", a.a, QNot(a.b)), "Negation der Äquivalenz", "¬(A ⇔ B) ≡ A ⇔ ¬B"
        if isinstance(a, QAtom) and simp:
            flipped = _negate_atom(a.text)
            if flipped:
                return QAtom(flipped), "Negation des Vergleichs", f"¬({a.text}) ≡ {flipped}"
        return None
    if isinstance(f, QBin):
        r = _rw(f.a, simp)
        if r:
            return QBin(f.op, r[0], f.b), r[1], r[2]
        r = _rw(f.b, simp)
        if r:
            return QBin(f.op, f.a, r[0]), r[1], r[2]
        return None
    if isinstance(f, QQuant):
        r = _rw(f.body, simp)
        if r:
            return QQuant(f.kind, f.var, f.universe, r[0]), r[1], r[2]
    return None


def _has_quant(f) -> bool:
    if isinstance(f, QQuant):
        return True
    if isinstance(f, QNot):
        return _has_quant(f.arg)
    if isinstance(f, QBin):
        return _has_quant(f.a) or _has_quant(f.b)
    return False


def _resolve_q_universe(q: QQuant, universes) -> Universe:
    universes = universes or {}
    u = q.universe
    if u is None:
        if q.var in universes:
            return as_universe(universes[q.var])
        raise ValueError(f"Kein Universum für '{q.var}': schreibe '∀{q.var} ∈ U' oder übergib "
                         f"universes={{'{q.var}': …}}.")
    if u in universes:
        v = universes[u]
        if isinstance(v, Universe):
            return v if v.name else Universe(v.items, u, v.finite, v.note)
        return as_universe(v, name=u)
    if u in _ALIASES:
        return named_universe(u)
    if u in ("∅", "{}"):
        return Universe((), "∅")
    if u[0] in "{[" or u.startswith("range"):
        return as_universe(safe_eval(u))
    raise ValueError(f"Unbekanntes Universum '{u}': übergib universes={{'{u}': […]}} "
                     "(oder benutze N, Z, R, Q, {…}, range(…)).")


def _body_pyexpr(f: QFormula) -> str:
    """Quantorenfreie Formel als Python-Ausdruck (Atome in Klammern)."""
    if isinstance(f, QAtom):
        return f"({f.text})"
    if isinstance(f, QNot):
        return f"(not {_body_pyexpr(f.arg)})"
    if isinstance(f, QBin):
        a, b = _body_pyexpr(f.a), _body_pyexpr(f.b)
        if f.op == "and":
            return f"({a} and {b})"
        if f.op == "or":
            return f"({a} or {b})"
        if f.op == "imp":
            return f"((not {a}) or {b})"
        return f"iff({a}, {b})"
    raise ValueError("Quantor innerhalb der Verknüpfung: nicht in Präfixform.")


def _subst_atoms(f: QFormula, values: dict) -> QFormula:
    if not values:
        return f
    if isinstance(f, QAtom):
        t = f.text
        for k, v in values.items():
            t = _map_outside_strings(t, lambda s, k=k, v=v: re.sub(rf"(?<![\w.]){re.escape(k)}(?!\w)",
                                                                  lambda m: _vs(v), s))
        return QAtom(t)
    if isinstance(f, QNot):
        return QNot(_subst_atoms(f.arg, values))
    if isinstance(f, QBin):
        return QBin(f.op, _subst_atoms(f.a, values), _subst_atoms(f.b, values))
    return f


class FormulaNegation(Result):
    """
    Negation einer Quantoren-Formel, Schritt für Schritt nach innen geschoben (Aufgabe 6).

    `original` (wie eingegeben), `start` (die umgeformte Formel, bei `negate_formula` mit ¬ davor),
    `result` (¬ steht nur noch bei Teilaussagen), `trail` (Liste (Regel, Gleichung, Formel danach)),
    `checked` (True, wenn eine Gegenprobe gelaufen ist), `check_equal`.
    """

    def __init__(self, original: QFormula, start: QFormula, result: QFormula, trail, check=None,
                 negation=True, check_equal=None):
        super().__init__()
        self.original = original
        self.start = start
        self.result = result
        self.trail = trail
        self.checked = check is not None
        self.check_equal = check_equal
        if negation:
            self.title = f"Negation von {original}"
            self.summary = f"¬({original}) ≡ {result}"
            self.steps.append(Step("Ausgangsaussage", items=[("A", str(original))]))
            self.steps.append(Step("Negation davorsetzen", "Das Gegenteil von A: die ganze Aussage wird verneint.",
                                   items=[("¬A", str(start))]))
        else:
            self.title = f"Negation nach innen schieben: {original}"
            self.summary = f"{original} ≡ {result}"
            self.steps.append(Step("Ausgangsformel", "Die Negation soll hinter alle Quantoren.",
                                   items=[("Formel", str(original))]))
        cur = str(start)
        for n, (rule, eq, after) in enumerate(trail, 1):
            self.steps.append(Step(f"Umformung {n}", eq, items=[("vorher", cur), ("nachher", str(after))],
                                   rule=rule))
            cur = str(after)
        self.steps.append(Step("Ergebnis", "Die Negation steht jetzt nur noch direkt vor den Teilaussagen "
                               "(oder ist in den Vergleich eingearbeitet).",
                               items=[("Ergebnis", str(result))]))
        if check is not None:
            self.steps.append(check)
        self.highlights = [("Negation" if negation else "Ergebnis", str(result))]


def _push(start: QFormula, simplify: bool):
    cur, trail = start, []
    for _ in range(300):
        r = _rw(cur, simplify)
        if r is None:
            return cur, trail
        cur = r[0]
        trail.append((r[1], r[2], cur))
    raise ValueError("Die Umformung endet nicht (Formel zu verschachtelt).")        # pragma: no cover


def _negation_run(original, start, simplify, universes, predicates, env, negation):
    cur, trail = _push(start, simplify)
    check, equal, warn = None, None, False
    if universes is not None or env or predicates:
        try:
            v0 = evaluate_formula(start, universes, predicates, **env)
            v1 = evaluate_formula(cur, universes, predicates, **env)
        except ValueError as e:
            check = Step("Gegenprobe", f"Nicht möglich: {e}")
        else:
            equal = v0.value == v1.value
            warn = not (v0.certain and v1.certain)
            check = Step("Gegenprobe", "Vor und nach der Umformung wird die Formel über dem Universum "
                         "ausgewertet; beide Werte müssen gleich sein."
                         + (" (Nur ein Ausschnitt eines unendlichen Universums.)" if warn else ""),
                         items=[("vor der Umformung", v0.value), ("nach der Umformung", v1.value),
                                ("gleich", equal)], rule="Äquivalenz prüfen")
            if not equal:                                                       # pragma: no cover
                raise AssertionError("Gegenprobe der Negation schlug fehl (interner Fehler).")
    res = FormulaNegation(original, start, cur, trail, check, negation, equal)
    if warn:
        res.warnings.append("Gegenprobe nur in einem endlichen Ausschnitt des Universums.")
    return res


def negate_formula(formula, universes=None, predicates=None, simplify: bool = True, **env) -> FormulaNegation:
    """
    Das *Gegenteil* einer Formel bilden und die Negation hinter alle Quantoren schieben
    (¬∀ ≡ ∃¬, ¬∃ ≡ ∀¬, De Morgan, ¬(A⇒B) ≡ A∧¬B, ¬¬A ≡ A).

        negate_formula("∀x ∈ N: ∃y ∈ N: x > y")             # ∃x ∈ N: ∀y ∈ N: x ≤ y
        negate_formula("∀u ∈ U: ∃a ∈ A: ∀f ∈ U: (Freund(u, f) → Geladen(f, a))")     # Aufgabe 7

    `simplify=False` lässt ¬(x < y) stehen statt x ≥ y zu schreiben. Mit `universes=…`
    (und `predicates=…` bzw. Schlüsselwörtern für feste Werte) werden zur Gegenprobe Original und
    Ergebnis über dem Universum ausgewertet (bei N, Z, R nur in einem Ausschnitt).
    """
    a = parse_formula(formula)
    return _negation_run(a, QNot(a), simplify, universes, predicates, env, True)


def push_negation(formula, universes=None, predicates=None, simplify: bool = True, **env) -> FormulaNegation:
    """
    Aufgabe 6: «Bringen Sie die Negation hinter alle Quantoren» — die gegebene Formel enthält
    schon ein ¬ (z. B. «¬∃y ∈ R: ∀x ∈ R: x < y»); es wird nach innen geschoben.

        push_negation("¬∃y ∈ R: ∀x ∈ R: x < y")               # ∀y ∈ R: ∃x ∈ R: x ≥ y
        push_negation("¬∀x ∈ Z: ∃y ∈ Z: x = y + 1")           # ∃x ∈ Z: ∀y ∈ Z: x ≠ y + 1
        push_negation("¬∃x ∈ U: ∀y ∈ U: ∃z ∈ U: P(x, y, z)")  # ∀x ∈ U: ∃y ∈ U: ∀z ∈ U: ¬P(x, y, z)
    """
    f = parse_formula(formula)
    return _negation_run(f, f, simplify, universes, predicates, env, False)


# -- Auswerten von Formeln --------------------------------------------------------

class FormulaEvaluation(Result):
    """Wahrheitswert einer Formel mit beliebig geschachtelten Quantoren und Verknüpfungen."""

    def __init__(self, formula: QFormula, universes, predicates, env):
        super().__init__()
        self.formula = formula
        self._preds = dict(predicates or {})
        self._cache: dict = {}
        self._count = 0
        self.value, self.certain = self._ev(formula, dict(env), universes)
        self.title = f"Auswertung von {formula}"
        self.summary = f"Die Aussage ist {'wahr' if self.value else 'falsch'}" \
                       + ("" if self.certain else " (im geprüften Ausschnitt)") + "."
        self.steps.append(Step("Formel lesen", items=[("Formel", str(formula)), ("in Worten", describe_formula(formula))],
                               rule="Definition 2"))
        self.steps.append(Step("Von innen nach aussen auswerten",
                               f"{self._count} Auswertung(en) von Teilaussagen; Quantoren laufen über ihr Universum."))
        self.steps.append(Step("Schluss", items=[("Wahrheitswert", self.value)]))
        if not self.certain:
            self.warnings.append("Nur ein endlicher Ausschnitt eines unendlichen Universums wurde geprüft; "
                                 "'nichts gefunden' beweist dort nichts.")
        self.highlights = [("Wahrheitswert", self.value)]

    def __bool__(self):
        return self.value

    def _atom(self, text, env):
        tree = self._cache.get(text)
        if tree is None:
            tree = self._cache[text] = _parse_expr(text, self._preds.keys())
        self._count += 1
        if self._count > MAX_COMBINATIONS:
            raise ValueError(f"Zu viele Auswertungen (mehr als {MAX_COMBINATIONS}). Verkleinere die Universen.")
        return _as_bool(_Eval(env, self._preds).run(tree), text)

    def _ev(self, f, env, universes):
        if isinstance(f, QAtom):
            return self._atom(f.text, env), True
        if isinstance(f, QNot):
            v, c = self._ev(f.arg, env, universes)
            return (not v), c
        if isinstance(f, QBin):
            va, ca = self._ev(f.a, env, universes)
            if f.op == "and" and not va and ca:
                return False, True
            if f.op == "or" and va and ca:
                return True, True
            if f.op == "imp" and not va and ca:
                return True, True
            vb, cb = self._ev(f.b, env, universes)
            if f.op == "and":
                return va and vb, (ca and cb) or (not vb and cb)
            if f.op == "or":
                return va or vb, (ca and cb) or (vb and cb)
            if f.op == "imp":
                return (not va) or vb, (ca and cb) or (vb and cb)
            return va == vb, ca and cb
        U = _resolve_q_universe(f, universes)
        want = f.kind == "exists"
        any_hit = False
        hit_certain = False
        all_certain = U.finite
        for item in U.items:
            v, c = self._ev(f.body, {**env, f.var: item}, universes)
            all_certain = all_certain and c
            if v == want:
                any_hit = True
                if c:
                    return want, True
        if any_hit:                                                     # nur unsichere Treffer
            return want, False
        return (not want), all_certain


def evaluate_formula(formula, universes=None, predicates=None, **env):
    """
    Wahrheitswert einer Quantoren-Formel (Text oder `parse_formula`-Baum).

        evaluate_formula("∀x ∈ U: ∃y ∈ U: mag(x, y)", {"U": people}, {"mag": mag})
        evaluate_formula("∃i ∈ I ∃j ∈ I: i != j and a[i] = a[j]", {"I": index_set(9)}, a=one_based(a))
        evaluate_formula("∃x ∈ R: x*x < x")        # ℝ: exakter Bruchausschnitt, Zeuge x = 1/2

    `universes`: Namen → Mengen/Listen (oder `Universe`); N, Z, R, Q sind als endliche Ausschnitte
    vordefiniert. `predicates`: Namen → `Predicate` oder Funktion für Aufrufe wie mag(x, y).
    Schlüsselwörter: feste Werte (Listen …). Formeln der Form ∀…∃… vor einer quantorenfreien
    Aussage liefern ein `QuantResult` (mit Tabelle, Zeuge, Gegenbeispiel), alle anderen ein
    `FormulaEvaluation`.
    """
    f = parse_formula(formula)
    preds = dict(predicates or {})
    for k, v in preds.items():
        if not callable(v):
            raise ValueError(f"predicates['{k}'] muss ein Predicate oder eine Funktion sein.")
    chain, body = [], f
    while isinstance(body, QQuant):
        chain.append(body)
        body = body.body
    if chain and not _has_quant(body):
        qs = [(q.kind, q.var, _resolve_q_universe(q, universes)) for q in chain]
        expr = _body_pyexpr(body)
        names = [v for _, v, _ in qs]
        # auch Quantorvariablen, die im Rumpf gar nicht vorkommen, sind Lücken (∀x ∃y S(x) ist erlaubt)
        params = names + [n for n in free_names(expr, preds.keys()) if n not in names]
        pred = Predicate("", params, expr=expr, funcs=preds,
                         label_fn=lambda vals, b=body: _subst_atoms(b, vals).to_str())
        fixed = {k: v for k, v in env.items() if k in pred.params and k not in names}
        res = _check_fixed_and_build(qs, pred, fixed)
        return res
    return FormulaEvaluation(f, universes, preds, env)


# -- Negation einer quantifizierten Aussage mit Gegenprobe ---------------------------

class NegationResult(Result):
    """¬(Q… P) ≡ Q'… ¬P an einem konkreten Prädikat, mit Gegenprobe über das Universum."""

    def __init__(self, original: QuantResult, negated: QuantResult):
        super().__init__()
        self.original = original
        self.negated = negated
        self.value_of_negation = not original.value
        self.consistent = self.value_of_negation == negated.value
        self.formula_negated = negated.formula
        self.title = f"Negation von {original.formula}"
        self.summary = (f"¬({original.formula}) ≡ {negated.formula}; beide Seiten sind "
                        f"{'wahr' if negated.value else 'falsch'}.")
        self.steps.append(Step("Ausgangsaussage", items=[("A", original.formula),
                                                         ("Wahrheitswert von A", original.value)]))
        self.steps.append(Step("Negation davorsetzen", "Das Gegenteil von A.",
                               items=[("¬A", f"¬({original.formula})"), ("Wahrheitswert von ¬A", self.value_of_negation)]))
        rules = []
        for k, v, _ in original.quantifiers:
            rules.append(f"¬∀{v} P ≡ ∃{v} ¬P" if k == "forall" else f"¬∃{v} P ≡ ∀{v} ¬P")
        self.steps.append(Step(
            "Negation hinter die Quantoren ziehen",
            "Jeder Quantor wechselt (∀ ↔ ∃), das Prädikat wird verneint: " + "; ".join(rules) + ".",
            items=[("Ergebnis", negated.formula)], rule="Negation von Quantoren"))
        tstep = Step("Gegenprobe: umgeformte Aussage auswerten",
                     "Die umgeformte Aussage wird über dasselbe Universum ausgewertet.",
                     items=[("Wahrheitswert", negated.value)], table=negated.table)
        self.steps.append(tstep)
        link = ""
        if original.counterexample and negated.witness:
            link = (f" Das Gegenbeispiel von A ({_bind_text(original.counterexample)}) ist der Zeuge der "
                    "umgeformten Aussage.")
        elif original.witness and negated.counterexample:
            link = (f" Der Zeuge von A ({_bind_text(original.witness)}) ist das Gegenbeispiel der "
                    "umgeformten Aussage.")
        self.steps.append(Step("Vergleich",
                               "¬A und die umgeformte Aussage haben denselben Wahrheitswert." + link,
                               items=[("¬A", self.value_of_negation), ("umgeformt", negated.value),
                                      ("gleich", self.consistent)], rule="Äquivalenz"))
        if not (original.certain and negated.certain):
            self.warnings.append("Nur ein endlicher Ausschnitt eines unendlichen Universums wurde geprüft.")
        self.highlights = [("Negation", negated.formula), ("Wahrheitswert", negated.value)]


def negate_quantifier(spec, pred, universes=None, **fixed) -> NegationResult:
    """
    ¬∀x P(x) ≡ ∃x ¬P(x) und ¬∃x P(x) ≡ ∀x ¬P(x) an einem konkreten Prädikat — mit Gegenprobe.

        negate_quantifier("∀x", "x % 2 == 0", universes={"x": range(10)})
        negate_quantifier("∃x ∀y", pred, universes={"x": U, "y": V})

    Die Folge `spec` wird wie bei `quantify` angegeben; jeder Quantor wird gewechselt und das
    Prädikat verneint. Es wird bewiesen (durch Auswerten), dass beide Seiten gleich sind.
    """
    qs = _parse_prefix(spec, universes)
    p = _as_predicate(pred)
    original = _check_fixed_and_build(qs, p, fixed)
    flipped = [("exists" if k == "forall" else "forall", v, u) for k, v, u in qs]
    negated = _check_fixed_and_build(flipped, ~p, fixed)
    res = NegationResult(original, negated)
    if not res.consistent:                                              # pragma: no cover
        raise AssertionError("Negationsregel verletzt (interner Fehler).")
    return res


# -- Reihenfolge der Quantoren (Aufgabe 4) --------------------------------------------

class OrderResult(Result):
    """
    Zwei Quantoren vertauschen: gleiche Quantoren ja, verschiedene nein (Aufgabe 4).

    `original`, `swapped` (je ein `QuantResult`), `same_kind`, `differ` (Wahrheitswerte
    verschieden), `stronger` ('original'|'swapped'|None: die Aussage mit ∃ vorne, die die
    andere impliziert).
    """

    def __init__(self, original: QuantResult, swapped: QuantResult, readings=None):
        super().__init__()
        self.original, self.swapped = original, swapped
        k1, k2 = original.quantifiers[0][0], original.quantifiers[1][0]
        self.same_kind = k1 == k2
        self.differ = original.value != swapped.value
        self.stronger = None
        if not self.same_kind:
            self.stronger = "original" if k1 == "exists" else "swapped"
        self.title = f"Reihenfolge der Quantoren: {original.formula} gegen {swapped.formula}"
        self.summary = (f"{'Gleiche' if self.same_kind else 'Verschiedene'} Quantoren: die beiden Aussagen sind "
                        + ("gleichwertig." if not self.differ else "verschieden (ein Wahrheitswert wechselt)."))
        its = [("Aussage 1", original.formula), ("in Worten", original.meaning)]
        if readings:
            its.append(("Lesart", readings[0]))
        its += [("Aussage 2", swapped.formula), ("in Worten", swapped.meaning)]
        if readings:
            its.append(("Lesart", readings[1]))
        self.steps.append(Step("Beide Aussagen aufschreiben", "Dieselben Quantoren in verschiedener Reihenfolge.",
                               items=its))
        self.steps.append(Step("Auswerten", items=[("Aussage 1", original.value), ("Aussage 2", swapped.value)]))
        for res, label in ((original, "Aussage 1"), (swapped, "Aussage 2")):
            ev = res._ev
            if ev:
                kind = "Zeuge" if res.witness else "Gegenbeispiel"
                self.steps.append(Step(f"{label}: {kind}", _bind_text(ev)))
        if self.same_kind:
            self.steps.append(Step(
                "Regel", "Gleiche Quantoren dürfen vertauscht werden: ∀x∀y P ≡ ∀y∀x P und ∃x∃y P ≡ ∃y∃x P.",
                items=[("gleicher Wahrheitswert", not self.differ)], rule="Vertauschen gleicher Quantoren"))
        else:
            strong = original if self.stronger == "original" else swapped
            weak = swapped if self.stronger == "original" else original
            txt = (f"Verschiedene Quantoren dürfen nicht vertauscht werden. Immer gilt: {strong.formula} ⇒ "
                   f"{weak.formula} (ein Zeuge für alle ist insbesondere für jeden da). Die Umkehrung gilt im "
                   "Allgemeinen nicht.")
            if strong.value is False and weak.value is True:
                txt += " Hier ist das der Fall: die schwächere Aussage ist wahr, die stärkere falsch."
            self.steps.append(Step("Regel", txt, items=[("stärkere ⇒ schwächere", (not strong.value) or weak.value)],
                                   rule="Verschiedene Quantoren nicht vertauschbar"))
        if not (original.certain and swapped.certain):
            self.warnings.append("Nur ein endlicher Ausschnitt eines unendlichen Universums wurde geprüft.")
        self.highlights = [("Aussage 1", original.value), ("Aussage 2", swapped.value)]


def swap_order(spec, pred, universes=None, **fixed) -> OrderResult:
    """
    Vertauscht die Reihenfolge von zwei Quantoren und vergleicht beide Aussagen.

        swap_order("∀x ∃y", mag, universes={"x": people, "y": people})    # ∀x∃y gegen ∃y∀x
        swap_order("∀x ∀y", pred, {...})                                  # immer gleich

    Es gilt immer ∃y∀x P ⇒ ∀x∃y P, die Umkehrung nicht ("Everybody loves somebody").
    """
    qs = _parse_prefix(spec, universes)
    if len(qs) != 2:
        raise ValueError("swap_order erwartet genau zwei Quantoren, z. B. '∀x ∃y'.")
    p = _as_predicate(pred)
    first = _check_fixed_and_build(qs, p, fixed)
    second = _check_fixed_and_build([qs[1], qs[0]], p, fixed)
    res = OrderResult(first, second)
    if res.stronger and not res.stronger == "":                         # Kontrolle der Regel
        strong = first if res.stronger == "original" else second
        weak = second if res.stronger == "original" else first
        if strong.value and not weak.value:                             # pragma: no cover
            raise AssertionError("∃∀ ⇒ ∀∃ verletzt (interner Fehler).")
    return res


def everybody_loves_somebody(people=None, loves=None, verb: str = "liebt") -> OrderResult:
    """
    Aufgabe 4 («Everybody Loves Somebody») an einem Beispiel.

    ∀x∃y liebt(x, y): jeder liebt irgendjemanden.  ∃y∀x liebt(x, y): es gibt jemanden, den alle lieben.
    Ohne Angaben: Anna → Ben → Cleo → Anna und Dora → Ben (erste Aussage wahr, zweite falsch).

        everybody_loves_somebody()
        everybody_loves_somebody(["A", "B"], {("A", "B"), ("B", "B")})   # beide wahr: B wird von allen geliebt
    """
    if people is None:
        people = ["Anna", "Ben", "Cleo", "Dora"]
        loves = {("Anna", "Ben"), ("Ben", "Cleo"), ("Cleo", "Anna"), ("Dora", "Ben")}
    elif loves is None:
        raise ValueError("Gib auch die Beziehungen an: loves={(x, y), …} oder {x: {y, …}}.")
    rel = relation_predicate(loves, "loves")
    U = as_universe(people, name="U")
    qs = [("forall", "x", U), ("exists", "y", U)]
    first = _check_fixed_and_build(qs, rel, {})
    second = _check_fixed_and_build([qs[1], qs[0]], rel, {})
    res = OrderResult(first, second, readings=(f"Jeder {verb} irgendjemanden (Everybody loves somebody)",
                                              f"Es gibt eine Person, die von allen geliebt wird"))
    return res


# ==========================================================================
# In Worten lesen und aus der deutschen Standardform übersetzen
# ==========================================================================

def _fwords(f: QFormula) -> str:
    if isinstance(f, QAtom):
        return f.text
    if isinstance(f, QNot):
        a = f.arg
        if isinstance(a, (QQuant, QBin)):
            return f"es ist nicht der Fall, dass {_fwords(a)}"
        return f"nicht {_fwords(a)}"
    if isinstance(f, QBin):
        wa, wb = _fwords(f.a), _fwords(f.b)
        if f.op == "and":
            return f"{wa} und {wb}"
        if f.op == "or":
            return f"{wa} oder {wb}"
        if f.op == "imp":
            return f"wenn {wa}, dann {wb}"
        return f"{wa} genau dann, wenn {wb}"
    where = f" ∈ {f.universe}" if f.universe else ""
    rest = _fwords(f.body)
    if f.kind == "forall":
        return f"für alle {f.var}{where} gilt: {rest}"
    return f"es gibt ein {f.var}{where}, sodass {rest}"


def describe_formula(formula) -> str:
    """
    Quantoren-Formel in Worten (Deutung).

        describe_formula("∀x ∈ U: ∃y ∈ U: mag(x, y)")
        # 'Für alle x ∈ U gilt: es gibt ein y ∈ U, sodass mag(x, y).'
    """
    return _cap(_fwords(parse_formula(formula))) + "."


_UNI_PAT = r"(?P<u>\{[^}]*\}|range\([^)]*\)|[^\s,:;()]+)"
_VAR_PAT = r"(?P<v>[^\W\d]\w*)"
_IN_PAT = r"(?:\s*∈\s*|\s+(?:aus|in)\s+)"
_NOUN_PAT = r"(?:[\w-]+\s+)??"
_ALL_RE = re.compile(
    rf"^\s*(?P<neg>nicht\s+)?f(?:ü|ue)r\s+(?:(?P<none>kein(?:e|en|em|er)?)|alle|jede[nrms]?|jedes)\s+"
    rf"{_NOUN_PAT}{_VAR_PAT}{_IN_PAT}{_UNI_PAT}\s*(?:,\s*|:\s*|\s+)?(?:gilt\s*:?\s*)?", re.I)
_EX_RE = re.compile(
    rf"^\s*(?:es\s+(?:gibt|existiert)|gibt\s+es|existiert)\s+(?:(?P<none>kein(?:e|en|em|er)?)|"
    rf"ein(?:e|en|em|er)?|mindestens\s+ein(?:e|en|em|er)?)\s+{_NOUN_PAT}{_VAR_PAT}{_IN_PAT}{_UNI_PAT}"
    rf"\s*,?\s*(?:so\s*,?\s*dass|sodass|derart,?\s+dass|f(?:ü|ue)r\s+(?:das|den|die|welches|welchen|welche)|"
    rf"mit|bei\s+(?:dem|der)|wobei|:)?\s*(?:gilt\s*:?\s*)?", re.I)


class TranslationResult(Result):
    """Deutsche Standardform → Formel: `formula` (Text), `tree` (QFormula), `reading`, `negation()`."""

    def __init__(self, sentence, parts, tree):
        super().__init__()
        self.sentence = sentence
        self.tree = tree
        self.formula = str(tree)
        self.reading = describe_formula(tree)
        self.title = "Satz in Prädikatenlogik übersetzen"
        self.summary = f"{self.formula}"
        for n, (phrase, sym, rule) in enumerate(parts, 1):
            self.steps.append(Step(f"Satzteil {n}", f"«{phrase.strip()}»", items=[("Quantor", sym)], rule=rule))
        self.steps.append(Step("Formel", "Die Quantoren stehen vorn, dahinter die Aussage über den Variablen.",
                               items=[("Formel", self.formula), ("in Worten", self.reading)]))
        self.highlights = [("Formel", self.formula)]

    def negation(self, **kw) -> FormulaNegation:
        """Das Gegenteil dieser Aussage mit Rechenweg (siehe `negate_formula`)."""
        return negate_formula(self.tree, **kw)


def to_formula(sentence: str) -> TranslationResult:
    """
    Zeigt die Formel zu einem deutschen Satz in Standardform (reine Darstellung, keine Sprachverarbeitung).

        to_formula("Für alle x aus U gilt P(x)")                     # ∀x ∈ U: P(x)
        to_formula("Es gibt ein x aus U, sodass P(x)")               # ∃x ∈ U: P(x)
        to_formula("Für jedes x aus N gibt es ein y aus N, sodass x > y")    # ∀x ∈ N: ∃y ∈ N: x > y
        to_formula("Es gibt kein x aus U mit P(x)")                  # ¬∃x ∈ U: P(x)
        to_formula("Für jeden iPhone-Benutzer u aus U gibt es eine App a aus A, sodass Freund(u, f) → …")

    Erkannt werden die Anfänge «Für alle/jede(n)/jedes … aus/in/∈ …», «Für kein …», «Nicht für alle …»,
    «Es gibt (mindestens) ein …», «Es gibt kein …», danach «, gilt», «sodass», «mit», «für das». Der Rest
    des Satzes ist die Aussage (Verknüpfungen und, oder, nicht, →, ⇔ werden gelesen). Mit
    `.negation()` erhält man das Gegenteil (Aufgabe 7).
    """
    if not isinstance(sentence, str) or not sentence.strip():
        raise ValueError("Der Satz ist leer.")
    if len(sentence) > 1000:
        raise ValueError("Der Satz ist zu lang (höchstens 1000 Zeichen).")
    rest = sentence.strip()
    quants, parts = [], []
    # ponytail: nur feste Satzanfänge (Regex), keine echte Sprachverarbeitung — genügt für die Standardform
    for _ in range(12):
        m = _ALL_RE.match(rest)
        kind = "forall"
        if not m:
            m = _EX_RE.match(rest)
            kind = "exists"
        if not m:
            break
        neg = bool(m.groupdict().get("neg")) or bool(m.groupdict().get("none"))
        if m.groupdict().get("none"):
            kind = "exists"
        quants.append((kind, neg, m.group("v"), m.group("u")))
        sym = ("¬" if neg else "") + f"{_SYM[kind]}{m.group('v')} ∈ {m.group('u')}"
        rule = ("Allquantor (Generalisierung)" if kind == "forall" and not neg else
                "Existenzquantor (Spezialisierung)" if kind == "exists" and not neg else
                "Negation von Quantoren")
        parts.append((rest[:m.end()], sym, rule))
        rest = rest[m.end():]
    if not quants:
        raise ValueError("Satz nicht erkannt. Erwartet z. B. «Für alle x aus U gilt P(x)», «Für jedes x aus U "
                         "gibt es ein y aus V, sodass L(x, y)» oder «Es gibt kein x aus U mit P(x)».")
    body = rest.strip().rstrip(".").strip()
    body = re.sub(r"\s+gilt$", "", body).strip()
    body = re.sub(r"^dass\s+", "", body).strip()
    if not body:
        raise ValueError("Nach den Quantoren fehlt die Aussage (z. B. P(x)).")
    try:
        node = parse_formula(body)
    except ValueError:
        node = QAtom(" ".join(body.split()))
    for kind, neg, var, uni in reversed(quants):
        node = QQuant(kind, var, uni, node)
        if neg:
            node = QNot(node)
    return TranslationResult(sentence, parts, node)


# ==========================================================================
# Aufgabe 1: wahre und falsche Aussagen zu einem Prädikat suchen
# ==========================================================================

class ExamplesResult(Result):
    """Aufgabe 1: `true_examples` und `false_examples` (Listen von Belegungen) zu einem Prädikat."""

    def __init__(self, pred: Predicate, true_ex: list, false_ex: list, n_true: int, n_false: int, truncated: bool):
        super().__init__()
        self.predicate = pred
        self.true_examples, self.false_examples = true_ex, false_ex
        self.title = f"Wahre und falsche Aussagen zu {pred.signature()}"
        self.summary = f"{len(true_ex)} wahre und {len(false_ex)} falsche Aussage(n) gefunden."
        d = pred.definition()
        self.steps.append(Step("Prädikat", "Alle Lücken mit Werten füllen: dann entsteht eine Aussage.",
                               items=[("Prädikat", pred.signature())] + ([("Bedeutung", d)] if d else []),
                               rule="Definition 1"))
        cols = list(pred.gaps or pred.open_params)
        rows, marks = [], {}
        for env, v in [(e, True) for e in true_ex] + [(e, False) for e in false_ex]:
            rows.append([_vs(env[c]) for c in cols] + [pred.label(env), bit(v)])
            marks[len(rows) - 1] = "ok" if v else "bad"
        self.steps.append(Step("Gefundene Aussagen", table=Table(cols + ["Aussage", "Wahrheitswert"], rows, marks)))
        if len(true_ex) < n_true or len(false_ex) < n_false:
            self.warnings.append("Nicht genug Beispiele im durchsuchten Bereich gefunden (Universum vergrössern?).")
        if truncated:
            self.warnings.append(f"Die Suche wurde nach {MAX_COMBINATIONS} Belegungen abgebrochen.")
        self.highlights = [("wahre Aussagen", len(true_ex)), ("falsche Aussagen", len(false_ex))]


def find_examples(pred, universes=None, n_true: int = 3, n_false: int = 3, /, **fixed) -> ExamplesResult:
    """
    Aufgabe 1: Werte finden, so dass drei wahre und drei falsche Aussagen entstehen.

        find_examples(FERMAT, naturals(13, start=1))          # a, b, c, n aus 1…13
        find_examples(SUM_EQUAL_TO_FIVE, {"x": range(5), "y": range(5)}, 2, 2)

    `universes`: ein Universum für alle Lücken oder ein Dictionary je Lücke (Standard: 1…10).
    Die Suche läuft in Tabellenreihenfolge (erste Lücke am langsamsten). Hinweis: Für das
    Fermat-Prädikat gibt es für n ≥ 3 keine wahre Aussage (Satz von Wiles) — wahre Beispiele
    haben n = 1 oder n = 2 (z. B. 3² + 4² = 5²).
    """
    p = _as_predicate(pred)
    cols = [c for c in p.open_params if c not in fixed]
    if not cols:
        raise ValueError("Alle Lücken sind schon fest vorgegeben.")
    if not all(isinstance(n, int) and not isinstance(n, bool) and 0 <= n <= 1000 for n in (n_true, n_false)):
        raise ValueError("n_true und n_false müssen ganze Zahlen von 0 bis 1000 sein.")
    if universes is None:
        unis = {c: naturals(10, start=1) for c in cols}
    elif isinstance(universes, dict):
        missing = [c for c in cols if c not in universes]
        if missing:
            raise ValueError(f"Kein Universum für: {', '.join(missing)}.")
        unis = {c: as_universe(universes[c]) for c in cols}
    else:
        u = as_universe(universes)
        unis = {c: u for c in cols}
    true_ex, false_ex = [], []
    count = 0
    truncated = False
    for combo in itertools.product(*[unis[c].items for c in cols]):
        if len(true_ex) >= n_true and len(false_ex) >= n_false:
            break
        count += 1
        if count > MAX_COMBINATIONS:
            truncated = True
            break
        env = dict(zip(cols, combo))
        v = p._apply({**fixed, **env})
        if v and len(true_ex) < n_true:
            true_ex.append(env)
        elif not v and len(false_ex) < n_false:
            false_ex.append(env)
    return ExamplesResult(p, true_ex, false_ex, n_true, n_false, truncated)


# ==========================================================================
# Aufgabe 3 f: Listen (Positionen ab 1)
# ==========================================================================

_DUP = Predicate("Q(i, j)", ("i", "j", "a"), expr="i != j and a[i] == a[j]")
_DISTINCT = Predicate("D(i, j)", ("i", "j", "a"), expr="implies(i != j, a[i] != a[j])")


class ListQuantResult(QuantResult):
    """
    Quantoren über die Positionen einer Liste (Aufgabe 3 f). Zusätzlich zu `QuantResult`:
    `list` (die Liste), `groups` (Liste (Wert, Positionen) in Reihenfolge des Auftretens),
    `pair` (Zeuge/Gegenbeispiel als Paar (i, j) oder None).
    """

    def _finish(self, seq, first_index: int, pairs):
        self.list = list(seq)
        groups: list = []
        for pos, v in enumerate(self.list, first_index):
            for g in groups:
                if g[0] == v:
                    g[1].append(pos)
                    break
            else:
                groups.append((v, [pos]))
        self.groups = groups
        ev = self.witness or self.counterexample
        self.pair = (ev.get("i"), ev.get("j")) if ev and "j" in ev else None
        steps = [Step(
            "Wo steht welcher Wert?", "Positionen (zählen ab " + str(first_index) + "), an denen derselbe Wert steht.",
            items=[(f"Wert {_vs(v)}", "Positionen " + ", ".join(map(str, ps))) for v, ps in groups[:14]])]
        use = list(pairs) if pairs is not None else ([self.pair] if self.pair and None not in self.pair else [])
        for (i, j) in use:
            env = {"i": i, "j": j, "a": Seq1(self.list) if first_index == 1 else self.list}
            try:
                text, leaves, lookups, val = _DUP._rechenweg(env)
            except ValueError as e:
                steps.append(Step(f"Beispiel i = {i}, j = {j}", f"Nicht möglich: {e}"))
                continue
            its = [("eingesetzt", text)]
            for k, (t, r) in enumerate(leaves, 1):
                its.append((f"Teil {k}", f"{t}  →  " + ("nicht nötig" if r is None else bit(r))))
            its += [("nachgeschlagen", s) for s in lookups]
            its.append((f"Q({i}, {j})", val))
            steps.append(Step(
                f"Beispielrechnung i = {i}, j = {j}",
                "Die beiden Positionen werden in Q(i, j) :⇔ ¬(i = j) ∧ a[i] = a[j] eingesetzt.", items=its,
                note="Wahr heisst: an zwei verschiedenen Positionen steht derselbe Wert."))
        self.steps[1:1] = steps


def _list_input(a, one_based_):
    if isinstance(a, str):
        a = list(a)
    if hasattr(a, "tolist"):
        a = a.tolist()
    if isinstance(a, Seq1):
        a, one_based_ = list(a), True
    if not isinstance(a, (list, tuple)):
        raise ValueError("Erwartet wird eine Liste (oder ein Text).")
    if len(a) > 200:
        raise ValueError("Die Liste ist zu lang (höchstens 200 Elemente bei Paaren i, j).")
    return list(a), one_based_


def _list_quant(spec, pred, a, one_based_, pairs, title):
    lst, ob = _list_input(a, one_based_)
    first = 1 if ob else 0
    I = index_set(len(lst), start=first)
    seq = Seq1(lst) if ob else lst
    qs = _parse_prefix(spec, {"i": I, "j": I})
    res = _check_fixed_and_build(qs, pred, {"a": seq}, cls=ListQuantResult)
    res._finish(lst, first, pairs)
    res.title = title
    return res


def has_duplicate(a, one_based: bool = True, pairs=None) -> ListQuantResult:
    """
    «Kommt ein Element doppelt oder mehrfach in der Liste vor?»  ∃i ∃j: ¬(i = j) ∧ a[i] = a[j]

        has_duplicate([3, 2, 1, 1, 4, 4, 2, 3, 1])                  # wahr, Zeuge i = 1, j = 8
        has_duplicate(a, pairs=[(1, 8), (2, 7)])                    # Rechenbeispiele der Mitschrift

    Positionen zählen ab 1 (`one_based=False`: ab 0). `pairs`: Positionspaare, für die der Rechenweg
    gezeigt wird (Standard: der Zeuge). Höchstens 200 Elemente.
    """
    return _list_quant("∃i ∃j", _DUP, a, one_based, pairs,
                       "Kommt ein Element mehrfach in der Liste vor?")


def all_distinct(a, one_based: bool = True, pairs=None) -> ListQuantResult:
    """
    Sind alle Elemente verschieden?  ∀i ∀j: (i ≠ j ⇒ a[i] ≠ a[j])  — das Gegenteil von `has_duplicate`.

        all_distinct([3, 2, 1, 4])        # wahr
        all_distinct([3, 2, 1, 3])        # falsch, Gegenbeispiel i = 1, j = 4
    """
    return _list_quant("∀i ∀j", _DISTINCT, a, one_based, pairs,
                       "Sind alle Elemente der Liste verschieden?")


def occurs_multiple(a, one_based: bool = True, pairs=None) -> ListQuantResult:
    """
    Aufgabe 3 f wörtlich:  ∀i ∈ {1…n}: ∃j ∈ {1…n}: ¬(i = j) ∧ a[i] = a[j]
    «Jedes Element kommt mindestens zweimal vor.»

        occurs_multiple([3, 2, 1, 1, 4, 4, 2, 3, 1])   # wahr
        occurs_multiple([3, 2, 1, 1])                  # falsch, Gegenbeispiel i = 1 (die 3 steht nur einmal)
    """
    return _list_quant("∀i ∃j", _DUP, a, one_based, pairs,
                       "Kommt jedes Element mindestens zweimal in der Liste vor?")


# ==========================================================================
# Aufgabe 5: Matrix-Prädikat P(M, i, j): m_ij = 1
# ==========================================================================

_MATRIX_STATEMENTS = [
    ("a", "∀i ∀j", "alle Einträge sind {t}"),
    ("b", "∀j ∀i", "alle Einträge sind {t} (andere Reihenfolge, gleiche Aussage)"),
    ("c", "∃i ∃j", "mindestens ein Eintrag ist {t}"),
    ("d", "∃j ∃i", "mindestens ein Eintrag ist {t} (andere Reihenfolge, gleiche Aussage)"),
    ("e", "∀i ∃j", "jede Zeile enthält mindestens eine {t}"),
    ("f", "∀j ∃i", "jede Spalte enthält mindestens eine {t}"),
    ("g", "∃i ∀j", "es gibt eine Zeile nur aus {t}"),
    ("h", "∃j ∀i", "es gibt eine Spalte nur aus {t}"),
]


class MatrixResult(Result):
    """
    Die acht Aussagen der Aufgabe 5 für eine Matrix. `statements`: Schlüssel wie "∀i∃j" →
    `QuantResult`; `values`: Buchstabe a…h → Wahrheitswert; `matrix[i-1][j-1]` ist m_ij.
    """

    def __init__(self, matrix, target, statements: dict):
        super().__init__()
        self.matrix = matrix
        self.target = target
        self.statements = statements
        self.values = {}
        t = _vs(target)
        self.title = f"Matrix-Prädikat P(M, i, j): m_ij = {t}"
        header = ["j = " + str(j) for j in range(1, len(matrix[0]) + 1)]
        self.steps.append(Step(
            "Matrix und Prädikat", "P(M, i, j) ist wahr, wenn der Eintrag in Zeile i und Spalte j gleich "
            f"{t} ist. Zeilen und Spalten zählen ab 1.", rule="Definition 1",
            table=Table([""] + header, [[f"i = {i}"] + [_vs(x) for x in row] for i, row in enumerate(matrix, 1)],
                        {})))
        rows, marks = [], {}
        for n, (letter, key, reading) in enumerate(_MATRIX_STATEMENTS):
            res = statements[key.replace(" ", "")]
            self.values[letter] = res.value
            ev = res._ev
            rows.append([letter, key.replace(" ", ""), reading.format(t=t), bit(res.value),
                         _bind_text(ev) if ev else ""])
            marks[n] = "ok" if res.value else "bad"
        self.steps.append(Step(
            "Acht Aussagen auswerten", "Von innen nach aussen: Zeilen und Spalten prüfen. In der Prüfung genügt "
            "es zu entscheiden, ob die Aussage wahr oder falsch ist.", rule="Verschachtelte Quantoren",
            table=Table(["Aufg.", "Aussage", "Lesart", "Wahrheitswert", "Zeuge/Gegenbeispiel"], rows, marks)))
        self.steps.append(Step(
            "Reihenfolge", "Gleiche Quantoren lassen sich vertauschen (a = b, c = d); verschiedene nicht "
            "(e ≠ f, g ≠ h). Immer gilt: g ⇒ f (eine Zeile voller Einsen gibt jeder Spalte eine 1) und "
            "h ⇒ e.", rule="Vertauschen von Quantoren"))
        self.summary = "Wahr: " + (", ".join(k for k, v in self.values.items() if v) or "keine") + "."
        self.highlights = [(k, v) for k, v in self.values.items()]

    def __getitem__(self, key) -> QuantResult:
        """res['e'] oder res['∀i∃j'] → QuantResult."""
        for letter, k, _ in _MATRIX_STATEMENTS:
            if key == letter or key.replace(" ", "") == k.replace(" ", ""):
                return self.statements[k.replace(" ", "")]
        raise KeyError(key)


def matrix_quantors(M, target=1, one_based: bool = True) -> MatrixResult:
    """
    Aufgabe 5: die acht Aussagen ∀i∀j, ∀j∀i, ∃i∃j, ∃j∃i, ∀i∃j, ∀j∃i, ∃i∀j, ∃j∀i über P(M, i, j): m_ij = 1.

        matrix_quantors([[0, 0, 0], [1, 1, 1], [0, 0, 0]])       # Zeile 2 nur aus Einsen
        matrix_quantors(E).values          # {'a': False, ..., 'g': True, ...}
        matrix_quantors(M)["g"].witness    # {'i': 3}

    `M`: Liste von Zeilen (oder numpy-Feld), `target`: gesuchter Eintrag (Standard 1). Wie die
    Notebook-Funktionen von Serie 3, aber mit Positionen ab 1 (`one_based=False`: ab 0).
    """
    if hasattr(M, "tolist"):
        M = M.tolist()
    if not isinstance(M, (list, tuple)) or not M or not all(isinstance(r, (list, tuple)) for r in M):
        raise ValueError("Die Matrix muss eine nichtleere Liste von Zeilen sein.")
    cols = len(M[0])
    if cols == 0 or any(len(r) != cols for r in M):
        raise ValueError("Alle Zeilen der Matrix müssen gleich lang (und nicht leer) sein.")
    if len(M) > 50 or cols > 50:
        raise ValueError("Die Matrix ist zu gross (höchstens 50 × 50).")
    if isinstance(target, bool) or not isinstance(target, (int, str)):
        raise ValueError("target muss eine ganze Zahl oder ein Text sein.")
    matrix = [list(r) for r in M]
    first = 1 if one_based else 0
    seq = one_based_(matrix) if one_based else matrix
    pred = Predicate("P(i, j)", ("M", "i", "j"), expr=f"M[i][j] == {target!r}")
    unis = {"i": index_set(len(matrix), start=first), "j": index_set(cols, start=first)}
    statements = {}
    for _, key, _r in _MATRIX_STATEMENTS:
        qs = _parse_prefix(key, unis)
        statements[key.replace(" ", "")] = _check_fixed_and_build(qs, pred, {"M": seq})
    return MatrixResult(matrix, target, statements)


one_based_ = one_based      # Hilfsname: in matrix_quantors verdeckt der Parameter `one_based` die Funktion
