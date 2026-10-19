#!/usr/bin/env python3
"""
Analyse von Algorithmen (Semesterwoche 5, Serie 5).

Was hier steht (alles exakt gerechnet, nie ``eval``/``exec`` auf Eingaben):

    count_steps(code, **eingaben)       Rechenschritte zählen nach den Zählregeln der Vorlesung
                                        (eigener, sicherer Interpreter für eine Python-Teilmenge)
    worst_case(algorithmus, n_werte)    Worst-Case: gemessen UND als geschlossene Formel in n
    best_case(algorithmus, n_werte)     Best-Case (kleinste Schrittzahl über alle Eingaben der Länge n)
    average_case(algorithmus, n)        Durchschnitt  E(X) = Σ p(a)·X(a)
    compare_growth({...}, [n-Werte])    Aufgabe 1: Tabelle der Schrittzahlen (n·log2 n, n², n!)
    max_input_size(f, 60, 1e-9)         Aufgabe 1: grösstes n, das in der Frist berechenbar ist
    max_input_sizes({...}, ...)         dasselbe für mehrere Algorithmen auf einmal
    growth_classes()                    die wichtigsten Wachstumsklassen (O(1) ⊂ O(log n) ⊂ …)
    classify(f)                         Θ-Klasse einer Schrittzahl-Funktion
    big_o(f, g) / big_omega / big_theta Aufgabe 8: Zeugen x0, C, c finden und begründen
    gauss_sum(n)                        der Gauss-Trick  1 + 2 + … + m = m·(m+1)/2

Zählregeln (``COUNTING_RULES``, Standard = Vorlesung, Ennios Mitschrift zu Aufgabe 2):

    * jede Zuweisung ........................ 1 Schritt   (``y = ...``, ``Liste[i] = ...``)
    * jede Rechenoperation (+ - * // % **) .. 1 Schritt
    * jeder Vergleich (== != < <= > >=) ..... 1 Schritt   (der Test von ``while``/``if`` besteht aus
                                                            seinen Vergleichen; auch der letzte,
                                                            fehlschlagende Test zählt)
    * ``return`` ............................ 1 Schritt
    * jeder Durchlauf einer ``for``-Schleife  1 Schritt   (Laufvariable setzen)
    * Listenzugriff ``Liste[i]``, ``len(...)``, Argumente von ``range`` und ``and/or/not`` zählen 0

Damit ergibt ``decToBin`` pro Schleifendurchlauf genau die 7 Schritte der Vorlesung
(``temp != 0``, ``temp % 2``, ``== 0``, ``'0' + y``, ``y = …``, ``temp // 2``, ``temp = …``), dazu
»3 initiale Schritte« (``y = ''``, ``temp = x`` und der letzte, fehlschlagende while-Test) und
»1 Return-Schritt«:   T(x) = 3 + 7·(⌊log₂ x⌋ + 1) + 1 = 11 + 7·⌊log₂ x⌋   für x ≥ 1.

Hinweise zu den Folien/Serie (im Abschlussbericht vermerkt):
  * Die Vorlesung zählt bei der Selection-Sort-Analyse die Zuweisung ``minIndex = j`` in jedem
    Innendurchlauf (obere Schranke 5n + 3·n(n−1)/2); tatsächlich wird sie nicht immer ausgeführt.
    Darum liefert ``worst_case`` beides: das echte Maximum und die Vorlesungs-Schranke
    (``branch="max"``: jeder ``if`` wird mit seinem teureren Zweig gezählt).
  * Im Lerngerüst steht ``anzzahl = anzzahl * 2`` (Tippfehler) — gemeint ist ``anzahl``.
"""

from __future__ import annotations

import ast
import copy
import itertools
import math
import re
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction

import sympy as sp

from .render import Result, Step, Table

# ==========================================================================
# Grenzen
# ==========================================================================

MAX_CODE_LEN = 6000            # Zeichen
MAX_CODE_LINES = 200
MAX_AST_NODES = 4000
MAX_AST_DEPTH = 60
DEFAULT_MAX_STEPS = 200_000    # Rechenschritte je Programmlauf (Standard)
MAX_STEPS_CAP = 5_000_000      # absolute Obergrenze für ``max_steps``
MAX_CALL_DEPTH = 30            # Rekursionstiefe der interpretierten Funktionen
MAX_LIST_LEN = 20_000
MAX_STR_LEN = 100_000
MAX_INT_BITS = 20_000
MAX_RANGE = 1_000_000
MAX_EXHAUSTIVE_N = 8           # Brute-Force über alle Permutationen höchstens bis n = 8
MAX_TERM_LEN = 300             # Länge von Funktionstermen (Strings)
MAX_N_VALUES = 60
MAX_N = 10 ** 15               # Suchgrenze bei max_input_size
MAX_WORST_N = 200              # grösstes n bei worst_case/best_case

__all__ = [
    "GrowthClass", "gauss_trick",
    "COUNTING_RULES", "default_rules", "count_steps", "StepCountResult",
    "REFERENCE_CODE", "REFERENCE_FUNCTIONS", "reference_code", "reference_algorithms",
    "worst_case", "best_case", "average_case", "CaseResult", "AverageResult",
    "compare_growth", "GrowthTableResult", "growth_factor",
    "max_input_size", "max_input_sizes", "MaxSizeResult", "MaxSizeTable",
    "growth_classes", "GrowthClassesResult", "classify", "ClassifyResult",
    "big_o", "big_omega", "big_theta", "AsymptoticResult",
    "gauss_sum", "GaussResult", "parse_term", "fit_formula", "Fit",
    "StepLimitError", "UnsafeCodeError", "AlgoRuntimeError",
    "decToBin", "linearSearch", "selectionSort", "insertionSort", "bubbleSort", "binarySearch",
    "nestedLoops",
]


# ==========================================================================
# Fehler
# ==========================================================================

class UnsafeCodeError(ValueError):
    """Der Code enthält etwas, das nicht zur erlaubten Python-Teilmenge gehört."""


class StepLimitError(ValueError):
    """Schrittlimit überschritten (vermutlich eine Endlosschleife)."""


class AlgoRuntimeError(ValueError):
    """Laufzeitfehler im analysierten Algorithmus (Index ausserhalb, Division durch 0 …)."""


# ==========================================================================
# Zählregeln
# ==========================================================================

# Schlüssel -> (Schritte, Kategorie, Erklärung, Beispiel)
COUNTING_RULES: dict[str, dict] = {
    "assign": {"steps": 1, "category": "Zuweisung",
               "text": "jede Zuweisung (auch Listenelement, Tausch zählt je Element)",
               "example": "y = '0' + y   →  1 für die Zuweisung"},
    "arith": {"steps": 1, "category": "Rechenoperation",
              "text": "jede Rechenoperation + − · // % ** (je Operator)",
              "example": "temp // 2  →  1;   j + 1  →  1"},
    "compare": {"steps": 1, "category": "Vergleich",
                "text": "jeder Vergleich == != < <= > >= (der Test von while/if besteht aus ihnen)",
                "example": "temp != 0  →  1;   Liste[j] < Liste[m]  →  1"},
    "return": {"steps": 1, "category": "Rückgabe",
               "text": "return",
               "example": "return y  →  1"},
    "for_iter": {"steps": 1, "category": "Schleifendurchlauf",
                 "text": "jeder Durchlauf einer for-Schleife (Laufvariable setzen)",
                 "example": "for i in range(n):  →  1 je Durchlauf"},
    "boolop": {"steps": 0, "category": "Logik",
               "text": "and / or / not (in Serie 5 nicht ausdrücklich gezählt)",
               "example": "j > 0 and Liste[j] < Liste[j-1]  →  nur die beiden Vergleiche"},
    "index": {"steps": 0, "category": "Zugriff",
              "text": "Listenzugriff Liste[i] (Speicherzugriff ist in Θ(1), wird nicht gezählt)",
              "example": "Liste[j]  →  0"},
    "len": {"steps": 0, "category": "Zugriff",
            "text": "len(...)", "example": "len(Liste)  →  0"},
    "range_args": {"steps": 0, "category": "Schleifendurchlauf",
                   "text": "Operatoren in den Argumenten von range(...) (0 = nicht zählen, 1 = zählen)",
                   "example": "range(i + 1, len(Liste))  →  das i + 1 zählt nicht"},
    "call": {"steps": 0, "category": "Zugriff",
             "text": "Aufruf einer eigenen Funktion", "example": "f(x)  →  0 (die Operationen darin zählen)"},
}


def default_rules() -> dict[str, int]:
    """Die Zählregeln der Vorlesung als Wörterbuch {Regel: Schritte}."""
    return {k: v["steps"] for k, v in COUNTING_RULES.items()}


def _check_rules(rules) -> dict[str, int]:
    out = default_rules()
    if rules is None:
        return out
    if not isinstance(rules, dict):
        raise ValueError("rules muss ein Wörterbuch {Regel: Schritte} sein, z. B. {'return': 0}")
    for key, val in rules.items():
        if key not in out:
            raise ValueError(f"Unbekannte Zählregel '{key}'. Erlaubt: {', '.join(out)}")
        if isinstance(val, bool) or not isinstance(val, int) or not 0 <= val <= 100:
            raise ValueError(f"Zählregel '{key}' muss eine ganze Zahl 0 … 100 sein")
        out[key] = val
    return out


# ==========================================================================
# Prüfung der Python-Teilmenge (AST)
# ==========================================================================

_FORBIDDEN_NAMES = {
    ast.Import: "import", ast.ImportFrom: "import", ast.Attribute: "Attributzugriff (Punkt-Schreibweise, z. B. liste.append)",
    ast.Lambda: "lambda", ast.ClassDef: "class", ast.Try: "try/except", ast.With: "with",
    ast.Global: "global", ast.Nonlocal: "nonlocal", ast.Delete: "del", ast.Raise: "raise",
    ast.Assert: "assert", ast.ListComp: "Listen-Abkürzung (Comprehension)", ast.SetComp: "Comprehension",
    ast.DictComp: "Comprehension", ast.GeneratorExp: "Generator", ast.JoinedStr: "f-String",
    ast.FormattedValue: "f-String", ast.Dict: "Wörterbuch {…}", ast.Set: "Menge {…}",
    ast.Starred: "Sternausdruck *", ast.Slice: "Slicing (Liste[a:b])", ast.Await: "await",
    ast.Yield: "yield", ast.YieldFrom: "yield", ast.NamedExpr: "Walrus-Operator :=",
    ast.AnnAssign: "Typannotation", ast.AsyncFunctionDef: "async", ast.AsyncFor: "async",
    ast.AsyncWith: "async", ast.Match: "match",
}
_OK_STMT = (ast.FunctionDef, ast.Return, ast.Assign, ast.AugAssign, ast.For, ast.While, ast.If,
            ast.Break, ast.Continue, ast.Pass, ast.Expr)
_OK_EXPR = (ast.Name, ast.Constant, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Call,
            ast.List, ast.Tuple, ast.Subscript, ast.IfExp)
_OK_OPS = (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Not,
           ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Load, ast.Store)
_BUILTINS = ("len", "range", "abs")
_BAD_CALLS = ("exec", "eval", "open", "compile", "__import__", "getattr", "setattr", "globals",
              "locals", "input", "print", "vars", "dir", "type", "id", "help", "breakpoint")


def _where(node) -> str:
    return f" (Zeile {getattr(node, 'lineno', '?')})"


def _validate(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    """Lässt nur die Python-Teilmenge zu; gibt die Funktionsdefinitionen zurück."""
    defs: dict[str, ast.FunctionDef] = {}
    for st in tree.body:
        if isinstance(st, ast.FunctionDef):
            if st.name in defs:
                raise UnsafeCodeError(f"Funktion '{st.name}' ist doppelt definiert" + _where(st))
            defs[st.name] = st
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise UnsafeCodeError(f"Der Code ist zu gross (mehr als {MAX_AST_NODES} Knoten)")

    def check(node, depth, in_func, in_loop, top):
        if depth > MAX_AST_DEPTH:
            raise UnsafeCodeError("Der Code ist zu tief verschachtelt")
        kind = type(node)
        if kind in _FORBIDDEN_NAMES:
            raise UnsafeCodeError(f"{_FORBIDDEN_NAMES[kind]} ist in der Python-Teilmenge nicht erlaubt"
                                  + _where(node))
        if isinstance(node, ast.stmt):
            if not isinstance(node, _OK_STMT):
                raise UnsafeCodeError(f"{kind.__name__} ist in der Python-Teilmenge nicht erlaubt" + _where(node))
            if isinstance(node, ast.FunctionDef):
                if not top:
                    raise UnsafeCodeError("Verschachtelte Funktionsdefinitionen sind nicht erlaubt" + _where(node))
                a = node.args
                if a.vararg or a.kwarg or a.kwonlyargs or a.defaults or a.kw_defaults or a.posonlyargs:
                    raise UnsafeCodeError("Nur einfache Parameter erlaubt (keine Vorgabewerte, *args, **kwargs)"
                                          + _where(node))
                if node.decorator_list or node.returns or any(x.annotation for x in a.args):
                    raise UnsafeCodeError("Dekoratoren und Typannotationen sind nicht erlaubt" + _where(node))
                if node.name.startswith("__") or node.name in _BUILTINS or node.name in _BAD_CALLS:
                    raise UnsafeCodeError(f"Der Funktionsname '{node.name}' ist nicht erlaubt" + _where(node))
                names = [x.arg for x in a.args]
                if len(set(names)) != len(names) or any("__" in x for x in names):
                    raise UnsafeCodeError("Ungültige Parameternamen" + _where(node))
                for st in node.body:
                    check(st, depth + 1, True, False, False)
                return
            if isinstance(node, ast.Return) and not in_func:
                raise UnsafeCodeError("return ausserhalb einer Funktion" + _where(node))
            if isinstance(node, (ast.Break, ast.Continue)) and not in_loop:
                raise UnsafeCodeError("break/continue ausserhalb einer Schleife" + _where(node))
            if isinstance(node, (ast.For, ast.While)) and node.orelse:
                raise UnsafeCodeError("else bei Schleifen ist nicht erlaubt" + _where(node))
            if isinstance(node, ast.For):
                if not isinstance(node.target, ast.Name):
                    raise UnsafeCodeError("for-Schleife: Laufvariable muss ein einfacher Name sein" + _where(node))
                it = node.iter
                if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range":
                    if not 1 <= len(it.args) <= 3 or it.keywords:
                        raise UnsafeCodeError("range(...) braucht 1 bis 3 Argumente" + _where(node))
            if isinstance(node, ast.Expr):
                v = node.value
                if type(v) in _FORBIDDEN_NAMES:
                    raise UnsafeCodeError(f"{_FORBIDDEN_NAMES[type(v)]} ist in der Python-Teilmenge nicht erlaubt"
                                          + _where(node))
                if not (isinstance(v, ast.Constant) or isinstance(v, ast.Call)):
                    raise UnsafeCodeError("Ein Ausdruck allein als Anweisung ist nicht erlaubt" + _where(node))
            loop = in_loop or isinstance(node, (ast.For, ast.While))
            for child in ast.iter_child_nodes(node):
                check(child, depth + 1, in_func, loop, False)
            return
        if isinstance(node, ast.Name):
            if "__" in node.id:
                raise UnsafeCodeError(f"Namen mit doppeltem Unterstrich sind nicht erlaubt: '{node.id}'" + _where(node))
            return
        if isinstance(node, ast.Constant):
            v = node.value
            if not (isinstance(v, (int, str, bool)) or v is None) or (isinstance(v, str) and len(v) > 1000):
                raise UnsafeCodeError("Erlaubte Konstanten: ganze Zahlen, Wahrheitswerte, Text und None "
                                      "(keine Kommazahlen)" + _where(node))
            if isinstance(v, int) and v.bit_length() > 2000:
                raise UnsafeCodeError("Konstante zu gross" + _where(node))
            return
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute):
                raise UnsafeCodeError(f"{_FORBIDDEN_NAMES[ast.Attribute]} ist in der Python-Teilmenge nicht "
                                      "erlaubt (keine Methoden wie liste.append)" + _where(node))
            if not isinstance(f, ast.Name):
                raise UnsafeCodeError("Aufgerufen werden dürfen nur Namen (len, range, abs, eigene Funktionen)"
                                      + _where(node))
            if node.keywords:
                raise UnsafeCodeError("Schlüsselwort-Argumente sind nicht erlaubt" + _where(node))
            if f.id in _BAD_CALLS or "__" in f.id:
                raise UnsafeCodeError(f"Aufruf von '{f.id}' ist nicht erlaubt" + _where(node))
            if f.id not in _BUILTINS and f.id not in defs:
                raise UnsafeCodeError(f"Aufruf von '{f.id}' ist nicht erlaubt: erlaubt sind nur "
                                      "len, range, abs und eigene Funktionen" + _where(node))
            if f.id == "len" and len(node.args) != 1 or f.id == "abs" and len(node.args) != 1:
                raise UnsafeCodeError(f"{f.id}(...) braucht genau ein Argument" + _where(node))
            for a in node.args:
                if isinstance(a, ast.Starred):
                    raise UnsafeCodeError("Sternausdruck * ist nicht erlaubt" + _where(node))
                check(a, depth + 1, in_func, in_loop, False)
            return
        if isinstance(node, ast.Compare) and any(isinstance(o, (ast.Is, ast.IsNot, ast.In, ast.NotIn))
                                                 for o in node.ops):
            raise UnsafeCodeError("is / in sind nicht erlaubt (sie würden Schritte verstecken) — "
                                  "schreibe die Suche als Schleife" + _where(node))
        if isinstance(node, _OK_EXPR) or isinstance(node, ast.expr_context):
            for child in ast.iter_child_nodes(node):
                check(child, depth + 1, in_func, in_loop, False)
            return
        if isinstance(node, _OK_OPS):
            return
        if isinstance(node, (ast.Div, ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift,
                             ast.MatMult, ast.Invert)):
            raise UnsafeCodeError("Dieser Operator ist nicht erlaubt (erlaubt: + − * // % ** und Vergleiche; "
                                  "für die Division // verwenden)" + _where(node))
        if isinstance(node, (ast.Is, ast.IsNot, ast.In, ast.NotIn)):
            raise UnsafeCodeError("is / in sind nicht erlaubt (sie würden Schritte verstecken) — "
                                  "schreibe die Suche als Schleife" + _where(node))
        if isinstance(node, ast.arguments) or isinstance(node, ast.arg):
            return
        raise UnsafeCodeError(f"{kind.__name__} ist in der Python-Teilmenge nicht erlaubt" + _where(node))

    for st in tree.body:
        check(st, 0, False, False, True)
    return defs


# ==========================================================================
# Interpreter
# ==========================================================================

class _Flow(Exception):
    pass


class _Return(_Flow):
    def __init__(self, value):
        self.value = value


class _Break(_Flow):
    pass


class _Continue(_Flow):
    pass


_CAT_OF = {k: v["category"] for k, v in COUNTING_RULES.items()}
_EXTRA = "Worst-Case-Zuschlag"


def _has_flow(stmts) -> bool:
    return any(isinstance(n, (ast.Return, ast.Break, ast.Continue)) for s in stmts for n in ast.walk(s))


class _Interp:
    """AST-Interpreter, der nach den Zählregeln Schritte zählt. Kein eval/exec."""

    def __init__(self, tree, lines, rules, branch, max_steps):
        self.tree, self.lines, self.rules, self.branch = tree, lines, rules, branch
        self.max_steps = max_steps
        self.max_work = 30 * max_steps + 100_000
        self.funcs: dict[str, ast.FunctionDef] = {}
        self.frames: list[dict] = [{}]
        self.steps = 0
        self.work = 0
        self.by_line: Counter = Counter()
        self.exec_line: Counter = Counter()
        self.by_cat: Counter = Counter()
        self.line = 0
        self.muted = 0
        self.dry = False
        self.dry_cost = 0
        self.depth = 0

    # -- Buchführung ------------------------------------------------------

    def src(self, line) -> str:
        return self.lines[line - 1].strip() if 1 <= line <= len(self.lines) else ""

    def tick(self):
        self.work += 1
        if self.work > self.max_work:
            self._limit()

    def _limit(self):
        raise StepLimitError(
            f"Abbruch: mehr als {self.max_steps} Rechenschritte (zuletzt Zeile {self.line}: "
            f"`{self.src(self.line)}`). Vermutlich eine Endlosschleife — wird die Schleifenbedingung je "
            "falsch? Wird die Laufvariable in der Schleife verändert? "
            "Mit max_steps=… lässt sich das Limit erhöhen.")

    def charge(self, key: str, times: int = 1):
        if self.muted:
            return
        n = self.rules[key] * times
        if not n:
            return
        if self.dry:
            self.dry_cost += n
            return
        self.steps += n
        self.by_line[self.line] += n
        self.by_cat[_CAT_OF[key]] += n
        if self.steps > self.max_steps:
            self._limit()

    def hit(self):
        if not self.dry:
            self.exec_line[self.line] += 1

    def err(self, msg):
        return AlgoRuntimeError(f"{msg} (Zeile {self.line}: `{self.src(self.line)}`)")

    # -- Umgebung ---------------------------------------------------------

    def lookup(self, name):
        frame = self.frames[-1]
        if name in frame:
            return frame[name]
        if name in self.frames[0]:
            return self.frames[0][name]
        raise self.err(f"Variable '{name}' ist nicht definiert")

    def setname(self, name, value):
        self.frames[-1][name] = value

    # -- Ausdrücke --------------------------------------------------------

    def ev(self, node):
        self.tick()
        return getattr(self, "_e_" + type(node).__name__)(node)

    def _e_Constant(self, n):
        return n.value

    def _e_Name(self, n):
        return self.lookup(n.id)

    def _e_List(self, n):
        if len(n.elts) > MAX_LIST_LEN:
            raise self.err("Liste zu lang")
        return [self.ev(e) for e in n.elts]

    def _e_Tuple(self, n):
        return tuple(self.ev(e) for e in n.elts)

    def _e_IfExp(self, n):
        return self.ev(n.body) if self.ev(n.test) else self.ev(n.orelse)

    def _e_UnaryOp(self, n):
        if isinstance(n.operand, ast.Constant) and isinstance(n.operand.value, int) \
                and isinstance(n.op, (ast.USub, ast.UAdd)):
            return -n.operand.value if isinstance(n.op, ast.USub) else n.operand.value
        v = self.ev(n.operand)
        if isinstance(n.op, ast.Not):
            self.charge("boolop")
            return not v
        if isinstance(v, bool) or not isinstance(v, int):
            raise self.err("Unäres Minus/Plus nur für ganze Zahlen")
        self.charge("arith")
        return -v if isinstance(n.op, ast.USub) else v

    def _e_BoolOp(self, n):
        is_and = isinstance(n.op, ast.And)
        v = None
        for i, part in enumerate(n.values):
            v = self.ev(part)
            if i < len(n.values) - 1:
                self.charge("boolop")
            if is_and and not v:
                return v
            if not is_and and v:
                return v
        return v

    def _e_BinOp(self, n):
        a = self.ev(n.left)
        b = self.ev(n.right)
        self.charge("arith")
        return self._binop(n.op, a, b)

    def _binop(self, op, a, b):
        isint = lambda v: isinstance(v, int)
        try:
            if isinstance(op, ast.Add):
                if isint(a) and isint(b):
                    r = a + b
                elif isinstance(a, str) and isinstance(b, str):
                    if len(a) + len(b) > MAX_STR_LEN:
                        raise self.err("Text zu lang")
                    return a + b
                elif isinstance(a, list) and isinstance(b, list):
                    if len(a) + len(b) > MAX_LIST_LEN:
                        raise self.err("Liste zu lang")
                    return a + b
                else:
                    raise TypeError
            elif isinstance(op, ast.Sub):
                if not (isint(a) and isint(b)):
                    raise TypeError
                r = a - b
            elif isinstance(op, ast.Mult):
                if isint(a) and isint(b):
                    if a.bit_length() + b.bit_length() > MAX_INT_BITS:
                        raise self.err("Zahl zu gross")
                    r = a * b
                elif isinstance(a, (list, str)) and isint(b) or isint(a) and isinstance(b, (list, str)):
                    seq, k = (a, b) if not isint(a) else (b, a)
                    if k > 0 and len(seq) * k > (MAX_LIST_LEN if isinstance(seq, list) else MAX_STR_LEN):
                        raise self.err("Ergebnis zu lang")
                    return seq * k
                else:
                    raise TypeError
            elif isinstance(op, ast.FloorDiv):
                if not (isint(a) and isint(b)):
                    raise TypeError
                r = a // b
            elif isinstance(op, ast.Mod):
                if not (isint(a) and isint(b)):
                    raise TypeError
                r = a % b
            elif isinstance(op, ast.Pow):
                if not (isint(a) and isint(b)):
                    raise TypeError
                if b < 0:
                    raise self.err("Negative Exponenten sind nicht erlaubt (Ergebnis wäre keine ganze Zahl)")
                if b > 20_000 or (abs(a) > 1 and abs(a).bit_length() * b > MAX_INT_BITS):
                    raise self.err("Potenz zu gross")
                r = a ** b
            else:
                raise TypeError
        except ZeroDivisionError:
            raise self.err("Division durch 0") from None
        except TypeError:
            raise self.err(f"Der Operator passt nicht zu den Typen {type(a).__name__} und "
                           f"{type(b).__name__}") from None
        if isinstance(r, int) and r.bit_length() > MAX_INT_BITS:
            raise self.err("Zahl zu gross")
        return r

    def _e_Compare(self, n):
        left = self.ev(n.left)
        for op, comp in zip(n.ops, n.comparators):
            right = self.ev(comp)
            self.charge("compare")
            try:
                if isinstance(op, ast.Eq):
                    ok = left == right
                elif isinstance(op, ast.NotEq):
                    ok = left != right
                elif isinstance(op, ast.Lt):
                    ok = left < right
                elif isinstance(op, ast.LtE):
                    ok = left <= right
                elif isinstance(op, ast.Gt):
                    ok = left > right
                else:
                    ok = left >= right
            except TypeError:
                raise self.err(f"Vergleich nicht möglich zwischen {type(left).__name__} und "
                               f"{type(right).__name__}") from None
            if not ok:
                return False
            left = right
        return True

    def _index(self, container, idx):
        if not isinstance(container, (list, str)):
            raise self.err("Indexzugriff nur auf Listen und Text")
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise self.err("Der Index muss eine ganze Zahl sein")
        if not -len(container) <= idx < len(container):
            raise self.err(f"Index {idx} ausserhalb der Liste (Länge {len(container)})")
        return idx

    def _e_Subscript(self, n):
        c = self.ev(n.value)
        i = self._index(c, self.ev(n.slice))
        self.charge("index")
        return c[i]

    def _e_Call(self, n):
        name = n.func.id
        if name == "range":
            raise self.err("range(...) darf nur im Kopf einer for-Schleife stehen")
        args = [self.ev(a) for a in n.args]
        if name == "len":
            if not isinstance(args[0], (list, str)):
                raise self.err("len(...) nur für Listen und Text")
            self.charge("len")
            return len(args[0])
        if name == "abs":
            if isinstance(args[0], bool) or not isinstance(args[0], int):
                raise self.err("abs(...) nur für ganze Zahlen")
            self.charge("arith")
            return abs(args[0])
        return self.call(name, args)

    def call(self, name, args):
        fd = self.funcs[name]
        params = [a.arg for a in fd.args.args]
        if len(args) != len(params):
            raise self.err(f"{name} erwartet {len(params)} Argument(e), bekam {len(args)}")
        if self.depth >= MAX_CALL_DEPTH:
            raise self.err(f"Rekursionstiefe {MAX_CALL_DEPTH} überschritten")
        self.charge("call")
        self.frames.append(dict(zip(params, args)))
        self.depth += 1
        saved_line = self.line
        try:
            self.exec_block(fd.body)
            value = None
        except _Return as r:
            value = r.value
        except RecursionError:
            raise self.err("Rekursionstiefe überschritten") from None
        finally:
            self.depth -= 1
            self.frames.pop()
            self.line = saved_line
        return value

    # -- Anweisungen ------------------------------------------------------

    def exec_block(self, stmts):
        for s in stmts:
            self.exec_stmt(s)

    def exec_stmt(self, s):
        self.tick()
        self.line = s.lineno
        getattr(self, "_s_" + type(s).__name__)(s)

    def _s_FunctionDef(self, s):
        self.funcs[s.name] = s

    def _s_Pass(self, s):
        self.hit()

    def _s_Break(self, s):
        self.hit()
        raise _Break()

    def _s_Continue(self, s):
        self.hit()
        raise _Continue()

    def _s_Expr(self, s):
        self.hit()
        if not isinstance(s.value, ast.Constant):
            self.ev(s.value)

    def _s_Return(self, s):
        self.hit()
        v = self.ev(s.value) if s.value is not None else None
        self.charge("return")
        raise _Return(v)

    def _store(self, target, value):
        if isinstance(target, ast.Name):
            self.setname(target.id, value)
        elif isinstance(target, ast.Subscript):
            c = self.ev(target.value)
            if not isinstance(c, list):
                raise self.err("Index-Zuweisung nur bei Listen")
            i = self._index(c, self.ev(target.slice))
            c[i] = value
        elif isinstance(target, ast.Tuple):
            # ponytail: Tausch a, b = b, a zählt einfach 2 Zuweisungen (kein eigener Tausch-Schritt)
            if not isinstance(value, (tuple, list)) or len(value) != len(target.elts):
                raise self.err("Auspacken: Anzahl der Werte passt nicht")
            for t, v in zip(target.elts, value):
                self._store(t, v)
            return
        else:
            raise self.err("Ungültiges Zuweisungsziel")
        self.charge("assign")

    def _s_Assign(self, s):
        self.hit()
        v = self.ev(s.value)
        for t in s.targets:
            self._store(t, v)

    def _s_AugAssign(self, s):
        self.hit()
        t = s.target
        if isinstance(t, ast.Name):
            cur = self.lookup(t.id)
            rhs = self.ev(s.value)
            self.charge("arith")
            self.setname(t.id, self._binop(s.op, cur, rhs))
            self.charge("assign")
        elif isinstance(t, ast.Subscript):
            c = self.ev(t.value)
            i = self._index(c, self.ev(t.slice))
            if not isinstance(c, list):
                raise self.err("Index-Zuweisung nur bei Listen")
            rhs = self.ev(s.value)
            self.charge("arith")
            c[i] = self._binop(s.op, c[i], rhs)
            self.charge("assign")
        else:
            raise self.err("Ungültiges Ziel für erweiterte Zuweisung")

    def _s_If(self, s):
        self.hit()
        test = self.ev(s.test)
        body, other = (s.body, s.orelse) if test else (s.orelse, s.body)
        before = self.steps
        self.exec_block(body)
        if self.branch == "max" and not self.dry and other and not _has_flow(other):
            extra = self._dry_run(other) - (self.steps - before)
            if extra > 0:
                self.steps += extra
                self.by_line[s.lineno] += extra
                self.by_cat[_EXTRA] += extra

    def _dry_run(self, stmts) -> int:
        """Kosten eines nicht genommenen Zweigs — auf einer Kopie des Zustands."""
        saved = (self.frames, self.dry, self.dry_cost, self.line, self.depth)
        # ponytail: Zustand tief kopieren und den Zweig wirklich ausführen, statt ihn symbolisch zu analysieren
        self.frames = copy.deepcopy(self.frames)
        self.dry, self.dry_cost = True, 0
        try:
            self.exec_block(stmts)
        except (_Flow, AlgoRuntimeError):
            pass
        finally:
            cost = self.dry_cost
            self.frames, self.dry, self.dry_cost, self.line, self.depth = saved
        return cost

    def _s_While(self, s):
        while True:
            self.line = s.lineno
            self.hit()
            self.tick()
            if not self.ev(s.test):
                break
            try:
                self.exec_block(s.body)
            except _Break:
                break
            except _Continue:
                continue

    def _loop_values(self, it):
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range":
            mute = not self.rules["range_args"]
            self.muted += mute
            try:
                args = [self.ev(a) for a in it.args]
            finally:
                self.muted -= mute
            if any(isinstance(a, bool) or not isinstance(a, int) for a in args):
                raise self.err("range(...) braucht ganze Zahlen")
            try:
                r = range(*args)
            except ValueError:
                raise self.err("range: Schrittweite darf nicht 0 sein") from None
            if len(r) > MAX_RANGE:
                raise self.err(f"range(...) ist zu gross (mehr als {MAX_RANGE} Durchläufe)")
            return r
        v = self.ev(it)
        if not isinstance(v, (list, str)):
            raise self.err("for-Schleife nur über range(...), Listen oder Text")
        return v

    def _s_For(self, s):
        seq = self._loop_values(s.iter)
        is_list = isinstance(seq, list)
        i = 0
        while True:
            if is_list:
                if i >= len(seq):
                    break
                v = seq[i]
            else:
                if i >= len(seq):
                    break
                v = seq[i]
            i += 1
            self.line = s.lineno
            self.hit()
            self.tick()
            self.setname(s.target.id, v)
            self.charge("for_iter")
            try:
                self.exec_block(s.body)
            except _Break:
                break
            except _Continue:
                continue


# ==========================================================================
# Eingaben prüfen
# ==========================================================================

def _clean_value(v, depth=0):
    if depth > 3:
        raise ValueError("Eingaben: Listen dürfen höchstens 3 Ebenen tief sein")
    if v is None or isinstance(v, (bool, str)):
        # ponytail: keine Kommazahlen in der Teilmenge (Serie 5 rechnet nur mit ganzen Zahlen)
        if isinstance(v, str) and len(v) > MAX_STR_LEN:
            raise ValueError("Eingabe: Text zu lang")
        return v
    if isinstance(v, int):
        if v.bit_length() > MAX_INT_BITS:
            raise ValueError("Eingabe: Zahl zu gross")
        return v
    if isinstance(v, (list, tuple, range)):
        if len(v) > MAX_LIST_LEN:
            raise ValueError(f"Eingabe: Liste zu lang (höchstens {MAX_LIST_LEN} Elemente)")
        return [_clean_value(e, depth + 1) for e in v]
    raise ValueError(f"Eingaben müssen ganze Zahlen, Text, Wahrheitswerte oder Listen davon sein "
                     f"(bekam {type(v).__name__}); Kommazahlen sind nicht Teil der Teilmenge")


# ==========================================================================
# Ergebnis: Schritte zählen
# ==========================================================================

@dataclass
class LineRow:
    line: int
    code: str
    executions: int
    steps: int


class StepCountResult(Result):
    """
    Ergebnis von ``count_steps``.

    Attribute: ``total`` (Gesamtschritte), ``value`` (Rückgabewert), ``arguments`` (Argumente nach dem
    Lauf, z. B. die sortierte Liste), ``rows`` (``LineRow`` je Zeile: Ausführungen und Schritte),
    ``by_category`` (Schritte je Art), ``rules``, ``func``, ``inputs``.
    """

    def __init__(self, code, func, inputs, rules, branch, total, value, arguments, rows, by_category,
                 variables):
        super().__init__()
        self.code, self.func, self.inputs, self.rules, self.branch = code, func, inputs, rules, branch
        self.total, self.value, self.arguments = total, value, arguments
        self.rows, self.by_category, self.variables = rows, by_category, variables
        shown = ", ".join(f"{k} = {_short(v)}" for k, v in inputs.items())
        self.title = f"Rechenschritte zählen: {func}({shown})" if func else "Rechenschritte zählen"
        self.summary = (f"Gesamtschritte = {total}"
                        + (f", Rückgabewert = {_short(value)}" if value is not None else "")
                        + (" (teuerster Zweig je if)" if branch == "max" else ""))
        self._build()

    def exec_count(self, line: int) -> int:
        """Wie oft die Zeile ausgeführt wurde (bei Schleifenköpfen: Tests bzw. Durchläufe)."""
        return next((r.executions for r in self.rows if r.line == line), 0)

    def steps_of(self, *lines: int) -> int:
        """Summe der Schritte der angegebenen Zeilen (z. B. nur die »rot gefärbten« Operationen)."""
        return sum(r.steps for r in self.rows if r.line in lines)

    def _build(self):
        r = self.rules
        rule_rows = [[v["text"], str(r[k]), v["example"]] for k, v in COUNTING_RULES.items()]
        changed = [k for k in r if r[k] != COUNTING_RULES[k]["steps"]]
        self.steps.append(Step(
            "Zählregeln", "Gezählt werden Zuweisungen, Vergleiche und Rechenoperationen "
            "(Serie 5, Aufgabe 2); jeder Durchlauf einer for-Schleife und das return je 1 Schritt."
            + (f" Abweichend von der Vorlesung: {', '.join(changed)}." if changed else ""),
            rule="Zählregeln der Vorlesung",
            table=Table(["Operation", "Schritte", "Beispiel"], rule_rows)))
        items = [(k, v) for k, v in self.inputs.items()]
        if self.func:
            items.append(("Rückgabewert", self.value))
            for k, v in self.arguments.items():
                if isinstance(v, list) and v != self.inputs.get(k):
                    items.append((f"{k} nach dem Lauf", v))
        self.steps.append(Step("Programm ausführen", "Der Interpreter führt den Code Schritt für Schritt "
                               "aus und zählt dabei mit.", items=items))
        rows = [[str(x.line), x.code, str(x.executions), str(x.steps)] for x in self.rows]
        hi = {i: "hi" for i, x in enumerate(self.rows) if x.steps and x.steps == max(y.steps for y in self.rows)}
        self.steps.append(Step(
            "Schritte je Zeile", "Ausführungen bei while/if/for = Anzahl der Tests bzw. Durchläufe "
            "(beim while zählt der letzte, fehlschlagende Test mit).",
            table=Table(["Zeile", "Code", "Ausführungen", "Schritte"], rows, marks=hi)))
        cat_rows = [[k, str(v)] for k, v in self.by_category.items() if v]
        self.steps.append(Step("Schritte nach Art", table=Table(["Art", "Schritte"], cat_rows)))
        parts = " + ".join(str(x.steps) for x in self.rows if x.steps)
        self.steps.append(Step("Summe", f"{parts or '0'} = {self.total}", items=[("Gesamtschritte", self.total)],
                               rule="Schritte addieren"))
        self.highlights = [("Gesamtschritte", self.total)]
        if self.func:
            self.highlights.append(("Rückgabewert", self.value))


def _short(v, n=40):
    s = repr(v) if not isinstance(v, str) else repr(v)
    return s if len(s) <= n else s[:n - 1] + "…"


def _prepare(code):
    if not isinstance(code, str):
        raise ValueError("code muss ein Text (Python-Quelltext oder Name eines Referenzalgorithmus) sein")
    key = _norm_name(code)
    if "\n" not in code and key in _REF_INDEX:
        code = REFERENCE_CODE[_REF_INDEX[key]]
    if len(code) > MAX_CODE_LEN:
        raise UnsafeCodeError(f"Der Code ist zu lang (mehr als {MAX_CODE_LEN} Zeichen)")
    code = textwrap.dedent(code.strip("\n").replace("\t", "    "))
    if code.count("\n") + 1 > MAX_CODE_LINES:
        raise UnsafeCodeError(f"Der Code hat zu viele Zeilen (mehr als {MAX_CODE_LINES})")
    if not code.strip():
        raise ValueError("Der Code ist leer")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"Syntaxfehler in Zeile {e.lineno}: {e.msg}") from None
    except (ValueError, RecursionError, MemoryError):
        raise ValueError("Der Code kann nicht gelesen werden (zu tief verschachtelt oder ungültig)") from None
    return code, tree


def _run(code, tree, defs, args, func, inputs, rules, branch, max_steps):
    # ponytail: als Funktion aufgerufen wird die letzte def (oder func=...), kein Aufrufausdruck im Code nötig
    """Führt aus; gibt (interp, value, arguments, func_name) zurück."""
    lines = code.split("\n")
    it = _Interp(tree, lines, rules, branch, max_steps)
    only_defs = all(isinstance(s, ast.FunctionDef) for s in tree.body)
    for name, fd in defs.items():
        it.funcs[name] = fd
    value, arguments, fname = None, {}, None
    try:
        if only_defs and defs:
            fname = func or list(defs)[-1]
            if fname not in defs:
                raise ValueError(f"Funktion '{fname}' ist im Code nicht definiert (vorhanden: {', '.join(defs)})")
            params = [a.arg for a in defs[fname].args.args]
            vals = dict(zip(params, args))
            if len(args) > len(params):
                raise ValueError(f"{fname} hat nur {len(params)} Parameter ({', '.join(params)})")
            for k, v in inputs.items():
                if k not in params:
                    raise ValueError(f"Unbekannter Parameter '{k}' für {fname}({', '.join(params)})")
                if k in vals:
                    raise ValueError(f"Parameter '{k}' doppelt angegeben")
                vals[k] = v
            missing = [p for p in params if p not in vals]
            if missing:
                raise ValueError(f"Eingaben fehlen: {', '.join(missing)} — z. B. count_steps(code, "
                                 + ", ".join(f"{p}=…" for p in params) + ")")
            vals = {k: _clean_value(v) for k, v in vals.items()}
            inputs_used = {p: copy.deepcopy(vals[p]) for p in params}      # Zustand VOR dem Lauf
            call_args = [vals[p] for p in params]
            it.line = defs[fname].lineno
            value = it.call(fname, call_args)
            arguments = dict(zip(params, call_args))
            return it, value, arguments, fname, inputs_used
        if args:
            raise ValueError("Positionelle Eingaben gibt es nur bei Code mit Funktionsdefinition")
        for k, v in inputs.items():
            it.frames[0][k] = _clean_value(v)
        inputs_used = copy.deepcopy(dict(it.frames[0]))
        for st in tree.body:
            it.exec_stmt(st)
        arguments = {}
        return it, None, arguments, None, inputs_used
    except RecursionError:
        raise AlgoRuntimeError("Rekursionstiefe überschritten") from None
    except _Flow:
        raise AlgoRuntimeError("Unerwartetes break/continue/return") from None


def count_steps(code: str, *args, func: str | None = None, rules: dict | None = None,
                branch: str = "actual", max_steps: int = DEFAULT_MAX_STEPS, **inputs) -> StepCountResult:
    """
    Zählt die Rechenschritte eines Programms nach den Zählregeln der Vorlesung.

    ``code``  Quelltext in der Python-Teilmenge (Funktion ``def …`` oder Skript) oder der Name eines
              Referenzalgorithmus (``"decToBin"``, ``"selectionSort"`` …).
    ``inputs`` Eingaben als Schlüsselwörter (Parametername = Wert); bei einem Skript werden sie als
              Variablen vorbelegt. ``func`` wählt die aufgerufene Funktion (Standard: die letzte).
    ``rules`` ändert Zählregeln, z. B. ``{"return": 0}`` (siehe ``COUNTING_RULES``).
    ``branch="max"`` zählt jedes ``if`` mit seinem teureren Zweig (Vorlesungs-Zählung für den Worst Case).

    >>> r = count_steps("decToBin", x=25)
    >>> r.total          # 3 + 7·5 + 1
    39
    >>> r.value
    '11001'
    >>> count_steps("linearSearch", Liste=[1, 2, 3], Element=9).total     # 2n + 1
    7

    Endlosschleifen werden mit ``StepLimitError`` abgebrochen, unerlaubter Code mit ``UnsafeCodeError``.
    """
    if branch not in ("actual", "max"):
        raise ValueError("branch muss 'actual' oder 'max' sein")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= MAX_STEPS_CAP:
        raise ValueError(f"max_steps muss eine ganze Zahl zwischen 1 und {MAX_STEPS_CAP} sein")
    rules = _check_rules(rules)
    code, tree = _prepare(code)
    defs = _validate(tree)
    it, value, arguments, fname, inputs_used = _run(code, tree, defs, args, func, inputs, rules, branch, max_steps)
    rows = [LineRow(ln, it.src(ln), it.exec_line.get(ln, 0), it.by_line.get(ln, 0))
            for ln in sorted(set(it.exec_line) | set(it.by_line)) if ln > 0]
    cats = {c: it.by_cat.get(c, 0) for c in dict.fromkeys(v["category"] for v in COUNTING_RULES.values())}
    if it.by_cat.get(_EXTRA):
        cats[_EXTRA] = it.by_cat[_EXTRA]
    variables = dict(it.frames[0]) if fname is None else {}
    res = StepCountResult(code, fname, inputs_used, rules, branch, it.steps, value,
                          {k: v for k, v in arguments.items()}, rows, cats, variables)
    return res


# ==========================================================================
# Referenzalgorithmen: Quelltext in der Teilmenge UND echte Python-Funktionen
# ==========================================================================

REFERENCE_CODE: dict[str, str] = {
    "decToBin": """
def decToBin(x):
    y = ''
    temp = x
    while temp != 0:
        if temp % 2 == 0:
            y = '0' + y
        else:
            y = '1' + y
        temp = temp // 2
    return y
""",
    "linearSearch": """
def linearSearch(Liste, Element):
    for i in range(len(Liste)):
        if Liste[i] == Element:
            return True
    return False
""",
    "selectionSort": """
def selectionSort(Liste):
    for i in range(len(Liste)):
        minIndex = i
        for j in range(i + 1, len(Liste)):
            if Liste[j] < Liste[minIndex]:
                minIndex = j
        temp = Liste[i]
        Liste[i] = Liste[minIndex]
        Liste[minIndex] = temp
""",
    "insertionSort": """
def insertionSort(Liste):
    for i in range(1, len(Liste)):
        j = i
        while j > 0 and Liste[j] < Liste[j - 1]:
            temp = Liste[j]
            Liste[j] = Liste[j - 1]
            Liste[j - 1] = temp
            j = j - 1
""",
    "bubbleSort": """
def bubbleSort(Liste):
    for i in range(len(Liste)):
        for j in range(len(Liste) - 1):
            if Liste[j] > Liste[j + 1]:
                temp = Liste[j]
                Liste[j] = Liste[j + 1]
                Liste[j + 1] = temp
""",
    "binarySearch": """
def binarySearch(Liste, Element):
    left = 0
    right = len(Liste) - 1
    while left <= right:
        middle = (left + right) // 2
        if Liste[middle] == Element:
            return True
        elif Liste[middle] > Element:
            right = middle - 1
        else:
            left = middle + 1
    return False
""",
    "nestedLoops": """
def nestedLoops(n):
    anzahl = 0
    i = 0
    while i < n:
        j = 0
        while j < i:
            anzahl = anzahl + 1
            anzahl = anzahl * 2
            j = j + 1
        i = i + 1
    return anzahl
""",
}
REFERENCE_CODE = {k: v.strip("\n") + "\n" for k, v in REFERENCE_CODE.items()}


# Echte Python-Funktionen (zum Gegenprüfen; Quelltext = REFERENCE_CODE, getestet per AST-Vergleich).
def decToBin(x):
    y = ''
    temp = x
    while temp != 0:
        if temp % 2 == 0:
            y = '0' + y
        else:
            y = '1' + y
        temp = temp // 2
    return y


def linearSearch(Liste, Element):
    for i in range(len(Liste)):
        if Liste[i] == Element:
            return True
    return False


def selectionSort(Liste):
    for i in range(len(Liste)):
        minIndex = i
        for j in range(i + 1, len(Liste)):
            if Liste[j] < Liste[minIndex]:
                minIndex = j
        temp = Liste[i]
        Liste[i] = Liste[minIndex]
        Liste[minIndex] = temp


def insertionSort(Liste):
    for i in range(1, len(Liste)):
        j = i
        while j > 0 and Liste[j] < Liste[j - 1]:
            temp = Liste[j]
            Liste[j] = Liste[j - 1]
            Liste[j - 1] = temp
            j = j - 1


def bubbleSort(Liste):
    for i in range(len(Liste)):
        for j in range(len(Liste) - 1):
            if Liste[j] > Liste[j + 1]:
                temp = Liste[j]
                Liste[j] = Liste[j + 1]
                Liste[j + 1] = temp


def binarySearch(Liste, Element):
    left = 0
    right = len(Liste) - 1
    while left <= right:
        middle = (left + right) // 2
        if Liste[middle] == Element:
            return True
        elif Liste[middle] > Element:
            right = middle - 1
        else:
            left = middle + 1
    return False


def nestedLoops(n):
    anzahl = 0
    i = 0
    while i < n:
        j = 0
        while j < i:
            anzahl = anzahl + 1
            anzahl = anzahl * 2
            j = j + 1
        i = i + 1
    return anzahl


REFERENCE_FUNCTIONS = {
    "decToBin": decToBin, "linearSearch": linearSearch, "selectionSort": selectionSort,
    "insertionSort": insertionSort, "bubbleSort": bubbleSort, "binarySearch": binarySearch,
    "nestedLoops": nestedLoops,
}


def _norm_name(s: str) -> str:
    return re.sub(r"[\s_\-]", "", s).lower()


_ALIASES = {"dectobin": "decToBin", "binary": "decToBin", "binaerdarstellung": "decToBin",
            "linearsearch": "linearSearch", "lineraesuche": "linearSearch",
            "selectionsort": "selectionSort", "insertionsort": "insertionSort", "bubblesort": "bubbleSort",
            "binarysearch": "binarySearch", "binaresuche": "binarySearch", "binäresuche": "binarySearch",
            "nestedloops": "nestedLoops"}
_REF_INDEX = {_norm_name(k): k for k in REFERENCE_CODE}
_REF_INDEX.update(_ALIASES)


def reference_code(name: str) -> str:
    """Quelltext eines Referenzalgorithmus: ``reference_code("selectionSort")``."""
    key = _REF_INDEX.get(_norm_name(str(name)))
    if key is None:
        raise ValueError(f"Unbekannter Referenzalgorithmus '{name}'. Verfügbar: {', '.join(REFERENCE_CODE)}")
    return REFERENCE_CODE[key]


def reference_algorithms() -> list[str]:
    """Namen der eingebauten Referenzalgorithmen (Serie 5)."""
    return list(REFERENCE_CODE)


# ==========================================================================
# Funktionsterme sicher einlesen (kein eval): Strings in n bzw. x -> SymPy
# ==========================================================================

_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def _sym(var: str = "n") -> sp.Symbol:
    if not isinstance(var, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,3}", var):
        raise ValueError("Der Variablenname muss ein kurzer Name wie 'n' oder 'x' sein")
    return sp.Symbol(var, positive=True)


_LOG_WRAP = {"none": None, "ceil": sp.ceiling, "floor": sp.floor, "round": lambda v: sp.floor(v + sp.Rational(1, 2))}


def parse_term(text, var: str = "n", log_mode: str = "none") -> sp.Expr:
    """
    Liest einen Funktionsterm (Text) in der Variablen ``var`` sicher ein — ohne ``eval``.

    Erlaubt: Zahlen, ``+ - * / ** ^``, ``n!``, Klammern und die Funktionen ``log(x)`` (natürlich),
    ``log(x, b)``, ``ln``, ``log2``, ``log10``, ``lg``, ``sqrt``, ``exp``, ``factorial``, ``floor``,
    ``ceil``, ``abs``, die Konstanten ``pi`` und ``E``. ``log_mode`` rundet jeden Logarithmus
    (``"ceil"`` = aufrunden, wie in der Vorlesung bei n·log₂ n mit n = 5 → 5·3 = 15).

    >>> parse_term("n*log(n,2)")
    n*log(n)/log(2)
    """
    if log_mode not in _LOG_WRAP:
        raise ValueError("log_mode muss 'none', 'ceil', 'floor' oder 'round' sein")
    sym = _sym(var)
    if isinstance(text, sp.Basic):
        return text
    if isinstance(text, bool):
        raise ValueError("Funktionsterm muss ein Text oder eine Zahl sein")
    if isinstance(text, int):
        return sp.Integer(text)
    if not isinstance(text, str):
        raise ValueError("Funktionsterm muss ein Text wie 'n*log(n,2)' sein")
    s = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+", lambda m: "**" + m.group().translate(_SUPERSCRIPTS), text.strip())
    if not s or len(s) > MAX_TERM_LEN:
        raise ValueError(f"Funktionsterm leer oder länger als {MAX_TERM_LEN} Zeichen")
    s = s.replace("^", "**").replace("·", "*").replace("×", "*").replace("−", "-").replace("÷", "/")
    s = re.sub(r"(?<![A-Za-z_\d.])(\d+)\s*(?=[A-Za-z(])", r"\1*", s)                       # 3n -> 3*n
    s = re.sub(r"(?<![\w.])([A-Za-z]\w*|\d+|\))\s*!", lambda m: f"factorial({m.group(1)})" if m.group(1) != ")" else m.group(0) + "!", s)
    if "!" in s:
        raise ValueError("Fakultät bitte als factorial(...) schreiben")
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError:
        raise ValueError(f"Der Term '{text}' kann nicht gelesen werden") from None
    wrap = _LOG_WRAP[log_mode]
    consts = {"pi": sp.pi, "E": sp.E}

    def logf(*a):
        if len(a) not in (1, 2):
            raise ValueError("log(...) braucht ein oder zwei Argumente")
        v = sp.log(*a)
        return wrap(v) if wrap else v

    funcs = {
        "log": logf, "ln": lambda a: logf(a), "log2": lambda a: logf(a, 2), "lg": lambda a: logf(a, 10),
        "log10": lambda a: logf(a, 10), "sqrt": sp.sqrt, "exp": sp.exp, "factorial": sp.factorial,
        "floor": sp.floor, "ceil": sp.ceiling, "ceiling": sp.ceiling, "abs": sp.Abs,
    }

    def conv(node, depth=0):
        if depth > 40:
            raise ValueError("Term zu tief verschachtelt")
        if isinstance(node, ast.Expression):
            return conv(node.body, depth + 1)
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError("Nur Zahlen sind als Konstanten erlaubt")
            if isinstance(v, int):
                if v.bit_length() > 200:
                    raise ValueError("Zahl zu gross")
                return sp.Integer(v)
            return sp.Rational(str(v))
        if isinstance(node, ast.Name):
            if node.id == var:
                return sym
            if node.id in consts:
                return consts[node.id]
            raise ValueError(f"Unbekannter Name '{node.id}' im Term (erlaubt: {var}, pi, E und die Funktionen "
                             "log, ln, log2, log10, sqrt, exp, factorial, floor, ceil, abs)")
        if isinstance(node, ast.UnaryOp):
            v = conv(node.operand, depth + 1)
            if isinstance(node.op, ast.USub):
                return -v
            if isinstance(node.op, ast.UAdd):
                return v
            raise ValueError("Operator nicht erlaubt")
        if isinstance(node, ast.BinOp):
            a, b = conv(node.left, depth + 1), conv(node.right, depth + 1)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                if b == 0:
                    raise ValueError("Division durch 0")
                return a / b
            if isinstance(node.op, ast.Pow):
                if b.is_Number and abs(b) > 10_000:
                    raise ValueError("Exponent zu gross")
                if a.is_Number and b.is_Number and a not in (0, 1, -1) and abs(b) * math.log2(abs(float(a)) + 1) > 50_000:
                    raise ValueError("Potenz zu gross")
                return a ** b
            raise ValueError("Operator nicht erlaubt (erlaubt: + - * / **)")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in funcs or node.keywords:
                raise ValueError("Funktion nicht erlaubt (erlaubt: log, ln, log2, log10, lg, sqrt, exp, "
                                 "factorial, floor, ceil, abs)")
            args = [conv(a, depth + 1) for a in node.args]
            if node.func.id != "log" and len(args) != 1:
                raise ValueError(f"{node.func.id}(...) braucht genau ein Argument")
            if node.func.id == "factorial" and args and args[0].is_Number and (args[0] > 5000 or args[0] < 0):
                raise ValueError("Fakultät nur für 0 … 5000 mit konkretem Zahlenwert")
            try:
                return funcs[node.func.id](*args)
            except TypeError:
                raise ValueError(f"Falsche Argumente für {node.func.id}(...)") from None
        raise ValueError("Der Term enthält etwas Unerlaubtes (nur Zahlen, Variable, + - * / ** und Funktionen)")

    expr = conv(tree)
    if expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise ValueError("Der Term ist nicht definiert")
    return expr


# ==========================================================================
# Formeln aus Messwerten exakt anpassen
# ==========================================================================

def _floor_log2(n: int) -> int:
    return n.bit_length() - 1


def _pretty(expr, var: str = "n") -> str:
    """Term als lesbarer Text: 5·n² − 2·n − 3, 7·⌊log₂ x⌋ + 11."""
    s = sp.sstr(expr)
    s = s.replace(f"floor(log({var})/log(2))", f"⌊log₂ {var}⌋").replace(f"floor(log({var}, 2))", f"⌊log₂ {var}⌋")
    s = re.sub(r"\*\*(\d)", lambda m: m.group(1).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")), s)
    s = s.replace("**", "^").replace("*", "·").replace(" - ", " − ")
    return s


@dataclass
class Fit:
    """
    Exakt angepasste Formel T(n) (``expr`` in SymPy), gültig für n ≥ ``valid_from``.

    ``kind``: 'konstant', 'Polynom' oder 'Floor-Log' (a + b·⌊log₂ n⌋). ``exceptions`` enthält die
    Messwerte unterhalb von ``valid_from`` (z. B. n = 0). ``checked`` = Zahl der Kontrollpunkte,
    die zusätzlich zu den Stützstellen exakt stimmen.
    """
    kind: str
    expr: sp.Expr
    var: sp.Symbol
    valid_from: int
    exceptions: dict
    checked: int
    _coef: tuple = ()

    def __call__(self, n: int):
        if n < self.valid_from:
            return self.exceptions.get(n)
        v = self.expr.subs(self.var, n)
        return int(v) if v.is_Integer else v

    @property
    def text(self) -> str:
        return _pretty(self.expr, self.var.name)

    @property
    def degree(self) -> int:
        return sp.Poly(self.expr, self.var).degree() if self.kind != "Floor-Log" else 0


def _as_frac(v) -> Fraction:
    return Fraction(int(v.p), int(v.q)) if isinstance(v, sp.Rational) else Fraction(v)


def fit_formula(points: dict, var: str = "n", max_degree: int = 4) -> Fit | None:
    """
    Passt eine Formel exakt an Messwerte an und verifiziert sie an den übrigen Punkten.

    ``points`` = {n: Wert}. Versucht der Reihe nach konstant, a + b·⌊log₂ n⌋ und Polynome vom Grad
    1 … ``max_degree`` (Lagrange-Interpolation mit exakten Brüchen); lässt zuerst keinen, dann die
    kleinsten n aus (Sonderfälle wie n = 0). Es braucht mindestens 2 Kontrollpunkte, sonst ``None``.

    >>> fit_formula({n: 5*n*n - 2*n - 3 for n in range(2, 12)}).text
    '5·n² − 2·n − 3'
    """
    if not points:
        raise ValueError("Keine Messwerte")
    sym = _sym(var)
    # ponytail: nur konstant, a + b·⌊log₂ n⌋ und Polynome; das echte Maximum von Selection Sort ist ein
    # Quasi-Polynom (⌊n²/4⌋) und bekommt bewusst keine Formel
    pts = sorted((int(k), Fraction(v)) for k, v in points.items())
    for start in range(0, min(4, max(1, len(pts) - 3))):
        sub = pts[start:]
        ex = {n: (int(v) if v.denominator == 1 else v) for n, v in pts[:start]}
        n0 = sub[0][0]
        # konstant
        if len(sub) >= 3 and len({v for _, v in sub}) == 1:
            c = sub[0][1]
            return Fit("konstant", sp.Rational(c.numerator, c.denominator), sym, n0, ex, len(sub) - 1)
        # a + b * floor(log2 n)
        if n0 >= 1 and len(sub) >= 4:
            (n1, y1) = sub[0]
            other = next(((n, y) for n, y in sub if _floor_log2(n) != _floor_log2(n1)), None)
            if other is not None:
                b = (other[1] - y1) / (_floor_log2(other[0]) - _floor_log2(n1))
                a = y1 - b * _floor_log2(n1)
                if all(a + b * _floor_log2(n) == y for n, y in sub) and b != 0:
                    expr = (sp.Rational(a.numerator, a.denominator)
                            + sp.Rational(b.numerator, b.denominator) * sp.floor(sp.log(sym, 2)))
                    return Fit("Floor-Log", expr, sym, n0, ex, len(sub) - 2)
        # Polynom
        for d in range(1, max_degree + 1):
            if len(sub) < d + 3:
                break
            base = sub[:d + 1]
            poly = sp.expand(sp.interpolate([(n, sp.Rational(v.numerator, v.denominator)) for n, v in base], sym))
            coeffs = sp.Poly(poly, sym).all_coeffs()
            cf = [_as_frac(c) for c in coeffs]
            ok = True
            for n, y in sub:
                val = Fraction(0)
                for c in cf:
                    val = val * n + c
                if val != y:
                    ok = False
                    break
            if ok:
                return Fit("Polynom", poly, sym, n0, ex, len(sub) - d - 1)
    return None


# ==========================================================================
# Eingabefamilien für Worst-/Best-/Average-Case
# ==========================================================================

import random as _random

_LIST_NAMES = {"liste", "list", "lst", "l", "a", "arr", "array", "xs", "daten", "data", "zahlen", "werte"}
_ELEM_NAMES = {"element", "elem", "key", "target", "ziel", "wert", "value", "suchwert", "e", "k", "needle",
               "gesucht"}
_SIZE_NAMES = {"n", "x", "m", "zahl", "anzahl", "z"}


def _sort_inputs(pname):
    def family(n):
        asc = list(range(n))
        cands = [asc, asc[::-1], [0] * n]
        if n >= 3:
            zig, lo, hi = [], 0, n - 1
            while lo <= hi:
                zig.append(lo)
                lo += 1
                if lo <= hi:
                    zig.append(hi)
                    hi -= 1
            cands += [zig, zig[::-1], asc[1:] + asc[:1], asc[-1:] + asc[:-1]]
        for seed in range(4):
            perm = asc[:]
            _random.Random(1000 * n + seed).shuffle(perm)
            cands.append(perm)
        seen, out = set(), []
        for c in cands:
            if tuple(c) not in seen:
                seen.add(tuple(c))
                out.append({pname: c})
        return out

    def exhaustive(n):
        return ({pname: list(p)} for p in itertools.permutations(range(n)))
    return family, exhaustive


def _search_inputs(lname, ename):
    def family(n):
        lst = [2 * i for i in range(n)]
        elems = [lst[k] for k in range(n)] + [-1] + [2 * k + 1 for k in range(n)]
        return [{lname: lst[:], ename: e} for e in elems]
    return family, None


def _scalar_inputs(pname):
    return (lambda n: [{pname: n}]), None


@dataclass
class _Spec:
    name: str
    var: str
    default_n: list
    max_n: int
    family: object
    exhaustive: object
    worst_desc: str
    best_desc: str
    size_note: str = ""


def _generic_spec(prog) -> _Spec:
    if prog.fname is None:
        raise ValueError("Für diese Analyse braucht der Code eine Funktion (def …), deren Parameter die Eingabe sind")
    params = prog.params
    low = [p.lower() for p in params]
    if len(params) == 1 and low[0] in _LIST_NAMES:
        fam, exh = _sort_inputs(params[0])
        return _Spec(prog.fname, "n", list(range(0, 13)), 120, fam, exh,
                     "Gemessen an Familien wie aufsteigend, absteigend, gleiche Werte, Zickzack, gemischt "
                     "(und Brute-Force über alle Permutationen für kleine n).",
                     "Kleinster Wert über dieselben Familien.")
    if len(params) == 2 and low[0] in _LIST_NAMES and low[1] in _ELEM_NAMES:
        fam, exh = _search_inputs(params[0], params[1])
        return _Spec(prog.fname, "n", list(range(0, 17)), 120, fam, exh,
                     "Sortierte Liste 0, 2, 4, …; gesucht wird jedes enthaltene Element und jede Lücke "
                     "(auch ausserhalb).", "Kleinster Wert über dieselben Suchwerte.")
    if len(params) == 1 and low[0] in _SIZE_NAMES:
        fam, exh = _scalar_inputs(params[0])
        return _Spec(prog.fname, params[0], list(range(0, 13)), 150, fam, exh,
                     "Für jede Eingabe gibt es genau einen Ablauf.", "Dito.")
    raise ValueError(
        f"Für die Parameter ({', '.join(params)}) kann ich keine Eingaben selbst bauen. Gib "
        "inputs=funktion an: n -> Liste von Eingaben (Wörterbüchern), z. B. "
        "inputs=lambda n: [{'Liste': list(range(n))}].")


def _ref_spec(name) -> _Spec:
    sort_f, sort_e = _sort_inputs("Liste")
    srch_f, srch_e = _search_inputs("Liste", "Element")
    if name == "decToBin":
        return _Spec(name, "x", list(range(0, 33)), 10 ** 12, *_scalar_inputs("x"),
                     "Für jedes x gibt es genau einen Ablauf; if- und else-Zweig kosten gleich viel "
                     "(Test 2 + Zuweisung mit Addition 2 Schritte).", "Dito: Worst und Best fallen zusammen.")
    if name == "nestedLoops":
        return _Spec(name, "n", list(range(0, 13)), 150, *_scalar_inputs("n"),
                     "Keine Verzweigung: für jedes n gibt es genau einen Ablauf.", "Dito.")
    if name == "linearSearch":
        return _Spec(name, "n", list(range(0, 17)), 120, srch_f, srch_e,
                     "Element nicht in der Liste: alle n Durchläufe, danach return False.",
                     "Element an der ersten Stelle: ein Durchlauf, dann return True.")
    if name == "binarySearch":
        return _Spec(name, "n", list(range(0, 17)), 120, srch_f, srch_e,
                     "Element nicht enthalten (oder erst im letzten Halbierungsschritt gefunden): jeder "
                     "Durchlauf halbiert die Liste, also ⌊log₂ n⌋ + 1 Durchläufe.",
                     "Element liegt genau in der Mitte: ein Durchlauf.", "geordnete Liste 0, 2, 4, …")
    if name == "selectionSort":
        return _Spec(name, "n", list(range(0, 13)), 120, sort_f, sort_e,
                     "Vorlesungs-Zählung: minIndex = j wird in jedem Durchlauf der inneren Schleife ausgeführt "
                     "(obere Schranke). Das tatsächliche Maximum ist kleiner (Brute-Force unten).",
                     "Liste bereits aufsteigend sortiert: der if-Test ist nie wahr.")
    if name == "insertionSort":
        return _Spec(name, "n", list(range(0, 13)), 120, sort_f, sort_e,
                     "Absteigend sortierte Liste: jedes Element wandert bis ganz nach vorn (j bis 0).",
                     "Bereits aufsteigend sortierte Liste: der while-Test scheitert sofort.")
    if name == "bubbleSort":
        return _Spec(name, "n", list(range(0, 13)), 120, sort_f, sort_e,
                     "Absteigend sortierte Liste: die meisten Vertauschungen (n(n−1)/2 Inversionen). "
                     "Vorlesungs-Schranke: Vertauschung in jedem Durchlauf (oben).",
                     "Bereits sortierte Liste: nie eine Vertauschung, die Schleifen laufen trotzdem komplett.")
    raise ValueError(name)


class _Program:
    def __init__(self, code):
        self.code, self.tree = _prepare(code)
        self.defs = _validate(self.tree)
        only_defs = all(isinstance(s, ast.FunctionDef) for s in self.tree.body)
        self.fname = list(self.defs)[-1] if (self.defs and only_defs) else None
        self.params = [a.arg for a in self.defs[self.fname].args.args] if self.fname else []

    def run(self, inputs, rules, branch="actual", max_steps=DEFAULT_MAX_STEPS):
        it, value, arguments, fname, used = _run(self.code, self.tree, self.defs, (), None, inputs, rules,
                                                 branch, max_steps)
        return it, value, used


def _resolve(algorithm):
    if not isinstance(algorithm, str):
        raise ValueError("algorithm muss ein Name (z. B. 'selectionSort') oder Quelltext sein")
    key = _REF_INDEX.get(_norm_name(algorithm)) if "\n" not in algorithm else None
    if key:
        prog = _Program(REFERENCE_CODE[key])
        return _ref_spec(key), prog
    prog = _Program(algorithm)
    return _generic_spec(prog), prog


def _inp_str(inp: dict, n=64) -> str:
    s = ", ".join(f"{k}={v}" for k, v in inp.items())
    return s if len(s) <= n else s[:n - 1] + "…"


def _check_n_values(n_values, spec):
    if n_values is None:
        return list(spec.default_n)
    if isinstance(n_values, (int,)) and not isinstance(n_values, bool):
        n_values = [n_values]
    try:
        ns = sorted({int(v) for v in n_values})
    except (TypeError, ValueError):
        raise ValueError("n_values muss eine Liste ganzer Zahlen sein") from None
    if not ns or len(ns) > MAX_N_VALUES:
        raise ValueError(f"n_values: 1 bis {MAX_N_VALUES} verschiedene Werte erwartet")
    if ns[0] < 0 or ns[-1] > spec.max_n:
        raise ValueError(f"n_values: Werte müssen zwischen 0 und {spec.max_n} liegen")
    return ns


# ==========================================================================
# Worst-/Best-Case
# ==========================================================================

def _fmt_formula(fit: Fit | None, var="n") -> str:
    return "—" if fit is None else fit.text


@dataclass
class _Run:
    steps: int
    inp: dict
    exec_: dict
    lsteps: dict


def _measure(prog, inp, rules, branch, max_steps) -> _Run:
    it, _, used = prog.run(inp, rules, branch, max_steps)
    return _Run(it.steps, used, dict(it.exec_line), dict(it.by_line))


class CaseResult(Result):
    """
    Worst-Case oder Best-Case eines Algorithmus: Messwerte und geschlossene Formel.

    Attribute: ``case`` ('worst'|'best'), ``algorithm``, ``var``, ``n_values``,
    ``measured`` {n: Schritte} (echtes Maximum/Minimum über alle geprüften Eingaben),
    ``inputs`` {n: Eingabe, die es erreicht}, ``formula`` (SymPy oder ``None``), ``fit`` (``Fit``),
    ``bound_measured``/``bound_fit`` (Vorlesungs-Schranke, nur Worst-Case), ``theta`` (Klasse),
    ``lines`` (Formeln je Zeile), ``exhaustive_up_to``.
    """

    def __init__(self):
        super().__init__()

    def at(self, n: int):
        """Schrittzahl für n: aus der Formel (bzw. Messwert, falls n gemessen wurde)."""
        if n in self.measured:
            return self.measured[n]
        fit = self.fit or self.bound_fit
        if fit is None:
            raise ValueError("Keine Formel vorhanden")
        return fit(n)


def _case(algorithm, case, n_values, inputs, rules, exhaustive_up_to, max_steps):
    if case not in ("worst", "best"):
        raise ValueError("case muss 'worst' oder 'best' sein")
    if isinstance(exhaustive_up_to, bool) or not isinstance(exhaustive_up_to, int) or \
            not -1 <= exhaustive_up_to <= MAX_EXHAUSTIVE_N:
        raise ValueError(f"exhaustive_up_to muss zwischen -1 und {MAX_EXHAUSTIVE_N} liegen "
                         "(Brute-Force über alle Permutationen, ab 8 dauert es einige Sekunden)")
    spec, prog = _resolve(algorithm)
    rules = _check_rules(rules)
    ns = _check_n_values(n_values, spec)
    # ponytail: Brute-Force über alle Permutationen bis exhaustive_up_to statt eines Beweises für den echten Worst Case
    pick = max if case == "worst" else min
    best_runs: dict[int, _Run] = {}
    bound_runs: dict[int, _Run] = {}
    total_runs = 0
    for n in ns:
        fam = list(inputs(n)) if inputs is not None else list(spec.family(n))
        if not fam:
            raise ValueError(f"inputs({n}) liefert keine Eingaben")
        seen, chosen = set(), None
        cand = list(fam)
        if inputs is None and spec.exhaustive is not None and n <= exhaustive_up_to:
            cand += list(spec.exhaustive(n))
        for c in cand:
            key = repr(sorted(c.items()))
            if key in seen:
                continue
            seen.add(key)
            total_runs += 1
            if total_runs > 100_000:
                raise ValueError("Zu viele Eingaben zu prüfen (> 100000); verkleinere n_values oder exhaustive_up_to")
            r = _measure(prog, c, rules, "actual", max_steps)
            if chosen is None or (r.steps > chosen.steps if case == "worst" else r.steps < chosen.steps):
                chosen = r
        best_runs[n] = chosen
        if case == "worst":
            bch = None
            for c in fam:
                r = _measure(prog, c, rules, "max", max_steps)
                if bch is None or r.steps > bch.steps:
                    bch = r
            bound_runs[n] = bch

    res = CaseResult()
    res.case, res.algorithm, res.var, res.n_values = case, spec.name, spec.var, ns
    res.rules, res.exhaustive_up_to = rules, exhaustive_up_to
    res.measured = {n: best_runs[n].steps for n in ns}
    res.inputs = {n: best_runs[n].inp for n in ns}
    var = spec.var
    res.fit = fit_formula(res.measured, var) if len(ns) >= 7 else None
    res.bound_measured = {n: bound_runs[n].steps for n in ns} if case == "worst" else {}
    res.bound_fit = fit_formula(res.bound_measured, var) if (case == "worst" and len(ns) >= 7) else None
    differs = case == "worst" and res.bound_measured != res.measured
    res.bound_differs = differs
    # Primäre Formel: die exakte; falls es keine gibt, die Vorlesungs-Schranke
    res.formula_is_bound = False
    primary_fit = res.fit
    primary_runs = best_runs
    if primary_fit is None and res.bound_fit is not None:
        primary_fit, primary_runs, res.formula_is_bound = res.bound_fit, bound_runs, True
    res.formula = primary_fit.expr if primary_fit else None
    res.formula_text = primary_fit.text if primary_fit else None
    res.valid_from = primary_fit.valid_from if primary_fit else None
    res.theta = classify(primary_fit.expr, var).notation if primary_fit else None
    res.lines = _line_formulas(prog, primary_runs, ns, var)
    _case_steps(res, spec, prog, primary_fit, primary_runs, inputs is not None)
    return res


def _line_formulas(prog, runs, ns, var):
    lines = sorted({ln for r in runs.values() for ln in (set(r.exec_) | set(r.lsteps)) if ln > 0})
    src = prog.code.split("\n")
    out = []
    for ln in lines:
        ex = {n: runs[n].exec_.get(ln, 0) for n in ns}
        st = {n: runs[n].lsteps.get(ln, 0) for n in ns}
        fe = fit_formula(ex, var) if len(ns) >= 7 else None
        fs = fit_formula(st, var) if len(ns) >= 7 else None
        nlast = ns[-1]
        per = (Fraction(st[nlast], ex[nlast]) if ex[nlast] else None)
        const_per = per is not None and all((Fraction(st[n], ex[n]) == per) for n in ns if ex[n])
        out.append({"line": ln, "code": src[ln - 1].strip(), "exec": ex, "steps": st, "exec_fit": fe,
                    "steps_fit": fs, "per": (int(per) if const_per and per.denominator == 1 else None)})
    return out


def _case_steps(res, spec, prog, fit, runs, custom_inputs):
    var, ns, case = res.var, res.n_values, res.case
    name = spec.name
    word = "Worst-Case" if case == "worst" else "Best-Case"
    res.title = f"{word}-Analyse: {name}({var})" if spec.var in ("x",) else f"{word}-Analyse: {name}"
    pick = "maximale" if case == "worst" else "kleinste"
    res.steps.append(Step(
        "Was wird bestimmt?",
        (f"Bei einer Worst-Case-Analyse wird in Abhängigkeit von der Grösse {var} des Inputs die maximale Anzahl "
         "Rechnungsschritte angegeben, die zur Berechnung nötig sein können."
         if case == "worst" else
         f"Best-Case: die kleinste Anzahl Rechnungsschritte, die für alle Eingaben der Grösse {var} nötig sind."),
        rule="Definition 1 (Worst-Case)" if case == "worst" else "Best-Case"))
    res.steps.append(Step(
        "Zählregeln",
        "Zuweisung, Vergleich und Rechenoperation je 1 Schritt; jeder for-Durchlauf und das return je 1 Schritt; "
        "der letzte, fehlschlagende while-Test zählt mit (siehe COUNTING_RULES)."
        + ("" if res.rules == default_rules() else " Achtung: veränderte Zählregeln: "
           + ", ".join(f"{k}={v}" for k, v in res.rules.items() if v != COUNTING_RULES[k]["steps"]) + ".")))
    desc = spec.worst_desc if case == "worst" else spec.best_desc
    res.steps.append(Step("Schlechtesten Fall wählen" if case == "worst" else "Besten Fall wählen", desc,
                          rule="Eingabe konstruieren"))
    # Messwerte
    rows = [[str(n), str(res.measured[n])] + ([str(res.bound_measured[n])] if res.bound_differs else [])
            + [_inp_str(res.inputs[n])] for n in ns]
    header = [var, f"{pick} Schritte"] + (["Schranke (jeder if teuer)"] if res.bound_differs else []) + ["Eingabe"]
    res.steps.append(Step(
        "Messen", f"Der Interpreter zählt die Schritte für jede Eingabe der Grösse {var}; behalten wird die "
        f"{pick}." + (f" Für {var} ≤ {res.exhaustive_up_to} werden alle Permutationen von 0 … {var}−1 durchprobiert "
                      "(Brute-Force), darüber die oben genannten Familien." if spec.exhaustive and res.exhaustive_up_to >= 0
                      and not custom_inputs else ""),
        table=Table(header, rows)))
    # Zeilen
    if res.lines and any(l["exec_fit"] is not None for l in res.lines):
        lrows = []
        for l in res.lines:
            ef, sf = l["exec_fit"], l["steps_fit"]
            lrows.append([str(l["line"]), l["code"], ef.text if ef else "—",
                          str(l["per"]) if l["per"] is not None else "—", sf.text if sf else "—"])
        res.steps.append(Step(
            "Zeile für Zeile", f"Wie oft wird jede Zeile ausgeführt (als Formel in {var}), und was kostet eine "
            "Ausführung? Bei Schleifenköpfen sind die Ausführungen die Tests bzw. Durchläufe.",
            table=Table(["Zeile", "Code", "Ausführungen", "Schritte je Ausf.", "Schritte gesamt"], lrows),
            rule="Schritte je Zeile × Anzahl Ausführungen",
            note="»—« heisst: hängt von den Werten der Eingabe ab (z. B. if/else-Zweig) und hat keine einfache "
                 "Formel; nur die Summe zählt." if any(l["exec_fit"] is None for l in res.lines) else ""))
    # Formel
    if fit is not None:
        extra = ""
        if fit.kind == "Floor-Log":
            a0, b0 = fit.expr.subs(sp.floor(sp.log(fit.var, 2)), 0), fit.expr.coeff(sp.floor(sp.log(fit.var, 2)))
            extra = (f" Die Zweierpotenzen entscheiden, wie oft die Schleife läuft: Mit {var} = 2^k gilt "
                     f"⌊log₂ {var}⌋ = k — jede Verdopplung von {var} bringt einen Durchlauf mehr, also "
                     f"⌊log₂ {var}⌋ + 1 Durchläufe. Umgeschrieben: T = {b0}·(⌊log₂ {var}⌋ + 1) + {a0 - b0} "
                     f"({b0} Schritte pro Durchlauf; {a0 - b0} Schritte ausserhalb: Initialisierung, "
                     "letzter fehlschlagender Test, return).")
        elif fit.kind == "Polynom" and fit.degree == 2:
            extra = (" Eine Doppelschleife liefert eine Summe 1 + 2 + … + (n−1): Gauss-Trick, "
                     "s = (n−1)·n/2 (siehe gauss_sum).")
        res.steps.append(Step(
            "Formel", f"Ansatz je nach Verlauf: {fit.kind}. Exakt angepasst und an {fit.checked} weiteren Werten von "
            f"{var} bestätigt (gilt für {var} ≥ {fit.valid_from})." + extra,
            items=[(f"T({var})", fit.text)] + [(f"T({n})", v) for n, v in fit.exceptions.items()],
            rule="exakte Anpassung (Interpolation) + Probe",
            note=("Obere Schranke nach Vorlesungs-Zählung (jeder if-Zweig teuer); das echte Maximum hat keine "
                  "einfache Formel (siehe Messwerte)." if res.formula_is_bound else "")))
        if res.bound_differs and not res.formula_is_bound and res.bound_fit is not None:
            res.steps.append(Step("Vorlesungs-Schranke (jeder if teuer)",
                                  "Zählt man — wie in der Vorlesung — in jedem Durchlauf den teureren Zweig, "
                                  "ergibt sich eine obere Schranke.",
                                  items=[(f"T_max({var})", res.bound_fit.text)], rule="Worst-Case-Zuschlag"))
        res.steps.append(Step("Wachstumsklasse", items=[("Klasse", res.theta)], rule="asymptotische Analyse"))
        res.highlights = [(f"T_{case}({var})", fit.text), ("Klasse", res.theta)]
    else:
        res.warnings.append("Keine Formel gefunden (zu wenige n-Werte oder kein Polynom/⌊log₂ n⌋-Verlauf): "
                            "es gelten nur die Messwerte.")
        res.highlights = [(f"T_{case}({n})", res.measured[n]) for n in ns[-3:]]
    if res.bound_differs and res.bound_fit is not None and not res.formula_is_bound:
        res.highlights.append((f"Schranke Vorlesung T_max({var})", res.bound_fit.text))
    if res.formula_is_bound:
        res.warnings.append("Die Formel ist die Vorlesungs-Schranke (obere Schranke); das tatsächliche Maximum "
                            "liegt darunter und folgt keinem einfachen Polynom (Messwerte oben).")
    if spec.exhaustive and res.exhaustive_up_to < max(ns) and not custom_inputs and case == "worst" \
            and spec.name in ("selectionSort", "insertionSort", "bubbleSort"):
        res.warnings.append(f"Für {var} > {res.exhaustive_up_to} wurden nur ausgewählte Eingaben geprüft "
                            "(lokale Familien); die Brute-Force-Absicherung gilt bis dahin.")


def worst_case(algorithm, n_values=None, *, inputs=None, rules=None, exhaustive_up_to: int = 6,
               max_steps: int = DEFAULT_MAX_STEPS) -> CaseResult:
    """
    Worst-Case-Analyse: grösste Schrittzahl über alle Eingaben der Grösse n — gemessen und als Formel.

    ``algorithm``   Name eines Referenzalgorithmus (``"selectionSort"``, ``"binarySearch"``, ``"decToBin"`` …)
                    oder Quelltext einer Funktion in der Python-Teilmenge.
    ``n_values``    Eingabegrössen (Standard je Algorithmus); die Formel braucht mindestens 7 Werte.
    ``inputs``      eigene Funktion ``n -> [ {Parameter: Wert}, … ]`` (sonst werden Eingaben passend zu den
                    Parameternamen gebaut: Liste / Liste + Element / Zahl).
    ``exhaustive_up_to``  bis zu diesem n werden alle Permutationen durchprobiert (höchstens 8).

    >>> r = worst_case("insertionSort")
    >>> r.formula_text
    '5·n² − 2·n − 3'
    >>> worst_case("decToBin").formula_text
    '7·⌊log₂ x⌋ + 11'
    """
    return _case(algorithm, "worst", n_values, inputs, rules, exhaustive_up_to, max_steps)


def best_case(algorithm, n_values=None, *, inputs=None, rules=None, exhaustive_up_to: int = 6,
              max_steps: int = DEFAULT_MAX_STEPS) -> CaseResult:
    """
    Best-Case: die kleinste Schrittzahl, die für alle Eingaben der Grösse n nötig ist (Serie 5, Aufgaben 4–7).

    >>> best_case("insertionSort").formula_text
    '5·n − 5'
    """
    return _case(algorithm, "best", n_values, inputs, rules, exhaustive_up_to, max_steps)


# ==========================================================================
# Average-Case
# ==========================================================================

class AverageResult(Result):
    """
    Durchschnittlicher Aufwand E(X) = Σ p(a)·X(a) (Zusammenfassung »Durchschnittliche Komplexität«).

    ``values`` {n: E als exakter Bruch (SymPy Rational)}, ``value`` (bei einem n), ``cases`` {n: [(Eingabe,
    p, Schritte)]}.
    """


def average_case(algorithm, n=5, *, inputs=None, weights=None, rules=None,
                 max_steps: int = DEFAULT_MAX_STEPS) -> AverageResult:
    """
    Durchschnittlicher Rechenaufwand, exakt: E(X) = Σ p(a)·X(a).

    Standard: alle Eingaben der Grösse n gleich wahrscheinlich — bei Sortierern alle Permutationen von
    0 … n−1 (n ≤ 8), bei Suchen alle 2n+1 Suchwerte (n Treffer und n+1 Lücken). Mit ``inputs=[{…}, …]``
    und ``weights=[p, …]`` (Brüche, Summe = 1 oder beliebig, wird normiert) eine eigene Verteilung.
    ``n`` darf eine Zahl oder eine Liste sein.

    >>> average_case("linearSearch", 3).value      # (3 + 5 + 7 + 4·7) / 7
    43/7
    """
    spec, prog = _resolve(algorithm)
    rules = _check_rules(rules)
    ns = [n] if isinstance(n, int) and not isinstance(n, bool) else list(n)
    if not ns or len(ns) > 12:
        raise ValueError("n: eine Zahl oder höchstens 12 Zahlen")
    res = AverageResult()
    res.algorithm, res.values, res.cases = spec.name, {}, {}
    rows = []
    for k in ns:
        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise ValueError("n muss eine ganze Zahl ≥ 0 sein")
        if inputs is not None:
            cases = list(inputs)
            if weights is not None:
                w = list(weights)
                if len(w) != len(cases) or any(x < 0 for x in w) or sum(w) == 0:
                    raise ValueError("weights: gleich viele nicht-negative Gewichte wie Eingaben nötig")
            else:
                w = [1] * len(cases)
        else:
            if spec.exhaustive is not None and len(prog.params) == 1:      # Sortierer: alle Permutationen
                if k > MAX_EXHAUSTIVE_N:
                    raise ValueError(f"Durchschnitt über alle Permutationen nur für n ≤ {MAX_EXHAUSTIVE_N}")
                cases = list(spec.exhaustive(k))
            else:
                if k > spec.max_n or (spec.var == "n" and k > 120):
                    raise ValueError("n zu gross")
                cases = list(spec.family(k))
            w = [1] * len(cases)
        if not cases or len(cases) > 50_000:
            raise ValueError("Keine oder zu viele Eingaben")
        wsum = sum(Fraction(x) for x in w)
        total = Fraction(0)
        detail = []
        for c, x in zip(cases, w):
            r = _measure(prog, c, rules, "actual", max_steps)
            p = Fraction(x) / wsum
            total += p * r.steps
            detail.append((r.inp, p, r.steps))
        res.values[k] = sp.Rational(total.numerator, total.denominator)
        res.cases[k] = detail
        rows.append([str(k), str(len(cases)), str(res.values[k]),
                     f"{float(total):.4f}".rstrip("0").rstrip(".")])
    res.value = res.values[ns[0]] if len(ns) == 1 else None
    res.title = f"Durchschnittlicher Aufwand: {spec.name}"
    res.summary = "E(X) = Σ p(a)·X(a) — jede Eingabe a mit ihrer Wahrscheinlichkeit p(a) und ihren Schritten X(a)."
    res.steps.append(Step("Erwartungswert", "Jede Eingabe der Grösse n wird mit p(a) = 1/(Anzahl Eingaben) "
                          "(bzw. den gegebenen Gewichten) gewichtet und die Schritte X(a) gezählt.",
                          rule="E(X) = Σ p(a)·X(a)",
                          table=Table(["n", "Eingaben", "E(X) exakt", "E(X) ≈"], rows)))
    if len(ns) == 1:
        det = res.cases[ns[0]]
        if len(det) <= 24:
            res.steps.append(Step("Alle Fälle", table=Table(
                ["Eingabe", "p(a)", "X(a)"], [[_inp_str(i), str(sp.Rational(p.numerator, p.denominator)), str(s)]
                                              for i, p, s in det])))
        res.highlights = [("E(X)", res.value)]
    else:
        res.highlights = [(f"E(X) für n = {k}", v) for k, v in res.values.items()]
    return res


# ==========================================================================
# Wachstumsklassen und Klassifikation
# ==========================================================================

@dataclass
class GrowthClass:
    """Eine Wachstumsklasse: ``term`` ist ein Term in n, ``polynomial`` heisst »real lösbar«."""
    key: str
    name: str
    notation: str
    term: str
    example: str
    polynomial: bool


_CLASSES = [
    GrowthClass("konstant", "konstant", "Θ(1)", "1", "Ein Speicherzugriff ist in Θ(1).", True),
    GrowthClass("log", "logarithmisch", "Θ(log n)", "log(n)", "Die binäre Suche in einer Liste mit n Komponenten.", True),
    GrowthClass("linear", "linear", "Θ(n)", "n", "Die lineare Suche in einer Liste mit n Komponenten.", True),
    GrowthClass("nlogn", "n log n", "Θ(n log n)", "n*log(n)", "Mergesort (John von Neumann).", True),
    GrowthClass("quad", "quadratisch", "Θ(n²)", "n**2", "Die meisten klassischen Sortierverfahren; Algorithmus B in Aufgabe 1.", True),
    GrowthClass("kub", "kubisch", "Θ(n³)", "n**3", "Naive Multiplikation zweier n×n-Matrizen.", True),
    GrowthClass("exp", "exponentiell", "Θ(2ⁿ)", "2**n", "Das SAT-Problem mit n Variablen (naives Vorgehen).", False),
    GrowthClass("fak", "faktoriell", "Θ(n!)", "factorial(n)", "Algorithmus C in Aufgabe 1: n! Rechnungsschritte.", False),
]
_SUBS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _human(v, digits=4) -> str:
    """Zahl lesbar: ganze Zahlen mit ’ als Tausendertrennung, sehr grosse in Zehnerpotenz."""
    v = sp.nsimplify(v) if isinstance(v, (int, float)) else v
    if v.is_Integer:
        iv = int(v)
        if abs(iv) < 10 ** 15:
            return f"{iv:,}".replace(",", "’")
        return _sci(sp.N(v, digits + 2), digits)
    f = sp.N(v, 30)
    if f == 0:
        return "0"
    if abs(f) >= 10 ** 15 or abs(f) < sp.Rational(1, 10 ** 6):
        return _sci(f, digits)
    s = f"{float(f):,.{digits}f}".rstrip("0").rstrip(".").replace(",", "’")
    return s


def _sci(f, digits=4) -> str:
    f = sp.N(f, 30)
    sign = "-" if f < 0 else ""
    f = abs(f)
    e = int(sp.floor(sp.log(f, 10)))
    m = f / sp.Integer(10) ** e
    ms = f"{float(m):.{digits - 1}f}".rstrip("0").rstrip(".")
    return f"{sign}{ms}·10^{e}"


def _value_at(expr, sym, n):
    v = expr.subs(sym, n)
    if not isinstance(v, sp.Number) and v.has(sp.log):
        v = sp.simplify(sp.expand_log(v, force=True))        # n·log2 n bei Zweierpotenzen exakt: 8·3 = 24
    if not isinstance(v, sp.Number):
        v = sp.N(v, 60)
    if not isinstance(v, sp.Number) or not v.is_real:
        raise ValueError(f"Der Term ist für n = {n} nicht als reelle Zahl auswertbar")
    return v


def _strip_rounding(e):
    return e.replace(lambda t: isinstance(t, (sp.floor, sp.ceiling)), lambda t: t.args[0])


class ClassifyResult(Result):
    """
    Ergebnis von ``classify``: ``theta`` (SymPy-Term der Klasse), ``notation`` ('Θ(n²)'),
    ``name`` ('quadratisch'), ``polynomial`` (True = polynomielles Wachstum, »real lösbar«), ``degree``.
    """


def _theta_name(kind, d=None):
    if kind == "poly":
        names = {0: "konstant", 1: "linear", 2: "quadratisch", 3: "kubisch"}
        return names.get(d, f"polynomiell (Grad {d})")
    return kind


def _notation(term: str, var: str) -> str:
    t = term.replace("**", "^").replace("*", " ").replace("factorial(n)", "n!").replace("log(n)", "log n")
    t = re.sub(r"\^(\d)", lambda m: m.group(1).translate(_SUBS), t).replace("^n", "ⁿ")
    return f"Θ({t.replace('n', var)})"


def classify(f, var: str = "n") -> ClassifyResult:
    """
    Ordnet eine Schrittzahl-Funktion f(n) ihrer Wachstumsklasse Θ(g) zu.

    Polynome: höchster Exponent. Sonst werden die Klassen der Reihe nach getestet: f ∈ Θ(g), wenn
    lim f/g endlich und ≠ 0 ist (Floor/Ceil werden ignoriert, Konstanten und Basen von log spielen
    keine Rolle).

    >>> classify("5*n**2 - 2*n - 3").notation
    'Θ(n²)'
    >>> classify("n*log(n,2)").name
    'n log n'
    """
    sym = _sym(var)
    expr = parse_term(f, var)
    res = ClassifyResult()
    res.f = expr
    res.title = "Wachstumsklasse bestimmen"
    res.steps.append(Step("Funktion", items=[(f"f({var})", expr)]))
    e2 = sp.expand(_strip_rounding(expr))
    if e2 != expr:
        res.steps.append(Step("Auf- und Abrunden ignorieren", "⌊·⌋ und ⌈·⌉ ändern das Wachstum nicht.",
                              items=[(f"f({var})", e2)]))
    theta, name, poly, deg = None, None, False, None
    if e2.is_polynomial(sym):
        P = sp.Poly(e2, sym)
        deg, lead = P.degree(), P.LC()
        if lead == 0:
            raise ValueError("Die Funktion ist 0")
        theta = sym ** deg
        name, poly = _theta_name("poly", deg), True
        res.steps.append(Step(
            "Dominanten Term bestimmen",
            f"Bei einem Polynom zählt der höchste Exponent: der Term {sp.sstr(lead)}·{var}^{deg} schlägt alle "
            "anderen, wenn n gross wird; Konstante Faktoren fallen weg.",
            items=[("höchster Term", lead * sym ** deg)], rule="Dominanter Term"))
    else:
        cands = []
        for k in range(0, 7):
            for j in range(0, 3):
                if k == 0 and j == 0:
                    cands.append(sp.Integer(1))
                else:
                    cands.append(sym ** k * sp.log(sym) ** j)
        cands += [sp.sqrt(sym)] + [b ** sym for b in (2, 3, 4, 5, 10)] + [sp.factorial(sym), sym ** sym]
        for g in cands:
            try:
                lim = sp.limit(sp.simplify(e2 / g), sym, sp.oo)
            except Exception:
                continue
            if lim.is_finite and lim != 0 and lim.is_real:
                theta = g
                break
        if theta is None:
            raise ValueError("Keine Standardklasse gefunden (erlaubt sind Terme aus Potenzen, log, a^n, n!)")
        res.steps.append(Step(
            "Grenzwert des Quotienten", "f ∈ Θ(g), wenn lim f(n)/g(n) endlich und ≠ 0 ist.",
            items=[(f"lim f/g (g = {sp.sstr(theta)})", lim)], rule="Quotientenkriterium für Θ"))
        key = {sp.Integer(1): "konstant", sp.log(sym): "logarithmisch", sym: "linear",
               sym * sp.log(sym): "n log n", sym ** 2: "quadratisch", sym ** 3: "kubisch",
               sp.Integer(2) ** sym: "exponentiell", sp.factorial(sym): "faktoriell"}
        poly = not (theta.has(sp.factorial) or any(isinstance(a, sp.Pow) and a.exp.has(sym)
                                                      for a in sp.preorder_traversal(theta)))
        name = key.get(theta)
        if name is None:
            name = "exponentiell" if not poly else "polynomiell"
    res.theta = theta
    res.name, res.polynomial, res.degree = name, poly, deg
    res.notation = _notation(sp.sstr(theta).replace(sp.sstr(sym), "n"), var)
    res.summary = f"f ∈ {res.notation} ({name}; {'polynomielles Wachstum: real lösbar' if poly else 'exponentielles Wachstum: real unlösbar'})"
    res.steps.append(Step("Ergebnis", items=[("Klasse", res.notation)], note=res.summary))
    res.highlights = [("Klasse", res.notation), ("Name", name)]
    return res


class GrowthClassesResult(Result):
    """Die wichtigsten Wachstumsklassen (``classes``: Liste von ``GrowthClass``)."""


def growth_classes() -> GrowthClassesResult:
    """
    Die wichtigsten Komplexitätsklassen der Vorlesung (Serie 5, Frage 4):

        O(1) ⊂ O(log n) ⊂ O(n) ⊂ O(n log n) ⊂ O(n²) ⊂ … ⊂ O(nᵏ) ⊂ …  polynomiell (real lösbar)
        O(2ⁿ) ⊂ O(3ⁿ) ⊂ … ⊂ O(aⁿ) ⊂ …                               exponentiell (real unlösbar)

    >>> [c.name for c in growth_classes().classes][:3]
    ['konstant', 'logarithmisch', 'linear']
    """
    res = GrowthClassesResult()
    res.classes = list(_CLASSES)
    res.title = "Wachstumsklassen"
    res.summary = "O(1) ⊂ O(log n) ⊂ O(n) ⊂ O(n log n) ⊂ O(n²) ⊂ … ⊂ O(nᵏ) ⊂ …  und  O(2ⁿ) ⊂ O(3ⁿ) ⊂ … ⊂ O(aⁿ) ⊂ …"
    n = _sym("n")
    rows = []
    for c in _CLASSES:
        g = parse_term(c.term, "n")
        v10, v100 = _value_at(g, n, 10), _value_at(g, n, 100)
        v200 = _value_at(g, n, 200)
        factor = "" if c.key == "konstant" else _human(sp.N(v200 / v100, 6), 3)
        rows.append([c.name, c.notation, c.example, _human(v10), _human(v100), factor or "1",
                     "polynomiell (real lösbar)" if c.polynomial else "exponentiell+ (real unlösbar)"])
    res.steps.append(Step(
        "Die Klassen im Überblick", "Je weiter rechts, desto schneller wächst die Schrittzahl. "
        "Die Spalte »n: 100 → 200« zeigt, um welchen Faktor die Schrittzahl wächst, wenn sich n verdoppelt.",
        table=Table(["Klasse", "Notation", "Beispiel", "g(10)", "g(100)", "n: 100→200", "Einordnung"], rows),
        rule="Komplexitätsklassen"))
    res.steps.append(Step(
        "Definition 2 (Landau-Notation)",
        "f ∈ O(g): f(x) ≤ C·g(x) für alle x > x0 (wächst nicht stärker). f ∈ Ω(g): f(x) ≥ c·g(x) für alle x > x0 "
        "(wächst nicht langsamer). f ∈ Θ(g): beides. Zu zeigen sind die Zeugen x0 und C bzw. c (siehe big_o, big_omega, big_theta).",
        rule="O, Ω, Θ"))
    res.highlights = [("polynomiell", "O(1) ⊂ O(log n) ⊂ O(n) ⊂ O(n log n) ⊂ O(n²) ⊂ …"),
                      ("exponentiell", "O(2ⁿ) ⊂ O(3ⁿ) ⊂ … ⊂ O(aⁿ)")]
    return res


# ==========================================================================
# Aufgabe 1: Wachstum vergleichen und maximale Eingabelänge
# ==========================================================================

def _terms(funcs, var, log_mode):
    if not isinstance(funcs, dict) or not funcs:
        raise ValueError("funcs muss ein Wörterbuch {Name: Term} sein, z. B. {'A': 'n*log(n,2)', 'B': 'n**2'}")
    if len(funcs) > 8:
        raise ValueError("Höchstens 8 Funktionen")
    return {str(k): parse_term(v, var, log_mode) for k, v in funcs.items()}


def _duration(seconds) -> str:
    s = sp.N(seconds, 30)
    if s == 0:
        return "0 s"
    units = [(sp.Rational(1, 10 ** 9), "ns"), (sp.Rational(1, 10 ** 6), "µs"), (sp.Rational(1, 1000), "ms"),
             (1, "s"), (60, "min"), (3600, "h"), (86400, "Tage"), (31557600, "Jahre")]
    if s < units[0][0]:
        return "< 1 ns"
    best = units[0]
    for u in units:
        if s >= u[0]:
            best = u
    val = s / best[0]
    if best[1] == "Jahre" and val > 10 ** 6:
        return f"{_sci(val, 3)} Jahre" + (" (länger als das Weltall alt ist)" if val > 1.4 * 10 ** 10 else "")
    return f"{_human(val, 4)} {best[1]}"


class GrowthTableResult(Result):
    """``values`` {Name: {n: SymPy-Zahl}}, ``terms`` {Name: Term}, ``n_values``, ``ranking`` (aufsteigend)."""


def compare_growth(funcs: dict, n_values=(5, 10, 20, 50, 100), *, var: str = "n", round_log: str = "none",
                   step_time_s=None) -> GrowthTableResult:
    """
    Tabelle der Schrittzahlen mehrerer Algorithmen (Serie 5, Aufgabe 1).

    ``round_log="ceil"`` rundet jeden Logarithmus auf — so entsteht in der Vorlesung für n = 5 die Zeile
    A: 15, B: 25, C: 120 (n·⌈log₂ n⌉ = 5·3, n² = 25, n! = 120). ``step_time_s`` (z. B. 1e-9) ergänzt die Rechenzeit.

    >>> r = compare_growth({"A": "n*log(n,2)", "B": "n**2", "C": "factorial(n)"}, [5], round_log="ceil")
    >>> [int(r.values[k][5]) for k in "ABC"]
    [15, 25, 120]
    """
    ns = _check_ints(n_values, "n_values", 0, 10 ** 6, 12)
    terms = _terms(funcs, var, round_log)
    sym = _sym(var)
    values = {k: {n: _value_at(t, sym, n) for n in ns} for k, t in terms.items()}
    res = GrowthTableResult()
    res.terms, res.values, res.n_values = terms, values, ns
    res.title = "Wachstum vergleichen"
    note = " (Logarithmen aufgerundet)" if round_log == "ceil" else ""
    res.steps.append(Step("Funktionen", f"Anzahl Rechnungsschritte in Abhängigkeit von {var}{note}.",
                          items=[(k, t) for k, t in terms.items()]))
    rows = [[str(n)] + [_human(values[k][n]) for k in terms] for n in ns]
    res.steps.append(Step("Schrittzahlen", f"Jede Funktion an den Stellen {var} = " + ", ".join(map(str, ns)) + ".",
                          table=Table([var] + list(terms), rows), rule="Einsetzen"))
    if len(ns) and ns[-1] * 10 <= 10 ** 6:
        base = [n for n in ns if n >= 1][:5]
        frows = []
        for n in base:
            row = [str(n)]
            for k, t in terms.items():
                a, b = _value_at(t, sym, n), _value_at(t, sym, 10 * n)
                row.append("—" if a == 0 else _human(sp.N(b / a, 8), 4))
            frows.append(row)
        if frows:
            res.steps.append(Step(
                f"Was passiert, wenn {var} verzehnfacht wird?",
                "Faktor f(10n)/f(n): wie viel langsamer wird der Algorithmus bei zehnmal so grosser Eingabe? "
                "Bei polynomiellem Wachstum bleibt der Faktor ungefähr gleich, bei exponentiellem explodiert er.",
                table=Table([var] + list(terms), frows)))
    if step_time_s is not None:
        st = _rational(step_time_s, "step_time_s")
        trow = [[str(n)] + [_duration(values[k][n] * st) for k in terms] for n in ns]
        res.steps.append(Step("Rechenzeit", f"Zeit pro Rechenschritt: {float(st):g} s.",
                              table=Table([var] + list(terms), trow)))
    last = ns[-1]
    order = sorted(terms, key=lambda k: values[k][last])
    res.ranking = order
    res.steps.append(Step("Rangfolge", f"An der Stelle {var} = {last}: " + " ≤ ".join(order) + ".",
                          note="Entscheidend ist das Verhalten für grosse Eingaben, nicht der Wert bei kleinem n."))
    res.highlights = [(f"{k}({ns[-1]})", values[k][ns[-1]]) for k in terms]
    return res


def _check_ints(vals, name, lo, hi, maxlen):
    try:
        out = sorted({int(v) for v in vals})
    except (TypeError, ValueError):
        raise ValueError(f"{name} muss eine Liste ganzer Zahlen sein") from None
    if not out or len(out) > maxlen or out[0] < lo or out[-1] > hi:
        raise ValueError(f"{name}: 1 bis {maxlen} ganze Zahlen zwischen {lo} und {hi} erwartet")
    return out


def _rational(v, name) -> sp.Rational:
    try:
        r = sp.Rational(str(v)) if not isinstance(v, (sp.Rational, Fraction)) else sp.Rational(v)
    except (ValueError, TypeError):
        raise ValueError(f"{name} muss eine positive Zahl sein") from None
    if r <= 0 or r > 10 ** 15 or r < sp.Rational(1, 10 ** 30):
        raise ValueError(f"{name} muss eine Zahl zwischen 1e-30 und 1e15 sein")
    return r


def growth_factor(f, n: int, factor: int = 10, *, var: str = "n", round_log: str = "none"):
    """Faktor f(factor·n)/f(n): wie viel langsamer wird der Algorithmus bei ``factor``-facher Eingabe?

    >>> growth_factor("n**2", 7)
    100
    """
    if not isinstance(n, int) or not isinstance(factor, int) or n < 1 or not 2 <= factor <= 10 ** 6 or n > 10 ** 9:
        raise ValueError("n ≥ 1 und factor ≥ 2 (ganze Zahlen) erwartet")
    e = parse_term(f, var, round_log)
    sym = _sym(var)
    a = _value_at(e, sym, n)
    if a == 0:
        raise ValueError("f(n) = 0: der Faktor ist nicht definiert")
    q = _value_at(e, sym, factor * n) / a
    return q if q.is_Rational else sp.N(q, 12)


class MaxSizeResult(Result):
    """
    ``n`` (grösste Eingabelänge), ``budget`` (Rechenschritte in der Frist), ``value`` = f(n), ``value_next`` = f(n+1),
    ``time_at_n`` (Sekunden), ``limited`` (True: Suchgrenze erreicht), ``log`` (Suchprotokoll).
    """


def _bisect_max(expr, sym, budget, rows=None):
    """Grösstes ganzes n ≥ 1 mit f(n) ≤ budget (f wachsend). Gibt (n, limited) zurück."""
    def ok(n):
        return _value_at(expr, sym, n) <= budget
    if not ok(1):
        return 0, False
    hi, prev = 1, _value_at(expr, sym, 1)
    while ok(hi):
        if rows is not None:
            rows.append(("Verdopplung", hi, _value_at(expr, sym, hi), True))
        if hi >= MAX_N:
            return MAX_N, True
        nxt = min(hi * 2, MAX_N)
        v = _value_at(expr, sym, nxt)
        if v < prev:
            raise ValueError("Die Funktion ist nicht monoton wachsend — das Verfahren braucht f(n) ≤ f(n+1)")
        prev, hi = v, nxt
    if rows is not None:
        rows.append(("Verdopplung", hi, _value_at(expr, sym, hi), False))
    lo = hi // 2 if hi > 1 else 0
    lo = max(lo, 1)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        v = _value_at(expr, sym, mid)
        good = v <= budget
        if rows is not None:
            rows.append(("Bisektion", mid, v, good))
        if good:
            lo = mid
        else:
            hi = mid
    return lo, False


def max_input_size(f, time_limit_s=60, step_time_s=None, *, steps_per_second: int = 10 ** 9, var: str = "n",
                   round_log: str = "none") -> MaxSizeResult:
    """
    Grösste Eingabelänge n, die innerhalb einer Frist berechnet werden kann (Serie 5, Aufgabe 1).

    Der Rechner führt ``steps_per_second`` Rechenschritte pro Sekunde aus (oder ``step_time_s`` Sekunden pro
    Schritt); in ``time_limit_s`` Sekunden sind das B = Frist / Schrittzeit Schritte. Gesucht ist das grösste
    ganze n mit f(n) ≤ B. Für n·log₂ n und n! gibt es keine algebraische Lösung — es wird numerisch gelöst
    (Verdopplung, dann Bisektion über ganze Zahlen, exakt).

    >>> max_input_size("factorial(n)").n          # 13! = 6 227 020 800 ≤ 6·10^10 < 14!
    13
    >>> max_input_size("n**2").n                  # ⌊√(6·10^10)⌋
    244948
    """
    if isinstance(time_limit_s, bool):
        raise ValueError("time_limit_s muss eine Zahl sein")
    T = _rational(time_limit_s, "time_limit_s")
    if step_time_s is not None:
        st = _rational(step_time_s, "step_time_s")
    else:
        if isinstance(steps_per_second, bool) or not isinstance(steps_per_second, int) or steps_per_second < 1:
            raise ValueError("steps_per_second muss eine positive ganze Zahl sein")
        st = sp.Rational(1, steps_per_second)
    budget = sp.floor(T / st)
    if budget > 10 ** 40:
        raise ValueError("Das Rechenbudget ist zu gross (> 10^40 Schritte)")
    expr = parse_term(f, var, round_log)
    sym = _sym(var)
    log: list = []
    n, limited = _bisect_max(expr, sym, budget, log)
    res = MaxSizeResult()
    res.n, res.budget, res.limited, res.log, res.expr = n, int(budget), limited, log, expr
    res.value = _value_at(expr, sym, n) if n >= 1 else None
    res.value_next = _value_at(expr, sym, n + 1)
    res.time_at_n = res.value * st if res.value is not None else None
    res.title = f"Maximale Eingabelänge für f({var}) = {sp.sstr(expr)}"
    res.summary = (f"{var}_max = {n}" + (" (Suchgrenze erreicht)" if limited else "")
                   + f": f({n}) ≤ {_human(budget)} < f({n + 1})")
    res.steps.append(Step(
        "Rechenbudget", f"Pro Rechenschritt {float(st):g} s, Frist {float(T):g} s.",
        items=[("B = Frist / Schrittzeit", int(budget))], rule="Zeit ÷ Schrittzeit"))
    res.steps.append(Step(
        "Ansatz", f"Gesucht: das grösste ganze {var} mit f({var}) ≤ B. Die Gleichung f({var}) = B lässt sich bei "
        f"{var}·log({var}) oder {var}! nicht mit Algebra lösen — sie wird numerisch mit einem Algorithmus gelöst: "
        f"f wächst monoton, also erst {var} verdoppeln, bis f > B, dann den Bereich halbieren (Bisektion).",
        rule="Numerische Lösung (Bisektion)"))
    dbl = [r for r in log if r[0] == "Verdopplung"]
    bis = [r for r in log if r[0] == "Bisektion"]
    res.steps.append(Step("Verdoppeln", table=Table([var, f"f({var})", "f ≤ B?"],
                                                    [[str(a), _human(v), "ja" if g else "nein"] for _, a, v, g in dbl],
                                                    marks={i: ("ok" if r[3] else "bad") for i, r in enumerate(dbl)})))
    if bis:
        shown = bis if len(bis) <= 60 else bis[:40] + bis[-20:]
        res.steps.append(Step("Halbieren (Bisektion)", f"{len(bis)} Halbierungen genügen.",
                              table=Table([var, f"f({var})", "f ≤ B?"],
                                          [[str(a), _human(v), "ja" if g else "nein"] for _, a, v, g in shown],
                                          marks={i: ("ok" if r[3] else "bad") for i, r in enumerate(shown)})))
    if expr.is_polynomial(sym) and sp.Poly(expr, sym).degree() >= 1:
        try:
            roots = [r for r in sp.Poly(expr - budget, sym).nroots(n=30) if abs(sp.im(r)) < 1e-20 and sp.re(r) > 0]
            if roots:
                res.steps.append(Step("Kontrolle algebraisch",
                                      f"Bei einem Polynom geht es auch mit der Wurzel: f({var}) = B hat die Lösung "
                                      f"{var} ≈ {_human(max(roots))}; abgerundet: {int(sp.floor(max(roots)))}.",
                                      rule="Gleichung auflösen"))
        except Exception:
            pass
    res.steps.append(Step(
        "Ergebnis", items=[(f"{var}_max", n), (f"f({n})", res.value if res.value is not None else "–"),
                           (f"f({n + 1})", res.value_next)],
        note=(f"Rechenzeit bei {var} = {n}: {_duration(res.time_at_n)}; bei {n + 1}: "
              f"{_duration(res.value_next * st)}.") if n >= 1 else ""))
    res.highlights = [(f"{var}_max", n)]
    return res


class MaxSizeTable(Result):
    """``results`` {Name: MaxSizeResult}, ``n`` {Name: n_max}."""


def max_input_sizes(funcs: dict, time_limit_s=60, step_time_s=None, *, steps_per_second: int = 10 ** 9,
                    var: str = "n", round_log: str = "none") -> MaxSizeTable:
    """
    ``max_input_size`` für mehrere Algorithmen auf einmal (Aufgabe 1: A = n·log₂ n, B = n², C = n!).

    >>> max_input_sizes({"A": "n*log(n,2)", "B": "n**2", "C": "factorial(n)"}).n["C"]
    13
    """
    _terms(funcs, var, round_log)
    res = MaxSizeTable()
    res.results = {k: max_input_size(v, time_limit_s, step_time_s, steps_per_second=steps_per_second, var=var,
                                     round_log=round_log) for k, v in funcs.items()}
    res.n = {k: r.n for k, r in res.results.items()}
    first = next(iter(res.results.values()))
    res.title = "Maximale Eingabelänge je Algorithmus"
    res.summary = f"Frist {time_limit_s} s, Budget B = {_human(first.budget)} Rechenschritte"
    rows = [[k, sp.sstr(r.expr), str(r.n), _human(r.value) if r.value is not None else "–", _human(r.value_next)]
            for k, r in res.results.items()]
    res.steps.append(Step("Grösstes n mit f(n) ≤ B", "Numerisch gelöst: Verdopplung, dann Bisektion (exakte Ganzzahl-Suche).",
                          table=Table(["Algorithmus", "f(n)", f"{var}_max", "f(n_max)", "f(n_max+1)"], rows),
                          rule="Numerische Lösung (Bisektion)"))
    res.highlights = [(f"{k}: {var}_max", r.n) for k, r in res.results.items()]
    return res


# ==========================================================================
# Gauss-Trick
# ==========================================================================

class GaussResult(Result):
    """``value`` = Summe 1 + 2 + … + m als SymPy-Term in n bzw. Zahl; ``m`` = letzter Summand."""


def gauss_sum(m=100, var: str = "n") -> GaussResult:
    """
    Der Gauss-Trick für s = 1 + 2 + … + m  (»sollte unbedingt geübt werden«, Serie 5 Aufgabe 4).

    ``m`` ist eine Zahl oder ein Term wie ``"n-1"``.

    >>> gauss_sum(100).value
    5050
    >>> gauss_sum("n-1").value
    n*(n - 1)/2
    """
    if isinstance(m, int) and not isinstance(m, bool) and not 0 <= m <= 10 ** 12:
        raise ValueError("m muss zwischen 0 und 10^12 liegen")
    e = parse_term(m, var)
    sym = _sym(var)
    res = GaussResult()
    res.m = e
    s = sp.factor(e * (e + 1) / 2)
    res.value = sp.Integer(s) if s.is_Integer else s
    es = sp.sstr(e) if e.is_Atom else f"({sp.sstr(e)})"
    em1, ep1 = sp.sstr(sp.expand(e - 1)), sp.sstr(sp.expand(e + 1))
    res.title = f"Gauss-Trick: 1 + 2 + … + {es}"
    res.steps.append(Step("Summe aufschreiben", f"s = 1 + 2 + … + ({em1}) + {es}"))
    res.steps.append(Step("Rückwärts darunter schreiben", f"s = {es} + ({em1}) + … + 2 + 1"))
    res.steps.append(Step(
        "Spaltenweise addieren", f"2s = ({ep1}) + ({ep1}) + … + ({ep1})  mit {es} Summanden. "
        f"Jedes Paar ergibt dieselbe Summe: 1 + {es} = 2 + ({em1}) = … = {ep1}.",
        items=[("2s", sp.factor(e * (e + 1)))], rule="Gauss-Trick"))
    res.steps.append(Step("Durch 2 teilen", items=[("s", res.value)],
                          note=f"Kurz: s = m·(m+1)/2 mit m = {sp.sstr(e)}."))
    if e.is_Integer and e <= 10 ** 6:
        chk = sum(range(1, int(e) + 1))
        res.steps.append(Step("Probe", f"Direkt addiert: {chk}.", items=[("Summe", chk)]))
    res.highlights = [("s", res.value)]
    return res


# ==========================================================================
# O, Ω, Θ: Zeugen finden und begründen (Serie 5, Aufgabe 8)
# ==========================================================================

import mpmath as _mp

_L = sp.Symbol("L_", positive=True)       # steht für ln(x)
_MAX_X0 = 10 ** 6


class AsymptoticResult(Result):
    """
    Ergebnis von ``big_o`` / ``big_omega`` / ``big_theta``.

    ``kind`` ('O'|'Omega'|'Theta'), ``holds`` (stimmt die Aussage?), ``x0`` (Startwert), ``C`` (obere Konstante),
    ``c`` (untere Konstante; bei Θ beide), ``f``, ``g`` (SymPy), ``checked`` (Zahl der numerisch geprüften Stellen).
    Zeugen: f(x) ≤ C·g(x) bzw. f(x) ≥ c·g(x) für alle x ≥ x0.
    """


def _poly_xl(expr, x):
    """Term als Polynom in x und L = ln x: Liste [((p, q), Koeffizient)]; Logarithmengesetze angewandt."""
    # ponytail: nur Polynome in x und ln x — keine 2^x, keine verschachtelten Logarithmen
    e = sp.expand(sp.expand_log(expr, force=True))
    if e.has(sp.floor, sp.ceiling, sp.Abs):
        raise ValueError("Auf-/Abrundungen und Beträge werden hier nicht unterstützt")
    e = sp.expand(e.subs(sp.log(x), _L))
    if e.free_symbols - {x, _L}:
        raise ValueError("Der Term enthält unbekannte Grössen")
    try:
        P = sp.Poly(e, x, _L)
    except sp.PolynomialError:
        raise ValueError("Unterstützt werden Terme, die sich als Polynom in x und log(x) schreiben lassen "
                         "(nach Anwendung der Logarithmengesetze), z. B. 4x^5 + 15x^3, x^2*log(x), "
                         "5x·log2(3^x) + 3x^2") from None
    terms = [(m, c) for m, c in P.terms() if c != 0]
    if not terms:
        raise ValueError("Die Funktion ist 0")
    return e, terms


def _num(c):
    return _mp.mpf(str(sp.N(c, 45)))


def _round_up_int(v) -> int:
    return max(1, int(_mp.ceil(v)))


def _lower_rational(v) -> sp.Rational:
    """Eine schöne rationale Zahl ≤ v (v > 0 als mpf)."""
    for den in (1, 2, 4, 10, 100, 1000, 10 ** 4, 10 ** 6):
        k = int(_mp.floor(v * den))
        if k >= 1:
            return sp.Rational(k, den)
    return sp.Rational(1, 10 ** 6)


def _mono(p, q, x, L=None):
    return x ** p * (_L if L is None else L) ** q


def _fx(terms, xv):
    """Wert Σ a·x^p·ln(x)^q an der Stelle xv (mpf)."""
    lx = _mp.log(xv)
    return sum(_num(c) * xv ** m[0] * lx ** m[1] for m, c in terms)


def _lex_key(m):
    return (m[0], m[1])


def _fmt_mono(p, q, var="x"):
    parts = []
    if p:
        parts.append(var if p == 1 else f"{var}^{p}")
    if q:
        parts.append("log(x)" if q == 1 else f"log(x)^{q}")
    return "·".join(parts) or "1"


def _show(e, x):
    return e.subs(_L, sp.log(x))


def _O_witness(terms, gm, ag):
    """Zeugen für f ≤ C·g. terms: [((p,q), a)] von s·f. Gibt (holds, C, x0, rows, reason)."""
    pg, qg = gm
    rows, x0, total = [], 1, _mp.mpf(0)
    pos = [(m, c) for m, c in terms if _num(c) > 0]
    for (p, q), c in sorted(pos, key=lambda t: _lex_key(t[0]), reverse=True):
        a = _num(c)
        if p > pg or (p == pg and q > qg):
            return False, None, None, rows, f"Der Term {_fmt_mono(p, q)} wächst stärker als g"
        if q <= qg:
            xi = 3 if q < qg else 1
            ci = a
            why = (f"{_fmt_mono(p, q)} ≤ {_fmt_mono(pg, qg)} für x ≥ {xi}"
                   + (" (dort sind x ≥ 1 und ln x ≥ 1)" if xi == 3 else ""))
        else:
            Ls = _mp.mpf(q - qg) / (pg - p)
            r = _mp.exp(-(q - qg)) * Ls ** (q - qg)
            xi, ci = 1, a * r
            why = (f"{_fmt_mono(p, q)} ≤ {float(r):.4g}·{_fmt_mono(pg, qg)} (grösster Wert des Quotienten bei ln x = "
                   f"{float(Ls):.4g})")
        x0 = max(x0, xi)
        total += ci
        rows.append([sp.sstr(sp.nsimplify(c) if c.is_Rational else sp.N(c, 5)) + "·" + _fmt_mono(p, q), why,
                     f"{float(ci):.5g}"])
    return True, _round_up_int(total / ag), x0, rows, ""


def _Omega_witness(terms, gm, ag):
    """Zeugen für f ≥ c·g (f positiv führend). Gibt (holds, c, x0, rows, reason, info)."""
    pg, qg = gm
    lead_m, lead_c = max(terms, key=lambda t: _lex_key(t[0]))
    p, q = lead_m
    a = _num(lead_c)
    if a <= 0:
        return False, None, None, [], "Der führende Koeffizient ist negativ: f wird negativ", None
    if (p, q) < (pg, qg):
        return False, None, None, [], f"Der führende Term {_fmt_mono(p, q)} wächst langsamer als g", None
    negs = [(m, c) for m, c in terms if m != lead_m and _num(c) < 0]
    # x1: Rest-Terme klein gegen den halben führenden Term
    x_start = 1
    for (pi, qi), c in negs:
        if pi < p and qi > q:
            x_start = max(x_start, int(_mp.ceil(_mp.exp(_mp.mpf(qi - q) / (p - pi)))))
    half = a / 2

    def neg_ok(xv):
        lx = _mp.log(xv)
        lhs = sum(-_num(c) * xv ** m[0] * lx ** m[1] for m, c in negs)
        return lhs <= half * xv ** p * lx ** q

    x1 = x_start
    while not neg_ok(_mp.mpf(x1)):
        x1 += 1
        if x1 > _MAX_X0:
            raise ValueError("Es wurde kein passendes x0 ≤ 10^6 gefunden")
    # x2: führender Term gegen g
    if p >= pg and q >= qg:
        x2 = 3 if q > qg else 1
    else:
        x_inc = max(1, int(_mp.ceil(_mp.exp(_mp.mpf(qg - q) / (p - pg)))))
        x2 = x_inc
        while not (_mp.mpf(x2) ** (p - pg) * _mp.log(_mp.mpf(x2)) ** (q - qg) >= 1):
            x2 += 1
            if x2 > _MAX_X0:
                raise ValueError("Es wurde kein passendes x0 ≤ 10^6 gefunden")
    x0 = max(x1, x2)
    cval = half / ag
    rows = [[sp.sstr(c if c.is_Rational else sp.N(c, 5)) + "·" + _fmt_mono(*m),
             "negativ: Betrag nach oben abschätzen"] for m, c in negs]
    info = dict(lead=(lead_m, lead_c), half=half, x1=x1, x2=x2, negs=negs)
    return True, cval, x0, rows, "", info


def _probe(terms, x0, ag, g_m, const, direction):
    # ponytail: die Probe ist numerisch (40 Stellen) als Gegenkontrolle; der Beweis steht in den Schritten davor
    """Numerische Probe: ('O': f ≤ C g, 'Omega': f ≥ c g) an vielen Stellen x ≥ x0."""
    pg, qg = g_m
    pts = sorted({x0, x0 + 1, x0 + 2, 2 * x0, 5 * x0, 10 * x0, 100 * x0, 10 ** 4 + x0, 10 ** 6 + x0, 10 ** 9 + x0,
                  10 ** 15 + x0, *range(x0, x0 + 60)})
    ok = True
    checked = 0
    for xi in pts:
        xv = _mp.mpf(xi)
        f = _fx(terms, xv)
        gv = ag * xv ** pg * _mp.log(xv) ** qg
        cond = f <= const * gv if direction == "O" else f >= const * gv
        if not cond:
            ok = False
            break
        checked += 1
    return ok, checked, pts


def _nice(v) -> str:
    return _human(sp.Float(str(v), 20), 5) if not isinstance(v, (int, sp.Basic)) else _human(sp.nsimplify(v) if isinstance(v, sp.Basic) and v.is_Rational else v, 5)


def _asym(f, g, kind, var):
    x = _sym(var)
    fe, ge = parse_term(f, var), parse_term(g, var)
    f_ex, f_terms = _poly_xl(fe, x)
    g_ex, g_terms = _poly_xl(ge, x)
    if len(g_terms) != 1:
        raise ValueError("g muss ein einzelner Term wie x^2·log(x) sein (eine Potenz von x mal eine Potenz von log x)")
    (gm, gc) = g_terms[0]
    if _num(gc) <= 0:
        raise ValueError("g muss positiv sein")
    ag = _num(gc)
    res = AsymptoticResult()
    res.kind, res.f, res.g = kind, fe, ge
    sym_k = {"O": "O", "Omega": "Ω", "Theta": "Θ"}[kind]
    res.title = f"Begründen: f({var}) ∈ {sym_k}(g({var}))"
    res.steps.append(Step("Behauptung", items=[(f"f({var})", fe), (f"g({var})", ge)],
                          note="Definition 2 (Serie 5): f ∈ O(g) ⇔ f(x) ≤ C·g(x), f ∈ Ω(g) ⇔ f(x) ≥ c·g(x), "
                               "jeweils für alle x ≥ x0; f ∈ Θ(g), wenn beides gilt. Gesucht sind die Zeugen x0 und C bzw. c.",
                          rule="Definition Landau-Notation"))
    pf = _show(f_ex, x)
    if fe.has(sp.log) and (pf - fe).expand() != 0 or fe.has(sp.log) and sp.srepr(pf) != sp.srepr(fe):
        res.steps.append(Step("Logarithmengesetze anwenden",
                              "log(u^w) = w·log(u), log(u·v) = log u + log v, log_b(u) = ln u / ln b: danach steht f "
                              "als Polynom in x und log(x).", items=[(f"f({var})", pf)],
                              rule="Logarithmengesetze"))
    if ge.has(sp.log) and sp.srepr(_show(g_ex, x)) != sp.srepr(ge):
        res.steps.append(Step("g umformen", items=[(f"g({var})", _show(g_ex, x))], rule="Logarithmengesetze"))
    holds_all, parts = True, {}
    # Vorzeichen: für O mit |f|
    lead_c = max(f_terms, key=lambda t: _lex_key(t[0]))[1]
    sgn = 1 if _num(lead_c) > 0 else -1
    s_terms = [(m, c * sgn) for m, c in f_terms]
    if sgn < 0 and kind in ("O", "Theta"):
        res.steps.append(Step("Betrag", "f ist für grosse x negativ; es gilt |f(x)| = −f(x). Wir schätzen |f| ab.",
                              items=[("|f|", sp.expand(-_show(f_ex, x)))]))
    # ---------- O ----------
    if kind in ("O", "Theta"):
        okO, C, x0O, rows, why = _O_witness(s_terms, gm, ag)
        if not okO:
            holds_all = False
            res.steps.append(Step("f ∈ O(g) gilt nicht", why + ": der Quotient f/g wird beliebig gross.",
                                  rule="Wachstum vergleichen"))
        else:
            dropped = [(m, c) for m, c in s_terms if _num(c) <= 0]
            detail = ("Jeder positive Term wird nach oben durch ein Vielfaches von g abgeschätzt; "
                      + ("negative Terme dürfen weggelassen werden (sie machen f nur kleiner). " if dropped else "")
                      + "Alles gilt für x ≥ x0.").replace("; Alles", ". Alles")
            res.steps.append(Step("Term für Term abschätzen", detail,
                                  table=Table(["Term", "Abschätzung", "Beitrag zu C"], rows),
                                  rule="Terme nach oben abschätzen"))
            ok, checked, pts = _probe(s_terms, x0O, ag, gm, C, "O")
            if not ok:
                raise AssertionError("interner Fehler: O-Zeuge besteht die Probe nicht")
            parts["O"] = (C, x0O, checked, pts)
            res.steps.append(Step(
                "Zeugen für O", f"Summe der Beiträge, aufgerundet: C = {C}. Damit gilt f(x) ≤ {C}·g(x) für alle x ≥ {x0O}.",
                items=[("x0", x0O), ("C", C)], rule="f ≤ C·g"))
    # ---------- Ω ----------
    if kind in ("Omega", "Theta"):
        okW, c, x0W, rows, why, info = _Omega_witness(f_terms, gm, ag)
        if not okW:
            holds_all = False
            res.steps.append(Step("f ∈ Ω(g) gilt nicht", why + ".", rule="Wachstum vergleichen"))
        else:
            (lm, lc) = info["lead"]
            cn = sp.Rational(lc) / 2 / sp.Rational(gc) if (lc.is_Rational and gc.is_Rational) \
                else _lower_rational(c)
            detail = (f"Führender Term: {sp.sstr(lc if lc.is_Rational else sp.N(lc, 5))}·{_fmt_mono(*lm)}. "
                      "Positive Reste weglassen (f wird nur kleiner), negative Reste durch den halben führenden Term "
                      "nach oben abschätzen:")
            if rows:
                res.steps.append(Step("Negative Terme gegen den führenden Term", detail
                                      + f" Ab x ≥ {info['x1']} gilt: Summe der Beträge der negativen Terme ≤ ½·(führender Term).",
                                      table=Table(["Negativer Term", "Behandlung"], rows),
                                      rule="Terme nach unten abschätzen"))
            else:
                res.steps.append(Step("Keine negativen Terme", detail.split(" Positive")[0] + " Es gibt keine negativen "
                                      "Terme: f ≥ führender Term.", rule="Terme nach unten abschätzen"))
            res.steps.append(Step(
                "Führenden Term mit g vergleichen",
                f"Für x ≥ {info['x2']} gilt {_fmt_mono(*lm)} ≥ {_fmt_mono(*gm)}. Zusammen: f(x) ≥ ½·"
                f"{sp.sstr(lc if lc.is_Rational else sp.N(lc, 5))}·{_fmt_mono(*lm)} ≥ c·g(x) mit c = "
                f"{sp.sstr(cn)} für x ≥ {x0W}.", items=[("x0", x0W), ("c", cn)], rule="f ≥ c·g"))
            ok, checked, pts = _probe(f_terms, x0W, ag, gm, _num(cn), "Omega")
            if not ok:
                raise AssertionError("interner Fehler: Ω-Zeuge besteht die Probe nicht")
            parts["Omega"] = (cn, x0W, checked, pts)
    res.holds = holds_all
    res.parts = parts
    res.checked = sum(v[2] for v in parts.values())
    # Probe-Tabelle
    if parts:
        key = "O" if "O" in parts else "Omega"
        const, x0k, checked, pts = (parts[key][0], parts[key][1], parts[key][2], parts[key][3])
        shown = [x0k, 2 * x0k, 10 * x0k, 100 * x0k, 10 ** 6]
        prow = []
        for kk, (cv, x0v, _, _) in parts.items():
            for xi in dict.fromkeys(max(x0v, s) for s in shown):
                xv = _mp.mpf(xi)
                fv = _fx(f_terms, xv) if kk == "Omega" else _fx(s_terms, xv)
                gv = ag * xv ** gm[0] * _mp.log(xv) ** gm[1]
                bound = _num(cv) * gv
                prow.append([("O" if kk == "O" else "Ω"), str(xi), f"{float(fv):.6g}", f"{float(bound):.6g}",
                             "f ≤ C·g ✓" if kk == "O" else "f ≥ c·g ✓"])
        res.steps.append(Step("Probe", f"Stichproben (zusätzlich wurde die Ungleichung an {res.checked} Stellen "
                                       "bis x = 10^15 mit 40 Stellen Genauigkeit geprüft).",
                              table=Table(["", "x", "f(x)", "Schranke", "Ergebnis"], prow)))
    if kind == "Theta" and holds_all:
        res.C, res.c = parts["O"][0], parts["Omega"][0]
        res.x0 = max(parts["O"][1], parts["Omega"][1])
        res.steps.append(Step("Zeugen für Θ", "f ∈ O(g) und f ∈ Ω(g) mit demselben x0 (dem grösseren):",
                              items=[("x0", res.x0), ("c", res.c), ("C", res.C)], rule="Θ = O ∩ Ω"))
        res.summary = f"f ∈ Θ(g): {res.c}·g ≤ f ≤ {res.C}·g für x ≥ {res.x0}"
    elif holds_all:
        if kind == "O":
            res.C, res.x0, res.c = parts["O"][0], parts["O"][1], None
            res.summary = f"f ∈ O(g): f(x) ≤ {res.C}·g(x) für x ≥ {res.x0}"
        else:
            res.c, res.x0, res.C = parts["Omega"][0], parts["Omega"][1], None
            res.summary = f"f ∈ Ω(g): f(x) ≥ {res.c}·g(x) für x ≥ {res.x0}"
    else:
        res.C = res.c = res.x0 = None
        res.summary = f"Die Aussage f ∈ {sym_k}(g) stimmt nicht."
        # kleinere Stütze: falls Θ nur eine Hälfte scheitert
    res.highlights = ([("x0", res.x0)] if res.x0 is not None else []) \
        + ([("C", res.C)] if res.C is not None else []) + ([("c", res.c)] if res.c is not None else []) \
        + [("gilt", res.holds)]
    return res


def big_o(f, g, var: str = "x") -> AsymptoticResult:
    """
    Zeigt f ∈ O(g): sucht Zeugen x0 und C mit f(x) ≤ C·g(x) für alle x ≥ x0 und rechnet es wie auf Papier vor.

    ``f``, ``g`` Terme in ``var`` (Polynome und log(x); Logarithmengesetze werden angewandt, z. B.
    ``5*x*log(3**x, 2) + 3*x**2``); ``g`` ein einzelner Term wie ``x**2*log(x)``.

    >>> r = big_o("4*x**5 + 15*x**3 - 10*x**2 - 8", "x**6")
    >>> (r.C, r.x0)
    (19, 1)
    """
    return _asym(f, g, "O", var)


def big_omega(f, g, var: str = "x") -> AsymptoticResult:
    """
    Zeigt f ∈ Ω(g): Zeugen x0 und c mit f(x) ≥ c·g(x) für alle x ≥ x0.

    >>> r = big_omega("4*x**5 + 15*x**3 - 10*x**2 - 8", "x**3")
    >>> r.holds
    True
    """
    return _asym(f, g, "Omega", var)


def big_theta(f, g, var: str = "x") -> AsymptoticResult:
    """
    Zeigt f ∈ Θ(g) = O(g) ∩ Ω(g) mit c, C, x0.

    >>> r = big_theta("5*x**2 - 2*x + 8", "x**2")
    >>> r.holds
    True
    """
    return _asym(f, g, "Theta", var)


gauss_trick = gauss_sum     # Alias, so heisst es in der Vorlesung
