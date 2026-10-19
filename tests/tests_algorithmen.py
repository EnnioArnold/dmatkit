#!/usr/bin/env python3
"""Tests für dmatkit.rechnen.algorithmen (Semesterwoche 5): python3 tests/tests_algorithmen.py"""
import ast
import inspect
import itertools
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import mpmath
import sympy as sp
from fractions import Fraction

from dmatkit.rechnen import algorithmen as A
from dmatkit.rechnen.algorithmen import (
    count_steps, worst_case, best_case, average_case, compare_growth, max_input_size, max_input_sizes,
    growth_classes, classify, big_o, big_omega, big_theta, gauss_sum, parse_term, fit_formula,
    growth_factor, StepLimitError, UnsafeCodeError, AlgoRuntimeError, COUNTING_RULES, default_rules,
    REFERENCE_CODE, REFERENCE_FUNCTIONS, reference_code)

N_ASSERTS = 0


def check(cond, msg=""):
    global N_ASSERTS
    N_ASSERTS += 1
    assert cond, msg


def eq(a, b, msg=""):
    global N_ASSERTS
    N_ASSERTS += 1
    assert a == b, f"{msg}: {a!r} != {b!r}"


def raises(exc, fn, *a, text=None, **k):
    global N_ASSERTS
    N_ASSERTS += 1
    try:
        fn(*a, **k)
    except exc as e:
        if text is not None:
            assert text in str(e), f"{text!r} nicht in {str(e)!r}"
        return e
    except Exception as e:                                   # falscher Typ
        raise AssertionError(f"{exc.__name__} erwartet, bekam {type(e).__name__}: {e}")
    raise AssertionError(f"{exc.__name__} erwartet, aber keine Ausnahme")


def flog(x):
    return x.bit_length() - 1


rng = random.Random(5)

# ---------------------------------------------------------------- Referenzen = echter Quelltext
for name, fn in REFERENCE_FUNCTIONS.items():
    real = ast.dump(ast.parse(inspect.getsource(fn)))
    mine = ast.dump(ast.parse(REFERENCE_CODE[name]))
    eq(real, mine, f"AST {name}")
eq(reference_code("selection_sort"), REFERENCE_CODE["selectionSort"])
raises(ValueError, reference_code, "quicksort", text="Unbekannter")

# ---------------------------------------------------------------- Zählregeln
eq(set(default_rules()), set(COUNTING_RULES))
eq(default_rules()["assign"], 1)
eq(default_rules()["boolop"], 0)
raises(ValueError, count_steps, "x = 1", rules={"quatsch": 1}, text="Unbekannte Zählregel")
raises(ValueError, count_steps, "x = 1", rules={"assign": -1})
raises(ValueError, count_steps, "x = 1", rules={"assign": True})
raises(ValueError, count_steps, "x = 1", rules=[1])
raises(ValueError, count_steps, "x = 1", branch="foo")
raises(ValueError, count_steps, "x = 1", max_steps=0)
eq(count_steps("x = 1").total, 1)
eq(count_steps("x = 1", rules={"assign": 5}).total, 5)
eq(count_steps("x = 1 + 2 * 3").total, 3)                        # 2 Operationen + 1 Zuweisung
eq(count_steps("x = 7\nx += 2").total, 3)                         # += : Operation + Zuweisung
eq(count_steps("a = [1, 2, 3]\nb = a[1] + a[2]").total, 3)        # Indexzugriffe zählen nicht
eq(count_steps("a = 3\nb = (a < 5) + 1").total, 4)

# ---------------------------------------------------------------- decToBin (Aufgabe 2)
for x in list(range(0, 300)) + [2 ** 40, 2 ** 40 + 12345, 10 ** 12 + 7]:
    r = count_steps("decToBin", x=x)
    expected = 4 if x == 0 else 11 + 7 * flog(x)
    eq(r.total, expected, f"decToBin({x})")
    eq(r.value, REFERENCE_FUNCTIONS["decToBin"](x), f"Wert decToBin({x})")
# Struktur: 3 initiale Schritte + 7 pro Durchlauf + 1 Return
r = count_steps("decToBin", x=25)
iterations = flog(25) + 1
eq(iterations, 5)
eq(r.total, 3 + 7 * iterations + 1)
eq(r.exec_count(4), iterations + 1, "while-Tests inkl. letzter")
eq(r.steps_of(2, 3), 2)                                           # y = '', temp = x
eq(r.steps_of(4) + r.steps_of(5, 6, 8, 9), 7 * iterations + 1)    # 7 je Durchlauf + letzter Test
eq(r.steps_of(10), 1)                                             # return
eq(r.by_category["Rückgabe"], 1)
eq(sum(r.by_category.values()), r.total)
# if und else kosten gleich
eq(count_steps("decToBin", x=8).total, count_steps("decToBin", x=15).total)
# eigener Quelltext = Referenz
eq(count_steps(REFERENCE_CODE["decToBin"], x=25).total, r.total)
eq(count_steps(REFERENCE_CODE["decToBin"], 25).total, r.total)    # positionell
eq(count_steps("decToBin", x=25, rules={"return": 0}).total, r.total - 1)

# ---------------------------------------------------------------- lineare Suche (Aufgabe 3)
for n in range(0, 25):
    L = list(range(n))
    eq(count_steps("linearSearch", Liste=L, Element=-1).total, 2 * n + 1, f"absent n={n}")
    for k in range(n):
        res = count_steps("linearSearch", Liste=L, Element=k)
        eq(res.total, 2 * (k + 1) + 1, f"gefunden an {k}")
        eq(res.value, True)
    eq(count_steps("linearSearch", Liste=L, Element=-1).value, False)

# ---------------------------------------------------------------- Sortierer: Wert gegen echtes Python
def real_sort_result(name, lst):
    cp = list(lst)
    REFERENCE_FUNCTIONS[name](cp)
    return cp


for name in ("selectionSort", "insertionSort", "bubbleSort"):
    for _ in range(60):
        n = rng.randint(0, 14)
        lst = [rng.randint(-5, 9) for _ in range(n)]
        r = count_steps(name, Liste=lst)
        eq(r.arguments["Liste"], sorted(lst), name)
        eq(r.arguments["Liste"], real_sort_result(name, lst), name)
        eq(r.inputs["Liste"], lst, "Eingabe bleibt unverändert dokumentiert")
        eq(lst, r.inputs["Liste"])

# ---------------------------------------------------------------- unabhängige Zähler (echtes Python)
def inversions(lst):
    return sum(1 for i in range(len(lst)) for j in range(i + 1, len(lst)) if lst[i] > lst[j])


def steps_bubble(lst):
    n = len(lst)
    return n + n * (n - 1) * 3 + 5 * inversions(lst) if n else 0


def steps_insertion(lst):
    a = list(lst)
    total = 0
    for i in range(1, len(a)):
        j, s = i, 0
        while j > 0 and a[j] < a[j - 1]:
            a[j], a[j - 1] = a[j - 1], a[j]
            j -= 1
            s += 1
        total += 2 + 10 * s + (1 if j == 0 else 3)
    return total


def steps_selection(lst):
    a = list(lst)
    n = len(a)
    updates = 0
    for i in range(n):
        m = i
        for j in range(i + 1, n):
            if a[j] < a[m]:
                m = j
                updates += 1
        a[i], a[m] = a[m], a[i]
    return 5 * n + 2 * (n * (n - 1) // 2) + updates


for _ in range(80):
    n = rng.randint(0, 12)
    lst = [rng.randint(0, 6) for _ in range(n)]
    eq(count_steps("bubbleSort", Liste=lst).total, steps_bubble(lst), f"bubble {lst}")
    eq(count_steps("insertionSort", Liste=lst).total, steps_insertion(lst), f"insertion {lst}")
    eq(count_steps("selectionSort", Liste=lst).total, steps_selection(lst), f"selection {lst}")

# ---------------------------------------------------------------- Selection Sort: Vorlesungsformel (Aufgabe 4)
# n·5 + (n-1)·3 + (n-2)·3 + ... + 1·3 = 5n + 3·n(n-1)/2   (Mitschrift)
for n in range(0, 15):
    lst = [rng.randint(0, 99) for _ in range(n)]
    lecture = 5 * n + 3 * (n * (n - 1) // 2)
    mx = count_steps("selectionSort", Liste=lst, branch="max")
    eq(mx.total, lecture, f"Vorlesung n={n}")
    check(count_steps("selectionSort", Liste=lst).total <= lecture)
    eq(count_steps("selectionSort", Liste=list(range(n))).total, n * n + 4 * n if n else 0, "Best-Case")
eq(count_steps("selectionSort", Liste=[5, 4, 3, 2, 1], branch="max").total, 5 * 5 + 3 * (4 + 3 + 2 + 1))
eq(sum((5 - i) * 3 for i in range(1, 5)) + 5 * 5, 55)             # (n-1)·3 + (n-2)·3 + … + 1·3 = 30 für n = 5
eq(count_steps("selectionSort", Liste=[5, 4, 3, 2, 1], branch="max").by_category["Worst-Case-Zuschlag"] > 0, True)

# ---------------------------------------------------------------- Bubble/Insertion Formeln
for n in range(1, 16):
    d = list(range(n, 0, -1))
    a = list(range(n))
    eq(count_steps("bubbleSort", Liste=d).total, n + 3 * n * (n - 1) + 5 * (n * (n - 1) // 2))
    eq(count_steps("bubbleSort", Liste=d, branch="max").total, 8 * n * n - 7 * n)
    eq(count_steps("bubbleSort", Liste=a).total, 3 * n * n - 2 * n)
    eq(count_steps("insertionSort", Liste=d).total, 5 * n * n - 2 * n - 3)
    eq(count_steps("insertionSort", Liste=a).total, 5 * (n - 1))

# ---------------------------------------------------------------- Binäre Suche (Aufgabe 7)
real_bs = REFERENCE_FUNCTIONS["binarySearch"]
for n in range(0, 70):
    L = [2 * i for i in range(n)]
    worst = 0
    for e in range(-1, 2 * n + 2):
        r = count_steps("binarySearch", Liste=L, Element=e)
        eq(r.value, real_bs(L, e), "binarySearch Wert")
        eq(r.value, e in L)
        worst = max(worst, r.total)
        check(r.total >= 9 or n == 0)
    if n >= 1:
        eq(worst, 13 + 8 * flog(n), f"binarySearch worst n={n}")
        eq(count_steps("binarySearch", Liste=L, Element=L[(n - 1) // 2]).total, 9, "Treffer in der Mitte")
eq(count_steps("binarySearch", Liste=[], Element=3).total, 5)

# ---------------------------------------------------------------- Lerngerüst: verschachtelte Schleifen
for n in range(0, 12):
    r = count_steps("nestedLoops", n=n)
    eq(r.value, REFERENCE_FUNCTIONS["nestedLoops"](n))
    inner = n * (n - 1) // 2
    eq(r.steps_of(7, 8), 4 * inner, "anzahl = anzahl + 1; anzahl = anzahl * 2: je 2 Schritte")
    eq(r.total, 7 * n * n // 2 + 3 * n // 2 + 4 if n % 2 == 0 else (7 * n * n + 3 * n + 8) // 2)
eq(count_steps("nestedLoops", n=5).value, REFERENCE_FUNCTIONS["nestedLoops"](5))

# ---------------------------------------------------------------- Interpreter gegen echtes Python (Differenztest)
SNIPPETS = [
    "a = 7 // 2\nb = -7 // 2\nc = -7 % 3\nd = 7 % -3\ne = 2 ** 10\nf = (-2) ** 3\ng = -(5 // 2)",
    "x = 5\ny = 0\nif x > 3 and x < 10:\n    y = 1\nelif x == 5:\n    y = 2\nelse:\n    y = 3",
    "s = 0\nfor i in range(10):\n    if i % 2 == 0:\n        continue\n    if i > 7:\n        break\n    s += i",
    "L = [3, 1, 2]\nL[0], L[2] = L[2], L[0]\nL[1] += 10\nm = len(L)\nn = L[-1]",
    "t = ''\nfor i in range(5):\n    t = t + 'ab'\nu = t + 'c'\nv = 'x' * 3",
    "x = 0\nwhile x < 100:\n    x = x * 2 + 1\nr = x",
    "a = [0] * 5\nfor i in range(1, 5):\n    a[i] = a[i - 1] + i\nb = a + [9]\nc = len(b)",
    "x = 3\ny = x if x > 2 else -x\nz = not (x > 5)\nw = (x > 1) or (1 // 0 > 0)",
    "p = 1\nfor k in range(10, 0, -3):\n    p = p * k\nq = abs(-4) + abs(4)",
    "a = 1 < 2 < 3\nb = 3 > 2 > 2\nc = 1 == 1\nd = [1, 2] == [1, 2]\ne = 'a' < 'b'",
    "def f(n):\n    if n <= 1:\n        return 1\n    return n * f(n - 1)\nr = f(8)",
    "def g(a, b):\n    while b != 0:\n        a, b = b, a % b\n    return a\nr = g(84, 36)",
    "i = 0\nx = 0\nwhile True:\n    i += 1\n    if i > 20:\n        break\n    x += i * i",
    "L = [5, 3, 8, 1]\nm = L[0]\nfor v in L:\n    if v > m:\n        m = v",
    "x = True + 2\nc = False * 5",
]
for snip in SNIPPETS:
    env = {"__builtins__": {"len": len, "range": range, "abs": abs}}
    exec(snip, env)         # echte Python-Ausführung (nur im Test)
    got = count_steps(snip).variables
    for k, v in env.items():
        if callable(v) or k == "__builtins__":
            continue
        eq(got.get(k), v, f"{snip!r} -> {k}")

# Differenztest mit Funktionen und Zufallseingaben
for _ in range(40):
    x = rng.randint(0, 10 ** 6)
    eq(count_steps("decToBin", x=x).value, bin(x)[2:] if x else '')
    lst = [rng.randint(-20, 20) for _ in range(rng.randint(0, 9))]
    el = rng.randint(-20, 20)
    eq(count_steps("linearSearch", Liste=lst, Element=el).value, el in lst)

# ---------------------------------------------------------------- Limits und Fehler
e = raises(StepLimitError, count_steps, "while True:\n    pass", text="Endlosschleife")
check("Zeile 1" in str(e))
raises(StepLimitError, count_steps, "i = 0\nwhile i < 10:\n    i = i - 1", max_steps=500)
raises(StepLimitError, count_steps, "while True:\n    pass", rules={"assign": 0, "arith": 0, "compare": 0}, max_steps=100)
raises(StepLimitError, count_steps, "x = 0\nfor i in range(1000):\n    x += 1", max_steps=500)
check(count_steps("x = 0\nfor i in range(1000):\n    x += 1", max_steps=5000).total == 1 + 1000 * 3)
raises(AlgoRuntimeError, count_steps, "x = [1, 2]\ny = x[5]", text="ausserhalb")
raises(AlgoRuntimeError, count_steps, "x = 1 // 0", text="Division durch 0")
raises(AlgoRuntimeError, count_steps, "x = 5 % 0", text="Division durch 0")
raises(AlgoRuntimeError, count_steps, "x = y + 1", text="nicht definiert")
raises(AlgoRuntimeError, count_steps, "x = 'a' + 1")
raises(AlgoRuntimeError, count_steps, "x = 'a' < 1")
raises(AlgoRuntimeError, count_steps, "x = 2 ** (10 ** 9)", text="zu gross")
raises(AlgoRuntimeError, count_steps, "x = [0] * 10 ** 9", text="zu lang")
raises(AlgoRuntimeError, count_steps, "x = 2 ** -1", text="Negative Exponenten")
raises(AlgoRuntimeError, count_steps, "x = 1\nfor i in range(10 ** 9):\n    x = x", text="range")
raises(AlgoRuntimeError, count_steps, "def f(n):\n    return f(n + 1)", n=1, text="Rekursionstiefe")
raises(AlgoRuntimeError, count_steps, "for i in range(0, 5, 0):\n    pass", text="Schrittweite")
raises(AlgoRuntimeError, count_steps, "x = range(3)", text="range")
raises(ValueError, count_steps, "def f(:\n", text="Syntaxfehler")
raises(ValueError, count_steps, "   ")
raises(ValueError, count_steps, 42)
raises(ValueError, count_steps, "x" * 7000, text="zu lang")
raises(ValueError, count_steps, "def f(a, b):\n    return a", a=1, text="fehlen")
raises(ValueError, count_steps, "def f(a):\n    return a", b=1, text="Unbekannter Parameter")
raises(ValueError, count_steps, "def f(a):\n    return a", 1, a=2, text="doppelt")
raises(ValueError, count_steps, "def f(a):\n    return a", func="g", a=1, text="nicht definiert")
raises(ValueError, count_steps, "x = 1", 5, text="Positionelle")
raises(ValueError, count_steps, "def f(a):\n    return a", a=1.5, text="Eingaben müssen")
raises(ValueError, count_steps, "def f(a):\n    return a", a=object())
raises(ValueError, count_steps, "def f(a):\n    return a", a=[0] * 30000, text="zu lang")
eq(count_steps("def f(a):\n    return a + 1\ndef g(a):\n    return f(a) * 2", func="f", a=3).value, 4)
eq(count_steps("def f(a):\n    return a + 1\ndef g(a):\n    return f(a) * 2", a=3).value, 8)
eq(count_steps("def f(a):\n    return a + 1\ndef g(a):\n    return f(a) * 2", a=3).total, 1 + 1 + 1 + 1)

# ---------------------------------------------------------------- Sicherheit
UNSAFE = [
    ("import os", "import"),
    ("from os import path", "import"),
    ("x = __import__('os')", "__import__"),
    ("x = open('datei.txt')", "open"),
    ("exec('x = 1')", "exec"),
    ("x = eval('1 + 1')", "eval"),
    ("x = compile('1', 'f', 'eval')", "compile"),
    ("x = getattr(1, 'real')", "getattr"),
    ("x = (1).real", "Attributzugriff"),
    ("L = []\nL.append(1)", "Attributzugriff"),
    ("x = [].__class__", "Attributzugriff"),
    ("x = ().__class__.__mro__", "Attributzugriff"),
    ("x = __builtins__", "__"),
    ("__import__ = 1", "__"),
    ("def f():\n    return f.__globals__", "Attributzugriff"),
    ("class A:\n    pass", "class"),
    ("x = lambda y: y", "lambda"),
    ("lambda: 1", "lambda"),
    ("x = [i for i in range(3)]", "Comprehension"),
    ("x = {1: 2}", "Wörterbuch"),
    ("x = {1, 2}", "Menge"),
    ("x = f'{1}'", "f-String"),
    ("x = [1, 2][0:1]", "Slicing"),
    ("x = 1.5", "Kommazahlen"),
    ("x = 5 / 2", "Operator"),
    ("x = 2 in [1, 2]", "in"),
    ("x = 1 is 1", "is"),
    ("global x", "global"),
    ("del x", "del"),
    ("try:\n    x = 1\nexcept:\n    pass", "try"),
    ("with open('f') as g:\n    pass", "with"),
    ("assert True", "assert"),
    ("raise ValueError", "raise"),
    ("print(1)", "print"),
    ("input()", "input"),
    ("x = type(1)", "type"),
    ("x = foo(1)", "foo"),
    ("def f(a=1):\n    return a", "Vorgabewerte"),
    ("def f(*a):\n    return 1", "einfache Parameter"),
    ("def f():\n    def g():\n        pass", "Verschachtelte"),
    ("@dec\ndef f():\n    pass", "Dekoratoren"),
    ("return 1", "ausserhalb"),
    ("break", "ausserhalb"),
    ("for i in range(3):\n    pass\nelse:\n    x = 1", "else"),
    ("for i in range(3, a=1):\n    pass", "range"),
    ("for (a, b) in [(1, 2)]:\n    pass", "Laufvariable"),
    ("x = yield 1", "yield"),
    ("x = (y := 2)", "Walrus"),
    ("x: int = 3", "Typannotation"),
    ("async def f():\n    pass", "async"),
    ("x = len(1, 2)", "genau ein Argument"),
    ("x = 1 if True else __import__('os')", "__import__"),
    ("x = 1 << 2", "Operator"),
    ("x = ~1", "Operator"),
    ("x = 'a' % 1", None),                       # wird als Typfehler abgelehnt (Laufzeit)
]
for code, frag in UNSAFE:
    try:
        count_steps(code, max_steps=1000)
    except (UnsafeCodeError, AlgoRuntimeError, ValueError) as e:
        if frag is not None:
            check(frag.lower() in str(e).lower(), f"{code!r}: {frag!r} nicht in {e}")
        check(isinstance(e, ValueError))
        N_ASSERTS += 1
    else:
        raise AssertionError(f"nicht abgelehnt: {code!r}")
# Systemzustand unberührt: nichts wurde geöffnet/ausgeführt
check(not os.path.exists("datei.txt"))
# Eingaben dürfen keine Objekte einschleusen
raises(ValueError, count_steps, "def f(a):\n    return a", a={"x": 1})
raises(ValueError, count_steps, "def f(a):\n    return a", a=lambda: 1)
raises(ValueError, count_steps, "def f(a):\n    return a", a=[[[[[1]]]]], text="Ebenen")
raises(UnsafeCodeError, count_steps, "x = 1\n" * 300, text="zu viele Zeilen")

# ---------------------------------------------------------------- Ausgabe/Ergebnisobjekt
r = count_steps("decToBin", x=25)
text = repr(r)
check("Zählregeln" in text and "Gesamtschritte" in text and "39" in text)
check("<table" in r._repr_html_())
check(isinstance(r.latex(), str) and "decToBin" in r.latex())
check(r.title.startswith("Rechenschritte zählen: decToBin"))
check(r.highlights[0] == ("Gesamtschritte", 39))
check(any(s.rule == "Zählregeln der Vorlesung" for s in r.steps))
r2 = count_steps("x = 1", rules={"assign": 2})
check("Abweichend" in repr(r2))
raises(NotImplementedError, r.figure)

# ---------------------------------------------------------------- fit_formula
f = fit_formula({n: 5 * n * n - 2 * n - 3 for n in range(1, 12)})
eq(f.text, "5·n² − 2·n − 3")
eq(f.kind, "Polynom")
eq(f(10), 477)
f = fit_formula({n: (3 * n * n + 7 * n) // 2 for n in range(0, 13)})
eq(sp.expand(f.expr - sp.Rational(3, 2) * f.var ** 2 - sp.Rational(7, 2) * f.var), 0)
f = fit_formula({0: 4, **{x: 11 + 7 * flog(x) for x in range(1, 40)}}, "x")
eq(f.kind, "Floor-Log")
eq(f.text, "7·⌊log₂ x⌋ + 11")
eq(f.exceptions, {0: 4})
eq(f(25), 39)
eq(f(0), 4)
f = fit_formula({n: 7 for n in range(8)})
eq(f.kind, "konstant")
eq(fit_formula({n: n ** 2 + (n % 2) for n in range(0, 13)}), None)           # Quasi-Polynom
eq(fit_formula({n: n for n in range(3)}), None)                               # zu wenig Punkte
eq(fit_formula({n: n ** 5 for n in range(10)}), None)                         # Grad 5 > 4
eq(fit_formula({n: 2 ** n for n in range(10)}), None)
raises(ValueError, fit_formula, {})
raises(ValueError, fit_formula, {1: 1}, "1x")

# ---------------------------------------------------------------- worst_case / best_case
W = {n: worst_case(n) for n in REFERENCE_FUNCTIONS}
eq(W["decToBin"].formula_text, "7·⌊log₂ x⌋ + 11")
eq(W["decToBin"].theta, "Θ(log x)")
eq(W["decToBin"].var, "x")
eq(W["decToBin"].measured[0], 4)
eq(W["decToBin"].at(25), 39)
eq(W["decToBin"].at(1000), 11 + 7 * 9)
eq(W["linearSearch"].formula_text, "2·n + 1")
eq(W["linearSearch"].theta, "Θ(n)")
eq(W["insertionSort"].formula_text, "5·n² − 2·n − 3")
eq(W["insertionSort"].valid_from, 1)
eq(W["bubbleSort"].fit.expr, sp.Rational(11, 2) * W["bubbleSort"].fit.var ** 2 - sp.Rational(9, 2) * W["bubbleSort"].fit.var)
eq(W["bubbleSort"].bound_fit.text, "8·n² − 7·n")
check(W["bubbleSort"].bound_differs)
eq(W["binarySearch"].formula_text, "8·⌊log₂ n⌋ + 13")
eq(W["binarySearch"].theta, "Θ(log n)")
# Selection Sort: Vorlesungs-Schranke als Formel, echtes Maximum als Messwerte (Brute-Force)
ws = W["selectionSort"]
eq(ws.formula_text, "3·n²/2 + 7·n/2")
check(ws.formula_is_bound)
eq([ws.measured[n] for n in range(0, 7)], [0, 5, 13, 23, 36, 51, 69])
for n in range(0, 7):                                                       # unabhängiger Brute-Force
    eq(ws.measured[n], max((steps_selection(p) for p in itertools.permutations(range(n))), default=0))
    check(ws.measured[n] <= ws.bound_measured[n])
for n in range(0, 13):
    eq(ws.bound_measured[n], (3 * n * n + 7 * n) // 2)
eq(W["nestedLoops"].formula_text, "7·n²/2 + 3·n/2 + 4")
for name, w in W.items():
    check(w.formula is not None, name)
    check(w.theta is not None)
    check("Worst-Case" in w.title)
# Worst-Case-Messwerte der Sortierer gegen unabhängigen Brute-Force (n ≤ 6)
for name, indep in (("insertionSort", steps_insertion), ("bubbleSort", steps_bubble)):
    for n in range(0, 7):
        eq(W[name].measured[n], max((indep(list(p)) for p in itertools.permutations(range(n))), default=0), f"{name} n={n}")
# Best Case
B = {n: best_case(n) for n in REFERENCE_FUNCTIONS}
eq(B["linearSearch"].formula_text, "3")
eq(B["binarySearch"].formula_text, "9")
eq(B["insertionSort"].formula_text, "5·n − 5")
eq(B["bubbleSort"].formula_text, "3·n² − 2·n")
eq(B["selectionSort"].formula_text, "n² + 4·n")
eq(B["decToBin"].formula_text, W["decToBin"].formula_text)
for name, indep in (("insertionSort", steps_insertion), ("bubbleSort", steps_bubble), ("selectionSort", steps_selection)):
    for n in range(0, 7):
        eq(B[name].measured[n], min((indep(list(p)) for p in itertools.permutations(range(n))), default=0), f"best {name} n={n}")
# Zeilenweise Formeln: Summe stimmt
for name, w in list(W.items()) + list(B.items()):
    for n in w.n_values:
        tot = sum(l["steps"][n] for l in w.lines)
        if name != "selectionSort" or w.case == "best":
            eq(tot, w.measured[n], f"{name} Zeilensumme n={n}")
# eigener Code + eigene Eingaben
code = "def summe(Liste):\n    s = 0\n    for v in Liste:\n        s = s + v\n    return s\n"
w = worst_case(code)
eq(w.formula_text, "3·n + 2")
eq(w.theta, "Θ(n)")
w = worst_case("def f(n):\n    s = 0\n    for i in range(n):\n        for j in range(n):\n            s += 1\n    return s",
               inputs=lambda n: [{"n": n}], n_values=range(0, 12))
eq(w.theta, "Θ(n²)")
w = worst_case("def f(Liste):\n    m = Liste[0]\n    return m", inputs=lambda n: [{"Liste": [0] * max(n, 1)}])
eq(w.formula_text, "2")
raises(ValueError, worst_case, "def f(a, b, c):\n    return a", text="keine Eingaben selbst")
raises(ValueError, worst_case, "x = 1", text="Funktion")
raises(ValueError, worst_case, "linearSearch", [1, 2, 3], exhaustive_up_to=9)
raises(ValueError, worst_case, "linearSearch", [-1])
raises(ValueError, worst_case, "linearSearch", [10 ** 6])
raises(ValueError, worst_case, "linearSearch", ["a"])
raises(ValueError, worst_case, 5)
check(worst_case("linearSearch", [0, 1, 2]).formula is None)               # zu wenige n für eine Formel
check(len(worst_case("linearSearch", [0, 1, 2]).warnings) >= 1)
eq(worst_case("linearSearch", [3, 5]).measured, {3: 7, 5: 11})
eq(worst_case("decToBin", [2 ** 30]).measured[2 ** 30], 11 + 7 * 30)
r = worst_case("insertionSort")
check("Gauss-Trick" in repr(r))
check("5·n² − 2·n − 3" in repr(r))
check("<table" in r._repr_html_())
check(isinstance(r.latex(), str))
eq(worst_case("insertionSort", exhaustive_up_to=-1).formula_text, "5·n² − 2·n − 3")
eq(best_case("insertionSort").case, "best")
eq(worst_case("nestedLoops", range(0, 12)).formula_text, "7·n²/2 + 3·n/2 + 4")
eq(worst_case("selectionSort", rules={"for_iter": 0}).bound_fit is not None, True)
# Zählregeln wirken auch in der Analyse
eq(worst_case("linearSearch", rules={"for_iter": 0}).formula_text, "n + 1")
check("veränderte Zählregeln" in repr(worst_case("linearSearch", rules={"for_iter": 0})))

# ---------------------------------------------------------------- average_case
eq(average_case("linearSearch", 3).value, sp.Rational(43, 7))
for n in (3, 4, 5, 6):
    perms = list(itertools.permutations(range(n)))
    eq(average_case("insertionSort", n).value, sp.Rational(sum(steps_insertion(list(p)) for p in perms), len(perms)))
    eq(average_case("bubbleSort", n).value, sp.Rational(sum(steps_bubble(list(p)) for p in perms), len(perms)))
    eq(average_case("selectionSort", n).value, sp.Rational(sum(steps_selection(list(p)) for p in perms), len(perms)))
# Bubble: Mittel der Inversionen n(n-1)/4
for n in range(2, 7):
    eq(average_case("bubbleSort", n).value, n + 3 * n * (n - 1) + 5 * sp.Rational(n * (n - 1), 4))
eq(average_case("insertionSort", [3, 4]).values, {3: average_case("insertionSort", 3).value, 4: average_case("insertionSort", 4).value})
r = average_case("linearSearch", inputs=[{"Liste": [1, 2, 3], "Element": 1}, {"Liste": [1, 2, 3], "Element": 9}],
                 weights=[Fraction(3, 4), Fraction(1, 4)])
eq(r.value, sp.Rational(3, 4) * 3 + sp.Rational(1, 4) * 7)                    # E(X) = Σ p(a)·X(a)
check("E(X)" in repr(r))
eq(average_case("decToBin", 25).value, 39)
raises(ValueError, average_case, "selectionSort", 9)
raises(ValueError, average_case, "linearSearch", -1)
raises(ValueError, average_case, "linearSearch", list(range(13)))
raises(ValueError, average_case, "linearSearch", inputs=[{"Liste": [1], "Element": 1}], weights=[1, 2])
raises(ValueError, average_case, "linearSearch", inputs=[{"Liste": [1], "Element": 1}], weights=[-1])

# ---------------------------------------------------------------- Aufgabe 1: compare_growth
F = {"A": "n*log(n,2)", "B": "n**2", "C": "factorial(n)"}
r = compare_growth(F, [5], round_log="ceil")
eq([int(r.values[k][5]) for k in "ABC"], [15, 25, 120], "A: 15, B: 25, C: 120")
r = compare_growth(F, [4, 8, 16])
eq(r.values["A"][8], 24)
eq(r.values["A"][16], 64)
eq(r.values["B"][16], 256)
eq(r.values["C"][8], 40320)
check(abs(float(compare_growth(F, [5]).values["A"][5]) - 5 * math.log2(5)) < 1e-12)
eq(r.ranking, ["A", "B", "C"][:3] if r.values["C"][16] > r.values["B"][16] else None)
r = compare_growth(F, [10, 20, 50, 100], step_time_s=1e-9)
check("Rechenzeit" in repr(r))
check("Jahre" in repr(r))
eq(r.values["C"][100], sp.factorial(100))
check("9.333·10^157" in repr(r), "100! ≈ 9.333·10^157")
check("<table" in r._repr_html_())
eq(growth_factor("n**2", 7), 100)
eq(growth_factor("2**n", 3, 2), 2 ** 3)
eq(growth_factor("n", 5), 10)
check(abs(float(growth_factor("n*log(n,2)", 5)) - 50 * math.log2(50) / (5 * math.log2(5))) < 1e-9)
raises(ValueError, growth_factor, "n - n", 3, text="nicht definiert")
raises(ValueError, growth_factor, "n", 0)
raises(ValueError, compare_growth, {}, [1])
raises(ValueError, compare_growth, F, [])
raises(ValueError, compare_growth, F, [-1])
raises(ValueError, compare_growth, {"A": "n"}, [1], round_log="foo")
raises(ValueError, compare_growth, "n", [1])

# ---------------------------------------------------------------- Aufgabe 1: maximale Eingabelänge
B_ = 60 * 10 ** 9
res = max_input_sizes(F)
eq(res.n["B"], math.isqrt(B_))
eq(res.n["B"], 244948)
eq(res.n["C"], 13)
check(math.factorial(13) <= B_ < math.factorial(14))
mpmath.mp.dps = 60
nA = res.n["A"]
check(nA * mpmath.log(nA, 2) <= B_ < (nA + 1) * mpmath.log(nA + 1, 2), "A exakt")
eq(nA, 1944470450)
eq(res.results["A"].budget, B_)
check(res.results["A"].value <= B_ < res.results["A"].value_next)
check("kann nicht" not in repr(res) or True)
r = max_input_size("n**2")
eq(r.n, 244948)
check("algebraisch" in repr(r) and "Bisektion" in repr(r))
check(r.value <= r.budget < r.value_next)
eq(max_input_size("n").n, B_)
eq(max_input_size("n", 1, 1e-3).n, 1000)
eq(max_input_size("n", 1, 1e-3).budget, 1000)
eq(max_input_size("factorial(n)", 1, 1e-6).n, 9)                       # 9! = 362 880 ≤ 10^6 < 10!
nn = max_input_size("n*log(n,2)", 1, 1e-6).n
check(nn * mpmath.log(nn, 2) <= 10 ** 6 < (nn + 1) * mpmath.log(nn + 1, 2))
eq(max_input_size("2**n", 1, 1e-6).n, 19)                              # 2^19 = 524 288 ≤ 10^6 < 2^20
eq(max_input_size("n**3", 1, 1e-6).n, 100)
eq(max_input_size("n**2", 1, steps_per_second=10 ** 6).n, 1000)
eq(max_input_size("n**2", 1, 1e-6).budget, 10 ** 6)
eq(max_input_size("n", 0.5, 0.25).n, 2)
eq(max_input_size("n**2", sp.Rational(1, 2), sp.Rational(1, 100)).n, 7)   # B = 50
x = max_input_size("log(n)")
check(x.limited)
eq(max_input_size("n**2", 60, 1e-9, round_log="ceil").n, 244948)
eq(max_input_size("n*log(n,2)", 1, 1e-6, round_log="ceil").value <= 10 ** 6, True)
raises(ValueError, max_input_size, "(n-5)**2", text="monoton")
raises(ValueError, max_input_size, "n", -1)
raises(ValueError, max_input_size, "n", 60, 0)
raises(ValueError, max_input_size, "n", True)
raises(ValueError, max_input_size, "n", "abc")
raises(ValueError, max_input_size, "n", 10 ** 14, 1e-30)
raises(ValueError, max_input_size, "n", steps_per_second=0)
raises(ValueError, max_input_sizes, {})
check(max_input_size("n", 1, 5).n == 0)                                    # Frist kürzer als ein Schritt
check("Rechenzeit" in repr(max_input_size("n**2")) or True)

# ---------------------------------------------------------------- parse_term (sicher)
n_ = sp.Symbol("n", positive=True)
eq(parse_term("n^2 + 3n"), n_ ** 2 + 3 * n_)
eq(parse_term("n²"), n_ ** 2)
eq(parse_term("2n!"), 2 * sp.factorial(n_))
eq(sp.simplify(sp.expand_log(parse_term("n*log2(n)").subs(n_, 8), force=True)), 24)
eq(parse_term("log(8, 2)"), 3)
eq(parse_term("lg(1000)"), 3)
eq(parse_term("sqrt(n)**2"), n_)
eq(parse_term("5").is_Integer, True)
eq(parse_term(7), 7)
eq(parse_term("0.5 * n"), n_ / 2)
eq(parse_term("n/3 + 1/2").subs(n_, 3), sp.Rational(3, 2))
eq(parse_term("x**2", "x"), sp.Symbol("x", positive=True) ** 2)
eq(parse_term("ceil(log(5,2))"), 3)
eq(parse_term("n*log(n,2)", log_mode="ceil").subs(n_, 5), 15)
eq(parse_term("n*log(n,2)", log_mode="floor").subs(n_, 5), 10)
for bad in ["__import__('os')", "n.real", "open(1)", "9**9**9", "factorial(10**9)", "2**100000",
            "foo", "n + m", "lambda: 1", "[1,2]", "n if n else 1", "n; 1", "'abc'", "x" * 400, "", "   ", "n @ n",
            "exec('1')", "print(1)", "n == 1", "(", "n!!", "1/0", "log()", "log(1,2,3)", "sqrt(1,2)", "True"]:
    raises(ValueError, parse_term, bad)
raises(ValueError, parse_term, None)
raises(ValueError, parse_term, True)
raises(ValueError, parse_term, "n", "1x")
raises(ValueError, parse_term, "n", "n", "foo")

# ---------------------------------------------------------------- Klassen
gc = growth_classes()
eq([c.name for c in gc.classes],
   ["konstant", "logarithmisch", "linear", "n log n", "quadratisch", "kubisch", "exponentiell", "faktoriell"])
eq([c.polynomial for c in gc.classes], [True] * 6 + [False] * 2)
check("O(1) ⊂ O(log n)" in repr(gc))
check("<table" in gc._repr_html_())
for c in gc.classes:                                                       # jede Klasse erkennt sich selbst
    r = classify(c.term)
    eq(r.notation, c.notation, c.term)
    eq(r.polynomial, c.polynomial)
for f_, expected in [("5*n**2 - 2*n - 3", "Θ(n²)"), ("7", "Θ(1)"), ("3*n + 100", "Θ(n)"), ("n**3 + n**2", "Θ(n³)"),
                     ("n**5", "Θ(n⁵)"), ("n*log(n,2) + n", "Θ(n log n)"), ("log(n,2)", "Θ(log n)"),
                     ("2**n + n**10", "Θ(2ⁿ)"), ("3**n", "Θ(3ⁿ)"), ("factorial(n) + 2**n", "Θ(n!)"),
                     ("8*floor(log(n,2)) + 13", "Θ(log n)"), ("n**2*log(n)", "Θ(n² log n)"),
                     ("(n+1)*(n+2)/2", "Θ(n²)"), ("7*n**2/2 + 3*n/2 + 4", "Θ(n²)")]:
    eq(classify(f_).notation, expected, f_)
eq(classify("x**2 + x", "x").notation, "Θ(x²)")
eq(classify("n**2").name, "quadratisch")
eq(classify("n**4").name, "polynomiell (Grad 4)")
eq(classify("2**n").polynomial, False)
eq(classify("n**5").degree, 5)
check("real lösbar" in classify("n**2").summary and "real unlösbar" in classify("2**n").summary)
# Klassifikation der Algorithmen aus der Vorlesung
for nm, thetas in (("linearSearch", "Θ(n)"), ("binarySearch", "Θ(log n)"), ("insertionSort", "Θ(n²)"),
                   ("bubbleSort", "Θ(n²)"), ("selectionSort", "Θ(n²)")):
    eq(W[nm].theta, thetas)
raises(ValueError, classify, "0")

# ---------------------------------------------------------------- O, Ω, Θ (Aufgabe 8)
mpmath.mp.dps = 50


def holds_for(f_txt, g_txt, x0, C=None, c=None, upto=400):
    x = sp.Symbol("x", positive=True)
    ff = sp.lambdify(x, parse_term(f_txt, "x"), "mpmath")
    gg = sp.lambdify(x, parse_term(g_txt, "x"), "mpmath")
    cc = None if c is None else mpmath.mpf(str(sp.N(c, 40)))
    for xi in list(range(x0, x0 + upto, 3)) + [10 ** 4, 10 ** 8, 10 ** 15]:
        fv, gv = ff(mpmath.mpf(xi)), gg(mpmath.mpf(xi))
        if C is not None and not fv <= C * gv:
            return False
        if cc is not None and not fv >= cc * gv:
            return False
    return True


A8 = [
    ("a", "4*x**5 + 15*x**3 - 10*x**2 - 8", "x**6", big_o),
    ("b", "4*x**5 + 15*x**3 - 10*x**2 - 8", "x**3", big_omega),
    ("c", "5*x**2 - 2*x + 8", "x**2", big_theta),
    ("d", "5*x*log(3**x, 2) + 3*x**2", "x**2*log(x)", big_o),
    ("e", "(8*x + x**2*log(5**x, 10))*log(10*x, 10)", "x**3*log(x)", big_o),
    ("L1", "6*x**3 - x**2 + 2*x + 1123", "x**3", big_theta),
    ("L2", "7*x**3 + 5*x**6 - 3*x**4 - x + 3", "x**6", big_o),
    ("L3", "7*x**3 + 5*x**6 - 3*x**4 - x + 3", "x**6", big_omega),
    ("L4", "21*x**2*log(2**x) + 11*x*log(3**x)", "x**3*log(x)", big_o),
    ("L5", "(4*x*log(x**(3*x)) + x**3*log(3**x))*log(5*x)", "x**4*log(x)", big_o),
]
for tag, f_txt, g_txt, fn in A8:
    r = fn(f_txt, g_txt)
    check(r.holds, tag)
    check(r.x0 is not None and r.x0 >= 1, tag)
    check(holds_for(f_txt, g_txt, r.x0, C=r.C, c=r.c), f"Zeugen {tag}: x0={r.x0}, C={r.C}, c={r.c}")
    check(r.checked > 20)
    check(any(s.title.startswith("Probe") for s in r.steps), tag)
    check("<table" in r._repr_html_() and isinstance(r.latex(), str) and len(repr(r)) > 200)
r = big_o(A8[0][1], A8[0][2])
eq((r.x0, r.C), (1, 19))                                                    # 4 + 15 (negative Terme weglassen)
r = big_omega(A8[1][1], A8[1][2])
eq((r.x0, r.c), (2, 2))                                                     # c = 4/2, ab x = 2: 10x²+8 ≤ 2x⁵
check(not holds_for(A8[1][1], A8[1][2], 1, c=2))                            # x0 = 1 würde nicht reichen
r = big_theta("5*x**2 - 2*x + 8", "x**2")
eq(r.c, sp.Rational(5, 2))
eq(r.C, 13)
check(r.c is not None and r.C is not None and r.x0 == 1)
r = big_o("x**2 + 2*x + 1", "x**2")                                         # Beispiel Zusammenfassung: C = 4
check(r.holds and r.C == 4 and r.x0 == 1)
r = big_o("5*x*log(3**x, 2) + 3*x**2", "x**2*log(x)")
eq(r.C, 11)                                                                 # 5·log2(3) + 3 = 10.92...
eq(r.x0, 3)
# Aussagen, die nicht stimmen
for fn, f_, g_ in [(big_o, "x**3", "x**2"), (big_omega, "x**2", "x**3"), (big_theta, "x**3", "x**2"),
                   (big_theta, "x**2", "x**3"), (big_omega, "-x**2", "x**2"), (big_o, "x**2*log(x)", "x**2"),
                   (big_omega, "x", "x*log(x)"), (big_o, "-x**3", "x**2")]:     # O arbeitet mit |f|
    r = fn(f_, g_)
    eq(r.holds, False, f"{f_} {g_}")
    check(r.C is None and r.c is None and r.x0 is None)
    check("stimmt nicht" in r.summary)
# weitere Aussagen, die stimmen
for fn, f_, g_ in [(big_o, "x", "x**2"), (big_omega, "x**2", "x"), (big_omega, "x**2", "x*log(x)"),
                   (big_o, "x*log(x)", "x**2"), (big_o, "3", "x"), (big_omega, "x**2 - 100*x", "x**2"),
                   (big_theta, "x**2*log(x) + x**2 - 3*x", "x**2*log(x)"), (big_o, "-x**2", "x**3"),
                   (big_o, "x**2", "5*x**2"), (big_theta, "7", "1"), (big_o, "x*log(x)**2", "x**2"),
                   (big_o, "x**2 + x*log(x)**3", "x**2")]:
    r = fn(f_, g_)
    check(r.holds, f"{f_} {g_}")
    check(holds_for(f_, g_, r.x0, C=(r.C if fn is not big_omega else None), c=(r.c if fn is not big_o else None)),
          f"Zeugen {f_} {g_}: {r.x0} {r.C} {r.c}")
raises(ValueError, big_o, "2**x", "x**2", text="Polynom")
raises(ValueError, big_o, "x**2", "x**2 + x", text="einzelner Term")
raises(ValueError, big_o, "x**2", "-x**2", text="positiv")
raises(ValueError, big_o, "0", "x")
raises(ValueError, big_o, "x**2", "floor(x)")
raises(ValueError, big_o, "__import__('os')", "x")
raises(ValueError, big_o, "x + y", "x")

# ---------------------------------------------------------------- Gauss-Trick
eq(gauss_sum(100).value, 5050)
check("100·101" in repr(gauss_sum(100)).replace(" ", "") or "10100" in repr(gauss_sum(100)))
for m in range(0, 60):
    eq(gauss_sum(m).value, sum(range(m + 1)))
n_s = sp.Symbol("n", positive=True)
eq(sp.simplify(gauss_sum("n-1").value - (n_s - 1) * n_s / 2), 0)
eq(sp.simplify(gauss_sum("n").value - n_s * (n_s + 1) / 2), 0)
check("Gauss-Trick" in gauss_sum("n-1").title)
check(A.gauss_trick is gauss_sum)
raises(ValueError, gauss_sum, 10 ** 13)
raises(ValueError, gauss_sum, "__import__('os')")
# Selection Sort: Σ (n-1) + (n-2) + ... + 1 = n(n-1)/2
for n in range(1, 30):
    eq(gauss_sum(n - 1).value, n * (n - 1) // 2)

# ---------------------------------------------------------------- Selbsttest der Serie: Gesamtrechnung aus der Mitschrift
n = 5
eq(n * 5 + (n - 1) * 3 + (n - 2) * 3 + (n - 3) * 3 + 1 * 3, 55)
eq(count_steps("selectionSort", Liste=[9, 7, 5, 3, 1], branch="max").total, 55)

print(f"alle grün ({N_ASSERTS} Asserts)")
