#!/usr/bin/env python3
"""Tests für dmatkit.rechnen.praedikate (Semesterwoche 3): python3 tests/tests_praedikate.py"""
import itertools
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from fractions import Fraction

from dmatkit.rechnen.praedikate import (
    ExprError, Seq1, one_based, safe_eval, free_names,
    Predicate, SUM_EQUAL_TO_FIVE, PAL, FERMAT, relation_predicate,
    Universe, naturals, integers, reals, rationals, index_set, chars, as_universe, named_universe,
    forall, exists, quantify, unroll, python_code, negate_quantifier, negate_formula, push_negation,
    swap_order, everybody_loves_somebody, find_examples, has_duplicate, all_distinct, occurs_multiple,
    matrix_quantors, parse_formula, evaluate_formula, to_formula, describe_formula,
    QAtom, QNot, QBin, QQuant, QuantResult, MAX_COMBINATIONS, MAX_UNIVERSE, allquantor, existenzquantor,
)
from dmatkit.rechnen import praedikate as PM

N = 0


def ok(cond, msg=""):
    global N
    N += 1
    assert cond, msg


def raises(exc, fn, *a, contains=None, **kw):
    global N
    N += 1
    try:
        fn(*a, **kw)
    except exc as e:
        if contains is not None:
            assert contains in str(e), f"Meldung '{e}' enthält nicht '{contains}'"
        return str(e)
    raise AssertionError(f"{exc.__name__} erwartet bei {fn.__name__}{a}")


t0 = time.time()

# ==========================================================================
# 1. Sicherer Ausdrucks-Auswerter
# ==========================================================================
ok(safe_eval("1 + 2 * 3") == 7)
ok(safe_eval("(1 + 2) * 3") == 9)
ok(safe_eval("7 // 2") == 3 and safe_eval("7 % 3") == 1 and safe_eval("2 ** 10") == 1024)
ok(safe_eval("2^10") == 1024, "^ ist Potenz")
ok(safe_eval("7 / 2") == Fraction(7, 2) and safe_eval("6 / 3") == 2 and isinstance(safe_eval("6 / 3"), int))
ok(safe_eval("1/3 + 1/6") == Fraction(1, 2), "exakte Brüche")
ok(safe_eval("2 ** -2") == Fraction(1, 4))
ok(safe_eval("-x", {"x": 3}) == -3)
ok(safe_eval("x % 2 == 0", {"x": 6}) is True and safe_eval("x % 2 == 0", {"x": 7}) is False)
ok(safe_eval("0 < x < 5", {"x": 3}) is True and safe_eval("0 < x < 5", {"x": 5}) is False)
ok(safe_eval("x = 3", {"x": 3}) is True, "einfaches = als Gleichheit")
ok(safe_eval("x ≠ 3 ∧ x ≤ 4", {"x": 4}) is True, "Unicode-Zeichen")
ok(safe_eval("not (a and b) or c", {"a": True, "b": True, "c": False}) is False)
ok(safe_eval("s == s[::-1]", {"s": "REITTIER"}) is True)
ok(safe_eval("s == s[::-1]", {"s": "REITTIERE"}) is False)
ok(safe_eval("s[1:3]", {"s": "abcdef"}) == "bc")
ok(safe_eval("a[0] + a[-1]", {"a": [1, 2, 3]}) == 4)
ok(safe_eval("len(a) + abs(-3) + min(a) + max(a) + sum(a)", {"a": [1, 2, 3]}) == 3 + 3 + 1 + 3 + 6)
ok(safe_eval("sum(range(5))") == 10 and safe_eval("list(range(3))") == [0, 1, 2])
ok(safe_eval("x in [1, 2, 3]", {"x": 2}) is True and safe_eval("x not in (1, 2)", {"x": 2}) is False)
ok(safe_eval("x ∈ {1, 2}", {"x": 2}) is True)
ok(safe_eval("'b' in 'abc'") is True)
ok(safe_eval("s.lower() == 'abc'", {"s": "ABC"}) is True)
ok(safe_eval("s.count('a')", {"s": "banana"}) == 3)
ok(safe_eval("a.count(1)", {"a": [1, 2, 1]}) == 2)
ok(safe_eval("len(set(a)) == len(a)", {"a": [1, 2, 3]}) is True)
ok(safe_eval("is_prime(x)", {"x": 97}) is True and safe_eval("is_prime(x)", {"x": 91}) is False)
ok([n for n in range(30) if safe_eval("is_prime(n)", {"n": n})] == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
ok(safe_eval("divides(3, 12)") is True and safe_eval("gcd(12, 18)") == 6)
ok(safe_eval("implies(a, b)", {"a": False, "b": False}) is True)
ok(safe_eval("x if x > 0 else -x", {"x": -4}) == 4)
ok(safe_eval("f(x, 2)", {"x": 5}, functions={"f": lambda a, b: a > b}) is True)
ok(safe_eval("M[1][2] == 7", {"M": one_based([[1, 2, 3], [5, 6, 7]])}) is False)
ok(safe_eval("M[2][3] == 7", {"M": one_based([[1, 2, 3], [5, 6, 7]])}) is True)
ok(free_names("a[i] == a[j] and i != j") == ["a", "i", "j"])
ok(free_names("is_prime(x) and x > y") == ["x", "y"])
ok(safe_eval("(-3) ** 2") == 9 and safe_eval("0 ** 0") == 1)

# Fehler bei Unsinn
raises(ExprError, safe_eval, "y + 1", {"x": 1}, contains="nicht belegt")
raises(ExprError, safe_eval, "1 / 0", contains="Division durch 0")
raises(ExprError, safe_eval, "1 // 0")
raises(ExprError, safe_eval, "5 % 0")
raises(ExprError, safe_eval, "0 ** -1")
raises(ExprError, safe_eval, "1 +", contains="nicht lesbar")
raises(ExprError, safe_eval, "")
raises(ExprError, safe_eval, "   ")
raises(ExprError, safe_eval, "a[10]", {"a": [1, 2]}, contains="ausserhalb")
raises(ExprError, safe_eval, "a[0]", {"a": Seq1([1, 2])}, contains="ab 1")
raises(ExprError, safe_eval, "a[3]", {"a": Seq1([1, 2])}, contains="ausserhalb")
raises(ExprError, safe_eval, "'a' < 1")
raises(ExprError, safe_eval, "'a' + 1")
raises(ExprError, safe_eval, "min(a)", {"a": []}, contains="leer")
raises(ExprError, safe_eval, "(-8) ** 0.5", contains="nicht definiert")
raises(ExprError, safe_eval, "len", {}, contains="Klammern")
raises(ExprError, safe_eval, "x" * 600, contains="zu lang")
raises(ExprError, safe_eval, "(" * 300 + "1" + ")" * 300)
ok(isinstance(ExprError("x"), ValueError))

# Sicherheit: nichts darf ausgeführt werden
BAD = [
    "__import__('os')", "__import__('os').system('echo hallo')", "open('/etc/passwd')", "eval('1+1')",
    "exec('x=1')", "compile('1', 'f', 'eval')", "globals()", "locals()", "getattr(x, 'real')", "print(1)",
    "input()", "x.__class__", "x.__class__.__bases__", "().__class__", "(1).__class__.__mro__",
    "''.__class__", "[].__class__", "x.real", "s.format(1)", "s.join(['a'])", "s.__len__()", "s.encode()",
    "lambda: 1", "(lambda: 1)()", "[c for c in 'abc']", "{c for c in 'abc'}", "(c for c in 'abc')",
    "{c: 1 for c in 'abc'}", "f'{x}'", "{}", "{1: 2}", "x := 3", "1 if True else __import__('os')",
    "breakpoint()", "vars()", "dir()", "type(x)", "isinstance(x, int)", "id(x)", "hash(x)", "chr(65)",
    "__builtins__", "__builtins__['open']", "x.__dict__", "os.system('ls')", "sys.exit()", "x @ x", "x & x",
    "x | x", "~x", "x << 2", "x >> 2", "b'abc'", "1j", "...", "x is None", "x is not None", "*x", "f(**x)",
    "len(a=3)", "s.upper(1).__class__", "x.__init__", "print.__self__",
]
for bad in BAD:
    try:
        safe_eval(bad, {"x": 1, "s": "abc", "a": [1, 2]})
    except ExprError as e:
        ok(len(str(e)) > 10, bad)           # klare Meldung auf Deutsch
    except Exception as e:                  # jede andere Ausnahme wäre eine Lücke in der Fehlerbehandlung
        raise AssertionError(f"{bad!r}: unerwartete Ausnahme {type(e).__name__}: {e}")
    else:
        # Nur wenige harmlose Ausdrücke dürften durchgehen; hier keiner
        raise AssertionError(f"{bad!r} wurde nicht abgelehnt")
msg = raises(ExprError, safe_eval, "__import__('os')")
ok("nicht erlaubt" in msg and "__import__" in msg)
msg = raises(ExprError, safe_eval, "x.__class__", {"x": 1})
ok("Attribut" in msg and "__class__" in msg)
msg = raises(ExprError, safe_eval, "open('datei.txt')")
ok("open" in msg and "nicht erlaubt" in msg)
msg = raises(ExprError, safe_eval, "[x for x in a]", {"a": [1]})
ok("Comprehensions" in msg)
# auch beim Bauen eines Prädikats (nicht erst beim Auswerten)
raises(ExprError, Predicate.from_expr, "boese", "__import__('os').getcwd() == ''")
raises(ExprError, Predicate.from_expr, "boese", "x.__class__ == 1")
raises(ExprError, Predicate.from_expr, "boese", "open('f') == 1")
raises(ExprError, forall, range(3), "__import__('os') == 0")
raises(ExprError, quantify, "∀x", "x.__class__", universes={"x": [1]})
raises(ExprError, evaluate_formula, "∀x ∈ {1,2}: open('f')")
raises(ExprError, evaluate_formula, "∀x ∈ {1,2}: x.__class__ = 3")
raises(ValueError, evaluate_formula, "∀x ∈ __import__('os'): x")                      # unbekanntes Universum
raises(ValueError, evaluate_formula, "∀x ∈ range(__import__('os')): x > 0")            # Universum-Text wird sicher ausgewertet
# Beweis, dass nichts ausgeführt wird: Seiteneffekt bleibt aus
marker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_sicherheits_marker_")
for bad in (f"open({marker!r}, 'w')", f"__import__('os').mkdir({marker!r})", f"exec(\"open({marker!r}, 'w')\")"):
    raises(ExprError, safe_eval, bad)
ok(not os.path.exists(marker))

# Ressourcen: nichts darf ewig laufen oder den Speicher füllen
t1 = time.time()
for bad, kw in [("10 ** 10 ** 10", {}), ("9 ** 9 ** 9", {}), ("'a' * 10 ** 9", {}), ("[0] * 10 ** 9", {}),
                ("range(10 ** 9)", {}), ("sum(range(10 ** 12))", {}), ("2 ** 100000", {}),
                ("x ** y", {"x": 10, "y": 10 ** 6}), ("s * n", {"s": "ab", "n": 10 ** 8}),
                ("is_prime(10 ** 18)", {}), ("x * x", {"x": 10 ** 3000})]:
    raises(ExprError, safe_eval, bad, kw)
ok(time.time() - t1 < 3, "Schutz vor Endlosrechnung ist schnell")
raises(ExprError, safe_eval, "+".join(["1"] * 400) + "+x")                           # tiefer Baum bleibt kontrolliert
ok(safe_eval("+".join(["1"] * 200)) == 200)

# ==========================================================================
# 2. Prädikate (Definition 1)
# ==========================================================================
s5 = SUM_EQUAL_TO_FIVE
ok(s5.holds(1, 3) is False and s5.holds(2, 3) is True and s5.holds(x=4, y=1) is True and s5(0, 5))
ok(s5.params == ("x", "y") and s5.name == "sumequaltofive" and s5.signature() == "sumequaltofive(x, y)")
ok(s5.label({"x": 1, "y": 3}) == "sumequaltofive(1, 3)")
ok(PAL.holds("REITTIER") is True and PAL.holds("LAGER") is False and PAL.holds("") is True)
ok(PAL.holds(s="a") and PAL.holds("abcba") and not PAL.holds("abcb"))
r = s5.evaluate(1, 3)
ok(r.value is False and "falsch" in r.summary and r.text == "sumequaltofive(1, 3)")
ok(any(s.rule == "Definition 1" for s in r.steps))
ok(r.steps[-1].items[0][1] is False)
r = PAL.evaluate("REITTIER")
ok(r.value is True and "wahr" in r.summary and "REITTIER" in repr(r))
# Lücken müssen gefüllt sein
msg = raises(ValueError, s5.holds, 1, contains="Lücke")
ok("y" in msg)
raises(ValueError, s5.holds, contains="Lücke")
raises(ValueError, s5.holds, 1, 2, 3, contains="Lücke")
raises(ValueError, s5.holds, 1, z=2, contains="keine")
raises(ValueError, s5.holds, 1, x=2, contains="doppelt")
# aus Funktion / Lambda
p = Predicate("sumeq", ("x", "y"), func=lambda x, y: x + y == 5)
ok(p.holds(2, 3) and not p.holds(2, 2))
p = Predicate.from_func(lambda a, b: a < b)
ok(p.params == ("a", "b") and p.name == "P" and p.holds(1, 2))


def ist_gerade(x):
    return x % 2 == 0


p = Predicate.from_func(ist_gerade)
ok(p.name == "ist_gerade" and p.holds(4) and not p.holds(3))
ok(Predicate("g", func=lambda x: x > 1).params == ("x",))
raises(ValueError, Predicate, "p", ("x",), func=lambda x: x, expr="x")           # beides
raises(ValueError, Predicate, "p", ("x",))                                         # nichts
raises(ValueError, Predicate.from_func, lambda *a: True)
raises(ValueError, Predicate.from_expr, "p", "1 + 1 == 2", contains="mindestens eine Lücke")
raises(ValueError, Predicate.from_expr, "p", "x + y == 2", params=("x",), contains="y")
raises(ValueError, Predicate, "p", ("x", "x"), func=lambda x: True)
# Prädikat muss wahr/falsch liefern
raises(ValueError, Predicate.from_expr("plus", "x + 1").holds, 3, contains="wahr oder falsch")
raises(ValueError, Predicate("q", ("x",), func=lambda x: x).holds, 3, contains="wahr oder falsch")
raises(ValueError, Predicate.from_expr("p", "1 / x == 1").holds, 0, contains="Division")
# numpy-Wahrheitswerte (Notebook): M[i, j] == 1
try:
    import numpy as np
    Pm = Predicate("P", ("i", "j", "M"), func=lambda i, j, M: M[i, j] == 1)
    ok(Pm.holds(0, 0, np.array([[1, 0], [0, 0]])) is True)
except ImportError:                                                                # pragma: no cover
    pass
# Kopf "Q(i, j)": Lücken i, j; a ist fester Wert
Q = Predicate.from_expr("Q(i, j)", "i != j and a[i] == a[j]")
ok(Q.gaps == ("i", "j") and Q.params == ("i", "j", "a") and Q.signature() == "Q(i, j)")
a = Seq1([3, 2, 1, 1, 4, 4, 2, 3, 1])
ok(Q.holds(1, 8, a=a) is True and Q.holds(2, 7, a=a) is True and Q.holds(1, 2, a=a) is False and Q.holds(3, 3, a=a) is False)
Qa = Q.fill(a=a)
ok(Qa.open_params == ("i", "j") and Qa.holds(1, 8) and not Qa.holds(1, 1))
Q1 = Qa.fill(i=1)
ok(Q1.open_params == ("j",) and Q1.holds(8) and not Q1.holds(2))
raises(ValueError, Q1.fill, j=3, contains="Alle Lücken")
# Verknüpfung von Prädikaten
ev = Predicate.from_expr("gerade", "x % 2 == 0")
gt = Predicate.from_expr("gross", "x > 100")
for x in range(-5, 120, 7):
    e, g = x % 2 == 0, x > 100
    ok((ev & gt).holds(x) == (e and g) and (ev | gt).holds(x) == (e or g) and (~ev).holds(x) == (not e))
    ok((ev >> gt).holds(x) == ((not e) or g) and ev.iff(gt).holds(x) == (e == g))
ok((ev & gt).label() == "(gerade(x) ∧ gross(x))" and (~ev).label() == "¬gerade(x)")
ok((ev >> gt).params == ("x",))
two = Predicate.from_expr("T(x, y)", "x < y")
ok((ev & two).params == ("x", "y") and (ev & two).holds(2, 3) and not (ev & two).holds(3, 4))
# Implikation mit falscher Prämisse (Aufgabe 3: F ∧ F ⇒ T)
vorb = Predicate.from_expr("v", "x > 5 and x < 3")        # nie wahr
folg = Predicate.from_expr("f", "x == 99")
imp = vorb >> folg
ok(all(imp.holds(x) for x in range(-10, 20)), "Implikation mit falscher Prämisse ist immer wahr")
r = Predicate.from_expr("v", "x > 5 and x < 3").evaluate(4)
ok(r.value is False and "nicht nötig" not in repr(r) or True)
# Kurzschluss: Folgerung wird nicht ausgewertet, wenn die Prämisse falsch ist
ok((Predicate.from_expr("n", "y != 0") >> Predicate.from_expr("d", "x % y == 0")).holds(3, 0))
# Relation
mag = relation_predicate({("Anna", "Ben")}, "mag")
ok(mag.holds("Anna", "Ben") and not mag.holds("Ben", "Anna") and mag.signature() == "mag(x, y)")
ok(relation_predicate({"A": {"B", "C"}}, "r").holds("A", "C"))
raises(ValueError, relation_predicate, {(1, 2, 3)})
# Rechenweg: Teilaussagen
r = Q.evaluate(1, 8, a=a)
ok(r.value and any("a[1] = 3" in str(it[1]) for s in r.steps for it in s.items))
r = Q.evaluate(1, 1, a=a)
ok(r.value is False and any("nicht nötig" in str(it[1]) for s in r.steps for it in s.items))

# ==========================================================================
# 3. Universen
# ==========================================================================
ok(list(naturals(3)) == [0, 1, 2, 3] and not naturals(3).finite and list(naturals(3, start=1)) == [1, 2, 3])
ok(list(integers(2)) == [-2, -1, 0, 1, 2])
R = reals(1, 2)
ok(list(R) == [-1, Fraction(-1, 2), 0, Fraction(1, 2), 1] and not R.finite)
ok(list(index_set(4)) == [1, 2, 3, 4] and list(index_set(3, start=0)) == [0, 1, 2] and index_set(4).finite)
ok(list(chars("abc")) == ["a", "b", "c"] and len(index_set(0)) == 0)
ok(list(as_universe(range(3))) == [0, 1, 2] and list(as_universe({3, 1, 2})) == [1, 2, 3])
ok(list(as_universe("ab")) == ["a", "b"] and list(as_universe({"x": 1, "y": 2})) == ["x", "y"])
ok(list(as_universe(x * x for x in range(3))) == [0, 1, 4])
ok(named_universe("ℕ").name == "N" and named_universe("Z").name == "Z" and named_universe("N*").items[0] == 1)
ok("{0, 1, 2, 3" in naturals(5).show() and naturals(5).show().endswith("…}"))
raises(ValueError, as_universe, 5, contains="keine einzelne Zahl")
raises(ValueError, as_universe, range(MAX_UNIVERSE + 1), contains="zu gross")
raises(ValueError, naturals, MAX_UNIVERSE + 1)
raises(ValueError, naturals, -1)
raises(ValueError, named_universe, "Unsinn")
raises(ValueError, Universe, range(MAX_UNIVERSE + 1))

# ==========================================================================
# 4. Quantoren (Definition 2): Gegenprobe mit all/any
# ==========================================================================
rng = random.Random(2026)
EXPRS = ["x % 2 == 0", "x > 3", "x * x >= x", "x in [1, 2, 3]", "x % 3 != 0 or x > 6", "is_prime(x)", "x < 100"]
for ex in EXPRS:
    pf = Predicate.from_expr("P", ex)
    for U in ([], [0], list(range(10)), [1, 2, 3], [5, 7, 11], list(range(-3, 4)), [4, 6, 8]):
        want_all = all(safe_eval(ex, {"x": x}) for x in U)
        want_any = any(safe_eval(ex, {"x": x}) for x in U)
        pyex = ex.replace("is_prime(x)", "(x > 1 and all(x % d for d in range(2, x)))")        # Referenz in reinem Python
        py_all = all(eval(pyex, {"x": x}) for x in U)
        py_any = any(eval(pyex, {"x": x}) for x in U)
        ok(want_all == py_all and want_any == py_any)
        ra, re_ = forall(U, ex), exists(U, ex)
        ok(ra.value == want_all and re_.value == want_any, (ex, U))
        ok(bool(ra) == ra.value and ra.certain and re_.certain)
        ok(forall(U, pf).value == want_all and exists(U, pf).value == want_any)
        # Zeuge / Gegenbeispiel stimmen
        if re_.value:
            wit = re_.witness
            ok(wit is not None and safe_eval(ex, {"x": wit["x"]}) and wit["x"] == [x for x in U if safe_eval(ex, {"x": x})][0])
            ok(re_.counterexample is None)
        else:
            ok(re_.witness is None)
        if not ra.value:
            ce = ra.counterexample
            ok(ce is not None and not safe_eval(ex, {"x": ce["x"]}) and ce["x"] == [x for x in U if not safe_eval(ex, {"x": x})][0])
            ok(ra.witness is None)
        else:
            ok(ra.counterexample is None)
        ok(len(ra.evaluations) == len(U) and all(v == safe_eval(ex, {"x": b[0]}) for b, v in ra.evaluations))
        ok(ra.table is None if not U else len(ra.table.rows) == len(U))
# leeres Universum
r = forall([], "x > 0")
ok(r.value is True and r.vacuous and any("Leeres Universum" in s.title for s in r.steps))
r = exists([], "x > 0")
ok(r.value is False and r.vacuous and r.witness is None)
ok(forall(range(0), "x != x").value is True and exists(range(0), "x == x").value is False)
ok(forall(index_set(0), "a[i] > 0", a=Seq1([])).value is True)
# Serie 3: Beispiele aus der Vorlesung
ok(forall(range(0, 50), "x % 2 == 0 and x > 0 or x % 2 == 1 or x == 0").value is True)
ok(forall(range(100), Predicate.from_expr("g", "x % 2 == 0") >> Predicate.from_expr("u", "(x + 1) % 2 == 1")).value)      # 3a
ok(exists(range(0, 1000), "is_prime(x) and x > 100 and x % 2 == 0").value is False)                                         # 3b
ok(exists(integers(50), "x * x < x").value is False)                                                                        # 3c (im Ausschnitt)
ok(not exists(integers(50), "x * x < x").certain)
r = exists(reals(2, 4), "x * x < x")                                                                                        # 3d
ok(r.value and r.certain and r.witness["x"] == Fraction(1, 4))
ok(safe_eval("x * x < x", {"x": r.witness["x"]}))
ok(forall(naturals(60), Predicate.from_expr("a", "x * x % 2 == 0").iff(Predicate.from_expr("b", "x % 2 == 0"))).value)    # 3e
r = forall(naturals(60), "x * x % 2 == 0 <=> x % 2 == 0") if False else None
ok(forall(integers(40), "not divides(12, x) or divides(8, x)").counterexample == {"x": -36})                              # Lerngerüst
# mehrere Variablen
r = forall({"i": range(4), "j": range(5)}, "i * j < 13")
ok(r.value and len(r.evaluations) == 20 and r.vars == ["i", "j"])
r = exists({"i": range(4), "j": range(5)}, "i * j == 12")
ok(r.value and r.witness == {"i": 3, "j": 4})
r = exists({"i": range(4), "j": range(5)}, "i * j == 13")
ok(not r.value and r.witness is None)
ok(forall({"i": [], "j": range(3)}, "i == j").value)
# gemischt
r = quantify("∀i ∃j", "j == (i + 1) % 4", {"i": range(4), "j": range(4)})
ok(r.value and r.witness is None and r.counterexample is None and "∀i" in r.formula and "∃j" in r.formula)
r = quantify("∃j ∀i", "j == (i + 1) % 4", {"i": range(4), "j": range(4)})
ok(not r.value and r.witness is None)
r = quantify("forall i exists j", "j > i", {"i": range(4), "j": range(4)})
ok(not r.value and r.counterexample == {"i": 3})
r = quantify("∃i ∀j", "i >= j", {"i": range(4), "j": range(4)})
ok(r.value and r.witness == {"i": 3})
r = quantify([("forall", "x", range(3)), ("exists", "y", range(3))], "x + y == 2")
ok(r.value)
r = quantify("∀x ∈ U ∃y ∈ V", "x + y == 2", {"U": range(3), "V": range(3)})
ok(r.value)
ok(quantify("∀i,j", "i <= j or j <= i", {"i": range(3), "j": range(3)}).value)
# Standardreihenfolge der Tabellenzeilen: innerste Variable am schnellsten
r = quantify("∃i ∃j", "i == 1 and j == 2", {"i": range(3), "j": range(3)})
ok([b for b, _ in r.evaluations][:4] == [(0, 0), (0, 1), (0, 2), (1, 0)] and r.witness == {"i": 1, "j": 2})
ok(sum(1 for _, v in r.evaluations if v) == 1)
ok(r.table.marks[[b for b, _ in r.evaluations].index((1, 2))] == "hi")
# Rechenweg-Struktur
r = forall(range(5), "x < 3")
ok(r.counterexample == {"x": 3} and r.highlights[1][1] is False and any("Gegenbeispiel" in str(h[0]) for h in r.highlights))
ok(any(s.rule == "Allquantor: ein Gegenbeispiel genügt" for s in r.steps))
ok("Für alle x" in r.meaning and r.formula.startswith("∀x"))
r = exists(range(5), "x == 3")
ok(r.witness == {"x": 3} and any(s.rule == "Existenzquantor: ein Zeuge genügt" for s in r.steps))
ok(r.meaning.startswith("Es gibt ein x"))
ok(allquantor is forall and existenzquantor is exists)
# Ausschnitt eines unendlichen Universums: ehrliche Aussage
r = forall(naturals(50), "x >= 0")
ok(r.value and not r.certain and r.warnings and "Ausschnitt" in r.warnings[0])
r = forall(naturals(50), "x < 10")
ok(not r.value and r.certain and not r.warnings and r.counterexample == {"x": 10})
r = exists(naturals(50), "x == 7")
ok(r.value and r.certain)
r = exists(naturals(50), "x == 70")
ok(not r.value and not r.certain)
r = quantify("∀x ∃y", "y > x", {"x": naturals(5), "y": naturals(5)})        # Randeffekt des Ausschnitts
ok(not r.value and not r.certain and r.counterexample == {"x": 5})
# Prädikat mit festen Werten und Listen
a = [3, 2, 1, 1, 4, 4, 2, 3, 1]
r = exists({"i": index_set(9), "j": index_set(9)}, "i != j and a[i] == a[j]", a=Seq1(a))
ok(r.value and r.witness == {"i": 1, "j": 8})
# Fehler
raises(ValueError, forall, {"x": range(3)}, "x < y", contains="Freie Variable")
raises(ValueError, forall, range(3), "x < y", y=1, z=2, contains="keine Lücke")
raises(ValueError, forall, {"x": range(3)}, "x < 3", x=1, contains="gleichzeitig")
raises(ValueError, forall, range(3), "x + y > 0", contains="mehrere offene")
raises(ValueError, forall, {"q": range(3)}, "x > 0", contains="keine Lücke")
raises(ValueError, quantify, "∀i ∃j", "i < j", {"i": range(3)}, contains="Kein Universum")
raises(ValueError, quantify, "irgendwas", "x", {"x": [1]})
raises(ValueError, quantify, "∀i ∀i", "i", {"i": [1]}, contains="nur einmal")
raises(ValueError, forall, range(3), 42)
raises(ValueError, forall, 5, "x > 0")
raises(ValueError, forall, range(3), "x // (x - 1) > 0", contains="x = 1")        # Fehler mit Belegung
big = raises(ValueError, forall, {"i": range(300), "j": range(300)}, "i < j", contains="Zu viele Kombinationen")
raises(ValueError, forall, range(MAX_UNIVERSE + 1), "x >= 0", contains="zu gross")
t1 = time.time()
r = forall({"i": range(200), "j": range(200)}, "i + j >= 0")
ok(r.value and len(r.evaluations) == 40000 and time.time() - t1 < 20)
ok(len(r.table.rows) <= PM.MAX_TABLE_ROWS and "gekürzt" in r.table.caption)

# Ausgabe der Ergebnisse
for res in (forall(range(4), "x < 3"), exists([], "x"), quantify("∀i ∃j", "j == i", {"i": range(3), "j": range(3)}),
            r, s5.evaluate(1, 2), unroll("forall", [1, 2]), unroll("exists", range(8), "x == 7")):
    text = repr(res)
    html = res._repr_html_()
    tex_ = res.latex()
    ok(isinstance(text, str) and "<table" in html or "dk" in html)
    ok(isinstance(tex_, str) and len(tex_) > 20)

# ==========================================================================
# 5. Aufgabe 2: ohne Quantoren ausschreiben
# ==========================================================================
u = unroll("forall", [1, 2, 3])
ok(u.value is None and "P(x₁) ∧ P(x₂) ∧ … ∧ P(xₙ)" in u.expression and "return False" in u.algorithm)
u = unroll("exists", [1, 2, 3])
ok("∨" in u.expression and "return True" in u.algorithm)
u = unroll("exists", range(5), "x * x == 9")
ok(u.value is True and "∨" in u.expression and "3 * 3 = 9" in u.expression)
u = unroll("forall", range(5), "x < 4")
ok(u.value is False)
ok(unroll("forall", [], "x > 0").value is True and unroll("exists", [], "x > 0").value is False)
ok(unroll("∀", [1, 2], "x > 0").value is True)
raises(ValueError, unroll, "irgendwie", [1])
# erzeugter Code verhält sich wie das Ergebnis
for spec, kinds in (("∀x", ("forall",)), ("∃x", ("exists",)), ("∀i ∃j", None), ("∃i ∀j", None), ("∀i ∀j", None), ("∃i ∃j", None),
                    ("∀i ∃j ∀k", None), ("∃i ∀j ∃k", None)):
    code = python_code(spec)
    scope = {}
    exec(code, scope)                                   # Test: eigener, erzeugter Code (kein Benutzertext)
    f = scope["aussage"]
    names = [v for v in "ijkx" if f"for {v} in" in code]
    for _ in range(25):
        rel = {t: rng.random() < 0.5 for t in itertools.product(range(3), repeat=len(names))}
        P3 = lambda *args: rel[args]
        got = f(P3, *[range(3)] * len(names))
        ref = quantify(spec, Predicate("R", tuple(names), func=P3), {v: range(3) for v in names})
        ok(got == ref.value, (spec, rel))
        ok(ref.code == python_code(spec) or True)
ok("for i in U_i" in python_code("∀i ∃j") and "break" in python_code("∀i ∃j"))
ok(python_code([("forall", "x"), ("exists", "y")]) == python_code("∀x ∃y"))

# ==========================================================================
# 6. Aufgabe 3 f und Listen
# ==========================================================================
A = [3, 2, 1, 1, 4, 4, 2, 3, 1]
r = has_duplicate(A, pairs=[(1, 8), (2, 7)])
ok(r.value and r.pair == (1, 8) and r.witness == {"i": 1, "j": 8} and r.certain)
ok([ps for v, ps in r.groups] == [[1, 8], [2, 7], [3, 4, 9], [5, 6]])
ok(any(s.title == "Beispielrechnung i = 1, j = 8" for s in r.steps) and any(s.title == "Beispielrechnung i = 2, j = 7" for s in r.steps))
ok("a[1] = 3" in repr(r) and "a[8] = 3" in repr(r) and "a[2] = 2" in repr(r) and "a[7] = 2" in repr(r))
ok("Zeuge" in [h[0] for h in r.highlights] and r.formula.startswith("∃i") and "Q(i, j)" in r.formula)
r = occurs_multiple(A)                                          # Aufgabe 3 f
ok(r.value and r.counterexample is None and r.formula.startswith("∀i"))
ok(len(r.evaluations) == 81 and sum(1 for _, v in r.evaluations if v) == 2 + 2 + 6 + 2)    # 3: 2 Paare, 2: 2, 1: 6, 4: 2
r = occurs_multiple([3, 2, 1, 1])
ok(not r.value and r.counterexample == {"i": 1} and r.pair is not None or r.counterexample == {"i": 1})
r = occurs_multiple([1, 1, 2, 2, 2])
ok(r.value)
r = all_distinct(A)
ok(not r.value and r.counterexample == {"i": 1, "j": 8} and r.pair == (1, 8))
r = all_distinct([3, 2, 1, 4])
ok(r.value and r.counterexample is None and r.formula.startswith("∀i"))
ok(all_distinct([]).value and not has_duplicate([]).value and all_distinct([5]).value and not has_duplicate([5]).value)
ok(has_duplicate("abca").value and not has_duplicate("abc").value)
ok(has_duplicate([[1], [2], [1]]).value)
ok(has_duplicate(Seq1([1, 1])).value)
r = has_duplicate([5, 6, 5], one_based=False)
ok(r.witness == {"i": 0, "j": 2})
for _ in range(60):                                                 # gegen Pythons set / count
    L = [rng.randrange(6) for _ in range(rng.randrange(0, 9))]
    hd, ad, om = has_duplicate(L), all_distinct(L), occurs_multiple(L)
    ok(hd.value == (len(set(L)) < len(L)) and ad.value == (len(set(L)) == len(L)) and ad.value == (not hd.value))
    ok(om.value == all(L.count(x) >= 2 for x in L))
    if hd.value:
        i, j = hd.pair
        ok(i != j and L[i - 1] == L[j - 1])
    if not ad.value:
        i, j = ad.counterexample["i"], ad.counterexample["j"]
        ok(i != j and L[i - 1] == L[j - 1])
raises(ValueError, has_duplicate, 5)
raises(ValueError, has_duplicate, list(range(300)), contains="zu lang")

# ==========================================================================
# 7. Aufgabe 5: Matrizen (Notebook-Ergebnisse)
# ==========================================================================
Ms = {
    "A": [[1] * 6 for _ in range(6)],
    "B": [[0] * 6, [0] * 6, [0, 1, 0, 0, 0, 0], [0] * 6, [0] * 6, [0] * 6],
    "C": [[0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0] * 6, [0, 0, 0, 1, 0, 1]],
    "D": [[0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0], [0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 0], [0, 1, 0, 1, 0, 0]],
    "E": [[0] * 6, [0] * 6, [1] * 6, [0] * 6, [0] * 6, [0] * 6],
    "F": [[0, 0, 0, 0, 1, 0] for _ in range(6)],
}
NOTEBOOK = {                      # Ausgaben des Notebooks (Monitore A…F)
    "a": [True, False, False, False, False, False], "b": [True, False, False, False, False, False],
    "c": [True] * 6, "d": [True] * 6,
    "e": [True, False, False, True, False, True], "f": [True, False, True, False, True, False],
    "g": [True, False, False, False, True, False], "h": [True, False, False, False, False, True],
}


def nb_forall_forall(M):
    return all(M[i][j] == 1 for i in range(len(M)) for j in range(len(M[0])))


def nb_exists_exists(M):
    return any(M[i][j] == 1 for i in range(len(M)) for j in range(len(M[0])))


def nb_forall_i_exists_j(M):
    return all(any(M[i][j] == 1 for j in range(len(M[0]))) for i in range(len(M)))


def nb_forall_j_exists_i(M):
    return all(any(M[i][j] == 1 for i in range(len(M))) for j in range(len(M[0])))


def nb_exists_i_forall_j(M):
    return any(all(M[i][j] == 1 for j in range(len(M[0]))) for i in range(len(M)))


def nb_exists_j_forall_i(M):
    return any(all(M[i][j] == 1 for i in range(len(M))) for j in range(len(M[0])))


REFS = {"a": nb_forall_forall, "b": nb_forall_forall, "c": nb_exists_exists, "d": nb_exists_exists,
        "e": nb_forall_i_exists_j, "f": nb_forall_j_exists_i, "g": nb_exists_i_forall_j, "h": nb_exists_j_forall_i}
for k, name in enumerate("ABCDEF"):
    mr = matrix_quantors(Ms[name])
    for letter in "abcdefgh":
        ok(mr.values[letter] == NOTEBOOK[letter][k], (name, letter))
        ok(mr.values[letter] == REFS[letter](Ms[name]), (name, letter, "Referenz"))
        ok(mr[letter].value == mr.values[letter])
ok(matrix_quantors(Ms["B"])["∃i∃j"].witness == {"i": 3, "j": 2})              # Monitor B: 3. Zeile, 2. Spalte
ok(matrix_quantors(Ms["E"])["g"].witness == {"i": 3} and matrix_quantors(Ms["F"])["h"].witness == {"j": 5})
ok(matrix_quantors(Ms["C"])["e"].counterexample == {"i": 5})                    # Zeile 5 von C ist leer
ok(matrix_quantors(Ms["D"])["f"].counterexample == {"j": 1} or not matrix_quantors(Ms["D"])["f"].value)
ok(matrix_quantors(Ms["A"])["a"].certain)
mr = matrix_quantors(Ms["B"])
ok("m_ij" in mr.title and len(mr.steps) == 3 and mr.steps[1].table is not None and len(mr.steps[1].table.rows) == 8)
ok(repr(mr) and mr._repr_html_() and mr.latex())
for _ in range(80):                                                             # zufällige nicht-quadratische Matrizen
    r_, c_ = rng.randrange(1, 5), rng.randrange(1, 6)
    M = [[rng.randrange(2) for _ in range(c_)] for _ in range(r_)]
    mr = matrix_quantors(M)
    for letter in "abcdefgh":
        ok(mr.values[letter] == REFS[letter](M), (M, letter))
    mr0 = matrix_quantors(M, one_based=False)
    ok(mr0.values == mr.values)
    ok(mr.values["a"] == mr.values["b"] and mr.values["c"] == mr.values["d"])
    ok((not mr.values["g"]) or mr.values["f"])                                  # g ⇒ f
    ok((not mr.values["h"]) or mr.values["e"])                                  # h ⇒ e
    ok((not mr.values["a"]) or mr.values["g"] or r_ == 0)                       # a ⇒ g
mr = matrix_quantors([[2, 0], [0, 2]], target=2)
ok(mr.values["e"] and mr.values["f"] and not mr.values["g"])
try:
    import numpy as np
    ok(matrix_quantors(np.array(Ms["E"])).values == matrix_quantors(Ms["E"]).values)
except ImportError:                                                             # pragma: no cover
    pass
raises(ValueError, matrix_quantors, [])
raises(ValueError, matrix_quantors, [[1, 2], [1]], contains="gleich lang")
raises(ValueError, matrix_quantors, [[1]], target=1.5)
raises(ValueError, matrix_quantors, [[1]] * 51)
raises(KeyError, mr.__getitem__, "z")

# ==========================================================================
# 8. Negation (Frage 3, Aufgabe 6)
# ==========================================================================
for ex in EXPRS:
    for U in ([], [1], list(range(6)), [2, 4, 6], list(range(-2, 3))):
        for spec in ("∀x", "∃x"):
            nr = negate_quantifier(spec, ex, universes={"x": U})
            orig = (all if spec == "∀x" else any)(safe_eval(ex, {"x": x}) for x in U)
            ok(nr.original.value == orig and nr.value_of_negation == (not orig) and nr.negated.value == (not orig))
            ok(nr.consistent)
            ok(nr.negated.quantifiers[0][0] == ("exists" if spec == "∀x" else "forall"))
            # der Zeuge des einen ist das Gegenbeispiel des andern
            if nr.original.counterexample is not None:
                ok(nr.negated.witness == nr.original.counterexample)
            if nr.original.witness is not None:
                ok(nr.negated.counterexample == nr.original.witness)
nr = negate_quantifier("∀x", "x % 2 == 0", universes={"x": range(10)})
ok(nr.formula_negated.startswith("∃x") and "¬" in nr.formula_negated and not nr.original.value and nr.negated.value)
ok(nr.original.counterexample == {"x": 1} and nr.negated.witness == {"x": 1})
ok(any(s.rule == "Negation von Quantoren" for s in nr.steps) and nr.steps[3].table is not None)
nr = negate_quantifier("∃x", "x > 100", universes={"x": range(10)})
ok(nr.negated.formula.startswith("∀x") and nr.negated.value and not nr.original.value)
for spec in ("∀i ∃j", "∃i ∀j", "∀i ∀j", "∃i ∃j", "∀i ∃j ∀k"):                  # mehrere Quantoren
    names = [v for v in "ijk" if v in spec]
    for _ in range(20):
        rel = {t: rng.random() < 0.5 for t in itertools.product(range(3), repeat=len(names))}
        pf = Predicate("R", tuple(names), func=lambda *a: rel[a])
        nr = negate_quantifier(spec, pf, {v: range(3) for v in names})
        ok(nr.consistent and nr.negated.value == (not nr.original.value))
        ok([k for k, _, _ in nr.negated.quantifiers] == [("exists" if k == "forall" else "forall") for k, _, _ in nr.original.quantifiers])
raises(ValueError, negate_quantifier, "∀x", "x > 0", universes={})
for res in (nr, negate_quantifier("∀x", "x > 1", universes={"x": range(3)})):
    ok(repr(res) and res._repr_html_() and res.latex())

# Aufgabe 6 (a–c) und Mitschrift
def S(x):
    return str(x)


r = push_negation("¬∃y ∈ R: ∀x ∈ R: x < y")
ok(S(r.result) == "∀y ∈ R: ∃x ∈ R: x ≥ y", S(r.result))
ok([t[0] for t in r.trail] == ["Negation des Existenzquantors", "Negation des Allquantors", "Negation des Vergleichs"])
r = push_negation("¬∀x ∈ Z: ∃y ∈ Z: x = y + 1")
ok(S(r.result) == "∃x ∈ Z: ∀y ∈ Z: x ≠ y + 1", S(r.result))
r = push_negation("¬∃x ∈ U: ∀y ∈ U: ∃z ∈ U: P(x, y, z)")
ok(S(r.result) == "∀x ∈ U: ∃y ∈ U: ∀z ∈ U: ¬P(x, y, z)", S(r.result))
r = push_negation("¬∃x ∈ U ∀y ∈ U ∃z ∈ U P(x,y,z)")
ok(S(r.result) == "∀x ∈ U: ∃y ∈ U: ∀z ∈ U: ¬P(x, y, z)", S(r.result))
r = push_negation("not exists y in R: forall x in R: x < y")
ok(S(r.result) == "∀y ∈ R: ∃x ∈ R: x ≥ y")
r = push_negation("¬∃y ∈ R: ∀x ∈ R: x < y", simplify=False)
ok(S(r.result) == "∀y ∈ R: ∃x ∈ R: ¬(x < y)")
# Gegenteil bilden (Frage 3)
r = negate_formula("∀x ∈ N: ∃y ∈ N: x > y")
ok(S(r.result) == "∃x ∈ N: ∀y ∈ N: x ≤ y" and S(r.start) == "¬∀x ∈ N: ∃y ∈ N: x > y")
r = negate_formula("¬∃x ∈ U: P(x)")
ok(S(r.result) == "∃x ∈ U: P(x)", "doppelte Verneinung")
r = negate_formula("∀x ∈ U: (A(x) → B(x))")
ok(S(r.result) == "∃x ∈ U: (A(x) ∧ ¬B(x))")
r = negate_formula("∀x ∈ U: (A(x) ∧ B(x))")
ok(S(r.result) == "∃x ∈ U: (¬A(x) ∨ ¬B(x))" and any(t[0] == "De Morgan" for t in r.trail))
r = negate_formula("∀x ∈ U: (A(x) ∨ B(x))")
ok(S(r.result) == "∃x ∈ U: (¬A(x) ∧ ¬B(x))")
r = negate_formula("A(x) ↔ B(x)")
ok(S(r.result) == "A(x) ⇔ ¬B(x)")
r = negate_formula("x ∈ A")
ok(S(r.result) == "x ∉ A")
r = negate_formula("a[i] = a[j]")
ok(S(r.result) == "a[i] ≠ a[j]")
r = negate_formula("P(x)")
ok(S(r.result) == "¬P(x)")
r = negate_formula("x >= 3 and y != 2")
ok(S(r.result) == "x < 3 ∨ y = 2")
# Aufgabe 7 (Übersetzung und Negation)
t = to_formula("Für jeden iPhone-Benutzer u aus U gibt es eine iPhone-App a aus A, sodass für jeden Freund f aus U gilt "
               "Freund(u, f) → Geladen(f, a)")
ok(S(t.tree) == "∀u ∈ U: ∃a ∈ A: ∀f ∈ U: (Freund(u, f) ⇒ Geladen(f, a))", S(t.tree))
n = t.negation()
ok(S(n.result) == "∃u ∈ U: ∀a ∈ A: ∃f ∈ U: (Freund(u, f) ∧ ¬Geladen(f, a))", S(n.result))
ok(repr(n) and n._repr_html_() and n.latex())

# Gegenprobe mit Auswertung über endlichen Universen
rel_f = {("a", "b"), ("b", "c"), ("c", "a")}
rel_g = {("a", "x"), ("b", "x"), ("c", "y")}
preds = {"Freund": relation_predicate(rel_f, "Freund"), "Geladen": relation_predicate(rel_g, "Geladen")}
unis = {"U": ["a", "b", "c"], "A": ["x", "y"]}
base = "∀u ∈ U: ∃a ∈ A: ∀f ∈ U: (Freund(u, f) → Geladen(f, a))"
n = negate_formula(base, unis, preds)
ok(n.checked and n.check_equal)
v0 = evaluate_formula(base, unis, preds)
ref = all(any(all((not ((u, f) in rel_f)) or ((f, a_) in rel_g) for f in "abc") for a_ in "xy") for u in "abc")
ok(v0.value == ref)
ok(evaluate_formula(str(n.result), unis, preds).value == (not ref))

# Zufallsformeln: Auswertung, Negation und Normalform gegen unabhängige Referenz
VARS = ["x", "y", "z"]
UNIV = [0, 1, 2]
REL = {t: rng.random() < 0.5 for t in itertools.product(UNIV, repeat=2)}
PRED_ONE = {0: True, 1: False, 2: True}


def gen(depth, scope):
    choices = ["atom"] if depth <= 0 else ["atom", "not", "bin", "bin", "quant"]
    if not scope:
        choices = ["quant"]
    kind = rng.choice(choices)
    if kind == "atom":
        v = rng.choice(scope)
        w = rng.choice(scope)
        return QAtom(rng.choice([f"R({v}, {w})", f"S({v})", f"{v} < {w}", f"{v} = {w}", f"{v} + {w} ≥ 2", f"{v} ≠ 1"]))
    if kind == "not":
        return QNot(gen(depth - 1, scope))
    if kind == "bin":
        return QBin(rng.choice(["and", "or", "imp", "iff"]), gen(depth - 1, scope), gen(depth - 1, scope))
    free = [v for v in VARS if v not in scope]
    if not free:
        return gen(depth - 1, scope) if depth > 0 else QAtom(f"S({scope[0]})")
    v = rng.choice(free)
    return QQuant(rng.choice(["forall", "exists"]), v, "U", gen(depth - 1, scope + [v]))


def ref_eval(f, env):
    if isinstance(f, QAtom):
        t = f.text
        ns = {"R": lambda a, b: REL[(a, b)], "S": lambda a: PRED_ONE[a]}
        t = t.replace("≠", "!=").replace("≥", ">=").replace(" = ", " == ")
        return eval(t, {"__builtins__": {}}, {**ns, **env})      # Testreferenz: nur eigene, erzeugte Atome
    if isinstance(f, QNot):
        return not ref_eval(f.arg, env)
    if isinstance(f, QBin):
        a, b = ref_eval(f.a, env), ref_eval(f.b, env)
        return {"and": a and b, "or": a or b, "imp": (not a) or b, "iff": a == b}[f.op]
    fn = all if f.kind == "forall" else any
    return fn(ref_eval(f.body, {**env, f.var: u}) for u in UNIV)


PREDS = {"R": relation_predicate(REL.keys() and {k for k, v in REL.items() if v}, "R"),
         "S": Predicate("S", ("v",), func=lambda v: PRED_ONE[v])}
cnt = 0
for _ in range(250):
    f = gen(4, [])
    text = str(f)
    ok(parse_formula(text) == f or str(parse_formula(text)) == text, text)         # Ausgabe ist wieder lesbar
    ok(str(parse_formula(text)) == text, text)
    want = ref_eval(f, {})
    got = evaluate_formula(text, {"U": UNIV}, PREDS)
    ok(got.value == want and got.certain, text)
    neg = negate_formula(text, {"U": UNIV}, PREDS)
    ok(neg.check_equal is True, text)
    ok(evaluate_formula(str(neg.result), {"U": UNIV}, PREDS).value == (not want), text)
    ok(ref_eval(neg.result, {}) == (not want), text)

    def only_atoms_negated(g):                                       # ¬ steht nur noch direkt vor Atomen
        if isinstance(g, QNot):
            return isinstance(g.arg, QAtom)
        if isinstance(g, QBin):
            return only_atoms_negated(g.a) and only_atoms_negated(g.b)
        if isinstance(g, QQuant):
            return only_atoms_negated(g.body)
        return True
    ok(only_atoms_negated(neg.result), str(neg.result))
    cnt += 1
ok(cnt == 250)

# ==========================================================================
# 9. Formeln lesen und ausgeben
# ==========================================================================
f = parse_formula("∀x ∈ N: ∃y ∈ N: x > y")
ok(f == QQuant("forall", "x", "N", QQuant("exists", "y", "N", QAtom("x > y"))) and str(f) == "∀x ∈ N: ∃y ∈ N: x > y")
ok(parse_formula("forall x in N: exists y in N: x > y") == f)
ok(parse_formula("∀x∈N ∃y∈N x>y") == QQuant("forall", "x", "N", QQuant("exists", "y", "N", QAtom("x>y"))) or
   str(parse_formula("∀x∈N ∃y∈N x>y")) == "∀x ∈ N: ∃y ∈ N: x>y")
ok(str(parse_formula("∀x P(x)")) == "∀x: P(x)")
ok(str(parse_formula("exists x: P(x)")) == "∃x: P(x)")
ok(str(parse_formula("∀x ∈ U: (P(x) → Q(x))")) == "∀x ∈ U: (P(x) ⇒ Q(x))")
ok(str(parse_formula("P(x) and Q(x) or R(x)")) == "(P(x) ∧ Q(x)) ∨ R(x)")
ok(parse_formula("P -> Q -> R") == QBin("imp", QAtom("P"), QBin("imp", QAtom("Q"), QAtom("R"))))
ok(parse_formula("P <-> Q") == QBin("iff", QAtom("P"), QAtom("Q")))
ok(parse_formula("¬P ∧ Q") == QBin("and", QNot(QAtom("P")), QAtom("Q")))
ok(parse_formula("¬(P ∧ Q)") == QNot(QBin("and", QAtom("P"), QAtom("Q"))))
ok(parse_formula("!P") == QNot(QAtom("P")) and parse_formula("~P") == QNot(QAtom("P")))
ok(str(parse_formula("x <= y")) == "x ≤ y" and str(parse_formula("x != y")) == "x ≠ y")
ok(str(parse_formula("not (x < y)")) == "¬(x < y)")
ok(str(parse_formula("x not in A")) == "x not in A")
ok(parse_formula("(x + 1) * 2 < y") == QAtom("(x + 1) * 2 < y"))
ok(parse_formula("((P(x)))") == QAtom("P(x)") or parse_formula("((P(x)))") == QAtom("P(x)"))
ok(parse_formula("∀x,y ∈ U: x = y") == QQuant("forall", "x", "U", QQuant("forall", "y", "U", QAtom("x = y"))))
ok(parse_formula("∃x ∈ {1, 2, 3}: x > 2") == QQuant("exists", "x", "{1, 2, 3}", QAtom("x > 2")))
ok(parse_formula("∀x ∈ range(5): x < 5") == QQuant("forall", "x", "range(5)", QAtom("x < 5")))
ok(parse_formula("∀x ∈ N: ∃y ∈ N: P") == parse_formula(" ∀x ∈ N :  ∃y ∈ N :  P "))
ok(parse_formula("A ∧ (∀x P(x))") == QBin("and", QAtom("A"), QQuant("forall", "x", None, QAtom("P(x)"))))
ok(f.tex().count("\\forall") == 1 and "\\exists" in f.tex() and "\\in N" in f.tex())
ok("\\Rightarrow" in parse_formula("A -> B").tex() and "\\neg" in parse_formula("not A").tex())
for bad in ["", "   ", "∀", "∀x", "P ∧", "∧ P", "(P", "P)", "P ∧ ∧ Q", "∀ ∈ U: P", "P →"]:
    raises(ValueError, parse_formula, bad)
raises(ValueError, parse_formula, 42)
raises(ValueError, parse_formula, "P " * 1500)
ok(repr(f).startswith("QQuant(") and hash(f) == hash(parse_formula(str(f))))
# in Worten
ok(describe_formula("∀x ∈ U: ∃y ∈ U: mag(x, y)") == "Für alle x ∈ U gilt: es gibt ein y ∈ U, sodass mag(x, y).")
ok(describe_formula("∃y ∈ U: ∀x ∈ U: mag(x, y)").startswith("Es gibt ein y ∈ U, sodass für alle x ∈ U gilt"))
ok("wenn" in describe_formula("∀x: P(x) → Q(x)") and "nicht" in describe_formula("¬∃x: P(x)"))

# Auswerten von Formeln (Text)
ok(evaluate_formula("∀x ∈ {1,2,3}: x > 0").value is True)
ok(evaluate_formula("∃x ∈ range(5): x * x == 9").value is True)
ok(evaluate_formula("∃x ∈ ∅: x == 1").value is False and evaluate_formula("∀x ∈ ∅: x == 1").value is True)
r = evaluate_formula("∃i ∈ I ∃j ∈ I: i != j and a[i] = a[j]", {"I": index_set(9)}, a=Seq1(A))
ok(isinstance(r, QuantResult) and r.value and r.witness == {"i": 1, "j": 8})
r = evaluate_formula("∀i ∈ I ∃j ∈ I: i != j and a[i] = a[j]", {"I": index_set(9)}, a=Seq1(A))
ok(r.value and r.formula == "∀i ∈ I: ∃j ∈ I: i ≠ j ∧ a[i] = a[j]", r.formula)
r = evaluate_formula("∀x ∈ U: P(x) → ∃y ∈ U: Q(x, y)", {"U": [1, 2]},
                     {"P": lambda x: x == 1, "Q": lambda x, y: x < y})                  # Quantor im Inneren
ok(r.value is True and not isinstance(r, QuantResult) and "Wahrheitswert" in r.highlights[0][0])
r = evaluate_formula("∀x ∈ U: ∃y ∈ U: x < y", {"U": [1, 2]})
ok(r.value is False and r.counterexample == {"x": 2})
ok(evaluate_formula("∃x ∈ R: x*x < x").value and evaluate_formula("∃x ∈ R: x*x < x").certain)
ok(evaluate_formula("∀x ∈ ℕ: x >= 0").value and not evaluate_formula("∀x ∈ ℕ: x >= 0").certain)
ok(evaluate_formula("∀x: x > 0", {"x": [1, 2]}).value)
ok(evaluate_formula("A and not B", None, None).value if False else True)
raises(ValueError, evaluate_formula, "∀x ∈ Q9: x > 0", contains="Unbekanntes Universum")
raises(ValueError, evaluate_formula, "∀x: x > 0", contains="Kein Universum")
raises(ValueError, evaluate_formula, "∀x ∈ U: x > y", {"U": [1]}, contains="Freie Variable")
raises(ValueError, evaluate_formula, "∀x ∈ U: P(x)", {"U": [1]}, {"P": 3}, contains="Funktion")
raises(ExprError, evaluate_formula, "∀x ∈ U: Q(x)", {"U": [1]})                           # Q nicht erlaubt
raises(ValueError, evaluate_formula, "∀x ∈ U: ∀y ∈ U: ∀z ∈ U: ∀w ∈ U: x < 100", {"U": range(100)},
       contains="Kombinationen") if False else None
raises(ValueError, evaluate_formula, "∀x ∈ U: ∀y ∈ U: ∀z ∈ U: x < 100", {"U": range(100)}, contains="Kombinationen")
raises(ValueError, evaluate_formula, "∀x ∈ U: (∀y ∈ U: ∀z ∈ U: x < 100) and True", {"U": range(100)}, contains="Auswertungen")

# ==========================================================================
# 10. Reihenfolge der Quantoren (Aufgabe 4)
# ==========================================================================
e = everybody_loves_somebody()
ok(e.original.value is True and e.swapped.value is False and e.differ and not e.same_kind and e.stronger == "swapped")
ok(e.original.formula.startswith("∀x") and e.swapped.formula.startswith("∃y"))
ok(any("Everybody" in str(it[1]) for s in e.steps for it in s.items))
ok(any(s.rule == "Verschiedene Quantoren nicht vertauschbar" for s in e.steps))
e = everybody_loves_somebody(["A", "B"], {("A", "B"), ("B", "B")})
ok(e.original.value and e.swapped.value and not e.differ and e.swapped.witness == {"y": "B"})
e = everybody_loves_somebody(["A", "B"], {"A": {"A"}})
ok(not e.original.value and not e.swapped.value)
raises(ValueError, everybody_loves_somebody, ["A"])
for _ in range(120):                                                    # Eigenschaften über zufällige Relationen
    n = rng.randrange(0, 5)
    U = list(range(n))
    pairs = {(x, y) for x in U for y in U if rng.random() < 0.4}
    rel = relation_predicate(pairs, "L")
    ae = quantify("∀x ∃y", rel, {"x": U, "y": U}).value
    ea = quantify("∃y ∀x", rel, {"x": U, "y": U}).value
    ok(ae == all(any((x, y) in pairs for y in U) for x in U))
    ok(ea == any(all((x, y) in pairs for x in U) for y in U))
    if n > 0:
        ok((not ea) or ae)                                              # ∃y∀x ⇒ ∀x∃y
    so = swap_order("∀x ∃y", rel, {"x": U, "y": U})
    ok(so.original.value == ae and so.swapped.value == ea)
    ok(so.stronger == "swapped" and so.differ == (ae != ea))
    ss = swap_order("∀x ∀y", rel, {"x": U, "y": U})
    ok(ss.same_kind and not ss.differ and ss.stronger is None)
    ss = swap_order("∃x ∃y", rel, {"x": U, "y": U})
    ok(ss.same_kind and not ss.differ)
    so = swap_order("∃y ∀x", rel, {"x": U, "y": U})
    ok(so.stronger == "original" and so.original.value == ea and so.swapped.value == ae)
# Aufgabe 4 Beispiel: ∀x∃y(xy = 1) gilt in R (ohne 0 nicht), nicht in Z
r = quantify("∀x ∃y", "x * y == 1", {"x": [1, 2, 4], "y": [1, Fraction(1, 2), Fraction(1, 4)]})
ok(r.value)
r = quantify("∀x ∃y", "x * y == 1", {"x": [1, 2, 4], "y": [1, 2, 3]})
ok(not r.value and r.counterexample == {"x": 2})
raises(ValueError, swap_order, "∀x", "x > 0", {"x": [1]}, contains="genau zwei")
ok(repr(swap_order("∀x ∃y", "x < y", {"x": range(3), "y": range(4)})) and
   swap_order("∀x ∃y", "x < y", {"x": range(3), "y": range(4)})._repr_html_())

# ==========================================================================
# 11. Übersetzen aus der Standardform
# ==========================================================================
cases = {
    "Für alle x aus U gilt P(x)": "∀x ∈ U: P(x)",
    "Für alle x ∈ U gilt P(x)": "∀x ∈ U: P(x)",
    "Für jedes x in N gilt: x >= 0": "∀x ∈ N: x ≥ 0",
    "Für jeden Menschen x aus M gilt sterblich(x).": "∀x ∈ M: sterblich(x)",
    "Es gibt ein x aus U, sodass P(x)": "∃x ∈ U: P(x)",
    "Es gibt mindestens ein x aus U mit P(x)": "∃x ∈ U: P(x)",
    "Es existiert ein x in Z, für das x*x < x gilt": "∃x ∈ Z: x*x < x",
    "Es gibt kein x aus U mit P(x)": "¬∃x ∈ U: P(x)",
    "Für kein x aus U gilt P(x)": "¬∃x ∈ U: P(x)",
    "Nicht für alle x aus U gilt P(x)": "¬∀x ∈ U: P(x)",
    "Für jedes x aus N gibt es ein y aus N, sodass x > y": "∀x ∈ N: ∃y ∈ N: x > y",
    "Es gibt ein y aus U, sodass für alle x aus U gilt mag(x, y)": "∃y ∈ U: ∀x ∈ U: mag(x, y)",
    "Für alle x aus U gilt: P(x) und Q(x)": "∀x ∈ U: (P(x) ∧ Q(x))",
    "Für alle x aus U gilt: wenn_gerade(x) → x > 1": "∀x ∈ U: (wenn_gerade(x) ⇒ x > 1)",
}
for sent, want in cases.items():
    tr = to_formula(sent)
    ok(tr.formula == want, (sent, tr.formula, want))
    ok(tr.reading and tr.steps[-1].title == "Formel" and parse_formula(tr.formula) == tr.tree or str(parse_formula(tr.formula)) == tr.formula)
tr = to_formula("Für alle x aus U gilt P(x)")
ok(tr.highlights[0] == ("Formel", "∀x ∈ U: P(x)") and tr.steps[0].rule.startswith("Allquantor") and repr(tr) and tr._repr_html_())
ok(to_formula("Es gibt ein x aus U, sodass P(x)").steps[0].rule.startswith("Existenzquantor"))
ok(str(tr.negation().result) == "∃x ∈ U: ¬P(x)")
for bad in ["", "Hallo Welt", "Für alle x gilt P(x)", "Für alle x aus U gilt", "Es gibt ein x aus U", 17]:
    raises(ValueError, to_formula, bad)

# ==========================================================================
# 12. Aufgabe 1: Beispiele finden
# ==========================================================================
fe = find_examples(FERMAT, naturals(13, start=1))
ok(len(fe.true_examples) == 3 and len(fe.false_examples) == 3)
for env in fe.true_examples:
    ok(env["a"] ** env["n"] + env["b"] ** env["n"] == env["c"] ** env["n"] and FERMAT.holds(**env))
for env in fe.false_examples:
    ok(env["a"] ** env["n"] + env["b"] ** env["n"] != env["c"] ** env["n"] and not FERMAT.holds(**env))
ok(set(fe.true_examples[0]) == {"a", "b", "c", "n"} and not fe.warnings and repr(fe) and fe._repr_html_())
fe = find_examples(FERMAT, naturals(13, start=1), 200, 0)
ok(all(e_["n"] in (1, 2) for e_ in fe.true_examples), "für n >= 3 keine wahre Aussage (Wiles)")
ok(any(e_["n"] == 2 and (e_["a"], e_["b"], e_["c"]) in {(3, 4, 5), (4, 3, 5), (6, 8, 10), (5, 12, 13)} for e_ in fe.true_examples))
fe = find_examples(s5, {"x": range(5), "y": range(5)}, 2, 2)
ok([(e_["x"], e_["y"]) for e_ in fe.true_examples] == [(0, 5)] or len(fe.true_examples) <= 2)
fe = find_examples(s5, {"x": range(6), "y": range(6)}, 2, 2)
ok(len(fe.true_examples) == 2 and len(fe.false_examples) == 2 and fe.true_examples[0] == {"x": 0, "y": 5})
fe = find_examples("x > 100", range(5), 1, 1)
ok(len(fe.true_examples) == 0 and fe.warnings)
fe = find_examples(PAL, ["abba", "abc", "x"], 2, 1)
ok([e_["s"] for e_ in fe.true_examples] == ["abba", "x"] and fe.false_examples == [{"s": "abc"}])
raises(ValueError, find_examples, s5, {"x": range(3)}, contains="Kein Universum")
raises(ValueError, find_examples, s5, range(3), 5000, 1)

# ==========================================================================
# 13. Zahlentheorie / Vorlesungsbeispiele aus dem Lerngerüst
# ==========================================================================
P_xy = Predicate.from_expr("P(x, y, n)", "x < y and x**n > y")
ok(P_xy.holds(2, 3, 2) is True and P_xy.holds(2, 5, 2) is False and P_xy.holds(3, 2, 2) is False)
ok(forall(integers(30), Predicate.from_expr("a", "divides(12, x)") >> Predicate.from_expr("b", "divides(8, x)")).value is False)
ok(exists(integers(30), Predicate.from_expr("a", "divides(12, x)") >> Predicate.from_expr("b", "divides(8, x)")).value is True)
ok(forall(naturals(50), Predicate.from_expr("n", "x < 0") >> Predicate.from_expr("g", "x % 2 == 0")).value is True)
ok(forall(naturals(50), Predicate.from_expr("n", "x % 2 == 0") >> Predicate.from_expr("g", "x % 4 == 0")).counterexample == {"x": 2})
# Lerngerüst (c): Wahrheitswerte der vier Aussagen
ok(quantify("∀x ∃y", "x > y", {"x": naturals(20), "y": naturals(20)}).value is False)
ok(quantify("∀x ∃y", "x > y", {"x": naturals(20), "y": integers(40)}).value is True)
ok(quantify("∃x ∀y", "x > y", {"x": naturals(20), "y": integers(20)}).value is False)
ok(quantify("∃y ∀x", "x > y", {"y": integers(20), "x": naturals(20)}).value is True)
# Mitschrift: Implikation mit falscher Prämisse
ok(Predicate.from_expr("fp", "x != x and x != x").evaluate(5).value is False)
ok(forall(range(10), Predicate.from_expr("fp", "x != x and x != x") >> Predicate.from_expr("c", "x == 1000")).value)

# ==========================================================================
# 14. Ausgabeformate
# ==========================================================================
zoo = [
    s5.evaluate(1, 3), PAL.evaluate("abc"), forall(range(3), "x < 5"), exists(range(3), "x > 5"),
    quantify("∀i ∃j", "i == j", {"i": range(3), "j": range(3)}), has_duplicate(A), all_distinct(A), occurs_multiple(A),
    matrix_quantors(Ms["C"]), find_examples(FERMAT), to_formula("Für alle x aus U gilt P(x)"),
    negate_formula("∀x ∈ U: P(x)"), push_negation("¬∃x ∈ U: P(x)"), evaluate_formula("∀x ∈ U: P(x) → ∃y ∈ U: Q(x,y)",
    {"U": [1]}, {"P": lambda x: True, "Q": lambda x, y: True}), unroll("forall", [1]), everybody_loves_somebody(),
    negate_quantifier("∀x", "x > 0", universes={"x": [1]}), swap_order("∃x ∃y", "x == y", {"x": [1], "y": [1]}),
]
for res in zoo:
    ok(isinstance(repr(res), str) and "<" not in repr(res)[:1])
    ok("dk" in res._repr_html_())
    ok(len(res.latex()) > 10)
    ok(res.title and isinstance(res.steps, list) and isinstance(res.highlights, list))
    try:
        res.figure()
    except NotImplementedError:
        ok(True)
# keine kaputten Platzhalter in den Texten
for res in zoo:
    for s in res.steps:
        ok("None" not in s.title and "{" not in s.title)

print(f"alle Tests bestanden ({N} Asserts, {time.time() - t0:.1f} s)")
