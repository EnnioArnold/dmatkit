#!/usr/bin/env python3
"""Tests für dmatkit.rechnen.rekursion (Semesterwoche 6): python3 tests/tests_rekursion.py

Soll-Werte stammen aus Serie 6 (Aufgabenstellung), von Hand oder aus unabhängiger Python-Rekursion /
Iteration — nie aus dem Ergebnis des Moduls selbst.
"""
import math
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import sympy as sp

import dmatkit as dk
from dmatkit.rechnen import rekursion as R
from dmatkit.rechnen.rekursion import (RECURSIVE_CODE, hanoi, master_theorem, recursion_tree,
                                       solve_recurrence)

N = [0]


def ok(cond, msg=""):
    N[0] += 1
    assert cond, msg


def eq(a, b, msg=""):
    N[0] += 1
    assert a == b, f"{msg}: {a!r} != {b!r}"


def raises(exc, fn, *a, text=None, **k):
    N[0] += 1
    try:
        fn(*a, **k)
    except exc as e:
        if text is not None:
            assert text in str(e), f"{text!r} nicht in {str(e)!r}"
        return e
    except Exception as e:                                   # falscher Typ
        raise AssertionError(f"{exc.__name__} erwartet, bekam {type(e).__name__}: {e}")
    raise AssertionError(f"{exc.__name__} erwartet, aber keine Ausnahme")


def shows(res):
    """Jede Darstellung läuft durch."""
    ok(isinstance(repr(res), str) and "Traceback" not in repr(res))
    ok(res._repr_html_().startswith("<style>") and isinstance(res.latex(), str))


# ==========================================================================
# 1. Rekursionsbäume (Aufgaben 1 bis 3)
# ==========================================================================
for alg, n, wert in (("summe", 5, 15), ("fakultaet", 5, 120)):
    r = recursion_tree(alg, n)
    eq(r.value, wert, alg)
    eq((r.calls, r.height, [len(lv) for lv in r.levels]), (n, n, [1] * n), alg)
    eq([k.label for k in r.nodes], [f"{alg}({m})" for m in range(n, 0, -1)])
    shows(r)
for n in range(1, 13):
    eq(recursion_tree("summe", n).value, n * (n + 1) // 2)
    eq(recursion_tree("fakultaet", n).value, math.factorial(n))
    eq(recursion_tree("summe", n).calls, n)

A = [4, 6, 5, 1, 7, 4, 3, 1]
r = recursion_tree("mergeSort", A)
eq(r.value, sorted(A))
eq((r.calls, r.height, [len(lv) for lv in r.levels]), (15, 4, [1, 2, 4, 8]))
eq([k.label for k in r.levels[1]], ["[4, 6, 5, 1]", "[7, 4, 3, 1]"])
eq([k.label for k in r.levels[2]], ["[4, 6]", "[5, 1]", "[7, 4]", "[3, 1]"])
eq([k.value for k in r.levels[2]], [[4, 6], [1, 5], [4, 7], [1, 3]])
shows(r)
ok(A == [4, 6, 5, 1, 7, 4, 3, 1], "Eingabe nicht verändert")
for liste in ([5], [2, 1], [3, 1, 2], list(range(10, 0, -1)), [1] * 7, [-3, 5, 0, 5, -1]):
    rr = recursion_tree("mergeSort", liste)
    eq(rr.value, sorted(liste), liste)
    eq(rr.calls, 2 * len(liste) - 1, "Binärbaum mit n Blättern hat 2n − 1 Knoten")
    eq(rr.height, (len(liste) - 1).bit_length() + 1 if len(liste) > 1 else 1)
eq(recursion_tree("mergeSort", [3, 1, 2]).levels[1][0].label, "[3]")        # len(A)//2: links 1 Element

for n, aufrufe in ((1, 1), (2, 3), (3, 7), (4, 15), (5, 31)):
    rr = recursion_tree("hanoi", n)
    eq((rr.calls, rr.value), (aufrufe, aufrufe))
    eq(rr.height, n)
eq(recursion_tree("hanoi", 2).nodes[0].label, "hanoi(2, B→C)")
for alias in ("sum", "Summe", "factorial", "Fakultät", "merge_sort", "mergesort", "Turm von Hanoi", "tuermevonhanoi"):
    ok(recursion_tree(alias, 3 if "ergesort" not in alias and "merge" not in alias.lower() else [2, 1]) is not None, alias)
raises(ValueError, recursion_tree, "quicksort", 3, text="Unbekannter Algorithmus")
raises(ValueError, recursion_tree, "summe", 0, text="zwischen 1 und")
raises(ValueError, recursion_tree, "summe", 13)
raises(ValueError, recursion_tree, "summe", 2.5, text="ganze Zahl")
raises(ValueError, recursion_tree, "summe", True)
raises(ValueError, recursion_tree, "hanoi", 8)
raises(ValueError, recursion_tree, "mergeSort", [])
raises(ValueError, recursion_tree, "mergeSort", 5, text="nichtleere Liste")
raises(ValueError, recursion_tree, "mergeSort", ["a", "b"], text="ganze Zahlen")
raises(ValueError, recursion_tree, "mergeSort", list(range(33)), text="höchstens")
ok(set(R.recursive_algorithms()) == {"summe", "fakultaet", "mergeSort", "hanoi"})
ok(all("def " in c for c in RECURSIVE_CODE.values()))
compile(RECURSIVE_CODE["summe"], "summe", "exec")
compile(RECURSIVE_CODE["fakultaet"], "fakultaet", "exec")
compile(RECURSIVE_CODE["mergeSort"].replace("return merge(L, R)", "return L + R"), "mergeSort", "exec")

# Brücke zu Woche 5: der Schrittzähler zählt die rekursive Summe linear; solve_recurrence löst die Beziehung dazu
steps = {n: dk.count_steps(RECURSIVE_CODE["summe"], n=n, func="summe").total for n in range(1, 9)}
eq(steps, {n: 4 * n - 2 for n in range(1, 9)}, "T(n) = T(n−1) + 4, T(1) = 2")
eq(solve_recurrence([1], 4, {1: 2}).solution, 4 * sp.Symbol("n", integer=True) - 2)

# ==========================================================================
# 2. Türme von Hanoi (Aufgabe 3)
# ==========================================================================
def spiel(zuege, n, start, ziel, hilf):
    """Unabhängiges Nachspielen der Regeln."""
    st = {start: list(range(n, 0, -1)), ziel: [], hilf: []}
    for d, von, nach in zuege:
        assert st[von] and st[von][-1] == d, "nicht die oberste Scheibe"
        assert not st[nach] or st[nach][-1] > d, "gross auf klein"
        st[nach].append(st[von].pop())
    return st[ziel] == list(range(n, 0, -1)) and not st[start] and not st[hilf]


eq(hanoi(2).moves, [(1, "B", "A"), (2, "B", "C"), (1, "A", "C")])
eq(hanoi(3).count, 7)
eq(hanoi(1).moves, [(1, "B", "C")])
for n in range(1, 11):
    h = hanoi(n)
    eq(h.count, 2 ** n - 1, n)
    ok(spiel(h.moves, n, "B", "C", "A"), n)
    eq(sum(1 for z in h.moves if z[0] == n), 1, "die grösste Scheibe zieht genau einmal")
    eq(sum(1 for z in h.moves if z[0] == 1), 2 ** (n - 1), "die kleinste zieht jeden zweiten Zug")
eq(sorted(set(z[0] for z in hanoi(4).moves)), [1, 2, 3, 4])
ok(spiel(hanoi(5, "L", "R", "M").moves, 5, "L", "R", "M"))
eq(hanoi(12).count, 4095)
shows(hanoi(3))
shows(hanoi(8))                                   # lange Zugtabelle wird gekürzt
raises(ValueError, hanoi, 0)
raises(ValueError, hanoi, 13)
raises(ValueError, hanoi, 3, "A", "A", "B", text="verschiedene")
raises(ValueError, hanoi, "3")

# ==========================================================================
# 3. Master-Satz (Aufgaben 5 bis 7)
# ==========================================================================
def T(a, b, k, n, c=1):
    return c if n == 1 else a * T(a, b, k, n // b, c) + c * n ** k


FAELLE = [   # (a, b, k, Fall, Θ)   Serie 6 Aufgabe 5 und 7
    (2, 2, 0, 1, "Θ(n)"), (2, 2, 1, 2, "Θ(n log n)"), (2, 2, 2, 3, "Θ(n²)"),                 # 5: TA, TB, TC
    (4, 3, 2, 3, "Θ(n²)"), (2, 3, 4, 3, "Θ(n⁴)"), (16, 4, 2, 2, "Θ(n² log n)"),             # 7a, 7b, 7c
    (4, 2, 0, 1, "Θ(n²)"), (4, 4, 1, 2, "Θ(n log n)"),                                       # 7d, 7e
    (1, 2, 0, 2, "Θ(log n)"), (8, 2, 3, 2, "Θ(n³ log n)"), (3, 2, 1, 1, "Θ(n^(log_2 3))"),
]
for a, b, k, fall, theta in FAELLE:
    m = master_theorem(a, b, k)
    eq((m.case, m.theta), (fall, theta), (a, b, k))
    eq(m.case, 1 if b ** k < a else 2 if b ** k == a else 3)
    for n, v in m.values.items():
        eq(int(v), T(a, b, k, n), (a, b, k, n))
    shows(m)
ok(master_theorem(2, 2, 1).log_factor and not master_theorem(2, 2, 2).log_factor)
eq(master_theorem(4, 2, 0).exponent, 2)
eq(master_theorem(2, 2, 1).exponent, 1)
# Θ-Probe: T(2^j)/g(2^j) beschränkt zwischen 1 und 2 für T(n) = 2T(n/2) + nᵏ
for k, g in ((0, lambda j: 2 ** j), (1, lambda j: 2 ** j * j), (2, lambda j: 4 ** j)):
    ok(all(1 <= T(2, 2, k, 2 ** j) / g(j) <= 2 for j in range(1, 20)), k)
# mergeSort Aufgabe 6: T(2^j) = 2^j·(j + 1); genaue Rekursion mit ⌈n/2⌉ + ⌊n/2⌋ liegt zwischen n·log₂ n und 2·n·log₂ n
mm = master_theorem(2, 2, 1)
eq({n: int(v) for n, v in mm.values.items()}, {2 ** j: 2 ** j * (j + 1) for j in range(5)})


def t_merge(n):
    return 1 if n == 1 else t_merge((n + 1) // 2) + t_merge(n // 2) + n


ok(all(n * math.log2(n) <= t_merge(n) <= 2 * n * math.log2(n) for n in range(2, 600)))
# c ≠ 1: Rekursion T(1) = c
m3 = master_theorem(2, 2, 1, c=3)
eq({n: int(v) for n, v in m3.values.items()}, {n: T(2, 2, 1, n, 3) for n in m3.values})
eq(master_theorem(2, 2, 1, c=Fraction(1, 2)).theta, "Θ(n log n)")
eq(len(master_theorem(2, 2, 1, levels=10).values), 11)
eq(master_theorem(2, 2, 0).levels.rows[2], ["2", "4", "n/2^2 = 4", "1", "4"])        # Baum für n = 16: Ebene 2 hat 4 Knoten der Grösse 4
eq(master_theorem(4, 2, 0).levels.rows[3][4], "64")                                  # 4³ Knoten · 1
raises(ValueError, master_theorem, 0, 2, 1)
raises(ValueError, master_theorem, 2, 1, 1)
raises(ValueError, master_theorem, 2, 2, -1)
raises(ValueError, master_theorem, 2.5, 2, 1, text="ganze Zahl")
raises(ValueError, master_theorem, 2, 2, 1, c=0, text="positiv")
raises(ValueError, master_theorem, 2, 2, 1, c="abc")
raises(ValueError, master_theorem, 2, 2, 1, levels=0)
raises(ValueError, master_theorem, 10 ** 7, 2, 1)
raises(ValueError, master_theorem, True, 2, 1)

# ==========================================================================
# 4. Lineare Rekursionsbeziehungen (Aufgaben 4 und 8)
# ==========================================================================
def iteriere(cs, f, init, n):
    seq = {i: Fraction(v) for i, v in init.items()}
    for m in range(max(seq) + 1, n + 1):
        seq[m] = sum(Fraction(c) * seq[m - i - 1] for i, c in enumerate(cs)) + f(m)
    return seq[n]


n_ = sp.Symbol("n", integer=True)
F8 = [  # (c, F, init, Formel von Hand, f(n))
    ([5], 0, {0: 4}, 4 * 5 ** n_, lambda m: 0),
    ([6, -9], 0, {0: 1, 1: 6}, (1 + n_) * 3 ** n_, lambda m: 0),
    ([1, 2], 0, {0: 2, 1: 7}, 3 * 2 ** n_ - (-1) ** n_, lambda m: 0),
    ([3], "2**n", {1: 3}, sp.Rational(7, 3) * 3 ** n_ - 2 ** (n_ + 1), lambda m: 2 ** m),
    ([5, -6], "7*n", {0: 1, 1: 15}, -33 * 2 ** n_ + sp.Rational(87, 4) * 3 ** n_ + sp.Rational(7, 2) * n_ + sp.Rational(49, 4), lambda m: 7 * m),
]
for cs, Fn, init, hand, f in F8:
    r = solve_recurrence(cs, Fn, init)
    eq(sp.simplify(r.solution - hand), 0, (cs, Fn))
    for m in range(min(init), 15):
        eq(Fraction(str(r.at(m))), iteriere(cs, f, init, m), (cs, Fn, m))
    shows(r)
    ok(len(r.steps) >= 5 and r.steps[-1].title == "Probe")
# einzelne Zwischenergebnisse des Rechenwegs
r = solve_recurrence([6, -9], 0, {0: 1, 1: 6})
eq(r.roots, {3: 2})                                          # doppelte Wurzel
eq(r.constants, {"a": 1, "b": 1})
r = solve_recurrence([1, 2], 0, {0: 2, 1: 7})
eq(r.roots, {2: 1, -1: 1})
eq(r.constants, {"a": -1, "b": 3})        # a·(−1)ⁿ + b·2ⁿ (Wurzeln aufsteigend)
eq(solve_recurrence([5, -6], "7*n", {0: 1, 1: 15}).particular, sp.Rational(7, 2) * n_ + sp.Rational(49, 4))
eq(solve_recurrence([3], "2**n", {1: 3}).particular, -2 * 2 ** n_)
# Resonanz: F(n) hat dieselbe Basis wie eine Wurzel -> Ansatz mit n multipliziert
r = solve_recurrence([2], "2**n", {0: 1})
eq(sp.simplify(r.solution - (1 + n_) * 2 ** n_), 0)         # T(n) = 2T(n−1) + 2ⁿ, T(0) = 1  ->  (n + 1)·2ⁿ
for m in range(10):
    eq(r.at(m), (m + 1) * 2 ** m)
# Aufgabe 4: T(n) = T(n−1) + c ist linear; Hanoi T(n) = 2T(n−1) + 1 hat die Lösung 2ⁿ − 1; Fibonacci
eq(solve_recurrence([1], 1, {1: 1}).solution, n_)
eq(sp.simplify(solve_recurrence([2], 1, {1: 1}).solution - (2 ** n_ - 1)), 0)
fib = solve_recurrence([1, 1], 0, {0: 0, 1: 1})
eq([int(fib.at(m)) for m in range(12)], [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89])
eq(len(fib.roots), 2)
# ohne Anfangswerte bleiben die Konstanten stehen
g = solve_recurrence([3], "2**n")
ok(g.constants == {} and sp.Symbol("a") in g.solution.free_symbols)
# Aufgabe 8 mit n bei F, Polynom Grad 2, und Resonanz mit n^2-Ansatz
r = solve_recurrence([2, -1], "n", {0: 0, 1: 1})              # doppelte Wurzel 1 und F = n  ->  Ansatz n²·(p0 + p1 n)
for m in range(12):
    eq(Fraction(str(r.at(m))), iteriere([2, -1], lambda x: x, {0: 0, 1: 1}, m))
r = solve_recurrence([1], "n**2 + 1", {0: 0})
for m in range(10):
    eq(r.at(m), sum(k * k + 1 for k in range(1, m + 1)))
r = solve_recurrence([Fraction(1, 2)], 3, {0: 1})
for m in range(8):
    eq(Fraction(str(r.at(m))), iteriere([Fraction(1, 2)], lambda x: 3, {0: 1}, m))
# Fehler
raises(ValueError, solve_recurrence, [], 0, {0: 1}, text="Liste")
raises(ValueError, solve_recurrence, [1, 2, 3], 0, {0: 1, 1: 1, 2: 1}, text="Grad 1 und 2")
raises(ValueError, solve_recurrence, [1, 0], 0, {0: 1, 1: 1}, text="≠ 0")
raises(ValueError, solve_recurrence, "5", 0, {0: 1})
raises(ValueError, solve_recurrence, [5], 0, {0: 1, 1: 2}, text="Anfangswert")
raises(ValueError, solve_recurrence, [1, 1], 0, {0: 1}, text="Anfangswert")
raises(ValueError, solve_recurrence, [1, 1], 0, {0: 1, 2: 3}, text="aufeinanderfolgen")
raises(ValueError, solve_recurrence, [5], "evil(n)", {0: 1})
raises(ValueError, solve_recurrence, [5], "n*log(n)", {0: 1}, text="unterstützt")
raises(ValueError, solve_recurrence, [5], "x + n", {0: 1}, text="Unbekannter Name")
raises(ValueError, solve_recurrence, [5], "n**n", {0: 1})
raises(NotImplementedError, solve_recurrence, [0, -1], 0, {0: 1, 1: 1}, text="Komplex")   # r² = −1
raises(ValueError, solve_recurrence, ["x"], 0, {0: 1})

# ==========================================================================
# 5. Zusammenspiel mit dem Paket
# ==========================================================================
ok(dk.WEEKS[6][1] is R and dk.rekursion is R)
for name in R.__all__:
    ok(hasattr(dk, name), name)
ok("SW6" in dk.info() and "master_theorem" in dk.info() and "plot_recursion_tree" in dk.info())
ok(dk.master_theorem(2, 2, 1).theta == "Θ(n log n)" and dk.hanoi(3).count == 7)
fig = recursion_tree("summe", 5).figure()
ok(fig.axes and hanoi(2).final["C"] == [2, 1])

print(f"tests_rekursion.py: alle grün ({N[0]} Prüfungen)")
