#!/usr/bin/env python3
"""
Tests für dmatkit.grafik.plots — laufen ohne Bildschirm (Agg-Backend).
Start: python3 tests/tests_plots.py
Einige Bilder landen in tests/shots/ (gitignored).
"""
import os
import sys
import math
import time

os.environ["MPLBACKEND"] = "Agg"
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', '..'))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

from dmatkit.grafik import plots as P  # noqa: E402
from dmatkit.grafik.plots import (has_plotly, plot_bits, plot_connectives,  # noqa: E402
                                  plot_entailment, plot_growth, plot_halving,
                                  plot_quantifier, plot_resolution, plot_steps,
                                  plot_syntax_tree, plot_truth_table, plotly_growth)
from dmatkit.rechnen.core import big_and, parse  # noqa: E402

SHOTS = os.path.join(_here, "shots")
os.makedirs(SHOTS, exist_ok=True)
CHECKS = 0


def check(cond, msg="Bedingung verletzt"):
    global CHECKS
    CHECKS += 1
    assert cond, msg


def raises(exc, fn, *a, **k):
    global CHECKS
    CHECKS += 1
    try:
        fn(*a, **k)
    except exc:
        return True
    return False


def _fig(fig, shot=None):
    check(isinstance(fig, Figure) and fig.axes, "keine Figure mit Axes")
    fig.canvas.draw()                      # wirklich rendern, nicht nur bauen
    if shot:
        fig.savefig(os.path.join(SHOTS, shot + ".png"), dpi=110)
    return fig


class Node:
    """Duck-Typing-Knoten wie rechnen/aussagenlogik.Node."""
    def __init__(self, label, *children):
        self.label, self.children = label, tuple(children)


def full_tree(depth):
    if depth == 0:
        return Node("x")
    return Node("∧", full_tree(depth - 1), full_tree(depth - 1))


GRAPH = [
    {"id": 1, "clause": "{¬A, B}", "parents": (), "origin": "Wissensbasis"},
    {"id": 2, "clause": "{A}", "parents": (), "origin": "Wissensbasis"},
    {"id": 3, "clause": "{¬B}", "parents": (), "origin": "Negierte Anfrage"},
    {"id": 4, "clause": "{B}", "parents": (1, 2), "origin": "Resolvent"},
    {"id": 5, "clause": "□", "parents": (4, 3), "origin": "Resolvent"},
]


def test_syntax_tree():
    _fig(plot_syntax_tree("(A and B) -> not C"), "tree_small")
    _fig(plot_syntax_tree(parse("A")))                              # einzelner Knoten
    _fig(plot_syntax_tree("true"))
    _fig(plot_syntax_tree(Node("Wurzel", Node("a"), Node("b", Node("c")))))
    _fig(plot_syntax_tree("A", dark=True))
    f = big_and([f"(P{i} or not Q{i}) -> R{i}" for i in range(6)])
    fig = _fig(plot_syntax_tree(f, title="Gross"), "tree_big")
    _fig(plot_syntax_tree(full_tree(5)), "tree_full5")             # 63 Knoten
    _fig(plot_syntax_tree(Node("Eine sehr lange Beschriftung mit vielen Wörtern darin",
                               Node("kurz"), Node("noch_ein_langer_Name_ohne_Leerzeichen"))), "tree_long")
    _fig(plot_syntax_tree(Node("a", Node("a", Node("a"), Node("a")), Node("a", Node("a"), Node("a")))))
    _fig(plot_syntax_tree("A", figsize=(3, 2)))
    _fig(plot_syntax_tree(Node("100$ or $B", Node("$A$"))))                           # Dollar darf keinen Formelsatz auslösen
    # Struktur: Knoten stehen nicht übereinander (Layout-Garantie)
    from dmatkit.grafik.plots import _build_tree, _layout_tree, _TNode
    root = _build_tree(full_tree(5))
    stack, allnodes = [root], []
    while stack:
        n = stack.pop(); allnodes.append(n); stack.extend(n.kids)
    for n in allnodes:
        n.w = 0.5
    nodes = _layout_tree(root, 0.2)
    bylevel = {}
    for n in nodes:
        bylevel.setdefault(n.depth, []).append(n.x)
    for xs in bylevel.values():
        xs.sort()
        check(all(b - a >= 0.5 + 0.2 - 1e-9 for a, b in zip(xs, xs[1:])), "Knoten überlappen")
    check(len(nodes) == 63)
    # Fehler
    check(raises(ValueError, plot_syntax_tree, ""))
    check(raises(ValueError, plot_syntax_tree, "A and"))
    check(raises(ValueError, plot_syntax_tree, None))
    check(raises(ValueError, plot_syntax_tree, 42))
    check(raises(ValueError, plot_syntax_tree, Node("x", 5)))
    check(raises(ValueError, plot_syntax_tree, full_tree(9)))       # > 300 Knoten
    deep = Node("x")
    for _ in range(400):
        deep = Node("¬", deep)
    check(raises(ValueError, plot_syntax_tree, deep))
    check(raises(ValueError, plot_syntax_tree, "A", figsize=(0, 1)))


def test_truth_table():
    _fig(plot_truth_table("A xor (B or C)"), "tt_xor")
    _fig(plot_truth_table("A -> B", names=["B", "A"]))
    _fig(plot_truth_table("A or not A"))                            # Tautologie
    _fig(plot_truth_table("A and not A"))                           # Widerspruch
    _fig(plot_truth_table("A"))
    _fig(plot_truth_table(True))                                    # keine Variablen
    _fig(plot_truth_table("A", names="A B"))
    _fig(plot_truth_table("(A -> B) and (B -> C) -> (A -> C)", dark=True), "tt_dark")
    _fig(plot_truth_table(" and ".join(f"V{i}" for i in range(6))))   # 64 Zeilen
    check(raises(ValueError, plot_truth_table, ""))
    check(raises(ValueError, plot_truth_table, "A and B", names=["A"]))
    check(raises(ValueError, plot_truth_table, "A", names=["A", "A"]))
    check(raises(ValueError, plot_truth_table, " and ".join(f"V{i}" for i in range(7))))
    check(raises(ValueError, plot_truth_table, None))


def test_connectives():
    _fig(plot_connectives(), "connectives")
    _fig(plot_connectives(dark=True, title="Verknüpfungen"))
    _fig(plot_connectives(figsize=(8, 3)))


def test_entailment():
    _fig(plot_entailment("A and (A -> B)", "B"), "ent_ok")
    _fig(plot_entailment(["A or B", "not A"], "A"), "ent_bad")
    _fig(plot_entailment("A and not A", "B"))                       # WB unerfüllbar
    _fig(plot_entailment("A", "A or B", dark=True))
    check(raises(ValueError, plot_entailment, [], "A"))
    check(raises(ValueError, plot_entailment, "A", ""))
    check(raises(ValueError, plot_entailment, None, "A"))


def test_resolution():
    _fig(plot_resolution(GRAPH), "res_small")
    _fig(plot_resolution(GRAPH, dark=True))
    _fig(plot_resolution(GRAPH[:3]))                                # keine Herleitung
    big = [{"id": i, "clause": "{" + ", ".join(f"{'¬' if (i + j) % 2 else ''}P{j}" for j in range(i % 4 + 1)) + "}",
            "parents": (), "origin": "Wissensbasis"} for i in range(1, 10)]
    big.append({"id": 10, "clause": "{¬P0}", "parents": (), "origin": "Negierte Anfrage"})
    nid = 11
    prev = [1, 2]
    for k in range(6):
        big.append({"id": nid, "clause": f"{{R{k}, ¬S{k}, T{k}}}", "parents": tuple(prev[:2]),
                    "origin": "Resolvent"})
        prev = [nid, (k % 9) + 3]
        nid += 1
    big.append({"id": nid, "clause": "□", "parents": (nid - 1, 10), "origin": "Resolvent"})
    _fig(plot_resolution(big), "res_big")
    check(raises(ValueError, plot_resolution, []))
    check(raises(ValueError, plot_resolution, "x"))
    check(raises(ValueError, plot_resolution, [{"id": 1}]))
    check(raises(ValueError, plot_resolution, [dict(GRAPH[0]), dict(GRAPH[0])]))      # doppelte id
    check(raises(ValueError, plot_resolution, [{"id": 1, "clause": "{A}", "parents": (9,), "origin": "Resolvent"}]))
    cyc = [{"id": 1, "clause": "{A}", "parents": (2,), "origin": "Resolvent"},
           {"id": 2, "clause": "{B}", "parents": (1,), "origin": "Resolvent"}]
    check(raises(ValueError, plot_resolution, cyc))


def test_bits():
    _fig(plot_bits(181), "bits_181")
    _fig(plot_bits(-37, width=8, signed=True), "bits_neg")
    _fig(plot_bits(0))
    _fig(plot_bits(255))
    _fig(plot_bits(-128, signed=True))
    _fig(plot_bits(127, signed=True, dark=True))
    _fig(plot_bits("10110101"))
    _fig(plot_bits("0b1111_1111", signed=True))
    _fig(plot_bits(2 ** 32 - 1, width=32), "bits_32")
    _fig(plot_bits(5, width=64))
    check(raises(ValueError, plot_bits, 256))
    check(raises(ValueError, plot_bits, -1))
    check(raises(ValueError, plot_bits, 128, 8, True))
    check(raises(ValueError, plot_bits, -129, 8, True))
    check(raises(ValueError, plot_bits, 1.5))
    check(raises(ValueError, plot_bits, True))
    check(raises(ValueError, plot_bits, "102"))
    check(raises(ValueError, plot_bits, "111111111"))
    check(raises(ValueError, plot_bits, 1, width=0))
    check(raises(ValueError, plot_bits, 1, width=65))
    check(raises(ValueError, plot_bits, ""))
    from dmatkit.grafik.plots import _bit_pattern
    check(_bit_pattern(-37, 8, True) == ([1, 1, 0, 1, 1, 0, 1, 1], -37))
    check(_bit_pattern("11111111", 8, True)[1] == -1)
    check(_bit_pattern("11111111", 8, False)[1] == 255)
    for v in range(-128, 128):
        check(_bit_pattern(v, 8, True)[1] == v)


def test_quantifier():
    _fig(plot_quantifier([1, 2, 3, 4], [False, True, True, False]), "q_1d")
    _fig(plot_quantifier(list(range(1, 6)), [True] * 5))
    _fig(plot_quantifier(list(range(1, 6)), [False] * 5))
    _fig(plot_quantifier(["a", "b"], [1, 0], name="Q", variables=("w",), dark=True))
    _fig(plot_quantifier(list(range(60)), [i % 3 == 0 for i in range(60)]), "q_60")
    _fig(plot_quantifier([(1, 2, 3), (4, 5, 6)], [True, False]))
    U = [(x, y) for x in range(1, 5) for y in range(1, 5)]
    _fig(plot_quantifier(U, [x < y for x, y in U], variables=("x", "y")), "q_2d")
    _fig(plot_quantifier(U[:-3], [x + y == 5 for x, y in U[:-3]]))     # unvollständiges Raster
    _fig(plot_quantifier([("a", "b")], [True]))
    check(raises(ValueError, plot_quantifier, [], []))
    check(raises(ValueError, plot_quantifier, [1, 2], [True]))
    check(raises(ValueError, plot_quantifier, [1, 1], [True, False]))
    check(raises(ValueError, plot_quantifier, [1, 2], [True, "ja"]))
    check(raises(ValueError, plot_quantifier, 5, [True]))
    check(raises(ValueError, plot_quantifier, list(range(300)), [True] * 300))


def test_growth():
    fig = _fig(plot_growth({"n": "n", "n log n": "n*log(n,2)", "n²": "n**2", "2ⁿ": "2**n"}, n_max=20),
               "growth_lin")
    _fig(plot_growth({"n": "n", "n log n": "n*log(n,2)", "n²": "n**2", "2ⁿ": "2**n",
                      "n!": "factorial(n)"}, n_max=30, log=True), "growth_log")
    _fig(plot_growth({"n²": "n**2", "n log n": "n*log(n,2)", "n": "n", "2ⁿ": "2^n"}, n_max=100,
                     log=True, mark_limit=6e10), "growth_limit")
    _fig(plot_growth({"n²": "n**2", "n": "n"}, n_max=1000, mark_limit=10 ** 6))
    _fig(plot_growth(["n", "n^2", "sqrt(n)", "ln(n)", "exp(n/5)", "floor(n/2)", "ceiling(n/3)",
                      "n^3 + 2n"], n_max=40, dark=True))
    _fig(plot_growth("n"))
    _fig(plot_growth({"c": "5"}))
    _fig(plot_growth({"f": lambda n: n * n, "g": lambda n: 2 ** n}, n_max=50, log=True, mark_limit=1e6))
    _fig(plot_growth({"x": "factorial(n)"}, n_max=500, log=True))             # weit über float
    _fig(plot_growth({"x": "2**n", "y": "n**n"}, n_max=2000, log=True))
    _fig(plot_growth({"n": "n"}, n_max=10 ** 8, log=True))
    _fig(plot_growth({"x": "log(n-5)"}, n_max=20))                           # Teile undefiniert
    _fig(plot_growth({"n": "n"}, n_max=2))
    check(raises(ValueError, plot_growth, {"n": "n"}, 1))
    # Sicherheit: nichts wird ausgeführt
    evil = ["__import__('os').system('echo hacked')", "n.__class__", "open('x')", "x + 1",
            "os", "lambda: 1", "exec('1')", "n; 1", "foo(n)", "[n]", "'a'", "sin(n)",
            "n" * 300, "9" * 20, "", "   ", "n +", "n ** ", "()", "n$", "a b", "import os"]
    for e in evil:
        check(raises(ValueError, plot_growth, {"f": e}), f"nicht abgelehnt: {e!r}")
    check(raises(ValueError, plot_growth, {}))
    check(raises(ValueError, plot_growth, []))
    check(raises(ValueError, plot_growth, None))
    check(raises(ValueError, plot_growth, {"n": "n"}, 5.5))
    check(raises(ValueError, plot_growth, {"n": "n"}, True))
    check(raises(ValueError, plot_growth, {"n": "n"}, 10, False, -3))
    check(raises(ValueError, plot_growth, {"n": "n"}, 10, False, 0))
    check(raises(ValueError, plot_growth, {1: "n"}))
    check(raises(ValueError, plot_growth, {f"f{i}": "n" for i in range(20)}))
    check(raises(ValueError, plot_growth, {"f": lambda n: n}, 10 ** 7))
    # Riesen-Potenz darf nicht hängen
    t = time.time()
    _fig(plot_growth({"x": "9^9^9", "y": "2^(2^n)"}, n_max=30, log=True))
    check(time.time() - t < 8, "zu langsam")
    # Schnittpunkte
    from dmatkit.grafik.plots import _Fn, _max_n
    check(_max_n(_Fn("n**2"), 6e10)[0] == 244948)
    check(_max_n(_Fn("n"), 100)[0] == 100)
    check(_max_n(_Fn("2**n"), 1000)[0] == 9)
    check(_max_n(_Fn("n+10"), 5)[0] == 0)
    check(_max_n(_Fn("n*log(n,2)"), 1000)[0] == 140)
    check(_Fn("n*log(n,2)")(8)[0] == 24.0)
    check(abs(_Fn("factorial(n)")(10)[0] - 3628800) < 1e-6)
    check(_Fn("floor(n/2)")(7)[0] == 3.0)
    check(_Fn("ceiling(n/2)")(7)[0] == 4.0)
    check(_Fn("2n^2 + 3n")(4)[0] == 44.0)
    check(abs(_Fn("n log(n)")(10)[1] - math.log10(10 * math.log(10))) < 1e-9)
    check(abs(_Fn("sqrt(n)")(16)[0] - 4) < 1e-12)
    check(abs(_Fn("exp(n)")(1)[0] - math.e) < 1e-12)
    check(_Fn("1/2")(5)[0] == 0.5)
    check(_Fn(lambda n: n ** 2)(12)[0] == 144.0)
    check(math.isnan(_Fn("log(n)")(0)[1]))




def test_steps():
    _fig(plot_steps([1, 2, 4, 8, 16], [3, 7, 15, 31, 63], fit="4*n - 1"), "steps_fit")
    _fig(plot_steps([10, 20, 30], [100, 400, 900], fit="n**2", dark=True))
    _fig(plot_steps([1, 2, 3], [5, 5, 5]))
    _fig(plot_steps([5], [9], fit="n"))                              # ein Punkt
    _fig(plot_steps([1, 10, 100, 1000], [1, 100, 10000, 10 ** 6], log=True, fit=lambda n: n * n))
    _fig(plot_steps([0.5, 1.5], [1, 2]))
    check(raises(ValueError, plot_steps, [], []))
    check(raises(ValueError, plot_steps, [1, 2], [1]))
    check(raises(ValueError, plot_steps, [1, "a"], [1, 2]))
    check(raises(ValueError, plot_steps, [1, float("nan")], [1, 2]))
    check(raises(ValueError, plot_steps, 5, 6))
    check(raises(ValueError, plot_steps, [1, 2], [1, 2], fit="__import__('os')"))
    check(raises(ValueError, plot_steps, [1, 2], [1, 2], fit="m*n"))


def test_halving():
    _fig(plot_halving(100), "halving_100")
    _fig(plot_halving(1))
    _fig(plot_halving(2))
    _fig(plot_halving(1023, dark=True))
    _fig(plot_halving(10 ** 6))
    _fig(plot_halving(2 ** 64 - 1))
    check(raises(ValueError, plot_halving, 0))
    check(raises(ValueError, plot_halving, -5))
    check(raises(ValueError, plot_halving, 2.5))
    check(raises(ValueError, plot_halving, "8"))
    check(raises(ValueError, plot_halving, 2 ** 64))
    check(raises(ValueError, plot_halving, True))


def test_plotly():
    if has_plotly():
        import plotly.graph_objects as go
        fig = plotly_growth({"n²": "n**2", "2ⁿ": "2**n"}, n_max=30, log=True, mark_limit=1e6)
        check(isinstance(fig, go.Figure) and len(fig.data) >= 2)
        check(isinstance(plotly_growth({"n": "n"}), go.Figure))
        check(isinstance(plotly_growth({"n": "n"}, 10, dark=True), go.Figure))
        check(raises(ValueError, plotly_growth, {"x": "evil(n)"}))
    else:
        check(raises(RuntimeError, plotly_growth, {"n": "n"}))


def test_misc():
    import matplotlib.pyplot as plt
    plt.close("all")
    plot_syntax_tree("A and B")
    plot_growth({"n": "n"})
    plot_halving(10)
    check(plt.get_fignums() == [], "pyplot-Figuren übrig (doppelte Anzeige im Notebook)")
    check(isinstance(has_plotly(), bool))
    check(all(hasattr(P, n) for n in P.__all__))
    # Rückgabe ist backendunabhängig speicherbar
    import io
    buf = io.BytesIO()
    plot_connectives().savefig(buf, format="png")
    check(len(buf.getvalue()) > 1000)
    buf = io.BytesIO()
    plot_truth_table("A and B").savefig(buf, format="svg")
    check(b"<svg" in buf.getvalue())


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                import traceback
                fails += 1
                print("FAIL", name, repr(e))
                traceback.print_exc()
    print(f"alle Tests bestanden ({CHECKS} Prüfungen)" if not fails else f"{fails} Test(s) fehlgeschlagen")
    sys.exit(1 if fails else 0)
