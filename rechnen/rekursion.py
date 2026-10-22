#!/usr/bin/env python3
"""
Rekursive Algorithmen (Semesterwoche 6, Serie 6).

Was hier steht (alles exakt gerechnet, nie ``eval``/``exec`` auf Eingaben):

    recursion_tree(alg, arg)            Rekursionsbaum von summe, fakultaet, mergeSort oder Hanoi (Aufgaben 1 bis 3)
    hanoi(n)                            Türme von Hanoi: die Züge, geprüft gegen die Spielregeln (Aufgabe 3)
    master_theorem(a, b, k, c)          T(n) = a·T(n/b) + c·nᵏ  ->  Θ(...) mit Rekursionsbaum (Aufgaben 5 bis 7)
    solve_recurrence(coeffs, F, init)   lineare Rekursionsbeziehung mit konstanten Koeffizienten lösen
                                        (Satz 1 und Satz 2, Aufgaben 4 und 8)
    RECURSIVE_CODE                      der Pseudocode der Algorithmen als Text

Zusammenhang mit Woche 5: Dort liessen sich die Schritte durch Zählen und Summieren bestimmen
(Schleifen). Bei Rekursion hängt T(n) von T(kleinere Eingabe) ab. Man stellt darum eine
Rekursionsbeziehung für die Schrittzahl auf und löst sie, entweder über den Rekursionsbaum
(divide-and-conquer, ``master_theorem``) oder über die charakteristische Gleichung
(``solve_recurrence``). Das Ergebnis wird dann wie in SW5 als Θ-Klasse angegeben.

Hinweise zu den Unterlagen (im Abschlussbericht vermerkt):
  * Die Rekursionsbeziehungen zu Aufgabe 4 (Summe, Fakultät) stehen in der PDF-Serie nicht im Text
    (nur die Leerseite zum Zeichnen). Hier: T(n) = T(n−1) + c, also Θ(n). TODO: mit der Vorlesung abgleichen.
  * ``merge`` braucht in der Serie keine ausgeschriebene Schrittzahl; für Aufgabe 6 gilt die übliche
    Annahme: ``merge`` einer Liste mit n Elementen kostet c·n Schritte, also T(n) = 2·T(n/2) + c·n.
"""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass, field
from fractions import Fraction

import sympy as sp

from .algorithmen import parse_term
from .render import Result, Step, Table

# ==========================================================================
# Grenzen
# ==========================================================================

MAX_TREE_NODES = 300           # so viele Knoten zeichnet plot_syntax_tree höchstens
MAX_LIST_LEN = 32              # mergeSort-Baum
MAX_HANOI_MOVES_N = 12         # hanoi(n): 2ⁿ − 1 Züge, 4095 bei n = 12
MAX_HANOI_TREE_N = 7
MAX_LINEAR_N = 12              # sum / fakultaet
MAX_LEVELS = 40                # master_theorem: Ebenen der Tabelle
MAX_PROBE = 14                 # solve_recurrence: Probe bis n0 + 14

__all__ = ["RecNode", "RecursionTreeResult", "recursion_tree", "HanoiResult", "hanoi",
           "MasterResult", "master_theorem", "RecurrenceResult", "solve_recurrence",
           "RECURSIVE_CODE", "recursive_algorithms"]

# ==========================================================================
# Pseudocode
# ==========================================================================

RECURSIVE_CODE = {
    "summe": textwrap.dedent("""\
        def summe(n):
            if n == 1:
                return 1
            else:
                return summe(n - 1) + n"""),
    "fakultaet": textwrap.dedent("""\
        def fakultaet(n):
            if n == 1:
                return 1
            else:
                return fakultaet(n - 1) * n"""),
    # genau der Text aus Serie 6 (merge ist dort nicht ausgeschrieben)
    "mergeSort": textwrap.dedent("""\
        def mergeSort(A):
            if len(A) == 1:
                return A
            L = mergeSort([A[i] for i in range(len(A)//2)])
            R = mergeSort([A[i] for i in range(len(A)//2, len(A))])
            return merge(L, R)"""),
    "hanoi": textwrap.dedent("""\
        def hanoi(n, start, ziel, hilf):
            if n == 1:
                lege Scheibe 1 von start nach ziel
            else:
                hanoi(n - 1, start, hilf, ziel)      # obere n-1 Scheiben zur Seite
                lege Scheibe n von start nach ziel   # die grösste Scheibe
                hanoi(n - 1, hilf, ziel, start)      # die n-1 Scheiben wieder darauf"""),
}


def recursive_algorithms() -> list[str]:
    """Die Namen der Algorithmen, für die ``recursion_tree`` einen Baum zeichnet."""
    return list(RECURSIVE_CODE)


# ==========================================================================
# Rekursionsbaum
# ==========================================================================

@dataclass
class RecNode:
    """Ein Aufruf im Rekursionsbaum. ``label`` ist der Aufruf, ``value`` das Ergebnis nach der Rückkehr.

    ``label`` und ``children`` genügen ``plot_syntax_tree`` — der Baum lässt sich direkt zeichnen.
    """
    label: str
    value: object = None
    children: list["RecNode"] = field(default_factory=list)
    depth: int = 0

    def walk(self):
        yield self
        for k in self.children:
            yield from k.walk()


def _alg_name(alg: str) -> str:
    key = str(alg).strip().lower().replace("_", "").replace("ä", "ae").replace(" ", "").replace("-", "")
    table = {"summe": "summe", "sum": "summe", "fakultaet": "fakultaet", "fakultat": "fakultaet",
             "factorial": "fakultaet", "fak": "fakultaet", "mergesort": "mergeSort", "merge": "mergeSort",
             "hanoi": "hanoi", "turmvonhanoi": "hanoi", "tuermevonhanoi": "hanoi"}
    if key not in table:
        raise ValueError(f"Unbekannter Algorithmus {alg!r}. Gemeint ist einer von: "
                         + ", ".join(RECURSIVE_CODE))
    return table[key]


def _int(v, name, lo, hi) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"{name} muss eine ganze Zahl sein, nicht {v!r}")
    if not lo <= v <= hi:
        raise ValueError(f"{name} muss zwischen {lo} und {hi} liegen")
    return v


def _linear_tree(name: str, n: int) -> RecNode:
    op = (lambda a, b: a + b) if name == "summe" else (lambda a, b: a * b)

    def build(m: int, depth: int) -> RecNode:
        node = RecNode(f"{name}({m})", depth=depth)
        if m == 1:
            node.value = 1
        else:
            kid = build(m - 1, depth + 1)
            node.children = [kid]
            node.value = op(kid.value, m)
        return node

    return build(n, 0)


def merge(left: list, right: list) -> list:
    """Das Verschmelzen zweier sortierter Listen (Reissverschluss)."""
    out, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
    return out + left[i:] + right[j:]


def _merge_tree(a: list) -> RecNode:
    def build(part: list, depth: int) -> RecNode:
        node = RecNode(str(part), depth=depth)
        if len(part) == 1:
            node.value = list(part)
        else:
            half = len(part) // 2                      # genau wie range(len(A)//2) in Serie 6
            node.children = [build(part[:half], depth + 1), build(part[half:], depth + 1)]
            node.value = merge(node.children[0].value, node.children[1].value)
        return node

    return build(list(a), 0)


def _hanoi_moves(n: int, start: str, ziel: str, hilf: str) -> list[tuple[int, str, str]]:
    if n == 1:
        return [(1, start, ziel)]
    return (_hanoi_moves(n - 1, start, hilf, ziel) + [(n, start, ziel)]
            + _hanoi_moves(n - 1, hilf, ziel, start))


def _hanoi_tree(n: int, start: str, ziel: str, hilf: str) -> RecNode:
    def build(m: int, s: str, z: str, h: str, depth: int) -> RecNode:
        node = RecNode(f"hanoi({m}, {s}→{z})", depth=depth)
        if m == 1:
            node.value = 1
        else:
            node.children = [build(m - 1, s, h, z, depth + 1), build(m - 1, h, z, s, depth + 1)]
            node.value = node.children[0].value + 1 + node.children[1].value
        return node

    return build(n, start, ziel, hilf, 0)


class RecursionTreeResult(Result):
    """
    ``root`` (``RecNode``), ``nodes`` (alle Knoten, Wurzel zuerst), ``calls`` (Anzahl Aufrufe),
    ``height`` (Anzahl Ebenen), ``levels`` (Knoten je Ebene), ``value`` (Rückgabewert der Wurzel),
    ``algorithm``, ``arg``.
    """

    def figure(self, *, values: bool = False, dark: bool = False, figsize=None, title=None):
        """Zeichnung des Baums; ``values=True`` schreibt die Rückgabewerte mit dazu (»Aufruf → Ergebnis«)."""
        from ..grafik.plots import plot_recursion_tree
        return plot_recursion_tree(self.root, values=values, dark=dark, figsize=figsize, title=title)


def _show(v) -> str:
    return str(v)


def recursion_tree(alg: str, arg) -> RecursionTreeResult:
    """
    Rekursionsbaum zu ``summe``, ``fakultaet``, ``mergeSort`` oder ``hanoi`` (Serie 6, Aufgaben 1 bis 3).

    ``arg`` ist n (summe, fakultaet, hanoi) oder die Liste für mergeSort.

    >>> recursion_tree("summe", 5).value
    15
    >>> recursion_tree("fakultaet", 5).calls
    5
    >>> recursion_tree("mergeSort", [4, 6, 5, 1, 7, 4, 3, 1]).value
    [1, 1, 3, 4, 4, 5, 6, 7]
    >>> recursion_tree("hanoi", 3).calls
    7
    """
    name = _alg_name(alg)
    res = RecursionTreeResult()
    res.algorithm, res.arg = name, arg
    if name in ("summe", "fakultaet"):
        n = _int(arg, "n", 1, MAX_LINEAR_N)
        root = _linear_tree(name, n)
        head = f"{name}({n})"
        rec = "T(n) = T(n−1) + c,  T(1) = c   (ein Aufruf mit kleinerer Eingabe, dazu eine Addition/Multiplikation)"
        theta = "Θ(n)"
        why = f"n Aufrufe übereinander, je mit konstantem Aufwand: {n}·c Schritte"
    elif name == "mergeSort":
        if not isinstance(arg, (list, tuple)) or not arg:
            raise ValueError("mergeSort braucht eine nichtleere Liste, z. B. [4, 6, 5, 1, 7, 4, 3, 1]")
        if len(arg) > MAX_LIST_LEN:
            raise ValueError(f"Die Liste darf höchstens {MAX_LIST_LEN} Elemente haben")
        if any(isinstance(x, bool) or not isinstance(x, int) for x in arg):
            raise ValueError("Die Liste darf nur ganze Zahlen enthalten")
        root = _merge_tree(list(arg))
        head = f"mergeSort({list(arg)})"
        rec = "T(n) = T(⌈n/2⌉) + T(⌊n/2⌋) + (Schritte für merge),  merge: c·n"
        theta = "Θ(n log n)"
        why = "⌈log₂ n⌉ + 1 Ebenen, auf jeder Ebene zusammen n Elemente zu verschmelzen"
    else:
        n = _int(arg, "n", 1, MAX_HANOI_TREE_N)
        root = _hanoi_tree(n, "B", "C", "A")
        head = f"hanoi({n}, B→C)"
        rec = "T(n) = 2·T(n−1) + 1,  T(1) = 1   (zwei Aufrufe mit n−1, dazu der eine Zug der grössten Scheibe)"
        theta = "Θ(2ⁿ)"
        why = f"{2 ** n - 1} Aufrufe, je ein Zug: T(n) = 2ⁿ − 1"
    res.root = root
    res.nodes = list(root.walk())
    res.calls = len(res.nodes)
    res.height = max(k.depth for k in res.nodes) + 1
    levels: list[list[RecNode]] = [[] for _ in range(res.height)]
    for k in res.nodes:
        levels[k.depth].append(k)
    res.levels = levels
    res.value = root.value
    if res.calls > MAX_TREE_NODES:
        raise ValueError(f"Der Baum hat {res.calls} Knoten, mehr als {MAX_TREE_NODES} lassen sich nicht zeichnen")
    res.title = f"Rekursionsbaum: {head}"
    res.summary = f"{res.calls} Aufrufe in {res.height} Ebenen. {rec}"
    res.steps.append(Step("Algorithmus", RECURSIVE_CODE[name], rule="Rekursion"))
    res.steps.append(Step(
        "Basisfall und Rekursionsschritt",
        {"summe": "n = 1 wird direkt berechnet; sonst wird summe(n−1) aufgerufen und n addiert.",
         "fakultaet": "n = 1 wird direkt berechnet; sonst wird fakultaet(n−1) aufgerufen und mit n multipliziert.",
         "mergeSort": "Hat die Liste die Länge 1, ist sie schon sortiert (Basisfall). Sonst wird sie halbiert, "
                      "beide Hälften werden sortiert (zwei Aufrufe) und mit merge verschmolzen.",
         "hanoi": "Eine Scheibe wird direkt gelegt. Sonst: n−1 Scheiben auf den Hilfsstab, die grösste "
                  "ans Ziel, dann die n−1 Scheiben vom Hilfsstab aufs Ziel."}[name],
        rule="Divide and Conquer" if name in ("mergeSort", "hanoi") else "Rekursion"))
    rows = [[str(d), str(len(lv)), "   ".join(k.label for k in lv)] for d, lv in enumerate(levels)]
    res.steps.append(Step("Rekursionsbaum", "Jeder Aufruf ist ein Knoten; die Kinder sind seine rekursiven Aufrufe. "
                          "Ebene 0 ist der erste Aufruf.",
                          table=Table(["Ebene", "Aufrufe", "Knoten"], rows), rule="Rekursionsbaum"))
    back = [[k.label, _show(k.value)] for k in reversed(res.nodes)]
    if len(back) > 24:
        back = back[:12] + [["…", "…"]] + back[-11:]
    res.steps.append(Step(
        "Rückgabewerte von unten nach oben",
        "Zuerst kehren die Blätter zurück, dann setzt jeder Elternknoten aus den Werten seiner Kinder zusammen.",
        table=Table(["Aufruf", "Rückgabe" if name != "hanoi" else "Züge"], back)))
    res.steps.append(Step("Rekursionsbeziehung für die Schrittzahl", rec, items=[("Komplexität", theta)],
                          note=why, rule="Rekursionsbeziehung"))
    if name == "summe":
        chk = _int(arg, "n", 1, MAX_LINEAR_N) * (arg + 1) // 2
        res.steps.append(Step("Probe", f"Gauss: n·(n+1)/2 = {chk}.", items=[("Summe", _show(res.value))]))
    elif name == "fakultaet":
        res.steps.append(Step("Probe", f"math.factorial({arg}) = {math.factorial(arg)}.", items=[("n!", _show(res.value))]))
    elif name == "mergeSort":
        res.steps.append(Step("Probe", f"sorted(A) = {sorted(arg)}.", items=[("Ergebnis", _show(res.value))]))
    res.highlights = [("Aufrufe", res.calls), ("Ebenen", res.height), ("Ergebnis", _show(res.value))]
    if name == "mergeSort" and res.value != sorted(arg):
        raise AssertionError("mergeSort-Baum liefert keine sortierte Liste")       # nie erwartet
    return res


# ==========================================================================
# Türme von Hanoi
# ==========================================================================

class HanoiResult(Result):
    """``moves`` [(Scheibe, von, nach)], ``count`` (= 2ⁿ − 1), ``final`` {Stab: Scheiben unten→oben}."""


def hanoi(n: int, start: str = "B", ziel: str = "C", hilf: str = "A") -> HanoiResult:
    """
    Türme von Hanoi (Serie 6, Aufgabe 3): rekursiv, in 2ⁿ − 1 Zügen. Im Bild der Serie steht der Turm
    auf dem mittleren Pfosten und soll nach rechts: A = links, B = Mitte (Start), C = rechts (Ziel).

    Jeder Zug wird beim Rechnen gegen die Spielregeln geprüft (eine Scheibe pro Zug, nie gross auf klein).

    >>> hanoi(3).count
    7
    >>> hanoi(2).moves
    [(1, 'B', 'A'), (2, 'B', 'C'), (1, 'A', 'C')]
    """
    n = _int(n, "n", 1, MAX_HANOI_MOVES_N)
    pegs = [str(start), str(ziel), str(hilf)]
    if len(set(pegs)) != 3 or not all(pegs):
        raise ValueError("start, ziel und hilf müssen drei verschiedene Pfosten sein")
    moves = _hanoi_moves(n, *pegs)
    state = {start: list(range(n, 0, -1)), ziel: [], hilf: []}
    for disc, src, dst in moves:
        if not state[src] or state[src][-1] != disc:
            raise AssertionError("Zug nimmt nicht die oberste Scheibe")             # nie erwartet
        if state[dst] and state[dst][-1] < disc:
            raise AssertionError("grosse Scheibe auf kleiner")                       # nie erwartet
        state[dst].append(state[src].pop())
    res = HanoiResult()
    res.n, res.moves, res.count, res.final = n, moves, len(moves), state
    assert res.count == 2 ** n - 1 and state[ziel] == list(range(n, 0, -1))
    res.title = f"Türme von Hanoi: n = {n}"
    res.summary = f"{res.count} = 2^{n} − 1 Züge von {start} nach {ziel} (Hilfspfosten {hilf})."
    res.steps.append(Step("Idee", "Die grösste Scheibe kann nur frei liegen, wenn die anderen n−1 auf dem Hilfspfosten sind. "
                          "Also: n−1 Scheiben auf den Hilfspfosten, die grösste ans Ziel, die n−1 Scheiben aufs Ziel.",
                          rule="Divide and Conquer"))
    res.steps.append(Step("Pseudocode", RECURSIVE_CODE["hanoi"]))
    rows = [[str(i), f"Scheibe {d}", s, t] for i, (d, s, t) in enumerate(moves, 1)]
    if len(rows) > 40:
        rows = rows[:20] + [["…", "…", "…", "…"]] + rows[-19:]
    res.steps.append(Step("Die Züge", f"Zug für Zug, {res.count} in Summe.",
                          table=Table(["Zug", "Scheibe", "von", "nach"], rows)))
    res.steps.append(Step(
        "Anzahl Züge", "T(n) = 2·T(n−1) + 1 mit T(1) = 1. Lösung nach Satz 2: T(n) = 2ⁿ − 1 "
        "(siehe solve_recurrence([2], 1, {1: 1})).", items=[("T(n)", f"2^{n} − 1 = {res.count}")],
        rule="Rekursionsbeziehung"))
    res.highlights = [("Züge", res.count)]
    return res


# ==========================================================================
# Master-Satz für divide-and-conquer: T(n) = a·T(n/b) + c·nᵏ
# ==========================================================================

class MasterResult(Result):
    """
    ``case`` (1: bᵏ < a, 2: bᵏ = a, 3: bᵏ > a), ``theta`` (Text), ``exponent`` (SymPy, Exponent von n),
    ``log_factor`` (True bei Fall 2), ``levels`` (Tabelle der Ebenen), ``values`` {n = bʲ: T(n)},
    ``closed`` {j: geschlossene Form von T(bʲ)}.
    """


def _pow_str(base: str, e) -> str:
    sup = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    if e == 0:
        return "1"
    if e == 1:
        return base
    if e.is_Integer:
        return base + str(int(e)).translate(sup)
    return f"{base}^({sp.sstr(e)})"


def _rat(v, name):
    try:
        r = sp.Rational(v)
    except (TypeError, ValueError):
        raise ValueError(f"{name} muss eine Zahl sein, nicht {v!r}") from None
    return r


def master_theorem(a, b, k, c=1, *, levels: int | None = None) -> MasterResult:
    """
    T(n) = a·T(n/b) + c·nᵏ mit T(1) = c (Serie 6, Satz auf Seite 7), Aufgaben 5 bis 7.

        bᵏ < a   ->  Θ(n^(log_b a))      die Blätter dominieren
        bᵏ = a   ->  Θ(nᵏ · log n)       jede Ebene kostet gleich viel
        bᵏ > a   ->  Θ(nᵏ)               die Wurzel dominiert

    Dazu der Rekursionsbaum für n = bʲ (Hinweis der Serie): Ebene i hat aⁱ Knoten der Grösse n/bⁱ,
    Gesamtaufwand der Ebene c·nᵏ·(a/bᵏ)ⁱ. ``a``, ``b`` und ``k`` sind ganze Zahlen (a ≥ 1, b ≥ 2, k ≥ 0).

    >>> master_theorem(2, 2, 1).theta
    'Θ(n log n)'
    >>> master_theorem(4, 3, 2).theta
    'Θ(n²)'
    """
    for v, nm in ((a, "a"), (b, "b"), (k, "k")):
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"{nm} muss eine ganze Zahl sein (shortcut: ganzzahliges k, wie in Serie 6)")
    if a < 1 or b < 2 or k < 0:
        raise ValueError("Es muss a ≥ 1, b ≥ 2 und k ≥ 0 gelten")
    if a > 10 ** 6 or b > 10 ** 6 or k > 100:
        raise ValueError("Zahlen zu gross (a, b ≤ 10^6, k ≤ 100)")
    c = _rat(c, "c")
    if c <= 0:
        raise ValueError("c muss positiv sein")
    bk = b ** k
    case = 1 if bk < a else 2 if bk == a else 3
    logba = sp.simplify(sp.log(a, b)) if a > 1 else sp.Integer(0)
    if case == 1:
        exponent, logf = logba, False
        theta = f"Θ({_pow_str('n', logba)})" if logba.is_Rational else f"Θ(n^(log_{b} {a}))"
    elif case == 2:
        exponent, logf = sp.Integer(k), True
        theta = "Θ(log n)" if k == 0 else f"Θ({_pow_str('n', exponent)} log n)"
    else:
        exponent, logf, theta = sp.Integer(k), False, f"Θ({_pow_str('n', sp.Integer(k))})"
    res = MasterResult()
    res.case, res.theta, res.exponent, res.log_factor = case, theta, exponent, logf
    res.a, res.b, res.k, res.c = a, b, k, c
    cs = "" if c == 1 else sp.sstr(c) + "·"
    work = (f"{sp.sstr(c)}" if k == 0 else f"{cs}{_pow_str('n', sp.Integer(k))}")
    res.title = f"Master-Satz: T(n) = {a}·T(n/{b}) + {work}"
    res.summary = f"b^k = {b}^{k} = {bk} {'<' if case == 1 else '=' if case == 2 else '>'} a = {a}  ->  {theta}"
    res.steps.append(Step("Form erkennen", f"T(n) = a·T(n/b) + c·nᵏ mit a = {a}, b = {b}, k = {k}, c = {sp.sstr(c)}; T(1) = {sp.sstr(c)}.",
                          items=[("a", a), ("b", b), ("k", k)]))
    res.steps.append(Step("bᵏ mit a vergleichen", f"b^k = {b}^{k} = {bk}, a = {a}.",
                          items=[("bᵏ", bk), ("a", a)],
                          note={1: "bᵏ < a: viele Teilprobleme, wenig Aufwand pro Teilung — die Blätter dominieren.",
                                2: "bᵏ = a: jede Ebene des Baums kostet gleich viel.",
                                3: "bᵏ > a: der Aufwand für Zerlegen und Zusammensetzen dominiert — die Wurzel."}[case],
                          rule="Satz (divide and conquer)"))
    # Rekursionsbaum für n = b^j
    j = levels if levels is not None else (4 if b ** 4 <= 10 ** 4 else 3)
    if isinstance(j, bool) or not isinstance(j, int) or not 1 <= j <= MAX_LEVELS:
        raise ValueError(f"levels muss eine ganze Zahl zwischen 1 und {MAX_LEVELS} sein")
    q = sp.Rational(a, bk)
    n_j = b ** j
    rows = []
    for i in range(j + 1):
        size = sp.Rational(n_j, b ** i)
        per = c * size ** k
        rows.append([str(i), str(a ** i), f"n/{b}^{i} = {sp.sstr(size)}", sp.sstr(per), sp.sstr(a ** i * per)])
    res.levels = Table(["Ebene i", "Knoten aⁱ", "Grösse n/bⁱ", "Aufwand je Knoten", "Aufwand der Ebene"], rows)
    res.steps.append(Step(
        f"Rekursionsbaum für n = {b}^{j} = {n_j}",
        f"Ebene i hat {a}^i Knoten der Grösse n/{b}^i; ein Knoten kostet {cs}(n/{b}^i)^{k}. "
        f"Pro Ebene zusammen {cs}{_pow_str('n', sp.Integer(k))}·({a}/{bk})^i. Ebene {j} sind die Blätter mit Grösse 1 und Aufwand T(1) = {sp.sstr(c)}.",
        table=res.levels, rule="Rekursionsbaum"))
    # exakte Werte T(b^i) per Rekursion, und die Summenformel
    vals = {1: c}
    for i in range(1, j + 1):
        vals[b ** i] = a * vals[b ** (i - 1)] + c * sp.Integer(b ** i) ** k
    res.values = vals
    geo = " + ".join(f"({a}/{bk})^{i}" for i in range(min(j, 2) + 1)) + (" + …" if j > 2 else "")
    closed = {}
    for i in range(j + 1):
        t = sum(a ** m * c * sp.Integer(b ** (i - m)) ** k for m in range(i + 1))
        assert t == vals[b ** i]
        closed[i] = t
    res.closed = closed
    res.steps.append(Step(
        "Alle Ebenen addieren", f"T(n) = {cs}{_pow_str('n', sp.Integer(k))}·(1 + q + q² + … + q^j) mit q = a/bᵏ = {a}/{bk} = {sp.sstr(q)} und j = log_{b} n  "
        f"(geometrische Reihe: {geo}).",
        table=Table(["n", "T(n) aus der Rekursion"], [[sp.sstr(m), sp.sstr(v)] for m, v in vals.items()]),
        rule="Geometrische Reihe"))
    res.steps.append(Step(
        "Welcher Term dominiert?",
        {1: f"q = {sp.sstr(q)} > 1: die Reihe wird von ihrem letzten Glied dominiert, also a^j = n^(log_{b} {a}) (die Blätter).",
         2: "q = 1: jede der j + 1 Ebenen kostet gleich viel, also (j + 1)·c·nᵏ mit j = log_" f"{b} n.",
         3: f"q = {sp.sstr(q)} < 1: die Reihe ist beschränkt (Summe ≤ 1/(1−q)), also zählt nur das erste Glied c·nᵏ (die Wurzel)."}[case],
        items=[("T(n)", theta)], rule="Θ-Abschätzung"))
    res.highlights = [("T(n)", theta)]
    return res


# ==========================================================================
# Lineare Rekursionsbeziehungen mit konstanten Koeffizienten
# ==========================================================================

class RecurrenceResult(Result):
    """
    ``solution`` (SymPy-Term in n), ``homogeneous`` (allgemeine Lösung Tₕ mit Konstanten a, b),
    ``particular`` (Tₚ), ``roots`` {r: Vielfachheit}, ``constants`` {a: Wert, …}, ``init`` (Anfangswerte oder None),
    ``at(n)`` (T(n) als Zahl).
    """

    def at(self, n):
        return sp.nsimplify(self.solution.subs(self.var, n))


def _decompose(F, n):
    """F(n) = Σ coef·nʲ·sⁿ  ->  [(coef, j, s)]; sonst ValueError."""
    out = []
    for term in sp.Add.make_args(sp.expand(F, power_exp=True)):
        if term == 0:
            continue
        coef, j, s = sp.Integer(1), 0, sp.Integer(1)
        for fac in sp.Mul.make_args(term):
            if not fac.has(n):
                coef *= fac
            elif fac == n:
                j += 1
            elif fac.is_Pow and fac.base == n and fac.exp.is_Integer and fac.exp > 0:
                j += int(fac.exp)
            elif fac.is_Pow and fac.exp == n and not fac.base.has(n) and fac.base.is_Rational and fac.base != 0:
                s *= fac.base
            else:
                raise ValueError(f"F(n) = {sp.sstr(F)}: unterstützt sind Summen von c·nʲ·sⁿ, "
                                 f"z. B. 7*n, 2**n, 3*n**2*5**n (nicht {sp.sstr(fac)})")
        out.append((coef, j, s))
    return out


def _poly_ansatz(m: int, j: int, n):
    ps = sp.symbols(f"p0:{j + 1}")
    return ps, n ** m * sum(p * n ** i for i, p in enumerate(ps))


def solve_recurrence(coeffs, F=0, init=None, var: str = "n") -> RecurrenceResult:
    """
    Löst T(n) = c₁·T(n−1) + … + c_k·T(n−k) + F(n) für k = 1 oder 2 (Serie 6, Satz 1 und Satz 2, Aufgabe 8).

    ``coeffs`` = [c₁, …, c_k]. ``F`` ist ein Term in n aus Summanden c·nʲ·sⁿ (z. B. ``"2**n"``, ``"7*n"``, ``0``).
    ``init`` = {Index: Wert} mit k aufeinanderfolgenden Indizes, z. B. {0: 1, 1: 6}; ohne ``init``
    bleiben die Konstanten a, b stehen.

    >>> solve_recurrence([5], 0, {0: 4}).solution
    4*5**n
    >>> solve_recurrence([6, -9], 0, {0: 1, 1: 6}).solution
    3**n*(n + 1)
    """
    n = sp.Symbol(var, integer=True)
    if not isinstance(coeffs, (list, tuple)) or len(coeffs) not in (1, 2):
        raise ValueError("coeffs: eine Liste [c1] oder [c1, c2] (Satz 1 behandelt nur Grad 1 und 2)")
    cs = [_rat(c, "ci") for c in coeffs]
    k = len(cs)
    if cs[-1] == 0:
        raise ValueError("Der letzte Koeffizient muss ≠ 0 sein (sonst ist der Grad kleiner)")
    if isinstance(F, (int, float, Fraction)) and not isinstance(F, bool):
        Fe = sp.nsimplify(F)
    else:
        Fe = parse_term(F, var) if str(F).strip() not in ("", "0") else sp.Integer(0)
        Fe = Fe.subs({x: n for x in Fe.free_symbols})
    if Fe.free_symbols - {n}:
        raise ValueError("F darf nur die Variable n enthalten")
    terms = _decompose(Fe, n) if Fe != 0 else []
    if init is not None:
        if not isinstance(init, dict) or len(init) != k or not all(
                isinstance(i, int) and not isinstance(i, bool) for i in init):
            raise ValueError(f"init: {k} Anfangswert(e) als {{Index: Wert}}, z. B. {{0: 1, 1: 6}}")
        idx = sorted(init)
        if idx != list(range(idx[0], idx[0] + k)):
            raise ValueError("init: die Indizes müssen aufeinanderfolgen (etwa 0 und 1)")
        init = {i: _rat(v, "Anfangswert") for i, v in init.items()}

    res = RecurrenceResult()
    res.var, res.init = n, init
    # -- charakteristische Gleichung -------------------------------------
    r = sp.Symbol("r")
    char = r ** k - sum(c * r ** (k - 1 - i) for i, c in enumerate(cs))
    roots = sp.roots(sp.Poly(char, r))
    if sum(roots.values()) != k:
        raise NotImplementedError("Die charakteristische Gleichung hat komplexe oder nicht darstellbare Lösungen — "
                                  "Satz 1 der Serie behandelt nur reelle Wurzeln. TODO: komplexer Fall")
    if any(not rt.is_real for rt in roots):
        raise NotImplementedError("Komplexe Wurzeln werden nicht behandelt (TODO).")
    res.roots = dict(roots)
    order = sorted(roots, key=lambda x: (float(x), ))
    consts = sp.symbols("a b")[:k]
    if k == 1:
        homo = consts[0] * order[0] ** n
    elif len(order) == 2:
        homo = consts[0] * order[0] ** n + consts[1] * order[1] ** n
    else:
        homo = (consts[0] + consts[1] * n) * order[0] ** n
    res.homogeneous = homo
    lhs = " + ".join(f"{sp.sstr(c)}·T({var}−{i + 1})" for i, c in enumerate(cs)).replace("+ -", "− ").replace("1·T", "T")
    rec_text = f"T({var}) = {lhs}" + (f" + {sp.sstr(Fe)}" if Fe != 0 else "")
    res.title = f"Rekursionsbeziehung lösen: {rec_text}"
    kind = "homogen" if Fe == 0 else "inhomogen"
    res.steps.append(Step("Typ", f"Linear vom Grad k = {k} mit konstanten Koeffizienten c = {', '.join(map(sp.sstr, cs))}; "
                          f"F({var}) = {sp.sstr(Fe)}, also {kind}.", rule="Definition 1"))
    ch = f"r = {sp.sstr(cs[0])}" if k == 1 else f"r² = {sp.sstr(cs[0])}·r + {sp.sstr(cs[1])}".replace("+ -", "− ")
    rr = ", ".join(f"r{'' if len(roots) == 1 and k == 1 else i + 1} = {sp.sstr(x)}" if len(order) == k else f"r = {sp.sstr(x)} (doppelt)"
                   for i, x in enumerate(order))
    res.steps.append(Step("Allgemeine Lösung der homogenen Beziehung",
                          f"Charakteristische Gleichung {ch}  ->  {rr}.",
                          items=[(f"Tₕ({var})", homo)],
                          note="" if k == 1 else ("zwei verschiedene Wurzeln: a·r₁ⁿ + b·r₂ⁿ" if len(order) == 2
                                                  else "doppelte Wurzel: a·rⁿ + b·n·rⁿ"),
                          rule="Satz 1"))
    # -- partikuläre Lösung ----------------------------------------------
    part = sp.Integer(0)
    ansatz_lines = []
    for coef, j, s in terms:
        m = roots.get(s, 0)
        ps, q = _poly_ansatz(m, j, n)
        # q(n) − Σ cᵢ·s^(−i)·q(n−i) = coef·nʲ   (s^n wurde herausgekürzt)
        eq = sp.expand(q - sum(c * s ** (-(i + 1)) * q.subs(n, n - (i + 1)) for i, c in enumerate(cs)) - coef * n ** j)
        sol = sp.solve(sp.Poly(eq, n).all_coeffs(), ps, dict=True)
        if not sol:
            raise ArithmeticError("Der Ansatz geht nicht auf")        # nie erwartet: m passt zur Vielfachheit
        tp = sp.expand(q.subs(sol[0])) * s ** n
        part += tp
        shown = " + ".join([f"{sp.sstr(p)}" if i == 0 else f"{sp.sstr(p)}·{var}" + (f"^{i}" if i > 1 else "")
                            for i, p in enumerate(ps)])
        mult = "" if m == 0 else (f"{var}·" if m == 1 else f"{var}²·")
        ansatz_lines.append((coef, j, s, m, f"{mult}({shown})" + ("" if s == 1 else f"·{sp.sstr(s)}^{var}"), tp))
    res.particular = part
    if Fe == 0:
        res.steps.append(Step("Partikuläre Lösung", "F = 0: nichts zu tun, Tₚ = 0.", rule="Satz 2"))
        general = homo
    else:
        det = []
        for coef, j, s, m, form, tp in ansatz_lines:
            det.append(f"Zu {sp.sstr(coef * n ** j * s ** n)}: Ansatz Tₚ = {form}"
                       + (f" (s = {sp.sstr(s)} ist {m}-fache Wurzel, darum mit {var}^{m})" if m else "")
                       + f"; Einsetzen und Koeffizienten vergleichen gibt Tₚ = {sp.sstr(tp)}.")
        res.steps.append(Step("Partikuläre Lösung durch Ansatz", "  ".join(det), items=[(f"Tₚ({var})", part)],
                              note="Gleiche Form wie F(n); ist die Basis schon Wurzel der charakteristischen Gleichung, "
                                   "wird der Ansatz mit n multipliziert.", rule="Satz 2"))
        general = homo + part
    res.general = general
    res.steps.append(Step("Allgemeine Lösung", "T = Tₕ + Tₚ.", items=[(f"T({var})", general)], rule="Satz 2"))
    # -- Konstanten aus den Anfangsbedingungen ----------------------------
    if init is None:
        res.constants = {}
        res.solution = general
        res.highlights = [(f"T({var})", general)]
    else:
        eqs = [sp.Eq(general.subs(n, i), v) for i, v in init.items()]
        sol = sp.solve(eqs, list(consts), dict=True)
        if not sol:
            raise ValueError("Die Anfangsbedingungen passen nicht zur Form der Lösung")
        res.constants = {str(x): sol[0][x] for x in consts}
        solution = sp.simplify(general.subs(sol[0]))
        res.solution = solution
        res.steps.append(Step(
            "Konstanten aus den Anfangsbedingungen",
            "  ".join(f"T({i}) = {sp.sstr(v)}  ->  {sp.sstr(sp.expand(general.subs(n, i)))} = {sp.sstr(v)}" for i, v in init.items()),
            items=[(nm, v) for nm, v in res.constants.items()] + [(f"T({var})", solution)], rule="Anfangsbedingungen"))
        # -- Probe: Rekursion von Hand iterieren ---------------------------
        i0 = min(init)
        seq = {i: init[i] for i in init}
        for m in range(i0 + k, i0 + k + MAX_PROBE):
            seq[m] = sum(c * seq[m - i - 1] for i, c in enumerate(cs)) + Fe.subs(n, m)
        bad = [m for m in seq if sp.simplify(solution.subs(n, m) - seq[m]) != 0]
        if bad:
            raise AssertionError(f"Probe schlägt bei n = {bad[:3]} fehl")           # nie erwartet
        shown = sorted(seq)[:6]
        res.steps.append(Step("Probe", "Die Beziehung von Hand weitergerechnet stimmt mit der Formel überein.",
                              table=Table([var] + [str(m) for m in shown],
                                          [["T(n) per Rekursion"] + [sp.sstr(seq[m]) for m in shown],
                                           ["T(n) per Formel"] + [sp.sstr(sp.nsimplify(solution.subs(n, m))) for m in shown]])))
        res.highlights = [(f"T({var})", solution)]
    return res
