#!/usr/bin/env python3
"""
dmatkit — Diskrete Mathematik für Datenstrukturen und Algorithmen, mit Rechenweg.

Gebaut zum Modul DMATH-ALGO der HSLU (Semesterwochen 1 bis 6: Datentypen,
Aussagenlogik, Prädikatenlogik, Beweiskalkül, Analyse von Algorithmen, Rekursion).
Jedes Verfahren zeigt den Rechenweg so, wie er in der Prüfung auf dem Papier
stünde, und nennt die angewandte Regel.

Kurzübersicht
-------------

    import sys; sys.path.insert(0, '..')       # dmatkit liegt neben den Week-Ordnern
    import dmatkit as dk

    dk.eval_bool("False or True and False and True or False")   # SW1
    dk.to_binary(409)                                           # SW1
    dk.truth_table("(A or B) and not C")                        # SW2
    dk.check_equivalence("A -> B", "not B -> not A")            # SW2
    dk.forall(range(1, 10), dk.Predicate.from_expr("p", "x < 10"))   # SW3
    dk.has_duplicate([3, 1, 4, 1, 5])                           # SW3
    dk.entails(["A -> B", "A"], "B")                            # SW4
    dk.resolution(["A -> B", "B -> C"], "A -> C")               # SW4
    dk.count_steps("decToBin", x=25)                            # SW5
    dk.max_input_size("n**2", time_limit_s=60)                  # SW5
    dk.recursion_tree("mergeSort", [4, 6, 5, 1, 7, 4, 3, 1])    # SW6
    dk.master_theorem(2, 2, 1)                                  # SW6: Θ(n log n)
    dk.solve_recurrence([6, -9], 0, {0: 1, 1: 6})               # SW6

Jedes Ergebnis zeigt sich im Notebook von selbst. `.latex()` liefert den
Rechenweg zum Einfügen in eine Abgabe, `.show()` zeigt ihn, und für die
Zeichnungen gibt es die Funktionen `plot_*`. `dk.info()` listet alles auf.

Zwei Namen kommen in zwei Wochen vor: `classify` und `parse_term` meinen die
Aussagenlogik (SW2); für SW5 heissen sie `classify_growth` und
`parse_growth_term`.
"""

from __future__ import annotations

import importlib as _il
import sys as _sys

from .rechnen import (algorithmen, aussagenlogik, core, datentypen, kalkuel,
                      praedikate, rekursion, render)
from .rechnen.core import (Formula, Var, Const, Not, And, Or, Xor, Implies, Iff,
                           TRUE, FALSE, parse, big_and, big_or, flatten, bit,
                           assignments, truth_vector, models, is_tautology,
                           is_contradiction, is_satisfiable, equivalent, fmt, tex)
from .rechnen.render import Result, Step, Table, show_tex, tex_to_text

# Die öffentlichen Namen jeder Semesterwoche. Bei Doppelungen gewinnt SW4
# (entails liefert ein Ergebnis mit Rechenweg; bool(r) ist die Antwort).
for _mod in (datentypen, aussagenlogik, praedikate, kalkuel, algorithmen, rekursion):
    for _name in _mod.__all__:
        if _mod is algorithmen and _name in ("classify", "parse_term"):
            continue
        globals()[_name] = getattr(_mod, _name)
classify_growth = algorithmen.classify
parse_growth_term = algorithmen.parse_term
del _mod, _name

from .grafik.plots import (COLORS, has_plotly, plot_bits, plot_connectives,   # noqa: E402
                           plot_entailment, plot_growth, plot_halving,
                           plot_quantifier, plot_recursion_tree, plot_resolution, plot_steps,
                           plot_syntax_tree, plot_truth_table, plotly_growth)

__version__ = "0.1.0"

#: Wochen -> Modul, für info() und die App
WEEKS = {
    1: ("Elementare Datentypen", datentypen),
    2: ("Aussagenlogik", aussagenlogik),
    3: ("Prädikatenlogik", praedikate),
    4: ("Beweiskalkül", kalkuel),
    5: ("Analyse von Algorithmen", algorithmen),
    6: ("Rekursive Algorithmen", rekursion),
}


def info() -> str:
    """Was dieses Paket kann, je Semesterwoche."""
    lines = [f"dmatkit {__version__} — Diskrete Mathematik mit Rechenweg", ""]
    for week, (title, mod) in WEEKS.items():
        names = [n for n in mod.__all__ if not n[:1].isupper()]
        lines.append(f"SW{week} {title}:")
        row = "  "
        for n in names:
            if len(row) + len(n) + 2 > 78:
                lines.append(row.rstrip(", "))
                row = "  "
            row += n + ", "
        lines.append(row.rstrip(", "))
    lines += ["", "Zeichnen:",
              "  plot_syntax_tree, plot_truth_table, plot_connectives, plot_entailment,",
              "  plot_resolution, plot_bits, plot_quantifier, plot_growth, plot_steps,",
              "  plot_halving, plot_recursion_tree, plotly_growth",
              f"  Plotly für Interaktives: {'vorhanden' if has_plotly() else 'fehlt'}",
              "", "Jedes Ergebnis kennt .latex(), .show() und (wo sinnvoll) .figure()."]
    return "\n".join(lines)


# Kurznamen wie bei lialkit/anakit: dmatkit.core, dmatkit.plots, dmatkit.exercises …
for _alias, _real in (("plots", "grafik.plots"), ("exercises", "aufgaben.exercises"),
                      ("uebung", "aufgaben.uebung")):
    _m = _il.import_module(f"{__name__}.{_real}")
    _sys.modules[f"{__name__}.{_alias}"] = _m
    globals()[_alias] = _m
for _alias in ("core", "render", "datentypen", "aussagenlogik", "praedikate", "kalkuel",
               "algorithmen", "rekursion"):
    _sys.modules[f"{__name__}.{_alias}"] = globals()[_alias]
del _il, _sys, _alias, _real, _m
