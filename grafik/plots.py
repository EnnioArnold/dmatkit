#!/usr/bin/env python3
"""
Zeichnen: Syntaxbäume, Wahrheitstabellen, Resolution, Binärzahlen, Quantoren
und Wachstum von Algorithmen.

Gestaltung im Swiss Style wie lialkit/anakit: Schwarz/Weiss, ein Rot (#e60000),
Grautöne, keine Rundungen. Jede Funktion gibt eine `matplotlib.figure.Figure`
zurück, mit `dark=True` die dunkle Variante. Es wird nie `plt.show()` benutzt
und kein pyplot-Zustand angelegt — die Figuren hängen an keinem Fenster und
funktionieren mit jedem Backend (auch Agg).

Die Funktionen nehmen Formeln, Zahlen und Listen — keine Result-Objekte:

    plot_syntax_tree("(A and B) -> not C")
    plot_truth_table("A xor (B or C)")
    plot_connectives()
    plot_entailment("A and (A -> B)", "B")
    plot_resolution(graph)                      # list[dict] aus der Resolution
    plot_bits(-37, width=8, signed=True)
    plot_quantifier([1, 2, 3, 4], [False, True, True, False])
    plot_growth({"n": "n", "n log n": "n*log(n,2)", "2^n": "2^n"}, n_max=20, log=True)
    plot_steps([1, 2, 4, 8], [3, 7, 15, 31], fit="4*n-1")
    plot_halving(100)
    plot_recursion_tree(recursion_tree("mergeSort", [4, 6, 5, 1]).root, values=True)

Plotly gibt es zusätzlich für die interaktive Variante (`plotly_growth`); fehlt
es, meldet nur diese eine Funktion es klar.
"""

from __future__ import annotations

import contextlib
import functools
import math
import re
import textwrap
import time
from collections.abc import Mapping
from typing import Callable

import mpmath as mp
import numpy as np
import sympy as sp

from ..rechnen.core import (Const, Formula, Var, assignments, big_and, bit,
                            entails, parse)

__all__ = [
    "plot_syntax_tree", "plot_truth_table", "plot_connectives", "plot_entailment",
    "plot_resolution", "plot_bits", "plot_quantifier", "plot_growth",
    "plot_steps", "plot_halving", "plot_recursion_tree", "has_plotly", "plotly_growth", "COLORS",
]

# Swiss Style wie die App: Schwarz/Weiss, ein Rot (#e60000), Grautöne.
INK = "#000000"
RED = "#e60000"
GREY = "#666666"
COLORS = [INK, RED, GREY, "#999999", "#333333", "#b30000"]

MAX_PLOT_VARS = 6          # 2^6 = 64 Tabellenzeilen passen noch gut aufs Blatt
MAX_TREE_NODES = 300
MAX_TREE_DEPTH = 300
MAX_GRAPH_NODES = 150
MAX_FUNCS = 12
MAX_EXPR_LEN = 200
MAX_EXPR_NODES = 120
MAX_ELEMENTS_1D = 200
MAX_MATRIX = 30
MAX_BITS = 64
MAX_TIME = 8.0             # Sekunden Rechenzeit für eine Wachstumszeichnung

_LIGHT = dict(
    dark=False, bg="#ffffff", fg="#000000", muted="#666666", grid="#e0e0e0",
    red="#e60000", soft="#f0f0f0", tint="#fdeaea", good="#1f8a4c",
    on_color="#ffffff", series=COLORS)
_DARK = dict(
    dark=True, bg="#000000", fg="#ffffff", muted="#999999", grid="#2a2a2a",
    red="#ff3333", soft="#1c1c1c", tint="#2e1010", good="#2fb36a",
    on_color="#000000", series=["#ffffff", "#ff3333", "#aaaaaa", "#777777",
                                "#dddddd", "#cc0000"])

_FONT_WISH = ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]
_STYLES = ["-", "--", "-.", ":"]


# ==========================================================================
# Stil, Figuren, Texthilfen
# ==========================================================================

@functools.lru_cache(maxsize=1)
def _fonts() -> tuple:
    """Nur Schriften, die es gibt (sonst meldet Matplotlib bei jedem Text eine Warnung)."""
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    out = [f for f in _FONT_WISH if f in have]
    if "DejaVu Sans" not in out:
        out.append("DejaVu Sans")      # liegt Matplotlib bei und kennt ¬ ∧ ∨ ⊕ → ↔ □ ⊨
    return tuple(out)


def _theme(dark: bool) -> dict:
    return dict(_DARK if dark else _LIGHT)


def _rc(th: dict) -> dict:
    return {
        "font.family": list(_fonts()), "font.size": 10.5, "axes.titlesize": 12,
        "axes.unicode_minus": True, "figure.facecolor": th["bg"],
        "axes.facecolor": th["bg"], "savefig.facecolor": th["bg"],
        "text.color": th["fg"], "axes.labelcolor": th["fg"],
        "axes.edgecolor": th["fg"], "xtick.color": th["fg"], "ytick.color": th["fg"],
        "grid.color": th["grid"], "grid.linewidth": 0.6,
        "lines.solid_capstyle": "butt", "lines.dash_capstyle": "butt",
        "legend.frameon": False, "legend.facecolor": th["bg"],
        "axes.formatter.use_mathtext": False, "figure.dpi": 100,
    }


@contextlib.contextmanager
def _style(dark: bool):
    import matplotlib
    th = _theme(bool(dark))
    with matplotlib.rc_context(_rc(th)):
        yield th


def _new_figure(size, th, **kwargs):
    """Figur ohne pyplot: kein Fensterzustand, kein doppeltes Anzeigen im Notebook."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=size, facecolor=th["bg"], **kwargs)
    FigureCanvasAgg(fig)
    return fig


def _check_size(figsize):
    if figsize is None:
        return None
    try:
        w, h = float(figsize[0]), float(figsize[1])
        if len(figsize) != 2:
            raise ValueError
    except Exception:
        raise ValueError("figsize muss ein Paar (Breite, Höhe) in Zoll sein") from None
    if not (0.5 <= w <= 60 and 0.5 <= h <= 60):
        raise ValueError("figsize: Breite und Höhe müssen zwischen 0.5 und 60 Zoll liegen")
    return (w, h)


def _esc(s) -> str:
    """Matplotlib deutet $…$ als Formelsatz — hier soll jeder Text so bleiben, wie er ist."""
    return str(s).replace("$", r"\$")


_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _sup(k: int) -> str:
    return str(k).translate(_SUP)


def _group(i: int) -> str:
    """1234567 -> 1’234’567 (Schweizer Tausendertrennung)."""
    return f"{i:,}".replace(",", "’")


def _sci(v) -> str:
    """Zahl kompakt: 244948 -> 244’948, 6e10 -> 6·10¹⁰, nan -> ?"""
    try:
        if isinstance(v, int):
            if abs(v) < 10 ** 7:
                return _group(v)
            lg = math.log10(abs(v))
            sign = "−" if v < 0 else ""
            e = int(math.floor(lg))
            m = abs(v) / 10 ** e if e < 300 else 10 ** (lg - e)
        else:
            v = float(v)
            if math.isnan(v):
                return "?"
            if math.isinf(v):
                return "∞" if v > 0 else "−∞"
            if v == int(v) and abs(v) < 10 ** 7:
                return _group(int(v))
            if v == 0:
                return "0"
            sign = "−" if v < 0 else ""
            e = int(math.floor(math.log10(abs(v))))
            m = abs(v) / 10 ** e
            if abs(v) < 1000 and abs(v) >= 0.01:
                return sign + f"{abs(v):.3g}"
        ms = f"{m:.2f}".rstrip("0").rstrip(".")
        if ms == "10":
            ms, e = "1", e + 1
        return f"{sign}{ms}·10{_sup(e)}" if ms != "1" else f"{sign}10{_sup(e)}"
    except Exception:
        return "?"


def _label(e, limit: int = 24) -> str:
    """Element eines Universums als kurzer Text."""
    if isinstance(e, bool):
        return "wahr" if e else "falsch"
    if isinstance(e, (tuple, list)):
        s = "(" + ", ".join(_label(v, limit) for v in e) + ")"
    elif isinstance(e, (set, frozenset)):
        s = "{" + ", ".join(sorted(_label(v, limit) for v in e)) + "}"
    elif isinstance(e, float):
        s = f"{e:.4g}"
    else:
        s = str(e)
    return s if len(s) <= limit else s[:limit - 1] + "…"


def _formula(f, what: str = "Formel") -> Formula:
    if f is None:
        raise ValueError(f"{what}: es wurde nichts übergeben")
    if isinstance(f, (int, float)) and not isinstance(f, bool):
        raise ValueError(f"{what}: {f!r} ist keine Formel (Text wie 'A and B' erwartet)")
    return parse(f)


def _names(f: Formula, names) -> list[str]:
    own = f.variables()
    if names is None:
        out = own
    else:
        if isinstance(names, str):
            names = [n for n in re.split(r"[\s,;]+", names) if n]
        out = [str(n) for n in names]
        if len(set(out)) != len(out):
            raise ValueError("names enthält eine Variable doppelt")
        missing = [v for v in own if v not in out]
        if missing:
            raise ValueError(f"Die Variable {missing[0]!r} kommt in der Formel vor, "
                             f"fehlt aber in names")
    if len(out) > MAX_PLOT_VARS:
        raise ValueError(f"Zu viele Variablen für eine Zeichnung ({len(out)}); "
                         f"höchstens {MAX_PLOT_VARS} (= {2 ** MAX_PLOT_VARS} Zeilen)")
    return out


# ==========================================================================
# Zeichenblatt in Zoll (für Bäume, Tabellen, Graphen)
# ==========================================================================

class _Sheet:
    """
    Sammelt Zeichenbefehle in Zoll (y wächst nach unten) und baut daraus am Ende
    eine Figur mit Titel. So lassen sich Texte vorher ausmessen, und die Figur
    wird genau so gross, wie der Inhalt es braucht.
    """

    MAXW, MAXH = 24.0, 26.0
    MARGIN = 0.25
    MINW = 4.4

    def __init__(self, th: dict):
        self.th = th
        self.ops: list[tuple] = []
        self._cache: dict = {}
        self._probe = _new_figure((1, 1), th)
        self._renderer = self._probe.canvas.get_renderer()

    # -- Messen -----------------------------------------------------------
    def w(self, s: str, size: float = 11, weight: str = "normal") -> float:
        key = (s, size, weight)
        if key not in self._cache:
            from matplotlib.font_manager import FontProperties
            prop = FontProperties(size=size, weight=weight)
            width, _h, _d = self._renderer.get_text_width_height_descent(str(s), prop, False)
            self._cache[key] = width / self._probe.dpi
        return self._cache[key]

    def wrap(self, s: str, size: float, max_w: float, weight: str = "normal") -> list[str]:
        """Bricht an Leerzeichen um, so dass jede Zeile höchstens max_w Zoll breit ist."""
        lines: list[str] = []
        for para in str(s).split("\n"):
            cur = ""
            for word in para.split(" "):
                trial = word if not cur else cur + " " + word
                if cur and self.w(trial, size, weight) > max_w:
                    lines.append(cur)
                    cur = word
                else:
                    cur = trial
            lines.append(cur)
        return lines

    # -- Befehle ----------------------------------------------------------
    def rect(self, x, y, w, h, fill=None, edge=None, lw=0.0, ls="-", hatch=None, z=1):
        self.ops.append(("rect", x, y, w, h, fill, edge, lw, ls, hatch, z))

    def circle(self, x, y, r, fill=None, edge=None, lw=1.0, z=2):
        self.ops.append(("circle", x, y, r, fill, edge, lw, z))

    def line(self, x0, y0, x1, y1, color=None, lw=1.0, ls="-", z=2):
        self.ops.append(("line", x0, y0, x1, y1, color or self.th["fg"], lw, ls, z))

    def arrow(self, x0, y0, x1, y1, color=None, lw=1.0, z=1):
        self.ops.append(("arrow", x0, y0, x1, y1, color or self.th["fg"], lw, z))

    def text(self, x, y, s, size=11, color=None, weight="normal", ha="center",
             va="center_baseline", z=3):
        self.ops.append(("text", x, y, str(s), size, color or self.th["fg"], weight, ha, va, z))

    # -- Fertigstellen ----------------------------------------------------
    def finish(self, cw: float, ch: float, title, subs=(), figsize=None):
        from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
        th, m = self.th, self.MARGIN
        tw = max(cw, self.MINW)
        tlines = self.wrap(title, 13, tw, "bold") if title else []
        slines = [ln for s in subs for ln in self.wrap(s, 10, tw)]
        head = len(tlines) * 0.27 + len(slines) * 0.2 + (0.2 if (tlines or slines) else 0)
        nat_w, nat_h = tw + 2 * m, head + ch + 2 * m
        size = _check_size(figsize)
        if size is None:
            # ponytail: zu grosse Blätter werden verkleinert statt umgebrochen
            k0 = max(min(1.0, self.MAXW / nat_w, self.MAXH / nat_h), 0.3)
            size = (nat_w * k0, nat_h * k0)
        k = min(size[0] / nat_w, size[1] / nat_h)
        fig = _new_figure(size, th)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(-m, tw + m)
        ax.set_ylim(ch + m, -(head + m))
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        dx = (tw - cw) / 2
        y = -head + 0.13
        for ln in tlines:
            ax.text(0, y + 0.01, _esc(ln), fontsize=13 * k, fontweight="bold", color=th["fg"],
                    ha="left", va="center_baseline")
            y += 0.27
        for ln in slines:
            ax.text(0, y, _esc(ln), fontsize=10 * k, color=th["muted"], ha="left",
                    va="center_baseline")
            y += 0.2
        for op in self.ops:
            kind = op[0]
            if kind == "rect":
                _k, x0, y0, w, h, fill, edge, lw, ls, hatch, z = op
                ax.add_patch(Rectangle((x0 + dx, y0), w, h, facecolor=fill or "none",
                                       edgecolor=edge or "none", linewidth=lw * k,
                                       linestyle=ls, hatch=hatch, zorder=z))
            elif kind == "circle":
                _k, x0, y0, r, fill, edge, lw, z = op
                ax.add_patch(Circle((x0 + dx, y0), r, facecolor=fill or "none",
                                    edgecolor=edge or "none", linewidth=lw * k, zorder=z))
            elif kind == "line":
                _k, xa, ya, xb, yb, color, lw, ls, z = op
                ax.plot([xa + dx, xb + dx], [ya, yb], color=color, linewidth=lw * k,
                        linestyle=ls, zorder=z, solid_capstyle="butt")
            elif kind == "arrow":
                _k, xa, ya, xb, yb, color, lw, z = op
                ax.add_patch(FancyArrowPatch((xa + dx, ya), (xb + dx, yb), arrowstyle="-|>",
                                             mutation_scale=9 * k, shrinkA=0, shrinkB=0,
                                             linewidth=lw * k, color=color, zorder=z))
            else:
                _k, x0, y0, s, fs, color, weight, ha, va, z = op
                ax.text(x0 + dx, y0, _esc(s), fontsize=fs * k, color=color, fontweight=weight,
                        ha=ha, va=va, zorder=z, clip_on=False)
        return fig


def _table(sh: _Sheet, x: float, y: float, headers, rows, *, fs=11, row_h=0.31, head_h=0.4,
           pad=0.3, min_w=0.5, head_fill=None, head_color=None, row_fill=None,
           cell_fill=None, cell_color=None, bold_cols=(), vsep=(), box_cols=()):
    """
    Tabelle im Buchsatz-Stil (Linien oben, unter der Kopfzeile, unten).
    Gibt Breite, Höhe und die Spaltengeometrie zurück.
    """
    th = sh.th
    head_fill, head_color = head_fill or {}, head_color or {}
    row_fill, cell_fill, cell_color = row_fill or {}, cell_fill or {}, cell_color or {}
    ncol = len(headers)
    cw = []
    for j in range(ncol):
        w = sh.w(str(headers[j]), fs, "bold") + pad
        for r in rows:
            w = max(w, sh.w(str(r[j]), fs, "bold" if j in bold_cols else "normal") + pad)
        cw.append(max(min_w, w))
    total = sum(cw)
    cx = [x + sum(cw[:j]) for j in range(ncol)]
    bottom = y + head_h + len(rows) * row_h
    for j in range(ncol):
        if j in head_fill:
            sh.rect(cx[j], y, cw[j], head_h, fill=head_fill[j], z=1)
    for i in range(len(rows)):
        ry = y + head_h + i * row_h
        if i in row_fill:
            sh.rect(x, ry, total, row_h, fill=row_fill[i], z=1)
        for j in range(ncol):
            if (i, j) in cell_fill:
                sh.rect(cx[j], ry, cw[j], row_h, fill=cell_fill[(i, j)], z=1)
        if i > 0:
            sh.line(x, ry, x + total, ry, color=th["grid"], lw=0.5, z=2)
    sh.line(x, y, x + total, y, lw=1.4)
    sh.line(x, y + head_h, x + total, y + head_h, lw=0.9)
    sh.line(x, bottom, x + total, bottom, lw=1.4)
    for j in vsep:
        sh.line(cx[j], y, cx[j], bottom, color=th["muted"], lw=0.7)
    for j in box_cols:
        sh.rect(cx[j], y, cw[j], bottom - y, edge=th["red"], lw=1.6, z=4)
    for j in range(ncol):
        sh.text(cx[j] + cw[j] / 2, y + head_h / 2, headers[j], fs, head_color.get(j), "bold")
    for i, r in enumerate(rows):
        ry = y + head_h + i * row_h + row_h / 2
        for j in range(ncol):
            sh.text(cx[j] + cw[j] / 2, ry, r[j], fs, cell_color.get((i, j)),
                    "bold" if j in bold_cols else "normal")
    return dict(w=total, h=bottom - y, cx=cx, cw=cw, bottom=bottom)


# ==========================================================================
# Syntaxbaum
# ==========================================================================

class _TNode:
    __slots__ = ("label", "kids", "lines", "w", "h", "round", "x", "y", "offs", "depth")

    def __init__(self, label: str, kids: list):
        self.label, self.kids = label, kids


def _build_tree(tree) -> _TNode:
    """Formel, Formeltext oder jedes Objekt mit .label und .children -> interner Baum."""
    if isinstance(tree, (str, bool)) or isinstance(tree, Formula):
        tree = _formula(tree, "Syntaxbaum")
    if tree is None:
        raise ValueError("Syntaxbaum: es wurde nichts übergeben")
    count = 0

    def conv(obj, depth: int) -> _TNode:
        nonlocal count
        count += 1
        if depth > MAX_TREE_DEPTH:
            raise ValueError(f"Syntaxbaum zu tief (mehr als {MAX_TREE_DEPTH} Ebenen)")
        if count > MAX_TREE_NODES:
            raise ValueError(f"Syntaxbaum zu gross (mehr als {MAX_TREE_NODES} Knoten)")
        if not hasattr(obj, "label"):
            raise ValueError(f"Knoten {obj!r} hat kein Attribut .label — "
                             f"erwartet wird eine Formel oder ein Objekt mit .label/.children")
        kids = getattr(obj, "children", ())
        if callable(kids):
            kids = kids()
        if kids is None:
            kids = ()
        if isinstance(kids, (str, bytes)):
            raise ValueError("Die .children eines Knotens müssen eine Folge von Knoten sein")
        try:
            kids = list(kids)
        except TypeError:
            raise ValueError("Die .children eines Knotens müssen eine Folge von Knoten sein") from None
        return _TNode(str(obj.label), [conv(k, depth + 1) for k in kids])

    return conv(tree, 0)


def _layout_tree(root: _TNode, gap: float):
    """
    Reingold-Tilford-artig: jeder Teilbaum wird so nah wie möglich rechts an seinen
    linken Nachbarn gerückt (Konturen je Ebene), der Elternknoten steht mittig über
    seinen Kindern. Dadurch überlappt nichts, auch bei vielen Knoten.
    """
    # ponytail: Teilbäume rücken nur nach links zusammen; mittlere Geschwister werden nicht
    # nachträglich gleichmässig verteilt (bei Formelbäumen gibt es höchstens zwei Kinder)

    def lay(node: _TNode):
        half = node.w / 2
        if not node.kids:
            node.offs = []
            return [-half], [half]
        subs = [lay(k) for k in node.kids]
        pos = [0.0]
        acc_l, acc_r = list(subs[0][0]), list(subs[0][1])
        for left, right in subs[1:]:
            shift = max(acc_r[d] - left[d] for d in range(min(len(acc_r), len(left)))) + gap
            pos.append(shift)
            for d in range(len(left)):
                if d < len(acc_l):
                    acc_r[d] = right[d] + shift
                else:
                    acc_l.append(left[d] + shift)
                    acc_r.append(right[d] + shift)
        mid = (pos[0] + pos[-1]) / 2
        node.offs = [p - mid for p in pos]
        return [-half] + [v - mid for v in acc_l], [half] + [v - mid for v in acc_r]

    lay(root)
    root.x, root.depth = 0.0, 0
    stack = [root]
    nodes = []
    while stack:
        n = stack.pop()
        nodes.append(n)
        for k, off in zip(n.kids, n.offs):
            k.x, k.depth = n.x + off, n.depth + 1
            stack.append(k)
    return nodes


def plot_syntax_tree(tree, *, dark: bool = False, figsize=None, title=None):
    """
    Syntaxbaum von oben nach unten (Wurzel oben). Innere Knoten sind Kreise
    (lange Beschriftungen Rechtecke), Blätter grau abgesetzte Rechtecke, die
    Wurzel hat einen roten Rand. Gleiche Teilformeln erscheinen als eigene Knoten.

    `tree`: eine `Formula`, ein Formeltext oder jedes Objekt mit `.label` (str)
    und `.children` (Folge solcher Objekte).

        plot_syntax_tree("(A and B) -> not C")
        plot_syntax_tree(parse("A ∨ ¬A"), dark=True)
    """
    root = _build_tree(tree)
    with _style(dark) as th:
        sh = _Sheet(th)
        stack = [root]
        nodes_all = []
        while stack:
            n = stack.pop()
            nodes_all.append(n)
            stack.extend(n.kids)
        for n in nodes_all:
            raw = n.label if len(n.label) <= 60 else n.label[:59] + "…"
            n.lines = textwrap.wrap(raw, 16, break_long_words=True) or [" "]
            n.lines = n.lines[:4]
            tw = max(sh.w(ln, 11) for ln in n.lines)
            leaf = not n.kids
            n.round = (not leaf) and len(n.lines) == 1 and len(raw) <= 2
            if n.round:
                n.w = n.h = max(0.46, tw + 0.24)
            else:
                n.w = max(0.42, tw + 0.3)
                n.h = max(0.38, 0.19 * len(n.lines) + 0.2)
        nodes = _layout_tree(root, gap=0.2)
        xmin = min(n.x - n.w / 2 for n in nodes)
        xmax = max(n.x + n.w / 2 for n in nodes)
        hmax = max(n.h for n in nodes)
        level = hmax + 0.5
        depth = max(n.depth for n in nodes)
        for n in nodes:
            n.x -= xmin
            n.y = n.depth * level + hmax / 2
        for n in nodes:
            for k in n.kids:
                sh.line(n.x, n.y, k.x, k.y, lw=1.1, z=1)
        for n in nodes:
            leaf = not n.kids
            is_root = n is root
            edge = th["red"] if is_root else th["fg"]
            lw = 2.0 if is_root else (1.0 if leaf else 1.3)
            fill = th["soft"] if leaf else th["bg"]
            if n.round:
                sh.circle(n.x, n.y, n.w / 2, fill=fill, edge=edge, lw=lw)
            else:
                sh.rect(n.x - n.w / 2, n.y - n.h / 2, n.w, n.h, fill=fill, edge=edge, lw=lw, z=2)
            size = 14 if n.round else 11
            if len(n.lines) == 1:
                sh.text(n.x, n.y, n.lines[0], size)
            else:
                top = n.y - 0.095 * (len(n.lines) - 1)
                for i, ln in enumerate(n.lines):
                    sh.text(n.x, top + 0.19 * i, ln, size)
        auto = f"Syntaxbaum  ({len(nodes)} Knoten, Tiefe {depth + 1})"
        ttl = auto if title is None else title
        return sh.finish(xmax - xmin, depth * level + hmax, ttl, figsize=figsize)


# ==========================================================================
# Wahrheitstabellen
# ==========================================================================

def _tt_cells(f, vars_, subs):
    rows = []
    for env in assignments(vars_):
        rows.append([bit(env[v]) for v in vars_] + [bit(s.eval(env)) for s in subs])
    return rows


def plot_truth_table(f, names=None, *, dark: bool = False, figsize=None, title=None):
    """
    Wahrheitstabelle: eine Spalte pro Variable, dann je Teilformel (von innen nach
    aussen), ganz rechts das Ergebnis (rot hervorgehoben). Zeilen mit Ergebnis 1
    sind leicht getönt.

    `names` legt die Variablen und ihre Reihenfolge fest (auch zusätzliche).

        plot_truth_table("A xor (B or C)")
        plot_truth_table("A -> B", names=["B", "A"])
    """
    f = _formula(f)
    vars_ = _names(f, names)
    subs = [s for s in f.subformulas() if not isinstance(s, (Var, Const))]
    if not subs or subs[-1] != f:
        subs.append(f)
    rows = _tt_cells(f, vars_, subs)
    res = len(vars_) + len(subs) - 1
    headers = list(vars_) + [s.to_str() for s in subs]
    with _style(dark) as th:
        sh = _Sheet(th)
        ones = [i for i, r in enumerate(rows) if r[res] == "1"]
        tab = _table(sh, 0, 0, headers, rows,
                     head_fill={res: th["red"]}, head_color={res: th["on_color"]},
                     row_fill={i: th["tint"] for i in ones}, bold_cols={res},
                     vsep={len(vars_)} if vars_ else (), box_cols={res})
        n = len(rows)
        if len(ones) == n:
            verdict = "Tautologie: bei jeder Belegung wahr"
        elif not ones:
            verdict = "Widerspruch: bei jeder Belegung falsch"
        else:
            verdict = f"erfüllbar: {len(ones)} von {n} Belegungen machen die Formel wahr"
        ttl = f"Wahrheitstabelle  {f}" if title is None else title
        return sh.finish(tab["w"], tab["h"], ttl, [verdict] if title is None else [], figsize)


def plot_connectives(*, dark: bool = False, figsize=None, title=None):
    """
    Die Grundverknüpfungen ¬, ∧, ∨, ⊕, →, ↔ mit ihren Wahrheitstabellen nebeneinander.

        plot_connectives()
    """
    A, B = Var("A"), Var("B")
    items = [("Negation", ~A), ("Konjunktion (und)", A & B), ("Disjunktion (oder)", A | B),
             ("Antivalenz (xor)", A ^ B), ("Implikation", A >> B), ("Äquivalenz", A.iff(B))]
    with _style(dark) as th:
        sh = _Sheet(th)
        x = 0.0
        height = 0.0
        for name, f in items:
            vars_ = f.variables()
            rows = _tt_cells(f, vars_, [f])
            res = len(vars_)
            head = max(sh.w(name, 10, "bold"), 0)
            sh.text(x, 0.12, name, 10, th["muted"], "bold", ha="left")
            tab = _table(sh, x, 0.32, vars_ + [f.to_str()], rows,
                         head_fill={res: th["red"]}, head_color={res: th["on_color"]},
                         row_fill={i: th["tint"] for i, r in enumerate(rows) if r[res] == "1"},
                         bold_cols={res}, vsep={res}, min_w=0.55)
            width = max(tab["w"], head)
            height = max(height, 0.32 + tab["h"])
            x += width + 0.45
        ttl = "Die Grundverknüpfungen der Aussagenlogik" if title is None else title
        return sh.finish(x - 0.45, height, ttl, figsize=figsize)


def _kb(wb) -> Formula:
    if isinstance(wb, (list, tuple, set, frozenset)):
        items = list(wb)
        if not items:
            raise ValueError("Die Wissensbasis ist leer — mindestens eine Formel wird gebraucht")
        return big_and([_formula(w, "Wissensbasis") for w in items])
    return _formula(wb, "Wissensbasis")


def plot_entailment(wb, q, *, dark: bool = False, figsize=None, title=None):
    """
    Wahrheitstabelle für WB ⊨ Q. Zeilen mit WB = 1 sind grau hinterlegt, dort muss
    Q = 1 gelten; Gegenbeispiele (WB = 1, Q = 0) sind rot. Das Urteil steht im Titel.
    `wb` darf eine Formel oder eine Liste von Formeln sein (deren Konjunktion).

        plot_entailment("A and (A -> B)", "B")          # gilt
        plot_entailment(["A or B", "not A"], "A")       # gilt nicht
    """
    wb = _kb(wb)
    q = _formula(q, "Anfrage Q")
    vars_ = sorted(set(wb.variables()) | set(q.variables()))
    if len(vars_) > MAX_PLOT_VARS:
        raise ValueError(f"Zu viele Variablen für eine Zeichnung ({len(vars_)}); "
                         f"höchstens {MAX_PLOT_VARS}")
    rows, hi, bad, models = [], [], [], 0
    first_bad = None
    for i, env in enumerate(assignments(vars_)):
        w, qq = wb.eval(env), q.eval(env)
        verdict = "–" if not w else ("✓" if qq else "✗")
        rows.append([bit(env[v]) for v in vars_] + [bit(w), bit(qq), verdict])
        if w:
            models += 1
            hi.append(i)
            if not qq:
                bad.append(i)
                if first_bad is None:
                    first_bad = env
    holds = not bad
    assert holds == entails(wb, q)
    cW, cQ, cU = len(vars_), len(vars_) + 1, len(vars_) + 2
    with _style(dark) as th:
        sh = _Sheet(th)
        row_fill = {i: th["soft"] for i in hi}
        row_fill.update({i: th["red"] for i in bad})
        cell_color = {(i, j): th["on_color"] for i in bad for j in range(len(rows[0]))}
        tab = _table(sh, 0, 0, vars_ + ["WB", "Q", "WB ⊨ Q?"], rows,
                     head_fill={cW: th["fg"], cQ: th["red"]},
                     head_color={cW: th["bg"], cQ: th["on_color"]},
                     row_fill=row_fill, cell_color=cell_color, bold_cols={cW, cQ, cU},
                     vsep={cW})
        subs = [f"WB = {wb}", f"Q = {q}"]
        if holds:
            head = "WB ⊨ Q gilt"
            if models == 0:
                subs.append("Die WB ist unerfüllbar (kein Modell): sie hat jede Folgerung "
                            "(ex falso quodlibet).")
            else:
                subs.append(f"Die WB hat {models} Modell(e) (graue Zeilen), in allen ist Q = 1.")
        else:
            head = "WB ⊭ Q: es gibt ein Gegenbeispiel"
            ce = ", ".join(f"{v} = {bit(first_bad[v])}" for v in vars_) or "–"
            subs.append(f"Gegenbeispiel (rot): {ce}. Dort ist WB = 1, aber Q = 0.")
        ttl = head if title is None else title
        return sh.finish(tab["w"], tab["h"], ttl, subs, figsize)


# ==========================================================================
# Resolutionsgraph
# ==========================================================================

_EMPTY_CLAUSES = {"□", "{}", "⊥", "[]"}


def _check_graph(graph) -> list[dict]:
    if not isinstance(graph, (list, tuple)) or not graph:
        raise ValueError("Der Resolutionsgraph muss eine nichtleere Liste von Einträgen sein")
    if len(graph) > MAX_GRAPH_NODES:
        raise ValueError(f"Resolutionsgraph zu gross (mehr als {MAX_GRAPH_NODES} Klauseln)")
    nodes, ids = [], set()
    for entry in graph:
        if not isinstance(entry, Mapping):
            raise ValueError("Jeder Eintrag des Resolutionsgraphen muss ein dict sein "
                             "(Schlüssel id, clause, parents, origin)")
        missing = [k for k in ("id", "clause", "parents", "origin") if k not in entry]
        if missing:
            raise ValueError(f"Eintrag {dict(entry)!r}: Schlüssel {missing[0]!r} fehlt")
        i = entry["id"]
        if isinstance(i, bool) or not isinstance(i, int):
            raise ValueError(f"id muss eine ganze Zahl sein, nicht {i!r}")
        if i in ids:
            raise ValueError(f"id {i} kommt doppelt vor")
        ids.add(i)
        parents = entry["parents"]
        if parents is None:
            parents = ()
        try:
            parents = tuple(parents)
        except TypeError:
            raise ValueError(f"parents von {i} muss eine Folge von ids sein") from None
        nodes.append(dict(id=i, clause=str(entry["clause"]), parents=parents,
                          origin=str(entry["origin"])))
    for n in nodes:
        for p in n["parents"]:
            if p not in ids:
                raise ValueError(f"Klausel {n['id']} verweist auf die unbekannte id {p}")
            if p == n["id"]:
                raise ValueError(f"Klausel {n['id']} ist ihr eigener Elternteil")
    return nodes


def _origin_kind(n: dict) -> str:
    o = n["origin"].lower()
    if o.startswith("negiert") or "anfrage" in o:
        return "query"
    if o.startswith("wissens") or (not n["parents"] and not o.startswith("resolv")):
        return "kb"
    return "res"


def plot_resolution(graph, *, dark: bool = False, figsize=None, title=None):
    """
    Resolutionsgraph aus `list[dict]` (Schlüssel `id`, `clause`, `parents`, `origin`):
    Klauseln der Wissensbasis (weiss) und die negierte Anfrage (rot umrandet) oben,
    Resolventen (grau) darunter in Ebenen, Pfeile von den Eltern, die leere Klausel □
    rot ausgefüllt.

        plot_resolution([
            {"id": 1, "clause": "{A}",  "parents": (),     "origin": "Wissensbasis"},
            {"id": 2, "clause": "{¬A}", "parents": (),     "origin": "Negierte Anfrage"},
            {"id": 3, "clause": "□",    "parents": (1, 2), "origin": "Resolvent"}])
    """
    nodes = _check_graph(graph)
    by_id = {n["id"]: n for n in nodes}
    # Ebenen: Klauseln ohne Eltern oben (0), Resolvent = 1 + höchste Ebene der Eltern
    state: dict[int, int] = {}

    def level(i: int, trail: tuple = ()) -> int:
        if i in trail:
            raise ValueError("Der Resolutionsgraph enthält einen Kreis")
        if i not in state:
            ps = by_id[i]["parents"]
            state[i] = 0 if not ps else 1 + max(level(p, trail + (i,)) for p in ps)
        return state[i]

    for n in nodes:
        n["level"] = level(n["id"])
        n["kind"] = _origin_kind(n)
        n["empty"] = n["clause"].strip() in _EMPTY_CLAUSES
    derived = any(n["empty"] and n["parents"] for n in nodes)

    with _style(dark) as th:
        sh = _Sheet(th)
        for n in nodes:
            if n["empty"]:
                n["lines"], n["w"], n["h"] = ["□"], 0.6, 0.55
            else:
                n["lines"] = sh.wrap(n["clause"], 11, 3.6)[:5]
                n["w"] = max(0.7, max(sh.w(ln, 11) for ln in n["lines"]) + 0.3)
                n["h"] = max(0.42, 0.19 * len(n["lines"]) + 0.2)
        # Zeilen des Blatts
        top = sorted((n for n in nodes if n["level"] == 0),
                     key=lambda n: 1 if n["kind"] == "query" else 0)   # stabil: WB zuerst
        rows: list[list[dict]] = []
        cur, cur_w, max_w = [], 0.0, 18.0
        for n in top:
            if cur and cur_w + n["w"] + 0.3 > max_w:
                rows.append(cur)
                cur, cur_w = [], 0.0
            cur.append(n)
            cur_w += n["w"] + 0.3
        rows.append(cur)
        for lv in range(1, max(n["level"] for n in nodes) + 1):
            rows.append([n for n in nodes if n["level"] == lv])
        y = 0.0
        for row in rows:
            rh = max(n["h"] for n in row)
            for n in row:
                n["y"] = y + rh / 2
            y += rh + 0.8
        # x-Positionen: oben der Reihe nach, darunter mittig unter den Eltern
        # ponytail: Mittelung über den Eltern statt Kreuzungsminimierung (Graphen sind klein)
        for ri, row in enumerate(rows):
            if row[0]["level"] == 0:
                total = sum(n["w"] for n in row) + 0.3 * (len(row) - 1)
                x = -total / 2
                for n in row:
                    n["x"] = x + n["w"] / 2
                    x += n["w"] + 0.3
            else:
                for n in row:
                    n["want"] = sum(by_id[p]["x"] for p in n["parents"]) / len(n["parents"])
                row.sort(key=lambda n: n["want"])
                xs, prev = [], None
                for n in row:
                    xp = n["want"] if prev is None else max(n["want"], prev["xx"] + prev["w"] / 2
                                                            + 0.35 + n["w"] / 2)
                    n["xx"] = xp
                    xs.append(xp)
                    prev = n
                shift = sum(n["want"] - n["xx"] for n in row) / len(row)
                for n in row:
                    n["x"] = n["xx"] + shift
        xmin = min(n["x"] - n["w"] / 2 for n in nodes)
        xmax = max(n["x"] + n["w"] / 2 for n in nodes)
        for n in nodes:
            n["x"] -= xmin
        # Pfeile, dann Kästen
        for n in nodes:
            k = len(n["parents"])
            for j, p in enumerate(n["parents"]):
                pa = by_id[p]
                tx = n["x"] - n["w"] / 2 + n["w"] * (j + 1) / (k + 1)
                col = th["red"] if (n["empty"]) else th["fg"]
                sh.arrow(pa["x"], pa["y"] + pa["h"] / 2, tx, n["y"] - n["h"] / 2, col, 1.1)
        for n in nodes:
            x0, y0 = n["x"] - n["w"] / 2, n["y"] - n["h"] / 2
            if n["empty"]:
                sh.rect(x0, y0, n["w"], n["h"], fill=th["red"], edge=th["red"], lw=1.5, z=2)
                sh.text(n["x"], n["y"], "□", 18, th["on_color"], "bold")
            else:
                fill = {"kb": th["bg"], "query": th["tint"], "res": th["soft"]}[n["kind"]]
                edge = th["red"] if n["kind"] == "query" else th["fg"]
                sh.rect(x0, y0, n["w"], n["h"], fill=fill, edge=edge,
                        lw=1.8 if n["kind"] == "query" else 1.1, z=2)
                top_y = n["y"] - 0.095 * (len(n["lines"]) - 1)
                for i, ln in enumerate(n["lines"]):
                    sh.text(n["x"], top_y + 0.19 * i, ln, 11)
            sh.text(x0, y0 - 0.02, n["id"], 7.5, th["muted"], ha="left", va="bottom")
        # Legende
        height = y - 0.8
        ly = height + 0.55
        lx = 0.0
        kinds = [("kb", "Wissensbasis", th["bg"], th["fg"]),
                 ("query", "Negierte Anfrage", th["tint"], th["red"]),
                 ("res", "Resolvent", th["soft"], th["fg"])]
        for key, name, fill, edge in kinds:
            if any(n["kind"] == key for n in nodes):
                sh.rect(lx, ly - 0.09, 0.3, 0.18, fill=fill, edge=edge, lw=1.2)
                sh.text(lx + 0.4, ly, name, 9, th["muted"], ha="left")
                lx += 0.55 + sh.w(name, 9)
        if any(n["empty"] for n in nodes):
            sh.rect(lx, ly - 0.09, 0.3, 0.18, fill=th["red"], edge=th["red"], lw=1.2)
            sh.text(lx + 0.4, ly, "leere Klausel □", 9, th["muted"], ha="left")
            lx += 0.55 + sh.w("leere Klausel □", 9)
        cw = max(xmax - xmin, lx)
        if title is None:
            if derived:
                ttl = "Resolution: leere Klausel □ hergeleitet, also WB ⊨ Q"
            else:
                ttl = "Resolution: keine leere Klausel hergeleitet"
            subs = [f"{sum(1 for n in nodes if n['kind'] == 'res')} Resolvent(en), "
                    f"{len(nodes)} Klauseln insgesamt"]
        else:
            ttl, subs = title, []
        return sh.finish(cw, ly + 0.2, ttl, subs, figsize)


# ==========================================================================
# Binärzahlen
# ==========================================================================

def _bit_pattern(value, width: int, signed: bool) -> tuple[list[int], int]:
    if isinstance(width, bool) or not isinstance(width, int) or not 1 <= width <= MAX_BITS:
        raise ValueError(f"width muss eine ganze Zahl von 1 bis {MAX_BITS} sein")
    if isinstance(value, str):
        s = value.strip().lower().replace("_", "").replace(" ", "")
        if s.startswith("0b"):
            s = s[2:]
        if not s or any(c not in "01" for c in s):
            raise ValueError(f"{value!r} ist kein Bitmuster (nur 0 und 1 erlaubt)")
        if len(s) > width:
            raise ValueError(f"Das Bitmuster {s} hat {len(s)} Stellen, width ist nur {width}")
        bits = [0] * (width - len(s)) + [int(c) for c in s]
        v = sum(b << (width - 1 - i) for i, b in enumerate(bits))
        if signed and bits[0]:
            v -= 1 << width
        return bits, v
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"value muss eine ganze Zahl oder ein Bitmuster-Text sein, nicht {value!r}")
    v = int(value)
    lo, hi = (-(1 << (width - 1)), (1 << (width - 1)) - 1) if signed else (0, (1 << width) - 1)
    if v < 0 and not signed:
        raise ValueError(f"{v} ist negativ — vorzeichenlos geht nur ab 0 (signed=True für Zweierkomplement)")
    if not lo <= v <= hi:
        raise ValueError(f"{v} passt nicht in {width} Bit "
                         f"({'Zweierkomplement' if signed else 'vorzeichenlos'}: {lo} … {hi})")
    pattern = v & ((1 << width) - 1)
    return [(pattern >> (width - 1 - i)) & 1 for i in range(width)], v


def plot_bits(value, width: int = 8, signed: bool = False, *, dark: bool = False,
              figsize=None, title=None):
    """
    Stellenwerttabelle einer Binärzahl: Potenzen 2⁷ … 2⁰, Stellenwerte 128 … 1 und
    die Bits darunter, dazu die Summe. Mit `signed=True` ist das höchste Bit das
    Vorzeichenbit des Zweierkomplements (Wert −2^(width−1), rot markiert).

    `value`: ganze Zahl (Dezimalwert) oder Bitmuster als Text ("10110101", "0b1011").

        plot_bits(181)                       # 10110101 = 181
        plot_bits(-37, width=8, signed=True) # 11011011 = −37
        plot_bits("11111111", signed=True)   # = −1
    """
    bits, v = _bit_pattern(value, width, signed)
    pat = "".join(map(str, bits))
    with _style(dark) as th:
        sh = _Sheet(th)
        weights = [(1 << (width - 1 - i)) for i in range(width)]
        if signed:
            weights[0] = -weights[0]
        cwid = max(0.5, max(sh.w(_group(abs(w_)) + ("−" if w_ < 0 else ""), 10.5) for w_ in weights) + 0.18)
        labw = 1.15
        gap = 0.16
        # bei vielen Bits in Blöcken von 16 (oder 8, wenn die Stellenwerte breit sind) umbrechen
        per = 16 if cwid * 16 + labw + 0.6 <= 12.5 else 8
        per = min(per, width)
        blockh = 1.85 if signed else 1.65
        total = labw
        nblocks = math.ceil(width / per)
        for blk in range(nblocks):
            oy = blk * blockh
            idx = list(range(blk * per, min(width, (blk + 1) * per)))
            xs, x = {}, labw
            for n_, i in enumerate(idx):
                if n_ and n_ % 4 == 0:
                    x += gap
                xs[i] = x
                x += cwid
            total = max(total, x)
            for key, name, dy in (("pow", "Potenz", 0.16), ("val", "Stellenwert", 0.48), ("bit", "Bit", 1.02)):
                sh.text(labw - 0.12, oy + dy, name, 9.5, th["muted"], ha="right")
            for i in idx:
                k = width - 1 - i
                neg = signed and i == 0
                sh.text(xs[i] + cwid / 2, oy + 0.16, ("−" if neg else "") + "2" + _sup(k), 9.5, th["muted"])
                sh.text(xs[i] + cwid / 2, oy + 0.48, ("−" if neg else "") + _group(1 << k), 10.5,
                        th["red"] if neg else th["fg"], "bold" if neg else "normal")
                if bits[i]:
                    fill = th["red"] if neg else th["fg"]
                    sh.rect(xs[i] + 0.03, oy + 0.68, cwid - 0.06, 0.68, fill=fill, z=1)
                    sh.text(xs[i] + cwid / 2, oy + 1.02, "1", 17, th["on_color"] if neg else th["bg"], "bold")
                else:
                    sh.rect(xs[i] + 0.03, oy + 0.68, cwid - 0.06, 0.68, edge=th["red"] if neg else th["grid"],
                            lw=1.4 if neg else 0.9, z=1)
                    sh.text(xs[i] + cwid / 2, oy + 1.02, "0", 17, th["muted"])
            sh.line(labw, oy + 0.68, x, oy + 0.68, lw=0.9)
            if signed and blk == 0:
                sh.text(xs[0] + cwid / 2, oy + 1.5, "Vorzeichen", 8.5, th["red"], "bold")
        # Summe
        terms = [w_ for w_, b in zip(weights, bits) if b]
        if terms:
            def term(w_):
                return ("−" if w_ < 0 else "") + _group(abs(w_))
            expr = term(terms[0])
            for w_ in terms[1:]:
                expr += (" − " + _group(-w_)) if w_ < 0 else (" + " + _group(w_))
            expr += f" = {'−' if v < 0 else ''}{_group(abs(v))}"
        else:
            expr = "0 = 0"
        lines = sh.wrap("Summe der Stellenwerte mit Bit 1:  " + expr, 10.5, max(total, 5.0))
        y = nblocks * blockh
        for ln in lines:
            sh.text(0, y, ln, 10.5, ha="left")
            y += 0.22
        info = [f"Wertebereich bei {width} Bit "
                f"({'Zweierkomplement' if signed else 'vorzeichenlos'}): "
                f"{'−' + _group(1 << (width - 1)) + ' … ' + _group((1 << (width - 1)) - 1) if signed else '0 … ' + _group((1 << width) - 1)}"
                f"  ·  Hex 0x{int(pat, 2):0{(width + 3) // 4}X}"]
        if signed and v < 0:
            mag = -v
            mp_ = [(mag >> (width - 1 - i)) & 1 for i in range(width)]
            inv = [1 - b for b in mp_]
            info.append(f"Negativ darstellen: {_group(mag)} = {''.join(map(str, mp_))}  →  "
                        f"Bits umkehren: {''.join(map(str, inv))}  →  +1: {pat}")
        elif signed:
            info.append("Vorzeichenbit 0: die Zahl ist nicht negativ, es zählen die Stellenwerte wie üblich.")
        for s in info:
            for ln in sh.wrap(s, 9.5, max(total, 5.0)):
                sh.text(0, y + 0.08, ln, 9.5, th["muted"], ha="left")
                y += 0.2
        art = "Zweierkomplement" if signed else "vorzeichenlos"
        ttl = (f"{width}-Bit-Zahl ({art}):  {pat} = {'−' if v < 0 else ''}{_group(abs(v))}"
               if title is None else title)
        return sh.finish(max(total, 5.0), y + 0.1, ttl, figsize=figsize)


# ==========================================================================
# Quantoren
# ==========================================================================

def _truth_list(truth, n: int) -> list[bool]:
    try:
        vals = list(truth)
    except TypeError:
        raise ValueError("truth muss eine Liste von Wahrheitswerten sein") from None
    if len(vals) != n:
        raise ValueError(f"universe hat {n} Elemente, truth aber {len(vals)} Wahrheitswerte")
    out = []
    for t in vals:
        if isinstance(t, (bool, np.bool_)) or (isinstance(t, (int, np.integer)) and int(t) in (0, 1)):
            out.append(bool(t))
        else:
            raise ValueError(f"truth darf nur wahr/falsch (bool oder 0/1) enthalten, nicht {t!r}")
    return out


def plot_quantifier(universe, truth, *, name: str = "P", variables=None,
                    dark: bool = False, figsize=None, title=None):
    """
    Auswertung eines Prädikats über einem Universum. `universe` ist eine Liste von
    Elementen (oder Paaren), `truth` die Liste der Wahrheitswerte P(element).

    Ein Variable: Raster aus grünen (wahr, ✓) und roten (falsch, ✗) Feldern;
    der erste Zeuge für ∃ hat einen durchgezogenen, das erste Gegenbeispiel für ∀
    einen gestrichelten Rahmen. Paare (x, y): Matrix mit Zeilen- und Spaltenauswertung
    (∃y/∀y rechts, ∃x/∀x unten), dazu die vier Quantorenfolgen im Untertitel.

        plot_quantifier([1, 2, 3, 4], [False, True, True, False], name="P")
        plot_quantifier([(x, y) for x in range(3) for y in range(3)],
                        [x < y for x in range(3) for y in range(3)], variables=("x", "y"))
    """
    try:
        uni = list(universe)
    except TypeError:
        raise ValueError("universe muss eine Liste von Elementen sein") from None
    if not uni:
        raise ValueError("Das Universum ist leer — es gibt nichts auszuwerten")
    tv = _truth_list(truth, len(uni))
    keys = [repr(u) for u in uni]
    if len(set(keys)) != len(keys):
        raise ValueError("Das Universum enthält ein Element doppelt")
    pair = all(isinstance(u, (tuple, list)) and len(u) == 2 for u in uni)
    if variables is None:
        variables = ("x", "y") if pair else ("x",)
    variables = tuple(str(v) for v in variables)
    if len(variables) < (2 if pair else 1):
        raise ValueError("variables hat zu wenige Namen")
    pname = str(name)

    with _style(dark) as th:
        sh = _Sheet(th)
        good, bad = th["good"], th["red"]
        on = th["on_color"]
        if not pair:
            if len(uni) > MAX_ELEMENTS_1D:
                raise ValueError(f"Universum zu gross für eine Zeichnung "
                                 f"(mehr als {MAX_ELEMENTS_1D} Elemente)")
            x = variables[0]
            labels = [_label(u) for u in uni]
            cw = max(0.62, max(sh.w(s, 11, "bold") for s in labels) + 0.22)
            ch = 0.72
            per = 10 if len(uni) > 10 else len(uni)
            if cw * per > 11:
                per = max(1, int(11 // cw))
            w_idx = next((i for i, t in enumerate(tv) if t), None)
            c_idx = next((i for i, t in enumerate(tv) if not t), None)
            for i, (s, t) in enumerate(zip(labels, tv)):
                r, c = divmod(i, per)
                x0, y0 = c * (cw + 0.06), r * (ch + 0.08)
                sh.rect(x0, y0, cw, ch, fill=good if t else bad, z=1)
                sh.text(x0 + cw / 2, y0 + 0.26, s, 11, on, "bold")
                sh.text(x0 + cw / 2, y0 + 0.54, "✓" if t else "✗", 12, on, "bold")
                if i == w_idx:
                    sh.rect(x0 - 0.04, y0 - 0.04, cw + 0.08, ch + 0.08, edge=th["fg"], lw=2.4, z=4)
                if i == c_idx:
                    sh.rect(x0 - 0.04, y0 - 0.04, cw + 0.08, ch + 0.08, edge=th["fg"], lw=2.4,
                            ls=(0, (3, 2)), z=4)
            rows_n = math.ceil(len(uni) / per)
            width = per * (cw + 0.06) - 0.06
            ly = rows_n * (ch + 0.08) + 0.2
            allv, anyv = all(tv), any(tv)
            subs = [f"∀{x} {pname}({x}):  {'wahr' if allv else 'falsch'}"
                    + ("" if allv else f"   (Gegenbeispiel {x} = {labels[c_idx]})"),
                    f"∃{x} {pname}({x}):  {'wahr' if anyv else 'falsch'}"
                    + (f"   (Zeuge {x} = {labels[w_idx]})" if anyv else "   (kein Element erfüllt es)"),
                    f"{sum(tv)} von {len(tv)} Elementen erfüllen {pname}({x})"]
            ttl = "Quantoren über dem Universum" if title is None else title
        else:
            xv, yv = variables[0], variables[1]
            xs, ys = [], []
            cell = {}
            for u, t in zip(uni, tv):
                a, b = u
                if _label(a) not in [k for k in xs]:
                    xs.append(_label(a))
                if _label(b) not in ys:
                    ys.append(_label(b))
                cell[(_label(a), _label(b))] = t
            if len(xs) > MAX_MATRIX or len(ys) > MAX_MATRIX:
                raise ValueError(f"Matrix zu gross für eine Zeichnung "
                                 f"(mehr als {MAX_MATRIX} Zeilen oder Spalten)")
            if len(cell) != len(uni):
                raise ValueError("Zwei verschiedene Paare haben dieselbe Beschriftung")
            cs = max(0.58, max(sh.w(s, 10.5, "bold") for s in xs + ys) + 0.2)
            lw_ = max(sh.w(s, 10.5, "bold") for s in xs) + 0.3
            lab_w = max(lw_, sh.w(f"{xv} \\ {yv}", 9) + 0.2)
            x0, y0 = lab_w, 0.55
            sh.text(lab_w - 0.08, 0.28, f"{xv} \\ {yv}", 9, th["muted"], ha="right")
            for j, yl in enumerate(ys):
                sh.text(x0 + j * cs + cs / 2, 0.28, yl, 10.5, weight="bold")
            row_ex, row_all = [], []
            w_cell = c_cell = None
            for i, xl in enumerate(xs):
                sh.text(lab_w - 0.1, y0 + i * cs + cs / 2, xl, 10.5, weight="bold", ha="right")
                vals = []
                for j, yl in enumerate(ys):
                    t = cell.get((xl, yl))
                    cx, cy = x0 + j * cs, y0 + i * cs
                    if t is None:
                        sh.rect(cx + 0.02, cy + 0.02, cs - 0.04, cs - 0.04, edge=th["muted"],
                                lw=0.8, ls=":", z=1)
                        sh.text(cx + cs / 2, cy + cs / 2, "–", 10, th["muted"])
                        continue
                    vals.append(t)
                    sh.rect(cx + 0.02, cy + 0.02, cs - 0.04, cs - 0.04, fill=good if t else bad, z=1)
                    sh.text(cx + cs / 2, cy + cs / 2, "✓" if t else "✗", 13, on, "bold")
                    if t and w_cell is None:
                        w_cell = (cx, cy)
                    if not t and c_cell is None:
                        c_cell = (cx, cy)
                row_ex.append(any(vals))
                row_all.append(all(vals))
            col_ex, col_all = [], []
            for j, yl in enumerate(ys):
                vals = [cell[(xl, yl)] for xl in xs if (xl, yl) in cell]
                col_ex.append(any(vals))
                col_all.append(all(vals))
            gx = x0 + len(ys) * cs + 0.15
            gy = y0 + len(xs) * cs + 0.15

            def badge(bx, by, ok, mark_text):
                sh.rect(bx + 0.05, by + 0.05, cs - 0.1, cs - 0.1, edge=good if ok else bad, lw=1.4, z=1)
                sh.text(bx + cs / 2, by + cs / 2, "✓" if ok else "✗", 12, good if ok else bad, "bold")

            sh.text(gx + cs / 2, 0.28, f"∃{yv}", 10.5, th["muted"], "bold")
            sh.text(gx + cs * 1.5, 0.28, f"∀{yv}", 10.5, th["muted"], "bold")
            for i in range(len(xs)):
                badge(gx, y0 + i * cs, row_ex[i], "")
                badge(gx + cs, y0 + i * cs, row_all[i], "")
            for j in range(len(ys)):
                badge(x0 + j * cs, gy, col_ex[j], "")
                badge(x0 + j * cs, gy + cs, col_all[j], "")
            sh.text(lab_w - 0.1, gy + cs / 2, f"∃{xv}", 10.5, th["muted"], "bold", ha="right")
            sh.text(lab_w - 0.1, gy + cs * 1.5, f"∀{xv}", 10.5, th["muted"], "bold", ha="right")
            if w_cell:
                sh.rect(w_cell[0] - 0.02, w_cell[1] - 0.02, cs + 0.0, cs + 0.0, edge=th["fg"], lw=2.4, z=4)
            if c_cell:
                sh.rect(c_cell[0] - 0.02, c_cell[1] - 0.02, cs + 0.0, cs + 0.0, edge=th["fg"], lw=2.4,
                        ls=(0, (3, 2)), z=4)
            width = gx + 2 * cs
            ly = gy + 2 * cs + 0.25
            allv, anyv = all(tv), any(tv)
            ae = all(row_ex)
            ea = any(row_all)
            fmt_b = lambda b: "wahr" if b else "falsch"  # noqa: E731
            subs = [f"∀{xv}∀{yv} {pname}({xv},{yv}):  {fmt_b(allv)}      "
                    f"∃{xv}∃{yv} {pname}({xv},{yv}):  {fmt_b(anyv)}",
                    f"∀{xv}∃{yv} {pname}({xv},{yv}):  {fmt_b(ae)}      "
                    f"∃{xv}∀{yv} {pname}({xv},{yv}):  {fmt_b(ea)}",
                    f"{sum(tv)} von {len(tv)} Paaren erfüllen {pname}"]
            ttl = "Quantoren über Paaren: Reihenfolge zählt" if title is None else title
        # Legende
        lx = 0.0
        sh.rect(lx, ly - 0.09, 0.3, 0.18, fill=good, z=1)
        sh.text(lx + 0.4, ly, "wahr", 9, th["muted"], ha="left")
        lx += 0.9
        sh.rect(lx, ly - 0.09, 0.3, 0.18, fill=bad, z=1)
        sh.text(lx + 0.4, ly, "falsch", 9, th["muted"], ha="left")
        lx += 1.0
        sh.rect(lx, ly - 0.09, 0.3, 0.18, edge=th["fg"], lw=2.0)
        sh.text(lx + 0.4, ly, "Zeuge für ∃", 9, th["muted"], ha="left")
        lx += 0.55 + sh.w("Zeuge für ∃", 9)
        sh.rect(lx, ly - 0.09, 0.3, 0.18, edge=th["fg"], lw=2.0, ls=(0, (3, 2)))
        sh.text(lx + 0.4, ly, "Gegenbeispiel für ∀", 9, th["muted"], ha="left")
        lx += 0.55 + sh.w("Gegenbeispiel für ∀", 9)
        return sh.finish(max(width, lx), ly + 0.2, ttl, subs, figsize)


# ==========================================================================
# Sicherer Formelparser für Rechenschritte in n
# ==========================================================================

_N = sp.Symbol("n")
_ALLOWED_NAMES = "n, log (log(n, 2) = Basis 2), ln, sqrt, factorial, exp, floor, ceiling, Zahlen"
_LOCALS = {"n": _N, "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt, "factorial": sp.factorial,
           "exp": sp.exp, "floor": sp.floor, "ceiling": sp.ceiling}
# parse_expr wertet intern aus — darum nur diese Namen, kein __builtins__, keine Attribute.
_GLOBALS = {"__builtins__": {}, "Integer": sp.Integer, "Float": sp.Float,
            "Rational": sp.Rational, "Symbol": sp.Symbol, "Function": sp.Function,
            "Mul": sp.Mul, "Add": sp.Add, "Pow": sp.Pow}
_UNSAFE = re.compile(r"[^\w\s+\-*/^(),.]|_|(?<=[^\W\d])\.|\.(?!\d)")
_SUPERS = {"²": "^2", "³": "^3", "⁴": "^4"}


def _compile_expr(text) -> sp.Basic:
    """Text in n -> SymPy-Term (unausgewertet). ValueError bei allem, was nicht erlaubt ist."""
    from sympy.parsing.sympy_parser import (convert_xor, implicit_application,
                                            implicit_multiplication, parse_expr,
                                            standard_transformations)
    if isinstance(text, bool) or text is None:
        raise ValueError(f"Keine Formel in n: {text!r}")
    s = str(text).strip()
    if not s:
        raise ValueError("Leere Formel — erwartet z. B. 'n**2' oder 'n*log(n,2)'")
    if len(s) > MAX_EXPR_LEN:
        raise ValueError(f"Formel zu lang (mehr als {MAX_EXPR_LEN} Zeichen)")
    for a, b in _SUPERS.items():
        s = s.replace(a, b)
    s = s.replace("−", "-").replace("·", "*").replace("×", "*")
    if _UNSAFE.search(s):
        raise ValueError(f"Unzulässige Zeichen in der Formel {text!r}. Erlaubt sind: "
                         f"{_ALLOWED_NAMES}, Operatoren + - * / ** ^ und Klammern")
    if re.search(r"\d{13,}", s):
        raise ValueError("Zahl zu gross in der Formel (höchstens 12 Stellen)")
    try:
        e = parse_expr(s, local_dict=dict(_LOCALS), global_dict=dict(_GLOBALS),
                       transformations=standard_transformations + (
                           implicit_multiplication, implicit_application, convert_xor),
                       evaluate=False)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Die Formel {text!r} lässt sich nicht lesen ({type(exc).__name__}). "
                         f"Erlaubt sind: {_ALLOWED_NAMES}") from None
    if not isinstance(e, sp.Basic):
        raise ValueError(f"Die Formel {text!r} ist kein Term in n")
    if sum(1 for _ in sp.preorder_traversal(e)) > MAX_EXPR_NODES:
        raise ValueError("Formel zu verschachtelt")
    from sympy.core.function import AppliedUndef
    unknown = sorted({str(sy) for sy in e.free_symbols if sy != _N}
                     | {type(a).__name__ for a in e.atoms(AppliedUndef)})
    if unknown:
        raise ValueError(f"Unbekannter Name {unknown[0]!r} in der Formel {text!r}. "
                         f"Erlaubt sind: {_ALLOWED_NAMES}")
    return e


# ponytail: mpmath rechnet mit der globalen Standardgenauigkeit (15 Stellen) — fürs Zeichnen genug
def _pow(b, x):
    if isinstance(b, mp.mpf) and isinstance(x, mp.mpf) and b > 0 and b != 1 and x > 0:
        if x * mp.log10(b) > 1e6:        # grösser als 10^1'000'000: nicht mehr ausrechnen
            return mp.inf
    return b ** x


def _ev(e, n):
    """Wertet einen erlaubten SymPy-Term mit mpmath aus (keine Python-Riesenzahlen)."""
    if e.is_Symbol:
        return n
    if e.is_Integer:
        return mp.mpf(int(e))
    if e.is_Rational:
        return mp.mpf(int(e.p)) / int(e.q)
    if e.is_Number:
        return mp.mpf(str(e))
    if e.is_Add:
        total = mp.mpf(0)
        for a in e.args:
            total += _ev(a, n)
        return total
    if e.is_Mul:
        prod = mp.mpf(1)
        for a in e.args:
            prod *= _ev(a, n)
        return prod
    if e.is_Pow:
        return _pow(_ev(e.base, n), _ev(e.exp, n))
    args = [_ev(a, n) for a in e.args]
    fn = e.func
    if fn == sp.log:
        return mp.log(args[0]) if len(args) == 1 else mp.log(args[0]) / mp.log(args[1])
    if fn == sp.exp:
        return mp.exp(args[0]) if args[0] < 1e7 else mp.inf
    if fn == sp.factorial:
        return mp.factorial(args[0]) if args[0] < 1e13 else mp.inf
    if fn == sp.floor:
        return mp.floor(args[0])
    if fn == sp.ceiling:
        return mp.ceil(args[0])
    raise ValueError(f"Nicht erlaubte Funktion {fn.__name__}")


def _pretty_expr(s: str) -> str:
    s = str(s).replace("**", "^").replace("*", "·")
    s = re.sub(r"\^\(?(-?\d+)\)?", lambda m: _sup(int(m.group(1))), s)
    return s


def _val_lg(v) -> tuple[float, float]:
    """Wert -> (Zahl als float oder inf, log10 oder nan). Nichtpositives hat keinen log10."""
    try:
        if isinstance(v, (bool, np.bool_)):
            v = int(v)
        if isinstance(v, (int, np.integer)):
            v = int(v)
            if v <= 0:
                return float(v), float("nan")
            lg = math.log10(v)
            return (float(v) if lg < 300 else float("inf")), lg
        if not isinstance(v, (mp.mpf, mp.mpc)):
            v = mp.mpf(float(v))
        if isinstance(v, mp.mpc):
            if abs(v.imag) > 1e-9 * (1 + abs(v.real)):
                return float("nan"), float("nan")
            v = v.real
        if mp.isnan(v):
            return float("nan"), float("nan")
        if mp.isinf(v):
            return (float("inf"), float("inf")) if v > 0 else (float("-inf"), float("nan"))
        if v <= 0:
            return float(v), float("nan")
        lg = float(mp.log10(v))
        return (float(v) if lg < 300 else float("inf")), lg
    except Exception:
        return float("nan"), float("nan")


class _Fn:
    """Eine Rechenschrittzahl in n: Formeltext (sicher geparst) oder Callable."""

    def __init__(self, f, label: str | None = None):
        if callable(f) and not isinstance(f, str):
            self.expr, self.call = None, f
            self.label = label or getattr(f, "__name__", "f")
            if self.label == "<lambda>":
                self.label = "f(n)"
        else:
            self.expr, self.call = _compile_expr(f), None
            self.label = label or _pretty_expr(str(f))

    @property
    def is_callable(self) -> bool:
        return self.call is not None

    def __call__(self, n) -> tuple[float, float]:
        try:
            if self.expr is not None:
                return _val_lg(_ev(self.expr, mp.mpf(n)))
            return _val_lg(self.call(int(n) if float(n) == int(n) else float(n)))
        except Exception:  # noqa: BLE001 — log(0), 0^-1, Überlauf …: einfach kein Punkt
            return float("nan"), float("nan")


def _funcs_of(funcs) -> list[_Fn]:
    if isinstance(funcs, str) or (callable(funcs) and not isinstance(funcs, Mapping)):
        items = [(None, funcs)]
    elif isinstance(funcs, Mapping):
        items = list(funcs.items())
    elif isinstance(funcs, (list, tuple)):
        items = [(None, f) for f in funcs]
    else:
        raise ValueError("funcs muss ein dict Name → Formel (oder eine Liste von Formeln) sein")
    if not items:
        raise ValueError("funcs ist leer — mindestens eine Funktion wird gebraucht")
    if len(items) > MAX_FUNCS:
        raise ValueError(f"Zu viele Funktionen ({len(items)}); höchstens {MAX_FUNCS}")
    out = []
    for name, f in items:
        if name is not None and not isinstance(name, str):
            raise ValueError(f"Der Name {name!r} muss ein Text sein")
        out.append(_Fn(f, name))
    return out


def _n_max_of(n_max, fns, default=20) -> int:
    if n_max is None:
        return default
    if isinstance(n_max, bool) or not isinstance(n_max, (int, float, np.integer, np.floating)) \
            or not math.isfinite(n_max) or n_max != int(n_max):
        raise ValueError(f"n_max muss eine ganze Zahl sein, nicht {n_max!r}")
    n_max = int(n_max)
    cap = 10 ** 5 if any(f.is_callable for f in fns) else 10 ** 9
    if not 2 <= n_max <= cap:
        raise ValueError(f"n_max muss zwischen 2 und {_group(cap)} liegen")
    return n_max


def _limit_of(mark_limit):
    if mark_limit is None:
        return None
    if isinstance(mark_limit, bool) or not isinstance(mark_limit, (int, float, np.integer, np.floating)) \
            or not math.isfinite(mark_limit) or mark_limit <= 0:
        raise ValueError("mark_limit muss eine positive Zahl sein (z. B. 6e10 Schritte pro Minute)")
    return float(mark_limit) if not isinstance(mark_limit, int) else mark_limit


def _ngrid(n_max: int) -> list[int]:
    if n_max <= 400:
        return list(range(1, n_max + 1))
    # ponytail: ab 400 nur noch ~300 logarithmisch verteilte Stützstellen statt jedes n
    pts = {max(1, int(round(v))) for v in np.geomspace(1, n_max, 300)}
    pts.add(n_max)
    return sorted(pts)


class _Clock:
    def __init__(self, seconds: float = MAX_TIME):
        self.end = time.monotonic() + seconds

    def check(self):
        if time.monotonic() > self.end:
            raise ValueError("Die Auswertung dauert zu lange — kleineres n_max oder einfachere "
                             "Funktionen wählen")


def _sample(fns: list[_Fn], ns: list[int]):
    clock = _Clock()
    vals = np.full((len(fns), len(ns)), np.nan)
    lgs = np.full((len(fns), len(ns)), np.nan)
    for i, f in enumerate(fns):
        for j, n in enumerate(ns):
            vals[i, j], lgs[i, j] = f(n)
            if j % 16 == 0:
                clock.check()
    return vals, lgs


def _max_n(f: _Fn, limit: float):
    """
    Grösstes ganzes n mit f(n) ≤ limit. Rückgabe (n, Obergrenze_erreicht);
    n = 0 heisst: schon f(1) ist zu gross.
    """
    # ponytail: Verdoppeln + Bisektion setzt voraus, dass f ab n = 1 nicht mehr fällt
    cap = 10 ** 5 if f.is_callable else 10 ** 15
    clock = _Clock()

    def ok(n):
        clock.check()
        v, _lg = f(n)
        return (not math.isnan(v)) and v <= limit

    if not ok(1):
        return 0, False
    lo, hi = 1, 2
    while hi <= cap and ok(hi):
        lo, hi = hi, hi * 2
    if hi > cap:
        if ok(cap):
            return cap, True
        hi = cap
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo, False


def _limit_labels(fns, limit):
    out = []
    for f in fns:
        n, capped = _max_n(f, limit)
        out.append((n, capped))
    return out


def _tick_pow10(k: float) -> str:
    k = int(round(k))
    return "1" if k == 0 else ("10" if k == 1 else "10" + _sup(k))


def _growth_data(funcs, n_max, mark_limit):
    fns = _funcs_of(funcs)
    n_max = _n_max_of(n_max, fns)
    limit = _limit_of(mark_limit)
    ns = _ngrid(n_max)           # ponytail: Standard n_max = 20 zeigt den Unterschied der Klassen gut
    vals, lgs = _sample(fns, ns)
    maxes = _limit_labels(fns, limit) if limit is not None else None
    return fns, n_max, limit, ns, vals, lgs, maxes


def _legend_label(f: _Fn, mx) -> str:
    if mx is None:
        return f.label
    n, capped = mx
    if n == 0:
        return f"{f.label}   (schon n = 1 zu viel)"
    return f"{f.label}   (n ≤ {_sci(n)}{' und mehr' if capped else ''})"


def plot_growth(funcs, n_max=None, log: bool = False, mark_limit=None, *, dark: bool = False,
                figsize=None, title=None):
    """
    Wachstum von Rechenschrittzahlen über der Eingabegrösse n.

    `funcs`: dict Name → Formel in n als Text (`"n*log(n,2)"`, `"n**2"`, `"2^n"`,
    `"factorial(n)"`) oder Callable. Erlaubte Namen: n, log (log(n, 2) zur Basis 2),
    ln, sqrt, factorial, exp, floor, ceiling und Zahlen — alles andere ist ein
    ValueError. Es wird nichts mit eval ausgeführt.

    `n_max` (Standard 20) ist das grösste gezeichnete n. `log=True` zeichnet eine
    logarithmische y-Achse (Zehnerpotenzen, geht auch bei 2ⁿ oder n! über den
    Zahlbereich hinaus). `mark_limit` zeichnet eine rote Linie (z. B. 6e10
    Schritte pro Minute), markiert die Schnittpunkte und nennt in der Legende das
    grösste n, das noch unter der Linie bleibt.

        plot_growth({"n": "n", "n log n": "n*log(n,2)", "n²": "n**2", "2ⁿ": "2**n"},
                    n_max=30, log=True)
        plot_growth({"n²": "n**2", "n log n": "n*log(n,2)"}, n_max=100, mark_limit=6e10)
    """
    fns, n_max, limit, ns, vals, lgs, maxes = _growth_data(funcs, n_max, mark_limit)
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    with _style(dark) as th:
        fig = _new_figure(_check_size(figsize) or (9.6, 5.2), th, layout="constrained")
        ax = fig.add_subplot()
        _axes_style(ax, th)
        xs = np.array(ns, dtype=float)
        series = th["series"]
        y_all = lgs if log else vals
        y_plot = np.where(np.isfinite(y_all), y_all, np.nan)
        for i, f in enumerate(fns):
            color = series[i % len(series)]
            ls = _STYLES[(i // len(series)) % len(_STYLES)] if i >= len(series) else _STYLES[0]
            if i >= len(series):
                ls = _STYLES[1 + (i // len(series) - 1) % 3]
            ax.plot(xs, y_plot[i], color=color, linestyle=ls, linewidth=2.0,
                    marker="o" if len(ns) <= 30 else None, markersize=3.5,
                    label=_legend_label(f, maxes[i] if maxes else None), zorder=3)
        ax.set_xlim(1 if n_max > 1 else 0, n_max)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
        ax.set_xlabel("n  (Eingabegrösse)")
        # y-Achse
        clipped = False
        if log:
            finite = lgs[np.isfinite(lgs)]
            lo = min(0.0, float(finite.min())) if finite.size else 0.0
            hi = float(finite.max()) if finite.size else 1.0
            if limit is not None:
                hi = max(hi, math.log10(limit) + 0.4)
            lo, hi = math.floor(lo), math.ceil(hi + 0.05)
            hi = max(hi, lo + 1)
            ax.set_ylim(lo - 0.05, hi + 0.05)
            step = max(1, math.ceil((hi - lo) / 8))
            ax.set_yticks(list(range(lo, hi + 1, step)))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: _tick_pow10(v)))
            ax.set_ylabel("Rechenschritte  (log₁₀-Achse)")
        else:
            finals = sorted(float(v) for v in vals[:, -1] if math.isfinite(v) and v > 0)
            if limit is not None:
                top = limit * 1.25 if (not finals or finals[-1] > limit) else finals[-1] * 1.1
            elif finals:
                idx = max(k for k, v in enumerate(finals) if v <= 30 * finals[0])
                top = finals[idx] * 1.1
            else:
                top = 1.0
            top = max(top, 1.0)
            clipped = bool(finals) and finals[-1] > top
            ax.set_ylim(0, top)
            for _ln in ax.lines:                  # n! und 2^n: weit ausserhalb kappen (Raster-Überlauf)
                _y = np.asarray(_ln.get_ydata(), dtype=float)
                if _y.size and np.nanmax(_y) > top * 50:
                    _ln.set_ydata(np.minimum(_y, top * 50))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: _sci(v)))
            ax.set_ylabel("Rechenschritte")
            if clipped:
                ax.text(0.995, 0.015, "Steile Kurven verlassen das Bild — log=True zeigt alles",
                        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                        color=th["muted"])
        if limit is not None:
            ly = math.log10(limit) if log else limit
            ax.axhline(ly, color=th["red"], linewidth=1.3, linestyle="--", zorder=2)
            ax.text(0.005, ly, f" Limit {_sci(limit)} Schritte", transform=ax.get_yaxis_transform(),
                    ha="left", va="bottom", fontsize=9, color=th["red"])
            for i, f in enumerate(fns):
                n_star = maxes[i][0]
                if 1 <= n_star <= n_max:
                    v, lg = f(n_star)
                    yv = lg if log else v
                    yv = min(yv, ax.get_ylim()[1])      # sonst Raster-Überlauf bei n! und 2^n
                    if math.isfinite(yv):
                        color = series[i % len(series)]
                        ax.plot([n_star, n_star], [ax.get_ylim()[0], yv], color=color,
                                linewidth=0.8, linestyle=":", zorder=2)
                        ax.scatter([n_star], [yv], s=42, color=color, edgecolor=th["red"],
                                   linewidth=1.4, zorder=5)
        fig.legend(*ax.get_legend_handles_labels(), loc="outside right upper", fontsize=9.5)
        ax.set_title(_esc("Wachstum der Rechenschritte" if title is None else title),
                     loc="left", fontsize=12, fontweight="bold")
        return fig


def _axes_style(ax, th):
    ax.grid(True, color=th["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3)


def plot_steps(ns, steps, fit=None, *, log: bool = False, dark: bool = False, figsize=None,
               title=None):
    """
    Gemessene Rechenschritte über n als Punkte, optional mit einer angepassten
    Formel in n (Text wie bei `plot_growth`, oder Callable) als roter Linie.

        plot_steps([1, 2, 4, 8, 16], [3, 7, 15, 31, 63], fit="4*n - 1")
    """
    try:
        xs = [float(v) for v in ns]
        ys = [float(v) for v in steps]
    except (TypeError, ValueError):
        raise ValueError("ns und steps müssen Listen von Zahlen sein") from None
    if not xs:
        raise ValueError("ns ist leer — mindestens ein Messpunkt wird gebraucht")
    if len(xs) != len(ys):
        raise ValueError(f"ns hat {len(xs)} Werte, steps aber {len(ys)}")
    if len(xs) > 5000:
        raise ValueError("Zu viele Messpunkte (mehr als 5000)")
    if not all(math.isfinite(v) for v in xs + ys):
        raise ValueError("ns und steps müssen endliche Zahlen sein")
    fn = _Fn(fit) if fit is not None else None
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    with _style(dark) as th:
        fig = _new_figure(_check_size(figsize) or (8.4, 5.0), th, layout="constrained")
        ax = fig.add_subplot()
        _axes_style(ax, th)
        lo, hi = min(xs), max(xs)
        span = (hi - lo) or 1.0
        if fn is not None:
            grid = np.linspace(lo, hi, 200) if hi > lo else np.array([lo - 1, lo + 1])
            clock = _Clock()
            gv = np.array([fn(float(g))[0] for g in grid], dtype=float)
            clock.check()
            gv = np.where(np.isfinite(gv), gv, np.nan)
            if not np.isfinite(gv).any():
                raise ValueError("Die Anpassung lässt sich für diese n nicht auswerten")
            ax.plot(grid, gv, color=th["red"], linewidth=2.0, zorder=2,
                    label="Anpassung  f(n) = " + fn.label)
        ax.scatter(xs, ys, s=38, marker="s", color=th["fg"], zorder=4, label="Messung")
        ax.set_xlim(lo - 0.05 * span, hi + 0.05 * span)
        ax.xaxis.set_major_locator(MaxNLocator(integer=all(v == int(v) for v in xs), nbins=10))
        ax.set_xlabel("n  (Eingabegrösse)")
        ax.set_ylabel("Rechenschritte")
        if log and all(y > 0 for y in ys):
            ax.set_yscale("log")
        else:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: _sci(v)))
            if min(ys) >= 0:
                ax.set_ylim(bottom=0)
        ax.legend(loc="upper left", fontsize=9.5)
        ax.set_title(_esc("Gemessene Rechenschritte" if title is None else title), loc="left",
                     fontsize=12, fontweight="bold")
        return fig


def plot_halving(x, *, dark: bool = False, figsize=None, title=None):
    """
    Halbierungs-Schleife `while x > 0: x = x // 2` als Balken pro Durchlauf (Wert von
    x zu Beginn des Durchlaufs). Es sind ⌊log₂ x⌋ + 1 Durchläufe, danach ist x = 0.

        plot_halving(100)      # 7 Durchläufe: 100, 50, 25, 12, 6, 3, 1
    """
    if isinstance(x, bool) or not isinstance(x, (int, np.integer)):
        raise ValueError(f"x muss eine ganze Zahl sein, nicht {x!r}")
    x = int(x)
    if x < 1:
        raise ValueError("x muss mindestens 1 sein (für x = 0 läuft die Schleife nie)")
    if x.bit_length() > 64:
        raise ValueError("x ist zu gross (höchstens 64 Bit)")
    from matplotlib.ticker import FuncFormatter
    seq = []
    cur = x
    while cur > 0:
        seq.append(cur)
        cur //= 2
    k = len(seq)
    assert k == x.bit_length()          # = ⌊log₂ x⌋ + 1
    with _style(dark) as th:
        fig = _new_figure(_check_size(figsize) or (max(6.4, 0.75 * k + 3.0), 4.8), th,
                          layout="constrained")
        ax = fig.add_subplot()
        _axes_style(ax, th)
        pos = np.arange(1, k + 1)
        bars = ax.bar(pos, [float(v) for v in seq], width=0.68, color=th["fg"], zorder=3)
        bars[-1].set_color(th["red"])
        top = float(max(seq))
        for p, v in zip(pos, seq):
            ax.text(p, v + top * 0.015, _sci(v) if v >= 10 ** 7 else str(v), ha="center",
                    va="bottom", fontsize=10, fontweight="bold")
        ax.text(k + 1, top * 0.015, "x = 0\nStopp", ha="center", va="bottom", fontsize=9.5,
                color=th["muted"])
        ax.set_xlim(0.4, k + 1.6)
        ax.set_ylim(0, top * 1.14)
        ax.set_xticks(list(pos))
        if x.bit_length() <= 10:
            ax.set_xticklabels([f"{i}\n{bin(v)[2:]}" for i, v in zip(pos, seq)], fontsize=9)
        ax.set_xlabel("Durchlauf" + ("   (darunter: x binär — jede Halbierung streicht das letzte Bit)"
                                     if x.bit_length() <= 10 else ""))
        ax.set_ylabel("x zu Beginn des Durchlaufs")
        if x.bit_length() > 6:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: _sci(v)))
        ttl = (f"x = {x}:  ⌊log₂ {x}⌋ + 1 = {k} Durchläufe" if title is None else title)
        ax.set_title(_esc(ttl), loc="left", fontsize=12, fontweight="bold")
        return fig


def plot_recursion_tree(root, *, values: bool = False, dark: bool = False, figsize=None, title=None):
    """
    Rekursionsbaum: jeder Aufruf ein Knoten, die rekursiven Aufrufe darunter (Serie 6).

    `root`: ein Objekt mit `.label` und `.children` — in der Regel `recursion_tree(...).root`
    (jeder Knoten hat zusätzlich `.value`, die Rückgabe). Mit `values=True` steht an jedem Knoten
    »Aufruf → Rückgabe«.

        plot_recursion_tree(recursion_tree("summe", 5).root)
    """
    if root is None or not hasattr(root, "label"):
        raise ValueError("plot_recursion_tree: erwartet die Wurzel eines Rekursionsbaums (Objekt mit .label)")

    class _N:
        def __init__(self, node):
            self.label = (f"{node.label} → {node.value}" if values and getattr(node, "value", None) is not None
                          else str(node.label))
            self.children = [_N(k) for k in (getattr(node, "children", None) or ())]

    proxy = _N(root)
    n = 0
    stack = [proxy]
    while stack:
        cur = stack.pop()
        n += 1
        stack.extend(cur.children)
    auto = f"Rekursionsbaum  ({n} Aufrufe)"
    return plot_syntax_tree(proxy, dark=dark, figsize=figsize, title=auto if title is None else title)


# ==========================================================================
# Plotly (interaktiv)
# ==========================================================================

def has_plotly() -> bool:
    try:
        import plotly.graph_objects  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


_PLOTLY_HINT = ("Plotly ist nicht installiert — die interaktive Zeichnung braucht es.\n"
                "Installation:  pip install plotly\n"
                "(Die Matplotlib-Zeichnungen wie plot_growth funktionieren auch ohne.)")


def plotly_growth(funcs, n_max=None, log: bool = False, mark_limit=None, *, dark: bool = False,
                  title=None):
    """
    Interaktive Wachstumszeichnung (zoomen, Werte mit der Maus ablesen); gleiche
    Eingaben wie `plot_growth`. Wirft RuntimeError, wenn Plotly fehlt.

        plotly_growth({"n²": "n**2", "2ⁿ": "2**n"}, n_max=30, log=True)
    """
    if not has_plotly():
        raise RuntimeError(_PLOTLY_HINT)
    import plotly.graph_objects as go
    fns, n_max, limit, ns, vals, lgs, maxes = _growth_data(funcs, n_max, mark_limit)
    th = _theme(dark)
    fig = go.Figure()
    for i, f in enumerate(fns):
        y = lgs[i] if log else vals[i]
        y = np.where(np.isfinite(y), y, np.nan)
        color = th["series"][i % len(th["series"])]
        shown = [_sci(v) if math.isfinite(v) else
                 (f"10^{l:.1f}" if math.isfinite(l) else "?")
                 for v, l in zip(vals[i], lgs[i])]
        fig.add_trace(go.Scatter(
            x=ns, y=y, mode="lines+markers" if len(ns) <= 30 else "lines",
            name=_legend_label(f, maxes[i] if maxes else None),
            line=dict(color=color, width=2.4, dash=["solid", "dash", "dashdot", "dot"][
                (i // len(th["series"])) % 4]),
            customdata=shown,
            hovertemplate="n = %{x}<br>" + _esc(f.label).replace("<", "&lt;") + " = %{customdata}<extra></extra>"))
    axis = dict(gridcolor=th["grid"], linecolor=th["fg"], zerolinecolor=th["fg"])
    yaxis = dict(title="Rechenschritte" + ("  (log₁₀)" if log else ""), **axis)
    if log:
        finite = lgs[np.isfinite(lgs)]
        top = int(math.ceil(float(finite.max()))) if finite.size else 1
        if limit is not None:
            top = max(top, int(math.ceil(math.log10(limit))))
        lo = int(math.floor(min(0.0, float(finite.min())))) if finite.size else 0
        step = max(1, math.ceil((top - lo) / 8))
        ticks = list(range(lo, top + 1, step))
        yaxis.update(tickvals=ticks, ticktext=[_tick_pow10(t) for t in ticks])
    if limit is not None:
        fig.add_hline(y=math.log10(limit) if log else limit, line=dict(color=th["red"], dash="dash"),
                      annotation_text=f"Limit {_sci(limit)}")
        for i, f in enumerate(fns):
            n_star = maxes[i][0]
            if 1 <= n_star <= n_max:
                v, lg = f(n_star)
                fig.add_trace(go.Scatter(x=[n_star], y=[lg if log else v], mode="markers",
                                         showlegend=False, marker=dict(size=10, color=th["red"]),
                                         hovertemplate=f"größtes n = {n_star}<extra></extra>"))
    fig.update_layout(
        title="Wachstum der Rechenschritte" if title is None else title,
        template="plotly_dark" if dark else "simple_white", paper_bgcolor=th["bg"],
        plot_bgcolor=th["bg"], font=dict(family="Helvetica Neue, Helvetica, Arial, sans-serif",
                                         color=th["fg"]),
        xaxis=dict(title="n  (Eingabegrösse)", **axis), yaxis=yaxis, height=520,
        margin=dict(l=10, r=10, t=50, b=10))
    return fig
