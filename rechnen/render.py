#!/usr/bin/env python3
"""
Darstellung: LaTeX für Jupyter, Text für die Konsole.

Jedes Ergebnis von dmatkit ist ein Objekt, das sich selbst anzeigen kann.
Im Notebook erscheint es gesetzt, im Terminal als lesbarer Text — dieselbe
Rechnung, zwei Ausgaben.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

import sympy as sp

from .core import Formula, fmt, tex

# ==========================================================================
# LaTeX -> lesbarer Text (HTML-Ansicht ohne MathJax)
# ==========================================================================

_TEX_REPLACEMENTS = [
    (r"\left", ""), (r"\right", ""), (r"\cdot", "\u00b7"), (r"\infty", "\u221e"),
    (r"\pi", "\u03c0"), (r"\to", "\u2192"), (r"\Rightarrow", "\u21d2"),
    (r"\leq", "\u2264"), (r"\geq", "\u2265"), (r"\le", "\u2264"), (r"\ge", "\u2265"),
    (r"\neq", "\u2260"), (r"\pm", "\u00b1"), (r"\in", "\u2208"), (r"\mathbb{R}", "\u211d"),
    (r"\Delta", "\u0394"), (r"\delta", "\u03b4"), (r"\varepsilon", "\u03b5"),
    (r"\epsilon", "\u03b5"), (r"\lim", "lim"), (r"\ln", "ln"), (r"\log", "log"),
    (r"\sin", "sin"), (r"\cos", "cos"), (r"\tan", "tan"), (r"\exp", "exp"),
    (r"\operatorname{sign}", "sgn"), (r"\operatorname{atan}", "arctan"),
    (r"\quad", "   "), (r"\qquad", "     "), (r"\;", " "), (r"\,", " "), (r"\!", ""),
    (r"\circ", "\u2218"), (r"\forall", "\u2200"), (r"\exists", "\u2203"),
    (r"\wedge", "\u2227"), (r"\lor", "\u2228"), (r"\land", "\u2227"),
    (r"\mid", "|"), (r"\vert", "|"), (r"\\", "   "),
    (r"\neg", "\u00ac"), (r"\vee", "\u2228"), (r"\oplus", "\u2295"),
    (r"\Leftrightarrow", "\u21d4"), (r"\leftrightarrow", "\u2194"),
    (r"\rightarrow", "\u2192"), (r"\models", "\u22a8"), (r"\vdash", "\u22a2"),
    (r"\nvDash", "\u22ad"), (r"\top", "\u22a4"), (r"\bot", "\u22a5"),
    (r"\equiv", "\u2261"), (r"\emptyset", "\u2205"), (r"\subseteq", "\u2286"),
    (r"\cup", "\u222a"), (r"\cap", "\u2229"), (r"\Theta", "\u0398"),
    (r"\Omega", "\u03a9"), (r"\lfloor", "\u230a"), (r"\rfloor", "\u230b"),
    (r"\lceil", "\u2308"), (r"\rceil", "\u2309"), (r"\times", "\u00d7"),
    (r"\ne", "\u2260"), (r"\approx", "\u2248"), (r"\sum", "\u03a3"),
    (r"\binom", "C"), (r"\mathcal{O}", "O"),
]

_SUB = str.maketrans("0123456789", "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089")
_SUP = str.maketrans("0123456789-n", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207b\u207f")


def _group(s: str, i: int):
    """Liest {...} ab Position i (s[i] == '{'); gibt (Inhalt, Position danach)."""
    depth, j = 0, i
    while j < len(s):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    return s[i + 1:], len(s)


def _wrap(t: str) -> str:
    t = t.strip()
    return t if re.fullmatch(r"[\w.\u00b2\u00b3\u00b9\u221e]+", t) else f"({t})"


def _expand(s: str) -> str:
    """Löst \\frac, \\sqrt, \\ln{..} u. ä. mit verschachtelten Klammern auf."""
    out, i = [], 0
    while i < len(s):
        for cmd in ("\\dfrac", "\\frac"):
            if s.startswith(cmd, i) and s[i + len(cmd):i + len(cmd) + 1] == "{":
                a, j = _group(s, i + len(cmd))
                if j < len(s) and s[j] == "{":
                    b, j = _group(s, j)
                    out.append(f"{_wrap(_expand(a))}/{_wrap(_expand(b))}")
                    i = j
                    break
        else:
            if s.startswith("\\sqrt{", i):
                a, j = _group(s, i + 5)
                out.append("\u221a(" + _expand(a) + ")")
                i = j
            elif s.startswith("\\operatorname{", i):
                a, j = _group(s, i + 13)
                out.append({"sign": "sgn", "atan": "arctan", "asin": "arcsin",
                            "acos": "arccos"}.get(a, a))
                i = j
            else:
                m = re.match(r"\\(ln|log|sin|cos|tan|cot|exp)\{", s[i:])
                if m:
                    a, j = _group(s, i + len(m.group(0)) - 1)
                    out.append(m.group(1) + "(" + _expand(a) + ")")
                    i = j
                else:
                    out.append(s[i])
                    i += 1
    return "".join(out)


def tex_to_text(s: str) -> str:
    """Macht aus einem LaTeX-Schnipsel lesbaren Text."""
    out = s.replace(r"\left", "").replace(r"\right", "")
    out = _expand(out)
    out = re.sub(r"\\text\{([^{}]*)\}", r"\1", out)
    out = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", out)
    out = re.sub(r"\\mathbf\{([^{}]*)\}", r"\1", out)
    out = re.sub(r"\\begin\{cases\}(.*?)\\end\{cases\}",
                 lambda m: "{ " + m.group(1).replace("&", " falls ") + " }", out, flags=re.S)
    for needle, replacement in sorted(_TEX_REPLACEMENTS, key=lambda p: -len(p[0])):
        out = out.replace(needle, replacement)
    out = re.sub(r"_\{([0-9]+)\}", lambda m: m.group(1).translate(_SUB), out)
    out = re.sub(r"_([0-9])", lambda m: m.group(1).translate(_SUB), out)
    out = re.sub(r"\^\{(-?[0-9n]+)\}", lambda m: m.group(1).translate(_SUP), out)
    out = re.sub(r"\^(-?[0-9n])", lambda m: m.group(1).translate(_SUP), out)
    out = out.replace("{", "").replace("}", "").replace("$", "")
    out = re.sub(r"\(\(([^()]*)\)\)", r"(\1)", out)
    out = re.sub(r"\(\s+", "(", out)
    out = re.sub(r"\s+\)", ")", out)
    out = re.sub(r"\s{2,}", "   ", out)
    return out.strip()


# ==========================================================================
# Einzelne Grössen
# ==========================================================================

def tex_any(obj, decimal: bool = False, digits: int = 4) -> str:
    try:
        return tex(obj, decimal, digits)
    except Exception:
        return r"\text{" + str(obj) + "}"


def text_any(obj, decimal: bool = False, digits: int = 4) -> str:
    if isinstance(obj, str):
        return tex_to_text(obj) if "\\" in obj else obj
    if isinstance(obj, Formula):
        return obj.to_str("unicode")
    if isinstance(obj, bool):
        return "wahr" if obj else "falsch"
    try:
        return tex_to_text(tex(obj, decimal, digits))
    except Exception:
        return str(obj)


# ==========================================================================
# Schritte
# ==========================================================================

@dataclass
class Table:
    """
    Eine Tabelle im Rechenweg (Wahrheitstabelle, Vorzeichentabelle, Schrittzähler …).

    `marks` färbt Zeilen: {Zeilenindex: 'ok' | 'bad' | 'hi'} — etwa die Zeilen,
    in denen die Wissensbasis wahr ist (hi) oder ein Gegenbeispiel steht (bad).
    """
    header: list[str]
    rows: list[list[str]]
    marks: dict[int, str] = field(default_factory=dict)
    caption: str = ""

    def text(self) -> str:
        cells = [[text_any(c) for c in row] for row in self.rows]
        head = [text_any(h) for h in self.header]
        widths = [max(len(head[i]), *(len(r[i]) for r in cells)) if cells else len(head[i])
                  for i in range(len(head))]
        line = lambda r: " | ".join(c.center(w) for c, w in zip(r, widths))
        out = [line(head), "-+-".join("-" * w for w in widths)]
        sign = {"ok": " \u2713", "bad": " \u2717", "hi": " \u25c0"}
        for i, r in enumerate(cells):
            out.append(line(r) + sign.get(self.marks.get(i, ""), ""))
        return "\n".join(out)

    def html(self) -> str:
        parts = ["<table class='dk-tab'><thead><tr>"]
        parts += [f"<th>{html.escape(tex_to_text(h))}</th>" for h in self.header]
        parts.append("</tr></thead><tbody>")
        for i, row in enumerate(self.rows):
            cls = self.marks.get(i, "")
            parts.append(f"<tr class='dk-{cls}'>" if cls else "<tr>")
            parts += [f"<td>{html.escape(text_any(c))}</td>" for c in row]
            parts.append("</tr>")
        parts.append("</tbody></table>")
        if self.caption:
            parts.append(f"<div class='dk-n'>{html.escape(self.caption)}</div>")
        return "".join(parts)

    def tex(self) -> str:
        cols = "c" * len(self.header)
        body = [" & ".join(tex_any(h) for h in self.header) + r" \\ \hline"]
        for row in self.rows:
            body.append(" & ".join(tex_any(c) for c in row) + r" \\")
        return r"\begin{array}{" + cols + "}" + " ".join(body) + r"\end{array}"


@dataclass
class Step:
    """
    Ein Schritt im Rechenweg.

    `detail` ist der erklärende Satz (darf LaTeX enthalten), `items` sind
    benannte Grössen — meist die Formel oder der Term nach dem Schritt.
    `rule` nennt die angewandte Regel (»Produktregel«), damit man sie in
    der Prüfung beim Namen nennen kann.
    """
    title: str
    detail: str = ""
    items: list[tuple[str, object]] = field(default_factory=list)
    note: str = ""
    rule: str = ""
    table: "Table | None" = None

    def tex_items(self, decimal: bool, digits: int) -> str:
        parts = []
        for label, obj in self.items:
            piece = tex_any(obj, decimal, digits)
            parts.append(f"{label} = {piece}" if label else piece)
        return r"\qquad ".join(parts)

    def html_items(self, decimal: bool, digits: int) -> str:
        parts = []
        for label, obj in self.items:
            piece = html.escape(text_any(obj, decimal, digits))
            name = (f"<span class='dk-lbl'>{html.escape(tex_to_text(label))} =</span>"
                    if label else "")
            parts.append(f"<span class='dk-item'>{name}<span class='dk-v'>{piece}</span></span>")
        return "".join(parts)


class Result:
    """
    Gemeinsame Grundlage aller Rechenergebnisse.

    Unterklassen setzen `title`, `summary`, `steps` und `highlights`
    (die Grössen, die am Ende zählen). `figure()` liefert, wenn sinnvoll,
    eine matplotlib-Zeichnung dazu.
    """

    title = "Ergebnis"
    summary = ""

    def __init__(self, decimal: bool = False, digits: int = 4):
        self.decimal = decimal
        self.digits = digits
        self.steps: list[Step] = []
        self.highlights: list[tuple[str, object]] = []
        self.warnings: list[str] = []

    # -- Anzeige ----------------------------------------------------------

    def as_decimal(self, digits: int = 4) -> "Result":
        self.decimal, self.digits = True, digits
        return self

    def as_exact(self) -> "Result":
        self.decimal = False
        return self

    def figure(self, **kwargs):
        """Zeichnung zur Rechnung (matplotlib). Unterklassen überschreiben."""
        raise NotImplementedError(f"{type(self).__name__} hat keine Zeichnung")

    def latex(self, with_steps: bool = True) -> str:
        """Der ganze Rechenweg als LaTeX — zum Einfügen in eine Abgabe."""
        out = [r"\textbf{" + self.title + "}"]
        if with_steps:
            for index, step in enumerate(self.steps, 1):
                head = f"{index}. {step.title}" + (f" ({step.rule})" if step.rule else "")
                out.append(r"\text{" + head + "}")
                if step.detail:
                    out.append(r"\quad " + step.detail)
                items = step.tex_items(self.decimal, self.digits)
                if items:
                    out.append(items)
                if step.table is not None:
                    out.append(step.table.tex())
        for label, obj in self.highlights:
            out.append(f"{label} = {tex_any(obj, self.decimal, self.digits)}")
        return "\n\\\\[6pt]\n".join(out)

    def _repr_html_(self) -> str:
        return self._html(with_steps=bool(self.steps))

    CSS = (
        "<style>"
        ".dk{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',"
        "Helvetica,Arial,sans-serif;color:#1d1d1f;line-height:1.5;}"
        ".dk-h{font-weight:600;font-size:15px;margin:0 0 2px;}"
        ".dk-sum{color:#6e6e73;font-size:13px;margin:0 0 14px;}"
        ".dk-step{background:#f5f5f7;border-radius:12px;padding:12px 16px;margin:8px 0;}"
        ".dk-t{font-weight:600;font-size:13px;}"
        ".dk-rule{font-weight:500;font-size:11px;color:#0a60c8;background:#e3effc;"
        "border-radius:6px;padding:1px 7px;margin-left:8px;}"
        ".dk-d{color:#6e6e73;font-size:12.5px;margin-top:3px;}"
        ".dk-n{color:#86868b;font-size:11.5px;margin-top:5px;}"
        ".dk-res{background:#e3effc;border-radius:12px;padding:12px 16px;margin:12px 0 0;}"
        ".dk-warn{background:#fdf3d8;border-radius:10px;padding:8px 14px;margin:8px 0;"
        "color:#a86400;font-size:12px;}"
        ".dk-item{display:inline-flex;align-items:center;gap:7px;margin:8px 22px 4px 0;}"
        ".dk-lbl{font-size:13px;font-weight:500;}"
        ".dk-v{font-size:14px;font-family:'SF Mono',Menlo,Consolas,monospace;}"
        ".dk-tab{border-collapse:collapse;margin:8px 0;font-size:13px;"
        "font-family:'SF Mono',Menlo,Consolas,monospace;}"
        ".dk-tab th,.dk-tab td{border:1px solid #d2d2d7;padding:3px 10px;text-align:center;}"
        ".dk-tab th{background:#ececf0;font-weight:600;}"
        ".dk-tab tr.dk-ok td{background:#e3f6e8;}.dk-tab tr.dk-bad td{background:#fde4e1;}"
        ".dk-tab tr.dk-hi td{background:#e3effc;}"
        "@media (prefers-color-scheme:dark){"
        ".dk-tab th{background:#2c2c2e;}.dk-tab th,.dk-tab td{border-color:#48484a;}"
        ".dk-tab tr.dk-ok td{background:#17331f;}.dk-tab tr.dk-bad td{background:#3d1d1a;}"
        ".dk-tab tr.dk-hi td{background:#16324f;}"
        ".dk{color:#f5f5f7;}.dk-sum,.dk-d{color:#a1a1a6;}"
        ".dk-step{background:#232325;}.dk-res{background:#16324f;}"
        ".dk-rule{background:#16324f;color:#6cb2ff;}"
        ".dk-warn{background:#3a3113;color:#ffd60a;}}"
        "</style>")

    def _html(self, with_steps: bool) -> str:
        parts = [self.CSS, "<div class='dk'>",
                 f"<p class='dk-h'>{html.escape(self.title)}</p>"]
        if self.summary:
            parts.append(f"<p class='dk-sum'>{html.escape(tex_to_text(self.summary))}</p>")
        for warning in self.warnings:
            parts.append(f"<div class='dk-warn'>{html.escape(warning)}</div>")
        if with_steps:
            for index, step in enumerate(self.steps, 1):
                parts.append("<div class='dk-step'>")
                rule = f"<span class='dk-rule'>{html.escape(step.rule)}</span>" if step.rule else ""
                parts.append(f"<div class='dk-t'>{index}. {html.escape(step.title)}{rule}</div>")
                if step.detail:
                    parts.append(f"<div class='dk-d'>{html.escape(tex_to_text(step.detail))}</div>")
                items = step.html_items(self.decimal, self.digits)
                if items:
                    parts.append(f"<div>{items}</div>")
                if step.table is not None:
                    parts.append(step.table.html())
                if step.note:
                    parts.append(f"<div class='dk-n'>{html.escape(step.note)}</div>")
                parts.append("</div>")
        if self.highlights:
            parts.append("<div class='dk-res'>")
            for label, obj in self.highlights:
                piece = html.escape(text_any(obj, self.decimal, self.digits))
                parts.append(f"<span class='dk-item'><span class='dk-lbl'>"
                             f"{html.escape(tex_to_text(label))} =</span>"
                             f"<span class='dk-v'>{piece}</span></span>")
            parts.append("</div>")
        parts.append("</div>")
        return "".join(parts)

    def __repr__(self) -> str:
        lines = [self.title]
        if self.summary:
            lines.append("  " + text_any(self.summary))
        for warning in self.warnings:
            lines.append("  ! " + warning)
        for index, step in enumerate(self.steps, 1):
            head = f"  {index}. {step.title}" + (f"  [{step.rule}]" if step.rule else "")
            lines.append(head)
            if step.detail:
                lines.append("     " + text_any(step.detail))
            for label, obj in step.items:
                name = f"{text_any(label)} = " if label else ""
                lines.append(f"     {name}{text_any(obj, self.decimal, self.digits)}")
            if step.table is not None:
                lines.extend("     " + row for row in step.table.text().split("\n"))
        for label, obj in self.highlights:
            lines.append(f"  {text_any(label)} = {text_any(obj, self.decimal, self.digits)}")
        return "\n".join(lines)

    def show(self):
        """Im Notebook anzeigen, sonst drucken."""
        try:
            from IPython.display import HTML, display
            display(HTML(self._repr_html_()))
        except ImportError:
            print(self)


def show_tex(expr, label: str = "", decimal: bool = False):
    """Einen Term gesetzt anzeigen."""
    piece = tex_any(expr, decimal)
    body = f"{label} = {piece}" if label else piece
    try:
        from IPython.display import Math, display
        display(Math(body))
    except ImportError:
        print(f"{label} = " if label else "", text_any(expr, decimal), sep="")
