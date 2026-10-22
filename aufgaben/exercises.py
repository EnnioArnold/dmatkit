#!/usr/bin/env python3
"""
Die Aufgaben der Übungsserien 1 bis 6 (DMATH-ALGO) — als Daten, mit Selbstkontrolle.

Jede Aufgabe ist ein Datensatz

    Exercise(id, week, title, task, call, expected, hint)

    id        »SW3.5a«: Serie 3, Aufgabe 5, Teilaufgabe a.  »SW3.B1« ist ein Beispiel aus der
              Vorlesung (Mitschrift), »LG2.c1« eine Aufgabe aus dem Lerngerüst (Serie 2, c, erster Term).
    task      der Aufgabentext, knapp und auf Deutsch
    call      eine Funktion ohne Argumente, die das Ergebnisobjekt (mit Rechenweg) liefert
    expected  maschinell prüfbare Erwartungen (Erwartung(...)). Die Soll-Werte stammen aus den
              Serien, aus der Mitschrift oder aus einer unabhängigen Rechnung (Python-Builtins,
              Brute-Force) — nie aus dem Ergebnis des Tools selbst.
    hint      ein Anstoss, keine Lösung

Benutzung

    from dmatkit.aufgaben.exercises import liste, aufgabe, loesen, pruefen, alle_pruefen

    liste(3)                  # was es in Serie 3 gibt
    aufgabe("SW3.5a")         # Aufgabentext und Hinweis (ohne Lösung)
    loesen("SW3.5a")          # Rechenweg mit Regelnamen
    pruefen("SW3.5a")         # stimmt das Ergebnis mit dem Soll überein?
    alle_pruefen()            # alle Aufgaben laufen lassen — alle müssen grün sein

Die Aufgaben zum Selbstprogrammieren (ohne Lösung) stehen in `dmatkit.aufgaben.uebung`.
"""

from __future__ import annotations

import difflib
import itertools
import math
import re
import struct
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable

import sympy as sp

from ..rechnen import algorithmen as ag
from ..rechnen import aussagenlogik as al
from ..rechnen import datentypen as dt
from ..rechnen import kalkuel as ka
from ..rechnen import praedikate as pr
from ..rechnen import rekursion as rk
from ..rechnen.render import Result, Step, Table, text_any

__all__ = ["Exercise", "Erwartung", "Bericht", "Gesamtbericht", "Sammlung", "EXERCISES",
           "aufgabe", "liste", "loesen", "pruefen", "alle_pruefen"]


# ==========================================================================
# Datenstrukturen
# ==========================================================================

@dataclass(frozen=True)
class Erwartung:
    """
    Eine maschinell prüfbare Erwartung.

    `lesen(ergebnis)` holt einen Wert aus dem Ergebnisobjekt, `soll` ist der erwartete Wert
    (aus der Serie oder unabhängig gerechnet), `quelle` sagt woher.
    """
    text: str
    lesen: Callable[[Any], Any]
    soll: Any
    quelle: str = ""

    def pruefen(self, ergebnis) -> tuple[bool, Any, str]:
        """(bestanden, ist, Fehlertext). Eine Ausnahme beim Lesen zählt als nicht bestanden."""
        try:
            ist = self.lesen(ergebnis)
        except Exception as exc:                     # noqa: BLE001 - jede Panne ist ein Befund
            return False, None, f"Wert nicht lesbar: {type(exc).__name__}: {exc}"
        return _gleich(ist, self.soll), ist, ""


def _gleich(ist, soll) -> bool:
    """Vergleich mit Sinn für Wahrheitswerte, Brüche, Gleitkommazahlen, Mengen und Listen."""
    if isinstance(soll, bool):
        return isinstance(ist, bool) and ist == soll
    if isinstance(ist, bool):                        # True ist nicht 1 (siehe tests_exercises)
        return False
    if isinstance(soll, float) or isinstance(ist, float):
        try:
            return math.isclose(float(ist), float(soll), rel_tol=1e-12, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    if isinstance(soll, (list, tuple)) and isinstance(ist, (list, tuple)):
        return len(soll) == len(ist) and all(_gleich(a, b) for a, b in zip(ist, soll))
    if isinstance(soll, dict) and isinstance(ist, dict):
        return set(soll) == set(ist) and all(_gleich(ist[k], soll[k]) for k in soll)
    try:
        return bool(ist == soll)
    except Exception:                                # noqa: BLE001
        return False


def _zeige(wert) -> str:
    if isinstance(wert, (frozenset, set)):
        return "{" + ", ".join(sorted(map(str, wert))) + "}"
    text = text_any(wert) if not isinstance(wert, (list, tuple, dict)) else str(wert)
    return text if len(text) <= 120 else text[:117] + "…"


@dataclass
class Exercise:
    """Eine Aufgabe mit Aufgabentext, Rechenweg (`call`) und Soll-Werten (`expected`)."""
    id: str
    week: int
    title: str
    task: str
    call: Callable[[], Any]
    expected: tuple = ()
    hint: str = ""
    kind: str = "Serie"        # "Serie" | "Beispiel" | "Lerngerüst" | "optional"
    source: str = ""           # z. B. "Serie 3, Aufgabe 5a"

    def __repr__(self) -> str:
        lines = [f"{self.id} — {self.title}   [{self.kind}]", f"  {self.task}"]
        if self.source:
            lines.append(f"  Quelle: {self.source}")
        if self.hint:
            lines.append(f"  Hinweis: {self.hint}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        import html
        out = (f"<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif\">"
               f"<p style='font-weight:600;margin:0'>{html.escape(self.id)} — {html.escape(self.title)}"
               f" <span style='color:#86868b;font-weight:400;font-size:12px'>[{html.escape(self.kind)}]</span></p>"
               f"<p style='color:#6e6e73;font-size:13px;margin:2px 0 6px'>{html.escape(self.task)}</p>")
        if self.source:
            out += f"<p style='color:#86868b;font-size:12px;margin:0'>Quelle: {html.escape(self.source)}</p>"
        if self.hint:
            out += f"<p style='color:#86868b;font-size:12px;margin:0'>Hinweis: {html.escape(self.hint)}</p>"
        return out + "</div>"

    # -- Lösen und Prüfen ---------------------------------------------------

    def loesen(self):
        """Das Ergebnisobjekt mit Rechenweg."""
        return self.call()

    def pruefen(self) -> "Bericht":
        """Rechnet die Aufgabe und vergleicht mit allen Erwartungen."""
        zeilen: list[tuple[bool, str, str]] = []
        try:
            res = self.call()
        except Exception as exc:                     # noqa: BLE001
            zeilen.append((False, "Aufgabe läuft", f"Ausnahme {type(exc).__name__}: {exc}"))
            return Bericht(self.id, self.title, zeilen)
        # Jede Aufgabe muss einen Rechenweg zeigen und lesbar ausgeben können.
        try:
            hat_weg = isinstance(res, Result) and len(res.steps) > 0
            ausgabe = repr(res)
            res._repr_html_()
            res.latex()
            zeilen.append((hat_weg and bool(ausgabe), "Rechenweg vorhanden und darstellbar",
                           "" if hat_weg else "das Ergebnis hat keine Schritte"))
        except Exception as exc:                     # noqa: BLE001
            zeilen.append((False, "Rechenweg vorhanden und darstellbar",
                           f"Darstellung scheitert: {type(exc).__name__}: {exc}"))
        for erw in self.expected:
            ok, ist, fehler = erw.pruefen(res)
            if ok:
                zeilen.append((True, erw.text, f"{_zeige(erw.soll)}" + (f"   ({erw.quelle})" if erw.quelle else "")))
            else:
                detail = fehler or f"erwartet {_zeige(erw.soll)}, erhalten {_zeige(ist)}"
                zeilen.append((False, erw.text, detail))
        return Bericht(self.id, self.title, zeilen)


class Bericht:
    """Ergebnis von `pruefen(id)`. `bool(bericht)` ist True, wenn alle Erwartungen stimmen."""

    def __init__(self, id: str, title: str, zeilen):
        self.id, self.title, self.zeilen = id, title, list(zeilen)

    @property
    def ok(self) -> bool:
        return all(z[0] for z in self.zeilen)

    def __bool__(self) -> bool:
        return self.ok

    @property
    def fehler(self) -> list[tuple[str, str]]:
        return [(z[1], z[2]) for z in self.zeilen if not z[0]]

    def __repr__(self) -> str:
        n_ok = sum(1 for z in self.zeilen if z[0])
        lines = [f"{self.id} — {self.title}: {n_ok} von {len(self.zeilen)} Prüfungen bestanden"]
        for ok, text, detail in self.zeilen:
            lines.append(f"  {'✓' if ok else '✗'} {text}" + (f" — {detail}" if detail else ""))
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        import html
        rows = "".join(
            f"<tr><td style='color:{'#1f9d55' if ok else '#d93025'};font-weight:700'>{'✓' if ok else '✗'}</td>"
            f"<td style='padding:0 12px 0 6px'>{html.escape(text)}</td>"
            f"<td style='color:#6e6e73'>{html.escape(detail)}</td></tr>"
            for ok, text, detail in self.zeilen)
        return (f"<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:13px\">"
                f"<p style='font-weight:600;margin:0 0 6px'>{html.escape(self.id)} — {html.escape(self.title)}</p>"
                f"<table style='border-collapse:collapse'>{rows}</table></div>")


class Gesamtbericht:
    """Ergebnis von `alle_pruefen()`: alle Berichte und eine Zusammenfassung."""

    def __init__(self, berichte: list[Bericht]):
        self.berichte = berichte

    @property
    def gesamt(self) -> int:
        return len(self.berichte)

    @property
    def bestanden(self) -> int:
        return sum(1 for b in self.berichte if b.ok)

    @property
    def ok(self) -> bool:
        return self.gesamt > 0 and self.bestanden == self.gesamt

    @property
    def fehlgeschlagen(self) -> list[str]:
        return [b.id for b in self.berichte if not b.ok]

    def __bool__(self) -> bool:
        return self.ok

    def __repr__(self) -> str:
        wochen: dict[int, list[Bericht]] = {}
        for b in self.berichte:
            wochen.setdefault(EXERCISES[b.id].week, []).append(b)
        lines = [f"{self.bestanden} von {self.gesamt} Aufgaben stimmen mit dem Soll überein"]
        for w in sorted(wochen):
            bs = wochen[w]
            lines.append(f"  SW{w}: {sum(1 for b in bs if b.ok)} von {len(bs)}")
        for b in self.berichte:
            if not b.ok:
                lines.append("")
                lines.append(repr(b))
        if self.ok:
            lines.append("Alles grün.")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        import html
        return "<pre>" + html.escape(repr(self)) + "</pre>"


class Sammlung(Result):
    """
    Mehrere Teilrechnungen einer Aufgabe in einem Ergebnis.

    `teile` bildet Beschriftungen auf Ergebnisobjekte (oder einfache Werte) ab; der Rechenweg
    enthält die Schritte jeder Teilrechnung mit der Beschriftung davor.
    """

    def __init__(self, title: str, teile: dict, summary: str = "", highlights=None):
        super().__init__()
        self.title = title
        self.summary = summary
        self.teile = dict(teile)
        self.werte = {k: getattr(v, "value", v) for k, v in self.teile.items()}
        for label, teil in self.teile.items():
            if isinstance(teil, Result):
                for s in teil.steps:
                    self.steps.append(Step(f"{label}: {s.title}", s.detail, list(s.items), s.note,
                                           s.rule, s.table))
                if not teil.steps:
                    self.steps.append(Step(label, teil.summary))
            else:
                self.steps.append(Step(label, "", [("", teil)]))
        self.highlights = list(highlights) if highlights is not None else [
            (k, v) for k, v in self.werte.items()]

    def __getitem__(self, key):
        return self.teile[key]


# ==========================================================================
# Sammlung und Zugriff
# ==========================================================================

EXERCISES: dict[str, Exercise] = {}
_LOWER: dict[str, str] = {}


def _week_of(id: str) -> int:
    m = re.match(r"(?:SW|LG)(\d+)\.", id)
    if not m:
        raise ValueError(f"Aufgabenschlüssel {id!r} ohne Woche")
    return int(m.group(1))


def _ex(id: str, title: str, task: str, call, expected, hint: str = "", kind: str = "", source: str = ""):
    if id in EXERCISES:
        raise ValueError(f"Doppelter Aufgabenschlüssel {id}")
    if not kind:
        kind = "Lerngerüst" if id.startswith("LG") else ("Beispiel" if ".B" in id else "Serie")
    if not source and kind == "Beispiel":
        source = f"Beispiel aus der Vorlesung (SW{_week_of(id)})"
    ex = Exercise(id, _week_of(id), title, task, call, tuple(expected), hint, kind, source)
    EXERCISES[id] = ex
    _LOWER[id.lower()] = id


def E(text, lesen, soll, quelle=""):
    """Kurzform für `Erwartung`."""
    return Erwartung(text, lesen, soll, quelle)


class Liste(list):
    """Liste von Aufgabenschlüsseln, die sich als Tabelle anzeigt."""

    def __repr__(self) -> str:
        if not self:
            return "(keine Aufgaben)"
        w = max(len(k) for k in self)
        return "\n".join(f"{k:<{w}}  {EXERCISES[k].title}" for k in self)

    def _repr_html_(self) -> str:
        import html
        rows = "".join(
            f"<tr><td style='padding:1px 14px 1px 0;font-family:monospace;font-size:13px'>{html.escape(k)}</td>"
            f"<td style='font-size:13px'>{html.escape(EXERCISES[k].title)}</td>"
            f"<td style='font-size:12px;color:#86868b;padding-left:12px'>{html.escape(EXERCISES[k].kind)}</td></tr>"
            for k in self)
        return f"<table style='border-collapse:collapse'>{rows}</table>"


def _woche(week) -> int | None:
    if week is None:
        return None
    if isinstance(week, int) and not isinstance(week, bool):
        w = week
    else:
        m = re.fullmatch(r"\s*(?:SW|LG|S)?\s*(\d+)\s*", str(week), flags=re.I)
        if not m:
            raise KeyError(f"Unbekannte Woche {week!r}. Gemeint ist eine Zahl von 1 bis 6 oder »SW3«.")
        w = int(m.group(1))
    if w not in {e.week for e in EXERCISES.values()}:
        raise KeyError(f"Für Woche {week!r} gibt es keine Aufgaben. Vorhanden sind Wochen 1 bis 6.")
    return w


def liste(week=None, art: str | None = None) -> Liste:
    """
    Die Aufgabenschlüssel, optional einer Woche (`3` oder `"SW3"`) und einer Art
    (`"Serie"`, `"Beispiel"`, `"Lerngerüst"`, `"optional"`).

        liste(3)
        liste(art="Lerngerüst")
    """
    w = _woche(week)
    out = [k for k, e in EXERCISES.items()
           if (w is None or e.week == w) and (art is None or e.kind.lower() == art.lower())]
    return Liste(out)


def _schluessel(id: str) -> str:
    key = str(id).strip()
    low = key.lower().replace(" ", "")
    if low in _LOWER:
        return _LOWER[low]
    if re.fullmatch(r"\d+\.\w+", low) and ("sw" + low) in _LOWER:       # »3.5a« statt »SW3.5a«
        return _LOWER["sw" + low]
    kinder = [v for k, v in _LOWER.items() if k.startswith(low) and low]
    if kinder:
        raise KeyError(f"Unbekannte Aufgabe {id!r}. Gemeint: {', '.join(kinder[:8])}"
                       + (" …" if len(kinder) > 8 else ""))
    nah = difflib.get_close_matches(low, list(_LOWER), n=4, cutoff=0.6)
    if nah:
        raise KeyError(f"Unbekannte Aufgabe {id!r}. Gemeint: {', '.join(_LOWER[n] for n in nah)}")
    raise KeyError(f"Unbekannte Aufgabe {id!r}. `liste()` zeigt alle Schlüssel (Woche: SW1 bis SW6).")


def aufgabe(id: str) -> Exercise:
    """Aufgabentext und Hinweis (ohne Lösung). `aufgabe("SW3.5a")`."""
    return EXERCISES[_schluessel(id)]


def loesen(id: str):
    """Rechnet die Aufgabe und liefert das Ergebnisobjekt mit Rechenweg."""
    return aufgabe(id).loesen()


def pruefen(id: str) -> Bericht:
    """Rechnet die Aufgabe und vergleicht mit den Soll-Werten. `bool(...)` ist True, wenn alles stimmt."""
    return aufgabe(id).pruefen()


def alle_pruefen(week=None, art: str | None = None) -> Gesamtbericht:
    """
    Lässt alle Aufgaben (oder die einer Woche) laufen und fasst zusammen.
    `bool(...)` ist True, wenn alle Aufgaben mit dem Soll übereinstimmen.
    """
    return Gesamtbericht([EXERCISES[k].pruefen() for k in liste(week, art)])


# ==========================================================================
# Unabhängige Hilfsrechnungen für die Soll-Werte (reines Python, ohne dmatkit)
# ==========================================================================

def _imp(a, b):
    return (not a) or b


def _iff(a, b):
    return a == b


def _vektor(f, n):
    """Wahrheitsvektor einer Python-Funktion über bool-Argumenten, Reihenfolge 0…0 zuerst."""
    return [bool(f(*v)) for v in itertools.product([False, True], repeat=n)]


def _klasse(f, n):
    v = _vektor(f, n)
    return "Tautologie" if all(v) else "Kontradiktion" if not any(v) else "erfüllbar (kontingent)"


def _modelle(f, n):
    return sum(_vektor(f, n))


def _vec_dk(formel, namen):
    """Wahrheitsvektor einer dmatkit-Formel über den Namen (sortiert), 0…0 zuerst."""
    return [bool(formel.eval(dict(zip(namen, v)))) for v in itertools.product([False, True], repeat=len(namen))]


def _bits(x, bits):
    return "".join(f"{b:08b}" for b in struct.pack(">f" if bits == 32 else ">d", x))


def _zerlege(muster, bits):
    e = 8 if bits == 32 else 11
    return muster[0], muster[1:1 + e], muster[1 + e:]


def _wert_binaer(text):
    """Wert einer Binärzahl mit Komma und Periode, z. B. '0.1(10)' oder '11.0(01)' — als Fraction."""
    m = re.fullmatch(r"(\d*)\.?(\d*)(?:\((\d+)\))?", text)
    ganz, pre, per = m.group(1) or "0", m.group(2), m.group(3)
    wert = Fraction(int(ganz, 2))
    if pre:
        wert += Fraction(int(pre, 2), 2 ** len(pre))
    if per:
        wert += Fraction(int(per, 2), (2 ** len(per) - 1) * 2 ** len(pre))
    return wert


def _ist_prim(n):
    return n > 1 and all(n % d for d in range(2, math.isqrt(n) + 1))


def _knoten(text):
    """Zählt Knoten eines Terms aus seinem Text: Variablen und Junktoren (ohne Klammern)."""
    return len(re.findall(r"<->|<=>|->|=>|[A-Za-z]|[&|~^∧∨¬⊕⇒⇔]", text))


# Schrittzähler für Algorithmen, von Hand aus den Zählregeln der Vorlesung abgeleitet (Serie 5,
# Aufgabe 2: Zuweisung, Rechenoperation, Vergleich, return und jeder for-Durchlauf zählen je 1;
# Listenzugriff, len() und die Argumente von range() zählen nicht). Sie laufen unabhängig vom
# Interpreter in dmatkit.

def _schritte_dectobin(x):
    """3 Anfangsschritte + 7 je Durchlauf + 1 für return (Mitschrift)."""
    return 3 + 7 * x.bit_length() + 1 if x else 4


def _schritte_linear(liste_, element):
    s = 0
    for v in liste_:
        s += 2                                      # for-Durchlauf und Vergleich
        if v == element:
            return s + 1                            # return True
    return s + 1                                    # return False


def _schritte_selection(liste_, worst_bound=False):
    """Je i: for + minIndex = i + Tausch (3) = 5; je j: for + Vergleich = 2, plus 1 für minIndex = j."""
    a = list(liste_)
    n = len(a)
    s = 5 * n
    for i in range(n):
        mi = i
        for j in range(i + 1, n):
            s += 2
            if worst_bound:
                s += 1                              # Vorlesung: Zuweisung minIndex = j immer mitgezählt
            if a[j] < a[mi]:
                if not worst_bound:
                    s += 1
                mi = j
        a[i], a[mi] = a[mi], a[i]
    return s


def _schritte_insertion(liste_):
    """Je i: for + j = i = 2. Je Schleifentest: j > 0 (1), j-1 und Vergleich (2). Je Durchlauf
    des Rumpfs 7 (temp = ...: 1; die beiden Listenzuweisungen je 2; j = j-1: 2)."""
    a = list(liste_)
    s = 0
    for i in range(1, len(a)):
        s += 2
        j = i
        while True:
            s += 1                                  # j > 0
            if not j > 0:
                break
            s += 2                                  # j-1 und Liste[j] < Liste[j-1]
            if not a[j] < a[j - 1]:
                break
            s += 7
            a[j], a[j - 1] = a[j - 1], a[j]
            j -= 1
    return s


def _schritte_bubble(liste_):
    """Je i: for = 1. Je j: for, j+1, Vergleich = 3; Tausch 5 (temp: 1; zwei Listenzuweisungen je 2)."""
    a = list(liste_)
    n = len(a)
    s = n
    for i in range(n):
        for j in range(n - 1):
            s += 3
            if a[j] > a[j + 1]:
                s += 5
                a[j], a[j + 1] = a[j + 1], a[j]
    return s


def _schritte_binaer(liste_, element):
    """left = 0 (1), right = len-1 (2); je Durchlauf: Test 1, middle 3, ==: 1, >: 1, neue Grenze 2."""
    s = 3
    left, right = 0, len(liste_) - 1
    while True:
        s += 1                                      # left <= right
        if not left <= right:
            return s + 1                            # return False
        mitte = (left + right) // 2
        s += 3 + 1                                  # middle berechnen und zuweisen; ==
        if liste_[mitte] == element:
            return s + 1                            # return True
        s += 1 + 2                                  # > und Grenze (right = middle-1 / left = middle+1)
        if liste_[mitte] > element:
            right = mitte - 1
        else:
            left = mitte + 1


def _schritte_geschachtelt(n):
    """Lerngerüst Serie 5 (a), rot gefärbt: Zuweisungen, <, + und * (ohne return)."""
    s = 2                                           # anzahl = 0; i = 0
    i = 0
    while True:
        s += 1                                      # i < n
        if not i < n:
            break
        s += 1                                      # j = 0
        j = 0
        while True:
            s += 1                                  # j < i
            if not j < i:
                break
            s += 2 + 2 + 2                          # anzahl = anzahl + 1; anzahl = anzahl * 2; j = j + 1
            j += 1
        s += 2                                      # i = i + 1
        i += 1
    return s


# ==========================================================================
# Serie 1 — Elementare Datentypen
# ==========================================================================

_QUELLE1 = "Serie 1"

# -- Aufgabe 1: boolesche Ausdrücke --------------------------------------------
# Soll: Python wertet mit denselben Regeln aus (not vor and vor or, von links nach rechts).
_BOOL = {
    "a": ("False or True and False and True or False", False, "Serie 1, Aufgabe 1a (Mitschrift: False)"),
    "b": ("False and True or False and not True or False", False, "Serie 1, Aufgabe 1b (Mitschrift: False)"),
    "c": ("not False or not not True and False or not True", True, "Serie 1, Aufgabe 1c (aufbereitet: True)"),
    "d": ("not(not True or not(False and not True)) or False", False, "Serie 1, Aufgabe 1d (Mitschrift: False)"),
    "e": ("not((False or not True) or True) and not(False or True)", False, "Serie 1, Aufgabe 1e (aufbereitet: False)"),
}
_BOOL_PY = {
    "a": lambda: False or True and False and True or False,
    "b": lambda: False and True or False and not True or False,
    "c": lambda: not False or not not True and False or not True,
    "d": lambda: not (not True or not (False and not True)) or False,
    "e": lambda: not ((False or not True) or True) and not (False or True),
}
for _k, (_text, _soll, _q) in _BOOL.items():
    assert _BOOL_PY[_k]() is _soll
    _ex(f"SW1.1{_k}", f"Boolescher Ausdruck {_k})",
        f"Bestimmen Sie Schritt für Schritt den Wert von: {_text}",
        (lambda t=_text: dt.eval_bool(t)),
        [E("Wert", lambda r: r.value, _BOOL_PY[_k](), _q),
         E("letzte Zeile des Rechenwegs", lambda r: r.lines[-1], "T" if _soll else "F", "Rechenweg endet mit dem Wert")],
        "Zuerst not, dann and, am Schluss or; gleiche Operatoren von links nach rechts; Klammern zuerst.",
        source=f"Serie 1, Aufgabe 1{_k}")

# -- Aufgaben 2 bis 4: Zahlen in Binärdarstellung und Brüche ----------------------
_ex("SW1.2", "Binärdarstellung von 409", "Bestimmen Sie die Binärdarstellung von 409.",
    lambda: dt.to_binary(409),
    [E("Bits", lambda r: r.text, bin(409)[2:], "Python bin(409); Mitschrift: 110011001"),
     E("Bits laut Mitschrift", lambda r: r.text, "110011001", "Serie 1, Aufgabe 2"),
     E("Reste von unten nach oben (409 = 204·2 + 1, …)", lambda r: [row[2] for row in r.rows],
       [(409 >> k) & 1 for k in range((409).bit_length())], "unabhängig: Bit k von 409"),
     E("Gegenprobe 256 + 128 + 16 + 8 + 1", lambda r: int(r.text, 2), 256 + 128 + 16 + 8 + 1, "Mitschrift")],
    "Fortlaufend durch 2 teilen, die Reste von unten nach oben lesen.", source="Serie 1, Aufgabe 2")

_ex("SW1.3", "Bruch für 2.10(365)", "Bestimmen Sie einen Bruch, der 2.10365365… darstellt (Periode 365).",
    lambda: dt.repeating_to_fraction("2.10(365)"),
    [E("gekürzter Bruch", lambda r: r.value, Fraction(210365 - 210, 99900),
       "Mitschrift: 99900·x = 210365 − 210, gekürzt 42031/19980"),
     E("ungekürzt (Zähler, Nenner)", lambda r: r.unreduced, (210155, 99900), "Mitschrift"),
     E("Dezimalwert", lambda r: float(r.value), 2.10365365365365, "Gegenprobe als Dezimalzahl")],
    "Mit 10^5 und 10^2 multiplizieren, so dass beide dieselbe Periode haben, dann subtrahieren.",
    source="Serie 1, Aufgabe 3")

_ex("SW1.4", "Binärdarstellung von 1/6", "Bestimmen Sie die Binärdarstellung von 1/6.",
    lambda: dt.frac_to_binary("1/6"),
    [E("Binärbruch mit Periode", lambda r: r.text, "0.0(01)", "Mitschrift: 0.0‾01"),
     E("Wert der Darstellung (Vorkomma, Periode)", lambda r: _wert_binaer(r.text), Fraction(1, 6),
       "unabhängig: 2⁻³ + 2⁻⁵ + … = (1/8)/(1 − 1/4)")],
    "Fortlaufend mal 2 rechnen; der ganzzahlige Anteil ist das Bit. Wenn ein Rest wieder auftaucht, beginnt die Periode.",
    source="Serie 1, Aufgabe 4")

for _bits_n in (32, 64):
    _ex(f"SW1.5a{_bits_n}", f"0.00(10) als {_bits_n}-Bit-Fliesskommazahl",
        f"Bestimmen Sie die Darstellung der Binärzahl 0.00‾10 im {_bits_n}-Bit-Format.",
        (lambda b=_bits_n: dt.ieee754("0.00(10)", b, base=2)),
        [E("Vorzeichen", lambda r: r.sign, "0", "positiv"),
         E("Exponent-Feld", lambda r: r.exponent_bits,
           "01111100" if _bits_n == 32 else "01111111100",
           "Mitschrift: −3 + " + ("127 = 124" if _bits_n == 32 else "1023 = 1020")),
         E("Mantisse", lambda r: r.mantissa_bits,
           "01010101010101010101011" if _bits_n == 32 else "0101" * 13,
           "Mitschrift: 0101… auf " + ("23" if _bits_n == 32 else "52") + " Bit gerundet"),
         E("ganzes Bitmuster", lambda r: r.pattern, _bits(1 / 6, _bits_n), "unabhängig: struct.pack von 1/6")],
        "Normalisieren auf 1.0101… · 2^−3, dann Exponent + Bias.", source="Serie 1, Aufgabe 5a")
    _ex(f"SW1.5b{_bits_n}", f"35.2 als {_bits_n}-Bit-Fliesskommazahl",
        f"Bestimmen Sie die Darstellung der Dezimalzahl 35.2 im {_bits_n}-Bit-Format.",
        (lambda b=_bits_n: dt.ieee754("35.2", b)),
        [E("Vorzeichen", lambda r: r.sign, "0"),
         E("Exponent (35.2 = 1.000110011… · 2^5)", lambda r: r.exponent, 5, "von Hand: 35 = 100011₂ → 5 Stellen"),
         E("Exponent-Feld = 5 + Bias", lambda r: r.exponent_field, 5 + (127 if _bits_n == 32 else 1023)),
         E("ganzes Bitmuster", lambda r: r.pattern, _bits(35.2, _bits_n), "unabhängig: struct.pack von 35.2")],
        "35 = 100011₂, 0.2 = 0.0011… periodisch. Danach wie bei Aufgabe 5a.",
        source="Serie 1, Aufgabe 5b (in der Mitschrift noch offen: »zuhause machen«)")

# -- Aufgabe 6: Teilmengen -----------------------------------------------------------
_ex("SW1.6", "Alle Teilmengen von {a, b, c, d}", "Bestimmen Sie alle Teilmengen der Menge {a, b, c, d}.",
    lambda: dt.power_set("{a,b,c,d}"),
    [E("Anzahl 2⁴", lambda r: r.count, 2 ** 4, "Mitschrift: 16 Stück"),
     E("alle Teilmengen", lambda r: set(r.subsets),
       {frozenset(c) for k in range(5) for c in itertools.combinations("abcd", k)}, "unabhängig: itertools.combinations"),
     E("enthält ∅ und die Menge selbst", lambda r: frozenset() in r.subsets and frozenset("abcd") in r.subsets, True)],
    "Jedes Element ist drin oder nicht: 0000 bis 1111.", source="Serie 1, Aufgabe 6 (in der Mitschrift offen)")

# -- Aufgabe 7 (optional) ---------------------------------------------------------------
_ex("SW1.7a", "Kommazahl zu Bruch (optional)", "Bruchdarstellung einer rationalen Kommazahl: 3.1(6).",
    lambda: dt.repeating_to_fraction("3.1(6)"),
    [E("Bruch", lambda r: r.value, Fraction(316 - 31, 90), "unabhängig: (316 − 31)/90 = 19/6")],
    "Siehe Aufgabe 3. Selbst programmieren: `uebung.pruefe_dezimal_zu_bruch`.", kind="optional",
    source="Serie 1, Aufgabe 7a")
_ex("SW1.7b", "Rationale Zahl zu Binärstring (optional)", "Binärdarstellung als Zeichenkette für 5/6.",
    lambda: dt.frac_to_binary("5/6"),
    [E("Wert der Darstellung", lambda r: _wert_binaer(r.text), Fraction(5, 6), "unabhängig: Periode zurückrechnen"),
     E("Periode gefunden", lambda r: "(" in r.text, True)],
    "Selbst programmieren: `uebung.pruefe_binaer_nachkomma`.", kind="optional", source="Serie 1, Aufgabe 7b")
_ex("SW1.7c", "Quadratwurzel in Binärdarstellung (optional)", "√2 in Binärdarstellung auf 10 Stellen nach dem Komma.",
    lambda: dt.sqrt_binary(2, 10),
    [E("Binärstellen", lambda r: r.text,
       bin(math.isqrt(2 * 4 ** 10) >> 10)[2:] + "." + format(math.isqrt(2 * 4 ** 10) & 1023, "010b"),
       "unabhängig: ganzzahlige Wurzel von 2·4¹⁰")],
    "Mit 4^n multiplizieren, ganzzahlige Wurzel ziehen, durch 2^n teilen.", kind="optional",
    source="Serie 1, Aufgabe 7c")

# -- Beispiele aus der Vorlesung (Mitschrift) ---------------------------------------------
_ex("SW1.B1", "Binärdarstellung von 25", "Bestimmen Sie die Binärdarstellung von 25 (Stellenwerttabelle und Division).",
    lambda: dt.to_binary(25),
    [E("Bits", lambda r: r.text, "11001", "Mitschrift"),
     E("Stellenwerte 16 8 4 2 1", lambda r: r.weights, [16, 8, 4, 2, 1], "Mitschrift"),
     E("Gegenprobe", lambda r: int(r.text, 2), 25)], "25 = 12·2 + 1, 12 = 6·2 + 0, …")
_ex("SW1.B2", "14.2(51) als Bruch", "Bestimmen Sie den Bruch von 14.2515151…",
    lambda: dt.repeating_to_fraction("14.2(51)"),
    [E("ungekürzt 14109/990", lambda r: r.unreduced, (14109, 990), "Mitschrift"),
     E("gekürzt", lambda r: r.value, Fraction(14109, 990))], "1000x und 10x subtrahieren.")
_ex("SW1.B3", "14.05 als Bruch", "Bestimmen Sie den Bruch von 14.05 (ohne Periode).",
    lambda: dt.repeating_to_fraction("14.05"),
    [E("ungekürzt 1405/100", lambda r: r.unreduced, (1405, 100), "Mitschrift"),
     E("gekürzt", lambda r: r.value, Fraction(1405, 100))], "Ohne Periode: Kommastellen als Zehnerpotenz im Nenner.")
_ex("SW1.B4", "0.625 binär", "Bestimmen Sie die Binärdarstellung von 0.625.",
    lambda: dt.frac_to_binary("0.625"),
    [E("Bits von oben nach unten", lambda r: r.text, "0.101", "Mitschrift"),
     E("Wert", lambda r: _wert_binaer(r.text), Fraction(5, 8))], "Mal 2, ganzzahligen Anteil ablesen, bis Rest 0.")
_ex("SW1.B5", "0.3 binär (periodisch)", "Bestimmen Sie die Binärdarstellung von 0.3.",
    lambda: dt.frac_to_binary("0.3"),
    [E("Binärbruch mit Periode", lambda r: r.text, "0.0(1001)", "Mitschrift: (0.3)₂ = 0.0‾1001"),
     E("Wert", lambda r: _wert_binaer(r.text), Fraction(3, 10))],
    "Der Rest 0.6 taucht wieder auf: die Rechnung stoppt nie, die Periode ist ablesbar.")
_ex("SW1.B6", "0.7 + 0.1 == 0.9 − 0.1", "Was liefert der Vergleich in Python — und warum?",
    lambda: dt.float_compare("0.7 + 0.1 == 0.9 - 0.1"),
    [E("Python-Vergleich", lambda r: r.python_equal, 0.7 + 0.1 == 0.9 - 0.1, "Python selbst (Serie: False)"),
     E("exakt gerechnet gleich", lambda r: r.exact_equal, True, "4/5 = 4/5")],
    "Nie Floats mit == vergleichen.")
_MENGEN = {
    "a": ("Durchschnitt", "{a,b,c}", "∩", "{b,c,d}", set("abc") & set("bcd")),
    "b": ("Vereinigung", "{a,b,c}", "∪", "{b,c,d}", set("abc") | set("bcd")),
    "c": ("Differenz", "{a,b,c}", "\\", "{b,c,d}", set("abc") - set("bcd")),
    "d": ("Durchschnitt zweier fremder Mengen", "{a,b}", "∩", "{c,d}", set("ab") & set("cd")),
}
for _k, (_name, _l, _op, _r, _soll) in _MENGEN.items():
    _ex(f"SW1.B7{_k}", f"Menge: {_name}", f"Berechnen Sie {_l} {_op} {_r}.",
        (lambda l=_l, o=_op, r=_r: dt.set_op(l, o, r)),
        [E("Ergebnis", lambda r: r.result, frozenset(_soll), "Mitschrift; unabhängig mit Python-Mengen")],
        "Relation gibt wahr/falsch, Operation gibt ein neues Objekt.")
_ex("SW1.B7e", "Komplement", "Grundmenge {a,b,c,d,e}: Was ist das Komplement von {a,b}?",
    lambda: dt.set_op("{a,b}", "komplement", universe="{a,b,c,d,e}"),
    [E("Komplement", lambda r: r.result, frozenset(set("abcde") - set("ab")), "Mitschrift: cde")],
    "Alles aus der Grundmenge, was nicht in der Teilmenge ist.")
_ex("SW1.B7f", "Teilmenge oder Element", "Ist b ∈ {a,b,c}? Ist {b} ⊂ {a,b,c}? Ist {a,d} ⊂ {a,b,c}?",
    lambda: Sammlung("Element- und Teilmengenrelationen", {
        "b ∈ {a,b,c}": dt.set_relation("b", "∈", "{a,b,c}"),
        "{b} ⊂ {a,b,c}": dt.set_relation("{b}", "⊂", "{a,b,c}"),
        "{a,d} ⊂ {a,b,c}": dt.set_relation("{a,d}", "⊂", "{a,b,c}")}),
    [E("b ∈ {a,b,c}", lambda r: r.werte["b ∈ {a,b,c}"], "b" in {"a", "b", "c"}),
     E("{b} ⊂ {a,b,c}", lambda r: r.werte["{b} ⊂ {a,b,c}"], {"b"} <= {"a", "b", "c"}),
     E("{a,d} ⊂ {a,b,c}", lambda r: r.werte["{a,d} ⊂ {a,b,c}"], {"a", "d"} <= {"a", "b", "c"})],
    "∈ vergleicht ein Objekt mit einer Menge, ⊂ vergleicht zwei Mengen.")
_ex("SW1.B7g", "Partition", "Zerlegen {a,b} und {c,d,e} die Menge {a,b,c,d,e} in eine Partition?",
    lambda: dt.is_partition(["{a,b}", "{c,d,e}"], "{a,b,c,d,e}"),
    [E("Partition", lambda r: r.value,
       {"a", "b"}.isdisjoint({"c", "d", "e"}) and {"a", "b"} | {"c", "d", "e"} == set("abcde"))],
    "Fremde, nicht leere Teile, die zusammen die Grundmenge ergeben.")
_ex("SW1.B8", "Listenindex", "a = (5, 3, 8, 8, 1): Was ist a₂ = a[2] (Index ab 1)?",
    lambda: dt.list_index([5, 3, 8, 8, 1], 2),
    [E("a₂", lambda r: r.value, [5, 3, 8, 8, 1][2 - 1], "Mitschrift: 3 (Mathematik zählt ab 1, Python ab 0)")],
    "Index-Falle: Mathematiker zählen ab 1, Informatiker ab 0.")
_LISTEN = {
    "a": ("(1, 3, −1, 0, 1.5) + (1, 2, 2, 1, 0.5)", "[1, 3, -1, 0, 1.5] + [1, 2, 2, 1, 0.5]",
          [x + y for x, y in zip([1, 3, -1, 0, 1.5], [1, 2, 2, 1, 0.5])]),
    "b": ("(True, True, False) and (False, True, False)", "[True, True, False] and [False, True, False]",
          [x and y for x, y in zip([True, True, False], [False, True, False])]),
    "c": ("3 · (2, 5, 4, −2)", "3 * [2, 5, 4, -2]", [3 * x for x in [2, 5, 4, -2]]),
    "d": ("3 − (0, 1, 2, 3, 2, 1, 0)", "3 - [0, 1, 2, 3, 2, 1, 0]", [3 - x for x in [0, 1, 2, 3, 2, 1, 0]]),
    "e": ("not (True, False, False, True, False)", "not [True, False, False, True, False]",
          [not x for x in [True, False, False, True, False]]),
}
for _k, (_math, _code, _soll) in _LISTEN.items():
    _ex(f"SW1.B9{_k}", f"Listen komponentenweise ({_k})", f"Berechnen Sie {_math}.",
        (lambda c=_code: dt.broadcast_calc(c)),
        [E("Ergebnis", lambda r: r.value, _soll, "unabhängig: Python-Listenausdruck")],
        "Gleich lange Listen komponentenweise; einzelne Werte zuerst zur Liste aufblasen (Broadcasting).")
_ex("SW1.B10", "Matrixindex und Dimension", "A = (5 3 8 8 1 / 1 0 0 9 7): Was ist a₂₄, und welche Dimension hat A?",
    lambda: Sammlung("Matrix A", {
        "a₂₄": dt.matrix_index([[5, 3, 8, 8, 1], [1, 0, 0, 9, 7]], 2, 4),
        "Dimension": dt.matrix_dim([[5, 3, 8, 8, 1], [1, 0, 0, 9, 7]])}),
    [E("a₂₄", lambda r: r.werte["a₂₄"], [[5, 3, 8, 8, 1], [1, 0, 0, 9, 7]][1][3], "Serie: 9"),
     E("Dimension m × n", lambda r: tuple(r.werte["Dimension"]), (2, 5), "Mitschrift: 2×5, also 10 Elemente")],
    "Zeile zuerst, dann Spalte; links in m × n steht die Höhe.")
_ex("SW1.B11a", "Matrizen komponentenweise", "Berechnen Sie (1 2 / 3 4 / 7 5) + (0 4 / 1 1 / 5 2).",
    lambda: dt.broadcast_calc("[[1,2],[3,4],[7,5]] + [[0,4],[1,1],[5,2]]"),
    [E("Ergebnis", lambda r: r.value,
       [[x + y for x, y in zip(ra, rb)] for ra, rb in zip([[1, 2], [3, 4], [7, 5]], [[0, 4], [1, 1], [5, 2]])])],
    "Gleiche Dimension: Eintrag für Eintrag.")
_ex("SW1.B11b", "Wahrheitswert-Matrizen", "Berechnen Sie (F T / F T) or (T F / F T).",
    lambda: dt.broadcast_calc("[[False,True],[False,True]] or [[True,False],[False,True]]"),
    [E("Ergebnis", lambda r: r.value,
       [[x or y for x, y in zip(ra, rb)] for ra, rb in zip([[False, True], [False, True]], [[True, False], [False, True]])])],
    "or komponentenweise.")
_A3 = [[1, 2, 1, 1], [3, 4, 2, 2], [3, 5, 1, 1]]
_B3 = [[1, 2, 3, 2], [1, 5, 3, 3], [1, 2, 2, 5]]
_ex("SW1.B11c", "Broadcasting mit Matrizen", "Berechnen Sie 2 + A − (1;2;3) + (0 1 1 2)·B mit A, B wie in der Serie.",
    lambda: dt.broadcast_calc("2 + A - [[1],[2],[3]] + [[0,1,1,2]] * B", A=_A3, B=_B3),
    [E("Ergebnis (3×4)", lambda r: r.value,
       [[2 + _A3[i][j] - (i + 1) + [0, 1, 1, 2][j] * _B3[i][j] for j in range(4)] for i in range(3)],
       "unabhängig: Doppelschleife")],
    "Einzelwert, Spalte und Zeile werden auf 3×4 aufgeblasen, dann komponentenweise rechnen.")
def _dec_to_bin_serie(x):
    """Die Funktion decToBin aus der Serie, wörtlich in Python."""
    y = ""
    temp = x
    while temp != 0:
        if temp % 2 == 0:
            y = "0" + y
        else:
            y = "1" + y
        temp = temp // 2
    return y


_ex("SW1.B12", "decToBin(25)", "Was liefert der Algorithmus decToBin für x = 25?",
    lambda: Sammlung("decToBin", {"decToBin(25)": dt.to_binary(25)}, "Division durch 2, Reste von vorn anhängen."),
    [E("Ergebnis", lambda r: r.teile["decToBin(25)"].text, _dec_to_bin_serie(25), "Funktion der Serie, wörtlich ausgeführt"),
     E("bin(25)", lambda r: r.teile["decToBin(25)"].text, bin(25)[2:])],
    "Pro Durchlauf: Rest ablesen, durch 2 teilen.")
_ex("SW1.B13", "// und %", "Berechnen Sie 10 // 3, 10 % 3, 10 % 7 und 10 % 5.",
    lambda: Sammlung("Ganzzahldivision und Rest", {
        "10 // 3": dt.int_op(10, "//", 3), "10 % 3": dt.int_op(10, "%", 3),
        "10 % 7": dt.int_op(10, "%", 7), "10 % 5": dt.int_op(10, "%", 5)}),
    [E("10 // 3", lambda r: r.werte["10 // 3"], 10 // 3, "Mitschrift: 3"),
     E("10 % 3", lambda r: r.werte["10 % 3"], 10 % 3, "Mitschrift: 1, weil 3·3 + 1 = 10"),
     E("10 % 7", lambda r: r.werte["10 % 7"], 10 % 7, "Mitschrift: 3, weil 1·7 + 3 = 10"),
     E("10 % 5", lambda r: r.werte["10 % 5"], 10 % 5, "Mitschrift: 0, weil 2·5 + 0 = 10")],
    "a = q·b + r mit 0 ≤ r < b.")
_ex("SW1.B14", "Ist das eine Funktion?", "Ordnet f: {1,2,3} → {a,b} mit 1↦a, 2↦b, 3↦a jedem Element genau ein Element zu?",
    lambda: dt.check_function({1, 2, 3}, {"a", "b"}, {1: "a", 2: "b", 3: "a"}),
    [E("ist Funktion", lambda r: r.is_function, True, "jedes x hat genau ein Bild"),
     E("surjektiv", lambda r: r.surjective, set({1: "a", 2: "b", 3: "a"}.values()) == {"a", "b"}, "Bildmenge {a, b} = Wertebereich"),
     E("injektiv", lambda r: r.injective, len(set({1: "a", 2: "b", 3: "a"}.values())) == 3, "3 Elemente, aber nur 2 Bilder")],
    "Definition 7: jedem Element in D genau ein Element in W.")

# -- Lerngerüst Serie 1 -----------------------------------------------------------------------
_LG1A = {
    "a1": ("not True or not (not False and True)", lambda: not True or not (not False and True)),
    "a2": ("False and True or False or not True and not False",
           lambda: False and True or False or not True and not False),
}
for _k, (_text, _py) in _LG1A.items():
    _ex(f"LG1.{_k}", f"Boolescher Ausdruck ({_k})", f"Bestimmen Sie Schritt für Schritt den Wert von: {_text}",
        (lambda t=_text: dt.eval_bool(t)),
        [E("Wert", lambda r: r.value, _py(), "unabhängig: Python")],
        "not vor and vor or.", source="Lerngerüst Serie 1 (a)")
for _n in (123, 741):
    _ex(f"LG1.b{1 if _n == 123 else 2}", f"Binärdarstellung von {_n}", f"Bestimmen Sie die Binärdarstellung von {_n}.",
        (lambda n=_n: dt.to_binary(n)),
        [E("Bits", lambda r: r.text, bin(_n)[2:], "Python bin()")],
        "Fortlaufend durch 2 teilen.", source="Lerngerüst Serie 1 (b)")
_ex("LG1.c1", "Bruch für 20.1(445)", "Bestimmen Sie einen Bruch, der 20.1445445… darstellt.",
    lambda: dt.repeating_to_fraction("20.1(445)"),
    [E("Bruch", lambda r: r.value, Fraction(201445 - 201, 9990), "unabhängig: (201445 − 201)/9990")],
    "Wie Serie 1, Aufgabe 3.", source="Lerngerüst Serie 1 (c)")
_ex("LG1.c2", "Bruch für 0.173(15)", "Bestimmen Sie einen Bruch, der 0.17315151… darstellt.",
    lambda: dt.repeating_to_fraction("0.173(15)"),
    [E("Bruch", lambda r: r.value, Fraction(17315 - 173, 99000), "unabhängig: (17315 − 173)/99000")],
    "Wie Serie 1, Aufgabe 3.", source="Lerngerüst Serie 1 (c)")
_ex("LG1.d", "Binärdarstellung von 0.1", "Bestimmen Sie die Binärdarstellung von 0.1.",
    lambda: dt.frac_to_binary("0.1"),
    [E("Binärbruch mit Periode", lambda r: r.text, "0.0(0011)", "von Hand: 0.1·2 = 0 + 0.2, …, Rest 0.2 kehrt wieder"),
     E("Wert", lambda r: _wert_binaer(r.text), Fraction(1, 10))],
    "Wie 0.3 in der Vorlesung.", source="Lerngerüst Serie 1 (d)")
_ex("LG1.e", "0.0(110) als 32-Bit-Zahl", "Geben Sie die Binärzahl 0.0110110… als Fliesskommazahl im 32-Bit-Format an.",
    lambda: dt.ieee754("0.0(110)", 32, base=2),
    [E("Wert der Binärzahl", lambda r: r.exact, Fraction(3, 7), "unabhängig: 0.0(110)₂ = 6/14"),
     E("ganzes Bitmuster", lambda r: r.pattern, _bits(3 / 7, 32), "unabhängig: struct.pack von 3/7")],
    "Normalisieren: 1.10110… · 2^−2.", source="Lerngerüst Serie 1 (e)")
_ex("LG1.f", "Broadcasting", "Berechnen Sie (1 1 2 2 / 2 2 0 0) + 1 − (2 ; 1) mit Broadcasting.",
    lambda: dt.broadcast_calc("[[1,1,2,2],[2,2,0,0]] + 1 - [[2],[1]]"),
    [E("Ergebnis", lambda r: r.value,
       [[x + 1 - s for x in row] for row, s in zip([[1, 1, 2, 2], [2, 2, 0, 0]], [2, 1])], "unabhängig: Schleife")],
    "Die Spalte (2;1) wird auf alle vier Spalten ausgedehnt.", source="Lerngerüst Serie 1 (f)")


# ==========================================================================
# Serie 2 — Aussagenlogik
# ==========================================================================

def _modelle_liste(f, namen):
    """Alle erfüllenden Belegungen einer Python-Funktion (Reihenfolge 0…0 zuerst), als Dicts."""
    return [dict(zip(namen, v)) for v in itertools.product([False, True], repeat=len(namen)) if f(*v)]


def _tabelle_ref(spalten, n):
    """Zeilen einer Wahrheitstabelle (0/1 als Text) aus Python-Funktionen, 0…0 zuerst oder 1…1 zuerst."""
    return [[str(int(x)) for x in v] + [str(int(bool(s(*v)))) for s in spalten]
            for v in itertools.product([False, True], repeat=n)]


# -- Aufgabe 1: Syntaxbäume ---------------------------------------------------------------------
_SB1A = "((A -> B) & (~B)) <-> (~A)"
_ex("SW2.1a", "Syntaxbaum von ((A ⇒ B) ∧ ¬B) ⇔ ¬A", "Zeichnen Sie den Syntaxbaum des Terms ((A ⇒ B) ∧ (¬B)) ⇔ (¬A).",
    lambda: al.syntax_tree(_SB1A),
    [E("Wurzel", lambda r: r.root.label, "↔", "oberster Junktor ist ⇔"),
     E("Anzahl Knoten", lambda r: r.size, _knoten(_SB1A), "unabhängig: Variablen und Junktoren zählen"),
     E("Ebenen (Tiefe)", lambda r: r.depth, 4, "von Hand: ⇔ / ∧ ¬ / ⇒ ¬ A / A B B"),
     E("Blätter von links nach rechts", lambda r: r.leaves, re.findall(r"[A-Z]", _SB1A), "unabhängig"),
     E("Signatur", lambda r: r.signature, ["A", "B"]),
     E("Wahrheitswerte im Baum",
       lambda r: [r.evaluate(A=a, B=b) for a, b in itertools.product([False, True], repeat=2)],
       _vektor(lambda A, B: _iff(_imp(A, B) and not B, not A), 2), "unabhängig: Python")],
    "Von oben nach unten: der Junktor, der zuletzt ausgewertet wird, ist die Wurzel.",
    source="Serie 2, Aufgabe 1a")
_SB1B = "((P | Q) -> R) <-> ((P -> R) & (Q -> R))"
_ex("SW2.1b", "Syntaxbaum von ((P ∨ Q) ⇒ R) ⇔ ((P ⇒ R) ∧ (Q ⇒ R))", "Zeichnen Sie den Syntaxbaum des Terms.",
    lambda: al.syntax_tree(_SB1B),
    [E("Wurzel", lambda r: r.root.label, "↔"),
     E("Anzahl Knoten", lambda r: r.size, _knoten(_SB1B), "unabhängig"),
     E("Ebenen (Tiefe)", lambda r: r.depth, 4, "von Hand: ⇔ / ⇒ ∧ / ∨ R ⇒ ⇒ / P Q P R Q R"),
     E("Signatur", lambda r: r.signature, ["P", "Q", "R"]),
     E("allgemeingültig (alle 8 Interpretationen wahr)",
       lambda r: all(r.evaluate(P=p, Q=q, R=rr) for p, q, rr in itertools.product([False, True], repeat=3)),
       all(_vektor(lambda P, Q, R: _iff(_imp(P or Q, R), _imp(P, R) and _imp(Q, R)), 3)), "unabhängig: Python")],
    "Zuerst die Wurzel bestimmen, dann die Teilbäume links und rechts.",
    source="Serie 2, Aufgabe 1b (in der Mitschrift offen)")

# -- Aufgabe 2: Spezifikation ---------------------------------------------------------------------
_SPEC = {"A1": "~L -> Q", "A2": "~L -> B", "A3": "~L <-> N", "A4": "~Q -> B", "A5": "~B"}


def _spec(L, Q, B, N):
    return (_imp(not L, Q) and _imp(not L, B) and _iff(not L, N) and _imp(not Q, B) and not B)


_ex("SW2.2", "Ist die Spezifikation konsistent?",
    "A1: ¬L ⇒ Q; A2: ¬L ⇒ B; A3: ¬L ⇔ N; A4: ¬Q ⇒ B; A5: ¬B. Gibt es eine Interpretation, die alle fünf wahr macht?",
    lambda: al.check_consistency(_SPEC),
    [E("konsistent", lambda r: r.consistent, True, "Mitschrift: Erfüllbarkeit = Konsistenz"),
     E("einzige erfüllende Interpretation", lambda r: sorted(sorted(m.items()) for m in r.models),
       sorted(sorted(m.items()) for m in _modelle_liste(lambda B, L, N, Q: _spec(L, Q, B, N), ["B", "L", "N", "Q"])),
       "unabhängig: alle 16 Interpretationen in Python"),
     E("16 Zeilen in der Tabelle", lambda r: len(r.rows), 2 ** 4)],
    "Wahrheitstabelle über L, Q, B, N; eine Zeile mit lauter Einsen genügt.", source="Serie 2, Aufgabe 2")

# -- Aufgabe 3: erfüllbar, allgemeingültig, unerfüllbar ----------------------------------------------
_f3a = lambda A, B: (A and B) and ((not A) or (not B))
_f3b = lambda P, Q, R: _imp(_imp(P or Q, R), _imp(P, R) or _imp(Q, R))
_ex("SW2.3a", "Klassifizieren: (A ∧ B) ∧ (¬A ∨ ¬B)", "Ist der Term erfüllbar, allgemeingültig oder unerfüllbar?",
    lambda: al.classify("(A & B) & (~A | ~B)"),
    [E("Klasse", lambda r: r.classification, _klasse(_f3a, 2), "unabhängig: Python-Tabelle"),
     E("Anzahl erfüllender Zeilen", lambda r: len(r.models), _modelle(_f3a, 2))],
    "Wahrheitstabelle Schritt für Schritt.", source="Serie 2, Aufgabe 3a")
_ex("SW2.3b", "Klassifizieren: ((P ∨ Q) ⇒ R) ⇒ ((P ⇒ R) ∨ (Q ⇒ R))", "Ist der Term erfüllbar, allgemeingültig oder unerfüllbar?",
    lambda: al.classify("((P | Q) -> R) -> ((P -> R) | (Q -> R))"),
    [E("Klasse", lambda r: r.classification, _klasse(_f3b, 3), "unabhängig: Python-Tabelle"),
     E("Anzahl erfüllender Zeilen", lambda r: len(r.models), _modelle(_f3b, 3))],
    "Wahrheitstabelle Schritt für Schritt.", source="Serie 2, Aufgabe 3b (in der Mitschrift »lösen«)")


def _gegenbeispiele(f, g, namen):
    return sorted(tuple(sorted(d.items())) for d in
                  [dict(zip(namen, v)) for v in itertools.product([False, True], repeat=len(namen))
                   if bool(f(*v)) != bool(g(*v))])


_f4a1 = lambda P, Q, R: _imp(P, Q or R)
_f4a2 = lambda P, Q, R: Q or ((not P) or R)
_f4b2 = lambda P, Q, R: _imp(Q and not R, P)
_ex("SW2.4a", "Äquivalenz: P ⇒ (Q ∨ R) und Q ∨ (¬P ∨ R)", "Sind die beiden Terme äquivalent?",
    lambda: al.check_equivalence("P -> (Q | R)", "Q | (~P | R)"),
    [E("äquivalent", lambda r: r.equivalent, not _gegenbeispiele(_f4a1, _f4a2, "PQR"), "unabhängig: Python-Tabelle"),
     E("keine Gegenbeispiele", lambda r: len(r.counterexamples), 0)],
    "Zeile für Zeile beide Terme auswerten und vergleichen.", source="Serie 2, Aufgabe 4a")
_ex("SW2.4b", "Äquivalenz: P ⇒ (Q ∨ R) und (Q ∧ ¬R) ⇒ P", "Sind die beiden Terme äquivalent?",
    lambda: al.check_equivalence("P -> (Q | R)", "(Q & ~R) -> P"),
    [E("äquivalent", lambda r: r.equivalent, not _gegenbeispiele(_f4a1, _f4b2, "PQR"), "unabhängig: Python-Tabelle"),
     E("Gegenbeispiele", lambda r: sorted(tuple(sorted(d.items())) for d in r.counterexamples),
       _gegenbeispiele(_f4a1, _f4b2, "PQR"), "P=0,Q=1,R=0 und P=1,Q=0,R=0")],
    "Eine einzige Zeile mit verschiedenen Werten genügt für »nicht äquivalent«.",
    source="Serie 2, Aufgabe 4b")

# -- Aufgabe 5: Grösse der Wahrheitstabelle ---------------------------------------------------------
_ex("SW2.5", "Zeilen bei 50 atomaren Aussagen", "Wie viele Zeilen hat die Wahrheitstabelle bei 50 atomaren Aussagen?",
    lambda: al.table_size(50),
    [E("Zeilen", lambda r: r.rows, 1 << 50, "Mitschrift: 2^50 Zeilen = 1 125 899 906 842 624")],
    "Jede Aussage kann wahr oder falsch sein: 2·2·…·2.", source="Serie 2, Aufgabe 5")

# -- Aufgabe 6: Rechengesetze bestätigen --------------------------------------------------------------
_GESETZE = {   # Name -> Python-Funktionen (a, b, c) -> (links, rechts)
    "Doppelnegation": [lambda a, b, c: (not (not a), a)],
    "Idempotenz": [lambda a, b, c: (a or a, a), lambda a, b, c: (a and a, a)],
    "Dominanz": [lambda a, b, c: (a or True, True), lambda a, b, c: (a and False, False)],
    "Identität": [lambda a, b, c: (a or False, a), lambda a, b, c: (a and True, a)],
    "Invers": [lambda a, b, c: (a or not a, True), lambda a, b, c: (a and not a, False)],
    "Kommutativität": [lambda a, b, c: (a or b, b or a), lambda a, b, c: (a and b, b and a)],
    "Absorption": [lambda a, b, c: (a or (a and b), a), lambda a, b, c: (a and (a or b), a)],
    "Assoziativität": [lambda a, b, c: ((a or b) or c, a or (b or c)),
                       lambda a, b, c: ((a and b) and c, a and (b and c))],
    "Distributivität": [lambda a, b, c: (a or (b and c), (a or b) and (a or c)),
                        lambda a, b, c: (a and (b or c), (a and b) or (a and c))],
    "De Morgan": [lambda a, b, c: (not (a or b), (not a) and (not b)),
                  lambda a, b, c: (not (a and b), (not a) or (not b))],
}


def _gesetz_gilt(name):
    return all(l == r for fn in _GESETZE[name] for l, r in (fn(*v) for v in itertools.product([False, True], repeat=3)))


def _gesetze_teile():
    teile = {}
    for name in _GESETZE:
        for i, res in enumerate(al.verify_law(name), 1):
            teile[f"{name} ({i})"] = res
    return teile


_ex("SW2.6", "Rechengesetze mit Wahrheitstabellen bestätigen",
    "Bestätigen Sie ein paar der Gesetze (Doppelnegation, Idempotenz, Dominanz, Identität, Invers, Kommutativität, "
    "Absorption, Assoziativität, Distributivität, De Morgan) Schritt für Schritt mit einer Wahrheitstabelle.",
    lambda: Sammlung("Rechengesetze bestätigt", _gesetze_teile(),
                     highlights=[("Gesetze", ", ".join(_GESETZE))]),
    [E("alle Gesetze als äquivalent erkannt", lambda r: all(t.equivalent for t in r.teile.values()), True),
     E("jedes Gesetz gilt (Python-Brute-Force)",
       lambda r: all(_gesetz_gilt(k.split(" (")[0]) for k in r.teile), True, "unabhängig: Python-Tabelle"),
     E("Anzahl Gesetze", lambda r: len({k.split(" (")[0] for k in r.teile}), len(_GESETZE))],
    "Beide Seiten in einer Tabelle auswerten; die letzten Spalten müssen gleich sein.",
    source="Serie 2, Aufgabe 6")

# -- Aufgabe 7: Vereinfachen ----------------------------------------------------------------------------
_VEREINFACHEN = {
    "a": ("(A & B) | (~A & B)", "A, B", lambda A, B: (A and B) or ((not A) and B), "B"),
    "b": ("(P | ~(Q & P)) & (R | (S | R))", "P, Q, R, S",
          lambda P, Q, R, S: (P or not (Q and P)) and (R or (S or R)), "R ∨ S"),
    "c": ("(P & Q) -> (P | Q)", "P, Q", lambda P, Q: _imp(P and Q, P or Q), None),
    "d": ("(~Q & (P -> Q)) -> ~P", "P, Q", lambda P, Q: _imp((not Q) and _imp(P, Q), not P), None),
    "e": ("(P -> Q) & (Q -> R) -> (P -> R)", "P, Q, R", lambda P, Q, R: _imp(_imp(P, Q) and _imp(Q, R), _imp(P, R)), None),
}
_SIMPL_TEXT = {"a": "(A ∧ B) ∨ (¬A ∧ B)", "b": "(P ∨ ¬(Q ∧ P)) ∧ (R ∨ (S ∨ R))", "c": "(P ∧ Q) ⇒ (P ∨ Q)",
               "d": "(¬Q ∧ (P ⇒ Q)) ⇒ ¬P", "e": "((P ⇒ Q) ∧ (Q ⇒ R)) ⇒ (P ⇒ R)"}
for _k, (_text, _vars, _py, _ziel) in _VEREINFACHEN.items():
    _namen = sorted(v.strip() for v in _vars.split(","))
    _soll_vec = _vektor(_py, len(_namen))
    _erw = [E("Ergebnis ist äquivalent zum Ausgangsterm", lambda r, n=_namen: _vec_dk(r.result, n), _soll_vec,
              "unabhängig: Python-Tabelle des Ausgangsterms")]
    if _ziel is not None:
        _erw.append(E("vereinfachter Term", lambda r: r.result.to_str("unicode"), _ziel, "von Hand vereinfacht"))
    else:
        _erw.append(E("allgemeingültig (Ergebnis ist wahr)", lambda r: r.is_tautology, all(_soll_vec),
                      "unabhängig: Python-Tabelle"))
    _erw.append(E("nennt die angewandten Regeln", lambda r: all(s.rule for s in r.steps[1:-1]) and len(r.steps) > 2, True,
                  "jeder Schritt mit Regelname"))
    _ex(f"SW2.7{_k}", f"Vereinfachen: {_SIMPL_TEXT[_k]}", f"Vereinfachen Sie mit den Rechengesetzen so weit wie möglich: {_SIMPL_TEXT[_k]}",
        (lambda t=_text: al.simplify(t)), _erw,
        "Implikation auflösen, De Morgan, Kommutativität, Absorption, Inverse, Dominanz — immer mit Regelname.",
        source=f"Serie 2, Aufgabe 7{_k}")

# -- Beispiele aus der Vorlesung -------------------------------------------------------------------------------
_ex("SW2.B1", "Wahrheitstabelle von (Q ⇒ P) ∧ ¬P", "Stellen Sie die Wahrheitstabelle auf (Signatur {P, Q}).",
    lambda: al.truth_table("(Q -> P) and not P", true_first=True),
    [E("Spalten", lambda r: r.table.header, ["P", "Q", "Q → P", "¬P", "(Q → P) ∧ ¬P"], "Serie 2, Seite 6"),
     E("Zeilen (1 1 zuerst, wie in der Serie)", lambda r: r.table.rows,
       [[str(int(P)), str(int(Q)), str(int(_imp(Q, P))), str(int(not P)), str(int(_imp(Q, P) and not P))]
        for P, Q in itertools.product([True, False], repeat=2)], "unabhängig: Python"),
     E("Klasse", lambda r: r.classification, "erfüllbar (kontingent)", "Mitschrift: nur die letzte Zeile ist wahr")],
    "Teilterme von innen nach aussen.", source="Serie 2, Frage 3")
_ex("SW2.B2", "(Q ⇒ P) ∧ ¬P ≡ ¬(P ∨ Q)", "Zeigen Sie die Äquivalenz mit einer Wahrheitstabelle.",
    lambda: al.check_equivalence("(Q -> P) & ~P", "~(P | Q)", true_first=True),
    [E("äquivalent", lambda r: r.equivalent,
       not _gegenbeispiele(lambda P, Q: _imp(Q, P) and not P, lambda P, Q: not (P or Q), "PQ"), "unabhängig: Python")],
    "Zwei Terme sind äquivalent, wenn sie für alle Interpretationen gleich sind.", source="Serie 2, Frage 3")
_VERKNUEPFUNGEN = {
    "Negation": ("A", lambda a, b=None: not a),
    "Konjunktion": ("A, B", lambda a, b: a and b),
    "Disjunktion": ("A, B", lambda a, b: a or b),
    "Kontravalenz": ("A, B", lambda a, b: a != b),
    "Implikation": ("A, B", lambda a, b: _imp(a, b)),
    "Äquivalenz": ("A, B", lambda a, b: a == b),
}
_ex("SW2.B3", "Wahrheitstabellen der Junktoren", "Geben Sie die Wahrheitstabellen von ¬, ∧, ∨, ⊕, ⇒, ⇔ an.",
    lambda: al.connective_tables(),
    [E(f"{name}: letzte Spalte",
       lambda r, name=name: [row[-1] for row in r.tables[name].rows],
       [str(int(f(*v))) for v in itertools.product([False, True], repeat=1 if name == "Negation" else 2)],
       "unabhängig: Python-Operatoren")
     for name, (_, f) in _VERKNUEPFUNGEN.items()],
    "Merke: A ⇒ B ist nur falsch, wenn A wahr und B falsch ist.", source="Serie 2, Frage 2")
_ex("SW2.B4", "Vereinfachen: (¬(¬P ∨ ¬Q) ∧ (P ⇒ Q)) ⇒ P", "Zeigen Sie mit den Rechengesetzen, dass der Term allgemeingültig ist.",
    lambda: al.simplify("(~(~P | ~Q) & (P -> Q)) -> P"),
    [E("allgemeingültig", lambda r: r.is_tautology, all(_vektor(lambda P, Q: _imp((not ((not P) or (not Q))) and _imp(P, Q), P), 2)),
       "unabhängig: Python-Tabelle; Serie 2, Seite 12"),
     E("benutzt De Morgan", lambda r: any(s.rule == "De Morgan" for s in r.steps), True),
     E("benutzt Implikation auflösen", lambda r: any(s.rule == "Implikation" for s in r.steps), True)],
    "Die Rechenregeln: De Morgan, Implikation, Kommutativität, Absorption, Inverse, Dominanz.",
    source="Serie 2, Frage 4 (Beispiel)")
_ex("SW2.B5", "A ⊕ B ≡ ¬(A ⇔ B) und A ⇔ B ≡ (A ⇒ B) ∧ (B ⇒ A)", "Bestätigen Sie die beiden Zusammenhänge der Junktoren.",
    lambda: Sammlung("Zusammenhänge der Junktoren", {
        "A ⊕ B ≡ ¬(A ⇔ B)": al.check_equivalence("A ^ B", "~(A <-> B)"),
        "A ⇔ B ≡ (A ⇒ B) ∧ (B ⇒ A)": al.check_equivalence("A <-> B", "(A -> B) & (B -> A)")}),
    [E("⊕ ist die Negation von ⇔", lambda r: r.teile["A ⊕ B ≡ ¬(A ⇔ B)"].equivalent,
       not _gegenbeispiele(lambda A, B: A != B, lambda A, B: not (A == B), "AB")),
     E("⇔ ist Implikation in beide Richtungen", lambda r: r.teile["A ⇔ B ≡ (A ⇒ B) ∧ (B ⇒ A)"].equivalent,
       not _gegenbeispiele(lambda A, B: A == B, lambda A, B: _imp(A, B) and _imp(B, A), "AB"))],
    "Mitschrift »Merke«.", source="Serie 2, Zusammenhänge der Junktoren")
_ex("SW2.B6", "Erfüllbarkeit (SAT)", "Finden Sie eine Interpretation, die (A ∨ B) ∧ ¬A wahr macht.",
    lambda: al.sat("(A | B) & ~A"),
    [E("erfüllbar", lambda r: r.satisfiable, True),
     E("Modell", lambda r: r.model, {"A": False, "B": True}, "unabhängig: nur A=0, B=1 macht den Term wahr")],
    "Im schlimmsten Fall muss man alle 2^n Zeilen prüfen — das ist das SAT-Problem.", source="Serie 2, Frage 4")

# -- Lerngerüst Serie 2 ----------------------------------------------------------------------------------------------
_LG2A = "A & ~(B -> (P | Q)) & C"
_ex("LG2.a", "Syntaxbaum von A ∧ ¬(B ⇒ (P ∨ Q)) ∧ C", "Zeichnen Sie den Syntaxbaum von A ∧ (¬(B ⇒ (P ∨ Q)) ∧ C) — ∧ bindet von links nach rechts.",
    lambda: al.syntax_tree(_LG2A),
    [E("Anzahl Knoten", lambda r: r.size, _knoten(_LG2A), "unabhängig"),
     E("Ebenen (Tiefe)", lambda r: r.depth, 6, "von Hand: ∧ / ∧ C / A ¬ / ⇒ / B ∨ / P Q"),
     E("Blätter", lambda r: r.leaves, re.findall(r"[A-Z]", _LG2A), "unabhängig")],
    "Das Lerngerüst schreibt A ∧ ¬(…) ∧ C ohne Klammer um die ersten beiden Terme: links geklammert.",
    source="Lerngerüst Serie 2 (a)")
_ex("LG2.b", "Signatur und Interpretationen", "Was ist die Signatur von A ∧ ¬(B ⇒ (P ∨ Q)) ∧ C, und wie viele Interpretationen hat sie?",
    lambda: al.signature(_LG2A),
    [E("Signatur", lambda r: r.names, sorted(set(re.findall(r"[A-Z]", _LG2A))), "unabhängig: Buchstaben im Term"),
     E("Interpretationen 2⁵", lambda r: r.count, 2 ** len(set(re.findall(r"[A-Z]", _LG2A))))],
    "Anzahl der verschiedenen atomaren Aussagen n; Interpretationen: 2^n.", source="Lerngerüst Serie 2 (b)")
_LG2C = {
    "c1": ("((P -> Q) & (~P -> Q)) <-> Q", "((P ⇒ Q) ∧ (¬P ⇒ Q)) ⇔ Q", 2,
           lambda P, Q: _iff(_imp(P, Q) and _imp(not P, Q), Q)),
    "c2": ("(~(A | ~B) & (A -> B)) -> A", "(¬(A ∨ ¬B) ∧ (A ⇒ B)) ⇒ A", 2,
           lambda A, B: _imp((not (A or not B)) and _imp(A, B), A)),
    "c3": ("R & ((S -> ~R) <-> (S | ~R))", "R ∧ ((S ⇒ ¬R) ⇔ (S ∨ ¬R))", 2,
           lambda R, S: R and _iff(_imp(S, not R), S or not R)),
}
for _k, (_text, _show, _n, _py) in _LG2C.items():
    _ex(f"LG2.{_k}", f"Klassifizieren: {_show}", f"Ist {_show} erfüllbar, allgemeingültig oder unerfüllbar?",
        (lambda t=_text: al.classify(t)),
        [E("Klasse", lambda r: r.classification, _klasse(_py, _n), "unabhängig: Python-Tabelle"),
         E("Anzahl erfüllender Zeilen", lambda r: len(r.models), _modelle(_py, _n))],
        "Wahrheitstabelle Schritt für Schritt.", source="Lerngerüst Serie 2 (c)")
_LG2D = {
    "d1": ("~Q & (Q -> P)", "~(Q | ~P)", "¬Q ∧ (Q ⇒ P) und ¬(Q ∨ ¬P)", "PQ",
           lambda P, Q: (not Q) and _imp(Q, P), lambda P, Q: not (Q or not P)),
    "d2": ("B -> (B -> ~A)", "~(A & B)", "B ⇒ (B ⇒ ¬A) und ¬(A ∧ B)", "AB",
           lambda A, B: _imp(B, _imp(B, not A)), lambda A, B: not (A and B)),
}
for _k, (_f, _g, _show, _namen, _pf, _pg) in _LG2D.items():
    _ex(f"LG2.{_k}", f"Äquivalenz: {_show}", f"Sind {_show} äquivalent?",
        (lambda f=_f, g=_g: al.check_equivalence(f, g)),
        [E("äquivalent", lambda r: r.equivalent, not _gegenbeispiele(_pf, _pg, _namen), "unabhängig: Python-Tabelle"),
         E("Gegenbeispiele", lambda r: sorted(tuple(sorted(d.items())) for d in r.counterexamples),
           _gegenbeispiele(_pf, _pg, _namen))],
        "Beide Terme Zeile für Zeile vergleichen.", source="Lerngerüst Serie 2 (d)")
_LG2E = {
    "e1": ("(A & ~B) | (A & ~(B | C)) | (B & ~(B | C))", "(A ∧ ¬B) ∨ (A ∧ ¬(B ∨ C)) ∨ (B ∧ ¬(B ∨ C))", "ABC",
           lambda A, B, C: (A and not B) or (A and not (B or C)) or (B and not (B or C)), "A ∧ ¬B"),
    "e2": ("(~(~A | ~B) & (A -> B)) -> A", "(¬(¬A ∨ ¬B) ∧ (A ⇒ B)) ⇒ A", "AB",
           lambda A, B: _imp((not ((not A) or (not B))) and _imp(A, B), A), None),
    "e3": ("((~P | ~Q) & (Q -> P)) -> (P & ~Q)", "((¬P ∨ ¬Q) ∧ (Q ⇒ P)) ⇒ (P ∧ ¬Q)", "PQ",
           lambda P, Q: _imp(((not P) or (not Q)) and _imp(Q, P), P and not Q), "P ∨ Q"),
    "e4": ("((A -> B) -> (B -> C)) -> C", "((A ⇒ B) ⇒ (B ⇒ C)) ⇒ C", "ABC",
           lambda A, B, C: _imp(_imp(_imp(A, B), _imp(B, C)), C), "B ∨ C"),
}
for _k, (_text, _show, _namen, _py, _ziel) in _LG2E.items():
    _soll_vec = _vektor(_py, len(_namen))
    _erw = [E("Ergebnis ist äquivalent zum Ausgangsterm", lambda r, n=list(_namen): _vec_dk(r.result, n), _soll_vec,
              "unabhängig: Python-Tabelle des Ausgangsterms")]
    if _ziel is not None:
        _erw.append(E("vereinfachter Term", lambda r: r.result.to_str("unicode"), _ziel, "von Hand vereinfacht"))
    else:
        _erw.append(E("allgemeingültig", lambda r: r.is_tautology, all(_soll_vec), "unabhängig: Python-Tabelle"))
    _ex(f"LG2.{_k}", f"Vereinfachen: {_show}", f"Vereinfachen Sie mit den Rechengesetzen Schritt für Schritt: {_show}",
        (lambda t=_text: al.simplify(t)), _erw,
        "Auf dem Lerngerüst steht bei e4 eine Klammer um (A ⇒ B) ⇒ (B ⇒ C): ((A ⇒ B) ⇒ (B ⇒ C)) ⇒ C.",
        source="Lerngerüst Serie 2 (e)")


# ==========================================================================
# Serie 3 — Prädikatenlogik
# ==========================================================================
# Unendliche Universen (ℕ, ℤ, ℝ) kann ein Programm nur in einem endlichen Ausschnitt prüfen. Ein Zeuge oder
# Gegenbeispiel ist ein echter Beweis, »nichts gefunden« ist keiner. Die Soll-Werte sind darum die
# mathematisch richtigen Antworten, die Ausschnitte (bis 200 usw.) enthalten das entscheidende Beispiel.

# -- Aufgabe 1: Fermat ----------------------------------------------------------------------------------------
def _fermat_ok(e):
    return e["a"] ** e["n"] + e["b"] ** e["n"] == e["c"] ** e["n"]


_ex("SW3.1", "Prädikat Fermat(a, b, c, n)", "Finden Sie Werte a, b, c, n aus ℕ, so dass drei wahre und drei falsche Aussagen entstehen.",
    lambda: pr.find_examples(pr.FERMAT, pr.naturals(13, start=1), 3, 3),
    [E("drei wahre Beispiele", lambda r: len(r.true_examples), 3),
     E("alle wahren Beispiele stimmen (aⁿ + bⁿ = cⁿ)", lambda r: all(_fermat_ok(e) for e in r.true_examples), True,
       "unabhängig: Python-Arithmetik"),
     E("drei falsche Beispiele", lambda r: len(r.false_examples), 3),
     E("alle falschen Beispiele stimmen (aⁿ + bⁿ ≠ cⁿ)", lambda r: not any(_fermat_ok(e) for e in r.false_examples), True,
       "unabhängig: Python-Arithmetik")],
    "Mit n = 1 oder n = 2 geht es (3² + 4² = 5²); für n ≥ 3 gibt es keine wahre Aussage (Wiles).", source="Serie 3, Aufgabe 1")

# -- Aufgabe 2: Quantoren ohne Quantor ------------------------------------------------------------------------------
_ex("SW3.2a", "∀x ∈ U: P(x) ohne Quantor", "Schreiben Sie ∀x ∈ U: P(x) ohne Quantor und geben Sie einen Algorithmus an.",
    lambda: Sammlung("∀ ohne Quantor", {
        "allgemein (U = {x₁, …, xₙ})": pr.unroll("forall", [1, 2, 3]),
        "konkret: x < 4 über {0, 1, 2, 3}": pr.unroll("forall", range(4), "x < 4")}),
    [E("Konjunktion P(x₁) ∧ … ∧ P(xₙ)", lambda r: "∧" in r.teile["allgemein (U = {x₁, …, xₙ})"].expression, True,
       "Serie: P(x₁) ∧ P(x₂) ∧ … ∧ P(xₙ)"),
     E("Algorithmus gibt beim ersten Gegenbeispiel False zurück", lambda r: "return False" in r.teile["allgemein (U = {x₁, …, xₙ})"].algorithm, True,
       "Mitschrift: wenn return True kommt, ist es der Allquantor"),
     E("konkreter Wahrheitswert", lambda r: r.werte["konkret: x < 4 über {0, 1, 2, 3}"], all(x < 4 for x in range(4)),
       "unabhängig: Python all()")],
    "Alle Elemente durchlaufen; findet sich eines, für das P nicht gilt, ist die Aussage falsch.", source="Serie 3, Aufgabe 2a")
_ex("SW3.2b", "∃x ∈ U: P(x) ohne Quantor", "Schreiben Sie ∃x ∈ U: P(x) ohne Quantor und geben Sie einen Algorithmus an.",
    lambda: Sammlung("∃ ohne Quantor", {
        "allgemein (U = {x₁, …, xₙ})": pr.unroll("exists", [1, 2, 3]),
        "konkret: x·x = 9 über {0, …, 4}": pr.unroll("exists", range(5), "x * x == 9")}),
    [E("Disjunktion P(x₁) ∨ … ∨ P(xₙ)", lambda r: "∨" in r.teile["allgemein (U = {x₁, …, xₙ})"].expression, True,
       "Serie: P(x₁) ∨ P(x₂) ∨ … ∨ P(xₙ)"),
     E("Algorithmus gibt beim ersten Treffer True zurück", lambda r: "return True" in r.teile["allgemein (U = {x₁, …, xₙ})"].algorithm, True),
     E("konkreter Wahrheitswert", lambda r: r.werte["konkret: x·x = 9 über {0, …, 4}"], any(x * x == 9 for x in range(5)),
       "unabhängig: Python any()")],
    "Alle Elemente durchlaufen; beim ersten Treffer ist die Aussage wahr.", source="Serie 3, Aufgabe 2b (in der Mitschrift noch offen)")

# -- Aufgabe 3: Wahrheitswerte ----------------------------------------------------------------------------------------
_ex("SW3.3a", "∀x ∈ ℕ: (x gerade ⇒ x + 1 ungerade)", "Bestimmen Sie den Wahrheitswert.",
    lambda: pr.forall(pr.naturals(300), "x % 2 != 0 or (x + 1) % 2 == 1"),
    [E("Wahrheitswert", lambda r: r.value, True, "mathematisch: gerade + 1 ist ungerade"),
     E("auch im Ausschnitt 0…300 kein Gegenbeispiel", lambda r: r.value, all(x % 2 != 0 or (x + 1) % 2 == 1 for x in range(301)),
       "unabhängig: Python all()")],
    "Für ungerade x ist die Prämisse falsch und die Implikation damit wahr (F ⇒ … ist wahr).", source="Serie 3, Aufgabe 3a")
_ex("SW3.3b", "∃x ∈ ℕ: (x prim ∧ x > 100 ∧ x gerade)", "Bestimmen Sie den Wahrheitswert.",
    lambda: pr.exists(pr.naturals(1000), "is_prime(x) and x > 100 and x % 2 == 0"),
    [E("Wahrheitswert", lambda r: r.value, False, "mathematisch: die einzige gerade Primzahl ist 2"),
     E("im Ausschnitt kein Zeuge", lambda r: any(_ist_prim(x) and x > 100 and x % 2 == 0 for x in range(1001)), False,
       "unabhängig: Python any()"),
     E("kein Beweis (nur Ausschnitt)", lambda r: r.certain, False, "∃ über ℕ: »nichts gefunden« beweist nichts")],
    "Gerade Zahlen über 2 sind durch 2 teilbar, also nicht prim.", source="Serie 3, Aufgabe 3b")
_ex("SW3.3c", "∃x ∈ ℤ: x² < x", "Bestimmen Sie den Wahrheitswert.",
    lambda: pr.exists(pr.integers(60), "x * x < x"),
    [E("Wahrheitswert", lambda r: r.value, False, "mathematisch: x² ≥ x für alle ganzen Zahlen"),
     E("im Ausschnitt kein Zeuge", lambda r: any(x * x < x for x in range(-60, 61)), False, "unabhängig: Python any()")],
    "x² < x heisst x(x − 1) < 0, also 0 < x < 1 — keine ganze Zahl.", source="Serie 3, Aufgabe 3c")
_ex("SW3.3d", "∃x ∈ ℝ: x² < x", "Bestimmen Sie den Wahrheitswert.",
    lambda: pr.exists(pr.reals(2, 4), "x * x < x"),
    [E("Wahrheitswert", lambda r: r.value, True, "mathematisch: x = 1/2"),
     E("Zeuge erfüllt x² < x (exakt gerechnet)", lambda r: r.witness["x"] * r.witness["x"] < r.witness["x"], True,
       "unabhängig: Bruchrechnung")],
    "Probieren Sie Brüche zwischen 0 und 1.", source="Serie 3, Aufgabe 3d")
_ex("SW3.3e", "∀x ∈ ℕ: (x² gerade ⇔ x gerade)", "Bestimmen Sie den Wahrheitswert.",
    lambda: pr.forall(pr.naturals(300), pr.Predicate.from_expr("a", "x * x % 2 == 0").iff(pr.Predicate.from_expr("b", "x % 2 == 0"))),
    [E("Wahrheitswert", lambda r: r.value, True, "mathematisch: gerade·gerade = gerade, ungerade·ungerade = ungerade"),
     E("im Ausschnitt 0…300", lambda r: all((x * x % 2 == 0) == (x % 2 == 0) for x in range(301)), True,
       "unabhängig: Python all()")],
    "Beide Richtungen prüfen: gerade → gerade und ungerade → ungerade.", source="Serie 3, Aufgabe 3e")
_A3F = [3, 2, 1, 1, 4, 4, 2, 3, 1]
_ex("SW3.3f", "∀i ∃j: ¬(i = j) ∧ a[i] = a[j]", "Liste a = [3, 2, 1, 1, 4, 4, 2, 3, 1]: Kommt jedes Element mehrfach vor?",
    lambda: pr.occurs_multiple(_A3F),
    [E("Wahrheitswert", lambda r: r.value, all(_A3F.count(v) >= 2 for v in _A3F), "unabhängig: Python count()"),
     E("81 Paare (i, j) ausgewertet", lambda r: len(r.evaluations), 9 * 9),
     E("wahre Paare", lambda r: sum(1 for _, v in r.evaluations if v),
       sum(1 for i in range(9) for j in range(9) if i != j and _A3F[i] == _A3F[j]), "unabhängig: Doppelschleife")],
    "Für jede Position i ein Partner j an anderer Stelle mit demselben Wert.", source="Serie 3, Aufgabe 3f")
_ex("SW3.B4", "Paare (i = 1, j = 8) und (i = 2, j = 7)", "Rechnen Sie Q(i, j) = ¬(i = j) ∧ a[i] = a[j] für die beiden Paare aus (Index ab 1).",
    lambda: pr.has_duplicate(_A3F, pairs=[(1, 8), (2, 7)]),
    [E("Q(1, 8): a[1] = 3 = a[8]", lambda r: r.pair, (1, 8) if _A3F[0] == _A3F[7] else None, "unabhängig: Python"),
     E("es gibt Duplikate", lambda r: r.value, len(set(_A3F)) < len(_A3F), "unabhängig: Python set()"),
     E("Q(2, 7): a[2] = 2 = a[7]", lambda r: _A3F[1] == _A3F[6], True)],
    "Mitschrift: »Bitte ausrechnen«. Index ab 1.", source="Serie 3, Aufgabe 3f (Beispiele)")

# -- Aufgabe 4: Reihenfolge der Quantoren -----------------------------------------------------------------------------------
_LEUTE = ["Anna", "Ben", "Carl"]
_MAG = {("Anna", "Ben"), ("Ben", "Carl"), ("Carl", "Anna")}
_ex("SW3.4", "Everybody Loves Somebody", "Beschreiben Sie ∀x ∃y x liebt y und ∃y ∀x x liebt y in natürlicher Sprache — und prüfen Sie beide an einem Beispiel.",
    lambda: pr.everybody_loves_somebody(_LEUTE, _MAG),
    [E("∀x ∃y (jeder liebt jemanden)", lambda r: r.original.value, all(any((x, y) in _MAG for y in _LEUTE) for x in _LEUTE),
       "unabhängig: Python all/any"),
     E("∃y ∀x (jemand wird von allen geliebt)", lambda r: r.swapped.value, any(all((x, y) in _MAG for x in _LEUTE) for y in _LEUTE),
       "unabhängig: Python any/all"),
     E("die beiden Aussagen sind verschieden", lambda r: r.differ, True, "Mitschrift: zwei völlig unterschiedliche Behauptungen"),
     E("∃y ∀x ⇒ ∀x ∃y", lambda r: (not r.swapped.value) or r.original.value, True)],
    "Reihenfolge verschiedener Quantoren ist wichtig; gleiche dürfen vertauscht werden.", source="Serie 3, Aufgabe 4")

# -- Aufgabe 5: Monitore ------------------------------------------------------------------------------------------------------
MONITORE = {
    "A": [[1] * 6 for _ in range(6)],
    "B": [[0] * 6, [0] * 6, [0, 1, 0, 0, 0, 0], [0] * 6, [0] * 6, [0] * 6],
    "C": [[0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0] * 6, [0, 0, 0, 1, 0, 1]],
    "D": [[0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0], [0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 0], [0, 1, 0, 1, 0, 0]],
    "E": [[0] * 6, [0] * 6, [1] * 6, [0] * 6, [0] * 6, [0] * 6],
    "F": [[0, 0, 0, 0, 1, 0] for _ in range(6)],
}
# Notebook (Serie03.ipynb) und Tabelle der aufbereiteten Mitschrift: je Aussage die Werte für A, B, C, D, E, F.
_NOTEBOOK = {
    "a": [True, False, False, False, False, False], "b": [True, False, False, False, False, False],
    "c": [True] * 6, "d": [True] * 6,
    "e": [True, False, False, True, False, True], "f": [True, False, True, False, True, False],
    "g": [True, False, False, False, True, False], "h": [True, False, False, False, False, True],
}
_AUSSAGE5 = {"a": "∀i ∀j", "b": "∀j ∀i", "c": "∃i ∃j", "d": "∃j ∃i", "e": "∀i ∃j", "f": "∀j ∃i", "g": "∃i ∀j", "h": "∃j ∀i"}
_REF5 = {   # unabhängige Referenz mit reinem Python (Indizes ab 0)
    "a": lambda M: all(M[i][j] == 1 for i in range(6) for j in range(6)),
    "b": lambda M: all(M[i][j] == 1 for j in range(6) for i in range(6)),
    "c": lambda M: any(M[i][j] == 1 for i in range(6) for j in range(6)),
    "d": lambda M: any(M[i][j] == 1 for j in range(6) for i in range(6)),
    "e": lambda M: all(any(M[i][j] == 1 for j in range(6)) for i in range(6)),
    "f": lambda M: all(any(M[i][j] == 1 for i in range(6)) for j in range(6)),
    "g": lambda M: any(all(M[i][j] == 1 for j in range(6)) for i in range(6)),
    "h": lambda M: any(all(M[i][j] == 1 for i in range(6)) for j in range(6)),
}


class MonitorErgebnis(Result):
    """Wahrheitswert einer Quantorenfolge für die Monitore A bis F, mit Zeuge oder Gegenbeispiel."""

    def __init__(self, buchstabe: str):
        super().__init__()
        self.buchstabe = buchstabe
        self.aussage = _AUSSAGE5[buchstabe]
        self.title = f"Aufgabe 5{buchstabe}: {self.aussage}: P(M, i, j) für die Monitore A bis F"
        self.ergebnisse = {name: pr.matrix_quantors(M)[buchstabe] for name, M in MONITORE.items()}
        self.werte = {name: r.value for name, r in self.ergebnisse.items()}
        self.summary = "Wahr für Monitor " + (", ".join(n for n, v in self.werte.items() if v) or "(keinen)") + "."
        self.steps.append(Step("Aussage", f"{self.aussage}: P(M, i, j) mit P(M, i, j): m_ij = 1; Index ab 1.", rule="Quantoren auswerten"))
        rows = []
        for name, r in self.ergebnisse.items():
            beleg = r.witness if r.witness else r.counterexample
            art = "Zeuge" if r.witness else ("Gegenbeispiel" if r.counterexample else "")
            rows.append([name, "wahr" if r.value else "falsch", (f"{art}: " + ", ".join(f"{k} = {v}" for k, v in beleg.items())) if beleg else ""])
        self.steps.append(Step("Wahrheitswert je Monitor", "Jede Matrix ist 6×6.", table=Table(["Monitor", "Aussage", "Beleg"], rows)))
        self.highlights = [(f"Monitor {n}", "wahr" if v else "falsch") for n, v in self.werte.items()]


for _k in "abcdefgh":
    _ex(f"SW3.5{_k}", f"Monitore: {_AUSSAGE5[_k]} P(M, i, j)",
        f"Bestimmen Sie alle Matrizen A bis F, für die {_AUSSAGE5[_k]}: P(M, i, j) wahr ist (P: m_ij = 1).",
        (lambda k=_k: MonitorErgebnis(k)),
        [E(f"Werte für A, B, C, D, E, F", lambda r: [r.werte[n] for n in "ABCDEF"], _NOTEBOOK[_k],
           "Notebook zur Serie (Mitschrift, Tabelle)"),
         E("unabhängige Referenz (Python all/any)", lambda r, k=_k: [r.werte[n] for n in "ABCDEF"],
           [_REF5[_k](MONITORE[n]) for n in "ABCDEF"], "Python all()/any() über die Matrizen")],
        {"a": "Alle Einträge 1: nur Monitor A.", "b": "Gleiche Quantoren darf man vertauschen — wie a).",
         "c": "Mindestens eine 1: jeder Monitor, der nicht ganz dunkel ist.", "d": "Wie c).",
         "e": "Jede Zeile enthält mindestens eine 1.", "f": "Jede Spalte enthält mindestens eine 1.",
         "g": "Es gibt eine Zeile nur aus Einsen.", "h": "Es gibt eine Spalte nur aus Einsen."}[_k],
        source=f"Serie 3, Aufgabe 5{_k}")

# -- Aufgabe 6: Negation hinter die Quantoren --------------------------------------------------------------------------------------
_NEG6 = {
    "a": ("¬∃y ∈ R: ∀x ∈ R: x < y", "∀y ∈ R: ∃x ∈ R: x ≥ y", "¬∃ wird ∀¬, ¬∀ wird ∃¬, ¬(x < y) wird x ≥ y"),
    "b": ("¬∀x ∈ Z: ∃y ∈ Z: x = y + 1", "∃x ∈ Z: ∀y ∈ Z: x ≠ y + 1", "¬∀ wird ∃¬, ¬∃ wird ∀¬, ¬(x = y + 1) wird x ≠ y + 1"),
    "c": ("¬∃x ∈ U: ∀y ∈ U: ∃z ∈ U: P(x, y, z)", "∀x ∈ U: ∃y ∈ U: ∀z ∈ U: ¬P(x, y, z)", "Quantoren kehren sich alle um, ¬ steht am Schluss vor P"),
}
for _k, (_vorher, _nachher, _regel) in _NEG6.items():
    _ex(f"SW3.6{_k}", f"Negation: {_vorher}", f"Bringen Sie die Negation hinter alle Quantoren: {_vorher}",
        (lambda t=_vorher: pr.push_negation(t)),
        [E("Negation hinter den Quantoren", lambda r: str(r.result), _nachher, "von Hand: " + _regel)],
        "¬∀ ≡ ∃¬ und ¬∃ ≡ ∀¬; ¬(x < y) ≡ x ≥ y.", source=f"Serie 3, Aufgabe 6{_k}")

# -- Aufgabe 7: Übersetzen und negieren ---------------------------------------------------------------------------------------------
_S7 = ("Für jeden iPhone-Benutzer u aus U gibt es eine iPhone-App a aus A, sodass für jeden Freund f aus U gilt "
       "Freund(u, f) → Geladen(f, a)")
_ex("SW3.7", "iPhone-Benutzer und Apps", "Übersetzen Sie »Für jeden iPhone-Benutzer gibt es eine iPhone-App, die jeder seiner "
    "iPhone-benutzenden Freunde heruntergeladen hat« in die Prädikatenlogik und negieren Sie den Satz.",
    lambda: pr.to_formula(_S7),
    [E("Formel", lambda r: r.formula, "∀u ∈ U: ∃a ∈ A: ∀f ∈ U: (Freund(u, f) ⇒ Geladen(f, a))", "von Hand übersetzt"),
     E("Negation", lambda r: str(r.negation().result), "∃u ∈ U: ∀a ∈ A: ∃f ∈ U: (Freund(u, f) ∧ ¬Geladen(f, a))",
       "von Hand: ¬∀ ≡ ∃¬, ¬∃ ≡ ∀¬, ¬(P ⇒ Q) ≡ P ∧ ¬Q")],
    "Brauchen Sie die Bezeichnungen U für iPhone users und A für apps.", source="Serie 3, Aufgabe 7")

# -- Beispiele aus der Vorlesung -------------------------------------------------------------------------------------------------------
_ex("SW3.B1", "Prädikate werden zu Aussagen", "Bestimmen Sie sumEqualToFive(1, 3), sumEqualToFive(2, 3) und Pal(REITTIER).",
    lambda: Sammlung("Prädikate einsetzen", {
        "sumEqualToFive(1, 3)": pr.SUM_EQUAL_TO_FIVE.evaluate(1, 3),
        "sumEqualToFive(2, 3)": pr.SUM_EQUAL_TO_FIVE.evaluate(2, 3),
        "Pal(REITTIER)": pr.PAL.evaluate("REITTIER")}),
    [E("sumEqualToFive(1, 3) hat Wahrheitswert 0", lambda r: r.werte["sumEqualToFive(1, 3)"], 1 + 3 == 5, "Serie 3, Seite 2"),
     E("sumEqualToFive(2, 3)", lambda r: r.werte["sumEqualToFive(2, 3)"], 2 + 3 == 5),
     E("Pal(REITTIER) hat Wahrheitswert 1", lambda r: r.werte["Pal(REITTIER)"], "REITTIER" == "REITTIER"[::-1], "Serie 3, Seite 2")],
    "Erst wenn alle Lücken gefüllt sind, entsteht eine Aussage.", source="Serie 3, Frage 1")
_ex("SW3.B2", "¬∀x P(x) ≡ ∃x ¬P(x)", "Negieren Sie ∀x ∈ {0,…,9}: x gerade — und prüfen Sie beide Seiten.",
    lambda: pr.negate_quantifier("∀x", "x % 2 == 0", universes={"x": range(10)}),
    [E("∀x gerade ist falsch", lambda r: r.original.value, all(x % 2 == 0 for x in range(10)), "unabhängig: Python all()"),
     E("∃x ¬gerade ist wahr (Gegenbeispiel x = 1)", lambda r: r.negated.value, any(x % 2 != 0 for x in range(10))),
     E("Negation von ∀ ist ∃¬", lambda r: r.formula_negated.startswith("∃x"), True, "Serie 3, Frage 3"),
     E("beide Seiten stimmen überein", lambda r: r.consistent, True)],
    "Ein einziges Gegenbeispiel widerlegt eine ∀-Aussage.", source="Serie 3, Frage 3")
_ex("SW3.B3", "¬∃x P(x) ≡ ∀x ¬P(x)", "Negieren Sie ∃x ∈ {0,…,9}: x > 100 — und prüfen Sie beide Seiten.",
    lambda: pr.negate_quantifier("∃x", "x > 100", universes={"x": range(10)}),
    [E("∃x x > 100 ist falsch", lambda r: r.original.value, any(x > 100 for x in range(10))),
     E("∀x ¬(x > 100) ist wahr", lambda r: r.negated.value, all(not x > 100 for x in range(10))),
     E("Negation von ∃ ist ∀¬", lambda r: r.negated.formula.startswith("∀x"), True, "Serie 3, Frage 3")],
    "Um ∃ zu widerlegen, muss man alle Werte prüfen.", source="Serie 3, Frage 3")

# -- Lerngerüst Serie 3 -----------------------------------------------------------------------------------------------------------------
def _p_lg3a(x, y, n):
    return x < y and x ** n > y


_ex("LG3.a", "P(x, y, n): x < y und xⁿ > y", "Finden Sie Werte x, y, n aus ℕ, so dass eine wahre bzw. eine falsche Aussage entsteht.",
    lambda: pr.find_examples(pr.Predicate.from_expr("P(x, y, n)", "x < y and x**n > y"), pr.naturals(8, start=1), 2, 2),
    [E("wahre Beispiele", lambda r: len(r.true_examples) >= 1, True),
     E("alle wahren Beispiele stimmen", lambda r: all(_p_lg3a(e["x"], e["y"], e["n"]) for e in r.true_examples), True,
       "unabhängig: Python-Arithmetik"),
     E("falsche Beispiele", lambda r: len(r.false_examples) >= 1, True),
     E("alle falschen Beispiele stimmen", lambda r: not any(_p_lg3a(e["x"], e["y"], e["n"]) for e in r.false_examples), True,
       "unabhängig: Python-Arithmetik")],
    "Zum Beispiel x = 2, y = 3, n = 2 (2 < 3 und 4 > 3).", source="Lerngerüst Serie 3 (a)")
_LG3B = {
    "b1": ("∀x ∈ ℤ: (x durch 12 teilbar ⇒ x durch 8 teilbar)", "forall", lambda: pr.forall(pr.integers(60), "not divides(12, x) or divides(8, x)"),
           False, lambda x: x % 12 != 0 or x % 8 == 0, "x = 12 ist ein Gegenbeispiel"),
    "b2": ("∃x ∈ ℤ: (x durch 12 teilbar ⇒ x durch 8 teilbar)", "exists", lambda: pr.exists(pr.integers(60), "not divides(12, x) or divides(8, x)"),
           True, lambda x: x % 12 != 0 or x % 8 == 0, "x = 1: die Prämisse ist falsch, die Implikation damit wahr"),
    "b3": ("∀x ∈ ℕ: (x negativ ⇒ x gerade)", "forall", lambda: pr.forall(pr.naturals(300), "not (x < 0) or x % 2 == 0"),
           True, lambda x: not (x < 0) or x % 2 == 0, "in ℕ gibt es keine negativen Zahlen: die Prämisse ist immer falsch"),
    "b4": ("∀x ∈ ℕ: (x gerade ⇒ x durch 4 teilbar)", "forall", lambda: pr.forall(pr.naturals(300), "x % 2 != 0 or x % 4 == 0"),
           False, lambda x: x % 2 != 0 or x % 4 == 0, "x = 2 ist ein Gegenbeispiel"),
}
for _k, (_text, _art, _call, _soll, _py, _hint) in _LG3B.items():
    _bereich = range(-60, 61) if _k in ("b1", "b2") else range(0, 301)
    _ex(f"LG3.{_k}", f"Wahrheitswert: {_text}", f"Bestimmen Sie den Wahrheitswert von {_text}.", _call,
        [E("Wahrheitswert", lambda r: r.value, _soll, "mathematisch richtig"),
         E("im Ausschnitt gerechnet", lambda r: r.value, (all if _art == "forall" else any)(_py(x) for x in _bereich),
           "unabhängig: Python all()/any()")],
        _hint, source="Lerngerüst Serie 3 (b)")
_LG3C = {
    "c1": ("∀x ∈ ℕ: ∃y ∈ ℕ: x > y", "∀x ∃y", "N", "N", lambda x, y: x > y, 0, "∃x ∈ N: ∀y ∈ N: x ≤ y"),
    "c2": ("∀x ∈ ℕ: ∃y ∈ ℤ: x > y", "∀x ∃y", "N", "Z", lambda x, y: x > y, 0, "∃x ∈ N: ∀y ∈ Z: x ≤ y"),
    "c3": ("∃x ∈ ℕ: ∀y ∈ ℤ: x > y", "∃x ∀y", "N", "Z", lambda x, y: x > y, 0, "∀x ∈ N: ∃y ∈ Z: x ≤ y"),
    "c4": ("∃y ∈ ℤ: ∀x ∈ ℕ: x > y", "∃y ∀x", "Z", "N", lambda y, x: x > y, 0, "∀y ∈ Z: ∃x ∈ N: x ≤ y"),
}


def _lg3c_werte(k):
    """Wahrheitswert von Hand/Python: Ausschnitt ℕ = 0…30, ℤ = −30…30 (mathematisch ist der Wert schon klar)."""
    N, Z = range(0, 31), range(-30, 31)
    return {
        "c1": all(any(x > y for y in N) for x in N),
        "c2": all(any(x > y for y in Z) for x in N),
        "c3": any(all(x > y for y in Z) for x in N),
        "c4": any(all(x > y for x in N) for y in Z),
    }[k]


def _lg3c_call(k):
    text, spec, u1, u2, _, _, neg = _LG3C[k]
    if k == "c4":
        uni = {"y": pr.integers(30), "x": pr.naturals(30)}
        wert = pr.quantify("∃y ∀x", "x > y", uni)
    else:
        uni = {"x": pr.naturals(30), "y": pr.integers(30) if u2 == "Z" else pr.naturals(30)}
        wert = pr.quantify(spec, "x > y", uni)
    return Sammlung(f"Wahrheitswert und Negation von {text}", {"Wahrheitswert": wert, "Negation": pr.negate_formula(text.replace("ℕ", "N").replace("ℤ", "Z"))},
                    highlights=[("Wahrheitswert", wert.value), ("Negation", neg)])


for _k, (_text, _spec, _u1, _u2, _f, _z, _neg) in _LG3C.items():
    _ex(f"LG3.{_k}", f"Wahrheitswert und Negation: {_text}",
        f"Bestimmen Sie den Wahrheitswert von {_text} und schreiben Sie die Negation ohne Negationsoperator.",
        (lambda k=_k: _lg3c_call(k)),
        [E("Wahrheitswert", lambda r: r.werte["Wahrheitswert"], _lg3c_werte(_k), "unabhängig: Python all()/any()"),
         E("Negation ohne ¬", lambda r: str(r.teile["Negation"].result), _neg, "von Hand: Quantoren umkehren, Vergleich umkehren")],
        "N = {0, 1, 2, …}. ¬∀ ≡ ∃¬, ¬∃ ≡ ∀¬, ¬(x > y) ≡ x ≤ y.", source="Lerngerüst Serie 3 (c)")


# ==========================================================================
# Serie 4 — Beweiskalküle
# ==========================================================================
# Soll-Werte: Tabelle der Serie (Seite 2), Mitschrift (K6 = {N}, K7 = {P}) und Python-Brute-Force über alle Belegungen.
import itertools as _it


def _folgt(wb, q, namen):
    """Unabhängig: WB ⊨ Q durch Probieren aller Belegungen (wb, q sind Python-Funktionen der Belegung)."""
    for w in _it.product([False, True], repeat=len(namen)):
        e = dict(zip(namen, w))
        if wb(e) and not q(e):
            return False
    return True


def _wb1(e):
    return (e["A"] or e["B"]) and (e["C"] or (not e["B"] and not e["A"]) or e["B"])


_WB1 = "(A or B) and (C or (not B and not A) or B)"
_PDF_TAB = {(1, 1, 1): (1, 1), (1, 1, 0): (1, 1), (1, 0, 1): (1, 1), (1, 0, 0): (0, 1),
            (0, 1, 1): (1, 1), (0, 1, 0): (1, 1), (0, 0, 1): (0, 0), (0, 0, 0): (0, 0)}

_ex("SW4.1", "Folgerung per Wahrheitstabelle", f"Prüfen Sie mit einer Wahrheitstabelle, dass {_WB1} ⇒ (A ∨ B) eine Tautologie ist (WB ⊨ Q).",
    lambda: ka.entails(_WB1, "A or B", method="table"),
    [E("WB ⊨ Q", lambda r: r.entailed, True, "Serie 4, Seite 2: »Es gilt also WB ⊨ Q«"),
     E("Tabelle stimmt Zeile für Zeile mit der Serie überein",
       lambda r: all(_PDF_TAB[(int(z[0]), int(z[1]), int(z[2]))] == (int(z[3]), int(z[4])) for z in r.tables["table"].rows), True,
       "WB- und Q-Spalte aus der PDF-Tabelle"),
     E("acht Zeilen", lambda r: len(r.tables["table"].rows), 8),
     E("Brute-Force", lambda r: r.entailed, _folgt(_wb1, lambda e: e["A"] or e["B"], "ABC"), "Python über alle 8 Belegungen")],
    "Wo WB falsch ist, muss nichts geprüft werden. Wo WB wahr ist, muss auch Q wahr sein.", source="Serie 4, Aufgabe 1")
_ex("SW4.1b", "Tautologie mit den Rechengesetzen", f"Weisen Sie ({_WB1}) ⇒ (A ∨ B) mit den Rechengesetzen der Aussagenlogik nach.",
    lambda: ka.tautology_by_laws(f"({_WB1}) -> (A or B)"),
    [E("bewiesen", lambda r: r.proved, True, "Tautologie: Wahrheitstabelle Serie 4 / S. 2"),
     E("Brute-Force: Tautologie", lambda r: r.proved,
       all((not _wb1(dict(zip("ABC", w)))) or (w[0] or w[1]) for w in _it.product([False, True], repeat=3)), "Python")],
    "Implikation auflösen, De Morgan, Distributivgesetz, A ∨ ¬A = W.", source="Serie 4, Aufgabe 1")
_ex("SW4.2", "Widerspruchsbeweis per Tabelle", f"Prüfen Sie mit einer Wahrheitstabelle, dass {_WB1} ∧ ¬(A ∨ B) eine Kontradiktion ist.",
    lambda: ka.entails(_WB1, "A or B", method="contradiction"),
    [E("WB ∧ ¬Q ist unerfüllbar", lambda r: r.entailed, True, "Serie 4: WB ⊨ Q genau dann, wenn WB ∧ ¬Q Kontradiktion"),
     E("alle Zeilen WB ∧ ¬Q sind 0", lambda r: all(z[-1] == "0" for z in r.tables["contradiction"].rows), True),
     E("Brute-Force", lambda r: any(_wb1(dict(zip("ABC", w))) and not (w[0] or w[1]) for w in _it.product([False, True], repeat=3)), False, "Python")],
    "Eine Kontradiktion hat in jeder Zeile den Wert 0.", source="Serie 4, Aufgabe 2")
_ex("SW4.2b", "Kontradiktion mit den Rechengesetzen", f"Weisen Sie ({_WB1}) ∧ ¬(A ∨ B) mit den Rechengesetzen nach.",
    lambda: ka.contradiction_by_laws(f"({_WB1}) and not (A or B)"),
    [E("bewiesen", lambda r: r.proved, True, "Serie 4, Aufgabe 2")],
    "Zuerst ¬(A ∨ B) mit De Morgan zu ¬A ∧ ¬B umformen.", source="Serie 4, Aufgabe 2")
_ex("SW4.3", "Resolutionsregel semantisch", "Prüfen Sie mit einer Wahrheitstabelle, dass (A ∨ B) ∧ (B ⇒ C) ⊨ (A ∨ C) gilt.",
    lambda: ka.entails("(A or B) and (B -> C)", "A or C", method="table"),
    [E("WB ⊨ Q", lambda r: r.entailed, _folgt(lambda e: (e["A"] or e["B"]) and (not e["B"] or e["C"]), lambda e: e["A"] or e["C"], "ABC"),
       "Python über alle Belegungen"),
     E("Folgerung gilt", lambda r: r.entailed, True, "Serie 4: Resolutionsregel (A ∨ B) ∧ (¬B ∨ C) ⊢ (A ∨ C)")],
    "Gegenbeispiel gesucht: WB wahr, Q falsch. Gibt es keines, gilt die Folgerung.", source="Serie 4, Aufgabe 3")
_ex("SW4.3b", "Folgerung mit den Rechengesetzen", "Weisen Sie (A ∨ B) ∧ (B ⇒ C) ⊨ (A ∨ C) mit den Rechengesetzen nach.",
    lambda: ka.entails_by_laws("(A or B) and (B -> C)", "A or C"),
    [E("bewiesen", lambda r: r.proved, True, "Serie 4, Aufgabe 3")], "Deduktionstheorem: (WB ⇒ Q) muss eine Tautologie sein.",
    source="Serie 4, Aufgabe 3")
_ex("SW4.4", "KNF von (A ∨ B) ⇒ (C ∧ D)", "Formen Sie mit den Rechengesetzen (A ∨ B) ⇒ (C ∧ D) in KNF um.",
    lambda: ka.to_cnf("(A or B) -> (C and D)"),
    [E("vier Klauseln (von Hand: (¬A ∨ C) ∧ (¬A ∨ D) ∧ (¬B ∨ C) ∧ (¬B ∨ D))", lambda r: r.clauses,
       frozenset({frozenset({"¬A", "C"}), frozenset({"¬A", "D"}), frozenset({"¬B", "C"}), frozenset({"¬B", "D"})}),
       "von Hand: ¬(A ∨ B) ∨ (C ∧ D) = (¬A ∧ ¬B) ∨ (C ∧ D), ausmultipliziert"),
     E("in KNF", lambda r: ka.is_cnf(r.formula), True),
     E("äquivalent zur Ausgangsformel (Brute-Force)", lambda r: all(
         (not (e[0] or e[1]) or (e[2] and e[3])) == all(any((e["ABCD".index(l.lstrip("¬"))] != l.startswith("¬")) for l in k) for k in r.clauses)
         for e in _it.product([False, True], repeat=4)), True, "Python über 16 Belegungen")],
    "Implikation auflösen, De Morgan, dann das Distributivgesetz.", source="Serie 4, Aufgabe 4")
_WB5 = "(S or N) and (not S) and (P or N) and (not N or P)"
_Q5 = "not S and N and P"
_ex("SW4.5", "Resolution: (¬S ∧ N ∧ P) aus WB", f"Beweisen Sie mit dem Resolutionskalkül, dass aus {_WB5} die Aussage ¬S ∧ N ∧ P abgeleitet werden kann.",
    lambda: ka.resolution(_WB5, _Q5),
    [E("leere Klausel □ abgeleitet", lambda r: r.derived_empty, True, "Serie 4, Aufgabe 5"),
     E("K1…K4 sind die Klauseln der WB", lambda r: r.clauses[:4], [{"S", "N"}, {"¬S"}, {"P", "N"}, {"¬N", "P"}],
       "Mitschrift: »bereits in KNF, 4 Klauseln«"),
     E("¬Q ist die Klausel {S, ¬N, ¬P} (K5)", lambda r: set(r.clauses[4]), {"S", "¬N", "¬P"}, "De Morgan von Hand"),
     E("K6 = {N} aus K1 und K2", lambda r: (set(r.clauses[5]), r.lines[5].parents), ({"N"}, (1, 2)), "Mitschrift"),
     E("K7 = {P} aus K3 und K4", lambda r: (set(r.clauses[6]), r.lines[6].parents), ({"P"}, (3, 4)), "Mitschrift"),
     E("Brute-Force: WB ⊨ Q", lambda r: r.entailed,
       _folgt(lambda e: (e["S"] or e["N"]) and not e["S"] and (e["P"] or e["N"]) and (not e["N"] or e["P"]),
              lambda e: not e["S"] and e["N"] and e["P"], "SNP"), "Python")],
    "Erst ¬Q mit De Morgan in eine Klausel verwandeln, dann Paare mit komplementären Literalen verbinden.", source="Serie 4, Aufgabe 5")
_ex("SW4.G", "Gödel: korrekt und vollständig", "Erklären Sie semantische Folgerung ⊨, syntaktische Ableitung ⊢, korrekt und vollständig.",
    lambda: ka.kalkuel_info(),
    [E("Begriffe vorhanden", lambda r: all(k in r.facts for k in ("Korrekt", "Vollständig", "Gödelscher Vollständigkeitssatz",
                                                                   "Gödelscher Unvollständigkeitssatz", "Entscheidbarkeit")), True,
       "Serie 4, Seiten 6 und 7")],
    "Korrekt: WB ⊢ Q ⇒ WB ⊨ Q. Vollständig: WB ⊨ Q ⇒ WB ⊢ Q.", source="Serie 4, Gödel")
_ex("SW4.P1", "Prolog: Wem gehört der Fisch?", "Lösen Sie das Zebrarätsel (15 Hinweise).",
    lambda: ka.fish_puzzle(),
    [E("Der Deutsche", lambda r: "Deutsche" in r.answer, True, "bekannte Lösung des Rätsels (Einstein-Rätsel, Variante der Serie)"),
     E("genau eine Lösung", lambda r: len(r.solutions), 1)],
    "Das Prolog-Programm der Serie probiert mit member/neben/links alle Möglichkeiten.", source="Serie 4, Frage 2, Aufgabe 1 (optional)")
_ex("SW4.P2", "Einstellungstest: acht Kästchen", "Füllen Sie acht Kästchen so, dass kein »durchgefallen«-Hinweis zutrifft.",
    lambda: ka.boxes_puzzle(),
    [E("genau eine Lösung", lambda r: len(r.solutions), 1)],
    "Alle 2⁸ Belegungen prüfen.", source="Serie 4, Frage 2, Aufgabe 2 (optional)")

# -- Beispiele aus der Vorlesung -------------------------------------------------------------------------------------------------------
_ex("SW4.B1", "Drei Prüfungen der Folgerung", "Prüfen Sie WB ⊨ Q mit Pfeilmethode, Deduktion und Widerspruch.",
    lambda: ka.entails(_WB1, "A or B"),
    [E("alle drei Prüfungen wahr", lambda r: r.verdicts, {"table": True, "deduction": True, "contradiction": True}, "Serie 4: Zusammenfassung der drei Prüfungen")],
    "Alle drei Methoden müssen dasselbe Ergebnis liefern.", source="Serie 4, Zusammenfassung")
_ex("SW4.B2", "Nicht-Folgerung mit Gegenbeispiel", "Folgt A aus A ∨ B?",
    lambda: ka.entails("A or B", "A"),
    [E("folgt nicht", lambda r: r.entailed, False), E("Gegenbeispiel A = 0, B = 1", lambda r: r.counterexample, {"A": False, "B": True}, "von Hand")],
    "A = 0, B = 1 macht A ∨ B wahr, aber A falsch.", source="Serie 4, Folgerung")
_ex("SW4.B3", "Resolutionsregel", "Resolvieren Sie {S, N} mit {¬S}.",
    lambda: ka.refute([{"S", "N"}, {"¬S"}]),
    [E("Resolvent {N} aus K1 und K2", lambda r: (set(r.clauses[2]), r.lines[2].parents), ({"N"}, (1, 2)), "Mitschrift: K6"),
     E("Einzelschritt resolve()", lambda r: [(l, set(k)) for l, k in ka.resolve({"S", "N"}, {"¬S"})], [("S", {"N"})], "von Hand")],
    "Das Paar S / ¬S streichen, den Rest vereinigen.", source="Serie 4, Resolutionsregel")
_ex("SW4.B4", "Semantisch gegen syntaktisch", "Beantworten Sie dieselbe Frage mit Tabelle und mit Resolution.",
    lambda: ka.semantic_vs_syntactic("(A or B) and (B -> C)", "A or C"),
    [E("beide Verfahren stimmen überein", lambda r: r.same, True), E("Wert wahr", lambda r: r.value, True)],
    "Gödel: ein korrekter und vollständiger Kalkül liefert dieselbe Antwort.", source="Serie 4, Gödel")
_ex("SW4.B5", "KNF von ¬Q", "Bringen Sie ¬(¬S ∧ N ∧ P) in KNF.",
    lambda: ka.to_cnf("not (not S and N and P)"),
    [E("eine Klausel {S, ¬N, ¬P}", lambda r: r.clauses, frozenset({frozenset({"S", "¬N", "¬P"})}), "von Hand: De Morgan")],
    "Mitschrift: De Morgan liefert eine Klausel.", source="Serie 4, Aufgabe 5")
_ex("SW4.B6", "Literal, Klausel, KNF", "Prüfen Sie A, A ∨ B und (A ∨ B) ∧ C.",
    lambda: Sammlung("Begriffe", {"Literal ¬A": ka.to_cnf("not A"), "Klausel": ka.to_cnf("A or not B"), "KNF": ka.to_cnf("(A or B) and (not A or C)")}),
    [E("Literal", lambda r: ka.is_literal("not A"), True), E("Klausel", lambda r: ka.is_clause("A or not B or C"), True),
     E("KNF", lambda r: ka.is_cnf("(A or B) and (not A or C)"), True),
     E("keine KNF", lambda r: ka.is_cnf("A or (B and C)"), False, "ODER über UND ist keine KNF")],
    "Literal: Variable mit oder ohne ¬. Klausel: ODER von Literalen. KNF: UND von Klauseln.", source="Serie 4, KNF")

# -- Lerngerüst Serie 4 -----------------------------------------------------------------------------------------------------------------
for _i, (_wb, _q, _t) in enumerate([("U and W", "U or W", "U"), ("(not H or S) and H", "S and H", "H"),
                                     ("(A or B or C) -> (P and Q)", "(A -> P) and (C -> Q)", "A")], 1):
    _ex(f"LG4.a{_i}", f"Folgerung: {_wb} ⊨ {_q}", f"Prüfen Sie mit einer Wahrheitstabelle, dass {_wb} ⊨ {_q}.",
        (lambda w=_wb, q=_q: ka.entails(w, q)),
        [E("WB ⊨ Q", lambda r: r.entailed, True, "Lerngerüst Serie 4 (a)"),
         E("Brute-Force", lambda r, i=_i: r.entailed, {
             1: _folgt(lambda e: e["U"] and e["W"], lambda e: e["U"] or e["W"], "UW"),
             2: _folgt(lambda e: (not e["H"] or e["S"]) and e["H"], lambda e: e["S"] and e["H"], "HS"),
             3: _folgt(lambda e: (not (e["A"] or e["B"] or e["C"])) or (e["P"] and e["Q"]),
                       lambda e: (not e["A"] or e["P"]) and (not e["C"] or e["Q"]), "ABCPQ")}[_i], "Python")],
        "Zeilen mit WB = 1 müssen auch Q = 1 haben.", source="Lerngerüst Serie 4 (a)")
_ex("LG4.b", "KNF von (A ∨ B ∨ C) ⇒ (P ∧ Q)", "Formen Sie den Term in KNF um.",
    lambda: ka.to_cnf("(A or B or C) -> (P and Q)"),
    [E("sechs Klauseln {¬X, Y}, X ∈ {A, B, C}, Y ∈ {P, Q}", lambda r: r.clauses,
       frozenset(frozenset({"¬" + x, y}) for x in "ABC" for y in "PQ"), "von Hand: ¬(A ∨ B ∨ C) ∨ (P ∧ Q)")],
    "(¬A ∧ ¬B ∧ ¬C) ∨ (P ∧ Q) ausmultiplizieren: 3 · 2 Klauseln.", source="Lerngerüst Serie 4 (b)")
_ex("LG4.c", "Resolution: (C ∧ B) aus (C ⇒ B) ∧ (A ∨ C) ∧ (A ⇒ C)", "Beweisen Sie mit dem Resolutionskalkül.",
    lambda: ka.resolution("(C -> B) and (A or C) and (A -> C)", "C and B"),
    [E("□ abgeleitet", lambda r: r.derived_empty, True, "Lerngerüst Serie 4 (c)"),
     E("Brute-Force", lambda r: r.entailed, _folgt(lambda e: (not e["C"] or e["B"]) and (e["A"] or e["C"]) and (not e["A"] or e["C"]),
                                                   lambda e: e["C"] and e["B"], "ABC"), "Python")],
    "A ∨ C und A ⇒ C ergeben C; mit C ⇒ B folgt B.", source="Lerngerüst Serie 4 (c)")
_ex("LG4.d", "Resolution: ¬C aus ¬(A ∧ B) ∧ (C ⇒ B) ∧ ((¬A ∧ B) ⇒ ¬C)", "Beweisen Sie mit dem Resolutionskalkül (Klammerung wie im PDF-Bild).",
    lambda: ka.resolution("not (A and B) and (C -> B) and ((not A and B) -> not C)", "not C"),
    [E("□ abgeleitet", lambda r: r.derived_empty, True, "Lerngerüst Serie 4 (d)"),
     E("Brute-Force", lambda r: r.entailed, _folgt(
         lambda e: not (e["A"] and e["B"]) and (not e["C"] or e["B"]) and (not (not e["A"] and e["B"]) or not e["C"]),
         lambda e: not e["C"], "ABC"), "Python")],
    "Das PDF-Bild zeigt (¬A ∧ B) ⇒ ¬C als eigene Klammer; ohne sie wäre die Folgerung falsch.", source="Lerngerüst Serie 4 (d)")


# ==========================================================================
# Serie 5 — Analyse von Algorithmen
# ==========================================================================
# Soll-Werte: Mitschrift (A: 15, B: 25, C: 120; 3 + 7 + … + 1), Python-Brute-Force und die von Hand hergeleiteten
# Zähler _schritte_* (Regeln der Vorlesung). Die Messung des Moduls ist nie die Quelle des Soll-Wertes.
import math as _math


def _max_ueber_permutationen(zaehler, n):
    return max(zaehler(list(p)) for p in _it.permutations(range(n)))


def _min_ueber_permutationen(zaehler, n):
    return min(zaehler(list(p)) for p in _it.permutations(range(n)))


def _groesstes_n(schritte, limit):
    """Unabhängig: grösstes n mit schritte(n) ≤ limit (ganzzahlig, monoton)."""
    n = 1
    while schritte(n + 1) <= limit:
        n += 1
    return n


def _binsuche_worst(n):
    return max(_schritte_binaer(list(range(n)), e) for e in range(-1, n + 1))


dk_classify = ag.classify  # ponytail: im Paket heisst es classify_growth (classify ist SW2), das Modul-Original ist dasselbe

_MINUTE = 60 * 10 ** 9      # Rechnungsschritte in einer Minute bei 10⁹ pro Sekunde

_ex("SW5.1a", "Rechnungsschritte für n = 5", "A: n·log₂(n), B: n², C: n!. Wie viele Schritte sind es bei n = 5 (log aufgerundet wie in der Vorlesung)?",
    lambda: ag.compare_growth({"A": "n*log(n,2)", "B": "n**2", "C": "factorial(n)"}, [5], round_log="ceil"),
    [E("A = 5·⌈log₂ 5⌉ = 15", lambda r: int(r.values["A"][5]), 5 * _math.ceil(_math.log2(5)), "Mitschrift: A: 15"),
     E("B = 5² = 25", lambda r: int(r.values["B"][5]), 25, "Mitschrift: B: 25"),
     E("C = 5! = 120", lambda r: int(r.values["C"][5]), _math.factorial(5), "Mitschrift: C: 120")],
    "log₂ 5 ≈ 2.32, aufgerundet 3.", source="Serie 5, Aufgabe 1")
_ex("SW5.1b", "Grösste Eingabe in einer Minute", "Wie viele Komponenten darf die Liste höchstens haben, damit der Output bei 10⁹ Schritten pro Sekunde innerhalb einer Minute berechnet wird?",
    lambda: ag.max_input_sizes({"A": "n*log(n,2)", "B": "n**2", "C": "factorial(n)"}),
    [E("A (n·log₂ n ≤ 6·10¹⁰)", lambda r: r.n["A"], 1944470450, "numerisch bestimmt, unten nachgerechnet"),
     E("A: n passt, n + 1 nicht (Python-Floats)", lambda r: (r.n["A"] * _math.log2(r.n["A"]) <= _MINUTE < (r.n["A"] + 1) * _math.log2(r.n["A"] + 1)), True,
       "unabhängig: math.log2"),
     E("B: n = ⌊√(6·10¹⁰)⌋", lambda r: r.n["B"], _math.isqrt(_MINUTE), "von Hand: n² ≤ 6·10¹⁰"),
     E("C: grösstes n mit n! ≤ 6·10¹⁰", lambda r: r.n["C"], _groesstes_n(_math.factorial, _MINUTE), "Python-Schleife")],
    "Nicht algebraisch lösbar: numerisch probieren.", source="Serie 5, Aufgabe 1 (Mitschrift: »Kann numerisch mit Algorithmik gelöst werden«)")

_ex("SW5.2a", "decToBin: Schritte für x = 25", "Bestimmen Sie die totale Anzahl Rechnungsschritte von decToBin(25).",
    lambda: ag.count_steps("decToBin", x=25),
    [E("3 + 7 · 5 + 1 = 39", lambda r: r.total, _schritte_dectobin(25), "Mitschrift: 3 Initialschritte + 7 je Durchlauf + 1 Return"),
     E("Ergebnis des Algorithmus", lambda r: r.value, bin(25)[2:], "Python bin()")],
    "x = 25 hat 5 Binärstellen, also 5 Durchläufe.", source="Serie 5, Aufgabe 2")
_ex("SW5.2b", "decToBin: Formel in x", "Bestimmen Sie die Schrittzahl abhängig von x.",
    lambda: ag.worst_case("decToBin", [1, 2, 3, 4, 8, 15, 16, 100, 1000]),
    [E("T(x) = 11 + 7·⌊log₂ x⌋", lambda r: [r.at(x) for x in (1, 2, 3, 4, 8, 15, 16, 100, 1000)],
       [11 + 7 * (x.bit_length() - 1) for x in (1, 2, 3, 4, 8, 15, 16, 100, 1000)], "Mitschrift: n + 1 = ⌊log₂ x⌋ + 1 Durchläufe"),
     E("von Hand für alle x ≤ 300", lambda r: all(_schritte_dectobin(x) == 11 + 7 * (x.bit_length() - 1) for x in range(1, 301)), True,
       "Python-Schleife")],
    "Anzahl Durchläufe = Anzahl Binärstellen = ⌊log₂ x⌋ + 1.", source="Serie 5, Aufgabe 2")

_ex("SW5.3", "Lineare Suche, Worst-Case", "Bestimmen Sie die Anzahl Rechnungsschritte der linearen Suche im Worst-Case.",
    lambda: ag.worst_case("linearSearch"),
    [E("2n + 1 für n = 1…12", lambda r: [r.at(n) for n in range(1, 13)], [2 * n + 1 for n in range(1, 13)], "von Hand: n × (for + Vergleich) + return"),
     E("Zähler unabhängig (Element fehlt)", lambda r: all(_schritte_linear(list(range(n)), -1) == 2 * n + 1 for n in range(13)), True, "Python-Zähler _schritte_linear")],
    "Worst-Case: das Element fehlt oder steht ganz hinten.", source="Serie 5, Aufgabe 3")

_ex("SW5.4a", "Selection Sort: Worst-Case", "Bestimmen Sie die Schrittzahl von selectionSort im Worst-Case (Vorlesung: 5n + 3·n(n−1)/2).",
    lambda: ag.worst_case("selectionSort"),
    [E("echtes Maximum (alle Permutationen) für n = 1…6",
       lambda r: [r.at(n) for n in range(1, 7)],
       [_max_ueber_permutationen(_schritte_selection, n) for n in range(1, 7)], "Python: Zähler über alle n! Permutationen"),
     E("Vorlesungsschranke 5n + 3n(n−1)/2 bei n = 5 ist 55", lambda r: 5 * 5 + 3 * (5 * 4 // 2), 55, "Mitschrift (Schranke: jeder Vergleich aktualisiert minIndex)"),
     E("Schranke liegt über dem Maximum", lambda r: all(5 * n + 3 * (n * (n - 1) // 2) >= r.at(n) for n in range(1, 13)), True)],
    "Der Gauss-Trick: (n−1) + (n−2) + … + 1 = n(n−1)/2.", source="Serie 5, Aufgabe 4")
_ex("SW5.4b", "Selection Sort: Best-Case", "Was ist die kleinste Schrittzahl von selectionSort?",
    lambda: ag.best_case("selectionSort"),
    [E("Minimum (alle Permutationen) für n = 1…6", lambda r: [r.at(n) for n in range(1, 7)],
       [_min_ueber_permutationen(_schritte_selection, n) for n in range(1, 7)], "Python über alle Permutationen"),
     E("n² + 4n", lambda r: [r.at(n) for n in range(1, 7)], [n * n + 4 * n for n in range(1, 7)], "von Hand: 5n + 2·n(n−1)/2 ohne Updates")],
    "Best-Case: sortierte Liste, minIndex wird nie überschrieben.", source="Serie 5, Aufgabe 4")
_ex("SW5.5a", "Insertion Sort: Worst-Case", "Bestimmen Sie die Schrittzahl von insertionSort im Worst-Case.",
    lambda: ag.worst_case("insertionSort"),
    [E("absteigende Liste für n = 1…6", lambda r: [r.at(n) for n in range(1, 7)],
       [_schritte_insertion(list(range(n, 0, -1))) for n in range(1, 7)], "Python-Zähler _schritte_insertion"),
     E("echtes Maximum (alle Permutationen)", lambda r: [r.at(n) for n in range(1, 7)],
       [_max_ueber_permutationen(_schritte_insertion, n) for n in range(1, 7)]),
     E("5n² − 2n − 3", lambda r: [r.at(n) for n in range(1, 7)], [5 * n * n - 2 * n - 3 for n in range(1, 7)], "von Hand")],
    "Worst-Case: absteigend sortiert, jedes Element wandert ganz nach vorn.", source="Serie 5, Aufgabe 5")
_ex("SW5.5b", "Insertion Sort: Best-Case", "Was ist die kleinste Schrittzahl von insertionSort?",
    lambda: ag.best_case("insertionSort"),
    [E("Minimum (alle Permutationen)", lambda r: [r.at(n) for n in range(1, 7)], [_min_ueber_permutationen(_schritte_insertion, n) for n in range(1, 7)]),
     E("5n − 5", lambda r: [r.at(n) for n in range(1, 7)], [5 * n - 5 for n in range(1, 7)], "von Hand: n−1 Durchläufe, je 5")],
    "Best-Case: sortiert, die while-Bedingung scheitert sofort.", source="Serie 5, Aufgabe 5")
_ex("SW5.6a", "Bubble Sort: Worst-Case", "Bestimmen Sie die Schrittzahl von bubbleSort im Worst-Case.",
    lambda: ag.worst_case("bubbleSort"),
    [E("echtes Maximum (alle Permutationen)", lambda r: [r.at(n) for n in range(1, 7)], [_max_ueber_permutationen(_schritte_bubble, n) for n in range(1, 7)]),
     E("11n²/2 − 9n/2", lambda r: [r.at(n) for n in range(1, 7)], [(11 * n * n - 9 * n) // 2 for n in range(1, 7)], "von Hand: n + 3n(n−1) + 5·n(n−1)/2")],
    "Worst-Case: absteigend, jeder Vergleich tauscht.", source="Serie 5, Aufgabe 6")
_ex("SW5.6b", "Bubble Sort: Best-Case", "Was ist die kleinste Schrittzahl von bubbleSort?",
    lambda: ag.best_case("bubbleSort"),
    [E("Minimum (alle Permutationen)", lambda r: [r.at(n) for n in range(1, 7)], [_min_ueber_permutationen(_schritte_bubble, n) for n in range(1, 7)]),
     E("3n² − 2n", lambda r: [r.at(n) for n in range(1, 7)], [3 * n * n - 2 * n for n in range(1, 7)], "von Hand: n + 3n(n−1)")],
    "Bubble Sort läuft immer alle n·(n−1) Vergleiche, auch bei sortierter Liste.", source="Serie 5, Aufgabe 6")
_ex("SW5.7a", "Binäre Suche: Worst-Case", "Bestimmen Sie die Schrittzahl von binarySearch im Worst-Case.",
    lambda: ag.worst_case("binarySearch"),
    [E("8·⌊log₂ n⌋ + 13 für n = 1…40", lambda r: [r.at(n) for n in range(1, 41)], [8 * (n.bit_length() - 1) + 13 for n in range(1, 41)], "von Hand"),
     E("echtes Maximum über alle Suchwerte", lambda r: [r.at(n) for n in range(1, 41)], [_binsuche_worst(n) for n in range(1, 41)],
       "Python-Zähler _schritte_binaer")],
    "Bei jedem Durchlauf halbiert sich der Suchbereich.", source="Serie 5, Aufgabe 7")
_ex("SW5.7b", "Binäre Suche: Best-Case", "Was ist die kleinste Schrittzahl von binarySearch?",
    lambda: ag.best_case("binarySearch"),
    [E("9 (Treffer in der Mitte)", lambda r: [r.at(n) for n in range(1, 21)], [9] * 20, "von Hand: 3 + Test 1 + middle 3 + == 1 + return 1"),
     E("Python-Zähler", lambda r: _schritte_binaer(list(range(7)), 3), 9)],
    "Das gesuchte Element liegt genau in der Mitte.", source="Serie 5, Aufgabe 7")

# -- Aufgabe 8: Big-O, Big-Ω, Big-Θ (Zeugen werden mit Floats/Brüchen unabhängig nachgerechnet) ----------------------------------
def _zeugen_ok(f, g, art, konst, x0, bis=400):
    """Prüft f(x) ≤ C·g(x) bzw. f(x) ≥ c·g(x) für alle ganzen x in [x0, bis] mit Bruchrechnung."""
    for x in range(max(int(x0), 2), bis + 1):
        fx, gx = f(x), g(x)
        if art == "O" and not fx <= konst * gx:
            return False
        if art == "Omega" and not fx >= konst * gx:
            return False
    return True


_F8 = {
    "a": ("4x⁵ + 15x³ − 10x² − 8", "4*x**5 + 15*x**3 - 10*x**2 - 8", "x**6", "O", "O(x⁶)", lambda x: 4 * x ** 5 + 15 * x ** 3 - 10 * x ** 2 - 8, lambda x: x ** 6),
    "b": ("4x⁵ + 15x³ − 10x² − 8", "4*x**5 + 15*x**3 - 10*x**2 - 8", "x**3", "Omega", "Ω(x³)", lambda x: 4 * x ** 5 + 15 * x ** 3 - 10 * x ** 2 - 8, lambda x: x ** 3),
    "d": ("5x·log₂(3ˣ) + 3x²", "5*x*log(3**x, 2) + 3*x**2", "x**2*log(x)", "O", "O(x² log x)",
          lambda x: 5 * x * x * _math.log2(3) + 3 * x * x, lambda x: x * x * _math.log(x)),
    "e": ("(8x + x²·log₁₀(5ˣ))·log₁₀(10x)", "(8*x + x**2*log(5**x, 10))*log(10*x, 10)", "x**3*log(x)", "O", "O(x³ log x)",
          lambda x: (8 * x + x * x * x * _math.log10(5)) * _math.log10(10 * x), lambda x: x ** 3 * _math.log(x)),
}
for _k, (_ft, _f, _g, _art, _gt, _pf, _pg) in _F8.items():
    _fn = {"O": ag.big_o, "Omega": ag.big_omega}[_art]
    _ex(f"SW5.8{_k}", f"f(x) = {_ft} in {_gt}", f"Begründen Sie, dass f(x) = {_ft} in {_gt} ist.",
        (lambda fn=_fn, f=_f, g=_g: fn(f, g)),
        [E("Aussage gilt", lambda r: r.holds, True, "Serie 5, Aufgabe 8"),
         E("Zeugen stimmen (Python, x0…400)", lambda r, pf=_pf, pg=_pg, a=_art: _zeugen_ok(pf, pg, a, r.C if a == "O" else r.c, r.x0), True,
           "unabhängig: Ungleichung f(x) ≤ C·g(x) bzw. ≥ c·g(x) Punkt für Punkt")],
        "Zeugen x₀ und C bzw. c angeben; Terme mit log umformen (log(aˣ) = x·log a).", source=f"Serie 5, Aufgabe 8{_k}")
_ex("SW5.8c", "f(x) = 5x² − 2x + 8 in Θ(x²)", "Begründen Sie, dass f(x) = 5x² − 2x + 8 in Θ(x²) ist.",
    lambda: ag.big_theta("5*x**2 - 2*x + 8", "x**2"),
    [E("Aussage gilt", lambda r: r.holds, True, "Serie 5, Aufgabe 8c"),
     E("c·x² ≤ f(x) ≤ C·x² für x ≥ x₀ (Bruchrechnung)", lambda r: all(r.c * x * x <= 5 * x * x - 2 * x + 8 <= r.C * x * x for x in range(max(int(r.x0), 1), 501)), True,
       "unabhängig: Python"),
     E("c ≤ 5 ≤ C (Leitkoeffizient)", lambda r: r.c <= 5 <= r.C, True, "von Hand")],
    "O und Ω getrennt zeigen.", source="Serie 5, Aufgabe 8c")

# -- Beispiele aus der Vorlesung -------------------------------------------------------------------------------------------------------
_ex("SW5.B1", "Gauss-Trick: 1 + 2 + … + 100", "Berechnen Sie 1 + 2 + … + 100 mit dem Gauss-Trick.",
    lambda: ag.gauss_sum(100),
    [E("5050", lambda r: r.value, sum(range(1, 101)), "Mitschrift: 100·101 = 2s, s = 5050")],
    "Erste + letzte Zahl = 101, das gibt 50 Paare.", source="Serie 5, Aufgabe 4 (Mitschrift)")
_ex("SW5.B2", "Gauss-Trick: 1 + … + (n−1)", "Berechnen Sie (n−1) + (n−2) + … + 1.",
    lambda: ag.gauss_sum("n-1"),
    [E("n(n−1)/2", lambda r: [int(r.value.subs(sorted(r.value.free_symbols, key=str)[0], k)) for k in (2, 5, 10)], [k * (k - 1) // 2 for k in (2, 5, 10)], "Mitschrift: (n−1)·n = 2s")],
    "N−1 + 1 = N, N−2 + 2 = N, …", source="Serie 5, Aufgabe 4 (Mitschrift)")
_ex("SW5.B3", "Komplexitätsklassen", "Nennen Sie die wichtigen Komplexitätsklassen.",
    lambda: ag.growth_classes(),
    [E("mindestens konstant, logarithmisch, linear, n log n, quadratisch", lambda r: [c.name for c in r.classes][:5],
       ["konstant", "logarithmisch", "linear", "n log n", "quadratisch"], "Serie 5, Frage 4: O(1) ⊂ O(log n) ⊂ O(n) ⊂ O(n log n) ⊂ O(n²)")],
    "O(1) ⊂ O(log n) ⊂ O(n) ⊂ O(n log n) ⊂ O(n²) ⊂ … ⊂ O(2ⁿ).", source="Serie 5, Frage 4")
_ex("SW5.B4", "n² + 3n + 3 und n² + 5n + 7", "Wie unterscheiden sich n² + 3n + 3 und n² + 5n + 7 asymptotisch?",
    lambda: Sammlung("Asymptotik", {"n² + 3n + 3": dk_classify("n**2+3*n+3"), "n² + 5n + 7": dk_classify("n**2+5*n+7")}),
    [E("beide in Θ(n²)", lambda r: [r.teile[k].notation for k in r.teile], ["Θ(n²)", "Θ(n²)"], "Serie 5, Frage 3: »dürfte keinen Unterschied machen«")],
    "Nur der höchste Exponent zählt.", source="Serie 5, Frage 3")
_ex("SW5.B5", "Wachstumsfaktor bei Verzehnfachung", "Wie viel langsamer wird ein Algorithmus mit n² Schritten, wenn man die Eingabe verzehnfacht?",
    lambda: Sammlung("Verzehnfachung", {"n²": ag.compare_growth({"B": "n**2"}, [7, 70])}),
    [E("Faktor 100", lambda r: ag.growth_factor("n**2", 7), (7 * 10) ** 2 // 7 ** 2, "von Hand: (10n)²/n² = 100"),
     E("Tabelle stimmt", lambda r: [int(r.teile["n²"].values["B"][n]) for n in (7, 70)], [49, 4900])],
    "Serie 5, Frage 1: »wie viel langsamer wird der Algorithmus, wenn wir die Grösse der Eingabe verzehnfachen?«", source="Serie 5, Frage 1")

# -- Lerngerüst Serie 5 -----------------------------------------------------------------------------------------------------------------
_ex("LG5.a", "Verschachtelte Schleifen (rot gefärbt)", "Bestimmen Sie die Anzahl Operationen (rot gefärbt: Zuweisungen, <, + und *) von nestedLoops(n) abhängig von n.",
    lambda: ag.count_steps("nestedLoops", n=5, rules={"return": 0}),
    [E("n = 5: 98 Schritte", lambda r: r.total, _schritte_geschachtelt(5), "von Hand gezählt (Python-Zähler _schritte_geschachtelt)"),
     E("Rückgabewert", lambda r: r.value, ag.REFERENCE_FUNCTIONS["nestedLoops"](5) if hasattr(ag, "REFERENCE_FUNCTIONS") else None)],
    "Innere Schleife läuft i-mal: 0 + 1 + … + (n−1) = n(n−1)/2 Durchläufe.", source="Lerngerüst Serie 5 (a)")
_P7 = lambda x: 7 * x ** 3 + 5 * x ** 6 - 3 * x ** 4 - x + 3
_LG5B = {   # Schlüssel: (Text, f, g, Art, Python-f, Python-g)
    "b1": ("6x³ − x² + 2x + 1123 in Θ(x³)", "6*x**3 - x**2 + 2*x + 1123", "x**3", "Theta", lambda x: 6 * x ** 3 - x ** 2 + 2 * x + 1123, lambda x: x ** 3),
    "b2": ("7x³ + 5x⁶ − 3x⁴ − x + 3 in O(x⁶)", "7*x**3 + 5*x**6 - 3*x**4 - x + 3", "x**6", "O", _P7, lambda x: x ** 6),
    "b3": ("7x³ + 5x⁶ − 3x⁴ − x + 3 in Ω(x⁶)", "7*x**3 + 5*x**6 - 3*x**4 - x + 3", "x**6", "Omega", _P7, lambda x: x ** 6),
    "b4": ("21x²·log(2ˣ) + 11x·log(3ˣ) in O(x³ log x)", "21*x**2*log(2**x, 2) + 11*x*log(3**x, 2)", "x**3*log(x)", "O",
           lambda x: 21 * x * x * x + 11 * x * x * _math.log2(3), lambda x: x ** 3 * _math.log(x)),
    "b5": ("(4x·log(x^(3x)) + x³·log(3ˣ))·log(5x) in O(x⁴ log x)", "(4*x*log(x**(3*x), 2) + x**3*log(3**x, 2))*log(5*x, 2)", "x**4*log(x)", "O",
           lambda x: (12 * x * x * _math.log2(x) + x ** 4 * _math.log2(3)) * _math.log2(5 * x), lambda x: x ** 4 * _math.log(x)),
}
for _k, (_t, _f, _g, _art, _pf, _pg) in _LG5B.items():
    _fn = {"O": ag.big_o, "Omega": ag.big_omega, "Theta": ag.big_theta}[_art]
    if _art == "Theta":
        _chk = E("c·g ≤ f ≤ C·g für x ≥ x₀ (Python, bis 500)", lambda r, pf=_pf, pg=_pg: all(r.c * pg(x) <= pf(x) <= r.C * pg(x) for x in range(max(int(r.x0), 2), 501)), True,
                 "unabhängig: Ungleichungen Punkt für Punkt")
    else:
        _chk = E("Zeugen stimmen (Python, bis 400)", lambda r, pf=_pf, pg=_pg, a=_art: _zeugen_ok(pf, pg, a, r.C if a == "O" else r.c, r.x0), True,
                 "unabhängig: Ungleichung Punkt für Punkt")
    _ex(f"LG5.{_k}", f"Asymptotik: {_t}", f"Begründen Sie Schritt für Schritt: f(x) = {_t}.",
        (lambda fn=_fn, f=_f, g=_g: fn(f, g)),
        [E("Aussage gilt", lambda r: r.holds, True, "Lerngerüst Serie 5 (b)"), _chk],
        "Zeugen x₀ und C bzw. c angeben; log(aˣ) = x·log a.", source="Lerngerüst Serie 5 (b)")


# ==========================================================================
# Serie 6 — Rekursive Algorithmen
# ==========================================================================
# In den Unterlagen steht nur die Aufgabenstellung (keine Musterlösung). Die Soll-Werte sind eigene Rechnungen,
# unabhängig vom Modul gerechnet: Python-Rekursion und Iteration der Rekursionsbeziehung mit Brüchen.
from fractions import Fraction as _Fr


def _t_dc(a, b, k, n):
    """T(n) = a·T(n/b) + n^k, T(1) = 1, für n = b^j — rein rekursiv gerechnet."""
    return 1 if n == 1 else a * _t_dc(a, b, k, n // b) + n ** k


def _t_iter(coeffs, f, init, n):
    """T(n) aus T(n) = Σ cᵢ·T(n−i) + f(n) und {Index: Wert} durch Iteration (Brüche)."""
    seq = {i: _Fr(v) for i, v in init.items()}
    for m in range(max(seq) + 1, n + 1):
        seq[m] = sum(_Fr(c) * seq[m - i - 1] for i, c in enumerate(coeffs)) + f(m)
    return seq[n]


def _t_merge(n):
    """mergeSort mit Aufteilung ⌈n/2⌉ + ⌊n/2⌋ und merge = n Schritte (Annahme), T(1) = 1."""
    return 1 if n == 1 else _t_merge((n + 1) // 2) + _t_merge(n // 2) + n


def _hanoi_regeln(zuege, n, start="B", ziel="C", hilf="A"):
    """Spielt die Züge nach: eine Scheibe pro Zug, nie gross auf klein, am Ende alles am Ziel."""
    stab = {start: list(range(n, 0, -1)), ziel: [], hilf: []}
    for scheibe, von, nach in zuege:
        if not stab[von] or stab[von][-1] != scheibe or (stab[nach] and stab[nach][-1] < scheibe):
            return False
        stab[nach].append(stab[von].pop())
    return stab[ziel] == list(range(n, 0, -1)) and not stab[start] and not stab[hilf]


def _hanoi_min(n):
    """Kürzeste Zugfolge per Breitensuche über alle Zustände (Position jeder Scheibe: 0, 1, 2)."""
    start, ziel = (1,) * n, (2,) * n        # Scheibe i liegt auf Stab t[i]; Start Mitte (1), Ziel rechts (2)
    dist, queue = {start: 0}, [start]
    for st in queue:
        if st == ziel:
            return dist[st]
        for d in range(n):                  # Scheibe d (0 = kleinste) darf ziehen, wenn keine kleinere auf ihrem Stab liegt
            if any(st[e] == st[d] for e in range(d)):
                continue
            for t in range(3):
                if t != st[d] and not any(st[e] == t for e in range(d)):
                    nxt = st[:d] + (t,) + st[d + 1:]
                    if nxt not in dist:
                        dist[nxt] = dist[st] + 1
                        queue.append(nxt)
    raise AssertionError("Ziel nicht erreichbar")


def _g_theta(k, j):
    """Vergleichsfunktion der Θ-Klasse von T(n) = 2T(n/2) + nᵏ bei n = 2ʲ: n, n·log₂ n, n²."""
    n = 2 ** j
    return {0: n, 1: n * j, 2: n * n}[k]


_A2 = [4, 6, 5, 1, 7, 4, 3, 1]

_ex("SW6.1a", "Rekursive Summe, Rekursionsbaum für n = 5", "Bestimmen Sie einen rekursiven Algorithmus für die Summe der ersten n natürlichen Zahlen und zeichnen Sie den Rekursionsbaum für n = 5.",
    lambda: rk.recursion_tree("summe", 5),
    [E("Summe = 15", lambda r: r.value, sum(range(1, 6)), "Python sum()"),
     E("5 Aufrufe in 5 Ebenen (summe(5) … summe(1))", lambda r: (r.calls, r.height), (5, 5), "von Hand"),
     E("Aufrufe von oben nach unten", lambda r: [k.label for k in r.nodes], [f"summe({m})" for m in (5, 4, 3, 2, 1)], "von Hand")],
    "summe(n) = summe(n−1) + n, Basisfall n = 1. Der Baum ist eine Kette.", source="Serie 6, Aufgabe 1; Lösung eigen")
_ex("SW6.1b", "Rekursive Fakultät, Rekursionsbaum für n = 5", "Bestimmen Sie einen rekursiven Algorithmus für die Fakultät n! und zeichnen Sie den Rekursionsbaum für n = 5.",
    lambda: rk.recursion_tree("fakultaet", 5),
    [E("5! = 120", lambda r: r.value, _math.factorial(5), "math.factorial"),
     E("5 Aufrufe in 5 Ebenen", lambda r: (r.calls, r.height), (5, 5), "von Hand"),
     E("Rückgaben von unten: 1, 2, 6, 24, 120", lambda r: [k.value for k in reversed(r.nodes)], [1, 2, 6, 24, 120], "von Hand")],
    "fakultaet(n) = fakultaet(n−1) · n, Basisfall n = 1.", source="Serie 6, Aufgabe 1; Lösung eigen")
_ex("SW6.2", "mergeSort-Rekursionsbaum", "Zeichnen Sie einen Rekursionsbaum von mergeSort für A = [4, 6, 5, 1, 7, 4, 3, 1].",
    lambda: rk.recursion_tree("mergeSort", _A2),
    [E("sortierte Liste", lambda r: r.value, sorted(_A2), "Python sorted()"),
     E("15 Aufrufe, 4 Ebenen (1, 2, 4, 8 Knoten)", lambda r: (r.calls, [len(lv) for lv in r.levels]), (15, [1, 2, 4, 8]), "von Hand: 2·8 − 1 Knoten"),
     E("Blätter sind die 8 Einzelelemente", lambda r: [k.label for k in r.levels[3]], [f"[{x}]" for x in _A2], "von Hand"),
     E("Ebene 1: beide Hälften", lambda r: [k.label for k in r.levels[1]], ["[4, 6, 5, 1]", "[7, 4, 3, 1]"], "range(len(A)//2) wie in der Serie")],
    "Jede Liste wird in der Mitte geteilt; bei Länge 1 hört die Rekursion auf. Danach verschmilzt merge.", source="Serie 6, Aufgabe 2; Lösung eigen")
_ex("SW6.3", "Türme von Hanoi, 3 Scheiben", "Entwickeln Sie einen rekursiven Algorithmus für die Türme von Hanoi (Turm in der Mitte, Ziel rechts) und führen Sie ihn für n = 3 aus.",
    lambda: rk.hanoi(3),
    [E("7 = 2³ − 1 Züge", lambda r: r.count, 2 ** 3 - 1, "von Hand"),
     E("alle Züge regelkonform, Turm am Ziel", lambda r: _hanoi_regeln(r.moves, 3), True, "unabhängig nachgespielt"),
     E("optimal: weniger geht nicht (Python-Breitensuche über alle Zustände)", lambda r: _hanoi_min(3), 7, "unabhängig: BFS")],
    "hanoi(n−1, start, hilf, ziel); grösste Scheibe start → ziel; hanoi(n−1, hilf, ziel, start). Start ist hier B (Mitte), Ziel C (rechts).",
    source="Serie 6, Aufgabe 3 (Mitschrift: »n=3 → 3 Plätze«); Lösung eigen")
_ex("SW6.4a", "Rekursionsbeziehung der Summe", "Bestimmen Sie für die rekursive Berechnung der Summe eine Big-Θ-Abschätzung.",
    lambda: rk.solve_recurrence([1], 1, {1: 1}),
    [E("T(n) = n für alle n (Iteration)", lambda r: [r.at(n) for n in range(1, 9)], [_t_iter([1], lambda m: 1, {1: 1}, n) for n in range(1, 9)], "eigene Rechnung: T(n) = T(n−1) + 1"),
     E("linear: T(n) = n", lambda r: str(r.solution), "n", "eigene Rechnung")],
    "Eigene Herleitung: T(n) = T(n−1) + c, T(1) = c. Die Beziehung in der PDF-Serie ist ein Bild und nicht lesbar. TODO: mit der Vorlesung abgleichen.",
    source="Serie 6, Aufgabe 4 (Beziehung eigene Annahme, TODO); Lösung eigen")
_ex("SW6.4b", "Rekursionsbeziehung der Fakultät", "Bestimmen Sie für die rekursive Berechnung der Fakultät eine Big-Θ-Abschätzung.",
    lambda: rk.solve_recurrence([1], 1, {1: 1}),
    [E("T(n) = n (je Aufruf eine Multiplikation)", lambda r: [r.at(n) for n in range(1, 9)], list(range(1, 9)), "eigene Rechnung: T(n) = T(n−1) + 1")],
    "Wie bei der Summe: ein Aufruf mit n−1 und konstanter Aufwand. TODO: Schrittzählung der Vorlesung abgleichen.",
    source="Serie 6, Aufgabe 4 (Beziehung eigene Annahme, TODO); Lösung eigen")
for _k, (_a, _b, _kk, _th) in {"a": (2, 2, 0, "Θ(n)"), "b": (2, 2, 1, "Θ(n log n)"), "c": (2, 2, 2, "Θ(n²)")}.items():
    _ex(f"SW6.5{_k}", f"T{_k.upper()}(n) = 2·T(n/2) + {'1' if _kk == 0 else 'n' if _kk == 1 else 'n²'}", f"Bestimmen Sie eine Big-Θ-Abschätzung für T(n) = 2·T(n/2) + {'1' if _kk == 0 else 'n' if _kk == 1 else 'n²'}, T(1) = 1.",
        (lambda a=_a, b=_b, kk=_kk: rk.master_theorem(a, b, kk)),
        [E(f"{_th}", lambda r: r.theta, _th, "Satz auf Seite 7 der Serie, Fall bᵏ " + ("< a" if _kk == 0 else "= a" if _kk == 1 else "> a")),
         E("T(2^j) stimmt mit der Rekursion überein", lambda r, a=_a, b=_b, kk=_kk: all(int(v) == _t_dc(a, b, kk, m) for m, v in r.values.items()), True, "unabhängig: Python-Rekursion"),
         E("T(n)/g(n) liegt zwischen 1 und 2 (n = 2^1 … 2^16)", lambda r, a=_a, b=_b, kk=_kk: all(1 <= _t_dc(a, b, kk, 2 ** j) / _g_theta(kk, j) <= 2 for j in range(1, 17)), True,
           "unabhängig: Θ-Probe mit g = n, n·log₂ n bzw. n²")],
        "Rekursionsbaum für n = 2ᵏ zeichnen: Ebene i hat 2ⁱ Knoten der Grösse n/2ⁱ.", source=f"Serie 6, Aufgabe 5 (Teil {_k}); Lösung eigen")
_ex("SW6.6", "mergeSort ist in Θ(n log n)", "Zeigen Sie, dass mergeSort in Θ(n log n) ist (merge: c·n Schritte, Annahme).",
    lambda: rk.master_theorem(2, 2, 1),
    [E("Θ(n log n)", lambda r: r.theta, "Θ(n log n)", "T(n) = 2T(n/2) + n: a = 2, bᵏ = 2"),
     E("T(2^j) = 2^j·(j + 1)", lambda r: all(int(v) == int(m) * (m.bit_length()) for m, v in r.values.items()), True, "von Hand: j + 1 Ebenen mit je n Schritten"),
     E("genaue Rekursion mit ⌈n/2⌉ + ⌊n/2⌋: n·log₂ n ≤ T(n) ≤ 2·n·log₂ n (2 ≤ n ≤ 1000)",
       lambda r: all(n * _math.log2(n) <= _t_merge(n) <= 2 * n * _math.log2(n) for n in range(2, 1001)), True, "unabhängig: Python-Rekursion")],
    "Ebenen: log₂ n + 1; auf jeder Ebene werden insgesamt n Elemente verschmolzen. Annahme: merge braucht c·n Schritte (steht nicht in der Serie).",
    source="Serie 6, Aufgabe 6; Lösung eigen")
_F7 = {"a": (4, 3, 2, "Θ(n²)"), "b": (2, 3, 4, "Θ(n⁴)"), "c": (16, 4, 2, "Θ(n² log n)"), "d": (4, 2, 0, "Θ(n²)"), "e": (4, 4, 1, "Θ(n log n)")}
for _k, (_a, _b, _kk, _th) in _F7.items():
    _w = "1" if _kk == 0 else "n" if _kk == 1 else f"n{'²³⁴'[_kk - 2]}"
    _ex(f"SW6.7{_k}", f"T(n) = {_a}·T(n/{_b}) + {_w}", f"Bestimmen Sie mit dem Satz eine Big-Θ-Abschätzung für T(n) = {_a}·T(n/{_b}) + {_w}, T(1) = 1.",
        (lambda a=_a, b=_b, kk=_kk: rk.master_theorem(a, b, kk)),
        [E(_th, lambda r: r.theta, _th, "Satz auf Seite 7 der Serie: bᵏ " + ("<" if _b ** _kk < _a else "=" if _b ** _kk == _a else ">") + " a"),
         E("Fall nach bᵏ ⋚ a", lambda r, a=_a, b=_b, kk=_kk: r.case, (1 if _b ** _kk < _a else 2 if _b ** _kk == _a else 3), "von Hand"),
         E("T(b^j) stimmt mit der Rekursion überein", lambda r, a=_a, b=_b, kk=_kk: all(int(v) == _t_dc(a, b, kk, m) for m, v in r.values.items()), True, "unabhängig: Python-Rekursion")],
        "b^k mit a vergleichen: kleiner → Θ(n^(log_b a)), gleich → Θ(n^k log n), grösser → Θ(n^k).", source=f"Serie 6, Aufgabe 7{_k}; Lösung eigen")
_F8 = {  # Teil: (c, F, Anfangswerte, Formel, Text der Beziehung, Text der Lösung)
    "a": ([5], 0, {0: 4}, lambda n: 4 * 5 ** n, "T(n) = 5·T(n−1), T(0) = 4", "T(n) = 4·5ⁿ"),
    "b": ([6, -9], 0, {0: 1, 1: 6}, lambda n: (1 + n) * 3 ** n, "T(n) = 6·T(n−1) − 9·T(n−2), T(0) = 1, T(1) = 6", "T(n) = (1 + n)·3ⁿ (doppelte Wurzel 3)"),
    "c": ([1, 2], 0, {0: 2, 1: 7}, lambda n: 3 * 2 ** n - (-1) ** n, "T(n) = T(n−1) + 2·T(n−2), T(0) = 2, T(1) = 7", "T(n) = 3·2ⁿ − (−1)ⁿ (Wurzeln 2 und −1)"),
    "d": ([3], "2**n", {1: 3}, lambda n: _Fr(7, 3) * 3 ** n - 2 ** (n + 1), "T(n) = 3·T(n−1) + 2ⁿ, T(1) = 3", "T(n) = 7/3·3ⁿ − 2ⁿ⁺¹"),
    "e": ([5, -6], "7*n", {0: 1, 1: 15}, lambda n: -33 * 2 ** n + _Fr(87, 4) * 3 ** n + _Fr(7, 2) * n + _Fr(49, 4),
          "T(n) = 5·T(n−1) − 6·T(n−2) + 7n, T(0) = 1, T(1) = 15", "T(n) = −33·2ⁿ + 87/4·3ⁿ + 7n/2 + 49/4"),
}
_FN8 = {"a": lambda n: 0, "b": lambda n: 0, "c": lambda n: 0, "d": lambda n: 2 ** n, "e": lambda n: 7 * n}
for _k, (_c, _f, _i, _fn, _rec, _txt) in _F8.items():
    _ex(f"SW6.8{_k}", f"Rekursionsbeziehung: {_rec}", f"Bestimmen Sie die Auflösung der Rekursionsbeziehung {_rec}.",
        (lambda c=_c, f=_f, i=_i: rk.solve_recurrence(c, f, i)),
        [E("Formel stimmt mit Iteration der Rekursion überein (bis n = 12)", lambda r, c=_c, i=_i, k=_k: all(_Fr(str(r.at(n))) == _t_iter(c, _FN8[k], i, n) for n in range(max(i), 13)), True, "unabhängig: Iteration mit Brüchen"),
         E("Formel von Hand hergeleitet", lambda r, fn=_fn, i=_i: all(_Fr(str(r.at(n))) == _Fr(fn(n)) for n in range(min(i), 13)), True, _txt)],
        "Charakteristische Gleichung (Satz 1), bei inhomogen Ansatz für Tₚ (Satz 2), Konstanten aus den Anfangswerten.", source=f"Serie 6, Aufgabe 8{_k}; Lösung eigen: {_txt}")

# -- Beispiele aus der Vorlesung / Mitschrift -------------------------------------------------------------------------------------------
_ex("SW6.B1", "mergeSort: Rekursionsbaum mit Rückgabewerten", "Wie kommt die sortierte Liste im Baum zustande?",
    lambda: rk.recursion_tree("mergeSort", [4, 6, 5, 1]),
    [E("Wurzel gibt [1, 4, 5, 6] zurück", lambda r: r.value, [1, 4, 5, 6], "Python sorted()"),
     E("7 Aufrufe", lambda r: r.calls, 7, "von Hand: 1 + 2 + 4")],
    "Base Case: Länge 1. Dann von unten nach oben verschmelzen.", source="Serie 6, Aufgabe 2 (Mitschrift: »Base Case = hat die Liste die Länge 1?«)")
_ex("SW6.B2", "Hanoi: Zuganzahl 2ⁿ − 1", "Wie viele Züge braucht der Turm mit n Scheiben?",
    lambda: rk.solve_recurrence([2], 1, {1: 1}),
    [E("T(n) = 2ⁿ − 1 (Iteration)", lambda r: [r.at(n) for n in range(1, 11)], [2 ** n - 1 for n in range(1, 11)], "von Hand: T(n) = 2T(n−1) + 1"),
     E("n = 64 (Legende der Serie): 2⁶⁴ − 1 Züge", lambda r: r.at(64), 2 ** 64 - 1, "von Hand")],
    "T(n) = 2·T(n−1) + 1: die n−1 Scheiben zweimal, dazu ein Zug.", source="Serie 6, Aufgabe 3 (Zuganzahl; Rekursionsbeziehung eigene Herleitung)")


# ==========================================================================
# Reihenfolge: je Woche erst die Serie, dann die Beispiele, dann das Lerngerüst
# ==========================================================================
def _sortierschluessel(e: Exercise):
    art = {"Serie": 0, "optional": 1, "Beispiel": 2, "Lerngerüst": 3}.get(e.kind, 4)
    teile = [(0, int(t), "") if t.isdigit() else (1, 0, t) for t in re.findall(r"\d+|[A-Za-z]+", e.id.split(".", 1)[1])]
    return e.week, art, teile


for _e in sorted(EXERCISES.values(), key=_sortierschluessel):
    EXERCISES[_e.id] = EXERCISES.pop(_e.id)
