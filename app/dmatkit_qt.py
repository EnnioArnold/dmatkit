#!/usr/bin/env python3
"""
dmatkit — Diskrete Mathematik mit Rechenweg, als Desktop-App im Swiss Style (Qt).

Gebaut zum Modul DMATH-ALGO der HSLU (Semesterwochen 1 bis 6). Die App rechnet
mit demselben Paket `dmatkit` wie das Notebook; sie ist nur eine andere
Oberfläche dafür. Sechs Seiten:

    Datentypen (SW1)        Boolesche Ausdrücke, Zahlsysteme, IEEE 754, Zweierkomplement, Mengen
    Aussagenlogik (SW2)     Wahrheitstabelle, Äquivalenz, Vereinfachen, Normalformen, Syntaxbaum
    Prädikatenlogik (SW3)   ∀ und ∃ über einem Universum, Duplikate, Quantoren negieren
    Beweiskalkül (SW4)      Folgerung ⊨, Resolution, KNF-Umformung, Ableitung
    Algorithmen (SW5)       Schritte zählen, Worst Case, Wachstum, grösstes n, Landau-Symbole
    Rekursion (SW6)         Rekursionsbaum, Hanoi, Master-Satz T(n) = a·T(n/b) + c·nᵏ, lineare Rekursionen

Jede Seite zeigt ihr Ergebnis als Karte mit zwei Ansichten: Rechenweg (gesetzte
Formeln und Tabellen) und Zeichnung. Die Prüfung mit Wolfram|Alpha ist optional
und braucht eine kostenlose AppID von https://developer.wolframalpha.com/
(Umgebungsvariable WOLFRAM_APPID oder einmalige Eingabe in der App).

Installation:
    pip install PySide6 matplotlib sympy numpy        # plotly freiwillig

Start (aus dem Ordner heraus, egal wie er heisst):
    python3 app/dmatkit_qt.py
"""

from __future__ import annotations

import ast
import base64
import html
import json
import math
import os
import re
import sys
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Sequence

# Matplotlib ohne eigenes Fenster: die Figuren entstehen im Hintergrund,
# gezeigt werden sie von der Qt-Leinwand der App.
try:
    import matplotlib
    matplotlib.use("agg")
except Exception:                                            # pragma: no cover
    pass

from PySide6.QtCore import (QBuffer, QByteArray, QIODevice, QObject, QPoint,
                            QPointF, QRectF, QSize, Qt, QThread, QTimer, QUrl, Signal)
from PySide6.QtGui import (QColor, QDesktopServices, QFont, QFontDatabase, QFontInfo, QFontMetricsF,
                           QGuiApplication, QImage, QPainter, QPen)
from PySide6.QtWidgets import (QApplication, QDialog, QFileDialog, QFrame,
                               QGraphicsDropShadowEffect, QHBoxLayout, QLabel,
                               QLineEdit, QMainWindow, QMenu, QPlainTextEdit,
                               QPushButton, QScrollArea, QSizePolicy,
                               QStackedWidget, QVBoxLayout, QWidget)

# Dieser Ordner ist selbst das Paket `dmatkit` — auch dann, wenn er nach dem
# GitHub-Download »dmatkit-main« heisst oder man `python3 app/dmatkit_qt.py`
# aufruft. Darum hier von Hand als Paket anmelden, vor allen Importen.
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # orb liegt neben dieser Datei
if "dmatkit" not in sys.modules:
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "dmatkit", os.path.join(_HERE, "__init__.py"),
        submodule_search_locations=[_HERE])
    _pkg = _ilu.module_from_spec(_spec)
    sys.modules["dmatkit"] = _pkg
    _spec.loader.exec_module(_pkg)

import numpy as np                                            # noqa: E402
import sympy as sp                                            # noqa: E402

import dmatkit as dk                                          # noqa: E402
from dmatkit.rechnen.core import fmt                          # noqa: E402
from dmatkit.rechnen.render import tex_any, text_any, tex_to_text   # noqa: E402
from orb import ThinkingOrb                                   # noqa: E402

# Animationen sind freiwillig: ohne manimgx sagt der Reiter, was zu tun ist.
try:
    from dmatkit.grafik import anim as dk_anim
except Exception:                                            # pragma: no cover
    dk_anim = None
try:
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
    from PySide6.QtMultimediaWidgets import QVideoWidget
    HAS_VIDEO = True
except Exception:                                            # pragma: no cover
    HAS_VIDEO = False

MINUS = "−"


# ==========================================================================
# Eingaben lesen und Fehler freundlich sagen
# ==========================================================================
#
# Alles in diesem und den folgenden Abschnitten bis »Farben und Schriften«
# kommt ohne Fenster aus und lässt sich im Test direkt aufrufen. Nichts davon
# benutzt eval: Zahlen, Listen und Mengen liest ast.literal_eval, Formeln und
# Prädikate liest der Kern selbst.

class InputError(ValueError):
    """Eine Eingabe, die der Mensch korrigieren kann — mit fertigem Satz."""


FORMULA_HELP = ("Erlaubt sind and/or/not/xor, ∧ ∨ ¬ ⊕, -> und <->, Klammern sowie "
                "Variablen wie A, B, P, Q (Beispiel: (A or B) and not C).")
MAX_INPUT = 4000


def _clip(text: str, what: str = "Die Eingabe") -> str:
    if len(text) > MAX_INPUT:
        raise InputError(f"{what} ist zu lang (höchstens {MAX_INPUT} Zeichen).")
    return text


def get_text(p: dict, key: str, what: str) -> str:
    s = (p.get(key) or "").strip()
    if not s:
        raise InputError(f"Bitte {what} eingeben.")
    return _clip(s)


def get_int(p: dict, key: str, what: str, lo: int | None = None, hi: int | None = None) -> int:
    raw = get_text(p, key, what)
    s = raw.replace(MINUS, "-").replace("'", "").replace("’", "").replace("_", "").replace(" ", "")
    try:
        value = int(s)
    except ValueError:
        raise InputError(f"»{raw}« ist keine ganze Zahl (gebraucht für {what}).") from None
    if lo is not None and value < lo:
        raise InputError(f"{what[0].upper() + what[1:]} muss mindestens {lo} sein (eingegeben: {value}).")
    if hi is not None and value > hi:
        raise InputError(f"{what[0].upper() + what[1:]} darf höchstens {hi} sein (eingegeben: {value}).")
    return value


def get_fraction(p: dict, key: str, what: str, positive: bool = True) -> Fraction:
    raw = get_text(p, key, what)
    try:
        value = Fraction(raw.replace(",", ".").replace(MINUS, "-").replace(" ", ""))
    except (ValueError, ZeroDivisionError):
        raise InputError(f"»{raw}« ist keine Zahl (gebraucht für {what}). Erlaubt sind 60, 0.5, "
                         "3/2 oder 1e9.") from None
    if positive and value <= 0:
        raise InputError(f"{what[0].upper() + what[1:]} muss grösser als 0 sein.")
    return value


def split_top(text: str, sep: str = ",") -> list[str]:
    """Teilt an `sep` ausserhalb von Klammern und Anführungszeichen."""
    parts, depth, quote, cur = [], 0, "", []
    for ch in text:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = ""
        elif ch in "'\"":
            quote = ch
            cur.append(ch)
        elif ch in "([{":
            depth += 1
            cur.append(ch)
        elif ch in ")]}":
            depth -= 1
            cur.append(ch)
        elif ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [x.strip() for x in parts if x.strip()]


def _literal(text: str, what: str):
    try:
        return ast.literal_eval(text.strip())
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        raise InputError(f"»{text.strip()[:40]}« lässt sich nicht lesen ({what}). Erlaubt sind "
                         "Zahlen, Texte in Anführungszeichen und Listen wie [3, 1, 4].") from None


def read_list(text: str, what: str = "die Liste") -> list:
    """»3, 2, 1« oder »[3, 2, 1]« oder »a b c« (Wörter) → Python-Liste."""
    s = _clip((text or "").strip())
    if not s:
        raise InputError(f"Bitte {what} eingeben, zum Beispiel 3, 1, 4, 1, 5.")
    if s[0] in "[(":
        value = _literal(s, what)
    else:
        value = [_literal(x, what) if re.fullmatch(r"[-+]?\d+(\.\d+)?|['\"].*['\"]", x)
                 else x for x in split_top(s.replace(";", ","))]
    if not isinstance(value, (list, tuple)):
        raise InputError(f"»{s[:40]}« ist keine Liste.")
    return list(value)


def read_kwargs(text: str) -> dict:
    """»x=25, Liste=[3, 1, 2]« → {'x': 25, 'Liste': [3, 1, 2]}."""
    out: dict = {}
    for part in split_top(_clip(text or "")):
        name, eq, value = part.partition("=")
        name = name.strip()
        if not eq or not re.fullmatch(r"[A-Za-z_]\w*", name):
            raise InputError(f"Eingaben bitte als Name=Wert schreiben (etwa x=25 oder "
                             f"Liste=[3, 1, 2], Element=2), nicht »{part[:30]}«.")
        out[name] = _literal(value, f"Wert von {name}")
    return out


_BOOLS = {"1": True, "0": False, "true": True, "false": False, "wahr": True, "falsch": False,
          "w": True, "f": False, "t": True}


def read_assign(text: str) -> dict:
    """»A=1, B=0« → {'A': True, 'B': False} (auch wahr/falsch, True/False)."""
    out = {}
    for part in split_top(_clip(text or "").replace(";", ",")):
        name, eq, value = part.partition("=")
        name, value = name.strip(), value.strip().lower()
        if not eq or not re.fullmatch(r"[A-Za-z_]\w*", name) or value not in _BOOLS:
            raise InputError(f"Belegungen bitte als A=1, B=0 schreiben (oder wahr/falsch), nicht »{part[:30]}«.")
        out[name] = _BOOLS[value]
    return out


def read_universe(text: str, what: str = "das Universum"):
    """
    Universum aus Text: range(1, 10), 1..9, {1, 2, 3}, [a, b], N, Z, R, Q.
    Gibt etwas zurück, das der Kern als Universum annimmt (range, Liste, Universum).
    """
    s = _clip((text or "").strip())
    if not s:
        raise InputError(f"Bitte {what} angeben, zum Beispiel range(1, 10) oder {{1, 2, 3}}.")
    if s.replace("ℕ", "N").replace("ℤ", "Z").replace("ℝ", "R").replace("ℚ", "Q") in (
            "N", "Z", "R", "Q", "N*"):
        return dk.named_universe(s.replace("ℕ", "N").replace("ℤ", "Z").replace("ℝ", "R")
                                 .replace("ℚ", "Q"))
    m = re.fullmatch(r"range\(\s*(-?\d+)\s*(?:,\s*(-?\d+)\s*)?(?:,\s*(-?\d+)\s*)?\)", s)
    if m:
        nums = [int(g) for g in m.groups() if g is not None]
        try:
            return range(*nums)
        except ValueError:
            raise InputError("Bei range(a, b, Schritt) darf der Schritt nicht 0 sein.") from None
    m = re.fullmatch(r"(-?\d+)\s*\.\.\.?\s*(-?\d+)", s)
    if m:
        return range(int(m.group(1)), int(m.group(2)) + 1)
    if s[0] in "{[(" and s[-1] in "}])":
        s = s[1:-1]
    items = []
    for x in split_top(s):
        if re.fullmatch(r"[-+]?\d+", x):
            items.append(int(x))
        elif re.fullmatch(r"[-+]?\d*\.\d+", x):
            items.append(float(x))
        else:
            items.append(x.strip("'\""))
    if not items:
        raise InputError(f"{what[0].upper() + what[1:]} ist leer. Schreibe zum Beispiel {{1, 2, 3}}.")
    return items


def read_universes(text: str, spec: str) -> dict:
    """
    »x: range(1, 4); y: {1, 2}« → {'x': range(1, 4), 'y': [1, 2]}. Ohne »name:«
    gilt das Universum für alle Variablen der Quantorenfolge.
    """
    try:
        prefix = dk.praedikate._parse_prefix(spec, {}, need_universe=False)  # ponytail: privat, aber die einzige Stelle, die Variablen zuverlässig liest
    except ValueError as exc:
        raise InputError(str(exc)) from None
    names = [v for _, v, _ in prefix]
    if not names:
        raise InputError("Die Quantorenfolge ist leer. Schreibe etwa ∀x, ∃y oder ∀x ∃y.")
    explicit = {n: u for _, n, u in prefix if u is not None}
    out: dict = {}
    pieces = split_top(_clip(text or ""), ";") if re.search(r"(^|;)\s*[A-Za-z_]\w*\s*:", text or "") else []
    if pieces:
        for piece in pieces:
            name, _, rest = piece.partition(":")
            out[name.strip()] = read_universe(rest, f"das Universum von {name.strip()}")
    elif (text or "").strip():
        uni = read_universe(text)
        out = {n: uni for n in names}
    for n in names:
        if n in explicit:
            out.setdefault(n, explicit[n])
        if n not in out:
            raise InputError(f"Für die Variable {n} fehlt das Universum. Schreibe zum Beispiel "
                             f"»{n}: range(1, 10)« oder ein gemeinsames Universum für alle.")
    return out


def read_funcs(text: str, what: str = "die Funktionen") -> dict:
    """Zeilen »A: n*log(n,2)« oder nur »n**2« → {'A': 'n*log(n,2)', …}."""
    out: dict = {}
    for k, line in enumerate(l.strip() for l in _clip(text or "").splitlines()):
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^\s*([A-Za-zÄÖÜäöü]\w{0,11}|[A-Za-z][²³]?)\s*[:=]\s*(.+)$", line)
        if m and not re.match(r"^\s*(log|ln|sqrt|factorial|exp|floor|ceiling)\s*\(", line):
            name, expr = m.group(1), m.group(2).strip()
        else:
            name, expr = chr(ord("A") + len(out)), line
        if name in out:
            raise InputError(f"Der Name {name} kommt zweimal vor.")
        out[name] = expr
    if not out:
        raise InputError(f"Bitte {what} eingeben, eine pro Zeile, zum Beispiel »A: n*log(n,2)«.")
    if len(out) > 8:
        raise InputError("Es sind höchstens 8 Funktionen möglich.")
    return out


def read_formulas(text: str, what: str = "die Wissensbasis") -> list[str]:
    """Eine Formel pro Zeile; Leerzeilen und Zeilen mit # werden übersprungen."""
    lines = [l.strip() for l in _clip(text or "").replace(";", "\n").splitlines()]
    out = [l for l in lines if l and not l.startswith("#")]
    if not out:
        raise InputError(f"Bitte {what} eingeben, eine Formel pro Zeile.")
    return out


def friendly_error(exc: BaseException) -> str:
    """Macht aus einer Ausnahme einen Satz, den man lesen will."""
    if isinstance(exc, InputError):
        return str(exc)
    if isinstance(exc, ZeroDivisionError):
        return "Hier wird durch 0 geteilt. Die Rechnung ist so nicht definiert — andere Werte wählen."
    if isinstance(exc, RecursionError):
        return "Der Ausdruck ist zu verschachtelt: die Rechnung wurde abgebrochen."
    if isinstance(exc, MemoryError):
        return "Die Rechnung braucht zu viel Speicher. Bitte kleinere Werte wählen."
    if isinstance(exc, SyntaxError):
        return f"Das lässt sich nicht lesen (Syntaxfehler). {FORMULA_HELP}"
    if isinstance(exc, NotImplementedError):
        return "Das kann dmatkit für diese Eingabe nicht rechnen" + (f": {exc}" if str(exc) else ".")
    if isinstance(exc, (ValueError, KeyError)) and str(exc):
        return str(exc).strip("'\"")
    return (f"Das hat nicht geklappt ({type(exc).__name__}: {exc}). Bitte die Eingabe prüfen — "
            "ein einfacheres Beispiel hilft herauszufinden, was stört.")


# ==========================================================================
# Die sechs Seiten als Daten: Felder, Rechnungen, Beispiele
# ==========================================================================
#
# Jede Seite hat eine Liste von Rechnungen (»Actions«). Jede Rechnung nennt
# ihre Eingabefelder; die Seite baut das Formular daraus selbst. Alle Eingaben
# sind Text in einem Wörterbuch `p`, so lassen sie sich merken und im Test
# direkt verwenden.

@dataclass
class Field:
    key: str
    caption: str
    kind: str = "line"          # line | lines | code | choice | preset
    default: str = ""
    ph: str = ""                # Platzhalter
    w: int = 0                  # > 0: kurzes Feld, steht mit Nachbarn in einer Zeile
    options: tuple = ()         # choice: ((Wert, Beschriftung), …); preset: ((Name, {Feld: Wert}), …)
    rows: int = 3
    pv: str = ""                # »formula«: Live-Vorschau der gelesenen Formel
    note: str = ""


@dataclass
class Action:
    key: str
    label: str
    desc: str
    fields: list
    decimal: bool = False       # nur dann zeigt die Seite den Schalter Exakt/Dezimal


def F(*a, **k) -> Field:
    return Field(*a, **k)


SET_OPS = (("∩", "∩"), ("∪", "∪"), ("\\", "∖"), ("komplement", "Aᶜ"))

DATENTYPEN = [
    Action("bool", "Boolescher Ausdruck", "Wahrheitswert Schritt für Schritt: not vor and vor or, Klammern zuerst",
           [F("expr", "Ausdruck", default="False or True and False and True or False",
              ph="False or True and False and True or False"),
            F("env", "Belegung der Variablen (optional)", ph="A=1, B=0", w=0),
            F("short", "Schreibweise", "choice", "short", options=(("short", "T / F"), ("long", "True / False")))]),
    Action("intop", "Ganzzahl-Operation", "a + b, a − b, a · b, a // b, a % b mit Probe a = q·b + r",
           [F("ia", "a", default="10", w=130), F("ib", "b", default="3", w=130),
            F("iop", "Operation", "choice", "//", options=(("+", "+"), ("-", "−"), ("*", "·"),
                                                        ("//", "//"), ("%", "%"), ("divmod", "beides")))]),
    Action("base", "Zahl → Zahlsystem", "Dezimalzahl in Binär, Hex oder jede Basis 2 bis 36: Division mit Rest, Stellenwerttabelle",
           [F("bn", "Dezimalzahl", default="25", w=170), F("bb", "Basis", default="2", w=80),
            F("bw", "Stellen (optional)", default="", w=130),
            F("bm", "Verfahren", "choice", "both", options=(("both", "beide"), ("division", "Division"),
                                                           ("table", "Tabelle")))]),
    Action("frombase", "Zahlsystem → Zahl", "Ziffernfolge zur Basis b in eine Dezimalzahl: Stellenwerte und Horner-Schema",
           [F("ft", "Ziffern", default="11001", w=220), F("fb", "Basis", default="2", w=80)]),
    Action("repeat", "Periodische Zahl → Bruch", "Kommazahl mit Periode in Klammern: zwei Zehnerpotenzen, subtrahieren",
           [F("rt", "Zahl mit Periode", default="2.10(365)", ph="2.10(365)", w=220),
            F("rb", "Basis", default="10", w=80)], decimal=True),
    Action("fracbin", "Bruch → Binär", "Nachkommaanteil fortlaufend mal 2 nehmen; die Periode wird erkannt",
           [F("xt", "Zahl oder Bruch", default="0.3", ph="0.3, 1/6 oder 35.2")]),
    Action("ieee", "IEEE 754", "Zahl als float (32 Bit) oder double (64 Bit): Vorzeichen, Exponent, Mantisse",
           [F("ix", "Zahl", default="35.2", ph="35.2 oder 0.00(10) (binär)"),
            F("ibits", "Genauigkeit", "choice", "32", options=(("32", "32 Bit"), ("64", "64 Bit"))),
            F("ibase", "Eingabe ist", "choice", "10", options=(("10", "Dezimal"), ("2", "Binär")))],
           decimal=True),
    Action("twos", "Zweierkomplement", "Ganze Zahl als Bitmuster fester Länge, auch rückwärts",
           [F("tn", "Zahl oder Muster", default="-10", w=200), F("tbits", "Bit", default="8", w=80),
            F("tdir", "Richtung", "choice", "fwd", options=(("fwd", "Zahl → Muster"), ("back", "Muster → Zahl")))]),
    Action("setop", "Mengenoperation", "Durchschnitt, Vereinigung, Differenz, Komplement mit Tabelle »∈ A?, ∈ B?«",
           [F("sa", "Menge A", default="{a, b, c}"), F("sop", "Operation", "choice", "∩", options=SET_OPS),
            F("sb", "Menge B", default="{b, c, d}"),
            F("su", "Grundmenge (nur für Aᶜ)", default="", ph="{a, b, c, d, e}")]),
]

LOGIK = [
    Action("tt", "Wahrheitstabelle", "Spalte je Teilformel von innen nach aussen, Einstufung und Modelle",
           [F("f", "Formel", default="(Q -> P) and not P", ph="(A or B) and not C", pv="formula"),
            F("order", "Zeilenreihenfolge", "choice", "0", options=(("0", "0 zuerst"), ("1", "1 zuerst")))]),
    Action("equiv", "Äquivalenz prüfen", "f ≡ g mit Wahrheitstabelle; sonst ein Gegenbeispiel",
           [F("f", "Formel f", default="P -> (Q | R)", pv="formula"),
            F("g", "Formel g", default="Q | (~P | R)", pv="formula")]),
    Action("impl", "Implikation prüfen", "Gilt f ⇒ g in jeder Zeile? Mit Umkehrung und Kontraposition",
           [F("f", "Formel f", default="P -> (Q | R)", pv="formula"),
            F("g", "Formel g", default="Q | (~P | R)", pv="formula")]),
    Action("simplify", "Vereinfachen", "Mit den Rechengesetzen, jeder Schritt mit Namen des Gesetzes",
           [F("f", "Formel", default="(~(~P | ~Q) & (P -> Q)) -> P", pv="formula")]),
    Action("nf", "Normalformen", "Kanonische DNF (Minterme) oder KNF (Maxterme) aus der Wahrheitstabelle",
           [F("f", "Formel", default="A xor B", pv="formula"),
            F("nfkind", "Normalform", "choice", "dnf", options=(("dnf", "DNF"), ("knf", "KNF")))]),
    Action("tree", "Syntaxbaum", "Aufbau einer Formel oder eines Terms wie 2*x+y als Baum",
           [F("expr", "Formel oder Term", default="((A -> B) & ~B) <-> ~A", ph="2*x + y"),
            F("tkind", "Art", "choice", "auto", options=(("auto", "automatisch"), ("formula", "Formel"),
                                                         ("term", "Term")))]),
]

QUANT_NOTE = ("Universum: range(1, 10), 1..9, {1, 2, 3}, N, Z. Bei mehreren Variablen "
              "»x: range(1, 4); y: {1, 2}«.")

PRAEDIKATE = [
    Action("quant", "Quantor auswerten", "∀ und ∃ (auch Folgen wie ∀x ∃y) über einem endlichen Universum",
           [F("qs", "Quantoren", default="∀x", ph="∀x   oder   ∃x   oder   ∀x ∃y", w=0),
            F("qp", "Prädikat P", default="x % 2 == 0", ph="x % 2 == 0"),
            F("qu", "Universum", default="range(1, 10)", ph="range(1, 10)", note=QUANT_NOTE)]),
    Action("list", "Duplikat in Liste", "∃i ∃j: i ≠ j ∧ a[i] = a[j] — und das Gegenteil, alle verschieden",
           [F("la", "Liste a", default="3, 2, 1, 1, 4, 4, 2, 3, 1", ph="3, 2, 1, 1, 4"),
            F("lm", "Frage", "choice", "dup", options=(("dup", "Doppelt vorhanden?"), ("distinct", "Alle verschieden?"))),
            F("lo", "Positionen", "choice", "1", options=(("1", "ab 1"), ("0", "ab 0")))]),
    Action("negq", "Quantor negieren", "¬∀ ≡ ∃¬ und ¬∃ ≡ ∀¬ an einem Prädikat, mit Gegenprobe",
           [F("qs", "Quantoren", default="∀x", ph="∀x   oder   ∃x ∀y"),
            F("qp", "Prädikat P", default="x % 2 == 0", ph="x % 2 == 0"),
            F("qu", "Universum", default="range(1, 10)", note=QUANT_NOTE)]),
    Action("negf", "Formel negieren", "Gegenteil einer Formel bilden, Negation hinter alle Quantoren schieben",
           [F("nfo", "Formel", default="∀x ∈ N: ∃y ∈ N: x > y",
              ph="∀x ∈ N: ∃y ∈ N: x > y", note="Quantoren ∀ ∃ mit »∈ Menge«, dazu ∧ ∨ ¬ → und Vergleiche."),
            F("nsimp", "Vergleiche", "choice", "1", options=(("1", "x ≥ y schreiben"), ("0", "¬(x < y) lassen")))]),
]

KALKUEL = [
    Action("entails", "Folgerung WB ⊨ Q", "Mit Wahrheitstabelle, Deduktionstheorem und Widerspruchsbeweis",
           [F("wb", "Wissensbasis (eine Formel pro Zeile)", "lines", "A -> B\nA", rows=4),
            F("q", "Anfrage Q", default="B", pv="formula"),
            F("em", "Methode", "choice", "all", options=(("all", "alle drei"), ("table", "Tabelle"),
                                                       ("deduction", "Deduktion"), ("contradiction", "Widerspruch")))]),
    Action("res", "Resolution", "KNF, negierte Anfrage, Resolventen bis zur leeren Klausel □",
           [F("wb", "Wissensbasis (eine Formel pro Zeile)", "lines", "A -> B\nA", rows=4),
            F("q", "Anfrage Q", default="B", pv="formula")]),
    Action("cnf", "KNF-Umformung", "Äquivalenz und Implikation auflösen, De Morgan, Distributivgesetz",
           [F("cf", "Formel", default="(A or B) -> (C and D)", pv="formula")]),
    Action("derive", "Ableitung WB ⊢ Q", "Schlussregeln: Modus ponens, tollens, Syllogismen, Resolution",
           [F("wb", "Wissensbasis (eine Formel pro Zeile)", "lines", "A -> B\nA", rows=4),
            F("q", "Anfrage Q", default="B", pv="formula")]),
]

#: Vorbefüllung des Code-Feldes: (Name, {Feld: Wert}); die Referenzalgorithmen kommen aus dem Kern.
REF_INPUTS = {
    "decToBin": "x=25", "linearSearch": "Liste=[3, 1, 4, 1, 5], Element=9",
    "selectionSort": "Liste=[5, 2, 4, 1, 3]", "insertionSort": "Liste=[5, 2, 4, 1, 3]",
    "bubbleSort": "Liste=[5, 2, 4, 1, 3]", "binarySearch": "Liste=[1, 3, 5, 7, 9, 11, 13], Element=7",
    "nestedLoops": "n=5"}
OWN_CODE = ("def summe(n):\n    s = 0\n    for i in range(1, n + 1):\n        s = s + i\n    return s")


def _presets() -> tuple:
    out = []
    for name in dk.reference_algorithms():
        out.append((name, {"code": dk.reference_code(name), "cin": REF_INPUTS.get(name, ""), "cfn": ""}))
    out.append(("Eigenes Beispiel: Summe 1 … n", {"code": OWN_CODE, "cin": "n=10", "cfn": ""}))
    return tuple(out)


FUNCS_SERIE5 = "A: n*log(n,2)\nB: n**2\nC: factorial(n)"
CODE_FIELD = F("code", "Code (Python-Teilmenge)", "code", dk.reference_code("decToBin"), rows=9,
               note="Zählregeln der Vorlesung; erlaubt sind def, for, while, if, Zuweisungen, Rechnen, Listen.")

ALGORITHMEN = [
    Action("count", "Schritte zählen", "Rechenschritte eines Programms nach den Zählregeln, Zeile für Zeile",
           [F("_ref", "Referenzalgorithmus laden", "preset", options=_presets()), CODE_FIELD,
            F("cin", "Eingaben", default="x=25", ph="x=25   oder   Liste=[3, 1, 2], Element=2"),
            F("cfn", "Funktion (optional)", default="", ph="Name der aufgerufenen Funktion", w=0),
            F("cbr", "if-Zählung", "choice", "actual", options=(("actual", "tatsächlich"), ("max", "teurer Zweig")))]),
    Action("case", "Worst / Best Case", "Grösste oder kleinste Schrittzahl über alle Eingaben der Grösse n, als Formel",
           [F("_ref", "Referenzalgorithmus laden", "preset", options=_presets()), CODE_FIELD,
            F("cn", "n-Werte (optional)", default="", ph="z. B. 1, 2, 3, 4, 5, 6, 7, 8"),
            F("ckind", "Fall", "choice", "worst", options=(("worst", "Worst Case"), ("best", "Best Case")))]),
    Action("growth", "Wachstum vergleichen", "Tabelle der Schrittzahlen mehrerer Funktionen und ihre Rangfolge",
           [F("gf", "Funktionen (eine pro Zeile)", "lines", FUNCS_SERIE5, rows=4,
              note="Namen n, log(n,2), ln, sqrt, factorial, exp, floor, ceiling; Potenz mit ** oder ^."),
            F("gn", "n-Werte", default="5, 10, 20, 50, 100", ph="5, 10, 20, 50"),
            F("gr", "Logarithmen", "choice", "none", options=(("none", "exakt"), ("ceil", "aufrunden"))),
            F("gt", "Zeit pro Schritt in s (optional)", default="", ph="1e-9", w=170)], decimal=True),
    Action("maxn", "Grösstes n", "Grösste Eingabe, die in einer Frist noch berechnet wird (z. B. 1 Minute)",
           [F("mf", "Funktionen (eine pro Zeile)", "lines", FUNCS_SERIE5, rows=4),
            F("mt", "Zeitlimit in Sekunden", default="60", w=170),
            F("ms", "Schritte pro Sekunde", default="1e9", w=170)], decimal=True),
    Action("landau", "Landau-Symbole", "f ∈ O(g), Ω(g) oder Θ(g) mit Zeugen C, c und x₀",
           [F("lf", "f(x)", default="4*x**5 + 15*x**3 - 10*x**2 - 8"),
            F("lg", "g(x)", default="x**6"),
            F("lr", "Symbol", "choice", "O", options=(("O", "O"), ("Omega", "Ω"), ("Theta", "Θ")))]),
]


REKURSION = [
    Action("tree", "Rekursionsbaum", "Aufrufe von summe, Fakultät, mergeSort oder Türme von Hanoi als Baum, mit Rückgabewerten",
           [F("ralg", "Algorithmus", "choice", "mergeSort", options=(("summe", "summe(n)"), ("fakultaet", "Fakultät n!"),
                                                                   ("mergeSort", "mergeSort(Liste)"), ("hanoi", "Türme von Hanoi"))),
            F("rarg", "Eingabe", default="4, 6, 5, 1, 7, 4, 3, 1", ph="n  oder  Liste: 4, 6, 5, 1",
              note="summe, Fakultät: n bis 12. Hanoi: n bis 7. mergeSort: Liste mit höchstens 32 ganzen Zahlen.")]),
    Action("master", "Master-Satz", "T(n) = a·T(n/b) + c·nᵏ: Rekursionsbaum je Ebene, dann Θ(n^log_b a), Θ(nᵏ log n) oder Θ(nᵏ)",
           [F("ma", "a (Teilprobleme)", default="2", w=130), F("mb", "b (Teilung n/b)", default="2", w=130),
            F("mk", "k (Zerlegen: nᵏ)", default="1", w=130), F("mc", "c", default="1", w=100)]),
    Action("linrec", "Lineare Rekursion", "T(n) = c₁·T(n−1) + c₂·T(n−2) + F(n): charakteristische Gleichung, Ansatz, Konstanten",
           [F("lc", "Koeffizienten c₁, c₂", default="6, -9", ph="5   oder   6, -9", w=0),
            F("lF", "F(n) (0 = homogen)", default="0", ph="0   oder   2**n   oder   7*n"),
            F("li", "Anfangswerte", default="T(0)=1, T(1)=6", ph="T(0)=1, T(1)=6")]),
]


@dataclass
class PageSpec:
    key: str
    title: str
    tab: str
    lede: str
    actions: list
    examples: list
    help: str = ""


def _ex(name, action, **p):
    return (name, dict(action=action, **p))


EX_DATENTYPEN = [
    _ex("Serie 1·1a: False or True and False and True or False", "bool"),
    _ex("Serie 1·1d: not(not True or not(False and not True)) or False", "bool",
        expr="not(not True or not(False and not True)) or False"),
    _ex("Mit Variablen: A and not B, A = 1, B = 0", "bool", expr="A and not B", env="A=1, B=0"),
    _ex("10 // 3 (mit Probe)", "intop", ia="10", ib="3", iop="//"),
    _ex("−7 // 2 (Rest ist nie negativ)", "intop", ia="-7", ib="2", iop="//"),
    _ex("10 % 3", "intop", ia="10", ib="3", iop="%"),
    _ex("17 : 5 mit Quotient und Rest", "intop", ia="17", ib="5", iop="divmod"),
    _ex("409 in Binär", "base", bn="409", bb="2"),
    _ex("25 in Binär, 8 Stellen", "base", bn="25", bb="2", bw="8"),
    _ex("255 in Hex (Basis 16)", "base", bn="255", bb="16"),
    _ex("(11001)₂ → Dezimal", "frombase", ft="11001", fb="2"),
    _ex("FF (Basis 16) → Dezimal", "frombase", ft="FF", fb="16"),
    _ex("Serie 1·3: 2.10(365) → Bruch", "repeat", rt="2.10(365)"),
    _ex("14.2(51) → Bruch", "repeat", rt="14.2(51)"),
    _ex("14.05 → Bruch", "repeat", rt="14.05"),
    _ex("Lerngerüst: 0.173(15) → Bruch", "repeat", rt="0.173(15)"),
    _ex("Binär: 0.0(1001) → Bruch (Basis 2)", "repeat", rt="0.0(1001)", rb="2"),
    _ex("0.625 → Binär", "fracbin", xt="0.625"),
    _ex("0.3 → Binär (Periode)", "fracbin", xt="0.3"),
    _ex("0.1 → Binär", "fracbin", xt="0.1"),
    _ex("35.2 → Binär", "fracbin", xt="35.2"),
    _ex("Serie 1·5a: 0.00(10) binär als float", "ieee", ix="0.00(10)", ibits="32", ibase="2"),
    _ex("Serie 1·5b: 35.2 als float", "ieee", ix="35.2", ibits="32", ibase="10"),
    _ex("35.2 als double (64 Bit)", "ieee", ix="35.2", ibits="64", ibase="10"),
    _ex("Lerngerüst: 0.0(110) binär als float", "ieee", ix="0.0(110)", ibits="32", ibase="2"),
    _ex("−10 im Zweierkomplement (8 Bit)", "twos", tn="-10", tbits="8", tdir="fwd"),
    _ex("5 im Zweierkomplement (8 Bit)", "twos", tn="5", tbits="8", tdir="fwd"),
    _ex("11110110 → Zahl (8 Bit)", "twos", tn="11110110", tbits="8", tdir="back"),
    _ex("{a,b,c} ∩ {b,c,d}", "setop", sa="{a, b, c}", sop="∩", sb="{b, c, d}", su=""),
    _ex("{a,b,c} ∪ {b,c,d}", "setop", sa="{a, b, c}", sop="∪", sb="{b, c, d}", su=""),
    _ex("{a,b,c} ∖ {b,c,d}", "setop", sa="{a, b, c}", sop="\\", sb="{b, c, d}", su=""),
    _ex("Komplement von {a,b} in {a,b,c,d,e}", "setop", sa="{a, b}", sop="komplement", sb="", su="{a, b, c, d, e}"),
]

EX_LOGIK = [
    _ex("(Q → P) ∧ ¬P", "tt", f="(Q -> P) and not P"),
    _ex("(A ∨ B) ∧ ¬C", "tt", f="(A or B) and not C"),
    _ex("A xor (B or C)", "tt", f="A xor (B or C)"),
    _ex("Tautologie: A ∨ ¬A", "tt", f="A or not A"),
    _ex("Serie 2·4: P → (Q ∨ R) ≡ Q ∨ (¬P ∨ R)", "equiv", f="P -> (Q | R)", g="Q | (~P | R)"),
    _ex("Nicht äquivalent: P → (Q ∨ R) und (Q ∧ ¬R) → P", "equiv", f="P -> (Q | R)", g="(Q & ~R) -> P"),
    _ex("Kontraposition: A → B ≡ ¬B → ¬A", "equiv", f="A -> B", g="not B -> not A"),
    _ex("Implikation: A ∧ B ⇒ A", "impl", f="A and B", g="A"),
    _ex("Vorlesung: (¬(¬P ∨ ¬Q) ∧ (P → Q)) → P", "simplify", f="(~(~P | ~Q) & (P -> Q)) -> P"),
    _ex("Absorption: A ∨ (A ∧ B)", "simplify", f="A or (A and B)"),
    _ex("De Morgan: ¬(A ∧ B) ∨ A", "simplify", f="not (A and B) or A"),
    _ex("DNF von A xor B", "nf", f="A xor B", nfkind="dnf"),
    _ex("KNF von A ↔ B", "nf", f="A <-> B", nfkind="knf"),
    _ex("Syntaxbaum ((A→B) ∧ ¬B) ↔ ¬A", "tree", expr="((A -> B) & ~B) <-> ~A", tkind="auto"),
    _ex("Syntaxbaum (P ∨ Q) → R ↔ (P→R) ∧ (Q→R)", "tree", expr="(P | Q) -> R <-> (P -> R) & (Q -> R)", tkind="auto"),
    _ex("Syntaxbaum des Terms 2·x + y", "tree", expr="2*x + y", tkind="auto"),
]

EX_PRAEDIKATE = [
    _ex("∀x gerade, x ∈ 1 … 9", "quant", qs="∀x", qp="x % 2 == 0", qu="range(1, 10)"),
    _ex("∃x gerade, x ∈ 1 … 9", "quant", qs="∃x", qp="x % 2 == 0", qu="range(1, 10)"),
    _ex("∀x: x < 10, x ∈ 1 … 9", "quant", qs="∀x", qp="x < 10", qu="range(1, 10)"),
    _ex("∃x: x² = 49", "quant", qs="∃x", qp="x*x == 49", qu="range(0, 12)"),
    _ex("∀x ∃y: x < y über 1 … 3", "quant", qs="∀x ∃y", qp="x < y", qu="x: range(1, 4); y: range(1, 4)"),
    _ex("∃x ∀y: x ≤ y über 1 … 3", "quant", qs="∃x ∀y", qp="x <= y", qu="x: range(1, 4); y: range(1, 4)"),
    _ex("Liste 3 2 1 1 4 4 2 3 1: Duplikat?", "list", la="3, 2, 1, 1, 4, 4, 2, 3, 1", lm="dup"),
    _ex("Liste 3 2 1 4: alle verschieden?", "list", la="3, 2, 1, 4", lm="distinct"),
    _ex("Liste 3 2 1 3: alle verschieden?", "list", la="3, 2, 1, 3", lm="distinct"),
    _ex("¬∀x gerade ≡ ∃x ¬gerade", "negq", qs="∀x", qp="x % 2 == 0", qu="range(10)"),
    _ex("¬∃x ∀y: x ≤ y", "negq", qs="∃x ∀y", qp="x <= y", qu="x: range(1, 4); y: range(1, 4)"),
    _ex("Negation von ∀x ∈ N: ∃y ∈ N: x > y", "negf", nfo="∀x ∈ N: ∃y ∈ N: x > y"),
    _ex("Serie 3·7: ∀u ∃a ∀f: Freund(u,f) → Geladen(f,a)", "negf",
        nfo="∀u ∈ U: ∃a ∈ A: ∀f ∈ U: (Freund(u, f) → Geladen(f, a))"),
]

EX_KALKUEL = [
    _ex("A → B und A ⊨ B", "entails", wb="A -> B\nA", q="B"),
    _ex("Vorlesung: (A ∨ B) ∧ (C ∨ (¬B ∧ ¬A) ∨ B) ⊨ A ∨ B", "entails",
        wb="(A or B) and (C or (not B and not A) or B)", q="A or B"),
    _ex("A ∨ B ⊭ A (Gegenbeispiel)", "entails", wb="A or B", q="A"),
    _ex("Serie 4·5: (S ∨ N) ∧ ¬S ∧ (P ∨ N) ∧ (¬N ∨ P) ⊨ ¬S ∧ N ∧ P", "res",
        wb="(S or N) and not S and (P or N) and (not N or P)", q="not S and N and P"),
    _ex("Resolution: A → B, B → C ⊨ A → C", "res", wb="A -> B\nB -> C", q="A -> C"),
    _ex("Resolution: A ∨ B ⊭ A", "res", wb="A or B", q="A"),
    _ex("Serie 4·4: KNF von (A ∨ B) → (C ∧ D)", "cnf", cf="(A or B) -> (C and D)"),
    _ex("KNF von A ↔ B", "cnf", cf="A <-> B"),
    _ex("Ableitung: (A ∨ B) ∧ (B → C) ⊢ A ∨ C", "derive", wb="(A or B) and (B -> C)", q="A or C"),
    _ex("Modus ponens: p → q, p ⊢ q", "derive", wb="p -> q\np", q="q"),
]

EX_ALGORITHMEN = [
    _ex("decToBin(25): 3 + 7·5 + 1 Schritte", "count", code=dk.reference_code("decToBin"), cin="x=25", cfn="", cbr="actual"),
    _ex("linearSearch: Element nicht vorhanden (2n + 1)", "count", code=dk.reference_code("linearSearch"),
        cin=REF_INPUTS["linearSearch"], cfn="", cbr="actual"),
    _ex("selectionSort einer 5er-Liste", "count", code=dk.reference_code("selectionSort"),
        cin=REF_INPUTS["selectionSort"], cfn="", cbr="actual"),
    _ex("Eigenes Programm: Summe 1 … 10", "count", code=OWN_CODE, cin="n=10", cfn="", cbr="actual"),
    _ex("Worst Case insertionSort", "case", code=dk.reference_code("insertionSort"), cn="", ckind="worst"),
    _ex("Worst Case decToBin", "case", code=dk.reference_code("decToBin"), cn="", ckind="worst"),
    _ex("Best Case linearSearch", "case", code=dk.reference_code("linearSearch"), cn="", ckind="best"),
    _ex("Serie 5·1: A, B, C bei n = 5 (aufgerundet)", "growth", gf=FUNCS_SERIE5, gn="5", gr="ceil", gt=""),
    _ex("Wachstum bei n = 5 … 100 (exakt)", "growth", gf=FUNCS_SERIE5, gn="5, 10, 20, 50, 100", gr="none", gt="1e-9"),
    _ex("Serie 5·1: grösstes n in 1 Minute (10⁹ Schritte/s)", "maxn", mf=FUNCS_SERIE5, mt="60", ms="1e9"),
    _ex("Nur n² in einer Stunde", "maxn", mf="n**2", mt="3600", ms="1e9"),
    _ex("4x⁵ + 15x³ − 10x² − 8 ∈ O(x⁶)", "landau", lf="4*x**5 + 15*x**3 - 10*x**2 - 8", lg="x**6", lr="O"),
    _ex("5x² − 2x + 8 ∈ Θ(x²)", "landau", lf="5*x**2 - 2*x + 8", lg="x**2", lr="Theta"),
    _ex("x³ ∉ O(x²)", "landau", lf="x**3", lg="x**2", lr="O"),
]

EX_REKURSION = [
    _ex("Serie 6·1: summe(5), Rekursionsbaum", "tree", ralg="summe", rarg="5"),
    _ex("Serie 6·1: Fakultät 5", "tree", ralg="fakultaet", rarg="5"),
    _ex("Serie 6·2: mergeSort [4, 6, 5, 1, 7, 4, 3, 1]", "tree", ralg="mergeSort", rarg="4, 6, 5, 1, 7, 4, 3, 1"),
    _ex("Serie 6·3: Türme von Hanoi, n = 3", "tree", ralg="hanoi", rarg="3"),
    _ex("Serie 6·5: TA(n) = 2·T(n/2) + 1", "master", ma="2", mb="2", mk="0", mc="1"),
    _ex("Serie 6·5: TB(n) = 2·T(n/2) + n (mergeSort)", "master", ma="2", mb="2", mk="1", mc="1"),
    _ex("Serie 6·5: TC(n) = 2·T(n/2) + n²", "master", ma="2", mb="2", mk="2", mc="1"),
    _ex("Serie 6·7a: 4·T(n/3) + n²", "master", ma="4", mb="3", mk="2", mc="1"),
    _ex("Serie 6·7c: 16·T(n/4) + n²", "master", ma="16", mb="4", mk="2", mc="1"),
    _ex("Serie 6·7d: 4·T(n/2) + 1", "master", ma="4", mb="2", mk="0", mc="1"),
    _ex("Serie 6·8a: T(n) = 5·T(n−1), T(0) = 4", "linrec", lc="5", lF="0", li="T(0)=4"),
    _ex("Serie 6·8b: T(n) = 6·T(n−1) − 9·T(n−2)", "linrec", lc="6, -9", lF="0", li="T(0)=1, T(1)=6"),
    _ex("Serie 6·8c: T(n) = T(n−1) + 2·T(n−2)", "linrec", lc="1, 2", lF="0", li="T(0)=2, T(1)=7"),
    _ex("Serie 6·8d: T(n) = 3·T(n−1) + 2ⁿ, T(1) = 3", "linrec", lc="3", lF="2**n", li="T(1)=3"),
    _ex("Serie 6·8e: T(n) = 5·T(n−1) − 6·T(n−2) + 7n", "linrec", lc="5, -6", lF="7*n", li="T(0)=1, T(1)=15"),
    _ex("Hanoi: T(n) = 2·T(n−1) + 1", "linrec", lc="2", lF="1", li="T(1)=1"),
]

PAGES: dict[str, PageSpec] = {
    "datentypen": PageSpec("datentypen", "Datentypen", "Datentypen (SW1)",
                           "Boolesche Ausdrücke, Zahlsysteme, Brüche, IEEE 754, Zweierkomplement und Mengen.",
                           DATENTYPEN, EX_DATENTYPEN),
    "logik": PageSpec("logik", "Aussagenlogik", "Aussagenlogik (SW2)",
                      "Wahrheitstabellen, Äquivalenz, Vereinfachen mit Gesetzen, Normalformen und Syntaxbäume.",
                      LOGIK, EX_LOGIK),
    "praedikate": PageSpec("praedikate", "Prädikatenlogik", "Prädikatenlogik (SW3)",
                           "Quantoren ∀ und ∃ über einem Universum, Duplikate in Listen und Negation.",
                           PRAEDIKATE, EX_PRAEDIKATE),
    "kalkuel": PageSpec("kalkuel", "Beweiskalkül", "Beweiskalkül (SW4)",
                        "Folgerung WB ⊨ Q mit drei Methoden, Resolution, KNF-Umformung und Ableitung.",
                        KALKUEL, EX_KALKUEL),
    "algorithmen": PageSpec("algorithmen", "Algorithmen", "Algorithmen (SW5)",
                            "Rechenschritte zählen, Worst Case, Wachstum vergleichen, grösstes n und Landau-Symbole.",
                            ALGORITHMEN, EX_ALGORITHMEN),
    "rekursion": PageSpec("rekursion", "Rekursion", "Rekursion (SW6)",
                          "Rekursionsbäume, Türme von Hanoi, Master-Satz für divide and conquer und lineare Rekursionsbeziehungen.",
                          REKURSION, EX_REKURSION),
}
PAGE_KEYS = tuple(PAGES)


def defaults_of(spec: PageSpec) -> dict:
    out = {"action": spec.actions[0].key}
    for act in spec.actions:
        for fld in act.fields:
            if fld.kind != "preset":
                out.setdefault(fld.key, fld.default)
    return out


DEFAULTS: dict[str, dict] = {k: defaults_of(s) for k, s in PAGES.items()}
EXAMPLES: dict[str, list] = {k: s.examples for k, s in PAGES.items()}


def action_of(page: str, key: str) -> Action:
    for act in PAGES[page].actions:
        if act.key == key:
            return act
    raise InputError("Unbekannte Rechnung.")


# ==========================================================================
# Die Rechnungen: Eingaben (Text) -> dmatkit-Ergebnis
# ==========================================================================

def calc_datentypen(p: dict):
    a = p.get("action", "bool")
    if a == "bool":
        env = read_assign(p.get("env", ""))
        return dk.eval_bool(get_text(p, "expr", "den Ausdruck"), short=p.get("short") != "long", **env)
    if a == "intop":
        x, y = get_int(p, "ia", "die Zahl a"), get_int(p, "ib", "die Zahl b")
        op = p.get("iop", "//")
        if op == "divmod":
            return dk.divmod_steps(x, y)
        return dk.int_op(x, op, y)
    if a == "base":
        width = get_int(p, "bw", "die Stellenzahl", 1, 128) if (p.get("bw") or "").strip() else None
        return dk.to_base(get_int(p, "bn", "die Zahl", 0), get_int(p, "bb", "die Basis", 2, 36),
                          p.get("bm", "both"), width)
    if a == "frombase":
        return dk.from_base(get_text(p, "ft", "die Ziffernfolge"), get_int(p, "fb", "die Basis", 2, 36))
    if a == "repeat":
        return dk.repeating_to_fraction(get_text(p, "rt", "die Zahl, etwa 2.10(365)"),
                                        get_int(p, "rb", "die Basis", 2, 36))
    if a == "fracbin":
        return dk.frac_to_binary(get_text(p, "xt", "die Zahl, etwa 0.3"))
    if a == "ieee":
        return dk.ieee754(get_text(p, "ix", "die Zahl, etwa 35.2"), int(p.get("ibits", "32")),
                          base=int(p.get("ibase", "10")))
    if a == "twos":
        if p.get("tdir") == "back":
            pattern = get_text(p, "tn", "das Bitmuster, etwa 11110110")
            if not re.fullmatch(r"[01 ]+", pattern):
                raise InputError(f"»{pattern}« ist kein Bitmuster: erlaubt sind nur 0 und 1.")
            return dk.from_twos_complement(pattern.replace(" ", ""))
        return dk.twos_complement(get_int(p, "tn", "die Zahl"), get_int(p, "tbits", "die Bitzahl", 2, 64))
    if a == "setop":
        op = p.get("sop", "∩")
        uni = (p.get("su") or "").strip()
        if op == "komplement":
            if not uni:
                raise InputError("Für das Komplement braucht es eine Grundmenge, etwa {a, b, c, d, e}.")
            return dk.set_op(get_text(p, "sa", "die Menge A"), op, None, uni)
        return dk.set_op(get_text(p, "sa", "die Menge A"), op, get_text(p, "sb", "die Menge B"))
    raise InputError("Unbekannte Rechnung.")


def calc_logik(p: dict):
    a = p.get("action", "tt")
    if a == "tt":
        return dk.truth_table(get_text(p, "f", "die Formel"), true_first=p.get("order") == "1")
    if a == "equiv":
        return dk.check_equivalence(get_text(p, "f", "die Formel f"), get_text(p, "g", "die Formel g"))
    if a == "impl":
        return dk.check_implication(get_text(p, "f", "die Formel f"), get_text(p, "g", "die Formel g"))
    if a == "simplify":
        return dk.simplify(get_text(p, "f", "die Formel"))
    if a == "nf":
        f = get_text(p, "f", "die Formel")
        return dk.canonical_knf(f) if p.get("nfkind") == "knf" else dk.canonical_dnf(f)
    if a == "tree":
        return dk.syntax_tree(get_text(p, "expr", "die Formel oder den Term"), p.get("tkind", "auto"))
    raise InputError("Unbekannte Rechnung.")


def calc_praedikate(p: dict):
    a = p.get("action", "quant")
    if a in ("quant", "negq"):
        spec = get_text(p, "qs", "die Quantoren, etwa ∀x")
        pred = get_text(p, "qp", "das Prädikat, etwa x % 2 == 0")
        unis = read_universes(p.get("qu", ""), spec)
        return (dk.quantify if a == "quant" else dk.negate_quantifier)(spec, pred, unis)
    if a == "list":
        values = read_list(p.get("la", ""), "die Liste")
        if len(values) > 200:
            raise InputError("Die Liste darf höchstens 200 Elemente haben.")
        one = p.get("lo", "1") != "0"
        return (dk.has_duplicate if p.get("lm") != "distinct" else dk.all_distinct)(values, one_based=one)
    if a == "negf":
        return dk.negate_formula(get_text(p, "nfo", "die Formel"), simplify=p.get("nsimp", "1") != "0")
    raise InputError("Unbekannte Rechnung.")


def calc_kalkuel(p: dict):
    a = p.get("action", "entails")
    if a == "cnf":
        return dk.to_cnf(get_text(p, "cf", "die Formel"))
    wb = read_formulas(p.get("wb", ""))
    q = get_text(p, "q", "die Anfrage Q")
    if a == "entails":
        return dk.entails(wb, q, method=p.get("em", "all"))
    if a == "res":
        return dk.resolution(wb, q)
    if a == "derive":
        return dk.derive(wb, q)
    raise InputError("Unbekannte Rechnung.")


def _reference_name(code: str) -> str | None:
    """Der Name eines Referenzalgorithmus, wenn der Code genau dessen Text ist."""
    norm = lambda s: "\n".join(l.rstrip() for l in s.strip().splitlines())
    for name in dk.reference_algorithms():
        if norm(code) == norm(dk.reference_code(name)):
            return name
    return None


def _int_list(text: str, what: str) -> list[int] | None:
    if not (text or "").strip():
        return None
    out = []
    for v in read_list(text, what):
        if isinstance(v, bool) or not isinstance(v, int):
            raise InputError(f"Bitte {what} als ganze Zahlen angeben, etwa 1, 2, 4, 8.")
        out.append(v)
    if len(out) > 60:
        raise InputError(f"Zu viele Werte in {what} (höchstens 60).")
    return out


def calc_algorithmen(p: dict):
    a = p.get("action", "count")
    if a in ("count", "case"):
        code = get_text(p, "code", "den Code")
    if a == "count":
        inputs = read_kwargs(p.get("cin", ""))
        func = (p.get("cfn") or "").strip() or None
        return dk.count_steps(code, func=func, branch=p.get("cbr", "actual"), **inputs)
    if a == "case":
        algo = _reference_name(code) or code
        nvals = _int_list(p.get("cn", ""), "die n-Werte")
        fn = dk.best_case if p.get("ckind") == "best" else dk.worst_case
        return fn(algo, nvals)
    if a == "growth":
        nvals = _int_list(p.get("gn", ""), "die n-Werte") or [5, 10, 20, 50, 100]
        step = float(get_fraction(p, "gt", "die Zeit pro Schritt")) if (p.get("gt") or "").strip() else None
        return dk.compare_growth(read_funcs(p.get("gf", "")), nvals, round_log=p.get("gr", "none"),
                                 step_time_s=step)
    if a == "maxn":
        funcs = read_funcs(p.get("mf", ""))
        limit = get_fraction(p, "mt", "das Zeitlimit")
        sps = int(get_fraction(p, "ms", "die Schritte pro Sekunde"))
        if len(funcs) == 1:
            return dk.max_input_size(next(iter(funcs.values())), float(limit), steps_per_second=sps)
        return dk.max_input_sizes(funcs, float(limit), steps_per_second=sps)
    if a == "landau":
        f, g = get_text(p, "lf", "die Funktion f"), get_text(p, "lg", "die Funktion g")
        fn = {"O": dk.big_o, "Omega": dk.big_omega, "Theta": dk.big_theta}[p.get("lr", "O")]
        return fn(f, g)
    raise InputError("Unbekannte Rechnung.")

def _read_init(text: str) -> dict:
    """»T(0)=1, T(1)=6« oder »1, 6« (dann ab Index 0) -> {0: 1, 1: 6}."""
    s = (text or "").replace(MINUS, "-")
    pairs = re.findall(r"T\(\s*(-?\d+)\s*\)\s*=\s*(-?\d+(?:/\d+)?)", s)
    if pairs:
        return {int(i): Fraction(v) for i, v in pairs}
    vals = read_list(s, "die Anfangswerte")
    if not vals or not all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
        raise InputError("Bitte die Anfangswerte als T(0)=1, T(1)=6 angeben (oder als 1, 6).")
    return {i: v for i, v in enumerate(vals)}


def calc_rekursion(p: dict):
    a = p.get("action", "tree")
    if a == "tree":
        alg = p.get("ralg", "mergeSort")
        if alg == "mergeSort":
            vals = read_list(p.get("rarg", ""), "die Liste")
            if not vals or not all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
                raise InputError("mergeSort braucht eine Liste ganzer Zahlen, etwa 4, 6, 5, 1.")
            return dk.recursion_tree("mergeSort", vals)
        n = get_int(p, "rarg", "n", 1, 7 if alg == "hanoi" else 12)
        return dk.hanoi(n) if alg == "hanoi" else dk.recursion_tree(alg, n)
    if a == "master":
        return dk.master_theorem(get_int(p, "ma", "a", 1, 10 ** 6), get_int(p, "mb", "b", 2, 10 ** 6),
                                 get_int(p, "mk", "k", 0, 100), get_fraction(p, "mc", "c"), levels=6)
    if a == "linrec":
        coeffs = [Fraction(v.replace(MINUS, "-")) if re.fullmatch(r"-?[\d/]+", v.replace(MINUS, "-")) else None
                  for v in split_top(get_text(p, "lc", "die Koeffizienten c₁, c₂"))]
        if None in coeffs:
            raise InputError("Die Koeffizienten sind Zahlen, etwa 6, -9.")
        coeffs = [int(c) if c.denominator == 1 else c for c in coeffs]
        f = (p.get("lF") or "0").strip() or "0"
        return dk.solve_recurrence(coeffs, f, _read_init(p.get("li", "")))
    raise InputError("Unbekannte Rechnung.")


CALC = {"datentypen": calc_datentypen, "logik": calc_logik, "praedikate": calc_praedikate,
        "kalkuel": calc_kalkuel, "algorithmen": calc_algorithmen, "rekursion": calc_rekursion}


def calc(page: str, p: dict):
    """Rechnet eine Seite. Wirft InputError mit freundlichem Satz bei Fehlern."""
    try:
        return CALC[page](p)
    except InputError:
        raise
    except Exception as exc:                  # der Kern wirft ValueError, SymPy und der Parser vielerlei
        raise InputError(friendly_error(exc)) from exc


# ==========================================================================
# Die Zeichnungen
# ==========================================================================

NO_FIGURE = "Zu dieser Rechnung gibt es keine Zeichnung."


def figure_names(page: str, p: dict) -> list[str]:
    """Die Zeichnungen, zwischen denen die Seite umschalten kann."""
    act = p.get("action")
    if page == "logik" and act == "simplify" or page == "kalkuel" and act == "cnf":
        return ["Vorher", "Nachher"]
    if page == "algorithmen" and act == "growth":
        return ["Logarithmisch", "Linear"]
    if page == "rekursion" and act == "tree":
        return ["Aufrufe", "Mit Rückgabe"]
    return ["Zeichnung"]


def _bit_width(n: int) -> int:
    return min(64, max(8, 8 * math.ceil(max(1, n.bit_length()) / 8)))


def _quant_figure(res, dark: bool):
    evals = getattr(res, "evaluations", None)
    if not evals:
        raise NotImplementedError(NO_FIGURE)
    nvars = len(evals[0][0])
    if nvars > 2:
        raise NotImplementedError("Die Zeichnung zeigt höchstens zwei Variablen.")
    if len(evals) > 1500:
        raise NotImplementedError("Das Universum ist für eine Zeichnung zu gross.")
    keys = [e[0] if nvars > 1 else e[0][0] for e in evals]
    name = getattr(res.predicate, "name", "") or "P"
    return dk.plot_quantifier(keys, [bool(e[1]) for e in evals], name=name,
                              variables=tuple(res.vars) if nvars > 1 else None, dark=dark)


def _nvals_of(p: dict, key: str, default: int) -> int:
    try:
        vals = _int_list(p.get(key, ""), "n") or []
    except InputError:
        vals = []
    return max(vals) if vals else default


def growth_args(p: dict, result):
    """(Funktionen, grösstes n, Rechenbudget oder None) für plot_growth und plotly_growth."""
    if p.get("action") == "growth":
        return read_funcs(p["gf"]), max(10, min(_nvals_of(p, "gn", 20), 100)), None
    results = result.results if hasattr(result, "results") else {"f": result}
    budget = float(next(iter(results.values())).budget)
    n_max = max(10, min(2 * min(r.n for r in results.values()) + 2, 200))
    return read_funcs(p["mf"]), n_max, budget


def make_figure(page: str, p: dict, result, dark: bool = False, variant: int = 0):
    """
    Die Matplotlib-Figur zur Rechnung. Wirft NotImplementedError, wenn es zu
    dieser Rechnung keine Zeichnung gibt (zum Beispiel die Ganzzahl-Operation).
    """
    a = p.get("action")
    if page == "datentypen":
        if a == "bool":
            return dk.plot_syntax_tree(result.formula, dark=dark)
        if a in ("base", "frombase"):
            return dk.plot_bits(int(result.number), width=_bit_width(int(result.number)), dark=dark)
        if a == "twos":
            return dk.plot_bits(result.pattern, width=len(result.pattern), signed=True, dark=dark)
        if a == "ieee":
            return dk.plot_bits(re.sub(r"[^01]", "", result.pattern), width=result.bits, dark=dark)
        raise NotImplementedError(NO_FIGURE)
    if page == "logik":
        if a == "tt" or a == "nf":
            return dk.plot_truth_table(result.formula, dark=dark)
        if a == "equiv":
            return dk.plot_truth_table(dk.Iff(result.f, result.g), dark=dark)
        if a == "impl":
            return dk.plot_truth_table(dk.Implies(result.f, result.g), dark=dark)
        if a == "simplify":
            return dk.plot_syntax_tree(result.original if variant == 0 else result.result, dark=dark)
        if a == "tree":
            return dk.plot_syntax_tree(result.root, dark=dark)
    if page == "praedikate":
        if a in ("quant", "list"):
            return _quant_figure(result, dark)
        if a == "negq":
            return _quant_figure(dk.quantify(p["qs"], p["qp"], read_universes(p.get("qu", ""), p["qs"])), dark)
        raise NotImplementedError(NO_FIGURE)
    if page == "kalkuel":
        if a == "entails":
            return dk.plot_entailment(result.wb_parts, result.q, dark=dark)
        if a == "res":
            return dk.plot_resolution(result.graph, dark=dark)
        if a == "cnf":
            return dk.plot_syntax_tree(result.original if variant == 0 else result.formula, dark=dark)
        if a == "derive":
            return dk.plot_entailment(result.wb_parts, result.q, dark=dark)
    if page == "algorithmen":
        if a == "count":
            x = getattr(result, "arguments", {}).get("x")
            if result.func == "decToBin" and isinstance(x, int) and 0 < x < 10**9:
                return dk.plot_halving(x, dark=dark)
            raise NotImplementedError("Für diese Zeichnung 'Worst / Best Case' wählen: sie zeigt "
                                      "die Schritte über n.")
        if a == "case":
            ns = sorted(result.measured)
            fit = None
            if getattr(result, "fit", None) is not None:
                fit = str(result.fit.expr.subs(result.fit.var, sp.Symbol("n"))).replace("**", "^") \
                    if result.fit.var != sp.Symbol("n") else str(result.fit.expr)
            try:
                return dk.plot_steps(ns, [result.measured[n] for n in ns], fit=fit, dark=dark)
            except Exception:
                return dk.plot_steps(ns, [result.measured[n] for n in ns], dark=dark)
        if a in ("growth", "maxn"):
            funcs, n_max, limit = growth_args(p, result)
            return dk.plot_growth(funcs, n_max=n_max, log=(variant == 0 or limit is not None),
                                  mark_limit=limit, dark=dark)   # ponytail: mit Rechenbudget nur logarithmisch, linear bricht plot_growth ab
        if a == "landau":
            if result.C is None and getattr(result, "c", None) is None:
                raise NotImplementedError("Die Aussage gilt nicht — es gibt keine Konstanten zum Einzeichnen.")
            to_n = lambda e: re.sub(r"\bx\b", "n", str(e))
            funcs = {"f": to_n(result.f)}
            if result.c is not None:
                funcs[f"{result.c}·g"] = f"({result.c})*({to_n(result.g)})"
            if result.C is not None:
                funcs[f"{result.C}·g"] = f"({result.C})*({to_n(result.g)})"
            x0 = int(result.x0) if result.x0 is not None else 1
            return dk.plot_growth(funcs, n_max=max(10, min(x0 * 3, 60)), dark=dark)
    if page == "rekursion":
        if a == "tree":
            res = result if p.get("ralg") != "hanoi" else dk.recursion_tree("hanoi", result.n)
            return res.figure(values=(variant == 1), dark=dark)
        if a == "master":
            ns = sorted(result.values)
            return dk.plot_steps(ns, [int(result.values[n]) for n in ns], dark=dark, log=True,
                                 title="T(n) aus der Rekursion für n = bʲ (logarithmische Achse)")
        if a == "linrec":
            i0 = min(int(k) for k in (result.init or {0: 0}))
            ns = list(range(i0, i0 + 13))
            vals = [result.at(n) for n in ns]
            vals = [int(v) if v == int(v) else float(v) for v in vals]
            return dk.plot_steps(ns, vals, dark=dark, title="T(n) für die ersten Werte")
    raise NotImplementedError(NO_FIGURE)


# ==========================================================================
# Wolfram|Alpha
# ==========================================================================

def ssl_context():
    """
    Der Zertifikatsspeicher für die Anfragen.

    Ein mit pyenv gebautes Python benutzt den Speicher der OpenSSL-Version,
    gegen die es gebaut wurde — der ist auf macOS oft leer. Ist das Paket
    certifi vorhanden, nehmen wir dessen Zertifikate; das ist derselbe Satz,
    den auch Browser verwenden, und erspart jede Bastelei am System.
    """
    import ssl
    if os.environ.get("SSL_CERT_FILE"):
        return ssl.create_default_context()      # bewusst gesetzt, respektieren
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


WOLFRAM_URL = "https://api.wolframalpha.com/v2/query"
WOLFRAM_SITE = "https://www.wolframalpha.com/input"
APPID_FILE = Path.home() / ".dmatkit_wolfram_appid"
#: Wer schon eine AppID für anakit oder lialkit hinterlegt hat, braucht sie nicht noch einmal.
APPID_FALLBACKS = (Path.home() / ".anakit_wolfram_appid", Path.home() / ".gauss_wolfram_appid")
USER_AGENT = "dmatkit-app/1.0"
_CACHE: dict[str, str] = {}


def load_appid() -> str:
    key = os.environ.get("WOLFRAM_APPID", "").strip()
    if key:
        return key
    for path in (APPID_FILE, *APPID_FALLBACKS):
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            return value
    return ""


def save_appid(key: str) -> None:
    APPID_FILE.write_text(key.strip(), encoding="utf-8")
    try:
        APPID_FILE.chmod(0o600)
    except OSError:
        pass

def browser_url(query: str) -> str:
    return f"{WOLFRAM_SITE}?i={urllib.parse.quote(query)}"


def explain_network_error(reason) -> str:
    """
    Macht aus dem technischen Grund einer fehlgeschlagenen Anfrage einen
    Satz, der sagt, woran es liegt und was zu tun ist.
    """
    import ssl
    import socket

    text = str(reason)
    if isinstance(reason, ssl.SSLCertVerificationError) or \
            "CERTIFICATE_VERIFY_FAILED" in text:
        try:
            import certifi                                    # noqa: F401
            return ("Das Sicherheitszertifikat liess sich nicht prüfen, obwohl "
                    "certifi vorhanden ist. Vermutlich hängt ein Netzwerkfilter "
                    "dazwischen, der den Verkehr aufbricht — probier ein "
                    "anderes Netz, etwa den Hotspot vom Handy.")
        except ImportError:
            return ("Das Sicherheitszertifikat liess sich nicht prüfen: diesem "
                    "Python fehlen die Wurzelzertifikate. „pip install certifi“ "
                    "ausführen und die App neu starten — sie nimmt die "
                    "Zertifikate dann von dort.")
    if isinstance(reason, socket.gaierror) or "Name or service not known" in text \
            or "nodename nor servname" in text:
        return ("Der Server api.wolframalpha.com war nicht auffindbar. Das "
                "deutet auf eine fehlende Internetverbindung oder einen "
                "blockierenden DNS hin.")
    if "Tunnel connection failed" in text or "proxy" in text.lower():
        return ("Ein Proxy im Netzwerk hat die Anfrage abgewiesen. In einem "
                "Firmen- oder Schulnetz hilft meist ein anderes Netz, etwa der "
                "Hotspot vom Handy.")
    if isinstance(reason, ConnectionRefusedError) or "refused" in text.lower():
        return ("Die Verbindung wurde abgewiesen — vermutlich blockiert eine "
                "Firewall oder ein Netzwerkfilter den Zugriff.")
    if isinstance(reason, (TimeoutError, socket.timeout)) or "timed out" in text:
        return ("Zeitüberschreitung beim Verbinden. Entweder ist die Leitung "
                "sehr langsam oder etwas blockiert die Anfrage.")
    return f"Keine Verbindung zu Wolfram|Alpha: {text}"


def cert_source() -> str:
    """Woher die Zertifikate kommen — gehört in die Diagnose."""
    if os.environ.get("SSL_CERT_FILE"):
        return "Zertifikat geprüft (SSL_CERT_FILE)."
    try:
        import certifi                                        # noqa: F401
        return "Zertifikat geprüft (certifi)."
    except ImportError:
        return "Zertifikat geprüft (Systemspeicher)."


def network_selftest(appid: str = "", timeout: float = 20) -> list[tuple[str, bool, str]]:
    """
    Prüft der Reihe nach: Namensauflösung, TLS-Verbindung, Antwort der API.
    Gibt je Schritt (Name, geklappt, Erläuterung) zurück.
    """
    import socket

    out: list[tuple[str, bool, str]] = []
    host = "api.wolframalpha.com"

    try:
        address = socket.gethostbyname(host)
        out.append(("Namensauflösung", True, f"{host} → {address}"))
    except OSError as exc:
        out.append(("Namensauflösung", False, explain_network_error(exc)))
        return out

    try:
        req = urllib.request.Request(f"https://{host}/",
                                     headers={"User-Agent": USER_AGENT})
        urllib.request.urlopen(req, timeout=timeout,
                               context=ssl_context()).read(64)
        out.append(("Verschlüsselte Verbindung", True, cert_source()))
    except urllib.error.HTTPError:
        # Eine Fehlerseite ist auch eine Antwort — die Leitung steht.
        out.append(("Verschlüsselte Verbindung", True, cert_source()))
    except urllib.error.URLError as exc:
        out.append(("Verschlüsselte Verbindung", False,
                    explain_network_error(exc.reason)))
        return out
    except OSError as exc:
        out.append(("Verschlüsselte Verbindung", False,
                    explain_network_error(exc)))
        return out

    if not appid:
        out.append(("AppID", False, "Noch keine AppID hinterlegt."))
        return out

    try:
        answer = wolfram_text(appid, "2+2", timeout=timeout)
        out.append(("Testabfrage 2+2", True, f"Antwort: {answer.splitlines()[0]}"))
    except RuntimeError as exc:
        out.append(("Testabfrage 2+2", False, str(exc)))
    return out


def wolfram_text(appid: str, query: str, timeout: float = 30) -> str:
    """Fragt die Full Results API und gibt den Klartext des Ergebnis-Pods."""
    if query in _CACHE:
        return _CACHE[query]
    params = urllib.parse.urlencode({
        "appid": appid, "input": query, "output": "json",
        "format": "plaintext", "podtimeout": "20",
    })
    req = urllib.request.Request(f"{WOLFRAM_URL}?{params}",
                                 headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout,
                                    context=ssl_context()) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise RuntimeError(
                "Die AppID wurde abgelehnt. Prüfe sie unter „AppID“ — eine "
                "frische AppID braucht manchmal ein paar Minuten.") from exc
        if exc.code == 429:
            raise RuntimeError("Zu viele Anfragen. Das kostenlose Kontingent "
                               "liegt bei 2000 Abfragen im Monat.") from exc
        raise RuntimeError(f"Wolfram antwortete mit HTTP {exc.code}.") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(explain_network_error(exc.reason)) from exc
    except TimeoutError as exc:
        raise RuntimeError("Wolfram hat nicht rechtzeitig geantwortet. "
                           "Versuch es gleich noch einmal.") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Wolfram hat keine lesbare Antwort geschickt.") from exc
    except OSError as exc:                       # Verbindung abgebrochen u. ä.
        raise RuntimeError(explain_network_error(exc)) from exc

    qr = data.get("queryresult", {})
    if qr.get("error"):
        detail = qr.get("error")
        msg = detail.get("msg") if isinstance(detail, dict) else None
        raise RuntimeError(msg or "Wolfram meldet einen Fehler.")
    if not qr.get("success"):
        raise RuntimeError("Wolfram konnte die Anfrage nicht auswerten.")

    pods = qr.get("pods", [])
    preferred = [p for p in pods if p.get("id") in
                 ("Result", "Solution", "DecimalApproximation", "BaseConversion",
                  "BaseRepresentations", "Value")]
    for pod in preferred + pods:
        if pod.get("id") in ("Input", "InputInterpretation"):
            continue
        for sub in pod.get("subpods", []):
            text = (sub.get("plaintext") or "").strip()
            if text:
                _CACHE[query] = text
                return text
    raise RuntimeError("Wolfram hat kein Ergebnis zurückgegeben.")



# -- Eingaben für Wolfram und Antworten von Wolfram lesen ------------------
#
# Wolfram|Alpha versteht Wolfram-Language-Ausdrücke wie BooleanTable[…]. Für
# die Logik fragen wir immer dieselbe Sache: die Wahrheitstabelle eines
# Ausdrucks. Alles andere (Äquivalenz, Folgerung, Vereinfachung, Normalform)
# lässt sich darauf zurückführen: der Ausdruck ist genau dann überall wahr,
# wenn die Aussage gilt. Variablen heissen v1, v2, … — einzelne Buchstaben wie
# C, D, E, N, I haben bei Wolfram eine feste Bedeutung.

def wl(f, rename: dict) -> str:
    """Eine Formel als Wolfram-Language-Ausdruck."""
    kind = type(f).__name__
    if kind == "Var":
        return rename[f.name]
    if kind == "Const":
        return "True" if f.value else "False"
    args = ", ".join(wl(c, rename) for c in f.children)
    return {"Not": "Not", "And": "And", "Or": "Or", "Xor": "Xor", "Implies": "Implies",
            "Iff": "Equivalent"}[kind] + "[" + args + "]"


def table_query(expr, names: Sequence[str]) -> str:
    """BooleanTable-Anfrage für eine Formel über den Variablen `names`."""
    rename = {n: f"v{i + 1}" for i, n in enumerate(names)}
    return f"BooleanTable[{wl(expr, rename)}, {{{', '.join(rename.values())}}}]"


def our_vector(expr, names: Sequence[str]) -> list[bool]:
    """Unsere Wahrheitswerte in der Reihenfolge 0 0 / 0 1 / 1 0 / 1 1."""
    return [bool(expr.eval(env)) for env in dk.assignments(list(names))]


def parse_wolfram_bools(text: str) -> list[bool]:
    """Liest {True, False, …} aus Wolframs Antwort; ValueError, wenn nichts Lesbares dasteht."""
    words = re.findall(r"\b(true|false|t|f)\b", (text or "").lower())
    if not words:
        raise ValueError("nicht auswertbar: keine Wahrheitswerte gefunden")
    return [w in ("true", "t") for w in words]


def parse_wolfram_int(text: str) -> int:
    """Eine ganze Zahl aus Wolframs Antwort (»6 227 020 800«, »−4«, »6227020800«)."""
    line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    line = line.replace("−", "-").replace("’", "").replace("'", "")
    if "×10^" in line or "x10^" in line:
        raise ValueError("nicht auswertbar: Näherung statt ganzer Zahl")
    m = re.search(r"-?\d[\d ,  ]*", line)
    if not m:
        raise ValueError("nicht auswertbar: keine Zahl gefunden")
    digits = re.sub(r"[ ,  ]", "", m.group(0))
    return int(digits)


def parse_wolfram_number(text: str) -> float:
    """Eine Zahl, auch als Näherung wie »6.227×10^9« oder »≈ 1.9 × 10^9«."""
    line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    line = line.replace("−", "-").replace("’", "").replace("≈", " ")
    m = re.search(r"(-?\d[\d ,  ]*(?:\.\d+)?)\s*(?:[×x*·]\s*10\s*\^\s*\(?(-?\d+)\)?)?", line)
    if not m:
        raise ValueError("nicht auswertbar: keine Zahl gefunden")
    mant = float(re.sub(r"[ ,  ]", "", m.group(1)))
    return mant * 10 ** int(m.group(2)) if m.group(2) else mant


def parse_wolfram_base(text: str) -> str:
    """Ziffernfolge aus Wolframs Antwort wie »11001_2«, »ff₁₆« oder »0.101_2 (binary)«."""
    line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    line = re.split("[\u2080-\u2089]", line)[0].replace("−", "-")      # tiefgestellte Basis abschneiden
    m = re.match(r"\s*(-?[0-9A-Za-z]+(?:\.[0-9A-Za-z]+)?)\s*(?:_\s*\{?\d+\}?|\(|$|\s)", line)
    if not m:
        raise ValueError("nicht auswertbar: keine Ziffernfolge gefunden")
    return m.group(1).upper()


def parse_wolfram_digits(text: str) -> str:
    """Die Dezimalziffern einer Näherung (»2.1036536536…«), ohne Auslassungspunkte."""
    line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    line = line.replace("≈", " ").replace("…", " ").replace("...", " ").replace(MINUS, "-")
    m = re.search(r"-?\d+(?:\.\d+)?", line)
    if not m:
        raise ValueError("nicht auswertbar: keine Zahl gefunden")
    return m.group(0)


def expand_periodic(text: str, base_digits: int = 40) -> str:
    """»2.10(365)« -> »2.10365365365…« (zum Vergleich mit Wolframs Näherung)."""
    m = re.fullmatch(r"(-?)(\d*)\.?(\d*)(?:\((\d+)\))?", text.replace(" ", ""))
    if not m:
        raise ValueError(text)
    sign, whole, pre, period = m.groups()
    frac = pre + ((period or "") * (base_digits // max(1, len(period or "1")) + 1))
    frac = frac[:base_digits]
    return f"{sign}{whole or '0'}" + ("." + frac if frac else "")


def prefix_equal(a: str, b: str, slack: int = 2) -> bool:
    """Stimmen zwei Ziffernfolgen in allen bis auf die letzten `slack` Stellen der kürzeren überein?"""
    n = min(len(a), len(b)) - slack
    return n >= 3 and a[:n] == b[:n]


# -- Vergleiche ---------------------------------------------------------------

@dataclass
class Check:
    """Ein einzelner Prüfpunkt im Bericht."""
    name: str
    status: str          # 'ok' | 'bad' | 'info' | 'error'
    ours: str = ""
    theirs: str = ""
    note: str = ""


class WolframReport:
    def __init__(self):
        self.checks: list[Check] = []
        self.texts: list[tuple[str, str]] = []     # (Frage, Wolframs Antwort)
        self.url: str = ""

    @property
    def verdict(self) -> str:
        if any(c.status == "bad" for c in self.checks):
            return "bad"
        if any(c.status == "error" for c in self.checks):
            return "warn"
        if any(c.status == "ok" for c in self.checks):
            return "ok"
        return "warn"

    @property
    def headline(self) -> str:
        return {
            "ok": "Wolfram|Alpha bestätigt das Ergebnis.",
            "bad": "Wolfram|Alpha kommt auf etwas anderes.",
            "warn": "Wolfram konnte nicht alles prüfen.",
        }[self.verdict]


def _ask(appid: str, queries: dict[str, str], ask=None) -> dict[str, object]:
    """Stellt die Anfragen nebeneinander und sammelt Antworten wie Fehler."""
    ask = ask or wolfram_text
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {key: pool.submit(ask, appid, q) for key, q in queries.items()}
        answers: dict[str, object] = {}
        for key, fut in futures.items():
            try:
                answers[key] = fut.result()
            except (RuntimeError, ValueError) as exc:
                answers[key] = exc
    # Scheitert alles, ist meist AppID oder Netz schuld — ein Bericht mit
    # lauter gleichen Fehlern hilft nicht.
    if answers and all(isinstance(v, Exception) for v in answers.values()):
        raise RuntimeError(str(next(iter(answers.values()))))
    return answers


def _first_line(answer) -> str:
    return str(answer).strip().splitlines()[0][:60] if str(answer).strip() else ""


def _note(report: WolframReport, label: str, answer) -> None:
    if isinstance(answer, str):
        report.texts.append((label, answer.replace("\n", "  ·  ")))


def _unreadable(report: WolframReport, name: str, ours: str, answer, exc) -> None:
    report.checks.append(Check(name, "info", ours=ours, note=(
        f"Wolframs Antwort „{_first_line(answer)}“ ist nicht auswertbar "
        f"({str(exc).replace('nicht auswertbar: ', '')}).")))


def check_table(report: WolframReport, name: str, expr, names: Sequence[str], answer,
                label: str = "", query: str = "", want_all_true: bool | None = None) -> None:
    """
    Vergleicht die Wahrheitstabelle von `expr` mit Wolframs BooleanTable.
    BooleanTable zählt mit »wahr« zuerst, wir mit »falsch« zuerst: der
    Vergleich dreht deshalb Wolframs Liste um.
    """
    ours = our_vector(expr, names)
    shown = "".join("1" if v else "0" for v in ours)
    if isinstance(answer, Exception):
        report.checks.append(Check(name, "error", ours=shown, note=str(answer)))
        return
    _note(report, label or name, answer)
    try:
        theirs = parse_wolfram_bools(answer)
    except ValueError as exc:
        _unreadable(report, name, shown, answer, exc)
        return
    theirs = theirs[::-1]
    if len(theirs) != len(ours):
        report.checks.append(Check(name, "info", ours=shown, note=(
            f"Wolfram lieferte {len(theirs)} statt {len(ours)} Werte — nicht vergleichbar.")))
        return
    same = theirs == ours
    report.checks.append(Check(name, "ok" if same else "bad", ours=shown,
                               theirs="" if same else "".join("1" if v else "0" for v in theirs),
                               note="Gleiche Wahrheitstabelle." if same else ""))


def check_int(report: WolframReport, name: str, ours: int, answer, label: str = "") -> None:
    if isinstance(answer, Exception):
        report.checks.append(Check(name, "error", ours=str(ours), note=str(answer)))
        return
    _note(report, label or name, answer)
    try:
        theirs = parse_wolfram_int(answer)
    except ValueError as exc:
        _unreadable(report, name, str(ours), answer, exc)
        return
    report.checks.append(Check(name, "ok" if theirs == ours else "bad", ours=str(ours),
                               theirs="" if theirs == ours else str(theirs)))


def check_digits(report: WolframReport, name: str, ours_text: str, theirs_text: str, answer,
                 label: str = "") -> None:
    """Wolframs Zifferndarstellung gegen unsere (nur die ersten Stellen, die Näherung rundet)."""
    if isinstance(answer, Exception):
        report.checks.append(Check(name, "error", ours=ours_text, note=str(answer)))
        return
    _note(report, label or name, answer)
    same = prefix_equal(ours_text.lower(), theirs_text.lower())
    report.checks.append(Check(name, "ok" if same else "bad", ours=ours_text,
                               theirs="" if same else theirs_text,
                               note="Die ersten Stellen stimmen überein." if same else ""))


def wolfram_term(expr, n: int) -> str:
    """f(n) als Anfrage: n wird in den Term eingesetzt, gerechnet wird bei Wolfram."""
    return re.sub(r"\bn\b", f"({n})", sp.sstr(expr)).replace("**", "^")


def check_maxn(report: WolframReport, name: str, res, ans_at, ans_next) -> None:
    """f(n) ≤ B < f(n+1) mit Wolframs Werten für f(n) und f(n+1)."""
    shown = f"n = {res.n}"
    for a in (ans_at, ans_next):
        if isinstance(a, Exception):
            report.checks.append(Check(name, "error", ours=shown, note=str(a)))
            return
    try:
        at, nxt = parse_wolfram_number(ans_at), parse_wolfram_number(ans_next)
    except ValueError as exc:
        _unreadable(report, name, shown, ans_at, exc)
        return
    budget = float(res.budget)
    ok = at <= budget * (1 + 1e-9) and nxt > budget * (1 - 1e-9)
    report.checks.append(Check(
        name, "ok" if ok else "bad", ours=shown,
        theirs="" if ok else f"f(n) = {at:.6g}, f(n+1) = {nxt:.6g}, B = {budget:.6g}",
        note=f"Wolfram: f({res.n}) = {at:.6g} ≤ B = {budget:.6g} < f({res.n + 1}) = {nxt:.6g}." if ok else ""))


UNSUPPORTED = ("Für diese Rechnung gibt es keine Prüfung mit Wolfram|Alpha — sie lässt sich dort "
               "nicht robust vergleichen.")


def wolfram_support(page: str, p: dict, result=None) -> str:
    """Leer, wenn sich die Rechnung prüfen lässt; sonst der Grund."""
    a = p.get("action")
    ok = {
        "datentypen": {"bool", "intop", "base", "frombase", "repeat", "fracbin"},
        "logik": {"tt", "equiv", "impl", "simplify", "nf"},
        "kalkuel": {"entails", "res", "cnf", "derive"},
        "algorithmen": {"maxn"},
    }
    if a not in ok.get(page, set()):
        return UNSUPPORTED
    if page == "datentypen" and result is not None:
        if a == "bool" and result.formula.variables():
            return "Mit Variablen lässt sich der Ausdruck bei Wolfram nicht prüfen."
        if a == "repeat" and getattr(result, "base", 10) != 10:
            return "Periodische Zahlen prüft Wolfram|Alpha nur im Dezimalsystem."
        if a == "base" and not 2 <= int(getattr(result, "base", 2)) <= 36:
            return UNSUPPORTED
    return ""


def wolfram_subtitle(page: str) -> str:
    return {
        "datentypen": "Vergleicht Umrechnungen, Brüche und Ganzzahl-Operationen mit Wolfram|Alpha.",
        "logik": "Vergleicht die Wahrheitstabelle mit BooleanTable bei Wolfram|Alpha.",
        "praedikate": "Für Quantoren gibt es keine Prüfung bei Wolfram|Alpha.",
        "kalkuel": "Prüft über die Wahrheitstabelle, ob die Folgerung beziehungsweise Umformung gilt.",
        "algorithmen": "Prüft das grösste n: f(n) ≤ Budget < f(n + 1) bei Wolfram|Alpha.",
        "rekursion": "Für Rekursion gibt es keine Prüfung bei Wolfram|Alpha. Die Rechnung prüft sich selbst (Probe im Rechenweg).",
    }[page]


def _vars_of(*formulas) -> list[str]:
    names: set[str] = set()
    for f in formulas:
        names |= set(f.variables())
    return sorted(names)


def _table_check_for(report, appid, label, expr, names, ask, title, want=None):
    q = table_query(expr, names)
    report.url = report.url or browser_url(q)
    ans = _ask(appid, {"t": q}, ask)["t"]
    check_table(report, title, expr, names, ans, label=label, query=q)


def run_wolfram_checks(appid: str, page: str, p: dict, result, ask=None) -> WolframReport:
    """
    Prüft eine Rechnung mit Wolfram|Alpha. `ask(appid, frage) -> Text` lässt
    sich im Test durch eine Attrappe ersetzen. Wirft RuntimeError, wenn
    Netz oder AppID versagen, und ValueError, wenn es nichts zu prüfen gibt.
    """
    why = wolfram_support(page, p, result)
    if why:
        raise ValueError(why)
    a = p.get("action")
    report = WolframReport()
    if page == "datentypen":
        if a == "bool":
            q = wl(result.formula, {})
            report.url = browser_url(q)
            ans = _ask(appid, {"b": q}, ask)["b"]
            if isinstance(ans, Exception):
                report.checks.append(Check("Wahrheitswert", "error", ours=fmt(result.value), note=str(ans)))
            else:
                _note(report, "Wahrheitswert", ans)
                try:
                    theirs = parse_wolfram_bools(ans)[0]
                    report.checks.append(Check("Wahrheitswert", "ok" if theirs == bool(result.value) else "bad",
                                               ours=fmt(result.value),
                                               theirs="" if theirs == bool(result.value) else fmt(theirs)))
                except ValueError as exc:
                    _unreadable(report, "Wahrheitswert", fmt(result.value), ans, exc)
        elif a == "intop":
            x, y, op = result.a, result.b, result.op
            qs = {"quotient": f"floor({x}/{y})", "remainder": f"mod({x}, {y})"} if op in ("//", "%", "divmod") \
                else {"value": f"({x}) {op} ({y})"}
            if op == "//":
                qs = {"quotient": qs["quotient"]}
            elif op == "%":
                qs = {"remainder": qs["remainder"]}
            report.url = browser_url(next(iter(qs.values())))
            answers = _ask(appid, qs, ask)
            names = {"quotient": ("Quotient", getattr(result, "quotient", result.value)),
                     "remainder": ("Rest", getattr(result, "remainder", result.value)),
                     "value": ("Ergebnis", result.value)}
            for key, ans in answers.items():
                label, ours = names[key]
                check_int(report, label, int(ours), ans, label=label)
        elif a == "base":
            b = result.base
            q = f"{result.number} in base {b}"
            report.url = browser_url(q)
            ans = _ask(appid, {"b": q}, ask)["b"]
            if isinstance(ans, Exception):
                report.checks.append(Check("Ziffernfolge", "error", ours=result.text, note=str(ans)))
            else:
                _note(report, "Ziffernfolge", ans)
                try:
                    theirs = parse_wolfram_base(ans).lstrip("0") or "0"
                    ours = result.text.upper().lstrip("0") or "0"
                    report.checks.append(Check("Ziffernfolge", "ok" if theirs == ours else "bad",
                                               ours=result.text, theirs="" if theirs == ours else theirs))
                except ValueError as exc:
                    _unreadable(report, "Ziffernfolge", result.text, ans, exc)
        elif a == "frombase":
            q = f"{result.text}_{result.base}"
            report.url = browser_url(q)
            ans = _ask(appid, {"b": q}, ask)["b"]
            check_int(report, "Dezimalzahl", int(result.number), ans, label="Dezimalzahl")
        elif a == "repeat":
            q = f"decimal expansion of {result.numerator}/{result.denominator}"
            report.url = browser_url(q)
            ans = _ask(appid, {"b": q}, ask)["b"]
            if isinstance(ans, Exception):
                report.checks.append(Check("Dezimalbruch", "error", ours=fmt(result.value), note=str(ans)))
            else:
                _note(report, "Dezimalentwicklung", ans)
                try:
                    check_digits(report, f"Bruch {result.numerator}/{result.denominator}",
                                 expand_periodic(result.text), parse_wolfram_digits(ans), ans)
                except ValueError as exc:
                    _unreadable(report, "Dezimalbruch", fmt(result.value), ans, exc)
        elif a == "fracbin":
            q = f"{result.source} in base 2"
            report.url = browser_url(q)
            ans = _ask(appid, {"b": q}, ask)["b"]
            if isinstance(ans, Exception):
                report.checks.append(Check("Binärdarstellung", "error", ours=result.text, note=str(ans)))
            else:
                _note(report, "Binärdarstellung", ans)
                try:
                    theirs = parse_wolfram_base(ans)
                    check_digits(report, "Binärdarstellung", expand_periodic(result.text), theirs, ans)
                except ValueError as exc:
                    _unreadable(report, "Binärdarstellung", result.text, ans, exc)
        return report
    if page == "logik":
        if a in ("tt", "nf"):
            f = result.formula
            names = list(result.variables)
            expr = f if a == "tt" else result.normal_form
            _table_check_for(report, appid, "Wahrheitstabelle", expr, names, ask,
                             "Wahrheitstabelle" if a == "tt" else f"{result.kind} hat dieselbe Tabelle wie die Formel")
            if a == "nf" and our_vector(f, names) != our_vector(expr, names):
                report.checks.append(Check("Normalform", "bad", note="Normalform und Formel unterscheiden sich."))
        elif a == "equiv":
            names = list(result.variables)
            _table_check_for(report, appid, "f ↔ g", dk.Iff(result.f, result.g), names, ask,
                             "f ↔ g ist überall wahr" if result.equivalent else "f ↔ g ist nicht überall wahr")
        elif a == "impl":
            names = list(result.variables)
            _table_check_for(report, appid, "f → g", dk.Implies(result.f, result.g), names, ask,
                             "f → g in allen Zeilen" if result.holds else "f → g nicht in allen Zeilen")
        elif a == "simplify":
            names = _vars_of(result.original)
            _table_check_for(report, appid, "Ergebnis", result.result, names, ask,
                             "Vereinfachte Formel hat dieselbe Tabelle")
            if our_vector(result.result, names) != our_vector(result.original, names):
                report.checks.append(Check("Vereinfachung", "bad", note="Ergebnis und Ausgangsformel unterscheiden sich."))
        return report
    if page == "kalkuel":
        if a == "cnf":
            names = _vars_of(result.original)
            _table_check_for(report, appid, "KNF", result.formula, names, ask,
                             "KNF hat dieselbe Tabelle wie die Formel")
            if our_vector(result.formula, names) != our_vector(result.original, names):
                report.checks.append(Check("KNF", "bad", note="KNF und Ausgangsformel unterscheiden sich."))
        else:
            wb = dk.big_and(result.wb_parts)
            q = result.q
            names = _vars_of(wb, q)
            _table_check_for(report, appid, "WB → Q", dk.Implies(wb, q), names, ask,
                             "WB ⊨ Q gilt (WB → Q überall wahr)" if bool(result.value)
                             else "WB ⊨ Q gilt nicht (WB → Q nicht überall wahr)")
        return report
    if page == "algorithmen":
        results = result.results if hasattr(result, "results") else {"f": result}
        queries, meta = {}, {}
        for name, r in results.items():
            for tag, nn in (("at", r.n), ("next", r.n + 1)):
                queries[f"{name}:{tag}"] = wolfram_term(r.expr, nn)
            meta[name] = r
        report.url = browser_url(next(iter(queries.values())))
        answers = _ask(appid, queries, ask)
        for name, r in meta.items():
            check_maxn(report, f"Grösstes n für {name}" if len(meta) > 1 else "Grösstes n",
                       r, answers[f"{name}:at"], answers[f"{name}:next"])
            if isinstance(answers[f"{name}:at"], str):
                _note(report, f"f({r.n})", answers[f"{name}:at"])
        return report
    raise ValueError(UNSUPPORTED)
# ==========================================================================
# Farben und Schriften
# ==========================================================================

# Swiss Style (wie lialkit und anakit): Schwarz/Weiss, ein Rot, keine Rundungen.
LIGHT = dict(
    bg="#ffffff", nav="#ffffff", card="#ffffff", text="#000000", sec="#333333",
    ter="#666666", line="#000000", hair="#cccccc", fill="#f4f4f4", fill2="#f4f4f4",
    seg="#ffffff", blue="#000000", blue_hi="#e60000", blue_soft="#f4f4f4",
    green="#18794e", green_soft="#e6f4ec", red="#e60000", red_soft="#fdecec",
    amber="#ab5200", amber_soft="#fff4e0", flash="#fff3c4",
    entry="#ffffff", hi_soft="#e3edfb", scroll="#bdbdbd", shadow=0, dark=False)

DARK = dict(
    bg="#000000", nav="#000000", card="#000000", text="#ffffff", sec="#cccccc",
    ter="#888888", line="#ffffff", hair="#333333", fill="#1a1a1a", fill2="#111111",
    seg="#333333", blue="#e60000", blue_hi="#ff3333", blue_soft="#2a0000",
    green="#46a758", green_soft="#102a16", red="#ff3333", red_soft="#2a0000",
    amber="#ffb224", amber_soft="#3a2a0a", flash="#3d3210",
    entry="#000000", hi_soft="#0e2540", scroll="#555555", shadow=0, dark=True)

T: dict = dict(LIGHT)


def pick_font(candidates: Sequence[str], fallback: str) -> str:
    families = {f.lower(): f for f in QFontDatabase.families()}
    for name in candidates:
        if name.lower() in families:
            return families[name.lower()]
    return fallback


class Fonts:
    def __init__(self):
        ui = pick_font(["Geist", "Inter", "SF Pro Text", ".AppleSystemUIFont", "Helvetica Neue",
                        "Segoe UI Variable Text", "Segoe UI", "Inter",
                        "DejaVu Sans"], "Sans Serif")
        display = pick_font(["Geist", "Inter", "SF Pro Display", "SF Pro Text", "Helvetica Neue",
                             "Segoe UI Variable Display", "Segoe UI", "Inter",
                             "DejaVu Sans"], ui)
        mono = pick_font(["SF Mono", "Menlo", "Cascadia Mono", "Consolas",
                          "DejaVu Sans Mono"], "Monospace")
        self.family, self.display, self.mono_family = ui, display, mono
        # Logik-Zeichen (∧ ∨ ¬ ⊕ ⊨ ⊢ ∀ ∃) fehlen vielen Schriften: in dieser Reihenfolge
        # wird pro Zeichen weitergesucht, bevor Qt selbst irgendeine Schrift nimmt.
        have = {f.lower() for f in QFontDatabase.families()}
        self.fallback = [n for n in ("Segoe UI Symbol", "Apple Symbols", "Cambria Math",
                                     "STIX Two Math", "Noto Sans Math", "Noto Sans Symbols 2",
                                     "DejaVu Sans", "Arial Unicode MS") if n.lower() in have]

    def f(self, size: int, weight: int = QFont.Normal,
          display: bool = False, mono: bool = False) -> QFont:
        fam = self.mono_family if mono else (self.display if display else self.family)
        font = QFont(fam, size)
        font.setFamilies([fam] + [n for n in self.fallback if n != fam])
        font.setWeight(weight)
        if display:
            font.setLetterSpacing(QFont.PercentageSpacing, 97)
        return font


FONTS: Fonts | None = None


def qcolor(key: str) -> QColor:
    return QColor(T[key])


# ==========================================================================
# Gesetzte Formeln: LaTeX -> Matplotlib-Mathtext -> Bild
# ==========================================================================
#
# Die Rechenwege von dmatkit tragen ihre Terme als LaTeX. Matplotlib kann einen
# grossen Teil davon selbst setzen (Brüche, Wurzeln, Potenzen, Grenzwerte),
# ohne dass ein LaTeX auf dem Rechner stehen muss. Aus dem Ergebnis wird eine
# Alpha-Maske; die App färbt sie je nach Thema ein. Was Mathtext nicht kann
# (Fallunterscheidungen setzt die App selbst zeilenweise), fällt auf den
# Unicode-Text von render.tex_to_text zurück.

MATH_SCALE = 3                  # Bildpunkte je logischem Punkt: scharf auch auf Retina
_MATH_LOCK = threading.Lock()
_PARSER = None


@dataclass
class Mask:
    """Eine gesetzte Formel als Alpha-Maske; `depth` = Tiefe unter der Grundlinie."""
    alpha: "np.ndarray"
    depth: float

    @property
    def size(self) -> tuple[float, float]:
        """Logische Breite und Höhe."""
        h, w = self.alpha.shape
        return w / MATH_SCALE, h / MATH_SCALE


_MASKS: "OrderedDict[tuple[str, int], Mask | None]" = OrderedDict()
_MASK_LIMIT = 6000


def _escape_text(inner: str) -> str:
    inner = (inner.replace("\\", "").replace("_", r"\_").replace("%", r"\%")
             .replace("&", r"\&").replace("#", r"\#").replace("$", "")
             .replace("{", r"\{").replace("}", r"\}").replace("^", r"\^{}"))
    return inner


_SUB = {"\u2080": "0", "\u2081": "1", "\u2082": "2", "\u2083": "3", "\u2084": "4",
        "\u2085": "5", "\u2086": "6", "\u2087": "7", "\u2088": "8", "\u2089": "9",
        "\u208a": "+", "\u208b": "-", "\u2093": "x", "\u2095": "h", "\u2099": "n"}
_SUP = {"\u2070": "0", "\u00b9": "1", "\u00b2": "2", "\u00b3": "3", "\u2074": "4",
        "\u2075": "5", "\u2076": "6", "\u2077": "7", "\u2078": "8", "\u2079": "9",
        "\u207a": "+", "\u207b": "-", "\u207f": "n"}
_SUBRUN = re.compile("[" + "".join(_SUB) + "]+")
_SUPRUN = re.compile("[" + "".join(_SUP) + "]+")
#: Zeichen der Logik und Mengenlehre -> Mathtext-Befehl (nur ausserhalb von \text{…})
_SYMBOLS = {"\u2227": r"\wedge ", "\u2228": r"\vee ", "\u00ac": r"\neg ", "\u2295": r"\oplus ",
            "\u2194": r"\leftrightarrow ", "\u21d2": r"\Rightarrow ", "\u21d4": r"\Leftrightarrow ",
            "\u22a8": r"\models ", "\u22a2": r"\vdash ", "\u22ad": r"\nvDash ", "\u2200": r"\forall ",
            "\u2203": r"\exists ", "\u2208": r"\in ", "\u2209": r"\notin ", "\u2261": r"\equiv ",
            "\u2260": r"\neq ", "\u22a4": r"\top ", "\u22a5": r"\bot ", "\u2205": r"\emptyset ",
            "\u222a": r"\cup ", "\u2229": r"\cap ", "\u2216": r"\setminus ", "\u2286": r"\subseteq ",
            "\u0398": r"\Theta ", "\u03a9": r"\Omega ", "\u230a": r"\lfloor ", "\u230b": r"\rfloor ",
            "\u2308": r"\lceil ", "\u2309": r"\rceil ", "\u00d7": r"\times ", "\u2248": r"\approx ",
            "\u25a1": "\\text{\u25a1}", "\u2218": r"\circ ", "\u03b1": r"\alpha ", "\u03b2": r"\beta "}


def _text_piece(inner: str) -> str:
    """Text im Mathtext-Satz. Die Schrift dort hat keine tiefgestellten Ziffern (₂): sie werden zu echten Indizes."""
    inner = inner.replace("\n", " ")
    out, pos = [], 0
    for m in re.finditer("[" + "".join(_SUB) + "]+|[" + "".join(_SUP) + "]+", inner):
        if m.start() > pos:
            out.append(r"\text{" + _escape_text(inner[pos:m.start()]) + "}")
        elif not out:
            out.append("{}")
        run = m.group()
        out.append("_{" + "".join(_SUB[c] for c in run) + "}" if run[0] in _SUB
                   else "^{" + "".join(_SUP[c] for c in run) + "}")
        pos = m.end()
    if pos < len(inner) or not out:
        out.append(r"\text{" + _escape_text(inner[pos:]) + "}")
    return "".join(out)


def _unicode_math(s: str) -> str:
    s = _SUBRUN.sub(lambda m: "_{" + "".join(_SUB[c] for c in m.group()) + "}", s)
    s = _SUPRUN.sub(lambda m: "^{" + "".join(_SUP[c] for c in m.group()) + "}", s)
    s = (s.replace("\u2212", "-").replace("\u221e", r"\infty").replace("\u03c0", r"\pi")
          .replace("\u00b1", r"\pm").replace("\u2264", r"\leq").replace("\u2265", r"\geq")
          .replace("\u2192", r"\to ").replace("\u22c5", r"\cdot").replace("\u00b7", r"\cdot"))
    return "".join(_SYMBOLS.get(ch, ch) for ch in s)


_MACROS = {r"\operatorname": r"\mathrm", r"\textbf": r"\mathbf", r"\tfrac": r"\frac",
           r"\land": r"\wedge", r"\lor": r"\vee", r"\lnot": r"\neg", r"\implies": r"\Rightarrow",
           r"\iff": r"\Leftrightarrow", r"\Box": "\\text{\u25a1}", r"\square": "\\text{\u25a1}",
           r"\lvert": "|", r"\rvert": "|", r"\bigl": "", r"\bigr": "", r"\left.": "", r"\right.": ""}


def mathtext_ready(tex: str) -> str:
    """Passt LaTeX an das an, was Matplotlibs Mathtext versteht."""
    s = " ".join(tex.split("\n")).strip()
    for old, new in _MACROS.items():
        s = re.sub(re.escape(old) + r"(?![A-Za-z])", lambda _m, n=new: n, s)
    if not re.search(r"\\(?:d?frac|sqrt|lim|sum|int|binom|begin)", s):
        # Ohne hohe Inhalte wirken die mitwachsenden Klammern nur winzig.
        for old in (r"\left(", r"\right)", r"\left[", r"\right]"):
            s = s.replace(old, old[-1])
    out, pos = [], 0
    while True:                                 # Text bleibt Text: Zeichen dort nicht in Befehle verwandeln
        start = s.find(r"\text{", pos)
        if start < 0:
            break
        depth, i = 1, start + 6
        while i < len(s) and depth:
            depth += {"{": 1, "}": -1}.get(s[i], 0)
            i += 1
        out.append(_unicode_math(s[pos:start]))
        out.append(_text_piece(s[start + 6:i - 1] if depth == 0 else s[start + 6:]))
        pos = i
    out.append(_unicode_math(s[pos:]))
    return "".join(out)


def render_mask(tex: str, px: int) -> "Mask | None":
    """
    Setzt `tex` in der Schriftgrösse px (logische Bildpunkte). Gibt None, wenn
    Mathtext es nicht kann — der Aufrufer nimmt dann Text. Ergebnisse werden
    gemerkt; der Aufruf ist von mehreren Fäden aus sicher.
    """
    global _PARSER
    key = (tex, px)
    if key in _MASKS:
        return _MASKS[key]
    mask = None
    with _MATH_LOCK:
        if key in _MASKS:
            return _MASKS[key]
        try:
            from matplotlib.font_manager import FontProperties
            from matplotlib.mathtext import MathTextParser
            if _PARSER is None:
                _PARSER = MathTextParser("agg")
            prop = FontProperties(size=px, math_fontfamily="cm")
            raster = _PARSER.parse("$" + mathtext_ready(tex) + "$",
                                   dpi=72 * MATH_SCALE, prop=prop)
            alpha = np.array(raster.image, dtype=np.uint8)
            if alpha.ndim == 2 and alpha.size:
                mask = Mask(alpha, float(raster.depth))
        except Exception:
            mask = None
        _MASKS[key] = mask
        if len(_MASKS) > _MASK_LIMIT:
            _MASKS.popitem(last=False)
    return mask


def mask_image(mask: Mask, color: str, inline_px: int | None = None) -> QImage:
    """
    Färbt eine Maske ein. Mit inline_px wird sie oben oder unten aufgefüllt, bis
    die Mathe-Achse (etwa ¼ Schrifthöhe über der Grundlinie) in der Bildmitte
    liegt — so sitzt sie in einem Satz mit `vertical-align: middle` richtig.
    """
    alpha = mask.alpha
    if inline_px is not None:
        h = alpha.shape[0]
        axis = 0.27 * inline_px * MATH_SCALE
        ascent = h - mask.depth
        above, below = ascent - axis, mask.depth + axis
        half = max(above, below)
        top = max(0, int(round(half - above)))
        bottom = max(0, int(round(half - below)))
        alpha = np.pad(alpha, ((top, bottom), (0, 0)))
    h, w = alpha.shape
    col = QColor(color)
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = col.red(), col.green(), col.blue()
    rgba[..., 3] = alpha
    image = QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888).copy()
    image.setDevicePixelRatio(MATH_SCALE)
    return image


def screen_dpr() -> float:
    screen = QGuiApplication.primaryScreen()
    return float(screen.devicePixelRatio()) if screen is not None else 1.0


def scaled_image(mask: Mask, color: str, scale: float = 1.0, dpr: float | None = None,
                 inline_px: int | None = None) -> "tuple[QImage, float, float]":
    """
    Die Maske in Bildschirmauflösung: aus dem dreifach gesetzten Bild glatt
    heruntergerechnet, damit dünne Striche (Klammern, Bruchstriche) nicht
    wegbrechen. Gibt (Bild, logische Breite, logische Höhe) zurück.
    """
    dpr = dpr or screen_dpr()
    image = mask_image(mask, color, inline_px)
    lw = image.width() / MATH_SCALE * scale
    lh = image.height() / MATH_SCALE * scale
    tw, th = max(1, round(lw * dpr)), max(1, round(lh * dpr))
    out = image.scaled(tw, th, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
    out.setDevicePixelRatio(dpr)
    return out, lw, lh


_B64: "OrderedDict[tuple, str]" = OrderedDict()


def mask_data_uri(tex: str, px: int, color: str) -> "tuple[str, float, float] | None":
    """(data-URI, Breite, Höhe) für ein Bild im Fliesstext, oder None."""
    dpr = screen_dpr()
    key = (tex, px, color, dpr)
    mask = render_mask(tex, px)
    if mask is None:
        return None
    if key not in _B64:
        image, lw, lh = scaled_image(mask, color, 1.0, dpr, inline_px=px)
        data = QByteArray()
        buf = QBuffer(data)
        buf.open(QIODevice.WriteOnly)
        image.save(buf, "PNG")
        buf.close()
        _B64[key] = ("data:image/png;base64," + base64.b64encode(bytes(data)).decode(), lw, lh)
        if len(_B64) > 4000:
            _B64.popitem(last=False)
    return _B64[key]


# -- Das Ansichtsmodell: was die Karten zeigen, ganz ohne Fenster ------------

_WORD = re.compile(r"[A-Za-zÄÖÜäöüéèàç]{3,}")


def label_tex(label: str) -> str:
    """Eine Beschriftung wie »f'(x_0)« oder »Newton« als Mathtext-Stück."""
    if not label:
        return ""
    if "\\" in label:
        return label.replace(r"\!", "").strip()
    out = []
    for token in label.split(" "):
        if _WORD.fullmatch(token) and not re.search(r"[_^]", token):
            out.append(r"\text{" + token + "}")
        elif token:
            out.append(token)
    return r"\ ".join(out)


def _tex_value(obj, decimal: bool, digits: int) -> str:
    try:
        return tex_any(obj, decimal, digits)
    except Exception:
        return r"\text{" + str(obj) + "}"


_CASES = re.compile(r"\\begin\{cases\}(.*)\\end\{cases\}", re.S)


def split_cases(value_tex: str) -> "list[str] | None":
    """Zerlegt \\begin{cases} … \\end{cases} in einzelne Zeilen »Term für Bedingung«."""
    m = _CASES.search(value_tex)
    if not m:
        return None
    rows = []
    for row in re.split(r"\\\\", m.group(1)):
        if not row.strip():
            continue
        left, _, cond = row.partition("&")
        cond = cond.strip()
        if cond.startswith(r"\text{otherwise}"):
            cond = r"\text{sonst}"
        elif cond.startswith(r"\text{for}"):
            cond = cond[len(r"\text{for}"):].strip()
        if cond.startswith(r"\text{sonst}") or cond == "sonst":
            tail = r"\text{sonst}"
        elif cond:
            tail = r"\text{falls}\ " + cond
        else:
            tail = ""
        rows.append(r"\quad " + left.strip() + (r"\qquad " + tail if tail else ""))
    return rows


@dataclass
class ItemVM:
    """Eine benannte Grösse: »Beschriftung = Wert«, gesetzt oder als Text."""
    tex: "str | None"
    text: str
    lines: "list[str] | None" = None      # Fallunterscheidung: Kopf und Zeilen


def item_vm(name: str, obj, decimal: bool, digits: int, big: bool = False) -> ItemVM:
    value = _tex_value(obj, decimal, digits)
    head = label_tex(name)
    fallback = (tex_to_text(name) + " = " if name else "") + text_any(obj, decimal, digits)
    cases = split_cases(value)
    if cases:
        return ItemVM(None, fallback, [(head + " =") if head else r"\text{abschnittsweise}"] + cases)
    value = re.sub(r"(?<!d)\\frac", r"\\dfrac", value)       # Brüche in voller Grösse
    body = (head + " = " if head else "") + value
    return ItemVM(body, fallback)


def split_math(text: str) -> "list[tuple[str, str]]":
    """Teilt einen Satz mit $…$ in [('t', Text), ('m', LaTeX), …]."""
    parts: list[tuple[str, str]] = []
    pieces = text.split("$")
    if len(pieces) % 2 == 0:               # unpaariges $ — alles als Text
        return [("t", tex_to_text(text) if "\\" in text else text)]
    for i, piece in enumerate(pieces):
        if not piece:
            continue
        if i % 2:
            parts.append(("m", piece))
        else:
            parts.append(("t", tex_to_text(piece) if "\\" in piece else piece))
    return parts


@dataclass
class StepVM:
    index: int
    title: str
    rule: str
    detail: "list[tuple[str, str]]"
    items: "list[ItemVM]"
    note: str
    table: object = None          # dmatkit.Table oder None


@dataclass
class ResultVM:
    title: str
    summary: "list[tuple[str, str]]"
    warnings: "list[list[tuple[str, str]]]"
    steps: "list[StepVM]"
    highlights: "list[ItemVM]"
    decimal: bool = False


def build_vm(result, decimal: bool = False) -> ResultVM:
    """Macht aus einem dmatkit-Ergebnis das, was die Karten zeigen."""
    digits = getattr(result, "digits", 4)
    result.decimal = decimal
    steps = []
    for index, step in enumerate(result.steps, 1):
        steps.append(StepVM(
            index=index, title=step.title, rule=step.rule or "",
            detail=split_math(step.detail) if step.detail else [],
            items=[item_vm(n, o, decimal, digits) for n, o in step.items],
            note=step.note or "", table=getattr(step, "table", None)))
    return ResultVM(
        title=tex_to_text(result.title),
        summary=split_math(result.summary) if result.summary else [],
        warnings=[split_math(w) for w in result.warnings],
        steps=steps,
        highlights=[item_vm(n, o, decimal, digits, big=True) for n, o in result.highlights],
        decimal=decimal)


#: Schriftgrössen der Formeln in logischen Bildpunkten; setzt `setup_math_sizes`.
MATH_PX = {"detail": 15, "item": 17, "big": 20}


def setup_math_sizes():
    """Nach dem Start: die Formelgrösse an die Schrift der App anpassen."""
    def em(pt: int) -> int:
        return QFontInfo(FONTS.f(pt)).pixelSize() or int(pt * 4 / 3)
    MATH_PX["detail"] = int(em(11) * 1.18)
    MATH_PX["item"] = int(em(12) * 1.2)
    MATH_PX["big"] = int(em(13) * 1.4)


def warm_vm(vm: ResultVM):
    """Setzt alle Formeln eines Ergebnisses vor — im Hintergrundfaden."""
    for step in vm.steps:
        for kind, piece in step.detail:
            if kind == "m":
                render_mask(piece, MATH_PX["detail"])
        for item in step.items:
            _warm_item(item, MATH_PX["item"])
    for kind, piece in vm.summary:
        if kind == "m":
            render_mask(piece, MATH_PX["detail"])
    for item in vm.highlights:
        _warm_item(item, MATH_PX["big"])


def _warm_item(item: ItemVM, px: int):
    if item.tex:
        render_mask(item.tex, px)
    for line in item.lines or []:
        render_mask(line, px)


# ==========================================================================
# Bausteine
# ==========================================================================

class Card(QFrame):
    """Fläche mit runden Ecken und weichem Schatten."""

    def __init__(self, parent=None, pad: int = 26):
        super().__init__(parent)
        self.setObjectName("card")
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(pad, pad, pad, pad)
        self.lay.setSpacing(0)
        self.shadow = QGraphicsDropShadowEffect(self)
        self.setGraphicsEffect(self.shadow)
        self.restyle()

    def restyle(self):
        self.shadow.setBlurRadius(30)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(4)
        self.shadow.setColor(QColor(0, 0, 0, T["shadow"] // (3 if T["dark"] else 1)))


class Segmented(QWidget):
    """Segmentierte Umschaltleiste wie in iOS; jedes Segment so breit wie sein Text."""

    changed = Signal(int)

    def __init__(self, options: Sequence[str], value: int = 0,
                 seg_width: int | None = None, parent=None, pad: int = 30):
        super().__init__(parent)
        self.options = list(options)
        self.value = value
        self.enabled = True
        self.disabled_hint = ""
        self.font_ = FONTS.f(11)
        metrics = QFontMetricsF(self.font_)
        self.widths = [seg_width or int(metrics.horizontalAdvance(o)) + pad
                       for o in self.options]
        self.setFixedSize(sum(self.widths) + 6, 30)
        self.setCursor(Qt.PointingHandCursor)

    def _box(self, i: int) -> QRectF:
        return QRectF(3 + sum(self.widths[:i]), 3, self.widths[i], self.height() - 6)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(qcolor("fill2") if not self.enabled else qcolor("fill"))
        p.drawRoundedRect(QRectF(0, 0, self.width(), self.height()), 0, 0)
        p.setFont(self.font_)
        for i, text in enumerate(self.options):
            box = self._box(i)
            if i == self.value:
                p.setBrush(qcolor("fill") if not self.enabled else qcolor("seg"))
                p.setPen(Qt.NoPen)
                p.drawRoundedRect(box, 0, 0)
            if not self.enabled:
                p.setPen(qcolor("ter"))
            else:
                p.setPen(qcolor("text") if i == self.value else qcolor("sec"))
            p.drawText(box, Qt.AlignCenter, text)

    def mousePressEvent(self, event):
        if not self.enabled:
            return
        px = event.position().x() - 3
        edge = 0
        for i, w in enumerate(self.widths):
            edge += w
            if px < edge:
                if i != self.value:
                    self.value = i
                    self.update()
                    self.changed.emit(i)
                return

    def set_value(self, i: int):
        self.value = i
        self.update()

    def set_enabled(self, on: bool):
        """Schaltet die Leiste stumm: sie bleibt stehen, nimmt aber keine Klicks."""
        if self.enabled == on:
            return
        self.enabled = on
        self.setCursor(Qt.PointingHandCursor if on else Qt.ArrowCursor)
        self.setToolTip("" if on else self.disabled_hint)
        self.update()


class Stepper(QWidget):
    """Minus / Wert / Plus in einer Pille."""

    stepped = Signal(int)

    def __init__(self, value: int, parent=None):
        super().__init__(parent)
        self.value = value
        self.font_ = FONTS.f(12)
        self.setFixedSize(96, 30)
        self.setCursor(Qt.PointingHandCursor)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(qcolor("fill"))
        p.drawRoundedRect(QRectF(0, 0, self.width(), self.height()), 0, 0)
        p.setFont(self.font_)
        p.setPen(qcolor("text"))
        h = self.height()
        p.drawText(QRectF(0, 0, 32, h), Qt.AlignCenter, MINUS)
        p.drawText(QRectF(32, 0, 32, h), Qt.AlignCenter, str(self.value))
        p.drawText(QRectF(64, 0, 32, h), Qt.AlignCenter, "+")

    def mousePressEvent(self, event):
        px = event.position().x()
        if px < 32:
            self.stepped.emit(-1)
        elif px > 64:
            self.stepped.emit(1)

    def set_value(self, value: int):
        self.value = value
        self.update()


class Dropdown(QPushButton):
    """Auswahlfeld mit Menü."""

    picked = Signal(str)

    def __init__(self, options: Sequence[str], value: str | None = None,
                 placeholder: str | None = None, parent=None):
        super().__init__(parent)
        self.setObjectName("dropdown")
        self.options = list(options)
        self.value = value if value is not None else (self.options[0] if self.options else "")
        self.placeholder = placeholder
        self.setText(f"  {placeholder or self.value}   ⌄")
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(30)
        self.clicked.connect(self._open)

    def set_options(self, options: Sequence[str]):
        self.options = list(options)

    def _open(self):
        if not self.options:
            return
        menu = QMenu(self)
        menu.setFont(FONTS.f(11))
        for opt in self.options:
            menu.addAction(opt, lambda o=opt: self._choose(o))
        menu.exec(self.mapToGlobal(QPoint(0, self.height() + 2)))

    def _choose(self, opt: str):
        self.value = opt
        if not self.placeholder:
            self.setText(f"  {opt}   ⌄")
        self.picked.emit(opt)

    def set_value(self, value: str):
        self.value = value
        if not self.placeholder:
            self.setText(f"  {value}   ⌄")


def pill(text: str, kind: str = "primary", on_click: Callable | None = None,
         small: bool = False) -> QPushButton:
    btn = QPushButton(text)
    btn.setObjectName(kind)
    btn.setCursor(Qt.PointingHandCursor)
    btn.setFixedHeight(30 if small else 38)
    btn.setFont(FONTS.f(11 if small else 12))
    if on_click:
        btn.clicked.connect(on_click)
    return btn


def label(text: str, size: int = 12, color: str = "text", weight=QFont.Normal,
          display: bool = False, mono: bool = False, wrap: bool = False) -> QLabel:
    lb = QLabel(text)
    lb.setFont(FONTS.f(size, weight, display, mono))
    lb.setProperty("colorKey", color)
    lb.setStyleSheet(f"color: {T[color]}; background: transparent;")
    lb.setWordWrap(wrap)
    lb.setTextInteractionFlags(Qt.TextSelectableByMouse)
    if wrap:
        lb.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
    return lb


def wrapped(text: str, width: int, size: int = 11, color: str = "sec") -> QLabel:
    """
    Umbrochener Text mit selbst berechneter Höhe.

    Qt bestimmt die Höhe eines Labels, bevor es den Umbruch kennt; in einem
    Rahmen mit fester Breite bleibt sonst nur die erste Zeile sichtbar.
    """
    lb = label(text, size, color, wrap=True)
    lb.setFixedWidth(width)
    metrics = QFontMetricsF(lb.font())
    box = metrics.boundingRect(QRectF(0, 0, width, 10000), int(Qt.TextWordWrap), text)
    lb.setMinimumHeight(int(box.height()) + 4)
    return lb


def entry(text: str = "", width: int | None = None, placeholder: str = "",
          size: int = 13, height: int = 36) -> QLineEdit:
    """Ein Eingabefeld im Stil der App."""
    field = QLineEdit(text)
    field.setObjectName("cell")
    field.setFont(FONTS.f(size))
    field.setFixedHeight(height)
    if width:
        field.setFixedWidth(width)
    if placeholder:
        field.setPlaceholderText(placeholder)
    return field


class Badge(QLabel):
    def __init__(self, text: str, fg: str, bg: str, parent=None, small: bool = False):
        super().__init__(text, parent)
        self.fg, self.bg, self.small = fg, bg, small
        self.setFont(FONTS.f(10 if small else 11, QFont.DemiBold))
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.restyle()

    def restyle(self):
        pad = "2px 9px" if self.small else "4px 14px"
        self.setStyleSheet(
            f"color: {T[self.fg]}; background: {T[self.bg]};"
            f"border-radius: 0px; padding: {pad};")


def restyle_labels(root: QWidget):
    """Färbt alle mit label() erzeugten Beschriftungen neu ein."""
    for lb in root.findChildren(QLabel):
        if isinstance(lb, Badge):
            lb.restyle()
        elif isinstance(lb, RichLabel):
            lb.restyle()
        elif lb.property("colorKey"):
            lb.setStyleSheet(
                f"color: {T[lb.property('colorKey')]}; background: transparent;")
    for fl in root.findChildren(FormulaLabel):
        fl.update()


class Toast(QFrame):
    """Kurze Rückmeldung unten rechts, die sich selbst wieder ausblendet."""

    HEIGHT = 40

    def __init__(self, parent: QWidget, text: str, kind: str = "info"):
        super().__init__(parent)
        self.setObjectName("toast")
        tone = {"info": "text", "ok": "green", "bad": "red"}[kind]
        glyph = {"info": "", "ok": "✓  ", "bad": "✕  "}[kind]
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 0, 18, 0)
        self.label = label(glyph + text, 11, tone, weight=QFont.DemiBold)
        lay.addWidget(self.label)
        self.setFixedHeight(self.HEIGHT)
        self.setStyleSheet(
            f"#toast {{ background: {T['card']}; border-radius: 0px;"
            f"border: 1px solid {T['line']}; }}")
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(26)
        shadow.setYOffset(5)
        shadow.setColor(QColor(0, 0, 0, 60 if not T["dark"] else 150))
        self.setGraphicsEffect(shadow)
        self.adjustSize()
        self.setFixedWidth(max(180, self.label.sizeHint().width() + 40))


class ToastArea(QObject):
    """Stapelt die Meldungen und räumt sie nach ein paar Sekunden weg."""

    def __init__(self, host: QWidget):
        super().__init__(host)
        self.host = host
        self.items: list[Toast] = []

    def show(self, text: str, kind: str = "info", ms: int = 2600):
        toast = Toast(self.host, text, kind)
        self.items.append(toast)
        if len(self.items) > 3:
            self._drop(self.items[0])
        toast.show()
        self._layout()
        QTimer.singleShot(ms, lambda: self._drop(toast))

    def _drop(self, toast: Toast):
        if toast not in self.items:
            return
        self.items.remove(toast)
        toast.hide()
        toast.deleteLater()
        self._layout()

    def _layout(self):
        bottom = self.host.height() - 24
        for toast in reversed(self.items):
            bottom -= Toast.HEIGHT + 8
            toast.move(self.host.width() - toast.width() - 24, bottom)
            toast.raise_()

    def reposition(self):
        self._layout()


# ==========================================================================
# Gesetzte Formeln im Fenster
# ==========================================================================

class RichLabel(QLabel):
    """
    Ein Absatz mit Text und eingebetteten Formeln; bricht um, wie ein Label es tut.

    Die Formeln stehen als Bilder (data-URI) im Fliesstext, mit der
    Mathe-Achse auf Höhe der Textmitte.
    """

    def __init__(self, parts: "list[tuple[str, str]]", size: int = 11,
                 color: str = "sec", weight=QFont.Normal, math_key: str = "detail",
                 parent=None):
        super().__init__(parent)
        self.parts, self.color_key, self.math_key = parts, color, math_key
        self.setFont(FONTS.f(size, weight))
        self.setWordWrap(True)
        self.setTextFormat(Qt.RichText)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
        self.restyle()

    def restyle(self):
        col = T[self.color_key]
        self.setStyleSheet(f"color: {col}; background: transparent;")
        math_col = T["text"] if self.color_key in ("sec", "ter") else col
        out = []
        for kind, piece in self.parts:
            if kind == "m":
                uri = mask_data_uri(piece, MATH_PX[self.math_key], math_col)
                if uri is not None:
                    src, w, h = uri
                    out.append(f'<img src="{src}" width="{w:.1f}" height="{h:.1f}" '
                               f'style="vertical-align: middle;">')
                else:
                    out.append(html.escape(tex_to_text(piece)))
            else:
                esc = html.escape(piece).replace("\n", "<br>")
                if esc.startswith(" "):
                    esc = "&nbsp;" + esc[1:]
                if esc.endswith(" "):
                    esc = esc[:-1] + "&nbsp;"
                out.append(esc)
        self.setText("".join(out))

    def heightForWidth(self, w: int) -> int:
        return super().heightForWidth(w)


class FormulaLabel(QWidget):
    """
    Eine Zeile (oder ein paar Zeilen) gesetzter Formel, in Textfarbe gezeichnet.

    Ist sie breiter als der Platz, wird sie verkleinert statt abgeschnitten.
    Hat Mathtext sie nicht setzen können, steht der Unicode-Text da.
    """

    MIN_SCALE = 0.55

    def __init__(self, item: ItemVM, px: int, parent=None):
        super().__init__(parent)
        self.item, self.px = item, px
        lines = item.lines if item.lines else ([item.tex] if item.tex else [])
        masks = [render_mask(t, px) for t in lines]
        self.masks = masks if masks and all(m is not None for m in masks) else None
        self._cache: dict = {}
        self.gap = 4
        if self.masks:
            sizes = [m.size for m in self.masks]
            self.nat_w = max(w for w, _ in sizes) + 2
            self.nat_h = sum(h for _, h in sizes) + self.gap * (len(sizes) - 1) + 2
        else:
            self.font_ = FONTS.f(12)
            fm = QFontMetricsF(self.font_)
            self.nat_w = fm.horizontalAdvance(item.text) + 4
            self.nat_h = fm.height() + 4
        self.setFixedHeight(int(self.nat_h + 0.5))

    def natural_width(self) -> float:
        return self.nat_w

    def fit(self, avail: float) -> QSize:
        """Grösse bei verfügbarer Breite: verkleinert, wenn nötig."""
        if self.masks and self.nat_w > avail > 0:
            s = max(self.MIN_SCALE, avail / self.nat_w)
        else:
            s = 1.0
        return QSize(int(min(self.nat_w, avail) if avail > 0 else self.nat_w),
                     int(self.nat_h * s + 0.5))

    def sizeHint(self) -> QSize:
        return QSize(int(self.nat_w), int(self.nat_h + 0.5))

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform
                         | QPainter.TextAntialiasing)
        color = T["text"]
        if not self.masks:
            p.setFont(self.font_)
            p.setPen(QColor(color))
            p.drawText(QRectF(0, 0, self.width(), self.height()),
                       Qt.AlignLeft | Qt.AlignVCenter, self.item.text)
            return
        scale = 1.0
        if self.nat_w > self.width() > 0:
            scale = max(self.MIN_SCALE, self.width() / self.nat_w)
        y = 1.0
        dpr = self.devicePixelRatioF()
        for mask in self.masks:
            key = (id(mask), color, round(scale * 100), dpr)
            if key not in self._cache:
                self._cache[key] = scaled_image(mask, color, scale, dpr)
            image, w, h = self._cache[key]
            p.drawImage(QPointF(1, y), image)
            y += h + self.gap * scale


class FlowBox(QWidget):
    """Legt Kinder nebeneinander und bricht in eine neue Zeile, wenn der Platz fehlt."""

    def __init__(self, hgap: int = 26, vgap: int = 8, parent=None):
        super().__init__(parent)
        self.kids: list[FormulaLabel] = []
        self.hgap, self.vgap = hgap, vgap
        self._last_width = -1

    def add(self, kid: "FormulaLabel"):
        kid.setParent(self)
        self.kids.append(kid)
        kid.show()
        self.relayout(max(self.width(), 560))

    def relayout(self, width: int):
        if width <= 0:
            return
        x0 = y0 = row_h = 0
        for kid in self.kids:
            size = kid.fit(width)
            if x0 and x0 + size.width() > width:
                x0, y0, row_h = 0, y0 + row_h + self.vgap, 0
            kid.setGeometry(x0, y0, size.width(), size.height())
            x0 += size.width() + self.hgap
            row_h = max(row_h, size.height())
        total = y0 + row_h
        if self.height() != total:
            self.setFixedHeight(total)
        self._last_width = width

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if event.size().width() != self._last_width:
            self.relayout(event.size().width())


class TableView(QWidget):
    """
    Eine Tabelle des Rechenwegs (Wahrheitstabelle, Schrittzähler, Beweiszeilen).

    Zeilen mit Markierung sind eingefärbt: ok grün, bad rot, hi blau. Das Zeichen
    am rechten Rand (✓ ✗ ◀) trägt dieselbe Bedeutung, damit sie nicht nur an der
    Farbe hängt.
    """

    PAD_X, PAD_Y, MAX_CELL = 12, 5, 380
    GLYPH = {"ok": "✓", "bad": "✗", "hi": "◀"}
    TONE = {"ok": ("green_soft", "green"), "bad": ("red_soft", "red"), "hi": ("hi_soft", "text")}

    def __init__(self, table, parent=None):
        super().__init__(parent)
        self.table = table
        self.head_font = FONTS.f(11, QFont.DemiBold)
        self.cell_font = FONTS.f(11)
        hm, cm = QFontMetricsF(self.head_font), QFontMetricsF(self.cell_font)
        self.header = [tex_to_text(h) if "\\" in str(h) else str(h) for h in table.header]
        self.rows = [[text_any(c) for c in row] for row in table.rows]
        self.marked = bool(table.marks)
        widths = []
        for i, h in enumerate(self.header):
            cells = [cm.horizontalAdvance(r[i]) for r in self.rows if i < len(r)]
            widths.append(min(self.MAX_CELL, max([hm.horizontalAdvance(h)] + cells)) + 2 * self.PAD_X)
        self.widths = [max(36, w) for w in widths]
        self.glyph_w = 28 if self.marked else 0
        self.row_h = int(max(hm.height(), cm.height())) + 2 * self.PAD_Y
        self.setFixedSize(int(sum(self.widths)) + self.glyph_w + 1, self.row_h * (len(self.rows) + 1) + 1)

    def sizeHint(self) -> QSize:
        return self.size()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.TextAntialiasing)
        total_w = self.width() - 1
        h = self.row_h
        # Kopf
        p.fillRect(QRectF(0, 0, total_w, h), qcolor("fill"))
        p.setFont(self.head_font)
        p.setPen(qcolor("text"))
        x = 0.0
        for i, text in enumerate(self.header):
            p.drawText(QRectF(x, 0, self.widths[i], h), Qt.AlignCenter, text)
            x += self.widths[i]
        # Zeilen
        p.setFont(self.cell_font)
        for r, row in enumerate(self.rows):
            y = (r + 1) * h
            mark = self.table.marks.get(r, "")
            if mark in self.TONE:
                p.fillRect(QRectF(0, y, total_w, h), qcolor(self.TONE[mark][0]))
            x = 0.0
            p.setPen(qcolor("text"))
            for i, text in enumerate(row[:len(self.widths)]):
                rect = QRectF(x, y, self.widths[i], h)
                fm = QFontMetricsF(self.cell_font)
                p.drawText(rect, Qt.AlignCenter, fm.elidedText(text, Qt.ElideRight, self.widths[i] - 2 * self.PAD_X))
                x += self.widths[i]
            if mark in self.GLYPH:
                p.setPen(qcolor(self.TONE[mark][1]))
                p.drawText(QRectF(x, y, self.glyph_w, h), Qt.AlignCenter, self.GLYPH[mark])
        # Linien
        p.setPen(QPen(qcolor("hair"), 1))
        for r in range(len(self.rows) + 2):
            p.drawLine(0, r * h, total_w, r * h)
        x = 0.0
        for w in self.widths:
            p.drawLine(int(x), 0, int(x), self.height() - 1)
            x += w
        p.drawLine(int(x), 0, int(x), self.height() - 1)
        p.drawLine(total_w, 0, total_w, self.height() - 1)
        p.setPen(QPen(qcolor("line"), 1))
        p.drawLine(0, h, total_w, h)


class TableBox(QScrollArea):
    """Hält eine TableView; ist sie breiter als der Platz, läuft sie seitlich, nicht über den Rand."""

    def __init__(self, table, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWidgetResizable(False)
        self.view = TableView(table)
        self.setWidget(self.view)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._fit(560)

    def _fit(self, width: int):
        extra = 14 if self.view.width() > width else 0
        self.setFixedHeight(self.view.height() + 2 + extra)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit(event.size().width())

    def sizeHint(self) -> QSize:
        return QSize(min(self.view.width() + 2, 900), self.view.height() + 2)


# ==========================================================================
# Ergebnisse darstellen
# ==========================================================================

class ResultView(QScrollArea):
    """Zeigt ein dmatkit-Ergebnis: Titel, Schrittkarten mit gesetzten Formeln, Endwerte."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.host = QWidget()
        self.host.setObjectName("page")
        self.lay = QVBoxLayout(self.host)
        self.lay.setContentsMargins(0, 0, 8, 0)
        self.lay.setSpacing(10)
        self.setWidget(self.host)
        self.vm: ResultVM | None = None
        self.message = ("info", "Noch nichts gerechnet.")
        self._placeholder()

    def _clear(self):
        while self.lay.count():
            item = self.lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()

    def _placeholder(self, text: str = "Noch nichts gerechnet."):
        self._clear()
        self.vm = None
        self.message = ("info", text)
        self.lay.addWidget(label(text, 12, "ter", wrap=True))
        self.lay.addStretch(1)

    def placeholder(self, text: str):
        self._placeholder(text)

    def busy(self, text: str = "Rechne …"):
        """Ladeanzeige: der Orb dreht sich, solange im Hintergrund gerechnet wird."""
        self._clear()
        self.vm = None
        self.message = ("busy", text)
        orb = ThinkingOrb(64)
        orb.set_colors([T["blue_hi"], T["text"], T["sec"]])
        self.lay.addSpacing(40)
        self.lay.addWidget(orb, 0, Qt.AlignHCenter)
        self.lay.addSpacing(8)
        self.lay.addWidget(label(text, 12, "ter"), 0, Qt.AlignHCenter)
        self.lay.addStretch(1)

    def show_error(self, message: str):
        self._clear()
        self.vm = None
        self.message = ("error", message)
        box = QFrame()
        box.setObjectName("errbox")
        box.setStyleSheet(f"#errbox {{ background: {T['red_soft']}; border-radius: 0px; }}")
        bl = QVBoxLayout(box)
        bl.setContentsMargins(16, 13, 16, 13)
        bl.setSpacing(4)
        bl.addWidget(label("Das hat nicht geklappt", 12, "red", QFont.DemiBold))
        bl.addWidget(label(message, 12, "red", wrap=True))
        self.lay.addWidget(box)
        self.lay.addStretch(1)

    def restyle(self):
        """Nach einem Themawechsel dieselbe Anzeige in neuen Farben aufbauen."""
        if self.vm is not None:
            self.show_vm(self.vm)
        elif self.message[0] == "error":
            self.show_error(self.message[1])
        elif self.message[0] == "busy":
            self.busy(self.message[1])
        else:
            self._placeholder(self.message[1])

    def show_result(self, result, decimal: bool = False):
        self.show_vm(build_vm(result, decimal))

    def show_vm(self, vm: ResultVM):
        self._clear()
        self.vm = vm
        self.message = ("info", "")
        head = QVBoxLayout()
        head.setSpacing(3)
        title = label(vm.title, 15, weight=QFont.DemiBold, wrap=True)
        head.addWidget(title)
        if vm.summary:
            head.addWidget(RichLabel(vm.summary, 11, "sec"))
        holder = QWidget()
        holder.setLayout(head)
        self.lay.addWidget(holder)

        for warning in vm.warnings:
            box = QFrame()
            box.setObjectName("warnbox")
            box.setStyleSheet(f"#warnbox {{ background: {T['amber_soft']}; border-radius: 0px; }}")
            bl = QVBoxLayout(box)
            bl.setContentsMargins(14, 10, 14, 10)
            bl.addWidget(RichLabel(warning, 11, "amber"))
            self.lay.addWidget(box)

        for step in vm.steps:
            self.lay.addWidget(self._step_card(step))

        if vm.highlights:
            self.lay.addWidget(self._result_card(vm.highlights))
        self.lay.addStretch(1)

    def _step_card(self, step: StepVM) -> QWidget:
        card = QFrame()
        card.setObjectName("stepcard")
        card.setStyleSheet(f"#stepcard {{ background: {T['fill2']}; border-radius: 0px;"
                           f"border: 1px solid {T['hair']}; }}")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(16, 13, 16, 13)
        lay.setSpacing(5)
        head = QHBoxLayout()
        head.setSpacing(8)
        head.addWidget(label(f"{step.index}. {step.title}", 12, weight=QFont.DemiBold),
                       0, Qt.AlignVCenter)
        if step.rule:
            head.addWidget(Badge(step.rule, "blue", "blue_soft", small=True), 0, Qt.AlignVCenter)
        head.addStretch(1)
        lay.addLayout(head)
        if step.detail:
            lay.addWidget(RichLabel(step.detail, 11, "sec"))
        if step.items:
            flow = FlowBox()
            for item in step.items:
                flow.add(FormulaLabel(item, MATH_PX["item"]))
            lay.addWidget(flow)
        if step.table is not None:
            lay.addWidget(TableBox(step.table))
            if step.table.caption:
                lay.addWidget(label(step.table.caption, 10, "ter", wrap=True))
        if step.note:
            lay.addWidget(label(step.note, 10, "ter", wrap=True))
        return card

    def _result_card(self, highlights: "list[ItemVM]") -> QWidget:
        card = QFrame()
        card.setObjectName("rescard")
        card.setStyleSheet(f"#rescard {{ background: {T['blue_soft']}; border-radius: 0px; }}")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(16, 13, 16, 14)
        lay.setSpacing(6)
        lay.addWidget(label("Ergebnis", 10, "blue", QFont.DemiBold))
        flow = FlowBox(hgap=30, vgap=8)
        for item in highlights:
            flow.add(FormulaLabel(item, MATH_PX["big"]))
        lay.addWidget(flow)
        return card


# ==========================================================================
# Zeichnungen im Fenster
# ==========================================================================

def theme_figure(figure):
    """
    Passt eine Matplotlib-Figur an das gerade gewählte Thema an.

    Die Zeichnungen aus dmatkit.grafik.plots kennen `dark=` und kommen schon im
    richtigen Thema an; hier werden nur noch Hintergründe, Achsen und
    Beschriftungen auf die Kartenfarbe der App gezogen, damit zwischen
    Zeichnung und Karte keine Kante steht. Die Linien der Zeichnung selbst
    bleiben unberührt — ihre Farben tragen Bedeutung.
    """
    card, text, sec, line = T["card"], T["text"], T["sec"], T["line"]
    figure.patch.set_facecolor(card)
    for ax in figure.axes:
        ax.set_facecolor(card)
        ax.title.set_color(text)
        ax.tick_params(colors=sec, which="both")
        for spine in ax.spines.values():
            spine.set_color(line)
        for axis in (ax.xaxis, ax.yaxis):
            axis.label.set_color(sec)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(card)
            legend.get_frame().set_edgecolor(line)
            for entry_ in legend.get_texts():
                entry_.set_color(text)


class PlotCanvas(QWidget):
    """Eine Zeichnung im Fenster selbst, nicht im Browser."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = None
        self.figure = None
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(10)
        self.empty = label("Noch keine Zeichnung — erst rechnen.", 13, "sec", wrap=True)
        self.lay.addWidget(self.empty, 1, Qt.AlignCenter)
        self.readout = label("", 11, "sec")
        self.lay.addWidget(self.readout)
        self.hint = label("", 11, "ter", wrap=True)
        self.lay.addWidget(self.hint)
        self._drag = None

    @staticmethod
    def available() -> bool:
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvas  # noqa
        except Exception:
            return False
        return True

    def show_figure(self, figure, hint: str = ""):
        """Zeigt eine Matplotlib-Figur; die vorige wird freigegeben."""
        from matplotlib.backends.backend_qtagg import FigureCanvas

        self.clear()
        theme_figure(figure)
        self.figure = figure
        self.canvas = FigureCanvas(figure)
        self.canvas.setMinimumHeight(300)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.empty.hide()
        self.lay.insertWidget(0, self.canvas, 1)
        self.canvas.draw_idle()
        self.hint.setText(hint + ("  " if hint else "") +
                          "Mausrad zoomt, Ziehen verschiebt, Doppelklick setzt zurück.")
        self._home = [(ax, ax.get_xlim(), ax.get_ylim()) for ax in figure.axes]
        for ev, fn in (("scroll_event", self._on_scroll), ("button_press_event", self._on_press),
                       ("button_release_event", self._on_release),
                       ("motion_notify_event", self._on_move), ("axes_leave_event", self._on_leave)):
            self.canvas.mpl_connect(ev, fn)

    # Zoom, Verschieben und Koordinaten-Anzeige direkt auf der Zeichnung.
    def _on_scroll(self, ev):
        ax = ev.inaxes
        if ax is None or ev.xdata is None:
            return
        k = 0.8 if ev.step > 0 else 1.25
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        ax.set_xlim(ev.xdata - (ev.xdata - x0) * k, ev.xdata + (x1 - ev.xdata) * k)
        ax.set_ylim(ev.ydata - (ev.ydata - y0) * k, ev.ydata + (y1 - ev.ydata) * k)
        self.canvas.draw_idle()

    def _on_press(self, ev):
        if ev.inaxes is None or ev.button != 1:
            return
        if ev.dblclick:
            for ax, xl, yl in self._home:
                ax.set_xlim(xl)
                ax.set_ylim(yl)
            self.canvas.draw_idle()
        else:
            self._drag = (ev.inaxes, ev.x, ev.y, ev.inaxes.get_xlim(), ev.inaxes.get_ylim())
            self.canvas.setCursor(Qt.ClosedHandCursor)

    def _on_release(self, ev):
        self._drag = None
        self.canvas.setCursor(Qt.ArrowCursor)

    def _on_move(self, ev):
        if self._drag is not None:
            ax, px, py, (x0, x1), (y0, y1) = self._drag
            inv = ax.transData.inverted()
            (ax_, ay_), (bx_, by_) = inv.transform((px, py)), inv.transform((ev.x, ev.y))
            dx, dy = ax_ - bx_, ay_ - by_
            ax.set_xlim(x0 + dx, x1 + dx)
            ax.set_ylim(y0 + dy, y1 + dy)
            self.canvas.draw_idle()
            return
        if ev.inaxes is None or ev.xdata is None:
            return self._on_leave(ev)
        text = f"x = {ev.xdata:.3f}   y = {ev.ydata:.3f}"
        for ln in ev.inaxes.get_lines():          # Wert der ersten Kurve unter dem Mauszeiger
            xs, ys = ln.get_xdata(), ln.get_ydata()
            if len(xs) > 50 and xs[0] <= ev.xdata <= xs[-1]:
                text += f"   Kurve: {np.interp(ev.xdata, xs, ys):.3f}"
                break
        self.readout.setText(text)

    def _on_leave(self, ev=None):
        self.readout.setText("")

    def show_message(self, text: str):
        self.clear()
        self.empty.setText(text)
        self.empty.show()
        self.hint.setText("")

    def clear(self):
        self._drag = None
        self.readout.setText("")
        if self.canvas is not None:
            self.canvas.hide()
            self.canvas.setParent(None)
            self.canvas.deleteLater()
            self.canvas = None
        if self.figure is not None:
            import matplotlib.pyplot as plt
            plt.close(self.figure)
            self.figure = None

    def restyle(self):
        if self.figure is not None and self.canvas is not None:
            theme_figure(self.figure)
            self.canvas.draw_idle()


class PlotPane(QWidget):
    """
    Die Ansicht »Zeichnung«: Leinwand, Umschalter zwischen Zeichnungen und —
    wenn Plotly da ist — dasselbe interaktiv im Browser. Speichern steht im
    Menü »Exportieren«.
    """

    variant_changed = Signal(int)
    browser_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        self.head_box = QWidget()
        self.head = QHBoxLayout(self.head_box)
        self.head.setContentsMargins(0, 0, 0, 0)
        self.head.setSpacing(8)
        self.seg_holder = QHBoxLayout()
        self.head.addLayout(self.seg_holder)
        self.head.addStretch(1)
        self.btn_browser = pill("Interaktiv im Browser", "plain",
                                lambda: self.browser_requested.emit(), small=True)
        self.btn_browser.setVisible(False)
        self.head.addWidget(self.btn_browser)
        self.head_box.hide()
        lay.addWidget(self.head_box)
        self.canvas = PlotCanvas()
        lay.addWidget(self.canvas, 1)
        self.seg = None
        self.names: list[str] = []

    @property
    def figure(self):
        return self.canvas.figure

    def _update_head(self):
        self.head_box.setVisible(self.seg is not None or not self.btn_browser.isHidden())

    def allow_browser(self, on: bool):
        self.btn_browser.setVisible(bool(on) and dk.has_plotly())
        self._update_head()

    def set_variants(self, names: Sequence[str], current: int = 0):
        """Zeigt den Umschalter nur, wenn es mehr als eine Zeichnung gibt."""
        if list(names) == self.names:
            if self.seg is not None:
                self.seg.set_value(current)
            return
        self.names = list(names)
        if self.seg is not None:
            self.seg.hide()
            self.seg.setParent(None)
            self.seg.deleteLater()
            self.seg = None
        if len(names) > 1:
            self.seg = Segmented(names, current)
            self.seg.changed.connect(self.variant_changed.emit)
            self.seg_holder.addWidget(self.seg)
        self._update_head()

    def show_figure(self, figure, hint: str = ""):
        self.canvas.show_figure(figure, hint)

    def show_message(self, text: str):
        self.canvas.show_message(text)

    def restyle(self):
        self.canvas.restyle()


def save_figure_dialog(parent, figure, default_name: str = "dmatkit-zeichnung") -> str:
    """Fragt nach einem Dateinamen und speichert die Figur; gibt den Pfad oder '' zurück."""
    if figure is None:
        return ""
    path, _ = QFileDialog.getSaveFileName(
        parent, "Zeichnung speichern", str(Path.home() / f"{default_name}.png"),
        "PNG-Bild (*.png);;PDF (*.pdf);;SVG (*.svg)")
    if not path:
        return ""
    figure.savefig(path, dpi=200, facecolor=figure.get_facecolor(), bbox_inches="tight")
    return path


# ==========================================================================
# Rechnen im Hintergrund
# ==========================================================================

def selftest_report(appid: str) -> WolframReport:
    """Die Verbindungsprüfung als Bericht: Namensauflösung, TLS, Testabfrage."""
    report = WolframReport()
    for name, ok, note in network_selftest(appid):
        report.checks.append(Check(name, "ok" if ok else "bad", note=note))
    return report


class ComputeWorker(QObject):
    """Rechnet eine Seite (Ergebnis, Ansichtsmodell, Zeichnung) ausserhalb des Fensters."""

    finished = Signal(int, object)

    def __init__(self, job_id: int, run):
        super().__init__()
        self.job_id, self.run_job = job_id, run

    def run(self):
        try:
            outcome = self.run_job()
        except BaseException as exc:            # nie den Faden mit Ausnahme beenden
            outcome = Outcome(error=friendly_error(exc))
        self.finished.emit(self.job_id, outcome)


@dataclass
class Outcome:
    """Was ein Rechenlauf liefert: Ergebnis samt Ansichtsmodell, oder einen Fehlersatz."""
    result: object = None
    vm: "ResultVM | None" = None
    figure: object = None
    figure_error: str = ""
    error: str = ""
    seconds: float = 0.0


#: pyplot teilt sich einen Zustand; zwei Zeichnungen gleichzeitig sind nicht sicher.
PLOT_LOCK = threading.Lock()


def run_page_job(page: str, p: dict, decimal: bool, dark: bool, variant: int = 0) -> Outcome:
    """
    Alles, was dauern kann, an einem Ort: rechnen, Formeln setzen, zeichnen.
    Läuft im Hintergrundfaden; gibt nie eine Ausnahme weiter.
    """
    import time
    start = time.time()
    try:
        result = calc(page, p)
    except InputError as exc:
        return Outcome(error=str(exc))
    except Exception as exc:
        return Outcome(error=friendly_error(exc))
    out = Outcome(result=result)
    try:
        out.vm = build_vm(result, decimal)
        warm_vm(out.vm)
    except Exception as exc:
        return Outcome(error=f"Der Rechenweg liess sich nicht darstellen: {exc}")
    try:
        with PLOT_LOCK:
            out.figure = make_figure(page, p, result, dark, variant)
    except NotImplementedError as exc:
        out.figure_error = str(exc) or "Zu dieser Rechnung gibt es keine Zeichnung."
    except Exception as exc:
        out.figure_error = f"Zeichnen ging nicht: {friendly_error(exc)}"
    out.seconds = time.time() - start
    return out


def run_figure_job(page: str, p: dict, result, dark: bool, variant: int) -> Outcome:
    """Nur die Zeichnung neu — nach Themawechsel oder Wechsel der Ansicht."""
    out = Outcome(result=result)
    try:
        with PLOT_LOCK:
            out.figure = make_figure(page, p, result, dark, variant)
    except NotImplementedError as exc:
        out.figure_error = str(exc) or "Zu dieser Rechnung gibt es keine Zeichnung."
    except Exception as exc:
        out.figure_error = f"Zeichnen ging nicht: {friendly_error(exc)}"
    return out


# ==========================================================================
# Prüfung mit Wolfram|Alpha: der Bereich links
# ==========================================================================

class AppIDDialog(QDialog):
    def __init__(self, parent, current: str):
        super().__init__(parent)
        self.setWindowTitle("Wolfram|Alpha AppID")
        self.setModal(True)
        self.setFixedWidth(430)
        self.setStyleSheet(parent.styleSheet())
        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 26, 28, 24)
        lay.setSpacing(10)
        lay.addWidget(label("AppID hinterlegen", 15, weight=QFont.DemiBold))
        lay.addWidget(label(
            "Eine kostenlose AppID bekommst du unter developer.wolframalpha.com. "
            "Sie wird in deinem Benutzerordner gespeichert und nur für die "
            "Prüfanfragen verwendet.", 11, "sec", wrap=True))
        self.field = QLineEdit(current)
        self.field.setObjectName("cell")
        self.field.setFont(FONTS.f(12))
        self.field.setFixedHeight(38)
        self.field.setPlaceholderText("z. B. ABCDEF-GHIJKLMNOP")
        self.field.returnPressed.connect(self.accept)
        lay.addSpacing(6)
        lay.addWidget(self.field)
        lay.addSpacing(8)

        row = QHBoxLayout()
        link = pill("Seite öffnen", "plain", lambda: webbrowser.open(
            "https://developer.wolframalpha.com/access"), small=True)
        row.addWidget(link)
        row.addStretch(1)
        row.addWidget(pill("Abbrechen", "secondary", self.reject, small=True))
        row.addWidget(pill("Speichern", "primary", self.accept, small=True))
        lay.addLayout(row)

    def key(self) -> str:
        return self.field.text().strip()



class ChecksWorker(QObject):
    """Führt eine Prüffunktion aus, ohne das Fenster anzuhalten."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, run):
        super().__init__()
        self.run_checks = run

    def run(self):
        try:
            self.finished.emit(self.run_checks())
        except (RuntimeError, ValueError) as exc:
            self.failed.emit(str(exc))
        except Exception as exc:                  # nichts darf den Faden still sterben lassen
            self.failed.emit(f"Die Prüfung ist fehlgeschlagen: {exc}")


class WolframPanel(QWidget):
    """
    Die Prüfung mit Wolfram|Alpha als eigene Karte.

    Sie kennt die Rechnung nicht: `build_checks(appid)` liefert die Aufgabe,
    die im Hintergrundfaden den Bericht baut; alles andere — AppID,
    Verbindungstest, Darstellung, Offline-Zustand — ist für jede Seite gleich.
    """

    def __init__(self, build_checks, subtitle: str, parent=None):
        super().__init__(parent)
        self.build_checks = build_checks
        self.report: WolframReport | None = None
        self.status: tuple[str, str] | None = None
        self.unavailable = ""
        self.thread = None
        self.worker = None

        self.card = Card()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.card)
        lay = self.card.lay
        lay.addWidget(label("Prüfung mit Wolfram|Alpha", 15, weight=QFont.DemiBold))
        lay.addWidget(label(subtitle, 11, "sec", wrap=True))
        lay.addSpacing(14)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn = pill("Jetzt prüfen", "primary", self.start)
        row.addWidget(self.btn)
        self.btn_key = pill("AppID", "secondary", self.ask_appid, small=True)
        row.addWidget(self.btn_key)
        self.btn_test = pill("Verbindung testen", "plain", self.start_selftest, small=True)
        row.addWidget(self.btn_test)
        row.addStretch(1)
        lay.addLayout(row)
        self.body = QVBoxLayout()
        self.body.setSpacing(0)
        lay.addSpacing(6)
        lay.addLayout(self.body)
        self.render()

    # -- Bedienung --------------------------------------------------------

    def ask_appid(self):
        dialog = AppIDDialog(self.window(), load_appid())
        if dialog.exec() == QDialog.Accepted and dialog.key():
            try:
                save_appid(dialog.key())
            except OSError as exc:
                self.status = ("bad", f"AppID nicht speicherbar: {exc}")
            self.render()

    def _run(self, job, busy_text: str):
        self.report = None
        self.status = ("busy", busy_text)
        self.render()
        self.thread = QThread(self)
        self.worker = ChecksWorker(job)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.thread.start()

    def start(self):
        if self.thread is not None:
            return
        appid = load_appid()
        if not appid:
            self.ask_appid()
            appid = load_appid()
            if not appid:
                return
        try:
            job = self.build_checks(appid)
        except (ValueError, RuntimeError) as exc:
            self.status = ("bad", str(exc))
            self.render()
            return
        self._run(job, "Frage Wolfram|Alpha …")

    def start_selftest(self):
        if self.thread is not None:
            return
        appid = load_appid()
        self._run(lambda: selftest_report(appid), "Prüfe die Verbindung …")

    def _stop_thread(self):
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait()
            self.thread = None
        self.worker = None

    def _done(self, report):
        self._stop_thread()
        self.report, self.status = report, None
        self.render()

    def _failed(self, message: str):
        self._stop_thread()
        self.report, self.status = None, ("bad", message)
        self.render()

    def wait(self, ms: int = 30000):
        """Wartet auf einen laufenden Lauf (für Tests); Ereignisse werden weiter verarbeitet."""
        import time
        end = time.time() + ms / 1000
        while self.thread is not None and time.time() < end:
            QApplication.processEvents()
            time.sleep(0.01)

    def invalidate(self):
        """Nach einer Änderung an der Eingabe gilt der alte Bericht nicht mehr."""
        if self.thread is None and (self.report is not None or self.status is not None):
            self.report, self.status = None, None
            self.render()

    def set_unavailable(self, reason: str):
        """Sagt, warum sich diese Rechnung nicht prüfen lässt (leer: sie lässt sich prüfen)."""
        if reason != self.unavailable:
            self.unavailable = reason
            self.render()

    # -- Darstellung ------------------------------------------------------

    def render(self):
        while self.body.count():
            item = self.body.takeAt(0)
            # Nach setParent(None) gibt das Element den Verweis nicht mehr
            # heraus — darum vorher festhalten.
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
        self.btn_key.setText("AppID ändern" if load_appid() else "AppID")

        if self.status:
            level, text = self.status
            color = {"busy": "sec", "bad": "red", "ok": "green"}[level]
            self.body.addSpacing(12)
            box = QFrame()
            box.setObjectName("wolferr" if level == "bad" else "wolfbusy")
            if level == "bad":
                box.setStyleSheet(f"#wolferr {{ background: {T['red_soft']}; border-radius: 0px; }}")
            bl = QVBoxLayout(box)
            bl.setContentsMargins(14, 10, 14, 10)
            bl.addWidget(label(text, 11, color, wrap=True))
            if level == "bad":
                bl.addWidget(label("Die Rechnung selbst ist davon nicht betroffen — "
                                   "nur die Gegenprobe fehlt.", 10, "ter", wrap=True))
            self.body.addWidget(box)
            self.btn.setEnabled(level != "busy")
            self.btn.setText("Prüfe …" if level == "busy" else "Jetzt prüfen")
            return

        self.btn.setEnabled(not self.unavailable)
        self.btn.setText("Jetzt prüfen")
        self.btn.setToolTip(self.unavailable)
        if self.report is None:
            if self.unavailable:
                self.body.addSpacing(12)
                self.body.addWidget(label(self.unavailable, 11, "ter", wrap=True))
            elif not load_appid():
                self.body.addSpacing(12)
                self.body.addWidget(label(
                    "Dafür braucht es einmalig eine kostenlose AppID.",
                    11, "ter", wrap=True))
            return

        report = self.report
        self.body.addSpacing(14)
        tone = {"ok": ("green", "green_soft"), "bad": ("red", "red_soft"),
                "warn": ("amber", "amber_soft")}[report.verdict]
        mark = {"ok": "✓", "bad": "✕", "warn": "!"}[report.verdict]
        self.body.addWidget(Badge(f"{mark}  {report.headline}", *tone))
        self.body.addSpacing(10)

        for check in report.checks:
            glyph = {"ok": "✓", "bad": "✕", "info": "•", "error": "!"}[check.status]
            color = {"ok": "green", "bad": "red", "info": "sec", "error": "amber"}[check.status]
            row = QHBoxLayout()
            row.setSpacing(8)
            row.addWidget(label(glyph, 11, color, weight=QFont.DemiBold), 0, Qt.AlignTop)
            text = check.name
            if check.theirs and check.theirs != check.ours:
                text += f":  wir {check.ours}, Wolfram {check.theirs}"
            elif check.ours:
                text += f":  {check.ours}"
            row.addWidget(label(text, 11, wrap=True), 1)
            holder = QWidget()
            holder.setLayout(row)
            self.body.addWidget(holder)
            if check.note:
                self.body.addWidget(label("     " + check.note, 10, "ter", wrap=True))

        if report.texts:
            self.body.addSpacing(12)
            self.body.addWidget(label("Wolframs Antworten", 11, "sec"))
            self.body.addSpacing(4)
            for name, text in report.texts[:6]:
                self.body.addWidget(label(f"{name}:", 10, "ter"))
                self.body.addWidget(label("  " + text.replace("-", MINUS), 11, mono=True, wrap=True))

        if report.url:
            self.body.addSpacing(10)
            row = QHBoxLayout()
            row.addWidget(pill("Bei Wolfram öffnen", "plain",
                               lambda: webbrowser.open(report.url), small=True))
            row.addStretch(1)
            holder = QWidget()
            holder.setLayout(row)
            self.body.addWidget(holder)

    def restyle(self):
        self.card.restyle()
        self.render()

    def shutdown(self):
        self._stop_thread()



# ==========================================================================
# Die Ergebniskarte: Rechenweg und Zeichnung
# ==========================================================================

def show_view(page, index: int):
    """Schaltet die Ergebniskarte um, auch wenn der Nutzer nichts geklickt hat."""
    page.seg_out.set_value(index)
    page.stack_out.setCurrentIndex(index)


class CodeEdit(QPlainTextEdit):
    """Mehrzeiliges Feld im Stil der Eingabefelder; Strg/Cmd + Enter rechnet."""

    submitted = Signal()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and \
                event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier):
            self.submitted.emit()
            return
        if event.key() == Qt.Key_Tab:
            self.insertPlainText("    ")
            return
        super().keyPressEvent(event)


# ==========================================================================
# Eine Seite: Eingabe links, Ergebnis rechts
# ==========================================================================

HISTORY_LIMIT = 10


class AnimationPane(QWidget):
    """
    Die Ansicht »Animation«: rendert den Rechenweg des aktuellen Ergebnisses mit
    manimgx (sonst Manim CE) und spielt das Video ab. Das Rendern läuft in einem
    eigenen Faden, die Engine selbst in einem eigenen Prozess.
    """

    progressed = Signal(int, str)
    finished = Signal(object, str)          # (Pfad oder None, Fehlertext)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result = None
        self.figure_source: Callable | None = None   # liefert die Figur zur Rechnung oder None
        self.path: Path | None = None
        self.busy = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        self.hint = label("", 11, "sec", wrap=True)
        lay.addWidget(self.hint)
        row = QHBoxLayout()
        row.setSpacing(10)
        self.btn = pill("Animation erstellen", "primary", self.start, small=True)
        row.addWidget(self.btn)
        self.btn_open = pill("Im Player öffnen", "plain", self.open_external, small=True)
        self.btn_open.hide()
        row.addWidget(self.btn_open)
        self.status = label("", 11, "ter")
        row.addWidget(self.status)
        row.addStretch(1)
        lay.addLayout(row)
        self.player = None
        self.video = None
        if HAS_VIDEO:
            self.video = QVideoWidget()
            self.video.setMinimumHeight(260)
            self.video.hide()
            self.player = QMediaPlayer(self)
            self.player.setAudioOutput(QAudioOutput(self))
            self.player.setVideoOutput(self.video)
            self.player.setLoops(QMediaPlayer.Loops.Infinite)
            lay.addWidget(self.video, 1)
        else:
            lay.addStretch(1)
        self.progressed.connect(lambda pct, text: self.status.setText(f"{text} {pct} %"))
        self.finished.connect(self._finished)
        self.set_result(None)

    def set_result(self, result):
        """Neues Ergebnis (oder None nach einem Fehler): altes Video verwerfen."""
        self.result, self.path = result, None
        if self.player is not None:
            self.player.stop()
            self.video.hide()
        self.btn_open.hide()
        self.status.setText("")
        ok, why = dk_anim.manim_available() if dk_anim else (False, "Das Modul grafik.anim lässt sich nicht laden.")
        if not ok:
            self.hint.setText(f"{why}\n{dk_anim.INSTALL_HINT if dk_anim else 'pip install manimgx'}")
        elif result is None:
            self.hint.setText("Erst rechnen, dann gibt es einen Rechenweg zum Animieren.")
        else:
            self.hint.setText("Zeigt den Rechenweg Schritt für Schritt als Video, gerendert mit manimgx.")
        self.btn.setEnabled(ok and result is not None and not self.busy)

    def start(self):
        if self.busy or self.result is None or dk_anim is None:
            return
        self.busy = True
        self.btn.setEnabled(False)
        self.status.setText("Animation startet …")
        result, source = self.result, self.figure_source
        theme = dk_anim.Theme.dark() if T["dark"] else dk_anim.Theme()

        def work():
            try:
                # Dieselbe Zeichnung wie im Reiter »Zeichnung«; gibt es keine, bleibt es beim Rechenweg.
                figure = source() if source else None
                job = dk_anim.job_from_result(result, quality="medium", theme=theme, figure=figure)
                self.finished.emit(dk_anim.render(job, progress=self.progressed.emit), "")
            except Exception as exc:                 # Fehlertext zeigen, nie den Faden beenden
                self.finished.emit(None, str(exc))

        threading.Thread(target=work, daemon=True).start()

    def _finished(self, path, error: str):
        self.busy = False
        self.btn.setEnabled(self.result is not None)
        if path is None:
            self.status.setText(error.splitlines()[0] if error else "Das Rendern ist fehlgeschlagen.")
            return
        self.path = Path(path)
        self.status.setText("Fertig.")
        self.btn_open.show()
        if self.player is not None:
            self.video.show()
            self.player.setSource(QUrl.fromLocalFile(str(self.path)))
            self.player.play()
        else:
            self.open_external()

    def open_external(self):
        if self.path is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.path)))

    def restyle(self):
        restyle_labels(self)

    def shutdown(self):
        if self.player is not None:
            self.player.stop()


class ToolPage(QWidget):
    """
    Gemeinsames Gerüst der sechs Seiten.

    Links die Aufgabe (Rechnung wählen, Felder ausfüllen) und die Prüfung mit
    Wolfram|Alpha, rechts die Ergebniskarte. Was eine Seite kann, steht als Daten
    in `PAGES`: die Seite baut ihr Formular aus den Feldlisten selbst.
    """

    def __init__(self, window, key: str, parent=None):
        super().__init__(parent)
        self.setObjectName("page")
        self.KEY = key
        self.spec = PAGES[key]
        self.window_ = window
        self.p: dict = dict(DEFAULTS[key])
        self.decimal = False
        self.result = None
        self.result_p: dict = {}
        self.vm: ResultVM | None = None
        self.stale = False
        self.history: list[tuple[str, dict]] = []
        self.job_id = 0
        self.running = False
        self.rerun = False
        self._jobs: dict = {}
        self.variant = 0
        self.inputs: dict[str, QWidget] = {}
        self.previews: dict[str, QLabel] = {}
        self.figure_job_id = 0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(32, 26, 32, 26)
        outer.setSpacing(0)
        outer.addWidget(label(self.spec.title, 26, weight=QFont.Bold, display=True))
        outer.addWidget(label(self.spec.lede, 13, "sec", display=True, wrap=True))
        outer.addSpacing(18)

        columns = QHBoxLayout()
        columns.setSpacing(20)
        outer.addLayout(columns, 1)

        # Links: eigene Rolle, damit lange Prüfberichte scrollen, ohne die Ergebniskarte zu dehnen.
        self.left_scroll = QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setFrameShape(QFrame.NoFrame)
        self.left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.left_scroll.setFixedWidth(540)
        left_host = QWidget()
        left_host.setObjectName("page")
        self.left_scroll.setWidget(left_host)
        left = QVBoxLayout(left_host)
        left.setContentsMargins(0, 0, 14, 8)
        left.setSpacing(18)
        columns.addWidget(self.left_scroll)
        right = QVBoxLayout()
        right.setSpacing(18)
        columns.addLayout(right, 1)

        self._build_input_card(left)
        self.wolfram = WolframPanel(self._wolfram_run, wolfram_subtitle(self.KEY))
        left.addWidget(self.wolfram)
        left.addStretch(1)
        self._build_output_card(right)
        self.sync_widgets()

    # -- Aufbau -----------------------------------------------------------

    def _build_input_card(self, left: QVBoxLayout):
        self.card_input = Card(pad=24)
        left.addWidget(self.card_input)
        lay = self.card_input.lay
        row = QHBoxLayout()
        row.setSpacing(8)
        self.dd_examples = Dropdown([n for n, _ in EXAMPLES[self.KEY]], placeholder="Beispiele")
        self.dd_examples.picked.connect(self.load_example)
        row.addWidget(self.dd_examples)
        self.dd_history = Dropdown([], placeholder="Verlauf")
        self.dd_history.picked.connect(self.load_history)
        self.dd_history.setEnabled(False)
        row.addWidget(self.dd_history)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addSpacing(16)
        lay.addWidget(label("Rechnung", 11, "sec"))
        lay.addSpacing(4)
        self.dd_action = Dropdown([a.label for a in self.spec.actions], value=self.action().label)
        self.dd_action.picked.connect(self.set_action_label)
        lay.addWidget(self.dd_action)
        lay.addSpacing(6)
        self.lbl_action = label("", 11, "sec", wrap=True)
        lay.addWidget(self.lbl_action)
        lay.addSpacing(14)
        self.form = QVBoxLayout()
        self.form.setSpacing(10)
        lay.addLayout(self.form)
        lay.addSpacing(16)
        row = QHBoxLayout()
        row.setSpacing(10)
        self.btn_compute = pill("Rechnen", "primary", self.compute)
        row.addWidget(self.btn_compute)
        self.seg_fmt = Segmented(["Exakt", "Dezimal"], 0)
        self.seg_fmt.changed.connect(self.set_format)
        row.addWidget(self.seg_fmt)
        row.addStretch(1)
        lay.addLayout(row)
        self.lbl_err = label("", 11, "red", wrap=True)
        self.lbl_err.hide()
        lay.addSpacing(6)
        lay.addWidget(self.lbl_err)

    def _build_output_card(self, right: QVBoxLayout):
        self.card_out = Card()
        right.addWidget(self.card_out, 1)
        head = QHBoxLayout()
        head.setSpacing(10)
        self.seg_out = Segmented(["Rechenweg", "Zeichnung", "Animation"], 0, seg_width=104)
        self.seg_out.changed.connect(lambda i: self.stack_out.setCurrentIndex(i))
        head.addWidget(self.seg_out)
        self.lbl_status = label("", 11, "ter")
        head.addWidget(self.lbl_status)
        head.addStretch(1)
        self.btn_export = pill("Exportieren  ⌄", "secondary", self.open_export_menu, small=True)
        head.addWidget(self.btn_export)
        self.card_out.lay.addLayout(head)
        self.card_out.lay.addSpacing(12)
        self.stack_out = QStackedWidget()
        # Mit Auswahl, nicht pauschal: eine Regel ohne Auswahl vererbt sich auf
        # alle Kinder und nähme den Knöpfen darin ihre Farbe.
        self.stack_out.setObjectName("outstack")
        self.stack_out.setStyleSheet("#outstack { background: transparent; }")
        self.view = ResultView()
        self.plot = PlotPane()
        self.plot.variant_changed.connect(self.set_variant)
        self.plot.browser_requested.connect(self.open_in_browser)
        self.stack_out.addWidget(self.view)
        self.stack_out.addWidget(self.plot)
        self.anim = AnimationPane()
        self.anim.figure_source = self._anim_figure
        self.stack_out.addWidget(self.anim)
        self.card_out.lay.addWidget(self.stack_out, 1)

    # -- Felder -----------------------------------------------------------

    def action(self) -> Action:
        try:
            return action_of(self.KEY, self.p.get("action", ""))
        except InputError:
            return self.spec.actions[0]

    def clear_form(self):
        self.inputs = {}
        self.previews = {}
        self._drop_layout(self.form)

    def _drop_layout(self, lay):
        while lay.count():
            item = lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()
            elif item.layout() is not None:
                self._drop_layout(item.layout())

    def build_form(self):
        """Baut die Felder zur gewählten Rechnung auf; kurze Felder stehen nebeneinander."""
        row: QHBoxLayout | None = None
        for fld in self.action().fields:
            if fld.kind == "choice":
                row = None
                self._add_choice(fld)
            elif fld.kind == "preset":
                row = None
                self._add_preset(fld)
            elif fld.w:
                if row is None:
                    row = QHBoxLayout()
                    row.setSpacing(12)
                    self.form.addLayout(row)
                row.addLayout(self._field_box(fld))
                row.addStretch(1)
                row.setStretch(row.count() - 1, 0)
            else:
                row = None
                self.form.addLayout(self._field_box(fld))

    def _field_box(self, fld: Field) -> QVBoxLayout:
        box = QVBoxLayout()
        box.setSpacing(4)
        box.addWidget(label(fld.caption, 11, "sec"))
        text = self.p.get(fld.key, "")
        if fld.kind in ("lines", "code"):
            ed = CodeEdit()
            ed.setObjectName("cell")
            ed.setFont(FONTS.f(12, mono=True))
            ed.setPlainText(text)
            ed.setLineWrapMode(QPlainTextEdit.NoWrap if fld.kind == "code" else QPlainTextEdit.WidgetWidth)
            ed.setFixedHeight(int(QFontMetricsF(ed.font()).lineSpacing() * fld.rows) + 18)
            ed.setPlaceholderText(fld.ph)
            ed.textChanged.connect(lambda k=fld.key, e=ed: self.set_param(k, e.toPlainText()))
            ed.submitted.connect(self.compute)
        else:
            ed = entry(text, width=fld.w or None, placeholder=fld.ph, size=13)
            ed.textChanged.connect(lambda t, k=fld.key: self.set_param(k, t))
            ed.returnPressed.connect(self.compute)
        self.inputs[fld.key] = ed
        box.addWidget(ed)
        if fld.pv:
            pv = label("", 10, "ter", wrap=True)
            pv.hide()
            self.previews[fld.key] = pv
            box.addWidget(pv)
            self.update_preview(fld.key)
        if fld.note:
            box.addWidget(label(fld.note, 10, "ter", wrap=True))
        return box

    def _add_choice(self, fld: Field):
        box = QVBoxLayout()
        box.setSpacing(4)
        box.addWidget(label(fld.caption, 11, "sec"))
        values = [v for v, _ in fld.options]
        current = self.p.get(fld.key, fld.default)
        seg = Segmented([t for _, t in fld.options], values.index(current) if current in values else 0)
        seg.changed.connect(lambda i, k=fld.key, vs=values: self.set_param(k, vs[i]))
        box.addWidget(seg)
        holder = QHBoxLayout()
        holder.addLayout(box)
        holder.addStretch(1)
        self.form.addLayout(holder)
        self.inputs[fld.key] = seg

    def _add_preset(self, fld: Field):
        box = QVBoxLayout()
        box.setSpacing(4)
        box.addWidget(label(fld.caption, 11, "sec"))
        dd = Dropdown([n for n, _ in fld.options], placeholder="Auswählen …")
        dd.picked.connect(lambda name, f=fld: self.apply_preset(f, name))
        box.addWidget(dd)
        holder = QHBoxLayout()
        holder.addLayout(box)
        holder.addStretch(1)
        self.form.addLayout(holder)

    def apply_preset(self, fld: Field, name: str):
        for n, values in fld.options:
            if n == name:
                self.p.update(values)
                self.sync_widgets()
                self.touch()
                return

    def _preview_text(self, fld_key: str) -> tuple[str, bool]:
        text = (self.p.get(fld_key) or "").strip()
        if not text:
            return "", False
        try:
            return "= " + dk.parse(text).to_str("unicode"), False
        except Exception as exc:                # der Kern sagt auf Deutsch, was nicht stimmt
            msg = friendly_error(exc)
            return (msg if len(msg) < 150 else msg[:147] + "…"), True

    def update_preview(self, key: str):
        pv = self.previews.get(key)
        if pv is None:
            return
        text, bad = self._preview_text(key)
        pv.setText(text)
        pv.setProperty("colorKey", "red" if bad else "ter")
        pv.setStyleSheet(f"color: {T['red' if bad else 'ter']}; background: transparent;")
        pv.setVisible(bool(text))

    # -- Zustand ----------------------------------------------------------

    def set_param(self, key: str, value: str):
        if self.p.get(key) == value:
            return
        self.p[key] = value
        self.update_preview(key)
        self.touch()

    def touch(self):
        """Die Eingabe hat sich geändert: Ergebnis und Prüfbericht gelten nicht mehr."""
        if self.result is not None and not self.stale:
            self.stale = True
            self.update_status()
        self.wolfram.invalidate()

    def update_status(self, text: str | None = None):
        if text is not None:
            self.lbl_status.setText(text)
        elif self.running:
            self.lbl_status.setText("Rechne …")
        elif self.stale:
            self.lbl_status.setText("Eingabe geändert — neu rechnen")
        else:
            self.lbl_status.setText("")
        key = "amber" if self.stale and not self.running else "ter"
        self.lbl_status.setProperty("colorKey", key)
        self.lbl_status.setStyleSheet(f"color: {T[key]}; background: transparent;")

    def sync_widgets(self):
        """Zeigt den Zustand von self.p in den Widgets (nach Beispiel, Verlauf oder Wechsel der Rechnung)."""
        act = self.action()
        self.dd_action.set_value(act.label)
        self.lbl_action.setText(act.desc)
        self.clear_form()
        self.build_form()
        self.seg_fmt.setVisible(act.decimal)
        if not act.decimal and self.decimal:
            self.decimal = False
            self.seg_fmt.set_value(0)
        self.wolfram.set_unavailable(wolfram_support(self.KEY, self.p))
        self.set_err("")

    def set_action_label(self, name: str):
        for act in self.spec.actions:
            if act.label == name:
                self.p["action"] = act.key
                self.sync_widgets()
                self.touch()
                return

    def set_format(self, index: int):
        self.decimal = bool(index)
        if self.result is not None:
            self.compute()

    def set_variant(self, index: int):
        """Eine andere Zeichnung zur selben Rechnung."""
        self.variant = index
        if self.result is not None:
            self.draw_variant()

    def set_err(self, text: str):
        self.lbl_err.setText(text)
        self.lbl_err.setVisible(bool(text))

    # -- Beispiele und Verlauf --------------------------------------------

    def load_example(self, name: str):
        for n, params in EXAMPLES[self.KEY]:
            if n == name:
                self.p = dict(DEFAULTS[self.KEY])
                self.p.update(params)
                self.variant = 0
                self.sync_widgets()
                self.compute()
                return

    def describe(self) -> str:
        act = self.action()
        for fld in act.fields:
            if fld.kind in ("line", "lines", "code") and fld.key != "code":
                first = (self.result_p.get(fld.key) or "").strip().splitlines()
                if first:
                    text = first[0]
                    return f"{act.label}: {text if len(text) < 38 else text[:36] + '…'}"
        return act.label

    def remember(self):
        text = self.describe()
        self.history = [(t, q) for t, q in self.history if q != self.result_p]
        self.history.insert(0, (text, dict(self.result_p)))
        del self.history[HISTORY_LIMIT:]
        self.dd_history.set_options([f"{i + 1}.  {t}" for i, (t, _) in enumerate(self.history)])
        self.dd_history.setEnabled(True)

    def load_history(self, name: str):
        index = int(name.split(".")[0]) - 1
        self.p = dict(self.history[index][1])
        self.sync_widgets()
        self.compute()

    # -- Rechnen ----------------------------------------------------------

    def params(self) -> dict:
        return dict(self.p)

    def compute(self):
        """Startet die Rechnung im Hintergrund; das Fenster bleibt bedienbar."""
        if self.running:
            self.rerun = True
            return
        self.job_id += 1
        job = self.job_id
        p, dark, variant = self.params(), T["dark"], self.variant
        decimal = self.decimal and self.action().decimal
        self.running = True
        self.btn_compute.setEnabled(False)
        self.btn_compute.setText("Rechne …")
        self.update_status()
        if self.result is None:
            self.view.busy("Rechne …")
        self.start_worker(job, lambda: run_page_job(self.KEY, p, decimal, dark, variant),
                          self._computed, (p, decimal))

    def start_worker(self, job: int, run, handler, extra=()):
        """
        Startet `run` in einem eigenen Faden. Das Ergebnis kommt über ein Signal
        an eine Methode dieser Seite zurück, damit `handler` im Fenster-Faden
        läuft — eine blosse Funktion als Empfänger liefe im Hintergrundfaden.
        """
        thread = QThread(self)
        worker = ComputeWorker(job, run)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._job_finished)
        self._jobs[job] = (thread, worker, handler, extra)
        thread.start()

    def _job_finished(self, job_id: int, outcome):
        entry_ = self._jobs.pop(job_id, None)
        if entry_ is None:
            return
        thread, worker, handler, extra = entry_
        thread.quit()
        thread.wait()
        worker.deleteLater()
        thread.deleteLater()
        handler(job_id, outcome, *extra)

    def _computed(self, job: int, outcome: Outcome, p: dict, decimal: bool):
        self.running = False
        self.btn_compute.setEnabled(True)
        self.btn_compute.setText("Rechnen")
        if outcome.error:
            self.result, self.vm = None, None
            self.stale = False
            self.view.show_error(outcome.error)
            self.plot.show_message("Keine Zeichnung — die Rechnung ist nicht gelungen.")
            self.plot.allow_browser(False)
            self.anim.set_result(None)
            self.wolfram.set_unavailable("")
            self.update_status("")
            show_view(self, 0)
        else:
            self.result, self.vm = outcome.result, outcome.vm
            self.result_p = dict(p)
            self.stale = False
            self.view.show_vm(outcome.vm)
            self._set_plot(outcome)
            self.anim.set_result(outcome.result)
            self.wolfram.set_unavailable(wolfram_support(self.KEY, p, outcome.result))
            self.remember()
            self.update_status("")
            self.lbl_status.setText(f"{outcome.seconds:.1f} s" if outcome.seconds >= 0.3 else "")
        self.wolfram.invalidate()
        if self.rerun:
            self.rerun = False
            self.compute()

    def wait(self, ms: int = 60000):
        """Wartet, bis keine Rechnung mehr läuft (für Tests)."""
        import time
        end = time.time() + ms / 1000
        while (self.running or self._jobs) and time.time() < end:
            QApplication.processEvents()
            time.sleep(0.005)
        QApplication.processEvents()

    def _set_plot(self, outcome: Outcome):
        names = figure_names(self.KEY, self.result_p)
        self.plot.set_variants(names, min(self.variant, len(names) - 1))
        self.plot.allow_browser(self.result_p.get("action") in ("growth", "maxn"))
        if outcome.figure is not None:
            self.plot.show_figure(outcome.figure)
        else:
            self.plot.show_message(outcome.figure_error or NO_FIGURE)

    def _anim_figure(self):
        """Die Figur des Reiters »Zeichnung« für die Animation; None, wenn es keine gibt."""
        if self.result is None:
            return None
        try:
            with PLOT_LOCK:
                return make_figure(self.KEY, self.result_p, self.result, T["dark"], 0)
        except Exception:
            return None

    def draw_variant(self):
        """Zeichnung neu bauen (Themawechsel, andere Ansicht) — im Hintergrund."""
        if self.result is None:
            return
        self.figure_job_id += 1
        job = self.figure_job_id
        page, p, result = self.KEY, dict(self.result_p), self.result
        dark, variant = T["dark"], self.variant

        def handler(job_id, outcome):
            if job_id != self.figure_job_id or self.result is not result:
                return
            self._set_plot(outcome)

        self.start_worker(job, lambda: run_figure_job(page, p, result, dark, variant), handler)

    # -- Prüfung mit Wolfram|Alpha ----------------------------------------

    def _wolfram_run(self, appid: str):
        """Gibt die Prüfung als Aufgabe zurück; ausgeführt wird sie im Faden."""
        if self.result is None:
            raise ValueError("Erst rechnen — die Prüfung vergleicht das Ergebnis der Rechnung.")
        if self.stale:
            raise ValueError("Die Eingabe hat sich seit der Rechnung geändert. Erst neu rechnen, "
                             "dann prüfen.")
        why = wolfram_support(self.KEY, self.result_p, self.result)
        if why:
            raise ValueError(why)
        page, p, result = self.KEY, dict(self.result_p), self.result
        return lambda: run_wolfram_checks(appid, page, p, result)

    # -- Exportieren ------------------------------------------------------

    def open_export_menu(self):
        menu = QMenu(self)
        menu.setFont(FONTS.f(11))
        menu.addAction("Rechenweg als LaTeX kopieren", self.copy_latex)
        menu.addAction("Rechenweg als Text kopieren", self.copy_text)
        menu.addSeparator()
        menu.addAction("Zeichnung speichern …", self.save_figure)
        menu.addAction("Zeichnung kopieren", self.copy_figure)
        menu.exec(self.btn_export.mapToGlobal(QPoint(0, self.btn_export.height() + 2)))

    def toast(self, text: str, kind: str = "info"):
        self.window_.toasts.show(text, kind)

    def latex_text(self) -> str:
        return self.result.latex() if self.result is not None else ""

    def copy_latex(self):
        if self.result is None:
            self.toast("Erst rechnen.", "bad")
            return
        QGuiApplication.clipboard().setText(self.latex_text())
        self.toast("LaTeX kopiert", "ok")

    def copy_text(self):
        if self.result is None:
            self.toast("Erst rechnen.", "bad")
            return
        QGuiApplication.clipboard().setText(repr(self.result))
        self.toast("Rechenweg als Text kopiert", "ok")

    def save_figure(self):
        figure = self.plot.figure
        if figure is None:
            self.toast("Es gibt keine Zeichnung zu speichern.", "bad")
            return
        try:
            path = save_figure_dialog(self, figure, f"dmatkit-{self.KEY}")
        except Exception as exc:
            self.toast(f"Speichern ging nicht: {exc}", "bad")
            return
        if path:
            self.toast(f"Gespeichert: {Path(path).name}", "ok")

    def copy_figure(self):
        canvas = self.plot.canvas.canvas
        if canvas is None:
            self.toast("Es gibt keine Zeichnung zu kopieren.", "bad")
            return
        QGuiApplication.clipboard().setPixmap(canvas.grab())
        self.toast("Zeichnung kopiert", "ok")

    def open_in_browser(self):
        """Dieselben Kurven mit Plotly, mit Zoom und Beschriftung beim Überfahren."""
        try:
            funcs, n_max, limit = growth_args(self.result_p, self.result)
            figure = dk.plotly_growth(funcs, n_max=n_max, log=True, mark_limit=limit, dark=T["dark"])
            path = Path(tempfile.mkdtemp(prefix="dmatkit-plot-")) / "plot.html"
            figure.write_html(str(path), include_plotlyjs="cdn")
            webbrowser.open(path.as_uri())
        except Exception as exc:
            self.toast(f"Browser-Zeichnung ging nicht: {friendly_error(exc)}", "bad")

    # -- Thema ------------------------------------------------------------

    def restyle(self):
        self.card_input.restyle()
        self.card_out.restyle()
        self.view.restyle()
        self.plot.restyle()
        self.anim.restyle()
        self.wolfram.restyle()
        self.update_status()
        for key in self.previews:
            self.update_preview(key)
        if self.result is not None:
            self.draw_variant()

    def shutdown(self):
        self.anim.shutdown()
        self.wolfram.shutdown()
        for thread, _worker, _h, _e in list(self._jobs.values()):
            thread.quit()
            thread.wait(3000)
        self._jobs.clear()


# ==========================================================================
# Hilfe
# ==========================================================================

class HelpDialog(QDialog):
    """Kurze Erklärung der Schreibweise."""

    SECTIONS = [
        ("Formeln eingeben", "(A or B) and not C",
         "and, or, not, xor oder ∧ ∨ ¬ ⊕; Implikation mit ->, Äquivalenz mit <->. "
         "Vorrang: not vor and vor xor vor or vor -> vor <->. Variablen sind Buchstaben wie A, B, P, Q."),
        ("Prädikate und Universum", "x % 2 == 0   über   range(1, 10)",
         "Das Prädikat ist ein Python-Ausdruck mit Vergleichen und Rechnen (nie ein beliebiger Code). "
         "Universum: range(1, 10), 1..9, {1, 2, 3}, N, Z. Bei zwei Variablen »x: range(1, 4); y: {1, 2}«."),
        ("Zahlen und Zahlsysteme", "2.10(365)   0.00(10)   FF",
         "Perioden stehen in Klammern. Binärzahlen mit Komma sind erlaubt, Ziffern bis Basis 36 (A bis Z). "
         "Mengen schreibt man {a, b, c}."),
        ("Code für Schrittzählung", "def f(n): …",
         "Erlaubt ist eine Teilmenge von Python (def, for, while, if, Rechnen, Listen). Eingaben als "
         "Name=Wert, z. B. Liste=[3, 1, 2], Element=2. Strg/Cmd + Enter rechnet auch aus dem Codefeld."),
        ("Exakt oder Dezimal", "1/3 ↔ 0.3333",
         "Gerechnet wird immer exakt. Wo es etwas ändert, zeigt die Seite den Schalter »Dezimal«; "
         "er rundet nur die Anzeige."),
    ]

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Hilfe")
        self.setModal(True)
        self.setFixedWidth(560)
        self.setStyleSheet(parent.styleSheet())
        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 24, 28, 22)
        lay.setSpacing(12)
        lay.addWidget(label("So gibst du etwas ein", 15, weight=QFont.DemiBold))
        for title, example, text in self.SECTIONS:
            box = QFrame()
            box.setObjectName("helpbox")
            box.setStyleSheet(f"#helpbox {{ background: {T['fill2']}; border-radius: 0px; }}")
            bl = QVBoxLayout(box)
            bl.setContentsMargins(16, 13, 16, 13)
            bl.setSpacing(4)
            row = QHBoxLayout()
            row.addWidget(label(title, 12, weight=QFont.DemiBold))
            row.addStretch(1)
            row.addWidget(label(example, 11, "blue", mono=True))
            bl.addLayout(row)
            bl.addWidget(label(text, 11, "sec", wrap=True))
            lay.addWidget(box)
        lay.addSpacing(4)
        lay.addWidget(label("Prüfen mit Wolfram|Alpha", 12, weight=QFont.DemiBold))
        lay.addWidget(label("Wo es sich robust vergleichen lässt (Zahlsysteme, Wahrheitstabellen, grösstes n), prüft die Seite gegen Wolfram|Alpha. Das ist "
                            "freiwillig und braucht eine kostenlose AppID. Ohne Netz bleibt "
                            "die Rechnung selbst unberührt, nur die Gegenprobe meldet, was fehlt.",
                            11, "sec", wrap=True))
        lay.addSpacing(8)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(pill("Verstanden", "primary", self.accept, small=True))
        lay.addLayout(row)


# ==========================================================================
# Hauptfenster
# ==========================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("dmatkit — Diskrete Mathematik mit Rechenweg")
        self.resize(1320, 900)
        self.setMinimumSize(1040, 600)
        self._build()
        self.toasts = ToastArea(self.centralWidget())
        self.apply_theme()

    # -- Aufbau -----------------------------------------------------------

    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.nav = QWidget()
        self.nav.setObjectName("nav")
        self.nav.setFixedHeight(56)
        nav_lay = QHBoxLayout(self.nav)
        nav_lay.setContentsMargins(32, 0, 32, 0)
        self.brand = label("dmatkit", 13, weight=QFont.DemiBold)
        nav_lay.addWidget(self.brand)
        nav_lay.addSpacing(18)
        self.seg_page = Segmented([PAGES[k].tab for k in PAGE_KEYS], 0, pad=24)
        self.seg_page.changed.connect(self.set_page)
        nav_lay.addWidget(self.seg_page)
        nav_lay.addStretch(1)
        self.seg_theme = Segmented(["Hell", "Dunkel"], 0)
        self.seg_theme.changed.connect(self.set_theme)
        nav_lay.addWidget(self.seg_theme)
        nav_lay.addSpacing(6)
        nav_lay.addWidget(pill("Hilfe", "plain", self.show_help, small=True))
        outer.addWidget(self.nav)

        self.hairline = QFrame()
        self.hairline.setFixedHeight(1)
        outer.addWidget(self.hairline)

        self.stack = QStackedWidget()
        outer.addWidget(self.stack, 1)
        self.pages = [ToolPage(self, k) for k in PAGE_KEYS]
        for page in self.pages:
            self.stack.addWidget(page)

    # -- Aussehen ---------------------------------------------------------

    def apply_theme(self):
        sheet = f"""
        QWidget {{ color: {T['text']}; }}
        QMainWindow, #page, QStackedWidget {{ background: {T['bg']}; }}
        QScrollArea {{ background: transparent; }}
        #nav {{ background: {T['nav']}; }}
        #card {{ background: {T['card']}; border-radius: 0px;
                 border: {'0px' if T['dark'] else '1px'} solid {T['hair']}; }}
        QPushButton#primary {{ background: {T['blue']}; color: #ffffff;
            border: none; border-radius: 0px; padding: 0 22px; }}
        QPushButton#primary:hover {{ background: {T['blue_hi']}; }}
        QPushButton#primary:disabled {{ background: {T['fill']};
            color: {T['ter']}; }}
        QPushButton#secondary {{ background: {T['fill']}; color: {T['text']};
            border: none; border-radius: 0px; padding: 0 18px; }}
        QPushButton#secondary:hover {{ background: {T['line']}; }}
        QPushButton#secondary:disabled {{ color: {T['ter']}; }}
        QPushButton#plain {{ background: transparent; color: {T['blue']};
            border: none; padding: 0 10px; }}
        QPushButton#plain:hover {{ text-decoration: underline; }}
        QPushButton#plain:disabled {{ color: {T['ter']}; }}
        QPushButton#dropdown {{ background: {T['fill']}; color: {T['text']};
            border: none; border-radius: 0px; padding: 0 10px; text-align: left; }}
        QPushButton#dropdown:hover {{ background: {T['line']}; }}
        QPushButton#dropdown:disabled {{ color: {T['ter']}; }}
        QLineEdit#cell, QPlainTextEdit#cell {{ background: {T['entry']}; color: {T['text']};
            border: 1px solid {T['line']}; border-radius: 0px; padding: 0 10px;
            selection-background-color: {T['blue']}; }}
        QLineEdit#cell:focus, QPlainTextEdit#cell:focus {{ border: 2px solid {T['blue']}; }}
        QMenu {{ background: {T['card']}; color: {T['text']};
            border: 1px solid {T['line']}; border-radius: 0px; padding: 5px; }}
        QMenu::item {{ padding: 6px 22px; border-radius: 0px; }}
        QMenu::item:selected {{ background: {T['blue']}; color: #ffffff; }}
        QMenu::separator {{ height: 1px; background: {T['line']}; margin: 4px 8px; }}
        QToolTip {{ background: {T['card']}; color: {T['text']};
            border: 1px solid {T['line']}; padding: 4px 8px; }}
        QScrollBar:vertical {{ background: transparent; width: 12px; margin: 2px; }}
        QScrollBar::handle:vertical {{ background: {T['scroll']};
            border-radius: 0px; min-height: 40px; }}
        QScrollBar::handle:vertical:hover {{ background: {T['ter']}; }}
        QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
        QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
        QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 2px; }}
        QScrollBar::handle:horizontal {{ background: {T['scroll']};
            border-radius: 0px; min-width: 40px; }}
        QDialog {{ background: {T['card']}; }}
        QProgressBar#progress {{ background: {T['fill']}; border: none;
            border-radius: 0px; }}
        QProgressBar#progress::chunk {{ background: {T['blue']};
            border-radius: 0px; }}
        """
        self.setStyleSheet(sheet)
        self.hairline.setStyleSheet(f"background: {T['line']};")
        for page in getattr(self, "pages", []):
            page.restyle()
        restyle_labels(self)

    def set_theme(self, index: int):
        T.clear()
        T.update(DARK if index else LIGHT)
        self.apply_theme()
        for widget in self.findChildren(QWidget):
            widget.update()

    def set_page(self, index: int):
        self.stack.setCurrentIndex(index)
        self.seg_page.set_value(index)

    @property
    def page(self) -> ToolPage:
        return self.pages[self.stack.currentIndex()]

    def show_help(self):
        HelpDialog(self).exec()

    # -- Tastatur ---------------------------------------------------------

    def keyPressEvent(self, event):
        mods = event.modifiers()
        if mods & (Qt.ControlModifier | Qt.MetaModifier):
            if Qt.Key_1 <= event.key() <= Qt.Key_6:
                self.set_page(event.key() - Qt.Key_1)
                return
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self.page.compute()
                return
        super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "toasts"):
            self.toasts.reposition()

    def closeEvent(self, event):
        """Hintergrundläufe beenden und temporäre Renderdateien aufräumen."""
        for page in self.pages:
            page.shutdown()
        super().closeEvent(event)


def _prewarm_matplotlib():
    try:
        import matplotlib.pyplot  # noqa: F401
        from matplotlib.backends.backend_qtagg import FigureCanvas  # noqa: F401
        render_mask(r"\frac{a}{b}", 14)         # Mathtext und Schriften einmal laden
    except Exception:
        pass            # fehlt Matplotlib, meldet »Zeichnung« das selbst


def create_app(argv: Sequence[str] | None = None):
    """Legt QApplication, Schriften und Fenster an; gibt (app, window) zurück."""
    global FONTS
    app = QApplication.instance() or QApplication(list(argv if argv is not None else sys.argv))
    app.setApplicationName("dmatkit")
    FONTS = Fonts()
    setup_math_sizes()
    window = MainWindow()
    return app, window


def main():
    app, window = create_app()
    window.show()
    # Matplotlib im Hintergrund vorladen, damit die erste Rechnung nicht stockt.
    threading.Thread(target=_prewarm_matplotlib, daemon=True).start()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
