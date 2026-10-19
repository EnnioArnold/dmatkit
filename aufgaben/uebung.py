#!/usr/bin/env python3
"""
Selbst programmieren: Gerüste und Prüfstände für die Algorithmen der Serien 1 bis 5.

Wie bei lialkit steht hier keine Lösung. Du schreibst die Funktion selbst, rufst `pruefe_...` damit auf
und siehst sofort, ob sie an den Beispielen der Serien und an den Randfällen besteht, die gern
übersehen werden. Die Meldungen sagen auf Deutsch, was nicht stimmt.

    from dmatkit.aufgaben.uebung import pruefe_dec_to_bin, vorlage

    print(vorlage("meine_dec_to_bin"))      # Gerüst zum Abschreiben

    def meine_dec_to_bin(n):
        ...

    pruefe_dec_to_bin(meine_dec_to_bin)

Schutz: Jede Funktion läuft mit Zeitlimit (Endlosschleifen werden abgebrochen), Ausnahmen werden
abgefangen und gemeldet, falsche Rückgabetypen werden erkannt. Eingaben werden als Kopie übergeben und
danach auf ungewollte Veränderung geprüft.
"""
from __future__ import annotations

import contextlib
import copy
import ctypes
import html as _html
import inspect
import io
import itertools
import random
import signal
import threading
import time
from fractions import Fraction
from typing import Any, Callable

ZEITLIMIT = 2.0        # Sekunden pro Aufruf

__all__ = [
    "VORLAGEN", "vorlage", "namen", "Fall", "Pruefung",
    "pruefe_dec_to_bin", "pruefe_binaer_zu_dezimal", "pruefe_forall", "pruefe_exists", "pruefe_hat_duplikate",
    "pruefe_palindrom", "pruefe_wahrheitstabelle", "pruefe_folgerung", "pruefe_resolvente",
    "pruefe_lineare_suche", "pruefe_selection_sort",
    "pruefe_klassifikation", "pruefe_potenzmenge", "pruefe_dezimal_zu_bruch", "pruefe_binaer_nachkomma",
]


# ==========================================================================
# Gerüste (keine Lösungen)
# ==========================================================================
VORLAGEN: dict[str, str] = {
    "meine_dec_to_bin": '''
def meine_dec_to_bin(n):
    """Binärdarstellung der natürlichen Zahl n als Text, z. B. 25 -> '11001'."""
    y = ''
    temp = n
    while temp != 0:
        # TODO: letzte Stelle (Rest bei Division durch 2) vorne anhängen
        # TODO: temp ganzzahlig durch 2 teilen
        pass
    return y
''',
    "meine_binaer_zu_dezimal": '''
def meine_binaer_zu_dezimal(text):
    """Wert einer Binärzahl als Text, z. B. '110011001' -> 409."""
    wert = 0
    for zeichen in text:
        # TODO: bisherigen Wert verdoppeln und die neue Stelle dazuzählen
        pass
    return wert
''',
    "mein_forall": '''
def mein_forall(universum, praedikat):
    """True, wenn praedikat(x) für ALLE x im Universum gilt (∀)."""
    for x in universum:
        # TODO: beim ersten Gegenbeispiel sofort entscheiden
        pass
    # TODO: was gilt, wenn kein Gegenbeispiel gefunden wurde (auch beim leeren Universum)?
''',
    "mein_exists": '''
def mein_exists(universum, praedikat):
    """True, wenn praedikat(x) für MINDESTENS ein x im Universum gilt (∃)."""
    for x in universum:
        # TODO: beim ersten Treffer sofort entscheiden
        pass
    # TODO: was gilt, wenn kein Treffer gefunden wurde?
''',
    "meine_hat_duplikate": '''
def meine_hat_duplikate(a):
    """True, wenn in der Liste a ein Wert an zwei verschiedenen Positionen steht:
    ∃i ∃j: ¬(i = j) ∧ a[i] = a[j]"""
    n = len(a)
    for i in range(n):
        for j in range(n):
            # TODO: Positionen vergleichen UND Werte vergleichen
            pass
    return False
''',
    "mein_palindrom": '''
def mein_palindrom(s):
    """True, wenn der Text vorwärts und rückwärts gleich gelesen wird, z. B. 'REITTIER'."""
    # TODO: Zeichen von vorne und von hinten paarweise vergleichen
    pass
''',
    "meine_wahrheitstabelle": '''
def meine_wahrheitstabelle(f_func, n_vars):
    """Liste aller Zeilen (belegung, wert) mit belegung = Tupel aus False/True der Länge n_vars
    und wert = f_func(*belegung). Es müssen 2**n_vars Zeilen sein."""
    zeilen = []
    # TODO: alle Belegungen erzeugen (Tipp: itertools.product([False, True], repeat=n_vars)
    #       oder die Zahlen 0 ... 2**n_vars - 1 als Binärzahl lesen)
    return zeilen
''',
    "meine_folgerung": '''
def meine_folgerung(wb_func, q_func, n_vars):
    """True, wenn WB ⊨ Q: in JEDER Belegung, in der wb_func(*belegung) wahr ist, ist auch q_func(*belegung) wahr."""
    # TODO: alle Belegungen durchgehen; Zeilen mit WB = 0 zählen nicht
    pass
''',
    "meine_resolvente": '''
def meine_resolvente(k1, k2):
    """Alle Resolventen der Klauseln k1 und k2 (Mengen von Texten wie 'A' und '¬B') als Liste von Mengen.
    Pro komplementärem Paar (L in k1, ¬L in k2) eine Resolvente: beide Literale streichen, den Rest vereinigen.
    Die Eingaben dürfen nicht verändert werden. Die leere Klausel ist set()."""
    ergebnis = []
    for lit in k1:
        # TODO: Gegenstück von lit bestimmen ('A' <-> '¬A') und in k2 suchen
        pass
    return ergebnis
''',
    "meine_lineare_suche": '''
def meine_lineare_suche(a, x):
    """Lineare Suche mit Schrittzähler nach den Zählregeln der Vorlesung.
    Gib (gefunden, schritte) zurück. Es zählen: jeder for-Durchlauf (1), jeder Vergleich (1), jedes return (1).
    len() und Listenzugriff zählen nicht."""
    schritte = 0
    for i in range(len(a)):
        # TODO: Schritte mitzählen
        pass
    # TODO: Rückgabe
''',
    "mein_selection_sort": '''
def mein_selection_sort(a):
    """Sortiert die Liste a aufsteigend mit Selection Sort und gibt sie zurück."""
    n = len(a)
    for i in range(n):
        min_index = i
        for j in range(i + 1, n):
            # TODO: kleineres Element merken
            pass
        # TODO: Elemente an Position i und min_index tauschen
    return a
''',
    "meine_klassifikation": '''
def meine_klassifikation(f_func, n_vars):
    """'Tautologie', 'Kontradiktion' oder 'Kontingenz' (erfüllbar, aber nicht immer wahr)."""
    # TODO: alle Belegungen durchgehen und Wahr/Falsch zählen
    pass
''',
    "meine_potenzmenge": '''
def meine_potenzmenge(menge):
    """Alle Teilmengen der Menge als Liste von Mengen, inklusive ∅ und der Menge selbst (2**n Stück)."""
    teilmengen = [set()]
    for element in menge:
        # TODO: zu jeder bisherigen Teilmenge eine Kopie mit dem neuen Element hinzufügen
        pass
    return teilmengen
''',
    "meine_dezimal_zu_bruch": '''
def meine_dezimal_zu_bruch(text):
    """Periodische Dezimalzahl als Text, z. B. '2.10(365)', in einen gekürzten Bruch (Fraction) umrechnen."""
    from fractions import Fraction
    # TODO: Ganzzahlteil, Vorperiode und Periode trennen
    # TODO: 10er-Potenzen so wählen, dass sich die Periode beim Subtrahieren wegkürzt
    pass
''',
    "meine_binaer_nachkomma": '''
def meine_binaer_nachkomma(zaehler, nenner):
    """Binärdarstellung von zaehler/nenner als Text mit Periode in Klammern, z. B. 1/10 -> '0.0(0011)', 3/8 -> '0.011'."""
    # TODO: Ganzzahlteil binär; dann den Rest wiederholt mit 2 multiplizieren, die Ziffer ist der ganzzahlige Teil.
    # TODO: Wiederholt sich ein Rest, beginnt die Periode.
    pass
''',
}
VORLAGEN = {k: v.strip("\n") for k, v in VORLAGEN.items()}


def namen() -> list[str]:
    """Alle Funktionsnamen, für die es ein Gerüst gibt."""
    return list(VORLAGEN)


def vorlage(name: str | None = None) -> str:
    """Gerüst zum Abschreiben. Ohne Namen: alle Gerüste hintereinander."""
    if name is None:
        return "\n\n\n".join(VORLAGEN.values())
    if name not in VORLAGEN:
        raise ValueError(f"Kein Gerüst namens {name!r}. Vorhanden: {', '.join(VORLAGEN)}")
    return VORLAGEN[name]


# ==========================================================================
# Ausführung mit Schutz
# ==========================================================================
class _Zeit(BaseException):
    """Zeitlimit überschritten (BaseException, damit ein `except Exception` im Schülercode sie nicht verschluckt)."""


def _in_hauptthread() -> bool:
    return threading.current_thread() is threading.main_thread() and hasattr(signal, "setitimer")


def _mit_zeitlimit(fn: Callable, args: tuple, limit: float):
    """Gibt (status, wert) zurück: ('ok', Wert) | ('fehler', Ausnahme) | ('zeit', None)."""
    puffer = io.StringIO()
    if _in_hauptthread():
        def _alarm(signum, frame):
            raise _Zeit()
        alt = signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, limit)
        try:
            with contextlib.redirect_stdout(puffer):
                return "ok", fn(*args)
        except _Zeit:
            return "zeit", None
        except Exception as e:                      # noqa: BLE001 — Schülercode, alles melden
            return "fehler", e
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, alt)
    # ponytail: ausserhalb des Hauptthreads (z. B. GUI) gibt es kein SIGALRM; stattdessen ein Hilfsthread, dem per
    # PyThreadState_SetAsyncExc eine Ausnahme zugestellt wird. Das greift nur in Python-Code, nicht in lange C-Aufrufe.
    box: dict[str, Any] = {}

    def _lauf():
        box["tid"] = threading.get_ident()
        try:
            with contextlib.redirect_stdout(puffer):
                box["res"] = ("ok", fn(*args))
        except _Zeit:
            box["res"] = ("zeit", None)
        except Exception as e:                      # noqa: BLE001
            box["res"] = ("fehler", e)

    t = threading.Thread(target=_lauf, daemon=True)
    t.start()
    t.join(limit)
    if t.is_alive():
        if "tid" in box:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(box["tid"]), ctypes.py_object(_Zeit))
        t.join(1.0)
        return "zeit", None
    return box.get("res", ("fehler", RuntimeError("keine Antwort")))


def _kurz(x: Any, n: int = 60) -> str:
    try:
        s = repr(x)
    except Exception:                                # noqa: BLE001
        s = f"<{type(x).__name__}>"
    return s if len(s) <= n else s[: n - 1] + "…"


def _tn(x: Any) -> str:
    return type(x).__name__


# ==========================================================================
# Bericht
# ==========================================================================
class Fall:
    """Ein Prüffall: Name, bestanden?, Kurztext, Einzelheiten."""

    def __init__(self, name: str, ok: bool, text: str = "", details: list[str] | None = None):
        self.name, self.ok, self.text, self.details = name, ok, text, details or []

    def __repr__(self):
        return f"Fall({self.name!r}, {'ok' if self.ok else 'FALSCH'})"


class Pruefung:
    """Ergebnis einer Prüfung: `.bestanden`, `.gesamt`, `.ok`; `bool(p)` ist True, wenn alles stimmt."""

    def __init__(self, titel: str):
        self.titel = titel
        self.faelle: list[Fall] = []
        self.stolpersteine: list[Fall] = []           # ok=False heisst: Hinweis, kein Fehler
        self.hinweise: list[str] = []
        self.infos: list[str] = []
        self.zeitfehler = 0                           # so oft lief die Funktion schon ins Zeitlimit

    @property
    def bestanden(self) -> int:
        return sum(f.ok for f in self.faelle)

    @property
    def gesamt(self) -> int:
        return len(self.faelle)

    @property
    def ok(self) -> bool:
        return self.gesamt > 0 and self.bestanden == self.gesamt

    def __bool__(self):
        return self.ok

    def bericht(self, ausfuehrlich: bool = True) -> str:
        z = [f"{self.titel}: {self.bestanden} von {self.gesamt} Fällen bestanden"]
        for f in self.faelle:
            z.append(f"  {'✓' if f.ok else '✕'} {f.name}" + (f": {f.text}" if f.text and (not f.ok or ausfuehrlich) else ""))
            if not f.ok and ausfuehrlich:
                z += [f"      {d}" for d in f.details]
        for h in self.hinweise:
            z.append(f"  Hinweis: {h}")
        if self.stolpersteine and ausfuehrlich:
            z += ["", "Stolpersteine (kein Fehler, aber gut zu wissen):"]
            for s in self.stolpersteine:
                z.append(f"  {'✓' if s.ok else '!'} {s.name}: {s.text}")
        for i in self.infos:
            z.append(f"  {i}")
        if self.ok:
            z.append("Alles richtig.")
        return "\n".join(z)

    __repr__ = bericht

    def html(self) -> str:
        e = _html.escape
        farbe = "#1a7f37" if self.ok else "#c62828"
        h = [f'<div style="font-family:system-ui,sans-serif"><b style="color:{farbe}">{e(self.titel)}: '
             f'{self.bestanden} von {self.gesamt} Fällen bestanden</b><ul style="list-style:none;padding-left:1em">']
        for f in self.faelle:
            h.append(f"<li>{'✓' if f.ok else '✕'} {e(f.name)}" + (f": {e(f.text)}" if f.text else "") + "</li>")
            if not f.ok:
                h += [f'<li style="margin-left:1.5em;color:#666">{e(d)}</li>' for d in f.details]
        h.append("</ul>")
        for s in self.stolpersteine:
            h.append(f"<div>{'✓' if s.ok else '!'} {e(s.name)}: {e(s.text)}</div>")
        for x in self.hinweise + self.infos:
            h.append(f"<div>{e(x)}</div>")
        h.append("</div>")
        return "".join(h)

    _repr_html_ = html

    def show(self):
        print(self.bericht())


def _abschluss(p: Pruefung, ausgabe: bool | None, ausfuehrlich: bool = True) -> Pruefung:
    """Ausgabe: True druckt immer, False nie, None druckt (Standard für interaktives Üben)."""
    if ausgabe is None or ausgabe:
        print(p.bericht(ausfuehrlich))
    return p


# ==========================================================================
# Gemeinsamer Prüfablauf
# ==========================================================================
def _quelle_hinweis(funktion: Callable, verboten: dict[str, str]) -> list[str]:
    try:
        q = inspect.getsource(funktion)
    except (OSError, TypeError):
        return []
    return [text for tok, text in verboten.items() if tok in q]


def _lauf(p: Pruefung, funktion: Callable, name: str, args: tuple, soll: Any, pruefer: Callable,
          hinweis: str = "", limit: float = ZEITLIMIT, mutation: bool = True, details_soll: bool = True):
    """Ruft `funktion(*kopie)` auf, prüft das Ergebnis mit `pruefer(ist, soll) -> None | fehlertext`."""
    if not callable(funktion):
        p.faelle.append(Fall(name, False, "Das ist keine Funktion.", [f"Übergeben wurde {_kurz(funktion)} ({_tn(funktion)})."]))
        return
    eingabe = ", ".join(_kurz(a, 40) for a in args)
    if p.zeitfehler >= 2:                             # ponytail: nach zwei Zeitüberschreitungen nicht jeden Fall 2 s warten lassen
        p.faelle.append(Fall(name, False, "übersprungen: die Funktion lief davor schon zweimal ins Zeitlimit.", [f"Aufruf: f({eingabe})"]))
        return
    kopie = copy.deepcopy(args)
    status, ist = _mit_zeitlimit(funktion, kopie, limit)
    if status == "zeit":
        p.zeitfehler += 1
        p.faelle.append(Fall(name, False, f"Zeitlimit von {limit:g} s überschritten (Endlosschleife?).",
                             [f"Aufruf: f({eingabe})", "Tipp: Wird die Schleifenvariable in jedem Durchlauf verändert?"]))
        return
    if status == "fehler":
        p.faelle.append(Fall(name, False, f"Fehler im Aufruf: {_tn(ist)}: {ist}", [f"Aufruf: f({eingabe})"]))
        return
    fehler = pruefer(ist, soll)
    if fehler is None and mutation and kopie != copy.deepcopy(args) and not getattr(pruefer, "erlaubt_aenderung", False):
        fehler = "Die Eingabe wurde verändert. Arbeite mit einer Kopie, die Eingabe soll unverändert bleiben."
        p.faelle.append(Fall(name, False, fehler, [f"Aufruf: f({eingabe})", f"danach war die Eingabe: {_kurz(kopie, 80)}"]))
        return
    if fehler is None:
        p.faelle.append(Fall(name, True, hinweis))
        return
    det = [f"Aufruf: f({eingabe})", f"Erwartet: {_kurz(soll, 80)}", f"Erhalten: {_kurz(ist, 80)}"] if details_soll else [f"Aufruf: f({eingabe})"]
    if hinweis:
        det.append(f"Tipp: {hinweis}")
    p.faelle.append(Fall(name, False, fehler, det))


def _typ_und_gleich(typ: type | tuple, norm: Callable = lambda x: x, name: str = "") -> Callable:
    def pruefer(ist, soll):
        if not isinstance(ist, typ) or (typ is not bool and isinstance(ist, bool) and not isinstance(True, typ)):
            erw = typ.__name__ if isinstance(typ, type) else " oder ".join(t.__name__ for t in typ)
            return f"Falscher Rückgabetyp: erwartet {erw}, erhalten {_tn(ist)}."
        if norm(ist) != norm(soll):
            return "Falsches Ergebnis."
        return None
    return pruefer


# ==========================================================================
# SW1: Datentypen
# ==========================================================================
def _binaer_ref(n: int) -> str:
    return bin(n)[2:] if n else ""


def pruefe_dec_to_bin(funktion: Callable, ausfuehrlich: bool = True, ausgabe: bool | None = None) -> Pruefung:
    """Prüft `meine_dec_to_bin(n)`: Binärdarstellung als Text (Serie 1, Aufgabe 2 und decToBin)."""
    p = Pruefung("Dezimal nach Binär")

    def pruefer(ist, soll):
        if not isinstance(ist, str):
            return f"Falscher Rückgabetyp: erwartet str (Text aus 0 und 1), erhalten {_tn(ist)}."
        if soll == "":
            return None if ist in ("", "0") else "Für 0 ist '' (wie in der Serie) oder '0' richtig."
        if set(ist) - {"0", "1"}:
            return "Der Text enthält andere Zeichen als 0 und 1."
        if ist != soll:
            if ist == soll[::-1]:
                return "Die Ziffern stehen in umgekehrter Reihenfolge (der erste Rest ist die letzte Stelle)."
            if ist.lstrip("0") == soll:
                return "Führende Nullen: die Binärdarstellung beginnt mit der höchsten 1."
            return "Falsches Ergebnis."
        return None

    for n, h in [(409, "Vorlesung: 409 = 110011001₂"), (25, "Rest 1, 0, 0, 1, 1 von hinten nach vorn gelesen"), (1, ""),
                 (2, "Zweierpotenz: 1 mit Nullen"), (5, ""), (255, "acht Einsen"), (256, "1 mit acht Nullen"),
                 (1023, ""), (2 ** 20, ""), (0, "Sonderfall: keine Schleifendurchläufe")]:
        _lauf(p, funktion, f"n = {n}", (n,), _binaer_ref(n), pruefer, h)
    # Stolperstein: negative Zahlen
    st, _ = _mit_zeitlimit(funktion, (-5,), 0.5) if callable(funktion) else ("fehler", None)
    p.stolpersteine.append(Fall("Negative Zahl", st != "zeit",
                                {"zeit": "Endlosschleife: -5 // 2 ist -3 und -1 // 2 ist -1, temp wird nie 0. "
                                         "Die Serie definiert decToBin nur für natürliche Zahlen; eine Fehlermeldung (ValueError) wäre sauberer.",
                                 "fehler": "bricht mit einer Meldung ab. Das ist in Ordnung.", "ok": "liefert ein Ergebnis."}[st]))
    p.hinweise += _quelle_hinweis(funktion, {"bin(": "bin() ist die fertige Lösung. Zum Üben die Division durch 2 selbst schreiben."})
    return _abschluss(p, ausgabe, ausfuehrlich)


def pruefe_binaer_zu_dezimal(funktion: Callable, ausfuehrlich: bool = True, ausgabe: bool | None = None) -> Pruefung:
    """Prüft `meine_binaer_zu_dezimal(text)`: Wert einer Binärzahl (Stellenwerte 2^k)."""
    p = Pruefung("Binär nach Dezimal")
    pr_ = _typ_und_gleich(int)
    for t, h in [("110011001", "Vorlesung: 409"), ("1", ""), ("0", ""), ("10", "Zweierpotenz"), ("11001", "25"),
                 ("11111111", "255"), ("100000000", "256"), ("0011", "Führende Nullen ändern den Wert nicht"),
                 ("1" * 40, "grosse Zahl: 2⁴⁰ − 1")]:
        _lauf(p, funktion, f"'{t}'", (t,), int(t, 2), pr_, h)
    p.hinweise += _quelle_hinweis(funktion, {"int(": "int(text, 2) ist die fertige Lösung. Zum Üben die Stellenwerte selbst addieren."})
    return _abschluss(p, ausgabe, ausfuehrlich)


# ==========================================================================
# SW3: Quantoren und Prädikate
# ==========================================================================
def _bool_pruefer(ist, soll):
    if not isinstance(ist, bool):
        return f"Falscher Rückgabetyp: erwartet bool (True oder False), erhalten {_tn(ist)}" + (
            " (return vergessen?)" if ist is None else ".")
    return None if ist == soll else "Falscher Wahrheitswert."


def _quantor_fall(p, funktion, art, name, universum, praedikat, soll, hinweis=""):
    _lauf(p, funktion, name, (universum, praedikat), soll, _bool_pruefer, hinweis)


def _pruefe_quantor(funktion, art: str, ausfuehrlich, ausgabe) -> Pruefung:
    alle = art == "forall"
    p = Pruefung("∀ (forall)" if alle else "∃ (exists)")
    gerade, positiv, gleich3 = (lambda x: x % 2 == 0), (lambda x: x > 0), (lambda x: x == 3)
    ref = all if alle else any
    faelle = [
        ("alle erfüllen", [2, 4, 6], gerade, ""),
        ("keiner erfüllt", [1, 3, 5], gerade, ""),
        ("nur das erste erfüllt", [2, 3, 5], gerade, "Das erste Element muss mitgeprüft werden."),
        ("nur das letzte erfüllt", [1, 3, 4], gerade, "Das letzte Element muss mitgeprüft werden (Off-by-one in der Schleife?)."),
        ("nur das erste erfüllt nicht", [-1, 2, 3], positiv, ""),
        ("nur das letzte erfüllt nicht", [1, 2, -3], positiv, "Das letzte Element muss mitgeprüft werden."),
        ("nur ein Element, erfüllt", [3], gleich3, ""),
        ("nur ein Element, erfüllt nicht", [4], gleich3, ""),
        ("leeres Universum", [], gerade,
         "∀ über der leeren Menge ist wahr (es gibt kein Gegenbeispiel), ∃ ist falsch (es gibt keinen Zeugen)."),
        ("range als Universum", range(6), lambda x: x < 6 if alle else x == 5, "Das Universum muss nicht nur eine Liste sein."),
        ("Tupel mit Texten", ("a", "bb", "ccc"), lambda s: len(s) > 0 if alle else len(s) == 3, ""),
        ("Wahrheitswerte als Elemente", [True, False], lambda b: b is True or b is False if alle else b is False, ""),
    ]
    for name, uni, pred, h in faelle:
        soll = ref(pred(x) for x in uni)
        _quantor_fall(p, funktion, art, name, uni, pred, soll, h)
    # Prädikat darf nicht zu oft aufgerufen werden: abbrechen beim ersten Gegenbeispiel / Treffer
    zaehler = [0]

    def zaehlend(x):
        zaehler[0] += 1
        return (x > 0) if alle else (x == 2)
    if callable(funktion):
        stat, ist = _mit_zeitlimit(funktion, ([1, 2, 3, 4, 5, 6, 7, 8] if not alle else [1, -2, 3, 4, 5, 6, 7, 8], zaehlend), ZEITLIMIT)
        if stat == "ok":
            p.stolpersteine.append(Fall("Frühes Abbrechen", zaehler[0] <= 3,
                                        f"das Prädikat wurde {zaehler[0]}-mal aufgerufen. "
                                        + ("Gut: beim ersten " + ("Gegenbeispiel" if alle else "Treffer") + " ist die Antwort klar."
                                           if zaehler[0] <= 3 else "Beim ersten " + ("Gegenbeispiel" if alle else "Treffer")
                                           + " kann man sofort zurückgeben (return), das spart Aufrufe.")))
    p.hinweise += _quelle_hinweis(funktion, {"all(": "all() ist die fertige Lösung.", "any(": "any() ist die fertige Lösung."})
    return _abschluss(p, ausgabe, ausfuehrlich)


def pruefe_forall(funktion: Callable, ausfuehrlich: bool = True, ausgabe: bool | None = None) -> Pruefung:
    """Prüft `mein_forall(universum, praedikat)` (Serie 3, Aufgabe 2a)."""
    return _pruefe_quantor(funktion, "forall", ausfuehrlich, ausgabe)


def pruefe_exists(funktion: Callable, ausfuehrlich: bool = True, ausgabe: bool | None = None) -> Pruefung:
    """Prüft `mein_exists(universum, praedikat)` (Serie 3, Aufgabe 2b)."""
    return _pruefe_quantor(funktion, "exists", ausfuehrlich, ausgabe)


def pruefe_hat_duplikate(funktion: Callable, ausfuehrlich: bool = True, ausgabe: bool | None = None) -> Pruefung:
    """Prüft `meine_hat_duplikate(a)`: ∃i ∃j: ¬(i = j) ∧ a[i] = a[j] (Serie 3, Aufgabe 3f)."""
    p = Pruefung("Duplikate in einer Liste")
    for name, a, h in [
        ("Beispiel der Serie", [3, 2, 1, 1, 4, 4, 2, 3, 1], "Paare (1, 8) und (2, 7) mit gleichem Wert (Index ab 1)"),
        ("leere Liste", [], ""),
        ("ein Element", [7], "i = j gilt nicht als Duplikat: ¬(i = j)"),
        ("lauter verschiedene", [1, 2, 3, 4], "Ein Element mit sich selbst zu vergleichen ist kein Duplikat."),
        ("zwei gleiche am Anfang", [5, 5, 1, 2], ""),
        ("zwei gleiche am Ende", [1, 2, 6, 6], "Die letzte Position muss mitgeprüft werden."),
        ("erstes und letztes gleich", [9, 1, 2, 9], "Weit auseinander liegende Paare nicht vergessen."),
        ("nur zwei Elemente gleich", [4, 4], ""),
        ("nur zwei Elemente verschieden", [4, 5], ""),
        ("Texte", ["a", "b", "a"], ""),
        ("absteigend ohne Duplikate", [9, 7, 5, 3, 1], "Die Liste nicht sortieren: das verändert die Eingabe."),
    ]:
        _lauf(p, funktion, name, (a,), len(set(a)) < len(a), _bool_pruefer, h)
    p.hinweise += _quelle_hinweis(funktion, {"set(": "set() ist die fertige Lösung. Zum Üben die Doppelschleife über i und j schreiben."})
    return _abschluss(p, ausgabe, ausfuehrlich)


def pruefe_palindrom(funktion: Callable, ausfuehrlich: bool = True, ausgabe: bool | None = None) -> Pruefung:
    """Prüft `mein_palindrom(s)`: Pal(s) aus Serie 3."""
    p = Pruefung("Palindrom")
    for s, h in [("REITTIER", "Vorlesung: Pal(REITTIER) hat den Wahrheitswert 1"), ("REITTIERS", ""), ("", "Leerer Text liest sich in beide Richtungen gleich"),
                 ("A", ""), ("AA", "gerade Länge"), ("AB", ""), ("ABA", "ungerade Länge: das mittlere Zeichen ist frei"),
                 ("ABCA", "Nur die äusseren Zeichen stimmen"), ("ABCBA", ""), ("ABCCA", "Mitte passt nicht"),
                 ("Aa", "Gross- und Kleinschreibung zählen als verschieden"), ("ANNA", ""), ("1221", "Ziffern sind auch Zeichen")]:
        _lauf(p, funktion, f"'{s}'", (s,), s == s[::-1], _bool_pruefer, h)
    p.hinweise += _quelle_hinweis(funktion, {"[::-1]": "[::-1] ist die fertige Lösung. Zum Üben Zeichen paarweise von aussen nach innen vergleichen.",
                                              "reversed(": "reversed() ist die fertige Lösung."})
    return _abschluss(p, ausgabe, ausfuehrlich)


# ==========================================================================
# SW2 / SW4: Aussagenlogik und Kalkül
# ==========================================================================
def _belegungen(n: int):
    return list(itertools.product([False, True], repeat=n))


def _tabelle_pruefer(n):
    def pruefer(ist, soll):
        if not isinstance(ist, (list, tuple)):
            return f"Falscher Rückgabetyp: erwartet Liste von Zeilen (belegung, wert), erhalten {_tn(ist)}."
        if len(ist) != 2 ** n:
            return f"Falsche Zeilenzahl: bei {n} Variablen sind es 2^{n} = {2 ** n} Zeilen, erhalten {len(ist)}."
        zeilen = {}
        for z in ist:
            try:
                b, w = z
                b = tuple(bool(x) for x in b)
            except (TypeError, ValueError):
                return f"Eine Zeile hat nicht die Form (belegung, wert): {_kurz(z)}"
            if len(b) != n:
                return f"Eine Belegung hat {len(b)} statt {n} Werte: {_kurz(z)}"
            if w not in (True, False, 0, 1):
                return f"Der Wert einer Zeile ist kein Wahrheitswert: {_kurz(z)}"
            if b in zeilen:
                return f"Die Belegung {tuple(int(x) for x in b)} kommt doppelt vor."
            zeilen[b] = bool(w)
        fehlend = [b for b in soll if b not in zeilen]
        if fehlend:
            return f"Es fehlt die Belegung {tuple(int(x) for x in fehlend[0])}."
        falsch = [b for b in soll if zeilen[b] != soll[b]]
        if falsch:
            b = falsch[0]
            return f"Falscher Wert in der Zeile {tuple(int(x) for x in b)}: erwartet {int(soll[b])}, erhalten {int(zeilen[b])}."
        return None
    return pruefer


_FUNKTIONEN_AL = [
    ("A ∧ B", lambda a, b: a and b, 2, ""), ("A ∨ B", lambda a, b: a or b, 2, ""),
    ("A ⇒ B", lambda a, b: (not a) or b, 2, "falsch nur bei A = 1, B = 0"),
    ("A ⇔ B", lambda a, b: a == b, 2, ""), ("A ⊕ B", lambda a, b: a != b, 2, ""),
    ("¬A", lambda a: not a, 1, "eine Variable: 2 Zeilen"),
    ("(A ∧ B) ∨ C", lambda a, b, c: (a and b) or c, 3, "acht Zeilen"),
    ("(A ∨ B) ∧ (C ∨ (¬B ∧ ¬A) ∨ B)", lambda a, b, c: (a or b) and (c or (not b and not a) or b), 3, "WB aus Serie 4"),
    ("A ∧ ¬A", lambda a: a and not a, 1, "Kontradiktion: überall 0"),
    ("vier Variablen", lambda a, b, c, d: (a and b) != (c or d), 4, "16 Zeilen"),
]


def pruefe_wahrheitstabelle(funktion: Callable, ausfuehrlich: bool = True, ausgabe: bool | None = None) -> Pruefung:
    """Prüft `meine_wahrheitstabelle(f_func, n_vars)`: Liste von (belegung, wert); Reihenfolge der Zeilen egal."""
    p = Pruefung("Wahrheitstabelle")
    for name, f, n, h in _FUNKTIONEN_AL:
        soll = {b: bool(f(*b)) for b in _belegungen(n)}
        _lauf(p, funktion, name, (f, n), soll, _tabelle_pruefer(n), h, mutation=False)
    return _abschluss(p, ausgabe, ausfuehrlich)


def pruefe_folgerung(funktion: Callable, ausfuehrlich: bool = True, ausgabe: bool | None = None) -> Pruefung:
    """Prüft `meine_folgerung(wb_func, q_func, n_vars)`: WB ⊨ Q (Serie 4)."""
    p = Pruefung("Semantische Folgerung")

    def folgt(wb, q, n):
        return all(q(*b) for b in _belegungen(n) if wb(*b))
    wb1 = lambda a, b, c: (a or b) and (c or (not b and not a) or b)
    faelle = [
        ("Serie 4: WB ⊨ A ∨ B", wb1, lambda a, b, c: a or b, 3, "Zeilen mit WB = 0 zählen nicht"),
        ("A ∨ B ⊨ A gilt nicht", lambda a, b: a or b, lambda a, b: a, 2, "Gegenbeispiel A = 0, B = 1"),
        ("Rückrichtung: A ∨ B ⊨ WB gilt nicht", lambda a, b, c: a or b, wb1, 3, "WB ⊨ Q ist nicht dasselbe wie Q ⊨ WB."),
        ("U ∧ W ⊨ U ∨ W", lambda u, w: u and w, lambda u, w: u or w, 2, "Lerngerüst"),
        ("(¬H ∨ S) ∧ H ⊨ S ∧ H", lambda h, s: ((not h) or s) and h, lambda h, s: s and h, 2, "Lerngerüst"),
        ("(A∨B∨C) ⇒ (P∧Q) ⊨ (A⇒P) ∧ (C⇒Q)", lambda a, b, c, p_, q: (not (a or b or c)) or (p_ and q),
         lambda a, b, c, p_, q: ((not a) or p_) and ((not c) or q), 5, "Lerngerüst, 32 Zeilen"),
        ("Rückrichtung dazu gilt nicht", lambda a, b, c, p_, q: ((not a) or p_) and ((not c) or q),
         lambda a, b, c, p_, q: (not (a or b or c)) or (p_ and q), 5, "Lerngerüst: nur die Hinrichtung gilt"),
        ("(A ∨ B) ∧ (B ⇒ C) ⊨ A ∨ C", lambda a, b, c: (a or b) and ((not b) or c), lambda a, b, c: a or c, 3, "Resolutionsregel, Aufgabe 3"),
        ("Gleiche Formeln", lambda a, b: a and b, lambda a, b: a and b, 2, ""),
        ("WB widersprüchlich: folgt alles", lambda a: a and not a, lambda a: False, 1,
         "Aus einem Widerspruch folgt jede Aussage: es gibt keine Zeile mit WB = 1."),
        ("Q immer wahr", lambda a, b: a, lambda a, b: b or not b, 2, ""),
        ("WB immer wahr, Q nicht", lambda a, b: a or not a, lambda a, b: a, 2, ""),
        ("WB immer wahr, Q immer falsch", lambda a: a or not a, lambda a: False, 1, ""),
    ]
    for name, wb, q, n, h in faelle:
        _lauf(p, funktion, name, (wb, q, n), folgt(wb, q, n), _bool_pruefer, h, mutation=False)
    return _abschluss(p, ausgabe, ausfuehrlich)


def pruefe_klassifikation(funktion: Callable, ausfuehrlich: bool = True, ausgabe: bool | None = None) -> Pruefung:
    """Prüft `meine_klassifikation(f_func, n_vars)`: 'Tautologie', 'Kontradiktion' oder 'Kontingenz'."""
    p = Pruefung("Klassifikation einer Formel")

    def pruefer(ist, soll):
        if not isinstance(ist, str):
            return f"Falscher Rückgabetyp: erwartet Text, erhalten {_tn(ist)}."
        return None if ist.strip().lower() == soll.lower() else "Falsche Klasse."
    for name, f, n, soll in [("A ∨ ¬A", lambda a: a or not a, 1, "Tautologie"), ("A ∧ ¬A", lambda a: a and not a, 1, "Kontradiktion"),
                             ("A ∧ B", lambda a, b: a and b, 2, "Kontingenz"), ("A ⇒ (B ⇒ A)", lambda a, b: (not a) or ((not b) or a), 2, "Tautologie"),
                             ("(A ⇒ B) ∧ A ∧ ¬B", lambda a, b: ((not a) or b) and a and not b, 2, "Kontradiktion"),
                             ("A ⊕ B", lambda a, b: a != b, 2, "Kontingenz")]:
        _lauf(p, funktion, name, (f, n), soll, pruefer, "", mutation=False)
    return _abschluss(p, ausgabe, ausfuehrlich)


def _norm_klauseln(ist) -> tuple[frozenset, str | None]:
    """Normalisiert eine Liste von Klauseln. Gibt (Menge von frozensets ohne Tautologien, Fehlertext) zurück."""
    if isinstance(ist, (set, frozenset)) and ist and all(isinstance(x, str) for x in ist):
        return frozenset(), "Gib eine Liste von Klauseln zurück (jede Klausel ist selbst eine Menge), nicht eine einzelne Klausel."
    if not isinstance(ist, (list, tuple, set, frozenset)):
        return frozenset(), f"Falscher Rückgabetyp: erwartet Liste von Mengen, erhalten {_tn(ist)}."
    res = set()
    for k in ist:
        if isinstance(k, str) or not isinstance(k, (set, frozenset, list, tuple)):
            return frozenset(), f"Eine Klausel muss eine Menge von Literalen sein, erhalten {_kurz(k)}."
        k = frozenset(k)
        if any(not isinstance(l, str) for l in k):
            return frozenset(), f"Literale sind Texte wie 'A' und '¬B': {_kurz(k)}"
        if any(("¬" + l) in k for l in k if not l.startswith("¬")):
            continue                              # tautologische Resolventen sind erlaubt und werden ignoriert
        res.add(k)
    return frozenset(res), None


def _resolventen_ref(k1, k2):
    out = set()
    for l in k1:
        g = l[1:] if l.startswith("¬") else "¬" + l
        if g in k2:
            r = frozenset((set(k1) - {l}) | (set(k2) - {g}))
            if not any(("¬" + x) in r for x in r if not x.startswith("¬")):
                out.add(r)
    return frozenset(out)


def pruefe_resolvente(funktion: Callable, ausfuehrlich: bool = True, ausgabe: bool | None = None) -> Pruefung:
    """Prüft `meine_resolvente(k1, k2)`: Liste der Resolventen als Mengen (Serie 4, Resolutionsregel)."""
    p = Pruefung("Resolutionsregel")

    def pruefer(ist, soll):
        norm, fehler = _norm_klauseln(ist)
        if fehler:
            return fehler
        if norm != soll:
            if not norm and soll:
                return "Es wurde keine Resolvente gefunden, obwohl ein komplementäres Paar existiert."
            if norm and not soll:
                return "Es gibt kein komplementäres Paar, trotzdem wurde eine Resolvente geliefert."
            return "Falsche Resolvente."
        return None
    faelle = [
        ("Serie 4, Aufgabe 5: {S, N} und {¬S}", {"S", "N"}, {"¬S"}, "Streiche S und ¬S: übrig bleibt {N} (K6)"),
        ("{¬N, P} und {P, N}", {"¬N", "P"}, {"P", "N"}, "Mitschrift K7 = {P}: gleiches Literal fällt nur einmal an"),
        ("Regel der Vorlesung: {A, B} und {¬B, C}", {"A", "B"}, {"¬B", "C"}, "(A ∨ B) ∧ (¬B ∨ C) ⊢ (A ∨ C)"),
        ("Umgekehrte Reihenfolge", {"¬B", "C"}, {"A", "B"}, "Das komplementäre Paar kann in k1 oder in k2 stehen."),
        ("Leere Klausel: {A} und {¬A}", {"A"}, {"¬A"}, "Das Ergebnis ist die leere Klausel □ = set(), die Liste ist [set()], nicht []."),
        ("Kein komplementäres Paar", {"A"}, {"B"}, "Keine Resolvente: leere Liste []."),
        ("Gleiche Klausel zweimal", {"A", "B"}, {"A", "B"}, "Kein Paar A / ¬A."),
        ("Mehrere Literale bleiben", {"A", "B", "C"}, {"¬A", "D"}, "Alles ausser A / ¬A bleibt: {B, C, D}"),
        ("Gleiches Restliteral", {"A", "C"}, {"¬A", "C"}, "Die Vereinigung zweier Mengen enthält C nur einmal: {C}"),
        ("Zwei komplementäre Paare", {"A", "B"}, {"¬A", "¬B"}, "Es entstehen {B, ¬B} und {A, ¬A}: beide sind tautologisch (immer wahr); sie dürfen fehlen."),
        ("Negierte Literale auf beiden Seiten", {"¬A", "¬B"}, {"A", "¬C"}, "{¬B, ¬C}"),
    ]
    for name, k1, k2, h in faelle:
        _lauf(p, funktion, name, (k1, k2), _resolventen_ref(k1, k2), pruefer, h)
    return _abschluss(p, ausgabe, ausfuehrlich)


# ==========================================================================
# SW5: Algorithmen
# ==========================================================================
def _schritte_linear_ref(n: int, k: int | None) -> int:
    """Lecture: k = Index des Treffers (0-basiert) oder None. 2·(k+1) + 1 bzw. 2n + 1."""
    return 2 * (k + 1) + 1 if k is not None else 2 * n + 1


def _schritte_linear_dk(a, x) -> int:
    # Referenz aus dem Rechenmodul; bei jedem Problem die Formel der Vorlesung.
    try:
        from ..rechnen import algorithmen as _ag
        return _ag.count_steps("linearSearch", Liste=list(a), Element=x).total
    except Exception:                                # noqa: BLE001
        k = a.index(x) if x in a else None
        return _schritte_linear_ref(len(a), k)


def pruefe_lineare_suche(funktion: Callable, ausfuehrlich: bool = True, ausgabe: bool | None = None) -> Pruefung:
    """Prüft `meine_lineare_suche(a, x)`: gibt (gefunden, schritte) zurück; Zählregeln der Vorlesung (Serie 5, Aufgabe 3)."""
    p = Pruefung("Lineare Suche mit Schrittzähler")

    def pruefer(ist, soll):
        if isinstance(ist, bool) or not isinstance(ist, (tuple, list)) or len(ist) != 2:
            return ("Gib ein Paar (gefunden, schritte) zurück, nicht nur " + (_tn(ist)) + ".")
        g, s = ist
        if not isinstance(g, bool):
            return f"gefunden muss True oder False sein, erhalten {_tn(g)}."
        if isinstance(s, bool) or not isinstance(s, int):
            return f"schritte muss eine ganze Zahl sein, erhalten {_tn(s)}."
        if g != soll[0]:
            return "Falsches Suchergebnis (gefunden)."
        if s != soll[1]:
            tipp = ""
            if s == soll[1] - 1:
                tipp = " Fehlt das return mit seinem Schritt?"
            elif s == soll[1] + 1:
                tipp = " Wurde etwas doppelt gezählt?"
            return f"Falsche Schrittzahl: {s} statt {soll[1]}." + tipp
        return None
    faelle = [
        ("Element in der Mitte", [4, 8, 15, 16], 8, "2 Schritte je Position (for + Vergleich), dazu 1 für return: 2·2 + 1 = 5"),
        ("Element fehlt (Worst-Case)", [4, 8, 15, 16], 23, "2n + 1 = 9"),
        ("Element vorne", [4, 8, 15], 4, "Treffer an Position 1: 2·1 + 1 = 3"),
        ("Element hinten (Worst-Case)", [4, 8, 15], 15, "ebenfalls 2n + 1 = 7"),
        ("Leere Liste", [], 1, "keine Durchläufe: nur das return, 1 Schritt"),
        ("Ein Element, gefunden", [5], 5, ""),
        ("Ein Element, fehlt", [5], 6, ""),
        ("Wert kommt mehrfach vor", [1, 2, 2, 2], 2, "Beim ersten Treffer aufhören: 2·2 + 1 = 5"),
        ("Länge 10, fehlt", list(range(10)), -1, "2·10 + 1 = 21"),
    ]
    for name, a, x, h in faelle:
        soll = (x in a, _schritte_linear_dk(a, x))
        _lauf(p, funktion, f"{name}: a = {_kurz(a, 30)}, x = {x}", (a, x), soll, pruefer, h)
    p.infos.append("Zählregeln: for-Durchlauf 1, Vergleich 1, return 1; len() und Listenzugriff zählen nicht.")
    return _abschluss(p, ausgabe, ausfuehrlich)


def pruefe_selection_sort(funktion: Callable, ausfuehrlich: bool = True, ausgabe: bool | None = None) -> Pruefung:
    """Prüft `mein_selection_sort(a)`: sortiert aufsteigend (in place mit return oder als neue Liste)."""
    p = Pruefung("Selection Sort")
    rng = random.Random(1)

    def pruefer(ist, soll):
        if ist is None:
            return "Es wurde nichts zurückgegeben. Füge am Ende return a ein (oder sortiere die übergebene Liste direkt)."
        if not isinstance(ist, list):
            return f"Falscher Rückgabetyp: erwartet list, erhalten {_tn(ist)}."
        if sorted(ist, key=repr) != sorted(soll, key=repr):
            return "Es fehlen oder es kommen Elemente dazu (sortiert wird nur umgeordnet)."
        if ist != sorted(soll):
            return "Die Liste ist nicht aufsteigend sortiert."
        return None
    pruefer.erlaubt_aenderung = True
    for name, a, h in [
        ("leer", [], ""), ("ein Element", [4], ""), ("zwei, falsch herum", [2, 1], "Der Tausch muss auch bei n = 2 funktionieren."),
        ("zwei, richtig", [1, 2], ""), ("schon sortiert", [1, 2, 3, 4, 5], ""),
        ("absteigend", [5, 4, 3, 2, 1], "Worst-Case der Vorlesung"),
        ("Beispiel", [3, 1, 2], ""), ("Minimum ganz hinten", [2, 3, 4, 1], "Der Tausch muss mit dem richtigen Index geschehen."),
        ("Maximum vorne", [9, 1, 2, 3], ""),
        ("doppelte Werte", [3, 2, 1, 1, 4, 4, 2, 3, 1], "Liste aus Serie 3"),
        ("negative Zahlen", [0, -3, 5, -1, -3], ""), ("alle gleich", [7, 7, 7], ""),
        ("zufällig, 40 Werte", [rng.randint(-50, 50) for _ in range(40)], ""),
    ]:
        # Eine Funktion ohne return, die direkt in place sortiert, gilt auch als richtig: Kopie nachprüfen.
        kopie = copy.deepcopy(a)
        if callable(funktion):
            st, ist = _mit_zeitlimit(funktion, (kopie,), ZEITLIMIT)
            if st == "ok" and ist is None and kopie == sorted(a):
                p.faelle.append(Fall(name, True, "sortiert direkt in der übergebenen Liste (ohne return) — auch in Ordnung"))
                continue
        _lauf(p, funktion, name, (a,), a, pruefer, h, mutation=False)
    p.hinweise += _quelle_hinweis(funktion, {"sorted(": "sorted() ist die fertige Lösung.", ".sort(": "list.sort() ist die fertige Lösung.",
                                              "min(": "min() erspart die innere Schleife. In der Serie ist sie Teil der Zählung."})
    return _abschluss(p, ausgabe, ausfuehrlich)


# ==========================================================================
# Weitere Übungen aus der Vorlesung
# ==========================================================================
def pruefe_potenzmenge(funktion: Callable, ausfuehrlich: bool = True, ausgabe: bool | None = None) -> Pruefung:
    """Prüft `meine_potenzmenge(menge)`: alle 2ⁿ Teilmengen (Serie 1, Mengen)."""
    p = Pruefung("Potenzmenge")

    def pruefer(ist, soll):
        if not isinstance(ist, (list, tuple, set, frozenset)):
            return f"Falscher Rückgabetyp: erwartet Liste von Mengen, erhalten {_tn(ist)}."
        try:
            teile = [frozenset(x) for x in ist]
        except TypeError:
            return "Jede Teilmenge muss eine Menge (set) sein."
        if len(teile) != len(soll):
            return f"Falsche Anzahl: 2^n = {len(soll)} Teilmengen erwartet, erhalten {len(teile)}."
        if len(set(teile)) != len(teile):
            return "Eine Teilmenge kommt doppelt vor."
        if set(teile) != soll:
            fehlt = [set(x) for x in soll if x not in set(teile)][:2]
            return f"Falsche Teilmengen, es fehlt z. B. {fehlt}."
        return None
    for menge, h in [([], "∅ hat genau eine Teilmenge: ∅ selbst"), ([1], "∅ und {1}"), ([1, 2], "4 Teilmengen"),
                     ([1, 2, 3], "8 Teilmengen"), (["a", "b", "c", "d"], "16 Teilmengen"), ([5, 6, 7, 8, 9], "32 Teilmengen")]:
        soll = {frozenset(c) for r in range(len(menge) + 1) for c in itertools.combinations(menge, r)}
        _lauf(p, funktion, f"{{{', '.join(map(str, menge))}}}", (menge,), soll, pruefer, h)
    return _abschluss(p, ausgabe, ausfuehrlich)


def _periodisch_ref(text: str) -> Fraction:
    """Bruch zu 'g.v(p)' mit Ganzzahlteil g, Vorperiode v, Periode p."""
    t = text.strip()
    ganz, _, rest = t.partition(".")
    vor, per = rest, ""
    if "(" in rest:
        vor, _, per = rest.partition("(")
        per = per.rstrip(")")
    wert = Fraction(int(ganz or "0"))
    if vor:
        wert += Fraction(int(vor), 10 ** len(vor))
    if per:
        wert += Fraction(int(per), (10 ** len(per) - 1) * 10 ** len(vor))
    return wert


def pruefe_dezimal_zu_bruch(funktion: Callable, ausfuehrlich: bool = True, ausgabe: bool | None = None) -> Pruefung:
    """Prüft `meine_dezimal_zu_bruch(text)`: periodische Dezimalzahl als Fraction (Serie 1)."""
    p = Pruefung("Periodische Dezimalzahl als Bruch")

    def pruefer(ist, soll):
        if isinstance(ist, Fraction):
            return None if ist == soll else "Falscher Bruch."
        if isinstance(ist, (tuple, list)) and len(ist) == 2 and all(isinstance(x, int) for x in ist):
            if ist[1] == 0:
                return "Der Nenner ist 0."
            if Fraction(ist[0], ist[1]) != soll:
                return "Falscher Bruch."
            return None if (ist[0], ist[1]) == (soll.numerator, soll.denominator) else "Der Bruch stimmt, ist aber nicht gekürzt."
        return f"Falscher Rückgabetyp: erwartet Fraction (oder Paar aus Zähler und Nenner), erhalten {_tn(ist)}."
    for t, h in [("2.10(365)", "Vorlesung: 42031/19980"), ("0.(3)", "1/3"), ("0.(9)", "Die Periode 9 ergibt genau 1"), ("0.5", "keine Periode"),
                 ("1.2(34)", "Vorperiode 2, Periode 34"), ("0.0(01)", "Vorperiode 0"), ("12.(142857)", "sechsstellige Periode"),
                 ("3", "ganze Zahl"), ("0.(6)", "2/3, muss gekürzt werden")]:
        _lauf(p, funktion, f"'{t}'", (t,), _periodisch_ref(t), pruefer, h)
    return _abschluss(p, ausgabe, ausfuehrlich)


def _binaer_nachkomma_ref(z: int, n: int) -> str:
    q = Fraction(z, n)
    ganz = int(q)
    text = bin(ganz)[2:]
    r = q - ganz
    if r == 0:
        return text
    stellen: list[str] = []
    gesehen: dict[Fraction, int] = {}
    while r != 0 and r not in gesehen:
        gesehen[r] = len(stellen)
        r *= 2
        stellen.append("1" if r >= 1 else "0")
        if r >= 1:
            r -= 1
    if r == 0:
        return text + "." + "".join(stellen)
    i = gesehen[r]
    return text + "." + "".join(stellen[:i]) + "(" + "".join(stellen[i:]) + ")"


def pruefe_binaer_nachkomma(funktion: Callable, ausfuehrlich: bool = True, ausgabe: bool | None = None) -> Pruefung:
    """Prüft `meine_binaer_nachkomma(zaehler, nenner)`: Text wie '0.0(0011)' (Serie 1, Binärbrüche)."""
    p = Pruefung("Binärer Bruch mit Periode")
    pr_ = _typ_und_gleich(str)
    for z, n, h in [(1, 10, "Vorlesung: 0.1 ist binär periodisch: 0.0(0011)"), (1, 2, "0.1"), (3, 8, "endlich: 0.011"), (1, 3, "0.(01)"),
                    (5, 6, "0.1(10): Vorperiode 1"), (7, 4, "Ganzzahlteil 1: 1.11"), (0, 1, "0"), (4, 1, "ganze Zahl: 100"), (1, 5, "0.(0011)")]:
        _lauf(p, funktion, f"{z}/{n}", (z, n), _binaer_nachkomma_ref(z, n), pr_, h)
    return _abschluss(p, ausgabe, ausfuehrlich)
