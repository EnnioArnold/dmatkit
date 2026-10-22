#!/usr/bin/env python3
"""Tests für dmatkit.aufgaben.exercises: python3 tests/tests_exercises.py

Jede Aufgabe der Serien 1 bis 6, des Lerngerüsts und der Vorlesungsbeispiele muss laufen und mit dem Soll
übereinstimmen. Die Soll-Werte stammen aus den Serien, der Mitschrift, Python-Builtins oder Brute-Force,
nie aus der Ausgabe von dmatkit selbst. Dieser Test rechnet die wichtigsten Werte noch einmal unabhängig nach.
"""
import itertools
import math
import os
import re
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import dmatkit as dk
from dmatkit.aufgaben import exercises as ex
from dmatkit.rechnen.render import Result

N = [0]


def ok(cond, msg=""):
    N[0] += 1
    assert cond, msg


def raises(exc, fn, *a, text=None, **kw):
    N[0] += 1
    try:
        fn(*a, **kw)
    except exc as e:
        if text is not None:
            assert text in str(e), f"{text!r} nicht in {str(e)!r}"
        return str(e)
    raise AssertionError(f"{exc.__name__} erwartet")


# ==========================================================================
# 1. Alle Aufgaben laufen und stimmen
# ==========================================================================
g = ex.alle_pruefen()
ok(g.ok and bool(g), repr(g))
ok(g.gesamt == len(ex.EXERCISES) == g.bestanden and g.fehlgeschlagen == [])
for k in ex.EXERCISES:
    b = ex.pruefen(k)
    ok(b.ok and bool(b), repr(b))
    ok(len(b.zeilen) >= 2, f"{k}: mindestens Rechenweg und eine Erwartung")

# ==========================================================================
# 2. Aufbau und Vollständigkeit
# ==========================================================================
wochen = {w: ex.liste(w) for w in range(1, 7)}
ok(sum(len(v) for v in wochen.values()) == len(ex.EXERCISES))
ok(all(len(v) > 0 for v in wochen.values()))
ok(ex.liste() == list(ex.EXERCISES))
ok(ex.liste("SW3") == ex.liste(3) == wochen[3] and ex.liste("sw3") == wochen[3])
ok(all(ex.EXERCISES[k].week == 3 for k in ex.liste(3)))
for art in ("Serie", "Beispiel", "Lerngerüst", "optional"):
    sub = ex.liste(art=art)
    ok(len(sub) > 0 and all(ex.aufgabe(k).kind == art for k in sub))
ok(ex.liste(3, "serie") == [k for k in wochen[3] if ex.aufgabe(k).kind == "Serie"])
raises(KeyError, ex.liste, 9, text="keine Aufgaben")
raises(KeyError, ex.liste, "Woche drei", text="Unbekannte Woche")
repr(ex.liste(2))
ex.liste(2)._repr_html_()
ok(repr(ex.liste(art="Beispiel")).count("\n") >= 10)

ids = list(ex.EXERCISES)
ok(len(ids) == len(set(ids)) and all(re.fullmatch(r"(SW|LG)[1-6]\.[A-Za-z0-9]+", k) for k in ids))
for k, e in ex.EXERCISES.items():
    ok(e.id == k and 1 <= e.week <= 6 and str(e.week) == k[2], k)
    ok(isinstance(e.title, str) and len(e.title) > 3 and isinstance(e.task, str) and len(e.task) > 10, k)
    ok(callable(e.call) and len(e.expected) >= 1, k)
    ok(isinstance(e.hint, str) and e.hint != "", f"{k}: Hinweis fehlt")
    ok(e.source != "", f"{k}: Quelle fehlt")
    ok(e.kind in ("Serie", "Beispiel", "Lerngerüst", "optional"), k)
    repr(e)
    e._repr_html_()
    ok("erwartet" not in e.task.lower(), k)

# Alle Aufgaben der Serien 1 bis 6 sind vorhanden (nach den Übungsblättern)
SOLL = {
    1: ["SW1.1a", "SW1.1b", "SW1.1c", "SW1.1d", "SW1.1e", "SW1.2", "SW1.3", "SW1.4", "SW1.5a32", "SW1.5b32", "SW1.5a64", "SW1.5b64", "SW1.6", "SW1.7a"],
    2: ["SW2.1a", "SW2.1b", "SW2.2", "SW2.3a", "SW2.3b", "SW2.4a", "SW2.4b", "SW2.5", "SW2.6", "SW2.7a", "SW2.7b", "SW2.7c", "SW2.7d", "SW2.7e"],
    3: ["SW3.1", "SW3.2a", "SW3.2b", "SW3.3a", "SW3.3b", "SW3.3c", "SW3.3d", "SW3.3e", "SW3.3f", "SW3.4"]
       + [f"SW3.5{c}" for c in "abcdefgh"] + ["SW3.6a", "SW3.6b", "SW3.6c", "SW3.7"],
    4: ["SW4.1", "SW4.1b", "SW4.2", "SW4.2b", "SW4.3", "SW4.3b", "SW4.4", "SW4.5", "SW4.G", "SW4.P1", "SW4.P2"],
    5: ["SW5.1a", "SW5.1b", "SW5.2a", "SW5.2b", "SW5.3", "SW5.4a", "SW5.4b", "SW5.5a", "SW5.5b", "SW5.6a", "SW5.6b", "SW5.7a", "SW5.7b",
        "SW5.8a", "SW5.8b", "SW5.8c", "SW5.8d", "SW5.8e"],
    6: ["SW6.1a", "SW6.1b", "SW6.2", "SW6.3", "SW6.4a", "SW6.4b", "SW6.5a", "SW6.5b", "SW6.5c", "SW6.6"]
       + [f"SW6.7{c}" for c in "abcde"] + [f"SW6.8{c}" for c in "abcde"],
}
for w, soll in SOLL.items():
    for k in soll:
        ok(k in ex.EXERCISES, f"fehlt: {k}")
    ok(set(soll) <= set(wochen[w]))
for pref in ("LG1.a1", "LG1.b1", "LG1.c1", "LG1.d", "LG1.e", "LG1.f", "LG2.a", "LG2.b", "LG2.c1", "LG2.d1", "LG2.e1", "LG3.a", "LG3.b1", "LG3.c1",
             "LG4.a1", "LG4.b", "LG4.c", "LG4.d", "LG5.a", "LG5.b1"):
    ok(pref in ex.EXERCISES, pref)
ok(len(ex.liste(art="Lerngerüst")) >= 35)
ok(len(ex.liste(art="Beispiel")) >= 35)

# ==========================================================================
# 3. Zugriff: aufgabe, loesen, pruefen
# ==========================================================================
ok(ex.aufgabe("SW3.5a") is ex.EXERCISES["SW3.5a"])
ok(ex.aufgabe("sw3.5A") is ex.aufgabe("SW3.5a") and ex.aufgabe("sw3.5a ") is ex.aufgabe("SW3.5a"))
ok(ex.aufgabe("3.5a") is ex.aufgabe("SW3.5a"))
msg = raises(KeyError, ex.aufgabe, "SW3.5")
ok("SW3.5a" in msg, "Vorschläge bei Teil-Schlüssel")
raises(KeyError, ex.aufgabe, "SW3.55")
raises(KeyError, ex.aufgabe, "quatsch", text="liste()")
raises(KeyError, ex.loesen, "SW9.1")
raises(KeyError, ex.pruefen, "SW9.1")
for k in ex.EXERCISES:
    r = ex.loesen(k)
    ok(isinstance(r, Result) and len(r.steps) > 0, k)
    ok(isinstance(repr(r), str) and r._repr_html_() and r.latex() is not None, k)
b = ex.pruefen("SW5.1a")
ok(b.ok and "SW5.1a" in repr(b) and "✓" in repr(b) and "<table" in b._repr_html_())
ok("Alles grün" in repr(g))
ok(repr(g).startswith(f"{g.gesamt} von {g.gesamt}") and "SW5:" in repr(g) and "<pre>" in g._repr_html_())
for w in range(1, 7):
    gw = ex.alle_pruefen(w)
    ok(gw.ok and gw.gesamt == len(wochen[w]), w)
ok(ex.alle_pruefen("SW2", "Lerngerüst").gesamt == len(ex.liste(2, "Lerngerüst")))


# Das Prüfwerk erkennt falsche Soll-Werte, Ausnahmen und fehlenden Rechenweg
def e_bad(soll, call=lambda: dk.gauss_sum(100), lesen=lambda r: r.value):
    return ex.Exercise("SW5.X", 5, "t", "Aufgabe zum Prüfen des Prüfers", call, (ex.E("Wert", lesen, soll),), "h")


ok(e_bad(5050).pruefen().ok)
b = e_bad(5051).pruefen()
ok(not b.ok and not bool(b) and b.fehler and "erwartet" in b.fehler[0][1] and "✗" in repr(b))
b = e_bad(5050, lesen=lambda r: r.quatsch).pruefen()
ok(not b.ok and "nicht lesbar" in repr(b))
b = e_bad(1, call=lambda: 1 / 0).pruefen()
ok(not b.ok and "ZeroDivisionError" in repr(b))
b = e_bad(1, call=lambda: 42, lesen=lambda r: r).pruefen()
ok(not b.ok, "ein Zahlenwert ohne Rechenweg ist keine Aufgabe")
# strikte Vergleiche: True ist nicht 1, Gleitkommazahlen mit Toleranz
ok(not ex._gleich(True, 1) and ex._gleich(True, True) and ex._gleich(0.1 + 0.2, 0.3) and not ex._gleich(0.31, 0.3))
ok(ex._gleich([1, 2], [1, 2]) and not ex._gleich([1, 2], [2, 1]) and ex._gleich({"a": 1}, {"a": 1}))
ok(not ex._gleich(5, 5.5) and ex._gleich(Fraction(1, 3), Fraction(2, 6)))

# ==========================================================================
# 4. Unabhängige Gegenrechnung der wichtigsten Werte (Python, ohne dmatkit)
# ==========================================================================
# Serie 1
ok(bin(409)[2:] == "110011001")
ok(Fraction(21, 10) + Fraction(365, 999 * 100) == Fraction(42031, 19980))          # 2.1 + 0.00(365)
ok(dk.repeating_to_fraction("2.10(365)").value == Fraction(42031, 19980))
# Serie 5, Aufgabe 1: 15 / 25 / 120 und maximale Eingabe
ok(5 * math.ceil(math.log2(5)) == 15 and 5 ** 2 == 25 and math.factorial(5) == 120)
mx = ex.loesen("SW5.1b").n
minute = 60 * 10 ** 9
ok(mx["B"] ** 2 <= minute < (mx["B"] + 1) ** 2)
ok(math.factorial(mx["C"]) <= minute < math.factorial(mx["C"] + 1))
ok(mx["A"] * math.log2(mx["A"]) <= minute < (mx["A"] + 1) * math.log2(mx["A"] + 1))


# decToBin: T(x) = 11 + 7·⌊log2 x⌋, von Hand gezählt: 3 Anfangsschritte, 7 je Durchlauf, 1 Return
def zaehle_dec(x):
    s = 3
    temp = x
    while temp != 0:
        s += 7
        temp //= 2
    return s + 1


ok(all(zaehle_dec(x) == 11 + 7 * (x.bit_length() - 1) for x in range(1, 500)))
ok(zaehle_dec(25) == 39 and ex.loesen("SW5.2a").total == 39)


# Sortieralgorithmen: Formeln gegen unabhängig gezählte Werte
def inv(lst):
    return sum(1 for i in range(len(lst)) for j in range(i + 1, len(lst)) if lst[i] > lst[j])


for n in range(1, 8):
    absteigend = list(range(n, 0, -1))
    ok(n + 3 * n * (n - 1) + 5 * inv(absteigend) == (11 * n * n - 9 * n) // 2)       # Bubble, Worst-Case
    ok(ex.loesen("SW5.6a").at(n) == (11 * n * n - 9 * n) // 2)
    ok(ex.loesen("SW5.6b").at(n) == 3 * n * n - 2 * n)
    ok(ex.loesen("SW5.5a").at(n) == 5 * n * n - 2 * n - 3)
    ok(ex.loesen("SW5.5b").at(n) == 5 * n - 5)
    ok(ex.loesen("SW5.3").at(n) == 2 * n + 1)
    ok(ex.loesen("SW5.7a").at(n) == 8 * (n.bit_length() - 1) + 13)
    # Selection Sort: Vorlesungsschranke 5n + 3n(n−1)/2 liegt über dem echten Maximum
    ok(ex.loesen("SW5.4a").at(n) <= 5 * n + 3 * (n * (n - 1) // 2))
ok(5 * 5 + 3 * 10 == 55)
ok(ex.loesen("SW5.B1").value == sum(range(101)) == 5050)

# Serie 3, Aufgabe 5: Monitorwerte A bis F mit reinem Python nachgerechnet und gegen die Tabelle des Notebooks
M = ex.MONITORE
ok(set(M) == set("ABCDEF") and all(len(m) == 6 and all(len(z) == 6 for z in m) for m in M.values()))
ok(all(v == 1 for z in M["A"] for v in z))
ok(M["B"][2][1] == 1 and sum(map(sum, M["B"])) == 1)
ok(M["E"][2] == [1] * 6 and sum(map(sum, M["E"])) == 6)
ok(all(z[4] == 1 for z in M["F"]) and sum(map(sum, M["F"])) == 6)
R = range(6)
aussagen = {
    "a": lambda m: all(m[i][j] for i in R for j in R), "b": lambda m: all(m[i][j] for j in R for i in R),
    "c": lambda m: any(m[i][j] for i in R for j in R), "d": lambda m: any(m[i][j] for j in R for i in R),
    "e": lambda m: all(any(m[i][j] for j in R) for i in R), "f": lambda m: all(any(m[i][j] for i in R) for j in R),
    "g": lambda m: any(all(m[i][j] for j in R) for i in R), "h": lambda m: any(all(m[i][j] for i in R) for j in R),
}
for c, f in aussagen.items():
    res = ex.loesen(f"SW3.5{c}")
    ok([res.werte[n] for n in "ABCDEF"] == [bool(f(M[n])) for n in "ABCDEF"], c)
    ok([res.werte[n] for n in "ABCDEF"] == ex._NOTEBOOK[c], c)
ok({c: sum(ex._NOTEBOOK[c]) for c in "abcdefgh"} == {"a": 1, "b": 1, "c": 6, "d": 6, "e": 3, "f": 3, "g": 2, "h": 2})

# Fisch-Rätsel (Serie 4, Prolog): unabhängig per Brute-Force über Permutationen gelöst
Nat = ["Brite", "Schwede", "Däne", "Norweger", "Deutsche"]
Farbe = ["rot", "grün", "weiss", "gelb", "blau"]
Getr = ["Tee", "Kaffee", "Milch", "Bier", "Wasser"]
Zig = ["PallMall", "Dunhill", "Marlboro", "Winfield", "Rothmans"]
Tier = ["Hund", "Vogel", "Katze", "Pferd", "Fisch"]
loesungen = []
for n in itertools.permutations(Nat):
    if n[0] != "Norweger":
        continue
    for f in itertools.permutations(Farbe):
        if f[n.index("Brite")] != "rot" or f.index("grün") + 1 != f.index("weiss") or abs(n.index("Norweger") - f.index("blau")) != 1:
            continue
        for gtr in itertools.permutations(Getr):
            if gtr[n.index("Däne")] != "Tee" or gtr[f.index("grün")] != "Kaffee" or gtr[2] != "Milch":
                continue
            for z in itertools.permutations(Zig):
                if z[f.index("gelb")] != "Dunhill" or gtr[z.index("Winfield")] != "Bier" or z[n.index("Deutsche")] != "Rothmans":
                    continue
                for t in itertools.permutations(Tier):
                    if (t[n.index("Schwede")] == "Hund" and t[z.index("PallMall")] == "Vogel"
                            and abs(z.index("Marlboro") - t.index("Katze")) == 1 and abs(t.index("Pferd") - z.index("Dunhill")) == 1
                            and abs(z.index("Marlboro") - gtr.index("Wasser")) == 1):
                        loesungen.append(n[t.index("Fisch")])
ok(loesungen == ["Deutsche"], loesungen)
ok("Deutsche" in ex.loesen("SW4.P1").answer and len(ex.loesen("SW4.P1").solutions) == 1)
ok(len(ex.loesen("SW4.P2").solutions) == 1)

# Serie 4: Tabelle der Serie und Resolution der Mitschrift
WB = lambda a, b, c: (a or b) and (c or (not b and not a) or b)
tab = {(1, 1, 1): (1, 1), (1, 1, 0): (1, 1), (1, 0, 1): (1, 1), (1, 0, 0): (0, 1), (0, 1, 1): (1, 1), (0, 1, 0): (1, 1), (0, 0, 1): (0, 0), (0, 0, 0): (0, 0)}
ok(all(tab[b] == (int(bool(WB(*map(bool, b)))), int(bool(b[0] or b[1]))) for b in tab))
res = ex.loesen("SW4.5")
ok(res.derived_empty and [set(k) for k in res.clauses[:4]] == [{"S", "N"}, {"¬S"}, {"P", "N"}, {"¬N", "P"}])
ok(set(res.clauses[5]) == {"N"} and set(res.clauses[6]) == {"P"})

# Serie 3, Aufgabe 6: Negationen (Strings von Hand hergeleitet)
ok(str(ex.loesen("SW3.6a").result) == "∀y ∈ R: ∃x ∈ R: x ≥ y")
ok(str(ex.loesen("SW3.6b").result) == "∃x ∈ Z: ∀y ∈ Z: x ≠ y + 1")
ok(str(ex.loesen("SW3.6c").result) == "∀x ∈ U: ∃y ∈ U: ∀z ∈ U: ¬P(x, y, z)")

# ==========================================================================
# 5. Stabilität: zweimal rechnen gibt dasselbe
# ==========================================================================
a1 = [(k, ex.pruefen(k).ok) for k in ex.EXERCISES]
a2 = [(k, ex.pruefen(k).ok) for k in ex.EXERCISES]
ok(a1 == a2 and all(v for _, v in a1))
ok(repr(ex.loesen("SW5.1a")) == repr(ex.loesen("SW5.1a")))

print(f"tests_exercises.py: {N[0]} Prüfungen bestanden, {len(ex.EXERCISES)} Aufgaben ("
      + ", ".join(f"SW{w}: {len(wochen[w])}" for w in range(1, 7)) + ")")
