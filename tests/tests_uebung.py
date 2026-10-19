#!/usr/bin/env python3
"""Tests für dmatkit.aufgaben.uebung: python3 tests/tests_uebung.py

Die Referenzlösungen stehen nur hier im Test. Geprüft wird: richtige Lösungen bestehen, typische Fehler
(Off-by-one, vertauschte Quantoren, vergessene Randfälle) fallen durch, Schutz gegen Endlosschleifen,
Ausnahmen, falsche Rückgabetypen und Veränderung der Eingabe.
"""
import os
import sys
import time
import itertools
import threading
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import dmatkit as dk
from dmatkit.aufgaben import uebung as u

N = [0]


def ok(cond, msg=""):
    N[0] += 1
    assert cond, msg


def gut(pruefer, f, msg=""):
    p = pruefer(f, ausgabe=False)
    N[0] += 1
    assert p.ok, f"{msg}: richtige Lösung abgelehnt\n{p.bericht()}"
    return p


def schlecht(pruefer, f, msg="", fall=None):
    p = pruefer(f, ausgabe=False)
    N[0] += 1
    assert not p.ok, f"{msg}: falsche Lösung akzeptiert\n{p.bericht()}"
    if fall is not None:
        N[0] += 1
        namen = [x.name for x in p.faelle if not x.ok]
        assert any(fall in n for n in namen), f"{msg}: erwarteter Fall {fall!r} nicht unter den Fehlern {namen}"
    return p


# ==========================================================================
# Referenzlösungen (richtig) und typische Fehler
# ==========================================================================
def r_dec(n):
    y, t = '', n
    while t != 0:
        y = str(t % 2) + y
        t //= 2
    return y


def f_dec_umgekehrt(n):
    y, t = '', n
    while t != 0:
        y = y + str(t % 2)
        t //= 2
    return y


def f_dec_int(n):
    return int(bin(n)[2:] or 0)


def f_dec_null(n):                       # Fehler: 0 ergibt '0' + führende Null bei allen? hier: führende Null immer
    return '0' + r_dec(n)


def f_dec_endlos(n):
    y, t = '', n
    while t != 0:
        y = str(t % 2) + y                # t wird nie verändert
    return y


def f_dec_off(n):                         # Fehler: Zahl 1 liefert ''
    y, t = '', n
    while t > 1:
        y = str(t % 2) + y
        t //= 2
    return y


def r_bin(t):
    w = 0
    for z in t:
        w = 2 * w + int(z)
    return w


def f_bin_umgekehrt(t):
    w = 0
    for i, z in enumerate(t):
        w += int(z) * 2 ** i
    return w


def f_bin_off(t):
    w = 0
    for z in t[:-1]:
        w = 2 * w + int(z)
    return w


def f_bin_str(t):
    return str(r_bin(t))


def r_forall(uni, p):
    for x in uni:
        if not p(x):
            return False
    return True


def r_exists(uni, p):
    for x in uni:
        if p(x):
            return True
    return False


def f_forall_vertauscht(uni, p):        # ∃ statt ∀
    return r_exists(uni, p)


def f_exists_vertauscht(uni, p):        # ∀ statt ∃
    return r_forall(uni, p)


def f_forall_off(uni, p):               # letztes Element vergessen
    uni = list(uni)
    for x in uni[:-1]:
        if not p(x):
            return False
    return True


def f_exists_off(uni, p):
    uni = list(uni)
    for x in uni[:-1]:
        if p(x):
            return True
    return False


def f_forall_leer(uni, p):              # leere Menge ergibt False
    uni = list(uni)
    return bool(uni) and all(p(x) for x in uni)


def f_exists_leer(uni, p):              # leere Menge ergibt True
    uni = list(uni)
    return (not uni) or any(p(x) for x in uni)


def f_forall_none(uni, p):              # return vergessen
    for x in uni:
        if not p(x):
            return False


def f_forall_nur_erstes(uni, p):
    for x in uni:
        return bool(p(x))
    return True


def r_dup(a):
    n = len(a)
    for i in range(n):
        for j in range(n):
            if i != j and a[i] == a[j]:
                return True
    return False


def f_dup_ohne_ij(a):                   # vergisst ¬(i = j)
    for i in range(len(a)):
        for j in range(len(a)):
            if a[i] == a[j]:
                return True
    return False


def f_dup_off(a):                       # j läuft nicht bis zum Ende
    for i in range(len(a)):
        for j in range(i + 1, len(a) - 1):
            if a[i] == a[j]:
                return True
    return False


def f_dup_sortiert(a):                  # verändert die Eingabe
    a.sort()
    return any(a[i] == a[i + 1] for i in range(len(a) - 1))


def f_dup_nachbarn(a):                  # nur direkte Nachbarn
    return any(a[i] == a[i + 1] for i in range(len(a) - 1))


def r_pal(s):
    i, j = 0, len(s) - 1
    while i < j:
        if s[i] != s[j]:
            return False
        i += 1
        j -= 1
    return True


def f_pal_halb(s):                      # prüft nur das erste und das letzte Zeichen
    return len(s) == 0 or s[0] == s[-1]


def f_pal_lower(s):
    s = s.lower()
    return s == s[::-1]


def f_pal_leer(s):
    return len(s) > 0 and s == s[::-1]


def r_tab(f, n):
    return [(b, f(*b)) for b in itertools.product([False, True], repeat=n)]


def f_tab_zeile_fehlt(f, n):
    return r_tab(f, n)[:-1]


def f_tab_falsch(f, n):
    z = r_tab(f, n)
    b, w = z[-1]
    z[-1] = (b, not w)
    return z


def f_tab_zahlen(f, n):                 # Binärzahlen 0 … 2^n−1, Rückgabe in anderer Reihenfolge (erlaubt)
    zeilen = []
    for k in reversed(range(2 ** n)):
        b = tuple(bool((k >> (n - 1 - i)) & 1) for i in range(n))
        zeilen.append((b, f(*b)))
    return zeilen


def f_tab_einsen(f, n):                 # 0/1 statt False/True (erlaubt)
    return [(tuple(int(x) for x in b), int(w)) for b, w in r_tab(f, n)]


def f_tab_nur_werte(f, n):
    return [w for _, w in r_tab(f, n)]


def r_folg(wb, q, n):
    for b in itertools.product([False, True], repeat=n):
        if wb(*b) and not q(*b):
            return False
    return True


def f_folg_alle_q(wb, q, n):            # Fehler: alle Zeilen müssen Q wahr haben
    return all(q(*b) for b in itertools.product([False, True], repeat=n))


def f_folg_aequiv(wb, q, n):            # Fehler: prüft Äquivalenz
    return all(bool(wb(*b)) == bool(q(*b)) for b in itertools.product([False, True], repeat=n))


def f_folg_umgekehrt(wb, q, n):         # Fehler: Q ⊨ WB
    return r_folg(q, wb, n)


def f_folg_sat(wb, q, n):               # Fehler: verlangt eine Zeile mit WB = 1
    zeilen = [b for b in itertools.product([False, True], repeat=n) if wb(*b)]
    return bool(zeilen) and all(q(*b) for b in zeilen)


def f_folg_eine(wb, q, n):              # Fehler: eine Zeile reicht
    return any(q(*b) for b in itertools.product([False, True], repeat=n) if wb(*b))


def r_res(k1, k2):
    out = []
    for l in k1:
        g = l[1:] if l.startswith('¬') else '¬' + l
        if g in k2:
            out.append((k1 - {l}) | (k2 - {g}))
    return out


def f_res_nur_erste(k1, k2):            # Fehler: nur ein Paar
    for l in k1:
        g = l[1:] if l.startswith('¬') else '¬' + l
        if g in k2:
            return [(k1 - {l}) | (k2 - {g})]
    return []


def f_res_mutiert(k1, k2):              # Fehler: verändert die Eingabe
    out = []
    for l in list(k1):
        g = l[1:] if l.startswith('¬') else '¬' + l
        if g in k2:
            k1.remove(l)
            k2.remove(g)
            out.append(k1 | k2)
    return out


def f_res_nur_k1(k1, k2):               # Fehler: sucht das Gegenstück nur in einer Richtung (¬L in k1, L in k2 fehlt)
    out = []
    for l in k1:
        if not l.startswith('¬') and '¬' + l in k2:
            out.append((k1 - {l}) | (k2 - {'¬' + l}))
    return out


def f_res_ohne_rest(k1, k2):            # Fehler: lässt den Rest von k2 weg
    return [k1 - {l} for l in k1 if (l[1:] if l.startswith('¬') else '¬' + l) in k2]


def f_res_eine_klausel(k1, k2):         # Fehler: gibt eine Klausel statt einer Liste zurück
    r = r_res(k1, k2)
    return r[0] if r else set()


def f_res_ohne_filter(k1, k2):          # erlaubt: gibt auch tautologische Resolventen aus
    return r_res(k1, k2)


def r_lin(a, x):
    s = 0
    for i in range(len(a)):
        s += 1
        s += 1
        if a[i] == x:
            s += 1
            return True, s
    s += 1
    return False, s


def f_lin_ohne_return(a, x):            # Fehler: return nicht gezählt
    g, s = r_lin(a, x)
    return g, s - 1


def f_lin_nur_bool(a, x):
    return r_lin(a, x)[0]


def f_lin_doppelt(a, x):                # Fehler: zählt len(a) mit
    g, s = r_lin(a, x)
    return g, s + 1


def f_lin_off(a, x):                    # Fehler: letzte Position fehlt
    for i in range(len(a) - 1):
        if a[i] == x:
            return True, 2 * (i + 1) + 1
    return False, 2 * (len(a) - 1) + 1


def r_sel(a):
    n = len(a)
    for i in range(n):
        m = i
        for j in range(i + 1, n):
            if a[j] < a[m]:
                m = j
        a[i], a[m] = a[m], a[i]
    return a


def r_sel_inplace(a):                    # ohne return: erlaubt
    r_sel(a)


def f_sel_off(a):                       # Fehler: letzte Runde fehlt, falsche Ausgabe bei Maximum hinten ist ok, aber [2,1] bleibt
    n = len(a)
    for i in range(n - 2):
        m = i
        for j in range(i + 1, n):
            if a[j] < a[m]:
                m = j
        a[i], a[m] = a[m], a[i]
    return a


def f_sel_bubble_fehler(a):             # Fehler: tauscht nicht mit dem Minimum, sondern mit a[j]
    n = len(a)
    for i in range(n):
        for j in range(i + 1, n):
            if a[j] < a[i]:
                a[i] = a[j]             # überschreibt statt zu tauschen: Elemente gehen verloren
    return a


def f_sel_absteigend(a):
    return r_sel(a)[::-1]


def f_sel_none(a):
    r_sel(a)


def f_sel_tuple(a):
    return tuple(r_sel(a))


def r_klass(f, n):
    w = [bool(f(*b)) for b in itertools.product([False, True], repeat=n)]
    return 'Tautologie' if all(w) else ('Kontradiktion' if not any(w) else 'Kontingenz')


def f_klass_zwei(f, n):                 # Fehler: kennt Kontingenz nicht
    w = [bool(f(*b)) for b in itertools.product([False, True], repeat=n)]
    return 'Tautologie' if all(w) else 'Kontradiktion'


def r_pot(menge):
    ts = [set()]
    for e in menge:
        ts += [t | {e} for t in ts]
    return ts


def f_pot_ohne_leer(menge):
    return r_pot(menge)[1:]


def f_pot_doppelt(menge):
    return r_pot(menge) + [set(menge)]


def r_dez(t):
    ganz, _, rest = t.partition('.')
    vor, per = rest, ''
    if '(' in rest:
        vor, _, per = rest.partition('(')
        per = per.rstrip(')')
    w = Fraction(int(ganz))
    if vor:
        w += Fraction(int(vor), 10 ** len(vor))
    if per:
        w += Fraction(int(per), (10 ** len(per) - 1) * 10 ** len(vor))
    return w


def f_dez_float(t):
    return float(r_dez(t))


def f_dez_ohne_vor(t):                  # Fehler: ignoriert die Vorperiode
    ganz, _, rest = t.partition('.')
    if '(' not in rest:
        return r_dez(t)
    vor, _, per = rest.partition('(')
    per = per.rstrip(')')
    return Fraction(int(ganz)) + Fraction(int(per), 10 ** len(per) - 1)


def f_dez_tuple_ungekuerzt(t):
    w = r_dez(t)
    return (w.numerator * 2, w.denominator * 2)


def f_dez_tuple(t):
    w = r_dez(t)
    return (w.numerator, w.denominator)


def r_nk(z, n):
    q = Fraction(z, n)
    g = int(q)
    t = bin(g)[2:]
    r = q - g
    if r == 0:
        return t
    st, gesehen = [], {}
    while r != 0 and r not in gesehen:
        gesehen[r] = len(st)
        r *= 2
        st.append('1' if r >= 1 else '0')
        if r >= 1:
            r -= 1
    if r == 0:
        return t + '.' + ''.join(st)
    i = gesehen[r]
    return t + '.' + ''.join(st[:i]) + '(' + ''.join(st[i:]) + ')'


def f_nk_ohne_periode(z, n):            # Fehler: bricht nach 8 Stellen ab
    r = Fraction(z, n)
    t = bin(int(r))[2:] + '.'
    r -= int(r)
    for _ in range(8):
        r *= 2
        t += '1' if r >= 1 else '0'
        if r >= 1:
            r -= 1
    return t


# ==========================================================================
# 1. Vorlagen
# ==========================================================================
ok(len(u.VORLAGEN) >= 11 and set(u.namen()) == set(u.VORLAGEN))
for nm in ("meine_dec_to_bin", "meine_binaer_zu_dezimal", "mein_forall", "mein_exists", "meine_hat_duplikate", "mein_palindrom",
           "meine_wahrheitstabelle", "meine_folgerung", "meine_resolvente", "meine_lineare_suche", "mein_selection_sort"):
    ok(nm in u.VORLAGEN, nm)
    src = u.vorlage(nm)
    ok(src.startswith("def " + nm), nm)
    compile(src, nm, "exec")                      # gültiges Python
    ns = {}
    exec(src, ns)                                 # Vorlage selbst ist ausführbar (nur eigener Test-Quelltext)
    ok(callable(ns[nm]))
for nm in u.namen():
    compile(u.vorlage(nm), nm, "exec")
ok("def meine_dec_to_bin" in u.vorlage())
try:
    u.vorlage("quatsch")
    ok(False, "ValueError erwartet")
except ValueError as e:
    ok("Kein Gerüst" in str(e))
# Vorlagen enthalten keine fertige Lösung: unvollständige Vorlagen bestehen die Prüfung nicht
for nm, pr in [("meine_dec_to_bin", u.pruefe_dec_to_bin), ("meine_binaer_zu_dezimal", u.pruefe_binaer_zu_dezimal),
               ("mein_forall", u.pruefe_forall), ("mein_exists", u.pruefe_exists), ("meine_hat_duplikate", u.pruefe_hat_duplikate),
               ("mein_palindrom", u.pruefe_palindrom), ("meine_wahrheitstabelle", u.pruefe_wahrheitstabelle),
               ("meine_folgerung", u.pruefe_folgerung), ("meine_resolvente", u.pruefe_resolvente),
               ("meine_lineare_suche", u.pruefe_lineare_suche), ("mein_selection_sort", u.pruefe_selection_sort),
               ("meine_klassifikation", u.pruefe_klassifikation), ("meine_potenzmenge", u.pruefe_potenzmenge),
               ("meine_dezimal_zu_bruch", u.pruefe_dezimal_zu_bruch), ("meine_binaer_nachkomma", u.pruefe_binaer_nachkomma)]:
    ns = {}
    exec(u.vorlage(nm), ns)
    if nm == "meine_dec_to_bin":
        continue                                  # Vorlage hat while-Schleife ohne Fortschritt (Endlosschleife, siehe eigener Test)
    p = pr(ns[nm], ausgabe=False)
    ok(not p.ok, f"Vorlage {nm} darf nicht schon bestehen")

# ==========================================================================
# 2. SW1
# ==========================================================================
p = gut(u.pruefe_dec_to_bin, r_dec, "dec_to_bin")
ok(p.gesamt == 10 and p.bestanden == 10 and bool(p))
gut(u.pruefe_dec_to_bin, lambda n: bin(n)[2:] if n else '0', "dec_to_bin mit '0'")
schlecht(u.pruefe_dec_to_bin, f_dec_umgekehrt, "umgekehrt", "409")
schlecht(u.pruefe_dec_to_bin, f_dec_int, "int statt str")
schlecht(u.pruefe_dec_to_bin, f_dec_null, "führende Null")
schlecht(u.pruefe_dec_to_bin, f_dec_off, "Off-by-one bei 1", "n = 1")
t0 = time.time()
p = schlecht(u.pruefe_dec_to_bin, f_dec_endlos, "Endlosschleife")
ok(time.time() - t0 < 15, "Zeitlimit greift")
ok(any("Zeitlimit" in f.text for f in p.faelle))
st = [s for s in p.stolpersteine if s.name == "Negative Zahl"]
p_neg = u.pruefe_dec_to_bin(r_dec, ausgabe=False)
ok(any("Endlosschleife" in s.text for s in p_neg.stolpersteine), "negative Zahl: Endlosschleife als Stolperstein, kein Fehler")
ok(p_neg.ok)
gut(u.pruefe_binaer_zu_dezimal, r_bin, "binär zu dezimal")
gut(u.pruefe_binaer_zu_dezimal, lambda t: int(t, 2), "int(t, 2)")
schlecht(u.pruefe_binaer_zu_dezimal, f_bin_umgekehrt, "umgekehrt", "110011001")
schlecht(u.pruefe_binaer_zu_dezimal, f_bin_off, "letzte Stelle fehlt")
schlecht(u.pruefe_binaer_zu_dezimal, f_bin_str, "Text statt Zahl")
def mit_int(t):
    return int(t, 2)


p = u.pruefe_binaer_zu_dezimal(mit_int, ausgabe=False)
ok(p.ok and any("fertige Lösung" in h for h in p.hinweise), "int(..., 2) wird als fertige Lösung bemerkt (Quelltext lesbar)")

# ==========================================================================
# 3. SW3
# ==========================================================================
gut(u.pruefe_forall, r_forall, "forall")
gut(u.pruefe_exists, r_exists, "exists")
gut(u.pruefe_forall, lambda uni, p: all(p(x) for x in uni), "forall mit all")
gut(u.pruefe_exists, lambda uni, p: any(p(x) for x in uni), "exists mit any")
schlecht(u.pruefe_forall, f_forall_vertauscht, "∀ und ∃ vertauscht")
schlecht(u.pruefe_exists, f_exists_vertauscht, "∃ und ∀ vertauscht")
schlecht(u.pruefe_forall, f_forall_off, "∀ letztes Element vergessen", "letzte")
schlecht(u.pruefe_exists, f_exists_off, "∃ letztes Element vergessen", "letzte")
schlecht(u.pruefe_forall, f_forall_leer, "∀ leere Menge", "leeres")
schlecht(u.pruefe_exists, f_exists_leer, "∃ leere Menge", "leeres")
p = schlecht(u.pruefe_forall, f_forall_none, "return vergessen")
ok(any("bool" in f.text for f in p.faelle))
schlecht(u.pruefe_forall, f_forall_nur_erstes, "nur erstes Element")
schlecht(u.pruefe_forall, lambda uni, p: 1 if all(p(x) for x in uni) else 0, "int statt bool")
p = u.pruefe_forall(r_forall, ausgabe=False)
ok(any(s.name == "Frühes Abbrechen" and s.ok for s in p.stolpersteine))
p = u.pruefe_forall(lambda uni, pr: all([pr(x) for x in uni]), ausgabe=False)
ok(any(s.name == "Frühes Abbrechen" and not s.ok for s in p.stolpersteine), "Liste statt Generator wertet alles aus")
gut(u.pruefe_hat_duplikate, r_dup, "Duplikate")
gut(u.pruefe_hat_duplikate, lambda a: len(set(a)) < len(a), "Duplikate mit set")
schlecht(u.pruefe_hat_duplikate, f_dup_ohne_ij, "i = j vergessen", "ein Element")
schlecht(u.pruefe_hat_duplikate, f_dup_off, "Off-by-one", "Ende")
schlecht(u.pruefe_hat_duplikate, f_dup_sortiert, "sortiert die Eingabe")
schlecht(u.pruefe_hat_duplikate, f_dup_nachbarn, "nur Nachbarn", "erstes und letztes")
gut(u.pruefe_palindrom, r_pal, "Palindrom")
gut(u.pruefe_palindrom, lambda s: s == s[::-1], "Palindrom mit Slice")
schlecht(u.pruefe_palindrom, f_pal_halb, "nur äussere Zeichen")
schlecht(u.pruefe_palindrom, f_pal_lower, "Kleinschreibung", "Aa")
schlecht(u.pruefe_palindrom, f_pal_leer, "leerer Text", "''")

# ==========================================================================
# 4. SW2 / SW4
# ==========================================================================
gut(u.pruefe_wahrheitstabelle, r_tab, "Tabelle")
gut(u.pruefe_wahrheitstabelle, f_tab_zahlen, "Tabelle in anderer Reihenfolge")
gut(u.pruefe_wahrheitstabelle, f_tab_einsen, "Tabelle mit 0 und 1")
schlecht(u.pruefe_wahrheitstabelle, f_tab_zeile_fehlt, "Zeile fehlt")
schlecht(u.pruefe_wahrheitstabelle, f_tab_falsch, "falscher Wert")
schlecht(u.pruefe_wahrheitstabelle, f_tab_nur_werte, "nur Werte")
gut(u.pruefe_folgerung, r_folg, "Folgerung")
for fn, nm in [(f_folg_alle_q, "alle Zeilen Q"), (f_folg_aequiv, "Äquivalenz statt Folgerung"), (f_folg_umgekehrt, "Richtung vertauscht"),
               (f_folg_sat, "WB muss erfüllbar sein"), (f_folg_eine, "eine Zeile reicht")]:
    schlecht(u.pruefe_folgerung, fn, nm)
p = schlecht(u.pruefe_folgerung, f_folg_sat, "WB widersprüchlich")
ok(any("widersprüchlich" in f.name for f in p.faelle if not f.ok))
gut(u.pruefe_klassifikation, r_klass, "Klassifikation")
schlecht(u.pruefe_klassifikation, f_klass_zwei, "Kontingenz fehlt")
gut(u.pruefe_resolvente, r_res, "Resolvente")
gut(u.pruefe_resolvente, f_res_ohne_filter, "mit tautologischen Resolventen")
gut(u.pruefe_resolvente, lambda k1, k2: [x for x in r_res(k1, k2) if not any(('¬' + l) in x for l in x)], "ohne Tautologien")
gut(u.pruefe_resolvente, lambda k1, k2: [frozenset(x) for x in r_res(k1, k2)], "frozenset als Rückgabe")
# zwei komplementäre Paare liefern nur tautologische Resolventen: "nur das erste Paar" ist darum nicht unterscheidbar und erlaubt
gut(u.pruefe_resolvente, f_res_nur_erste, "nur erstes Paar (gleichwertig)")
schlecht(u.pruefe_resolvente, f_res_mutiert, "verändert die Eingabe")
schlecht(u.pruefe_resolvente, f_res_nur_k1, "nur eine Richtung", "Umgekehrte Reihenfolge")
schlecht(u.pruefe_resolvente, f_res_ohne_rest, "Rest von k2 fehlt")
p = schlecht(u.pruefe_resolvente, f_res_eine_klausel, "eine Klausel statt Liste")
schlecht(u.pruefe_resolvente, lambda k1, k2: [], "nie eine Resolvente", "Serie 4")
schlecht(u.pruefe_resolvente, lambda k1, k2: [{"A"}], "immer dasselbe")
leere = u.pruefe_resolvente(lambda k1, k2: [x for x in r_res(k1, k2) if x], ausgabe=False)
ok(any(not f.ok and "Leere Klausel" in f.name for f in leere.faelle), "die leere Klausel darf nicht verschwinden")

# ==========================================================================
# 5. SW5
# ==========================================================================
# Die Referenzzählung stimmt mit der Formel der Vorlesung und mit dk.count_steps überein.
for a, x in [([4, 8, 15, 16], 8), ([4, 8, 15, 16], 23), ([], 1), ([5], 5), ([1, 2, 2, 2], 2), (list(range(10)), -1)]:
    ref = u._schritte_linear_dk(a, x)
    ok(ref == r_lin(a, x)[1], (a, x, ref))
    ok(ref == dk.count_steps("linearSearch", Liste=a, Element=x).total)
    n = len(a)
    k = a.index(x) if x in a else None
    ok(ref == (2 * (k + 1) + 1 if k is not None else 2 * n + 1))
gut(u.pruefe_lineare_suche, r_lin, "lineare Suche")
gut(u.pruefe_lineare_suche, lambda a, x: [r_lin(a, x)[0], r_lin(a, x)[1]], "Rückgabe als Liste")
schlecht(u.pruefe_lineare_suche, f_lin_ohne_return, "return nicht gezählt")
schlecht(u.pruefe_lineare_suche, f_lin_nur_bool, "nur bool")
schlecht(u.pruefe_lineare_suche, f_lin_doppelt, "zu viele Schritte")
schlecht(u.pruefe_lineare_suche, f_lin_off, "letzte Position vergessen")
schlecht(u.pruefe_lineare_suche, lambda a, x: (x in a, 0), "keine Zählung")
schlecht(u.pruefe_lineare_suche, lambda a, x: (1, 2 * len(a) + 1), "int statt bool")
gut(u.pruefe_selection_sort, r_sel, "Selection Sort")
gut(u.pruefe_selection_sort, r_sel_inplace, "Selection Sort ohne return")
gut(u.pruefe_selection_sort, lambda a: sorted(a), "neue Liste")
schlecht(u.pruefe_selection_sort, f_sel_off, "letzte Runde fehlt", "zwei, falsch herum")
schlecht(u.pruefe_selection_sort, f_sel_bubble_fehler, "überschreibt statt zu tauschen")
schlecht(u.pruefe_selection_sort, f_sel_absteigend, "absteigend")
schlecht(u.pruefe_selection_sort, f_sel_tuple, "Tupel statt Liste")
schlecht(u.pruefe_selection_sort, lambda a: a, "tut nichts")
schlecht(u.pruefe_selection_sort, lambda a: [], "leere Liste statt sortierte")

# Extras
gut(u.pruefe_potenzmenge, r_pot, "Potenzmenge")
gut(u.pruefe_potenzmenge, lambda m: [frozenset(t) for t in r_pot(m)], "Potenzmenge mit frozenset")
schlecht(u.pruefe_potenzmenge, f_pot_ohne_leer, "∅ fehlt")
schlecht(u.pruefe_potenzmenge, f_pot_doppelt, "doppelte Teilmenge")
gut(u.pruefe_dezimal_zu_bruch, r_dez, "Dezimal zu Bruch")
gut(u.pruefe_dezimal_zu_bruch, f_dez_tuple, "Dezimal zu Bruch als Paar")
schlecht(u.pruefe_dezimal_zu_bruch, f_dez_float, "float statt Bruch")
schlecht(u.pruefe_dezimal_zu_bruch, f_dez_ohne_vor, "Vorperiode ignoriert")
schlecht(u.pruefe_dezimal_zu_bruch, f_dez_tuple_ungekuerzt, "nicht gekürzt")
ok(r_dez("2.10(365)") == Fraction(42031, 19980), "Vorlesung: 2.10(365) = 42031/19980")
ok(u._periodisch_ref("2.10(365)") == Fraction(42031, 19980))
gut(u.pruefe_binaer_nachkomma, r_nk, "Binärbruch")
schlecht(u.pruefe_binaer_nachkomma, f_nk_ohne_periode, "ohne Periode")
# Referenz im Modul stimmt mit dk überein
for z, n in [(1, 10), (1, 3), (3, 8), (5, 6), (1, 2), (7, 4), (1, 5), (0, 1)]:
    ok(u._binaer_nachkomma_ref(z, n) == dk.frac_to_binary(f"{z}/{n}").text, (z, n))

# ==========================================================================
# 6. Schutz: Zeitlimit, Ausnahmen, Typen, Veränderung, keine Funktion, Ausgabe
# ==========================================================================
def wirft(*a):
    raise ValueError("kaputt")


p = schlecht(u.pruefe_palindrom, wirft, "Ausnahme")
ok(all("ValueError" in f.text for f in p.faelle), "Ausnahme wird mit Typ gemeldet")
p = schlecht(u.pruefe_palindrom, lambda s: "ja", "falscher Rückgabetyp")
ok(any("Rückgabetyp" in f.text for f in p.faelle))
p = schlecht(u.pruefe_palindrom, 42, "keine Funktion")
ok(all("keine Funktion" in f.text for f in p.faelle))
p = schlecht(u.pruefe_palindrom, lambda: True, "falsche Parameterzahl")
ok(all("TypeError" in f.text for f in p.faelle))


def rekursiv(s):
    return rekursiv(s)


p = schlecht(u.pruefe_palindrom, rekursiv, "Rekursion ohne Ende")
ok(all("RecursionError" in f.text for f in p.faelle))


def endlos(s):
    while True:
        pass


t0 = time.time()
u.ZEITLIMIT = 0.2                                  # kurz für den Test
p = u.pruefe_palindrom(endlos, ausgabe=False)
u.ZEITLIMIT = 2.0
ok(not p.ok and time.time() - t0 < 15)
ok(all("Zeitlimit" in f.text for f in p.faelle))


def fange_alles(s):
    while True:
        try:
            while True:
                pass
        except Exception:
            pass


# _Zeit ist BaseException: ein `except Exception` im Schülercode verschluckt den Abbruch nicht
status, _ = u._mit_zeitlimit(fange_alles, ("x",), 0.2)
ok(status == "zeit")


def druckt(s):
    print("Lärm")
    return s == s[::-1]


import io, contextlib
puffer = io.StringIO()
with contextlib.redirect_stdout(puffer):
    u.pruefe_palindrom(druckt, ausgabe=False)
ok("Lärm" not in puffer.getvalue(), "print im Schülercode stört nicht")

# Signal-Zustand wird sauber zurückgesetzt
import signal
alt = signal.getsignal(signal.SIGALRM)
u._mit_zeitlimit(lambda: 1, (), 0.2)
ok(signal.getsignal(signal.SIGALRM) == alt and signal.getitimer(signal.ITIMER_REAL)[0] == 0)

# Thread-Fallback (kein Hauptthread): Endlosschleife wird trotzdem abgebrochen
ergebnis = {}


def im_thread():
    ergebnis["a"] = u._mit_zeitlimit(lambda: sum(range(10)), (), 1.0)
    ergebnis["b"] = u._mit_zeitlimit(endlos, ("x",), 0.3)
    ergebnis["c"] = u._mit_zeitlimit(wirft, (), 1.0)


t = threading.Thread(target=im_thread)
t.start()
t.join(20)
ok(ergebnis["a"] == ("ok", 45) and ergebnis["b"][0] == "zeit" and ergebnis["c"][0] == "fehler")

# Eingabe wird als Kopie übergeben: die Originalliste des Aufrufers ändert sich nie
orig = [3, 1, 2]
u._lauf(u.Pruefung("t"), lambda a: a.sort(), "x", (orig,), None, lambda i, s: None)
ok(orig == [3, 1, 2])

# Ausgabe: ausgabe=False druckt nichts, ausgabe=True/None druckt
puffer = io.StringIO()
with contextlib.redirect_stdout(puffer):
    u.pruefe_palindrom(r_pal, ausgabe=False)
ok(puffer.getvalue() == "")
puffer = io.StringIO()
with contextlib.redirect_stdout(puffer):
    u.pruefe_palindrom(r_pal)
ok("Palindrom" in puffer.getvalue() and "Alles richtig" in puffer.getvalue())

# Berichte: Text und HTML
p = u.pruefe_palindrom(f_pal_halb, ausgabe=False)
txt = p.bericht()
ok("✕" in txt and "Erwartet" in txt and "Erhalten" in txt and "von" in txt.splitlines()[0])
ok(len(p.bericht(ausfuehrlich=False)) <= len(txt))
h = p._repr_html_()
ok(h.startswith("<div") and "Palindrom" in h and "✕" in h)
ok(repr(p) == txt)
p = u.pruefe_palindrom(r_pal, ausgabe=False)
ok(bool(p) and p.bestanden == p.gesamt and "Alles richtig" in p.bericht() and "✕" not in p.bericht())

# Referenzlösungen im Test sind unabhängig von dmatkit: Spotchecks gegen Python-Builtins
for n in range(0, 300):
    ok(r_dec(n) == (bin(n)[2:] if n else ''))
    ok(r_bin(bin(n)[2:]) == n)
ok(r_dec(409) == "110011001")

print(f"tests_uebung.py: {N[0]} Prüfungen bestanden")
