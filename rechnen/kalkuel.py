#!/usr/bin/env python3
"""
Beweiskalküle (Semesterwoche 4): Folgern, Ableiten, KNF, Resolution.

Die Vorlesung unterscheidet zwei Dinge, die am Ende dasselbe liefern:

    WB ⊨ Q   semantisch folgern   (Bedeutung: Wahrheitstabelle, Interpretationen)
    WB ⊢ Q   syntaktisch ableiten (nur Symbole und Regeln eines Kalküls)

Funktionen (alle liefern ein Ergebnisobjekt mit Rechenweg):

    entails(wb, q, method="all")      WB ⊨ Q mit den drei Prüfmethoden der Vorlesung
                                      (Pfeilmethode, Deduktionstheorem, Widerspruchsbeweis)
    derive(wb, q, rules=None)         WB ⊢ Q mit den Schlussregeln (Modus ponens, …)
    to_cnf(f)                         Umrechnung in KNF mit benannten Rechengesetzen
    resolution(wb, q)                 Resolutionskalkül: WB ∧ ¬Q → leere Klausel □
    refute(clauses)                   Resolution auf einer fertigen Klauselmenge
    tautology_by_laws(f)              Tautologie mit den Rechengesetzen nachweisen
    entails_by_laws(wb, q)            WB ⊨ Q mit den Rechengesetzen (Aufgaben 1–3)
    contradiction_by_laws(f)          Kontradiktion mit den Rechengesetzen (Aufgabe 2)
    semantic_vs_syntactic(wb, q)      ⊨ und ⊢ am selben Beispiel nebeneinander
    kalkuel_info()                    Begriffe: Kalkül, korrekt, vollständig, Gödel, …
    fish_puzzle() / boxes_puzzle()    Logik + Kontrolle: die Rätsel der Prolog-Aufgaben

Klauseln sind `frozenset`s von Literalen (Texte wie "A" und "¬B"); die
Schreibweise der Vorlesung ist `{A, ¬B}`, die leere Klausel ist `□`.

Bekannte Stolpersteine im Material (hier richtig gerechnet):
* Serie 4 nennt Beweiser der Prädikatenlogik »habentscheidbar« — gemeint ist
  *halbentscheidbar* (stoppt, wenn die Aussage gilt; sonst evtl. nie).
* In der Aufgabe (d) des Lerngerüsts fehlt im PDF-Text eine Klammer: gemeint ist
  `¬(A ∧ B) ∧ (C ⇒ B) ∧ ((¬A ∧ B) ⇒ ¬C)`. Ohne die innere Klammer bindet ⇒ am
  schwächsten und die Folgerung ¬C gälte *nicht*.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass
from functools import lru_cache

from . import core
from .core import (And, Const, FALSE, Formula, Iff, Implies, Not, Or, TRUE, Var, Xor,
                   assignments, big_and, big_or, bit, flatten, parse)
from .render import Result, Step, Table

__all__ = [
    "entails", "EntailmentResult",
    "derive", "DerivationResult", "ProofLine", "RULES",
    "to_cnf", "CNFResult", "is_literal", "is_clause", "is_cnf", "clauses_of",
    "lit_var", "lit_neg", "clause_str", "clause_set_str", "clause_formula", "make_clause",
    "resolve", "resolution", "refute", "ResolutionResult", "ResolutionLine",
    "tautology_by_laws", "entails_by_laws", "contradiction_by_laws", "LawsProof",
    "semantic_vs_syntactic", "ComparisonResult",
    "kalkuel_info", "KalkuelInfo",
    "fish_puzzle", "boxes_puzzle", "PuzzleResult",
    "MAX_CNF_SIZE", "MAX_CLAUSES", "MAX_RES_STEPS",
]

MAX_CNF_SIZE = 3000      # Knoten im Formelbaum, ab hier bricht to_cnf ab (Blow-up-Schutz)
MAX_CNF_STEPS = 300      # Umformungsdurchgänge in to_cnf
MAX_CLAUSES = 300        # Klauseln in der Resolution (inkl. Resolventen)
MAX_RES_STEPS = 20000    # geprüfte Resolventen-Kandidaten
MAX_DERIVE_LINES = 200   # Zeilen der Vorwärtsableitung

_SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _S(title, detail="", **kw):
    """Step; ein `note` wird an den Text angehängt, weil die Konsolenansicht Notizen nicht zeigt."""
    note = kw.pop("note", "")
    if note:
        detail = (detail + " " + note).strip()
    return Step(title, detail, **kw)


def _vars_of(*fs) -> list[str]:
    return sorted(set().union(*(f.variables() for f in fs)))


def _parts(x) -> list[Formula]:
    """Eine Formel oder eine Liste von Formeln -> Liste von Formeln."""
    # ponytail: Mehrere Formeln nur als Python-Liste, kein Aufspalten von Text an Kommas/Semikolons
    if isinstance(x, (list, tuple)):
        return [parse(p) for p in x]
    return [parse(x)]


def _env_str(env: dict) -> str:
    return ", ".join(f"{k}={bit(v)}" for k, v in env.items()) or "(keine Variablen)"


# ==========================================================================
# Schnelle Äquivalenzprüfung (Bitmasken statt Zeile für Zeile)
# ==========================================================================

@lru_cache(maxsize=64)
def _var_masks(names: tuple) -> dict:
    n = len(names)
    rows = 1 << n
    full = (1 << rows) - 1
    out = {}
    for i, name in enumerate(names):
        k = 1 << (n - 1 - i)                     # erste Variable = höchstes Bit (Tabellenreihenfolge)
        unit = ((1 << k) - 1) << k               # k Nullen, dann k Einsen
        rep = full // ((1 << (2 * k)) - 1)
        out[name] = unit * rep
    return out


def _truth_mask(f: Formula, names: tuple) -> int:
    """Wahrheitswerte aller 2^n Zeilen als eine grosse Ganzzahl."""
    # ponytail: Bitmasken-Auswertung statt core.eval je Zeile, damit jeder KNF-Schritt schnell prüfbar ist
    vm = _var_masks(names)
    full = (1 << (1 << len(names))) - 1
    memo: dict[int, int] = {}

    def ev(g: Formula) -> int:
        k = id(g)
        if k in memo:
            return memo[k]
        if isinstance(g, Var):
            r = vm[g.name]
        elif isinstance(g, Const):
            r = full if g.value else 0
        elif isinstance(g, Not):
            r = full ^ ev(g.a)
        else:
            a, b = ev(g.a), ev(g.b)
            if isinstance(g, And):
                r = a & b
            elif isinstance(g, Or):
                r = a | b
            elif isinstance(g, Xor):
                r = a ^ b
            elif isinstance(g, Implies):
                r = (full ^ a) | b
            else:
                r = full ^ (a ^ b)
        memo[k] = r
        return r

    return ev(f)


def _equiv(f: Formula, g: Formula):
    """f ≡ g per Wahrheitstabelle; None, wenn zu viele Variablen für eine Tabelle."""
    names = tuple(_vars_of(f, g))
    if len(names) > core.MAX_VARS:
        return None
    return _truth_mask(f, names) == _truth_mask(g, names)


def _entails_fast(wb: Formula, q: Formula):
    names = tuple(_vars_of(wb, q))
    if len(names) > core.MAX_VARS:
        return None
    full = (1 << (1 << len(names))) - 1
    return (_truth_mask(wb, names) & (full ^ _truth_mask(q, names))) == 0


# ==========================================================================
# Literale und Klauseln
# ==========================================================================

def lit_var(lit: str) -> str:
    """Variable eines Literals: lit_var('¬B') == 'B'."""
    return lit[1:] if lit.startswith("¬") else lit


def lit_neg(lit: str) -> str:
    """Komplementäres Literal: lit_neg('A') == '¬A', lit_neg('¬A') == 'A'."""
    return lit[1:] if lit.startswith("¬") else "¬" + lit


def _lit_key(lit: str):
    return (lit_var(lit), lit.startswith("¬"))


def _lit_formula(lit: str) -> Formula:
    return Not(Var(lit[1:])) if lit.startswith("¬") else Var(lit)


def _lit_of(f: Formula):
    """Formel -> Literal-Text oder None, falls f kein Literal ist."""
    if isinstance(f, Var):
        return f.name
    if isinstance(f, Not) and isinstance(f.a, Var):
        return "¬" + f.a.name
    return None


def make_clause(literals) -> frozenset:
    """Klausel aus Literalen: make_clause(["A", "¬B"]) -> frozenset({'A', '¬B'})."""
    out = []
    for lit in literals:
        lit = str(lit).strip().replace("~", "¬").replace("!", "¬").replace("-", "¬") \
            if str(lit).strip()[:1] in "~!-" else str(lit).strip()
        if not lit or lit == "¬" or lit_var(lit) == "":
            raise ValueError(f"Ungültiges Literal {lit!r}")
        if lit.startswith("¬¬"):
            raise ValueError(f"Ungültiges Literal {lit!r} (doppelte Negation)")
        out.append(lit)
    return frozenset(out)


def clause_str(clause) -> str:
    """Klausel in der Schreibweise der Vorlesung: {A, ¬B}; die leere Klausel ist □."""
    if not clause:
        return "□"
    return "{" + ", ".join(sorted(clause, key=_lit_key)) + "}"


def clause_set_str(clauses) -> str:
    """Klauselmenge als Text: {{¬A, C}, {¬A, D}} (Reihenfolge wie übergeben)."""
    items = list(clauses)
    if not isinstance(clauses, (list, tuple)):
        items = sorted(items, key=lambda c: (len(c), [_lit_key(l) for l in sorted(c, key=_lit_key)]))
    return "{" + ", ".join(clause_str(c) for c in items) + "}"


def clause_formula(clause) -> Formula:
    """Klausel -> Formel (ODER-Verbindung der Literale; leere Klausel = falsch)."""
    return big_or(_lit_formula(l) for l in sorted(clause, key=_lit_key))


def _tautologic(lits) -> bool:
    s = set(lits)
    return any(lit_neg(l) in s for l in s)


def is_literal(f) -> bool:
    """Variable oder negierte Variable?"""
    return _lit_of(parse(f)) is not None


def _split_cnf(f: Formula):
    """KNF-Formel -> Liste von Literallisten (Reihenfolge wie im Text) oder None."""
    if isinstance(f, Const):
        return [] if f.value else [[]]
    rows = []
    for c in flatten(f, "And"):
        lits = []
        for d in flatten(c, "Or"):
            l = _lit_of(d)
            if l is None:
                return None
            lits.append(l)
        rows.append(lits)
    return rows


def is_clause(f) -> bool:
    """ODER-Verbindung von Literalen?"""
    f = parse(f)
    return all(_lit_of(d) is not None for d in flatten(f, "Or"))


def is_cnf(f) -> bool:
    """UND-Verbindung von Klauseln (Konjunktive Normalform)?"""
    return _split_cnf(parse(f)) is not None


def clauses_of(f) -> list:
    """Klauseln einer Formel, die schon in KNF ist (Reihenfolge wie im Text, ohne Doppelte).

        clauses_of("(S or N) and not S")  ->  [frozenset({'N','S'}), frozenset({'¬S'})]
    """
    f = parse(f)
    rows = _split_cnf(f)
    if rows is None:
        raise ValueError(f"{f} ist nicht in KNF (mit to_cnf umrechnen)")
    out, seen = [], set()
    for r in rows:
        c = frozenset(r)
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _build_cnf(rows) -> Formula:
    if not rows:
        return TRUE
    return big_and(big_or(_lit_formula(l) for l in r) for r in rows)


# ==========================================================================
# 1. Semantische Folgerung  WB ⊨ Q
# ==========================================================================

_METHODS = {
    "table": "table", "tabelle": "table", "pfeil": "table", "pfeilmethode": "table", "arrow": "table",
    "wahrheitstabelle": "table",
    "deduction": "deduction", "deduktion": "deduction", "deduktionstheorem": "deduction",
    "contradiction": "contradiction", "widerspruch": "contradiction",
    "widerspruchsbeweis": "contradiction", "unsat": "contradiction",
    "all": "all", "alle": "all",
}
_METHOD_NAMES = {"table": "Pfeilmethode (Wahrheitstabelle)", "deduction": "Deduktionstheorem",
                 "contradiction": "Widerspruchsbeweis"}


class EntailmentResult(Result):
    """
    Ergebnis von `entails`.

    Attribute: wb (Formel), q (Formel), wb_parts (Liste), method, entailed (bool),
    verdicts ({'table','deduction','contradiction'} -> bool), consistent (alle Methoden
    einig), counterexample (Belegung oder None), counterexamples (alle),
    table (Haupttabelle), tables (Dict je Methode + 'all'), variables.
    `bool(ergebnis)` ist `entailed`.
    """

    title = "Semantische Folgerung WB ⊨ Q"

    def __init__(self):
        super().__init__()
        self.wb_parts: list[Formula] = []
        self.wb: Formula = TRUE
        self.q: Formula = TRUE
        self.method = "all"
        self.entailed = False
        self.value = False
        self.verdicts: dict = {}
        self.consistent = True
        self.counterexample = None
        self.counterexamples: list = []
        self.table = None
        self.tables: dict = {}
        self.variables: list = []

    def __bool__(self):
        return self.entailed


def entails(wb, q, method: str = "all") -> EntailmentResult:
    """
    Semantische Folgerung WB ⊨ Q (Definition 1) mit den drei Prüfmethoden.

    `wb` ist eine Formel oder eine Liste von Formeln (Konjunktion), `q` die Anfrage.
    `method`: "table" (Pfeilmethode), "deduction" (Deduktionstheorem),
    "contradiction" (Widerspruchsbeweis) oder "all" (alle drei + Zusammenfassungstabelle).

        r = entails("(A or B) and (C or (not B and not A) or B)", "A or B")
        r.entailed            # True
        r.verdicts            # {'table': True, 'deduction': True, 'contradiction': True}
        entails(["A", "A -> B"], "B", method="contradiction")
        entails("A or B", "A").counterexample   # {'A': False, 'B': True}

    Die Methoden sind logisch gleichwertig; bei `all` wird das nachgeprüft.
    """
    # ponytail: Tabellen immer komplett (höchstens 2^12 Zeilen) statt nur bis zum ersten Gegenbeispiel
    m = _METHODS.get(str(method).strip().lower())
    if m is None:
        raise ValueError("method muss 'table', 'deduction', 'contradiction' oder 'all' sein")
    parts = _parts(wb)
    W, Q = big_and(parts), big_and(_parts(q))
    names = _vars_of(W, Q)
    rows = assignments(names)                    # ValueError bei mehr als MAX_VARS Variablen
    wv = [W.eval(e) for e in rows]
    qv = [Q.eval(e) for e in rows]
    r = EntailmentResult()
    r.wb_parts, r.wb, r.q, r.method, r.variables = parts, W, Q, m, names
    n_rows = len(rows)

    def row_i(idx):
        return rows[idx]

    # -- Methode 1: Pfeilmethode ------------------------------------------------
    heads = ["WB", "Q"]
    if len(parts) > 1:
        heads = [f"WB{str(i + 1).translate(_SUBSCRIPT)}" for i in range(len(parts))] + heads
    t1_rows, t1_marks = [], {}
    for idx, env in enumerate(rows):
        cells = [bit(env[n]) for n in names]
        if len(parts) > 1:
            cells += [bit(p.eval(env)) for p in parts]
        cells += [bit(wv[idx]), bit(qv[idx])]
        t1_rows.append(cells)
        if wv[idx]:
            t1_marks[idx] = "bad" if not qv[idx] else "hi"
    t1 = Table(names + heads, t1_rows, t1_marks,
               "blau: Zeilen mit WB = 1 (hier muss Q = 1 sein); rot: Gegenbeispiel (WB = 1, Q = 0)")
    bad = [row_i(i) for i in range(n_rows) if wv[i] and not qv[i]]
    ok_table = not bad
    n_wb = sum(wv)

    # -- Methode 2: Deduktionstheorem --------------------------------------------
    imp = [(not wv[i]) or qv[i] for i in range(n_rows)]
    t2_rows = [[bit(env[n]) for n in names] + [bit(wv[i]), bit(qv[i]), bit(imp[i])]
               for i, env in enumerate(rows)]
    t2 = Table(names + ["WB", "Q", "WB ⇒ Q"], t2_rows,
               {i: "bad" for i in range(n_rows) if not imp[i]},
               "WB ⇒ Q muss in jeder Zeile 1 sein (Tautologie); rot: Zeile mit 0")
    ok_ded = all(imp)

    # -- Methode 3: Widerspruchsbeweis --------------------------------------------
    con = [wv[i] and not qv[i] for i in range(n_rows)]
    t3_rows = [[bit(env[n]) for n in names] + [bit(wv[i]), bit(qv[i]), bit(not qv[i]), bit(con[i])]
               for i, env in enumerate(rows)]
    t3 = Table(names + ["WB", "Q", "¬Q", "WB ∧ ¬Q"], t3_rows,
               {i: "bad" for i in range(n_rows) if con[i]},
               "WB ∧ ¬Q muss in jeder Zeile 0 sein (Kontradiktion); rot: Zeile mit 1")
    ok_con = not any(con)

    # -- Zusammenfassung --------------------------------------------------------------
    t_all_rows = [[bit(env[n]) for n in names] +
                  [bit(wv[i]), bit(qv[i]), bit(imp[i]), bit(not qv[i]), bit(con[i])]
                  for i, env in enumerate(rows)]
    t_all = Table(names + ["WB", "Q", "WB ⇒ Q", "¬Q", "WB ∧ ¬Q"], t_all_rows,
                  {i: ("bad" if con[i] else "hi") for i in range(n_rows) if wv[i]},
                  "blau: WB = 1; rot: Gegenbeispiel. Pfeil: Q in allen blauen Zeilen 1. "
                  "Deduktion: Spalte WB ⇒ Q nur Einsen. Widerspruch: Spalte WB ∧ ¬Q nur Nullen.")

    verdict = {"table": ok_table, "deduction": ok_ded, "contradiction": ok_con}
    if len(set(verdict.values())) != 1:           # kann mathematisch nicht passieren
        raise RuntimeError("Die drei Prüfmethoden widersprechen sich: Fehler im Programm")

    r.tables = {"table": t1, "deduction": t2, "contradiction": t3, "all": t_all}
    r.entailed = r.value = ok_table
    r.counterexamples = bad
    r.counterexample = bad[0] if bad else None
    r.verdicts = {k: v for k, v in verdict.items() if m in ("all", k)}
    r.table = t_all if m == "all" else r.tables[m]
    r.consistent = True

    # -- Schritte ----------------------------------------------------------------
    what = [("WB", W if len(parts) == 1 else parts), ("Q", Q)]
    r.steps.append(_S("Aufgabe", "Zu prüfen ist, ob die Anfrage Q aus der Wissensbasis WB semantisch folgt. "
                      + ("Die Wissensbasis ist die UND-Verbindung aller angegebenen Formeln. "
                         if len(parts) > 1 else "")
                      + f"Signatur: {', '.join(names) or '–'}  →  {n_rows} Interpretationen.",
                      items=what, rule="Definition 1: WB ⊨ Q"))
    if m in ("all", "table"):
        r.steps.append(_S(
            "Methode 1: Pfeilmethode (Wahrheitstabelle)",
            "Nur dort, wo WB wahr ist, muss auch Q wahr sein. In Zeilen mit WB = 0 muss nichts geprüft werden "
            "(dort ist WB ⇒ Q automatisch wahr).",
            items=[("Anzahl Zeilen mit WB wahr", n_wb), ("davon mit Q falsch", len(bad))],
            note="Gegenbeispiel: " + _env_str(bad[0]) if bad else "Kein Gegenbeispiel: überall, wo WB = 1, ist Q = 1.",
            rule="WB = 1 ⇒ Q = 1", table=t1))
    if m in ("all", "deduction"):
        r.steps.append(_S(
            "Methode 2: Deduktionstheorem",
            "Q folgt genau dann semantisch aus WB, wenn WB ⇒ Q allgemeingültig (Tautologie) ist.",
            items=[("Anzahl Zeilen mit WB ⇒ Q falsch", sum(1 for v in imp if not v))],
            note="WB ⇒ Q ist " + ("eine Tautologie." if ok_ded else "keine Tautologie."),
            rule="Deduktionstheorem", table=t2))
    if m in ("all", "contradiction"):
        r.steps.append(_S(
            "Methode 3: Widerspruchsbeweis",
            "Q folgt genau dann aus WB, wenn WB ∧ ¬Q unerfüllbar (eine Kontradiktion) ist, "
            "denn ¬(WB ⇒ Q) ≡ ¬(¬WB ∨ Q) ≡ WB ∧ ¬Q. Die Logikprogrammierung (und die Resolution) verwendet diese Methode.",
            items=[("Anzahl Zeilen mit WB ∧ ¬Q wahr", sum(1 for v in con if v))],
            note="WB ∧ ¬Q ist " + ("eine Kontradiktion." if ok_con else "erfüllbar."),
            rule="Widerspruchsbeweis", table=t3))
    if m == "all":
        r.steps.append(_S(
            "Zusammenfassung aller Prüfungen",
            "Alle drei Methoden lesen dieselbe Tabelle: Pfeil (blaue Zeilen: Q = 1?), Deduktion "
            "(Spalte WB ⇒ Q nur Einsen?), Widerspruch (Spalte WB ∧ ¬Q nur Nullen?).",
            items=[("Pfeilmethode", ok_table), ("Deduktionstheorem", ok_ded), ("Widerspruchsbeweis", ok_con)],
            note="Alle drei Methoden liefern dasselbe Ergebnis (geprüft).", table=t_all))
    if ok_table:
        r.steps.append(_S("Ergebnis", "Q folgt semantisch aus WB.", items=[("", "WB ⊨ Q gilt")], rule="⊨"))
        r.summary = f"{fmt_short(W)} ⊨ {fmt_short(Q)}: gilt"
    else:
        r.steps.append(_S("Ergebnis", "Q folgt nicht aus WB: es gibt eine Interpretation, in der WB wahr "
                          "und Q falsch ist.", items=[("Gegenbeispiel", _env_str(bad[0]))], rule="⊭"))
        r.summary = f"{fmt_short(W)} ⊭ {fmt_short(Q)}: gilt nicht"
    r.highlights = [("WB ⊨ Q", ok_table)]
    if bad:
        r.highlights.append(("Gegenbeispiel", _env_str(bad[0])))
    return r


def fmt_short(f: Formula) -> str:
    s = f.to_str("unicode")
    return s if len(s) <= 70 else s[:67] + "…"


# ==========================================================================
# 2. Syntaktische Ableitung  WB ⊢ Q
# ==========================================================================

RULES = {
    "mp": ("Modus ponens", "p, p → q ⊢ q"),
    "mt": ("Modus tollens", "¬q, p → q ⊢ ¬p"),
    "hs": ("Hypothetischer Syllogismus", "p → q, q → r ⊢ p → r"),
    "ds": ("Disjunktiver Syllogismus", "p ∨ q, ¬p ⊢ q"),
    "add": ("Addition", "p ⊢ p ∨ q"),
    "simp": ("Simplifikation", "p ∧ q ⊢ p"),
    "conj": ("Konjunktion", "p, q ⊢ p ∧ q"),
    "res": ("Resolution", "p ∨ q, ¬p ∨ r ⊢ q ∨ r"),
}
_RULE_LOOKUP = {}
for _k, (_n, _s) in RULES.items():
    _RULE_LOOKUP[_k] = _k
    _RULE_LOOKUP[_n.lower()] = _k
_RULE_LOOKUP.update({"und-beseitigung": "simp", "und-einführung": "conj", "abtrennungsregel": "mp",
                     "aufhebender modus": "mt"})


def _neg(f: Formula) -> Formula:
    return f.a if isinstance(f, Not) else Not(f)


def _key(f: Formula):
    """Vergleichsschlüssel: Reihenfolge in ∧/∨-Ketten und ¬¬ spielen keine Rolle."""
    while isinstance(f, Not) and isinstance(f.a, Not):
        f = f.a.a
    if isinstance(f, Or):
        return ("or", frozenset(_key(c) for c in flatten(f, "Or")))
    if isinstance(f, And):
        return ("and", frozenset(_key(c) for c in flatten(f, "And")))
    if isinstance(f, Not):
        return ("not", _key(f.a))
    if isinstance(f, (Implies, Iff, Xor)):
        return (f.op, _key(f.a), _key(f.b))
    return f


def _disj(f: Formula) -> list:
    """Disjunkte einer Formel: A ∨ B ∨ C -> [A, B, C]; p → q wird als ¬p ∨ q gelesen."""
    if isinstance(f, Or):
        return flatten(f, "Or")
    if isinstance(f, Implies):
        return [_neg(f.a)] + _disj(f.b)
    return [f]


@dataclass
class ProofLine:
    """Eine Zeile des nummerierten Beweises: Nr., Formel, Regel, Verweise auf frühere Zeilen."""
    nr: int
    formula: Formula
    rule: str
    refs: tuple = ()
    note: str = ""


class DerivationResult(Result):
    """
    Ergebnis von `derive`.

    Attribute: derived (bool), proof (Liste von ProofLine, nur die benötigten Zeilen),
    table (nummerierter Beweis als Tabelle), n_generated (alle abgeleiteten Formeln),
    saturated (nichts Neues mehr ableitbar), limit_reached, entailed (Gegenprobe ⊨ oder None),
    rules (benutzte Regelnamen), contradiction (F wurde abgeleitet).
    """

    title = "Syntaktische Ableitung WB ⊢ Q"

    def __init__(self):
        super().__init__()
        self.derived = False
        self.value = False
        self.proof: list[ProofLine] = []
        self.table = None
        self.n_generated = 0
        self.saturated = False
        self.limit_reached = False
        self.entailed = None
        self.rules: list[str] = []
        self.contradiction = False
        self.wb_parts: list[Formula] = []
        self.q = TRUE

    def __bool__(self):
        return self.derived


def derive(wb, q, rules=None, max_steps: int = MAX_DERIVE_LINES) -> DerivationResult:
    """
    Einfache Vorwärtsableitung WB ⊢ Q mit den Schlussregeln der Vorlesung.

    Regeln (`RULES`): Modus ponens, Modus tollens, Hypothetischer Syllogismus,
    Disjunktiver Syllogismus, Addition, Simplifikation, Konjunktion, Resolution.
    `rules=None` nimmt alle; sonst eine Liste von Schlüsseln oder Namen
    (z. B. ["mp", "Modus tollens"]).

    Der Kalkül sättigt: er wendet alle Regeln auf alle schon abgeleiteten Zeilen an, bis Q
    erscheint, nichts Neues mehr entsteht oder `max_steps` (höchstens 1000) Zeilen erreicht sind. Die einfachen
    Regeln kommen zuerst; Disjunktiver/Hypothetischer Syllogismus und Resolution nur, wenn es sonst nicht weitergeht.
    Addition und Konjunktion werden nur *zielgerichtet* benutzt (Ergebnis muss eine Teilformel von
    WB oder Q sein), sonst gäbe es unendlich viele Ableitungen. p → q wird bei der Resolution als ¬p ∨ q gelesen.

        r = derive("(A or B) and (B -> C)", "A or C")
        r.derived         # True
        print(r)          # nummerierter Beweis wie in der Vorlesung
        derive(["p -> q", "p"], "q", rules=["mp"]).derived

    Wird Q nicht gefunden, obwohl WB ⊨ Q gilt, zeigt das Ergebnis eine Warnung: dieser
    Regelsatz ist dann nicht vollständig (Resolution mit ¬Q, siehe `resolution`, ist es).
    """
    # ponytail: Sättigung mit Schrittlimit statt Beweissuche mit Heuristik; Addition/Konjunktion nur zielgerichtet
    parts, goal_parts = _parts(wb), _parts(q)
    goal = big_and(goal_parts)
    if not isinstance(max_steps, int) or max_steps < 1 or max_steps > 1000:
        raise ValueError("max_steps muss eine ganze Zahl zwischen 1 und 1000 sein")
    if rules is None:
        enabled = list(RULES)
    else:
        enabled = []
        for rr in (rules if isinstance(rules, (list, tuple, set)) else [rules]):
            k = _RULE_LOOKUP.get(str(rr).strip().lower())
            if k is None:
                raise ValueError(f"Unbekannte Regel {rr!r}. Erlaubt: " +
                                 ", ".join(f"{k} ({v[0]})" for k, v in RULES.items()))
            if k not in enabled:
                enabled.append(k)

    res = DerivationResult()
    res.wb_parts, res.q = parts, goal
    res.rules = [RULES[k][0] for k in enabled]
    gk = _key(goal)

    lines: list[dict] = []          # f, rule(Schlüssel), refs (Indizes), hyp (Nr. der Hypothese), note
    index: dict = {}
    state = {"found": None, "limit": False}

    def add(f, rule, refs=(), note="", hyp=0):
        k = _key(f)
        if k in index:
            return False
        if len(lines) >= max_steps:
            state["limit"] = True
            return False
        index[k] = len(lines)
        lines.append({"f": f, "rule": rule, "refs": tuple(refs), "note": note, "hyp": hyp})
        if k == gk:
            state["found"] = index[k]
        return True

    for h, p in enumerate(parts, 1):
        add(p, "hyp", hyp=h)

    # Teilformeln von WB und Q: nur solche darf Addition/Konjunktion erzeugen
    relevant = []
    seen_rel = set()
    for p in parts + [goal]:
        for s in p.subformulas():
            if _key(s) not in seen_rel:
                seen_rel.add(_key(s))
                relevant.append(s)

    F = lambda i: lines[i]["f"]

    def r_simp(snap):
        for i in snap:
            if isinstance(F(i), And):
                for c in (F(i).a, F(i).b):
                    add(c, "simp", (i,))

    def r_mp(snap):
        for i in snap:
            if isinstance(F(i), Implies):
                for j in snap:
                    if _key(F(j)) == _key(F(i).a):
                        add(F(i).b, "mp", (j, i))

    def r_mt(snap):
        for i in snap:
            if isinstance(F(i), Implies):
                nq = _key(_neg(F(i).b))
                for j in snap:
                    if _key(F(j)) == nq:
                        add(_neg(F(i).a), "mt", (j, i))

    def r_hs(snap):
        for i in snap:
            for j in snap:
                if i != j and isinstance(F(i), Implies) and isinstance(F(j), Implies) \
                        and _key(F(i).b) == _key(F(j).a) and _key(F(i).a) != _key(F(j).b):
                    add(Implies(F(i).a, F(j).b), "hs", (i, j))

    def r_conj(snap):
        for g in relevant:
            if isinstance(g, And) and _key(g) not in index \
                    and _key(g.a) in index and _key(g.b) in index:
                add(g, "conj", (index[_key(g.a)], index[_key(g.b)]))

    def r_add(snap):
        for g in relevant:
            if isinstance(g, Or) and _key(g) not in index:
                for c in flatten(g, "Or"):
                    if _key(c) in index:
                        add(g, "add", (index[_key(c)],))
                        break

    def r_disj(stage):
        def run(snap):
            for i in snap:
                di = _disj(F(i))
                for j in snap:
                    if j == i:
                        continue
                    dj = _disj(F(j))
                    for xi, x in enumerate(di):
                        nx = _key(_neg(x))
                        yj = next((t for t, y in enumerate(dj) if _key(y) == nx), None)
                        if yj is None:
                            continue
                        rest, rk = [], set()
                        for d in di[:xi] + di[xi + 1:] + dj[:yj] + dj[yj + 1:]:
                            if _key(d) not in rk:
                                rk.add(_key(d))
                                rest.append(d)
                        if any(_key(_neg(d)) in rk for d in rest):
                            continue                      # tautologischer Resolvent
                        unit = len(di) == 1 or len(dj) == 1
                        if stage == "ds" and not (unit and "ds" in enabled):
                            continue
                        if stage == "res" and "res" not in enabled:
                            continue
                        note = ""
                        if isinstance(F(i), Implies) or isinstance(F(j), Implies):
                            note = "Implikation p → q wird als ¬p ∨ q gelesen"
                        if not rest:
                            add(FALSE, stage, (i, j), note or "leeres Ergebnis: Widerspruch F")
                        else:
                            add(big_or(rest), stage, (i, j), note)
        return run

    # Reihenfolge: eine Regel nach der anderen; sobald eine etwas Neues liefert, beginnt die Suche wieder oben.
    # So kommen die einfachen Regeln zuerst, Disjunktiver Syllogismus, Hypothetischer Syllogismus und
    # Resolution nur, wenn es sonst nicht weitergeht (kurze Beweise, kein Überfluten mit p → r).
    order_ = (("simp", r_simp), ("mp", r_mp), ("mt", r_mt), ("conj", r_conj), ("add", r_add),
              ("ds", r_disj("ds")), ("hs", r_hs), ("res", r_disj("res")))
    groups = [[(k, f)] for k, f in order_ if k in enabled]
    while state["found"] is None and not state["limit"]:
        progressed = False
        for grp in groups:
            before = len(lines)
            snap = list(range(before))
            for _, fn in grp:
                fn(snap)
                if state["found"] is not None or state["limit"]:
                    break
            if len(lines) > before:
                progressed = True
                break
        if not progressed:
            res.saturated = True
            break

    res.n_generated = len(lines)
    res.limit_reached = state["limit"] and state["found"] is None
    res.derived = res.value = state["found"] is not None
    res.contradiction = any(isinstance(l["f"], Const) and not l["f"].value for l in lines)
    res.entailed = _entails_fast(big_and(parts), goal)

    res.steps.append(_S("Wissensbasis und Anfrage",
                        "Gesucht ist eine Ableitung von Q aus den Hypothesen mit den Schlussregeln des Kalküls. "
                        "Erlaubte Regeln: " + ", ".join(f"{RULES[k][0]}" for k in enabled) + ".",
                        items=[("WB", parts if len(parts) > 1 else big_and(parts)), ("Q", goal)],
                        rule="Definition 2: WB ⊢ Q"))
    if res.derived:
        need, stack = set(), [state["found"]]
        while stack:
            i = stack.pop()
            if i not in need:
                need.add(i)
                stack.extend(lines[i]["refs"])
        order = sorted(need)
        nr = {i: n for n, i in enumerate(order, 1)}
        for i in order:
            L = lines[i]
            refs = tuple(nr[x] for x in L["refs"])
            if L["rule"] == "hyp":
                rule_name, extra = "Hypothese", f"H{L['hyp']}"
            else:
                rule_name, extra = RULES[L["rule"]][0], ""
            pl = ProofLine(nr[i], L["f"], rule_name if not extra else f"Hypothese ({extra})", refs, L["note"])
            res.proof.append(pl)
            if L["rule"] == "hyp":
                detail = f"Hypothese {extra} aus der Wissensbasis."
            else:
                detail = f"{rule_name} unter Verwendung von " + " und ".join(f"({x})" for x in refs) + "."
            res.steps.append(_S(f"Zeile {nr[i]}", detail, items=[(f"({nr[i]})", L["f"])],
                                note=L["note"] or (RULES[L["rule"]][1] if L["rule"] != "hyp" else ""),
                                rule=rule_name))
        last = res.proof[-1]
        if last.formula != goal:                     # gleich bis auf ¬¬ bzw. Reihenfolge in ∧/∨-Ketten
            nrn = len(res.proof) + 1
            res.proof.append(ProofLine(nrn, goal, "Äquivalenzumformung", (last.nr,),
                                       "Doppelnegation, Kommutativ- und Assoziativgesetz"))
            res.steps.append(_S(f"Zeile {nrn}", f"Äquivalenzumformung von ({last.nr}) mit der Doppelnegation "
                                "bzw. dem Kommutativ- und Assoziativgesetz: Q steht jetzt genau so da.",
                                items=[(f"({nrn})", goal)], rule="Äquivalenzumformung"))
        res.table = Table(["Nr.", "Formel", "Regel", "Verweise"],
                          [[str(p.nr), str(p.formula), p.rule, ", ".join(map(str, p.refs))] for p in res.proof],
                          {len(res.proof) - 1: "ok"}, "nummerierter Beweis; die letzte Zeile ist Q")
        res.steps.append(_S("Ergebnis", f"Q ist in Zeile {len(res.proof)} abgeleitet: WB ⊢ Q.",
                            items=[("", "WB ⊢ Q gilt")], rule="⊢"))
        res.summary = f"{fmt_short(big_and(parts))} ⊢ {fmt_short(goal)}: abgeleitet in {len(res.proof)} Zeilen"
        if res.entailed is False:
            raise RuntimeError("Interner Fehler: eine Ableitung bewies etwas, das semantisch nicht folgt")
    else:
        why = ("Das Schrittlimit wurde erreicht." if res.limit_reached else
               "Es lässt sich nichts Neues mehr ableiten.")
        res.table = Table(["Nr.", "Formel", "Regel"],
                          [[str(n), str(L["f"]), "Hypothese" if L["rule"] == "hyp" else RULES[L["rule"]][0]]
                           for n, L in enumerate(lines, 1)], {}, "alle abgeleiteten Formeln (Q fehlt)")
        res.steps.append(_S("Keine Ableitung gefunden",
                            why + f" Insgesamt wurden {len(lines)} Formeln abgeleitet, Q war nicht dabei.",
                            items=[("Formeln", len(lines))], table=res.table))
        res.summary = f"{fmt_short(big_and(parts))} ⊢ {fmt_short(goal)}: nicht abgeleitet"
        if res.entailed:
            res.warnings.append("WB ⊨ Q gilt, aber mit diesen Regeln ist Q nicht ableitbar: der Regelsatz ist "
                                "nicht vollständig. Vollständig ist die Resolution mit ¬Q (siehe resolution()).")
        if res.limit_reached:
            res.warnings.append(f"Abbruch beim Limit von {max_steps} Zeilen.")
    if res.contradiction and not res.derived:
        res.warnings.append("Es wurde der Widerspruch F abgeleitet: die Wissensbasis ist unerfüllbar.")
    res.steps.append(_S("Gegenprobe (semantisch)",
                        "Korrektheit des Kalküls: was abgeleitet wird, folgt auch semantisch (WB ⊢ Q ⇒ WB ⊨ Q).",
                        items=[("WB ⊨ Q", res.entailed if res.entailed is not None else "zu viele Variablen")]))
    res.highlights = [("WB ⊢ Q", res.derived), ("WB ⊨ Q", res.entailed if res.entailed is not None else "?")]
    return res


# ==========================================================================
# 3. Konjunktive Normalform
# ==========================================================================

class CNFResult(Result):
    """
    Ergebnis von `to_cnf`.

    Attribute: original (Formel), formula (KNF-Formel), clauses (frozenset von Klauseln;
    jede Klausel ein frozenset von Literalen wie "A", "¬B"), clause_list (dieselben Klauseln als
    Liste in Reihenfolge des Terms), already_cnf, verified (jeder Schritt per Tabelle geprüft; None bei
    zu vielen Variablen), is_tautology (alle Klauseln tautologisch -> formula = 1),
    is_contradiction (leere Klausel).
    """

    title = "Umrechnung in Konjunktive Normalform (KNF)"

    def __init__(self):
        super().__init__()
        self.original: Formula = TRUE
        self.formula: Formula = TRUE
        self.clauses: frozenset = frozenset()
        self.clause_list: list = []
        self.already_cnf = False
        self.verified = True
        self.is_tautology = False
        self.is_contradiction = False
        self.value = None


def _rebuild(f: Formula, new: list) -> Formula:
    return Not(new[0]) if isinstance(f, Not) else type(f)(new[0], new[1])


def _top(f: Formula, rule, cnt: list) -> Formula:
    """Ein Durchgang: Regel an den äussersten passenden Stellen (ohne Verschachtelung)."""
    r = rule(f)
    if r is not None:
        cnt[0] += 1
        return r
    ch = f.children
    if not ch:
        return f
    new = [_top(c, rule, cnt) for c in ch]
    if all(a is b for a, b in zip(new, ch)):
        return f
    return _rebuild(f, new)


def _bottom(f: Formula, rule, cnt: list) -> Formula:
    """Ein Durchgang von innen nach aussen (für Regeln ohne Verdopplung)."""
    ch = f.children
    if ch:
        new = [_bottom(c, rule, cnt) for c in ch]
        if any(a is not b for a, b in zip(new, ch)):
            f = _rebuild(f, new)
    r = rule(f)
    if r is not None:
        cnt[0] += 1
        return r
    return f


def _r_const(n):
    if isinstance(n, Not) and isinstance(n.a, Const):
        return Const(not n.a.value)
    if isinstance(n, (And, Or, Implies, Iff, Xor)):
        a, b = n.a, n.b
        ca, cb = isinstance(a, Const), isinstance(b, Const)
        if not (ca or cb):
            return None
        c, o = (a, b) if ca else (b, a)
        v = c.value
        if isinstance(n, And):
            return o if v else FALSE
        if isinstance(n, Or):
            return TRUE if v else o
        if isinstance(n, Xor):
            return _neg(o) if v else o
        if isinstance(n, Iff):
            return o if v else _neg(o)
        if ca:                                                    # Implikation, links Konstante
            return b if v else TRUE
        return TRUE if v else _neg(a)                            # Implikation, rechts Konstante
    return None


def _r_iff(n):
    if isinstance(n, Iff):
        return And(Implies(n.a, n.b), Implies(n.b, n.a))


def _r_xor(n):
    if isinstance(n, Xor):
        return And(Or(n.a, n.b), Or(Not(n.a), Not(n.b)))


def _r_imp(n):
    if isinstance(n, Implies):
        return Or(Not(n.a), n.b)


def _r_dm(n):
    if isinstance(n, Not) and isinstance(n.a, (And, Or)):
        op = "And" if isinstance(n.a, And) else "Or"
        items = [Not(x) for x in flatten(n.a, op)]
        out = items[0]
        for x in items[1:]:
            out = Or(out, x) if op == "And" else And(out, x)
        return out


def _r_dn(n):
    if isinstance(n, Not) and isinstance(n.a, Not):
        return n.a.a


def _r_dist(n):
    if isinstance(n, Or):
        if isinstance(n.a, And):
            return And(Or(n.a.a, n.b), Or(n.a.b, n.b))
        if isinstance(n.b, And):
            return And(Or(n.a, n.b.a), Or(n.a, n.b.b))


def _size_error(max_size: int, where: str) -> ValueError:
    return ValueError(
        f"Die KNF-Umformung wird zu gross (mehr als {max_size} Knoten im Formelbaum nach dem Schritt "
        f"»{where}«). Das Ausmultiplizieren kann exponentiell wachsen, z. B. hat "
        "(A1 ∧ B1) ∨ (A2 ∧ B2) ∨ … ∨ (An ∧ Bn) in KNF 2^n Klauseln. "
        "Vereinfache die Formel oder erhöhe max_size.")


def to_cnf(f, max_size: int = MAX_CNF_SIZE, verify: bool = True, subsumption: bool = False) -> CNFResult:
    """
    Formel in Konjunktive Normalform (UND von ODER-Klauseln) umrechnen.

    Schritte (jeweils mit dem Namen des Rechengesetzes, jeder per Wahrheitstabelle als äquivalent geprüft):
    Konstanten vereinfachen · Äquivalenz auflösen · Exklusiv-Oder auflösen · Implikation auflösen ·
    De Morgan · doppelte Negation · Distributivgesetz · Aufräumen (Idempotenz, tautologische Klauseln
    p ∨ ¬p ≡ 1 streichen, doppelte Klauseln; mit `subsumption=True` auch Absorption p ∧ (p ∨ q) ≡ p).

        r = to_cnf("(A or B) -> (C and D)")      # Aufgabe 4 der Serie 4
        r.formula        # (¬A ∨ C) ∧ (¬A ∨ D) ∧ (¬B ∨ C) ∧ (¬B ∨ D)
        r.clauses        # frozenset({frozenset({'¬A','C'}), …})
        clause_set_str(r.clause_list)

    `max_size` begrenzt den Formelbaum (Schutz vor exponentiellem Blow-up; `ValueError` mit Meldung).
    """
    orig = parse(f)
    if not isinstance(max_size, int) or max_size < 10:
        raise ValueError("max_size muss eine ganze Zahl ≥ 10 sein")
    if orig.size() > max_size:
        raise _size_error(max_size, "Eingabe")
    res = CNFResult()
    res.original = orig
    cur = orig
    nv = len(orig.variables())
    check = verify and nv <= core.MAX_VARS
    res.verified = True if check else None
    if verify and not check:
        res.warnings.append(f"{nv} Variablen: für die Wahrheitstabellen-Prüfung zu viele "
                            f"(höchstens {core.MAX_VARS}); die Schritte sind nicht numerisch gegengeprüft.")
    n_steps = [0]

    def note_for(prev, new):
        if not check:
            return ""
        ok = _equiv(prev, new)
        if not ok:
            raise RuntimeError(f"Interner Fehler: Umformung nicht äquivalent: {prev}  →  {new}")
        return f"Äquivalenz per Wahrheitstabelle geprüft ({1 << nv} Zeilen)."

    def record(title, detail, rule, new):
        nonlocal cur
        if new.size() > max_size:
            raise _size_error(max_size, title)
        n_steps[0] += 1
        if n_steps[0] > MAX_CNF_STEPS:
            raise ValueError(f"Mehr als {MAX_CNF_STEPS} Umformungsschritte: Abbruch (Formel zu verschachtelt)")
        res.steps.append(_S(title, detail, items=[("", new)], rule=rule, note=note_for(cur, new)))
        cur = new

    res.steps.append(_S("Ausgangsterm", "Der Term, der in KNF gebracht werden soll.",
                        items=[("", orig)],
                        note="KNF = UND-Verbindung von Klauseln; Klausel = ODER-Verbindung von Literalen; "
                             "Literal = Variable oder negierte Variable."))

    rows = _split_cnf(cur)
    if rows is not None:
        res.already_cnf = True
        res.steps.append(_S("Bereits in KNF", f"Der Term besteht schon aus {len(rows)} Klausel(n); "
                            "es sind keine Umformungen nötig."))
    else:
        def loop(name, detail, rule, fn, mode="top"):
            while True:
                cnt = [0]
                new = (_top if mode == "top" else _bottom)(cur, fn, cnt)
                if cnt[0] == 0:
                    return
                record(name, detail, rule, new)

        has_const = any(isinstance(s, Const) for s in cur.subformulas())
        if has_const:
            loop("Konstanten vereinfachen", "Wahr (1) und falsch (0) mit den Identitäts- und Dominanzgesetzen entfernen.",
                 "p ∧ 1 ≡ p, p ∨ 0 ≡ p, p ∧ 0 ≡ 0, p ∨ 1 ≡ 1", _r_const, "bottom")
        if isinstance(cur, Const):
            pass
        else:
            loop("Äquivalenz auflösen", "p ↔ q durch (p → q) ∧ (q → p) ersetzen.",
                 "p ↔ q ≡ (p → q) ∧ (q → p)", _r_iff)
            loop("Exklusiv-Oder auflösen", "p ⊕ q durch (p ∨ q) ∧ (¬p ∨ ¬q) ersetzen.",
                 "p ⊕ q ≡ (p ∨ q) ∧ (¬p ∨ ¬q)", _r_xor)
            loop("Implikation auflösen", "p → q durch ¬p ∨ q ersetzen.", "p → q ≡ ¬p ∨ q", _r_imp, "bottom")
            guard = 0
            while True:
                guard += 1
                if guard > MAX_CNF_STEPS:
                    break
                changed = False
                for name, detail, rule, fn in (
                        ("De Morgan", "Negation über ∧ bzw. ∨ nach innen ziehen: aus ∧ wird ∨ und umgekehrt, "
                         "jedes Glied wird negiert.", "¬(p ∧ q) ≡ ¬p ∨ ¬q,  ¬(p ∨ q) ≡ ¬p ∧ ¬q", _r_dm),
                        ("Doppelte Negation", "Zwei Negationen heben sich auf.", "¬¬p ≡ p", _r_dn)):
                    cnt = [0]
                    new = _top(cur, fn, cnt)
                    if cnt[0]:
                        record(name, detail, rule, new)
                        changed = True
                if not changed:
                    break
            loop("Distributivgesetz", "∨ über ∧ ausmultiplizieren, bis nur noch Klauseln verbunden mit ∧ übrig sind.",
                 "p ∨ (q ∧ r) ≡ (p ∨ q) ∧ (p ∨ r)", _r_dist)

    # -- Aufräumen --------------------------------------------------------------------
    rows = _split_cnf(cur)
    if rows is None:
        raise RuntimeError("Interner Fehler: Ergebnis ist nicht in KNF")
    if not isinstance(cur, Const):
        rows2 = [list(dict.fromkeys(r)) for r in rows]
        if rows2 != rows:
            rows = rows2
            record("Doppelte Literale streichen", "Wiederholte Literale in einer Klausel fallen weg.",
                   "p ∨ p ≡ p", _build_cnf(rows))
        keep = [r for r in rows if not _tautologic(r)]
        if len(keep) != len(rows):
            gone = len(rows) - len(keep)
            rows = keep
            record("Tautologische Klauseln streichen",
                   f"{gone} Klausel(n) enthalten ein Literal und sein Gegenteil: p ∨ ¬p ≡ 1. "
                   "Eine UND-Verbindung mit 1 ändert nichts, die Klausel entfällt.",
                   "p ∨ ¬p ≡ 1,  q ∧ 1 ≡ q", _build_cnf(rows))
        seen, keep = set(), []
        for r in rows:
            k = frozenset(r)
            if k not in seen:
                seen.add(k)
                keep.append(r)
        if len(keep) != len(rows):
            rows = keep
            record("Doppelte Klauseln streichen", "Gleiche Klauseln müssen nur einmal vorkommen.",
                   "p ∧ p ≡ p", _build_cnf(rows))
        if subsumption:
            sets = [frozenset(r) for r in rows]
            keep = [r for r, s in zip(rows, sets) if not any(t < s for t in sets)]
            if len(keep) != len(rows):
                rows = keep
                record("Absorption", "Eine Klausel, die eine kürzere Klausel enthält, ist überflüssig.",
                       "p ∧ (p ∨ q) ≡ p", _build_cnf(rows))
    rows_final = [r for r in rows]
    res.clause_list = [frozenset(r) for r in rows_final]
    seen_c, uniq = set(), []
    for c in res.clause_list:
        if c not in seen_c:
            seen_c.add(c)
            uniq.append(c)
    res.clause_list = uniq
    res.clauses = frozenset(uniq)
    res.formula = cur
    if rows_final == [[]] or any(len(r) == 0 for r in rows_final):
        res.is_contradiction = True
        res.formula, res.clause_list, res.clauses = FALSE, [frozenset()], frozenset({frozenset()})
    res.is_tautology = not res.clause_list
    if res.is_tautology:
        res.formula = TRUE
    res.value = res.formula
    res.steps.append(_S("Ergebnis: KNF",
                        "Tautologie: keine Klausel übrig (≡ 1)." if res.is_tautology else
                        "Kontradiktion: die leere Klausel □ ist enthalten." if res.is_contradiction else
                        f"Die KNF besteht aus {len(res.clause_list)} Klausel(n).",
                        items=[("KNF", res.formula), ("Klauselmenge", clause_set_str(res.clause_list))],
                        rule="Konjunktive Normalform"))
    res.summary = f"{fmt_short(orig)} ≡ {fmt_short(res.formula)}"
    res.highlights = [("KNF", res.formula), ("Klauseln", clause_set_str(res.clause_list))]
    if check and not _equiv(orig, res.formula):
        raise RuntimeError("Interner Fehler: die KNF ist nicht äquivalent zum Ausgangsterm")
    return res


# ==========================================================================
# 4. Resolution
# ==========================================================================

def resolve(c1, c2, drop_tautologies: bool = True) -> list:
    """
    Resolutionsregel für zwei Klauseln: Liste der Resolventen als (Pivot-Variable, Resolvent).

        resolve({"S", "N"}, {"¬S"})          # [('S', frozenset({'N'}))]
        resolve({"A", "B"}, {"¬B", "C"})     # [('B', frozenset({'A', 'C'}))]

    Das komplementäre Paar B / ¬B wird gestrichen, der Rest vereinigt. Hat man mehrere
    komplementäre Paare, wäre jeder Resolvent tautologisch (und wird weggelassen).
    """
    a, b = make_clause(c1), make_clause(c2)
    out = []
    for l in sorted(a, key=_lit_key):
        if lit_neg(l) in b:
            r = (a - {l}) | (b - {lit_neg(l)})
            if drop_tautologies and _tautologic(r):
                continue
            out.append((lit_var(l), frozenset(r)))
    return out


@dataclass
class ResolutionLine:
    """Eine Klausel K_id der Resolution: id, clause, parents, origin, pivot (Variable oder None)."""
    id: int
    clause: frozenset
    parents: tuple
    origin: str
    pivot: str | None = None

    def as_dict(self) -> dict:
        return {"id": self.id, "clause": clause_str(self.clause),
                "parents": tuple(self.parents), "origin": self.origin}


class ResolutionResult(Result):
    """
    Ergebnis von `resolution` / `refute`.

    derived_empty (□ entstanden), entailed (WB ⊨ Q laut Resolution; None bei Limit),
    clauses (Liste aller Klauseln K1, K2, … als frozensets; Index = id-1),
    lines (Liste ResolutionLine), proof_ids (benötigte Klauseln), wb_inconsistent,
    graph (Vertrag mit grafik: Liste von dicts id/clause/parents/origin; bei derived_empty nur die für □ nötigen
    Klauseln, sonst alle), full_graph (immer alle), check (Gegenprobe core.entails oder None),
    agrees, counterexample, limit_reached, table (Klauselliste), proof_table.
    """

    title = "Resolutionskalkül (Aussagenlogik)"

    def __init__(self):
        super().__init__()
        self.derived_empty = False
        self.entailed = None
        self.value = None
        self.clauses: list = []
        self.lines: list[ResolutionLine] = []
        self.proof_ids: list = []
        self.wb_inconsistent = False
        self.graph: list = []
        self.full_graph: list = []
        self.check = None
        self.agrees = None
        self.counterexample = None
        self.limit_reached = False
        self.table = None
        self.proof_table = None
        self.cnf_wb: list = []
        self.cnf_query = None
        self.wb_parts: list = []
        self.q = None

    def __bool__(self):
        return bool(self.entailed)


def _saturate(initial: list, max_clauses: int, max_steps: int):
    """
    Resolution bis □ oder Sättigung. `initial`: Liste (Klausel, Herkunft).
    Wählt jeweils den kürzesten Resolventen (Einheitsklauseln zuerst), bei Gleichstand die
    Eltern mit den kleinsten Nummern.
    """
    # ponytail: kürzester Resolvent zuerst (Heap) statt Subsumption/Stützmengen; reicht für Vorlesungsgrösse
    lines: list[ResolutionLine] = []
    known: dict = {}
    for c, origin in initial:
        if c not in known:
            lines.append(ResolutionLine(len(lines) + 1, c, (), origin))
            known[c] = lines[-1].id
    heap: list = []
    store: dict = {}
    state = {"empty": any(not l.clause for l in lines), "limit": False, "tries": 0}

    def push(k):
        ck = lines[k - 1].clause
        for i in range(1, k):
            ci = lines[i - 1].clause
            comp = [l for l in ci if lit_neg(l) in ck]
            if len(comp) != 1:
                continue
            l = comp[0]
            r = frozenset((ci - {l}) | (ck - {lit_neg(l)}))
            if _tautologic(r):
                continue
            store[(i, k)] = (r, lit_var(l))
            heapq.heappush(heap, (len(r), k, i))

    if not state["empty"]:
        for k in range(2, len(lines) + 1):
            push(k)
    while heap and not state["empty"]:
        state["tries"] += 1
        if state["tries"] > max_steps:
            state["limit"] = True
            break
        _, k, i = heapq.heappop(heap)
        r, pivot = store[(i, k)]
        if r in known:
            continue
        if len(lines) >= max_clauses:
            state["limit"] = True
            break
        lines.append(ResolutionLine(len(lines) + 1, r, (i, k), "Resolvent", pivot))
        known[r] = lines[-1].id
        if not r:
            state["empty"] = True
            break
        push(len(lines))
    return lines, state["empty"], state["limit"]


def _ancestors(lines: list, target: int) -> list:
    need, stack = set(), [target]
    while stack:
        i = stack.pop()
        if i not in need:
            need.add(i)
            stack.extend(lines[i - 1].parents)
    return sorted(need)


def _clause_origin(l: ResolutionLine) -> str:
    if l.origin != "Resolvent":
        return l.origin
    return f"Resolvent von K{l.parents[0]} und K{l.parents[1]} (Pivot {l.pivot})"


def _finish_resolution(res: ResolutionResult, lines, empty, limit, W=None, Q=None):
    res.lines = lines
    res.clauses = [l.clause for l in lines]
    res.limit_reached = limit and not empty
    res.derived_empty = empty
    res.full_graph = [l.as_dict() for l in lines]
    if empty:
        # ponytail: .graph zeigt bei Erfolg nur die Klauseln des Beweises (ids bleiben die K-Nummern), alle in .full_graph
        target = next(l.id for l in lines if not l.clause)
        res.proof_ids = _ancestors(lines, target)
        res.graph = [lines[i - 1].as_dict() for i in res.proof_ids]
        res.wb_inconsistent = not any(lines[i - 1].origin == "Negierte Anfrage" for i in res.proof_ids)
    else:
        res.graph = list(res.full_graph)
    res.table = Table(["Nr.", "Klausel", "Herkunft"],
                      [[f"K{l.id}", clause_str(l.clause), _clause_origin(l)] for l in lines], {},
                      "alle Klauseln in der Reihenfolge ihrer Entstehung")
    if empty:
        res.proof_table = Table(["Nr.", "Klausel", "Herkunft"],
                                [[f"K{i}", clause_str(lines[i - 1].clause), _clause_origin(lines[i - 1])]
                                 for i in res.proof_ids], {len(res.proof_ids) - 1: "ok"},
                                "nur die Klauseln, die zur leeren Klausel führen")


def _resolution_steps(res: ResolutionResult, lines):
    for l in lines:
        if l.origin != "Resolvent":
            continue
        a, b = lines[l.parents[0] - 1], lines[l.parents[1] - 1]
        pv = l.pivot
        pos_k = a.id if pv in a.clause else b.id
        neg_k = b.id if pos_k == a.id else a.id
        if not l.clause:
            detail = (f"K{a.id} und K{b.id} enthalten {pv} bzw. ¬{pv} und sonst nichts: "
                      "nach dem Streichen bleibt nichts übrig — die leere Klausel □ (Widerspruch).")
        else:
            detail = (f"Pivot {pv}: K{pos_k} enthält {pv}, K{neg_k} enthält ¬{pv}. Das Paar {pv} / ¬{pv} wird "
                      "gestrichen, der Rest beider Klauseln mit ∨ verbunden.")
        res.steps.append(_S(f"K{l.id} aus K{a.id} und K{b.id}", detail,
                            items=[(f"K{a.id}", clause_str(a.clause)), (f"K{b.id}", clause_str(b.clause)),
                                   (f"K{l.id}", clause_str(l.clause))],
                            rule="Resolutionsregel"))


def resolution(wb, q, max_clauses: int = MAX_CLAUSES, max_steps: int = MAX_RES_STEPS,
               detail: bool = True) -> ResolutionResult:
    """
    Resolutionskalkül für die Aussagenlogik: zeigt WB ⊨ Q als Widerspruchsbeweis.

    Ablauf wie in der Vorlesung: (1) WB in KNF, Klauseln K1, K2, … (2) ¬Q bilden und in KNF
    bringen (De Morgan …), (3) wiederholt Resolventen bilden, bis die leere Klausel □ entsteht
    (⇒ WB ⊨ Q) oder keine neue Klausel mehr entsteht (⇒ WB ⊭ Q). Tautologische Klauseln
    werden verworfen. Als nächster Resolvent wird der kürzeste gewählt (Einheitsklauseln zuerst).
    Die Gegenprobe mit der Wahrheitstabelle (`core.entails`) läuft immer mit.

        r = resolution("(S or N) and not S and (P or N) and (not N or P)", "not S and N and P")  # Aufgabe 5
        r.derived_empty, r.entailed     # (True, True)
        r.graph                         # [{'id': 1, 'clause': '{N, S}', 'parents': (), 'origin': 'Wissensbasis'}, …]
        r.clauses                       # [frozenset({'N','S'}), frozenset({'¬S'}), …]

    `detail=False` lässt die einzelnen KNF-Umformungsschritte weg. Bei Erreichen von `max_clauses`
    oder `max_steps` bricht das Verfahren mit Warnung ab (`entailed` ist dann None).
    """
    if not isinstance(max_clauses, int) or not 1 <= max_clauses <= 5000:
        raise ValueError("max_clauses muss eine ganze Zahl zwischen 1 und 5000 sein")
    if not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps muss eine positive ganze Zahl sein")
    # ponytail: nur ¬Q als Stützklausel-Herkunft markiert, keine Set-of-Support-Strategie; Limits fangen Ausreisser ab
    parts, qparts = _parts(wb), _parts(q)
    W, Q = big_and(parts), big_and(qparts)
    res = ResolutionResult()
    res.wb_parts, res.q = parts, Q
    nQ = Not(Q)

    res.steps.append(_S("Widerspruchsbeweis",
                        "Q folgt genau dann aus WB, wenn WB ∧ ¬Q unerfüllbar ist. Der Resolutionskalkül zeigt das, "
                        "indem er aus den Klauseln von WB ∧ ¬Q die leere Klausel □ (die Kontradiktion F) ableitet.",
                        items=[("WB", parts if len(parts) > 1 else big_and(parts)), ("Q", Q)],
                        rule="Resolutionskalkül"))
    initial = []
    for n, p in enumerate(parts, 1):
        c = to_cnf(p, verify=True)
        res.cnf_wb.append(c)
        label = "Wissensbasis" + (f" (Formel {n})" if len(parts) > 1 else "")
        if detail and not c.already_cnf:
            res.steps.append(_S(f"{label} in KNF", "Die Formel ist nicht in KNF; Umrechnung folgt.", items=[("", p)]))
            res.steps.extend(c.steps[1:-1])
        res.steps.append(_S(f"{label}: Klauseln",
                            "Schon in KNF." if c.already_cnf else "Ergebnis der Umrechnung.",
                            items=[("KNF", c.formula), ("Klauseln", clause_set_str(c.clause_list))]))
        initial += [(cl, "Wissensbasis") for cl in c.clause_list]
    cq = to_cnf(nQ, verify=True)
    res.cnf_query = cq
    if detail:
        res.steps.append(_S("Negierte Anfrage ¬Q in KNF",
                            "¬Q ist meist keine Klausel; erst umformen (De Morgan, doppelte Negation …).",
                            items=[("¬Q", nQ)]))
        res.steps.extend(cq.steps[1:-1])
    res.steps.append(_S("Negierte Anfrage: Klauseln", "", items=[("KNF von ¬Q", cq.formula),
                                                                 ("Klauseln", clause_set_str(cq.clause_list))]))
    initial += [(cl, "Negierte Anfrage") for cl in cq.clause_list]

    lines, empty, limit = _saturate(initial, max_clauses, max_steps)
    _finish_resolution(res, lines, empty, limit)
    res.steps.append(_S("Klauselmenge", f"Ausgangsklauseln K1 bis K{sum(1 for l in lines if l.origin != 'Resolvent')}.",
                        table=Table(["Nr.", "Klausel", "Herkunft"],
                                    [[f"K{l.id}", clause_str(l.clause), l.origin]
                                     for l in lines if l.origin != "Resolvent"], {})))
    _resolution_steps(res, lines)
    # -- Ergebnis und Gegenprobe ----------------------------------------------------------
    chk = _entails_fast(W, Q)
    res.check = chk
    if empty:
        res.entailed = True
        txt = "Die leere Klausel □ ist abgeleitet: WB ∧ ¬Q ist unerfüllbar, also gilt WB ⊨ Q."
        if res.wb_inconsistent:
            txt += " (Die Wissensbasis ist schon allein widersprüchlich; ¬Q wurde gar nicht gebraucht.)"
        res.steps.append(_S("Ergebnis", txt, items=[("", "WB ⊨ Q (leere Klausel □ abgeleitet)")],
                            rule="Widerspruchsbeweis", table=res.proof_table,
                            note="Tabelle: nur die Klauseln, die zur leeren Klausel führen."))
    elif limit:
        res.entailed = None
        res.warnings.append(f"Abbruch: Limit erreicht ({len(lines)} Klauseln); kein Ergebnis aus der Resolution.")
        res.steps.append(_S("Abbruch", "Das Limit für Klauseln oder Schritte wurde erreicht, ohne □ zu finden.",
                            items=[("Klauseln", len(lines))]))
    else:
        res.entailed = False
        res.steps.append(_S("Ergebnis", "Es entstehen keine neuen Klauseln mehr, ohne dass □ vorkam: "
                            "WB ∧ ¬Q ist erfüllbar, also gilt WB ⊭ Q.",
                            items=[("", "WB ⊭ Q (Resolution gesättigt, kein □)")], rule="Vollständigkeit der Resolution"))
        if chk is False:
            rows = [e for e in assignments(_vars_of(W, Q)) if W.eval(e) and not Q.eval(e)]
            res.counterexample = rows[0] if rows else None
            if res.counterexample is not None:
                res.steps.append(_S("Gegenbeispiel", "Diese Interpretation macht WB wahr und Q falsch.",
                                    items=[("Belegung", _env_str(res.counterexample))]))
    if chk is not None:
        res.agrees = (res.entailed is None) or (res.entailed == chk)
        res.steps.append(_S("Gegenprobe mit Wahrheitstabelle", "Semantische Folgerung zum Vergleich "
                            "(Korrektheit und Vollständigkeit des Resolutionskalküls).",
                            items=[("WB ⊨ Q (Tabelle)", chk), ("Resolution", res.entailed if res.entailed is not None else "Limit")]))
        if res.agrees is False:
            raise RuntimeError("Interner Fehler: Resolution und Wahrheitstabelle widersprechen sich")
    else:
        res.warnings.append(f"Mehr als {core.MAX_VARS} Variablen: keine Gegenprobe mit Wahrheitstabelle.")
    res.value = res.entailed
    res.summary = (f"{fmt_short(W)} ⊨ {fmt_short(Q)}: " +
                   ("bewiesen (□ abgeleitet)" if empty else "nicht ableitbar" if res.entailed is False else "Limit erreicht"))
    res.highlights = [("WB ⊨ Q", res.entailed if res.entailed is not None else "?"),
                      ("leere Klausel □", empty), ("Klauseln", len(lines))]
    return res


def refute(clauses, max_clauses: int = MAX_CLAUSES, max_steps: int = MAX_RES_STEPS) -> ResolutionResult:
    """
    Resolution auf einer fertigen Klauselmenge: ist sie unerfüllbar (□ ableitbar)?

    Klauseln als Listen/Mengen von Literal-Texten oder als Klausel-Formeln:
        refute([{"A", "B"}, {"¬A"}, {"¬B"}]).derived_empty      # True
        refute(["A or B", "not A or B"]).derived_empty          # False
    """
    initial = []
    for c in clauses:
        if isinstance(c, (str, Formula)):
            f = parse(c)
            if not is_clause(f):
                raise ValueError(f"{f} ist keine Klausel (ODER von Literalen)")
            cl = frozenset(_lit_of(d) for d in flatten(f, "Or"))
        else:
            cl = make_clause(c)
        initial.append((cl, "Wissensbasis"))
    if not initial:
        raise ValueError("Die Klauselmenge ist leer")
    res = ResolutionResult()
    res.steps.append(_S("Klauselmenge", "Gegeben ist eine Klauselmenge; □ abzuleiten heisst: sie ist unerfüllbar.",
                        items=[("", clause_set_str([c for c, _ in initial]))], rule="Resolutionskalkül"))
    lines, empty, limit = _saturate(initial, max_clauses, max_steps)
    _finish_resolution(res, lines, empty, limit)
    _resolution_steps(res, lines)
    res.entailed = True if empty else (None if limit else False)
    res.value = empty
    res.wb_inconsistent = empty
    res.summary = "unerfüllbar (□ abgeleitet)" if empty else "erfüllbar (keine neue Klausel, kein □)" if not limit else "Limit erreicht"
    res.steps.append(_S("Ergebnis", res.summary, items=[("leere Klausel □", empty)], table=res.proof_table))
    res.highlights = [("leere Klausel □", empty), ("Klauseln", len(lines))]
    if limit and not empty:
        res.warnings.append("Limit erreicht, kein Ergebnis.")
    return res


# ==========================================================================
# 5. Nachweise mit den Rechengesetzen (Aufgaben 1–3: »auch mit den Rechengesetzen?«)
# ==========================================================================

class LawsProof(Result):
    """Ergebnis von `tautology_by_laws` & Co.: proved (bool), cnf (CNFResult), formula (geprüfte Formel), kind."""

    title = "Nachweis mit den Rechengesetzen"

    def __init__(self):
        super().__init__()
        self.proved = False
        self.value = False
        self.cnf = None
        self.formula = TRUE
        self.kind = ""

    def __bool__(self):
        return self.proved


def tautology_by_laws(f, max_size: int = MAX_CNF_SIZE) -> LawsProof:
    """
    Tautologie mit den Rechengesetzen nachweisen: in KNF umformen; jede Klausel einer Tautologie
    enthält ein Paar p / ¬p (Komplementgesetz p ∨ ¬p ≡ 1), sodass am Ende 1 übrig bleibt.

        tautology_by_laws("(A or B) and (C or (not B and not A) or B) -> (A or B)").proved   # Aufgabe 1
    """
    f = parse(f)
    c = to_cnf(f, max_size=max_size)
    r = LawsProof()
    r.formula, r.cnf, r.kind = f, c, "Tautologie"
    r.proved = r.value = c.is_tautology
    r.steps.append(_S("Idee", "Ein Term ist genau dann eine Tautologie, wenn in seiner KNF jede Klausel ein "
                      "Literal und dessen Gegenteil enthält (p ∨ ¬p ≡ 1). Also in KNF umformen und tautologische "
                      "Klauseln streichen.", items=[("Term", f)]))
    r.steps.extend(c.steps)
    r.steps.append(_S("Ergebnis", "Es bleibt 1 übrig: der Term ist eine Tautologie." if r.proved else
                      "Es bleiben Klauseln ohne Paar p / ¬p: der Term ist keine Tautologie.",
                      items=[("KNF", c.formula)], rule="Komplementgesetz"))
    r.summary = f"{fmt_short(f)} ist " + ("eine Tautologie" if r.proved else "keine Tautologie")
    r.highlights = [("Tautologie", r.proved)]
    return r


def contradiction_by_laws(f, max_size: int = MAX_CNF_SIZE) -> LawsProof:
    """
    Kontradiktion mit den Rechengesetzen nachweisen (Aufgabe 2): f ist genau dann eine Kontradiktion,
    wenn ¬f eine Tautologie ist.

        contradiction_by_laws("(A or B) and (C or (not B and not A) or B) and not (A or B)").proved
    """
    f = parse(f)
    t = tautology_by_laws(Not(f), max_size=max_size)
    r = LawsProof()
    r.formula, r.cnf, r.kind = f, t.cnf, "Kontradiktion"
    r.proved = r.value = t.proved
    r.steps.append(_S("Idee", "f ist genau dann eine Kontradiktion, wenn ¬f eine Tautologie ist.",
                      items=[("f", f), ("¬f", Not(f))]))
    r.steps.extend(t.steps[1:])
    r.summary = f"{fmt_short(f)} ist " + ("eine Kontradiktion" if r.proved else "keine Kontradiktion")
    r.highlights = [("Kontradiktion", r.proved)]
    return r


def entails_by_laws(wb, q, via: str = "deduction", max_size: int = MAX_CNF_SIZE) -> LawsProof:
    """
    WB ⊨ Q mit den Rechengesetzen (Aufgaben 1–3 »Können Sie dies auch mit den Rechengesetzen nachweisen?«).

    via="deduction": WB ⇒ Q ist eine Tautologie (Deduktionstheorem).
    via="contradiction": WB ∧ ¬Q ist eine Kontradiktion (Widerspruchsbeweis).

        entails_by_laws("(A or B) and (B -> C)", "A or C").proved     # Aufgabe 3
    """
    W, Q = big_and(_parts(wb)), big_and(_parts(q))
    v = str(via).strip().lower()
    if v in ("deduction", "deduktion", "deduktionstheorem"):
        r = tautology_by_laws(Implies(W, Q), max_size=max_size)
        r.kind = "Deduktionstheorem"
        r.steps.insert(0, _S("Deduktionstheorem", "WB ⊨ Q genau dann, wenn WB ⇒ Q eine Tautologie ist.",
                             items=[("WB", W), ("Q", Q), ("WB ⇒ Q", Implies(W, Q))], rule="Deduktionstheorem"))
    elif v in ("contradiction", "widerspruch", "widerspruchsbeweis"):
        r = contradiction_by_laws(And(W, Not(Q)), max_size=max_size)
        r.kind = "Widerspruchsbeweis"
        r.steps.insert(0, _S("Widerspruchsbeweis", "WB ⊨ Q genau dann, wenn WB ∧ ¬Q eine Kontradiktion ist.",
                             items=[("WB", W), ("Q", Q), ("WB ∧ ¬Q", And(W, Not(Q)))], rule="Widerspruchsbeweis"))
    else:
        raise ValueError("via muss 'deduction' oder 'contradiction' sein")
    r.summary = f"{fmt_short(W)} ⊨ {fmt_short(Q)}: " + ("nachgewiesen" if r.proved else "nicht nachweisbar (gilt nicht)")
    r.highlights = [("WB ⊨ Q", r.proved)]
    return r


# ==========================================================================
# 6. ⊨ gegen ⊢
# ==========================================================================

class ComparisonResult(Result):
    """Ergebnis von `semantic_vs_syntactic`: semantic, resolution, derivation, same (alle Antworten gleich), table."""

    title = "Semantisch folgern (⊨) und syntaktisch ableiten (⊢)"

    def __init__(self):
        super().__init__()
        self.semantic = None
        self.resolution = None
        self.derivation = None
        self.same = True
        self.table = None
        self.value = None


def semantic_vs_syntactic(wb, q) -> ComparisonResult:
    """
    Dieselbe Frage zweimal beantworten: mit Wahrheitstabelle (⊨) und mit Resolution (⊢).

        semantic_vs_syntactic("(A or B) and (B -> C)", "A or C").table
    """
    sem = entails(wb, q, method="table")
    rs = resolution(wb, q, detail=False)
    dv = derive(wb, q)
    c = ComparisonResult()
    c.semantic, c.resolution, c.derivation = sem, rs, dv
    n = len(sem.variables)
    c.value = sem.entailed
    c.same = (rs.entailed == sem.entailed) and (not dv.derived or sem.entailed)
    n_res = sum(1 for l in rs.lines if l.origin == "Resolvent")
    c.table = Table(["Verfahren", "Antwort", "Aufwand"],
                    [["Wahrheitstabelle  WB ⊨ Q", "ja" if sem.entailed else "nein", f"{1 << n} Zeilen (2^{n})"],
                     ["Resolution  WB ∧ ¬Q ⊢ □", "ja" if rs.entailed else "nein",
                      f"{len(rs.lines)} Klauseln, {n_res} Resolventen"],
                     ["Schlussregeln  WB ⊢ Q", "ja" if dv.derived else "nein (nicht gefunden)",
                      f"{dv.n_generated} abgeleitete Formeln"]], {})
    c.steps.append(_S("Semantisch (⊨)", "Rechnen mit Bedeutungen: alle Interpretationen durchgehen, Zeilen mit WB = 1 prüfen. "
                      "Der Aufwand wächst mit 2^n.", items=[("WB ⊨ Q", sem.entailed), ("Zeilen", 1 << n)],
                      rule="Definition 1"))
    c.steps.append(_S("Syntaktisch (⊢)", "Rechnen mit Symbolen: nur Regeln auf Klauseln anwenden, ohne Wahrheitswerte. "
                      "Mit Resolution wird aus WB ∧ ¬Q die leere Klausel □ abgeleitet.",
                      items=[("Resolution", rs.entailed), ("Klauseln", len(rs.lines)),
                             ("Schlussregeln finden Q", dv.derived)], rule="Definition 2"))
    c.steps.append(_S("Zusammenhang", "Der Resolutionskalkül ist korrekt (WB ⊢ Q ⇒ WB ⊨ Q) und für die Aussagenlogik "
                      "vollständig (WB ⊨ Q ⇒ WB ⊢ Q): beide Antworten stimmen immer überein.",
                      table=c.table, rule="Korrektheit und Vollständigkeit"))
    c.summary = "⊨ und ⊢ stimmen überein: " + ("WB ⊨ Q" if sem.entailed else "WB ⊭ Q")
    c.highlights = [("WB ⊨ Q", sem.entailed), ("WB ⊢ Q (Resolution)", rs.entailed)]
    return c


# ==========================================================================
# 7. Begriffe und Gödel
# ==========================================================================

class KalkuelInfo(Result):
    """Begriffsübersicht zu Beweiskalkülen. `facts` (Schlagwort -> Text), `table` (⊨ gegen ⊢)."""

    title = "Beweiskalküle: Begriffe und Gödel"

    def __init__(self):
        super().__init__()
        self.facts: dict = {}
        self.table = None


def kalkuel_info() -> KalkuelInfo:
    """
    Zusammenfassung der Begriffe aus Serie 4 (Kalkül, ⊨ gegen ⊢, korrekt, vollständig, Gödel,
    Entscheidbarkeit, Resolution, Logikprogrammierung).

        kalkuel_info().facts["Gödelscher Vollständigkeitssatz"]
    """
    k = KalkuelInfo()
    facts = {
        "Kalkül": "Ein Beweiskalkül ist die Gesamtheit der erlaubten Regeln, mit denen man Terme rein "
                  "syntaktisch (nach Form, ohne Wahrheitswerte) umformt — eine Art Regelbüchlein: was man darf und was nicht.",
        "Semantische Folgerung ⊨": "WB ⊨ Q (Definition 1): Für alle Interpretationen, in denen WB wahr ist, "
                  "ist auch Q wahr. Prüfbar mit Wahrheitstabelle (Pfeilmethode), Deduktionstheorem (WB ⇒ Q Tautologie) "
                  "oder Widerspruchsbeweis (WB ∧ ¬Q unerfüllbar).",
        "Syntaktische Ableitung ⊢": "WB ⊢ Q (Definition 2): Q lässt sich mit den Regeln eines Beweiskalküls aus WB "
                  "herleiten, ohne Interpretationen mit Wahrheitswerten zu betrachten.",
        "Unterschied ⊨ und ⊢": "⊨ fragt nach der Bedeutung (alle Interpretationen, Aufwand 2^n Zeilen), ⊢ nach der "
                  "Herleitbarkeit (Regeln auf Symbole, ein Computer kann das Schritt für Schritt suchen). Ob beide "
                  "dasselbe leisten, sagen Korrektheit und Vollständigkeit.",
        "Korrekt": "Ein Kalkül ist korrekt, wenn jede ableitbare Aussage auch semantisch folgt: aus WB ⊢ Q folgt WB ⊨ Q "
                  "(er beweist nichts Falsches).",
        "Vollständig": "Ein Kalkül ist vollständig, wenn sich jede semantische Folgerung auch ableiten lässt: "
                  "aus WB ⊨ Q folgt WB ⊢ Q (er beweist alles Wahre).",
        "Gödelscher Vollständigkeitssatz": "Für die Prädikatenlogik erster Stufe gibt es vollständige und korrekte "
                  "Beweiskalküle. (1965 wurde ein dafür angepasster Resolutionskalkül vorgestellt, der vollständig und "
                  "korrekt ist; er löste in den 70ern eine Logikeuphorie in der KI aus.)",
        "Gödelscher Unvollständigkeitssatz": "Für die Prädikatenlogik zweiter Stufe (Quantifizieren auch über Prädikate) "
                  "gibt es wahre Aussagen, die kein Beweiskalkül ableiten kann. (Allgemeiner gilt das für jedes "
                  "widerspruchsfreie, hinreichend starke formale System, das die Arithmetik enthält.)",
        "Entscheidbarkeit": "Beweiser der Aussagenlogik sind entscheidbar: sie stoppen immer nach endlicher Zeit (die "
                  "Wahrheitstabelle zeigt es). In der Prädikatenlogik erster Stufe stoppt ein Beweiser, wenn die Aussage "
                  "allgemeingültig ist; sonst kann er endlos laufen: halbentscheidbar (die Folien schreiben »habentscheidbar«, ein Tippfehler).",
        "Resolutionskalkül": "Eine einzige Regel auf Klauseln der KNF: (A ∨ B) ∧ (¬B ∨ C) ⊢ (A ∨ C). Man nimmt ¬Q zur WB "
                  "dazu und leitet die leere Klausel □ (Widerspruch) ab. Für die Aussagenlogik korrekt und vollständig.",
        "Logikprogrammierung": "Algorithm = Logic + Control (Kowalski): die Logik beschreibt, was gilt (Wissensbasis); "
                  "die Kontrolle bestimmt die Reihenfolge der Regelanwendung. PROLOG setzt das um und beweist per Widerspruch.",
    }
    k.facts = facts
    for name, txt in facts.items():
        k.steps.append(_S(name, txt))
    k.table = Table(["", "Semantisch  WB ⊨ Q", "Syntaktisch  WB ⊢ Q"],
                    [["Idee", "Bedeutung, Interpretationen", "Symbole und Regeln eines Kalküls"],
                     ["Methode", "Wahrheitstabelle, Deduktionstheorem, Widerspruch", "Ableitungsregeln, Resolution"],
                     ["Aufwand (Aussagenlogik)", "2^n Zeilen", "Regelanwendungen, oft viel weniger"],
                     ["Verbindung", "korrekt: ⊢ ⇒ ⊨", "vollständig: ⊨ ⇒ ⊢"]], {})
    k.steps.append(_S("Vergleich", "⊨ gegen ⊢ in einer Tabelle.", table=k.table))
    k.summary = "Kalkül, semantisch ⊨ gegen syntaktisch ⊢, korrekt/vollständig, Gödel, Resolution"
    return k


# ==========================================================================
# 8. Logik + Kontrolle: die Rätsel der Prolog-Aufgaben
# ==========================================================================

class PuzzleResult(Result):
    """Ergebnis eines Rätsels: solutions (Liste), table, answer (Text), inferences (Anzahl geprüfter Teillösungen)."""

    title = "Rätsel: Logik + Kontrolle"

    def __init__(self):
        super().__init__()
        self.solutions: list = []
        self.table = None
        self.answer = ""
        self.inferences = 0
        self.value = None


def fish_puzzle() -> PuzzleResult:
    """
    Optionale Aufgabe 1 (Zebra/Fisch-Rätsel): Wem gehört der Fisch?

    Die 15 Hinweise sind die *Logik*, das Durchprobieren mit Zurücksetzen (Backtracking) die
    *Kontrolle* — wie beim PROLOG-Programm der Serie (member/links/neben).

        r = fish_puzzle(); r.answer   # 'Der Deutsche hat den Fisch.'
    """
    # ponytail: Backtracking über Permutationen je Eigenschaft statt Prolog-Unifikation
    nat = ["Brite", "Schwede", "Däne", "Norweger", "Deutsche"]
    col = ["rot", "grün", "weiss", "gelb", "blau"]
    drk = ["Tee", "Kaffee", "Milch", "Bier", "Wasser"]
    smk = ["Pall Mall", "Dunhill", "Marlboro", "Winfield", "Rothmans"]
    pet = ["Hund", "Vogel", "Katze", "Pferd", "Fisch"]

    def pos(perm, v):
        return perm.index(v)

    def same(p1, v1, p2, v2):
        return pos(p1, v1) == pos(p2, v2)

    def nb(p1, v1, p2, v2):
        return abs(pos(p1, v1) - pos(p2, v2)) == 1

    count = [0]
    sols = []
    for N in itertools.permutations(nat):
        if pos(N, "Norweger") != 0:                                     # 9
            continue
        for C in itertools.permutations(col):
            count[0] += 1
            if not same(N, "Brite", C, "rot"):                          # 1
                continue
            if pos(C, "grün") + 1 != pos(C, "weiss"):                   # 4
                continue
            if not nb(N, "Norweger", C, "blau"):                        # 13
                continue
            for D in itertools.permutations(drk):
                count[0] += 1
                if not same(N, "Däne", D, "Tee") or not same(C, "grün", D, "Kaffee"):   # 3, 5
                    continue
                if pos(D, "Milch") != 2:                                # 7
                    continue
                for S in itertools.permutations(smk):
                    count[0] += 1
                    if not same(C, "gelb", S, "Dunhill") or not same(S, "Winfield", D, "Bier"):  # 8, 12
                        continue
                    if not same(N, "Deutsche", S, "Rothmans"):         # 14
                        continue
                    if not nb(S, "Marlboro", D, "Wasser"):              # 15
                        continue
                    for P in itertools.permutations(pet):
                        count[0] += 1
                        if not same(N, "Schwede", P, "Hund"):           # 2
                            continue
                        if not same(S, "Pall Mall", P, "Vogel"):        # 6
                            continue
                        if not nb(S, "Marlboro", P, "Katze"):           # 10
                            continue
                        if not nb(P, "Pferd", S, "Dunhill"):            # 11
                            continue
                        sols.append((N, C, D, S, P))
    r = PuzzleResult()
    r.solutions = sols
    r.inferences = count[0]
    if len(sols) != 1:
        raise RuntimeError("Interner Fehler: das Rätsel hat nicht genau eine Lösung")
    N, C, D, S, P = sols[0]
    owner = N[P.index("Fisch")]
    r.answer = f"Der {owner} hat den Fisch."
    r.table = Table(["Haus", "Farbe", "Nationalität", "Getränk", "Zigaretten", "Haustier"],
                    [[str(i + 1), C[i], N[i], D[i], S[i], P[i]] for i in range(5)],
                    {P.index("Fisch"): "hi"})
    r.value = owner
    r.steps.append(_S("Logik", "Jeder der 15 Hinweise ist eine Bedingung an die Reihenfolge (Position 1–5) der Farben, "
                      "Nationalitäten, Getränke, Zigaretten und Haustiere. »links von« heisst Position + 1, "
                      "»neben« heisst Positionsabstand 1."))
    r.steps.append(_S("Kontrolle", "Probiert wird eigenschaftsweise (Nationalität, Farbe, Getränk, Zigaretten, Haustier); "
                      "sobald ein Hinweis verletzt ist, wird zurückgesetzt (Backtracking).",
                      items=[("geprüfte Teillösungen", count[0]), ("Lösungen", len(sols))]))
    r.steps.append(_S("Lösung", r.answer, table=r.table))
    r.summary = r.answer
    r.highlights = [("Fisch gehört", owner)]
    return r


def boxes_puzzle() -> PuzzleResult:
    """
    Optionale Aufgabe 2 (Einstellungstest): acht Kästchen mit Kreuz (1) oder Kreis (0) füllen, sodass
    keiner der 16 »durchgefallen«-Hinweise zutrifft.

        r = boxes_puzzle(); r.solutions   # Liste von Tupeln (Kästchen 1 … 8), 1 = Kreuz
    """
    # ponytail: alle 2^8 Belegungen durchprobieren; Aufgabe 3 (Schlange im Shoppingcenter) ist nicht umgesetzt
    def fails(x):
        o = lambda i: x[i - 1] == 0
        k = lambda i: x[i - 1] == 1
        return [
            sum(k(i) for i in (4, 5, 6)) > 1,
            k(3) and k(5) and k(7),
            o(1) and k(2) and o(5),
            not (k(2) or k(5) or k(8)),
            o(4) and o(5) and o(7),
            o(2) and o(4),
            sum(x) > 5,
            o(4) and o(6) and k(7),
            k(3) != k(7),
            not (k(1) or k(6) or k(8)),
            not (o(3) or o(4) or o(6)),
            o(1) and o(5) and o(6),
            sum(o(i) for i in (4, 6, 8)) > 1,
            o(3) and o(7),
            sum(1 for v in x if v == 0) > 5,
            k(2) and k(4),
        ]
    sols = [x for x in itertools.product((0, 1), repeat=8) if not any(fails(x))]
    r = PuzzleResult()
    r.solutions = sols
    r.inferences = 256
    r.table = Table([f"K{i}" for i in range(1, 9)], [["×" if v else "○" for v in s] for s in sols], {})
    r.answer = (f"{len(sols)} Lösung(en)" if sols else "keine Lösung")
    r.value = sols
    r.steps.append(_S("Logik", "Jeder der 16 Hinweise beschreibt eine Belegung, bei der man durchfällt. Gesucht ist eine "
                      "Belegung, die keinen davon erfüllt."))
    r.steps.append(_S("Kontrolle", "Alle 2^8 = 256 Belegungen werden durchprobiert.", items=[("Lösungen", len(sols))]))
    r.steps.append(_S("Lösung", r.answer, table=r.table if sols else None))
    r.summary = r.answer
    r.highlights = [("Lösungen", len(sols))]
    return r
