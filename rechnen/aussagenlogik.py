#!/usr/bin/env python3
"""
Semesterwoche 2: Aussagenlogik.

Aussagen, Junktoren, Wahrheitstabellen, Einstufung (Tautologie / Kontradiktion /
erfüllbar), Äquivalenz, Äquivalenzgesetze mit benannten Umformungsschritten,
kanonische Normalformen (DNF / KNF aus der Wahrheitstabelle) und Syntaxbäume
(auch für arithmetische Terme wie ``2 * x + y``).

    from dmatkit.rechnen.aussagenlogik import *
    truth_table("(Q -> P) and not P")        # Serie 2, Frage 3
    check_equivalence("(Q -> P) & ~P", "~(P | Q)")
    simplify("(~(~P | ~Q) & (P -> Q)) -> P") # Rechengesetze, Schritt für Schritt
    canonical_dnf("A xor B")                 # (¬A ∧ B) ∨ (A ∧ ¬B)
    syntax_tree("2 * x + y").evaluate(x=3, y=4)

Hinweise zu den Folien (Serie 2):
  * Die Wahrheitstabellen der PDF-Serie beginnen mit ``1 1`` (wahr zuerst); die
    Mitschrift/Vorlesung verwendet ``0 0 / 0 1 / 1 0 / 1 1``. Standard hier:
    ``0 0`` zuerst; mit ``true_first=True`` erscheint die Reihenfolge der PDF-Serie.
  * Vorrang: ¬ vor ∧ vor ⊕ vor ∨ vor ⇒ vor ⇔ (wie in der Vorlesung; die
    Zusammenfassung v1.0 stellt ∧ und ∨ fälschlich auf dieselbe Stufe).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from fractions import Fraction

from .core import (FALSE, MAX_VARS, TRUE, And, Const, Formula, Iff, Implies, Not, Or, Var,
                   Xor, assignments, big_and, big_or, bit, equivalent, flatten, parse)
from .render import Result, Step, Table

__all__ = [
    # Hilfsklassen
    "Node", "Definition", "Law",
    # Ergebnisklassen
    "GlossaryResult", "ConnectiveTablesResult", "TruthTableResult", "SignatureResult",
    "TableSizeResult", "ConsistencyResult", "EquivalenceResult", "ImplicationCheckResult",
    "ImplicationVariantsResult", "LawApplicationResult", "SimplifyResult",
    "DerivationCheckResult", "NormalFormResult", "SyntaxTreeResult", "SatResult",
    # Glossar / Grundverknüpfungen
    "definitions", "glossary", "connective_tables", "connective_table",
    # Wahrheitstabellen
    "truth_table", "classify", "signature", "table_size", "check_consistency", "sat",
    # Äquivalenz und Folgerung
    "check_equivalence", "check_implication", "implication_variants",
    # Gesetze und Umformung
    "laws", "find_law", "verify_law", "apply_law", "apply_laws", "simplify", "check_derivation",
    # Normalformen
    "canonical_dnf", "canonical_knf",
    # Syntaxbäume
    "syntax_tree", "parse_term",
    # Deutsche Namen
    "wahrheitstabelle", "aequivalenz_pruefen", "vereinfachen", "syntaxbaum",
]

MAX_TERM_LEN = 300       # Länge arithmetischer Terme
MAX_TERM_DEPTH = 60      # Schachtelungstiefe arithmetischer Terme
MAX_STEPS = 200          # Obergrenze für Umformungsschritte (keine Endlosschleifen)


# ==========================================================================
# Hilfsfunktionen
# ==========================================================================

def _vars_of(*fs) -> list[str]:
    return sorted(set().union(*(f.variables() for f in fs)))


def _check_vars(names, what: str = "Die Formel") -> None:
    if len(names) > MAX_VARS:
        raise ValueError(f"{what} enthält {len(names)} Variablen; erlaubt sind höchstens "
                         f"{MAX_VARS} (die Wahrheitstabelle hätte sonst mehr als "
                         f"{2 ** MAX_VARS} Zeilen).")


def _fs(f) -> str:
    """Formel als Text mit Klammern um jeden zusammengesetzten Teilterm (wie auf dem Papier)."""
    return _pretty(f)


def _col(f: Formula) -> str:
    """Spaltenüberschrift einer Wahrheitstabelle."""
    return _fs(f)


def _is_compound(f: Formula) -> bool:
    return not isinstance(f, (Var, Const))


def _n_rows(n: int) -> str:
    return f"2^{n} = {2 ** n}"


def _compound_columns(f: Formula) -> list[Formula]:
    """Spalten der Tabelle: zusammengesetzte Teilformeln innen -> aussen (f ist zuletzt)."""
    cols = [s for s in f.subformulas() if _is_compound(s)]
    if not cols or cols[-1] != f:
        cols.append(f)
    return cols


def _pretty(f: Formula) -> str:
    """Wie ``_fs``, aber Teilterme einer ∧/∨-Kette stehen in Klammern: (¬A ∧ B) ∨ (A ∧ ¬B)."""
    if isinstance(f, (And, Or)):
        glue = " ∧ " if isinstance(f, And) else " ∨ "
        parts = [(_pretty(e) if isinstance(e, (Var, Const, Not)) and not
                  (isinstance(e, Not) and not isinstance(e.a, (Var, Const)))
                  else "(" + _pretty(e) + ")") for e in flatten(f, f.op)]
        return glue.join(parts)
    if isinstance(f, Not):
        inner = _pretty(f.a)
        return "¬" + (inner if isinstance(f.a, (Var, Const, Not)) else "(" + inner + ")")
    if isinstance(f, (Var, Const)):
        return f.to_str("unicode")
    a = _pretty(f.a) if isinstance(f.a, (Var, Const, Not)) else "(" + _pretty(f.a) + ")"
    b = _pretty(f.b) if isinstance(f.b, (Var, Const, Not)) else "(" + _pretty(f.b) + ")"
    return f"{a} {_SYM_NAME[f.op]} {b}"


_SYM_NAME = {"Xor": "⊕", "Implies": "→", "Iff": "↔"}


def _neg(x: Formula) -> Formula:
    """Komplement eines Literals/Terms: ¬x, bzw. y für x = ¬y."""
    return x.a if isinstance(x, Not) else Not(x)


# ==========================================================================
# Node: ein Knoten eines Syntaxbaums (Formeln und arithmetische Terme)
# ==========================================================================

class Node:
    """
    Knoten eines Syntaxbaums.

        Node("+", (Node("·", (Node("2"), Node("x"))), Node("y")))

    ``label`` ist die Beschriftung im Baum, ``children`` die Teilbäume von links
    nach rechts, ``op`` die Art des Knotens (z. B. ``"+"``, ``"num"``, ``"name"``).
    Formeln aus ``core`` haben ebenfalls ``.label`` und ``.children``; die
    Zeichen- und Textfunktionen arbeiten mit beiden (Duck-Typing).
    ``depth()`` zählt die Ebenen (Blatt = 1, wie ``Formula.depth``),
    ``size()`` die Knoten.
    """

    __slots__ = ("label", "children", "kind", "value")

    def __init__(self, label, children=(), kind: str = "", value=None):
        self.label = str(label)
        self.children = tuple(children)
        for c in self.children:
            if not hasattr(c, "children") or not hasattr(c, "label"):
                raise ValueError("Die Teilbäume eines Knotens müssen selbst Knoten sein.")
        self.kind = kind
        self.value = value

    @property
    def op(self) -> str:
        return self.kind or self.label

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def depth(self) -> int:
        return 1 + max((c.depth() for c in self.children), default=0)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def __eq__(self, other):
        return (isinstance(other, Node) and self.label == other.label
                and self.kind == other.kind and self.children == other.children)

    def __hash__(self):
        return hash((self.label, self.kind, self.children))

    def __repr__(self):
        if self.is_leaf:
            return f"Node({self.label!r})"
        return f"Node({self.label!r}, {self.children!r})"

    def __str__(self):
        return _infix(self)


def _kids(n) -> tuple:
    return tuple(n.children)


def _is_leaf(n) -> bool:
    return not _kids(n)


_ARITH_PREC = {"+": 1, "-": 1, "*": 2, "/": 2, "//": 2, "%": 2, "neg": 3, "**": 4,
               "num": 5, "name": 5}


def _infix(n) -> str:
    """Teilterm als Text (für Formeln: Unicode-Schreibweise, für Terme: infix)."""
    if isinstance(n, Formula):
        return _fs(n)
    kind = getattr(n, "kind", "") or n.label
    kids = _kids(n)
    if not kids:
        return n.label
    if kind == "neg" and len(kids) == 1:
        inner = _infix(kids[0])
        return ("−(" + inner + ")") if _ARITH_PREC.get(_kind(kids[0]), 5) < 3 else "−" + inner
    if len(kids) == 2 and kind in _ARITH_PREC:
        prec = _ARITH_PREC[kind]
        a, b = kids
        sa, sb = _infix(a), _infix(b)
        pa, pb = _ARITH_PREC.get(_kind(a), 5), _ARITH_PREC.get(_kind(b), 5)
        if kind == "**":
            if pa <= prec:
                sa = f"({sa})"
            if pb < 3:
                sb = f"({sb})"
        else:
            if pa < prec:
                sa = f"({sa})"
            if pb < prec or (pb == prec and kind in ("-", "/", "//", "%")):
                sb = f"({sb})"
        return f"{sa} {n.label} {sb}"
    return n.label + "(" + ", ".join(_infix(c) for c in kids) + ")"


def _kind(n) -> str:
    return getattr(n, "kind", "") or n.label


# ==========================================================================
# Glossar: Begriffe und Definitionen der Vorlesung
# ==========================================================================

@dataclass(frozen=True)
class Definition:
    """Ein Begriff der Vorlesung (Serie 2) mit Definition, Beispiel und Alias-Namen."""
    term: str
    text: str
    example: str = ""
    aliases: tuple = ()
    source: str = ""

    def matches(self, name: str) -> bool:
        key = name.strip().lower()
        return key == self.term.lower() or key in (a.lower() for a in self.aliases)


_DEFINITIONS: list[Definition] = [
    Definition(
        "Aussage",
        "Eine Aussage ist ein Satz (interpretierbarer Text), der entweder wahr oder falsch ist. "
        "Mit 1 codieren wir den Wert wahr, mit 0 den Wert falsch.",
        "Aussagen: «Die Summe von Zwei und Drei ist Fünf.» (wahr), «Vierundvierzig ist eine "
        "Primzahl.» (falsch), «Jede gerade Zahl grösser als Zwei ist die Summe zweier Primzahlen.» "
        "(Goldbach: wahr oder falsch, nur noch unbekannt). Keine Aussagen: «Wie alt wurde "
        "Leonhard Euler?» (Frage), «Sei freundlicher zu deinem Lehrer!» (Aufforderung).",
        ("Proposition", "Aussagen"), "Serie 2, Definition 1"),
    Definition(
        "atomare Aussage",
        "Eine Aussage ist atomar, wenn wir sie nicht in zwei kürzere Aussagen aufteilen können.",
        "«Das Haus steht zwischen Cham und Zug.» ist atomar; «Das Haus ist alt und steht direkt "
        "am See.» nicht (alt  UND  steht am See).",
        ("atomar", "atomare Aussagen"), "Serie 2, Frage 1"),
    Definition(
        "zusammengesetzte Aussage",
        "Atomare Aussagen werden mit Junktoren (¬, ∧, ∨, ⊕, ⇒, ⇔) zu neuen, zusammengesetzten "
        "Aussagen verbunden. Atomare Aussagen bezeichnen wir mit grossen lateinischen Buchstaben "
        "A, B, C, … (aussagenlogische Variablen).",
        "A: «Pierre ist Kanadier.», B: «Pierre ist Schweizer.»; B ∧ ¬A: «Pierre ist Schweizer "
        "und nicht Kanadier.»",
        ("zusammengesetzt", "zusammengesetzte Aussagen"), "Serie 2, Frage 2"),
    Definition(
        "aussagenlogische Variable",
        "Ein grosser lateinischer Buchstabe (A, B, C, …), der für eine atomare Aussage steht "
        "und die Werte 0 oder 1 annehmen kann.",
        "A: «Pierre ist Kanadier.»", ("Variable", "Aussagenvariable"), "Serie 2, Frage 2"),
    Definition(
        "Negation",
        "¬A (nicht): ¬A ist genau dann wahr, wenn A falsch ist.",
        "«Pierre ist nicht Kanadier.»", ("¬", "nicht"), "Serie 2, Frage 2"),
    Definition(
        "Konjunktion",
        "A ∧ B (und): wahr genau dann, wenn A und B beide wahr sind.",
        "«Pierre ist Schweizer und in Bern geboren.»", ("∧", "und"), "Serie 2, Frage 2"),
    Definition(
        "Disjunktion",
        "A ∨ B (oder, nicht ausschliessend): wahr, wenn mindestens eine der beiden Aussagen "
        "wahr ist.",
        "«Pierre ist Kanadier oder Schweizer.»", ("∨", "oder"), "Serie 2, Frage 2"),
    Definition(
        "Kontravalenz",
        "A ⊕ B (entweder … oder, Xor): wahr genau dann, wenn genau eine der beiden Aussagen "
        "wahr ist. Es gilt A ⊕ B ≡ ¬(A ⇔ B).",
        "«Pierre ist entweder in Bern oder in Quebec geboren.»",
        ("⊕", "Xor", "exklusives Oder", "entweder oder"), "Serie 2, Frage 2"),
    Definition(
        "Implikation",
        "A ⇒ B (wenn A, dann B): falsch genau dann, wenn A wahr und B falsch ist, sonst wahr. "
        "A heisst Prämisse (Annahme), B Konklusion (Folgerung). Merksatz: nur linke Seite wahr "
        "und rechte Seite falsch gibt 0, sonst überall 1. Ist die Prämisse falsch, so ist die "
        "Implikation wahr.",
        "«Wenn Pierre in Quebec geboren ist, dann ist er Kanadier.»",
        ("⇒", "→", "wenn dann", "wenn-dann"), "Serie 2, Frage 2"),
    Definition(
        "Prämisse",
        "Die linke Aussage einer Implikation (Annahme, Hypothese).", "D in D ⇒ A",
        ("Annahme", "Hypothese"), "Serie 2, Frage 2"),
    Definition(
        "Konklusion",
        "Die rechte Aussage einer Implikation (Folgerung).", "A in D ⇒ A",
        ("Folgerung",), "Serie 2, Frage 2"),
    Definition(
        "Äquivalenz",
        "A ⇔ B (genau dann, wenn): wahr genau dann, wenn A und B denselben Wahrheitswert haben. "
        "Es gilt A ⇔ B ≡ (A ⇒ B) ∧ (B ⇒ A).",
        "«Pierre ist genau dann in Quebec geboren, wenn er Kanadier ist.»",
        ("⇔", "↔", "genau dann wenn", "Bikonditional"), "Serie 2, Frage 2"),
    Definition(
        "aussagenlogischer Term",
        "Definition 2: (1) Eine atomare Aussage ist ein aussagenlogischer Term. (2) Ist A ein "
        "Term, so ist auch ¬A ein Term. (3) Sind A und B Terme, so sind auch A ∧ B, A ∨ B, "
        "A ⊕ B, A ⇒ B und A ⇔ B Terme.",
        "((A ⇒ B) ∧ (¬B)) ⇔ (¬A)", ("Term", "Formel"), "Serie 2, Definition 2"),
    Definition(
        "Syntaxbaum",
        "Zeichnung des Aufbaus eines Terms von oben nach unten: Die Wurzel ist der Operator, "
        "der zuletzt angewendet wird; die Blätter sind die atomaren Aussagen (bzw. Zahlen und "
        "Variablen bei arithmetischen Termen). Jeder Knoten ist ein Operator oder ein Blatt; "
        "die Äste unter einem Knoten heissen Teilbäume.",
        "2 · x + y hat + als Wurzel, links den Teilbaum 2 · x, rechts das Blatt y.",
        ("Baum", "Parsebaum", "Knoten", "Teilbaum", "Wurzel", "Blatt"), "Serie 2, Aufgabe 1"),
    Definition(
        "Signatur",
        "Die Menge aller atomaren Aussagen in einem aussagenlogischen Term.",
        "Signatur von (Q ⇒ P) ∧ ¬P: {P, Q}", ("Variablenmenge",), "Serie 2, Frage 3"),
    Definition(
        "Interpretation",
        "Eine Belegung aller atomaren Aussagen der Signatur mit Wahrheitswerten. Bei n atomaren "
        "Aussagen gibt es 2^n Interpretationen (= Zeilen der Wahrheitstabelle).",
        "Für die Signatur {P, Q}: vier Interpretationen.", ("Belegung",), "Serie 2, Frage 3"),
    Definition(
        "erfüllbar",
        "Ein Term heisst erfüllbar, wenn er für mindestens eine Interpretation der Signatur "
        "wahr ist (diese Interpretation heisst Modell).",
        "(Q ⇒ P) ∧ ¬P ist erfüllbar: nur in der Zeile P = 0, Q = 0 wahr.",
        ("erfüllbarkeit", "Modell"), "Serie 2, Definition 3"),
    Definition(
        "allgemeingültig",
        "Ein Term heisst allgemeingültig (Tautologie), wenn er für alle Interpretationen der "
        "Signatur wahr ist. Schreibweise: T.",
        "A ∨ ¬A ≡ T", ("Tautologie", "T"), "Serie 2, Definition 3"),
    Definition(
        "unerfüllbar",
        "Ein Term heisst unerfüllbar (Kontradiktion), wenn er für keine Interpretation der "
        "Signatur wahr ist. Schreibweise: F.",
        "A ∧ ¬A ≡ F", ("Kontradiktion", "Widerspruch", "F"), "Serie 2, Definition 3"),
    Definition(
        "Äquivalenz von Termen",
        "Zwei Terme sind aussagenlogisch äquivalent (gleichwertig), wenn sie für alle "
        "Interpretationen den gleichen Wahrheitswert haben. Schreibweise: T1 ≡ T2.",
        "(Q ⇒ P) ∧ ¬P ≡ ¬(P ∨ Q)", ("äquivalent", "gleichwertig", "logische Äquivalenz"),
        "Serie 2, Frage 3"),
    Definition(
        "Konsistenz",
        "Eine Spezifikation (Menge von Anforderungen) heisst konsistent, wenn es eine "
        "Interpretation gibt, für die jede Anforderung wahr ist. Konsistenz = Erfüllbarkeit "
        "der Konjunktion aller Anforderungen.",
        "Serie 2, Aufgabe 2: A1 … A5 über L, Q, B, N.", ("konsistent", "Spezifikation"),
        "Serie 2, Aufgabe 2"),
    Definition(
        "SAT",
        "Das Erfüllbarkeitsproblem der Aussagenlogik (satisfiability): Gibt es zu einem Term "
        "eine Interpretation, die ihn wahr macht? Im schlimmsten Fall müssen alle 2^n Zeilen "
        "der Wahrheitstabelle geprüft werden; kein effizienter Algorithmus bekannt. SAT liegt "
        "in der Klasse NP: eine Lösung lässt sich schnell prüfen (eine Zeile ausrechnen). "
        "SAT-Solver lösen in der Praxis auch sehr grosse Fälle.",
        "50 atomare Aussagen: 2^50 Zeilen.", ("Erfüllbarkeitsproblem", "SAT-Problem", "NP"),
        "Serie 2, Frage 4"),
    Definition(
        "Vollständigkeit von ¬, ∧, ∨",
        "Jede aussagenlogische Aussage lässt sich allein mit Negation, Konjunktion und "
        "Disjunktion ausdrücken: A ⇒ B ≡ ¬A ∨ B und A ⇔ B ≡ (A ⇒ B) ∧ (B ⇒ A).",
        "A ⊕ B ≡ (A ∨ B) ∧ (¬A ∨ ¬B)", ("Junktoren", "Basis"), "Serie 2, Frage 4"),
    Definition(
        "Kanonische DNF",
        "Disjunktive Normalform aus der Wahrheitstabelle: Disjunktion der Minterme (Konjunktion "
        "aller Variablen oder ihrer Negationen) der Zeilen, in denen der Term wahr ist.",
        "A ⊕ B ≡ (¬A ∧ B) ∨ (A ∧ ¬B)", ("DNF", "Minterm"), "Prüfungsvorbereitung"),
    Definition(
        "Kanonische KNF",
        "Konjunktive Normalform aus der Wahrheitstabelle: Konjunktion der Maxterme (Disjunktion "
        "aller Variablen, die in der Zeile 0 sind, und der Negationen der Variablen, die 1 "
        "sind) der Zeilen, in denen der Term falsch ist.",
        "A ⊕ B ≡ (A ∨ B) ∧ (¬A ∨ ¬B)", ("KNF", "Maxterm"), "Prüfungsvorbereitung"),
]


class GlossaryResult(Result):
    """Begriffe der Vorlesung (Aussagenlogik). ``entries`` ist eine Liste von ``Definition``."""

    title = "Begriffe der Aussagenlogik"

    def __init__(self, entries: list[Definition], query: str = ""):
        super().__init__()
        self.entries = entries
        self.query = query
        self.summary = (f"Begriff «{query}»" if query else
                        f"{len(entries)} Begriffe aus Serie 2 und der Prüfungsvorbereitung.")
        for d in entries:
            self.steps.append(Step(d.term, d.text,
                                   items=[("Beispiel", d.example)] if d.example else [],
                                   note=d.source))
        self.highlights = [("Anzahl Begriffe", len(entries))]

    def __getitem__(self, term: str) -> Definition:
        for d in self.entries:
            if d.matches(term):
                return d
        raise KeyError(term)

    def __len__(self):
        return len(self.entries)


def definitions() -> list[Definition]:
    """Alle Begriffe und Definitionen der Vorlesung als Liste von ``Definition``."""
    return list(_DEFINITIONS)


def glossary(term: str | None = None) -> GlossaryResult:
    """
    Begriffe der Vorlesung nachschlagen.

        glossary()                 # alle Begriffe
        glossary("Tautologie")     # eine Definition (auch über Alias-Namen)
    """
    if term is None:
        return GlossaryResult(list(_DEFINITIONS))
    if not isinstance(term, str) or not term.strip():
        raise ValueError("Bitte einen Begriff als Text angeben, z. B. glossary('Tautologie').")
    hits = [d for d in _DEFINITIONS if d.matches(term)]
    if not hits:
        low = term.strip().lower()
        hits = [d for d in _DEFINITIONS if low in d.term.lower()]
    if not hits:
        known = ", ".join(d.term for d in _DEFINITIONS)
        raise ValueError(f"Unbekannter Begriff {term!r}. Bekannt sind: {known}.")
    return GlossaryResult(hits, term.strip())


# ==========================================================================
# Wahrheitstabellen der Grundverknüpfungen
# ==========================================================================

_CONNECTIVES = {
    # Name: (Symbol, Formel, Definitionssatz, Beispielsatz)
    "Negation": ("¬", "¬A", "¬A ist wahr genau dann, wenn A falsch ist.",
                 "«Pierre ist nicht Kanadier.»"),
    "Konjunktion": ("∧", "A ∧ B", "A ∧ B ist wahr genau dann, wenn A und B beide wahr sind.",
                    "«Pierre ist Schweizer und in Bern geboren.»"),
    "Disjunktion": ("∨", "A ∨ B", "A ∨ B ist wahr, wenn mindestens eine der beiden wahr ist "
                    "(nicht ausschliessendes Oder).", "«Pierre ist Kanadier oder Schweizer.»"),
    "Kontravalenz": ("⊕", "A ⊕ B", "A ⊕ B ist wahr genau dann, wenn genau eine der beiden "
                     "wahr ist (entweder oder).",
                     "«Pierre ist entweder in Bern oder in Quebec geboren.»"),
    "Implikation": ("⇒", "A ⇒ B", "A ⇒ B ist falsch genau dann, wenn A wahr und B falsch ist.",
                    "«Wenn Pierre in Quebec geboren ist, dann ist er Kanadier.»"),
    "Äquivalenz": ("⇔", "A ⇔ B", "A ⇔ B ist wahr genau dann, wenn A und B denselben "
                   "Wahrheitswert haben.",
                   "«Pierre ist genau dann in Quebec geboren, wenn er Kanadier ist.»"),
}
_CONN_ALIASES = {
    "not": "Negation", "nicht": "Negation", "¬": "Negation",
    "and": "Konjunktion", "und": "Konjunktion", "∧": "Konjunktion",
    "or": "Disjunktion", "oder": "Disjunktion", "∨": "Disjunktion",
    "xor": "Kontravalenz", "⊕": "Kontravalenz", "exor": "Kontravalenz",
    "implies": "Implikation", "implikation": "Implikation", "⇒": "Implikation",
    "→": "Implikation", "->": "Implikation",
    "iff": "Äquivalenz", "aequivalenz": "Äquivalenz", "äquivalenz": "Äquivalenz",
    "⇔": "Äquivalenz", "↔": "Äquivalenz", "<->": "Äquivalenz",
}
IMPLICATION_RULE = ("Nur linke Seite wahr, rechte Seite falsch: dann 0, sonst überall 1. "
                    "Ist die Prämisse falsch, ist die Implikation automatisch wahr.")


def _conn_name(name: str) -> str:
    key = str(name).strip()
    if key in _CONNECTIVES:
        return key
    low = key.lower()
    for k in _CONNECTIVES:
        if k.lower() == low:
            return k
    if low in _CONN_ALIASES:
        return _CONN_ALIASES[low]
    raise ValueError(f"Unbekannte Verknüpfung {name!r}. Bekannt: {', '.join(_CONNECTIVES)}.")


def _conn_formula(name: str) -> Formula:
    a, b = Var("A"), Var("B")
    return {"Negation": Not(a), "Konjunktion": And(a, b), "Disjunktion": Or(a, b),
            "Kontravalenz": Xor(a, b), "Implikation": Implies(a, b), "Äquivalenz": Iff(a, b)}[name]


# ponytail: Tabellen werden vollständig aufgebaut (höchstens 2^12 Zeilen), nicht seitenweise
def _simple_table(f: Formula, true_first: bool = False) -> Table:
    names = f.variables()
    rows, marks = [], {}
    for i, env in enumerate(assignments(names, true_first)):
        v = f.eval(env)
        rows.append([bit(env[n]) for n in names] + [bit(v)])
        if v:
            marks[i] = "ok"
    return Table(names + [_col(f)], rows, marks)


class ConnectiveTablesResult(Result):
    """Wahrheitstabellen der Grundverknüpfungen. ``tables`` = {Name: Table}."""

    title = "Wahrheitstabellen der Grundverknüpfungen"

    def __init__(self, names: list[str], true_first: bool = False):
        super().__init__()
        self.names = names
        self.tables: dict[str, Table] = {}
        self.formulas: dict[str, Formula] = {}
        self.summary = ("Zeilenreihenfolge " + ("1 1 / 1 0 / 0 1 / 0 0" if true_first
                                                 else "0 0 / 0 1 / 1 0 / 1 1") + ".")
        for n in names:
            sym, shown, definition, example = _CONNECTIVES[n]
            f = _conn_formula(n)
            self.formulas[n] = f
            t = _simple_table(f, true_first)
            self.tables[n] = t
            self.steps.append(Step(f"{n} {sym}", definition,
                                   items=[("Beispiel", example)], table=t,
                                   note=IMPLICATION_RULE if n == "Implikation" else "",
                                   rule=n))
        self.highlights = [("Merksatz Implikation", IMPLICATION_RULE)] \
            if "Implikation" in names else []


def connective_tables(true_first: bool = False) -> ConnectiveTablesResult:
    """
    Wahrheitstabellen von ¬, ∧, ∨, ⊕, ⇒, ⇔ (Serie 2, Frage 2) mit dem Merksatz zur Implikation.

        connective_tables().tables["Implikation"]
    """
    return ConnectiveTablesResult(list(_CONNECTIVES), true_first)


def connective_table(name: str, true_first: bool = False) -> ConnectiveTablesResult:
    """Wahrheitstabelle einer einzelnen Verknüpfung: ``connective_table("Implikation")``."""
    return ConnectiveTablesResult([_conn_name(name)], true_first)


# ==========================================================================
# Wahrheitstabelle eines Terms
# ==========================================================================

def _classification(values: list[bool]) -> tuple[str, str]:
    if all(values):
        return "Tautologie", "allgemeingültig (Tautologie, T)"
    if not any(values):
        return "Kontradiktion", "unerfüllbar (Kontradiktion, F)"
    return "erfüllbar (kontingent)", "erfüllbar, aber nicht allgemeingültig"


def _env_text(env: dict, names) -> str:
    return ", ".join(f"{n} = {bit(env[n])}" for n in names) if names else "(keine Variablen)"


class TruthTableResult(Result):
    """
    Wahrheitstabelle eines Terms Schritt für Schritt.

    Attribute: ``formula``, ``variables``, ``columns`` (Teilformeln innen -> aussen, die
    letzte ist die Formel selbst), ``rows`` (Liste der Belegungen), ``values``
    (je Spalte eine Liste von bool), ``vector`` (Werte der Formel), ``table``,
    ``models`` / ``counter_models`` (Belegungen mit wahr / falsch), ``classification``
    ('Tautologie', 'Kontradiktion' oder 'erfüllbar (kontingent)') sowie
    ``is_tautology``, ``is_contradiction``, ``is_satisfiable``, ``is_contingent``.
    """

    title = "Wahrheitstabelle"

    def __init__(self, f: Formula, true_first: bool = False):
        super().__init__()
        names = f.variables()
        _check_vars(names)
        self.formula, self.variables, self.true_first = f, names, true_first
        self.columns = _compound_columns(f)
        self.rows = assignments(names, true_first)
        self.values = {c: [c.eval(env) for env in self.rows] for c in self.columns}
        self.vector = self.values[f]
        self.models = [env for env, v in zip(self.rows, self.vector) if v]
        self.counter_models = [env for env, v in zip(self.rows, self.vector) if not v]
        self.classification, long_text = _classification(self.vector)
        self.is_tautology = self.classification == "Tautologie"
        self.is_contradiction = self.classification == "Kontradiktion"
        self.is_satisfiable = not self.is_contradiction
        self.is_contingent = self.is_satisfiable and not self.is_tautology
        header = names + [_col(c) for c in self.columns]
        rows, marks = [], {}
        for i, env in enumerate(self.rows):
            rows.append([bit(env[n]) for n in names] + [bit(self.values[c][i]) for c in self.columns])
            if self.vector[i]:
                marks[i] = "ok"
        self.table = Table(header, rows, marks,
                           "Grün: Zeilen, in denen der ganze Term wahr ist.")
        self.title = f"Wahrheitstabelle von {_fs(f)}"
        self.summary = f"{_fs(f)} ist {long_text}."

        n = len(names)
        sig = "{" + ", ".join(names) + "}" if names else "∅"
        self.steps.append(Step(
            "Signatur und Interpretationen",
            f"Die Signatur ist die Menge der atomaren Aussagen; bei {n} Variablen gibt es "
            f"2^{n} = {2 ** n} Interpretationen.",
            items=[("Signatur", sig), ("Zeilen", 2 ** n)], rule="Definition 3"))
        inner = [c for c in self.columns if c != f]
        self.steps.append(Step(
            "Teilformeln von innen nach aussen",
            "Für jede zusammengesetzte Teilformel gibt es eine Spalte; die letzte Spalte ist "
            "der ganze Term." if inner else "Der Term hat keine echten Teilformeln.",
            items=[(f"Spalte {i}", c) for i, c in enumerate(self.columns, 1)]))
        self.steps.append(Step(
            "Tabelle ausfüllen",
            "Zeilenreihenfolge " + ("1 1 / 1 0 / 0 1 / 0 0." if true_first else
                                    "0 0 / 0 1 / 1 0 / 1 1."), table=self.table))
        ones = sum(self.vector)
        self.steps.append(Step(
            "Einstufung",
            f"Der Term ist in {ones} von {len(self.vector)} Zeilen wahr.",
            items=[("Ergebnis", long_text)],
            rule={"Tautologie": "allgemeingültig", "Kontradiktion": "unerfüllbar"}
            .get(self.classification, "erfüllbar")))
        if self.models and not self.is_tautology:
            shown = "; ".join(_env_text(m, names) for m in self.models[:16])
            if len(self.models) > 16:
                shown += f"; … ({len(self.models)} Modelle)"
            self.steps.append(Step("Modelle", "Belegungen, die den Term wahr machen.",
                                   items=[("Modelle", shown)]))
        elif not self.models:
            self.steps.append(Step("Modelle", "Keine Belegung macht den Term wahr."))
        self.highlights = [("Einstufung", self.classification),
                           ("Wahrheitsvektor", "".join(bit(v) for v in self.vector[:64]))]

    def __bool__(self):
        return True


def truth_table(f, true_first: bool = False) -> TruthTableResult:
    """
    Wahrheitstabelle mit einer Spalte je Teilformel (innen -> aussen) und Einstufung.

        r = truth_table("(Q -> P) and not P")
        r.classification        # 'erfüllbar (kontingent)'
        r.models                # [{'P': False, 'Q': False}]
        print(r)

    ``true_first=True`` beginnt mit 1 1 (Reihenfolge des PDF-Blatts).
    """
    return TruthTableResult(parse(f), true_first)


def classify(f, true_first: bool = False) -> TruthTableResult:
    """Wie ``truth_table``: erfüllbar / allgemeingültig / unerfüllbar (Serie 2, Aufgabe 3)."""
    return truth_table(f, true_first)


class SignatureResult(Result):
    """Signatur eines Terms und Anzahl der Interpretationen."""

    title = "Signatur"

    def __init__(self, f: Formula):
        super().__init__()
        self.formula, self.names = f, f.variables()
        self.count = 2 ** len(self.names)
        self.summary = (f"Die Signatur von {_fs(f)} hat {len(self.names)} Variablen und damit "
                        f"{self.count} Interpretationen.")
        sig = "{" + ", ".join(self.names) + "}" if self.names else "∅"
        self.steps = [
            Step("Atomare Aussagen sammeln", "Alle Variablen des Terms ergeben die Signatur.",
                 items=[("Signatur", sig)], rule="Signatur"),
            Step("Interpretationen zählen",
                 "Jede Variable hat zwei Werte (0 oder 1): Produktregel, 2 · 2 · … · 2.",
                 items=[("Anzahl", f"2^{len(self.names)} = {self.count}")]),
        ]
        self.highlights = [("Signatur", sig), ("Interpretationen", self.count)]


def signature(f) -> SignatureResult:
    """Signatur und Anzahl Interpretationen (Lerngerüst 2b): ``signature("A & ~(B -> (P | Q)) & C")``."""
    f = parse(f)
    return SignatureResult(f)


class TableSizeResult(Result):
    """Zeilenzahl einer Wahrheitstabelle bei n atomaren Aussagen (Aufgabe 5)."""

    title = "Grösse der Wahrheitstabelle"

    def __init__(self, n: int):
        super().__init__()
        self.n, self.rows = n, 2 ** n
        self.summary = (f"Bei {n} atomaren Aussagen hat die Wahrheitstabelle 2^{n} = "
                        f"{self.rows} Zeilen.")
        self.steps = [
            Step("Zeilen zählen", "Jede der n Variablen ist 0 oder 1, die Wahl ist unabhängig.",
                 items=[("Zeilen", f"2^{n} = {self.rows}")], rule="Produktregel"),
            Step("SAT", "Im schlimmsten Fall müssen alle Zeilen geprüft werden, um die "
                 "Erfüllbarkeit zu entscheiden (SAT, Klasse NP). Eine einzelne Zeile "
                 "auszurechnen ist dagegen effizient.", rule="SAT"),
        ]
        self.highlights = [("Zeilen", self.rows)]


def table_size(n: int) -> TableSizeResult:
    """Zeilenzahl 2^n (Serie 2, Aufgabe 5): ``table_size(50).rows == 2**50``."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("Die Anzahl atomarer Aussagen muss eine ganze Zahl ≥ 0 sein.")
    if n > 10_000:
        raise ValueError("n ist zu gross (höchstens 10000).")
    return TableSizeResult(n)


class SatResult(Result):
    """Ergebnis der Brute-Force-Suche nach einem Modell (SAT)."""

    title = "Erfüllbarkeit (SAT)"

    def __init__(self, f: Formula):
        super().__init__()
        names = f.variables()
        _check_vars(names)
        self.formula, self.model, self.checked = f, None, 0
        for env in assignments(names):
            self.checked += 1
            if f.eval(env):
                self.model = env
                break
        self.satisfiable = self.model is not None
        total = 2 ** len(names)
        if self.satisfiable:
            self.summary = (f"{_fs(f)} ist erfüllbar; Modell nach {self.checked} von {total} "
                            f"Zeilen gefunden.")
        else:
            self.summary = f"{_fs(f)} ist unerfüllbar: alle {total} Zeilen sind falsch."
        self.steps = [Step("Zeilen der Wahrheitstabelle durchprobieren",
                           "Brute Force, im schlimmsten Fall 2^n Zeilen.",
                           items=[("geprüfte Zeilen", self.checked), ("Zeilen insgesamt", total)],
                           rule="SAT")]
        if self.satisfiable:
            self.steps.append(Step("Modell", "Diese Interpretation macht den Term wahr.",
                                   items=[("Modell", _env_text(self.model, names))]))
        self.highlights = [("erfüllbar", self.satisfiable)]


def sat(f) -> SatResult:
    """Sucht per Brute Force ein Modell: ``sat("A and not A").satisfiable == False``."""
    return SatResult(parse(f))


# ==========================================================================
# Spezifikation / Konsistenz (Aufgabe 2)
# ==========================================================================

class ConsistencyResult(Result):
    """
    Konsistenz einer Spezifikation: gibt es eine Interpretation, die alle Anforderungen wahr macht?

    Attribute: ``requirements`` (dict Name -> Formula), ``consistent``, ``models``, ``table``.
    """

    title = "Konsistenz einer Spezifikation"

    def __init__(self, reqs: dict[str, Formula], true_first: bool = False):
        super().__init__()
        self.requirements = reqs
        conj = big_and(reqs.values())
        names = _vars_of(*reqs.values())
        _check_vars(names, "Die Spezifikation")
        self.conjunction, self.variables = conj, names
        self.rows = assignments(names, true_first)
        self.values = {k: [f.eval(e) for e in self.rows] for k, f in reqs.items()}
        self.all_true = [all(self.values[k][i] for k in reqs) for i in range(len(self.rows))]
        self.models = [e for e, ok in zip(self.rows, self.all_true) if ok]
        self.consistent = bool(self.models)
        header = names + list(reqs) + ["alle"]
        rows = [[bit(e[n]) for n in names] + [bit(self.values[k][i]) for k in reqs]
                + [bit(self.all_true[i])] for i, e in enumerate(self.rows)]
        marks = {i: "ok" for i, ok in enumerate(self.all_true) if ok}
        self.table = Table(header, rows, marks, "Grün: alle Anforderungen sind wahr.")
        self.summary = ("Die Spezifikation ist konsistent." if self.consistent else
                        "Die Spezifikation ist inkonsistent (unerfüllbar).")
        self.steps = [
            Step("Anforderungen als Terme", "Jede Anforderung ist ein aussagenlogischer Term.",
                 items=[(k, f) for k, f in reqs.items()]),
            Step("Wahrheitstabelle", f"{len(self.rows)} Interpretationen; die letzte Spalte ist "
                 f"wahr, wenn alle Anforderungen gleichzeitig wahr sind.", table=self.table),
            Step("Konsistenz = Erfüllbarkeit",
                 "Konsistent heisst: mindestens eine Zeile, in der alle Anforderungen wahr sind.",
                 items=[("Modelle", "; ".join(_env_text(m, names) for m in self.models[:16])
                         or "keine")], rule="Erfüllbarkeit = Konsistenz"),
        ]
        self.highlights = [("konsistent", self.consistent), ("Anzahl Modelle", len(self.models))]


def check_consistency(requirements, true_first: bool = False) -> ConsistencyResult:
    """
    Prüft, ob eine Menge von Anforderungen gemeinsam erfüllbar ist (Serie 2, Aufgabe 2).

        check_consistency({"A1": "~L -> Q", "A2": "~L -> B", "A3": "~L <-> N",
                           "A4": "~Q -> B", "A5": "~B"}).consistent    # True

    ``requirements``: dict Name -> Formel oder Liste von Formeln (Namen A1, A2, …).
    """
    if isinstance(requirements, dict):
        reqs = {str(k): parse(v) for k, v in requirements.items()}
    else:
        items = list(requirements)
        reqs = {f"A{i}": parse(v) for i, v in enumerate(items, 1)}
    if not reqs:
        raise ValueError("Es wurde keine Anforderung angegeben.")
    if len(reqs) > 30:
        raise ValueError("Zu viele Anforderungen (höchstens 30).")
    return ConsistencyResult(reqs, true_first)


# ==========================================================================
# Äquivalenz und Folgerung
# ==========================================================================

def _merged_columns(*fs: Formula) -> list[Formula]:
    cols: list[Formula] = []
    for f in fs:
        for c in _compound_columns(f):
            if c not in cols:
                cols.append(c)
    return cols


class EquivalenceResult(Result):
    """
    Äquivalenzprüfung f ≡ g mit Wahrheitstabelle.

    Attribute: ``f``, ``g``, ``equivalent``, ``table``, ``counterexamples`` (Belegungen mit
    verschiedenen Werten), ``counterexample`` (die erste oder None), ``variables``.
    """

    title = "Äquivalenz prüfen"

    def __init__(self, f: Formula, g: Formula, true_first: bool = False):
        super().__init__()
        names = _vars_of(f, g)
        _check_vars(names, "Die beiden Terme enthalten zusammen")
        self.f, self.g, self.variables = f, g, names
        cols = _merged_columns(f, g)           # Spalten von f innen -> aussen, dann die von g
        self.columns = cols
        self.rows = assignments(names, true_first)
        fv = [f.eval(e) for e in self.rows]
        gv = [g.eval(e) for e in self.rows]
        self.counterexamples = [e for e, a, b in zip(self.rows, fv, gv) if a != b]
        self.counterexample = self.counterexamples[0] if self.counterexamples else None
        self.equivalent = not self.counterexamples
        rows, marks = [], {}
        for i, e in enumerate(self.rows):
            rows.append([bit(e[n]) for n in names] + [bit(c.eval(e)) for c in cols]
                        + ["ja" if fv[i] == gv[i] else "nein"])
            if fv[i] != gv[i]:
                marks[i] = "bad"
        self.table = Table(names + [_col(c) for c in cols] + ["gleich?"], rows, marks,
                           "Rot: Gegenbeispiel (die Werte unterscheiden sich).")
        sym = "≡" if self.equivalent else "≢"
        self.title = f"Äquivalenz: {_fs(f)}  {sym}  {_fs(g)}"
        self.summary = (f"Die Terme sind äquivalent: in allen {len(self.rows)} Zeilen gleich."
                        if self.equivalent else
                        f"Die Terme sind nicht äquivalent: {len(self.counterexamples)} von "
                        f"{len(self.rows)} Zeilen unterscheiden sich.")
        self.steps = [
            Step("Gemeinsame Signatur", "Alle Variablen beider Terme bilden die Signatur; "
                 f"{len(self.rows)} Interpretationen.",
                 items=[("Signatur", "{" + ", ".join(names) + "}" if names else "∅")]),
            Step("Wahrheitstabelle beider Terme",
                 "Teilformeln von innen nach aussen, am Ende die beiden Terme.",
                 table=self.table, rule="Wahrheitstabelle"),
            Step("Vergleich der letzten Spalten",
                 "Äquivalent heisst: in jeder Zeile haben beide Terme denselben Wahrheitswert.",
                 items=[("Ergebnis", f"{_fs(f)} {sym} {_fs(g)}")], rule="Äquivalenz"),
        ]
        if self.counterexample is not None:
            e = self.counterexample
            self.steps.append(Step(
                "Gegenbeispiel", "Bei dieser Belegung haben die Terme verschiedene Werte.",
                items=[("Belegung", _env_text(e, names)), (_fs(f), f.eval(e)), (_fs(g), g.eval(e))]))
        self.highlights = [("äquivalent", self.equivalent)]


def check_equivalence(f, g, true_first: bool = False) -> EquivalenceResult:
    """
    Prüft f ≡ g mit Wahrheitstabelle und nennt ein Gegenbeispiel (Serie 2, Aufgabe 4).

        check_equivalence("P -> (Q | R)", "Q | (~P | R)").equivalent     # True
        check_equivalence("P -> (Q | R)", "(Q & ~R) -> P").counterexample # {'P': False, ...}
    """
    return EquivalenceResult(parse(f), parse(g), true_first)


class ImplicationCheckResult(Result):
    """
    Prüft f ⇒ g auf Allgemeingültigkeit, d. h. die Folgerung f ⊨ g.

    Attribute: ``holds`` (f ⊨ g), ``table``, ``counterexamples`` (f wahr, g falsch).
    """

    title = "Folgerung prüfen"

    def __init__(self, f: Formula, g: Formula, true_first: bool = False):
        super().__init__()
        names = _vars_of(f, g)
        _check_vars(names, "Die beiden Terme enthalten zusammen")
        self.f, self.g, self.variables = f, g, names
        imp = Implies(f, g)
        cols = [c for c in _merged_columns(f, g) if c not in (f, g)]
        cols += [f] + ([g] if g != f else []) + [imp]
        self.columns = cols
        self.rows = assignments(names, true_first)
        self.counterexamples = [e for e in self.rows if f.eval(e) and not g.eval(e)]
        self.holds = not self.counterexamples
        rows, marks = [], {}
        for i, e in enumerate(self.rows):
            rows.append([bit(e[n]) for n in names] + [bit(c.eval(e)) for c in cols])
            if f.eval(e) and not g.eval(e):
                marks[i] = "bad"
        self.table = Table(names + [_col(c) for c in cols], rows, marks,
                           "Rot: f wahr, g falsch (Gegenbeispiel).")
        self.title = f"Folgerung: {_fs(f)}  ⊨  {_fs(g)}"
        self.summary = ((f"{_fs(f)} ⇒ {_fs(g)} ist allgemeingültig, also folgt g aus f."
                         if self.holds else
                         f"{_fs(f)} ⇒ {_fs(g)} ist nicht allgemeingültig: g folgt nicht aus f."))
        self.steps = [
            Step("Wahrheitstabelle", "Zeilen, in denen die Prämisse wahr ist, müssen auch die "
                 "Konklusion wahr machen.", table=self.table),
            Step("Auswertung", "f ⊨ g gilt genau dann, wenn f ⇒ g allgemeingültig ist, also "
                 "keine Zeile mit f = 1 und g = 0 existiert.",
                 items=[("Gegenbeispiele", len(self.counterexamples))],
                 rule="Folgerung"),
        ]
        self.highlights = [("f ⊨ g", self.holds)]


def check_implication(f, g, true_first: bool = False) -> ImplicationCheckResult:
    """``check_implication("U & W", "U | W").holds`` -> True (Folgerung f ⊨ g)."""
    return ImplicationCheckResult(parse(f), parse(g), true_first)


class ImplicationVariantsResult(Result):
    """
    Varianten von A ⇒ B: Umkehrung, Kontraposition, Inverse, Negation.

    Attribute: ``original``, ``variants`` (dict Name -> Formel), ``equivalent`` (dict Name -> bool,
    äquivalent zur Originalimplikation?), ``table``.
    """

    title = "Varianten der Implikation"

    def __init__(self, a: Formula, b: Formula, true_first: bool = False):
        super().__init__()
        self.a, self.b = a, b
        self.original = Implies(a, b)
        self.variants = {
            "Umkehrung (Konverse)": Implies(b, a),
            "Kontraposition": Implies(_neg(b), _neg(a)),
            "Inverse": Implies(_neg(a), _neg(b)),
            "Negation": Not(self.original),
            "Negation vereinfacht": And(a, _neg(b)),
        }
        names = _vars_of(a, b)
        _check_vars(names, "Die Terme enthalten zusammen")
        self.variables = names
        self.equivalent = {k: equivalent(self.original, v) for k, v in self.variants.items()}
        self.negation_ok = equivalent(self.variants["Negation"], self.variants["Negation vereinfacht"])
        cols = [self.original] + list(self.variants.values())
        rows = [[bit(e[n]) for n in names] + [bit(c.eval(e)) for c in cols]
                for e in assignments(names, true_first)]
        self.table = Table(names + [_col(c) for c in cols], rows)
        self.summary = ("Die Kontraposition ist äquivalent zur Implikation; Umkehrung und Inverse "
                        "sind es im Allgemeinen nicht.")
        self.steps = [Step("Ausgangsimplikation", "A heisst Prämisse, B Konklusion.",
                           items=[("Original", self.original)])]
        for k, v in self.variants.items():
            eq = self.equivalent[k]
            self.steps.append(Step(
                k, "äquivalent zur Implikation" if eq else "nicht äquivalent zur Implikation",
                items=[(k, v), ("äquivalent?", eq)],
                rule="Kontraposition" if k == "Kontraposition" and eq else ""))
        self.steps.append(Step("Wahrheitstabelle", "Alle Varianten nebeneinander.",
                               table=self.table))
        self.highlights = [(k, v) for k, v in self.variants.items()]


def implication_variants(a, b=None, true_first: bool = False) -> ImplicationVariantsResult:
    """
    Umkehrung, Kontraposition, Inverse und Negation von A ⇒ B samt Äquivalenzprüfung.

        implication_variants("A", "B").equivalent["Kontraposition"]   # True
        implication_variants("A -> B")                                # eine Implikation genügt
    """
    fa = parse(a)
    if b is None:
        if not isinstance(fa, Implies):
            raise ValueError("Bitte eine Implikation A -> B oder zwei Terme (Prämisse, Konklusion) angeben.")
        return ImplicationVariantsResult(fa.a, fa.b, true_first)
    return ImplicationVariantsResult(fa, parse(b), true_first)


# ==========================================================================
# Äquivalenzgesetze (Tabelle) und Mustererkennung
# ==========================================================================

def _mask(f: Formula, names: list, cache: dict | None = None) -> int:
    """Wahrheitsvektor von f als grosse Ganzzahl (Bit i = Zeile i); sehr schnell zum Prüfen."""
    n = len(names)
    full = (1 << (1 << n)) - 1
    cache = {} if cache is None else cache
    if not cache:
        for k, name in enumerate(names):
            # Zeile r (0 0 / 0 1 / 1 0 / …): erste Variable ist das höchste Bit von r
            period = 1 << (n - 1 - k)
            block = ((1 << period) - 1) << period          # Zeilen, in denen die Variable 1 ist
            m, step = 0, period * 2
            for start in range(0, 1 << n, step):
                m |= block << start
            cache[Var(name)] = m

    def ev(g: Formula) -> int:
        if g in cache:
            return cache[g]
        if isinstance(g, Const):
            v = full if g.value else 0
        elif isinstance(g, Not):
            v = ~ev(g.a) & full
        else:
            a, b = ev(g.a), ev(g.b)
            if isinstance(g, And):
                v = a & b
            elif isinstance(g, Or):
                v = a | b
            elif isinstance(g, Xor):
                v = a ^ b
            elif isinstance(g, Implies):
                v = (~a | b) & full
            else:
                v = ~(a ^ b) & full
        cache[g] = v
        return v

    return ev(f)


def _same(f: Formula, g: Formula) -> bool:
    """Schnelle Äquivalenzprüfung per Wahrheitsvektor (Sicherheitsnetz für jeden Schritt)."""
    names = _vars_of(f, g)
    _check_vars(names, "Die Terme enthalten zusammen")
    return _mask(f, names) == _mask(g, names)


@dataclass(frozen=True)
class Law:
    """
    Ein Äquivalenzgesetz ``lhs ≡ rhs`` mit Mustervariablen A, B, C.

    ``reducing`` = darf von links nach rechts beliebig oft angewendet werden, ohne
    dass die Umformung in eine Endlosschleife läuft.
    """
    name: str
    lhs: Formula
    rhs: Formula
    variant: str = ""
    note: str = ""
    reducing: bool = False

    def text(self) -> str:
        return f"{_law_str(self.lhs)}  ≡  {_law_str(self.rhs)}"

    def __str__(self):
        return f"{self.name}{' (' + self.variant + ')' if self.variant else ''}: {self.text()}"


def _law_str(f: Formula) -> str:
    return _fs(f).replace("1", "T").replace("0", "F")


def _L(name, lhs, rhs, variant="", note="", reducing=False) -> Law:
    return Law(name, parse(lhs), parse(rhs), variant, note, reducing)


_LAWS: list[Law] = [
    _L("Doppelnegation", "~~A", "A", reducing=True),
    _L("Idempotenz", "A | A", "A", "∨", reducing=True),
    _L("Idempotenz", "A & A", "A", "∧", reducing=True),
    _L("Dominanz", "A | 1", "1", "∨", reducing=True),
    _L("Dominanz", "A & 0", "0", "∧", reducing=True),
    _L("Identität", "A | 0", "A", "∨", reducing=True),
    _L("Identität", "A & 1", "A", "∧", reducing=True),
    _L("Invers", "A | ~A", "1", "∨", "auch Komplement genannt", reducing=True),
    _L("Invers", "A & ~A", "0", "∧", "auch Komplement genannt", reducing=True),
    _L("Kommutativität", "A | B", "B | A", "∨"),
    _L("Kommutativität", "A & B", "B & A", "∧"),
    _L("Absorption", "A | (A & B)", "A", "∨",
       "Aussen und innen müssen verschiedene Junktoren sein.", reducing=True),
    _L("Absorption", "A & (A | B)", "A", "∧",
       "Aussen und innen müssen verschiedene Junktoren sein.", reducing=True),
    _L("Assoziativität", "(A | B) | C", "A | (B | C)", "∨"),
    _L("Assoziativität", "(A & B) & C", "A & (B & C)", "∧"),
    _L("Distributivität", "A | (B & C)", "(A | B) & (A | C)", "∨ über ∧"),
    _L("Distributivität", "A & (B | C)", "(A & B) | (A & C)", "∧ über ∨"),
    _L("De Morgan", "~(A & B)", "~A | ~B", "∧", "Negation in die Klammer: Junktor kehrt sich um.",
       reducing=True),
    _L("De Morgan", "~(A | B)", "~A & ~B", "∨", "Negation in die Klammer: Junktor kehrt sich um.",
       reducing=True),
    _L("Implikation", "A -> B", "~A | B", note="Implikation als Disjunktion.", reducing=True),
    _L("Äquivalenz", "A <-> B", "(A -> B) & (B -> A)", note="Äquivalenz als zwei Implikationen.",
       reducing=True),
    _L("Kontravalenz", "A ^ B", "(A | B) & (~A | ~B)", note="Xor mit nicht, und, oder.",
       reducing=True),
    _L("Negation der Implikation", "~(A -> B)", "A & ~B", reducing=True),
    _L("Kontraposition", "A -> B", "~B -> ~A"),
    _L("Äquivalenz (Wertegleichheit)", "A <-> B", "(A & B) | (~A & ~B)"),
    _L("Negation der Kontravalenz", "~(A ^ B)", "A <-> B", reducing=True),
]
_LAW_ALIASES = {"komplement": "invers", "inverse": "invers", "tautologie": "invers",
                "demorgan": "de morgan", "de-morgan": "de morgan", "ausklammern": "distributivität",
                "distributivgesetz": "distributivität", "doppelte negation": "doppelnegation",
                "implikation als disjunktion": "implikation",
                "äquivalenz als zwei implikationen": "äquivalenz"}
_DEFAULT_LAWS = ["Doppelnegation", "Implikation", "Äquivalenz", "Kontravalenz",
                 "Negation der Implikation", "De Morgan", "Dominanz", "Identität", "Invers",
                 "Idempotenz", "Absorption"]
_META = {"A", "B", "C"}


def _norm(name: str) -> str:
    key = str(name).strip().lower()
    return _LAW_ALIASES.get(key, key)


def laws() -> list[Law]:
    """
    Tabelle der Äquivalenzgesetze (Serie 2, Frage 4, sowie Zusammenfassung).

        for g in laws(): print(g)        # z. B. «De Morgan (∧): ¬(A ∧ B)  ≡  ¬A ∨ ¬B»
    """
    return list(_LAWS)


def find_law(name: str) -> list[Law]:
    """Alle Gesetze (Varianten ∧ / ∨ …) zu einem Namen: ``find_law("De Morgan")``."""
    key = _norm(name)
    hits = [g for g in _LAWS if g.name.lower() == key]
    if not hits:
        known = ", ".join(sorted({g.name for g in _LAWS}))
        raise ValueError(f"Unbekanntes Gesetz {name!r}. Bekannt: {known}.")
    return hits


def _match(pat: Formula, f: Formula, b: dict) -> bool:
    if isinstance(pat, Var):
        if pat.name in b:
            return b[pat.name] == f
        b[pat.name] = f
        return True
    if isinstance(pat, Const):
        return isinstance(f, Const) and f.value == pat.value
    if type(pat) is not type(f):
        return False
    return all(_match(p, c, b) for p, c in zip(pat.children, f.children))


def _rebuild(f: Formula, kids) -> Formula:
    if not kids:
        return f
    if isinstance(f, Not):
        return Not(kids[0])
    return type(f)(kids[0], kids[1])


def _inst(pat: Formula, b: dict) -> Formula:
    if isinstance(pat, Var):
        return b[pat.name]
    if isinstance(pat, Const):
        return pat
    return _rebuild(pat, [_inst(c, b) for c in pat.children])


def _positions(f: Formula, path: tuple = ()):
    """(Pfad, Teilformel) in Vorordnung: erst die Formel, dann links, dann rechts."""
    yield path, f
    for i, c in enumerate(f.children):
        yield from _positions(c, path + (i,))


def _replace(f: Formula, path: tuple, new: Formula) -> Formula:
    if not path:
        return new
    kids = list(f.children)
    kids[path[0]] = _replace(kids[path[0]], path[1:], new)
    return _rebuild(f, kids)


def _path_text(f: Formula, path: tuple) -> str:
    sub = f
    for i in path:
        sub = sub.children[i]
    return "auf den ganzen Term" if not path else f"auf den Teilterm {_fs(sub)}"


def _rewrite_candidates(f: Formula, law: Law, reverse: bool):
    """Alle Anwendungen von law (vorwärts oder rückwärts) auf Teilterme von f."""
    src, dst = (law.rhs, law.lhs) if reverse else (law.lhs, law.rhs)
    if reverse and (_vars_in(dst) - _vars_in(src)):
        raise ValueError(f"Das Gesetz «{law.name}» lässt sich nicht rückwärts anwenden, weil "
                         f"die Rückrichtung nicht eindeutig bestimmt ist.")
    for path, sub in _positions(f):
        b: dict = {}
        if _match(src, sub, b):
            yield path, sub, _inst(dst, b)


def _vars_in(f: Formula) -> set:
    return {n for n in f.variables() if n in _META}


class LawApplicationResult(Result):
    """Eine einzelne Gesetzesanwendung: ``before`` -> ``after`` (geprüft äquivalent)."""

    title = "Gesetz anwenden"

    def __init__(self, law: Law, before: Formula, after: Formula, path: tuple, reverse: bool):
        super().__init__()
        self.law, self.before, self.after, self.path, self.reverse = law, before, after, path, reverse
        self.equivalent = _same(before, after)
        arrow = "≡" if not reverse else "≡ (rückwärts)"
        self.title = f"{law.name}{' (' + law.variant + ')' if law.variant else ''}"
        self.summary = f"{_fs(before)}  {arrow}  {_fs(after)}"
        self.steps = [
            Step("Gesetz", law.note or "", items=[("Gesetz", law.text())], rule=law.name),
            Step("Anwendung", _path_text(before, path),
                 items=[("vorher", _fs(before)), ("nachher", _fs(after))]),
            Step("Kontrolle per Wahrheitstabelle",
                 "Beide Terme haben in allen Zeilen denselben Wert.",
                 items=[("äquivalent", self.equivalent)]),
        ]
        self.highlights = [("Ergebnis", _fs(after))]


def apply_law(f, name: str, reverse: bool = False, occurrence: int = 1,
              variant: str | None = None) -> LawApplicationResult:
    """
    Wendet ein benanntes Gesetz genau einmal an (auch Kommutativität, Distributivität …).

        apply_law("¬(A ∧ B)", "De Morgan").after        # ¬A ∨ ¬B
        apply_law("A | (B & C)", "Distributivität").after
        apply_law("(A | B) & (A | C)", "Distributivität", reverse=True).after   # Ausklammern

    ``occurrence`` wählt die n-te Fundstelle (1 = erste, Vorordnung von links nach rechts),
    ``variant`` schränkt auf eine Variante ein (z. B. ``"∧"``).
    """
    f = parse(f)
    cands = []
    group = find_law(name)
    if variant is not None:
        group = [g for g in group if g.variant == variant]
        if not group:
            raise ValueError(f"Das Gesetz {name!r} hat keine Variante {variant!r}.")
    for law in group:
        try:
            for path, sub, new in _rewrite_candidates(f, law, reverse):
                cands.append((law, path, new))
        except ValueError:
            if len(group) == 1:
                raise
    if not cands:
        raise ValueError(f"Das Gesetz «{name}» ist auf {_fs(f)} nicht anwendbar.")
    if not isinstance(occurrence, int) or not 1 <= occurrence <= len(cands):
        raise ValueError(f"Es gibt {len(cands)} Fundstelle(n); occurrence muss zwischen 1 und "
                         f"{len(cands)} liegen.")
    law, path, new = cands[occurrence - 1]
    after = _replace(f, path, new)
    if not _same(f, after):                                      # pragma: no cover
        raise RuntimeError(f"Interner Fehler: {law.name} ist hier nicht äquivalent.")
    return LawApplicationResult(law, f, after, path, reverse)


# ==========================================================================
# Vereinfachen: benannte Schritte, jeder per Wahrheitsvektor geprüft
# ==========================================================================

class SimplifyResult(Result):
    """
    Umformung eines Terms mit benannten Gesetzen.

    Attribute: ``original``, ``result`` (kürzester erreichter Term), ``trace`` (Liste von
    (Gesetz, Term nach dem Schritt) bis zum Ergebnis), ``equivalent`` (immer True: jeder Schritt
    wurde per Wahrheitsvektor geprüft), ``is_tautology``, ``is_contradiction``,
    ``size_before`` / ``size_after`` (Anzahl Knoten im Syntaxbaum).
    """

    title = "Term vereinfachen"

    def __init__(self, original: Formula, trace: list, full_trace: list | None = None):
        super().__init__()
        self.original = original
        self.trace = trace
        self.result = trace[-1][2] if trace else original
        self.equivalent = _same(original, self.result)
        self.size_before, self.size_after = original.size(), self.result.size()
        self.is_tautology = isinstance(self.result, Const) and self.result.value
        self.is_contradiction = isinstance(self.result, Const) and not self.result.value
        self.title = f"Vereinfachen: {_fs(original)}"
        self.steps.append(Step("Ausgangsterm", items=[("", _fs(original))]))
        for law, detail, formula in trace:
            self.steps.append(Step(law, detail, items=[("", _fs(formula))], rule=law))
        if not trace:
            self.summary = "Der Term lässt sich mit den Gesetzen nicht weiter verkürzen."
        elif self.is_tautology:
            self.summary = f"Der Term ist allgemeingültig (≡ T): {_fs(original)} ≡ 1."
        elif self.is_contradiction:
            self.summary = f"Der Term ist unerfüllbar (≡ F): {_fs(original)} ≡ 0."
        else:
            self.summary = f"{_fs(original)}  ≡  {_fs(self.result)}"
        if full_trace is not None and len(full_trace) > len(trace):
            self.warnings.append("Die weiteren Umformungen hätten den Term wieder verlängert; "
                                 "der Rechenweg endet beim kürzesten Term.")
        self.highlights = [("Ergebnis", _fs(self.result))]
        if self.is_tautology:
            self.highlights.append(("Einstufung", "allgemeingültig (Tautologie)"))
        elif self.is_contradiction:
            self.highlights.append(("Einstufung", "unerfüllbar (Kontradiktion)"))


def _map_all(f: Formula, fn) -> Formula:
    """Wendet fn (Formel -> Formel oder None) von unten nach oben auf alle Teilformeln an."""
    kids = [_map_all(c, fn) for c in f.children]
    g = _rebuild(f, kids) if kids else f
    new = fn(g)
    return g if new is None else new


def _pass(f: Formula, fn):
    new = _map_all(f, fn)
    return None if new == f else new


def _fn_iff(g):
    if isinstance(g, Iff):
        return And(Implies(g.a, g.b), Implies(g.b, g.a))


def _fn_xor(g):
    if isinstance(g, Xor):
        return And(Or(g.a, g.b), Or(Not(g.a), Not(g.b)))


def _fn_imp(g):
    if isinstance(g, Implies):
        return Or(Not(g.a), g.b)


def _fn_dn(g):
    if isinstance(g, Not) and isinstance(g.a, Not):
        return g.a.a


def _fn_demorgan(g):
    if isinstance(g, Not):
        if isinstance(g.a, And):
            return Or(Not(g.a.a), Not(g.a.b))
        if isinstance(g.a, Or):
            return And(Not(g.a.a), Not(g.a.b))


def _fn_const(g):
    if isinstance(g, Not) and isinstance(g.a, Const):
        return Const(not g.a.value)


_PREP_PASSES = [   # (Gesetz, Beschreibung, Funktion) — Phase A: Junktoren ersetzen
    ("Äquivalenz", "Äquivalenz als zwei Implikationen ersetzen: A ⇔ B ≡ (A ⇒ B) ∧ (B ⇒ A).", _fn_iff),
    ("Kontravalenz", "Kontravalenz mit nicht, und, oder schreiben: A ⊕ B ≡ (A ∨ B) ∧ (¬A ∨ ¬B).", _fn_xor),
    ("Implikation", "Implikationen als Disjunktion schreiben: A ⇒ B ≡ ¬A ∨ B.", _fn_imp),
]
_NEG_PASSES = [    # Phase B: Negation nach innen
    ("Doppelnegation", "Doppelte Negation streichen: ¬¬A ≡ A.", _fn_dn),
    ("De Morgan", "Negation in die Klammer hineinrechnen, der Junktor kehrt sich um.", _fn_demorgan),
    ("Negation der Konstanten", "¬T ≡ F und ¬F ≡ T.", _fn_const),
]


def _is_lit(x: Formula) -> bool:
    return isinstance(x, Var) or (isinstance(x, Not) and isinstance(x.a, Var))


def _chain_sets(es: list, inner: str) -> list[list]:
    return [flatten(e, inner) for e in es]


def _chain_step(node: Formula):
    """
    Erste anwendbare Regel für eine Kette gleicher Junktoren (∨ oder ∧), oder None.

    Liefert eine Liste von Schritten (Gesetz, Beschreibung, neuer Knoten). Reihenfolge der
    Regeln: Dominanz, Identität, Idempotenz, Invers, Absorption, Ausklammern, abgeleitete Regel.
    Kommutativität und Assoziativität werden dabei still mitbenutzt und im Text genannt.
    """
    is_or = isinstance(node, Or)
    opname = "Or" if is_or else "And"
    inner = "And" if is_or else "Or"
    outr, inn = (Or, And) if is_or else (And, Or)
    dom, unit = (TRUE, FALSE) if is_or else (FALSE, TRUE)
    mk = big_or if is_or else big_and
    mk_in = big_and if is_or else big_or
    es = flatten(node, opname)
    sy, sy_in = ("∨", "∧") if is_or else ("∧", "∨")
    tf = lambda c: "T" if c.value else "F"
    reorder = " (Summanden bei Bedarf mit Kommutativität/Assoziativität umordnen)"

    if dom in es:
        return [("Dominanz", f"A {sy} {tf(dom)} ≡ {tf(dom)}: eine Konstante {tf(dom)} in der "
                 f"Kette macht alles zu {tf(dom)}.", dom)]
    if unit in es:
        rest = [e for e in es if e != unit]
        return [("Identität", f"A {sy} {tf(unit)} ≡ A: die Konstante {tf(unit)} fällt weg.",
                 mk(rest) if rest else unit)]
    sets = _chain_sets(es, inner)
    n = len(es)
    for i in range(n):
        for j in range(i + 1, n):
            if set(sets[i]) == set(sets[j]):
                rest = [e for k, e in enumerate(es) if k != j]
                return [("Idempotenz", f"A {sy} A ≡ A: {_fs(es[j])} kommt doppelt vor."
                         + (reorder if j != i + 1 else ""), mk(rest))]
    for i in range(n):
        for j in range(i + 1, n):
            if es[j] == Not(es[i]) or es[i] == Not(es[j]):
                return [("Invers", f"A {sy} ¬A ≡ {tf(dom)}: {_fs(es[i])} und {_fs(es[j])} "
                         "sind Gegenstücke." + (reorder if j != i + 1 else ""), dom)]
    for i in range(n):
        for j in range(n):
            if i != j and set(sets[i]) < set(sets[j]):
                rest = [e for k, e in enumerate(es) if k != j]
                return [("Absorption", f"A {sy} (A {sy_in} B) ≡ A: {_fs(es[j])} wird von "
                         f"{_fs(es[i])} aufgesogen." + (reorder if abs(i - j) != 1 else ""),
                         mk(rest))]
    for i in range(n):
        for j in range(i + 1, n):
            common = [x for x in sets[i] if x in sets[j]]
            if common:
                ri = [x for x in sets[i] if x not in common]
                rj = [x for x in sets[j] if x not in common]
                merged = mk_in(common + [mk([mk_in(ri), mk_in(rj)])])
                rest = [merged if k == i else e for k, e in enumerate(es) if k != j]
                return [("Distributivität", f"Ausklammern von {', '.join(_fs(x) for x in common)}: "
                         f"(A {sy_in} B) {sy} (A {sy_in} C) ≡ A {sy_in} (B {sy} C)."
                         + (reorder if j != i + 1 else ""), mk(rest))]
    # ponytail: abgeleitete Regel A ∨ (¬A ∧ B) ≡ A ∨ B nur für Literale A, nicht für ganze Teilterme
    for i in range(n):
        x = es[i]
        if not _is_lit(x):
            continue
        comp = _neg(x)
        for j in range(n):
            if j == i or comp not in sets[j] or len(sets[j]) < 2:
                continue
            bpart = mk_in([y for y in sets[j] if y != comp])
            step1 = inn(outr(x, comp), outr(x, bpart))
            with_ = lambda new: mk([new if k == i else e for k, e in enumerate(es) if k != j])
            node1 = with_(step1)
            node2 = with_(inn(dom, outr(x, bpart)))
            node3 = with_(outr(x, bpart))
            return [
                ("Distributivität", f"{_fs(x)} {sy} ({_fs(comp)} {sy_in} B) ≡ "
                 f"({_fs(x)} {sy} {_fs(comp)}) {sy_in} ({_fs(x)} {sy} B)", node1),
                ("Invers", f"{_fs(x)} {sy} {_fs(comp)} ≡ {tf(dom)}.", node2),
                ("Identität", f"{tf(dom)} {sy_in} A ≡ A (mit Kommutativität).", node3),
            ]
    return None


def _find_chain(f: Formula, path: tuple = (), parent_op: str = ""):
    """Erste Kettenregel in Nachordnung (innere Ketten zuerst): (Pfad, Schritte) oder None."""
    for i, c in enumerate(f.children):
        r = _find_chain(c, path + (i,), f.op if isinstance(f, (And, Or)) else "")
        if r:
            return r
    if isinstance(f, (And, Or)) and f.op != parent_op:
        steps = _chain_step(f)
        if steps:
            return path, steps
    return None


def _sort_literal_chains(f: Formula):
    """Nur kosmetisch: Ketten aus Literalen alphabetisch ordnen (Kommutativität)."""
    def key(x):
        v = x.a if isinstance(x, Not) else x
        return (v.name, isinstance(x, Not))

    def fn(g):
        if isinstance(g, (And, Or)):
            es = flatten(g, g.op)
            if len(es) > 1 and all(_is_lit(e) for e in es):
                srt = sorted(es, key=key)
                if srt != es:
                    return (big_and if isinstance(g, And) else big_or)(srt)
    # ponytail: Bottom-up-Variante ändert verschachtelte Ketten nur auf ihrer obersten Ebene
    return _pass(f, fn)


def _simplify_trace(f: Formula, max_steps: int = MAX_STEPS):
    """Rechenweg als Liste (Gesetz, Beschreibung, Term nach dem Schritt)."""
    trace: list = []
    cur = f

    def push(law, detail, new):
        nonlocal cur
        if not _same(cur, new):                                # pragma: no cover
            raise RuntimeError(f"Interner Fehler: Schritt «{law}» ist nicht äquivalent.")
        cur = new
        trace.append((law, detail, new))

    for law, detail, fn in _PREP_PASSES:
        new = _pass(cur, fn)
        if new is not None:
            push(law, detail, new)
    while len(trace) < max_steps:
        for law, detail, fn in _NEG_PASSES:
            new = _pass(cur, fn)
            if new is not None:
                push(law, detail, new)
                break
        else:
            break
    while len(trace) < max_steps:
        found = _find_chain(cur)
        if not found:
            break
        path, steps = found
        where = _path_text(cur, path)
        for law, detail, new_sub in steps:
            push(law, detail + f" (angewandt {where})",
                 _replace(cur, path, new_sub))
            if len(trace) >= max_steps:
                break
    new = _sort_literal_chains(cur)
    if new is not None:
        push("Kommutativität", "Variablen alphabetisch ordnen.", new)
    return trace


def _best_prefix(f: Formula, trace: list) -> list:
    """Kürzester Term entlang des Wegs (bei Gleichstand der spätere); Rechenweg bis dort."""
    best_i, best_size = 0, f.size()
    for i, (_, _, g) in enumerate(trace, 1):
        if g.size() <= best_size:
            best_i, best_size = i, g.size()
    return trace[:best_i]


def simplify(f) -> SimplifyResult:
    """
    Vereinfacht einen Term mit den Rechengesetzen, Schritt für Schritt mit Gesetzesnamen.

        r = simplify("(~(~P | ~Q) & (P -> Q)) -> P")
        r.is_tautology                  # True  (Beispiel der Vorlesung)
        print(r)                        # Implikation, De Morgan, Absorption, Invers, Dominanz …

    Vorgehen: (1) ⇔, ⊕, ⇒ durch ¬, ∧, ∨ ersetzen, (2) Negation nach innen (De Morgan,
    Doppelnegation), (3) Ketten aus ∧ bzw. ∨ mit Dominanz, Identität, Idempotenz, Invers,
    Absorption und Ausklammern verkürzen. Jeder Schritt wird per Wahrheitstabelle als äquivalent
    geprüft; das Ergebnis ist nie länger als der Ausgangsterm (sonst endet der Weg früher).
    """
    f = parse(f)
    _check_vars(f.variables())
    full = _simplify_trace(f)
    return SimplifyResult(f, _best_prefix(f, full), full)


def apply_laws(f, names=None, max_steps: int = 50) -> SimplifyResult:
    """
    Wendet die angegebenen Gesetze von links nach rechts an, bis nichts mehr passt.

        apply_laws("~(A & B) | ~~C", ["De Morgan", "Doppelnegation"])

    Ohne ``names``: Doppelnegation, Implikation, Äquivalenz, Kontravalenz, De Morgan, Dominanz,
    Identität, Invers, Idempotenz, Absorption. Gesetze, die in Endlosschleifen führen könnten
    (Kommutativität, Assoziativität, Distributivität, Kontraposition), gehen nur mit ``apply_law``.
    """
    f = parse(f)
    _check_vars(f.variables())
    if not isinstance(max_steps, int) or not 1 <= max_steps <= MAX_STEPS:
        raise ValueError(f"max_steps muss zwischen 1 und {MAX_STEPS} liegen.")
    selected: list[Law] = []
    for name in (names if names is not None else _DEFAULT_LAWS):
        for law in find_law(name):
            if not law.reducing:
                raise ValueError(f"Das Gesetz «{law.name}» kann in eine Endlosschleife führen und "
                                 "lässt sich nur einzeln mit apply_law() anwenden.")
            selected.append(law)
    trace, cur = [], f
    while len(trace) < max_steps:
        for law in selected:
            hit = next(iter(_rewrite_candidates(cur, law, False)), None)
            if hit:
                path, _, new = hit
                after = _replace(cur, path, new)
                if not _same(cur, after):                      # pragma: no cover
                    raise RuntimeError(f"Interner Fehler: {law.name} nicht äquivalent.")
                trace.append((law.name, f"{law.text()}, angewandt {_path_text(cur, path)}.", after))
                cur = after
                break
        else:
            break
    return SimplifyResult(f, trace)


class DerivationCheckResult(Result):
    """
    Prüfung einer eigenen Umformungskette f0 ≡ f1 ≡ f2 …

    ``ok`` (alle Schritte äquivalent), ``checks`` = Liste von (Index, äquivalent, Gesetzesname oder "").
    """

    title = "Umformungskette prüfen"

    def __init__(self, formulas: list[Formula]):
        super().__init__()
        self.formulas, self.checks = formulas, []
        self.steps.append(Step("Ausgangsterm", items=[("", _fs(formulas[0]))]))
        for i in range(1, len(formulas)):
            a, b = formulas[i - 1], formulas[i]
            eq = _same(a, b)
            law = _identify_law(a, b) if eq else ""
            self.checks.append((i, eq, law))
            note = ("" if eq else "Dieser Schritt ist nicht äquivalent! Gegenbeispiel: "
                    + _env_text(_first_difference(a, b), _vars_of(a, b)))
            self.steps.append(Step(f"Schritt {i}", "äquivalent" if eq else "NICHT äquivalent",
                                   items=[("", _fs(b))], rule=law, note=note))
        self.ok = all(c[1] for c in self.checks)
        self.first_error = next((c[0] for c in self.checks if not c[1]), None)
        self.summary = ("Alle Schritte sind äquivalent." if self.ok else
                        f"Schritt {self.first_error} ist nicht äquivalent.")
        self.highlights = [("alle Schritte korrekt", self.ok)]


def _first_difference(a: Formula, b: Formula) -> dict:
    return next(e for e in assignments(_vars_of(a, b)) if a.eval(e) != b.eval(e))


def _identify_law(a: Formula, b: Formula) -> str:
    """Welches Gesetz erklärt a -> b in einem Schritt? (Leerer Text, wenn keines passt.)"""
    for law in _LAWS:
        for rev in (False, True):
            try:
                for path, _, new in _rewrite_candidates(a, law, rev):
                    if _replace(a, path, new) == b:
                        return law.name
            except ValueError:
                continue
    return ""


def check_derivation(*formulas) -> DerivationCheckResult:
    """
    Prüft eine selbst gerechnete Umformungskette und benennt, wo möglich, das Gesetz.

        check_derivation("A -> B", "~A | B", "~(A & ~B)")
    """
    items = list(formulas[0]) if len(formulas) == 1 and isinstance(formulas[0], (list, tuple)) else list(formulas)
    if len(items) < 2:
        raise ValueError("Bitte mindestens zwei Terme angeben.")
    if len(items) > 100:
        raise ValueError("Zu viele Schritte (höchstens 100).")
    return DerivationCheckResult([parse(x) for x in items])


def verify_law(name: str, variant: str | None = None):
    """
    Bestätigt ein Gesetz Schritt für Schritt per Wahrheitstabelle (Serie 2, Aufgabe 6).

        verify_law("De Morgan")      # -> Liste von EquivalenceResult (eine je Variante)
        verify_law("Absorption", "∨")[0].equivalent
    """
    group = find_law(name)
    if variant is not None:
        group = [g for g in group if g.variant == variant]
        if not group:
            raise ValueError(f"Das Gesetz {name!r} hat keine Variante {variant!r}.")
    out = []
    for g in group:
        r = EquivalenceResult(g.lhs, g.rhs)
        r.title = f"Gesetz bestätigen: {g.name}{' (' + g.variant + ')' if g.variant else ''}"
        out.append(r)
    return out


# ==========================================================================
# Kanonische Normalformen aus der Wahrheitstabelle
# ==========================================================================

class NormalFormResult(Result):
    """
    Kanonische DNF bzw. KNF aus der Wahrheitstabelle.

    Attribute: ``kind`` ('DNF' | 'KNF'), ``formula`` (Ausgangsterm), ``normal_form`` (die Normalform),
    ``text`` (Normalform mit Klammern als Text), ``terms`` (Minterme bzw. Maxterme als Formeln), ``literals`` (je Term Liste von (Name, bool),
    bool = positiv), ``row_indices`` (gewählte Zeilen, 0-basiert), ``table``, ``equivalent`` (Probe).
    """

    title = "Kanonische Normalform"

    def __init__(self, f: Formula, kind: str, true_first: bool = False):
        super().__init__()
        names = f.variables()
        _check_vars(names)
        self.formula, self.kind, self.variables = f, kind, names
        dnf = kind == "DNF"
        rows = assignments(names, true_first)
        vals = [f.eval(e) for e in rows]
        pick = [i for i, v in enumerate(vals) if v == dnf]       # DNF: Zeilen mit 1, KNF: Zeilen mit 0
        self.row_indices = pick
        self.literals, self.terms = [], []
        for i in pick:
            env = rows[i]
            if dnf:     # Minterm: Variable, wenn 1; Negation, wenn 0
                lits = [(n, env[n]) for n in names]
            else:       # Maxterm: Variable, wenn 0; Negation, wenn 1
                lits = [(n, not env[n]) for n in names]
            self.literals.append(lits)
            parts = [Var(n) if pos else Not(Var(n)) for n, pos in lits]
            self.terms.append((big_and if dnf else big_or)(parts))
        self.normal_form = (big_or if dnf else big_and)(self.terms)
        self.equivalent = _same(f, self.normal_form)
        marks = {i: ("ok" if dnf else "bad") for i in pick}
        header = names + [_col(f)] + (["Minterm"] if dnf else ["Maxterm"])
        trows = []
        for i, env in enumerate(rows):
            term = _fs(self.terms[pick.index(i)]) if i in pick else ""
            trows.append([bit(env[n]) for n in names] + [bit(vals[i]), term])
        self.table = Table(header, trows, marks,
                           "Gewählte Zeilen: " + ("f = 1" if dnf else "f = 0"))
        long = "disjunktive" if dnf else "konjunktive"
        self.title = f"Kanonische {kind} von {_fs(f)}"
        self.text = _pretty(self.normal_form)
        self.summary = self.text
        self.steps = [
            Step("Wahrheitstabelle", "Wert des Terms in jeder Zeile.", table=self.table,
                 rule="Wahrheitstabelle"),
            Step("Zeilen auswählen",
                 ("Bei der DNF zählen die Zeilen, in denen der Term wahr (1) ist."
                  if dnf else "Bei der KNF zählen die Zeilen, in denen der Term falsch (0) ist."),
                 items=[("Zeilen", ", ".join(str(i + 1) for i in pick) or "keine")]),
            Step("Minterme bilden" if dnf else "Maxterme bilden",
                 ("Je Zeile eine Konjunktion aller Variablen: Variable, wenn ihr Wert 1 ist, "
                  "sonst ihre Negation." if dnf else
                  "Je Zeile eine Disjunktion aller Variablen: Variable, wenn ihr Wert 0 ist, "
                  "sonst ihre Negation."),
                 items=[(f"Zeile {i + 1}", _pretty(t)) for i, t in zip(pick, self.terms)]),
            Step(f"{long} Normalform zusammensetzen",
                 ("Alle Minterme durch ∨ verbinden." if dnf else "Alle Maxterme durch ∧ verbinden."),
                 items=[(kind, self.text)], rule=f"kanonische {kind}"),
            Step("Probe", "Normalform und Ausgangsterm haben denselben Wahrheitsvektor.",
                 items=[("äquivalent", self.equivalent)]),
        ]
        if not pick:
            self.steps[1].note = ("Keine Zeile: " + ("der Term ist unerfüllbar, die DNF ist F."
                                                     if dnf else "der Term ist allgemeingültig, die KNF ist T."))
        self.highlights = [(kind, self.text)]


def canonical_dnf(f, true_first: bool = False) -> NormalFormResult:
    """
    Kanonische disjunktive Normalform: Disjunktion der Minterme der Zeilen mit Wert 1.

        canonical_dnf("A xor B").normal_form      # (¬A ∧ B) ∨ (A ∧ ¬B)
    """
    return NormalFormResult(parse(f), "DNF", true_first)


def canonical_knf(f, true_first: bool = False) -> NormalFormResult:
    """
    Kanonische konjunktive Normalform: Konjunktion der Maxterme der Zeilen mit Wert 0.

        canonical_knf("A xor B").normal_form      # (A ∨ B) ∧ (¬A ∨ ¬B)

    Das ist die Normalform aus der Wahrheitstabelle; die symbolische KNF-Umformung folgt in SW4.
    """
    return NormalFormResult(parse(f), "KNF", true_first)


# ==========================================================================
# Syntaxbäume (Formeln und arithmetische Terme)
# ==========================================================================

_ARITH_LABEL = {"+": "+", "-": "−", "*": "·", "/": "/", "//": "//", "%": "%", "**": "**"}
_BINOPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//",
           ast.Mod: "%", ast.Pow: "**"}


def parse_term(text: str) -> Node:
    """
    Arithmetischen Term einlesen (nur + − * / // % **, Klammern, Zahlen, Namen, unäres Minus).

        parse_term("2 * x + y")      # Node('+', (Node('·', ...), Node('y')))

    Es wird nie ``eval`` benutzt; alles andere gibt einen ``ValueError``.
    """
    if not isinstance(text, str):
        raise ValueError("Ein Term muss als Text angegeben werden, z. B. '2 * x + y'.")
    src = text.strip()
    if not src:
        raise ValueError("Leerer Term.")
    if len(src) > MAX_TERM_LEN:
        raise ValueError(f"Term zu lang (mehr als {MAX_TERM_LEN} Zeichen).")
    if "^" in src:
        raise ValueError("Das Zeichen ^ ist hier nicht erlaubt; Potenzen schreibt man mit **.")
    try:
        tree = ast.parse(src, mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        raise ValueError(f"Der Term {text!r} ist syntaktisch nicht korrekt.") from None
    return _conv(tree.body, 0, text)


def _conv(n, depth: int, text: str) -> Node:
    if depth > MAX_TERM_DEPTH:
        raise ValueError("Term zu tief verschachtelt.")
    if isinstance(n, ast.Constant):
        v = n.value
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"Nicht erlaubte Konstante {v!r} im Term {text!r}.")
        if isinstance(v, float):
            if v != v or v in (float("inf"), float("-inf")):
                raise ValueError("Nicht erlaubte Zahl im Term.")
            # ponytail: Dezimalzahlen als exakte Brüche aus ihrer Textform statt als Float
            val = Fraction(repr(v))
            label = repr(v)
        else:
            if abs(v) > 10 ** 30:
                raise ValueError("Zahl zu gross (höchstens 30 Stellen).")
            val, label = v, str(v)
        return Node(label, (), "num", val)
    if isinstance(n, ast.Name):
        return Node(n.id, (), "name", n.id)
    if isinstance(n, ast.UnaryOp):
        if isinstance(n.op, ast.USub):
            return Node("−", (_conv(n.operand, depth + 1, text),), "neg")
        if isinstance(n.op, ast.UAdd):
            return _conv(n.operand, depth + 1, text)
        raise ValueError(f"Nicht erlaubter Operator im Term {text!r} (nur +, −, *, /, //, %, **).")
    if isinstance(n, ast.BinOp):
        sym = _BINOPS.get(type(n.op))
        if sym is None:
            raise ValueError(f"Nicht erlaubter Operator im Term {text!r} (nur +, −, *, /, //, %, **).")
        return Node(_ARITH_LABEL[sym], (_conv(n.left, depth + 1, text),
                                        _conv(n.right, depth + 1, text)), sym)
    raise ValueError(f"Der Term {text!r} enthält etwas, das kein Zahlenterm ist "
                     f"({type(n).__name__}); erlaubt sind Zahlen, Namen, + − * / // % ** und Klammern.")


def _num_out(v):
    if isinstance(v, Fraction) and v.denominator == 1:
        return int(v)
    return v


def _eval_term(n: Node, env: dict):
    k = n.op
    if k == "num":
        return n.value
    if k == "name":
        if n.value not in env:
            raise ValueError(f"Belegung fehlt für {n.value}")
        v = env[n.value]
        if isinstance(v, bool) or not isinstance(v, (int, float, Fraction)):
            raise ValueError(f"Der Wert von {n.value} muss eine Zahl sein.")
        return Fraction(v) if isinstance(v, float) else v
    kids = [_eval_term(c, env) for c in n.children]
    if k == "neg":
        return -kids[0]
    a, b = kids
    if k == "+":
        return a + b
    if k == "-":
        return a - b
    if k == "*":
        return a * b
    if k in ("/", "//", "%"):
        if b == 0:
            raise ValueError("Division durch 0.")
        return a / b if k == "/" else (a // b if k == "//" else a % b)
    if k == "**":
        if isinstance(b, Fraction) and b.denominator != 1:
            raise ValueError("Der Exponent muss eine ganze Zahl sein.")
        e = int(b)
        if abs(e) > 1000 or (abs(a) > 1 and abs(e) * (a.bit_length() if isinstance(a, int)
                                                         else 64) > 20000):
            raise ValueError("Potenz zu gross.")
        if a == 0 and e < 0:
            raise ValueError("Division durch 0.")
        return Fraction(a) ** e if e < 0 else a ** e
    raise ValueError(f"Unbekannter Operator {k!r}.")      # pragma: no cover


# ponytail: Baumzeichnung als eingerückter Text (├──); die gezeichnete Grafik liefert grafik/plots.py
def _tree_lines(n, prefix: str = "", last: bool = True, root: bool = True) -> list[str]:
    lab = n.label
    if root:
        out = [lab]
        child_prefix = ""
    else:
        out = [prefix + ("└── " if last else "├── ") + lab]
        child_prefix = prefix + ("    " if last else "│   ")
    kids = _kids(n)
    for i, c in enumerate(kids):
        out += _tree_lines(c, child_prefix, i == len(kids) - 1, False)
    return out


def _tree_nodes(root) -> list[dict]:
    """Knoten in Vorordnung: id, label, depth (Wurzel 0), parent, kind, subterm, n_children."""
    out: list[dict] = []

    def visit(n, depth, parent):
        i = len(out)
        out.append({"id": i, "label": n.label, "depth": depth, "parent": parent,
                    "kind": "", "subtree": _infix(n), "children": [], "size": n.size(),
                    "node": n})
        for c in _kids(n):
            out[i]["children"].append(visit(c, depth + 1, i))
        out[i]["kind"] = ("Wurzel und Blatt" if depth == 0 and not _kids(n) else
                          "Wurzel" if depth == 0 else "Blatt" if not _kids(n) else "innerer Knoten")
        return i
    visit(root, 0, None)
    return out


class SyntaxTreeResult(Result):
    """
    Syntaxbaum eines Terms (Formel oder arithmetischer Term), von oben nach unten.

    Attribute: ``root`` (Node bzw. Formula), ``kind`` ('formula' | 'term'), ``text`` (eingerückte
    Baumzeichnung), ``nodes`` (Liste von dict mit id, label, depth, parent, kind, subtree, children,
    size), ``levels`` (Beschriftungen je Ebene), ``leaves``, ``subtrees`` (Teilterm je Knoten als Text),
    ``depth`` (Anzahl Ebenen), ``height`` (längster Pfad in Kanten), ``size`` (Anzahl Knoten),
    ``signature`` (Variablen/Namen), ``positions`` (id -> (x, Ebene) für Zeichnungen).
    ``evaluate(**belegung)`` wertet den Term aus (Wahrheitswerte bzw. Zahlen).
    """

    title = "Syntaxbaum"

    def __init__(self, root, kind: str, source: str):
        super().__init__()
        self.root, self.kind, self.source = root, kind, source
        self.nodes = _tree_nodes(root)
        self.size = len(self.nodes)
        self.depth = root.depth()
        self.height = self.depth - 1
        self.text = "\n".join(_tree_lines(root))
        self.levels = [[d["label"] for d in self.nodes if d["depth"] == k]
                       for k in range(self.depth)]
        self.leaves = [d["label"] for d in self.nodes if not d["children"]]
        self.subtrees = [d["subtree"] for d in self.nodes]
        if kind == "formula":
            self.signature = root.variables()
        else:
            self.signature = sorted({d["label"] for d in self.nodes if d["node"].op == "name"})
        self.positions = self._layout()          # ponytail: einfache Mittenzentrierung, kein Reingold-Tilford
        shown = _infix(root)
        self.title = f"Syntaxbaum von {shown}"
        what = "Operator" if kind == "formula" else "Operator"
        self.summary = (f"{self.size} Knoten, {len([d for d in self.nodes if not d['children']])} "
                        f"Blätter, {self.depth} Ebenen (längster Pfad: {self.height} Kanten).")
        self.steps = [
            Step("Wurzel bestimmen",
                 f"Von oben nach unten: Die Wurzel ist der {what}, der zuletzt angewendet wird.",
                 items=[("Wurzel", root.label)], rule="Wurzel"),
        ]
        for k in range(1, self.depth):
            names = [d for d in self.nodes if d["depth"] == k]
            self.steps.append(Step(
                f"Ebene {k}", "Teilbäume der Knoten darüber; Blätter sind Variablen bzw. Zahlen.",
                items=[(f"Knoten {d['label']}" + (" (Blatt)" if not d["children"] else ""),
                        d["subtree"]) for d in names]))
        self.steps.append(Step("Baum", "Eingerückte Darstellung (Wurzel oben, Teilbäume darunter).",
                               items=[("", "\n" + self.text)]))
        self.highlights = [("Wurzel", root.label), ("Blätter", ", ".join(self.leaves)),
                           ("Ebenen", self.depth), ("Knoten", self.size)]

    def _layout(self) -> dict:
        """Positionen für eine Zeichnung: Blätter von links nach rechts, Eltern mittig darüber."""
        pos: dict = {}
        counter = [0]

        def place(i: int):
            kids = self.nodes[i]["children"]
            if not kids:
                x = float(counter[0])
                counter[0] += 1
            else:
                xs = [place(c) for c in kids]
                x = (xs[0] + xs[-1]) / 2
            pos[i] = (x, self.nodes[i]["depth"])
            return x
        place(0)
        return pos

    def evaluate(self, **env):
        """Wertet den Term aus: Formel -> bool, arithmetischer Term -> int bzw. Fraction."""
        if self.kind == "formula":
            return bool(self.root.eval(env))
        return _num_out(_eval_term(self.root, env))

    def subtree(self, node_id: int):
        """Teilbaum ab Knoten ``node_id`` (Vorordnungs-Nummer) als Node bzw. Formula."""
        if not isinstance(node_id, int) or not 0 <= node_id < len(self.nodes):
            raise ValueError(f"Knotennummer muss zwischen 0 und {len(self.nodes) - 1} liegen.")
        return self.nodes[node_id]["node"]

    def __str__(self):
        return self.text


# ponytail: Auto-Erkennung formula/term über Junktor-Zeichen und -Wörter statt über eine Grammatik
_LOGIC_WORDS = ("∧", "∨", "¬", "⊕", "→", "↔", "⇒", "⇔", "&", "|", "~", "!", "^", "->", "=>",
                "<->", "⊤", "⊥")


def _looks_logical(s: str) -> bool:
    import re
    if any(w in s for w in _LOGIC_WORDS):
        return True
    return bool(re.search(r"\b(and|or|not|xor|und|oder|nicht|iff|implies)\b", s, re.I))


def syntax_tree(expr, kind: str = "auto") -> SyntaxTreeResult:
    """
    Syntaxbaum einer Formel oder eines arithmetischen Terms (Serie 2, Aufgabe 1).

        syntax_tree("((A -> B) & ~B) <-> ~A")        # Formel
        syntax_tree("(P | Q) -> R <-> (P -> R) & (Q -> R)")
        syntax_tree("2 * x + y").evaluate(x=3, y=4)  # 10

    ``kind``: 'auto' (erkennt Junktoren wie ∧ ∨ ¬ -> and or not), 'formula' oder 'term'.
    """
    if kind not in ("auto", "formula", "term"):
        raise ValueError("kind muss 'auto', 'formula' oder 'term' sein.")
    if isinstance(expr, Formula) or isinstance(expr, bool):
        f = parse(expr)
        _check_vars(f.variables())
        return SyntaxTreeResult(f, "formula", _fs(f))
    if isinstance(expr, Node):
        return SyntaxTreeResult(expr, "term", _infix(expr))
    if not isinstance(expr, str):
        raise ValueError("Bitte einen Term als Text oder eine Formel angeben.")
    if kind == "formula" or (kind == "auto" and _looks_logical(expr)):
        f = parse(expr)
        _check_vars(f.variables())
        return SyntaxTreeResult(f, "formula", expr)
    if kind == "term":
        return SyntaxTreeResult(parse_term(expr), "term", expr)
    try:
        return SyntaxTreeResult(parse_term(expr), "term", expr)
    except ValueError as err:
        try:
            f = parse(expr)
        except ValueError:
            raise err from None
        _check_vars(f.variables())
        return SyntaxTreeResult(f, "formula", expr)


# ==========================================================================
# Deutsche Namen (so heissen sie in der Vorlesung)
# ==========================================================================

wahrheitstabelle = truth_table
aequivalenz_pruefen = check_equivalence
vereinfachen = simplify
syntaxbaum = syntax_tree
