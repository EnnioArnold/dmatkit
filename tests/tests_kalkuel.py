#!/usr/bin/env python3
"""Tests für dmatkit.rechnen.kalkuel (Semesterwoche 4): python3 tests/tests_kalkuel.py"""
import os, sys, random, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from dmatkit.rechnen import core
from dmatkit.rechnen import kalkuel as K
from dmatkit.rechnen.core import parse, assignments, Var, Not, And, Or, Xor, Implies, Iff, Const

N = [0]


def ok(cond, msg=""):
    N[0] += 1
    assert cond, msg


def raises(exc, fn, *a, **kw):
    N[0] += 1
    try:
        fn(*a, **kw)
    except exc as e:
        return str(e)
    except Exception as e:                      # falscher Typ
        raise AssertionError(f"{fn.__name__}: erwartet {exc.__name__}, bekommen {type(e).__name__}: {e}")
    raise AssertionError(f"{fn.__name__}: erwartet {exc.__name__}, aber keine Ausnahme")


def brute_entails(wb, q):
    wb, q = parse(wb), parse(q)
    names = sorted(set(wb.variables()) | set(q.variables()))
    return all(q.eval(e) for e in assignments(names) if wb.eval(e))


def show_all(obj):
    repr(obj); obj._repr_html_(); obj.latex()


# ==========================================================================
# 1. Semantische Folgerung
# ==========================================================================
WB1 = "(A or B) and (C or (not B and not A) or B)"
r = K.entails(WB1, "A or B")
ok(r.entailed and r.value and bool(r) and r.consistent)
ok(r.verdicts == {"table": True, "deduction": True, "contradiction": True})
ok(r.variables == ["A", "B", "C"] and r.counterexample is None and r.counterexamples == [])
# Tabelle der Vorlesung (Serie 4, Seite 2): Spalten WB und Q in der Reihenfolge der PDF-Tabelle (1 1 1 zuerst)
pdf_rows = {(1, 1, 1): (1, 1), (1, 1, 0): (1, 1), (1, 0, 1): (1, 1), (1, 0, 0): (0, 1),
            (0, 1, 1): (1, 1), (0, 1, 0): (1, 1), (0, 0, 1): (0, 0), (0, 0, 0): (0, 0)}
t = r.tables["table"]
ok(t.header == ["A", "B", "C", "WB", "Q"] and len(t.rows) == 8)
for row in t.rows:
    a, b, c, w, q_ = (int(x) for x in row)
    ok(pdf_rows[(a, b, c)] == (w, q_), row)
ok(t.rows[0] == ["0", "0", "0", "0", "0"] and t.rows[-1] == ["1", "1", "1", "1", "1"])   # Tabellenreihenfolge 0 0 0 zuerst
ok({i for i, m in t.marks.items() if m == "hi"} == {i for i, row in enumerate(t.rows) if row[3] == "1"})
ok(not any(m == "bad" for m in t.marks.values()))
ok(r.tables["deduction"].header[-1] == "WB ⇒ Q" and all(row[-1] == "1" for row in r.tables["deduction"].rows))
ok(r.tables["contradiction"].header[-2:] == ["¬Q", "WB ∧ ¬Q"] and all(row[-1] == "0" for row in r.tables["contradiction"].rows))
ok(r.table is r.tables["all"] and r.table.header[-5:] == ["WB", "Q", "WB ⇒ Q", "¬Q", "WB ∧ ¬Q"])
ok(any("Zusammenfassung" in s.title for s in r.steps))
ok(sum(1 for s in r.steps if s.table is not None) == 4)
show_all(r)

# Aufgabe 1 / 2 / 3
ok(core.is_tautology(f"({WB1}) -> (A or B)"))
ok(core.is_contradiction(f"({WB1}) and not (A or B)"))
ok(K.entails("(A or B) and (B -> C)", "A or C").entailed)

# Gegenbeispiel
r = K.entails("A or B", "A")
ok(not r.entailed and not bool(r) and r.counterexample == {"A": False, "B": True})
ok(r.verdicts == {"table": False, "deduction": False, "contradiction": False} and r.consistent)
ok(set(r.tables["table"].marks.values()) == {"bad", "hi"})
bad_rows = [i for i, m in r.tables["table"].marks.items() if m == "bad"]
ok(len(bad_rows) == 1 and r.tables["table"].rows[bad_rows[0]][:4] == ["0", "1", "1", "0"])
ok("Gegenbeispiel" in repr(r) and "A=0, B=1" in repr(r))
show_all(r)

# einzelne Methoden, Liste als WB, Aliasnamen
for m in ("table", "pfeil", "deduction", "deduktion", "contradiction", "widerspruch", "Pfeilmethode"):
    rr = K.entails(["A", "A -> B"], "B", method=m)
    ok(rr.entailed and len(rr.verdicts) == 1)
rr = K.entails(["A", "A -> B"], "B")
ok(rr.entailed and rr.tables["table"].header == ["A", "B", "WB₁", "WB₂", "WB", "Q"])
ok(rr.wb == parse("A and (A -> B)") and len(rr.wb_parts) == 2)
ok(K.entails(["A", "A -> B"], ["B", "A or C"]).entailed)
ok(not K.entails(["A", "B -> C"], "C").entailed)
ok(K.entails([], "A or not A").entailed and not K.entails([], "A").entailed)
ok(K.entails("A and not A", "B").entailed)                       # aus Widerspruch folgt alles
ok(K.entails("A", True).entailed and not K.entails("A", False).entailed)
ok(K.entails(False, "A").entailed)
raises(ValueError, K.entails, "A", "B", method="quatsch")
raises(ValueError, K.entails, "A and", "B")
raises(ValueError, K.entails, " ".join(f"x{i} or" for i in range(13)) + " y", "z")     # 14 Variablen
# Beispiele der Lerngerüst-Aufgabe (a)
ok(K.entails("U and W", "U or W").entailed)
ok(K.entails("(not H or S) and H", "S and H").entailed)
ok(K.entails("(A or B or C) -> (P and Q)", "(A -> P) and (C -> Q)").entailed)
ok(not K.entails("(A -> P) and (C -> Q)", "(A or B or C) -> (P and Q)").entailed)

# ==========================================================================
# 2. Konjunktive Normalform
# ==========================================================================
c = K.to_cnf("(A or B) -> (C and D)")                           # Aufgabe 4
ok(core.equivalent(c.formula, "(A or B) -> (C and D)"))
ok(c.clauses == {frozenset({"¬A", "C"}), frozenset({"¬A", "D"}), frozenset({"¬B", "C"}), frozenset({"¬B", "D"})})
ok(K.is_cnf(c.formula) and len(c.clause_list) == 4 and c.verified is True and not c.already_cnf)
ok([s.title for s in c.steps if s.rule] [:2] == ["Implikation auflösen", "De Morgan"])
ok(any(s.title == "Distributivgesetz" for s in c.steps))
ok(K.clause_set_str(c.clause_list) == "{{¬A, C}, {¬A, D}, {¬B, C}, {¬B, D}}")
show_all(c)

c = K.to_cnf("(A or B or C) -> (P and Q)")                      # Lerngerüst (b)
ok(len(c.clauses) == 6 and c.clauses == {frozenset({"¬" + x, y}) for x in "ABC" for y in "PQ"})
ok(core.equivalent(c.formula, "(A or B or C) -> (P and Q)"))

c = K.to_cnf("(S or N) and not S and (P or N) and (not N or P)")
ok(c.already_cnf and len(c.clause_list) == 4 and c.clause_list[1] == frozenset({"¬S"}))
c = K.to_cnf("not (not S and N and P)")                          # ¬Q von Aufgabe 5
ok(c.clauses == {frozenset({"S", "¬N", "¬P"})} and c.formula == parse("S or not N or not P"))
ok([s.title for s in c.steps[1:-1]] == ["De Morgan", "Doppelte Negation"])
c = K.to_cnf("A <-> B")
ok(c.clauses == {frozenset({"¬A", "B"}), frozenset({"A", "¬B"})} and c.steps[1].title == "Äquivalenz auflösen")
c = K.to_cnf("A xor B")
ok(c.clauses == {frozenset({"A", "B"}), frozenset({"¬A", "¬B"})})
c = K.to_cnf("A or not A")
ok(c.is_tautology and c.clauses == frozenset() and c.formula == core.TRUE and K.clause_set_str(c.clause_list) == "{}")
c = K.to_cnf("A and not A")
ok(c.clauses == {frozenset({"A"}), frozenset({"¬A"})} and not c.is_contradiction)
c = K.to_cnf(False)
ok(c.is_contradiction and c.clauses == {frozenset()} and c.formula == core.FALSE and K.clause_str(frozenset()) == "□")
c = K.to_cnf(True)
ok(c.is_tautology and c.clauses == frozenset())
c = K.to_cnf("A and (B or True) and (C or False)")
ok(c.clauses == {frozenset({"A"}), frozenset({"C"})} and any(s.title == "Konstanten vereinfachen" for s in c.steps))
c = K.to_cnf("A")
ok(c.already_cnf and c.clauses == {frozenset({"A"})})
c = K.to_cnf("A and (A or B)")
ok(len(c.clauses) == 2)
c = K.to_cnf("A and (A or B)", subsumption=True)
ok(c.clauses == {frozenset({"A"})} and any(s.title == "Absorption" for s in c.steps))
c = K.to_cnf("(A or A or B) and (A or B) and (C or not C)")
ok(c.clauses == {frozenset({"A", "B"})})
ok(c.verified)
# viele Variablen: ohne Wahrheitstabellen-Prüfung, mit Warnung
many = " and ".join(f"(x{i} or y{i})" for i in range(8))
c = K.to_cnf(many)
ok(c.verified is None and c.warnings and len(c.clauses) == 8)
# Blow-up-Schutz
big = " or ".join(f"(a{i} and b{i})" for i in range(14))
msg = raises(ValueError, K.to_cnf, big)
ok("exponentiell" in msg and "max_size" in msg, msg)
raises(ValueError, K.to_cnf, "A", max_size=3)
ok(len(K.to_cnf(" or ".join(f"(a{i} and b{i})" for i in range(3))).clauses) == 8)
raises(ValueError, K.to_cnf, "A and or B")
# Klausel-Hilfen
ok(K.lit_neg("A") == "¬A" and K.lit_neg("¬A") == "A" and K.lit_var("¬B") == "B")
ok(K.clause_str({"A", "¬B"}) == "{A, ¬B}" and K.clause_str({"¬B", "A", "C"}) == "{A, ¬B, C}")
ok(K.is_literal("A") and K.is_literal("not A") and not K.is_literal("not not A") and not K.is_literal("A or B"))
ok(K.is_clause("A or not B or C") and not K.is_clause("A and B") and K.is_clause("A"))
ok(K.is_cnf("(A or B) and C") and not K.is_cnf("A or (B and C)") and K.is_cnf("A"))
ok(K.clauses_of("(S or N) and not S") == [frozenset({"S", "N"}), frozenset({"¬S"})])
raises(ValueError, K.clauses_of, "A -> B")
ok(K.clause_formula({"A", "¬B"}) == parse("A or not B") and K.clause_formula(set()) == core.FALSE)
ok(K.make_clause(["A", "~B"]) == frozenset({"A", "¬B"}))
raises(ValueError, K.make_clause, [""])

# ==========================================================================
# 3. Resolution
# ==========================================================================
WB5 = "(S or N) and (not S) and (P or N) and (not N or P)"
Q5 = "not S and N and P"
r = K.resolution(WB5, Q5)                                       # Aufgabe 5
ok(r.derived_empty and r.entailed is True and r.value is True and r.check is True and r.agrees)
ok(r.clauses[0] == {"S", "N"} and r.clauses[1] == {"¬S"} and r.clauses[2] == {"P", "N"} and r.clauses[3] == {"¬N", "P"})
ok(r.clauses[4] == {"S", "¬N", "¬P"} and r.lines[4].origin == "Negierte Anfrage")
ok(r.clauses[5] == {"N"} and r.lines[5].parents == (1, 2))      # K6 aus K1 und K2 (Mitschrift)
ok(r.clauses[6] == {"P"} and r.lines[6].parents == (3, 4))      # K7 aus K3 und K4 (Mitschrift)
ok(not r.clauses[-1] and r.lines[-1].origin == "Resolvent")
ok(r.wb_inconsistent is False)
# Vertrag: Graph
for gl in (r.graph, r.full_graph):
    ok(isinstance(gl, list) and all(isinstance(d, dict) and set(d) == {"id", "clause", "parents", "origin"} for d in gl))
    ok(all(isinstance(d["id"], int) and isinstance(d["clause"], str) and isinstance(d["parents"], tuple) for d in gl))
    ok(all(d["origin"] in ("Wissensbasis", "Negierte Anfrage", "Resolvent") for d in gl))
    ids = {d["id"] for d in gl}
    ok(all(p in ids for d in gl for p in d["parents"]))
    ok(all((d["origin"] == "Resolvent") == (len(d["parents"]) == 2) for d in gl))
ok(r.graph[-1]["clause"] == "□" and r.full_graph[-1]["clause"] == "□")
ok({d["id"] for d in r.graph} == set(r.proof_ids) and len(r.graph) <= len(r.full_graph))
ok({"id": 6, "clause": "{N}", "parents": (1, 2), "origin": "Resolvent"} in r.full_graph)
ok({"id": 1, "clause": "{N, S}", "parents": (), "origin": "Wissensbasis"} in r.full_graph)
ok(sum(1 for d in r.graph if d["origin"] == "Negierte Anfrage") == 1)
ok(r.table.header == ["Nr.", "Klausel", "Herkunft"] and len(r.table.rows) == len(r.clauses))
ok(r.proof_table is not None and r.proof_table.rows[-1][1] == "□")
ok(any(s.rule == "Resolutionsregel" for s in r.steps) and any(s.title.startswith("De Morgan") for s in r.steps))
show_all(r)
r2 = K.resolution(WB5, Q5, detail=False)
ok(len(r2.steps) < len(r.steps) and r2.entailed)
ok(K.resolution([ "S or N", "not S", "P or N", "not N or P"], Q5).clauses[:4] == r.clauses[:4])   # WB als Liste

# Lerngerüst (c), (d)
r = K.resolution("(C -> B) and (A or C) and (A -> C)", "C and B")
ok(r.derived_empty and r.check)
r = K.resolution("not (A and B) and (C -> B) and ((not A and B) -> not C)", "not C")
ok(r.derived_empty and r.check)
# ohne die innere Klammer (so steht es im PDF-Text) gilt die Folgerung nicht
r = K.resolution("not (A and B) and (C -> B) and (not A and B) -> not C", "not C")
ok(not r.derived_empty and r.entailed is False and r.check is False)
ok(r.counterexample is not None and parse("not (A and B) and (C -> B) and (not A and B) -> not C").eval(r.counterexample)
   and not parse("not C").eval(r.counterexample))
ok(r.graph == r.full_graph and all(d["clause"] != "□" for d in r.graph))
ok("WB ⊭ Q" in repr(r) and "Gegenbeispiel" in repr(r))
show_all(r)
# Aufgabe 3 per Resolution
r = K.resolution("(A or B) and (B -> C)", "A or C")
ok(r.derived_empty and r.check)
# Nicht-Folgerung, Sättigung
r = K.resolution("A or B", "A")
ok(r.entailed is False and not r.derived_empty and r.counterexample == {"A": False, "B": True})
ok(len(r.clauses) == 3 and r.clauses[2] == {"B"} and r.lines[2].parents == (1, 2))
# Widerspruch in der WB
r = K.resolution("A and not A", "Z")
ok(r.derived_empty and r.wb_inconsistent and r.check is True)
# Spezialfälle
r = K.resolution("A", "A")
ok(r.derived_empty and len(r.clauses) == 3 and r.graph[-1]["parents"] == (1, 2))
r = K.resolution([], "A or not A")
ok(r.derived_empty and len(r.clauses) == 3 and r.graph[-1]["parents"] == (1, 2))     # ¬Q = ¬A ∧ A
r = K.resolution([], True)
ok(r.derived_empty and r.clauses[0] == frozenset() and r.graph[0]["clause"] == "□" and r.graph[0]["parents"] == ())
r = K.resolution([], "A")
ok(r.entailed is False and r.clauses == [frozenset({"¬A"})])
r = K.resolution("A", True)
ok(r.derived_empty)
r = K.resolution("A", False)
ok(r.entailed is False and r.check is False)
r = K.resolution(False, "X")
ok(r.derived_empty and r.wb_inconsistent)
# Limits
r = K.resolution("(A or B) and (not A or B) and (A or not B) and (not A or not B)", "C", max_clauses=5)
ok(r.limit_reached and r.entailed is None and r.warnings and r.check is True and r.agrees)
ok(K.resolution("(A or B) and (not A or B) and (A or not B) and (not A or not B)", "C").derived_empty)
r = K.resolution("(A or B) and (not A or B) and (A or not B) and (not A or not B)", "C", max_steps=1)
ok(r.limit_reached or r.derived_empty)
raises(ValueError, K.resolution, "A", "B", max_clauses=0)
raises(ValueError, K.resolution, "A", "B", max_steps=0)
raises(ValueError, K.resolution, "A or", "B")
# mehr als 12 Variablen: Resolution läuft, Gegenprobe entfällt
chain = " and ".join(f"(v{i} -> v{i+1})" for i in range(14)) + " and v0"
r = K.resolution(chain, "v14")
ok(r.derived_empty and r.check is None and r.agrees is None and r.warnings)
# Tautologische Resolventen werden verworfen, kein Doppelter
r = K.refute([{"A", "B"}, {"¬A", "¬B"}])
ok(not r.derived_empty and r.entailed is False and len(r.clauses) == 2)
# Einzelschritt
ok(K.resolve({"S", "N"}, {"¬S"}) == [("S", frozenset({"N"}))])
ok(K.resolve({"A", "B"}, {"¬B", "C"}) == [("B", frozenset({"A", "C"}))])
ok(K.resolve({"A", "B"}, {"¬A", "¬B"}) == [] and len(K.resolve({"A", "B"}, {"¬A", "¬B"}, drop_tautologies=False)) == 2)
ok(K.resolve({"A"}, {"¬A"}) == [("A", frozenset())] and K.resolve({"A"}, {"B"}) == [])
# refute
r = K.refute([{"A", "B"}, {"¬A"}, {"¬B"}])
ok(r.derived_empty and r.entailed is True and r.graph[-1]["clause"] == "□" and r.wb_inconsistent)
ok(all(d["origin"] in ("Wissensbasis", "Resolvent") for d in r.graph))
r = K.refute(["A or B", "not A or B"])
ok(not r.derived_empty and r.entailed is False)
raises(ValueError, K.refute, [])
raises(ValueError, K.refute, ["A and B"])
show_all(r)

# ==========================================================================
# 4. Syntaktische Ableitung
# ==========================================================================
d = K.derive("(A or B) and (B -> C)", "A or C")
ok(d.derived and bool(d) and d.entailed is True and d.proof[-1].formula == parse("A or C"))
ok([p.rule for p in d.proof] == ["Hypothese (H1)", "Simplifikation", "Simplifikation", "Resolution"])
ok(d.proof[-1].refs == (2, 3) and d.table.header == ["Nr.", "Formel", "Regel", "Verweise"])
show_all(d)
# Beispiel der Zusammenfassung: Hypothesen H1–H4, Ziel h
d = K.derive(["not n and k", "s -> n", "not s -> f", "f -> h"], "h")
ok(d.derived and d.proof[-1].formula == parse("h"))
ok(all(any(p.rule.startswith(n) for p in d.proof) for n in ("Simplifikation", "Modus tollens", "Modus ponens")))
ok(len(d.proof) == 8)
# Aufgabe 5 auch syntaktisch
d = K.derive(WB5, Q5)
ok(d.derived and d.proof[-1].formula == parse(Q5))
ok(any(p.rule == "Konjunktion" for p in d.proof) and any(p.rule == "Disjunktiver Syllogismus" for p in d.proof))
# Beweis lückenlos: jede Zeile folgt semantisch aus den Hypothesen
hyps = [p.formula for p in d.proof if p.rule.startswith("Hypothese")]
for p in d.proof:
    ok(brute_entails(core.big_and(hyps), p.formula), p)
    ok(all(1 <= x < p.nr for x in p.refs))
# einzelne Regeln
d = K.derive(["p", "p -> q"], "q", rules=["mp"])
ok(d.derived and d.proof[-1].rule == "Modus ponens" and d.proof[-1].refs == (1, 2))
d = K.derive(["p", "p -> q"], "q", rules=["Modus tollens"])
ok(not d.derived and d.saturated and d.entailed is True and d.warnings)
d = K.derive(["not q", "p -> q"], "not p", rules="mt")
ok(d.derived and d.proof[-1].rule == "Modus tollens")
d = K.derive(["p -> q", "q -> r"], "p -> r", rules=["hs"])
ok(d.derived and d.proof[-1].rule == "Hypothetischer Syllogismus")
d = K.derive(["p or q", "not p"], "q", rules=["ds"])
ok(d.derived and d.proof[-1].rule == "Disjunktiver Syllogismus")
d = K.derive("p", "p or q", rules=["add"])
ok(d.derived and d.proof[-1].rule == "Addition")
d = K.derive("p and q", "q", rules=["simp"])
ok(d.derived and d.proof[-1].rule == "Simplifikation")
d = K.derive(["p", "q"], "p and q", rules=["conj"])
ok(d.derived and d.proof[-1].rule == "Konjunktion")
d = K.derive(["p or q", "not p or r"], "q or r", rules=["res"])
ok(d.derived and d.proof[-1].rule == "Resolution")
d = K.derive("p", "p")
ok(d.derived and len(d.proof) == 1)
d = K.derive("A or B", "A")
ok(not d.derived and d.entailed is False and d.saturated)
d = K.derive("A and not A", "B")
ok(not d.derived and d.contradiction and d.entailed is True)     # F wird abgeleitet, Ex falso ist keine Regel
d = K.derive("not not A", "A", rules=["mp"])
ok(d.derived and d.proof[-1].rule == "Äquivalenzumformung" and d.proof[-1].formula == parse("A"))
d = K.derive("not not A", "A", rules=["mp"])
d = K.derive("A and B", "B and A")
ok(d.derived and d.proof[-1].formula == parse("B and A"))
d = K.derive(["A -> B", "not B"], "not A")
ok(d.derived)
d = K.derive(["not A -> B", "not B"], "A")
ok(d.derived)                                                  # ¬¬A wird als A gelesen
d = K.derive(["A", "B", "C"], "A and B and C")
ok(d.derived and d.proof[-1].rule == "Konjunktion")
d = K.derive("A", "A or B or C")
ok(d.derived and d.proof[-1].rule == "Addition")
d = K.derive(["a -> b", "b -> c", "c -> d"], "a -> d")
ok(d.derived)
d = K.derive(["a -> b", "b -> c", "c -> d"], "a -> d", max_steps=3)
ok(not d.derived and d.limit_reached and d.warnings)
raises(ValueError, K.derive, "A", "B", rules=["quatsch"])
raises(ValueError, K.derive, "A", "B", max_steps=0)
raises(ValueError, K.derive, "A", "B", max_steps=10 ** 6)
ok(set(K.RULES) == {"mp", "mt", "hs", "ds", "add", "simp", "conj", "res"})
show_all(d)
# Regel Resolution aus Serie 4: (A ∨ B) ∧ (B ⇒ C) ⊢ (A ∨ C)
ok(K.derive("(A or B) and (not B or C)", "A or C").derived)

# ==========================================================================
# 5. Nachweise mit den Rechengesetzen
# ==========================================================================
p = K.tautology_by_laws(f"({WB1}) -> (A or B)")                  # Aufgabe 1
ok(p.proved and bool(p) and p.cnf.is_tautology)
p = K.contradiction_by_laws(f"({WB1}) and not (A or B)")         # Aufgabe 2
ok(p.proved and p.kind == "Kontradiktion")
p = K.entails_by_laws("(A or B) and (B -> C)", "A or C")         # Aufgabe 3
ok(p.proved and p.kind == "Deduktionstheorem")
p = K.entails_by_laws("(A or B) and (B -> C)", "A or C", via="contradiction")
ok(p.proved and p.kind == "Widerspruchsbeweis")
p = K.entails_by_laws("A or B", "A")
ok(not p.proved)
ok(not K.tautology_by_laws("A or B").proved and not K.contradiction_by_laws("A or B").proved)
ok(K.contradiction_by_laws("A and not A").proved and K.tautology_by_laws("A -> A").proved)
raises(ValueError, K.entails_by_laws, "A", "B", via="x")
show_all(p)

# ==========================================================================
# 6. Zufallstests gegen Wahrheitstabellen
# ==========================================================================
rng = random.Random(4242)
VARS = ["A", "B", "C", "D"]


def rand_formula(depth):
    if depth == 0 or rng.random() < 0.25:
        if rng.random() < 0.04:
            return Const(rng.random() < 0.5)
        return Var(rng.choice(VARS))
    op = rng.choice(["Not", "And", "Or", "Implies", "Iff", "Xor", "And", "Or", "Implies"])
    if op == "Not":
        return Not(rand_formula(depth - 1))
    cls = {"And": And, "Or": Or, "Implies": Implies, "Iff": Iff, "Xor": Xor}[op]
    return cls(rand_formula(depth - 1), rand_formula(depth - 1))


skipped = 0
for i in range(200):
    f = rand_formula(rng.randint(1, 4))
    try:
        c = K.to_cnf(f)
    except ValueError:
        skipped += 1
        continue
    ok(core.equivalent(f, c.formula), (f, c.formula))
    ok(K.is_cnf(c.formula))
    ok(all(not K._tautologic(cl) for cl in c.clauses))
    ok(core.equivalent(f, core.big_and(K.clause_formula(cl) for cl in c.clause_list)) if c.clause_list or True else True)
    ok(c.clauses == frozenset(c.clause_list))
    ok(c.verified is True)
    # jeder Zwischenschritt war äquivalent: die letzten Formeln der Schritte
    prev = f
    for s in c.steps:
        for label, obj in s.items:
            if label == "" and isinstance(obj, core.Formula) and s.title not in ("Ausgangsterm",):
                ok(core.equivalent(prev, obj), (s.title, prev, obj))
                prev = obj
ok(skipped < 20, skipped)

skipped = 0
n_true = n_false = 0
for i in range(200):
    wb = [rand_formula(rng.randint(1, 3)) for _ in range(rng.randint(1, 3))]
    q = rand_formula(rng.randint(1, 3))
    truth = brute_entails(core.big_and(wb), q)
    try:
        r = K.resolution(wb, q, detail=False)
    except ValueError:
        skipped += 1
        continue
    ok(r.entailed == truth, (wb, q))
    ok(r.derived_empty == truth and r.check == truth and r.agrees)
    ok(core.entails(core.big_and(wb), q) == truth)
    e = K.entails(wb, q)
    ok(e.entailed == truth and e.consistent and len(set(e.verdicts.values())) == 1)
    if not truth:
        ok(r.counterexample is not None and core.big_and(wb).eval(r.counterexample) and not q.eval(r.counterexample))
        ok(e.counterexample is not None and core.big_and(wb).eval(e.counterexample) and not q.eval(e.counterexample))
        n_false += 1
    else:
        n_true += 1
        ok(r.graph[-1]["clause"] == "□")
    # Graph-Vertrag
    ids = {d_["id"] for d_ in r.graph}
    ok(all(pp in ids for d_ in r.graph for pp in d_["parents"]))
    ok(len(ids) == len(r.graph))
    # Beweis mit Schlussregeln: wenn gefunden, dann korrekt
    dv = K.derive(wb, q, max_steps=60)
    if dv.derived:
        ok(truth, (wb, q))
        ok(dv.proof[-1].formula == core.big_and([q]) or dv.proof[-1].formula == q)
    if dv.contradiction:
        ok(not core.is_satisfiable(core.big_and(wb)))
ok(skipped < 20 and n_true > 20 and n_false > 20, (skipped, n_true, n_false))

# Resolution: jede Resolvente folgt semantisch aus ihren Eltern
rr = K.resolution("(A or B) and (not A or C) and (not B or C) and (not C or D)", "D")
ok(rr.derived_empty)
for l in rr.lines:
    if l.origin == "Resolvent":
        pa, pb = rr.lines[l.parents[0] - 1].clause, rr.lines[l.parents[1] - 1].clause
        ok(brute_entails(core.big_and([K.clause_formula(pa), K.clause_formula(pb)]), K.clause_formula(l.clause)))
        ok(l.pivot in {K.lit_var(x) for x in pa} and l.pivot not in {K.lit_var(x) for x in l.clause})

# ==========================================================================
# 7. Vergleich, Begriffe, Rätsel
# ==========================================================================
cmp_ = K.semantic_vs_syntactic("(A or B) and (B -> C)", "A or C")
ok(cmp_.same and cmp_.value is True and cmp_.table.rows[0][1] == "ja" and cmp_.table.rows[1][1] == "ja")
cmp_ = K.semantic_vs_syntactic("A or B", "A")
ok(cmp_.same and cmp_.value is False and cmp_.table.rows[1][1] == "nein")
show_all(cmp_)

info = K.kalkuel_info()
ok({"Kalkül", "Semantische Folgerung ⊨", "Syntaktische Ableitung ⊢", "Korrekt", "Vollständig",
    "Gödelscher Vollständigkeitssatz", "Gödelscher Unvollständigkeitssatz", "Unterschied ⊨ und ⊢",
    "Entscheidbarkeit", "Resolutionskalkül", "Logikprogrammierung"} <= set(info.facts))
ok("Prädikatenlogik erster Stufe" in info.facts["Gödelscher Vollständigkeitssatz"])
ok("zweiter Stufe" in info.facts["Gödelscher Unvollständigkeitssatz"])
ok("Algorithm = Logic + Control" in info.facts["Logikprogrammierung"])
ok(info.table.rows[3][1] == "korrekt: ⊢ ⇒ ⊨")
show_all(info)

fp = K.fish_puzzle()
ok(fp.answer == "Der Deutsche hat den Fisch." and len(fp.solutions) == 1 and fp.value == "Deutsche")
ok(fp.table.rows[0][2] == "Norweger" and fp.table.rows[2][3] == "Milch" and len(fp.table.rows) == 5)
show_all(fp)
bp = K.boxes_puzzle()
ok(len(bp.solutions) >= 1 and all(len(s) == 8 for s in bp.solutions))
# Gegenprobe unabhängig als Logik: jede Lösung erfüllt die Hinweise 9 (3 und 7 gleich) und 16 (nicht 2 und 4 beide Kreuz)
ok(all(s[2] == s[6] and not (s[1] and s[3]) and sum(s) <= 5 and 8 - sum(s) <= 5 for s in bp.solutions))
show_all(bp)

# Darstellung aller Ergebnisse
for obj in (K.entails("A", "A or B"), K.to_cnf("A <-> B"), K.resolution("A", "A"), K.derive("A", "A or B"),
            K.refute([{"A"}, {"¬A"}]), K.tautology_by_laws("A or not A")):
    show_all(obj)
    ok(obj.steps and obj.highlights is not None)

# Namen exportiert
for name in K.__all__:
    ok(hasattr(K, name), name)

print(f"alle Tests bestanden ({N[0]} Asserts)")
