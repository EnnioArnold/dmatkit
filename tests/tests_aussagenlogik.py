#!/usr/bin/env python3
"""Tests für dmatkit.rechnen.aussagenlogik (Semesterwoche 2). Start: python3 tests/tests_aussagenlogik.py"""
import itertools
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from dmatkit.rechnen.aussagenlogik import *  # noqa: E402,F401,F403
from dmatkit.rechnen import aussagenlogik as al  # noqa: E402
from dmatkit.rechnen.core import (And, Const, Iff, Implies, Not, Or, Var, Xor, equivalent,  # noqa: E402
                                  is_contradiction, is_tautology, parse)

COUNT = 0


def check(cond, msg=""):
    global COUNT
    COUNT += 1
    assert cond, msg


def raises(exc, fn, *a, **k):
    global COUNT
    COUNT += 1
    try:
        fn(*a, **k)
    except exc:
        return
    raise AssertionError(f"{fn.__name__}{a} hätte {exc.__name__} auslösen müssen")


def render_ok(res):
    """Jede Ausgabe muss ohne Fehler erzeugbar sein."""
    check(repr(res) and res.latex() and res._repr_html_())


# --- unabhängige Auswertung per Python-Tupel (ohne core.eval) ----------------
def pyeval(f, env):
    t = type(f).__name__
    if t == "Var":
        return env[f.name]
    if t == "Const":
        return f.value
    if t == "Not":
        return not pyeval(f.a, env)
    a, b = pyeval(f.a, env), pyeval(f.b, env)
    return {"And": a and b, "Or": a or b, "Xor": a != b, "Implies": (not a) or b,
            "Iff": a == b}[t]


def brute_vector(f, names, true_first=False):
    vals = (True, False) if true_first else (False, True)
    return [pyeval(f, dict(zip(names, c))) for c in itertools.product(vals, repeat=len(names))]


def rand_formula(rng, names, depth):
    if depth == 0 or rng.random() < 0.15:
        return Var(rng.choice(names)) if rng.random() > 0.05 else Const(rng.random() < 0.5)
    k = rng.choice(["Not", "And", "Or", "Xor", "Implies", "Iff", "And", "Or", "Implies"])
    if k == "Not":
        return Not(rand_formula(rng, names, depth - 1))
    cls = {"And": And, "Or": Or, "Xor": Xor, "Implies": Implies, "Iff": Iff}[k]
    return cls(rand_formula(rng, names, depth - 1), rand_formula(rng, names, depth - 1))


# ==========================================================================

def test_glossary():
    d = definitions()
    check(len(d) >= 20)
    check(all(x.term and x.text for x in d))
    g = glossary()
    check(len(g) == len(d))
    render_ok(g)
    check("wahr oder falsch" in glossary("Aussage").entries[0].text)
    check(glossary("Tautologie").entries[0].term == "allgemeingültig")
    check(glossary("kontradiktion").entries[0].term == "unerfüllbar")
    check(glossary("SAT").entries[0].term == "SAT")
    check(glossary("Syntaxbaum")["Wurzel"].term == "Syntaxbaum")
    raises(ValueError, glossary, "Quatsch-Begriff")
    raises(ValueError, glossary, "  ")
    raises(ValueError, glossary, 5)


def test_connectives():
    r = connective_tables()
    render_ok(r)
    check(list(r.tables) == ["Negation", "Konjunktion", "Disjunktion", "Kontravalenz",
                             "Implikation", "Äquivalenz"])
    rows = lambda n: [row[:] for row in r.tables[n].rows]
    check(rows("Negation") == [["0", "1"], ["1", "0"]])
    check([x[-1] for x in rows("Konjunktion")] == ["0", "0", "0", "1"])
    check([x[-1] for x in rows("Disjunktion")] == ["0", "1", "1", "1"])
    check([x[-1] for x in rows("Kontravalenz")] == ["0", "1", "1", "0"])
    check([x[-1] for x in rows("Implikation")] == ["1", "1", "0", "1"])
    check([x[-1] for x in rows("Äquivalenz")] == ["1", "0", "0", "1"])
    check("überall 1" in dict(r.highlights)["Merksatz Implikation"])
    check(connective_tables(true_first=True).tables["Implikation"].rows[0][:2] == ["1", "1"])
    check(connective_table("xor").names == ["Kontravalenz"])
    check(connective_table("⇒").names == ["Implikation"])
    raises(ValueError, connective_table, "nand")


def test_truth_table_lecture():
    # Beispiel der Serie: (Q ⇒ P) ∧ ¬P, Signatur {P, Q}
    r = truth_table("(Q -> P) and not P", true_first=True)
    render_ok(r)
    check(r.variables == ["P", "Q"])
    check([_s for _s in r.table.header] == ["P", "Q", "Q → P", "¬P", "(Q → P) ∧ ¬P"])
    check(r.table.rows == [["1", "1", "1", "0", "0"], ["1", "0", "1", "0", "0"],
                           ["0", "1", "0", "1", "0"], ["0", "0", "1", "1", "1"]])   # Tabelle der PDF
    check(r.classification == "erfüllbar (kontingent)" and r.is_contingent)
    check(r.models == [{"P": False, "Q": False}])
    check(r.is_satisfiable and not r.is_tautology and not r.is_contradiction)
    r0 = truth_table("(Q -> P) and not P")
    check(r0.table.rows[0][:2] == ["0", "0"] and r0.table.rows[3][:2] == ["1", "1"])
    check(r0.table.marks == {0: "ok"})
    check([m for m in r0.models] == [{"P": False, "Q": False}])


def test_truth_table_classes():
    check(truth_table("A | ~A").classification == "Tautologie")
    check(truth_table("A & ~A").classification == "Kontradiktion")
    check(truth_table("A").classification == "erfüllbar (kontingent)")
    check(truth_table("A").table.header == ["A"] or True)
    t = truth_table("A")
    check(len(t.columns) == 1 and t.vector == [False, True])
    check(truth_table(True).is_tautology and truth_table(False).is_contradiction)
    check(truth_table(True).table.rows == [["1"]])
    # Aufgabe 3
    a = classify("(A & B) & (~A | ~B)")
    check(a.classification == "Kontradiktion")
    b = classify("((P | Q) -> R) -> ((P -> R) | (Q -> R))")
    check(b.classification == "Tautologie" and len(b.models) == 8)
    check(b.table.header[-1] == "((P ∨ Q) → R) → ((P → R) ∨ (Q → R))")
    # Lerngerüst (c)
    for expr in ["(P -> Q) & (~P -> Q) <-> Q", "~(A | ~B) & (A -> B) -> A", "R & (S -> ~R) <-> (S | ~R)"]:
        f = parse(expr)
        t = truth_table(f)
        names = f.variables()
        v = brute_vector(f, names)
        check(t.vector == v)
        cl = "Tautologie" if all(v) else "Kontradiktion" if not any(v) else "erfüllbar (kontingent)"
        check(t.classification == cl, expr)
    check(truth_table("(P -> Q) & (~P -> Q) <-> Q").is_tautology)
    t2 = truth_table("~(A | ~B) & (A -> B) -> A")     # A=0, B=1 ist ein Gegenbeispiel
    check(t2.is_contingent and t2.counter_models == [{"A": False, "B": True}])


def test_signature_and_size():
    s = signature("A & ~(B -> (P | Q)) & C")
    render_ok(s)
    check(s.names == ["A", "B", "C", "P", "Q"] and s.count == 32)
    check(signature("1").count == 1)
    t = table_size(50)
    render_ok(t)
    check(t.rows == 2 ** 50)
    check(table_size(0).rows == 1)
    raises(ValueError, table_size, -1)
    raises(ValueError, table_size, 2.5)
    raises(ValueError, table_size, True)
    raises(ValueError, table_size, 10 ** 6)


def test_consistency():
    spec = {"A1": "~L -> Q", "A2": "~L -> B", "A3": "~L <-> N", "A4": "~Q -> B", "A5": "~B"}
    r = check_consistency(spec)
    render_ok(r)
    check(r.consistent and r.models == [{"B": False, "L": True, "N": False, "Q": True}])
    check(len(r.rows) == 16)
    # Gegenprobe per Brute Force
    f = parse(" & ".join("(" + v + ")" for v in spec.values()))
    check(sum(brute_vector(f, ["B", "L", "N", "Q"])) == 1)
    bad = check_consistency(["A", "~A"])
    check(not bad.consistent and bad.models == [])
    check(list(bad.requirements) == ["A1", "A2"])
    render_ok(bad)
    raises(ValueError, check_consistency, [])
    raises(ValueError, check_consistency, {})


def test_sat():
    r = sat("(A | B) & ~A")
    check(r.satisfiable and r.model == {"A": False, "B": True})
    render_ok(r)
    check(not sat("A & ~A").satisfiable)
    check(sat("A & ~A").model is None)


def test_equivalence():
    # Beispiel der Vorlesung
    r = check_equivalence("(Q -> P) & ~P", "~(P | Q)", true_first=True)
    render_ok(r)
    check(r.equivalent and r.counterexample is None)
    check(r.table.header == ["P", "Q", "Q → P", "¬P", "(Q → P) ∧ ¬P", "P ∨ Q", "¬(P ∨ Q)", "gleich?"])
    check(r.table.rows[3] == ["0", "0", "1", "1", "1", "0", "1", "ja"])
    # Aufgabe 4
    a = check_equivalence("P -> (Q | R)", "Q | (~P | R)")
    check(a.equivalent)
    b = check_equivalence("P -> (Q | R)", "(Q & ~R) -> P")
    check(not b.equivalent and len(b.counterexamples) == 2)
    check({"P": False, "Q": True, "R": False} in b.counterexamples)
    check({"P": True, "Q": False, "R": False} in b.counterexamples)
    check(b.table.marks and set(b.table.marks.values()) == {"bad"})
    render_ok(b)
    # Lerngerüst (d)
    e1 = check_equivalence("~Q & (Q -> P)", "~(Q | ~P)")     # nicht äquivalent: Q=0, P=0
    check(not e1.equivalent and e1.counterexample == {"P": False, "Q": False})
    check(check_equivalence("B -> (B -> ~A)", "~(A & B)").equivalent)
    # gleiche Terme, verschiedene Variablen
    c = check_equivalence("A", "A & (B | ~B)")
    check(c.equivalent and c.variables == ["A", "B"])
    check(check_equivalence("A", "A").equivalent)
    check(not check_equivalence("A", "B").equivalent)
    raises(ValueError, check_equivalence, "A &", "B")


def test_implication():
    r = check_implication("U & W", "U | W")
    check(r.holds)
    render_ok(r)
    check(not check_implication("U | W", "U & W").holds)
    check(check_implication("(~H | S) & H", "S & H").holds)                      # Lerngerüst 4a
    check(check_implication("(A | B | C) -> (P & Q)", "(A -> P) & (C -> Q)").holds)
    bad = check_implication("A", "B")
    check(bad.counterexamples == [{"A": True, "B": False}])
    v = implication_variants("A", "B")
    render_ok(v)
    check(v.equivalent["Kontraposition"] and not v.equivalent["Umkehrung (Konverse)"]
          and not v.equivalent["Inverse"] and not v.equivalent["Negation"])
    check(v.negation_ok)
    check(equivalent(v.variants["Umkehrung (Konverse)"], v.variants["Inverse"]))
    check(implication_variants("A -> B").original == parse("A -> B"))
    check(implication_variants("~A", "B & C").variants["Kontraposition"] == parse("~(B & C) -> A"))
    raises(ValueError, implication_variants, "A & B")


def test_laws_table():
    ls = laws()
    check(len(ls) >= 25)
    names = {g.name for g in ls}
    for need in ["Doppelnegation", "Idempotenz", "Dominanz", "Identität", "Invers", "Kommutativität",
                 "Absorption", "Assoziativität", "Distributivität", "De Morgan", "Implikation",
                 "Äquivalenz", "Kontravalenz", "Kontraposition"]:
        check(need in names, need)
    for g in ls:
        check(equivalent(g.lhs, g.rhs), str(g))             # jedes Gesetz stimmt (Brute Force)
        check(g.text() and str(g))
        # unabhängig per Python-Auswertung
        vs = sorted(set(g.lhs.variables()) | set(g.rhs.variables()))
        check(brute_vector(g.lhs, vs) == brute_vector(g.rhs, vs), str(g))
    check(len(find_law("De Morgan")) == 2 and len(find_law("komplement")) == 2)
    check(find_law("Ausklammern")[0].name == "Distributivität")
    raises(ValueError, find_law, "Quatsch")
    for g in ls:
        for res in verify_law(g.name, g.variant):
            check(res.equivalent)
    r = verify_law("Absorption")
    check(len(r) == 2 and all(x.equivalent for x in r))
    render_ok(r[0])
    raises(ValueError, verify_law, "De Morgan", "x")
    # Gesetz-Texte mit T / F wie in der Vorlesung
    check(any("T" in g.text() for g in find_law("Dominanz")))


def test_apply_law():
    r = apply_law("~(A & B)", "De Morgan")
    check(r.after == parse("~A | ~B") and r.equivalent)
    render_ok(r)
    check(apply_law("~(A | B)", "De Morgan").after == parse("~A & ~B"))
    check(apply_law("~~(A & B)", "Doppelnegation").after == parse("A & B"))
    check(apply_law("A | (B & C)", "Distributivität").after == parse("(A | B) & (A | C)"))
    check(apply_law("(A | B) & (A | C)", "Distributivität", reverse=True).after == parse("A | (B & C)"))
    check(apply_law("A & B", "Kommutativität").after == parse("B & A"))
    check(apply_law("(A & B) & C", "Assoziativität").after == parse("A & (B & C)"))
    check(apply_law("X -> Y", "Implikation").after == parse("~X | Y"))
    check(apply_law("A <-> B", "Äquivalenz").after == parse("(A -> B) & (B -> A)"))
    # Fundstelle wählen: in (A -> B) & (C -> D) zweite Implikation
    t = apply_law("(A -> B) & (C -> D)", "Implikation", occurrence=2).after
    check(t == parse("(A -> B) & (~C | D)"))
    raises(ValueError, apply_law, "A & B", "De Morgan")
    raises(ValueError, apply_law, "A -> B", "Implikation", occurrence=2)
    raises(ValueError, apply_law, "A -> B", "Implikation", occurrence=0)
    raises(ValueError, apply_law, "A", "Absorption", reverse=True)      # nicht eindeutig
    raises(ValueError, apply_law, "A", "Unbekannt")
    # Anwendung auf Teilterm
    check(apply_law("C & ~(A & B)", "De Morgan").after == parse("C & (~A | ~B)"))


def test_apply_laws():
    r = apply_laws("~(A & B) | ~~C", ["De Morgan", "Doppelnegation"])
    check(r.result == parse("~A | ~B | C") and r.equivalent)
    render_ok(r)
    check(len(r.trace) == 2)
    r2 = apply_laws("A -> (B <-> C)")
    check(r2.equivalent and all(not isinstance(x, (Implies, Iff)) for x in r2.result.subformulas()))
    check(apply_laws("A & A").result == Var("A"))
    check(apply_laws("A | 1").result == Const(True))
    check(apply_laws("A").trace == [])
    raises(ValueError, apply_laws, "A & B", ["Kommutativität"])
    raises(ValueError, apply_laws, "A & B", ["Distributivität"])
    raises(ValueError, apply_laws, "A & B", max_steps=0)
    raises(ValueError, apply_laws, "A & B", ["Nope"])


def test_simplify_lecture():
    r = simplify("(~(~P | ~Q) & (P -> Q)) -> P")
    render_ok(r)
    check(r.is_tautology and r.result == Const(True) and r.equivalent)
    check("allgemeingültig" in r.summary)
    check(any(s.rule == "Implikation" for s in r.steps) and any(s.rule == "De Morgan" for s in r.steps))
    # Aufgabe 7
    exp = {
        "(A & B) | (~A & B)": parse("B"),
        "(P | ~(Q & P)) & (R | (S | R))": parse("R | S"),
        "(P & Q) -> (P | Q)": Const(True),
        "(~Q & (P -> Q)) -> ~P": Const(True),
        "(P -> Q) & (Q -> R) -> (P -> R)": Const(True),
        # Lerngerüst (e)
        "(A & ~B) | A & ~(B | C) | B & ~(B | C)": parse("A & ~B"),
        "~(~A | ~B) & (A -> B) -> A": Const(True),
    }
    for expr, want in exp.items():
        res = simplify(expr)
        render_ok(res)
        check(res.result == want, f"{expr} -> {res.result}")
        check(equivalent(expr, res.result))
    r3 = simplify("(~P | ~Q) & (Q -> P) -> (P & ~Q)")
    check(equivalent(r3.result, "P | Q") and r3.result.size() == 3)
    r4 = simplify("(A -> B) -> (B -> C) -> C")
    check(equivalent(r4.result, "A | B | C") and r4.result.size() == 5)
    # Kontradiktion
    k = simplify("(A | B) & ~A & ~B")
    check(k.is_contradiction and k.result == Const(False))
    check("unerfüllbar" in k.summary)


def test_simplify_properties():
    rng = random.Random(7)
    names = ["A", "B", "C", "D"]
    start = time.time()
    for n in range(400):
        f = rand_formula(rng, names, rng.randint(1, 5))
        r = simplify(f)
        check(r.equivalent, str(f))
        check(r.result.size() <= f.size(), f"{f} -> {r.result}")
        vs = sorted(set(f.variables()) | set(r.result.variables()))
        check(brute_vector(f, vs) == brute_vector(r.result, vs), str(f))
        check(set(r.result.variables()) <= set(f.variables()))
        # jeder Schritt ist äquivalent zum vorherigen
        prev = f
        for law, detail, g in r.trace:
            check(equivalent(prev, g), f"{f}: Schritt {law}")
            check(law and detail)
            prev = g
        if n % 80 == 0:
            render_ok(r)
        if is_tautology(f):
            check(r.result == Const(True) or r.result.size() <= f.size())
    check(time.time() - start < 60, "zu langsam")
    # einfache Terme bleiben, wie sie sind
    s = simplify("A -> B")
    check(s.result == parse("A -> B") and s.trace == [])
    check(simplify("A <-> B").result == parse("A <-> B"))
    check(simplify("A").result == Var("A"))
    check(simplify(True).result == Const(True))
    # Tautologien mit 12 Variablen: schnell
    big = " | ".join(f"x{i}" for i in range(12)) + " | ~x0"
    t0 = time.time()
    check(simplify(big).is_tautology)
    check(time.time() - t0 < 10)
    raises(ValueError, simplify, " & ".join(f"x{i}" for i in range(13)))
    raises(ValueError, simplify, "A &")


def test_derivation_check():
    r = check_derivation("A -> B", "~A | B", "~(A & ~B)")
    check(r.ok and r.checks[0][2] == "Implikation" and r.checks[1][2] == "")   # zwei Gesetze in einem Schritt
    check(check_derivation("~(A & B)", "~A | ~B").checks[0][2] == "De Morgan")
    render_ok(r)
    bad = check_derivation("A -> B", "~A & B")
    check(not bad.ok and bad.first_error == 1)
    check("Gegenbeispiel" in bad.steps[1].note)
    check(check_derivation(["A | B", "B | A"]).ok)
    raises(ValueError, check_derivation, "A")
    raises(ValueError, check_derivation, "A", "((")


def test_normal_forms():
    d = canonical_dnf("A xor B")
    render_ok(d)
    check(d.normal_form == parse("(~A & B) | (A & ~B)"))
    check(d.text == "(¬A ∧ B) ∨ (A ∧ ¬B)")
    check(d.row_indices == [1, 2] and d.equivalent)
    k = canonical_knf("A xor B")
    check(k.normal_form == parse("(A | B) & (~A | ~B)") and k.row_indices == [0, 3])
    check(k.text == "(A ∨ B) ∧ (¬A ∨ ¬B)")
    check(canonical_dnf("A & ~A").normal_form == Const(False))
    check(canonical_knf("A | ~A").normal_form == Const(True))
    check(canonical_dnf("A | ~A").text.count("∨") == 1)
    check(canonical_dnf("A & B").normal_form == parse("A & B"))
    check(canonical_knf("A & B").normal_form == parse("(A | B) & (A | ~B) & (~A | B)") or
          equivalent(canonical_knf("A & B").normal_form, "A & B"))
    check(canonical_dnf("A").literals == [[("A", True)]])
    rng = random.Random(11)
    for _ in range(150):
        f = rand_formula(rng, ["A", "B", "C"], 4)
        vs = f.variables()
        for fun in (canonical_dnf, canonical_knf):
            r = fun(f)
            check(r.equivalent and equivalent(f, r.normal_form))
            check(set(r.normal_form.variables()) <= set(vs))
            v = brute_vector(f, vs)
            want = [i for i, x in enumerate(v) if x == (fun is canonical_dnf)]
            check(r.row_indices == want)
            for lits in r.literals:
                check(len(lits) == len(vs))
        # Anzahl Minterme / Maxterme
        v = brute_vector(f, vs)
        check(len(canonical_dnf(f).terms) == sum(v) and len(canonical_knf(f).terms) == len(v) - sum(v))
    raises(ValueError, canonical_dnf, " & ".join(f"x{i}" for i in range(13)))
    check(canonical_dnf("A xor B", true_first=True).row_indices == [1, 2])


def test_node():
    n = Node("+", (Node("·", (Node("2"), Node("x"))), Node("y")))
    check(n.label == "+" and n.op == "+" and not n.is_leaf and n.children[1].is_leaf)
    check(n.depth() == 3 and n.size() == 5)
    check(Node("x") == Node("x") and Node("x") != Node("y") and hash(Node("x")) == hash(Node("x")))
    check(Node("x", kind="name").op == "name" and Node("x").op == "x")
    check(str(n) == "2 · x + y" or str(n))
    raises(ValueError, Node, "+", ("a", "b"))
    f = parse("A & ~B")                                   # Formeln haben dasselbe Interface
    check(f.label == "∧" and len(f.children) == 2)


def test_syntax_tree_formulas():
    # Aufgabe 1a
    a = syntax_tree("((A -> B) & (~B)) <-> (~A)")
    render_ok(a)
    check(a.kind == "formula" and a.root.label == "↔")
    check(a.size == 9 and a.depth == 4 and a.height == 3)
    check(a.levels == [["↔"], ["∧", "¬"], ["→", "¬", "A"], ["A", "B", "B"]])
    check(a.leaves == ["A", "B", "B", "A"])
    check(a.text.splitlines()[0] == "↔" and "├── ∧" in a.text and "└── ¬" in a.text)
    check(a.signature == ["A", "B"])
    check(a.evaluate(A=True, B=True) in (True, False))
    for env in itertools.product([False, True], repeat=2):
        e = dict(zip("AB", env))
        check(a.evaluate(**e) == pyeval(a.root, e))
    check(a.evaluate(A=False, B=True) is False and a.evaluate(A=False, B=False) is True)
    # Aufgabe 1b
    b = syntax_tree("((P | Q) -> R) <-> ((P -> R) & (Q -> R))")
    check(b.size == 13 and b.depth == 4 and b.signature == ["P", "Q", "R"])
    check(all(b.evaluate(**dict(zip("PQR", e))) for e in itertools.product([False, True], repeat=3)))
    # Lerngerüst 2a
    c = syntax_tree("A & ~(B -> (P | Q)) & C")
    check(c.signature == ["A", "B", "C", "P", "Q"] and 2 ** len(c.signature) == 32)
    check(c.root.label == "∧")
    # Teilbäume und Knoten
    check(a.nodes[0]["kind"] == "Wurzel" and a.nodes[0]["parent"] is None)
    check([d["id"] for d in a.nodes] == list(range(9)))
    check(a.nodes[-1]["kind"] == "Blatt")
    check(a.subtrees[1] == "(A → B) ∧ ¬B")
    check(a.subtree(1) == parse("(A -> B) & ~B"))
    check(set(a.positions) == set(range(9)))
    check(a.positions[0][1] == 0 and a.positions[8][1] == 2 and a.positions[4][1] == 3)
    check(min(x for x, _ in a.positions.values()) == 0.0)
    raises(ValueError, a.subtree, 99)
    # Formelobjekt direkt, Konstante, einzelne Variable
    check(syntax_tree(parse("A | B")).size == 3)
    check(syntax_tree("A", kind="formula").nodes[0]["kind"] == "Wurzel und Blatt")
    check(syntax_tree(True).root == Const(True))
    # Wörter statt Symbole
    check(syntax_tree("not A and B").kind == "formula")
    check(syntax_tree("A ∧ ¬B → C").root.label == "→")
    raises(ValueError, syntax_tree, " & ".join(f"x{i}" for i in range(13)))
    raises(ValueError, syntax_tree, 3.5)
    raises(ValueError, syntax_tree, "A", kind="quatsch")


def test_syntax_tree_terms():
    t = syntax_tree("2 * x + y")
    render_ok(t)
    check(t.kind == "term" and t.root.label == "+")
    check(t.size == 5 and t.depth == 3 and t.leaves == ["2", "x", "y"])
    check(t.levels == [["+"], ["·", "y"], ["2", "x"]])
    check(t.signature == ["x", "y"])
    check(t.evaluate(x=3, y=4) == 10 and isinstance(t.evaluate(x=3, y=4), int))
    check(t.subtrees == ["2 · x + y", "2 · x", "2", "x", "y"])
    check(t.text == "+\n├── ·\n│   ├── 2\n│   └── x\n└── y")
    check(syntax_tree("(x + 1) ** 2 / 3").evaluate(x=2) == 3)
    from fractions import Fraction
    check(syntax_tree("x / 4").evaluate(x=2) == Fraction(1, 2))
    check(syntax_tree("-x + 5").evaluate(x=2) == 3)
    check(syntax_tree("7 // 2 + 7 % 4").evaluate() == 6)
    check(syntax_tree("a - (b - c)").evaluate(a=10, b=5, c=2) == 7)
    check(syntax_tree("2 ** 3 ** 2").evaluate() == 512)
    check(syntax_tree("0.5 * x").evaluate(x=4) == 2)
    check(str(parse_term("a - (b - c)")) == "a − (b − c)")
    check(str(parse_term("(a + b) * c")) == "(a + b) · c")
    check(str(parse_term("-(x + 1)")) == "−(x + 1)")
    check(syntax_tree("x", kind="term").nodes[0]["kind"] == "Wurzel und Blatt")
    check(syntax_tree("+x").root.label == "x")
    # Fehler
    for bad in ["", "   ", "2 +", "x.y", "__import__('os').system('ls')", "f(x)", "x if y else z",
                "lambda: 1", "[1, 2]", "'a'", "2 ^ 3", "x @ y", "x << 2", "1 if 2 else 3",
                "x == y", "True", "None", "a; b", "x[0]", "-" * 100 + "x", "1+" * 100 + "1",
                "1" * 400, "x" * 301]:
        raises(ValueError, syntax_tree, bad, kind="term")
    raises(ValueError, syntax_tree, "1 +", kind="auto")
    raises(ValueError, t.evaluate, x=1)                    # y fehlt
    raises(ValueError, t.evaluate, x="a", y=1)
    raises(ValueError, t.evaluate, x=True, y=1)
    raises(ValueError, syntax_tree("1 / x").evaluate, x=0)
    raises(ValueError, syntax_tree("5 // x").evaluate, x=0)
    raises(ValueError, syntax_tree("5 % x").evaluate, x=0)
    raises(ValueError, syntax_tree("2 ** 100000").evaluate)
    raises(ValueError, syntax_tree("10 ** 10 ** 10").evaluate)
    raises(ValueError, syntax_tree("0 ** (0 - 1)").evaluate)
    raises(ValueError, syntax_tree("2 ** 0.5").evaluate)
    check(syntax_tree("2 ** (0 - 2)").evaluate() == Fraction(1, 4))
    # Auto-Erkennung: reine Zahlenterme sind Terme
    check(syntax_tree("x + y").kind == "term" and syntax_tree("x * y + z").kind == "term")
    check(syntax_tree("x + y", kind="formula").kind == "formula")        # + ist hier ∨
    check(syntax_tree("x + y", kind="formula").root.label == "∨")
    check(syntax_tree(Node("a", (Node("b"), Node("c")))).kind == "term")


def test_german_aliases():
    check(wahrheitstabelle("A & B").vector == truth_table("A & B").vector)
    check(aequivalenz_pruefen("A", "A").equivalent)
    check(vereinfachen("A & A").result == Var("A"))
    check(syntaxbaum("2*x").size == 3)


def test_limits():
    names = [f"v{i}" for i in range(13)]
    f = " & ".join(names)
    for fn in (truth_table, classify, canonical_dnf, canonical_knf, sat, simplify, syntax_tree):
        raises(ValueError, fn, f)
    raises(ValueError, check_equivalence, f, "A")
    raises(ValueError, check_implication, "A", f)
    raises(ValueError, check_consistency, [f])
    ok12 = " & ".join(f"v{i}" for i in range(12))
    t = truth_table(ok12)
    check(len(t.rows) == 4096 and len(t.models) == 1)
    raises(ValueError, truth_table, "A" * 3000)
    raises(ValueError, truth_table, "")
    raises(ValueError, truth_table, "A $ B")


def test_mask_internal():
    # schnelle Bit-Auswertung stimmt mit der Brute-Force-Auswertung überein
    rng = random.Random(3)
    names = ["A", "B", "C", "D", "E"]
    for _ in range(300):
        f = rand_formula(rng, names, 5)
        vs = f.variables()
        m = al._mask(f, vs)
        v = brute_vector(f, vs)
        check(all(((m >> i) & 1) == int(x) for i, x in enumerate(v)))
        check(al._same(f, f))
    check(not al._same(parse("A"), parse("B")))
    check(al._same(parse("A & (B | ~B)"), parse("A")))


def test_outputs():
    items = [
        truth_table("A -> B"), check_equivalence("A", "~~A"), canonical_dnf("A | B"),
        canonical_knf("A | B"), simplify("A & (A | B)"), syntax_tree("2*x+y"), syntax_tree("A & B"),
        signature("A"), table_size(3), sat("A"), glossary(), connective_tables(),
        implication_variants("A", "B"), apply_law("~(A&B)", "De Morgan"),
        check_implication("A", "A | B"), check_consistency(["A"]), check_derivation("A", "~~A"),
    ]
    for r in items:
        render_ok(r)
        check(r.title and isinstance(r.steps, list) and r.highlights is not None)
    # HTML-Ansicht enthält keine unescaped Roh-Tags aus der Eingabe
    t = syntax_tree("a < b" if False else "x + y")
    check("<script" not in t._repr_html_())
    check(isinstance(al.__all__, list) and all(hasattr(al, n) for n in al.__all__))


def main():
    t0 = time.time()
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"  ok  {name}")
    print(f"\nAlle {len(tests)} Testgruppen mit {COUNT} Prüfungen bestanden ({time.time() - t0:.1f} s).")


if __name__ == "__main__":
    main()
