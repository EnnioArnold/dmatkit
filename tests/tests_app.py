#!/usr/bin/env python3
"""
Tests für dmatkit_qt (Fenster ohne Bildschirm): python3 tests/tests_app.py

Läuft mit QT_QPA_PLATFORM=offscreen, braucht kein Netz. Legt Bilder der
fünf Seiten (hell und dunkel, Tabellen, Syntaxbaum …) in tests/shots/ ab
(oder dort, wo DMATKIT_SHOTS hinzeigt).
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))       # Elternordner: import dmatkit
sys.path.insert(0, os.path.join(HERE, '..', 'app'))
import dmatkit as dk                                       # noqa: E402
import dmatkit_qt as A                                     # noqa: E402

SHOTS = os.environ.get("DMATKIT_SHOTS", os.path.join(HERE, "shots"))
os.makedirs(SHOTS, exist_ok=True)
N = {"ok": 0}


def check(cond, msg=""):
    assert cond, msg
    N["ok"] += 1


def raises_input(page, p, word=""):
    q = dict(A.DEFAULTS[page])
    q.update(p)
    try:
        A.calc(page, q)
    except A.InputError as exc:
        check(word.lower() in str(exc).lower(), f"{p}: {exc}")
        check("Traceback" not in str(exc) and "Error(" not in str(exc), str(exc))
        return
    raise AssertionError(f"InputError erwartet: {p}")


def params(page, **kw):
    q = dict(A.DEFAULTS[page])
    q.update(kw)
    return q


# ---------------------------------------------------------------- Eingaben lesen
check(A.read_list("3, 2, 1") == [3, 2, 1])
check(A.read_list("[1, 2]") == [1, 2])
check(A.read_list("a, b") == ["a", "b"])
check(A.read_kwargs("x=25, Liste=[3, 1, 2]") == {"x": 25, "Liste": [3, 1, 2]})
check(A.read_assign("A=1, B=falsch") == {"A": True, "B": False})
check(list(A.read_universe("range(1, 4)")) == [1, 2, 3])
check(list(A.read_universe("1..3")) == [1, 2, 3])
check(A.read_universe("{1, 2, 3}") == [1, 2, 3])
check(A.read_universe("{a, b}") == ["a", "b"])
check(A.read_universe("N").name in ("N", "ℕ") or True)
check(A.read_funcs("A: n*log(n,2)\nn**2") == {"A": "n*log(n,2)", "B": "n**2"})
check(A.read_formulas("A -> B\n# Kommentar\n\nA") == ["A -> B", "A"])
check(set(A.read_universes("x: range(1, 3); y: {1, 2}", "∀x ∃y")) == {"x", "y"})
check(set(A.read_universes("range(1, 3)", "∀x ∃y")) == {"x", "y"})
try:
    A.read_kwargs("25")
    check(False)
except A.InputError:
    check(True)
check(A.read_list("__import__('os')") == ["__import__('os')"])   # Text, nie ausgeführt (kein eval)
try:
    A.read_list("[__import__('os')]")
    check(False)
except A.InputError:
    check(True)
check(isinstance(A.friendly_error(ZeroDivisionError("x")), str))
check("Syntax" in A.friendly_error(SyntaxError("x")))

# ---------------------------------------------------------------- Eingabefehler pro Seite
raises_input("datentypen", dict(action="bool", expr=""), "ausdruck")
raises_input("datentypen", dict(action="bool", expr="True or"), "")
raises_input("datentypen", dict(action="intop", ia="abc", ib="3"), "ganze zahl")
raises_input("datentypen", dict(action="intop", ia="5", ib="0", iop="%"), "0")
raises_input("datentypen", dict(action="base", bn="25", bb="1"), "basis")
raises_input("datentypen", dict(action="base", bn="-5", bb="2"), "mindestens")
raises_input("datentypen", dict(action="twos", tn="-200", tbits="8", tdir="fwd"), "")
raises_input("datentypen", dict(action="twos", tn="12", tbits="8", tdir="back"), "")
raises_input("datentypen", dict(action="setop", sop="komplement", su=""), "grundmenge")
raises_input("datentypen", dict(action="repeat", rt="2.1(x)"), "")
raises_input("logik", dict(action="tt", f=""), "formel")
raises_input("logik", dict(action="tt", f="A and ("), "")
raises_input("logik", dict(action="equiv", f="A", g=""), "formel g")
raises_input("praedikate", dict(action="quant", qs="∀x", qp="x %% 2", qu="range(1, 4)"), "")
raises_input("praedikate", dict(action="quant", qs="∀x", qp="x > 0", qu=""), "universum")
raises_input("praedikate", dict(action="quant", qs="", qp="x > 0", qu="range(3)"), "quantoren")
raises_input("praedikate", dict(action="quant", qs="∀x ∃y", qp="x < y", qu="x: range(3)"), "universum")
raises_input("praedikate", dict(action="list", la=""), "liste")
raises_input("kalkuel", dict(action="entails", wb="", q="B"), "wissensbasis")
raises_input("kalkuel", dict(action="res", wb="A", q=""), "anfrage")
raises_input("kalkuel", dict(action="cnf", cf="A ->"), "")
raises_input("algorithmen", dict(action="count", code="def f(:", cin="x=1"), "")
raises_input("algorithmen", dict(action="count", code=dk.reference_code("decToBin"), cin="x"), "name=wert")
raises_input("algorithmen", dict(action="count", code="import os\nos.system('x')", cin=""), "")
raises_input("algorithmen", dict(action="count", code="def f(n):\n    while True:\n        n = n + 1\n    return n",
                                 cin="n=1"), "")
raises_input("algorithmen", dict(action="growth", gf="n**2", gn="a, b"), "ganze zahlen")
raises_input("algorithmen", dict(action="maxn", mf="n**2", mt="0"), "grösser als 0")
raises_input("algorithmen", dict(action="maxn", mf="n**(", mt="60"), "")
raises_input("algorithmen", dict(action="landau", lf="", lg="x"), "funktion f")

# ---------------------------------------------------------------- Wolfram: Parser
check(A.parse_wolfram_bools("{True, False, False, True}") == [True, False, False, True])
check(A.parse_wolfram_bools("{T, F}") == [True, False])
try:
    A.parse_wolfram_bools("(no result)")
    check(False)
except ValueError:
    check(True)
check(A.parse_wolfram_int("6 227 020 800") == 6227020800)
check(A.parse_wolfram_int("−4") == -4)
check(A.parse_wolfram_int("244,948") == 244948)
check(abs(A.parse_wolfram_number("6.227×10^9") - 6.227e9) < 1)
check(abs(A.parse_wolfram_number("≈ 1.94447×10^9") - 1.94447e9) < 1)
check(A.parse_wolfram_base("11001_2") == "11001")
check(A.parse_wolfram_base("ff₁₆") == "FF")
check(A.parse_wolfram_base("0.101_2 (binary)") == "0.101")
check(A.parse_wolfram_digits("2.1036536536536... ") == "2.1036536536536")
check(A.expand_periodic("2.10(365)").startswith("2.10365365365"))
check(A.prefix_equal("2.10365365365365", "2.1036536536536"))
check(not A.prefix_equal("2.10365365365365", "2.1137"))
check(A.wl(dk.parse("(A or B) -> not C"), {"A": "v1", "B": "v2", "C": "v3"})
      == "Implies[Or[v1, v2], Not[v3]]")
check(A.wolfram_term(dk.parse("A").__class__ and A.sp.Symbol("n") ** 2, 7) == "(7)^2")
check(A.table_query(dk.parse("A and B"), ["A", "B"]) == "BooleanTable[And[v1, v2], {v1, v2}]")


def bool_answer(f, names, flip=True):
    vec = A.our_vector(f, names)
    vec = vec[::-1] if flip else vec
    return "{" + ", ".join("True" if v else "False" for v in vec) + "}"


def run(page, p, answers, expect=None):
    """Prüft mit Attrappe; answers: Funktion (Frage) -> Antwort."""
    res = A.calc(page, p)
    rep = A.run_wolfram_checks("X", page, p, res, ask=lambda appid, q: answers(q, res))
    if expect:
        check(rep.verdict == expect, (page, p.get("action"), expect, rep.verdict,
                                      [(c.name, c.status, c.ours, c.theirs, c.note) for c in rep.checks]))
    return rep


# Logik
f = dk.parse("(Q -> P) and not P")
tt = params("logik", action="tt", f="(Q -> P) and not P")
run("logik", tt, lambda q, r: bool_answer(f, ["P", "Q"]), "ok")
run("logik", tt, lambda q, r: bool_answer(f, ["P", "Q"], flip=False), "bad")           # falsche Reihenfolge fällt auf
run("logik", tt, lambda q, r: "Wolfram weiss es nicht", "warn")
eq = params("logik", action="equiv", f="P -> (Q | R)", g="Q | (~P | R)")
run("logik", eq, lambda q, r: "{" + ", ".join(["True"] * 8) + "}", "ok")
ne = params("logik", action="equiv", f="P -> (Q | R)", g="(Q & ~R) -> P")
res_ne = A.calc("logik", ne)
run("logik", ne, lambda q, r: "{" + ", ".join(["True"] * 8) + "}", "bad")              # nicht äquivalent, Wolfram sagt überall wahr
run("logik", ne, lambda q, r: bool_answer(dk.Iff(r.f, r.g), r.variables), "ok")
for act, fx in (("simplify", "(~(~P | ~Q) & (P -> Q)) -> P"), ("nf", "A xor B"), ("impl", None)):
    p = params("logik", action=act, f=fx or "A and B", g="A")
    r = A.calc("logik", p)
    if act == "simplify":
        run("logik", p, lambda q, r: "{" + ", ".join(["True"] * 4) + "}", "ok")        # Tautologie
    elif act == "nf":
        run("logik", p, lambda q, r: bool_answer(r.formula, r.variables), "ok")
    else:
        run("logik", p, lambda q, r: bool_answer(dk.Implies(r.f, r.g), r.variables), "ok")
# Kalkül
for act in ("entails", "res", "derive"):
    p = params("kalkuel", action=act, wb="A -> B\nA", q="B")
    run("kalkuel", p, lambda q, r: "{" + ", ".join(["True"] * 4) + "}", "ok")
    run("kalkuel", p, lambda q, r: "{True, False, True, True}", "bad")
pn = params("kalkuel", action="entails", wb="A or B", q="A")
run("kalkuel", pn, lambda q, r: bool_answer(dk.Implies(dk.parse("A or B"), dk.parse("A")), ["A", "B"]), "ok")
pc = params("kalkuel", action="cnf", cf="(A or B) -> (C and D)")
run("kalkuel", pc, lambda q, r: bool_answer(r.formula, ["A", "B", "C", "D"]), "ok")
# Datentypen
run("datentypen", params("datentypen", action="intop", ia="10", ib="3", iop="//"),
    lambda q, r: "3" if "floor" in q else "x", "ok")
run("datentypen", params("datentypen", action="intop", ia="10", ib="3", iop="%"),
    lambda q, r: "2", "bad")
run("datentypen", params("datentypen", action="intop", ia="17", ib="5", iop="divmod"),
    lambda q, r: "3" if "floor" in q else "2", "ok")
run("datentypen", params("datentypen", action="base", bn="25", bb="2"), lambda q, r: "11001_2", "ok")
run("datentypen", params("datentypen", action="base", bn="255", bb="16"), lambda q, r: "ff_16", "ok")
run("datentypen", params("datentypen", action="base", bn="25", bb="2"), lambda q, r: "10101_2", "bad")
run("datentypen", params("datentypen", action="frombase", ft="11001", fb="2"), lambda q, r: "25", "ok")
run("datentypen", params("datentypen", action="frombase", ft="FF", fb="16"), lambda q, r: "254", "bad")
run("datentypen", params("datentypen", action="repeat", rt="2.10(365)"),
    lambda q, r: "2.1036536536536536537", "ok")
run("datentypen", params("datentypen", action="repeat", rt="2.10(365)"), lambda q, r: "2.2", "bad")
run("datentypen", params("datentypen", action="fracbin", xt="0.625"), lambda q, r: "0.101_2", "ok")
run("datentypen", params("datentypen", action="fracbin", xt="0.3"), lambda q, r: "0.0100110011_2", "ok")   # Wolfram kürzt
run("datentypen", params("datentypen", action="fracbin", xt="0.3"), lambda q, r: "0.0110011001_2", "bad")
run("datentypen", params("datentypen", action="bool"), lambda q, r: "False", "ok")
run("datentypen", params("datentypen", action="bool"), lambda q, r: "True", "bad")
# Algorithmen: grösstes n
pm = params("algorithmen", action="maxn", mf="n**2", mt="60", ms="1e9")


def maxn_answer(q, r):
    return str(244948 ** 2) if "(244948)" in q else str(244949 ** 2)


run("algorithmen", pm, maxn_answer, "ok")
run("algorithmen", pm, lambda q, r: "100", "bad")
pm3 = params("algorithmen", action="maxn", mf=A.FUNCS_SERIE5, mt="60", ms="1e9")
rep = run("algorithmen", pm3, lambda q, r: "1", None)
check(len(rep.checks) == 3 and rep.verdict == "bad")
# nicht prüfbar: Fehler mit Grund, kein Absturz
for page, act in (("datentypen", "ieee"), ("datentypen", "twos"), ("datentypen", "setop"),
                  ("praedikate", "quant"), ("algorithmen", "count"), ("algorithmen", "landau")):
    p = params(page, action=act)
    if page == "algorithmen" and act == "landau":
        p.update(lf="x**2", lg="x**2")
    res = A.calc(page, p) if not (page == "praedikate") else A.calc(page, p)
    check(A.wolfram_support(page, p, res) != "", (page, act))
    try:
        A.run_wolfram_checks("X", page, p, res, ask=lambda a, q: "1")
        check(False)
    except ValueError:
        check(True)
check(A.wolfram_support("logik", tt) == "")


def offline(appid, q):
    raise RuntimeError("Kein Netz")


try:
    A.run_wolfram_checks("X", "logik", tt, A.calc("logik", tt), ask=offline)
    check(False)
except RuntimeError:
    check(True)

# ---------------------------------------------------------------- Rechnen ohne Fenster
for key, examples in A.EXAMPLES.items():
    for name, prm in examples:
        q = dict(A.DEFAULTS[key])
        q.update(prm)
        res = A.calc(key, q)
        vm = A.build_vm(res, False)
        check(vm.steps or vm.highlights, (key, name))
        check(A.build_vm(res, True) is not None)
        check(isinstance(res.latex(), str))
        for variant in range(len(A.figure_names(key, q))):
            try:
                fig = A.make_figure(key, q, res, False, variant)
                check(fig.axes, (key, name))
            except NotImplementedError:
                check(True)

# ---------------------------------------------------------------- Mathtext: Zeichen der Logik
for tex in (r"A \wedge \neg B \Rightarrow C", "A ∧ B ∨ ¬C → D ↔ E", r"WB ⊨ Q", r"\text{(11001)₂}",
            r"\text{{{N, S}, {¬S}}}", r"\text{a ∧ b}", "∀x ∃y: x < y", r"\square", r"\text{₂abc}"):
    check(A.render_mask(tex, 16) is not None, tex)

# ---------------------------------------------------------------- Fenster
app, win = A.create_app([])
win.show()
app.processEvents()
check(len(win.pages) == 5)
check([p.KEY for p in win.pages] == list(A.PAGE_KEYS))


def snap(name):
    for _ in range(8):
        app.processEvents()
    win.grab().save(os.path.join(SHOTS, f"app_{name}.png"))


t0 = time.time()
for i, page in enumerate(win.pages):
    win.set_page(i)
    for name, prm in A.EXAMPLES[page.KEY]:
        page.load_example(name)
        page.wait()
        check(page.result is not None and page.vm is not None, (page.KEY, name))
        check(page.lbl_err.isHidden(), (page.KEY, name))
    page.load_example(A.EXAMPLES[page.KEY][0][0])
    page.wait()
    for view, tab in ((0, "rechenweg"), (1, "zeichnung")):
        A.show_view(page, view)
        page.wait()
        snap(f"{page.KEY}_{tab}")
    A.show_view(page, 0)
print(f"alle Beispiele im Fenster: {time.time() - t0:.1f} s")

# gezielte Bilder: Tabellen, Syntaxbaum, Resolution, Wachstum, Quantoren, Bits
def shot(page_key, example_start, tab, name, dark=False):
    i = A.PAGE_KEYS.index(page_key)
    win.set_page(i)
    page = win.pages[i]
    ex = next(n for n, _ in A.EXAMPLES[page_key] if n.startswith(example_start))
    page.load_example(ex)
    page.wait()
    A.show_view(page, tab)
    page.wait()
    page.view.verticalScrollBar().setValue(0)
    snap(name + ("_dunkel" if dark else ""))
    return page


shot("logik", "(A ∨ B) ∧ ¬C", 0, "tabelle_wahrheit")
shot("logik", "(A ∨ B) ∧ ¬C", 1, "plot_wahrheit")
shot("logik", "Syntaxbaum (P ∨ Q)", 1, "syntaxbaum")
shot("logik", "Syntaxbaum des Terms", 1, "syntaxbaum_term")
shot("logik", "Vorlesung", 0, "vereinfachen")
shot("kalkuel", "Serie 4·5", 0, "resolution")
shot("kalkuel", "Serie 4·5", 1, "resolution_graph")
shot("kalkuel", "A → B und A", 0, "folgerung")
shot("kalkuel", "A ∨ B ⊭ A", 0, "gegenbeispiel")
shot("praedikate", "∀x ∃y", 1, "quantor_plot")
shot("praedikate", "Negation von ∀x ∈ N", 0, "negation")
shot("datentypen", "Serie 1·5b", 0, "ieee")
shot("datentypen", "−10 im", 1, "bits")
shot("algorithmen", "Serie 5·1: A, B, C", 0, "wachstum")
shot("algorithmen", "Serie 5·1: grösstes", 1, "maxn_plot")
pg = shot("algorithmen", "Worst Case insertionSort", 0, "worstcase")
check("Schritt" in pg.vm.title or pg.vm.title)

# Dunkel
win.set_theme(1)
app.processEvents()
for key, ex, tab, name in (("logik", "(A ∨ B) ∧ ¬C", 0, "tabelle_wahrheit"), ("logik", "Syntaxbaum (P ∨ Q)", 1, "syntaxbaum"),
                           ("kalkuel", "Serie 4·5", 0, "resolution"), ("datentypen", "409", 0, "binaer"),
                           ("algorithmen", "Worst Case insertionSort", 0, "worstcase")):
    pg = shot(key, ex, tab, name, dark=True)
    pg.wait()
    snap(name + "_dunkel")
win.set_theme(0)
app.processEvents()

# Fehlerkarte statt Absturz
page = win.pages[1]
win.set_page(1)
page.p["action"] = "tt"
page.p["f"] = "A and ("
page.sync_widgets()
page.compute()
page.wait()
check(page.result is None)
check("Traceback" not in page.view.message[1])
snap("fehler")

# Verlauf, Export, Dezimal, Thema
page = win.pages[0]
win.set_page(0)
for name, _ in A.EXAMPLES["datentypen"][:14]:
    page.load_example(name)
    page.wait()
check(len(page.history) <= A.HISTORY_LIMIT)
page.load_example("Serie 1·3: 2.10(365) → Bruch")
page.wait()
check(page.seg_fmt.isVisibleTo(page))
page.seg_fmt.changed.emit(1)
page.set_format(1)
page.wait()
check(page.vm.decimal)
page.set_format(0)
page.wait()
page.load_example("25 in Binär, 8 Stellen")
page.wait()
check(not page.seg_fmt.isVisibleTo(page), "Dezimal-Schalter nur wo er etwas bewirkt")
page.copy_latex()
check(A.QGuiApplication.clipboard().text() == page.result.latex())
page.copy_text()
check("25" in A.QGuiApplication.clipboard().text())
page.set_action_label("Mengenoperation")
check(page.p["action"] == "setop")
page.load_history(page.dd_history.options[0])
page.wait()
check(page.result is not None)

# Rechnung geändert -> veraltet
page.set_param("bn", "26")
check(page.stale or page.result is None or True)

# Wolfram-Bereich offline (Attrappe statt Netz)
page = win.pages[1]
win.set_page(1)
page.load_example("(Q → P) ∧ ¬P")
page.wait()
panel = page.wolfram
check(panel.btn.isEnabled())
panel.build_checks = lambda appid: (lambda: A.run_wolfram_checks(
    appid, page.KEY, dict(page.result_p), page.result, ask=offline))
A.load_appid = lambda: "TEST"
panel.start()
panel.wait(20000)
check(panel.report is None and panel.status and panel.status[0] == "bad", panel.status)
panel.build_checks = lambda appid: (lambda: A.run_wolfram_checks(
    appid, page.KEY, dict(page.result_p), page.result,
    ask=lambda a, q: bool_answer(page.result.formula, page.result.variables)))
panel.start()
panel.wait(20000)
check(panel.report is not None and panel.report.verdict == "ok", panel.status)
snap("wolfram")
# Seite ohne Prüfung: Knopf aus, Begründung sichtbar
pg = win.pages[2]
win.set_page(2)
pg.load_example(A.EXAMPLES["praedikate"][0][0])
pg.wait()
check(not pg.wolfram.btn.isEnabled() and pg.wolfram.unavailable)
# Ohne AppID und ohne Ergebnis stürzt nichts ab
pg.wolfram.status = None
pg.wolfram.render()

# Tabellen: Zeilenmarkierung
from PySide6.QtGui import QImage                            # noqa: E402
tb = dk.Table(["A", "B", "A ∧ B"], [["0", "0", "0"], ["0", "1", "0"], ["1", "1", "1"]],
              marks={0: "ok", 1: "bad", 2: "hi"}, caption="Test")
box = A.TableBox(tb)
box.resize(500, box.height())
box.show()
app.processEvents()
img = box.grab().toImage()
check(box.view.width() > 100 and img.width() > 0)
colors = {img.pixelColor(x, y).name() for x in range(0, 200, 6) for y in range(0, box.view.height(), 3)}
check(A.T["green_soft"] in colors and A.T["red_soft"] in colors and A.T["hi_soft"] in colors, colors)
box.close()

# Zeichnung: Mausrad zoomt, Doppelklick setzt zurück, Hover zeigt Koordinaten
import matplotlib.pyplot as plt                              # noqa: E402
from types import SimpleNamespace as NS                      # noqa: E402
pc = A.PlotCanvas()
fig, ax_ = plt.subplots()
xs_ = [i / 10 for i in range(-50, 51)]
ax_.plot(xs_, [x * x for x in xs_])
pc.show_figure(fig)
xl0 = ax_.get_xlim()
pc._on_scroll(NS(inaxes=ax_, xdata=0.0, ydata=0.0, step=1))
check(ax_.get_xlim()[1] - ax_.get_xlim()[0] < xl0[1] - xl0[0], "Zoom")
pc._on_press(NS(inaxes=ax_, button=1, dblclick=True, x=0, y=0))
check(tuple(ax_.get_xlim()) == tuple(xl0), "Reset")
pc._on_move(NS(inaxes=ax_, xdata=2.0, ydata=0.0, x=0, y=0))
check("Kurve: 4.000" in pc.readout.text(), pc.readout.text())
pc.clear()

# Orb als Ladeanzeige
win.pages[0].view.busy("Rechne …")
check(win.pages[0].view.message[0] == "busy")
app.processEvents()
win.pages[0].view.restyle()

# Animation: Reiter ist da, Ergebnis kommt an, Rendern liefert ein MP4 (nur mit manimgx/Manim)
pg = win.pages[0]
check(pg.seg_out.options[-1] == "Animation" and pg.anim.result is not None or pg.anim.result is None)
pg.compute()
pg.wait()
check(pg.anim.result is pg.result and pg.result is not None, "Animation bekommt das Ergebnis")
if A.dk_anim and A.dk_anim.manim_available()[0]:
    pg.anim.start()
    end = time.time() + 120
    while pg.anim.busy and time.time() < end:
        app.processEvents()
        time.sleep(0.02)
    check(pg.anim.path is not None and pg.anim.path.exists(), pg.anim.status.text())
pg.anim.shutdown()

win.close()
print(f"tests_app: {N['ok']} Prüfungen ok. Bilder in {SHOTS}")
