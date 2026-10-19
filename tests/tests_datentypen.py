#!/usr/bin/env python3
"""Tests für dmatkit.rechnen.datentypen (Semesterwoche 1): python3 tests/tests_datentypen.py"""
import os
import random
import struct
import sys
import math
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import sympy as sp
import numpy as np
from dmatkit.rechnen import datentypen as dt
from dmatkit.rechnen.datentypen import *
from dmatkit.rechnen.core import parse
from dmatkit.rechnen.render import Result

N = 0


def check(cond, msg=""):
    global N
    N += 1
    assert cond, msg


def raises(exc, fn, *a, text=None, **kw):
    global N
    N += 1
    try:
        fn(*a, **kw)
    except exc as e:
        if text is not None:
            assert text in str(e), (text, str(e))
        return
    raise AssertionError(f"{fn.__name__}{a} sollte {exc.__name__} auslösen")


def smoke(r):
    """Jedes Ergebnis lässt sich ausgeben (Text, HTML, LaTeX)."""
    check(isinstance(r, Result))
    check(isinstance(repr(r), str) and len(repr(r)) > 0)
    check("<div" in r._repr_html_())
    check(isinstance(r.latex(), str))
    check(r.title and r.summary is not None)


rnd = random.Random(20260922)

# ==========================================================================
# 1. Boolesche Ausdrücke (Serie 1, Aufgabe 1)
# ==========================================================================

# Aufgabe 1a (Mitschrift und aufbereitete Fassung)
r = eval_bool("False or True and False and True or False")
check(r.value is False and r.checked)
check(r.lines[0] == "False or True and False and True or False")
check(r.lines[1] == "F or ==T and F== and T or F", r.lines[1])
check(r.lines[2] == "F or ==F and T== or F", r.lines[2])
check(r.lines[3] == "==F or F== or F")
check(r.lines[4] == "==F or F==")
check(r.lines[-1] == "F" and len(r.lines) == 6)
check([x["op"] for x in r.reductions] == ["and", "and", "or", "or"])
smoke(r)

# 1b (Übungsblatt) und die Fassung der Mitschrift
r = eval_bool("False and True or False and not True or False")
check(r.value is False)
check(r.lines[1] == "F and T or F and ==not T== or F", r.lines[1])
check(r.reductions[0]["op"] == "not" and r.reductions[0]["sub"] == "not T")
r = eval_bool("False and True and not True or False")
check(r.value is False)

# 1c (in der Mitschrift unvollständig): richtig zu Ende gerechnet
r = eval_bool("not False or not not True and False or not True")
check(r.value is True)
check(r.lines[1] == "==not F== or not not T and F or not T", r.lines[1])
check(r.lines[-1] == "T")
check("not ==not T==" in r.lines[2])          # innerste not zuerst

# 1d mit Klammern
r = eval_bool("not(not True or not (False and not True)) or False")
check(r.value is False)
check(r.lines[1] == "not(not T or not(F and ==not T==)) or F", r.lines[1])
check(r.lines[2] == "not(not T or not(==F and F==)) or F", r.lines[2])
check(r.lines[-2] == "==F or F==" and r.lines[-1] == "F")
# 1e
r = eval_bool("not((False or not True) or True) and not(False or True)")
check(r.value is False)
check(r.lines[1] == "not((F or ==not T==) or T) and not(F or T)", r.lines[1])

# Lerngerüst Serie 1 (a)
check(eval_bool("not True or not (not False and True)").value is False)
check(eval_bool("False and True or False or not True and not False").value is False)

# Variablen, Abkürzung, Kurzschreibweisen
check(eval_bool("A and not B", A=True, B=False).value is True)
r = eval_bool("A or B and C", A=False, B=True, C=False)
check(r.value is False and r.steps[1].title == "Werte einsetzen")
r = eval_bool("True and False", short=False)
check(r.lines[1] == "==True and False==")
check(eval_bool("T and not F").value is True)
check(eval_bool("wahr oder falsch").value is True)
check(eval_bool("¬(True ∧ False) ∨ False").value is True)
check(eval_bool("True").value is True and eval_bool("True").reductions == [])
check(eval_bool("(True)").value is True and eval_bool("(True)").reductions[0]["op"] == "paren")
check(eval_bool("((False))").value is False)

# Zufallsvergleich gegen Python-eval (nur auf selbst erzeugten festen Strings) und core.Formula
def rand_bool(depth):
    if depth == 0 or rnd.random() < 0.25:
        return rnd.choice(["True", "False"])
    k = rnd.random()
    if k < 0.2:
        return "not " + rand_bool(depth - 1)
    if k < 0.4:
        return "(" + rand_bool(depth - 1) + ")"
    op = rnd.choice(["and", "or"])
    return f"{rand_bool(depth - 1)} {op} {rand_bool(depth - 1)}"


for _ in range(300):
    s = rand_bool(5)
    r = eval_bool(s)
    check(r.value == eval(s), s)                      # eval nur im Test, auf eigenen Strings
    check(r.value == parse(s).eval({}), s)
    check(r.lines[-1] == ("T" if r.value else "F"))
    check(all(l.count("==") == 2 for l in r.lines[1:-1]), s)   # genau ein markierter Teil
    strip = lambda t: "".join(t.replace("(", "").replace(")", "").split())
    check(strip(r.lines[0]) == strip(s))

# Fehler
raises(ValueError, eval_bool, "", text="Leer")
raises(ValueError, eval_bool, "True and", text="endet zu früh")
raises(ValueError, eval_bool, "(True and False", text="Klammer")
raises(ValueError, eval_bool, "True and False)", text="Unerwartetes")
raises(ValueError, eval_bool, "A and B", text="Unbekannter Name")
raises(ValueError, eval_bool, "True xor False", text="Aussagenlogik")
raises(ValueError, eval_bool, "1 + 2", text="boolesch")
raises(ValueError, eval_bool, "True $ False", text="Unerwartetes Zeichen")
raises(ValueError, eval_bool, "True + False")
raises(ValueError, eval_bool, "x" * 400, text="zu lang")
raises(ValueError, eval_bool, "(" * 100 + "True" + ")" * 100, text="verschachtelt")
raises(ValueError, eval_bool, 5)
raises(ValueError, eval_bool, "A and B", A=True, B=3)
# tiefe, aber erlaubte Ausdrücke
check(eval_bool("not " * 50 + "True").value is True)
check(eval_bool("True" + " and True" * 30).value is True)

# ==========================================================================
# 2. integer: + - * // % (Aufgabe 2)
# ==========================================================================

r = int_op(10, "//", 3)
check(r.value == 3 and r.quotient == 3 and r.remainder == 1 and r.check)
r = int_op(10, "%", 3)
check(r.value == 1 and r.equation == "10 = 3 · 3 + 1", r.equation)
check(int_op(10, "%", 7).value == 3 and int_op(10, "%", 7).equation == "10 = 1 · 7 + 3")
check(int_op(10, "%", 5).value == 0 and int_op(10, "%", 5).equation == "10 = 2 · 5 + 0")
check(int_op(7, "+", 5).value == 12 and int_op(7, "-", 9).value == -2 and int_op(-4, "*", 6).value == -24)
check(any("weil 3 · 3 + 1 = 10" in s.detail for s in int_op(10, "%", 3).steps))
smoke(int_op(10, "%", 3))
smoke(int_op(3, "+", 4))
# Vorzeichenregeln wie Python, vollständig geprüft
for a in range(-25, 26):
    for b in list(range(-9, 0)) + list(range(1, 10)):
        q, rem = int_op(a, "//", b), int_op(a, "%", b)
        check(q.value == a // b and rem.value == a % b)
        check(a == q.value * b + rem.value and rem.check and q.check)
        check(abs(rem.value) < abs(b) and (rem.value == 0 or (rem.value > 0) == (b > 0)))
r = int_op(-7, "//", 2)
check(r.value == -4 and r.remainder == 1 and any("Java" in s.note for s in r.steps))
check(int_op(7, "//", -2).value == -4 and int_op(7, "%", -2).value == -1)
r = divmod_steps(25, 2)
check(r.value == (12, 1) and r.quotient == 12 and r.remainder == 1 and r.equation == "25 = 12 · 2 + 1")
check(int_op(25, "divmod", 2).value == (12, 1))
smoke(r)
raises(ValueError, int_op, 5, "//", 0, text="Division durch 0")
raises(ValueError, int_op, 5, "%", 0, text="Division durch 0")
raises(ValueError, divmod_steps, 5, 0, text="Division durch 0")
raises(ValueError, int_op, 1.5, "+", 2, text="ganze Zahl")
raises(ValueError, int_op, True, "+", 2)
raises(ValueError, int_op, 10 ** 19, "+", 2, text="zu gross")
raises(ValueError, int_op, 5, "**", 2, text="Unbekannter Operator")
check(int_op("12", "+", " -3").value == 9)
check(int_op(10 ** 18, "*", 10 ** 18).value == 10 ** 36)

# Ausdrücke
r = eval_int("10 // 3 + 10 % 3 * 2")
check(r.value == 5 and r.lines[1] == "==10 // 3== + 10 % 3 * 2", r.lines)
check(eval_int("(-7) // 2").value == -4 and eval_int("-7 // 2").value == -4)
check(eval_int("-(3 + 4) * 2").value == -14 and eval_int("2 - -3").value == 5)
check(eval_int("(2 + 3) * (4 - 1)").value == 15)
check(eval_int("a * b + c", a=3, b=4, c=-2).value == 10)
for _ in range(400):
    parts = []
    for i in range(rnd.randint(1, 5)):
        parts.append(str(rnd.randint(0, 30)))
        parts.append(rnd.choice(["+", "-", "*", "//", "%"]))
    s = " ".join(parts[:-1])
    if rnd.random() < 0.4 and len(parts) > 3:
        s = "(" + " ".join(parts[:3]) + ") " + parts[3] + " " + " ".join(parts[4:-1]) if len(parts) > 5 else s
    try:
        want = eval(s)
    except ZeroDivisionError:
        raises(ValueError, eval_int, s, text="durch 0")
        continue
    r = eval_int(s)
    check(r.value == want, (s, r.value, want))
    check(r.lines[-1] == str(want))
raises(ValueError, eval_int, "3 / 2", text="Ganzzahldivision")
raises(ValueError, eval_int, "1.5 + 2", text="kein integer")
raises(ValueError, eval_int, "True + 1")
raises(ValueError, eval_int, "5 // (3 - 3)", text="durch 0")
raises(ValueError, eval_int, "2 +")
raises(ValueError, eval_int, "99999999999999999999 + 1", text="zu gross")
raises(ValueError, eval_int, "9" * 18 + " * " + "9" * 18 + " * " + "9" * 18 + " * " + "9" * 18 + " * " + "9" * 18 + " * " + "9" * 18 + " * " + "9" * 18)
smoke(r)

# ==========================================================================
# 3. integer in Bits
# ==========================================================================

r = to_binary(25)
check(r.text == "11001" and r.number == 25 and r.digits == [1, 1, 0, 0, 1] and r.weights == [16, 8, 4, 2, 1])
check([(a, b, c) for a, b, c in r.rows] == [(25, 12, 1), (12, 6, 0), (6, 3, 0), (3, 1, 1), (1, 0, 1)])
tab = [s.table for s in r.steps if s.table is not None]
check(any(t.header == ["16", "8", "4", "2", "1", "Zahl"] and t.rows == [["1", "1", "0", "0", "1", "25"]] for t in tab))
check(any(t.rows[0] == ["25 = 12 · 2 + 1", "1"] for t in tab))
smoke(r)
check(to_binary(409).text == "110011001")
check([row[2] for row in to_binary(409).rows] == [1, 0, 0, 1, 1, 0, 0, 1, 1])
check(to_binary(123).text == "1111011" and to_binary(741).text == "1011100101")   # Lerngerüst (b)
check(to_binary(0).text == "0" and to_binary(1).text == "1")
check(to_binary(25, width=8).text == "00011001" and to_binary(25, width=8).bits == "00011001")
check(to_binary(409, method="division").steps[1].table is not None and
      all(s.table is None or s.table.header[0] != "256" for s in to_binary(409, method="division").steps))
check(any(s.table is not None and s.table.header[0] == "256" for s in to_binary(409, method="table").steps))
check(dec_to_bin(25) == "11001" and dec_to_bin(0) == "0")
for _ in range(500):
    n = rnd.randint(0, 10 ** rnd.randint(1, 18))
    check(to_binary(n).text == bin(n)[2:] and from_binary(to_binary(n).text).number == n)
    check(to_base(n, 16).text == format(n, "X") and to_base(n, 8).text == format(n, "o"))
check(to_base(255, 16).text == "FF" and from_base("FF", 16).number == 255 and from_base("ff", 16).number == 255)
check(from_binary("11001").number == 25 and from_binary("0b11001").number == 25
      and from_binary("1 1001").number == 25 and from_binary("(11001)₂").number == 25)
check(from_base("zz", 36).number == 35 * 36 + 35)
smoke(from_binary("11001"))
raises(ValueError, to_binary, -3, text="Zweierkomplement")
raises(ValueError, to_binary, 25, width=3, text="reicht nicht")
raises(ValueError, to_binary, 10 ** 19, text="zu gross")
raises(ValueError, to_binary, 2.5)
raises(ValueError, to_base, 5, 1)
raises(ValueError, from_binary, "1021", text="keine Ziffer")
raises(ValueError, from_binary, "")
raises(ValueError, from_binary, "1" * 65, text="zu lang")
raises(ValueError, from_base, "G", 16, text="keine Ziffer")

# ==========================================================================
# 4. Kommazahlen: Periode -> Bruch (Aufgabe 3)
# ==========================================================================

r = repeating_to_fraction("2.10(365)")
check(r.value == sp.Rational(42031, 19980) and r.unreduced == (210155, 99900))
check(r.numerator == 42031 and r.denominator == 19980 and r.pre == "10" and r.period == "365")
check(any("100000" in str(i) for s in r.steps for i in s.items))
check(any("99900" in str(s.detail) or "99900" in str([x[1] for x in s.items]) for s in r.steps))
smoke(r)
r = repeating_to_fraction("14.2(51)")
check(r.unreduced == (14109, 990) and r.value == sp.Rational(14109, 990) == sp.Rational(4703, 330))
r = repeating_to_fraction("14.05")
check(r.value == sp.Rational(281, 20) and r.unreduced == (1405, 100))
check(repeating_to_fraction("20.1(445)").value == sp.Rational(201445 - 201, 9990))            # Lerngerüst (c)
check(repeating_to_fraction("0.173(15)").value == sp.Rational(17315 - 173, 99000))
check(repeating_to_fraction("20.1(445)").value == sp.Rational(201244, 9990) == sp.Rational(100622, 4995))
check(repeating_to_fraction("0.(3)").value == sp.Rational(1, 3) and repeating_to_fraction("0.(9)").value == 1)
check(repeating_to_fraction("0.(9)").warnings)
check(repeating_to_fraction("-1.5(3)").value == sp.Rational(-23, 15))
check(repeating_to_fraction("7").value == 7 and repeating_to_fraction("0,25").value == sp.Rational(1, 4))
check(repeating_to_fraction("0.0(1001)", 2).value == sp.Rational(3, 10))
check(from_binary("0.0(1001)").value == sp.Rational(3, 10) and from_binary("101.011").value == sp.Rational(5 * 8 + 3, 8))
# Gegenprobe über lange Entwicklung
for _ in range(200):
    ip = str(rnd.randint(0, 99))
    pre = "".join(rnd.choice("0123456789") for _ in range(rnd.randint(0, 4)))
    per = "".join(rnd.choice("0123456789") for _ in range(rnd.randint(1, 5)))
    txt = f"{ip}.{pre}({per})"
    v = repeating_to_fraction(txt).value
    s = f"{ip}.{pre}{per * 30}"
    check(abs(sp.Rational(s) - v) < sp.Rational(1, 10 ** (len(pre) + 28 * len(per))), txt)
raises(ValueError, repeating_to_fraction, "abc")
raises(ValueError, repeating_to_fraction, "1.2(3a)", text="keine Ziffer")
raises(ValueError, repeating_to_fraction, "12(3)", text="Komma")
raises(ValueError, repeating_to_fraction, "1." + "3" * 60, text="Zu viele")
raises(ValueError, repeating_to_fraction, "1.5", 99)

# ==========================================================================
# 5. Kommazahlen in Binär (Aufgabe 4, Lerngerüst d, Aufgabe 7)
# ==========================================================================

r = frac_to_binary("0.625")
check(r.text == "0.101" and r.pre == "101" and r.period == "" and r.complete and r.value == sp.Rational(5, 8))
check([(row[0], row[1]) for row in r.rows] == [(Fraction(5, 8), 1), (Fraction(1, 4), 0), (Fraction(1, 2), 1)])
smoke(r)
r = frac_to_binary("0.3")
check(r.pre == "0" and r.period == "1001" and r.text == "0.0(1001)" and r.expand(12) == "010011001100")
check(len(r.rows) == 5 and r.rows[-1][2] == Fraction(3, 5))
check(frac_to_binary("1/6").text == "0.0(01)" and frac_to_binary("1/6").pre == "0")
check(frac_to_binary("0.1").text == "0.0(0011)")
check(frac_to_binary("35.2").text == "100011.(0011)" and frac_to_binary(35.2).text == "100011.(0011)")
check(frac_to_binary("-0.5").text == "-0.1" and frac_to_binary("-0.5").sign == -1)
check(frac_to_binary(5).text == "101" and frac_to_binary("0").text == "0")
check(frac_to_binary("2.10(365)").value == sp.Rational(42031, 19980))
check(frac_to_binary("0.(3)").text == "0.(01)")
check(frac_to_binary(sp.Rational(1, 3)).text == "0.(01)" and frac_to_binary(Fraction(7, 8)).text == "0.111")
# 1/997: Periode länger als 64 -> abgebrochen, mit Warnung
r = frac_to_binary("1/997")
check(not r.complete and r.warnings and r.text.endswith("…"))
check(frac_to_binary("1/997", max_bits=2000).complete)
check(frac_to_binary("1/7").text == "0.(001)")
for _ in range(300):
    q = rnd.randint(1, 60)
    p = rnd.randint(0, 3 * q)
    f = Fraction(p, q)
    r = frac_to_binary(f"{p}/{q}", max_bits=200)
    check(r.complete)
    n = 40
    want = bin((f - int(f)).numerator * 2 ** n // (f - int(f)).denominator)[2:].rjust(n, "0") if f != int(f) else "0" * n
    check(r.expand(n) == want, (p, q))
    check(from_binary(r.text).value == sp.Rational(p, q) if "." in r.text else
          from_binary(r.text).number == p // q)
raises(ValueError, frac_to_binary, "abc")
raises(ValueError, frac_to_binary, "1/0")
raises(ValueError, frac_to_binary, float("inf"))
raises(ValueError, frac_to_binary, "0.5", max_bits=0)
raises(ValueError, frac_to_binary, "0.5", max_bits=10 ** 6)
raises(ValueError, frac_to_binary, True)

# Wurzel in Binär (Aufgabe 7c)
r = sqrt_binary(2, 10)
check(r.text == "1.0110101000" and r.digits == 10 and not r.exact)
check(r.value == sp.Rational(math.isqrt(2 * 4 ** 10), 2 ** 10))
check(sqrt_binary("0.25", 4).text == "0.1000" and sqrt_binary("0.25", 4).exact)
check(sqrt_binary(16, 0).text == "100" and sqrt_binary(0, 3).text == "0.000")
for x in (3, 5, 10, 12345, Fraction(1, 3), Fraction(7, 2)):
    for n in (1, 7, 30):
        r = sqrt_binary(x, n)
        fx = Fraction(x)
        v = r.value
        v = Fraction(int(v.p), int(v.q))
        check(v * v <= fx < (v + Fraction(1, 2 ** n)) ** 2)
        check(from_binary(r.text).value == r.value or r.text.isdigit())
smoke(sqrt_binary(2, 5))
raises(ValueError, sqrt_binary, -1, text="negativ")
raises(ValueError, sqrt_binary, 2, 5000, text="zu gross")
raises(ValueError, sqrt_binary, 2, -1)

# ==========================================================================
# 6. IEEE 754 (Aufgabe 5)
# ==========================================================================

def struct_bits(x, bits):
    fmt_ = ">f" if bits == 32 else ">d"
    return "".join(f"{b:08b}" for b in struct.pack(fmt_, x))


# Aufgabe 5a, 32 Bit wie in der Mitschrift
r = ieee754("0.00(10)", 32, base=2)
check(r.sign == "0" and r.exponent_bits == "01111100" and r.mantissa_bits == "01010101010101010101011")
check(r.exponent == -3 and r.exponent_field == 124 and r.rounded and r.kind == "normal")
check(r.pattern == "0" + "01111100" + "01010101010101010101011" and r.hex == "3E2AAAAB")
check(r.pattern == struct_bits(1 / 6, 32))
check(r.exact == sp.Rational(1, 6) and r.error == r.stored - r.exact and r.error > 0)
smoke(r)
r = ieee754("0.00(10)", 64, base=2)
check(r.exponent_bits == "01111111100" and r.exponent_field == 1020)
check(r.mantissa_bits == "0101" * 13 and r.pattern == struct_bits(1 / 6, 64))
# Aufgabe 5b (zuhause): 35.2
r = ieee754("35.2", 32)
check(r.pattern == struct_bits(35.2, 32) and r.exponent == 5 and r.exponent_bits == "10000100")
check(r.mantissa_bits == "00011001100110011001101")
check(ieee754("35.2", 64).pattern == struct_bits(35.2, 64) and ieee754(35.2, 64).exponent_bits == "10000000100")
# Lerngerüst (e): 0.0(110)
r = ieee754("0.0(110)", 32, base=2)
x = repeating_to_fraction("0.0(110)", 2).value
check(x == sp.Rational(3, 7), x)
check(r.exact == x and r.pattern == struct_bits(3 / 7, 32), (r.pattern, struct_bits(3 / 7, 32)))
check(r.exponent == -2 and r.mantissa_bits == "10110110110110110110111")
# viele Werte gegen struct
vals = [0.1, 0.5, 1.0, 2.0, 3.14159, 100.0, 1e10, 1e-10, 123456.789, 0.3, 1 / 3, 255.5, 0.7, 1e30, 3.4e38,
        1.5e-45, 1e-40, 5e-324, 2.2250738585072014e-308, 1.7976931348623157e308, 16777217.0]
for v in vals:
    for sgn in (1, -1):
        for bits in (32, 64):
            fv = sgn * v
            try:
                want = struct_bits(fv, bits)
            except OverflowError:
                continue
            r = ieee754(repr(fv), bits)
            check(r.pattern == want, (fv, bits, r.pattern, want))
for _ in range(400):
    v = rnd.uniform(-1, 1) * 10 ** rnd.randint(-8, 8)
    for bits in (32, 64):
        r = ieee754(repr(v), bits)
        if bits == 64:
            check(r.pattern == struct_bits(v, 64))
        else:
            # Rundung dezimal → float32 (ohne Doppelrundung über double): Nachbarn prüfen
            s = r.stored
            fs = Fraction(int(s.p), int(s.q))
            ulp = Fraction(2) ** (r.exponent - 23)
            check(abs(fs - Fraction(repr(v))) <= ulp / 2)
            check(abs(float(s) - float(struct.unpack('>f', struct.pack('>f', v))[0])) <= float(ulp))
# Sonderfälle
r = ieee754(0, 32)
check(r.pattern == "0" * 32 and r.kind == "zero")
check(ieee754("-1", 32).pattern == struct_bits(-1.0, 32) and ieee754("1", 64).pattern == struct_bits(1.0, 64))
r = ieee754("1e-40", 32)
check(r.kind == "subnormal" and r.pattern == struct_bits(1e-40, 32) and r.warnings)
r = ieee754(Fraction(10) ** 39, 32)
check(r.kind == "inf" and r.exponent_bits == "1" * 8 and r.mantissa_bits == "0" * 23 and r.warnings)
check(ieee754("16777217", 32).pattern == struct_bits(16777216.0, 32))        # Gleichstand → gerade
# Dekodieren
for _ in range(300):
    bits = rnd.choice((32, 64))
    pat = "".join(rnd.choice("01") for _ in range(bits))
    r = ieee754_decode(pat)
    fv = struct.unpack(">f" if bits == 32 else ">d", int(pat, 2).to_bytes(bits // 8, "big"))[0]
    if math.isnan(fv):
        check(r.kind == "nan")
    elif math.isinf(fv):
        check(r.kind == "inf" and r.value == (sp.oo if fv > 0 else -sp.oo))
    else:
        check(r.value == sp.Rational(Fraction(fv).numerator, Fraction(fv).denominator), pat)
        # Rückweg: zurückcodieren ergibt dieselbe Bitfolge (ausser -0)
        if fv != 0:
            check(ieee754(Fraction(fv), bits).pattern == pat)
check(ieee754_decode("0 01111100 01010101010101010101011").value == sp.Rational(11184811, 67108864))
check(ieee754_decode("0" + "01111100" + "01010101010101010101011", 32).kind == "normal")
check(ieee754_decode("1" + "0" * 31).value == 0 and ieee754_decode("0" * 31 + "1").kind == "subnormal")
smoke(ieee754_decode("0" * 31 + "1"))
raises(ValueError, ieee754, "0.5", 16, text="32 Bit")
raises(ValueError, ieee754, 0.5, 32, 2, text="Text")
raises(ValueError, ieee754, "abc")
raises(ValueError, ieee754, "0.12", 32, 2, text="keine Ziffer")
raises(ValueError, ieee754_decode, "0101")
raises(ValueError, ieee754_decode, "x" * 32)

# float-Vergleich
r = float_compare("0.7 + 0.1 == 0.9 - 0.1")
check(r.python_equal is False and r.exact_equal is True and r.left == 0.7 + 0.1 and r.right == 0.9 - 0.1)
check(r.left == 0.7999999999999999 and r.right == 0.8 and r.left_exact == r.right_exact == sp.Rational(4, 5))
check(r.difference > 0 and r.difference < 1e-15)
smoke(r)
r = float_compare("0.5 + 0.25 == 0.75")
check(r.python_equal is True and r.exact_equal is True)
r = float_compare("0.1 * 3")
check(r.compared is False and r.left == 0.1 * 3 and r.left_exact == sp.Rational(3, 10))
check(float_compare("1/3 + 1/3 + 1/3 == 1").exact_equal is True)
raises(ValueError, float_compare, "1 / 0", text="durch 0")
raises(ValueError, float_compare, "1 == 1 == 1")
raises(ValueError, float_compare, "True + 1")

# ==========================================================================
# 7. Zweierkomplement, Wertebereich, Überlauf
# ==========================================================================

check(twos_complement(-10, 8).pattern == "11110110" and twos_complement(5, 8).pattern == "00000101")
check(twos_complement(-10, 8).ones == "11110101")
check(twos_complement(-1, 8).pattern == "11111111" and twos_complement(-128, 8).pattern == "10000000")
check(twos_complement(127, 8).pattern == "01111111" and twos_complement(0, 4).pattern == "0000")
smoke(twos_complement(-10, 8))
smoke(twos_complement(10, 8))
for b in (2, 3, 4, 8):
    for n in range(-2 ** (b - 1), 2 ** (b - 1)):
        r = twos_complement(n, b)
        check(r.pattern == format(n & (2 ** b - 1), f"0{b}b") and from_twos_complement(r.pattern).value == n)
for b in (16, 32, 64):
    for n in (0, 1, -1, 2 ** (b - 1) - 1, -2 ** (b - 1), rnd.randint(-10 ** 18, 10 ** 18)):
        if -2 ** (b - 1) <= n < 2 ** (b - 1):
            check(twos_complement(n, b).pattern == format(n & (2 ** b - 1), f"0{b}b"))
            check(from_twos_complement(twos_complement(n, b).pattern).value == n)
check(from_twos_complement("11110110").value == -10 and from_twos_complement("0111").value == 7)
check(from_twos_complement("1 000").value == -8)
smoke(from_twos_complement("11110110"))
raises(ValueError, twos_complement, -200, 8, text="ausserhalb")
raises(ValueError, twos_complement, 128, 8, text="ausserhalb")
raises(ValueError, twos_complement, 5, 65, text="Bitbreite")
raises(ValueError, twos_complement, 5, 1)
raises(ValueError, from_twos_complement, "12")
raises(ValueError, from_twos_complement, "1" * 65)
r = int_range(8)
check((r.min, r.max, r.count) == (-128, 127, 256) and int_range(8, False).max == 255 and int_range(8, False).min == 0)
check(int_range(32).max == 2147483647 and int_range(64).min == -2 ** 63 and int_range(1).min == -1)
smoke(r)
raises(ValueError, int_range, 65)
raises(ValueError, int_range, 0)
r = overflow_check(100, "+", 100, 8)
check(r.exact == 200 and r.wrapped == -56 and r.overflow and r.pattern == "11001000" and r.warnings)
check(not overflow_check(50, "+", 50, 8).overflow and overflow_check(50, "+", 50, 8).wrapped == 100)
check(overflow_check(255, "+", 1, 8, False).wrapped == 0 and overflow_check(255, "+", 1, 8, False).overflow)
check(overflow_check(0, "-", 1, 8, False).wrapped == 255)
smoke(r)
for b in (3, 4):
    lo, hi = -2 ** (b - 1), 2 ** (b - 1) - 1
    for a in range(lo, hi + 1):
        for c in range(lo, hi + 1):
            for op in "+-*":
                r = overflow_check(a, op, c, b)
                ex = eval(f"{a}{op}{c}")
                w = ((ex - lo) % 2 ** b) + lo
                check(r.exact == ex and r.wrapped == w and r.overflow == (not lo <= ex <= hi))
raises(ValueError, overflow_check, 200, "+", 1, 8, text="Wertebereich")
raises(ValueError, overflow_check, 1, "/", 1, 8)

# ==========================================================================
# 8. Mengen
# ==========================================================================

check(set_op("{a,b,c}", "∩", "{b,c,d}").result == frozenset("bc"))
check(set_op("{a,b,c}", "∪", "{b,c,d}").result == frozenset("abcd"))
check(set_op("{a,b,c}", "\\", "{b,c,d}").result == frozenset("a"))
check(set_op("{a,b}", "∩", "{c,d}").result == frozenset() and set_op("{a,b}", "∩", "{c,d}").summary.endswith("∅"))
check(set_op("{a,b}", "komplement", universe="{a,b,c,d,e}").result == frozenset("cde"))
check(set_op("abc", "cap", "bcd").result == frozenset("bc"))
check(set_op({1, 2, 3}, "cup", [3, 4]).result == frozenset({1, 2, 3, 4}))
check(set_op("{1,2,3}", "-", "{2}").result == frozenset({1, 3}))
for _ in range(100):
    A = frozenset(rnd.sample(range(12), rnd.randint(0, 8)))
    B = frozenset(rnd.sample(range(12), rnd.randint(0, 8)))
    U = frozenset(range(12))
    check(set_op(A, "∩", B).result == (A & B) and set_op(A, "∪", B).result == (A | B))
    check(set_op(A, "\\", B).result == (A - B) and set_op(A, "c", universe=U).result == (U - A))
smoke(set_op("{a,b,c}", "∩", "{b,c,d}"))
smoke(set_op("{a,b}", "c", universe="{a,b,c}"))
raises(ValueError, set_op, "{a}", "∩", text="zwei Mengen")
raises(ValueError, set_op, "{a}", "c", text="Grundmenge")
raises(ValueError, set_op, "{a,z}", "c", universe="{a,b}", text="Teilmenge")
raises(ValueError, set_op, "{a}", "xor", "{b}", text="Unbekannte")
raises(ValueError, set_op, 5, "∪", "{a}")
raises(ValueError, set_op, list(range(300)), "∪", "{1}", text="zu viele")
# Relationen
check(set_relation("{a,b}", "⊂", "{a,b,c}").value is True)
check(set_relation("{a,d}", "⊂", "{a,b,c}").value is False and set_relation("{a,d}", "⊂", "{a,b,c}").witness == "d")
check(set_relation("{a,b}", "⊂", "{a,b}").value is True and set_relation("{a,b}", "⊊", "{a,b}").value is False)
check(set_relation("", "⊂", "{a}").value is True and set_relation("{}", "⊂", "{}").value is True)
check(set_relation("b", "∈", "{a,b,c}").value is True and set_relation("z", "∈", "{a,b,c}").value is False)
check(set_relation(2, "∈", {1, 2}).value is True and set_relation("2", "∉", "{1,2}").value is False)
check(set_relation("{a,b}", "=", "{b,a,a}").value is True and set_relation("{a}", "=", "{b}").value is False)
smoke(set_relation("{a,d}", "⊂", "{a,b,c}"))
smoke(set_relation("b", "∈", "{a,b,c}"))
smoke(set_relation("{a}", "=", "{a}"))
raises(ValueError, set_relation, "{a}", "~", "{a}", text="Unbekannte Relation")
# Potenzmenge (Aufgabe 6)
r = power_set("{a,b,c,d}")
check(r.count == 16 and len(set(r.subsets)) == 16 and frozenset() in r.subsets and frozenset("abcd") in r.subsets)
check(r.bitstrings[0] == "0000" and r.bitstrings[-1] == "1111" and sorted(map(len, r.subsets)) == list(map(len, r.subsets)))
check(power_set("{1,2}").count == 4 and power_set("").count == 1)
check(all(set_bits == frozenset(e for e, bch in zip(r.elements, b) if bch == "1")
          for set_bits, b in zip(r.subsets, r.bitstrings)))
smoke(r)
raises(ValueError, power_set, list(range(11)), text="höchstens")
r = cartesian_product("{1,2,3}", "{a,b}")
check(r.size == 6 and (1, "a") in r.pairs and ("a", 1) not in r.pairs)
check(set(cartesian_product("{a,b}", "{1,2,3}").pairs) != set(r.pairs))
smoke(r)
raises(ValueError, cartesian_product, list(range(100)), list(range(100)), text="zu viele")
check(is_partition(["{a,b}", "{c,d,e}"], "{a,b,c,d,e}").value is True)
check(is_partition(["{a,b}", "{b,c,d,e}"], "{a,b,c,d,e}").value is False)
check(is_partition(["{a,b}", "{c,d}"], "{a,b,c,d,e}").value is False)
check(is_partition(["{a,b}", "{}", "{c}"], "{a,b,c}").value is False)
smoke(is_partition(["{a,b}", "{c,d,e}"], "{a,b,c,d,e}"))
raises(ValueError, is_partition, [], "{a}")
check(set_bitstring("{2,a}", "{1,2,a,*}").bits in ("0110", "0101", "0011", "1010") and
      set_bitstring("{}", "{1,2}").bits == "00" and set_bitstring("{1,2}", "{1,2}").bits == "11")
smoke(set_bitstring("{1}", "{1,2}"))
raises(ValueError, set_bitstring, "{9}", "{1,2}", text="gehört nicht")

# ==========================================================================
# 9. Listen, Matrizen, Broadcasting
# ==========================================================================

check(list_index([5, 3, 8, 8, 1], 2).value == 3 and list_index([5, 3, 8, 8, 1], 2).python_index == 1)
check(list_index([5, 3, 8, 8, 1], 2, start=0).value == 8)
raises(ValueError, list_index, [1, 2], 3, text="ausserhalb")
raises(ValueError, list_index, [1, 2], 0, text="ausserhalb")
raises(ValueError, list_index, [], 1)
raises(ValueError, list_index, [1], 1, 2)
A = [[5, 3, 8, 8, 1], [1, 0, 0, 9, 7]]
check(matrix_index(A, 2, 4).value == 9 and matrix_index(A, 2, 4).python_index == (1, 3))
check(matrix_index(A, 1, 1).value == 5 and matrix_index(A, 0, 0, start=0).value == 5)
check(matrix_dim(A).value == (2, 5) and matrix_dim(A).count == 10)
smoke(matrix_index(A, 2, 4))
smoke(matrix_dim(A))
smoke(list_index([1, 2], 1))
raises(ValueError, matrix_index, A, 3, 1, text="ausserhalb")
raises(ValueError, matrix_dim, [[1, 2], [3]], text="rechteckig")
raises(ValueError, matrix_dim, [])

# Serie 1, Seite 10/11
check(broadcast_calc("[1, 3, -1, 0, 1.5] + [1, 2, 2, 1, 0.5]").value == [2, 5, 1, 1, 2])
check(broadcast_calc("[True, True, False] and [False, True, False]").value == [False, True, False])
check(broadcast_calc("3 * [2, 5, 4, -2]").value == [6, 15, 12, -6])
check(broadcast_calc("3 - [0, 1, 2, 3, 2, 1, 0]").value == [3, 2, 1, 0, 1, 2, 3])
check(broadcast_calc("not [True, False, False, True, False]").value == [False, True, True, False, True])
check(broadcast_calc("[[1,2],[3,4],[7,5]] + [[0,4],[1,1],[5,2]]").value == [[1, 6], [4, 5], [12, 7]])
check(broadcast_calc("[[False,True],[False,True]] or [[True,False],[False,True]]").value ==
      [[True, True], [False, True]])
M3 = [[1, 2, 1, 1], [3, 4, 2, 2], [3, 5, 1, 1]]
M4 = [[1, 2, 3, 2], [1, 5, 3, 3], [1, 2, 2, 5]]
r = broadcast_calc("2 + A - [[1],[2],[3]] + [[0,1,1,2]] * B", A=M3, B=M4)
want = 2 + np.array(M3) - np.array([[1], [2], [3]]) + np.array([[0, 1, 1, 2]]) * np.array(M4)
check(r.value == want.tolist() and r.shape == (3, 4) and r.kind == "matrix", r.value)
check(any(s.items and s.items[-1][0] == "Broadcasting" for s in r.steps))
# Lerngerüst (f)
r = broadcast_calc("[[1,1,2,2],[2,2,0,0]] + 1 - [[2],[1]]")
check(r.value == (np.array([[1, 1, 2, 2], [2, 2, 0, 0]]) + 1 - np.array([[2], [1]])).tolist() == [[0, 0, 1, 1], [2, 2, 0, 0]])
smoke(r)
# Zufall gegen NumPy
for _ in range(150):
    m, n = rnd.randint(1, 4), rnd.randint(1, 4)
    big = np.array([[rnd.randint(-9, 9) for _ in range(n)] for _ in range(m)])
    small = rnd.choice([(1, 1), (m, 1), (1, n), (m, n)])
    sm = np.array([[rnd.randint(-9, 9) for _ in range(small[1])] for _ in range(small[0])])
    op = rnd.choice(["+", "-", "*"])
    r = broadcast_calc(f"A {op} B", A=big.tolist(), B=sm.tolist())
    want = eval(f"big {op} sm")
    check(r.value == want.tolist() and r.shape == (m, n))
    r = broadcast_calc(f"B {op} A", A=big.tolist(), B=sm.tolist())
    check(r.value == eval(f"sm {op} big").tolist())
    c = rnd.randint(-5, 5)
    check(broadcast_calc(f"{c} {op} A", A=big.tolist()).value == eval(f"c {op} big").tolist())
    sm2 = sm.copy()
    sm2[sm2 == 0] = 1
    check(broadcast_calc("A // B", A=big.tolist(), B=sm2.tolist()).value == (big // sm2).tolist())
    check(broadcast_calc("A % B", A=big.tolist(), B=sm2.tolist()).value == (big % sm2).tolist())
raises(ValueError, broadcast_calc, "[1,2,3] + [1,2]", text="gleich lang")
raises(ValueError, broadcast_calc, "[[1,2],[3,4]] + [[1,2,3]]", text="Broadcasting")
raises(ValueError, broadcast_calc, "[[1],[2],[3]] + [[1,2,3,4]]", text="Broadcasting")
raises(ValueError, broadcast_calc, "[1,2] + [[1,2]]", text="Liste")
raises(ValueError, broadcast_calc, "[1,2] + True", text="mischen")
raises(ValueError, broadcast_calc, "[1,2] and [True,False]")
raises(ValueError, broadcast_calc, "[1,2] // [0,1]", text="durch 0")
raises(ValueError, broadcast_calc, "[[1,2],[3]]", text="rechteckig")
raises(ValueError, broadcast_calc, "[[[1]]]")
raises(ValueError, broadcast_calc, "[1,2", text="Klammer")
raises(ValueError, broadcast_calc, "[1 2]", text="Komma")
raises(ValueError, broadcast_calc, "A + 1", text="Unbekannter Name")
raises(ValueError, broadcast_calc, "A + 1", A=[[1, 2], [3]], text="rechteckig")
raises(ValueError, broadcast_calc, "A + 1", A=[1, True], text="mischen")
raises(ValueError, broadcast_calc, "A + 1", A=list(range(500)), text="zu viele")
check(broadcast_calc("5").value == 5 and broadcast_calc("5").kind == "scalar")

# ==========================================================================
# 10. Funktionen
# ==========================================================================

r = check_function("{1,2,3}", "{a,b}", {1: "a", 2: "b", 3: "a"})
check(r.is_function and r.surjective and not r.injective and not r.bijective)
r = check_function("{1,2}", "{a,b}", [(1, "a"), (1, "b"), (2, "a")])
check(not r.is_function and r.bijective is None and any("mehrere Bilder" in p for p in r.problems))
r = check_function("{1,2,3}", "{a,b}", {1: "a", 2: "b"})
check(not r.is_function and any("kein Bild" in p for p in r.problems))
r = check_function("{1,2}", "{a,b}", {1: "a", 2: "c"})
check(not r.is_function and any("nicht in W" in p for p in r.problems))
r = check_function("{1,2}", "{a,b}", {1: "a", 2: "b"})
check(r.bijective)
r = check_function(range(5), range(10), lambda x: 2 * x)
check(r.is_function and r.injective and not r.surjective)
check(check_function(range(8), range(8), lambda x: (x + 1) % 8).bijective)
smoke(r)
smoke(check_function("{1,2}", "{a,b}", [(1, "a"), (1, "b"), (2, "a")]))
raises(ValueError, check_function, "{1}", "{a}", [(1,)], text="Paare")

# ==========================================================================
# 11. Allgemeine Eigenschaften
# ==========================================================================

check(all(hasattr(dt, n) for n in dt.__all__))
check(MAX_NUMBER == 10 ** 18 and MAX_BITS == 64)
for cls in (BoolEvalResult, IntEvalResult, ArrayEvalResult, IntOpResult, BinaryResult, FractionResult,
            FracBinaryResult, SqrtBinaryResult, FloatBitsResult, FloatDecodeResult, FloatCompareResult,
            TwosComplementResult, RangeResult, OverflowResult, SetOpResult, SetRelationResult,
            PowerSetResult, ProductResult, PartitionResult, BitstringResult, IndexResult,
            DimensionResult, FunctionCheckResult):
    check(issubclass(cls, Result))
# kein eval/exec im Modul
src = open(dt.__file__, encoding="utf-8").read()
import re as _re
check(not _re.search(r"(?<![\w.])(eval|exec)\(", src), "eval/exec im Modul")
r = eval_bool("False or True and False and True or False")
check("==" in r.latex())
print(r)
print(f"alle grün: {N} Prüfungen")
