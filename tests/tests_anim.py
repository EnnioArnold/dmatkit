#!/usr/bin/env python3
"""
Tests für dmatkit.grafik.anim: Auftrag und Szenentabelle, ohne Render.
Start: python3 tests/tests_anim.py

Gerendert wird hier nicht — das braucht manimgx und dauert.
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', '..'))

from dmatkit.grafik import anim  # noqa: E402


import dmatkit as dk  # noqa: E402

#: Je Woche eine Rechnung: jeder Rechenweg muss sich in einen Auftrag wandeln lassen.
BEISPIELE = [
    lambda: dk.to_binary(409),
    lambda: dk.eval_bool("False or True and False"),
    lambda: dk.truth_table("(A or B) and not C"),
    lambda: dk.check_equivalence("A -> B", "not B -> not A"),
    lambda: dk.has_duplicate([3, 1, 4, 1, 5]),
    lambda: dk.entails(["A -> B", "A"], "B"),
    lambda: dk.resolution(["A -> B", "B -> C"], "A -> C"),
    lambda: dk.count_steps("decToBin", x=25),
    lambda: dk.max_input_size("n**2", time_limit_s=60),
]


def test_job_from_every_week():
    for make in BEISPIELE:
        job = anim.job_from_result(make())
        assert job.slides and job.title and job.scene in anim.SCENE_NAMES


def test_binary_job_content():
    job = anim.job_from_result(dk.to_binary(409))
    assert job.slides[-1]["final"] and "110011001" in " ".join(job.slides[-1]["items"])
    assert anim.SCENE_NAMES["rechenweg"] == "RechenwegScene"


def test_long_table_is_cut():
    job = anim.job_from_result(dk.truth_table("A and B and C and D"))
    tables = [s["table"] for s in job.slides if s["table"]]
    assert tables and all(len(t["rows"]) <= anim.MAX_TABLE_ROWS + 1 for t in tables)
    assert all(len(t["rows"]) == len(t["marks"]) for t in tables)


def test_no_steps_is_rejected():
    class Leer:
        title, decimal, digits, steps, highlights = "x", False, 4, [], []
    try:
        anim.job_from_result(Leer())
    except ValueError:
        return
    raise AssertionError("ValueError erwartet")


def test_job_json_roundtrip():
    job = anim.job_from_result(dk.truth_table("A and B"), theme=anim.Theme.dark())
    assert anim.AnimationJob.from_json(job.to_json()) == job


def test_engine_forced():
    os.environ["DMATKIT_ENGINE"] = "manim"
    try:
        assert anim.engine() == "manim"
    finally:
        del os.environ["DMATKIT_ENGINE"]


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                import traceback
                fails += 1
                print("FAIL", name, repr(e))
                traceback.print_exc()
    print("alle Tests bestanden" if not fails else f"{fails} Test(s) fehlgeschlagen")
    sys.exit(1 if fails else 0)
