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


def test_dec_to_bin_job():
    job = anim.job_from_dec_to_bin(409)
    assert job.scene == "steps" and anim.SCENE_NAMES["steps"] == "StepsScene"
    assert job.lines[0] == "409 : 2 = 204  Rest 1" and len(job.lines) == 9
    assert "110011001₂" in job.result


def test_dec_to_bin_zero_and_bad_input():
    assert anim.job_from_dec_to_bin(0).lines == ["0 : 2 = 0  Rest 0"]
    for bad in (-1, 2.5, True):
        try:
            anim.job_from_dec_to_bin(bad)
        except ValueError:
            continue
        raise AssertionError(f"ValueError erwartet für {bad!r}")


def test_job_json_roundtrip():
    job = anim.job_from_dec_to_bin(25, theme=anim.Theme.dark())
    again = anim.AnimationJob.from_json(job.to_json())
    assert again == job


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
