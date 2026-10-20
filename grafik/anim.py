#!/usr/bin/env python3
"""
Manim-Animationen (freiwillig): der Rechenweg jedes Ergebnisses, Schritt für Schritt.

Swiss Style wie die App: Schwarz/Weiss mit einem Rot (#e60000).

Engine: manimgx wird bevorzugt (schnell, bringt die Videoausgabe selbst mit,
kein ffmpeg und kein Cairo/Pango nötig). Ist es nicht installiert, fällt
alles auf Manim CE zurück. DMATKIT_ENGINE=manim erzwingt den Rückfall.

Installation:
    pip install manimgx
    (Rückfall: brew install cairo pango pkg-config ffmpeg; pip install manim)

Verwendung:
    import dmatkit as dk
    from dmatkit.grafik.anim import job_from_result, render
    pfad = render(job_from_result(dk.to_binary(409)), progress=print)

Von der Kommandozeile (rendert 409 → Binär):
    python3 -m dmatkit.grafik.anim [zahl]
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

#: Grösse und Bildrate je Qualitätsstufe (Manim-CE-Kürzel dahinter).
QUALITIES = {
    "low": ("854x480", "15", "-ql"),
    "medium": ("1280x720", "30", "-qm"),
    "high": ("1920x1080", "60", "-qh"),
}

SCENE_NAMES = {"rechenweg": "RechenwegScene"}

#: Mehr Tabellenzeilen passen nicht aufs Bild; der Rest wird mit »…« angedeutet.
MAX_TABLE_ROWS = 10


@dataclass
class Theme:
    """Farben der Animation: Swiss Style, Schwarz/Weiss mit einem Rot."""
    background: str = "#ffffff"
    text: str = "#000000"
    muted: str = "#666666"
    accent: str = "#e60000"

    @staticmethod
    def dark() -> "Theme":
        return Theme(background="#000000", text="#ffffff", muted="#999999", accent="#ff3333")


@dataclass
class AnimationJob:
    """
    Ein Rendering-Auftrag: der Rechenweg als Folge von Bildern.

    slides — je Bild ein dict: num, head, rule, detail (Text), items
             ([Name, Wert]-Paare), table ({header, rows, marks} oder None),
             final (das Ergebnisbild am Ende)
    title  — Überschrift, bleibt über allen Bildern stehen
    """
    slides: list[dict] = field(default_factory=list)
    title: str = ""
    scene: str = "rechenweg"
    theme: Theme = field(default_factory=Theme)
    quality: str = "low"

    def to_json(self) -> str:
        data = asdict(self)
        data["theme"] = asdict(self.theme)
        return json.dumps(data)

    @staticmethod
    def from_json(text: str) -> "AnimationJob":
        data = json.loads(text)
        return AnimationJob(theme=Theme(**data.pop("theme")), **data)


def job_from_result(result, *, theme: Theme | None = None, quality: str = "low",
                    title: str = "") -> AnimationJob:
    """
    Baut den Auftrag aus dem Rechenweg irgendeines dmatkit-Ergebnisses
    (`to_binary(409)`, `truth_table(...)`, `resolution(...)`, ...): ein Bild je
    Schritt mit Regel, Erklärung, Zwischenergebnissen und Tabelle, am Ende die
    Ergebnisgrössen in Rot.
    """
    from ..rechnen.render import text_any, tex_to_text

    dec, digits = result.decimal, result.digits

    def piece(label, obj):
        return [tex_to_text(label) if label else "", text_any(obj, dec, digits)]

    slides = []
    for i, step in enumerate(result.steps, 1):
        table = None
        if step.table is not None:
            t = step.table
            rows = [[text_any(c, dec, digits) or "·" for c in r] for r in t.rows]
            marks = [t.marks.get(k, "") for k in range(len(rows))]
            if len(rows) > MAX_TABLE_ROWS:
                rows = rows[:MAX_TABLE_ROWS] + [["…"] + ["·"] * (len(t.header) - 1)]
                marks = marks[:MAX_TABLE_ROWS] + [""]
            table = {"header": [text_any(h, dec, digits) or "·" for h in t.header],
                     "rows": rows, "marks": marks}
        slides.append({"num": i, "head": tex_to_text(step.title), "rule": step.rule,
                       "detail": tex_to_text(step.detail),
                       "items": [piece(l, o) for l, o in step.items],
                       "table": table, "final": False})
    if result.highlights:
        slides.append({"num": 0, "head": "Fazit", "rule": "", "detail": "",
                       "items": [piece(l, o) for l, o in result.highlights],
                       "table": None, "final": True})
    if not slides:
        raise ValueError("Dieses Ergebnis hat keinen Rechenweg zum Zeigen.")
    return AnimationJob(slides=slides, title=title or tex_to_text(result.title),
                        theme=theme or Theme(), quality=quality)


# --------------------------------------------------------------------------
# Verfügbarkeit
# --------------------------------------------------------------------------

def engine() -> str:
    """'manimgx' wenn installiert, sonst 'manim'. DMATKIT_ENGINE=manim erzwingt den Rückfall."""
    forced = os.environ.get("DMATKIT_ENGINE")
    if forced in ("manim", "manimgx"):
        return forced
    return "manimgx" if importlib.util.find_spec("manimgx") else "manim"


def manim_available() -> tuple[bool, str]:
    """(verfügbar, Begründung) — prüft die Engine, ohne dass ein Import-Fehler hochkommt."""
    name = engine()
    try:
        importlib.import_module(name)
    except ImportError:
        return False, "manimgx (oder Manim CE) ist nicht installiert."
    except Exception as exc:                       # defekte Installation
        return False, f"{name} lässt sich nicht laden: {exc}"
    if name == "manim":
        import shutil
        if shutil.which("ffmpeg") is None:
            return False, "ffmpeg fehlt — nötig für die Videoausgabe."
    return True, ""


INSTALL_HINT = ("pip install manimgx   # Rendering und Videoausgabe in einem Paket\n"
                "(Rückfall: Manim CE mit brew install cairo pango pkg-config ffmpeg; pip install manim)")


# --------------------------------------------------------------------------
# Die Szene
# --------------------------------------------------------------------------

def _build_scenes():
    """Erzeugt die Szenenklassen erst beim Aufruf, damit das Modul ohne Manim importierbar bleibt."""
    mod = importlib.import_module(engine())
    (Scene, VGroup, FadeIn, FadeOut, Create, Line, Rectangle, LaggedStart, Indicate, Flash,
     GrowFromCenter, Transform, ReplacementTransform, DOWN, LEFT, RIGHT, UP, DR, UL) = (
        getattr(mod, n) for n in ("Scene VGroup FadeIn FadeOut Create Line Rectangle LaggedStart Indicate "
                                  "Flash GrowFromCenter Transform ReplacementTransform DOWN LEFT RIGHT UP DR UL").split())
    font = "Noto Sans" if engine() == "manimgx" else None

    def Text(*args, **kw):
        if font:
            kw.setdefault("font", font)
        return mod.Text(*args, **kw)

    def wrap(text, width):
        return "\n".join(textwrap.wrap(text, width) or [""])

    class RechenwegScene(Scene):
        """
        Titelkarte, dann je Schritt ein Bild: grosse Schrittnummer, Regel als Chip,
        Zwischenergebnisse, Tabelle (zeilenweise, markierte Zeilen rot hinterlegt).
        Unten läuft eine Fortschrittsleiste mit; das Ergebnis steht zuletzt auf einem roten Balken.
        """

        W, BAR_Y = 14.2, -3.85

        def grid(self, table, th):
            rows = [table["header"]] + table["rows"]
            cells = [[Text(c, font_size=26, color=th.muted if r == 0 else th.text) for c in row]
                     for r, row in enumerate(rows)]
            widths = [max(row[i].width for row in cells) + 0.6 for i in range(len(rows[0]))]
            row_groups = []
            for r, row in enumerate(cells):
                x = 0.0
                for i, c in enumerate(row):
                    c.move_to([x + widths[i] / 2, -r * 0.46, 0])
                    x += widths[i]
                row_groups.append(VGroup(*row))
            return VGroup(*row_groups), row_groups

        def item_row(self, name, value, th, big=False, on_band=False):
            ink = th.background if on_band else th.text
            soft = th.background if on_band else th.muted
            val = Text(wrap(value, 46), font_size=46 if big else 36, color=ink)
            col = VGroup(*([Text(name, font_size=22, color=soft)] if name else []), val)
            col.arrange(DOWN, aligned_edge=LEFT, buff=0.1)
            bar = Line(col.get_top(), col.get_bottom(), color=ink if on_band else th.accent, stroke_width=6)
            return VGroup(bar, col).arrange(RIGHT, buff=0.3)

        def construct(self):
            job = AnimationJob.from_json(os.environ["DMATKIT_JOB"])
            th, slides = job.theme, job.slides
            n = len(slides)
            W, y = self.W, self.BAR_Y
            self.camera.background_color = th.background

            # --- Titelkarte
            big = Text(job.title, font_size=56, color=th.text)
            big.scale(min(1.0, 12.0 / big.width))
            rule = Line(LEFT * 1.6, RIGHT * 1.6, color=th.accent, stroke_width=10).next_to(big, DOWN, buff=0.5)
            sub = Text(f"Rechenweg in {sum(1 for s in slides if not s['final'])} Schritten",
                       font_size=28, color=th.muted).next_to(rule, DOWN, buff=0.4)
            self.play(FadeIn(big, shift=UP * 0.3), Create(rule), run_time=1.0)
            self.play(FadeIn(sub), run_time=0.4)
            self.wait(0.7)
            small = Text(job.title, font_size=22, color=th.muted).to_corner(UL, buff=0.45)
            track = Line([-W / 2, y, 0], [W / 2, y, 0], color=th.muted, stroke_width=2).set_opacity(0.35)
            bar = Line([-W / 2, y, 0], [-W / 2 + 0.01, y, 0], color=th.accent, stroke_width=7)
            self.play(ReplacementTransform(big, small), FadeOut(rule), FadeOut(sub), FadeIn(track),
                      run_time=0.8)
            counter = None

            for k, s in enumerate(slides):
                final = s["final"]
                # Fortschritt
                new_bar = Line([-W / 2, y, 0], [-W / 2 + W * (k + 1) / n, y, 0], color=th.accent, stroke_width=7)
                new_counter = Text(f"{k + 1:02d} / {n:02d}", font_size=20, color=th.muted).to_corner(DR, buff=0.45)
                new_counter.shift(UP * 0.35)

                # Inhalt
                ink = th.background if final else th.text
                head_parts = []
                if not final:
                    head_parts.append(Text(f"{s['num']:02d}", font_size=84, color=th.accent))
                head_parts.append(Text(wrap(s["head"], 34), font_size=44 if not final else 40, color=ink))
                head = VGroup(*head_parts).arrange(RIGHT, buff=0.45, aligned_edge=DOWN if not final else UP)
                body = [head]
                chip = None
                if s["rule"]:
                    label = Text("Regel   " + s["rule"], font_size=24, color=th.accent)
                    chip = VGroup(Rectangle(width=label.width + 0.6, height=label.height + 0.36,
                                            color=th.accent, stroke_width=2.5), label)
                    label.move_to(chip[0])
                    body.append(chip)
                if s["detail"]:
                    body.append(Text(wrap(s["detail"], 95), font_size=24, color=th.muted))
                rows_ = [self.item_row(nm, v, th, big=final, on_band=final) for nm, v in s["items"]]
                if rows_:
                    # Kurze Werte nebeneinander: spart Höhe, damit nichts klein geschrumpft wird.
                    short = (not final and 1 < len(rows_) <= 4 and
                             all(len(v) <= 22 and len(nm) <= 30 for nm, v in s["items"]))
                    body.append(VGroup(*rows_).arrange(RIGHT if short else DOWN, aligned_edge=UP if short else LEFT,
                                                       buff=0.9 if short else 0.35))
                table_rows = marks = None
                if s["table"]:
                    tbl, table_rows = self.grid(s["table"], th)
                    marks = [""] + s["table"]["marks"]
                    body.append(tbl)
                group = VGroup(*body).arrange(DOWN, aligned_edge=LEFT, buff=0.38)
                group.scale(min(1.0, 12.4 / group.width, 5.4 / group.height))
                if final:
                    group.move_to([0, 0.2, 0]).to_edge(LEFT, buff=0.85)
                else:
                    group.to_edge(UP, buff=1.15).to_edge(LEFT, buff=0.85)

                # Hintergrund-Elemente: roter Balken (Ergebnis), Zeilenfarbe (Tabelle)
                extras = []
                if final:
                    band = Rectangle(width=W + 0.4, height=group.height + 1.3, color=th.accent,
                                     fill_color=th.accent, fill_opacity=1, stroke_width=0)
                    band.move_to(group)
                    extras.append(band)
                if table_rows:
                    for r, rg in enumerate(table_rows):
                        if marks[r] in ("hi", "bad", "ok") and r > 0:
                            hl = Rectangle(width=tbl.width + 0.5, height=0.44, stroke_width=0,
                                           fill_color=th.accent, fill_opacity=0.14 if marks[r] != "ok" else 0.06)
                            hl.move_to([tbl.get_center()[0], rg.get_center()[1], 0])
                            extras.append(hl)

                # Ablauf des Bildes
                first = [FadeIn(head_parts[0], shift=RIGHT * 0.4)] if not final else []
                if final:
                    self.play(GrowFromCenter(band), Transform(bar, new_bar), run_time=0.7)
                    first = [FadeIn(head, shift=UP * 0.2)]
                    self.play(*first, run_time=0.4)
                else:
                    self.play(*first, FadeIn(head_parts[-1], shift=RIGHT * 0.4), Transform(bar, new_bar),
                              run_time=0.6)
                if counter is None:
                    counter = new_counter
                    self.play(FadeIn(counter), run_time=0.2)
                else:
                    self.play(Transform(counter, new_counter), run_time=0.2)
                if chip is not None:
                    self.play(Create(chip[0]), FadeIn(chip[1], shift=RIGHT * 0.2), run_time=0.5)
                if s["detail"]:
                    self.play(FadeIn(body[2 if chip is not None else 1], shift=UP * 0.1), run_time=0.5)
                if rows_:
                    self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.35) for r in rows_], lag_ratio=0.35),
                              run_time=0.5 + 0.35 * len(rows_))
                    if not final:
                        self.play(Indicate(rows_[-1], color=th.accent, scale_factor=1.06), run_time=0.7)
                    else:
                        self.play(Flash(rows_[0], color=th.background, line_length=0.3, flash_radius=0.5),
                                  run_time=0.8)
                if table_rows:
                    self.play(LaggedStart(*[FadeIn(rg, shift=DOWN * 0.12) for rg in table_rows],
                                          lag_ratio=0.18), run_time=0.8 + 0.18 * len(table_rows))
                    if len(extras):
                        self.play(*[FadeIn(e) for e in extras], run_time=0.5)

                chars = sum(len(x) for x in (s["detail"], s["rule"])) + sum(len(v) for _, v in s["items"])
                self.wait(min(4.5, max(1.0, 0.6 + 0.035 * chars)))
                if k < n - 1:
                    self.play(FadeOut(VGroup(group, *extras), shift=LEFT * 0.6), run_time=0.45)
            self.wait(2.0)

    return {"rechenweg": RechenwegScene}


# --------------------------------------------------------------------------
# Rendern
# --------------------------------------------------------------------------

#: Wird vom Kindprozess ausgeführt. Die Engine sucht Szenenklassen nur in
#: diesem Modul, darum wird __module__ der hier erzeugten Klassen umgehängt.
_RUNNER = '''
import importlib, importlib.util, os, sys
if {pkg!r} not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        {pkg!r}, os.path.join({here!r}, "__init__.py"),
        submodule_search_locations=[{here!r}])
    _p = importlib.util.module_from_spec(_spec)
    sys.modules[{pkg!r}] = _p
    _spec.loader.exec_module(_p)
_mod = importlib.import_module({pkg!r} + ".grafik.anim")
for _key, _cls in _mod._build_scenes().items():
    _cls.__module__ = __name__
    globals()[_mod.SCENE_NAMES[_key]] = _cls
'''

_PROGRESS = re.compile(r"(\d+)\s*%")


def render(job: AnimationJob, *, progress: Callable[[int, str], None] | None = None,
           out_dir: str | Path | None = None, timeout: float = 600) -> Path:
    """
    Rendert den Auftrag und gibt den Pfad des MP4 zurück.

    Die Engine läuft in einem eigenen Prozess: Sie schreibt viel in globale
    Konfiguration und lässt sich im selben Prozess nicht sauber mehrfach
    starten. progress bekommt (Prozent, Text). Fehlt die Engine, gibt es einen
    RuntimeError mit Installationshinweis.
    """
    ok, why = manim_available()
    if not ok:
        raise RuntimeError(f"{why}\nInstallation:\n{INSTALL_HINT}")
    if job.scene not in SCENE_NAMES:
        raise ValueError(f"Unbekannte Szene {job.scene!r}")

    here = Path(__file__).resolve().parent.parent
    work = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="dmatkit-anim-"))
    work.mkdir(parents=True, exist_ok=True)
    script = work / "scene_runner.py"
    script.write_text(_RUNNER.format(pkg=(__package__ or here.name).split(".")[0], here=str(here)),
                      encoding="utf-8")

    res, fps, flag = QUALITIES.get(job.quality, QUALITIES["low"])
    name = SCENE_NAMES[job.scene]
    if engine() == "manimgx":
        out = work / "media" / "dmatkit.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "manimgx", "render", str(script), name,
               "-o", str(out), "-r", res, "--fps", fps]
    else:
        cmd = [sys.executable, "-m", "manim", flag, "--disable_caching",
               "--media_dir", str(work / "media"), "-o", "dmatkit", str(script), name]

    env = dict(os.environ)
    env["DMATKIT_JOB"] = job.to_json()
    env["PYTHONPATH"] = str(here.parent) + os.pathsep + env.get("PYTHONPATH", "")

    if progress:
        progress(2, "Animation startet …")
    process = subprocess.Popen(cmd, cwd=str(work), env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    # Die Engine zählt je Animation von 0 bis 100; für einen Balken, der nicht
    # zurückspringt, zählen wir die abgeschlossenen Animationen mit.
    expected = len(job.slides) * 8 + 3
    log: list[str] = []
    done = last = 0
    timed_out = threading.Event()

    def stop():
        timed_out.set()
        process.kill()

    timer = threading.Timer(timeout, stop)
    timer.start()
    try:
        for line in process.stdout:
            log.append(line)
            for match in _PROGRESS.finditer(line):
                within = int(match.group(1))
                if within >= 100:
                    done += 1
                overall = min(97, int((done + within / 100) / expected * 95) + 3)
                if overall > last and progress:
                    last = overall
                    progress(overall, "Rendere …")
        process.wait()
    finally:
        timer.cancel()
    if timed_out.is_set():
        raise RuntimeError("Das Rendern hat zu lange gedauert.")

    # Manche Fehler stehen nur im Text, der Prozess endet trotzdem mit 0.
    if process.returncode != 0 or "Traceback" in "".join(log):
        tail = "\n".join(l.rstrip() for l in log if l.strip())[-1200:] or "Keine Ausgabe."
        raise RuntimeError(f"Das Rendern ist fehlgeschlagen:\n{tail}")

    found = sorted((work / "media").rglob("dmatkit*.mp4"), key=lambda p: p.stat().st_mtime)
    if not found:
        raise RuntimeError("Es wurde keine Datei erzeugt.")
    if progress:
        progress(100, "Fertig.")
    return found[-1]


if __name__ == "__main__":
    if not __package__:
        # Direkt gestartet: das Paket über seinen Ordner importierbar machen.
        _here = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(_here.parent))
        _mod = importlib.import_module(_here.name + ".grafik.anim")
    else:
        _mod = sys.modules[__name__]
    _ok, _why = _mod.manim_available()
    if not _ok:
        print(_why)
        print(_mod.INSTALL_HINT)
        sys.exit(1)
    _dk = importlib.import_module(_mod.__name__.rsplit(".grafik", 1)[0])
    _n = int(sys.argv[1]) if len(sys.argv) > 1 else 409
    print("Fertig:", _mod.render(_mod.job_from_result(_dk.to_binary(_n)),
                                 progress=lambda p, t: print(f"{p:3d}%  {t}")))
