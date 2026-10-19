"""
Thinking-Orb für Qt: ein animierter Bogen aus mehreren Bändern.

Nachbau des »composing«-Zustands aus github.com/Jakubantalik/thinking-orbs
(64px-Preset) mit QPainter — nicht eine Kopie der Originalquelle.

Start:
    python3 orb.py   # rendert einen Frame und prüft ihn
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

BANDS = 3          # ponytail: feste Bandzahl, das Preset hat auch genau drei
SAMPLES = 48       # ponytail: Kurve aus Punkten statt exakter Bézierkurve


class ThinkingOrb(QWidget):
    """Läuft nur, solange das Widget sichtbar ist — der Timer hält sonst die CPU wach."""

    def __init__(self, size: int = 64, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.phase = 0.0
        self.colors: list[QColor] = [QColor("#e60000"), QColor("#000000"),
                                     QColor("#666666")]
        self.timer = QTimer(self)
        self.timer.setInterval(33)  # ~30 fps
        self.timer.timeout.connect(self._tick)

    def set_colors(self, colors: list[str]):
        self.colors = [QColor(c) for c in colors[:BANDS]]
        self.update()

    def _tick(self):
        self.phase += 0.09
        self.update()

    def showEvent(self, event):
        super().showEvent(event)
        self.timer.start()

    def hideEvent(self, event):
        super().hideEvent(event)
        self.timer.stop()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        size = self.width()
        mid = size / 2
        for i, color in enumerate(self.colors):
            amp = size * (0.24 - 0.05 * i)
            offset = i * 0.9
            path = QPainterPath()
            for k in range(SAMPLES + 1):
                t = k / SAMPLES
                # Sinus, an den Enden auf Null gedämpft: der Bogen wirkt
                # wie ein Band, das im Zentrum verankert ist.
                y = mid + amp * math.sin(2 * math.pi * 1.2 * t + self.phase + offset) \
                    * math.sin(math.pi * t)
                point = QPointF(size * t, y)
                if k == 0:
                    path.moveTo(point)
                else:
                    path.lineTo(point)
            band = QColor(color)
            band.setAlphaF(1.0)
            pen = QPen(band, max(2.0, size * 0.09))
            pen.setCapStyle(Qt.FlatCap)   # Swiss: eckige Enden
            pen.setJoinStyle(Qt.MiterJoin)
            p.setPen(pen)
            p.drawPath(path)
        p.end()


def demo():
    """Rendert einen Frame; schlägt fehl, wenn nichts gezeichnet wurde."""
    import os
    import sys
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    orb = ThinkingOrb(64)
    orb.resize(64, 64)
    image = QImage(64, 64, QImage.Format_ARGB32)
    image.fill(0)
    orb.render(image)
    painted = sum(1 for x in range(64) for y in range(64)
                  if image.pixelColor(x, y).alpha() > 0)
    assert painted > 50, f"Orb hat zu wenig gezeichnet: {painted} Pixel"
    print(f"ok — {painted} Pixel gezeichnet")


if __name__ == "__main__":
    demo()
