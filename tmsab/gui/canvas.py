"""Canvas matplotlib embutido em Qt (FigureCanvasQTAgg + barra de navegação)."""
from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyside6")

from PySide6.QtWidgets import QVBoxLayout, QWidget  # noqa: E402  (antes do backend, para o qt_compat escolher PySide6)

import matplotlib  # noqa: E402

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


class MplCanvas(QWidget):
    """Figura única com barra de ferramentas; `fresh_axes()` limpa e devolve um eixo novo."""

    def __init__(self, parent=None, width: float = 4.5, height: float = 4.0, dpi: int = 90, toolbar: bool = True):
        super().__init__(parent)
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvasQTAgg(self.figure)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        if toolbar:
            self.toolbar = NavigationToolbar2QT(self.canvas, self)
            lay.addWidget(self.toolbar)
        else:
            self.toolbar = None
        lay.addWidget(self.canvas)
        self.setMinimumSize(260, 220)

    def fresh_axes(self):
        self.figure.clear()
        return self.figure.add_subplot(111)

    def clear(self, message: str | None = None):
        self.figure.clear()
        if message:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, wrap=True, fontsize=9)
            ax.set_axis_off()
        self.draw()

    def draw(self):
        try:
            self.figure.tight_layout()
        except Exception:
            pass
        self.canvas.draw_idle()
