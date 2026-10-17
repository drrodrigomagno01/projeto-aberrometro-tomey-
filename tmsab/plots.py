"""Figuras compartilhadas entre PDF e interface (matplotlib).

Regras: mapa de elevação = escala sequencial (um matiz); frente de onda e resíduos =
escala divergente com meio neutro e limites simétricos; cobertura = categorias com legenda
e marcadores (nunca só cor). Títulos e unidades explícitos em todo mapa; escalas iguais
entre mapas comparados (ver compare_scale).
"""
from __future__ import annotations

from typing import Optional

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .model import Analysis, NormalizedExam, PointState, WavefrontResult

SEQ_CMAP = "viridis"
DIV_CMAP = "RdBu_r"
STATE_STYLE = {
    PointState.MEASURED: ("#2E6F9E", "o", "medido"),
    PointState.INTERPOLATED: ("#E08A1E", "^", "interpolado"),
    PointState.MISSING: ("#9A9A9A", "x", "ausente"),
}


def _circle(ax: Axes, r: float, cx: float = 0.0, cy: float = 0.0, **kw):
    t = np.linspace(0, 2 * np.pi, 361)
    ax.plot(cx + r * np.cos(t), cy + r * np.sin(t), **kw)


def plot_elevation(ax: Axes, exam: NormalizedExam, title: Optional[str] = None, vmin=None, vmax=None):
    s = exam.samples
    m = np.isfinite(s.z)
    sc = ax.scatter(s.x[m], s.y[m], c=s.z[m] * 1000.0, s=5, cmap=SEQ_CMAP, vmin=vmin, vmax=vmax, linewidths=0)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm) — direita do observador")
    ax.set_ylabel("y (mm) — cima")
    ax.set_title(title or f"Altura sagital HIT (µm) — {exam.identity.eye or '?'}")
    cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("µm (referência: plano apical, + posterior)")
    return sc


def plot_coverage(ax: Axes, exam: NormalizedExam, apertures: Optional[list[tuple[float, float, float]]] = None,
                  title: Optional[str] = None):
    s = exam.samples
    for st, (color, marker, label) in STATE_STYLE.items():
        m = (s.point_state == st)
        if st == PointState.MISSING:
            # pontos ausentes não têm coordenadas: desenhar na posição nominal do anel
            rn = np.nanmean(s.r, axis=1)
            rr = np.where(np.isfinite(s.r), s.r, rn[:, None])
            th = np.radians(s.theta_deg)[None, :]
            xx, yy = rr * np.cos(th), rr * np.sin(th)
            if m.any():
                ax.scatter(xx[m], yy[m], c=color, s=8, marker=marker, label=f"{label} ({int(m.sum())})", linewidths=0.6)
        elif m.any():
            ax.scatter(s.x[m], s.y[m], c=color, s=4, marker=marker, label=f"{label} ({int(m.sum())})", linewidths=0)
    for (cx, cy, r) in (apertures or []):
        _circle(ax, r, cx, cy, color="black", lw=1, ls="--")
        ax.plot([cx], [cy], marker="+", color="black", ms=8)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title or "Cobertura e estado dos pontos")
    ax.legend(loc="upper right", fontsize=7, markerscale=2)


def plot_wavefront(ax: Axes, wf: WavefrontResult, key: str = "W", title: Optional[str] = None,
                   vlim: Optional[float] = None):
    g = wf.map_grid
    if g is None:
        ax.text(0.5, 0.5, "sem mapa", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return None
    Z = g[key]
    if vlim is None:
        vlim = float(np.nanmax(np.abs(Z))) if np.isfinite(Z).any() else 1.0
        vlim = max(vlim, 1e-3)
    a = wf.profile.radius_mm
    im = ax.imshow(Z, origin="lower", extent=[-a, a, -a, a], cmap=DIV_CMAP, vmin=-vlim, vmax=vlim, interpolation="nearest")
    _circle(ax, a, color="black", lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm) rel. ao centro de análise")
    ax.set_ylabel("y (mm)")
    labels = {"W": "Frente de onda traçada W (µm)", "W_fit": "Frente de onda ajustada (Zernike) (µm)", "residual": "Resíduo traçado − ajuste (µm)"}
    ax.set_title(title or f"{labels.get(key, key)} — {wf.profile.diameter_mm:g} mm, centro {wf.profile.center_mode}")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("µm (+ = adiantada)")
    return im


def plot_zernike_bars(ax: Axes, wf: WavefrontResult, title: Optional[str] = None, ylim: Optional[float] = None):
    co = [c for c in wf.coefficients if c.n >= 2]
    if not co:
        ax.text(0.5, 0.5, "sem coeficientes", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    vals = np.array([c.value_um for c in co])
    labels = [f"{c.n},{c.m:+d}" for c in co]
    colors = ["#2E6F9E" if c.n >= 3 else "#9FB7C9" for c in co]
    ax.bar(range(len(co)), vals, color=colors, width=0.7)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(range(len(co)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("coeficiente (µm, OSA)")
    if ylim:
        ax.set_ylim(-ylim, ylim)
    ax.set_title(title or f"Coeficientes de Zernike (n ≥ 2) — {wf.profile.diameter_mm:g} mm")
    for i, c in enumerate(co):
        if (c.n, c.m) in ((4, 0), (3, 1), (3, -1), (3, 3), (3, -3)):
            ax.annotate(c.name, (i, vals[i]), fontsize=5, rotation=90, ha="center", va="bottom" if vals[i] >= 0 else "top")


def compare_scale(wfs: list[WavefrontResult], key: str = "W") -> float:
    """Limite simétrico comum para mapas comparados."""
    v = 0.0
    for w in wfs:
        if w.map_grid is not None and np.isfinite(w.map_grid[key]).any():
            v = max(v, float(np.nanmax(np.abs(w.map_grid[key]))))
    return max(v, 1e-3)


def figure_overview(exam: NormalizedExam, analysis: Analysis, wf: Optional[WavefrontResult]) -> Figure:
    fig = Figure(figsize=(11, 8.5))
    axs = fig.subplots(2, 2)
    plot_elevation(axs[0, 0], exam)
    aps = [(w.center_xy[0], w.center_xy[1], w.profile.radius_mm) for w in analysis.wavefronts if w.validity == "disponivel"]
    plot_coverage(axs[0, 1], exam, apertures=aps)
    if wf is not None and wf.map_grid is not None:
        plot_wavefront(axs[1, 0], wf, "W")
        plot_wavefront(axs[1, 1], wf, "residual")
    else:
        for ax in axs[1]:
            ax.text(0.5, 0.5, "perfil indisponível", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
    fig.tight_layout()
    return fig
