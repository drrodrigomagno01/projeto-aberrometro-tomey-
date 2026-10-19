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


def plot_zernike_bars(ax: Axes, wf: WavefrontResult, title: Optional[str] = None, ylim: Optional[float] = None, annotate: bool = True):
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
        if annotate and (c.n, c.m) in ((4, 0), (3, 1), (3, -1), (3, 3), (3, -3)):
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


# ---------------------------------------------------------------------------
# Mapas topográficos (DIO/ROC/POW/elevação) e página "Resultado completo" (A4, impressão)
# ---------------------------------------------------------------------------
TOPO_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "topo_clinico",
    ["#1a1aff", "#2d7bff", "#2ec9ff", "#19d6a0", "#2ecc40", "#b5e61d", "#ffee00", "#ffa500", "#ff4d00", "#c80000", "#7a0000"],
)
TOPO_MAPS = (
    ("DIO", "Potência axial (sagital)", "D"),
    ("ROC", "Potência tangencial (instantânea)", "D"),
    ("POW", "Potência refrativa", "D"),
)


def _closed_polar_grid(s, values: np.ndarray):
    """Coordenadas fechadas em 360° para pcolormesh; pontos sem coordenada recebem a posição nominal do anel;
    centro preenchido com a média do primeiro anel (ver topo._grid_closed)."""
    from .topo import _grid_closed
    return _grid_closed(s, values, fill_center=True)


def _step_scale(v: np.ndarray, step: float, min_span: float) -> tuple[float, float]:
    ok = np.isfinite(v)
    if not ok.any():
        return (0.0, 1.0)
    lo, hi = np.nanpercentile(v[ok], 1), np.nanpercentile(v[ok], 99)
    if hi - lo < min_span:
        c = 0.5 * (lo + hi); lo, hi = c - min_span / 2, c + min_span / 2
    return float(np.floor(lo / step) * step), float(np.ceil(hi / step) * step)


def plot_topo_map(ax: Axes, exam: NormalizedExam, role: str, title: Optional[str] = None, step: float = 0.5,
                  pupil: bool = True, zones: tuple[float, ...] = (3.0, 6.0), colorbar: bool = True, cax=None, fontsize: float = 7.5):
    """Mapa de potência (D) do TMS na escala clínica (azul plano → vermelho curvo), passos de `step` D.
    Pontos interpolados pelo TMS entram no mapa; ausentes ficam em branco."""
    s = exam.samples
    ft = s.fields.get(role)
    if ft is None:
        ax.text(0.5, 0.5, f"tabela {role} ausente", ha="center", va="center", transform=ax.transAxes, fontsize=7)
        ax.set_axis_off()
        return None
    vals = np.where(ft.usable_mask(), ft.values, np.nan)
    X, Y, V = _closed_polar_grid(s, vals)
    lo, hi = _step_scale(vals, step, 4.0)
    levels = np.arange(lo, hi + step / 2, step)
    norm = matplotlib.colors.BoundaryNorm(levels, TOPO_CMAP.N, clip=True)
    pm = ax.pcolormesh(X, Y, V, cmap=TOPO_CMAP, norm=norm, shading="gouraud", rasterized=True)
    for d in zones:
        _circle(ax, d / 2, color="black", lw=0.4, ls=":", alpha=0.7)
    if pupil and exam.pupil.state == "registrada" and getattr(exam.pupil, "edge_r", None) is not None:
        th = np.radians(s.theta_deg); re_ = np.asarray(exam.pupil.edge_r, float)
        ax.plot(re_ * np.cos(th), re_ * np.sin(th), color="black", lw=0.8, ls="--")
    lim = float(np.nanmax(np.abs(np.concatenate([X.ravel(), Y.ravel()])))) * 1.04 if np.isfinite(X).any() else 5.0
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xticks([-4, -2, 0, 2, 4]); ax.set_yticks([-4, -2, 0, 2, 4]); ax.tick_params(labelsize=6)
    name = dict((r, t) for r, t, _ in TOPO_MAPS).get(role, role)
    ax.set_title(title or f"{name} ({ft.unit})", fontsize=fontsize)
    if colorbar or cax is not None:
        ticks = levels[:: max(1, len(levels) // 8)]
        cb = ax.figure.colorbar(pm, cax=cax, ticks=ticks) if cax is not None else ax.figure.colorbar(pm, ax=ax, fraction=0.05, pad=0.03, ticks=ticks)
        cb.ax.tick_params(labelsize=6); cb.set_label(ft.unit, fontsize=6)
    return pm


def best_fit_sphere(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float, float, float]:
    """Esfera de melhor ajuste (mínimos quadrados lineares): devolve (a, b, c, R) com centro (a, b, c)."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]
    A = np.column_stack([2 * x, 2 * y, 2 * z, np.ones_like(x)])
    b = x**2 + y**2 + z**2
    (a, bb, c, d), *_ = np.linalg.lstsq(A, b, rcond=None)
    R = float(np.sqrt(max(d + a**2 + bb**2 + c**2, 0.0)))
    return float(a), float(bb), float(c), R


def plot_elevation_bfs(ax: Axes, exam: NormalizedExam, title: Optional[str] = None, colorbar: bool = True,
                       zones: tuple[float, ...] = (3.0, 6.0), cax=None, fontsize: float = 7.5):
    """Elevação (µm) em relação à esfera de melhor ajuste (EMA) sobre os pontos usados: + = acima da esfera (anterior)."""
    s = exam.samples
    usable = (s.point_state == PointState.MEASURED) | (s.point_state == PointState.INTERPOLATED)
    x, y, z = np.where(usable, s.x, np.nan), np.where(usable, s.y, np.nan), np.where(usable, s.z, np.nan)
    a, b, c, R = best_fit_sphere(x, y, z)
    # z da esfera (ramo anterior: z cresce para trás; a córnea fica "à frente" do centro)
    zs = c - np.sqrt(np.maximum(R**2 - (x - a) ** 2 - (y - b) ** 2, 0.0))
    elev = (zs - z) * 1000.0            # + = ponto mais anterior (menor z) que a esfera
    X, Y, V = _closed_polar_grid(s, elev)
    vmax = float(np.nanpercentile(np.abs(elev[np.isfinite(elev)]), 99)) if np.isfinite(elev).any() else 10.0
    vmax = max(5.0, float(np.ceil(vmax / 5.0) * 5.0))
    pm = ax.pcolormesh(X, Y, V, cmap=DIV_CMAP, vmin=-vmax, vmax=vmax, shading="gouraud", rasterized=True)
    for d in zones:
        _circle(ax, d / 2, color="black", lw=0.4, ls=":", alpha=0.7)
    lim = float(np.nanmax(np.abs(np.concatenate([X.ravel(), Y.ravel()])))) * 1.04 if np.isfinite(X).any() else 5.0
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xticks([-4, -2, 0, 2, 4]); ax.set_yticks([-4, -2, 0, 2, 4]); ax.tick_params(labelsize=6)
    ax.set_title(title or f"Elevação vs. esfera de melhor ajuste (R {R:.2f} mm) (µm)", fontsize=fontsize)
    if colorbar or cax is not None:
        cb = ax.figure.colorbar(pm, cax=cax) if cax is not None else ax.figure.colorbar(pm, ax=ax, fraction=0.05, pad=0.03)
        cb.ax.tick_params(labelsize=6); cb.set_label("µm (+ anterior)", fontsize=6)
    return pm


def hoa_map(wf: WavefrontResult, n: int = 121) -> Optional[np.ndarray]:
    """Mapa das aberrações de alta ordem (n ≥ 3) reconstruído dos coeficientes, na abertura do perfil."""
    from .zernike import ZernikeBasis
    if not wf.coefficients:
        return None
    a = wf.profile.radius_mm
    u = np.linspace(-a, a, n); X, Y = np.meshgrid(u, u)
    rho = np.hypot(X, Y) / a; th = np.arctan2(Y, X)
    inside = rho <= 1.0
    basis = ZernikeBasis(wf.profile.zernike_order)
    D = basis.design(rho[inside], th[inside])
    coef = np.zeros(basis.J)
    for c in wf.coefficients:
        if c.n >= 3 and (c.n, c.m) in basis.terms:
            coef[basis.terms.index((c.n, c.m))] = c.value_um
    Z = np.full(X.shape, np.nan); Z[inside] = D @ coef
    return Z


def _sq(fig: Figure, x: float, y: float, w: float):
    """Eixo quadrado (em polegadas) com canto inferior esquerdo (x, y) e largura w, em frações da página."""
    fw, fh = fig.get_size_inches()
    return fig.add_axes([x, y, w, w * fw / fh])


def _data_pt(iso: Optional[str]) -> str:
    """'2026-08-05T11:56:35' → '05/08/2026 11:56'."""
    if not iso:
        return "não informada"
    d, _, t = iso.partition("T")
    try:
        y, m, dd = d.split("-")
        return f"{dd}/{m}/{y}" + (f" {t[:5]}" if t else "")
    except ValueError:
        return iso


def figure_resultado_completo(exam: NormalizedExam, analysis: Analysis, fig: Optional[Figure] = None,
                              patient_name: Optional[str] = None, view=None) -> Figure:
    """Página única A4 (retrato) com tudo: identificação, índices do TMS, 4 mapas topográficos (axial, tangencial,
    refrativo, elevação), frente de onda (total e alta ordem), coeficientes de Zernike e tabelas de resultados.
    `fig` (opcional) é limpa e reutilizada — usado pelo canvas da interface. `patient_name` (opcional, lido do
    XREF a pedido do usuário) aparece SÓ nesta página impressa — nunca em JSON/CSV. `view` (topo.ViewOptions):
    escala, zona, diâmetro pupilar e centro escolhidos nos "botões" — a página imprime o que está na tela."""
    from . import CORE_VERSION
    if fig is None:
        fig = Figure(figsize=(8.27, 11.69), dpi=100)
    else:
        fig.clear()
        fig.set_size_inches(8.27, 11.69)
    ident, g = analysis.identity, analysis.geometry
    ix = exam.bundle.export_settings.get("tms_indices", {}) or {}

    def iv(k, fmt="{:.2f}"):
        v = ix.get(k)
        return fmt.format(v) if isinstance(v, (int, float)) and np.isfinite(v) else "—"

    # ---- cabeçalho
    cap = f" · captura {ident.capture_index}" if ident.capture_index is not None else ""
    fig.text(0.04, 0.978, "Aberrometria corneana anterior calculada — resultado completo", fontsize=12.5, weight="bold", va="top")
    fig.text(0.97, 0.978, f"tmsab {CORE_VERSION} · impresso {_data_pt(analysis.created_at)}", fontsize=7, va="top", ha="right", color="#444")
    dev = analysis.device.model.replace(" (presumido pelo formato)", "")
    if patient_name:
        fig.text(0.04, 0.9585, f"Paciente: {patient_name}", fontsize=10, weight="bold", va="top")
        fig.text(0.97, 0.9585, f"Data do exame: {_data_pt(ident.acquired_at)}   ·   Olho {ident.eye or '?'}{cap}", fontsize=9.5, weight="bold", va="top", ha="right")
        fig.text(0.04, 0.9415, f"Exame {ident.exam_id}  ·  código {ident.study_id}  ·  {dev}  ·  "
                 f"TMS: SimK1 {iv('SimK1')} D @ {iv('SimK1 Ang.', '{:.0f}')}°  SimK2 {iv('SimK2')} D @ {iv('SimK2 Ang.', '{:.0f}')}°  "
                 f"CYL {iv('CYL')} D  SRI {iv('SRI')}  SAI {iv('SAI')}  Q {iv('Q')}" if ix else
                 f"Exame {ident.exam_id}   ·   código do estudo {ident.study_id}   ·   {dev}   ·   índices do TMS: não disponíveis (sem XREF)",
                 fontsize=7.3, va="top", color="#222")
    else:
        fig.text(0.04, 0.957, f"Código do estudo {ident.study_id}   ·   Exame {ident.exam_id}   ·   Olho {ident.eye or '?'}{cap}   ·   "
                 f"Data do exame {_data_pt(ident.acquired_at)}   ·   {dev}", fontsize=8.5, va="top")
    if not patient_name:
        tms_line = (f"Índices do TMS (XREF): SimK1 {iv('SimK1')} D @ {iv('SimK1 Ang.', '{:.0f}')}°   SimK2 {iv('SimK2')} D @ {iv('SimK2 Ang.', '{:.0f}')}°   "
                    f"CYL {iv('CYL')} D   SRI {iv('SRI')}   SAI {iv('SAI')}   Q {iv('Q')}   Es {iv('Es')}" if ix else "Índices do TMS: não disponíveis (sem XREF)")
        fig.text(0.04, 0.940, tms_line, fontsize=7.5, va="top", color="#222")
    if analysis.synthetic:
        fig.text(0.5, 0.925, "*** EXAME SINTÉTICO — gerado por software; não corresponde a um paciente ***", fontsize=8, color="#a00", ha="center", va="top", weight="bold")

    # ---- linha 1: mapa axial no estilo do TMS (grande, escala Standard, meridianos, zonas, pupila) + 4 mapas pequenos
    from .topo import ViewOptions, draw_tms_legend, draw_tms_map, select_profile
    vo = view or ViewOptions()
    y1, w1 = 0.60, 0.44                       # mapa grande: quadrado de 0,44 da largura
    h1 = w1 * 8.27 / 11.69
    axb = fig.add_axes([0.03, y1, w1, h1])
    big = ViewOptions(kind="axial", scale=vo.scale, zone_mm=vo.zone_mm, pupil_mm=vo.pupil_mm, center=vo.center, annotate=True, protractor=True)
    pm_big, levels_big, _ = draw_tms_map(axb, exam, analysis, big, title=False, fontsize=6.3)
    axl = fig.add_axes([0.04, y1 - 0.028, 0.42, 0.022])
    if pm_big is not None:
        draw_tms_legend(axl, levels_big, big.scale, "D", fontsize=4.6)
    else:
        axl.set_axis_off()
    fig.text(0.03, y1 + h1 + 0.004, "TOPOGRAFIA — potência axial, escala " + ("Standard do TMS (40,5–53,0 D, 0,5 D)" if big.scale == "tms" else "ajustada ao exame")
             + (f", zona {big.zone_mm:g} mm" if big.zone_mm else "") + "; meridianos íngreme/plano com as potências a 3/5/7 mm; pupila tracejada",
             fontsize=6.3, weight="bold", color="#333", va="bottom")
    # 2×2 à direita: tangencial, refrativa, elevação, frente de onda total (diâmetro pupilar escolhido)
    sw = 0.2; sh = sw * 8.27 / 11.69
    grid = [("tangencial", 0.52, y1 + h1 - sh - 0.018), ("refrativa", 0.775, y1 + h1 - sh - 0.018), ("elevacao", 0.52, y1 - 0.004), ("frente_onda", 0.775, y1 - 0.004)]
    for kind, x, y in grid:
        ax = fig.add_axes([x, y, sw, sh])
        o_ = ViewOptions(kind=kind, scale=vo.scale, zone_mm=vo.zone_mm, pupil_mm=vo.pupil_mm, center=vo.center, annotate=False, protractor=False)
        pm, lv, _ = draw_tms_map(ax, exam, analysis, o_, title=True, fontsize=5.6)
        if pm is not None:
            cax = fig.add_axes([x + sw + 0.004, y + 0.01, 0.008, sh - 0.02])
            cb = fig.colorbar(pm, cax=cax); cb.ax.tick_params(labelsize=4.8)
            if kind in ("tangencial", "refrativa") and lv is not None and len(lv) > 10:
                cb.set_ticks(lv[::4])

    # ---- linha 2: alta ordem + Zernike do perfil escolhido (diâmetro pupilar / centro)
    wf = select_profile(analysis, vo) if analysis.wavefronts else None
    y2 = 0.415; w2 = 0.19; h2 = w2 * 8.27 / 11.69
    ax2 = fig.add_axes([0.06, y2, w2, h2]); cax2 = fig.add_axes([0.06 + w2 + 0.005, y2 + 0.01, 0.009, h2 - 0.02])
    ax3 = fig.add_axes([0.335, y2, 0.275, h2])
    if wf is not None and wf.map_grid is not None:
        a = wf.profile.radius_mm
        Zh = hoa_map(wf)
        if Zh is not None:
            vl2 = max(float(np.nanmax(np.abs(Zh))), 1e-3)
            im2 = ax2.imshow(Zh, origin="lower", extent=[-a, a, -a, a], cmap=DIV_CMAP, vmin=-vl2, vmax=vl2, interpolation="bilinear")
            _circle(ax2, a, color="black", lw=0.8); ax2.set_aspect("equal")
            ax2.set_title(f"Alta ordem (n ≥ 3) — {wf.profile.diameter_mm:g} mm, {wf.profile.center_mode} — RMS {wf.metrics.get('rms_hoa_n3plus', float('nan')):.3f} µm", fontsize=6)
            cb2 = fig.colorbar(im2, cax=cax2); cb2.ax.tick_params(labelsize=5); cb2.set_label("µm", fontsize=5)
            ax2.tick_params(labelsize=5); ax2.set_xlabel("x (mm) rel. ao centro", fontsize=5); ax2.set_ylabel("y (mm)", fontsize=5)
        plot_zernike_bars(ax3, wf, title=f"Coeficientes de Zernike OSA (n ≥ 2, µm) — {wf.profile.diameter_mm:g} mm, centro {wf.profile.center_mode}", annotate=False)
        ax3.tick_params(labelsize=5); ax3.yaxis.label.set_size(5.5); ax3.title.set_size(6)
    else:
        for ax in (ax2, ax3):
            ax.text(0.5, 0.5, "frente de onda indisponível\n(ver motivos na tabela)", ha="center", va="center", fontsize=6.5, transform=ax.transAxes)
            ax.set_axis_off()
        cax2.set_axis_off()

    # ---- linha 3: geometria/pupila/astigmatismo (esquerda) e tabela de perfis (direita)
    geo_lines = [
        ("GEOMETRIA RECONSTRUÍDA", True),
        (f"R = {g.conic_R_mm:.3f} mm   Q = {g.conic_Q:+.3f}   K eq. {g.k_mean_D_keratometric:.2f} D", False),
        (f"resíduo RMS {g.residual_rms_mm*1000:.2f} µm (máx {g.residual_max_mm*1000:.2f}); ordem {g.surface_order}", False),
        (f"{g.n_points_used} pts; raio medido {g.outer_radius_mm:.2f} mm; lacuna {g.central_gap_mm:.2f} mm", False),
        ("", False),
        ("PUPILA (PUP)", True),
    ]
    p = analysis.pupil
    if p.state == "registrada":
        geo_lines += [(f"centro ({p.center_x:+.2f}, {p.center_y:+.2f}) mm; diâmetro {p.diameter:.2f} mm" if p.diameter else f"centro ({p.center_x:+.2f}, {p.center_y:+.2f}) mm", False)]
    else:
        geo_lines += [("não detectada nesta captura — perfis pupilares indisponíveis", False)]
    geo_lines += [("", False), ("ASTIGMATISMO (refrativo equiv., centro no vértice)", True)]
    for w in analysis.wavefronts:
        if w.validity == "disponivel" and w.profile.center_mode == "vertice" and w.metrics.get("cilindro_equivalente_D") is not None:
            ax_ = w.metrics["eixo_equivalente_deg"]
            geo_lines.append((f"{w.profile.diameter_mm:g} mm: {w.metrics['cilindro_equivalente_D']:+.2f} D × {ax_:.0f}° (mais curvo {(ax_ - 90) % 180:.0f}°)", False))
    if ix:
        geo_lines.append((f"TMS SimK: CYL {iv('CYL')} D, meridiano mais curvo {iv('SimK1 Ang.', '{:.0f}')}°", False))
    y = 0.552
    for ln, bold in geo_lines:
        fig.text(0.635, y, ln, fontsize=5.9, va="top", weight="bold" if bold else "normal", color="#111" if bold else "#222")
        y -= 0.0112
    axt = fig.add_axes([0.045, 0.175, 0.935, 0.15]); axt.set_axis_off()
    cols = ["Perfil", "Estado", "SA c(4,0)", "|Coma|", "|Trefoil|", "RMS HOA (n≥3)", "SA 2ª c(6,0)", "Cil. equiv. × eixo", "Defocus c(2,0)", "Raios"]
    rows, reasons = [], []
    for w in analysis.wavefronts:
        pf = f"{w.profile.center_mode} {w.profile.diameter_mm:g} mm"
        if w.validity != "disponivel":
            rows.append([pf, w.validity.upper() + "*", "—", "—", "—", "—", "—", "—", "—", "—"])
            reasons.append(f"* {pf}: " + "; ".join(w.block_reasons))
            continue
        m = w.metrics
        cil = m.get("cilindro_equivalente_D"); eixo = m.get("eixo_equivalente_deg")
        rows.append([pf, "DISPONÍVEL", f"{m['sa_c40']:+.3f}", f"{m['coma_magnitude']:.3f}", f"{m['trefoil_magnitude']:.3f}",
                     f"{m['rms_hoa_n3plus']:.3f}", f"{m.get('sa_secundaria_c60', float('nan')):+.3f}",
                     (f"{cil:+.2f} D × {eixo:.0f}°" if cil is not None and np.isfinite(cil) else "—"),
                     f"{m.get('defocus_c20', float('nan')):+.2f}", f"{w.n_valid_rays}/{w.n_rays}"])
    fig.text(0.045, 0.332, "RESULTADOS POR PERFIL — diâmetros pupilares 3/4/5/6 mm e pupila medida, centro no vértice e na pupila (µm; Zernike OSA ortonormal)", fontsize=6.3, weight="bold", color="#333", va="bottom")
    if rows:
        tb = axt.table(cellText=rows, colLabels=cols, loc="upper center", cellLoc="center",
                       colWidths=[0.11, 0.11, 0.08, 0.08, 0.08, 0.1, 0.09, 0.14, 0.1, 0.09])
        tb.auto_set_font_size(False); tb.set_fontsize(5.6); tb.scale(1.0, 1.0)
        for (r_, c_), cell in tb.get_celld().items():
            cell.set_linewidth(0.4)
            if r_ == 0:
                cell.set_facecolor("#eef1f7"); cell.set_text_props(weight="bold", fontsize=5.8)
            elif c_ == 1:
                cell.set_text_props(color="#1b5e20" if rows[r_ - 1][1] == "DISPONÍVEL" else "#8B0000", weight="bold")
    yy = 0.168
    seen = set()
    for rs in reasons:
        key = rs.split(": ", 1)[-1]
        if key in seen:
            continue
        seen.add(key)
        if len(seen) > 2:
            break
        fig.text(0.045, yy, ("* " + key)[:170], fontsize=5.6, va="top", color="#8B0000"); yy -= 0.011
    fig.text(0.045, yy, "Cilindro refrativo equivalente (n = 1,376) a partir de c(2,±2) — não é ceratometria; eixo = meridiano do cilindro negativo. "
             "RMS HOA = n ≥ 3. Limites de disponibilidade provisórios (não clínicos); nenhuma faixa de normalidade é atribuída.",
             fontsize=5.8, va="top", color="#333", wrap=True); yy -= 0.026
    for wmsg in [w for w in analysis.warnings if "perfil verificado" not in w][:3]:
        fig.text(0.045, yy, ("AVISO: " + wmsg)[:175], fontsize=5.6, va="top", color="#8a6d00"); yy -= 0.012

    # ---- rodapé
    hashes = "  ".join(f"{k.split(':')[0]}:{v[:8]}" for k, v in list(analysis.bundle_hashes.items())[:6])
    fig.text(0.045, 0.030, f"Análise {analysis.analysis_id}  ·  arquivos {hashes}  ·  " + ", ".join(f"{k}={v}" for k, v in list(analysis.versions.items())[:3]),
             fontsize=5.4, color="#444", va="bottom")
    fig.text(0.045, 0.010, "Ferramenta de pesquisa — aberrometria corneana ANTERIOR calculada das Data Tables do Tomey TMS; não é aberrometria ocular total, "
             "não é alegação diagnóstica, seleção de LIO nem planejamento terapêutico.", fontsize=5.4, color="#555", va="bottom")
    return fig
