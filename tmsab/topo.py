"""Topografia no estilo do TMS (Tomey): mapa de potência com a escala "Standard" (40,5–53,0 D, passos de 0,5 D e a
paleta do aparelho) ou ajustada ao exame no passo escolhido (0,125 a 1,00 D), transferidor angular, meridianos
íngreme/plano com as potências nas zonas de 3/5/7 mm, caixa "Astig Stats / Power Axis", pupila (medida ou diâmetro
escolhido) e filtros de visualização (tipo de mapa, escala, zona, diâmetro pupilar). Os valores vêm das tabelas
exportadas (DIO/ROC/POW/HIT) — nada é recalculado.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .model import Analysis, NormalizedExam, PointState, WavefrontResult

# Paleta "Standard" do TMS (26 faixas de 0,5 D de 40,5 a 53,0 D), amostrada da legenda impressa pelo TmsSw.
TMS_STANDARD_COLORS = [
    "#000030", "#000050", "#000070", "#00009F", "#0000C0", "#0100FC", "#00609C", "#00906C", "#01C03C", "#00FC01",
    "#B0FD01", "#FBFB00", "#FCBD00", "#FC8600", "#FB5C00", "#FC0B00", "#FE3C3C", "#FC5C5C", "#FC7C7B", "#FC8B8B",
    "#FC9C9D", "#FCACAD", "#FCBCBC", "#FCCCCC", "#FCDCDD", "#FCFCFC",
]
TMS_STANDARD_LEVELS = np.round(np.arange(40.5, 53.5 + 0.25, 0.5), 2)       # 27 limites: faixa i = [L_i, L_i+0,5); rótulo = L_i
TMS_CMAP = matplotlib.colors.ListedColormap(TMS_STANDARD_COLORS, name="tms_standard")
TMS_NORM = matplotlib.colors.BoundaryNorm(TMS_STANDARD_LEVELS, TMS_CMAP.N, clip=True)
# escala ajustada ao exame: passos (D) dos botões e número de faixas — no mínimo 8 (4 D com 0,5 D, como antes), no
# máximo as 26 cores da paleta; além disso a janela de 26 faixas fica centrada na mediana do exame
PASSOS_ESCALA_D = (0.125, 0.25, 0.5, 0.75, 1.0)
FAIXAS_AJUSTADA_MIN, FAIXAS_AJUSTADA_MAX = 8, len(TMS_STANDARD_COLORS)

MAP_KINDS = {
    "axial": ("DIO", "Potência axial", "D"),
    "tangencial": ("ROC", "Potência tangencial", "D"),
    "refrativa": ("POW", "Potência refrativa", "D"),
    "elevacao": ("HIT", "Elevação vs. esfera", "µm"),
    "frente_onda": (None, "W total", "µm"),
    "alta_ordem": (None, "Alta ordem (n ≥ 3)", "µm"),
}
MAP_LABELS_PT = {"axial": "Axial", "tangencial": "Tangencial", "refrativa": "Refrativa", "elevacao": "Elevação",
                 "frente_onda": "Frente de onda", "alta_ordem": "Alta ordem"}
ZONES_MM = (3.0, 5.0, 7.0)


@dataclass
class ViewOptions:
    """Filtros de visualização (os "botões")."""
    kind: str = "axial"                 # axial | tangencial | refrativa | elevacao | frente_onda | alta_ordem
    scale: str = "tms"                  # tms (Standard 40,5–53 D, sempre 0,5 D) | auto (ajustada ao exame, no passo `step`)
    step: float = 0.5                   # passo da escala ajustada (D): um de PASSOS_ESCALA_D
    zone_mm: Optional[float] = None     # limitar o mapa a um diâmetro (3/5/7/9 mm); None = tudo
    pupil_mm: Optional[float] = None    # diâmetro pupilar a desenhar/usar; None = pupila medida (PUP)
    center: str = "vertice"             # centro do perfil de frente de onda: vertice | pupila
    annotate: bool = True               # meridianos, potências por zona, caixas
    protractor: bool = True


@dataclass
class ZoneAxisStats:
    zone_mm: float
    rows: list[tuple[float, float]] = field(default_factory=list)   # (potência D, semimeridiano °) na ordem do TMS
    steep_deg: float = float("nan")
    flat_deg: float = float("nan")
    steep_power: float = float("nan")
    flat_power: float = float("nan")


# ---------------------------------------------------------------------------
# potência ao longo de um círculo de diâmetro dado (interpolação radial por ângulo)
# ---------------------------------------------------------------------------
def power_on_circle(exam: NormalizedExam, role: str, diameter_mm: float) -> np.ndarray:
    """Valor da tabela `role` no raio diameter/2 para cada ângulo (interpolação linear em r por semimeridiano);
    NaN onde não há cobertura."""
    s = exam.samples
    ft = s.fields.get(role)
    if ft is None:
        return np.full(s.n_angles, np.nan)
    vals = np.where(ft.usable_mask(), ft.values, np.nan)
    rr = s.r
    out = np.full(s.n_angles, np.nan)
    rt = diameter_mm / 2.0
    for k in range(s.n_angles):
        r_k, v_k = rr[:, k], vals[:, k]
        ok = np.isfinite(r_k) & np.isfinite(v_k)
        if ok.sum() < 2 or rt < np.nanmin(r_k[ok]) or rt > np.nanmax(r_k[ok]):
            continue
        out[k] = np.interp(rt, r_k[ok], v_k[ok])
    return out


def _smooth_circular(v: np.ndarray, width: int = 9) -> np.ndarray:
    ok = np.isfinite(v)
    if ok.sum() < width:
        return v
    vv = np.where(ok, v, np.nanmean(v))
    k = np.ones(width) / width
    sm = np.convolve(np.concatenate([vv[-width:], vv, vv[:width]]), k, mode="same")[width:-width]
    return np.where(ok, sm, np.nan)


def zone_axis_stats(exam: NormalizedExam, diameter_mm: float, role: str = "DIO") -> ZoneAxisStats:
    """"Astig Stats / Power Axis" do TMS para uma zona: potência e semimeridiano dos dois máximos (íngreme) e dos
    dois mínimos (plano) ao longo do círculo — cada metade procurada separadamente, como no aparelho."""
    s = exam.samples
    th = np.asarray(s.theta_deg) % 360.0
    p = _smooth_circular(power_on_circle(exam, role, diameter_mm), 7)
    st = ZoneAxisStats(zone_mm=diameter_mm)
    if np.isfinite(p).sum() < s.n_angles * 0.5:
        return st
    # ajuste cos(2θ) para o eixo global (robusto), depois extremos locais por metade
    ok = np.isfinite(p)
    A = np.column_stack([np.ones(ok.sum()), np.cos(2 * np.radians(th[ok])), np.sin(2 * np.radians(th[ok]))])
    (c0, c1, c2), *_ = np.linalg.lstsq(A, p[ok], rcond=None)
    steep = (0.5 * np.degrees(np.arctan2(c2, c1))) % 180.0
    flat = (steep + 90.0) % 180.0
    st.steep_deg, st.flat_deg = float(steep), float(flat)
    st.steep_power = float(c0 + np.hypot(c1, c2)); st.flat_power = float(c0 - np.hypot(c1, c2))

    def local_extreme(center_deg: float, maximum: bool, halfwidth: float = 40.0) -> tuple[float, float]:
        d = (th - center_deg + 180.0) % 360.0 - 180.0
        sel = ok & (np.abs(d) <= halfwidth)
        if not sel.any():
            return float("nan"), float("nan")
        idx = np.where(sel)[0]
        j = idx[np.nanargmax(p[idx])] if maximum else idx[np.nanargmin(p[idx])]
        return float(p[j]), float(th[j])

    rows = [local_extreme(flat, False), local_extreme(steep, True), local_extreme(flat + 180.0, False), local_extreme(steep + 180.0, True)]
    # ordem do TMS: por ângulo crescente a partir de 180°
    rows.sort(key=lambda t: ((t[1] - 180.0) % 360.0) if np.isfinite(t[1]) else 1e9)
    st.rows = rows
    return st


# ---------------------------------------------------------------------------
# desenho
# ---------------------------------------------------------------------------
def _grid_closed(s, values: np.ndarray, fill_center: bool = True):
    """Grade polar fechada em 360° para pcolormesh. Pontos sem coordenada recebem a posição nominal do anel; o
    centro (lacuna do anel 1, ~0,2 mm) é preenchido com a média angular do primeiro anel, como faz o TMS."""
    rn = np.nanmean(s.r, axis=1)
    rr = np.where(np.isfinite(s.r), s.r, rn[:, None])
    V = np.where(np.isfinite(values) & np.isfinite(rr), values, np.nan)
    if fill_center and V.shape[0] > 0 and np.isfinite(V[0]).any():
        rr = np.concatenate([np.zeros((1, rr.shape[1])), rr], axis=0)
        V = np.concatenate([np.full((1, V.shape[1]), np.nanmean(V[0])), V], axis=0)
    th = np.radians(s.theta_deg)[None, :]
    X, Y = rr * np.cos(th), rr * np.sin(th)
    X = np.concatenate([X, X[:, :1]], axis=1); Y = np.concatenate([Y, Y[:, :1]], axis=1); V = np.concatenate([V, V[:, :1]], axis=1)
    return X, Y, np.ma.masked_invalid(V)


def passo_valido(step) -> float:
    """O passo pedido, se for um dos botões (PASSOS_ESCALA_D); senão 0,5 D."""
    try:
        s = float(step)
    except (TypeError, ValueError):
        return 0.5
    return next((p for p in PASSOS_ESCALA_D if abs(p - s) < 1e-9), 0.5)


def escala_ajustada(v: np.ndarray, step: float = 0.5) -> tuple[np.ndarray, bool]:
    """Limites das faixas da escala ajustada ao exame e se ela corta os extremos.

    Faixas de `step` D em múltiplos exatos do passo, do percentil 1 ao 99 dos valores; no mínimo 8 faixas (centradas
    nos dados) e no máximo 26 (as cores da paleta do TMS). Quando seriam necessárias mais de 26 (passo fino ou exame
    com muita variação, como no ceratocone), a janela de 26 faixas fica centrada na mediana e os valores fora dela saem
    nas cores das pontas (`cortada` = True)."""
    s = passo_valido(step)
    ok = np.isfinite(v)
    if not ok.any():
        return np.round(np.arange(FAIXAS_AJUSTADA_MIN + 1) * s, 6), False
    p1, p99 = np.nanpercentile(v[ok], [1, 99])
    k0, k1 = math.floor(p1 / s + 1e-9), math.ceil(p99 / s - 1e-9)
    n, cortada = max(k1 - k0, 1), False
    if n < FAIXAS_AJUSTADA_MIN:
        k0 -= (FAIXAS_AJUSTADA_MIN - n) // 2
        n = FAIXAS_AJUSTADA_MIN
    elif n > FAIXAS_AJUSTADA_MAX:
        k0 = int(round(float(np.nanmedian(v[ok])) / s - FAIXAS_AJUSTADA_MAX / 2))
        n, cortada = FAIXAS_AJUSTADA_MAX, True
    return np.round((k0 + np.arange(n + 1)) * s, 6), cortada


def _valores_do_mapa(exam: NormalizedExam, o: ViewOptions) -> Optional[np.ndarray]:
    """Valores de potência (D) que o mapa mostra (tabela do tipo de mapa, só os utilizáveis, recortados à zona)."""
    if o.kind not in ("axial", "tangencial", "refrativa"):
        return None
    s = exam.samples
    ft = s.fields.get(MAP_KINDS[o.kind][0])
    vals = np.where(ft.usable_mask(), ft.values, np.nan) if ft is not None else np.full_like(s.r, np.nan)
    if o.zone_mm:
        vals = np.where(s.r <= o.zone_mm / 2.0 + 1e-9, vals, np.nan)
    return vals


def escala_do_mapa(exam: NormalizedExam, opts: Optional[ViewOptions] = None) -> tuple[Optional[np.ndarray], bool]:
    """(limites das faixas, cortada) da escala de cores de um mapa de potência — a Standard do TMS (fixa) ou a ajustada
    ao exame no passo escolhido; (None, False) para elevação e frente de onda (escala contínua)."""
    o = opts or ViewOptions()
    vals = _valores_do_mapa(exam, o)
    if vals is None:
        return None, False
    if o.scale == "tms":
        return TMS_STANDARD_LEVELS, False
    return escala_ajustada(vals, o.step)


def norma_da_escala(levels: np.ndarray) -> matplotlib.colors.BoundaryNorm:
    """Faixa → cor: com 26 faixas, uma cor da paleta por faixa; com menos, as cores espalhadas pela paleta inteira
    (a mesma conta no mapa e na legenda)."""
    return TMS_NORM if levels is TMS_STANDARD_LEVELS else matplotlib.colors.BoundaryNorm(levels, TMS_CMAP.N, clip=True)


def _protractor(ax: Axes, R: float, fontsize: float = 6.5):
    """Transferidor do TMS: traços a cada 5°, rótulos a cada 15° com o meridiano (0–180 nas duas metades)."""
    for a in range(0, 360, 5):
        t = np.radians(a)
        r0 = R * (1.045 if a % 15 == 0 else 1.03)
        ax.plot([R * 1.015 * np.cos(t), r0 * np.cos(t)], [R * 1.015 * np.sin(t), r0 * np.sin(t)], color="#333", lw=0.5)
        if a % 15 == 0:
            lab = a if a <= 180 else a - 180
            ax.text(R * 1.13 * np.cos(t), R * 1.13 * np.sin(t), f"{lab:d}", ha="center", va="center", fontsize=fontsize, color="#222")
    ax.add_patch(matplotlib.patches.Circle((0, 0), R * 1.015, fill=False, lw=0.6, color="#333"))


def draw_tms_map(ax: Axes, exam: NormalizedExam, analysis: Optional[Analysis] = None, opts: Optional[ViewOptions] = None,
                 title: bool = True, fontsize: float = 6.5):
    """Mapa no estilo do TMS com os filtros de `opts`. Devolve (mappable, levels, stats por zona)."""
    o = opts or ViewOptions()
    s = exam.samples
    kind = o.kind if o.kind in MAP_KINDS else "axial"
    role, name, unit = MAP_KINDS[kind]
    R_meas = float(np.nanmax(s.r)) if np.isfinite(s.r).any() else 4.5
    R = min(R_meas, o.zone_mm / 2.0) if o.zone_mm else R_meas
    pm, levels, stats = None, None, []

    if kind in ("axial", "tangencial", "refrativa"):
        vals = _valores_do_mapa(exam, ViewOptions(kind=kind, zone_mm=o.zone_mm))
        X, Y, V = _grid_closed(s, vals)
        levels, _ = escala_do_mapa(exam, ViewOptions(kind=kind, scale=o.scale, step=o.step, zone_mm=o.zone_mm))
        pm = ax.pcolormesh(X, Y, V, cmap=TMS_CMAP, norm=norma_da_escala(levels), shading="gouraud", rasterized=True)
    elif kind == "elevacao":
        from .plots import best_fit_sphere
        usable = (s.point_state == PointState.MEASURED) | (s.point_state == PointState.INTERPOLATED)
        x, y, z = np.where(usable, s.x, np.nan), np.where(usable, s.y, np.nan), np.where(usable, s.z, np.nan)
        a, b, c, Rs = best_fit_sphere(x, y, z)
        zs = c - np.sqrt(np.maximum(Rs**2 - (x - a) ** 2 - (y - b) ** 2, 0.0))
        elev = (zs - z) * 1000.0
        if o.zone_mm:
            elev = np.where(s.r <= o.zone_mm / 2.0 + 1e-9, elev, np.nan)
        X, Y, V = _grid_closed(s, elev)
        vmax = max(5.0, float(np.ceil((np.nanpercentile(np.abs(elev[np.isfinite(elev)]), 99) if np.isfinite(elev).any() else 10.0) / 5.0) * 5.0))
        pm = ax.pcolormesh(X, Y, V, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="gouraud", rasterized=True)
        levels = np.linspace(-vmax, vmax, 9)
        name = f"{name} R {Rs:.2f}"
    else:   # frente de onda / alta ordem de um perfil
        wf = select_profile(analysis, o) if analysis is not None else None
        if wf is None or wf.map_grid is None:
            ax.text(0.5, 0.5, "frente de onda indisponível\npara este diâmetro/centro", ha="center", va="center", fontsize=fontsize, transform=ax.transAxes)
            ax.set_axis_off()
            return None, None, []
        a_ = wf.profile.radius_mm
        if kind == "frente_onda":
            Z = wf.map_grid["W"]
        else:
            from .plots import hoa_map
            Z = hoa_map(wf)
        vl = max(float(np.nanmax(np.abs(Z))), 1e-3)
        pm = ax.imshow(Z, origin="lower", extent=[-a_, a_, -a_, a_], cmap="RdBu_r", vmin=-vl, vmax=vl, interpolation="bilinear")
        levels = np.linspace(-vl, vl, 9)
        R = a_
        ax.plot([wf.center_xy[0]], [wf.center_xy[1]], marker="+", color="black", ms=6, mew=0.8)
        name = f"{name} ({unit}) — {wf.profile.diameter_mm:g} mm, {('vértice' if wf.profile.center_mode == 'vertice' else 'pupila')}"
        unit = None

    # anotações comuns
    lim = R * (1.22 if o.protractor else 1.05)
    boxes = o.annotate and kind in ("axial", "tangencial", "refrativa")
    ax.set_xlim(-lim * (1.42 if boxes else 1.0), lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.set_axis_off()
    if o.protractor and kind not in ("frente_onda", "alta_ordem"):
        _protractor(ax, R, fontsize)
    ax.plot([0], [0], marker="+", color="black", ms=7, mew=0.8)
    # pupila: medida (PUP) ou diâmetro escolhido centrado na pupila medida (ou no vértice)
    cx = exam.pupil.center_x if exam.pupil.state == "registrada" and exam.pupil.center_x is not None else 0.0
    cy = exam.pupil.center_y if exam.pupil.state == "registrada" and exam.pupil.center_y is not None else 0.0
    pd = o.pupil_mm if o.pupil_mm else (exam.pupil.diameter if exam.pupil.state == "registrada" else None)
    if pd and kind not in ("frente_onda", "alta_ordem"):
        if o.pupil_mm is None and getattr(exam.pupil, "edge_r", None) is not None:
            th = np.radians(s.theta_deg); re_ = np.asarray(exam.pupil.edge_r, float)
            ax.plot(re_ * np.cos(th), re_ * np.sin(th), color="black", lw=0.9, ls=(0, (1.5, 1.5)))
        else:
            ax.add_patch(matplotlib.patches.Circle((cx, cy), pd / 2.0, fill=False, lw=0.9, ls=(0, (1.5, 1.5)), color="black"))
    if o.annotate and kind in ("axial", "tangencial", "refrativa"):
        stats = [zone_axis_stats(exam, d, role) for d in ZONES_MM if d / 2.0 <= R + 1e-6]
        _draw_axes_and_labels(ax, exam, role, stats, R, fontsize)
        _draw_stats_box(ax, stats, R, lim * 1.42, fontsize)
        _draw_pupil_box(ax, exam, pd, cx, cy, lim * 1.42, fontsize)
    if title:
        ax.set_title((f"{name} ({unit})" if unit else name) + (f" — zona {o.zone_mm:g} mm" if o.zone_mm and kind not in ("frente_onda", "alta_ordem") else ""), fontsize=fontsize + 1.5)
    return pm, levels, stats


def _draw_axes_and_labels(ax: Axes, exam: NormalizedExam, role: str, stats: list[ZoneAxisStats], R: float, fontsize: float):
    """Meridianos íngreme (vermelho) e plano (azul) da maior zona, com a potência de cada zona escrita no ponto."""
    if not stats:
        return
    ref = stats[-1]
    if not np.isfinite(ref.steep_deg):
        return
    for ang, color in ((ref.steep_deg, "#D00000"), (ref.flat_deg, "#1030C0")):
        t = np.radians(ang)
        ax.plot([-R * np.cos(t), R * np.cos(t)], [-R * np.sin(t), R * np.sin(t)], color=color, lw=1.0, alpha=0.85)
    for st in stats:
        for (pw, ang) in st.rows:
            if not np.isfinite(pw):
                continue
            t = np.radians(ang); r = st.zone_mm / 2.0
            ax.text(r * np.cos(t), r * np.sin(t), f"{pw:.2f}", fontsize=fontsize - 1.2, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


def _draw_stats_box(ax: Axes, stats: list[ZoneAxisStats], R: float, lim: float, fontsize: float):
    if not stats:
        return
    lines = ["Astig Stats", "Potência  Eixo"]
    for st in stats:
        lines.append(f"── {st.zone_mm:.1f} mm ──")
        for pw, ang in st.rows:
            lines.append(f"{pw:6.2f}   {ang:3.0f}°" if np.isfinite(pw) else "   —      —")
    ax.text(-lim * 0.995, lim / 1.42 * 0.78, "\n".join(lines), fontsize=fontsize - 1.3, family="monospace", va="top", ha="left",
            bbox=dict(boxstyle="square,pad=0.3", fc="#111", ec="#444", alpha=0.95), color="#eee")


def _draw_pupil_box(ax: Axes, exam: NormalizedExam, pd: Optional[float], cx: float, cy: float, lim: float, fontsize: float):
    if exam.pupil.state == "registrada":
        txt = f"Pupila\nX= {cx:+.2f}\nY= {cy:+.2f}\nD= {pd:.2f}" if pd else f"Pupila\nX= {cx:+.2f}\nY= {cy:+.2f}"
    else:
        txt = "Pupila\nnão\ndetectada" if not pd else f"Pupila\nD= {pd:.2f}\n(no vértice)"
    ax.text(-lim * 0.995, -lim / 1.42 * 0.995, txt, fontsize=fontsize - 1.3, family="monospace", va="bottom", ha="left",
            bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="#333"))


def fmt_passo(step: float) -> str:
    """Passo da escala como nos botões: 0,125 · 0,25 · 0,5 · 0,75 · 1,00."""
    s = passo_valido(step)
    return "1,00" if s == 1.0 else f"{s:g}".replace(".", ",")


def titulo_da_escala(scale: str, step: float = 0.5, cortada: bool = False) -> str:
    """Nome da escala para legendas e cabeçalhos ("Standard do TMS" ou "ajustada ao exame, passo 0,25 D")."""
    if scale == "tms":
        return "Standard do TMS"
    return f"ajustada ao exame, passo {fmt_passo(step)} D" + (" (extremos nas cores das pontas)" if cortada else "")


def draw_tms_legend(ax: Axes, levels: np.ndarray, scale: str = "tms", unit: str = "D", fontsize: float = 5.5,
                    cortada: bool = False):
    """Barra de cores horizontal com uma caixa por faixa e os limites escritos, como na impressão do TMS. As cores vêm
    da mesma norma do mapa (norma_da_escala); os rótulos têm as casas do passo (0,125 → 3) e, quando não cabem lado a
    lado, saem em faixas alternadas."""
    ax.set_axis_off()
    if levels is None:
        return
    n = len(levels) - 1
    if n <= 0:
        return
    norm = norma_da_escala(levels)

    def casas(vals) -> int:            # as casas que os valores pedem (0,125 → 3), no mínimo 1, como no TMS
        return max(1, max(len(f"{v:.3f}".rstrip("0").split(".")[1]) for v in vals))

    from .plots import _largura_pt     # largura medida pela fonte (FreeType): programa, PDF e site quebram igual

    pos = ax.get_position()
    caixa_pt = pos.width * ax.figure.get_figwidth() * 72.0 / max(n + 0.6, 1)
    passo_rot = 1
    while True:                        # casas e largura só dos rótulos que aparecem; ≥ 1,2 pt entre vizinhos
        c = casas(levels[:-1:passo_rot])   # (alternados de 0,125 podem cair em 0,25 e pedir só 2 casas)
        larg_pt = max(_largura_pt(f"{v:.{c}f}", fontsize) for v in levels[:-1:passo_rot])
        if larg_pt + 1.2 <= passo_rot * caixa_pt or passo_rot >= n:
            break
        passo_rot += 1
    rotulos = [f"{v:.{c}f}" for v in levels[:-1]]
    for i in range(n):
        color = TMS_CMAP(int(norm(0.5 * (levels[i] + levels[i + 1]))))
        ax.add_patch(matplotlib.patches.Rectangle((i, 0), 0.94, 1, color=color))
        if i % passo_rot == 0:
            ax.text(i + 0.47, -0.35, rotulos[i], fontsize=fontsize, ha="center", va="top")   # rótulo = limite inferior, como no TMS
    ax.set_xlim(-0.3, n + 0.3); ax.set_ylim(-1.3, 1.2)
    step = float(levels[1] - levels[0])
    ax.text(n / 2, 1.1, f"escala {titulo_da_escala(scale, step, cortada)} ({unit})", fontsize=fontsize + 0.5, ha="center", va="bottom")


def select_profile(analysis: Analysis, o: ViewOptions) -> Optional[WavefrontResult]:
    """Perfil disponível mais próximo do diâmetro pupilar/centro pedidos."""
    cands = [w for w in analysis.wavefronts if w.validity == "disponivel" and w.map_grid is not None]
    if not cands:
        return None
    # sem diâmetro escolhido: 6 mm no vértice (padrão da página) ou a pupila medida quando o centro é a pupila
    want_c = o.center
    if o.pupil_mm:
        want_d = o.pupil_mm
    elif want_c == "pupila" and analysis.pupil.state == "registrada" and analysis.pupil.diameter:
        want_d = float(analysis.pupil.diameter)
    else:
        want_d = 6.0
    cands.sort(key=lambda w: (w.profile.center_mode != want_c, abs(w.profile.diameter_mm - want_d)))
    return cands[0]


def tms_header_lines(exam: NormalizedExam) -> list[str]:
    """Linha de índices como no cabeçalho do TMS: Ks/Kf/AvgK/MinK/Q/Cyl/SRI/SAI (do XREF)."""
    ix = exam.bundle.export_settings.get("tms_indices", {}) or {}

    def iv(k, fmt="{:.2f}"):
        v = ix.get(k)
        return fmt.format(v) if isinstance(v, (int, float)) and np.isfinite(v) else "—"
    if not ix:
        return ["Índices do TMS: não disponíveis (sem XREF)"]
    avg = (ix["SimK1"] + ix["SimK2"]) / 2.0 if isinstance(ix.get("SimK1"), (int, float)) and isinstance(ix.get("SimK2"), (int, float)) else None
    return [
        f"Ks: {iv('SimK1')} @ {iv('SimK1 Ang.', '{:.0f}')}°   Kf: {iv('SimK2')} @ {iv('SimK2 Ang.', '{:.0f}')}°   AvgK: {avg:.2f}" if avg is not None else f"Ks: {iv('SimK1')}   Kf: {iv('SimK2')}",
        f"MinK: {iv('MinK')} @ {iv('MinK Ang.', '{:.0f}')}°   Q: {iv('Q')}   Cyl: {iv('CYL')}",
        f"SRI: {iv('SRI')}   SAI: {iv('SAI')}   Es: {iv('Es')}",
    ]


def figure_tms_view(exam: NormalizedExam, analysis: Optional[Analysis], opts: Optional[ViewOptions] = None,
                    fig: Optional[Figure] = None, width_in: float = 6.0) -> Figure:
    """Figura completa de uma visualização: cabeçalho com índices, mapa e legenda (para a aba/HTML)."""
    o = opts or ViewOptions()
    if fig is None:
        fig = Figure(figsize=(width_in, width_in * 1.18), dpi=100)
    else:
        fig.clear()          # no canvas da interface o tamanho é o do widget
    ax = fig.add_axes([0.02, 0.17, 0.96, 0.75])
    pm, levels, _ = draw_tms_map(ax, exam, analysis, o, title=True, fontsize=7.5)
    axl = fig.add_axes([0.05, 0.035, 0.9, 0.09])
    unit = MAP_KINDS.get(o.kind, MAP_KINDS["axial"])[2]
    if pm is not None and o.kind in ("axial", "tangencial", "refrativa"):
        draw_tms_legend(axl, levels, o.scale, unit, fontsize=6, cortada=escala_do_mapa(exam, o)[1])
    elif pm is not None:
        axl.set_axis_off()
        cb = fig.colorbar(pm, cax=fig.add_axes([0.2, 0.07, 0.6, 0.03]), orientation="horizontal")
        cb.ax.tick_params(labelsize=6); cb.set_label(unit, fontsize=6)
    else:
        axl.set_axis_off()
    hdr = tms_header_lines(exam)
    fig.text(0.02, 0.985, "   ".join(hdr), fontsize=6.5, va="top", color="#222")
    return fig
