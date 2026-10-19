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
    """Coordenadas fechadas em 360° para pcolormesh; pontos sem coordenada recebem a posição nominal do anel."""
    rn = np.nanmean(s.r, axis=1)
    rr = np.where(np.isfinite(s.r), s.r, rn[:, None])
    th = np.radians(s.theta_deg)[None, :]
    X, Y = rr * np.cos(th), rr * np.sin(th)
    V = np.where(np.isfinite(values) & np.isfinite(rr), values, np.nan)
    X = np.concatenate([X, X[:, :1]], axis=1); Y = np.concatenate([Y, Y[:, :1]], axis=1); V = np.concatenate([V, V[:, :1]], axis=1)
    return X, Y, np.ma.masked_invalid(V)


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


# ---------------------------------------------------------------------------
# Texto medido sem renderizador (quebra de linhas e ajuste de tabelas na página A4)
# ---------------------------------------------------------------------------
_TTP = None
# Folga sobre a largura medida: com hinting, em fontes de 5–7 pt e 72–200 dpi (tela do programa, PNG do HTML) o texto
# desenhado sai até ~11% mais largo que a métrica sem hinting; no PDF (vetorial) sai igual.
_FOLGA = 1.12


def _largura_pt(s: str, fontsize: float, weight: str = "normal", style: str = "normal") -> float:
    """Largura (pt) de uma linha de texto na fonte padrão, medida pelo FreeType (TextToPath): não depende do
    canvas, então programa, PDF e navegador quebram as linhas no mesmo lugar. Estimativa folgada se falhar."""
    global _TTP
    if not s:
        return 0.0
    try:
        from matplotlib.font_manager import FontProperties
        from matplotlib.textpath import TextToPath
        if _TTP is None:
            _TTP = TextToPath()
        w, _, _ = _TTP.get_text_width_height_descent(s, FontProperties(size=fontsize, weight=weight, style=style), ismath=False)
        return float(w)
    except Exception:  # noqa: BLE001
        return 0.62 * fontsize * len(s)


def _quebrar(texto: str, largura_pt: float, fontsize: float, weight: str = "normal") -> list[str]:
    """Quebra o texto em linhas que cabem em `largura_pt` (com a folga de hinting), por palavra; palavra maior que
    a linha é partida. Linha que já cabe fica intacta (espaços preservados)."""
    lim = largura_pt / _FOLGA

    def cabe(t):
        return _largura_pt(t, fontsize, weight) <= lim

    linhas: list[str] = []
    for par in str(texto).split("\n"):
        if cabe(par):
            if par.strip():
                linhas.append(par)
            continue
        atual = ""
        for palavra in par.split():
            cand = f"{atual} {palavra}" if atual else palavra
            if cabe(cand):
                atual = cand
                continue
            if atual:
                linhas.append(atual)
            atual = palavra
            while len(atual) > 1 and not cabe(atual):
                k = len(atual) - 1
                while k > 1 and not cabe(atual[:k]):
                    k -= 1
                linhas.append(atual[:k])
                atual = atual[k:]
        if atual:
            linhas.append(atual)
    return linhas or [""]


def _texto_cabendo(fig: Figure, x: float, y: float, s: str, fontsize: float, max_pt: float, weight: str = "normal",
                   min_fs: float = 5.5, **kw):
    """fig.text com a fonte reduzida (até `min_fs`) para a linha caber em `max_pt` (com a folga de hinting)."""
    w = _largura_pt(s, fontsize, weight) * _FOLGA
    if w > max_pt > 0:
        fontsize = max(min_fs, fontsize * max_pt / w)
    return fig.text(x, y, s, fontsize=fontsize, weight=weight, **kw)


def _momento_local(iso: Optional[str]) -> str:
    """Instante ISO-8601 (UTC) → 'dd/mm/aaaa hh:mm (Brasília)'. Sem a base de fusos horários (o Pyodide e o Python do
    Windows não trazem tzdata) usa UTC−3 fixo (Brasília sem horário de verão desde 2019), escrito por extenso."""
    import datetime as _dt
    if not iso:
        return "não informado"
    try:
        t = _dt.datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except ValueError:
        return _data_pt(iso)
    if t.tzinfo is None:
        t = t.replace(tzinfo=_dt.timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        return t.astimezone(ZoneInfo("America/Sao_Paulo")).strftime("%d/%m/%Y %H:%M") + " (Brasília)"
    except Exception:  # noqa: BLE001 — sem tzdata (navegador, Windows) ou fuso desconhecido
        return t.astimezone(_dt.timezone(_dt.timedelta(hours=-3))).strftime("%d/%m/%Y %H:%M") + " (UTC−3)"


def _sinal(v: float, casas: int = 2) -> str:
    """Número com sinal, sem '-0.00'."""
    s = f"{v:+.{casas}f}"
    return "+" + s[1:] if s.startswith("-") and float(s) == 0 else s


def _eixo(v) -> str:
    """Eixo TABO 1–180 com °, ou '—'."""
    from .keratometry import fmt_eixo
    try:
        return fmt_eixo(None if v is None else float(v))
    except (TypeError, ValueError):
        return "—"


# ---------------------------------------------------------------------------
# Bloco "CERATOMETRIA POR ZONA" (página A4)
# ---------------------------------------------------------------------------
KERATO_GID = "ceratometria"            # gid dos artistas do bloco (rótulo, tabela, legenda) — usado nos testes
_KERATO_FUNDO = {"pupila": "#fff4d6", "zona": "#eef4fb"}


def _kerato_arranjo(linhas: list[dict], cab: list[str], fs: float, w_pt: float):
    """Textos, larguras de coluna e alturas de linha (pt) da tabela na fonte `fs`; None se não couber na largura."""
    hfs = fs - 0.4
    pad = 2.0
    estilos = [("bold" if l.get("destaque") == "pupila" else "normal",
                "italic" if l.get("destaque") == "referencia" else "normal") for l in linhas]
    corpo = [[str(l.get("rotulo", ""))] + [str(t) for t in l.get("texto", ["—"] * 6)] for l in linhas]

    def larg(txt, f, weight="normal", style="normal"):
        return _FOLGA * max(_largura_pt(p, f, weight, style) for p in txt.split("\n"))

    # rótulo muito mais longo que os demais (ex.: "SimK do TMS (anéis 6–8, …)") vai em duas linhas, antes do "("
    simples = [larg(r[0], fs, *e) for r, e in zip(corpo, estilos) if " (" not in r[0]]
    limite = 1.15 * max(simples) if simples else 0.35 * w_pt
    for r, e in zip(corpo, estilos):
        if " (" in r[0] and larg(r[0], fs, *e) > limite:
            i = r[0].index(" (")
            r[0] = r[0][:i] + "\n" + r[0][i + 1:]

    def colunas(cab_t):
        req = []
        for c in range(len(cab_t)):
            wc = larg(cab_t[c], hfs, "bold")
            for r, e in zip(corpo, estilos):
                wc = max(wc, larg(r[c], fs, *e))
            req.append(wc + 2 * pad)
        return req

    cab_t = list(cab)
    req = colunas(cab_t)
    if sum(req) > w_pt:                       # cabeçalho em duas linhas ("K1 plano" / "(D)")
        cab_t = [t.rsplit(" ", 1)[0] + "\n" + t.rsplit(" ", 1)[1] if " " in t else t for t in cab]
        req = colunas(cab_t)
    if sum(req) > w_pt:
        return None
    colw = [v * w_pt / sum(req) for v in req]
    vpad = 1.8
    rowh = [max(t.count("\n") + 1 for t in cab_t) * hfs * 1.25 + 2 * vpad]
    rowh += [max(t.count("\n") + 1 for t in r) * fs * 1.25 + 2 * vpad for r in corpo]
    return cab_t, corpo, colw, rowh, pad


def plot_keratometry_table(fig: Figure, rect, kres, fontsize: float = 6.2) -> Axes:
    """Bloco "CERATOMETRIA POR ZONA" da página A4 dentro de `rect` = [esquerda, base, largura, altura] (frações da
    página): rótulo da seção, tabela (linhas de `linhas_resumo`, colunas `COLUNAS_TABELA`) e legenda
    (`LEGENDA_TABELA`). Pupila com fundo #fff4d6 e negrito; SimK do TMS em cinza itálico. A fonte encolhe (até
    5 pt) se as linhas não couberem. `kres` None (JSON anterior à 0.4.0) → aviso no lugar da tabela.
    Devolve o eixo da tabela; todos os artistas do bloco levam gid KERATO_GID."""
    from matplotlib.table import Table

    from .keratometry import COLUNAS_TABELA, LEGENDA_TABELA, linhas_resumo

    x, y, w, h = (float(v) for v in rect)
    fw, fh = fig.get_size_inches()
    py = 1.0 / (fh * 72.0)                    # 1 pt em fração da altura da página
    w_pt = w * fw * 72.0
    topo = y + h
    fig.text(x, topo, "CERATOMETRIA POR ZONA (D; potência axial do TMS, n 1,3375)", fontsize=6.8, weight="bold", color="#333",
             va="top", gid=KERATO_GID)
    topo -= 11.5 * py
    linhas = linhas_resumo(kres) if kres is not None else []
    if not linhas:
        ax = fig.add_axes([x, y, w, max(topo - y, 0.01)], gid=KERATO_GID)
        ax.set_axis_off()
        if kres is None:
            msg = "ceratometria não calculada nesta versão\n(análise gravada antes da 0.4.0: recalcule o exame)"
        else:
            msg = "\n".join(["ceratometria indisponível"] + (_quebrar(kres.notes[0], w_pt * 0.9, 6.6) if kres.notes else []))
        ax.text(0.5, 0.6, msg, ha="center", va="center", fontsize=6.6, color="#555", transform=ax.transAxes,
                linespacing=1.4, gid=KERATO_GID)
        return ax

    leg_fs = 5.4
    extras = [(ln, "#333") for ln in _quebrar(LEGENDA_TABELA, w_pt, leg_fs)]
    for a in getattr(kres, "alertas", None) or []:          # informativo (ex.: astigmatismo dependente da zona)
        extras += [(ln, "#8a5300") for ln in _quebrar("Alerta (informativo): " + a, w_pt, leg_fs)]
    # motivos das linhas sem valor (a pupila "não registrada" já se explica no rótulo), agrupados, até 2 linhas
    grupos: dict[str, list[str]] = {}
    for lin in linhas:
        if not lin.get("disponivel", True) and lin.get("motivo") and "não registrada" not in str(lin.get("rotulo", "")):
            grupos.setdefault(str(lin["motivo"]), []).append(str(lin.get("rotulo", "")).split(" — ")[0])
    if grupos:
        mot = _quebrar("— sem valor: " + "; ".join(f"{', '.join(r)}: {m}" for m, r in grupos.items()), w_pt, leg_fs)
        if len(mot) > 2:
            ult = mot[1].split()
            while ult and _FOLGA * _largura_pt(" ".join(ult) + " …", leg_fs) > w_pt:
                ult.pop()
            mot = [mot[0], " ".join(ult) + " …"]
        extras += [(ln, "#8B0000") for ln in mot]
    for c in getattr(kres, "conferencia", []) or []:          # leitura do XREF que não confere: aviso visível
        if "NÃO" in c:
            extras += [(ln, "#8B0000") for ln in _quebrar(c, w_pt, leg_fs)]
    leg_h_pt = len(extras) * leg_fs * 1.3
    tab_max_pt = (topo - y) / py - leg_h_pt - 3.0

    cab = list(COLUNAS_TABELA)
    arr, fs = None, fontsize
    for fs in [round(fontsize - 0.2 * i, 2) for i in range(int(round((fontsize - 5.0) / 0.2)) + 1)]:
        arr = _kerato_arranjo(linhas, cab, fs, w_pt)
        if arr is not None and sum(arr[3]) <= tab_max_pt:
            break
    if arr is None:                            # nem a 5 pt coube na largura: aceita o menor arranjo possível
        arr = _kerato_arranjo(linhas, cab, fs, 1e9)
        arr = (arr[0], arr[1], [v * w_pt / sum(arr[2]) for v in arr[2]], arr[3], arr[4])
    cab_t, corpo, colw, rowh, pad = arr
    escala = min(max(tab_max_pt, 1.0) / sum(rowh), 1.3)   # folga vertical: linhas um pouco mais altas (até 30%)
    rowh = [v * escala for v in rowh]
    tab_h = sum(rowh) * py
    ax = fig.add_axes([x, topo - tab_h, w, tab_h], gid=KERATO_GID)
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])
    tb.auto_set_font_size(False)
    tot_h = sum(rowh)
    for r, textos in enumerate([cab_t] + corpo):
        lin = linhas[r - 1] if r else {}
        dest = lin.get("destaque", "")
        fundo = "#eef1f7" if r == 0 else _KERATO_FUNDO.get(dest, "white")
        for c, txt in enumerate(textos):
            cell = tb.add_cell(r, c, width=colw[c] / w_pt, height=rowh[r] / tot_h, text=txt,
                               loc="left" if c == 0 else "center", facecolor=fundo, edgecolor="#444")
            cell.set_linewidth(0.4)
            if c == 0:
                cell.PAD = pad / colw[c]
            t = cell.get_text()
            t.set_fontsize(fs - 0.4 if r == 0 else fs)
            t.set_linespacing(1.1)
            if r == 0 or dest == "pupila":
                t.set_fontweight("bold")
            if dest == "referencia":
                t.set_fontstyle("italic")
                t.set_color("#666")
            elif r and not lin.get("disponivel", True) and dest != "pupila":
                t.set_color("#777")
    tb.set_gid(KERATO_GID)
    ax.add_table(tb)
    yl = topo - tab_h - 2.5 * py
    for ln, cor in extras:
        fig.text(x, yl, ln, fontsize=leg_fs, va="top", color=cor, gid=KERATO_GID)
        yl -= leg_fs * 1.3 * py
    return ax


def figure_resultado_completo(exam: NormalizedExam, analysis: Analysis, fig: Optional[Figure] = None,
                              patient_name: Optional[str] = None) -> Figure:
    """Página única A4 (retrato) com tudo: identificação, índices do TMS, 4 mapas topográficos (axial, tangencial,
    refrativo, elevação), frente de onda (total e alta ordem), coeficientes de Zernike, ceratometria por zona e
    tabelas de resultados. `fig` (opcional) é limpa e reutilizada — usado pelo canvas da interface. `patient_name`
    (opcional, lido do XREF a pedido do usuário) aparece SÓ nesta página impressa — nunca em JSON/CSV.
    Todo o conteúdo fica entre 3% e 97% da página (o HTML recorta a imagem com bbox_inches='tight')."""
    from . import CORE_VERSION
    if fig is None:
        fig = Figure(figsize=(8.27, 11.69), dpi=100)
    else:
        fig.clear()
        fig.set_size_inches(8.27, 11.69)
    fw, fh = fig.get_size_inches()
    py = 1.0 / (fh * 72.0)                     # 1 pt em fração da altura da página
    XL, XR = 0.045, 0.965                      # margens do conteúdo
    larg_pt = (XR - XL) * fw * 72.0
    ident, g = analysis.identity, analysis.geometry
    ix = exam.bundle.export_settings.get("tms_indices", {}) or {}

    def iv(k, fmt="{:.2f}"):
        v = ix.get(k)
        return fmt.format(v) if isinstance(v, (int, float)) and np.isfinite(v) else "—"

    # ---- cabeçalho (versão do programa e data de impressão ficam no rodapé)
    cap = f" · captura {ident.capture_index}" if ident.capture_index is not None else ""
    dev = analysis.device.model.replace(" (presumido pelo formato)", "")
    olho = f"Olho {ident.eye or '?'}{cap}"
    simk = (f"Índices do XREF: SimK curvo {iv('SimK1')} D @ {_eixo(ix.get('SimK1 Ang.'))} / plano {iv('SimK2')} D @ "
            f"{_eixo(ix.get('SimK2 Ang.'))} (TMS)    CYL {iv('CYL')} D    SRI {iv('SRI')}    SAI {iv('SAI')}    "
            f"Q {iv('Q')}    Es {iv('Es')}" if ix else "Índices do TMS: não disponíveis (sem XREF)")
    x_cab = XL - 0.005
    fig.text(x_cab, 0.968, "Aberrometria corneana anterior calculada — resultado completo", fontsize=12.5, weight="bold", va="top")
    if patient_name:
        _texto_cabendo(fig, x_cab, 0.9485, f"Paciente: {patient_name}", 10.0, larg_pt, weight="bold", va="top")
        esq = f"Data do exame: {_data_pt(ident.acquired_at)}   ·   {olho}"
        dir_ = f"Exame {ident.exam_id}  ·  código {ident.study_id}  ·  {dev}"
        fig.text(x_cab, 0.9335, esq, fontsize=9.0, weight="bold", va="top")
        _texto_cabendo(fig, XR, 0.9335, dir_, 7.5, larg_pt - _FOLGA * _largura_pt(esq, 9.0, "bold") - 14.0, va="top", ha="right", color="#222")
        aviso_sint = "*** EXAME SINTÉTICO ***" if analysis.synthetic else ""
        if aviso_sint:
            fig.text(XR, 0.9185, aviso_sint, fontsize=7.5, color="#a00", ha="right", va="top", weight="bold")
        _texto_cabendo(fig, x_cab, 0.9185, simk, 7.5,
                       larg_pt - (_FOLGA * _largura_pt(aviso_sint, 7.5, "bold") + 14.0 if aviso_sint else 0.0), va="top", color="#222")
    else:
        _texto_cabendo(fig, x_cab, 0.9485, f"Código do estudo {ident.study_id}   ·   Exame {ident.exam_id}   ·   {olho}   ·   "
                       f"Data do exame {_data_pt(ident.acquired_at)}   ·   {dev}", 8.5, larg_pt, va="top")
        _texto_cabendo(fig, x_cab, 0.9335, simk, 7.5, larg_pt, va="top", color="#222")
        if analysis.synthetic:
            fig.text(0.5, 0.9185, "*** EXAME SINTÉTICO — gerado por software; não corresponde a um paciente ***", fontsize=8,
                     color="#a00", ha="center", va="top", weight="bold")

    # ---- linha 1: topografia (4 mapas quadrados com barra de cor própria; a unidade está no título)
    y1, w1, x1, pitch = 0.7605, 0.158, 0.075, 0.2225
    h1 = w1 * fw / fh
    titles = {"DIO": "Potência axial (D)", "ROC": "Potência tangencial (D)", "POW": "Potência refrativa (D)"}
    for i, (role, _, _) in enumerate(TOPO_MAPS):
        x = x1 + i * pitch
        ax = _sq(fig, x, y1, w1)
        cax = fig.add_axes([x + w1 + 0.005, y1, 0.010, h1])
        if plot_topo_map(ax, exam, role, title=titles[role], cax=cax, fontsize=7.2) is None:
            cax.set_axis_off()
        cax.set_ylabel("")
        ax.set_xlabel("x (mm)", fontsize=6)
        if i == 0:
            ax.set_ylabel("y (mm)", fontsize=6, labelpad=1.5)
        else:
            ax.set_yticklabels([])
    x = x1 + 3 * pitch
    ax = _sq(fig, x, y1, w1)
    cax = fig.add_axes([x + w1 + 0.005, y1, 0.010, h1])
    plot_elevation_bfs(ax, exam, cax=cax, fontsize=7.2, title=None)
    ax.set_title(ax.get_title().replace("Elevação vs. esfera de melhor ajuste", "Elevação vs. esfera"), fontsize=7.2)
    ax.set_xlabel("x (mm)", fontsize=6); ax.set_yticklabels([])
    fig.text(XL, y1 + h1 + 0.030, "TOPOGRAFIA (Data Tables do TMS; círculos 3 e 6 mm; pupila tracejada; pontos interpolados pelo TMS incluídos)",
             fontsize=6.8, weight="bold", color="#333", va="top")

    # ---- linha 2: frente de onda do melhor perfil disponível (vértice 6 mm de preferência)
    order = sorted(analysis.wavefronts, key=lambda w: (w.validity != "disponivel", w.profile.center_mode != "vertice", -w.profile.diameter_mm))
    wf = order[0] if order and order[0].validity == "disponivel" else None
    y2, w2 = 0.5625, 0.19
    h2 = w2 * fw / fh
    xa1 = 0.075
    xa2 = xa1 + w2 + 0.096
    xa3 = xa2 + w2 + 0.123
    ax1 = _sq(fig, xa1, y2, w2); cax1 = fig.add_axes([xa1 + w2 + 0.006, y2, 0.011, h2])
    ax2 = _sq(fig, xa2, y2, w2); cax2 = fig.add_axes([xa2 + w2 + 0.006, y2, 0.011, h2])
    ax3 = fig.add_axes([xa3, y2, XR - xa3, h2])
    if wf is not None and wf.map_grid is not None:
        a = wf.profile.radius_mm
        ticks = np.arange(-int(np.floor(a + 1e-9)), int(np.floor(a + 1e-9)) + 1)
        centro = "vértice" if wf.profile.center_mode == "vertice" else wf.profile.center_mode
        Z = wf.map_grid["W"]; vl = max(float(np.nanmax(np.abs(Z))), 1e-3)
        im = ax1.imshow(Z, origin="lower", extent=[-a, a, -a, a], cmap=DIV_CMAP, vmin=-vl, vmax=vl, interpolation="nearest")
        _circle(ax1, a, color="black", lw=0.8); ax1.set_aspect("equal")
        ax1.set_title(f"W total (µm) — {wf.profile.diameter_mm:g} mm, {centro}", fontsize=7.2)
        cb = fig.colorbar(im, cax=cax1); cb.ax.tick_params(labelsize=6); cb.set_label("µm (+ adiantada)", fontsize=6)
        Zh = hoa_map(wf)
        if Zh is not None:
            vl2 = max(float(np.nanmax(np.abs(Zh))), 1e-3)
            im2 = ax2.imshow(Zh, origin="lower", extent=[-a, a, -a, a], cmap=DIV_CMAP, vmin=-vl2, vmax=vl2, interpolation="bilinear")
            _circle(ax2, a, color="black", lw=0.8); ax2.set_aspect("equal")
            ax2.set_title(f"Alta ordem n ≥ 3 (µm) — RMS {wf.metrics.get('rms_hoa_n3plus', float('nan')):.3f} µm", fontsize=7.2)
            cb2 = fig.colorbar(im2, cax=cax2); cb2.ax.tick_params(labelsize=6)
        else:
            cax2.set_axis_off()
        for ax in (ax1, ax2):
            ax.set_xticks(ticks); ax.set_yticks(ticks)
            ax.tick_params(labelsize=6); ax.set_xlabel("x (mm) rel. ao centro", fontsize=6)
        ax1.set_ylabel("y (mm)", fontsize=6, labelpad=1.5); ax2.set_yticklabels([])
        plot_zernike_bars(ax3, wf, title=f"Zernike OSA (n ≥ 2, µm) — {wf.profile.diameter_mm:g} mm", annotate=False)
        ax3.tick_params(labelsize=5.5); ax3.yaxis.label.set_size(6); ax3.title.set_size(7.2)
        ax3.set_ylabel("coeficiente (µm)", fontsize=6, labelpad=1.5)
    else:
        for ax in (ax1, ax2, ax3):
            ax.text(0.5, 0.5, "frente de onda indisponível\n(ver motivos na tabela)", ha="center", va="center", fontsize=7, transform=ax.transAxes)
            ax.set_axis_off()
        for c in (cax1, cax2):
            c.set_axis_off()
    fig.text(XL, y2 + h2 + 0.030, "FRENTE DE ONDA DA SUPERFÍCIE ANTERIOR (traçado de raios, n 1,000 → 1,376; referência paraxial)",
             fontsize=6.8, weight="bold", color="#333", va="top")

    # ---- linha 3: geometria/pupila/astigmatismo (esquerda) e ceratometria por zona (direita)
    y3 = 0.524                                 # topo dos dois blocos
    xk = 0.50                                  # início do bloco de ceratometria
    geo_lines = [
        ("GEOMETRIA RECONSTRUÍDA", True),
        (f"R = {g.conic_R_mm:.3f} mm    Q = {g.conic_Q:+.3f}    K equiv. {g.k_mean_D_keratometric:.2f} D", False),
        (f"resíduo RMS {g.residual_rms_mm*1000:.2f} µm (máx {g.residual_max_mm*1000:.2f}); {g.method}, ordem {g.surface_order}", False),
        (f"{g.n_points_used} pontos; raio medido {g.outer_radius_mm:.2f} mm; lacuna central {g.central_gap_mm:.2f} mm", False),
        ("", False),
        ("PUPILA (PUP)", True),
    ]
    p = analysis.pupil
    if p.state == "registrada" and p.center_x is not None and p.center_y is not None:
        cen = f"centro ({_sinal(p.center_x)}, {_sinal(p.center_y)}) mm rel. ao vértice"
        geo_lines += [(cen + (f"; diâmetro {float(p.diameter):.2f} mm" if p.diameter else ""), False)]
    else:
        geo_lines += [("não detectada nesta captura — perfis pupilares indisponíveis", False)]
    geo_lines += [("", False), ("ASTIGMATISMO CORNEANO (refrativo equivalente, vértice)", True)]
    for w in analysis.wavefronts:
        if w.validity == "disponivel" and w.profile.center_mode == "vertice" and w.metrics.get("cilindro_equivalente_D") is not None:
            ax_ = w.metrics["eixo_equivalente_deg"]
            geo_lines.append((f"zona {w.profile.diameter_mm:g} mm: {_sinal(w.metrics['cilindro_equivalente_D'])} D × {_eixo(ax_)}  "
                              f"(meridiano mais curvo {_eixo(ax_ - 90.0)})", False))
    if ix:
        geo_lines.append((f"SimK do TMS (ceratométrico): CYL {iv('CYL')} D, meridiano mais curvo {_eixo(ix.get('SimK1 Ang.'))}", False))
    geo_pt = (xk - 0.012 - XL) * fw * 72.0
    y = y3
    for ln, bold in geo_lines:
        partes = _quebrar(ln, geo_pt, 6.6, "bold" if bold else "normal") if ln else [""]
        for k, parte in enumerate(partes):
            fig.text(XL + (0.012 if k else 0.0), y, parte, fontsize=6.6, va="top", weight="bold" if bold else "normal",
                     color="#111" if bold else "#222")
            y -= 0.0125
    y_geo = y + 0.0125 - 6.6 * 1.2 * py        # base da última linha
    yk0 = 0.284
    plot_keratometry_table(fig, [xk, yk0, XR - xk, y3 - yk0], getattr(analysis, "keratometry", None))

    # ---- rodapé (de baixo para cima): aviso de uso e proveniência com versão e data de impressão
    rod_fs = 5.4
    hashes = "  ".join(f"{k.split(':')[0]}:{v[:8]}" for k, v in list(analysis.bundle_hashes.items())[:6])
    prov = (f"tmsab {CORE_VERSION} · impresso {_momento_local(analysis.created_at)} · análise {analysis.analysis_id} · "
            f"arquivos {hashes} · " + ", ".join(f"{k}={v}" for k, v in list(analysis.versions.items())[:3]))
    uso = ("Ferramenta de pesquisa — aberrometria corneana ANTERIOR calculada das Data Tables do Tomey TMS; não é aberrometria "
           "ocular total, não é alegação diagnóstica, seleção de LIO nem planejamento terapêutico.")
    yr = 0.031
    for ln, cor in [(t, "#555") for t in reversed(_quebrar(uso, larg_pt, rod_fs))] + [(t, "#444") for t in reversed(_quebrar(prov, larg_pt, rod_fs))]:
        fig.text(XL, yr, ln, fontsize=rod_fs, color=cor, va="bottom")
        yr += rod_fs * 1.3 * py
    piso = yr + 0.004                          # nada do corpo da página desce abaixo disto

    # ---- resultados por perfil (altura proporcional ao número de perfis), motivos, nota e avisos
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
                     (f"{_sinal(cil)} D × {_eixo(eixo)}" if cil is not None and np.isfinite(cil) else "—"),
                     f"{m.get('defocus_c20', float('nan')):+.2f}", f"{w.n_valid_rays}/{w.n_rays}"])
    y = min(y_geo, yk0) - 0.010
    fig.text(XL, y, "RESULTADOS POR PERFIL (µm; Zernike OSA ortonormal; abertura no plano corneano)", fontsize=6.8, weight="bold",
             color="#333", va="top")
    y -= 0.0125
    if rows:
        n = len(rows) + 1
        rh = min(0.0145, max(0.0095, (y - piso - 0.06) / n))
        rh = min(rh, (y - piso - 0.012) / n)   # muitos perfis: linhas mais baixas, nunca sobre o rodapé
        fs_t = min(6.2, max(4.5, 0.62 * rh / py))
        axt = fig.add_axes([XL, y - n * rh, XR - XL, n * rh]); axt.set_axis_off()
        tb = axt.table(cellText=rows, colLabels=cols, cellLoc="center", bbox=[0, 0, 1, 1],
                       colWidths=[0.11, 0.11, 0.08, 0.08, 0.08, 0.1, 0.09, 0.14, 0.1, 0.09])
        tb.auto_set_font_size(False); tb.set_fontsize(fs_t)
        for (r_, c_), cell in tb.get_celld().items():
            cell.set_linewidth(0.4)
            if r_ == 0:
                cell.set_facecolor("#eef1f7"); cell.set_text_props(weight="bold", fontsize=fs_t - 0.4)
            elif c_ == 1:
                cell.set_text_props(color="#1b5e20" if rows[r_ - 1][1] == "DISPONÍVEL" else "#8B0000", weight="bold")
        y -= n * rh + 0.006
    else:
        fig.text(XL, y, "nenhum perfil calculado", fontsize=6.2, va="top", color="#555")
        y -= 0.012
    nota = ("Cilindro refrativo equivalente (n = 1,376) a partir de c(2,±2) — não é ceratometria (essa está na tabela "
            "CERATOMETRIA POR ZONA, n 1,3375); eixo = meridiano do cilindro negativo. RMS HOA = n ≥ 3. Limites de "
            "disponibilidade provisórios (não clínicos); nenhuma faixa de normalidade é atribuída.")
    avisos = [w for w in analysis.warnings if "perfil verificado" not in w]

    def _linhas(blocos):                       # [(texto, fonte, cor, espaço extra depois, índice do bloco)]
        out = []
        for k, (texto, fs, cor) in enumerate(blocos):
            partes = _quebrar(texto, larg_pt, fs)
            out += [(ln, fs, cor, 0.002 if j == len(partes) - 1 else 0.0, k) for j, ln in enumerate(partes)]
        return out

    def _alt(lns):
        return sum(fs * 1.3 * py + extra for _, fs, _, extra, _ in lns)

    l_mot = _linhas([(rs, 5.8, "#8B0000") for rs in reasons])
    l_nota = _linhas([(nota, 5.8, "#333")])
    l_av = _linhas([("AVISO: " + a, 5.6, "#8a6d00") for a in avisos])
    if y - _alt(l_mot + l_nota + l_av) < piso:   # não cabe tudo: a nota fica; motivos e avisos são cortados no fim
        livre = y - piso - _alt(l_nota) - 5.4 * 1.3 * py
        resto = l_mot + [(ln, fs, cor, ex, k + 1000) for ln, fs, cor, ex, k in l_av]
        cabe = 0
        while cabe < len(resto) and _alt(resto[: cabe + 1]) <= livre:
            cabe += 1
        fora = len({k for *_, k in resto[cabe:]})
        l_mot = [t for t in resto[:cabe] if t[4] < 1000]
        l_av = [t for t in resto[:cabe] if t[4] >= 1000] + [(f"… (+{fora} item(ns) no relatório detalhado)", 5.4, "#555", 0.0, -1)]
    for ln, fs, cor, extra, _ in l_mot + l_nota + l_av:
        fig.text(XL, y, ln, fontsize=fs, va="top", color=cor)
        y -= fs * 1.3 * py + extra
    return fig
