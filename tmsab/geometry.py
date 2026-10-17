"""Geometria: superfícies analíticas, reconstrução da superfície anterior e cobertura (plano §5).

Altura e frente de onda são objetos distintos: este módulo só trata da superfície
z = S(x, y) em milímetros, no sistema canônico (z positivo posterior).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np
from scipy.interpolate import RBFInterpolator

from .model import PointState, ReconstructionConfig, Samples, SupportMetrics
from .zernike import ZernikeBasis


# ---------------------------------------------------------------------------
# Protocolo de superfície
# ---------------------------------------------------------------------------
class Surface(Protocol):
    def z(self, x: np.ndarray, y: np.ndarray) -> np.ndarray: ...
    def grad(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


def hessian_numeric(surface: Surface, x0: float, y0: float, h: float = 0.002) -> np.ndarray:
    """Hessiano de S em (x0, y0) por diferenças centrais do gradiente analítico."""
    x = np.array([x0 + h, x0 - h, x0, x0])
    y = np.array([y0, y0, y0 + h, y0 - h])
    gx, gy = surface.grad(x, y)
    sxx = (gx[0] - gx[1]) / (2 * h)
    syy = (gy[2] - gy[3]) / (2 * h)
    sxy = 0.5 * ((gx[2] - gx[3]) / (2 * h) + (gy[0] - gy[1]) / (2 * h))
    return np.array([[sxx, sxy], [sxy, syy]])


def principal_curvatures(surface: Surface, x0: float, y0: float) -> tuple[float, float, float]:
    """Curvaturas principais (1/mm) e ângulo (graus) da curvatura máxima, a partir do Hessiano.

    Aproximação paraxial: curvatura ≈ segunda derivada (válida onde a inclinação é pequena,
    como no centro de análise). A escolha fica registrada no perfil óptico.
    """
    H = hessian_numeric(surface, x0, y0)
    w, v = np.linalg.eigh(H)
    k_min, k_max = float(w[0]), float(w[1])
    ang = float(np.degrees(np.arctan2(v[1, 1], v[0, 1])))
    if ang < 0:
        ang += 180.0
    return k_max, k_min, ang


# ---------------------------------------------------------------------------
# Superfícies analíticas (usadas em testes e no gerador sintético)
# ---------------------------------------------------------------------------
@dataclass
class ConicSurface:
    """Cônica de revolução: z = r² / (R + sqrt(R² − (1+Q) r²))."""

    R: float
    Q: float = 0.0

    def z(self, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        r2 = x * x + y * y
        s = np.sqrt(self.R**2 - (1.0 + self.Q) * r2)
        return r2 / (self.R + s)

    def grad(self, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        r2 = x * x + y * y
        s = np.sqrt(self.R**2 - (1.0 + self.Q) * r2)
        # dz/dr = r/s  ->  dz/dx = x/s
        return x / s, y / s


@dataclass
class BiconicSurface:
    """Bicônica com eixos principais rotacionados: curvaturas cx, cy (1/mm), cônicas Qx, Qy, eixo (graus).

    z' = (cx x'² + cy y'²) / (1 + sqrt(1 − (1+Qx) cx² x'² − (1+Qy) cy² y'²)), (x', y') = rot(−axis)·(x, y)
    """

    Rx: float
    Ry: float
    Qx: float = 0.0
    Qy: float = 0.0
    axis_deg: float = 0.0

    def _rot(self, x, y):
        a = np.radians(self.axis_deg)
        c, s = np.cos(a), np.sin(a)
        return c * x + s * y, -s * x + c * y

    def z(self, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        xp, yp = self._rot(x, y)
        cx, cy = 1.0 / self.Rx, 1.0 / self.Ry
        u = cx * xp**2 + cy * yp**2
        v = (1 + self.Qx) * cx**2 * xp**2 + (1 + self.Qy) * cy**2 * yp**2
        return u / (1.0 + np.sqrt(1.0 - v))

    def grad(self, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        xp, yp = self._rot(x, y)
        cx, cy = 1.0 / self.Rx, 1.0 / self.Ry
        u = cx * xp**2 + cy * yp**2
        v = (1 + self.Qx) * cx**2 * xp**2 + (1 + self.Qy) * cy**2 * yp**2
        s = np.sqrt(1.0 - v)
        D = 1.0 + s
        ux, uy = 2 * cx * xp, 2 * cy * yp
        vx, vy = 2 * (1 + self.Qx) * cx**2 * xp, 2 * (1 + self.Qy) * cy**2 * yp
        Dx, Dy = -vx / (2 * s), -vy / (2 * s)
        zxp = (ux * D - u * Dx) / D**2
        zyp = (uy * D - u * Dy) / D**2
        a = np.radians(self.axis_deg)
        c, sn = np.cos(a), np.sin(a)
        # gradiente de volta ao sistema canônico: ∂/∂x = c ∂/∂x' − s ∂/∂y' ; ∂/∂y = s ∂/∂x' + c ∂/∂y'
        return c * zxp - sn * zyp, sn * zxp + c * zyp


@dataclass
class ZernikeDeformation:
    """Perturbação de elevação Σ c_j Z_j(ρ,θ) (c em mm, ρ = r/a)."""

    radius: float
    order: int
    coeffs_mm: dict[tuple[int, int], float]

    def __post_init__(self):
        self.basis = ZernikeBasis(self.order)
        self.vec = np.zeros(self.basis.J)
        for (n, m), c in self.coeffs_mm.items():
            self.vec[self.basis.terms.index((n, m))] = c

    def z(self, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        rho = np.hypot(x, y) / self.radius
        th = np.arctan2(y, x)
        return self.basis.design(rho.ravel(), th.ravel()) @ self.vec

    def grad(self, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        rho = np.hypot(x, y) / self.radius
        th = np.arctan2(y, x)
        _, Ax, Ay = self.basis.design_with_gradient(rho.ravel(), th.ravel())
        return (Ax @ self.vec) / self.radius, (Ay @ self.vec) / self.radius


@dataclass
class CompositeSurface:
    """Soma de superfícies (base + perturbações)."""

    parts: list

    def z(self, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        out = np.zeros(np.broadcast(x, y).shape).ravel()
        for p in self.parts:
            out = out + np.asarray(p.z(x, y)).ravel()
        return out.reshape(np.broadcast(x, y).shape)

    def grad(self, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        shp = np.broadcast(x, y).shape
        gx = np.zeros(shp).ravel(); gy = np.zeros(shp).ravel()
        for p in self.parts:
            a, b = p.grad(x, y)
            gx = gx + np.asarray(a).ravel(); gy = gy + np.asarray(b).ravel()
        return gx.reshape(shp), gy.reshape(shp)


@dataclass
class MirroredSurface:
    """Espelha a superfície em x (x -> -x). Usado em ensaios de orientação."""

    inner: Surface

    def z(self, x, y):
        return self.inner.z(-np.asarray(x, float), y)

    def grad(self, x, y):
        gx, gy = self.inner.grad(-np.asarray(x, float), y)
        return -gx, gy


@dataclass
class RotatedSurface:
    """Rotaciona a superfície por `angle_deg` (anti-horário na vista do observador)."""

    inner: Surface
    angle_deg: float

    def _inv(self, x, y):
        a = np.radians(self.angle_deg)
        c, s = np.cos(a), np.sin(a)
        return c * x + s * y, -s * x + c * y

    def z(self, x, y):
        xp, yp = self._inv(np.asarray(x, float), np.asarray(y, float))
        return self.inner.z(xp, yp)

    def grad(self, x, y):
        xp, yp = self._inv(np.asarray(x, float), np.asarray(y, float))
        gx, gy = self.inner.grad(xp, yp)
        a = np.radians(self.angle_deg)
        c, s = np.cos(a), np.sin(a)
        return c * gx - s * gy, s * gx + c * gy


# ---------------------------------------------------------------------------
# Ajuste de cônica (linear): r² = 2 R z − (1+Q) z²
# ---------------------------------------------------------------------------
def fit_conic(r: np.ndarray, z: np.ndarray, w: Optional[np.ndarray] = None) -> tuple[float, float, float]:
    """Ajusta R e Q por mínimos quadrados lineares na forma implícita. Retorna (R, Q, rms do resíduo em z)."""
    r = np.asarray(r, float).ravel(); z = np.asarray(z, float).ravel()
    ok = np.isfinite(r) & np.isfinite(z) & (z > 0)
    if w is None:
        w = np.ones_like(r)
    w = np.asarray(w, float).ravel()
    ok &= np.isfinite(w) & (w > 0)
    A = np.column_stack([z[ok], z[ok] ** 2])
    b = r[ok] ** 2
    sw = np.sqrt(w[ok])
    sol, *_ = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)
    a, bq = sol
    R = a / 2.0
    Q = -bq - 1.0
    conic = ConicSurface(R, Q)
    zz = conic.z(r[ok], np.zeros_like(r[ok]))
    rms = float(np.sqrt(np.sum(w[ok] * (zz - z[ok]) ** 2) / np.sum(w[ok])))
    return float(R), float(Q), rms


# ---------------------------------------------------------------------------
# Superfície reconstruída
# ---------------------------------------------------------------------------
@dataclass
class ReconstructedSurface:
    method: str
    conic: ConicSurface
    fit_radius: float
    surface_order: int
    zern_basis: Optional[ZernikeBasis] = None
    zern_coeffs: Optional[np.ndarray] = None
    rbf: Optional[RBFInterpolator] = None
    n_points_used: int = 0
    residual_rms: float = float("nan")
    residual_max: float = float("nan")
    residual_by_ring_rms: list[float] = field(default_factory=list)
    holdout_residual_rms: Optional[float] = None
    condition_number: float = float("nan")
    central_gap: float = float("nan")
    outer_radius: float = float("nan")
    measured_fraction: float = float("nan")
    residuals: Optional[np.ndarray] = None      # resíduo por ponto usado (mm)
    used_mask: Optional[np.ndarray] = None

    # --- avaliação
    def _resid_z(self, x, y):
        if self.method == "conica_zernike":
            rho = np.hypot(x, y) / self.fit_radius
            th = np.arctan2(y, x)
            return self.zern_basis.design(rho.ravel(), th.ravel()) @ self.zern_coeffs
        else:
            pts = np.column_stack([x.ravel(), y.ravel()])
            return self.rbf(pts)

    def z(self, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        shp = np.broadcast(x, y).shape
        base = self.conic.z(x, y).ravel()
        return (base + self._resid_z(x, y)).reshape(shp)

    def grad(self, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        shp = np.broadcast(x, y).shape
        gx, gy = self.conic.grad(x, y)
        gx = gx.ravel(); gy = gy.ravel()
        if self.method == "conica_zernike":
            rho = np.hypot(x, y).ravel() / self.fit_radius
            th = np.arctan2(y, x).ravel()
            _, Ax, Ay = self.zern_basis.design_with_gradient(rho, th)
            gx = gx + (Ax @ self.zern_coeffs) / self.fit_radius
            gy = gy + (Ay @ self.zern_coeffs) / self.fit_radius
        else:
            h = 1e-3
            xr, yr = x.ravel(), y.ravel()
            gx = gx + (self._resid_z(xr + h, yr) - self._resid_z(xr - h, yr)) / (2 * h)
            gy = gy + (self._resid_z(xr, yr + h) - self._resid_z(xr, yr - h)) / (2 * h)
        return gx.reshape(shp), gy.reshape(shp)

    def inside_domain(self, x, y, margin: float = 0.0) -> np.ndarray:
        return np.hypot(np.asarray(x, float), np.asarray(y, float)) <= self.fit_radius * (1.0 + margin)


def area_weights(samples: Samples) -> np.ndarray:
    """Pesos proporcionais à área representada por cada amostra em anéis (r·Δr·Δθ), normalizados (média 1)."""
    r = samples.r
    assert r is not None
    nr, na = r.shape
    rn = np.nanmean(r, axis=1)           # raio nominal do anel
    if np.all(~np.isfinite(rn)):
        return np.ones_like(r)
    idx = np.arange(nr)
    good = np.isfinite(rn)
    rn_full = np.interp(idx, idx[good], rn[good]) if good.sum() >= 2 else np.where(good, rn, 1.0)
    dr = np.gradient(rn_full) if nr > 1 else np.ones(nr)
    w = np.where(np.isfinite(r), r, rn_full[:, None]) * np.abs(dr)[:, None]
    w = np.where(np.isfinite(w) & (w > 0), w, np.nanmedian(w))
    return w / np.nanmean(w)


def reconstruct_surface(samples: Samples, cfg: ReconstructionConfig) -> ReconstructedSurface:
    """Reconstrói z = S(x, y) a partir das amostras normalizadas.

    Método "conica_zernike": cônica (R, Q) por ajuste linear + resíduo em base de Zernike
    (ordem cfg.surface_order, raio cfg.fit_radius_mm). Método "rbf_tps": cônica + spline de
    placa fina (RBF) com suavização. Pesos de área por padrão. Pontos reservados (anéis)
    não entram no ajuste e fornecem o resíduo de verificação.
    """
    assert samples.x is not None and samples.y is not None and samples.z is not None and samples.point_state is not None
    x, y, z, st, r = samples.x, samples.y, samples.z, samples.point_state, samples.r
    usable = (st == PointState.MEASURED)
    if cfg.use_interpolated:
        usable |= (st == PointState.INTERPOLATED)
    usable &= np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if usable.sum() < 50:
        raise ValueError(f"pontos utilizáveis insuficientes para reconstrução: {int(usable.sum())}")

    rr = np.hypot(x, y)
    outer = float(np.nanmax(rr[usable]))
    fit_radius = cfg.fit_radius_mm if cfg.fit_radius_mm else outer * 1.0005
    w_all = area_weights(samples) if cfg.weights == "area" else np.ones_like(x)

    holdout = np.zeros_like(usable)
    for k in cfg.holdout_rings:
        if 0 <= k < samples.n_rings:
            holdout[k, :] = True
    use = usable & ~holdout & (rr <= fit_radius)

    # 1) cônica
    R, Q, _ = fit_conic(rr[use], z[use], w_all[use])
    conic = ConicSurface(R, Q)
    dz = z - conic.z(x, y)

    rec = ReconstructedSurface(method=cfg.method, conic=conic, fit_radius=float(fit_radius), surface_order=cfg.surface_order)
    if cfg.method == "conica_zernike":
        basis = ZernikeBasis(cfg.surface_order)
        rho = rr / fit_radius
        th = np.arctan2(y, x)
        fr = basis.fit(rho[use], th[use], dz[use], w_all[use])
        rec.zern_basis, rec.zern_coeffs, rec.condition_number = basis, fr.coeffs, fr.condition_number
    elif cfg.method == "rbf_tps":
        pts = np.column_stack([x[use], y[use]])
        rec.rbf = RBFInterpolator(pts, dz[use], kernel="thin_plate_spline", smoothing=cfg.rbf_smoothing, degree=1)
        rec.condition_number = float("nan")
    else:
        raise ValueError(f"método de reconstrução desconhecido: {cfg.method}")

    # 2) resíduos
    zfit = rec.z(x, y)
    res = z - zfit
    rec.residuals = np.where(use, res, np.nan)
    rec.used_mask = use
    rec.n_points_used = int(use.sum())
    rec.residual_rms = float(np.sqrt(np.nanmean(res[use] ** 2)))
    rec.residual_max = float(np.nanmax(np.abs(res[use])))
    rec.residual_by_ring_rms = [
        float(np.sqrt(np.nanmean(res[k, use[k]] ** 2))) if use[k].any() else float("nan") for k in range(samples.n_rings)
    ]
    ho = usable & holdout & (rr <= fit_radius)
    rec.holdout_residual_rms = float(np.sqrt(np.nanmean(res[ho] ** 2))) if ho.any() else None
    meas = (st == PointState.MEASURED) & np.isfinite(rr)
    rec.central_gap = float(np.nanmin(rr[meas])) if meas.any() else float("nan")
    rec.outer_radius = outer
    rec.measured_fraction = float(meas.sum() / max(1, np.isfinite(rr).sum()))
    return rec


# ---------------------------------------------------------------------------
# Suporte / cobertura de uma abertura
# ---------------------------------------------------------------------------
def support_metrics(samples: Samples, rec: ReconstructedSurface, cx: float, cy: float, radius: float,
                    n_meridians: int = 72, edge_margin: float = 0.98) -> SupportMetrics:
    """Mede o suporte real dos dados dentro da abertura (centro, raio) em mm."""
    x, y, st, r = samples.x, samples.y, samples.point_state, samples.r
    assert x is not None and y is not None and st is not None and r is not None
    # pontos ausentes não têm coordenadas: usar posição nominal do anel (raio médio do anel) para contá-los dentro da abertura
    rn = np.nanmean(r, axis=1)
    idx = np.arange(r.shape[0]); good = np.isfinite(rn)
    rn_full = np.interp(idx, idx[good], rn[good]) if good.sum() >= 2 else np.where(good, rn, 0.0)
    th = np.radians(samples.theta_deg)[None, :]
    xn = np.where(np.isfinite(x), x, rn_full[:, None] * np.cos(th))
    yn = np.where(np.isfinite(y), y, rn_full[:, None] * np.sin(th))
    dx, dy = xn - cx, yn - cy
    d = np.hypot(dx, dy)
    finite = np.isfinite(d)
    inside = finite & (d <= radius)
    meas = st == PointState.MEASURED
    usable = meas | (st == PointState.INTERPOLATED)
    n_in = int(inside.sum())
    n_meas = int((inside & meas).sum())
    n_us = int((inside & usable).sum())
    measured_fraction = n_meas / n_in if n_in else 0.0
    usable_fraction = n_us / n_in if n_in else 0.0

    # alcance por meridiano a partir do centro de análise
    ang = np.degrees(np.arctan2(dy, dx)) % 360.0
    reach = np.zeros(n_meridians)
    width = 360.0 / n_meridians
    for k in range(n_meridians):
        a0 = k * width
        sel = meas & finite & (((ang - a0) % 360.0) < width)
        reach[k] = np.max(d[sel]) if sel.any() else 0.0
    edge_reach = float(np.mean(reach >= radius * edge_margin))
    # maior setor contíguo sem alcance na borda (graus)
    bad = reach < radius * edge_margin
    if bad.all():
        max_sector = 360.0
    elif not bad.any():
        max_sector = 0.0
    else:
        b2 = np.concatenate([bad, bad])
        best = cur = 0
        for v in b2:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        max_sector = min(360.0, best * width)
    # lacuna central ao redor do centro de análise
    central_gap = float(np.min(d[meas & finite])) if (meas & finite).any() else float("inf")
    inside_fit = bool(np.hypot(cx, cy) + radius <= rec.fit_radius * 1.0 + 1e-9)
    return SupportMetrics(
        measured_fraction=float(measured_fraction),
        usable_fraction=float(usable_fraction),
        central_gap_mm=central_gap,
        edge_reach_fraction=edge_reach,
        max_missing_sector_deg=float(max_sector),
        n_points_inside=n_in,
        n_measured_inside=n_meas,
        inside_fit_domain=inside_fit,
    )
