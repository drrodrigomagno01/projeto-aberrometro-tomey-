"""Motor óptico: traçado de raios, diferença de caminho óptico e frente de onda (plano §5).

Modelo "anterior_v1": superfície anterior única separando ar (n_in) e córnea (n_out),
objeto axial no infinito, abertura circular no plano corneano (z = 0) com centro
(cx, cy) e raio a. Raios paralelos ao eixo z entram pela abertura, interceptam a
superfície em (x, y, S(x,y)), refratam pela lei de Snell vetorial e seguem em meio n_out.

Referência: esfera centrada no ponto imagem F sobre o raio principal refratado,
a uma distância s' = n_out / P_eq do ponto principal P_c, com P_eq = (n_out − n_in)·κ_média
(curvatura média da superfície no centro de análise, aproximação paraxial). A esfera
passa por P_c. O caminho óptico de cada raio é medido até a esfera ao longo do raio
em espaço imagem (extensão virtual permitida, como na definição clássica de
aberração de frente de onda).

W(x, y) = OPL_principal − OPL(x, y)   [positivo = frente de onda adiantada]

Com essa convenção, uma superfície esférica refratora (n_out > n_in) produz
aberração esférica primária positiva (c(4,0) > 0), e um elipsoide com Q = −1/n_out²
(oval de Descartes para objeto no infinito) produz W ≡ 0.

Unidades: geometria em mm; a saída W é convertida para micrômetros em um único
ponto (função trace_wavefront).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar

from .geometry import Surface, principal_curvatures
from .model import OpticalProfile
from .zernike import FitResult, ZernikeBasis

MM_TO_UM = 1000.0


@dataclass
class TraceResult:
    x: np.ndarray             # coordenadas na abertura (mm, relativas ao centro)
    y: np.ndarray
    W_um: np.ndarray          # frente de onda (µm), NaN onde o raio é inválido
    valid: np.ndarray         # máscara de raios válidos
    s_image_mm: float         # distância do ponto principal ao ponto imagem de referência
    power_eq_D: float         # potência equivalente usada na referência
    k_max: float
    k_min: float
    axis_deg: float
    chief_dir: tuple[float, float, float]
    n_rays: int
    n_valid: int
    notes: list


def _refract(d: np.ndarray, n_hat: np.ndarray, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """Lei de Snell vetorial. d: direções incidentes unitárias [N,3]; n_hat: normais unitárias
    apontando para o meio incidente [N,3]; eta = n_in/n_out. Retorna (t, válido)."""
    c1 = -np.einsum("ij,ij->i", n_hat, d)
    disc = 1.0 - eta**2 * (1.0 - c1**2)
    valid = disc >= 0.0
    c2 = np.sqrt(np.where(valid, disc, 0.0))
    t = eta * d + (eta * c1 - c2)[:, None] * n_hat
    norm = np.linalg.norm(t, axis=1)
    t = t / np.where(norm > 0, norm, 1.0)[:, None]
    return t, valid


def _surface_normals_incident_side(surface: Surface, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    gx, gy = surface.grad(x, y)
    n = np.column_stack([gx.ravel(), gy.ravel(), -np.ones(x.size)])   # (zx, zy, -1): aponta para o ar
    return n / np.linalg.norm(n, axis=1)[:, None]


def aperture_grid(radius: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Grade cartesiana n×n no quadrado circunscrito, filtrada ao disco (relativa ao centro)."""
    u = np.linspace(-radius, radius, n)
    X, Y = np.meshgrid(u, u)
    m = (X**2 + Y**2) <= radius**2 * (1.0 + 1e-12)
    return X[m], Y[m]


def _opl_to_reference_sphere(P: np.ndarray, t: np.ndarray, F: np.ndarray, rho_ref: float, n_in: float,
                             n_out: float, z_entry: float) -> tuple[np.ndarray, np.ndarray]:
    """Caminho óptico de cada raio desde o plano z_entry até a esfera de referência (centro F, raio rho_ref),
    medido ao longo do raio refratado (extensão virtual permitida). Raiz do lado próximo de F."""
    q = P - F
    qt = np.einsum("ij,ij->i", q, t)
    disc = qt**2 - np.einsum("ij,ij->i", q, q) + rho_ref**2
    valid = disc >= 0.0
    s = -qt - np.sqrt(np.where(valid, disc, 0.0))
    opl = n_in * (P[:, 2] - z_entry) + n_out * s
    return opl, valid


def trace_wavefront(surface: Surface, profile: OpticalProfile, cx: float = 0.0, cy: float = 0.0,
                    s_image_override: Optional[float] = None) -> TraceResult:
    """Calcula a frente de onda W (µm) sobre a abertura do perfil centrada em (cx, cy)."""
    a = profile.radius_mm
    n_in, n_out = profile.n_in, profile.n_out
    eta = n_in / n_out
    notes: list[str] = []

    # raio principal
    k_max, k_min, axis = principal_curvatures(surface, cx, cy)
    k_mean = 0.5 * (k_max + k_min)
    power_eq = (n_out - n_in) * k_mean          # 1/mm
    if not np.isfinite(power_eq) or power_eq <= 0:
        raise ValueError("potência paraxial equivalente inválida no centro de análise")
    s_image = n_out / power_eq if s_image_override is None else float(s_image_override)

    Pc = np.array([[cx, cy, float(np.asarray(surface.z(np.array([cx]), np.array([cy]))).ravel()[0])]])
    nc = _surface_normals_incident_side(surface, np.array([cx]), np.array([cy]))
    d = np.array([[0.0, 0.0, 1.0]])
    tc, okc = _refract(d, nc, eta)
    if not okc[0]:
        raise ValueError("raio principal sem solução de refração")
    F = Pc[0] + s_image * tc[0]
    rho_ref = s_image

    # feixe
    xr, yr = aperture_grid(a, profile.ray_grid)
    X, Y = xr + cx, yr + cy
    Z = np.asarray(surface.z(X, Y)).ravel()
    P = np.column_stack([X, Y, Z])
    nh = _surface_normals_incident_side(surface, X, Y)
    D = np.tile(d, (P.shape[0], 1))
    T, ok_refr = _refract(D, nh, eta)
    z_entry = 0.0
    opl, ok_sph = _opl_to_reference_sphere(P, T, F, rho_ref, n_in, n_out, z_entry)
    opl_c, _ = _opl_to_reference_sphere(Pc, tc, F[None, :], rho_ref, n_in, n_out, z_entry)
    valid = ok_refr & ok_sph & np.isfinite(opl)
    W_mm = np.where(valid, opl_c[0] - opl, np.nan)
    n_invalid = int((~valid).sum())
    if n_invalid:
        notes.append(f"{n_invalid} raios sem solução (refração ou esfera de referência)")
    return TraceResult(
        x=xr, y=yr, W_um=W_mm * MM_TO_UM, valid=valid, s_image_mm=float(s_image),
        power_eq_D=float(power_eq * 1000.0), k_max=k_max, k_min=k_min, axis_deg=axis,
        chief_dir=tuple(float(v) for v in tc[0]), n_rays=int(P.shape[0]), n_valid=int(valid.sum()), notes=notes,
    )


def refocus_rms(surface: Surface, profile: OpticalProfile, cx: float, cy: float) -> TraceResult:
    """Opção 'refoco_rms': escolhe s' minimizando o RMS de W após remover pistão e tilt (registrado)."""
    base = trace_wavefront(surface, profile, cx, cy)
    basis1 = ZernikeBasis(1)
    a = profile.radius_mm

    def cost(s):
        tr = trace_wavefront(surface, profile, cx, cy, s_image_override=s)
        v = tr.valid
        fr = basis1.fit(np.hypot(tr.x[v], tr.y[v]) / a, np.arctan2(tr.y[v], tr.x[v]), tr.W_um[v])
        return fr.residual_rms

    res = minimize_scalar(cost, bracket=(base.s_image_mm * 0.9, base.s_image_mm, base.s_image_mm * 1.1))
    tr = trace_wavefront(surface, profile, cx, cy, s_image_override=float(res.x))
    tr.notes.append(f"refoco por RMS: s'={res.x:.4f} mm (paraxial {base.s_image_mm:.4f} mm)")
    return tr


def fit_wavefront(tr: TraceResult, radius: float, order: int) -> FitResult:
    """Ajuste de Zernike OSA da frente de onda sobre a abertura (ρ = r/a, θ anti-horário)."""
    v = tr.valid & np.isfinite(tr.W_um)
    rho = np.hypot(tr.x[v], tr.y[v]) / radius
    th = np.arctan2(tr.y[v], tr.x[v])
    basis = ZernikeBasis(order)
    return basis.fit(rho, th, tr.W_um[v])


def wavefront_map_grid(tr: TraceResult, fit: FitResult, radius: float, n: int = 161) -> dict[str, np.ndarray]:
    """Grade regular para mapas: W traçada (interpolada por vizinho mais próximo da grade de raios),
    W ajustada e resíduo, em µm."""
    u = np.linspace(-radius, radius, n)
    X, Y = np.meshgrid(u, u)
    inside = (X**2 + Y**2) <= radius**2
    rho = np.hypot(X, Y) / radius
    th = np.arctan2(Y, X)
    basis = ZernikeBasis(int(max(nm[0] for nm in fit.terms)))
    Wfit = np.full(X.shape, np.nan)
    Wfit[inside] = basis.design(rho[inside], th[inside]) @ fit.coeffs
    # resíduo (traçado − ajuste) nos raios, interpolado para a grade; W traçada = ajuste + resíduo
    from scipy.interpolate import griddata
    v = tr.valid & np.isfinite(tr.W_um)
    res_grid = np.full(X.shape, np.nan)
    if v.sum() > 10:
        res_rays = tr.W_um[v] - basis.design(np.hypot(tr.x[v], tr.y[v]) / radius, np.arctan2(tr.y[v], tr.x[v])) @ fit.coeffs
        res_grid[inside] = griddata((tr.x[v], tr.y[v]), res_rays, (X[inside], Y[inside]), method="nearest")
    return {"x": X, "y": Y, "W": Wfit + res_grid, "W_fit": Wfit, "residual": res_grid, "mask": inside}
