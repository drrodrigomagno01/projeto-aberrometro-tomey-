"""Base de Zernike — convenção OSA/ANSI Z80.28 (Thibos et al. 2002).

- Índices duplos (n, m); índice único j = (n(n+2) + m)/2.
- Z_n^m(ρ,θ) = N_n^m R_n^|m|(ρ) cos(mθ)  para m ≥ 0
            = N_n^m R_n^|m|(ρ) sin(|m|θ) para m < 0
- N_n^m = sqrt(2(n+1)/(1+δ_m0))  (base ortonormal no disco unitário)
- θ anti-horário a partir do eixo +x (vista do observador); ρ = r/a.

As fórmulas aqui descrevem a base; a conversão de coordenadas físicas para (ρ, θ)
é responsabilidade do chamador, que deve registrar centro e raio da abertura.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import factorial, sqrt
from typing import Optional

import numpy as np

_NAMES = {
    (0, 0): "piston",
    (1, -1): "tilt vertical",
    (1, 1): "tilt horizontal",
    (2, -2): "astigmatismo obliquo",
    (2, 0): "defocus",
    (2, 2): "astigmatismo vertical/horizontal",
    (3, -3): "trefoil vertical",
    (3, -1): "coma vertical",
    (3, 1): "coma horizontal",
    (3, 3): "trefoil obliquo",
    (4, -4): "tetrafoil obliquo",
    (4, -2): "astigmatismo secundario obliquo",
    (4, 0): "aberracao esferica primaria",
    (4, 2): "astigmatismo secundario vertical",
    (4, 4): "tetrafoil vertical",
    (5, -5): "pentafoil",
    (5, -3): "trefoil secundario vertical",
    (5, -1): "coma secundario vertical",
    (5, 1): "coma secundario horizontal",
    (5, 3): "trefoil secundario obliquo",
    (5, 5): "pentafoil obliquo",
    (6, 0): "aberracao esferica secundaria",
}


def osa_j(n: int, m: int) -> int:
    return (n * (n + 2) + m) // 2


def osa_nm(j: int) -> tuple[int, int]:
    n = int(np.ceil((-3.0 + sqrt(9.0 + 8.0 * j)) / 2.0))
    m = 2 * j - n * (n + 2)
    return n, m


def nm_terms(order: int) -> list[tuple[int, int]]:
    """Lista (n, m) em ordem OSA (j crescente) até a ordem radial `order`."""
    return [(n, m) for n in range(order + 1) for m in range(-n, n + 1, 2)]


def term_name(n: int, m: int) -> str:
    return _NAMES.get((n, m), f"Z({n},{m:+d})")


def norm_factor(n: int, m: int) -> float:
    return sqrt(2.0 * (n + 1) / (2.0 if m == 0 else 1.0))


def radial_coeffs(n: int, m: int) -> list[tuple[int, float]]:
    """Coeficientes (potência, valor) de R_n^|m|(ρ) = Σ_k (-1)^k (n-k)! / (k! ((n+m)/2-k)! ((n-m)/2-k)!) ρ^(n-2k)."""
    m = abs(m)
    out = []
    for k in range((n - m) // 2 + 1):
        c = ((-1) ** k) * factorial(n - k) / (
            factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k)
        )
        out.append((n - 2 * k, float(c)))
    return out


@dataclass
class FitResult:
    coeffs: np.ndarray            # [J]
    terms: list[tuple[int, int]]
    fitted: np.ndarray            # valores ajustados nos pontos
    residual: np.ndarray          # valores - ajustados
    residual_rms: float
    condition_number: float
    rank: int
    n_points: int


class ZernikeBasis:
    """Base de Zernike OSA até a ordem radial `order`, com derivadas analíticas."""

    def __init__(self, order: int):
        if order < 0:
            raise ValueError("ordem radial deve ser >= 0")
        self.order = order
        self.terms = nm_terms(order)
        self.J = len(self.terms)
        self._norm = np.array([norm_factor(n, m) for n, m in self.terms])
        self._rcoef = [radial_coeffs(n, m) for n, m in self.terms]

    # -- avaliação ------------------------------------------------------
    def _radial(self, rho: np.ndarray, coefs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retorna R(ρ), R'(ρ) e R(ρ)/ρ (polinomial, finito em ρ=0 quando m≥1)."""
        R = np.zeros_like(rho)
        dR = np.zeros_like(rho)
        R_over_rho = np.zeros_like(rho)
        for p, c in coefs:
            R += c * rho ** p
            if p >= 1:
                dR += c * p * rho ** (p - 1)
                R_over_rho += c * rho ** (p - 1)
        return R, dR, R_over_rho

    def design(self, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Matriz [N, J] com Z_j(ρ_i, θ_i)."""
        rho = np.asarray(rho, dtype=float)
        theta = np.asarray(theta, dtype=float)
        A = np.empty((rho.size, self.J))
        for j, (n, m) in enumerate(self.terms):
            R, _, _ = self._radial(rho, self._rcoef[j])
            if m > 0:
                ang = np.cos(m * theta)
            elif m < 0:
                ang = np.sin(-m * theta)
            else:
                ang = np.ones_like(theta)
            A[:, j] = self._norm[j] * R * ang
        return A

    def design_with_gradient(self, rho: np.ndarray, theta: np.ndarray):
        """Retorna (A, dA/dX, dA/dY) em coordenadas normalizadas X=ρcosθ, Y=ρsinθ.

        Para derivada física dividir por a (raio de normalização).
        """
        rho = np.asarray(rho, dtype=float)
        theta = np.asarray(theta, dtype=float)
        ct, st = np.cos(theta), np.sin(theta)
        A = np.empty((rho.size, self.J))
        Ax = np.empty_like(A)
        Ay = np.empty_like(A)
        for j, (n, m) in enumerate(self.terms):
            R, dR, Rr = self._radial(rho, self._rcoef[j])
            mm = abs(m)
            if m > 0:
                ang = np.cos(mm * theta)
                dang = -mm * np.sin(mm * theta)       # d/dθ
            elif m < 0:
                ang = np.sin(mm * theta)
                dang = mm * np.cos(mm * theta)
            else:
                ang = np.ones_like(theta)
                dang = np.zeros_like(theta)
            N = self._norm[j]
            A[:, j] = N * R * ang
            # d/dX = cosθ ∂/∂ρ − (sinθ/ρ) ∂/∂θ ; d/dY = sinθ ∂/∂ρ + (cosθ/ρ) ∂/∂θ
            # (R/ρ)·dang é polinomial e finito em ρ=0 (dang=0 quando m=0)
            Ax[:, j] = N * (ct * dR * ang - st * Rr * dang)
            Ay[:, j] = N * (st * dR * ang + ct * Rr * dang)
        return A, Ax, Ay

    # -- ajuste ---------------------------------------------------------
    def fit(
        self,
        rho: np.ndarray,
        theta: np.ndarray,
        values: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> FitResult:
        """Ajuste por mínimos quadrados (SVD) com pesos opcionais; reporta condicionamento."""
        rho = np.asarray(rho, dtype=float).ravel()
        theta = np.asarray(theta, dtype=float).ravel()
        values = np.asarray(values, dtype=float).ravel()
        ok = np.isfinite(values) & np.isfinite(rho) & np.isfinite(theta)
        if weights is not None:
            weights = np.asarray(weights, dtype=float).ravel()
            ok &= np.isfinite(weights) & (weights > 0)
        if ok.sum() < self.J:
            raise ValueError(f"pontos insuficientes para ajuste: {int(ok.sum())} < {self.J}")
        A = self.design(rho[ok], theta[ok])
        b = values[ok]
        if weights is not None:
            sw = np.sqrt(weights[ok])
            Aw, bw = A * sw[:, None], b * sw
        else:
            Aw, bw = A, b
        coeffs, _, rank, sv = np.linalg.lstsq(Aw, bw, rcond=None)
        cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
        fitted_ok = A @ coeffs
        fitted = np.full(values.shape, np.nan)
        fitted[ok] = fitted_ok
        residual = values - fitted
        res_ok = residual[ok]
        if weights is not None:
            rms = float(np.sqrt(np.sum(weights[ok] * res_ok**2) / np.sum(weights[ok])))
        else:
            rms = float(np.sqrt(np.mean(res_ok**2)))
        return FitResult(coeffs, list(self.terms), fitted, residual, rms, cond, int(rank), int(ok.sum()))

    def evaluate(self, coeffs: np.ndarray, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return self.design(rho, theta) @ np.asarray(coeffs, dtype=float)


# ---------------------------------------------------------------------------
# Métricas a partir dos coeficientes
# ---------------------------------------------------------------------------
def coefficient(coeffs: np.ndarray, terms: list[tuple[int, int]], n: int, m: int) -> float:
    try:
        return float(coeffs[terms.index((n, m))])
    except ValueError:
        return float("nan")


def metrics_from_coefficients(coeffs: np.ndarray, terms: list[tuple[int, int]]) -> dict[str, float]:
    """Métricas padrão (µm). RMS por família = raiz da soma dos quadrados dos termos incluídos.

    Válido como RMS no disco completo apenas para base ortonormal e abertura circular
    completa (plano §5); com suporte irregular, verificar resíduo e cobertura.
    """
    c = np.asarray(coeffs, dtype=float)
    nn = np.array([n for n, _ in terms])
    out: dict[str, float] = {}
    out["piston_c00"] = coefficient(c, terms, 0, 0)
    out["tilt_c1m1"] = coefficient(c, terms, 1, -1)
    out["tilt_c11"] = coefficient(c, terms, 1, 1)
    out["defocus_c20"] = coefficient(c, terms, 2, 0)
    out["astig_c2m2"] = coefficient(c, terms, 2, -2)
    out["astig_c22"] = coefficient(c, terms, 2, 2)
    out["astig_magnitude"] = float(np.hypot(out["astig_c2m2"], out["astig_c22"]))
    out["coma_c3m1"] = coefficient(c, terms, 3, -1)
    out["coma_c31"] = coefficient(c, terms, 3, 1)
    out["coma_magnitude"] = float(np.hypot(out["coma_c3m1"], out["coma_c31"]))
    out["trefoil_c3m3"] = coefficient(c, terms, 3, -3)
    out["trefoil_c33"] = coefficient(c, terms, 3, 3)
    out["trefoil_magnitude"] = float(np.hypot(out["trefoil_c3m3"], out["trefoil_c33"]))
    out["sa_c40"] = coefficient(c, terms, 4, 0)
    out["sa_secundaria_c60"] = coefficient(c, terms, 6, 0)
    out["rms_lo_n2"] = float(np.sqrt(np.sum(c[nn == 2] ** 2)))
    out["rms_hoa_n3plus"] = float(np.sqrt(np.sum(c[nn >= 3] ** 2)))
    out["rms_n3"] = float(np.sqrt(np.sum(c[nn == 3] ** 2)))
    out["rms_n4"] = float(np.sqrt(np.sum(c[nn == 4] ** 2)))
    out["rms_n5plus"] = float(np.sqrt(np.sum(c[nn >= 5] ** 2)))
    out["rms_total_excl_piston_tilt"] = float(np.sqrt(np.sum(c[nn >= 2] ** 2)))
    return out


def astigmatism_from_coeffs(c2m2: float, c22: float, radius_mm: float) -> tuple[float, float]:
    """Cilindro (D) e eixo (graus) equivalentes dos termos de 2ª ordem (Thibos 2002, vetores de potência).

    J0 = -2√6 c22 / r², J45 = -2√6 c2m2 / r²  (c em µm, r em mm -> resultado em dioptrias)
    Cilindro (negativo) = -2 sqrt(J0² + J45²); eixo = ½ atan2(J45, J0).
    Trata-se da potência refrativa da frente de onda sobre a abertura (n = n_out),
    não da ceratometria (337,5/R); diferenças de ~10% são esperadas por convenção.
    """
    r2 = radius_mm**2
    J0 = -2.0 * sqrt(6.0) * c22 / r2
    J45 = -2.0 * sqrt(6.0) * c2m2 / r2
    cyl = -2.0 * sqrt(J0**2 + J45**2)
    axis = 0.5 * np.degrees(np.arctan2(J45, J0))
    if axis < 0:
        axis += 180.0
    return float(cyl), float(axis)
