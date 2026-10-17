"""Base de Zernike (plano §8: 'Base de Zernike' e 'Saídas')."""
import numpy as np
import pytest

from tmsab.zernike import ZernikeBasis, astigmatism_from_coeffs, metrics_from_coefficients, nm_terms, osa_j, osa_nm, term_name


def test_osa_indices_roundtrip():
    for j in range(45):
        n, m = osa_nm(j)
        assert osa_j(n, m) == j
    assert osa_j(4, 0) == 12 and osa_j(3, -1) == 7 and osa_j(3, 1) == 8 and osa_j(2, 0) == 4
    assert nm_terms(2) == [(0, 0), (1, -1), (1, 1), (2, -2), (2, 0), (2, 2)]
    assert term_name(4, 0).startswith("aberracao esferica")


def test_orthonormality_on_disk():
    b = ZernikeBasis(6)
    n = 600
    u = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(u, u)
    m = X**2 + Y**2 <= 1
    A = b.design(np.hypot(X[m], Y[m]), np.arctan2(Y[m], X[m]))
    G = (A.T @ A) / m.sum()
    assert np.abs(G - np.eye(b.J)).max() < 5e-3


def test_gradient_matches_numeric():
    b = ZernikeBasis(8)
    rng = np.random.default_rng(0)
    rho = rng.uniform(0.05, 0.98, 50)
    th = rng.uniform(-np.pi, np.pi, 50)
    x, y = rho * np.cos(th), rho * np.sin(th)
    _, Ax, Ay = b.design_with_gradient(rho, th)
    h = 1e-6
    Axn = (b.design(np.hypot(x + h, y), np.arctan2(y, x + h)) - b.design(np.hypot(x - h, y), np.arctan2(y, x - h))) / (2 * h)
    Ayn = (b.design(np.hypot(x, y + h), np.arctan2(y + h, x)) - b.design(np.hypot(x, y - h), np.arctan2(y - h, x))) / (2 * h)
    assert np.abs(Ax - Axn).max() < 1e-6 and np.abs(Ay - Ayn).max() < 1e-6
    # origem: derivadas finitas (termos m=1 têm derivada não nula)
    _, Ax0, Ay0 = b.design_with_gradient(np.array([0.0]), np.array([0.0]))
    assert np.all(np.isfinite(Ax0)) and np.all(np.isfinite(Ay0))
    assert abs(Ax0[0, osa_j(1, 1)] - 2.0) < 1e-12   # Z(1,1) = 2ρcosθ -> dZ/dX = 2


def test_fit_recovers_known_coefficients_with_noise():
    b = ZernikeBasis(6)
    rng = np.random.default_rng(1)
    rho = np.sqrt(rng.uniform(0, 1, 4000))
    th = rng.uniform(-np.pi, np.pi, 4000)
    truth = np.zeros(b.J)
    truth[osa_j(4, 0)] = 0.25
    truth[osa_j(3, 1)] = -0.12
    truth[osa_j(2, 2)] = 0.8
    truth[osa_j(3, -3)] = 0.05
    W = b.design(rho, th) @ truth + rng.normal(0, 0.01, rho.size)
    fr = b.fit(rho, th, W)
    assert np.abs(fr.coeffs - truth).max() < 3e-3
    assert fr.condition_number < 10
    m = metrics_from_coefficients(fr.coeffs, fr.terms)
    assert abs(m["sa_c40"] - 0.25) < 3e-3
    assert abs(m["coma_c31"] + 0.12) < 3e-3
    assert abs(m["rms_hoa_n3plus"] - np.sqrt(0.25**2 + 0.12**2 + 0.05**2)) < 5e-3


def test_fit_rejects_insufficient_points():
    b = ZernikeBasis(6)
    with pytest.raises(ValueError):
        b.fit(np.array([0.1, 0.2]), np.array([0.0, 1.0]), np.array([1.0, 2.0]))


def test_astigmatism_conversion_sign_and_axis():
    # c22 negativo -> J0 positivo -> cilindro negativo no eixo 180 (a favor da regra)
    cyl, ax = astigmatism_from_coeffs(0.0, -1.0, 3.0)
    assert cyl < 0 and (abs(ax - 180) < 1e-6 or abs(ax) < 1e-6)
    cyl2, ax2 = astigmatism_from_coeffs(0.0, +1.0, 3.0)
    assert abs(ax2 - 90) < 1e-6
    # magnitude: c22 = -1 µm em r=3 mm -> J0 = 2√6/9 D -> cil = -2·J0
    assert abs(cyl - (-2 * 2 * np.sqrt(6) / 9)) < 1e-9
