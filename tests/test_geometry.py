"""Geometria (plano §8: 'Geometria', 'Orientação', 'Lacunas')."""
import numpy as np

from tmsab.adapters import import_tms_bundle
from tmsab.geometry import (
    BiconicSurface,
    ConicSurface,
    MirroredSurface,
    RotatedSurface,
    ZernikeDeformation,
    fit_conic,
    principal_curvatures,
    reconstruct_surface,
    support_metrics,
)
from tmsab.model import PointState, ReconstructionConfig


def _num_grad(s, x, y, h=1e-6):
    return (s.z(x + h, y) - s.z(x - h, y)) / (2 * h), (s.z(x, y + h) - s.z(x, y - h)) / (2 * h)


def test_conic_and_biconic_gradients():
    rng = np.random.default_rng(0)
    x = rng.uniform(-3, 3, 100); y = rng.uniform(-3, 3, 100)
    for s in (ConicSurface(7.8, -0.3), BiconicSurface(7.9, 7.5, -0.2, -0.4, axis_deg=35.0),
              ZernikeDeformation(4.5, 5, {(3, 1): 0.003, (4, 0): 0.001, (5, -3): 0.0005})):
        gx, gy = s.grad(x, y)
        nx, ny = _num_grad(s, x, y)
        assert np.abs(gx - nx).max() < 1e-7 and np.abs(gy - ny).max() < 1e-7


def test_conic_fit_recovers_parameters():
    s = ConicSurface(7.65, -0.35)
    r = np.linspace(0.3, 5.0, 300)
    R, Q, rms = fit_conic(r, s.z(r, np.zeros_like(r)))
    assert abs(R - 7.65) < 1e-9 and abs(Q + 0.35) < 1e-9 and rms < 1e-12


def test_principal_curvatures_toric():
    s = BiconicSurface(Rx=8.0, Ry=7.5, axis_deg=30.0)     # mais curvo em y' (eixo 30+90=120°)
    kmax, kmin, ang = principal_curvatures(s, 0.0, 0.0)
    assert abs(kmax - 1 / 7.5) < 1e-6 and abs(kmin - 1 / 8.0) < 1e-6
    assert abs(ang - 120.0) < 0.5


def test_mirror_and_rotation_wrappers():
    base = ZernikeDeformation(4.5, 3, {(3, 1): 0.002})   # coma horizontal
    mir = MirroredSurface(base)
    x = np.array([1.0, -2.0]); y = np.array([0.5, 1.5])
    assert np.allclose(mir.z(x, y), base.z(-x, y))
    rot = RotatedSurface(base, 90.0)
    # rotacionar coma horizontal 90° anti-horário vira coma vertical: z_rot(x,y) = z(y, -x)... conferir ponto a ponto
    assert np.allclose(rot.z(x, y), base.z(y, -x))
    gx, gy = rot.grad(x, y)
    nx, ny = _num_grad(rot, x, y)
    assert np.allclose(gx, nx, atol=1e-7) and np.allclose(gy, ny, atol=1e-7)


def test_reconstruction_residual_and_holdout(synth_root):
    exam = import_tms_bundle(synth_root["paths"]["coma"])
    rec = reconstruct_surface(exam.samples, ReconstructionConfig(holdout_rings=(5, 15)))
    assert rec.residual_rms < 2e-6                      # exportação com 6 casas: resíduo ~ nulo
    assert rec.holdout_residual_rms is not None and rec.holdout_residual_rms < 2e-5
    assert abs(rec.conic.R - 7.8) < 0.02
    assert rec.n_points_used == 23 * 256
    rec2 = reconstruct_surface(exam.samples, ReconstructionConfig(method="rbf_tps"))
    assert rec2.residual_rms < 5e-5
    # os dois métodos concordam na altura dentro de 4 mm
    u = np.linspace(-2, 2, 21); X, Y = np.meshgrid(u, u); m = X**2 + Y**2 <= 4
    assert np.abs(rec.z(X[m], Y[m]) - rec2.z(X[m], Y[m])).max() < 1e-4


def test_support_metrics_with_missing_sector(synth_root):
    exam = import_tms_bundle(synth_root["paths"]["gapmiss"])
    rec = reconstruct_surface(exam.samples, ReconstructionConfig())
    s4 = support_metrics(exam.samples, rec, 0.0, 0.0, 2.0)
    s6 = support_metrics(exam.samples, rec, 0.0, 0.0, 3.0)
    assert s4.measured_fraction == 1.0 and s4.max_missing_sector_deg == 0.0
    assert s6.measured_fraction < 1.0 and s6.max_missing_sector_deg >= 60.0
    assert s6.edge_reach_fraction < 0.9
    assert (exam.samples.point_state == PointState.MISSING).sum() > 0
