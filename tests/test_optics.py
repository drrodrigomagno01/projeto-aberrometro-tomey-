"""Óptica (plano §8: 'Traçado de raios', 'Convergência', 'Orientação', 'Casos ópticos que evitam falsos sucessos')."""
import numpy as np

from tmsab.geometry import BiconicSurface, CompositeSurface, ConicSurface, MirroredSurface, RotatedSurface, ZernikeDeformation
from tmsab.model import OpticalProfile
from tmsab.optics import fit_wavefront, refocus_rms, trace_wavefront
from tmsab.reference.meridional import meridional_wavefront_conic
from tmsab.zernike import osa_j

N_OUT = 1.376


def _coef(fr, n, m):
    return float(fr.coeffs[fr.terms.index((n, m))])


def test_descartes_oval_has_zero_wavefront():
    s = ConicSurface(7.8, -1.0 / N_OUT**2)
    for D in (4.0, 6.0, 8.0):
        tr = trace_wavefront(s, OpticalProfile(diameter_mm=D))
        assert tr.n_valid == tr.n_rays
        assert np.nanmax(np.abs(tr.W_um)) < 1e-4


def test_sphere_has_positive_primary_sa_matching_seidel():
    s = ConicSurface(7.8, 0.0)
    a = 1.5
    fr = fit_wavefront(trace_wavefront(s, OpticalProfile(diameter_mm=2 * a)), a, 6)
    c40 = _coef(fr, 4, 0)
    seidel = (N_OUT - 1) * (1 / N_OUT**2) / (8 * 7.8**3) * a**4 / (6 * np.sqrt(5)) * 1000
    assert c40 > 0 and abs(c40 - seidel) / seidel < 0.05
    # simetria axial: termos não simétricos nulos
    for (n, m) in ((3, 1), (3, -1), (3, 3), (3, -3), (2, 2), (2, -2), (1, 1), (1, -1)):
        assert abs(_coef(fr, n, m)) < 1e-9


def test_conic_q_dependence_matches_seidel_factor():
    """W040 ∝ (1/n² + Q): Q=-0.26 reduz a SA pela razão (1/n²-0.26)/(1/n²)."""
    a = 1.5
    f0 = fit_wavefront(trace_wavefront(ConicSurface(7.8, 0.0), OpticalProfile(diameter_mm=2 * a)), a, 6)
    f1 = fit_wavefront(trace_wavefront(ConicSurface(7.8, -0.26), OpticalProfile(diameter_mm=2 * a)), a, 6)
    ratio = _coef(f1, 4, 0) / _coef(f0, 4, 0)
    expected = (1 / N_OUT**2 - 0.26) / (1 / N_OUT**2)
    assert abs(ratio - expected) < 0.02


def test_3d_trace_matches_independent_meridional_reference():
    for R, Q in ((7.7, 0.0), (7.7, -0.3), (8.2, -0.6)):
        tr = trace_wavefront(ConicSurface(R, Q), OpticalProfile(diameter_mm=6.0))
        m = np.abs(tr.y) < 1e-9
        Wref = meridional_wavefront_conic(R, Q, N_OUT, tr.x[m])
        assert np.nanmax(np.abs(tr.W_um[m] - Wref)) < 1e-4


def test_toric_gives_astigmatism_with_correct_axis_and_sign():
    # mais curvo no meridiano vertical (Ry < Rx): a favor da regra -> c22 < 0, c2-2 ≈ 0
    s = BiconicSurface(Rx=7.9, Ry=7.6, Qx=-0.2, Qy=-0.2, axis_deg=0.0)
    fr = fit_wavefront(trace_wavefront(s, OpticalProfile(diameter_mm=6.0)), 3.0, 6)
    assert _coef(fr, 2, 2) < -0.5 and abs(_coef(fr, 2, -2)) < 1e-6
    s45 = BiconicSurface(Rx=7.9, Ry=7.6, Qx=-0.2, Qy=-0.2, axis_deg=45.0)
    fr45 = fit_wavefront(trace_wavefront(s45, OpticalProfile(diameter_mm=6.0)), 3.0, 6)
    assert abs(_coef(fr45, 2, 2)) < 1e-6 and abs(abs(_coef(fr45, 2, -2)) - abs(_coef(fr, 2, 2))) < 1e-3


def test_mirror_flips_horizontal_coma_sign_only():
    base = CompositeSurface([ConicSurface(7.8, -0.2), ZernikeDeformation(4.5, 3, {(3, 1): 0.003, (3, -1): 0.001})])
    fr = fit_wavefront(trace_wavefront(base, OpticalProfile(diameter_mm=6.0)), 3.0, 6)
    frm = fit_wavefront(trace_wavefront(MirroredSurface(base), OpticalProfile(diameter_mm=6.0)), 3.0, 6)
    assert abs(_coef(frm, 3, 1) + _coef(fr, 3, 1)) < 1e-6      # coma horizontal inverte
    assert abs(_coef(frm, 3, -1) - _coef(fr, 3, -1)) < 1e-6     # coma vertical mantém
    assert abs(_coef(frm, 4, 0) - _coef(fr, 4, 0)) < 1e-9       # SA mantém


def test_rotation_by_90_moves_coma_between_components():
    base = CompositeSurface([ConicSurface(7.8, -0.2), ZernikeDeformation(4.5, 3, {(3, 1): 0.003})])
    fr = fit_wavefront(trace_wavefront(base, OpticalProfile(diameter_mm=6.0)), 3.0, 6)
    frr = fit_wavefront(trace_wavefront(RotatedSurface(base, 90.0), OpticalProfile(diameter_mm=6.0)), 3.0, 6)
    assert abs(_coef(frr, 3, -1) - _coef(fr, 3, 1)) < 1e-6       # coma horizontal -> vertical (+)
    assert abs(_coef(frr, 3, 1)) < 1e-6


def test_convergence_with_ray_grid():
    s = CompositeSurface([ConicSurface(7.8, -0.2), ZernikeDeformation(4.5, 4, {(3, 1): 0.002, (4, 0): 0.001})])
    vals = []
    for g in (49, 97, 193):
        fr = fit_wavefront(trace_wavefront(s, OpticalProfile(diameter_mm=6.0, ray_grid=g)), 3.0, 6)
        vals.append((_coef(fr, 4, 0), _coef(fr, 3, 1)))
    assert abs(vals[1][0] - vals[2][0]) < 2e-4 and abs(vals[1][1] - vals[2][1]) < 2e-4
    assert abs(vals[0][0] - vals[2][0]) < 2e-3


def test_refocus_option_keeps_sa_and_registers_note():
    s = ConicSurface(7.8, -0.2)
    p = OpticalProfile(diameter_mm=6.0, reference="refoco_rms")
    tr = refocus_rms(s, p, 0.0, 0.0)
    assert any("refoco" in n for n in tr.notes)
    fr = fit_wavefront(tr, 3.0, 6)
    tr0 = trace_wavefront(s, OpticalProfile(diameter_mm=6.0))
    fr0 = fit_wavefront(tr0, 3.0, 6)
    assert abs(_coef(fr, 4, 0) - _coef(fr0, 4, 0)) < 0.01   # SA quase inalterada; defocus muda
    assert abs(_coef(fr, 2, 0)) < abs(_coef(fr0, 2, 0))


def test_units_mm_to_um_single_conversion_point():
    s = ConicSurface(7.8, 0.0)
    tr = trace_wavefront(s, OpticalProfile(diameter_mm=6.0))
    # ordem de grandeza: esfera 7,8 mm a 6 mm -> W máximo de alguns µm (não mm nem nm)
    assert 1.0 < np.nanmax(np.abs(tr.W_um)) < 20.0
