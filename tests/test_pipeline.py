"""Cadeia completa (plano §8: 'Lacunas e deslocamento', 'Saídas', identidade do cálculo)."""
import json
import math

import numpy as np
import pytest

from tmsab.adapters import import_tms_bundle
from tmsab.model import OpticalProfile, QualityConfig, ReconstructionConfig, to_jsonable
from tmsab.optics import fit_wavefront, trace_wavefront
from tmsab.pipeline import analyze_exam, default_profiles
from tmsab.synthetic import build_surface


def _truth(spec, profile, cx=0.0, cy=0.0):
    fr = fit_wavefront(trace_wavefront(build_surface(spec), profile, cx, cy), profile.radius_mm, profile.zernike_order)
    return dict(zip(fr.terms, fr.coeffs))


def _wf(an, name):
    return next(w for w in an.wavefronts if w.profile.name == name)


@pytest.mark.parametrize("case,key,tol", [("con", (4, 0), 0.002), ("coma", (3, 1), 0.002), ("tref", (3, -3), 0.002), ("tor", (2, 2), 0.01)])
def test_recovery_against_truth(synth_root, case, key, tol):
    spec = synth_root["specs"][case]
    an, _ = analyze_exam(import_tms_bundle(synth_root["paths"][case]), with_maps=False)
    for name in ("anterior_v1_vertice_4mm", "anterior_v1_vertice_6mm"):
        w = _wf(an, name)
        assert w.validity == "disponivel", w.block_reasons
        got = next(c.value_um for c in w.coefficients if (c.n, c.m) == key)
        exp = _truth(spec, w.profile)[key]
        assert abs(got - exp) < tol and (np.sign(got) == np.sign(exp)), (name, got, exp)


def test_pupil_centered_profile_shows_decentration_coma(synth_root):
    spec = synth_root["specs"]["pup"]
    an, _ = analyze_exam(import_tms_bundle(synth_root["paths"]["pup"]), with_maps=False)
    w = _wf(an, "anterior_v1_pupila_6mm")
    assert w.validity == "disponivel" and w.center_xy == (0.3, -0.2)
    t = _truth(spec, w.profile, 0.3, -0.2)
    assert abs(w.metrics["coma_c31"] - t[(3, 1)]) < 0.003 and w.metrics["coma_c31"] > 0   # deslocamento +x -> coma +c31
    assert abs(w.metrics["coma_c3m1"] - t[(3, -1)]) < 0.003 and w.metrics["coma_c3m1"] < 0
    v = _wf(an, "anterior_v1_vertice_6mm")
    assert abs(v.metrics["coma_magnitude"]) < 1e-3     # no vértice não há coma


def test_pupil_profile_blocked_when_pupil_absent(synth_root):
    an, _ = analyze_exam(import_tms_bundle(synth_root["paths"]["nopup"]), with_maps=False)
    w = _wf(an, "anterior_v1_pupila_6mm")
    assert w.validity == "indisponivel" and any("pupilar" in r for r in w.block_reasons)
    assert w.coefficients == [] and w.metrics == {}
    assert _wf(an, "anterior_v1_vertice_6mm").validity == "disponivel"


def test_missing_sector_blocks_6mm_but_not_4mm(synth_root):
    an, _ = analyze_exam(import_tms_bundle(synth_root["paths"]["gapmiss"]), with_maps=False)
    assert _wf(an, "anterior_v1_vertice_4mm").validity == "disponivel"
    w6 = _wf(an, "anterior_v1_vertice_6mm")
    assert w6.validity == "indisponivel" and any("setor" in r or "borda" in r or "medidos" in r for r in w6.block_reasons)


def test_interpolated_sector_is_used_but_reported(synth_root):
    an, _ = analyze_exam(import_tms_bundle(synth_root["paths"]["gapint"]), with_maps=False)
    w = _wf(an, "anterior_v1_vertice_6mm")
    assert w.validity == "disponivel"
    assert w.support.measured_fraction == 1.0   # lacuna começa no anel 16 (> 3 mm)


def test_coarse_export_blocked_by_residual(synth_root):
    an, _ = analyze_exam(import_tms_bundle(synth_root["paths"]["coarse"]), with_maps=False)
    assert all(w.validity == "indisponivel" for w in an.wavefronts)
    assert all(any("resíduo" in r for r in w.block_reasons) for w in an.wavefronts)


def test_os_eye_keeps_laterality_and_no_mirroring(synth_root):
    od, _ = analyze_exam(import_tms_bundle(synth_root["paths"]["tor"]), with_maps=False)
    os_, _ = analyze_exam(import_tms_bundle(synth_root["paths"]["os"]), with_maps=False)
    assert od.identity.eye == "OD" and os_.identity.eye == "OS"
    a, b = _wf(od, "anterior_v1_vertice_6mm"), _wf(os_, "anterior_v1_vertice_6mm")
    assert abs(a.metrics["astig_c22"] - b.metrics["astig_c22"]) < 1e-6   # mesma superfície -> mesmos números; nenhuma simetrização automática


def test_analysis_identity_depends_on_configuration(synth_root):
    e = import_tms_bundle(synth_root["paths"]["con"])
    a1, _ = analyze_exam(e, with_maps=False)
    a2, _ = analyze_exam(e, with_maps=False)
    a3, _ = analyze_exam(e, default_profiles((4.0, 6.0), ("vertice",), 8), with_maps=False)
    a4, _ = analyze_exam(e, recon=ReconstructionConfig(surface_order=8), with_maps=False)
    assert a1.analysis_id == a2.analysis_id
    assert a1.analysis_id != a3.analysis_id and a1.analysis_id != a4.analysis_id
    assert a1.created_at != "" and a1.versions["nucleo"]


def test_json_serializable_without_nonfinite(synth_root):
    an, _ = analyze_exam(import_tms_bundle(synth_root["paths"]["nopup"]), with_maps=False)
    d = to_jsonable(an)
    s = json.dumps(d, allow_nan=False)      # falha se houver NaN/inf
    assert "indisponivel" in s
    w = next(x for x in d["wavefronts"] if x["validity"] == "indisponivel")
    assert w["center_xy"] == [None, None]   # ausência explícita, não zero


def test_quality_thresholds_are_provisional_and_configurable(synth_root):
    e = import_tms_bundle(synth_root["paths"]["gapmiss"])
    qc = QualityConfig(min_measured_fraction=0.5, min_usable_fraction=0.5, min_edge_meridians_fraction=0.5, max_missing_sector_deg=120)
    an, _ = analyze_exam(e, qc=qc, with_maps=False)
    assert _wf(an, "anterior_v1_vertice_6mm").validity == "disponivel"
    assert an.quality_config.provisional is True


def test_maps_generated_when_requested(synth_root):
    an, _ = analyze_exam(import_tms_bundle(synth_root["paths"]["con"]), [OpticalProfile(diameter_mm=6.0)], with_maps=True)
    g = an.wavefronts[0].map_grid
    assert g is not None and g["W"].shape == g["W_fit"].shape and np.isfinite(g["W"]).sum() > 1000
    assert np.nanmax(np.abs(g["residual"])) < 0.05
