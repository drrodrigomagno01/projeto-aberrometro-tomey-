"""Topografia no estilo do TMS: escala Standard, estatísticas por zona, filtros de visualização e página completa."""
import numpy as np
import pytest

from tmsab.adapters import import_tms_bundle
from tmsab.pipeline import analyze_exam, profiles_for_exam
from tmsab.synthetic import SyntheticSpec, write_export_folder_v1
from tmsab.topo import (TMS_STANDARD_COLORS, TMS_STANDARD_LEVELS, ViewOptions, draw_tms_map, figure_tms_view, power_on_circle,
                        select_profile, zone_axis_stats)


@pytest.fixture(scope="module")
def toric(tmp_path_factory):
    d = tmp_path_factory.mktemp("toric")
    write_export_folder_v1([SyntheticSpec(exam_id="X", R=7.8, Q=-0.2, cyl_D=2.0, axis_deg=30.0, eye="OD", pupil_center=(0.3, -0.2), decimals=4)], str(d))
    ex = import_tms_bundle(str(d), exam_id="90001")
    an, _ = analyze_exam(ex, profiles_for_exam(ex), with_maps=True)
    return str(d), ex, an


def test_standard_scale():
    assert len(TMS_STANDARD_COLORS) == 26 and len(TMS_STANDARD_LEVELS) == 27
    assert TMS_STANDARD_LEVELS[0] == 40.5 and TMS_STANDARD_LEVELS[-2] == 53.0 and TMS_STANDARD_LEVELS[-1] == 53.5
    assert np.allclose(np.diff(TMS_STANDARD_LEVELS), 0.5)


def test_zone_axis_stats_recover_axis_and_cylinder(toric):
    _, ex, _ = toric
    for z in (3.0, 5.0, 7.0):
        st = zone_axis_stats(ex, z, "DIO")
        assert len(st.rows) == 4 and all(np.isfinite(p) for p, _ in st.rows)
        # eixo do cilindro negativo 30° = meridiano plano; o íngreme fica a 120°
        assert abs(((st.steep_deg - 120.0) + 90) % 180 - 90) < 4.0, (z, st.steep_deg)
        assert abs(((st.flat_deg - 30.0) + 90) % 180 - 90) < 4.0, (z, st.flat_deg)
        assert 1.3 < st.steep_power - st.flat_power < 2.7, (z, st.steep_power, st.flat_power)
    p = power_on_circle(ex, "DIO", 5.0)
    assert np.isfinite(p).sum() > 200


def test_profiles_for_exam_includes_measured_pupil(toric):
    _, ex, an = toric
    names = [w.profile.name for w in an.wavefronts]
    assert "anterior_v1_vertice_3mm" in names and "anterior_v1_pupila_6mm" in names
    # pupila sintética de 4,0 mm: o perfil "pupila medida" coincide com o de 4 mm e não é duplicado
    assert any(w.profile.center_mode == "pupila" and abs(w.profile.diameter_mm - an.pupil.diameter) < 0.05 for w in an.wavefronts)
    assert sum(1 for w in an.wavefronts if w.profile.center_mode == "pupila" and abs(w.profile.diameter_mm - 4.0) < 0.05) == 1
    wf = select_profile(an, ViewOptions(pupil_mm=5.0, center="pupila"))
    assert wf.profile.diameter_mm == 5.0 and wf.profile.center_mode == "pupila"
    wf2 = select_profile(an, ViewOptions(pupil_mm=None, center="pupila"))     # "medida": pupila 4,0 mm centrada na pupila
    assert wf2.profile.center_mode == "pupila" and abs(wf2.profile.diameter_mm - 4.0) < 0.05
    wf3 = select_profile(an, ViewOptions(pupil_mm=None, center="vertice"))    # padrão da página: 6 mm no vértice
    assert wf3.profile.center_mode == "vertice" and wf3.profile.diameter_mm == 6.0


def test_draw_views(toric):
    from matplotlib.figure import Figure
    _, ex, an = toric
    for kind in ("axial", "tangencial", "refrativa", "elevacao", "frente_onda", "alta_ordem"):
        fig = Figure(figsize=(5, 5)); ax = fig.add_subplot(111)
        pm, levels, stats = draw_tms_map(ax, ex, an, ViewOptions(kind=kind, zone_mm=5.0, pupil_mm=4.0))
        assert pm is not None, kind
        if kind in ("axial", "tangencial", "refrativa"):
            assert len(stats) == 2                     # zonas 3 e 5 (7 mm fica fora da zona de 5 mm)
    f = figure_tms_view(ex, an, ViewOptions(kind="axial", scale="auto"))
    assert len(f.axes) >= 2
    fig2 = Figure(figsize=(5, 5)); ax2 = fig2.add_subplot(111)
    pm, _, _ = draw_tms_map(ax2, ex, None, ViewOptions(kind="frente_onda"))
    assert pm is None                                   # sem análise: aviso em vez de mapa


def test_web_topo_endpoints(toric):
    from tmsab.web import analyze_exam_html, completo_png, topo_png
    d, _, _ = toric
    assert "error" in completo_png(d, "90001", {})     # antes de calcular
    r = topo_png(d, "90001", {"kind": "axial", "scale": "tms", "zone_mm": 5, "pupil_mm": 4})
    assert r["png"].startswith("data:image/png;base64,") and len(r["stats"]) == 3 and r["profile"] is None
    import json
    json.loads(json.dumps(r))                           # sem NaN
    analyze_exam_html(d, "90001", {"pdf": False, "include_name": False})
    r2 = topo_png(d, "90001", {"kind": "alta_ordem", "pupil_mm": 5, "center": "pupila"})
    assert r2["profile"]["diameter_mm"] == 5.0 and r2["profile"]["center"] == "pupila"
    c = completo_png(d, "90001", {"scale": "auto", "pupil_mm": 3}, include_name=False)
    assert c["png"].startswith("data:image/png;base64,")
