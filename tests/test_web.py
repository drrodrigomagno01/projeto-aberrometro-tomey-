"""Versão HTML (navegador): o módulo tmsab.web roda em CPython com o mesmo contrato usado pelo Pyodide."""
import json
import os
import re

import numpy as np

import pytest

from tmsab.synthetic import SyntheticSpec, write_export_folder_v1
from tmsab.web import analyze_exam_html, list_folder, report_document


@pytest.fixture(scope="module")
def export_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("web_export")
    write_export_folder_v1([
        SyntheticSpec(exam_id="X", R=7.8, Q=-0.2, cyl_D=2.0, axis_deg=30.0, eye="OD", pupil_center=(0.3, -0.2), decimals=4),
        SyntheticSpec(exam_id="X", R=7.8, Q=-0.2, eye="OS", decimals=4),
    ], str(d))
    return str(d)


def test_list_folder(export_dir):
    r = list_folder(export_dir)
    assert [e["exam_id"] for e in r["exams"]] == ["90001", "90002"]
    assert r["exams"][0]["simk"]["CYL"] == 2.0 and r["patients"] and r["warnings"] == []
    assert "SINTETICO" not in json.dumps(r)


def test_analyze_exam_html_contract(export_dir):
    r = analyze_exam_html(export_dir, "90001", {"pdf": False, "diameters": [4.0, 6.0], "centers": ["vertice", "pupila"]})
    assert set(["html", "json", "csv", "summary", "stem", "analysis_id", "study_id"]) <= set(r)
    assert "pdf_b64" not in r
    profs = {p["name"]: p for p in r["summary"]["profiles"]}
    assert profs["anterior_v1_vertice_6mm"]["validity"] == "disponivel"
    assert abs(profs["anterior_v1_vertice_6mm"]["metrics"]["eixo_equivalente_deg"] - 30.0) < 1.0
    html = r["html"]
    assert html.startswith("<article class='exame'") and "DISPONÍVEL" in html and "data:image/png;base64," in html
    assert "Código do estudo" in html and r["study_id"] in html
    assert "SINTETICO" not in html and "PACIENTE" not in html
    json.loads(r["json"])                      # JSON canônico válido
    assert r["csv"].splitlines()[0].startswith("analysis_id;study_id")
    doc = report_document([html])
    assert doc.startswith("<!doctype html>") and "@media print" in doc


def test_analyze_exam_html_pdf_and_error(export_dir):
    r = analyze_exam_html(export_dir, "90002", {"pdf": True, "maps": False, "diameters": [6.0], "centers": ["vertice"]})
    assert r["pdf_b64"][:5] == "JVBER"              # %PDF- em base64
    assert "data:image/png" not in r["html"]        # sem mapas
    bad = analyze_exam_html(export_dir, "99999")
    assert "error" in bad and "não encontrado" in bad["error"]


def test_build_html(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_html", os.path.join(os.path.dirname(__file__), "..", "web", "build_html.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    path = mod.build(str(tmp_path))
    txt = open(path, encoding="utf-8").read()
    assert not re.search(r"\{\{[A-Z_]+\}\}", txt)
    assert "cdn.jsdelivr.net/pyodide/v" in txt and "setupEngine" in txt and "tmsab.web" in txt
    assert os.path.getsize(path) < 2_000_000


def test_resultado_completo_figure_and_html(export_dir, tmp_path):
    """Página única: 4 mapas topográficos + frente de onda + tabelas; entra como 1ª página do PDF e 1ª figura do HTML."""
    from pypdf import PdfReader
    from tmsab.adapters import import_tms_bundle
    from tmsab.pipeline import analyze_exam
    from tmsab.plots import best_fit_sphere, figure_resultado_completo, hoa_map
    from tmsab.reporting import export_pdf
    ex = import_tms_bundle(export_dir, exam_id="90001")
    an, _ = analyze_exam(ex, with_maps=True)
    fig = figure_resultado_completo(ex, an)
    titles = [a.get_title() for a in fig.axes if a.get_title()]
    assert any("axial" in t for t in titles) and any("tangencial" in t for t in titles) and any("refrativa" in t for t in titles)
    assert any("Elevação" in t for t in titles) and any("Alta ordem" in t for t in titles) and any("Zernike" in t for t in titles)
    assert abs(fig.get_size_inches()[0] - 8.27) < 0.01 and abs(fig.get_size_inches()[1] - 11.69) < 0.01
    # esfera de melhor ajuste de uma esfera exata recupera o raio
    t = np.linspace(0, 2 * np.pi, 200); r = np.linspace(0.3, 4.0, 15)[:, None]
    x, y = r * np.cos(t)[None, :], r * np.sin(t)[None, :]
    z = 7.8 - np.sqrt(7.8**2 - x**2 - y**2)
    a, b, c, R = best_fit_sphere(x, y, z)
    assert abs(R - 7.8) < 1e-6 and abs(a) < 1e-6 and abs(b) < 1e-6 and abs(c - 7.8) < 1e-6
    wf = next(w for w in an.wavefronts if w.profile.name == "anterior_v1_vertice_6mm")
    Z = hoa_map(wf, n=41)
    assert Z.shape == (41, 41) and np.isfinite(Z[20, 20]) and np.isnan(Z[0, 0])
    pdf = export_pdf(ex, an, str(tmp_path / "r.pdf"))
    assert len(PdfReader(pdf).pages) >= 4
    r = analyze_exam_html(export_dir, "90001", {"pdf": False})
    assert "<figure class='completo'>" in r["html"] and r["html"].index("completo") < r["html"].index("class='detalhes'")
