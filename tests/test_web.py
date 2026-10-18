"""Versão HTML (navegador): o módulo tmsab.web roda em CPython com o mesmo contrato usado pelo Pyodide."""
import json
import os
import re

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
