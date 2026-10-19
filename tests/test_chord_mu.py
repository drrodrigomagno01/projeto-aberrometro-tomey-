"""Chord mu aparente (tmsab/pupila.py): centro da pupila desta captura em relação ao vértice do TMS.

OD e OS com a mesma pupila deslocada no referencial do exame (vista do examinador): +x é nasal em OD e temporal em
OS. Sem pupila registrada não há chord mu (nunca vira o vértice). Todas as saídas leem o mesmo objeto.
"""
import csv
import io
import json

import pytest

from tmsab.adapters import import_tms_bundle
from tmsab.cli import _kerato_long_rows
from tmsab.model import PupilData, to_jsonable
from tmsab.pipeline import analyze_exam
from tmsab.plots import figure_resultado_completo
from tmsab.pupila import chord_mu_aparente, fmt_chord_mu
from tmsab.reporting import PUPILA_CSV_PERFIL, export_csv, report_lines
from tmsab.synthetic import SyntheticSpec, write_export_folder_v1
from tmsab.web import analyze_exam_html


@pytest.fixture(scope="module")
def exames(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("chordmu"))
    write_export_folder_v1([
        SyntheticSpec(exam_id="C", R=7.8, Q=-0.2, eye="OD", decimals=4, pupil_diameter=5.0, pupil_center=(0.3, -0.2)),
        SyntheticSpec(exam_id="C", R=7.8, Q=-0.2, eye="OS", decimals=4, pupil_diameter=5.0, pupil_center=(0.3, -0.2)),
        SyntheticSpec(exam_id="C", R=7.8, Q=-0.2, eye="OD", decimals=4, pupil_registered=False),
    ], d)
    out = {"dir": d}
    for i, chave in enumerate(("od", "os", "sem")):
        ex = import_tms_bundle(d, exam_id=str(90001 + i))
        an, _ = analyze_exam(ex, with_maps=False)
        out[chave] = (ex, an)
    return out


def test_od_pupila_nasal_e_inferior(exames):
    _, an = exames["od"]
    cm = an.chord_mu
    assert cm.distancia_mm == pytest.approx(0.3606, abs=0.02)
    assert cm.nasal_mm == pytest.approx(0.30, abs=0.02) and cm.superior_mm == pytest.approx(-0.20, abs=0.02)
    assert fmt_chord_mu(cm).startswith("0.36 mm — nasal 0.30, inferior 0.20") and "pupila 5.00 mm" in fmt_chord_mu(cm)


def test_os_mesmo_deslocamento_vira_temporal(exames):
    _, an = exames["os"]
    assert an.chord_mu.nasal_mm == pytest.approx(-0.30, abs=0.02)
    assert fmt_chord_mu(an.chord_mu, curto=True) == "0.36 mm — temporal 0.30, inferior 0.20"


def test_sem_pupila_nao_ha_chord_mu(exames):
    _, an = exames["sem"]
    assert an.chord_mu is None and fmt_chord_mu(None).startswith("não calculado")
    assert not any("chord mu" in ln for ln in report_lines(an))


def test_olho_desconhecido_mostra_so_x_e_y():
    cm = chord_mu_aparente(PupilData(state="registrada", center_x=0.1, center_y=-0.0001, diameter=6.0), None)
    assert cm.nasal_mm is None and fmt_chord_mu(cm, curto=True) == "0.10 mm — x +0.10, y +0.00 mm (olho não identificado)"


def test_todas_as_saidas_trazem_o_mesmo_chord_mu(exames, tmp_path):
    ex, an = exames["od"]
    cm = an.chord_mu
    d = to_jsonable(an)
    json.dumps(d, allow_nan=False)
    assert d["chord_mu"]["distancia_mm"] == cm.distancia_mm and d["chord_mu"]["nasal_mm"] == cm.nasal_mm
    # CSV principal: linhas de métrica da pupila (nome;valor), 18 colunas como as dos perfis
    p = tmp_path / "a.csv"
    export_csv(an, str(p))
    rows = list(csv.reader(io.StringIO(p.read_text(encoding="utf-8")), delimiter=";"))
    lin = {r[16]: r[17] for r in rows if len(r) == 18 and r[4] == PUPILA_CSV_PERFIL}
    assert all(len(r) <= 18 for r in rows)
    assert float(lin["chord_mu_aparente_mm"]) == pytest.approx(cm.distancia_mm, abs=1e-4)
    assert float(lin["chord_mu_nasal_mm"]) == pytest.approx(cm.nasal_mm, abs=1e-4) and lin["pupila_estado"] == "registrada"
    # relatório em texto, folha A4 e resumo do lote
    assert any(ln.startswith("Chord mu aparente") and fmt_chord_mu(cm) in ln for ln in report_lines(an))
    fig = figure_resultado_completo(ex, an, patient_name=None)
    assert any(t.get_text().startswith("chord mu aparente " + fmt_chord_mu(cm, curto=True)) for t in fig.texts)
    longo = {r["metrica"]: r["valor"] for r in _kerato_long_rows(an)}
    assert float(longo["chord_mu_aparente_mm"]) == pytest.approx(cm.distancia_mm, abs=1e-4)
    assert float(longo["chord_mu_nasal_mm"]) == pytest.approx(cm.nasal_mm, abs=1e-4)
    assert "SINTETICO" not in json.dumps(d) and "PACIENTE" not in p.read_text(encoding="utf-8")
    # versão HTML: resumo da página e detalhes
    w = analyze_exam_html(exames["dir"], "90001", {"maps": False, "pdf": False, "include_name": False})
    assert w["summary"]["chord_mu"]["distancia_mm"] == pytest.approx(cm.distancia_mm, abs=1e-12)
    assert "chord mu aparente " + fmt_chord_mu(cm) in w["html"].replace("&#x27;", "'")


@pytest.mark.parametrize("olho, x, y, esperado", [
    ("OD", +0.20, +0.10, "0.22 mm — nasal 0.20, superior 0.10"),
    ("OD", -0.20, +0.10, "0.22 mm — temporal 0.20, superior 0.10"),
    ("OD", -0.20, -0.10, "0.22 mm — temporal 0.20, inferior 0.10"),
    ("OS", +0.20, +0.10, "0.22 mm — temporal 0.20, superior 0.10"),
    ("OS", -0.20, -0.10, "0.22 mm — nasal 0.20, inferior 0.10"),
    ("OS", +0.20, -0.10, "0.22 mm — temporal 0.20, inferior 0.10"),
])
def test_quadrantes_e_olhos(olho, x, y, esperado):
    cm = chord_mu_aparente(PupilData(state="registrada", center_x=x, center_y=y, diameter=6.0), olho)
    assert fmt_chord_mu(cm, curto=True) == esperado
    assert cm.nasal_mm == pytest.approx(x if olho == "OD" else -x) and cm.superior_mm == pytest.approx(y)


def test_resumo_do_lote_tem_o_chord_mu_mesmo_sem_ceratometria(exames):
    _, an = exames["od"]
    import copy
    sem = copy.copy(an)
    sem.keratometry = None                         # análise antiga ou ceratometria desligada
    longo = {r["metrica"]: r["valor"] for r in _kerato_long_rows(sem)}
    assert float(longo["chord_mu_aparente_mm"]) == pytest.approx(an.chord_mu.distancia_mm, abs=1e-4)
    assert not any("@" in m for m in longo)
