"""Formato real do TMS-4N (perfil v1): pasta exportada com vários exames, XREF, PUP por ângulo, colunas de preenchimento."""
import json
import os

import numpy as np
import pytest

from tmsab.adapters import AdapterError, detect_profile, import_tms_bundle, list_exams, read_xref
from tmsab.model import PointState
from tmsab.optics import fit_wavefront, trace_wavefront
from tmsab.pipeline import analyze_exam
from tmsab.synthetic import SyntheticSpec, build_surface, write_export_folder_v1


@pytest.fixture(scope="module")
def export_v1(tmp_path_factory):
    d = tmp_path_factory.mktemp("export_v1")
    specs = [
        SyntheticSpec(exam_id="X", R=7.8, Q=-0.2, cyl_D=2.0, axis_deg=30.0, eye="OD", pupil_center=(0.3, -0.2), lid_gap_from_ring=16, decimals=4),
        SyntheticSpec(exam_id="X", R=7.8, Q=-0.2, zernike_um={"3,1": 3.0}, eye="OS", decimals=4),
        SyntheticSpec(exam_id="X", R=7.8, Q=-0.2, cyl_D=2.0, axis_deg=30.0, eye="OD", decimals=4),
    ]
    write_export_folder_v1(specs, str(d))
    return {"dir": str(d), "specs": specs}


def test_detect_and_list(export_v1):
    d = export_v1["dir"]
    assert detect_profile(d) == "tms4_datatables_v1"
    exams = list_exams(d)
    assert [e["exam_id"] for e in exams] == ["90001", "90002", "90003"]
    assert [e["eye"] for e in exams] == ["OD", "OS", "OD"]
    assert [e["capture_index"] for e in exams] == [1, 1, 2]       # segunda captura do OD
    assert all(e["acquired_at"].startswith("2026-06-17T09:3") for e in exams)


def test_xref_never_exposes_names(export_v1):
    x = read_xref(os.path.join(export_v1["dir"], "XREF.dat"))
    assert set(x) == {"90001", "90002", "90003"}
    for v in x.values():
        assert "Last Name" not in v["indices"] and "First Name" not in v["indices"] and "ID" not in v["indices"]
        assert "SINTETICO" not in json.dumps(v)
    assert x["90001"]["eye"] == "OD" and abs(x["90001"]["indices"]["CYL"] - 2.0) < 1e-9


def test_multi_exam_folder_requires_selection(export_v1):
    with pytest.raises(AdapterError, match="exam_id"):
        import_tms_bundle(export_v1["dir"])
    with pytest.raises(AdapterError, match="não encontrado"):
        import_tms_bundle(export_v1["dir"], exam_id="12345")


def test_import_real_format_structure(export_v1):
    ex = import_tms_bundle(export_v1["dir"], exam_id="90001")
    s = ex.samples
    assert s.n_rings == 25 and s.n_angles == 256                   # colunas 26–34 removidas
    assert ex.identity.eye == "OD" and ex.identity.capture_index == 1 and ex.identity.date_precision == "segundo"
    assert ex.identity.study_id.startswith("P-") and "SINTETICO" not in ex.identity.study_id
    assert (s.point_state == PointState.INTERPOLATED).sum() > 0     # lacuna de pálpebra marcada por sinal negativo
    assert (s.point_state == PointState.MISSING).sum() == 0
    assert ex.pupil.state == "registrada"
    assert abs(ex.pupil.center_x - 0.3) < 0.01 and abs(ex.pupil.center_y + 0.2) < 0.01 and abs(ex.pupil.diameter - 4.0) < 0.02
    assert "XREF" in {f.role for f in ex.bundle.files} and "AVE" in {f.role for f in ex.bundle.files}
    assert ex.bundle.export_settings["tms_indices"]["SimK1"] == 44.70
    assert ex.frame.transform_verified is True


def test_real_format_recovers_astigmatism_axis_and_coma(export_v1):
    d, specs = export_v1["dir"], export_v1["specs"]
    an, _ = analyze_exam(import_tms_bundle(d, exam_id="90001"), with_maps=False)
    w = next(x for x in an.wavefronts if x.profile.name == "anterior_v1_vertice_6mm")
    fr = fit_wavefront(trace_wavefront(build_surface(specs[0]), w.profile), 3.0, 6)
    assert abs(w.metrics["astig_c22"] - fr.coeffs[fr.terms.index((2, 2))]) < 0.01
    assert abs(w.metrics["astig_c2m2"] - fr.coeffs[fr.terms.index((2, -2))]) < 0.01
    assert abs(w.metrics["eixo_equivalente_deg"] - 30.0) < 1.0
    an2, _ = analyze_exam(import_tms_bundle(d, exam_id="90002"), with_maps=False)
    w2 = next(x for x in an2.wavefronts if x.profile.name == "anterior_v1_vertice_6mm")
    fr2 = fit_wavefront(trace_wavefront(build_surface(specs[1]), w2.profile), 3.0, 6)
    assert abs(w2.metrics["coma_c31"] - fr2.coeffs[fr2.terms.index((3, 1))]) < 0.01 and w2.metrics["coma_c31"] > 0


def test_reports_contain_no_names(export_v1, tmp_path):
    from tmsab.reporting import export_all
    ex = import_tms_bundle(export_v1["dir"], exam_id="90001")
    an, _ = analyze_exam(ex, with_maps=False)
    paths = export_all(ex, an, str(tmp_path / "out"))
    for k in ("json", "csv"):
        txt = open(paths[k], encoding="utf-8").read()
        assert "SINTETICO" not in txt and "PACIENTE" not in txt
    assert os.path.basename(paths["json"]).startswith(ex.identity.study_id + "_90001_OD_")


def test_study_code_from_xref_is_stable_across_folders(tmp_path):
    """O código de estudo vem do XREF (hash do nome/ID): o mesmo paciente exportado para pastas com nomes
    diferentes recebe o mesmo código; pacientes diferentes recebem códigos diferentes; o nome nunca aparece."""
    spec = [SyntheticSpec(exam_id="X", R=7.8, Q=-0.2, eye="OD", decimals=4)]
    write_export_folder_v1(spec, str(tmp_path / "pasta um"), first_exam=100, patient_name=("SOBRENOME", "NOME"))
    write_export_folder_v1(spec, str(tmp_path / "pasta dois"), first_exam=100, patient_name=("SOBRENOME", "NOME"))
    write_export_folder_v1(spec, str(tmp_path / "pasta tres"), first_exam=100, patient_name=("OUTRO", "NOME"))
    a = import_tms_bundle(str(tmp_path / "pasta um"))
    b = import_tms_bundle(str(tmp_path / "pasta dois"))
    c = import_tms_bundle(str(tmp_path / "pasta tres"))
    assert a.identity.study_id == b.identity.study_id != c.identity.study_id
    assert a.identity.study_id.startswith("P-") and len(a.identity.study_id) == 10
    for ex in (a, b, c):
        blob = json.dumps(ex.bundle.export_settings) + ex.identity.study_id + " ".join(ex.diagnostics + ex.warnings)
        assert "SOBRENOME" not in blob and "OUTRO" not in blob and "NOME" not in blob
    # exportações repetidas do mesmo exame produzem arquivos idênticos → mesma identidade de cálculo
    an_a, _ = analyze_exam(a, with_maps=False)
    an_b, _ = analyze_exam(b, with_maps=False)
    assert an_a.analysis_id == an_b.analysis_id


def test_two_patients_in_one_folder(tmp_path):
    """Exportar de novo para a mesma pasta: o TMS acrescenta linhas ao XREF.dat → dois pacientes na pasta.
    O programa separa-os pelo código, conta capturas por paciente/olho e avisa."""
    d = str(tmp_path / "mista")
    write_export_folder_v1([SyntheticSpec(exam_id="X", R=7.8, Q=-0.2, eye="OD", decimals=4),
                            SyntheticSpec(exam_id="X", R=7.8, Q=-0.2, eye="OD", decimals=4)],
                           d, first_exam=200, patient_name=("PRIMEIRO", "PACIENTE"))
    write_export_folder_v1([SyntheticSpec(exam_id="X", R=7.6, Q=-0.3, cyl_D=2.5, axis_deg=172.0, eye="OD", decimals=4)],
                           d, first_exam=300, patient_name=("SEGUNDO", "PACIENTE"), append=True, date_str="05/08/2026")
    x = read_xref(os.path.join(d, "XREF.dat"))
    assert set(x) == {"200", "201", "300"}
    exams = {e["exam_id"]: e for e in list_exams(d)}
    assert exams["200"]["patient_key"] == exams["201"]["patient_key"] != exams["300"]["patient_key"]
    assert [exams[k]["capture_index"] for k in ("200", "201", "300")] == [1, 2, 1]
    e1 = import_tms_bundle(d, exam_id="201")
    e2 = import_tms_bundle(d, exam_id="300")
    assert e1.identity.study_id != e2.identity.study_id
    assert e1.identity.capture_index == 2 and e2.identity.capture_index == 1
    assert e1.bundle.export_settings["patients_in_folder"] == 2
    assert any("2 pacientes" in w for w in e1.warnings) and any("2 pacientes" in w for w in e2.warnings)
    assert "PRIMEIRO" not in " ".join(e1.warnings + e2.warnings + e1.diagnostics + e2.diagnostics)
    an, _ = analyze_exam(e2, with_maps=False)
    w = next(x for x in an.wavefronts if x.profile.name == "anterior_v1_vertice_4mm")
    assert abs(((w.metrics["eixo_equivalente_deg"] - 172.0) + 90) % 180 - 90) < 1.5
    assert w.metrics["cilindro_equivalente_D"] < -2.0
