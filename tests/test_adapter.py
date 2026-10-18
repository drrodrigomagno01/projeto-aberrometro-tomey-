"""Importador (plano §8: 'Leitura lexical', 'Angle e Ring', 'Pareamento', 'Sinais e ausências')."""
import json
import os
import shutil

import numpy as np
import pytest

from tmsab.adapters import AdapterError, import_tms_bundle, list_profiles, load_profile
from tmsab.model import PointState


def _copy(src, dst):
    shutil.copytree(src, dst)
    return dst


def test_profiles_listed():
    assert "tms4_datatables_v0" in list_profiles()
    prof = load_profile("tms4_datatables_v0")
    assert prof["status"] != "verificado"        # nunca declarar compatibilidade sem arquivos reais


def test_roundtrip_angle_and_ring_orientations_equivalent(synth_root):
    a = import_tms_bundle(synth_root["paths"]["con"])
    b = import_tms_bundle(synth_root["paths"]["ring"])
    assert a.samples.n_rings == b.samples.n_rings == 25 and a.samples.n_angles == 256
    assert np.allclose(a.samples.z, b.samples.z, equal_nan=True)
    assert np.allclose(a.samples.r, b.samples.r, equal_nan=True)
    assert a.bundle.export_settings["orientations"]["HIT"] == "angle_rows"
    assert b.bundle.export_settings["orientations"]["HIT"] == "ring_rows"
    assert b.bundle.files[0].delimiter == ";" or any(f.delimiter == ";" for f in b.bundle.files)


def test_states_and_sign_rules(synth_root):
    e = import_tms_bundle(synth_root["paths"]["gapint"])
    st = e.samples.point_state
    assert (st == PointState.INTERPOLATED).sum() > 0 and (st == PointState.MISSING).sum() == 0
    # valores interpolados foram decodificados sem sinal negativo (regra do campo), não por valor absoluto global
    assert np.nanmin(e.samples.z) > 0
    e2 = import_tms_bundle(synth_root["paths"]["gapmiss"])
    st2 = e2.samples.point_state
    assert (st2 == PointState.MISSING).sum() > 0
    assert np.all(np.isnan(e2.samples.z[st2 == PointState.MISSING]))   # zero-ausente vira ausência, não geometria


def test_hashes_and_identity(synth_root):
    e = import_tms_bundle(synth_root["paths"]["con"])
    assert e.identity.exam_id == "T-CON" and e.identity.eye == "OD"
    assert all(len(f.sha256) == 64 for f in e.bundle.files)
    assert e.bundle.synthetic is True
    assert e.frame.transform_verified is True      # sintético: transformação conhecida


def test_truncated_file_rejected(synth_root, tmp_path):
    d = _copy(synth_root["paths"]["con"], tmp_path / "trunc")
    p = d / "T-CON_OD_HIT.csv"
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines[: len(lines) // 2]) + "\n")
    with pytest.raises(AdapterError, match="dimens"):
        import_tms_bundle(str(d))


def test_corrupted_cell_rejected(synth_root, tmp_path):
    d = _copy(synth_root["paths"]["con"], tmp_path / "corrupt")
    p = d / "T-CON_OD_HIT.csv"
    txt = p.read_text().splitlines()
    parts = txt[10].split(",")
    parts[3] = "abc"
    txt[10] = ",".join(parts)
    p.write_text("\n".join(txt) + "\n")
    with pytest.raises(AdapterError):
        import_tms_bundle(str(d))


def test_mixed_eyes_rejected(synth_root, tmp_path):
    d = _copy(synth_root["paths"]["con"], tmp_path / "mixed")
    os.rename(d / "T-CON_OD_RAD.csv", d / "T-CON_OS_RAD.csv")
    (d / "manifest.json").unlink()
    with pytest.raises(AdapterError, match="olhos"):
        import_tms_bundle(str(d))


def test_manifest_eye_conflict_rejected(synth_root, tmp_path):
    d = _copy(synth_root["paths"]["con"], tmp_path / "conflict")
    m = json.loads((d / "manifest.json").read_text())
    m["eye"] = "OS"
    (d / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(AdapterError, match="lateralidade"):
        import_tms_bundle(str(d))


def test_missing_required_table_rejected(synth_root, tmp_path):
    d = _copy(synth_root["paths"]["con"], tmp_path / "norad")
    (d / "T-CON_OD_RAD.csv").unlink()
    with pytest.raises(AdapterError, match="RAD"):
        import_tms_bundle(str(d))


def test_wrong_unit_rejected(synth_root, tmp_path):
    """RAD em µm (1000x) deve ser bloqueado pela faixa física, não convertido silenciosamente."""
    d = _copy(synth_root["paths"]["con"], tmp_path / "unit")
    p = d / "T-CON_OD_RAD.csv"
    out = []
    for ln in p.read_text().splitlines():
        if ln.startswith("#"):
            out.append(ln); continue
        out.append(",".join(f"{float(v)*1000:.1f}" for v in ln.split(",")))
    p.write_text("\n".join(out) + "\n")
    with pytest.raises(AdapterError, match="unidade"):
        import_tms_bundle(str(d))


def test_unknown_file_is_ignored_with_diagnostic(synth_root, tmp_path):
    d = _copy(synth_root["paths"]["con"], tmp_path / "extra")
    (d / "leiame.txt").write_text("nota")
    e = import_tms_bundle(str(d))
    assert any("ignorado" in x for x in e.diagnostics)


def test_decimal_comma_supported(synth_root, tmp_path):
    d = _copy(synth_root["paths"]["ring"], tmp_path / "comma")
    for name in os.listdir(d):
        if name.endswith(".csv"):
            p = d / name
            p.write_text(p.read_text().replace(".", ","))
    e = import_tms_bundle(str(d))
    ref = import_tms_bundle(synth_root["paths"]["ring"])
    assert np.allclose(e.samples.z, ref.samples.z, equal_nan=True)


def test_pupil_absent_state(synth_root):
    e = import_tms_bundle(synth_root["paths"]["nopup"])
    assert e.pupil.state == "ausente" and e.pupil.center_x is None
    e2 = import_tms_bundle(synth_root["paths"]["pup"])
    assert e2.pupil.state == "registrada" and abs(e2.pupil.center_x - 0.3) < 1e-9 and abs(e2.pupil.center_y + 0.2) < 1e-9


def test_real_profile_marks_unverified_and_no_manifest(synth_root, tmp_path):
    d = _copy(synth_root["paths"]["con"], tmp_path / "real")
    (d / "manifest.json").unlink()
    e = import_tms_bundle(str(d))
    assert e.bundle.synthetic is False
    assert e.frame.transform_verified is False
    assert any("NÃO verificado" in w for w in e.warnings) and any("manifesto e XREF ausentes" in w for w in e.warnings)
