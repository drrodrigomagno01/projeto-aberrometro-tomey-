"""Saídas (plano §8: 'Saídas' — JSON, CSV, PDF representam os mesmos valores) e estatística (§9)."""
import csv
import json
import os

import numpy as np

from tmsab.adapters import import_tms_bundle
from tmsab.pipeline import analyze_exam
from tmsab.reporting import export_all, report_lines
from tmsab.stats import LongTable, agreement_repeated, repeatability, write_results_csv


def test_exports_consistent(synth_root, tmp_path):
    exam = import_tms_bundle(synth_root["paths"]["coma"])
    an, _ = analyze_exam(exam, with_maps=True)
    paths = export_all(exam, an, str(tmp_path / "out"))
    assert set(paths) >= {"json", "csv", "pdf", "npz"}
    assert os.path.getsize(paths["pdf"]) > 10_000
    with open(paths["json"], encoding="utf-8") as fh:
        d = json.load(fh)
    assert d["schema"] == "tmsab.analysis/1" and d["analysis_id"] == an.analysis_id and d["synthetic"] is True
    w6 = next(w for w in d["wavefronts"] if w["profile"]["name"] == "anterior_v1_vertice_6mm")
    c31_json = next(c["value_um"] for c in w6["coefficients"] if c["n"] == 3 and c["m"] == 1)
    with open(paths["csv"], encoding="utf-8") as fh:
        rows = [r for r in csv.reader(fh, delimiter=";")]
    c31_csv = next(float(r[-1]) for r in rows[1:] if len(r) > 16 and r[4] == "anterior_v1_vertice_6mm" and r[13] == "3" and r[14] == "1")
    c31_obj = next(c.value_um for c in next(w for w in an.wavefronts if w.profile.name == "anterior_v1_vertice_6mm").coefficients if (c.n, c.m) == (3, 1))
    assert abs(c31_json - c31_obj) < 1e-12 and abs(c31_csv - c31_obj) < 1e-6
    lines = report_lines(an)
    assert any("SINTÉTICO" in ln for ln in lines) and any("Identificador da análise" in ln for ln in lines)


def test_blocked_profile_reported_as_absence(synth_root, tmp_path):
    exam = import_tms_bundle(synth_root["paths"]["nopup"])
    an, _ = analyze_exam(exam, with_maps=False)
    paths = export_all(exam, an, str(tmp_path / "out2"))
    with open(paths["csv"], encoding="utf-8") as fh:
        txt = fh.read()
    assert "BLOQUEADO" in txt
    lines = report_lines(an)
    assert any("BLOQUEIO" in ln for ln in lines) and any("ausência, não zero" in ln for ln in lines)


def _simulate_table(path, n=25, k=3, sw_a=0.02, sw_b=0.03, bias=0.05, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        true = rng.normal(0.25, 0.1)
        for c in range(1, k + 1):
            rows.append(("P%02d" % i, "OD", "TMS_calc", c, "sa_c40@6mm", true + rng.normal(0, sw_a)))
            rows.append(("P%02d" % i, "OD", "OPD_corneal", c, "sa_c40@6mm", true - bias + rng.normal(0, sw_b)))
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["participante", "olho", "metodo", "captura", "metrica", "valor"])
        w.writerows(rows)


def test_repeatability_and_agreement_recover_simulation(tmp_path):
    p = tmp_path / "tabela.csv"
    _simulate_table(str(p))
    tb = LongTable.read_csv(str(p))
    r = repeatability(tb.select("TMS_calc", "sa_c40@6mm"), "TMS_calc", "sa_c40@6mm")
    assert r.n_participantes == 25 and abs(r.sw - 0.02) < 0.008 and r.sw_ic95[0] < r.sw < r.sw_ic95[1]
    g = agreement_repeated(tb.select("TMS_calc", "sa_c40@6mm"), tb.select("OPD_corneal", "sa_c40@6mm"), "TMS_calc", "OPD_corneal", "sa_c40@6mm")
    assert abs(g.vies - 0.05) < 0.02 and g.vies_ic95[0] < g.vies < g.vies_ic95[1]
    assert g.loa[0] < g.vies < g.loa[1]
    assert abs(g.sd_dif_total - np.sqrt(0.02**2 + 0.03**2)) < 0.012
    out = write_results_csv(str(tmp_path / "res.csv"), [r], [g])
    assert os.path.getsize(out) > 100
