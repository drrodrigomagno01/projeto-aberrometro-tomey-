"""Ceratometria por zona — casos-limite apontados na revisão independente da 0.4.0.

Exame sem DIO, SimK com o meridiano plano incompleto, ponto ausente dentro da zona, contorno pupilar parcial, zona
dentro da lacuna central, anel inteiro ausente, perfil presumido (v0), eixo a um fio de 0°/180°, margem do PDF
técnico e impressão do HTML salvo. Só dados sintéticos.
"""
import os
import warnings

import numpy as np
import pytest

from tmsab.adapters import import_tms_bundle
from tmsab.keratometry import compute_keratometry, eixo_tabo, fmt_eixo
from tmsab.model import KeratometryConfig, PointState
from tmsab.pipeline import analyze_exam
from tmsab.reporting import _paginar, _text_page, _wrap, keratometry_records, report_lines
from tmsab.synthetic import SyntheticSpec, write_export_folder_v1


def _exame(tmp_path, nome, **kw):
    d = str(tmp_path / nome)
    base = dict(exam_id="R", R=7.8, Q=-0.2, cyl_D=1.5, axis_deg=30.0, eye="OD", decimals=4, pupil_diameter=5.0)
    base.update(kw)
    write_export_folder_v1([SyntheticSpec(**base)], d)
    return d, import_tms_bundle(d, exam_id="90001")


def test_eixo_a_um_fio_de_zero_vira_180_como_no_texto():
    assert eixo_tabo(0.0003) == 180.0 and eixo_tabo(179.9999) == 180.0 and eixo_tabo(-0.001) == 180.0
    assert eixo_tabo(0.01) == pytest.approx(0.01) and eixo_tabo(90.0003) == pytest.approx(90.0003)
    assert fmt_eixo(eixo_tabo(0.0003)) == "180°"


def test_sem_dio_o_simk_do_xref_continua_e_a_conferencia_diz_o_motivo(tmp_path):
    _, ex = _exame(tmp_path, "semdio")
    ex.samples.fields = {k: v for k, v in ex.samples.fields.items() if k != "DIO"}
    k = compute_keratometry(ex)
    assert k.simk_calculado is None and {"SimK1", "SimK2"} <= set(k.simk_tms)
    assert k.conferencia and "não conferido" in k.conferencia[0] and "DIO ausente" in k.conferencia[0]
    assert not any("NÃO confere" in c for c in k.conferencia)
    rec = {r["tipo"]: r for r in keratometry_records(k)}
    assert rec["simk_tms"]["validade"] == "disponivel" and rec["simk_tms"]["k2_curvo_D"] == k.simk_tms["SimK1"]
    assert rec["simk_calculado"]["validade"] == "indisponivel" and "DIO ausente" in rec["simk_calculado"]["motivo"]


def test_simk_com_meridiano_plano_incompleto_nao_acusa_erro_de_leitura(tmp_path):
    # pálpebra AUSENTE (zeros) desde o anel 5 em ±25° do meridiano de 90°, onde fica o meridiano plano
    d, ex = _exame(tmp_path, "simkgap", axis_deg=90.0, lid_gap_from_ring=5, lid_gap_half_angle_deg=25.0,
                   lid_gap_as_interpolated=False)
    k = compute_keratometry(ex)
    assert k.simk_calculado is None
    assert any(n.startswith("SimK recalculado indisponível") for n in k.notes)
    assert not any("NÃO confere" in c or "nan" in c for c in k.conferencia)
    with open(os.path.join(d, "XREF.dat"), encoding="ascii") as fh:
        assert "nan" not in fh.read().lower()                 # o gerador nunca grava "nan" no índice
    rec = {r["tipo"]: r for r in keratometry_records(k)}
    assert rec["simk_calculado"]["validade"] == "indisponivel" and rec["simk_calculado"]["motivo"]


def test_ponto_ausente_dentro_da_zona_conta_como_falta_de_dados(tmp_path):
    _, ex = _exame(tmp_path, "furo")
    tab = ex.samples.fields["DIO"]
    j, sl = 8, slice(0, 20)                                    # anel 9 (Ø ≈ 4 mm), 20 meridianos (28°)
    tab.states[j, sl] = PointState.MISSING
    tab.values[j, sl] = np.nan
    ex.samples.r[j, sl] = np.nan                               # como no formato real: RAD ausente = NaN
    k = compute_keratometry(ex)
    z6 = next(z for z in k.discos if z.diametro_mm == 6.0)
    z3 = next(z for z in k.discos if z.diametro_mm == 3.0)
    assert z6.validity == "indisponivel" and z6.fracao_cobertura < 1.0 and any("setor" in m for m in z6.block_reasons)
    assert z3.validity == "disponivel" and z3.fracao_cobertura == 1.0      # o furo está fora da zona de 3 mm


def test_contorno_pupilar_parcial_reduz_a_cobertura(tmp_path):
    _, ex = _exame(tmp_path, "pupparcial", pupil_center=(0.2, -0.1))
    ex.pupil.edge_r = np.asarray(ex.pupil.edge_r, dtype=float).copy()
    ex.pupil.edge_r[::6] = 0.0                                 # 43 de 256 meridianos sem borda detectada
    k = compute_keratometry(ex)
    assert k.pupila.fracao_cobertura < 0.85 and k.pupila.validity == "indisponivel"
    assert any("sem borda" in n for n in k.pupila.notes)


def test_zona_dentro_da_lacuna_central_fica_indisponivel_com_motivo(tmp_path):
    _, ex = _exame(tmp_path, "lacuna")
    z = compute_keratometry(ex, KeratometryConfig(zonas_disco_mm=(0.3,))).discos[0]
    assert z.validity == "indisponivel" and "lacuna central" in z.block_reasons[0]


def test_anel_inteiro_ausente_nao_emite_runtimewarning(tmp_path):
    _, ex = _exame(tmp_path, "anel")
    tab = ex.samples.fields["DIO"]
    tab.states[19, :] = PointState.MISSING
    tab.values[19, :] = np.nan
    ex.samples.r[19, :] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        k = compute_keratometry(ex)
    assert k.aneis[19].validity == "indisponivel"
    assert all(z.validity == "disponivel" for z in k.discos if z.diametro_mm <= 6.0)


def test_perfil_presumido_v0_avisa_que_o_dio_nao_foi_verificado(synth_root):
    k = compute_keratometry(import_tms_bundle(synth_root["paths"]["tor"]))
    assert any("presumido" in n and "tms4_datatables_v1" in n for n in k.notes)


def test_perfil_real_v1_nao_leva_o_aviso_de_perfil_presumido(tmp_path):
    _, ex = _exame(tmp_path, "v1")
    assert not any("presumido" in n for n in compute_keratometry(ex).notes)


def test_pdf_tecnico_texto_dentro_da_margem(tmp_path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    _, ex = _exame(tmp_path, "pdf")
    an, _ = analyze_exam(ex, with_maps=False)
    maior = 0.0
    for pagina in _paginar(_wrap(report_lines(an))):
        fig = _text_page(pagina, "t")
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        rend = canvas.get_renderer()
        for t in fig.texts:
            maior = max(maior, t.get_window_extent(rend).x1 / fig.bbox.width)
    assert maior <= 0.95, maior


def test_html_salvo_tambem_abre_os_aneis_na_impressao():
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(raiz, "web", "template.html"), encoding="utf-8") as fh:
        html = fh.read()
    assert html.count("details.zern, details.kerato-aneis") == 2     # impressão no app e no arquivo salvo
