"""Ceratometria por zona — ensaios de método com referência independente da implementação (revisão da 0.4.0).

Peso por área (regra do ACP) contra a integral analítica; conferência SimK × XREF reprovando de fato; regra do SimK
só nos anéis 6–8; sinal de J0/J45; alerta de zona de ponta a ponta; limites de qualidade nas bordas (setor que cruza
0°, furos espalhados, marca † nas zonas, * de eixo pouco definido, anel ausente no meio); diâmetro dos anéis contra o
AVE gravado; assimetria/irregularidade e interpolação linear com valores construídos. Só dados sintéticos.
"""
import math
import os

import numpy as np
import pytest

from tmsab.adapters import import_tms_bundle
from tmsab.keratometry import (
    _valores_na_circunferencia,
    compute_keratometry,
    conferir_simk,
    diferenca_eixo,
    fourier_fit,
    linhas_resumo,
    simk_regra_tms,
)
from tmsab.model import KeratometryConfig, PointState, QualityConfig
from tmsab.pipeline import analyze_exam
from tmsab.plots import figure_resultado_completo
from tmsab.reporting import report_lines
from tmsab.synthetic import SyntheticSpec, build_surface, write_export_folder_v1
from tmsab.web import report_html


def _exame(tmp_path, nome, **kw):
    d = str(tmp_path / nome)
    base = dict(exam_id="M", R=7.8, Q=-0.2, cyl_D=1.5, axis_deg=30.0, eye="OD", decimals=4)
    base.update(kw)
    write_export_folder_v1([SyntheticSpec(**base)], d)
    return d, import_tms_bundle(d, exam_id="90001")


# ---------------------------------------------------------------------------
# Peso por área (regra do ACP) contra a integral da potência axial analítica
# ---------------------------------------------------------------------------
def _km_por_area_analitico(spec, raio, nr=600, na=256):
    surf = build_surface(spec)
    th = np.radians(np.arange(na) * 360.0 / na)
    rs = (np.arange(nr) + 0.5) * raio / nr
    R, T = np.meshgrid(rs, th, indexing="ij")
    x, y = (R * np.cos(T)).ravel(), (R * np.sin(T)).ravel()
    gx, gy = surf.grad(x, y)
    s = np.asarray(gx).ravel() * np.cos(T.ravel()) + np.asarray(gy).ravel() * np.sin(T.ravel())
    p = 337.5 * s / (R.ravel() * np.sqrt(1.0 + s**2))
    return float(np.sum(R.ravel() * p) / np.sum(R.ravel()))


def test_zona_0_6_mm_e_a_media_por_area_da_potencia_axial(tmp_path):
    # prolata forte: a potência axial cai ~1 D do centro a r = 3 mm, então pesar ou não por r muda o Km
    spec = SyntheticSpec(exam_id="A", R=7.7, Q=-0.6, cyl_D=1.0, axis_deg=20.0, eye="OD", decimals=4)
    d = str(tmp_path / "area")
    write_export_folder_v1([spec], d)
    k = compute_keratometry(import_tms_bundle(d, exam_id="90001"))
    z6 = next(z for z in k.discos if z.diametro_mm == 6.0)
    ref = _km_por_area_analitico(spec, 3.0)
    # a regra discreta do TMS (r·Δr por anel, sem a lacuna central) fica ~0,04 D abaixo da integral contínua nesta
    # córnea muito prolata; sem o peso r (média por anel) o erro passa de +0,2 D
    assert abs(z6.k_medio_D - ref) < 0.08, (z6.k_medio_D, ref)


def test_zona_0_d_continua_com_o_diametro_e_igual_a_media_por_area(tmp_path):
    # ceratometria_v2: a amostra da borda pesa só a parte da sua célula radial dentro da zona. Com a amostra inteira
    # (v1), um anel entrava na zona primeiro pelos meridianos curvos: nesta córnea (3 D) Km/K1/K2 davam degraus de
    # 0,3 D entre décimos vizinhos e se afastavam 0,27 D da média por área (0,15 D nos exames reais)
    spec = SyntheticSpec(exam_id="S", R=7.0, Q=-0.2, cyl_D=3.0, axis_deg=85.0, eye="OS", decimals=4)
    d = str(tmp_path / "continua")
    write_export_folder_v1([spec], d)
    ex = import_tms_bundle(d, exam_id="90001")
    ds = [k / 10.0 for k in range(15, 61)]
    zs = [compute_keratometry(ex, KeratometryConfig(diametros_mm=(), zonas_disco_mm=(x,), aneis_medidos=False,
                                                    zona_pupilar="nenhuma")).discos[0] for x in ds]
    assert all(z.validity == "disponivel" for z in zs)
    v = np.array([[z.k1_plano_D, z.k2_curvo_D, z.k_medio_D, z.delta_k_D] for z in zs])
    degrau = np.abs(v[1:-1] - 0.5 * (v[:-2] + v[2:])).max(axis=0)
    assert degrau.max() < 0.02, degrau
    erro = max(abs(z.k_medio_D - _km_por_area_analitico(spec, x / 2)) for z, x in zip(zs, ds))
    assert erro < 0.04, erro


# ---------------------------------------------------------------------------
# Conferência SimK × XREF e regra do SimK
# ---------------------------------------------------------------------------
_CALC = {"k2_curvo_D": 44.0, "eixo_k2_deg": 90.0, "k1_plano_D": 42.0, "eixo_k1_deg": 180.0, "delta_k_D": 2.0}
_XREF = {"SimK1": 44.0, "SimK1 Ang.": 90.0, "SimK2": 42.0, "SimK2 Ang.": 180.0, "CYL": 2.0}


@pytest.mark.parametrize("xref, confere", [
    (_XREF, True),
    ({**_XREF, "SimK1": 42.0, "SimK2": 44.0, "SimK1 Ang.": 180.0, "SimK2 Ang.": 90.0}, False),   # SimK1 ↔ SimK2
    ({**_XREF, "SimK1": 44.06}, False),                                                     # 0,06 D > 0,05 D
    ({**_XREF, "SimK1 Ang.": 96.0}, False),                                                 # 6° com CYL 2 D
    ({**_XREF, "SimK1 Ang.": 96.0, "CYL": 0.5}, True),                                      # eixo não conta com CYL < 0,75
])
def test_conferencia_simk_reprova_troca_potencia_e_eixo(xref, confere):
    txt = conferir_simk(_CALC, xref, QualityConfig())[0]
    assert ("NÃO confere" not in txt) == confere and (txt.endswith("— confere") == confere), txt


def test_xref_com_simk_trocado_aparece_na_folha_e_no_relatorio(tmp_path):
    d, _ = _exame(tmp_path, "troca", cyl_D=2.0)
    p = os.path.join(d, "XREF.dat")
    linhas = open(p, encoding="ascii", newline="").read().split("\r\n")
    cab = linhas[1].split(",")
    i1, i1a, i2, i2a = (cab.index(c) for c in ("SimK1", "SimK1 Ang.", "SimK2", "SimK2 Ang."))
    c = linhas[2].split(",")
    c[i1], c[i2], c[i1a], c[i2a] = c[i2], c[i1], c[i2a], c[i1a]
    linhas[2] = ",".join(c)
    open(p, "w", encoding="ascii", newline="").write("\r\n".join(linhas))
    ex = import_tms_bundle(d, exam_id="90001")
    an, _ = analyze_exam(ex, with_maps=False)
    assert "NÃO confere" in an.keratometry.conferencia[0]
    assert any("NÃO confere" in ln for ln in report_lines(an))
    fig = figure_resultado_completo(ex, an, patient_name=None)
    assert any("NÃO confere" in t.get_text() for t in fig.texts)


def test_simk_usa_so_os_aneis_6_a_8_e_mink_e_o_minimo():
    na, nr = 256, 25
    th = np.arange(na) * 360.0 / na
    fi = np.radians(th - 60.0)
    anel = 40.0 + 0.1 * np.arange(1, nr + 1)[:, None]            # anel k vale 40 + 0,1·k: 6–8 → 40,7
    dio = anel + 0.75 * np.cos(2 * fi)[None, :] + 0.2 * np.cos(4 * fi)[None, :]
    s = simk_regra_tms(dio, np.zeros((nr, na), dtype=np.int8), th)
    # tolerância de amostragem (meridianos a cada 1,4°); usar os anéis 5–7 ou 7–9 erraria 0,1 D
    assert abs(s["k2_curvo_D"] - (40.7 + 0.75 + 0.2)) < 0.005 and diferenca_eixo(s["eixo_k2_deg"], 60.0) < 0.8
    assert abs(s["k1_plano_D"] - (40.7 - 0.75 + 0.2)) < 0.005 and diferenca_eixo(s["eixo_k1_deg"], 150.0) < 0.8
    # o 4º harmônico tira o mínimo do meridiano a 90°: MinK é o mínimo de verdade, não o plano
    assert s["mink_D"] < s["k1_plano_D"] - 1e-3 and diferenca_eixo(s["eixo_mink_deg"], 150.0) > 5.0


# ---------------------------------------------------------------------------
# Sinal de J0/J45 (Thibos: positivo = a favor da regra) com o eixo do K2 conhecido
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("eixo_plano, j0, j45", [
    (0.0, +1.0, 0.0),        # K2 a 90°: a favor da regra → J0 > 0
    (90.0, -1.0, 0.0),       # K2 a 180°: contra a regra → J0 < 0
    (135.0, 0.0, -1.0),      # K2 a 45°  → J45 = −ΔK/2
    (45.0, 0.0, +1.0),       # K2 a 135° → J45 = +ΔK/2
])
def test_sinal_de_j0_e_j45(tmp_path, eixo_plano, j0, j45):
    _, ex = _exame(tmp_path, f"j{int(eixo_plano)}", Q=0.0, cyl_D=2.0, axis_deg=eixo_plano)
    z = next(z for z in compute_keratometry(ex).discos if z.diametro_mm == 3.0)
    assert abs(z.j0_D - j0) < 0.05 and abs(z.j45_D - j45) < 0.05, (z.j0_D, z.j45_D)


# ---------------------------------------------------------------------------
# Alerta de astigmatismo dependente da zona, de ponta a ponta
# ---------------------------------------------------------------------------
def test_alerta_de_zona_de_ponta_a_ponta(tmp_path):
    # astigmatismo que cresce com o raio (Zernike Z(2,2)): a zona 0–3 mm e a pupila de 6,5 mm divergem
    _, ex = _exame(tmp_path, "alerta", cyl_D=1.0, pupil_diameter=6.5, zernike_um={"2,2": 50.0})
    an, _ = analyze_exam(ex, with_maps=False)
    k = an.keratometry
    assert len(k.alertas) == 1 and k.alertas[0].startswith("astigmatismo dependente da zona")
    assert any(ln.startswith("Alerta (informativo") and k.alertas[0] in ln for ln in report_lines(an))
    assert "class='aviso'" in report_html(ex, an, with_maps=False)
    fig = figure_resultado_completo(ex, an, patient_name=None)
    assert any(t.get_text().startswith("Alerta (informativo)") for t in fig.texts)


@pytest.mark.parametrize("dk, alerta", [(0.49, False), (0.52, True)])
def test_limite_do_alerta_com_margem(dk, alerta):
    # mesmos ΔK, eixos a 30°: diferença vetorial = ΔK (abaixo de 0,75 D o eixo sozinho não dispara)
    from tmsab.keratometry import alerta_dependencia_zona
    from tmsab.model import KeratometryResult, KeratometryZone

    def zona(tipo, d, eixo):
        c, s = math.cos(math.radians(2 * eixo)), math.sin(math.radians(2 * eixo))
        return KeratometryZone(rotulo=f"{tipo} {d}", tipo=tipo, diametro_mm=d, validity="disponivel", delta_k_D=dk,
                               eixo_k2_deg=eixo, k_medio_D=44.0, j0_D=-(dk / 2) * c, j45_D=-(dk / 2) * s)
    res = KeratometryResult(config=KeratometryConfig(), discos=[zona("zona", 3.0, 90.0)], pupila=zona("pupila", 6.0, 120.0))
    assert (alerta_dependencia_zona(res, QualityConfig()) is not None) == alerta


# ---------------------------------------------------------------------------
# Limites de qualidade nas bordas
# ---------------------------------------------------------------------------
def _furar(ex, meridianos, raio=2.0):
    """Marca como AUSENTES (RAD = NaN, como no formato real) os dois anéis em volta de `raio` nos meridianos dados."""
    tab, r = ex.samples.fields["DIO"], ex.samples.r
    rn = np.nanmean(r, axis=1)
    j = int(np.searchsorted(rn, raio))
    for jj in (j - 1, j):
        tab.states[jj, meridianos] = PointState.MISSING
        tab.values[jj, meridianos] = np.nan
        r[jj, meridianos] = np.nan


def test_setor_sem_dados_que_cruza_0_grau_e_contado_inteiro(tmp_path):
    _, ex = _exame(tmp_path, "setor0")
    _furar(ex, list(range(250, 256)) + list(range(0, 11)))      # 17 meridianos = 23,9° em volta de 0°
    z = next(z for z in compute_keratometry(ex).zonas if z.diametro_mm == 4.0)
    assert z.validity == "indisponivel" and any("setor" in m for m in z.block_reasons), z.block_reasons
    assert z.maior_setor_sem_dados_deg > 20.0


def test_furos_espalhados_acima_de_15_por_cento_bloqueiam_pela_cobertura(tmp_path):
    _, ex = _exame(tmp_path, "furos")
    _furar(ex, list(range(0, 256, 6)))                         # 43 meridianos isolados (16,8%), nenhum setor > 20°
    z = next(z for z in compute_keratometry(ex).zonas if z.diametro_mm == 4.0)
    assert z.validity == "indisponivel" and z.maior_setor_sem_dados_deg < 5.0
    assert any(m.startswith("meridianos com dados") for m in z.block_reasons), z.block_reasons


def test_anel_ausente_no_meio_nao_e_atravessado_pela_interpolacao():
    nr, na = 10, 64
    r = np.tile((0.2 * np.arange(1, nr + 1))[:, None], (1, na))
    dio = 40.0 + 2.0 * r
    st = np.zeros((nr, na), dtype=np.int8)
    st[4, :] = PointState.MISSING                              # anel em r = 1,0 mm ausente
    vals, _, ok = _valores_na_circunferencia(dio, st, r, 0.95)
    assert not ok.any()                                        # entre 0,8 e 1,2 mm não se interpola por cima do furo


def test_interpolacao_linear_em_r_e_exata():
    nr, na = 10, 64
    r = np.tile((0.2 * np.arange(1, nr + 1))[:, None], (1, na))
    dio = 40.0 + 2.0 * r                                       # potência linear em r
    vals, _, ok = _valores_na_circunferencia(dio, np.zeros((nr, na), dtype=np.int8), r, 1.07)
    assert ok.all() and np.allclose(vals, 40.0 + 2.0 * 1.07, atol=1e-12)


def test_marca_de_interpolados_nas_zonas_e_eixo_pouco_definido(tmp_path):
    _, ex = _exame(tmp_path, "interp", lid_gap_from_ring=8, lid_gap_half_angle_deg=60.0)
    k = compute_keratometry(ex)
    z6 = next(z for z in k.discos if z.diametro_mm == 6.0)
    assert z6.validity == "disponivel" and z6.marca_interpolados and z6.fracao_interpolados >= 0.10
    assert next(l for l in linhas_resumo(k, incluir_aneis=True) if l["rotulo"].startswith("zona 0–6 mm"))["rotulo"].endswith("†")
    _, ex2 = _exame(tmp_path, "baixo", cyl_D=0.3)
    z3 = next(z for z in compute_keratometry(ex2).zonas if z.diametro_mm == 3.0)
    assert z3.validity == "disponivel" and z3.eixo_pouco_definido
    _, ex3 = _exame(tmp_path, "medio", cyl_D=0.8)
    assert not next(z for z in compute_keratometry(ex3).zonas if z.diametro_mm == 3.0).eixo_pouco_definido


# ---------------------------------------------------------------------------
# Definições contra fontes independentes
# ---------------------------------------------------------------------------
def test_diametro_dos_aneis_igual_ao_ave_gravado(tmp_path):
    # pálpebra interpolada: o AVE usa só os pontos MEDIDOS (como o TMS); incluir os interpolados mudaria o diâmetro
    d, ex = _exame(tmp_path, "ave", lid_gap_from_ring=14, lid_gap_half_angle_deg=40.0)
    with open(os.path.join(d, "AVE_1_90001.dat"), encoding="ascii") as fh:
        ave = [tuple(float(x) for x in ln.split(",")) for ln in fh.read().splitlines() if ln.strip()]
    k = compute_keratometry(ex)
    for j, z in enumerate(k.aneis):
        assert abs(z.diametro_mm - ave[j][1]) < 2e-4, (j, z.diametro_mm, ave[j][1])


def test_assimetria_e_irregularidade_com_harmonicos_construidos():
    th = np.radians(np.arange(256) * 360.0 / 256)
    p = 44.0 + 0.3 * np.cos(th - math.radians(40.0)) + 0.75 * np.cos(2 * (th - math.radians(60.0))) + 0.2 * np.cos(3 * th)
    f = fourier_fit(th, p)
    assert abs(f["h1"] - 0.3) < 1e-9 and abs(f["eixo_assimetria_deg"] - 40.0) < 1e-6
    assert abs(f["irregularidade"] - 0.2 / math.sqrt(2.0)) < 1e-9 and abs(2 * f["h2"] - 1.5) < 1e-9
    plano = fourier_fit(th, 44.0 + 0.75 * np.cos(2 * th))
    assert plano["eixo_assimetria_deg"] is None                # sem 1º harmônico mensurável, sem eixo
