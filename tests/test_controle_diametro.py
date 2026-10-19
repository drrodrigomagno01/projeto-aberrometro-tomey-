"""Controle de diâmetro da ceratometria (aba Ceratometria do programa e versão HTML).

O usuário escolhe um diâmetro (passo de 0,1 mm) e vê a circunferência Ø d e a zona 0–d centradas no vértice.
O controle usa as MESMAS funções da tabela, então nos diâmetros da tabela os números são idênticos; entre eles, a
verdade independente é a potência axial analítica da superfície sintética (como em test_keratometria.py).
Só exportações sintéticas no formato real (v1): o repositório é público.
"""
import copy
import json
import math

import numpy as np
import pytest

from tmsab.adapters import import_tms_bundle
from tmsab.keratometry import (
    CONTROLE_INICIAL_MM,
    CONTROLE_MAX_SEM_RAD_MM,
    CONTROLE_MIN_MM,
    ceratometria_no_diametro,
    compute_keratometry,
    dados_do_controle,
    diametros_do_controle,
    diferenca_eixo,
    eixo_tabo,
    faixa_do_controle,
    fourier_fit,
    linha_da_zona,
    linha_do_controle,
)
from tmsab.model import KeratometryConfig, QualityConfig
from tmsab.synthetic import SyntheticSpec, build_surface, write_export_folder_v1


@pytest.fixture(scope="module")
def exames(tmp_path_factory):
    d = tmp_path_factory.mktemp("controle_v1")
    casos = [
        SyntheticSpec(exam_id="C", R=7.8, Q=-0.2, cyl_D=2.0, axis_deg=30.0, eye="OD", decimals=4,
                      pupil_diameter=5.0, pupil_center=(0.3, -0.2)),
        SyntheticSpec(exam_id="C", R=7.7, Q=-0.15, cyl_D=1.5, axis_deg=100.0, eye="OS", decimals=4),
        SyntheticSpec(exam_id="C", R=7.8, Q=-0.2, cyl_D=1.0, axis_deg=10.0, eye="OD", decimals=4,
                      lid_gap_from_ring=12, lid_gap_half_angle_deg=35.0, lid_gap_as_interpolated=False),
        SyntheticSpec(exam_id="C", R=7.8, Q=-0.2, eye="OD", decimals=4, pupil_registered=False),
    ]
    write_export_folder_v1(casos, str(d))
    return {"specs": casos, "exams": [import_tms_bundle(str(d), exam_id=str(90001 + i)) for i in range(len(casos))]}


def _axial_analitica(spec, d, n=256):
    """K1/K2/eixo do K2 da potência axial da superfície analítica na circunferência de diâmetro d."""
    surf = build_surface(spec)
    th = np.radians(np.arange(n) * 360.0 / n)
    r = d / 2.0
    gx, gy = surf.grad(r * np.cos(th), r * np.sin(th))
    s = np.asarray(gx).ravel() * np.cos(th) + np.asarray(gy).ravel() * np.sin(th)
    f = fourier_fit(th, 337.5 * s / (r * np.sqrt(1.0 + s**2)))
    return f["a0"] - f["h2"], f["a0"] + f["h2"], f["eixo_k2_deg"]


@pytest.mark.parametrize("i", [0, 1, 2, 3])
def test_nos_diametros_da_tabela_o_controle_mostra_a_mesma_linha(exames, i):
    ex = exames["exams"][i]
    k = compute_keratometry(ex)
    qc = QualityConfig()
    for z in k.zonas:                                        # circunferências Ø 1–8 mm (inclui 2,4 e 3,3)
        zc, _ = ceratometria_no_diametro(ex, z.diametro_mm, k.config, qc)
        assert linha_da_zona(zc) == linha_da_zona(z), z.rotulo
    for z in k.discos:                                       # zonas 0–3 e 0–6 mm
        _, zd = ceratometria_no_diametro(ex, z.diametro_mm, k.config, qc)
        assert linha_da_zona(zd, "zona") == linha_da_zona(z, "zona"), z.rotulo
    # a grade do HTML passa pelos mesmos décimos: 2.4 e 3.3 da grade são os floats da tabela
    dados = dados_do_controle(ex, k.config, qc)
    por_d = {p["d"]: p for p in dados["pontos"]}
    for z in k.zonas:
        assert por_d[z.diametro_mm]["circ"] == linha_do_controle(z), z.rotulo
    for z in k.discos:
        assert por_d[z.diametro_mm]["zona"] == linha_do_controle(z), z.rotulo


@pytest.mark.parametrize("d", [1.7, 2.7, 3.7, 4.5, 5.3, 6.6])
def test_entre_as_linhas_da_tabela_contra_a_potencia_axial_analitica(exames, d):
    spec, ex = exames["specs"][0], exames["exams"][0]
    zc, zd = ceratometria_no_diametro(ex, d)
    k1, k2, e2 = _axial_analitica(spec, d)
    assert zc.validity == "disponivel", zc.block_reasons
    assert abs(zc.k1_plano_D - k1) < 0.03 and abs(zc.k2_curvo_D - k2) < 0.03, (d, zc.k1_plano_D, k1, zc.k2_curvo_D, k2)
    assert diferenca_eixo(zc.eixo_k2_deg, e2) < 0.5
    assert diferenca_eixo(zc.eixo_k1_deg, spec.axis_deg) < 0.5          # o eixo do gerador é o meridiano mais plano
    # zona 0–d: a verdade por área está em test_keratometria_metodo (mesma função); aqui só a sanidade — na asférica o
    # ΔK axial muda com o raio (≈ 1,94 D na zona 0–6,6 mm para cilindro nominal de 2 D)
    assert zd.validity == "disponivel" and abs(zd.delta_k_D - spec.cyl_D) < 0.1
    assert zc.rotulo == f"Ø {d:g} mm" and zd.rotulo == f"zona 0–{d:g} mm"


def test_grade_em_decimos_exatos():
    g = diametros_do_controle(0.5, 9.0)
    assert len(g) == 86 and g[0] == 0.5 and g[-1] == 9.0
    assert all(d == k / 10.0 for d, k in zip(g, range(5, 91)))
    assert 2.4 in g and 3.3 in g and 3.0 in g                     # os mesmos floats dos literais da tabela
    assert diametros_do_controle(3.0, 3.0) == [3.0]


def test_faixa_vai_de_meio_milimetro_ao_ultimo_anel(exames):
    ex = exames["exams"][0]
    rn = np.nanmean(ex.samples.r, axis=1)
    dmin, dmax = faixa_do_controle(ex)
    assert dmin == CONTROLE_MIN_MM == 0.5
    assert dmax == math.ceil(20.0 * float(np.nanmax(rn)) - 1e-9) / 10.0
    assert dmax - 0.1 < 2.0 * float(np.nanmax(rn)) <= dmax + 1e-9
    # fora da cobertura dos anéis: indisponível com o motivo, nunca extrapolado
    zc, zd = ceratometria_no_diametro(ex, dmax + 0.5)
    assert zc.validity == "indisponivel" and "fora da cobertura" in "; ".join(zc.block_reasons)
    assert zd.validity == "indisponivel" and "além da cobertura" in "; ".join(zd.block_reasons)
    assert linha_da_zona(zc)["texto"] == ["—"] * 6
    # dentro da lacuna central do Plácido
    zc, zd = ceratometria_no_diametro(ex, 0.9 * 2.0 * float(rn[0]))
    assert zc.validity == "indisponivel" and "lacuna central" in zc.block_reasons[0]
    assert zd.validity == "indisponivel" and "lacuna central" in zd.block_reasons[0]


def test_palpebra_ausente_bloqueia_no_controle_como_na_tabela(exames):
    ex = exames["exams"][2]
    d12 = float(np.nanmean(ex.samples.r[12])) * 2.0
    perto = round(d12 + 0.6, 1)
    zc, _ = ceratometria_no_diametro(ex, perto)
    assert zc.validity == "indisponivel" and any("setor" in m for m in zc.block_reasons), zc.block_reasons
    zc, _ = ceratometria_no_diametro(ex, 2.5)
    assert zc.validity == "disponivel"


def test_json_do_html_sem_nan_e_com_eixos_tabo(exames):
    for ex in exames["exams"]:
        dados = dados_do_controle(ex)
        texto = json.dumps(dados, allow_nan=False, ensure_ascii=False)
        assert json.loads(texto) == dados
        assert dados["inicial"] == CONTROLE_INICIAL_MM == 3.0 and dados["passo"] == 0.1
        assert [p["d"] for p in dados["pontos"]] == diametros_do_controle(dados["dmin"], dados["dmax"])
        for p in dados["pontos"]:
            for ln in (p["circ"], p["zona"]):
                if ln["disponivel"]:
                    assert 0.0 < ln["eixo_k2"] <= 180.0 and ln["eixo_k1"] == eixo_tabo(ln["eixo_k2"] + 90.0)
                    assert ln["k1"] <= ln["k2"] and ln["texto"][4] == f"{ln['dk']:.2f}"
                else:
                    assert ln["texto"] == ["—"] * 6 and ln["motivo"]
                    assert ln["k1"] is None and ln["eixo_k2"] is None and not ln["pouco_definido"]
    # sem astigmatismo: o eixo vem marcado como pouco definido (o controle desenha o meridiano tracejado)
    p3 = next(p for p in dados_do_controle(exames["exams"][3])["pontos"] if p["d"] == 3.0)
    assert p3["circ"]["disponivel"] and p3["circ"]["pouco_definido"] and p3["circ"]["rotulo"].endswith("*")


def test_sem_dio_ou_sem_rad_fica_indisponivel_com_o_motivo(exames):
    sem_dio = copy.deepcopy(exames["exams"][0])
    del sem_dio.samples.fields["DIO"]
    zc, zd = ceratometria_no_diametro(sem_dio, 3.0)
    assert zc.validity == zd.validity == "indisponivel" and zc.block_reasons == ["tabela DIO ausente na exportação"]
    dados = dados_do_controle(sem_dio)
    assert dados["dmax"] == faixa_do_controle(exames["exams"][0])[1]            # a faixa vem do RAD
    assert all(not p["circ"]["disponivel"] and not p["zona"]["disponivel"] for p in dados["pontos"])
    sem_rad = copy.deepcopy(exames["exams"][0])
    sem_rad.samples.r = None
    assert faixa_do_controle(sem_rad) == (CONTROLE_MIN_MM, CONTROLE_MAX_SEM_RAD_MM)
    zc, _ = ceratometria_no_diametro(sem_rad, 3.0)
    assert zc.block_reasons == ["posições radiais (RAD) indisponíveis"]


def test_controle_nao_altera_o_exame_nem_a_configuracao(exames):
    ex = exames["exams"][0]
    antes = (ex.samples.fields["DIO"].values.copy(), ex.samples.r.copy())
    cfg = KeratometryConfig()
    for d in (0.5, 3.0, 4.4, 9.5):
        ceratometria_no_diametro(ex, d, cfg)
    dados_do_controle(ex, cfg)
    assert np.array_equal(antes[0], ex.samples.fields["DIO"].values, equal_nan=True)
    assert np.array_equal(antes[1], ex.samples.r, equal_nan=True)
    assert cfg == KeratometryConfig()


def test_zona_logo_acima_do_primeiro_anel_usa_todos_os_meridianos(exames):
    # no astigmatismo o raio do 1º anel muda com o meridiano: entre a mediana e o máximo dele a circunferência fica sem
    # dados em parte dos meridianos, mas a zona 0–d usa em TODOS a parte da célula do 1º anel que fica dentro da borda
    # (antes da ceratometria_v2 só os meridianos curvos entravam: Km +0,65 D acima do vizinho, com "cobertura 100%")
    spec, ex = exames["specs"][0], exames["exams"][0]
    r0 = ex.samples.r[0]
    d = float(np.nanmedian(r0) + np.nanmax(r0))
    zc, zd = ceratometria_no_diametro(ex, d)
    _, zd2 = ceratometria_no_diametro(ex, d + 0.1)
    assert zc.validity == "indisponivel" and zd.validity == zd2.validity == "disponivel", (zc.block_reasons, zd.block_reasons)
    assert zd.fracao_cobertura == 1.0 and zd.n_amostras == r0.size
    assert abs(zd.k_medio_D - zd2.k_medio_D) < 0.05 and abs(zd.delta_k_D - spec.cyl_D) < 0.1


def test_controle_usa_a_configuracao_e_os_limites_da_analise(exames):
    ex = exames["exams"][0]                                  # cilindro de 2 D
    cfg = KeratometryConfig(diametros_mm=(1.5, 2.5, 3.5), zonas_disco_mm=(2.5, 4.5))
    qc = QualityConfig(k_axis_min_cyl_D=3.0)                 # 2 D < 3 D: todo eixo vira "pouco definido" (*)
    k = compute_keratometry(ex, cfg, qc)
    por_d = {round(p["d"] * 10): p for p in dados_do_controle(ex, cfg, qc)["pontos"]}
    for z in k.zonas:
        zc, _ = ceratometria_no_diametro(ex, z.diametro_mm, cfg, qc)
        assert linha_da_zona(zc) == linha_da_zona(z) and zc.eixo_pouco_definido, z.rotulo
        assert por_d[round(z.diametro_mm * 10)]["circ"] == linha_do_controle(z), z.rotulo
    for z in k.discos:
        _, zd = ceratometria_no_diametro(ex, z.diametro_mm, cfg, qc)
        assert linha_da_zona(zd, "zona") == linha_da_zona(z, "zona"), z.rotulo
        assert por_d[round(z.diametro_mm * 10)]["zona"] == linha_do_controle(z), z.rotulo
    # com os limites padrão o mesmo diâmetro não leva a marca: o controle usou o qc recebido
    assert not ceratometria_no_diametro(ex, 2.5)[0].eixo_pouco_definido
