"""Escala de cores da topografia no estilo do TMS: a Standard do aparelho (fixa, 40,5–53 D, 0,5 D) ou ajustada ao exame
no passo escolhido nos botões (0,125 · 0,25 · 0,5 · 0,75 · 1,00 D) — mapa, legenda, folha A4 e site.

Só exportações sintéticas no formato real (v1): o repositório é público.
"""
import json
import re

import numpy as np
import pytest
from matplotlib.figure import Figure

from tmsab.adapters import import_tms_bundle
from tmsab.pipeline import analyze_exam, profiles_for_exam
from tmsab.plots import figure_resultado_completo
from tmsab.synthetic import SyntheticSpec, write_export_folder_v1
from tmsab.topo import (
    FAIXAS_AJUSTADA_MAX,
    FAIXAS_AJUSTADA_MIN,
    PASSOS_ESCALA_D,
    TMS_CMAP,
    TMS_STANDARD_LEVELS,
    ViewOptions,
    _valores_do_mapa,
    draw_tms_legend,
    draw_tms_map,
    escala_ajustada,
    escala_do_mapa,
    figure_tms_view,
    fmt_passo,
    norma_da_escala,
    passo_valido,
    titulo_da_escala,
)


@pytest.fixture(scope="module")
def exames(tmp_path_factory):
    d = tmp_path_factory.mktemp("escala")
    write_export_folder_v1([
        # tórica de 2 D: potências de ~40,6 a 44,2 D
        SyntheticSpec(exam_id="E", R=7.8, Q=-0.2, cyl_D=2.0, axis_deg=30.0, eye="OD", decimals=4, pupil_center=(0.3, -0.2)),
        # curva e irregular, como no ceratocone: ~36 a 60 D (mais de 13 D — com 0,5 D passaria de 26 faixas)
        SyntheticSpec(exam_id="E", R=6.6, Q=-0.9, cyl_D=6.0, axis_deg=100.0, eye="OS", decimals=4, zernike_um={"3,-1": 12.0}),
    ], str(d))
    return {"dir": str(d), "exams": [import_tms_bundle(str(d), exam_id=str(90001 + i)) for i in range(2)]}


def _percentis(ex, kind="axial"):
    v = _valores_do_mapa(ex, ViewOptions(kind=kind))
    return np.nanpercentile(v[np.isfinite(v)], [1, 99])


def test_passos_dos_botoes():
    assert PASSOS_ESCALA_D == (0.125, 0.25, 0.5, 0.75, 1.0)
    assert [fmt_passo(p) for p in PASSOS_ESCALA_D] == ["0,125", "0,25", "0,5", "0,75", "1,00"]
    assert passo_valido("0.25") == 0.25 and passo_valido(1) == 1.0
    assert passo_valido(0.3) == passo_valido(None) == passo_valido("x") == passo_valido(-1) == 0.5
    assert titulo_da_escala("tms", 0.25) == "Standard do TMS"
    assert titulo_da_escala("auto", 0.125) == "ajustada ao exame, passo 0,125 D"
    assert titulo_da_escala("auto", 1.0, cortada=True) == "ajustada ao exame, passo 1,00 D (extremos nas cores das pontas)"


@pytest.mark.parametrize("passo", PASSOS_ESCALA_D)
def test_escala_ajustada_em_multiplos_exatos_do_passo(exames, passo):
    for ex in exames["exams"]:
        for kind in ("axial", "tangencial", "refrativa"):
            levels, cortada = escala_do_mapa(ex, ViewOptions(kind=kind, scale="auto", step=passo))
            n = len(levels) - 1
            assert np.allclose(np.diff(levels), passo, atol=1e-9)
            assert np.allclose(levels / passo, np.round(levels / passo), atol=1e-9)        # 43,125 / 0,125 = 345
            assert FAIXAS_AJUSTADA_MIN <= n <= FAIXAS_AJUSTADA_MAX
            p1, p99 = _percentis(ex, kind)
            if cortada:                                      # janela de 26 faixas centrada na mediana do exame
                assert n == FAIXAS_AJUSTADA_MAX and (p1 < levels[0] or p99 > levels[-1])
                v = _valores_do_mapa(ex, ViewOptions(kind=kind))
                assert abs(0.5 * (levels[0] + levels[-1]) - np.nanmedian(v)) <= passo
            else:                                            # cobre do percentil 1 ao 99
                assert levels[0] <= p1 + 1e-9 and levels[-1] >= p99 - 1e-9


def test_minimo_8_faixas_e_corte_no_passo_fino(exames):
    ex = exames["exams"][0]                                  # 3,6 D de potências
    lv, cortada = escala_do_mapa(ex, ViewOptions(scale="auto", step=1.0))
    assert len(lv) - 1 == 8 and not cortada                  # 4 faixas bastariam: o mínimo é 8, centradas
    p1, p99 = _percentis(ex)
    assert lv[0] <= p1 and lv[-1] >= p99 and abs((lv[0] + lv[-1]) / 2 - (p1 + p99) / 2) <= 1.0
    lv, cortada = escala_do_mapa(ex, ViewOptions(scale="auto", step=0.125))
    assert len(lv) - 1 == 26 and cortada                     # 3,25 D de janela para 3,6 D de exame
    lv, cortada = escala_do_mapa(ex, ViewOptions(scale="auto", step=0.25))
    assert 8 < len(lv) - 1 < 26 and not cortada
    assert escala_ajustada(np.full(10, np.nan), 0.25)[0].size == 9          # sem valores: 8 faixas, sem erro


def test_standard_e_fixa_em_0_5_d_qualquer_que_seja_o_passo(exames):
    for ex in exames["exams"]:
        for passo in PASSOS_ESCALA_D:
            lv, cortada = escala_do_mapa(ex, ViewOptions(scale="tms", step=passo))
            assert lv is TMS_STANDARD_LEVELS and not cortada
    assert escala_do_mapa(exames["exams"][0], ViewOptions(kind="elevacao", scale="auto", step=0.25)) == (None, False)


@pytest.mark.parametrize("passo", PASSOS_ESCALA_D)
def test_mapa_de_muita_variacao_nao_quebra_e_usa_a_mesma_escala(exames, passo):
    # antes: a escala ajustada pedia mais faixas que as 26 cores e o matplotlib recusava (ValueError)
    ex = exames["exams"][1]
    for kind in ("axial", "tangencial", "refrativa"):
        o = ViewOptions(kind=kind, scale="auto", step=passo)
        fig = Figure(figsize=(5, 5)); ax = fig.add_subplot(111)
        pm, levels, _ = draw_tms_map(ax, ex, None, o)
        esperado, cortada = escala_do_mapa(ex, o)
        assert pm is not None and np.array_equal(levels, esperado) and np.array_equal(pm.norm.boundaries, esperado)
        p1, p99 = _percentis(ex, kind)
        necessarias = int(np.ceil(p99 / passo - 1e-9) - np.floor(p1 / passo + 1e-9))
        assert cortada == (necessarias > 26) and (len(levels) - 1 == 26 if cortada else levels[0] <= p1 <= p99 <= levels[-1])
    assert escala_do_mapa(ex, ViewOptions(scale="auto", step=0.5))[1]          # 24 D de exame: 48 faixas de 0,5 D
    fig = figure_tms_view(ex, None, ViewOptions(scale="auto", step=0.5))
    textos = [t.get_text() for a in fig.axes for t in a.texts]
    assert any("passo 0,5 D (extremos nas cores das pontas)" in t for t in textos), textos[-3:]


@pytest.mark.parametrize("passo", [1.0, 0.25, 0.125])
def test_legenda_tem_as_cores_do_mapa(exames, passo):
    ex = exames["exams"][0]
    o = ViewOptions(scale="auto", step=passo)
    fig = Figure(figsize=(6.2, 7.3))
    ax = fig.add_axes([0.02, 0.17, 0.96, 0.75])
    pm, levels, _ = draw_tms_map(ax, ex, None, o)
    axl = fig.add_axes([0.05, 0.035, 0.9, 0.09])
    draw_tms_legend(axl, levels, o.scale, "D", fontsize=6)
    caixas = [p.get_facecolor() for p in axl.patches]
    assert len(caixas) == len(levels) - 1
    for i, cor in enumerate(caixas):              # a cor da caixa i é a que o mapa usa para um valor dentro da faixa i
        meio = 0.5 * (levels[i] + levels[i + 1])
        assert np.allclose(cor, pm.cmap(pm.norm(meio)), atol=1e-6), (i, cor, pm.cmap(pm.norm(meio)))
    assert np.allclose(caixas[0], TMS_CMAP(0)) and np.allclose(caixas[-1], TMS_CMAP(TMS_CMAP.N - 1))   # paleta toda
    rotulos = [t.get_text() for t in axl.texts if re.fullmatch(r"\d+\.\d+", t.get_text())]
    rotulados = levels[:-1][:: (len(levels) - 1 + len(rotulos) - 1) // len(rotulos)]
    assert rotulos and [float(r) for r in rotulos] == pytest.approx(list(rotulados), abs=1e-9)   # limite inferior exato
    casas = max(1, max(len(f"{v:.3f}".rstrip("0").split(".")[1]) for v in rotulados))      # as casas que os rótulos pedem
    assert all(len(r.split(".")[1]) == casas for r in rotulos) and casas <= {1.0: 1, 0.25: 2, 0.125: 3}[passo]
    if passo == 0.125:                            # 26 rótulos não cabem lado a lado: faixas alternadas
        assert len(rotulos) < len(levels) - 1


def test_legenda_standard_inalterada(exames):
    fig = Figure(figsize=(6.2, 7.3))
    axl = fig.add_axes([0.05, 0.035, 0.9, 0.09])
    draw_tms_legend(axl, TMS_STANDARD_LEVELS, "tms", "D", fontsize=6)
    rotulos = [t.get_text() for t in axl.texts if re.fullmatch(r"\d+\.\d+", t.get_text())]
    assert rotulos == [f"{v:.1f}" for v in TMS_STANDARD_LEVELS[:-1]]
    assert [tuple(p.get_facecolor()) for p in axl.patches] == [TMS_CMAP(i) for i in range(26)]
    assert any(t.get_text() == "escala Standard do TMS (D)" for t in axl.texts)
    assert norma_da_escala(TMS_STANDARD_LEVELS).boundaries is TMS_STANDARD_LEVELS


@pytest.fixture(scope="module")
def analisado(exames):
    ex = exames["exams"][0]
    an, _ = analyze_exam(ex, profiles_for_exam(ex, (6.0,), ("vertice",)), with_maps=True)
    return ex, an


@pytest.mark.parametrize("passo", [0.25, 1.0])
def test_folha_a4_usa_o_passo_escolhido(analisado, passo):
    ex, an = analisado
    fig = figure_resultado_completo(ex, an, view=ViewOptions(scale="auto", step=passo))
    textos = [t.get_text() for t in fig.texts] + [t.get_text() for a in fig.axes for t in a.texts]
    assert any(f"escala ajustada ao exame, passo {fmt_passo(passo)} D" in t for t in textos)
    fig = figure_resultado_completo(ex, an, view=ViewOptions(scale="tms", step=passo))
    textos = [t.get_text() for t in fig.texts] + [t.get_text() for a in fig.axes for t in a.texts]
    assert any("escala Standard do TMS (40,5–53,0 D)" in t for t in textos) and not any("passo" in t for t in textos)


def test_site_recebe_o_passo_e_informa_a_escala(exames):
    from tmsab.web import topo_png
    d = exames["dir"]
    r = topo_png(d, "90001", {"kind": "axial", "scale": "auto", "step": 0.25})
    json.dumps(r, allow_nan=False)
    e = r["escala"]
    assert e["tipo"] == "auto" and e["passo"] == 0.25 and 8 <= e["faixas"] <= 26 and not e["cortada"]
    assert e["titulo"] == "ajustada ao exame, passo 0,25 D" and e["ate"] - e["de"] == pytest.approx(0.25 * e["faixas"])
    assert topo_png(d, "90001", {"scale": "auto", "step": 0.3})["escala"]["passo"] == 0.5            # passo fora dos botões
    s = topo_png(d, "90001", {"scale": "tms", "step": 0.25})["escala"]
    assert (s["tipo"], s["passo"], s["de"], s["ate"], s["faixas"]) == ("tms", 0.5, 40.5, 53.5, 26)
    assert topo_png(d, "90001", {"kind": "elevacao", "scale": "auto", "step": 0.25})["escala"] is None
    c = topo_png(d, "90002", {"scale": "auto", "step": 0.125})["escala"]
    assert c["cortada"] and c["faixas"] == 26 and "extremos nas cores das pontas" in c["titulo"]


# ---------------------------------------------------------------------------
# Revisão independente: legenda e cabeçalho da folha A4, mapas pequenos, zona, botão do site
# ---------------------------------------------------------------------------
def _caixas_dos_rotulos(fig, axl, dpi):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig.set_dpi(dpi)
    renderer = FigureCanvasAgg(fig).get_renderer()
    textos = [t for t in axl.texts if re.fullmatch(r"\d+\.\d+", t.get_text())]
    return sorted((t.get_window_extent(renderer) for t in textos), key=lambda b: b.x0)


@pytest.mark.parametrize("inicio", [44.625, 44.5, 42.125])
def test_rotulos_da_legenda_da_folha_a4_nao_se_encostam(inicio):
    # a geometria da legenda da folha A4 (0,36 da largura de 8,27 in, fonte 4,4): 26 faixas de 0,125 D começando num
    # oitavo ímpar pedem rótulos de 3 casas — antes, a largura estimada deixava 0,3 pt entre eles (colados no PNG)
    levels = np.round(inicio + np.arange(27) * 0.125, 6)
    fig = Figure(figsize=(8.27, 11.69))
    axl = fig.add_axes([0.05, 0.5, 0.36, 0.02])
    draw_tms_legend(axl, levels, "auto", "D", fontsize=4.4, cortada=True)
    for dpi in (88, 96, 100, 144, 150, 176, 192, 200, 240):   # telas comuns, Retina (176), PNG (150/200), impressão
        caixas = _caixas_dos_rotulos(fig, axl, dpi)
        assert len(caixas) >= 2
        folga_pt = min((b.x0 - a.x1) * 72.0 / dpi for a, b in zip(caixas, caixas[1:]))
        assert folga_pt >= 1.0, (dpi, round(folga_pt, 2))   # a estimativa antiga deixava −0,4 a 0,3 pt


@pytest.fixture(scope="module")
def ceratocone(exames):
    ex = exames["exams"][1]
    an, _ = analyze_exam(ex, profiles_for_exam(ex, (6.0,), ("vertice",)), with_maps=True)
    return ex, an


def test_folha_a4_mapas_pequenos_usam_o_mesmo_passo(analisado):
    ex, an = analisado
    for passo in (0.25, 0.75):
        fig = figure_resultado_completo(ex, an, view=ViewOptions(scale="auto", step=passo))
        normas = list({id(c.norm): c.norm for a in fig.axes for c in a.collections       # a barra de cor repete a norma
                       if isinstance(getattr(c, "norm", None), type(norma_da_escala(TMS_STANDARD_LEVELS)))}.values())
        assert len(normas) == 3                              # axial (grande), tangencial e refrativa (2×2)
        assert all(np.allclose(np.diff(n.boundaries), passo) for n in normas)
        esperado = {k: escala_do_mapa(ex, ViewOptions(kind=k, scale="auto", step=passo))[0] for k in ("axial", "tangencial", "refrativa")}
        assert all(any(np.array_equal(n.boundaries, e) for n in normas) for e in esperado.values())


def test_cabecalho_da_folha_a4_cabe_na_margem_com_escala_cortada_e_zona(ceratocone):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    ex, an = ceratocone
    vo = ViewOptions(scale="auto", step=0.125, zone_mm=7.0)
    assert escala_do_mapa(ex, vo)[1]                         # o pior caso: janela cortada e zona escolhida
    fig = figure_resultado_completo(ex, an, view=vo)
    cab = next(t for t in fig.texts if t.get_text().startswith("TOPOGRAFIA"))
    assert "passo 0,125 D" in cab.get_text() and "extremos" not in cab.get_text()     # o aviso fica na legenda
    leg = [t.get_text() for a in fig.axes for t in a.texts if t.get_text().startswith("escala ajustada")]
    assert leg and "extremos nas cores das pontas" in leg[0]
    for dpi in (88, 100, 150):
        fig.set_dpi(dpi)
        bb = cab.get_window_extent(FigureCanvasAgg(fig).get_renderer())
        assert bb.x1 <= 0.965 * fig.bbox.width + 1, (dpi, bb.x1 / fig.bbox.width)


@pytest.mark.parametrize("kind", ["axial", "tangencial", "refrativa"])
def test_zona_com_escala_ajustada_usa_so_a_zona(exames, kind):
    from tmsab.web import topo_png
    ex = exames["exams"][0]
    o = ViewOptions(kind=kind, scale="auto", step=0.25, zone_mm=3.0)
    v = _valores_do_mapa(ex, o)
    assert np.all(ex.samples.r[np.isfinite(v)] <= 1.5 + 1e-9)          # só a zona entra
    toda = _valores_do_mapa(ex, ViewOptions(kind=kind))
    assert np.isfinite(v).sum() < np.isfinite(toda).sum()
    levels, cortada = escala_do_mapa(ex, o)
    p1, p99 = np.nanpercentile(v[np.isfinite(v)], [1, 99])
    assert not cortada and levels[0] <= p1 and levels[-1] >= p99
    assert levels[-1] - levels[0] < np.ptp(escala_do_mapa(ex, ViewOptions(kind=kind, scale="auto", step=0.25))[0])
    fig = Figure(figsize=(5, 5)); ax = fig.add_subplot(111)
    assert np.array_equal(draw_tms_map(ax, ex, None, o)[1], levels)
    e = topo_png(exames["dir"], "90001", {"kind": kind, "escala": "auto:0.25", "zone_mm": 3})["escala"]
    assert (e["de"], e["ate"], e["faixas"]) == (float(levels[0]), float(levels[-1]), len(levels) - 1)


def test_site_le_a_escolha_do_botao_da_escala(exames):
    # o painel manda o estado inteiro; a escolha do botão ("tms" ou "auto:<passo>") vale mesmo se scale/step faltarem
    from tmsab.web import _view_from, topo_png
    for escolha, esperado in [("tms", ("tms", 0.5)), ("auto:0.125", ("auto", 0.125)), ("auto:0.25", ("auto", 0.25)),
                              ("auto:0.5", ("auto", 0.5)), ("auto:0.75", ("auto", 0.75)), ("auto:1", ("auto", 1.0))]:
        vo = _view_from({"escala": escolha, "scale": "tms", "step": 0.5})            # scale/step antigos não mandam
        assert (vo.scale, vo.step) == esperado, escolha
    assert (lambda v: (v.scale, v.step))(_view_from({"scale": "auto", "step": 0.75})) == ("auto", 0.75)   # sem o botão
    assert (lambda v: (v.scale, v.step))(_view_from({"escala": "auto:0.3"})) == ("auto", 0.5)             # passo inválido
    assert topo_png(exames["dir"], "90001", {"escala": "auto:0.125"})["escala"]["passo"] == 0.125
