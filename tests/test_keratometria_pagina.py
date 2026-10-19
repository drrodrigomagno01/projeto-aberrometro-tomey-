"""Página A4 "Resultado completo" (tmsab/plots.py): bloco CERATOMETRIA POR ZONA e layout da folha.

A mesma figura é a aba do programa, a 1ª página do PDF e a imagem do HTML (recortada com bbox_inches='tight').
Só exportações sintéticas no formato real (v1): o repositório é público.
"""
import types

import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from tmsab.adapters import import_tms_bundle
from tmsab.keratometry import COLUNAS_TABELA, linhas_resumo
from tmsab.pipeline import analyze_exam, default_profiles
from tmsab.plots import KERATO_GID, _momento_local, figure_resultado_completo, plot_keratometry_table
from tmsab.synthetic import SyntheticSpec, write_export_folder_v1

A4 = (8.27, 11.69)


@pytest.fixture(scope="module")
def casos(tmp_path_factory):
    d = tmp_path_factory.mktemp("pagina_v1")
    write_export_folder_v1([
        SyntheticSpec(exam_id="P", R=7.8, Q=-0.2, cyl_D=2.0, axis_deg=30.0, eye="OD", decimals=4,
                      pupil_diameter=5.0, pupil_center=(0.3, -0.2)),
        SyntheticSpec(exam_id="P", R=7.8, Q=-0.2, eye="OS", decimals=4, pupil_registered=False),
        SyntheticSpec(exam_id="P", R=7.8, Q=-0.2, cyl_D=1.0, axis_deg=10.0, eye="OD", decimals=4,
                      lid_gap_from_ring=12, lid_gap_half_angle_deg=35.0, lid_gap_as_interpolated=False),
    ], str(d))
    ex = [import_tms_bundle(str(d), exam_id=str(90001 + i)) for i in range(3)]
    return {
        "pupila": (ex[0], analyze_exam(ex[0], with_maps=True)[0]),
        "sem_pupila": (ex[1], analyze_exam(ex[1], with_maps=True)[0]),
        "seis_perfis": (ex[2], analyze_exam(ex[2], profiles=default_profiles(diameters=(3.0, 4.0, 6.0)), with_maps=True)[0]),
    }


def _desenhar(fig, dpi=None):
    if dpi:
        fig.set_dpi(dpi)
    FigureCanvasAgg(fig)
    fig.canvas.draw()
    return fig.canvas.get_renderer()


def _eixo_kerato(fig):
    return next(a for a in fig.axes if a.get_gid() == KERATO_GID)


def _celulas(ax):
    """{(linha, coluna): texto} da tabela do eixo (quebras de linha viram espaço)."""
    return {k: c.get_text().get_text().replace("\n", " ") for k, c in ax.tables[0].get_celld().items()}


def _intersecta(a, b, tol=0.5):
    return min(a.x1, b.x1) - max(a.x0, b.x0) > tol and min(a.y1, b.y1) - max(a.y0, b.y0) > tol


def _caixas(fig, r):
    """(nome, caixa em pixels, é do bloco de ceratometria) de todos os eixos (com rótulos e títulos) e textos."""
    out = []
    for ax in fig.axes:
        bloco = ax.get_gid() == KERATO_GID
        bb = ax.tables[0].get_window_extent(r) if bloco and ax.tables else ax.get_tightbbox(r)
        if bb is not None and bb.width > 0:
            out.append((ax.get_title() or ax.get_gid() or "eixo", bb, bloco))
        if bloco:
            out += [(t.get_text(), t.get_window_extent(r), True) for t in ax.texts if t.get_text()]
    out += [(t.get_text(), t.get_window_extent(r), t.get_gid() == KERATO_GID) for t in fig.texts if t.get_text()]
    return out


def test_bloco_de_ceratometria_bate_com_linhas_resumo(casos):
    ex, an = casos["pupila"]
    fig = figure_resultado_completo(ex, an)
    assert any(t.get_text().startswith("CERATOMETRIA POR ZONA") for t in fig.texts)
    ax = _eixo_kerato(fig)
    cel = _celulas(ax)
    linhas = linhas_resumo(an.keratometry)
    assert [cel[(0, c)] for c in range(len(COLUNAS_TABELA))] == list(COLUNAS_TABELA)
    assert max(r for r, _ in cel) == len(linhas)
    for i, lin in enumerate(linhas, start=1):
        assert [cel[(i, c)] for c in range(7)] == [lin["rotulo"]] + list(lin["texto"]), lin["rotulo"]
    assert [l["rotulo"] for l in linhas[:8]] == [f"Ø {d} mm" for d in range(1, 9)]
    # pupila destacada (fundo e negrito) e SimK do TMS em cinza itálico
    tb = ax.tables[0].get_celld()
    ip = next(i for i, l in enumerate(linhas, start=1) if l["destaque"] == "pupila")
    assert tb[(ip, 0)].get_facecolor()[:3] == pytest.approx((1.0, 0xF4 / 255, 0xD6 / 255))
    assert all(tb[(ip, c)].get_text().get_fontweight() == "bold" for c in range(7))
    iref = next(i for i, l in enumerate(linhas, start=1) if l["destaque"] == "referencia")
    assert cel[(iref, 0)].startswith("SimK do TMS") and tb[(iref, 2)].get_text().get_fontstyle() == "italic"
    assert tb[(iref, 2)].get_text().get_color() == "#666"
    # convenção: K1 = plano (menor), K2 = curvo; no XREF o SimK1 é o curvo
    simk = an.keratometry.simk_tms
    assert cel[(iref, 1)] == f"{simk['SimK2']:.2f}" and cel[(iref, 3)] == f"{simk['SimK1']:.2f}"


def test_bloco_nao_sobrepoe_nada_e_pagina_fica_entre_3_e_97_por_cento(casos):
    for chave in ("pupila", "sem_pupila", "seis_perfis"):
        ex, an = casos[chave]
        fig = figure_resultado_completo(ex, an)
        for dpi in (88, 100, 150):                 # tela do programa, PDF/padrão, imagem do HTML
            r = _desenhar(fig, dpi)
            caixas = _caixas(fig, r)
            bloco = [c for c in caixas if c[2]]
            outros = [c for c in caixas if not c[2]]
            assert len(bloco) >= 3
            for nb, bb, _ in bloco:
                for no, bo, _ in outros:
                    assert not _intersecta(bb, bo), (chave, dpi, nb[:40], no[:40])
            # textos soltos não se sobrepõem entre si (cabeçalho, geometria, notas, avisos, rodapé)
            textos = [(t.get_text(), t.get_window_extent(r)) for t in fig.texts if t.get_text()]
            for i in range(len(textos)):
                for j in range(i + 1, len(textos)):
                    assert not _intersecta(textos[i][1], textos[j][1]), (chave, dpi, textos[i][0][:40], textos[j][0][:40])
            tb = fig.get_tightbbox(r)              # polegadas
            assert tb.x0 >= 0.03 * A4[0] and tb.x1 <= 0.97 * A4[0], (chave, dpi, tb)
            assert tb.y0 >= 0.03 * A4[1] and tb.y1 <= 0.97 * A4[1], (chave, dpi, tb)


def test_sem_pupila_mostra_linha_nao_registrada(casos):
    ex, an = casos["sem_pupila"]
    fig = figure_resultado_completo(ex, an)
    cel = _celulas(_eixo_kerato(fig))
    i = next(r for (r, c), t in cel.items() if c == 0 and t == "pupila — não registrada")
    assert [cel[(i, c)] for c in range(1, 7)] == ["—"] * 6
    assert _eixo_kerato(fig).tables[0].get_celld()[(i, 3)].get_facecolor()[:3] == pytest.approx((1.0, 0xF4 / 255, 0xD6 / 255))


def test_seis_perfis_tabela_de_perfis_nao_invade_motivos_nem_nota(casos):
    ex, an = casos["seis_perfis"]
    assert len(an.wavefronts) == 6 and any(w.validity != "disponivel" for w in an.wavefronts)
    fig = figure_resultado_completo(ex, an)
    r = _desenhar(fig)
    axp = next(a for a in fig.axes if a.tables and a.get_gid() != KERATO_GID)
    assert max(rr for rr, _ in axp.tables[0].get_celld()) == 6
    bt = axp.tables[0].get_window_extent(r)
    abaixo = [t for t in fig.texts if t.get_text().startswith(("* ", "Cilindro refrativo"))]
    assert len(abaixo) >= 3
    for t in abaixo:
        bb = t.get_window_extent(r)
        assert bb.y1 <= bt.y0 + 0.5, t.get_text()[:50]
    # motivo das zonas sem valor aparece no bloco de ceratometria
    assert any(t.get_text().startswith("— sem valor:") for t in fig.texts if t.get_gid() == KERATO_GID)


def test_cabecalho_sem_simk1_com_e_sem_nome(casos):
    ex, an = casos["pupila"]
    for nome in (None, "NOME DE TESTE"):
        fig = figure_resultado_completo(ex, an, patient_name=nome)
        cab = " | ".join(t.get_text() for t in fig.texts if t.get_position()[1] > 0.9)
        assert "SimK1" not in cab and "SimK2" not in cab
        assert "SimK curvo" in cab and "/ plano" in cab and "(TMS)" in cab and "Es " in cab
        assert "impresso" not in cab and "tmsab" not in cab          # versão e data de impressão foram para o rodapé
        rod = [t for t in fig.texts if t.get_text().startswith("tmsab ")]
        assert rod and rod[0].get_position()[1] < 0.1 and "calculado em" in rod[0].get_text()
        if nome:
            assert "Paciente: NOME DE TESTE" in cab and "Data do exame: " in cab
    # a página diz "meridiano mais curvo" no mesmo eixo do "SimK curvo" do cabeçalho
    txt = " | ".join(t.get_text() for t in fig.texts)
    eixo = int(round(ex.bundle.export_settings["tms_indices"]["SimK1 Ang."])) % 180 or 180
    assert f"SimK curvo {ex.bundle.export_settings['tms_indices']['SimK1']:.2f} D @ {eixo}°" in txt
    assert f"meridiano mais curvo {eixo}°" in txt


def test_json_antigo_sem_ceratometria(casos):
    ex, an = casos["pupila"]
    k, an.keratometry = an.keratometry, None
    try:
        fig = figure_resultado_completo(ex, an)
    finally:
        an.keratometry = k
    ax = _eixo_kerato(fig)
    assert not ax.tables and any("ceratometria não calculada nesta versão" in t.get_text() for t in ax.texts)


def test_textos_longos_quebram_sem_cortar_e_numeros_formatados(casos):
    ex, an = casos["pupila"]
    aviso = "aviso de teste com um texto bem comprido " * 8 + "FIMDOAVISO"
    d0 = an.pupil.diameter
    an.warnings.append(aviso)
    an.pupil.diameter = 6.005941916410005
    try:
        fig = figure_resultado_completo(ex, an)
        r = _desenhar(fig, 150)
    finally:
        an.warnings.remove(aviso)
        an.pupil.diameter = d0
    textos = [t.get_text() for t in fig.texts]
    linhas_aviso = [t.get_text() for t in fig.texts if t.get_color() == "#8a6d00"]
    assert len(linhas_aviso) >= 2 and ("AVISO: " + aviso) in " ".join(linhas_aviso)   # quebrado em linhas, não cortado
    assert any("diâmetro 6.01 mm" in t for t in textos) and not any("6.0059" in t for t in textos)
    for t in fig.texts:
        assert t.get_window_extent(r).x1 <= 0.97 * fig.bbox.width, t.get_text()[:50]
    titulos = [a.get_title() for a in fig.axes]
    assert any(t.startswith("W total (µm) — 6 mm") for t in titulos)


def test_plot_keratometry_table_fica_dentro_do_retangulo_e_encolhe_a_fonte(casos, monkeypatch):
    ex, an = casos["pupila"]
    rect = [0.30, 0.40, 0.45, 0.24]
    fig = Figure(figsize=A4)
    ax = plot_keratometry_table(fig, rect, an.keratometry)
    r = _desenhar(fig)
    W, H = fig.bbox.width, fig.bbox.height
    caixa = (rect[0] * W - 1, rect[1] * H - 1, (rect[0] + rect[2]) * W + 1, (rect[1] + rect[3]) * H + 1)
    for _, bb, _ in _caixas(fig, r):
        assert bb.x0 >= caixa[0] and bb.y0 >= caixa[1] and bb.x1 <= caixa[2] and bb.y1 <= caixa[3]
    assert ax.tables[0].get_celld()[(1, 1)].get_text().get_fontsize() == pytest.approx(6.2)
    # mais linhas (ex.: núcleo com mais zonas no resumo): continua dentro do retângulo, com fonte menor (≥ 5 pt)
    base = linhas_resumo(an.keratometry)
    muitas = [dict(base[0], rotulo=f"Ø {d / 2:g} mm") for d in range(2, 16)]
    monkeypatch.setattr("tmsab.keratometry.linhas_resumo", lambda kres, incluir_aneis=False: muitas)
    fig2 = Figure(figsize=A4)
    ax2 = plot_keratometry_table(fig2, rect, types.SimpleNamespace(notes=[], conferencia=[]))
    r2 = _desenhar(fig2)
    for _, bb, _ in _caixas(fig2, r2):
        assert bb.y0 >= caixa[1] and bb.y1 <= caixa[3]
    fs = ax2.tables[0].get_celld()[(1, 1)].get_text().get_fontsize()
    assert 5.0 <= fs < 6.2 and max(rr for rr, _ in ax2.tables[0].get_celld()) == 14


def test_hora_de_impressao_em_brasilia_ou_utc_menos_3_explicito(monkeypatch):
    try:
        from zoneinfo import ZoneInfo
        ZoneInfo("America/Sao_Paulo")
        tem_tz = True
    except Exception:  # noqa: BLE001
        tem_tz = False
    s = _momento_local("2026-10-19T01:02:03+00:00")
    assert s == ("18/10/2026 22:02 (Brasília)" if tem_tz else "18/10/2026 22:02 (UTC−3)")

    def sem_base(_nome):
        raise KeyError("sem tzdata")

    monkeypatch.setattr("zoneinfo.ZoneInfo", sem_base)                  # como no navegador (Pyodide sem tzdata)
    assert _momento_local("2026-10-19T01:02:03+00:00") == "18/10/2026 22:02 (UTC−3)"
    assert _momento_local("2026-10-19T01:02:03") == "18/10/2026 22:02 (UTC−3)"
    assert _momento_local(None) == "não informado"


def test_pagina_dentro_de_97_por_cento_com_escala_de_elevacao_grande(tmp_path):
    # cilindro alto: elevação vs. esfera passa de ±100 µm e a barra de cor ganha rótulos de 4 caracteres
    d = str(tmp_path / "elev")
    write_export_folder_v1([SyntheticSpec(exam_id="E", R=7.8, Q=-0.2, cyl_D=8.0, axis_deg=20.0, eye="OD", decimals=4)], d)
    ex = import_tms_bundle(d, exam_id="90001")
    an, _ = analyze_exam(ex, with_maps=True)
    fig = figure_resultado_completo(ex, an)
    for dpi in (100, 150):
        r = _desenhar(fig, dpi)
        tb = fig.get_tightbbox(r)
        assert tb.x0 >= 0.03 * A4[0] and tb.x1 <= 0.97 * A4[0], (dpi, tb)


def test_motivos_resumidos_mostram_a_pupila_sem_valor(tmp_path):
    # pálpebra AUSENTE desde o anel 12: periferia e pupila de 6 mm ficam sem valor, cada uma com cobertura diferente
    d = str(tmp_path / "palp")
    write_export_folder_v1([SyntheticSpec(exam_id="M", R=7.8, Q=-0.2, cyl_D=1.0, axis_deg=10.0, eye="OD", decimals=4,
                                          pupil_diameter=6.0, lid_gap_from_ring=12, lid_gap_half_angle_deg=40.0,
                                          lid_gap_as_interpolated=False)], d)
    ex = import_tms_bundle(d, exam_id="90001")
    an, _ = analyze_exam(ex, with_maps=False)
    assert an.keratometry.pupila.validity == "indisponivel"
    fig = figure_resultado_completo(ex, an)
    mot = " ".join(t.get_text() for t in fig.texts if t.get_gid() == KERATO_GID and t.get_color() == "#8B0000")
    assert mot.startswith("— sem valor: pupila") and "cobertura insuficiente" in mot and "…" not in mot
