"""Controle de diâmetro da aba Ceratometria do programa (PySide6, sem tela).

Barra, caixa ou clique no mapa escolhem Ø d; a leitura, a circunferência e os meridianos K1/K2 sobre o mapa axial e a
linha nos gráficos usam ceratometria_no_diametro com a configuração da própria análise — no Ø 3 mm são exatamente as
linhas da tabela. Sem números antigos (desatualizado = controle vazio e desabilitado).
Só exportações sintéticas no formato real (v1): o repositório é público.
"""
import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_API", "pyside6")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6")

from matplotlib.backend_bases import MouseButton, MouseEvent  # noqa: E402
from PySide6.QtGui import QTextDocument  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from tmsab.keratometry import (  # noqa: E402
    CONTROLE_LEGENDA,
    ceratometria_no_diametro,
    diferenca_eixo,
    faixa_do_controle,
    fmt_eixo,
    linha_da_zona,
)
from tmsab.pipeline import default_profiles  # noqa: E402
from tmsab.synthetic import SyntheticSpec, write_export_folder_v1  # noqa: E402

PERFIS = default_profiles([4.0], ("vertice",), 6)       # a ceratometria não depende dos perfis ópticos; um basta
# aviso interno do backend Qt do matplotlib com o Qt 6 recente (não é do programa)
pytestmark = pytest.mark.filterwarnings("ignore:Enum value 'Qt..ApplicationAttribute.AA_UseHighDpiPixmaps':DeprecationWarning")


def _aba(win, nome: str) -> int:
    """Índice da aba pelo nome (a ordem das abas muda entre versões)."""
    return next(i for i in range(win.tabs.count()) if win.tabs.tabText(i) == nome)


@pytest.fixture(scope="module")
def janela(tmp_path_factory):
    app = QApplication.instance() or QApplication([])
    from tmsab.gui import main_window
    pasta = tmp_path_factory.mktemp("controle_gui")
    casos = [
        # A: astigmatismo de 2 D, meridiano plano (eixo do gerador) a 30°; anéis até r = 5,2 mm (faixa até 10,5 mm)
        SyntheticSpec(exam_id="G", R=7.8, Q=-0.2, cyl_D=2.0, axis_deg=30.0, eye="OD", decimals=4,
                      pupil_diameter=5.0, pupil_center=(0.3, -0.2)),
        # B: anéis só até r = 4,4 mm (faixa até 8,9 mm)
        SyntheticSpec(exam_id="G", R=7.7, Q=-0.15, cyl_D=1.5, axis_deg=100.0, eye="OS", decimals=4, r_last=4.4),
        # C: sem a tabela DIO na exportação (arquivo removido abaixo)
        SyntheticSpec(exam_id="G", R=7.8, Q=-0.2, cyl_D=1.0, axis_deg=10.0, eye="OD", decimals=4),
    ]
    write_export_folder_v1(casos, str(pasta))
    os.remove(os.path.join(str(pasta), "DIO_1_90003.dat"))
    win = main_window.MainWindow(interactive=False)
    win.auto_export.setChecked(False); win.auto_compute.setChecked(False); win.with_name.setChecked(False)
    chaves = [win.load_package(str(pasta), exam_id=str(90001 + i)) for i in range(len(casos))]
    win.compute_sync(chaves, PERFIS)
    win.tabs.setCurrentIndex(_aba(win, "Ceratometria"))
    win.resize(1366, 768)
    win.show()
    app.processEvents()
    yield {"win": win, "app": app, "chaves": chaves, "mod": main_window}
    win.close()


def _seleciona(j, i: int):
    win = j["win"]
    win.list.setCurrentRow(i)
    j["app"].processEvents()
    st = win.packages[j["chaves"][i]]
    assert win.selected_state() is st and st.analysis is not None, st.message
    return st


def _leitura(win) -> list[str]:
    doc = QTextDocument()
    doc.setHtml(win.k_leitura.text())
    return [ln.replace("\xa0", " ") for ln in doc.toPlainText().splitlines()]


def _esperada(z, descricao: str) -> str:
    ln = linha_da_zona(z)
    if not ln["disponivel"]:
        return f"{ln['rotulo']}: {'inválida' if ln['validity'] == 'invalida' else 'indisponível'} — {ln['motivo']}"
    t = ln["texto"]
    return f"{ln['rotulo']} ({descricao}): K1 plano {t[0]} D @ {t[1]} · K2 curvo {t[2]} D @ {t[3]} · ΔK {t[4]} D · Km {t[5]} D"


def _no_controle(st, d: float):
    a = st.analysis
    return ceratometria_no_diametro(st.exam, d, a.keratometry.config, a.quality_config)


def _evento(win, nome: str, xy_dados, botao=MouseButton.LEFT, apertados=()):
    """Evento do matplotlib no canvas do mapa, na posição (x, y) em mm do mapa; no movimento, `apertados` são os botões
    seguros (como o backend Qt informa)."""
    cv = win.cv_kmapa.canvas
    x, y = win._k_ax.transData.transform(xy_dados)
    if nome == "motion_notify_event":
        ev = MouseEvent(nome, cv, x, y)
        ev.buttons = frozenset(apertados)          # depois do construtor: o argumento `buttons` só existe no matplotlib ≥ 3.10
    else:
        ev = MouseEvent(nome, cv, x, y, button=botao)
    cv.callbacks.process(nome, ev)


def _clique(win, xy_dados, botao=MouseButton.LEFT):
    win.cv_kmapa.canvas.draw()                     # posições finais dos eixos (leiaute)
    _evento(win, "button_press_event", xy_dados, botao)
    _evento(win, "button_release_event", xy_dados, botao)


def test_abre_em_3_mm_com_as_linhas_da_tabela(janela):
    win = janela["win"]
    st = _seleciona(janela, 0)
    assert [win.k_subtabs.tabText(i) for i in range(win.k_subtabs.count())] == ["Mapa axial — diâmetro",
                                                                                 "Gráficos por diâmetro"]
    assert win.k_subtabs.currentIndex() == 0                      # o mapa é a sub-aba padrão
    assert win.k_slider.isEnabled() and win.k_spin.isEnabled()
    assert win.k_slider.value() == 30 and win.k_spin.value() == 3.0 and win.k_spin.text() == "3.0 mm"
    k = st.analysis.keratometry
    z3 = next(z for z in k.zonas if z.diametro_mm == 3.0)
    d3 = next(z for z in k.discos if z.diametro_mm == 3.0)
    linhas = _leitura(win)
    assert linhas[0] == _esperada(z3, "circunferência no vértice")
    assert linhas[1] == _esperada(d3, "média pela área")
    # ... e são os textos das linhas "Ø 3 mm" e "zona 0–3 mm" da tabela na tela
    tela = {win.k_table.item(r, 0).text(): [win.k_table.item(r, c).text() for c in range(1, 7)]
            for r in range(win.k_table.rowCount()) if win.k_table.item(r, 1) is not None}
    assert tela[linha_da_zona(z3)["rotulo"]] == linha_da_zona(z3)["texto"]
    assert tela[linha_da_zona(d3)["rotulo"]] == linha_da_zona(d3)["texto"]
    assert win.k_leitura.toolTip() == CONTROLE_LEGENDA
    assert win.k_dica.text() == "Clique ou arraste no mapa para escolher o diâmetro."
    assert "Ø 3.0 mm" in win._k_ax.get_title()


def test_faixa_do_controle_em_decimos(janela):
    win = janela["win"]
    for i in (0, 1):
        st = _seleciona(janela, i)
        dmin, dmax = faixa_do_controle(st.exam)
        assert (win.k_slider.minimum(), win.k_slider.maximum()) == (round(dmin * 10), round(dmax * 10))
        assert (win.k_spin.minimum(), win.k_spin.maximum()) == (dmin, dmax)
    assert faixa_do_controle(win.packages[janela["chaves"][1]].exam)[1] < faixa_do_controle(win.packages[janela["chaves"][0]].exam)[1]


def test_barra_e_caixa_sincronizadas_sem_laco(janela, monkeypatch):
    win = janela["win"]
    st = _seleciona(janela, 0)
    chamadas = []
    original = win._k_atualizar
    monkeypatch.setattr(win, "_k_atualizar", lambda: (chamadas.append(win.k_slider.value()), original())[1])
    win.k_slider.setValue(45)
    assert win.k_spin.value() == 4.5 and chamadas == [45]
    assert _leitura(win)[0] == _esperada(_no_controle(st, 4.5)[0], "circunferência no vértice")
    win.k_spin.setValue(2.4)
    assert win.k_slider.value() == 24 and chamadas == [45, 24]
    zc, zd = _no_controle(st, 2.4)
    assert _leitura(win)[:2] == [_esperada(zc, "circunferência no vértice"), _esperada(zd, "média pela área")]
    # a linha "Ø 2.4 mm" da tabela (mesmo float 2.4 da grade em décimos)
    z24 = next(z for z in st.analysis.keratometry.zonas if z.diametro_mm == 2.4)
    assert _leitura(win)[0] == _esperada(z24, "circunferência no vértice")
    # digitado com vírgula (teclado ABNT): vira ponto
    win.k_spin.lineEdit().selectAll()
    QTest.keyClicks(win.k_spin, "5,5")
    assert win.k_spin.value() == 5.5 and win.k_slider.value() == 55 and win.k_spin.text() == "5.5 mm"


def test_clique_e_arrasto_no_mapa_escolhem_o_diametro(janela):
    win = janela["win"]
    _seleciona(janela, 0)
    win.k_spin.setValue(3.0)
    _clique(win, (1.2, 1.6))                         # r = 2,0 mm -> Ø 4,0 mm
    assert win.k_slider.value() == 40 and win.k_spin.value() == 4.0
    assert _leitura(win)[0].startswith("Ø 4 mm (circunferência no vértice): K1 plano ")
    # arrastar: segue o ponteiro enquanto o botão está apertado
    win.cv_kmapa.canvas.draw()
    esquerdo = [MouseButton.LEFT]
    _evento(win, "button_press_event", (1.2, 1.6))
    _evento(win, "motion_notify_event", (0.0, 2.5), apertados=esquerdo)
    assert win.k_slider.value() == 50
    _evento(win, "motion_notify_event", (-1.5, 0.0), apertados=esquerdo)
    assert win.k_slider.value() == 30
    _evento(win, "button_release_event", (-1.5, 0.0))
    _evento(win, "motion_notify_event", (3.0, 0.0), apertados=esquerdo)    # depois de soltar: nada muda
    assert win.k_slider.value() == 30
    # soltou fora da janela (sem o evento de soltar): o primeiro movimento sem botão encerra o arrasto
    _evento(win, "button_press_event", (1.2, 1.6))
    assert win.k_slider.value() == 40
    _evento(win, "motion_notify_event", (0.0, 2.5))
    _evento(win, "motion_notify_event", (0.0, 3.0), apertados=esquerdo)
    assert win.k_slider.value() == 40 and not win._k_arrastando
    win.k_spin.setValue(3.0)
    # botão direito e clique fora do mapa (na barra de cores) não escolhem diâmetro
    _clique(win, (1.2, 1.6), botao=MouseButton.RIGHT)
    assert win.k_slider.value() == 30
    barra = next(a for a in win.cv_kmapa.figure.axes if a is not win._k_ax)
    cv = win.cv_kmapa.canvas
    x, y = barra.transAxes.transform((0.5, 0.5))
    cv.callbacks.process("button_press_event", MouseEvent("button_press_event", cv, x, y, button=MouseButton.LEFT))
    cv.callbacks.process("button_release_event", MouseEvent("button_release_event", cv, x, y, button=MouseButton.LEFT))
    assert win.k_slider.value() == 30
    # longe do vértice: limitado à faixa do controle
    _clique(win, (5.3, 0.0))
    assert win.k_slider.value() == win.k_slider.maximum()


def test_circunferencia_e_meridianos_sobre_o_mapa(janela):
    win = janela["win"]
    st = _seleciona(janela, 0)
    win.k_spin.setValue(4.0)
    zc, _ = _no_controle(st, 4.0)
    assert zc.validity == "disponivel"
    c = win._k_circulo
    assert c.get_radius() == pytest.approx(2.0) and tuple(c.center) == (0.0, 0.0) and c.get_linestyle() == "solid"
    for chave, eixo in (("k2", zc.eixo_k2_deg), ("k1", zc.eixo_k1_deg)):
        linha = win._k_merid[chave]
        (x0, x1), (y0, y1) = linha.get_data()
        assert linha.get_visible() and win._k_rot[chave].get_visible()
        assert diferenca_eixo(math.degrees(math.atan2(y1 - y0, x1 - x0)), eixo) < 0.01
        assert (x0, y0) == pytest.approx((-x1, -y1)) and math.hypot(x1, y1) == pytest.approx(1.12 * 2.0)
        assert win._k_rot[chave].get_text() == f"{chave.upper()} {fmt_eixo(eixo)}"
    # o K1 é o meridiano plano do gerador (30°) e o K2, o curvo (120°), ortogonais
    assert diferenca_eixo(zc.eixo_k1_deg, 30.0) < 0.5 and diferenca_eixo(zc.eixo_k2_deg, 120.0) < 0.5
    # cores da convenção (K1 plano azul, K2 curvo laranja; as do site) nas linhas, nos rótulos e na leitura
    mw = janela["mod"]
    assert (mw.K_COR_K1, mw.K_COR_K2) == ("#2a78d6", "#eb6834")
    for chave, cor in (("k1", mw.K_COR_K1), ("k2", mw.K_COR_K2)):
        assert win._k_merid[chave].get_color() == cor and win._k_rot[chave].get_color() == cor
    leitura = win.k_leitura.text()
    assert f'color:{mw.K_TEXTO_K1};">K1&nbsp;plano' in leitura and f'color:{mw.K_TEXTO_K2};">K2&nbsp;curvo' in leitura
    # no referencial do mapa (x à direita do examinador, y para cima), a direção desenhada do K2 é a mais curva do DIO
    s = st.exam.samples
    dio, r, th = s.fields["DIO"].values, np.asarray(s.r), np.asarray(s.theta_deg)

    def potencia(direcao_deg):
        j = int(np.argmin(np.abs((th - direcao_deg + 180.0) % 360.0 - 180.0)))
        i = int(np.nanargmin(np.abs(r[:, j] - 2.0)))
        return float(dio[i, j])

    direcao = {ch: math.degrees(math.atan2(win._k_merid[ch].get_data()[1][1], win._k_merid[ch].get_data()[0][1]))
               for ch in ("k1", "k2")}
    assert potencia(direcao["k2"]) - potencia(direcao["k1"]) > 1.5               # cilindro de 2 D
    assert win._k_vertice.get_visible() and "Ø 4.0 mm" in win._k_ax.get_title()


def test_diametro_sem_cobertura_fica_indisponivel_e_esconde_os_meridianos(janela):
    win = janela["win"]
    st = _seleciona(janela, 0)
    for dec in (win.k_slider.minimum(), win.k_slider.maximum()):      # lacuna central do Plácido / além dos anéis
        win.k_slider.setValue(dec)
        zc, zd = _no_controle(st, dec / 10.0)
        assert zc.validity != "disponivel"
        linhas = _leitura(win)
        assert linhas[0] == _esperada(zc, "") and linhas[0].startswith(f"Ø {dec / 10.0:g} mm: indisponível — ")
        assert linhas[1] == _esperada(zd, "média pela área")
        assert "0.00" not in linhas[0] and "—" in linhas[0]                       # ausência com motivo, nunca zero
        assert not any(win._k_merid[ch].get_visible() or win._k_rot[ch].get_visible() for ch in ("k1", "k2"))
        assert win._k_circulo.get_linestyle() != "solid" and win._k_circulo.get_radius() == pytest.approx(dec / 20.0)
    win.k_spin.setValue(3.0)                                          # de volta à cobertura: meridianos de novo
    assert all(win._k_merid[ch].get_visible() for ch in ("k1", "k2")) and win._k_circulo.get_linestyle() == "solid"


def test_zoom_ou_arrasto_da_barra_do_mapa_ignoram_o_clique(janela):
    win = janela["win"]
    _seleciona(janela, 0)
    win.k_spin.setValue(3.0)
    tb = win.cv_kmapa.toolbar
    for modo in (tb.zoom, tb.pan):
        modo()                                     # liga o modo da barra
        assert tb.mode
        try:
            _clique(win, (1.2, 1.6))
            assert win.k_slider.value() == 30
        finally:
            modo()                                 # desliga
        assert not tb.mode
    _clique(win, (1.2, 1.6))
    assert win.k_slider.value() == 40


def test_diametro_mantido_ao_trocar_de_exame_e_limitado_a_faixa(janela):
    win = janela["win"]
    _seleciona(janela, 0)
    win.k_spin.setValue(4.5)
    stb = _seleciona(janela, 1)
    assert win.k_slider.value() == 45
    assert _leitura(win)[0] == _esperada(_no_controle(stb, 4.5)[0], "circunferência no vértice")
    assert {chave[0] for chave in win._k_cache} == {stb.analysis.analysis_id}      # o cache só guarda a análise atual
    # 10 mm cabe em A (até 10,5) mas não em B (até 8,9): B mostra 8,9; de volta a A, os 10 mm escolhidos
    _seleciona(janela, 0)
    win.k_spin.setValue(10.0)
    _seleciona(janela, 1)
    assert win.k_slider.value() == 89 and _leitura(win)[0].startswith("Ø 8.9 mm")
    _seleciona(janela, 0)
    assert win.k_slider.value() == 100 and _leitura(win)[0].startswith("Ø 10 mm")
    win.k_spin.setValue(3.0)


def test_linha_do_diametro_escolhido_nos_graficos(janela):
    win = janela["win"]
    _seleciona(janela, 0)
    win.k_spin.setValue(5.2)
    assert len(win._k_marcas) == 2
    for m in win._k_marcas:
        assert list(m.get_xdata()) == [5.2, 5.2] and m.get_linestyle() == "-."
    legenda = win.cv_kerato.figure.legends[0]
    assert "diâmetro escolhido" in [t.get_text() for t in legenda.get_texts()]
    win.k_subtabs.setCurrentIndex(1)               # mostrar os gráficos redesenha com a linha no lugar
    janela["app"].processEvents()
    win.k_subtabs.setCurrentIndex(0)
    win.k_spin.setValue(3.0)
    assert all(list(m.get_xdata()) == [3.0, 3.0] for m in win._k_marcas)


def test_sem_tabela_dio_mostra_o_motivo(janela):
    win = janela["win"]
    st = _seleciona(janela, 2)
    assert "DIO" not in st.exam.samples.fields
    assert win.k_slider.isEnabled() and win._k_ax is None and win._k_marcas == []
    d = win.k_slider.value() / 10.0
    assert _leitura(win)[:2] == [f"Ø {d:g} mm: indisponível — tabela DIO ausente na exportação",
                                 f"zona 0–{d:g} mm: indisponível — tabela DIO ausente na exportação"]
    textos = [t.get_text() for a in win.cv_kmapa.figure.axes for t in a.texts]
    assert any("DIO ausente" in t for t in textos)
    _evento_sem_eixos = MouseEvent("button_press_event", win.cv_kmapa.canvas, 50, 50, button=MouseButton.LEFT)
    win.cv_kmapa.canvas.callbacks.process("button_press_event", _evento_sem_eixos)     # sem mapa: ignorado
    assert win.k_slider.value() / 10.0 == d


def test_erros_viram_mensagem_sem_derrubar_a_janela(janela, monkeypatch):
    win = janela["win"]
    st = _seleciona(janela, 0)
    mod = janela["mod"]

    def falha(*_a, **_k):
        raise RuntimeError("falha simulada")

    monkeypatch.setattr(mod, "ceratometria_no_diametro", falha)
    win._k_cache.clear()
    win.k_spin.setValue(3.7)
    assert "erro no cálculo — RuntimeError: falha simulada" in _leitura(win)[0]
    assert not any(win._k_merid[ch].get_visible() for ch in ("k1", "k2"))
    monkeypatch.setattr(mod, "plot_topo_map", falha)
    win.fill_keratometry_tab(st)
    assert win._k_ax is None
    assert any("erro ao desenhar o mapa" in t.get_text() for a in win.cv_kmapa.figure.axes for t in a.texts)
    monkeypatch.undo()
    win._k_cache.clear()
    win.fill_keratometry_tab(st)
    assert win._k_ax is not None and _leitura(win)[0] == _esperada(_no_controle(st, 3.7)[0], "circunferência no vértice")
    win.k_spin.setValue(3.0)


def test_linha_invalida_aparece_como_invalida(janela, monkeypatch):
    # ajuste de Fourier sem amostras (inalcançável com os 256 meridianos do TMS-4N, mas é estado da tabela): "inválida"
    # com o motivo, como na tabela, e o mapa sem meridianos
    win = janela["win"]
    st = _seleciona(janela, 0)
    mod = janela["mod"]
    real = mod.ceratometria_no_diametro

    def invalida(exam, d, cfg=None, qc=None):
        zc, zd = real(exam, d, cfg, qc)
        zc = mod.KeratometryZone(rotulo=zc.rotulo, tipo=zc.tipo, diametro_mm=zc.diametro_mm, validity="invalida",
                                 block_reasons=["ajuste de Fourier sem amostras suficientes"])
        return zc, zd

    monkeypatch.setattr(mod, "ceratometria_no_diametro", invalida)
    win._k_cache.clear()
    win.k_spin.setValue(3.6)
    zc, zd = invalida(st.exam, 3.6, st.analysis.keratometry.config, st.analysis.quality_config)
    assert _leitura(win)[:2] == ["Ø 3.6 mm: inválida — ajuste de Fourier sem amostras suficientes",
                                 _esperada(zd, "média pela área")]
    assert not any(win._k_merid[ch].get_visible() for ch in ("k1", "k2"))
    monkeypatch.undo()
    win._k_cache.clear()
    win.k_spin.setValue(3.0)

def test_nome_do_paciente_nunca_aparece_na_ceratometria(janela):
    win = janela["win"]
    win.with_name.setChecked(True)                 # mesmo com a opção do nome ligada (só o resultado completo o usa)
    try:
        for i in (0, 1):
            _seleciona(janela, i)
            textos = [win.k_state.text(), win.k_leitura.text(), win.k_dica.text(), win.keratometry_tsv()]
            for fig in (win.cv_kmapa.figure, win.cv_kerato.figure):
                textos += [t.get_text() for t in fig.findobj(lambda o: hasattr(o, "get_text"))]
            junto = " ".join(textos).upper()
            assert "SINTETICO" not in junto and "PACIENTE" not in junto
    finally:
        win.with_name.setChecked(False)


def test_desatualizado_desabilita_e_limpa_o_controle(janela):
    win = janela["win"]
    _seleciona(janela, 0)
    win.k_spin.setValue(4.4)
    win.order_spin.setValue(7)                     # parâmetros alterados -> análises desatualizadas
    janela["app"].processEvents()
    st = win.selected_state()
    assert st.status == "desatualizado" and st.analysis is None
    assert not win.k_slider.isEnabled() and not win.k_spin.isEnabled()
    assert win.k_leitura.text() == "" and win._k_marcas == [] and win._k_ax is None and not win._k_cache
    assert any("desatualizados" in t.get_text() for a in win.cv_kmapa.figure.axes for t in a.texts)
    win.order_spin.setValue(6)
    win.compute_sync(janela["chaves"], PERFIS)
    _seleciona(janela, 0)
    assert win.k_slider.isEnabled() and win.k_slider.value() == 44     # a escolha volta com a análise nova
    assert _leitura(win)[0] == _esperada(_no_controle(win.selected_state(), 4.4)[0], "circunferência no vértice")


def test_janela_cabe_em_1366_px_em_todas_as_abas(janela):
    """Nenhuma aba (nem as sub-abas da Ceratometria) alarga a janela além de 1366 px: o texto pequeno dos números em
    destaque quebra a linha (antes, a aba Frente de onda levava a janela a 1399 px numa tela de 1366 px)."""
    win, app = janela["win"], janela["app"]
    _seleciona(janela, 0)
    try:
        for i in range(win.tabs.count()):
            win.tabs.setCurrentIndex(i)
            app.processEvents()
            if win.tabs.tabText(i) == "Ceratometria":
                for j in (1, 0):
                    win.k_subtabs.setCurrentIndex(j)
                    app.processEvents()
            largura = win.minimumSizeHint().width()
            assert largura <= 1366 and win.width() == 1366, (win.tabs.tabText(i), largura, win.width())
    finally:
        win.tabs.setCurrentIndex(_aba(win, "Ceratometria"))
        app.processEvents()
