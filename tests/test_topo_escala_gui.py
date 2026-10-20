"""Botões da escala de cores na aba "Topografia (TMS)" do programa (PySide6, sem tela): Standard do TMS ou ajustada ao
exame no passo de 0,125 · 0,25 · 0,5 · 0,75 · 1,00 D, com o mapa, a legenda e a folha A4 atualizados no clique.
Só exportações sintéticas no formato real (v1): o repositório é público.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_API", "pyside6")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from tmsab.pipeline import default_profiles  # noqa: E402
from tmsab.synthetic import SyntheticSpec, write_export_folder_v1  # noqa: E402
from tmsab.topo import PASSOS_ESCALA_D, ViewOptions, escala_do_mapa, fmt_passo  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:Enum value 'Qt..ApplicationAttribute.AA_UseHighDpiPixmaps':DeprecationWarning")


@pytest.fixture(scope="module")
def janela(tmp_path_factory):
    app = QApplication.instance() or QApplication([])
    from tmsab.gui import main_window
    pasta = tmp_path_factory.mktemp("escala_gui")
    write_export_folder_v1([SyntheticSpec(exam_id="T", R=7.8, Q=-0.2, cyl_D=2.0, axis_deg=30.0, eye="OD", decimals=4,
                                          pupil_center=(0.3, -0.2))], str(pasta))
    win = main_window.MainWindow(interactive=False)
    win.auto_export.setChecked(False); win.auto_compute.setChecked(False); win.with_name.setChecked(False)
    chave = win.load_package(str(pasta), exam_id="90001")
    win.compute_sync([chave], default_profiles([6.0], ("vertice",), 6))
    win.tabs.setCurrentIndex(next(i for i in range(win.tabs.count()) if win.tabs.tabText(i) == "Topografia (TMS)"))
    win.resize(1366, 768)
    win.show()
    app.processEvents()
    yield {"win": win, "app": app, "st": win.packages[chave]}
    win.close()


def _textos(fig) -> list[str]:
    return [t.get_text() for t in fig.texts] + [t.get_text() for a in fig.axes for t in a.texts]


def _botao(win, valor: str):
    return next(b for b in win.vg_scale.buttons() if b.property("value") == valor)


def test_seis_botoes_standard_e_cinco_passos(janela):
    win = janela["win"]
    valores = [b.property("value") for b in win.vg_scale.buttons()]
    assert valores == ["tms"] + [f"auto:{p}" for p in PASSOS_ESCALA_D]
    assert [b.text() for b in win.vg_scale.buttons()][1:] == [fmt_passo(p) for p in PASSOS_ESCALA_D]
    assert _botao(win, "tms").isChecked() and win._view_options().scale == "tms"
    assert "escala Standard do TMS (D)" in _textos(win.cv_topo.figure)


@pytest.mark.parametrize("passo", PASSOS_ESCALA_D)
def test_clique_no_passo_muda_mapa_legenda_e_folha_a4(janela, passo):
    win, app, st = janela["win"], janela["app"], janela["st"]
    _botao(win, f"auto:{passo}").click()
    app.processEvents()
    vo = win._view_options()
    assert (vo.scale, vo.step) == ("auto", passo)
    niveis, cortada = escala_do_mapa(st.exam, vo)
    assert np.allclose(np.diff(niveis), passo)
    titulo = f"escala ajustada ao exame, passo {fmt_passo(passo)} D" + (" (extremos nas cores das pontas)" if cortada else "") + " (D)"
    assert titulo in _textos(win.cv_topo.figure)
    malha = next(c for a in win.cv_topo.figure.axes for c in a.collections if hasattr(c, "norm") and hasattr(c.norm, "boundaries"))
    assert np.array_equal(malha.norm.boundaries, niveis)                     # o mapa desenhado usa a escala do passo
    assert any(f"passo {fmt_passo(passo)} D" in t for t in _textos(win.cv_summary.figure))   # folha A4 junto


def test_voltar_a_standard(janela):
    win, app = janela["win"], janela["app"]
    _botao(win, "auto:0.25").click(); app.processEvents()
    _botao(win, "tms").click(); app.processEvents()
    assert win._view_options() == ViewOptions(scale="tms", step=0.5, center=win._view_options().center)
    assert "escala Standard do TMS (D)" in _textos(win.cv_topo.figure)
    assert any("escala Standard do TMS (40,5–53,0 D)" in t for t in _textos(win.cv_summary.figure))
