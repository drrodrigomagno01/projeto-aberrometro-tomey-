"""Janela principal do programa (plano §7).

Estados por pacote: carregado (importado, sem cálculo) · calculando · calculado · desatualizado
(parâmetros mudaram; nenhum número antigo é exibido) · inválida (importação ou análise bloqueada).
"""
from __future__ import annotations

import dataclasses
import html
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from matplotlib import patheffects
from matplotlib.backend_bases import MouseButton
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.text import Text
from PySide6.QtCore import QLocale, QSize, Qt, Slot
from PySide6.QtGui import QAction, QColor, QFont, QFontMetrics
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from ..adapters import AdapterError, detect_profile, import_tms_bundle, list_exams, patient_display_name
from ..keratometry import (
    COLUNAS_TABELA,
    CONTROLE_INICIAL_MM,
    CONTROLE_LEGENDA,
    CONTROLE_MAX_SEM_RAD_MM,
    CONTROLE_MIN_MM,
    LEGENDA_TABELA,
    ceratometria_no_diametro,
    eixo_tabo,
    faixa_do_controle,
    fmt_eixo,
    linha_da_zona,
    linhas_resumo,
)
from ..model import Analysis, KeratometryResult, KeratometryZone, NormalizedExam, OpticalProfile, QualityConfig, ReconstructionConfig, WavefrontResult
from ..pipeline import default_profiles, profiles_for_exam
from ..plots import (
    compare_scale,
    figure_resultado_completo,
    plot_coverage,
    plot_elevation,
    plot_topo_map,
    plot_wavefront,
    plot_zernike_bars,
)
from ..pupila import chord_mu_aparente, fmt_chord_mu
from ..topo import PASSOS_ESCALA_D, ViewOptions, figure_tms_view, fmt_passo, passo_valido, zone_axis_stats
from ..reporting import export_all, report_lines
from .canvas import MplCanvas
from .compute import profile_settings_diff, run_analysis
from .worker import AnalysisWorker

STATUS_TEXT = "Ferramenta de pesquisa — sem validação clínica; resultados vinculados a abertura, centro, modelo e versões"
ABOUT_TEXT = (
    "<b>Aberrometria corneana anterior calculada — TMS (pesquisa)</b><br>"
    f"versão {__version__}<br><br>"
    "<b>Finalidade:</b> estimar a contribuição óptica modelada da superfície anterior da córnea a partir de "
    "exportações numéricas do topógrafo Tomey TMS (mapa de frente de onda, coeficientes de Zernike OSA, "
    "aberração esférica, coma, trefoil e RMS de alta ordem), sempre vinculados à abertura, ao centro, ao modelo "
    "óptico e às versões usadas.<br><br>"
    "<b>Limitações:</b> não é frente de onda ocular total; não inclui superfície posterior, cristalino ou LIO; "
    "não realiza aquisição nem controla o instrumento; não emite alegações diagnósticas, seleção de LIO ou "
    "planejamento terapêutico. Perfis de exportação presumidos exigem verificação com arquivos reais (G0/G1). "
    "Limites de disponibilidade são provisórios (orçamento de erro), não critérios clínicos."
)


@dataclass
class PackageState:
    key: str
    directory: str
    exam_id: Optional[str] = None          # exame dentro de uma pasta exportada com vários (formato real do TMS)
    exam: Optional[NormalizedExam] = None
    analysis: Optional[Analysis] = None
    rec: object = None
    status: str = "carregado"
    message: str = ""
    profiles_used: list = field(default_factory=list)

    @property
    def label(self) -> str:
        if self.exam is not None:
            cap = f" c{self.exam.identity.capture_index}" if self.exam.identity.capture_index else ""
            ident = f"{self.exam.identity.exam_id} {self.exam.identity.eye or '?'}{cap}"
            sid = self.exam.identity.study_id
            if sid and sid.startswith("P-") and not self.exam.bundle.synthetic:
                ident += f" · {sid}"          # código pseudonimizado: distingue pacientes numa mesma pasta
        elif self.exam_id:
            ident = f"{self.exam_id}"
        else:
            ident = os.path.basename(self.directory.rstrip("/\\"))
        return f"{ident} — {self.status}"


class BigNumber(QWidget):
    """Valor em destaque com legenda; mostra '—' quando ausente (nunca zero)."""

    def __init__(self, caption: str, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(0)
        self.cap = QLabel(caption)
        self.cap.setStyleSheet("color: #555; font-size: 10px;")
        self.cap.setWordWrap(True)
        self.val = QLabel("—")
        f = QFont()
        f.setPointSize(15)
        f.setBold(True)
        self.val.setFont(f)
        self.sub = QLabel("")
        self.sub.setStyleSheet("color: #555; font-size: 10px;")
        self.sub.setWordWrap(True)
        self.sub.setMinimumWidth(120)        # texto longo quebra a linha em vez de alargar a janela (tela de 1366 px)
        lay.addWidget(self.cap)
        lay.addWidget(self.val)
        lay.addWidget(self.sub)
        self.setFrameStyle = None

    def set_value(self, text: Optional[str], sub: str = "") -> None:
        self.val.setText(text if text else "—")
        self.sub.setText(sub)


class _CaixaDiametro(QDoubleSpinBox):
    """Caixa do diâmetro com ponto decimal, como as tabelas e o mapa; a vírgula digitada (teclado ABNT) vira ponto."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLocale(QLocale.c())

    def validate(self, text, pos):
        return super().validate(text.replace(",", "."), pos)

    def valueFromText(self, text):  # noqa: N802
        return super().valueFromText(text.replace(",", "."))


def _fmt(v, nd: int = 4, sign: bool = True) -> str:
    try:
        if v is None or not np.isfinite(v):
            return "—"
        return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    except TypeError:
        return "—"


# ---- aba "Ceratometria": as 7 colunas do núcleo (COLUNAS_TABELA) + cobertura interpolada e vetor de potência
K_COLUNAS = tuple(COLUNAS_TABELA) + ("% interp.", "J0 (D)", "J45 (D)")
K_LEGENDA_COLUNAS_EXTRAS = ("% interp. = fração dos meridianos (na zona pupilar e nas zonas 0–d, da área) com pontos "
                            "interpolados pelo aparelho; J0/J45 = vetor de potência do ΔK (Thibos), positivo = a favor "
                            "da regra (J0 = −ΔK/2·cos 2·eixo K2).")
K_SEPARADOR_ANEIS = "anéis medidos (Ø medido)"
K_COR_K1, K_COR_K2 = "#2a78d6", "#eb6834"          # K1 plano / K2 curvo — as mesmas cores nos dois gráficos
K_TINTA, K_TINTA_FRACA, K_GRADE, K_EIXO = "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# controle de diâmetro: linha do diâmetro escolhido nos gráficos (traço-ponto escuro; a pupila é tracejada e mais clara)
K_COR_ESCOLHIDO = "#1c1b19"
K_TEXTO_K1, K_TEXTO_K2 = "#1f5fae", "#b4501a"     # os mesmos matizes, mais escuros: texto pequeno legível (≥ 4,5:1)
K_HALO = [patheffects.withStroke(linewidth=3.4, foreground="white")]          # traços sobre o mapa colorido
K_HALO_TEXTO = [patheffects.withStroke(linewidth=2.6, foreground="white")]
K_DICA_MAPA = "Clique ou arraste no mapa para escolher o diâmetro."


class _RotuloMeridiano(Text):
    """Rótulo na ponta do meridiano sobre o mapa. Ao desenhar (leiaute final, qualquer tamanho de janela), entra no mapa
    se passaria da borda; com o mapa ampliado e a ponta fora da vista, o recorte dos eixos o esconde."""

    def draw(self, renderer):
        ax = self.axes
        if not self.get_visible() or ax is None or not self.get_text():
            return super().draw(renderer)
        pos = self.get_position()
        x, y = ax.transData.transform(pos)
        ab = ax.bbox
        if not (ab.x0 <= x <= ab.x1 and ab.y0 <= y <= ab.y1):
            return super().draw(renderer)
        bb = self.get_window_extent(renderer)
        dx = max(0.0, ab.x0 + 2 - bb.x0) - max(0.0, bb.x1 - ab.x1 + 2)
        dy = max(0.0, ab.y0 + 2 - bb.y0) - max(0.0, bb.y1 - ab.y1 + 2)
        if not (dx or dy):
            return super().draw(renderer)
        self.set_position(tuple(ax.transData.inverted().transform((x + dx, y + dy))))
        try:
            return super().draw(renderer)
        finally:
            self.set_position(pos)


def _fmt_j(v) -> str:
    """Componente do vetor de potência com sinal e 2 casas, sem "−0.00" (arredonda antes de formatar)."""
    if v is None or not np.isfinite(v):
        return "—"
    return f"{round(float(v), 2) + 0.0:+.2f}"


def _cabecalho_k(titulo: str) -> str:
    """Título de coluna em duas linhas: a tabela tem 10 colunas e divide a largura com os gráficos."""
    return titulo.replace(" (", "\n(").replace("eixo ", "eixo\n").replace("% ", "%\n")


def _zonas_das_linhas(kres: KeratometryResult, linhas: list[dict]) -> list:
    """A KeratometryZone de cada linha de linhas_resumo(…, incluir_aneis=True) — None na linha do SimK do TMS —,
    para as colunas extras e o realce do eixo. Casa por tipo, na ordem das listas do resultado (circunferências,
    zonas 0–d, pupila, anéis); linha que não confere com a zona esperada fica sem extras (nunca com os de outra)."""
    grupos: dict[str, list] = {}
    todas = list(kres.zonas) + list(getattr(kres, "discos", None) or []) + list(kres.aneis)
    for z in todas + ([kres.pupila] if kres.pupila is not None else []):
        grupos.setdefault(z.tipo, []).append(z)
    fontes = {tipo: iter(zs) for tipo, zs in grupos.items()}
    out = []
    for ln in linhas:
        z = next(fontes[ln["tipo"]], None) if ln["tipo"] in fontes else None
        if z is not None and (z.k1_plano_D != ln["k1"] or z.validity != ln["validity"]):
            z = None
        out.append(z)
    return out


def _diametro_simk(kres: KeratometryResult) -> float:
    """Posição do SimK do TMS no eixo dos diâmetros: média dos diâmetros medidos dos anéis da regra (6–8)."""
    d = [z.diametro_mm for z in kres.aneis if z.anel in kres.config.simk_aneis and z.diametro_mm]
    return float(np.mean(d)) if d else 2.5


def _estilo_eixos_k(ax) -> None:
    ax.grid(True, color=K_GRADE, lw=0.6)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(K_EIXO)
    ax.tick_params(labelsize=7, colors=K_TINTA)


class MainWindow(QMainWindow):
    def __init__(self, adapter_profile: str = "auto", interactive: bool = True):
        super().__init__()
        self.adapter_profile = adapter_profile
        self.interactive = interactive
        self.packages: dict[str, PackageState] = {}
        self.worker: Optional[AnalysisWorker] = None
        self._params_block = False
        self.setWindowTitle(f"Aberrometria corneana anterior calculada — TMS (pesquisa) v{__version__}")
        self.resize(1366, 768)
        self._build_menu()
        self._build_central()
        self.statusBar().addPermanentWidget(QLabel(STATUS_TEXT))
        self._show_no_selection()

    # ------------------------------------------------------------------ menu
    def _build_menu(self):
        m = self.menuBar().addMenu("&Arquivo")
        for text, slot in (
            ("Abrir pasta exportada do TMS (paciente)…", self.open_export_folder_dialog),
            ("Abrir pacote…", self.open_package_dialog),
            ("Abrir pasta com vários pacotes…", self.open_folder_dialog),
            ("Gerar sintéticos…", self.generate_synthetic_dialog),
            (None, None),
            ("Exportar relatório (JSON/CSV/PDF)…", self.export_selected),
            ("Exportar todos…", self.export_all_packages),
            (None, None),
            ("Sair", self.close),
        ):
            if text is None:
                m.addSeparator()
                continue
            a = QAction(text, self)
            a.triggered.connect(slot)
            m.addAction(a)
        h = self.menuBar().addMenu("A&juda")
        a = QAction("Sobre", self)
        a.triggered.connect(self.about)
        h.addAction(a)

    # --------------------------------------------------------------- central
    def _build_central(self):
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_tabs())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([330, 1030])
        self.setCentralWidget(splitter)

    def _build_left_panel(self) -> QWidget:
        w = QWidget()
        w.setMinimumWidth(340)
        w.setMaximumWidth(460)
        lay = QVBoxLayout(w)

        g = QGroupBox("Pacotes carregados")
        gl = QVBoxLayout(g)
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.currentItemChanged.connect(lambda *_: self.on_selection_changed())
        gl.addWidget(self.list)
        row = QHBoxLayout()
        b1 = QPushButton("Abrir pacote…"); b1.clicked.connect(self.open_package_dialog)
        b2 = QPushButton("Abrir pasta com vários pacotes…"); b2.clicked.connect(self.open_folder_dialog)
        row.addWidget(b1); row.addWidget(b2)
        gl.addLayout(row)
        row2 = QHBoxLayout()
        b3 = QPushButton("Gerar sintéticos…"); b3.clicked.connect(self.generate_synthetic_dialog)
        b4 = QPushButton("Remover"); b4.clicked.connect(self.remove_selected)
        row2.addWidget(b3); row2.addWidget(b4)
        gl.addLayout(row2)
        lay.addWidget(g, 1)

        p = QGroupBox("Perfil de análise")
        pl = QVBoxLayout(p)
        form = QFormLayout()
        pl.addLayout(form)
        self.center_combo = QComboBox()
        self.center_combo.addItems(["vértice", "pupila", "ambos"])
        self.center_combo.setCurrentIndex(2)
        form.addRow("Centro:", self.center_combo)
        drow = QGridLayout(); drow.setHorizontalSpacing(4)
        drow.addWidget(QLabel("Diâmetros:"), 0, 0)
        self.d3 = QCheckBox("3"); self.d3.setChecked(True)
        self.d4 = QCheckBox("4"); self.d4.setChecked(True)
        self.d5 = QCheckBox("5"); self.d5.setChecked(True)
        self.d6 = QCheckBox("6 mm"); self.d6.setChecked(True)
        self.d_pupil = QCheckBox("pupila medida"); self.d_pupil.setChecked(True)
        self.d_pupil.setToolTip("Perfil centrado na pupila com o diâmetro pupilar medido pelo TMS (PUP)")
        self.d_other_check = QCheckBox("outro")
        self.d_other = QDoubleSpinBox(); self.d_other.setRange(2.0, 8.0); self.d_other.setSingleStep(0.5); self.d_other.setValue(5.0); self.d_other.setSuffix(" mm")
        self.d_other.setEnabled(False)
        self.d_other_check.toggled.connect(self.d_other.setEnabled)
        for i, x in enumerate((self.d3, self.d4, self.d5, self.d6)):
            drow.addWidget(x, 0, 1 + i)
        drow.addWidget(self.d_pupil, 1, 0, 1, 2)
        drow.addWidget(self.d_other_check, 1, 2, 1, 2)
        drow.addWidget(self.d_other, 1, 4, 1, 2)
        pl.addLayout(drow)
        self.order_spin = QSpinBox(); self.order_spin.setRange(4, 10); self.order_spin.setValue(6)
        form.addRow("Ordem de Zernike:", self.order_spin)
        self.method_combo = QComboBox(); self.method_combo.addItems(["conica_zernike", "rbf_tps"])
        form.addRow("Reconstrução:", self.method_combo)
        self.surf_order_spin = QSpinBox(); self.surf_order_spin.setRange(6, 14); self.surf_order_spin.setValue(10)
        form.addRow("Ordem da superfície:", self.surf_order_spin)
        self.ref_combo = QComboBox(); self.ref_combo.addItems(["paraxial_media_principais", "refoco_rms"])
        form.addRow("Referência:", self.ref_combo)
        lay.addWidget(p)
        for sig in (self.center_combo.currentIndexChanged, self.d3.toggled, self.d4.toggled, self.d5.toggled, self.d6.toggled,
                    self.d_pupil.toggled, self.d_other_check.toggled,
                    self.d_other.valueChanged, self.order_spin.valueChanged, self.method_combo.currentIndexChanged,
                    self.surf_order_spin.valueChanged, self.ref_combo.currentIndexChanged):
            sig.connect(lambda *_: self.on_params_changed())

        self.auto_compute = QCheckBox("Calcular automaticamente ao abrir a pasta"); self.auto_compute.setChecked(True)
        self.auto_export = QCheckBox("Salvar relatórios em <pasta>/analises ao terminar"); self.auto_export.setChecked(True)
        self.with_name = QCheckBox("Nome do paciente (XREF) e data no resultado completo"); self.with_name.setChecked(True)
        self.with_name.setToolTip("O nome entra só na página impressa/1ª página do PDF; JSON e CSV levam apenas o código do estudo.")
        self.with_name.toggled.connect(lambda *_: self._refresh_summary_only())
        for cb in (self.auto_compute, self.auto_export, self.with_name):
            cb.setStyleSheet("font-size:11px;")
            lay.addWidget(cb)
        crow = QHBoxLayout()
        self.btn_compute = QPushButton("Calcular"); self.btn_compute.clicked.connect(self.start_compute)
        self.btn_cancel = QPushButton("Cancelar"); self.btn_cancel.clicked.connect(self.cancel_compute); self.btn_cancel.setEnabled(False)
        crow.addWidget(self.btn_compute); crow.addWidget(self.btn_cancel)
        lay.addLayout(crow)
        self.progress = QProgressBar(); self.progress.setRange(0, 100); self.progress.setValue(0)
        lay.addWidget(self.progress)
        self.progress_label = QLabel("")
        self.progress_label.setWordWrap(True)
        self.progress_label.setStyleSheet("color:#444; font-size:10px;")
        lay.addWidget(self.progress_label)
        return w

    def _build_tabs(self) -> QWidget:
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_summary_tab(), "Resultado completo")
        self.tabs.addTab(self._build_topo_tab(), "Topografia (TMS)")
        self.tabs.addTab(self._build_keratometry_tab(), "Ceratometria")
        self.tabs.addTab(self._build_import_tab(), "Importação")
        self.tabs.addTab(self._build_geometry_tab(), "Geometria")
        self.tabs.addTab(self._build_wavefront_tab(), "Frente de onda")
        self.tabs.addTab(self._build_zernike_tab(), "Zernike")
        self.tabs.addTab(self._build_compare_tab(), "Comparação")
        self.tabs.addTab(self._build_technical_tab(), "Técnico")
        return self.tabs

    # ---- abas
    def _build_keratometry_tab(self) -> QWidget:
        """K1 (plano), K2 (curvo) e eixos por zona, da potência axial do TMS (tmsab/keratometry.py).
        Identificação só por exame/código de estudo: o nome do XREF fica restrito ao resultado completo."""
        w = QWidget(); lay = QVBoxLayout(w)
        self._k_ident = ""
        self.k_state = QLabel("")
        f = QFont(); f.setBold(True)
        self.k_state.setFont(f); self.k_state.setWordWrap(True)
        lay.addWidget(self.k_state)
        box = QGroupBox("Potência axial do TMS (D) — K1 = meridiano mais plano, K2 = mais curvo; eixos TABO")
        box.setMinimumHeight(118)                         # legenda + valor + subtexto de 2 linhas sem cortar
        grid = QGridLayout(box)
        grid.setContentsMargins(6, 2, 6, 2)
        self.k_bn = {
            "pupila": BigNumber("K na zona pupilar — K1 plano @ eixo / K2 curvo @ eixo"),
            "simk": BigNumber("SimK do TMS (≈2.5 mm) — plano @ eixo / curvo @ eixo"),
            "diametro": BigNumber("Diâmetro pupilar desta captura"),
        }
        for i, key in enumerate(("pupila", "simk", "diametro")):
            grid.addWidget(self.k_bn[key], 0, i)
            grid.setColumnStretch(i, 2 if key == "diametro" else 3)
        lay.addWidget(box)
        # à esquerda: tabela, cópia/legenda e notas; à direita, na altura toda: o mapa axial com o controle de diâmetro
        # (padrão) e os gráficos por diâmetro — na janela padrão (1366×768) o mapa não caberia só ao lado da tabela
        split = QSplitter(Qt.Horizontal)
        esq = QWidget(); el = QVBoxLayout(esq); el.setContentsMargins(0, 0, 0, 0)
        self.k_table = QTableWidget(0, len(K_COLUNAS))
        self.k_table.setHorizontalHeaderLabels([_cabecalho_k(c) for c in K_COLUNAS])
        self.k_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.k_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)   # largura medida em _fill_k_table
        self.k_table.setWordWrap(True)
        self.k_table.verticalHeader().setVisible(False)
        self.k_table.verticalHeader().setDefaultSectionSize(self.k_table.fontMetrics().height() + 8)
        self.k_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.k_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        el.addWidget(self.k_table, 1)
        base = QHBoxLayout()
        self.btn_k_copy = QPushButton("Copiar tabela (TSV)")
        self.btn_k_copy.setToolTip("Copia a tabela como está na tela (colunas separadas por tabulação), para colar numa planilha. "
                                   "O exame é identificado só pelo número e pelo código de estudo.")
        self.btn_k_copy.clicked.connect(self.copy_keratometry_tsv)
        self.btn_k_copy.setEnabled(False)
        base.addWidget(self.btn_k_copy, 0, Qt.AlignTop)
        self.k_legend = QLabel(LEGENDA_TABELA + " " + K_LEGENDA_COLUNAS_EXTRAS)
        self.k_legend.setWordWrap(True)
        self.k_legend.setStyleSheet("color:#444; font-size:10px;")
        base.addWidget(self.k_legend, 1)
        el.addLayout(base)
        self.k_notes = QPlainTextEdit(); self.k_notes.setReadOnly(True); self.k_notes.setMaximumHeight(96)
        self.k_notes.setPlaceholderText("Conferência com o TMS (SimK e ACP) e notas da ceratometria (após calcular)")
        el.addWidget(self.k_notes)
        split.addWidget(esq)
        self.k_subtabs = QTabWidget()
        self.k_subtabs.setDocumentMode(True)
        self.k_subtabs.addTab(self._build_k_mapa(), "Mapa axial — diâmetro")
        self.cv_kerato = MplCanvas(width=4.4, height=5.0)
        self.k_subtabs.addTab(self.cv_kerato, "Gráficos por diâmetro")
        self.k_subtabs.currentChanged.connect(self._k_subaba_mudou)
        split.addWidget(self.k_subtabs)
        split.setStretchFactor(0, 0); split.setStretchFactor(1, 1)     # a tabela tem largura fixa; a sobra vai para o mapa
        split.setSizes([560, 480])
        self.k_split, self._k_divisao_manual = split, False
        split.splitterMoved.connect(self._k_divisao_movida)
        lay.addWidget(split, 1)
        return w

    def _build_k_mapa(self) -> QWidget:
        """Controle de diâmetro: o usuário escolhe Ø d (barra, caixa ou clique no mapa) e vê a circunferência Ø d e a zona
        0–d no vértice, calculadas pelo núcleo (ceratometria_no_diametro) com a configuração da própria análise — no
        Ø 3 mm são exatamente as linhas da tabela. Só exibe números; não altera a análise nem os relatórios."""
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 2); lay.setSpacing(2)
        self._k_decimos = int(round(CONTROLE_INICIAL_MM * 10))     # escolha do usuário (décimos de mm), mantida entre exames
        self._k_st: Optional[PackageState] = None                  # pacote mostrado no controle (None = controle vazio)
        self._k_cache: dict[tuple[str, int], tuple] = {}           # (analysis_id, décimos) -> (circunferência, zona 0–d)
        self._k_arrastando = False
        self._k_marcas: list = []                                  # linha do diâmetro escolhido nos dois gráficos
        self._k_sem_mapa()
        row = QHBoxLayout(); row.setSpacing(6)
        row.addWidget(QLabel("Diâmetro:"))
        dica = ("Diâmetro da circunferência e da zona 0–d, centradas no vértice: arraste, use as setas ou clique no mapa "
                "(passo de 0.1 mm).")
        self.k_slider = QSlider(Qt.Horizontal)
        self.k_slider.setSingleStep(1); self.k_slider.setPageStep(5)          # setas: 0.1 mm; PgUp/PgDn: 0.5 mm
        self.k_slider.setTickPosition(QSlider.TicksBelow); self.k_slider.setTickInterval(10)
        self.k_spin = _CaixaDiametro()
        self.k_spin.setDecimals(1); self.k_spin.setSingleStep(0.1); self.k_spin.setSuffix(" mm")
        lo, hi = int(round(CONTROLE_MIN_MM * 10)), int(round(CONTROLE_MAX_SEM_RAD_MM * 10))
        self.k_slider.setRange(lo, hi); self.k_spin.setRange(lo / 10.0, hi / 10.0)
        self.k_slider.setValue(self._k_decimos); self.k_spin.setValue(self._k_decimos / 10.0)
        for x in (self.k_slider, self.k_spin):
            x.setToolTip(dica)
            x.setEnabled(False)
        self.k_slider.valueChanged.connect(self._k_slider_mudou)
        self.k_spin.valueChanged.connect(self._k_spin_mudou)
        row.addWidget(self.k_slider, 1); row.addWidget(self.k_spin)
        lay.addLayout(row)
        self.cv_kmapa = MplCanvas(width=4.4, height=4.0)
        if self.cv_kmapa.toolbar is not None:              # ícones menores: sobra altura para o mapa na janela padrão
            self.cv_kmapa.toolbar.setIconSize(QSize(18, 18))
        for nome, fn in (("button_press_event", self._k_mapa_press), ("motion_notify_event", self._k_mapa_move),
                         ("button_release_event", self._k_mapa_release)):
            self.cv_kmapa.canvas.mpl_connect(nome, fn)
        lay.addWidget(self.cv_kmapa, 1)
        self.k_leitura = QLabel("")
        self.k_leitura.setTextFormat(Qt.RichText)
        self.k_leitura.setWordWrap(True)
        self.k_leitura.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.k_leitura.setToolTip(CONTROLE_LEGENDA)
        self.k_leitura.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        # altura das 2 linhas quebradas + legenda das marcas: o mapa não "pula" ao arrastar entre diâmetros
        self.k_leitura.setMinimumHeight(5 * self.k_leitura.fontMetrics().lineSpacing())
        lay.addWidget(self.k_leitura)
        self.k_dica = QLabel(K_DICA_MAPA)
        self.k_dica.setStyleSheet("color:#6b6b6b; font-size:10px;")
        self.k_dica.setWordWrap(True)
        self.k_dica.setToolTip(CONTROLE_LEGENDA)
        lay.addWidget(self.k_dica)
        return w

    def _build_import_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        self.import_state = QLabel("Nenhum pacote selecionado")
        f = QFont(); f.setBold(True); f.setPointSize(11)
        self.import_state.setFont(f); self.import_state.setWordWrap(True)
        lay.addWidget(self.import_state)
        split = QSplitter(Qt.Vertical)
        self.import_diag = QPlainTextEdit(); self.import_diag.setReadOnly(True)
        self.import_diag.setPlaceholderText("Diagnósticos e avisos da importação")
        split.addWidget(self.import_diag)
        self.files_table = QTableWidget(0, 5)
        self.files_table.setHorizontalHeaderLabels(["Papel", "Arquivo", "SHA-256", "Linhas × colunas", "Orientação"])
        self.files_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.files_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        split.addWidget(self.files_table)
        bottom = QWidget(); bl = QHBoxLayout(bottom); bl.setContentsMargins(0, 0, 0, 0)
        self.manifest_text = QPlainTextEdit(); self.manifest_text.setReadOnly(True)
        self.manifest_text.setPlaceholderText("Manifesto (manifest.json)")
        self.pupil_label = QLabel("Pupila: —"); self.pupil_label.setWordWrap(True); self.pupil_label.setAlignment(Qt.AlignTop)
        self.pupil_label.setMinimumWidth(260)
        bl.addWidget(self.manifest_text, 2); bl.addWidget(self.pupil_label, 1)
        split.addWidget(bottom)
        split.setSizes([180, 160, 180])
        lay.addWidget(split)
        return w

    def _build_geometry_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        row = QHBoxLayout()
        self.cv_elev = MplCanvas(width=4.6, height=4.2)
        self.cv_cov = MplCanvas(width=4.6, height=4.2)
        row.addWidget(self.cv_elev); row.addWidget(self.cv_cov)
        lay.addLayout(row, 1)
        self.geom_text = QPlainTextEdit(); self.geom_text.setReadOnly(True); self.geom_text.setMaximumHeight(120)
        self.geom_text.setPlaceholderText("Resumo da reconstrução (após calcular)")
        lay.addWidget(self.geom_text)
        return w

    def _build_wavefront_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        top = QHBoxLayout()
        top.addWidget(QLabel("Perfil:"))
        self.profile_combo = QComboBox()
        self.profile_combo.currentIndexChanged.connect(lambda *_: self.on_profile_combo_changed())
        top.addWidget(self.profile_combo, 1)
        self.wf_state = QLabel("")
        f = QFont(); f.setBold(True)
        self.wf_state.setFont(f)
        top.addWidget(self.wf_state, 1)
        lay.addLayout(top)
        row = QHBoxLayout()
        self.cv_w = MplCanvas(width=3.6, height=3.4)
        self.cv_wfit = MplCanvas(width=3.6, height=3.4)
        self.cv_res = MplCanvas(width=3.6, height=3.4)
        for c in (self.cv_w, self.cv_wfit, self.cv_res):
            row.addWidget(c)
        lay.addLayout(row, 1)
        self.block_label = QLabel("")
        self.block_label.setWordWrap(True)
        self.block_label.setStyleSheet("color: #8B0000; font-weight: bold; padding: 6px; border: 1px solid #8B0000;")
        self.block_label.hide()
        lay.addWidget(self.block_label)
        grid_w = QGroupBox("Números do perfil (µm, convenção OSA; sinais preservados)")
        grid = QGridLayout(grid_w)
        self.bn = {
            "sa": BigNumber("Aberração esférica primária c(4,0)"),
            "coma": BigNumber("|Coma| — c(3,−1) / c(3,+1)"),
            "trefoil": BigNumber("|Trefoil| — c(3,−3) / c(3,+3)"),
            "hoa": BigNumber("RMS de alta ordem (n ≥ 3)"),
            "astig": BigNumber("Astigmatismo c(2,−2) / c(2,+2)"),
            "defocus": BigNumber("Defocus c(2,0) (referência paraxial)"),
        }
        for i, key in enumerate(["sa", "coma", "trefoil", "hoa", "astig", "defocus"]):
            grid.addWidget(self.bn[key], 0, i)
        self.wf_notes = QLabel("")
        self.wf_notes.setWordWrap(True)
        self.wf_notes.setStyleSheet("color:#444; font-size:10px;")
        grid.addWidget(self.wf_notes, 1, 0, 1, 6)
        lay.addWidget(grid_w)
        return w

    def _build_zernike_tab(self) -> QWidget:
        split = QSplitter(Qt.Horizontal)
        self.z_table = QTableWidget(0, 5)
        self.z_table.setHorizontalHeaderLabels(["j", "n", "m", "Nome", "Valor (µm)"])
        self.z_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.z_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.z_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        split.addWidget(self.z_table)
        self.cv_zbars = MplCanvas(width=6, height=4)
        split.addWidget(self.cv_zbars)
        split.setSizes([480, 700])
        return split

    def _build_compare_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        top = QHBoxLayout()
        top.addWidget(QLabel("Pacote A:")); self.cmp_a = QComboBox(); top.addWidget(self.cmp_a, 1)
        top.addWidget(QLabel("Pacote B:")); self.cmp_b = QComboBox(); top.addWidget(self.cmp_b, 1)
        top.addWidget(QLabel("Perfil:")); self.cmp_profile = QComboBox(); top.addWidget(self.cmp_profile, 1)
        b = QPushButton("Comparar"); b.clicked.connect(self.run_compare); top.addWidget(b)
        lay.addLayout(top)
        self.cmp_msg = QLabel("Selecione dois pacotes calculados e um perfil.")
        self.cmp_msg.setWordWrap(True)
        lay.addWidget(self.cmp_msg)
        row = QHBoxLayout()
        self.cv_cmp_a = MplCanvas(width=3.6, height=3.4)
        self.cv_cmp_b = MplCanvas(width=3.6, height=3.4)
        self.cv_cmp_d = MplCanvas(width=3.6, height=3.4)
        for c in (self.cv_cmp_a, self.cv_cmp_b, self.cv_cmp_d):
            row.addWidget(c)
        lay.addLayout(row, 1)
        self.cmp_table = QTableWidget(0, 7)
        self.cmp_table.setHorizontalHeaderLabels(["j", "n", "m", "Nome", "A (µm)", "B (µm)", "A − B (µm)"])
        self.cmp_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.cmp_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.cmp_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.cmp_table.setMaximumHeight(220)
        lay.addWidget(self.cmp_table)
        return w

    # ------------------------------------------------------------ topografia (estilo TMS) com filtros
    def _radio_row(self, title: str, options: list[tuple[str, object]], default_index: int = 0) -> tuple[QGroupBox, QButtonGroup]:
        box = QGroupBox(title); lay = QHBoxLayout(box); lay.setContentsMargins(6, 2, 6, 2); lay.setSpacing(4)
        grp = QButtonGroup(box); grp.setExclusive(True)
        for i, (label, value) in enumerate(options):
            rb = QRadioButton(label); rb.setProperty("value", value); rb.setChecked(i == default_index)
            grp.addButton(rb, i); lay.addWidget(rb)
        lay.addStretch(1)
        grp.buttonClicked.connect(lambda *_: self._on_view_changed())
        return box, grp

    def _escala_row(self) -> tuple[QGroupBox, QButtonGroup]:
        """Escala de cores num só grupo exclusivo: a Standard do TMS (fixa: 40,5–53 D, passo 0,5 D) ou a ajustada ao
        exame no passo do botão (0,125 a 1,00 D). Valor de cada botão: "tms" ou "auto:<passo>"."""
        box = QGroupBox("Escala de cores (D)")
        grid = QGridLayout(box); grid.setContentsMargins(6, 2, 6, 2); grid.setHorizontalSpacing(4); grid.setVerticalSpacing(0)
        grp = QButtonGroup(box); grp.setExclusive(True)
        rb = QRadioButton("Standard TMS (40,5–53, passo 0,5)"); rb.setProperty("value", "tms"); rb.setChecked(True)
        rb.setToolTip("Escala fixa do aparelho: 26 faixas de 0,5 D de 40,5 a 53,0 D (as mesmas cores em todo exame).")
        grp.addButton(rb, 0); grid.addWidget(rb, 0, 0, 1, len(PASSOS_ESCALA_D) + 1)
        rot = QLabel("ajustada:")                    # curto: o grupo cabe ao lado do "Mapa" numa tela de 1366 px
        rot.setToolTip("Escala ajustada ao exame no passo escolhido (D).")
        grid.addWidget(rot, 1, 0)
        for i, passo in enumerate(PASSOS_ESCALA_D):
            rb = QRadioButton(fmt_passo(passo)); rb.setProperty("value", f"auto:{passo}")
            rb.setToolTip(f"Escala ajustada ao exame em faixas de {fmt_passo(passo)} D, com as cores do TMS (8 a 26 faixas; "
                          "acima de 26, a janela fica centrada no exame e os extremos saem nas cores das pontas).")
            grp.addButton(rb, i + 1); grid.addWidget(rb, 1, i + 1)
        grid.setColumnStretch(len(PASSOS_ESCALA_D) + 1, 1)
        grp.buttonClicked.connect(lambda *_: self._on_view_changed())
        return box, grp

    def _build_topo_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        row1 = QHBoxLayout(); row2 = QHBoxLayout()
        b, self.vg_kind = self._radio_row("Mapa", [("Axial", "axial"), ("Tangencial", "tangencial"), ("Refrativa", "refrativa"),
                                                   ("Elevação", "elevacao"), ("Frente de onda", "frente_onda"), ("Alta ordem", "alta_ordem")])
        row1.addWidget(b, 3)
        b, self.vg_scale = self._escala_row()
        row1.addWidget(b, 2)
        b, self.vg_zone = self._radio_row("Zona (diâmetro)", [("tudo", None), ("3 mm", 3.0), ("5 mm", 5.0), ("7 mm", 7.0)])
        row2.addWidget(b, 2)
        b, self.vg_pupil = self._radio_row("Diâmetro pupilar", [("medida (PUP)", None), ("3 mm", 3.0), ("4 mm", 4.0), ("5 mm", 5.0), ("6 mm", 6.0)])
        row2.addWidget(b, 3)
        b, self.vg_center = self._radio_row("Centro (frente de onda)", [("vértice", "vertice"), ("pupila", "pupila")])
        row2.addWidget(b, 2)
        lay.addLayout(row1); lay.addLayout(row2)
        body = QHBoxLayout()
        self.cv_topo = MplCanvas(width=6.0, height=7.0, dpi=90, toolbar=True)
        body.addWidget(self.cv_topo, 3)
        side = QVBoxLayout()
        self.topo_text = QPlainTextEdit(); self.topo_text.setReadOnly(True)
        f = QFont("Monospace"); f.setStyleHint(QFont.TypeWriter); f.setPointSize(9); self.topo_text.setFont(f)
        self.topo_text.setPlaceholderText("Astig Stats por zona (3/5/7 mm), índices do TMS e perfil de frente de onda escolhido")
        side.addWidget(self.topo_text, 1)
        bp = QPushButton("Imprimir esta visualização…"); bp.clicked.connect(self.print_topo_view)
        bs = QPushButton("Salvar esta visualização (PDF/PNG)…"); bs.clicked.connect(self.save_topo_view)
        side.addWidget(bp); side.addWidget(bs)
        hint = QLabel("Os filtros (escala, zona, diâmetro pupilar, centro) também valem para a aba \"Resultado completo\" e para a impressão.")
        hint.setWordWrap(True); hint.setStyleSheet("color:#555; font-size:11px;")
        side.addWidget(hint)
        body.addLayout(side, 2)
        lay.addLayout(body, 1)
        return w

    def _view_options(self, kind: Optional[str] = None) -> ViewOptions:
        def val(grp: QButtonGroup):
            b = grp.checkedButton()
            return b.property("value") if b is not None else None
        escala, _, passo = str(val(self.vg_scale) or "tms").partition(":")
        return ViewOptions(kind=kind or val(self.vg_kind) or "axial", scale=escala if escala in ("tms", "auto") else "tms",
                           step=passo_valido(passo or 0.5), zone_mm=val(self.vg_zone),
                           pupil_mm=val(self.vg_pupil), center=val(self.vg_center) or "vertice")

    def _on_view_changed(self) -> None:
        st = self.selected_state()
        if st is None:
            return
        self.fill_topo_tab(st)
        self.fill_summary_tab(st)

    def fill_topo_tab(self, st: PackageState) -> None:
        if st.exam is None:
            self.cv_topo.clear("importação inválida"); self.topo_text.clear()
            return
        vo = self._view_options()
        try:
            figure_tms_view(st.exam, st.analysis, vo, fig=self.cv_topo.figure, width_in=6.0)
            self.cv_topo.canvas.draw_idle()
        except Exception as e:  # noqa: BLE001
            self.cv_topo.clear(f"Não foi possível desenhar: {type(e).__name__}: {e}")
        lines = []
        from ..topo import ZONES_MM, tms_header_lines
        lines += tms_header_lines(st.exam) + [""]
        role = {"axial": "DIO", "tangencial": "ROC", "refrativa": "POW"}.get(vo.kind, "DIO")
        lines.append(f"Astig Stats / Power Axis ({ {'DIO': 'axial', 'ROC': 'tangencial', 'POW': 'refrativa'}[role] }):")
        for z in ZONES_MM:
            zs = zone_axis_stats(st.exam, z, role)
            if zs.rows:
                lines.append(f"  {z:.1f} mm: íngreme {zs.steep_deg:.0f}° ({zs.steep_power:.2f} D)  plano {zs.flat_deg:.0f}° ({zs.flat_power:.2f} D)  "
                             f"cil. ceratométrico {zs.steep_power - zs.flat_power:.2f} D")
                lines.append("        " + "   ".join(f"{pw:.2f}@{ang:.0f}°" for pw, ang in zs.rows if np.isfinite(pw)))
            else:
                lines.append(f"  {z:.1f} mm: sem cobertura")
        if st.analysis is not None:
            from ..topo import select_profile
            wf = select_profile(st.analysis, vo)
            if wf is not None:
                m = wf.metrics
                lines += ["", f"Perfil mostrado: {wf.profile.name} (diâmetro {wf.profile.diameter_mm:g} mm, centro {wf.profile.center_mode})",
                          f"  SA c(4,0) {m['sa_c40']:+.3f} µm   |coma| {m['coma_magnitude']:.3f}   |trefoil| {m['trefoil_magnitude']:.3f}   RMS HOA {m['rms_hoa_n3plus']:.3f} µm",
                          f"  cilindro refrativo equiv. {m.get('cilindro_equivalente_D', float('nan')):+.2f} D × {m.get('eixo_equivalente_deg', float('nan')):.0f}°"]
            else:
                lines += ["", "Nenhum perfil de frente de onda disponível para o diâmetro/centro escolhidos."]
        else:
            lines += ["", "Calcule o exame para ver a frente de onda neste diâmetro pupilar."]
        self.topo_text.setPlainText("\n".join(lines))

    @Slot()
    def print_topo_view(self):
        st = self.selected_state()
        if st is None or st.exam is None:
            return
        fig = figure_tms_view(st.exam, st.analysis, self._view_options(), width_in=7.5)
        self._print_figure(fig, f"topografia_{st.exam.identity.exam_id}_{st.exam.identity.eye or 'X'}", landscape=False)

    @Slot()
    def save_topo_view(self):
        st = self.selected_state()
        if st is None or st.exam is None:
            return
        vo = self._view_options()
        stem = f"topografia_{vo.kind}_{st.exam.identity.study_id}_{st.exam.identity.exam_id}_{st.exam.identity.eye or 'X'}"
        if self.interactive:
            path, _ = QFileDialog.getSaveFileName(self, "Salvar visualização", stem + ".png", "PNG (*.png);;PDF (*.pdf)")
            if not path:
                return
        else:
            path = os.path.join(tempfile.mkdtemp(prefix="tmsab_"), stem + ".png")
        figure_tms_view(st.exam, st.analysis, vo, width_in=7.5).savefig(path, dpi=200)
        self.statusBar().showMessage(f"Visualização salva em {path}", 10000)
        return path

    def _print_figure(self, fig, doc_name: str, landscape: bool = False) -> None:
        try:
            from PySide6.QtGui import QImage, QPageLayout, QPageSize, QPainter
            from PySide6.QtPrintSupport import QPrintDialog, QPrinter
        except Exception as e:  # noqa: BLE001
            self._msg("Imprimir", f"Impressão indisponível neste sistema: {e}", "warn"); return
        printer = QPrinter(QPrinter.HighResolution)
        printer.setPageSize(QPageSize(QPageSize.A4)); printer.setPageOrientation(QPageLayout.Landscape if landscape else QPageLayout.Portrait)
        printer.setDocName(doc_name)
        if self.interactive:
            dlg = QPrintDialog(printer, self); dlg.setWindowTitle("Imprimir")
            if dlg.exec() != QPrintDialog.Accepted:
                return
        import io
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=220)
        img = QImage.fromData(buf.getvalue())
        painter = QPainter(printer)
        rect = painter.viewport(); size = img.size(); size.scale(rect.size(), Qt.KeepAspectRatio)
        painter.setViewport(rect.x(), rect.y(), size.width(), size.height()); painter.setWindow(img.rect())
        painter.drawImage(0, 0, img); painter.end()
        self.statusBar().showMessage("Enviado para a impressora.", 8000)

    def _build_summary_tab(self) -> QWidget:
        """Página única (A4) com topografia, frente de onda, Zernike e tabelas — pronta para imprimir."""
        w = QWidget(); lay = QVBoxLayout(w)
        row = QHBoxLayout()
        self.btn_print_summary = QPushButton("Imprimir resultado completo…"); self.btn_print_summary.clicked.connect(self.print_summary)
        self.btn_save_summary = QPushButton("Salvar esta página (PDF/PNG)…"); self.btn_save_summary.clicked.connect(self.save_summary)
        self.btn_print_summary.setEnabled(False); self.btn_save_summary.setEnabled(False)
        row.addWidget(self.btn_print_summary); row.addWidget(self.btn_save_summary)
        self.summary_hint = QLabel("A mesma página é a 1ª do PDF salvo em <pasta>/analises.")
        self.summary_hint.setStyleSheet("color:#555;")
        row.addWidget(self.summary_hint, 1)
        lay.addLayout(row)
        scroll = QScrollArea(); scroll.setWidgetResizable(False); scroll.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.cv_summary = MplCanvas(width=8.27, height=11.69, dpi=88, toolbar=False)
        self.cv_summary.setFixedSize(int(8.27 * 88) + 2, int(11.69 * 88) + 2)   # tamanho da página; rolagem no restante
        scroll.setWidget(self.cv_summary)
        lay.addWidget(scroll, 1)
        return w

    def _patient_name(self, st: PackageState) -> Optional[str]:
        """Nome do XREF para o relatório impresso — só quando a opção está marcada e o exame é real."""
        if not self.with_name.isChecked() or st.exam is None or st.exam.bundle.synthetic:
            return None
        return patient_display_name(st.directory, st.exam.identity.exam_id)

    def _refresh_summary_only(self) -> None:
        st = self.selected_state()
        if st is not None:
            self.fill_summary_tab(st)

    def fill_summary_tab(self, st: PackageState) -> None:
        ok = st.analysis is not None and st.exam is not None
        self.btn_print_summary.setEnabled(ok); self.btn_save_summary.setEnabled(ok)
        if not ok:
            self.cv_summary.clear("Calcule o exame para ver o resultado completo (topografia + frente de onda + tabelas).")
            return
        try:
            figure_resultado_completo(st.exam, st.analysis, fig=self.cv_summary.figure, patient_name=self._patient_name(st), view=self._view_options())
            self.cv_summary.canvas.draw_idle()
        except Exception as e:  # noqa: BLE001
            self.cv_summary.clear(f"Não foi possível montar a página: {type(e).__name__}: {e}")

    @Slot()
    def print_summary(self):
        st = self.selected_state()
        if st is None or st.analysis is None or st.exam is None:
            return
        try:
            from PySide6.QtGui import QImage, QPageLayout, QPageSize, QPainter
            from PySide6.QtPrintSupport import QPrintDialog, QPrinter
        except Exception as e:  # noqa: BLE001
            self._msg("Imprimir", f"Impressão indisponível neste sistema: {e}", "warn"); return
        printer = QPrinter(QPrinter.HighResolution)
        printer.setPageSize(QPageSize(QPageSize.A4)); printer.setPageOrientation(QPageLayout.Portrait)
        printer.setDocName(f"resultado_completo_{st.analysis.identity.exam_id}_{st.analysis.identity.eye or 'X'}")
        if self.interactive:
            dlg = QPrintDialog(printer, self); dlg.setWindowTitle("Imprimir resultado completo")
            if dlg.exec() != QPrintDialog.Accepted:
                return
        import io
        fig = figure_resultado_completo(st.exam, st.analysis, patient_name=self._patient_name(st), view=self._view_options())
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=220)
        img = QImage.fromData(buf.getvalue())
        painter = QPainter(printer)
        rect = painter.viewport(); size = img.size(); size.scale(rect.size(), Qt.KeepAspectRatio)
        painter.setViewport(rect.x(), rect.y(), size.width(), size.height()); painter.setWindow(img.rect())
        painter.drawImage(0, 0, img); painter.end()
        self.statusBar().showMessage("Resultado completo enviado para a impressora.", 8000)

    @Slot()
    def save_summary(self):
        st = self.selected_state()
        if st is None or st.analysis is None or st.exam is None:
            return
        stem = f"resultado_completo_{st.analysis.identity.study_id}_{st.analysis.identity.exam_id}_{st.analysis.identity.eye or 'X'}"
        if self.interactive:
            path, _ = QFileDialog.getSaveFileName(self, "Salvar resultado completo", stem + ".pdf", "PDF (*.pdf);;PNG (*.png)")
            if not path:
                return
        else:
            path = os.path.join(tempfile.mkdtemp(prefix="tmsab_"), stem + ".pdf")
        fig = figure_resultado_completo(st.exam, st.analysis, patient_name=self._patient_name(st), view=self._view_options())
        fig.savefig(path, dpi=200)
        self.statusBar().showMessage(f"Resultado completo salvo em {path}", 10000)
        return path

    def _build_technical_tab(self) -> QWidget:
        self.tech_text = QPlainTextEdit(); self.tech_text.setReadOnly(True)
        f = QFont("Monospace"); f.setStyleHint(QFont.TypeWriter); f.setPointSize(9)
        self.tech_text.setFont(f)
        self.tech_text.setPlaceholderText("Parâmetros do modelo, proveniência e relatório técnico (após calcular)")
        return self.tech_text

    # ------------------------------------------------------------ utilidades
    def _msg(self, title: str, text: str, kind: str = "info") -> None:
        if not self.interactive:
            self.statusBar().showMessage(f"{title}: {text}", 8000)
            return
        if kind == "warn":
            QMessageBox.warning(self, title, text)
        elif kind == "error":
            QMessageBox.critical(self, title, text)
        else:
            QMessageBox.information(self, title, text)

    def selected_state(self) -> Optional[PackageState]:
        it = self.list.currentItem()
        if it is None:
            return None
        return self.packages.get(it.data(Qt.UserRole))

    def _refresh_list_labels(self) -> None:
        for i in range(self.list.count()):
            it = self.list.item(i)
            st = self.packages.get(it.data(Qt.UserRole))
            if st is not None:
                it.setText(st.label)
                color = {"calculado": "#1b5e20", "inválida": "#8B0000", "desatualizado": "#8a6d00", "calculando": "#1a4f8a"}.get(st.status)
                it.setForeground(QColor(color) if color else QColor("#000000"))
        self.refresh_compare_controls()

    # ------------------------------------------------------- carregamento
    @Slot()
    def open_export_folder_dialog(self):
        d = QFileDialog.getExistingDirectory(self, "Abrir pasta exportada do TMS (Data Tables de um paciente)")
        if d:
            self.open_export_folder(d)

    def open_export_folder(self, directory: str, auto: bool = True) -> list[str]:
        """Fluxo principal: pasta exportada pelo TMS (um paciente, vários exames) → um item por exame,
        cálculo automático e relatórios em <pasta>/analises (quando 'auto' estiver ligado)."""
        directory = os.path.abspath(directory)
        try:
            exams = list_exams(directory)
        except Exception as e:  # noqa: BLE001
            self._msg("Abrir pasta exportada", f"Não foi possível listar os exames: {e}", "warn")
            return []
        if not exams:
            # pasta de pacote único (v0/sintético) ou pasta com subpastas (ex.: uma subpasta por exportação)
            if any(os.path.isdir(os.path.join(directory, n)) for n in os.listdir(directory)):
                keys = self.load_folder(directory)
                if keys and auto and self.auto_compute.isChecked():
                    self.start_compute()
                return keys
            return [self.load_package(directory)]
        keys = []
        for e in exams:
            keys.append(self.load_package(directory, exam_id=e["exam_id"]))
        patients = sorted({e["patient_key"] for e in exams if e.get("patient_key")})
        if len(patients) > 1:
            self._msg(
                "Pasta com mais de um paciente",
                f"Esta pasta contém {len(exams)} exames de {len(patients)} pacientes diferentes (pelo XREF.dat).\n\n"
                "Isso acontece quando se exporta de novo para uma pasta que já tinha uma exportação: o TMS acrescenta "
                "as linhas ao XREF.dat existente.\n\nCada exame recebeu o código do seu paciente (P-…), visível na lista; "
                "os relatórios levam esse código no nome. Para evitar confusão, use uma pasta nova por paciente.",
                "warn",
            )
        if auto and self.auto_compute.isChecked():
            self.start_compute()
        return keys

    @Slot()
    def open_package_dialog(self):
        d = QFileDialog.getExistingDirectory(self, "Abrir pacote exportado do TMS")
        if d:
            self.open_export_folder(d)

    @Slot()
    def open_folder_dialog(self):
        d = QFileDialog.getExistingDirectory(self, "Abrir pasta com vários pacotes (cada subpasta = pacote)")
        if d:
            self.load_folder(d)

    def load_folder(self, folder: str) -> list[str]:
        """Cada subpasta = uma exportação do TMS (vários exames) ou um pacote único; devolve todas as chaves."""
        keys = []
        for name in sorted(os.listdir(folder)):
            sub = os.path.join(folder, name)
            if not os.path.isdir(sub) or name.lower() == "analises":
                continue
            try:
                found = list_exams(sub)
            except Exception:  # noqa: BLE001
                found = []
            if len(found) > 1:
                keys.extend(self.load_package(sub, exam_id=e["exam_id"]) for e in found)
            else:
                keys.append(self.load_package(sub))
        return keys

    @Slot()
    def generate_synthetic_dialog(self):
        d = QFileDialog.getExistingDirectory(self, "Pasta de destino dos pacotes sintéticos")
        if d:
            self.generate_synthetic(d)

    def generate_synthetic(self, out_dir: str) -> list[str]:
        from ..synthetic import standard_cases, write_bundle
        keys = []
        for spec in standard_cases():
            keys.append(self.load_package(write_bundle(spec, out_dir)))
        return keys

    def load_package(self, directory: str, exam_id: Optional[str] = None) -> str:
        directory = os.path.abspath(directory)
        if exam_id is None:
            try:
                found = list_exams(directory)
            except Exception:  # noqa: BLE001
                found = []
            if len(found) == 1:
                exam_id = found[0]["exam_id"]
            elif len(found) > 1:
                keys = [self.load_package(directory, exam_id=e["exam_id"]) for e in found]
                return keys[0]
        key = directory if exam_id is None else f"{directory}#{exam_id}"
        st = self.packages.get(key)
        if st is None:
            st = PackageState(key=key, directory=directory, exam_id=exam_id)
            self.packages[key] = st
            item = QListWidgetItem(st.label)
            item.setData(Qt.UserRole, key)
            item.setToolTip(directory)      # a pasta (que pode ter o nome do paciente) só aparece ao passar o mouse
            self.list.addItem(item)
            self.list.setCurrentItem(item)
        st.analysis = None; st.rec = None
        try:
            st.exam = import_tms_bundle(directory, self._profile_for(directory), exam_id=exam_id)
            st.status, st.message = "carregado", ""
        except AdapterError as e:
            st.exam = None
            st.status, st.message = "inválida", f"importação inválida: {e}"
        except Exception as e:  # noqa: BLE001
            st.exam = None
            st.status, st.message = "inválida", f"erro inesperado: {type(e).__name__}: {e}"
        self._refresh_list_labels()
        if self.selected_state() is st:
            self.refresh_all_tabs(st)
        return key

    def _profile_for(self, directory: str):
        return detect_profile(directory) if self.adapter_profile in (None, "", "auto") else self.adapter_profile

    @Slot()
    def remove_selected(self):
        it = self.list.currentItem()
        if it is None:
            return
        key = it.data(Qt.UserRole)
        self.packages.pop(key, None)
        self.list.takeItem(self.list.row(it))
        self._clear_compare("Selecione dois pacotes calculados e um perfil.")
        self._refresh_list_labels()
        self.on_selection_changed()

    # ------------------------------------------------------------ parâmetros
    def current_profiles(self, exam=None) -> list[OpticalProfile]:
        """Perfis pedidos; com `exam`, acrescenta o da pupila medida (se marcado e se há pupila)."""
        centers = {0: ("vertice",), 1: ("pupila",), 2: ("vertice", "pupila")}[self.center_combo.currentIndex()]
        diam = [d for cb, d in ((self.d3, 3.0), (self.d4, 4.0), (self.d5, 5.0), (self.d6, 6.0)) if cb.isChecked()]
        if self.d_other_check.isChecked():
            diam.append(float(self.d_other.value()))
        diam = sorted(set(diam))
        if not diam and not self.d_pupil.isChecked():
            return []
        if exam is not None:
            return profiles_for_exam(exam, diam, centers, int(self.order_spin.value()), measured_pupil=self.d_pupil.isChecked(),
                                     reference=self.ref_combo.currentText())
        profs = default_profiles(diam, centers, int(self.order_spin.value()))
        for p in profs:
            p.reference = self.ref_combo.currentText()
        return profs

    def _profile_request(self) -> dict:
        """Pedido de perfis independente do exame (o worker completa com a pupila medida por exame)."""
        centers = {0: ("vertice",), 1: ("pupila",), 2: ("vertice", "pupila")}[self.center_combo.currentIndex()]
        diam = [d for cb, d in ((self.d3, 3.0), (self.d4, 4.0), (self.d5, 5.0), (self.d6, 6.0)) if cb.isChecked()]
        if self.d_other_check.isChecked():
            diam.append(float(self.d_other.value()))
        return {"diameters": sorted(set(diam)), "centers": centers, "order": int(self.order_spin.value()),
                "measured_pupil": self.d_pupil.isChecked(), "reference": self.ref_combo.currentText()}

    def current_recon(self) -> ReconstructionConfig:
        return ReconstructionConfig(method=self.method_combo.currentText(), surface_order=int(self.surf_order_spin.value()))

    def set_default_params(self) -> None:
        self._params_block = True
        try:
            self.center_combo.setCurrentIndex(2); self.d3.setChecked(True); self.d4.setChecked(True); self.d5.setChecked(True); self.d6.setChecked(True); self.d_pupil.setChecked(True)
            self.d_other_check.setChecked(False); self.order_spin.setValue(6)
            self.method_combo.setCurrentIndex(0); self.surf_order_spin.setValue(10); self.ref_combo.setCurrentIndex(0)
        finally:
            self._params_block = False

    @Slot()
    def on_params_changed(self):
        if self._params_block:
            return
        changed = False
        for st in self.packages.values():
            if st.analysis is not None:
                st.analysis = None; st.rec = None
                st.status, st.message = "desatualizado", "parâmetros alterados — recalcular para ver resultados"
                changed = True
        if changed:
            self._refresh_list_labels()
            self._clear_compare("Resultados desatualizados — recalcular e comparar novamente.")
            st = self.selected_state()
            if st is not None:
                self.refresh_all_tabs(st)

    # --------------------------------------------------------------- cálculo
    @Slot()
    def start_compute(self):
        if self.worker is not None and self.worker.isRunning():
            return
        profiles = self.current_profiles()
        if not profiles:
            self._msg("Perfil de análise", "Selecione pelo menos um diâmetro.", "warn")
            return
        jobs = [(k, st.directory, st.exam_id, self._profile_for(st.directory)) for k, st in self.packages.items() if st.status != "calculado"]
        if not jobs:
            self._msg("Calcular", "Nenhum pacote pendente (todos calculados com os parâmetros atuais).")
            return
        for job in jobs:
            self.packages[job[0]].status = "calculando"
        self._refresh_list_labels()
        self.worker = AnalysisWorker(jobs, profiles, self.current_recon(), QualityConfig(), parent=self,
                                     profile_request=self._profile_request())
        self.worker.progress.connect(self.on_progress)
        self.worker.package_done.connect(self.on_package_done)
        self.worker.package_failed.connect(self.on_package_failed)
        self.worker.finished_all.connect(self.on_finished_all)
        self.btn_compute.setEnabled(False); self.btn_cancel.setEnabled(True)
        self._set_params_enabled(False)
        self.worker.start()

    def _set_params_enabled(self, on: bool) -> None:
        for x in (self.center_combo, self.d3, self.d4, self.d5, self.d6, self.d_pupil, self.d_other_check, self.order_spin, self.method_combo,
                  self.surf_order_spin, self.ref_combo):
            x.setEnabled(on)
        self.d_other.setEnabled(on and self.d_other_check.isChecked())

    @Slot()
    def cancel_compute(self):
        if self.worker is not None:
            self.worker.cancel()
            self.progress_label.setText("cancelando…")

    @Slot(int, str)
    def on_progress(self, pct: int, msg: str):
        self.progress.setValue(pct)
        self.progress_label.setText(msg)

    @Slot(str, object, object, object)
    def on_package_done(self, key: str, exam, analysis, rec):
        st = self.packages.get(key)
        if st is None:
            return
        st.exam, st.analysis, st.rec = exam, analysis, rec
        st.status, st.message = "calculado", ""
        st.profiles_used = [w.profile for w in analysis.wavefronts]
        self._refresh_list_labels()
        if self.selected_state() is st:
            self.refresh_all_tabs(st)

    @Slot(str, str)
    def on_package_failed(self, key: str, msg: str):
        st = self.packages.get(key)
        if st is None:
            return
        st.analysis = None; st.rec = None
        st.status, st.message = "inválida", msg
        self._refresh_list_labels()
        if self.selected_state() is st:
            self.refresh_all_tabs(st)

    @Slot(bool)
    def on_finished_all(self, cancelled: bool):
        for st in self.packages.values():
            if st.status == "calculando":
                st.status, st.message = ("carregado", "cálculo cancelado") if cancelled else ("inválida", "sem resultado")
        self._refresh_list_labels()
        self.btn_compute.setEnabled(True); self.btn_cancel.setEnabled(False)
        self._set_params_enabled(True)
        self.worker = None
        st = self.selected_state()
        if st is not None:
            self.refresh_all_tabs(st)
        self._auto_compare()
        if not cancelled and self.auto_export.isChecked():
            self._auto_export_reports()

    def _auto_export_reports(self) -> None:
        """Salva JSON/CSV/PDF de cada exame calculado em <pasta exportada>/analises/ (sem dados nominais nos arquivos)."""
        n = 0
        for st in self.packages.values():
            if st.analysis is None or st.exam is None or st.exam.bundle.synthetic:
                continue
            out = os.path.join(st.directory, "analises")
            try:
                self.export_package(st, out)
                n += 1
            except Exception as e:  # noqa: BLE001
                self.statusBar().showMessage(f"Falha ao salvar relatórios em {out}: {e}", 15000)
        if n:
            self.statusBar().showMessage(f"{n} relatório(s) salvos automaticamente na subpasta 'analises' da pasta exportada", 15000)

    def _auto_compare(self) -> None:
        """Refaz a comparação automaticamente quando os dois pacotes selecionados têm análise atual."""
        try:
            self.refresh_compare_controls()
            ka, kb = self.cmp_a.currentData(), self.cmp_b.currentData()
            if ka and kb and ka != kb and self.packages[ka].analysis is not None and self.packages[kb].analysis is not None:
                self.run_compare()
        except Exception:   # a comparação nunca pode derrubar a janela
            pass

    def compute_sync(self, keys: Optional[list[str]] = None, profiles: Optional[list[OpticalProfile]] = None) -> None:
        """Cálculo síncrono (sem thread) — usado pelo teste de fumaça."""
        for key in (keys or list(self.packages)):
            st = self.packages[key]
            if st.exam is None:
                continue
            try:
                analysis, rec = run_analysis(st.exam, profiles or self.current_profiles(st.exam), self.current_recon(), QualityConfig())
            except (ValueError, AdapterError) as e:
                self.on_package_failed(key, f"análise inválida: {e}")
                continue
            self.on_package_done(key, st.exam, analysis, rec)
        self._auto_compare()

    # ----------------------------------------------------------- exibição
    @Slot()
    def on_selection_changed(self):
        st = self.selected_state()
        if st is None:
            self._show_no_selection()
        else:
            self.refresh_all_tabs(st)

    def _show_no_selection(self):
        self.import_state.setText("Nenhum pacote selecionado")
        self.import_diag.clear(); self.files_table.setRowCount(0); self.manifest_text.clear(); self.pupil_label.setText("Pupila: —")
        for c in (self.cv_elev, self.cv_cov, self.cv_w, self.cv_wfit, self.cv_res, self.cv_zbars):
            c.clear("sem pacote selecionado")
        self.geom_text.clear(); self.tech_text.clear()
        self.profile_combo.blockSignals(True); self.profile_combo.clear(); self.profile_combo.blockSignals(False)
        self.wf_state.setText(""); self.block_label.hide(); self.wf_notes.setText("")
        for b in self.bn.values():
            b.set_value(None)
        self.z_table.setRowCount(0)
        self._clear_keratometry_tab("sem pacote selecionado")

    def refresh_all_tabs(self, st: PackageState) -> None:
        self.fill_summary_tab(st)
        self.fill_topo_tab(st)
        self.fill_keratometry_tab(st)
        self.fill_import_tab(st)
        self.fill_geometry_tab(st)
        self.fill_wavefront_tab(st)
        self.fill_technical_tab(st)
        self.refresh_compare_controls()

    def fill_import_tab(self, st: PackageState) -> None:
        if st.exam is None:
            self.import_state.setText(f"IMPORTAÇÃO INVÁLIDA — {st.message}")
            self.import_state.setStyleSheet("color:#8B0000;")
            self.import_diag.setPlainText(st.message)
            self.files_table.setRowCount(0); self.manifest_text.clear(); self.pupil_label.setText("Pupila: —")
            return
        ex = st.exam
        prefix = "SINTÉTICO — " if ex.bundle.synthetic else ""
        self.import_state.setText(f"{prefix}Importação válida — exame {ex.identity.exam_id}, olho {ex.identity.eye or 'não informado'}, "
                                  f"perfil {ex.bundle.adapter_profile} — estado: {st.status}" + (f" ({st.message})" if st.message else ""))
        self.import_state.setStyleSheet("color:#1b5e20;" if st.status != "inválida" else "color:#8B0000;")
        lines = ["Diagnósticos:"] + [f"  • {d}" for d in ex.diagnostics]
        if ex.warnings:
            lines += ["", "Avisos:"] + [f"  ! {w}" for w in ex.warnings]
        if st.message and st.status == "inválida":
            lines += ["", f"Erro: {st.message}"]
        self.import_diag.setPlainText("\n".join(lines))
        self.files_table.setRowCount(0)
        for f in ex.bundle.files:
            r = self.files_table.rowCount(); self.files_table.insertRow(r)
            dims = f"{f.n_rows} × {f.n_cols}" if f.n_rows else "—"
            for c, v in enumerate([f.role, os.path.basename(f.path), f.sha256[:12] + "…", dims, f.orientation or "—"]):
                self.files_table.setItem(r, c, QTableWidgetItem(v))
        self.manifest_text.setPlainText(json.dumps(ex.bundle.manifest, ensure_ascii=False, indent=2) if ex.bundle.manifest else "(sem manifesto)")
        p = ex.pupil
        if p.state == "registrada":
            diam = f"{p.diameter:.2f} mm" if p.diameter is not None and np.isfinite(p.diameter) else "—"
            cm = chord_mu_aparente(p, ex.identity.eye)
            self.pupil_label.setText(f"Pupila: registrada\ncentro ({p.center_x:+.3f}, {p.center_y:+.3f}) mm\n"
                                     f"diâmetro {diam}\nchord mu aparente {fmt_chord_mu(cm, curto=True)}\n"
                                     f"referencial {p.frame}\nmétodo {p.method}")
        else:
            self.pupil_label.setText(f"Pupila: {p.state} — perfis centrados na pupila ficam bloqueados (a pupila não é preenchida com o vértice)")

    # ---- ceratometria por zona
    def _clear_keratometry_tab(self, canvas_msg: str, state: str = "", color: str = "#555") -> None:
        """Sem números antigos: sem pacote, sem cálculo, desatualizado ou inválido."""
        self._k_ident = ""
        self.k_state.setText(state)
        self.k_state.setStyleSheet(f"color:{color};")
        for b in self.k_bn.values():
            b.set_value(None)
            b.setToolTip("")
        self.k_table.clearSpans()
        self.k_table.setRowCount(0)
        self.k_notes.clear()
        self.btn_k_copy.setEnabled(False)
        self._k_canvas_msg(canvas_msg)
        self._k_limpar_controle(canvas_msg)

    def fill_keratometry_tab(self, st: PackageState) -> None:
        if st.exam is None:
            self._clear_keratometry_tab("importação inválida", f"IMPORTAÇÃO INVÁLIDA — {st.message}", "#8B0000")
            return
        if st.analysis is None:
            if st.status == "desatualizado":
                self._clear_keratometry_tab("resultados desatualizados — recalcular",
                                            "DESATUALIZADO — parâmetros alterados; recalcular", "#8a6d00")
            elif st.status == "inválida":
                self._clear_keratometry_tab("sem resultado", f"SEM RESULTADO — {st.message}", "#8B0000")
            else:
                self._clear_keratometry_tab("sem cálculo", f"Sem cálculo ({st.status}{': ' + st.message if st.message else ''}) "
                                            "— a ceratometria aparece depois de 'Calcular'.")
            return
        kres = st.analysis.keratometry
        if kres is None:
            self._clear_keratometry_tab("ceratometria não calculada", "Ceratometria ausente nesta análise — recalcular.", "#8a6d00")
            return
        linhas = linhas_resumo(kres, incluir_aneis=True)
        zonas = _zonas_das_linhas(kres, linhas)
        qc = st.analysis.quality_config
        ident = st.label.split(" — ")[0]          # exame, olho, captura e código de estudo (nunca o nome)
        self._k_ident = f"{ident} · análise {st.analysis.analysis_id[:12]}"
        alertas = list(getattr(kres, "alertas", None) or [])
        if any("NÃO confere" in c for c in kres.conferencia):
            self.k_state.setText(f"{ident} — ATENÇÃO: o SimK recalculado não confere com o XREF; verificar a leitura antes de usar os números")
            self.k_state.setStyleSheet("color:#8B0000;")
        elif not any(ln["disponivel"] for ln in linhas if ln["tipo"] != "simk_tms"):
            motivo = next((n for n in kres.notes if n.startswith("ceratometria indisponível")), "nenhuma zona disponível")
            self.k_state.setText(f"{ident} — {motivo} (motivos abaixo da tabela)")
            self.k_state.setStyleSheet("color:#8B0000;")
        elif alertas:                              # informativo (ex.: astigmatismo dependente da zona), não bloqueio
            self.k_state.setText(f"{ident} — " + "; ".join(alertas))
            self.k_state.setStyleSheet("color:#8a6d00;")
        else:
            self.k_state.setText(f"{ident} — ceratometria por zona: circunferências e anéis centrados no vértice; "
                                 "zona pupilar = área dentro do contorno pupilar do TMS")
            self.k_state.setStyleSheet("color:#1b5e20;")
        self._fill_k_numbers(st.analysis, kres, linhas)
        self._fill_k_table(linhas, zonas, qc)
        self._k_ajustar_divisao()
        self._fill_k_notes(kres, linhas)
        self.btn_k_copy.setEnabled(self.k_table.rowCount() > 0)
        try:
            self._k_preparar_controle(st)           # faixa e diâmetro antes dos gráficos (a linha do escolhido)
        except Exception as e:  # noqa: BLE001
            self._k_limpar_controle(f"controle de diâmetro indisponível: {type(e).__name__}: {e}")
        try:
            self._plot_keratometry(st.analysis, kres, qc)
        except Exception as e:  # noqa: BLE001
            self._k_canvas_msg(f"erro ao desenhar: {type(e).__name__}: {e}")
        if self._k_st is not None:
            try:
                self._k_desenhar_mapa(st.exam)
            except Exception as e:  # noqa: BLE001
                self._k_mapa_msg(f"erro ao desenhar o mapa: {type(e).__name__}: {e}")
            self._k_atualizar()

    def _fill_k_numbers(self, analysis: Analysis, kres: KeratometryResult, linhas: list[dict]) -> None:
        lp = next((ln for ln in linhas if ln["destaque"] == "pupila"), None)
        if lp is not None and lp["disponivel"]:
            t = lp["texto"]
            sub = [f"ΔK {t[4]} D", f"Km {t[5]} D"]
            if kres.pupila.eixo_pouco_definido:
                sub.append("* eixo pouco definido")
            if kres.pupila.marca_interpolados:
                sub.append("† pontos interpolados")
            self.k_bn["pupila"].set_value(f"{t[0]} @ {t[1]} / {t[2]} @ {t[3]}", " · ".join(sub))
        else:                                       # rótulo curto; o motivo completo fica na dica e nas notas
            self.k_bn["pupila"].set_value(None, lp["rotulo"] if lp is not None else "zona pupilar não calculada")
        self.k_bn["pupila"].setToolTip("" if lp is None or lp["disponivel"] else (lp["motivo"] or lp["validity"]))
        ls = next((ln for ln in linhas if ln["destaque"] == "referencia"), None)
        if ls is not None:
            t = ls["texto"]
            self.k_bn["simk"].set_value(f"{t[0]} @ {t[1]} / {t[2]} @ {t[3]}", f"ΔK {t[4]} D (CYL) · no XREF: SimK2 = plano, SimK1 = curvo")
        else:
            self.k_bn["simk"].set_value(None, "sem índices do TMS (XREF) neste pacote")
        p = analysis.pupil
        if p.state == "registrada" and p.diameter is not None and np.isfinite(p.diameter):
            sub = "pupila de entrada, sob a iluminação do Plácido"
            cm = getattr(analysis, "chord_mu", None)
            if cm is not None:                    # subtexto curto: cabe em 2 linhas na janela padrão
                sub = f"chord mu aparente {fmt_chord_mu(cm, curto=True)}"
            elif p.center_x is not None and p.center_y is not None:
                sub = f"centro ({p.center_x:+.2f}, {p.center_y:+.2f}) mm do vértice · " + sub
            self.k_bn["diametro"].set_value(f"{p.diameter:.2f} mm", sub)
        else:
            estado = {"ausente": "não registrada", "nao_verificada": "não verificada"}.get(p.state, p.state)
            self.k_bn["diametro"].set_value(None, f"pupila {estado} nesta captura")

    def _fill_k_table(self, linhas: list[dict], zonas: list, qc: QualityConfig) -> None:
        t = self.k_table
        t.clearSpans()
        t.setRowCount(0)
        cinza, cinza_claro = QColor("#6b6b6b"), QColor("#8a8a8a")
        separador = False
        linhas_longas = []
        for ln, z in zip(linhas, zonas):
            if ln["tipo"] == "anel" and not separador:
                r = t.rowCount(); t.insertRow(r)
                it = QTableWidgetItem(K_SEPARADOR_ANEIS)
                it.setFlags(Qt.ItemIsEnabled)
                fi = it.font(); fi.setItalic(True); it.setFont(fi)
                it.setBackground(QColor("#f0efec")); it.setForeground(QColor(K_TINTA))
                t.setItem(r, 0, it)
                t.setSpan(r, 0, 1, t.columnCount())
                separador = True
            extras = ["—"] * 3
            if z is not None and ln["disponivel"]:
                extras = [f"{100.0 * z.fracao_interpolados:.0f}%", _fmt_j(z.j0_D), _fmt_j(z.j45_D)]
            if not ln["disponivel"]:
                dica = "indisponível: " + (ln["motivo"] or ln["validity"])
            elif ln["destaque"] == "referencia":
                dica = ("SimK lido do XREF do TMS (no aparelho SimK1 = curvo e SimK2 = plano; aqui K1 = plano, K2 = curvo); "
                        "ΔK = CYL do TMS; não recalculado aqui (a conferência fica abaixo da tabela)")
            else:
                dica = "; ".join(z.notes) if z is not None else ""
            r = t.rowCount(); t.insertRow(r)
            for col, v in enumerate([ln["rotulo"]] + list(ln["texto"]) + extras):
                it = QTableWidgetItem(v)
                if col > 0:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if dica:
                    it.setToolTip(dica)
                if ln["destaque"] == "pupila":
                    it.setBackground(QColor("#dbe9f8"))
                    fb = it.font(); fb.setBold(True); it.setFont(fb)
                elif ln["destaque"] == "zona":         # zonas 0–d (média da área), à parte das circunferências
                    it.setBackground(QColor("#eef4fb"))
                if not ln["disponivel"]:
                    it.setForeground(cinza_claro)
                elif ln["destaque"] == "referencia":
                    it.setForeground(cinza)
                elif z is not None and z.eixo_pouco_definido and col in (2, 4):
                    it.setForeground(cinza_claro)
                    it.setToolTip(f"eixo pouco definido: ΔK < {qc.k_axis_min_cyl_D:.2f} D (o meridiano varia com a zona e entre capturas)")
                t.setItem(r, col, it)
            if ln["tipo"] == "simk_tms":
                linhas_longas.append(r)
        # coluna Zona na largura dos rótulos (menos o do SimK, que quebra em duas linhas): as 10 colunas cabem ao lado dos gráficos
        fm, negrito = QFontMetrics(t.font()), QFont(t.font())
        negrito.setBold(True)
        fmb = QFontMetrics(negrito)
        larg = [(fmb if ln["destaque"] == "pupila" else fm).horizontalAdvance(ln["rotulo"]) for ln in linhas if ln["tipo"] != "simk_tms"]
        t.setColumnWidth(0, max(larg + [fm.horizontalAdvance("anéis medidos")]) + 16)
        for r in linhas_longas:
            t.resizeRowToContents(r)

    def _fill_k_notes(self, kres: KeratometryResult, linhas: list[dict]) -> None:
        alertas = list(getattr(kres, "alertas", None) or [])
        out = ["Alertas (informativos, não são recomendação):"] + [f"  • {a}" for a in alertas] if alertas else []
        out += ["Conferência com o TMS:"]
        out += [f"  • {c}" for c in kres.conferencia] or ["  • sem SimK do TMS (XREF) neste pacote — nada a conferir"]
        if kres.pupila is not None and kres.pupila.validity == "disponivel" and kres.pupila.notes:
            out += ["Zona pupilar:"] + [f"  • {n}" for n in kres.pupila.notes]
        por_motivo: dict[str, list[str]] = {}          # o mesmo motivo em várias linhas vira uma linha só
        for ln in linhas:
            if not ln["disponivel"]:
                por_motivo.setdefault(ln["motivo"] or ln["validity"], []).append(ln["rotulo"])
        if por_motivo:
            out += ["Linhas indisponíveis (ausência, não zero):"] + [f"  • {', '.join(r)}: {m}" for m, r in por_motivo.items()]
        notas = [n for n in kres.notes if n not in alertas]     # análises antigas gravavam o alerta também nas notas
        if notas:
            out += ["Notas:"] + [f"  • {n}" for n in notas]
        self.k_notes.setPlainText("\n".join(out))

    def _k_canvas_msg(self, msg: str) -> None:
        """Mensagem no lugar dos gráficos; antes devolve a figura ao leiaute simples (o MplCanvas usa tight_layout)."""
        self._k_marcas = []
        self.cv_kerato.figure.set_layout_engine("none")
        self.cv_kerato.clear(msg)

    def _plot_keratometry(self, analysis: Analysis, kres: KeratometryResult, qc: QualityConfig) -> None:
        """K1/K2 × diâmetro e eixo do K2 × diâmetro (circunferências e anéis; zonas de área ficam só na tabela).
        Só números do núcleo; cores K1 = azul, K2 = laranja nos dois gráficos; legenda única embaixo."""
        def serie(zs, attr):
            x = np.array([z.diametro_mm if z.diametro_mm is not None else np.nan for z in zs], dtype=float)
            y = np.array([getattr(z, attr) if z.validity == "disponivel" and getattr(z, attr) is not None else np.nan
                          for z in zs], dtype=float)
            return x, y

        ks = np.concatenate([serie(zs, a)[1] for zs in (kres.zonas, kres.aneis) for a in ("k1_plano_D", "k2_curvo_D")])
        if not np.isfinite(ks).any():
            self._k_canvas_msg("nenhuma circunferência ou anel disponível para o gráfico (motivos abaixo da tabela)")
            return
        fig = self.cv_kerato.figure
        fig.clear()
        self._k_marcas = []
        fig.set_layout_engine("constrained", h_pad=0.04, w_pad=0.04)   # refaz o leiaute a cada redimensionamento
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
        for attr, cor in (("k1_plano_D", K_COR_K1), ("k2_curvo_D", K_COR_K2)):
            x, y = serie(kres.zonas, attr)
            ax1.plot(x, y, "-o", color=cor, lw=1.6, ms=5.5, mec="white", mew=0.9, zorder=3)
            xa, ya = serie(kres.aneis, attr)
            ax1.plot(xa, ya, linestyle="none", marker="o", ms=3.2, color=cor, alpha=0.55, mew=0, zorder=2)
        # eixo do K2: vazado = eixo pouco definido (ΔK baixo); sem linha ligando (o eixo dá a volta em 180°)
        for zs, grande in ((kres.zonas, True), (kres.aneis, False)):
            disp = [z for z in zs if z.validity == "disponivel" and z.eixo_k2_deg is not None and z.diametro_mm is not None]
            for pouco in (False, True):
                sel = [z for z in disp if bool(z.eixo_pouco_definido) == pouco]
                if not sel:
                    continue
                ax2.plot([z.diametro_mm for z in sel], [z.eixo_k2_deg for z in sel], linestyle="none", marker="o",
                         ms=5.5 if grande else 3.8, mfc="white" if pouco else K_COR_K2,
                         mec=K_COR_K2 if (pouco or not grande) else "white", mew=1.0 if pouco else (0.9 if grande else 0.0),
                         alpha=1.0 if grande else 0.6, zorder=3 if grande else 2)
        t = kres.simk_tms or {}
        tem_simk = {"SimK1", "SimK2"} <= set(t)
        if tem_simk:
            ds = _diametro_simk(kres)
            ax1.plot([ds, ds], [t["SimK2"], t["SimK1"]], linestyle="none", marker="D", ms=5, color=K_TINTA, mec="white", mew=0.8, zorder=4)
            if t.get("SimK1 Ang.") is not None:
                ax2.plot([ds], [eixo_tabo(t["SimK1 Ang."])], linestyle="none", marker="D", ms=5, color=K_TINTA, mec="white", mew=0.8, zorder=4)
            ks = np.concatenate([ks, [t["SimK1"], t["SimK2"]]])
        diams = [z.diametro_mm for z in list(kres.zonas) + list(kres.aneis) if z.diametro_mm]
        xmax = max(9.0, (max(diams) + 0.3) if diams else 9.0)
        dpup = kres.pupila.diametro_mm if kres.pupila is not None else None
        if analysis.pupil.state == "registrada" and dpup and np.isfinite(dpup):
            for ax in (ax1, ax2):
                ax.axvline(dpup, color=K_TINTA, lw=1.0, ls=(0, (4, 3)), zorder=1)
            a_direita = dpup < 0.7 * xmax
            ax1.text(dpup + (0.1 if a_direita else -0.1), 0.97, f"pupila Ø {dpup:.2f} mm", transform=ax1.get_xaxis_transform(),
                     ha="left" if a_direita else "right", va="top", fontsize=7.5, color=K_TINTA)
        if self._k_st is not None:      # diâmetro do controle (sub-aba do mapa): traço-ponto, movido em _k_atualizar
            d = self.k_slider.value() / 10.0
            self._k_marcas = [ax.axvline(d, color=K_COR_ESCOLHIDO, lw=1.2, ls="-.", zorder=2.5) for ax in (ax1, ax2)]
        k = ks[np.isfinite(ks)]
        meio, meia = 0.5 * (k.min() + k.max()), 0.5 * max(k.max() - k.min(), 1.0) * 1.15   # escala mínima de 1 D
        ax1.set_ylim(meio - meia, meio + meia)
        ax1.set_xlim(0.0, xmax)
        ax2.set_ylim(0.0, 180.0)
        ax2.set_yticks([0, 45, 90, 135, 180])
        ax1.set_title("K1 plano e K2 curvo por diâmetro", fontsize=9, color="#0b0b0b", loc="left")
        ax2.set_title("Eixo do K2 (meridiano mais curvo)", fontsize=9, color="#0b0b0b", loc="left")
        ax1.set_ylabel("potência axial (D)", fontsize=8, color=K_TINTA)
        ax2.set_ylabel("eixo do K2 (° TABO)", fontsize=8, color=K_TINTA)
        ax2.set_xlabel("diâmetro (mm): circunferência centrada no vértice ou anel (Ø medido)", fontsize=8, color=K_TINTA)
        ax1.tick_params(labelbottom=False)
        for ax in (ax1, ax2):
            _estilo_eixos_k(ax)
        alcas = [Line2D([], [], color=K_COR_K1, lw=1.6, marker="o", ms=5, mec="white", label="K1 plano"),
                 Line2D([], [], color=K_COR_K2, lw=1.6, marker="o", ms=5, mec="white", label="K2 curvo"),
                 Line2D([], [], color=K_TINTA_FRACA, ls="none", marker="o", ms=3.2, mew=0, label="anel (Ø medido)"),
                 Line2D([], [], color=K_TINTA_FRACA, ls="none", marker="o", ms=5, mfc="white", mew=1.0,
                        label=f"eixo pouco definido (ΔK < {qc.k_axis_min_cyl_D:.2f} D)")]
        if tem_simk:
            alcas.append(Line2D([], [], color=K_TINTA, ls="none", marker="D", ms=4.5, mec="white", label="SimK do TMS"))
        if self._k_marcas:
            alcas.append(Line2D([], [], color=K_COR_ESCOLHIDO, lw=1.2, ls="-.", label="diâmetro escolhido"))
        fig.legend(handles=alcas, loc="outside lower center", ncol=2, fontsize=7, frameon=False, labelcolor=K_TINTA,
                   handletextpad=0.4, columnspacing=1.0)
        self.cv_kerato.canvas.draw_idle()

    # ---- controle de diâmetro (sub-aba "Mapa axial — diâmetro")
    @Slot(int, int)
    def _k_divisao_movida(self, _pos: int, _indice: int) -> None:
        self._k_divisao_manual = True

    def _k_ajustar_divisao(self) -> None:
        """Tabela na largura do conteúdo (todas as colunas, sem barra horizontal com a fonte do sistema) e o resto para o
        mapa — até o usuário mover a divisória."""
        if self._k_divisao_manual:
            return
        t = self.k_table
        larg = t.horizontalHeader().length() + t.verticalScrollBar().sizeHint().width() + 2 * t.frameWidth()
        tam = self.k_split.sizes()
        if larg > 0 and len(tam) == 2 and tam[0] != larg:
            self.k_split.setSizes([larg, max(sum(tam) - larg, 1)])

    def _k_sem_mapa(self) -> None:
        """Esquece os artistas do mapa (figura limpa ou com mensagem): cliques e atualizações passam a ser ignorados."""
        self._k_ax = None
        self._k_lim = 5.0
        self._k_circulo = None
        self._k_vertice = None
        self._k_merid: dict = {}
        self._k_rot: dict = {}

    def _k_mapa_msg(self, msg: str) -> None:
        self._k_sem_mapa()
        self.cv_kmapa.figure.set_layout_engine("none")
        self.cv_kmapa.clear(msg)

    def _k_limpar_controle(self, msg: str) -> None:
        """Sem números antigos: controle desabilitado, mapa com a mensagem, leitura vazia (as linhas dos gráficos já
        saíram com a figura)."""
        self._k_st = None
        self._k_cache.clear()
        self._k_arrastando = False
        for x in (self.k_slider, self.k_spin):
            x.setEnabled(False)
        self.k_leitura.setText("")
        self._k_mapa_msg(msg)

    def _k_preparar_controle(self, st: PackageState) -> None:
        """Faixa do exame (décimos exatos) e o diâmetro escolhido, limitado a ela; sem disparar atualizações."""
        if any(chave[0] != st.analysis.analysis_id for chave in self._k_cache):
            self._k_cache.clear()                   # outro exame ou outra análise: o cache só guarda a atual
        self._k_st = st
        dmin, dmax = faixa_do_controle(st.exam)
        lo, hi = int(round(dmin * 10)), int(round(dmax * 10))
        dec = min(max(self._k_decimos, lo), hi)
        for x in (self.k_slider, self.k_spin):
            x.blockSignals(True)
        try:
            self.k_slider.setRange(lo, hi); self.k_spin.setRange(lo / 10.0, hi / 10.0)
            self.k_slider.setValue(dec); self.k_spin.setValue(dec / 10.0)
        finally:
            for x in (self.k_slider, self.k_spin):
                x.blockSignals(False)
                x.setEnabled(True)

    @Slot(int)
    def _k_slider_mudou(self, dec: int) -> None:
        self.k_spin.blockSignals(True)
        self.k_spin.setValue(dec / 10.0)
        self.k_spin.blockSignals(False)
        self._k_decimos = dec
        self._k_atualizar()

    @Slot(float)
    def _k_spin_mudou(self, v: float) -> None:
        dec = int(round(v * 10.0))
        self.k_slider.blockSignals(True)
        self.k_slider.setValue(dec)
        self.k_slider.blockSignals(False)
        self._k_decimos = self.k_slider.value()
        self._k_atualizar()

    def _k_zonas_no_diametro(self, st: PackageState, dec: int) -> tuple[KeratometryZone, KeratometryZone]:
        """(circunferência Ø d, zona 0–d) com a configuração da análise (os mesmos números da tabela), em cache."""
        a = st.analysis
        chave = (a.analysis_id, dec)
        if chave not in self._k_cache:
            self._k_cache[chave] = ceratometria_no_diametro(st.exam, dec / 10.0, a.keratometry.config, a.quality_config)
        return self._k_cache[chave]

    @staticmethod
    def _k_linha_html(z: KeratometryZone, descricao: str) -> str:
        """Uma linha da leitura com o texto de linha_da_zona (a formatação da tabela); indisponível = motivo, nunca zero."""
        ln = linha_da_zona(z)
        rot = html.escape(ln["rotulo"])
        if not ln["disponivel"]:
            motivo = html.escape(ln["motivo"] or ln["validity"])
            estado = "inválida" if ln["validity"] == "invalida" else "indisponível"
            return f'<span style="color:#7a7a7a;"><b>{rot}</b>: {estado} — {motivo}</span>'
        t = ln["texto"]
        juntos = lambda s: s.replace(" ", "&nbsp;")       # noqa: E731  (a linha só quebra depois de um " · ")
        termos = [f'<span style="color:{K_TEXTO_K1};">{juntos(f"K1 plano {t[0]} D @ {t[1]}")}</span>',
                  f'<span style="color:{K_TEXTO_K2};">{juntos(f"K2 curvo {t[2]} D @ {t[3]}")}</span>',
                  juntos(f"ΔK {t[4]} D"), juntos(f"Km {t[5]} D")]
        return f"<b>{juntos(rot)}</b> ({descricao}): " + "&nbsp;· ".join(termos)

    def _k_leitura_html(self, zc: KeratometryZone, zd: KeratometryZone, qc: QualityConfig) -> str:
        linhas = [self._k_linha_html(zc, "circunferência no vértice"), self._k_linha_html(zd, "média pela área")]
        disp = [z for z in (zc, zd) if z.validity == "disponivel"]
        marcas = []
        if any(z.marca_interpolados for z in disp):
            marcas.append("† pontos interpolados pelo aparelho")
        if any(z.eixo_pouco_definido for z in disp):
            marcas.append(f"* eixo pouco definido (ΔK &lt; {qc.k_axis_min_cyl_D:.2f} D)")
        if marcas:
            linhas.append(f'<span style="color:#6b6b6b; font-size:10px;">{"; ".join(marcas)}</span>')
        return "<br>".join(linhas)

    def _k_atualizar(self) -> None:
        """Diâmetro escolhido -> leitura, circunferência e meridianos no mapa, linha nos gráficos (nunca derruba a janela)."""
        st = self._k_st
        if st is None or st.exam is None or st.analysis is None or st.analysis.keratometry is None:
            return
        dec = self.k_slider.value()
        d = dec / 10.0
        zc = None
        try:
            zc, zd = self._k_zonas_no_diametro(st, dec)
            self.k_leitura.setText(self._k_leitura_html(zc, zd, st.analysis.quality_config))
        except Exception as e:  # noqa: BLE001
            self.k_leitura.setText(f'<span style="color:#8B0000;">Ø {d:.1f} mm: erro no cálculo — '
                                   f'{html.escape(type(e).__name__)}: {html.escape(str(e))}</span>')
        try:
            self._k_desenhar_sobreposicao(d, zc)
        except Exception as e:  # noqa: BLE001
            self._k_mapa_msg(f"erro ao desenhar o mapa: {type(e).__name__}: {e}")
        try:
            for m in self._k_marcas:
                m.set_xdata([d, d])
            if self._k_marcas and self.cv_kerato.isVisible():     # oculta: redesenha ao trocar de sub-aba
                self.cv_kerato.canvas.draw_idle()
        except Exception as e:  # noqa: BLE001
            self._k_canvas_msg(f"erro ao desenhar: {type(e).__name__}: {e}")

    @Slot(int)
    def _k_subaba_mudou(self, i: int) -> None:
        """Os gráficos não são redesenhados enquanto ocultos (arrastar no mapa fica leve): redesenha ao mostrá-los."""
        if self.k_subtabs.widget(i) is self.cv_kerato and self._k_marcas:
            try:
                self.cv_kerato.canvas.draw()
            except Exception as e:  # noqa: BLE001
                self._k_canvas_msg(f"erro ao desenhar: {type(e).__name__}: {e}")

    def _k_desenhar_mapa(self, exam: NormalizedExam) -> None:
        """Mapa axial (DIO) uma vez por exame, com os artistas do controle; cada diâmetro só os atualiza."""
        fig = self.cv_kmapa.figure
        fig.clear()
        self._k_sem_mapa()
        fig.set_layout_engine("constrained", h_pad=0.02, w_pad=0.02)
        ax = fig.add_subplot(111)
        pm = plot_topo_map(ax, exam, "DIO", title="Potência axial (D)", zones=(), pupil=True, colorbar=True, fontsize=8.5)
        if self.cv_kmapa.toolbar is not None:
            self.cv_kmapa.toolbar.update()          # "início" do zoom volta a este mapa
        if pm is None:                              # sem a tabela DIO: o mapa diz "ausente"; a leitura traz o motivo
            fig.set_layout_engine("none")
            self.cv_kmapa.canvas.draw_idle()
            return
        ax.tick_params(labelsize=7)
        self._k_lim = float(ax.get_xlim()[1])
        self._k_circulo = Circle((0.0, 0.0), 1.5, fill=False, ec="black", lw=1.6, zorder=6, path_effects=K_HALO)
        ax.add_patch(self._k_circulo)
        for chave, cor in (("k1", K_COR_K1), ("k2", K_COR_K2)):
            self._k_merid[chave], = ax.plot([], [], color=cor, lw=2.0, solid_capstyle="round", zorder=7,
                                            path_effects=K_HALO)
            self._k_rot[chave] = _RotuloMeridiano(0.0, 0.0, "", color=cor, fontsize=7.5, fontweight="bold", zorder=8,
                                                  clip_on=True, path_effects=K_HALO_TEXTO)
            ax.add_artist(self._k_rot[chave])
        self._k_vertice, = ax.plot([0.0], [0.0], marker="+", ms=9, mew=1.5, color="black", zorder=9, path_effects=K_HALO)
        self._k_ax = ax

    def _k_desenhar_sobreposicao(self, d: float, zc: Optional[KeratometryZone]) -> None:
        """Circunferência Ø d no vértice e meridianos K2 (curvo) e K1 (plano) nos eixos TABO da circunferência (x para a
        direita do examinador, y para cima: o meridiano a° passa pelo vértice na direção (cos a, sen a))."""
        ax = self._k_ax
        if ax is None:
            return
        disp = zc is not None and zc.validity == "disponivel"
        self._k_circulo.set_radius(d / 2.0)
        self._k_circulo.set_edgecolor("black" if disp else "#6b6b6b")
        self._k_circulo.set_linestyle("solid" if disp else (0, (4, 3)))
        pouco = disp and zc.eixo_pouco_definido
        meia = 1.12 * d / 2.0
        for chave, nome in (("k2", "K2"), ("k1", "K1")):
            linha, rot = self._k_merid[chave], self._k_rot[chave]
            eixo = (zc.eixo_k2_deg if chave == "k2" else zc.eixo_k1_deg) if disp else None
            if eixo is None or not math.isfinite(eixo):
                linha.set_visible(False); rot.set_visible(False)
                continue
            ux, uy = math.cos(math.radians(eixo)), math.sin(math.radians(eixo))
            linha.set_data([-meia * ux, meia * ux], [-meia * uy, meia * uy])
            linha.set_linestyle((0, (3, 2)) if pouco else "-")
            linha.set_linewidth(1.3 if pouco else 2.0)
            linha.set_visible(True)
            # rótulo na ponta do meridiano do lado do eixo TABO (metade de cima), dentro do mapa
            rr = min(meia + 0.12, 0.84 * self._k_lim)
            rot.set_position((rr * ux, rr * uy))
            rot.set_horizontalalignment("left" if ux > 0.3 else ("right" if ux < -0.3 else "center"))
            rot.set_verticalalignment("bottom" if uy > 0.3 else ("top" if uy < -0.3 else "center"))
            rot.set_text(f"{nome} {fmt_eixo(eixo)}")
            rot.set_visible(True)
        ax.set_title(f"Potência axial (D) — Ø {d:.1f} mm", fontsize=8.5)
        self.cv_kmapa.canvas.draw_idle()

    def _k_evento_no_mapa(self, ev) -> bool:
        """Clique válido para escolher o diâmetro: dentro do mapa, controle ativo e sem zoom/arrasto da barra."""
        tb = self.cv_kmapa.toolbar
        return (self._k_ax is not None and ev.inaxes is self._k_ax and ev.xdata is not None and ev.ydata is not None
                and self.k_slider.isEnabled() and not (tb is not None and tb.mode))

    def _k_diametro_do_evento(self, ev) -> None:
        dec = int(round(2.0 * math.hypot(ev.xdata, ev.ydata) * 10.0))
        self.k_slider.setValue(min(max(dec, self.k_slider.minimum()), self.k_slider.maximum()))   # dispara a atualização

    def _k_mapa_press(self, ev) -> None:
        if ev.button == MouseButton.LEFT and self._k_evento_no_mapa(ev):
            self._k_arrastando = True
            self._k_diametro_do_evento(ev)

    def _k_mapa_move(self, ev) -> None:
        botoes = getattr(ev, "buttons", None)              # matplotlib recente: botões seguros durante o movimento
        if botoes is not None and MouseButton.LEFT not in botoes:
            self._k_arrastando = False                     # soltou fora da janela: o arrasto acabou
        if self._k_arrastando and self._k_evento_no_mapa(ev):
            self._k_diametro_do_evento(ev)

    def _k_mapa_release(self, _ev) -> None:
        self._k_arrastando = False

    def keratometry_tsv(self) -> str:
        """A tabela da aba Ceratometria como está na tela, em texto separado por tabulação (sem nome)."""
        t = self.k_table
        if t.rowCount() == 0:
            return ""
        out = [f"Ceratometria — {self._k_ident}"] if self._k_ident else []
        out.append("\t".join(K_COLUNAS))
        for r in range(t.rowCount()):
            cells = [t.item(r, c).text() if t.item(r, c) is not None else "" for c in range(t.columnCount())]
            out.append("\t".join(cells).rstrip("\t"))
        out.append(LEGENDA_TABELA)
        out.append(K_LEGENDA_COLUNAS_EXTRAS)
        return "\n".join(out) + "\n"

    @Slot()
    def copy_keratometry_tsv(self) -> str:
        tsv = self.keratometry_tsv()
        if not tsv:
            self.statusBar().showMessage("Nada a copiar: calcule o exame para ver a ceratometria.", 6000)
            return ""
        QApplication.clipboard().setText(tsv)
        self.statusBar().showMessage(f"Tabela de ceratometria copiada ({self.k_table.rowCount()} linhas, separadas por tabulação).", 8000)
        return tsv

    def fill_geometry_tab(self, st: PackageState) -> None:
        if st.exam is None:
            self.cv_elev.clear("importação inválida"); self.cv_cov.clear("importação inválida"); self.geom_text.clear()
            return
        try:
            ax = self.cv_elev.fresh_axes(); plot_elevation(ax, st.exam); self.cv_elev.draw()
            aps = []
            if st.analysis is not None:
                aps = [(w.center_xy[0], w.center_xy[1], w.profile.radius_mm) for w in st.analysis.wavefronts if w.validity == "disponivel"]
            ax = self.cv_cov.fresh_axes(); plot_coverage(ax, st.exam, apertures=aps); self.cv_cov.draw()
        except Exception as e:  # noqa: BLE001
            self.cv_elev.clear(f"erro ao desenhar: {e}")
        if st.analysis is None:
            self.geom_text.setPlainText(f"Sem cálculo ({st.status}{': ' + st.message if st.message else ''}). Geometria exibida = dados importados; reconstrução só após 'Calcular'.")
            return
        g = st.analysis.geometry
        self.geom_text.setPlainText(
            f"Método {g.method}, ordem {g.surface_order}, raio de ajuste {g.fit_radius_mm:.2f} mm, {g.n_points_used} pontos\n"
            f"Cônica: R = {g.conic_R_mm:.4f} mm, Q = {g.conic_Q:+.4f} (K ceratométrico equiv. {g.k_mean_D_keratometric:.2f} D, só conferência)\n"
            f"Resíduo: RMS {g.residual_rms_mm*1000:.3f} µm, máx {g.residual_max_mm*1000:.3f} µm; condicionamento {g.condition_number:.2e}\n"
            f"Lacuna central {g.central_gap_mm:.2f} mm; raio externo {g.outer_radius_mm:.2f} mm; fração medida {g.measured_fraction:.0%}"
            + (f"; resíduo em pontos reservados {g.holdout_residual_rms_mm*1000:.3f} µm" if g.holdout_residual_rms_mm else "")
        )

    def fill_wavefront_tab(self, st: PackageState) -> None:
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        if st.analysis is not None:
            for i, w in enumerate(st.analysis.wavefronts):
                self.profile_combo.addItem(f"{w.profile.name} — {w.validity}", i)
        self.profile_combo.blockSignals(False)
        if st.analysis is None:
            for c in (self.cv_w, self.cv_wfit, self.cv_res, self.cv_zbars):
                c.clear("sem cálculo" if st.status != "desatualizado" else "resultados desatualizados — recalcular")
            self.wf_state.setText("" if st.status != "desatualizado" else "DESATUALIZADO — parâmetros alterados; recalcular")
            self.wf_state.setStyleSheet("color:#8a6d00;")
            self.block_label.setText(st.message if st.status in ("inválida", "desatualizado") else "")
            self.block_label.setVisible(bool(self.block_label.text()))
            self.wf_notes.setText("")
            for b in self.bn.values():
                b.set_value(None)
            self.z_table.setRowCount(0)
            return
        self.on_profile_combo_changed()

    def _current_wavefront(self, st: PackageState) -> Optional[WavefrontResult]:
        if st.analysis is None or self.profile_combo.currentIndex() < 0:
            return None
        idx = self.profile_combo.currentData()
        if idx is None or idx >= len(st.analysis.wavefronts):
            return None
        return st.analysis.wavefronts[idx]

    @Slot()
    def on_profile_combo_changed(self):
        st = self.selected_state()
        if st is None:
            return
        wf = self._current_wavefront(st)
        if wf is None:
            return
        if wf.validity != "disponivel":
            self.wf_state.setText(f"{wf.validity.upper()} — números indisponíveis (ausência, não zero)")
            self.wf_state.setStyleSheet("color:#8B0000;")
            self.block_label.setText("Motivos:\n" + "\n".join(f"• {r}" for r in wf.block_reasons))
            self.block_label.show()
            for c in (self.cv_w, self.cv_wfit, self.cv_res, self.cv_zbars):
                c.clear(f"perfil {wf.validity}")
            for b in self.bn.values():
                b.set_value(None)
            self.wf_notes.setText("")
            self.z_table.setRowCount(0)
            return
        self.wf_state.setText(f"DISPONÍVEL — abertura {wf.profile.diameter_mm:g} mm, centro {wf.profile.center_mode} "
                              f"({wf.center_xy[0]:+.2f}, {wf.center_xy[1]:+.2f}) mm, Zernike OSA ordem {wf.profile.zernike_order}")
        self.wf_state.setStyleSheet("color:#1b5e20;")
        self.block_label.hide()
        short = {"W": "W traçada (µm)", "W_fit": "W ajustada — Zernike (µm)", "residual": "Resíduo traçada − ajuste (µm)"}
        try:
            for cv, key in ((self.cv_w, "W"), (self.cv_wfit, "W_fit"), (self.cv_res, "residual")):
                ax = cv.fresh_axes(); plot_wavefront(ax, wf, key, title=f"{short[key]} — {wf.profile.diameter_mm:g} mm"); cv.draw()
        except Exception as e:  # noqa: BLE001
            self.cv_w.clear(f"erro ao desenhar: {e}")
        m = wf.metrics
        self.bn["sa"].set_value(_fmt(m.get("sa_c40")), f"c(6,0) = {_fmt(m.get('sa_secundaria_c60'))}")
        self.bn["coma"].set_value(_fmt(m.get("coma_magnitude"), sign=False), f"{_fmt(m.get('coma_c3m1'))} / {_fmt(m.get('coma_c31'))}")
        self.bn["trefoil"].set_value(_fmt(m.get("trefoil_magnitude"), sign=False), f"{_fmt(m.get('trefoil_c3m3'))} / {_fmt(m.get('trefoil_c33'))}")
        self.bn["hoa"].set_value(_fmt(m.get("rms_hoa_n3plus"), sign=False), f"3ª {_fmt(m.get('rms_n3'), sign=False)} · 4ª {_fmt(m.get('rms_n4'), sign=False)} · ≥5ª {_fmt(m.get('rms_n5plus'), sign=False)}")
        cyl, axd = m.get("cilindro_equivalente_D"), m.get("eixo_equivalente_deg")
        self.bn["astig"].set_value(f"{_fmt(m.get('astig_c2m2'))} / {_fmt(m.get('astig_c22'))}",
                                   f"cil. refrativo equiv. {_fmt(cyl, 2)} D × {axd:.0f}° (não é ceratometria)" if cyl is not None and np.isfinite(cyl) else "")
        self.bn["defocus"].set_value(_fmt(m.get("defocus_c20")), "informado; removido do RMS HOA")
        notes = [f"Ajuste: resíduo RMS {wf.fit_residual_rms_um:.2e} µm, condicionamento {wf.condition_number:.2e}, raios válidos {wf.n_valid_rays}/{wf.n_rays}."]
        notes += wf.notes
        self.wf_notes.setText(" ".join(notes))
        self.fill_zernike_tab(wf)

    def fill_zernike_tab(self, wf: WavefrontResult) -> None:
        self.z_table.setRowCount(0)
        for c in wf.coefficients:
            r = self.z_table.rowCount(); self.z_table.insertRow(r)
            vals = [str(c.j), str(c.n), f"{c.m:+d}", c.name, f"{c.value_um:+.5f}"]
            for col, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if col in (0, 1, 2, 4):
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if (c.n, c.m) in ((4, 0), (3, 1), (3, -1), (3, 3), (3, -3)):
                    it.setBackground(QColor("#eef4fb"))
                self.z_table.setItem(r, col, it)
        try:
            ax = self.cv_zbars.fresh_axes(); plot_zernike_bars(ax, wf); self.cv_zbars.draw()
        except Exception as e:  # noqa: BLE001
            self.cv_zbars.clear(f"erro ao desenhar: {e}")

    def fill_technical_tab(self, st: PackageState) -> None:
        if st.analysis is None:
            self.tech_text.setPlainText(f"Sem cálculo ({st.status}). " + (st.message or ""))
            return
        a = st.analysis
        lines = [f"analysis_id: {a.analysis_id}", f"executada em: {a.created_at}",
                 "versões: " + ", ".join(f"{k}={v}" for k, v in a.versions.items()), "",
                 "Parâmetros de reconstrução: " + json.dumps(dataclasses.asdict(a.reconstruction), ensure_ascii=False),
                 "Limites de qualidade (provisórios): " + json.dumps(dataclasses.asdict(a.quality_config), ensure_ascii=False), "",
                 "Perfis ópticos:"]
        for w in a.wavefronts:
            lines.append("  " + json.dumps(dataclasses.asdict(w.profile), ensure_ascii=False))
        lines += ["", "Hashes de entrada:"] + [f"  {k}: {v}" for k, v in a.bundle_hashes.items()] + [""]
        lines += ["=" * 100, "RELATÓRIO TÉCNICO", "=" * 100] + report_lines(a)
        self.tech_text.setPlainText("\n".join(lines))

    # ----------------------------------------------------------- comparação
    def refresh_compare_controls(self) -> None:
        computed = [(k, st) for k, st in self.packages.items() if st.analysis is not None]
        cur_a, cur_b, cur_p = self.cmp_a.currentData(), self.cmp_b.currentData(), self.cmp_profile.currentText()
        for combo in (self.cmp_a, self.cmp_b):
            combo.blockSignals(True); combo.clear()
            for k, st in computed:
                combo.addItem(st.label.split(" — ")[0] + f"  [{os.path.basename(st.directory)}]", k)
            combo.blockSignals(False)
        names: list[str] = []
        for _, st in computed:
            for w in st.analysis.wavefronts:
                if w.profile.name not in names:
                    names.append(w.profile.name)
        self.cmp_profile.blockSignals(True); self.cmp_profile.clear(); self.cmp_profile.addItems(names); self.cmp_profile.blockSignals(False)
        for combo, cur in ((self.cmp_a, cur_a), (self.cmp_b, cur_b)):
            i = combo.findData(cur)
            if i >= 0:
                combo.setCurrentIndex(i)
        if self.cmp_b.count() > 1 and self.cmp_b.currentIndex() == self.cmp_a.currentIndex():
            self.cmp_b.setCurrentIndex(1 if self.cmp_a.currentIndex() == 0 else 0)
        i = self.cmp_profile.findText(cur_p)
        if i >= 0:
            self.cmp_profile.setCurrentIndex(i)

    def _clear_compare(self, message: str) -> None:
        for c in (self.cv_cmp_a, self.cv_cmp_b, self.cv_cmp_d):
            c.clear()
        self.cmp_table.setRowCount(0)
        self.cmp_msg.setStyleSheet("")
        self.cmp_msg.setText(message)

    @Slot()
    def run_compare(self):
        ka, kb, pname = self.cmp_a.currentData(), self.cmp_b.currentData(), self.cmp_profile.currentText()
        for c in (self.cv_cmp_a, self.cv_cmp_b, self.cv_cmp_d):
            c.clear()
        self.cmp_table.setRowCount(0)
        if not ka or not kb or not pname:
            self.cmp_msg.setText("Selecione dois pacotes calculados e um perfil.")
            return
        if ka == kb:
            self.cmp_msg.setText("Escolha dois pacotes diferentes.")
            return
        sa, sb = self.packages[ka], self.packages[kb]
        if sa.analysis is None or sb.analysis is None:
            self.cmp_msg.setText("Ambos os pacotes precisam estar calculados com os parâmetros atuais.")
            return
        wa = next((w for w in sa.analysis.wavefronts if w.profile.name == pname), None)
        wb = next((w for w in sb.analysis.wavefronts if w.profile.name == pname), None)
        if wa is None or wb is None:
            self.cmp_msg.setText(f"Perfil '{pname}' ausente em um dos pacotes.")
            return
        diff = profile_settings_diff(wa.profile, wb.profile)
        if diff:
            self.cmp_msg.setText("INCOMPATÍVEL — definições diferentes, diferença não calculada: " + "; ".join(diff))
            self.cmp_msg.setStyleSheet("color:#8B0000; font-weight:bold;")
            return
        if wa.validity != "disponivel" or wb.validity != "disponivel":
            self.cmp_msg.setText(f"Perfil indisponível em A ({wa.validity}) ou B ({wb.validity}); sem diferença. "
                                 + " | ".join(wa.block_reasons + wb.block_reasons))
            self.cmp_msg.setStyleSheet("color:#8B0000;")
            return
        notes = []
        if dataclasses.asdict(sa.analysis.reconstruction) != dataclasses.asdict(sb.analysis.reconstruction):
            notes.append("ATENÇÃO: configurações de reconstrução diferentes entre A e B (comparação metodológica).")
        if abs(wa.center_xy[0] - wb.center_xy[0]) > 1e-9 or abs(wa.center_xy[1] - wb.center_xy[1]) > 1e-9:
            notes.append(f"Centros físicos diferentes: A ({wa.center_xy[0]:+.2f}, {wa.center_xy[1]:+.2f}) mm, B ({wb.center_xy[0]:+.2f}, {wb.center_xy[1]:+.2f}) mm (mesmo modo '{wa.profile.center_mode}').")
        vlim = compare_scale([wa, wb], "W")
        try:
            ax = self.cv_cmp_a.fresh_axes(); plot_wavefront(ax, wa, "W", title=f"A: {sa.exam.identity.exam_id} {sa.exam.identity.eye or ''} — W (µm)", vlim=vlim); self.cv_cmp_a.draw()
            ax = self.cv_cmp_b.fresh_axes(); plot_wavefront(ax, wb, "W", title=f"B: {sb.exam.identity.exam_id} {sb.exam.identity.eye or ''} — W (µm)", vlim=vlim); self.cv_cmp_b.draw()
            if wa.map_grid is not None and wb.map_grid is not None and wa.map_grid["W"].shape == wb.map_grid["W"].shape:
                g = dict(wa.map_grid); g["W"] = wa.map_grid["W"] - wb.map_grid["W"]
                wd = dataclasses.replace(wa, map_grid=g)
                ax = self.cv_cmp_d.fresh_axes(); plot_wavefront(ax, wd, "W", title="Diferença A − B (µm)"); self.cv_cmp_d.draw()
            else:
                self.cv_cmp_d.clear("mapas com grades diferentes — diferença não calculada")
        except Exception as e:  # noqa: BLE001
            self.cv_cmp_d.clear(f"erro ao desenhar: {e}")
        cb = {(c.n, c.m): c.value_um for c in wb.coefficients}
        for c in wa.coefficients:
            r = self.cmp_table.rowCount(); self.cmp_table.insertRow(r)
            vb = cb.get((c.n, c.m))
            vals = [str(c.j), str(c.n), f"{c.m:+d}", c.name, f"{c.value_um:+.5f}", _fmt(vb, 5), _fmt(c.value_um - vb, 5) if vb is not None else "—"]
            for col, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if col != 3:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.cmp_table.setItem(r, col, it)
        self.cmp_msg.setStyleSheet("color:#1b5e20;")
        self.cmp_msg.setText(f"Compatível — perfil {pname}; escala de cor comum ±{vlim:.3f} µm. " + " ".join(notes))

    # ------------------------------------------------------------- exportação
    @Slot()
    def export_selected(self):
        st = self.selected_state()
        if st is None or st.analysis is None or st.exam is None:
            self._msg("Exportar", "Selecione um pacote calculado com os parâmetros atuais.", "warn")
            return
        d = QFileDialog.getExistingDirectory(self, "Pasta de destino do relatório") if self.interactive else tempfile.mkdtemp(prefix="tmsab_")
        if d:
            self.export_package(st, d)

    def export_package(self, st: PackageState, out_dir: str) -> dict[str, str]:
        paths = export_all(st.exam, st.analysis, out_dir, patient_name=self._patient_name(st))
        self.statusBar().showMessage("Exportado: " + ", ".join(os.path.basename(p) for p in paths.values()), 10000)
        return paths

    @Slot()
    def export_all_packages(self):
        ready = [st for st in self.packages.values() if st.analysis is not None and st.exam is not None]
        if not ready:
            self._msg("Exportar todos", "Nenhum pacote calculado.", "warn")
            return
        d = QFileDialog.getExistingDirectory(self, "Pasta de destino dos relatórios") if self.interactive else tempfile.mkdtemp(prefix="tmsab_")
        if not d:
            return
        n = 0
        for st in ready:
            sub = os.path.basename(st.directory) + (f"_{st.exam_id}" if st.exam_id else "")
            self.export_package(st, os.path.join(d, sub))
            n += 1
        self._msg("Exportar todos", f"{n} pacote(s) exportado(s) em {d}")

    @Slot()
    def about(self):
        if self.interactive:
            QMessageBox.about(self, "Sobre", ABOUT_TEXT)

    def closeEvent(self, event):  # noqa: N802
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(5000)
        super().closeEvent(event)
