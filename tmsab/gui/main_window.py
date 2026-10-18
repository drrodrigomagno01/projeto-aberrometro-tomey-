"""Janela principal do programa (plano §7).

Estados por pacote: carregado (importado, sem cálculo) · calculando · calculado · desatualizado
(parâmetros mudaram; nenhum número antigo é exibido) · inválida (importação ou análise bloqueada).
"""
from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction, QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
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
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from ..adapters import AdapterError, detect_profile, import_tms_bundle, list_exams
from ..model import Analysis, NormalizedExam, OpticalProfile, QualityConfig, ReconstructionConfig, WavefrontResult
from ..pipeline import default_profiles
from ..plots import compare_scale, plot_coverage, plot_elevation, plot_wavefront, plot_zernike_bars
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
        lay.addWidget(self.cap)
        lay.addWidget(self.val)
        lay.addWidget(self.sub)
        self.setFrameStyle = None

    def set_value(self, text: Optional[str], sub: str = "") -> None:
        self.val.setText(text if text else "—")
        self.sub.setText(sub)


def _fmt(v, nd: int = 4, sign: bool = True) -> str:
    try:
        if v is None or not np.isfinite(v):
            return "—"
        return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    except TypeError:
        return "—"


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
        drow = QHBoxLayout()
        drow.addWidget(QLabel("Diâmetros:"))
        self.d4 = QCheckBox("4 mm"); self.d4.setChecked(True)
        self.d6 = QCheckBox("6 mm"); self.d6.setChecked(True)
        self.d_other_check = QCheckBox("outro")
        self.d_other = QDoubleSpinBox(); self.d_other.setRange(2.0, 8.0); self.d_other.setSingleStep(0.5); self.d_other.setValue(5.0); self.d_other.setSuffix(" mm")
        self.d_other.setEnabled(False)
        self.d_other_check.toggled.connect(self.d_other.setEnabled)
        for x in (self.d4, self.d6, self.d_other_check, self.d_other):
            drow.addWidget(x)
        drow.addStretch(1)
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
        for sig in (self.center_combo.currentIndexChanged, self.d4.toggled, self.d6.toggled, self.d_other_check.toggled,
                    self.d_other.valueChanged, self.order_spin.valueChanged, self.method_combo.currentIndexChanged,
                    self.surf_order_spin.valueChanged, self.ref_combo.currentIndexChanged):
            sig.connect(lambda *_: self.on_params_changed())

        self.auto_compute = QCheckBox("Calcular automaticamente ao abrir a pasta"); self.auto_compute.setChecked(True)
        self.auto_export = QCheckBox("Salvar relatórios em <pasta>/analises ao terminar"); self.auto_export.setChecked(True)
        for cb in (self.auto_compute, self.auto_export):
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
        self.tabs.addTab(self._build_import_tab(), "Importação")
        self.tabs.addTab(self._build_geometry_tab(), "Geometria")
        self.tabs.addTab(self._build_wavefront_tab(), "Frente de onda")
        self.tabs.addTab(self._build_zernike_tab(), "Zernike")
        self.tabs.addTab(self._build_compare_tab(), "Comparação")
        self.tabs.addTab(self._build_technical_tab(), "Técnico")
        return self.tabs

    # ---- abas
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
    def current_profiles(self) -> list[OpticalProfile]:
        centers = {0: ("vertice",), 1: ("pupila",), 2: ("vertice", "pupila")}[self.center_combo.currentIndex()]
        diam = []
        if self.d4.isChecked():
            diam.append(4.0)
        if self.d6.isChecked():
            diam.append(6.0)
        if self.d_other_check.isChecked():
            diam.append(float(self.d_other.value()))
        diam = sorted(set(diam))
        profs = default_profiles(diam, centers, int(self.order_spin.value())) if diam else []
        for p in profs:
            p.reference = self.ref_combo.currentText()
        return profs

    def current_recon(self) -> ReconstructionConfig:
        return ReconstructionConfig(method=self.method_combo.currentText(), surface_order=int(self.surf_order_spin.value()))

    def set_default_params(self) -> None:
        self._params_block = True
        try:
            self.center_combo.setCurrentIndex(2); self.d4.setChecked(True); self.d6.setChecked(True)
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
        self.worker = AnalysisWorker(jobs, profiles, self.current_recon(), QualityConfig(), parent=self)
        self.worker.progress.connect(self.on_progress)
        self.worker.package_done.connect(self.on_package_done)
        self.worker.package_failed.connect(self.on_package_failed)
        self.worker.finished_all.connect(self.on_finished_all)
        self.btn_compute.setEnabled(False); self.btn_cancel.setEnabled(True)
        self._set_params_enabled(False)
        self.worker.start()

    def _set_params_enabled(self, on: bool) -> None:
        for x in (self.center_combo, self.d4, self.d6, self.d_other_check, self.order_spin, self.method_combo,
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
        profiles = profiles or self.current_profiles()
        for key in (keys or list(self.packages)):
            st = self.packages[key]
            if st.exam is None:
                continue
            try:
                analysis, rec = run_analysis(st.exam, profiles, self.current_recon(), QualityConfig())
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

    def refresh_all_tabs(self, st: PackageState) -> None:
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
            self.pupil_label.setText(f"Pupila: registrada\ncentro ({p.center_x:+.3f}, {p.center_y:+.3f}) mm\n"
                                     f"diâmetro {p.diameter if p.diameter is not None else '—'} mm\nreferencial {p.frame}\nmétodo {p.method}")
        else:
            self.pupil_label.setText(f"Pupila: {p.state} — perfis centrados na pupila ficam bloqueados (a pupila não é preenchida com o vértice)")

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
        paths = export_all(st.exam, st.analysis, out_dir)
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
