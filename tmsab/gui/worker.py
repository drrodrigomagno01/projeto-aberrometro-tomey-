"""Processamento em segundo plano (QThread): importação + análise por pacote, com progresso e cancelamento."""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import QThread, Signal

from ..adapters import AdapterError, import_tms_bundle
from ..model import OpticalProfile, QualityConfig, ReconstructionConfig
from .compute import Cancelled, run_analysis


class AnalysisWorker(QThread):
    """jobs: lista de (chave, diretório). Emite resultados por pacote; nunca propaga exceções."""

    progress = Signal(int, str)                       # percentual 0–100, mensagem
    package_done = Signal(str, object, object, object)  # chave, exam, analysis, rec
    package_failed = Signal(str, str)                 # chave, mensagem
    finished_all = Signal(bool)                       # True se cancelado

    def __init__(self, jobs: list[tuple[str, str]], profiles: list[OpticalProfile], recon: ReconstructionConfig,
                 qc: QualityConfig | None = None, adapter_profile: str = "tms4_datatables_v0", parent: Any = None):
        super().__init__(parent)
        self.jobs = list(jobs)
        self.profiles = list(profiles)
        self.recon = recon
        self.qc = qc or QualityConfig()
        self.adapter_profile = adapter_profile
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def _should_cancel(self) -> bool:
        return self._cancel

    def run(self) -> None:
        n = max(1, len(self.jobs))
        cancelled = False
        for i, (key, directory) in enumerate(self.jobs):
            if self._cancel:
                cancelled = True
                break
            base = int(100 * i / n)
            span = max(1, int(100 / n))
            label = directory.rstrip("/\\").split("/")[-1].split("\\")[-1]
            self.progress.emit(base, f"{label}: importação")
            try:
                exam = import_tms_bundle(directory, self.adapter_profile)
            except AdapterError as e:
                self.package_failed.emit(key, f"importação inválida: {e}")
                continue
            except Exception as e:  # noqa: BLE001 — nunca derrubar a thread
                self.package_failed.emit(key, f"erro inesperado na importação: {type(e).__name__}: {e}")
                continue
            steps = len(self.profiles) + 3
            counter = {"k": 0}

            def _progress(msg: str, _base=base, _span=span, _label=label):
                counter["k"] += 1
                self.progress.emit(min(99, _base + int(_span * counter["k"] / steps)), f"{_label}: {msg}")

            try:
                analysis, rec = run_analysis(exam, self.profiles, self.recon, self.qc, progress=_progress,
                                             should_cancel=self._should_cancel, with_maps=True)
            except Cancelled:
                cancelled = True
                break
            except (ValueError, AdapterError) as e:
                self.package_failed.emit(key, f"análise inválida: {e}")
                continue
            except Exception as e:  # noqa: BLE001
                self.package_failed.emit(key, f"erro inesperado na análise: {type(e).__name__}: {e}")
                continue
            self.package_done.emit(key, exam, analysis, rec)
        self.progress.emit(100 if not cancelled else 0, "cancelado" if cancelled else "concluído")
        self.finished_all.emit(cancelled)
