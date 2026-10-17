"""Execução da cadeia do núcleo com progresso e cancelamento cooperativo.

Envelope fino sobre `tmsab.pipeline.analyze_exam` (que já aceita callbacks de progresso e
cancelamento), para que a interface e a linha de comando usem exatamente a mesma montagem
(mesmo identificador de análise, mesmos objetos).
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional

from ..geometry import ReconstructedSurface
from ..model import Analysis, NormalizedExam, OpticalProfile, QualityConfig, ReconstructionConfig
from ..pipeline import Cancelled, analyze_exam  # noqa: F401  (Cancelled reexportado para o worker)


def run_analysis(
    exam: NormalizedExam,
    profiles: list[OpticalProfile],
    recon: ReconstructionConfig,
    qc: Optional[QualityConfig] = None,
    progress: Optional[Callable[[str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    with_maps: bool = True,
) -> tuple[Analysis, ReconstructedSurface]:
    return analyze_exam(exam, profiles, recon, qc or QualityConfig(), with_maps=with_maps,
                        progress=progress, should_cancel=should_cancel)


def profile_settings_diff(a: OpticalProfile, b: OpticalProfile) -> list[str]:
    """Campos que diferem entre dois perfis (ignorando 'name'); vazio = compatíveis."""
    da, db = dataclasses.asdict(a), dataclasses.asdict(b)
    da.pop("name", None); db.pop("name", None)
    out = []
    for k in sorted(set(da) | set(db)):
        va, vb = da.get(k), db.get(k)
        if isinstance(va, (list, tuple)) and isinstance(vb, (list, tuple)):
            va, vb = list(va), list(vb)
        if va != vb:
            out.append(f"{k}: {va} ≠ {vb}")
    return out
