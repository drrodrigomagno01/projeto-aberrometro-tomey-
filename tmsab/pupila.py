"""Chord mu aparente: centro da pupila de entrada (PUP, nesta captura) em relação ao vértice topográfico do TMS.

O vértice é a origem do mapa do TMS: o reflexo corneano no centro dos anéis com o olho fixando o alvo (≈ reflexo
corneano coaxial). A distância dele ao centro da pupila é o chord mu (Chang & Waring 2014) medido pelo TMS — com a
pupila da captura (iluminação do Plácido), não com pupilometria padronizada; o centro da pupila se desloca com o
tamanho dela e a medida pressupõe boa fixação no alvo. Referencial do exame: vista do examinador, +x à direita da
imagem (nasal em OD, temporal em OS), +y superior.
"""
from __future__ import annotations

import math
from typing import Optional

from .model import ApparentChordMu, PupilData


def chord_mu_aparente(pupil: Optional[PupilData], eye: Optional[str]) -> Optional[ApparentChordMu]:
    """Deslocamento do centro da pupila em relação ao vértice; None sem pupila registrada (nunca usa o vértice)."""
    if pupil is None or pupil.state != "registrada" or pupil.center_x is None or pupil.center_y is None:
        return None
    x, y = float(pupil.center_x), float(pupil.center_y)
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    olho = (eye or "").strip().upper()
    nasal = x if olho == "OD" else (-x if olho == "OS" else None)
    d = float(pupil.diameter) if pupil.diameter is not None and math.isfinite(float(pupil.diameter)) and pupil.diameter > 0 else None
    return ApparentChordMu(distancia_mm=math.hypot(x, y), x_mm=x, y_mm=y, nasal_mm=nasal, superior_mm=y,
                           diametro_pupila_mm=d)


def fmt_chord_mu(cm: Optional[ApparentChordMu], curto: bool = False) -> str:
    """'0.23 mm — temporal 0.15, superior 0.17 (pupila 5.85 mm desta captura)'; `curto` omite a pupila."""
    if cm is None:
        return "não calculado (pupila não detectada nesta captura)"

    def comp(v: float, pos: str, neg: str) -> str:
        return f"{pos if round(v, 2) >= 0 else neg} {abs(v):.2f}"

    if cm.nasal_mm is not None:
        dirs = f"{comp(cm.nasal_mm, 'nasal', 'temporal')}, {comp(cm.superior_mm, 'superior', 'inferior')}"
    else:
        dirs = f"x {round(cm.x_mm, 2) + 0.0:+.2f}, y {round(cm.y_mm, 2) + 0.0:+.2f} mm (olho não identificado)"
    txt = f"{cm.distancia_mm:.2f} mm — {dirs}"
    if not curto and cm.diametro_pupila_mm:
        txt += f" (pupila {cm.diametro_pupila_mm:.2f} mm desta captura)"
    return txt
