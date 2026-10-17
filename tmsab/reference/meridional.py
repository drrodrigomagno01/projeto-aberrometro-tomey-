"""Traçado meridional 2D de referência para cônicas de revolução (código independente do núcleo 3D).

Usa trigonometria escalar (ângulo de incidência/refração) em vez da forma vetorial da lei de
Snell, e a esfera de referência resolvida em 2D. Serve para verificar optics.trace_wavefront
em superfícies com solução conhecida. Mesma convenção: W = OPL_principal − OPL(raio), em µm.
"""
from __future__ import annotations

import numpy as np


def conic_sag(x: np.ndarray, R: float, Q: float) -> np.ndarray:
    x = np.asarray(x, float)
    return x**2 / (R + np.sqrt(R**2 - (1 + Q) * x**2))


def meridional_wavefront_conic(R: float, Q: float, n_out: float, x: np.ndarray, n_in: float = 1.0) -> np.ndarray:
    """W(x) em µm para raios paralelos ao eixo na altura x, superfície cônica (R, Q)."""
    x = np.asarray(x, float)
    z = conic_sag(x, R, Q)
    slope = x / np.sqrt(R**2 - (1 + Q) * x**2)          # dz/dx
    phi = np.arctan(slope)                                 # ângulo da normal com o eixo = ângulo de incidência
    i = np.abs(phi)
    ip = np.arcsin(np.clip(np.sin(i) * n_in / n_out, -1, 1))
    delta = i - ip                                         # desvio em direção ao eixo
    tx = -np.sign(x) * np.sin(delta)
    tz = np.cos(delta)
    L = n_out * R / (n_out - n_in)                         # distância paraxial da imagem (vértice -> F)
    # esfera de referência: centro F=(0, L), raio L; raiz do lado próximo de F
    qx, qz = x, z - L
    qt = qx * tx + qz * tz
    disc = qt**2 - (qx**2 + qz**2) + L**2
    s = -qt - np.sqrt(disc)
    opl = n_in * z + n_out * s
    opl_chief = 0.0
    return (opl_chief - opl) * 1000.0
