"""Regras de disponibilidade e motivos de bloqueio (plano §7 e §8).

Três estados por perfil: "disponivel", "indisponivel" (dados insuficientes para o perfil),
"invalida" (importação ou reconstrução inválida). Os limites são provisórios — derivados de
orçamento de erro e estabilidade numérica, não de critérios clínicos publicados.
"""
from __future__ import annotations

from typing import Optional

from .geometry import ReconstructedSurface
from .model import OpticalProfile, PupilData, QualityConfig, SupportMetrics


def evaluate_profile_availability(
    profile: OpticalProfile,
    support: SupportMetrics,
    rec: ReconstructedSurface,
    pupil: PupilData,
    qc: QualityConfig,
    center_xy: tuple[float, float],
) -> tuple[str, list[str]]:
    """Retorna (estado, motivos). Estado 'indisponivel' bloqueia o cálculo óptico do perfil."""
    reasons: list[str] = []
    a = profile.radius_mm

    if profile.center_mode == "pupila" and pupil.state != "registrada":
        reasons.append("centro pupilar não registrado no exame (perfil pupilar bloqueado; pupila não é preenchida com o vértice)")
    if not support.inside_fit_domain:
        reasons.append(
            f"abertura (centro {center_xy[0]:+.2f},{center_xy[1]:+.2f} mm, raio {a:.2f} mm) excede o domínio reconstruído (raio {rec.fit_radius:.2f} mm)"
        )
    if support.measured_fraction < qc.min_measured_fraction:
        reasons.append(f"fração de pontos medidos na abertura {support.measured_fraction:.0%} < {qc.min_measured_fraction:.0%}")
    if support.usable_fraction < qc.min_usable_fraction:
        reasons.append(f"fração de pontos utilizáveis na abertura {support.usable_fraction:.0%} < {qc.min_usable_fraction:.0%}")
    if support.edge_reach_fraction < qc.min_edge_meridians_fraction:
        reasons.append(
            f"cobertura medida atinge a borda em {support.edge_reach_fraction:.0%} dos meridianos < {qc.min_edge_meridians_fraction:.0%}"
        )
    if support.max_missing_sector_deg > qc.max_missing_sector_deg:
        reasons.append(f"maior setor sem dados medidos na borda {support.max_missing_sector_deg:.0f}° > {qc.max_missing_sector_deg:.0f}°")
    if support.central_gap_mm > qc.max_central_gap_mm:
        reasons.append(f"lacuna central {support.central_gap_mm:.2f} mm > {qc.max_central_gap_mm:.2f} mm (região estimada pelo modelo)")
    if rec.residual_rms * 1000.0 > qc.max_surface_residual_rms_um:
        reasons.append(f"resíduo RMS da reconstrução {rec.residual_rms*1000:.2f} µm > {qc.max_surface_residual_rms_um:.2f} µm")
    if rec.condition_number == rec.condition_number and rec.condition_number > qc.max_condition_number:  # not NaN
        reasons.append(f"condicionamento da reconstrução {rec.condition_number:.1e} > {qc.max_condition_number:.0e}")
    state = "disponivel" if not reasons else "indisponivel"
    return state, reasons


def evaluate_trace(n_rays: int, n_valid: int, cond: Optional[float], qc: QualityConfig) -> list[str]:
    reasons = []
    if n_rays and (n_rays - n_valid) / n_rays > qc.max_invalid_ray_fraction:
        reasons.append(f"{n_rays - n_valid} de {n_rays} raios sem solução (> {qc.max_invalid_ray_fraction:.0%})")
    if cond is not None and cond > qc.max_condition_number:
        reasons.append(f"condicionamento do ajuste de Zernike {cond:.1e} > {qc.max_condition_number:.0e}")
    return reasons
