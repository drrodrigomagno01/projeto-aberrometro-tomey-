"""Encadeamento de funções (plano Apêndice A1).

import_tms_bundle -> validate_exam -> reconstruct_surface -> compute_anterior_wavefront
-> fit_zernike -> summarize_analysis -> export_result (reporting.py)

O núcleo óptico recebe geometria normalizada; não conhece nomes de arquivos TMS nem
identificadores pessoais. Cada análise tem identificador derivado dos hashes de entrada,
configuração e versões; reprocessar com outro centro, diâmetro ou método gera nova análise.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

from . import CORE_VERSION
from .geometry import ReconstructedSurface, reconstruct_surface, support_metrics
from .model import (
    Analysis,
    GeometrySummary,
    NormalizedExam,
    OpticalProfile,
    QualityConfig,
    ReconstructionConfig,
    WavefrontResult,
    ZernikeCoefficient,
    now_iso,
    stable_hash,
)
from .optics import fit_wavefront, refocus_rms, trace_wavefront, wavefront_map_grid
from .quality import evaluate_profile_availability, evaluate_trace
from .zernike import astigmatism_from_coeffs, metrics_from_coefficients, osa_j, term_name


def validate_exam(exam: NormalizedExam) -> list[str]:
    """Qualidade estrutural e geométrica inicial (não bloqueante além do que o adaptador já bloqueia)."""
    notes = []
    s = exam.samples
    if s.r is None:
        return ["sem coordenadas"]
    rn = np.nanmean(s.r, axis=1)
    notes.append(f"raio médio por anel: {np.nanmin(rn):.2f}–{np.nanmax(rn):.2f} mm")
    if exam.pupil.state == "registrada":
        d = float(np.hypot(exam.pupil.center_x, exam.pupil.center_y))
        notes.append(f"deslocamento centro pupilar–vértice: {d:.2f} mm")
        if d > 1.5:
            notes.append("deslocamento pupilar > 1,5 mm: verificar registro")
    if not exam.frame.transform_verified:
        notes.append("transformação de coordenadas não verificada (perfil presumido)")
    return notes


def default_profiles(diameters=(4.0, 6.0), centers=("vertice", "pupila"), order: int = 6) -> list[OpticalProfile]:
    out = []
    for c in centers:
        for d in diameters:
            out.append(OpticalProfile(name=f"anterior_v1_{c}_{d:g}mm", center_mode=c, diameter_mm=float(d), zernike_order=order))
    return out


def analysis_center(exam: NormalizedExam, profile: OpticalProfile) -> Optional[tuple[float, float]]:
    if profile.center_mode == "vertice":
        return (0.0, 0.0)
    if profile.center_mode == "pupila":
        if exam.pupil.state == "registrada" and exam.pupil.center_x is not None:
            return (float(exam.pupil.center_x), float(exam.pupil.center_y))
        return None
    raise ValueError(f"modo de centro desconhecido: {profile.center_mode}")


def compute_profile(exam: NormalizedExam, rec: ReconstructedSurface, profile: OpticalProfile, qc: QualityConfig,
                    with_maps: bool = True) -> WavefrontResult:
    center = analysis_center(exam, profile)
    if center is None:
        return WavefrontResult(profile=profile, center_xy=(float("nan"), float("nan")), validity="indisponivel",
                               block_reasons=["centro pupilar não registrado no exame"
                                              + (f" — {exam.pupil.method}" if exam.pupil.state != "registrada" and exam.pupil.method not in (None, "", "desconhecido") else "")],
                               support=None)
    cx, cy = center
    sup = support_metrics(exam.samples, rec, cx, cy, profile.radius_mm, edge_margin=qc.edge_margin)
    state, reasons = evaluate_profile_availability(profile, sup, rec, exam.pupil, qc, (cx, cy))
    res = WavefrontResult(profile=profile, center_xy=(cx, cy), validity=state, block_reasons=reasons, support=sup)
    if state != "disponivel":
        return res
    try:
        tr = refocus_rms(rec, profile, cx, cy) if profile.reference == "refoco_rms" else trace_wavefront(rec, profile, cx, cy)
    except ValueError as e:
        res.validity = "invalida"
        res.block_reasons = [f"traçado inválido: {e}"]
        return res
    fit = fit_wavefront(tr, profile.radius_mm, profile.zernike_order)
    more = evaluate_trace(tr.n_rays, tr.n_valid, fit.condition_number, qc)
    if more:
        res.validity = "indisponivel"
        res.block_reasons = more
    res.coefficients = [
        ZernikeCoefficient(n=n, m=m, j=osa_j(n, m), value_um=float(c), name=term_name(n, m)) for (n, m), c in zip(fit.terms, fit.coeffs)
    ]
    res.metrics = metrics_from_coefficients(fit.coeffs, fit.terms)
    cyl, ax = astigmatism_from_coeffs(res.metrics["astig_c2m2"], res.metrics["astig_c22"], profile.radius_mm)
    res.metrics["cilindro_equivalente_D"] = cyl
    res.metrics["eixo_equivalente_deg"] = ax
    res.fit_residual_rms_um = fit.residual_rms
    res.condition_number = fit.condition_number
    res.reference = {
        "s_imagem_mm": tr.s_image_mm, "potencia_equivalente_D": tr.power_eq_D,
        "curvatura_max_1_mm": tr.k_max, "curvatura_min_1_mm": tr.k_min, "eixo_curvatura_max_deg": tr.axis_deg,
        "raio_esfera_referencia_mm": tr.s_image_mm,
    }
    res.n_rays, res.n_valid_rays = tr.n_rays, tr.n_valid
    res.notes = list(tr.notes) + [
        f"abertura no plano corneano, centro ({cx:+.2f}, {cy:+.2f}) mm, diâmetro {profile.diameter_mm:g} mm",
        f"índices {profile.n_in} / {profile.n_out} (assunções do modelo); objeto axial no infinito; sem dispersão",
        f"esfera de referência: centro sobre o raio principal refratado a s'={tr.s_image_mm:.3f} mm, passa pelo ponto principal",
        "termos removidos para RMS HOA: " + ", ".join(profile.hoa_removed_terms),
    ]
    if with_maps:
        res.map_grid = wavefront_map_grid(tr, fit, profile.radius_mm)
    return res


def summarize_geometry(rec: ReconstructedSurface, cfg: ReconstructionConfig) -> GeometrySummary:
    return GeometrySummary(
        method=rec.method, fit_radius_mm=rec.fit_radius, surface_order=rec.surface_order, n_points_used=rec.n_points_used,
        conic_R_mm=rec.conic.R, conic_Q=rec.conic.Q, k_mean_D_keratometric=337.5 / rec.conic.R,
        residual_rms_mm=rec.residual_rms, residual_max_mm=rec.residual_max, residual_by_ring_rms_mm=rec.residual_by_ring_rms,
        holdout_residual_rms_mm=rec.holdout_residual_rms, condition_number=rec.condition_number,
        central_gap_mm=rec.central_gap, outer_radius_mm=rec.outer_radius, measured_fraction=rec.measured_fraction,
    )


class Cancelled(Exception):
    """Cálculo cancelado pelo chamador (verificado entre etapas)."""


def analyze_exam(exam: NormalizedExam, profiles: Optional[list[OpticalProfile]] = None,
                 recon: Optional[ReconstructionConfig] = None, qc: Optional[QualityConfig] = None,
                 with_maps: bool = True, progress=None, should_cancel=None) -> tuple[Analysis, ReconstructedSurface]:
    """Executa a cadeia completa para um exame normalizado.

    progress(msg) é chamado a cada etapa; should_cancel() é consultado entre etapas e, se verdadeiro,
    a execução termina com Cancelled (nenhum resultado parcial é devolvido).
    """
    profiles = profiles or default_profiles()
    recon = recon or ReconstructionConfig()
    qc = qc or QualityConfig()

    def _p(msg: str):
        if progress:
            progress(msg)

    def _c():
        if should_cancel and should_cancel():
            raise Cancelled()

    _p("validação estrutural")
    diag = validate_exam(exam)
    _c()
    _p("reconstrução da superfície")
    rec = reconstruct_surface(exam.samples, recon)
    results = []
    for i, p in enumerate(profiles, 1):
        _c()
        _p(f"perfil {i} de {len(profiles)}: {p.name}")
        results.append(compute_profile(exam, rec, p, qc, with_maps=with_maps))
    _c()
    _p("montagem do resultado")
    versions = {"nucleo": CORE_VERSION, "adaptador": f"{exam.bundle.adapter_profile}@{exam.bundle.adapter_version}", "numpy": np.__version__}
    import scipy
    versions["scipy"] = scipy.__version__
    payload = {
        "hashes": exam.bundle.hashes(), "recon": dataclasses.asdict(recon), "qc": dataclasses.asdict(qc),
        "profiles": [dataclasses.asdict(p) for p in profiles], "versions": versions,
        "frame": dataclasses.asdict(exam.frame),
    }
    analysis = Analysis(
        analysis_id=stable_hash(payload)[:24], created_at=now_iso(), versions=versions, identity=exam.identity,
        device=exam.device, frame=exam.frame, bundle_hashes=exam.bundle.hashes(), synthetic=exam.bundle.synthetic,
        reconstruction=recon, geometry=summarize_geometry(rec, recon), pupil=exam.pupil, wavefronts=results,
        quality_config=qc, import_diagnostics=list(exam.diagnostics) + diag, warnings=list(exam.warnings),
    )
    return analysis, rec
