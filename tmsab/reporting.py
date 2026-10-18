"""Relatórios (plano §7): JSON canônico, CSV e PDF derivados do mesmo objeto de resultado.

- JSON: null para ausências, nunca números não finitos; mapas não entram (ficam em arquivos NPZ opcionais).
- CSV: coeficientes assinados com índices n, m, j, unidade, perfil e identificação da análise.
- PDF: identificação, finalidade, perfil, números, qualidade, proveniência; mapas; tabela de Zernike.
Um perfil bloqueado gera relatório com o motivo; números indisponíveis aparecem como ausência.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import os
from typing import Optional

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from . import CORE_VERSION
from .model import Analysis, NormalizedExam, WavefrontResult, to_jsonable
from .plots import figure_resultado_completo, plot_coverage, plot_elevation, plot_wavefront, plot_zernike_bars


# ---------------------------------------------------------------------------
# JSON canônico
# ---------------------------------------------------------------------------
def canonical_result(analysis: Analysis) -> dict:
    d = to_jsonable(analysis)
    for w in d["wavefronts"]:
        w.pop("map_grid", None)      # mapas ficam fora do JSON canônico
    d["schema"] = "tmsab.analysis/1"
    d["purpose_note"] = (
        "Aberrometria corneana anterior calculada a partir de dados de videoceratoscopia; "
        "não representa a frente de onda ocular total nem substitui aberrometria ocular. "
        "Resultados vinculados a abertura, centro, modelo e versões registrados."
    )
    return d


def export_json(analysis: Analysis, path: str) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(canonical_result(analysis), fh, ensure_ascii=False, indent=2, allow_nan=False)
    return path


def export_maps_npz(analysis: Analysis, path: str) -> str:
    arrays = {}
    for i, w in enumerate(analysis.wavefronts):
        if w.map_grid is None:
            continue
        for k, v in w.map_grid.items():
            arrays[f"perfil{i}_{w.profile.name}_{k}"] = np.asarray(v)
    np.savez_compressed(path, **arrays)
    return path


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
def export_csv(analysis: Analysis, path: str) -> str:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["analysis_id", "study_id", "exam_id", "eye", "perfil", "centro", "centro_x_mm", "centro_y_mm",
                    "diametro_mm", "plano_abertura", "ordem_max", "normalizacao", "validade", "n", "m", "j", "nome", "valor_um"])
        for wf in analysis.wavefronts:
            p = wf.profile
            base = [analysis.analysis_id, analysis.identity.study_id, analysis.identity.exam_id, analysis.identity.eye or "",
                    p.name, p.center_mode, _fmt(wf.center_xy[0]), _fmt(wf.center_xy[1]), p.diameter_mm, p.aperture_plane,
                    p.zernike_order, p.normalization, wf.validity]
            if wf.validity != "disponivel" or not wf.coefficients:
                w.writerow(base + ["", "", "", "BLOQUEADO: " + "; ".join(wf.block_reasons), ""])
                continue
            for c in wf.coefficients:
                w.writerow(base + [c.n, c.m, c.j, c.name, f"{c.value_um:.6f}"])
        w.writerow([])
        w.writerow(["# metricas (um) por perfil"])
        for wf in analysis.wavefronts:
            if wf.validity != "disponivel":
                continue
            for k, v in wf.metrics.items():
                w.writerow([analysis.analysis_id, "", "", "", wf.profile.name, "", "", "", "", "", "", "", "", "", "", "", k, _fmt(v)])
    return path


def _fmt(v) -> str:
    try:
        return "" if v is None or not np.isfinite(v) else f"{float(v):.6f}"
    except TypeError:
        return str(v)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def _wrap(lines: list[str]) -> list[str]:
    import textwrap
    out: list[str] = []
    for ln in lines:
        if ln.startswith("## "):
            out.append(ln)
            continue
        indent = "  " if ln.startswith("  ") else ""
        width = 96 if indent else 118
        parts = textwrap.wrap(ln.strip(), width=width, subsequent_indent="    ") or [""]
        out.extend(indent + p if not p.startswith("    ") else indent + p for p in parts)
    return out


def _text_page(lines: list[str], title: str) -> Figure:
    fig = Figure(figsize=(8.27, 11.69))
    fig.text(0.07, 0.95, title, fontsize=13, weight="bold", va="top")
    y = 0.91
    for ln in lines:
        if y < 0.05:
            break
        size = 8.5
        weight = "bold" if ln.startswith("## ") else "normal"
        txt = ln[3:] if ln.startswith("## ") else ln
        fig.text(0.07, y, txt, fontsize=size, va="top", weight=weight, family="monospace" if ln.startswith("  ") else "sans-serif")
        y -= 0.0165 if not ln.startswith("## ") else 0.024
    return fig


def report_lines(analysis: Analysis) -> list[str]:
    a = analysis
    L: list[str] = []
    L.append("## Identificação")
    L.append(f"Código do estudo: {a.identity.study_id}   Exame: {a.identity.exam_id}   Olho: {a.identity.eye or 'não informado'}")
    L.append(f"Data de aquisição: {a.identity.acquired_at or 'não informada'} (precisão: {a.identity.date_precision})   Captura: {a.identity.capture_index if a.identity.capture_index is not None else '—'}")
    L.append(f"Equipamento: {a.device.model} / software {a.device.software_version} / calibração: {a.device.calibration_state}")
    if a.synthetic:
        L.append("*** EXAME SINTÉTICO — gerado por software, identificado como tal; não corresponde a um paciente. ***")
    L.append("## Finalidade")
    L.append(a.purpose + ". Não é aberrometria ocular total; não é alegação diagnóstica, seleção de LIO nem planejamento terapêutico.")
    L.append("## Geometria reconstruída")
    g = a.geometry
    L.append(f"Método: {g.method} (ordem {g.surface_order}, raio de ajuste {g.fit_radius_mm:.2f} mm, {g.n_points_used} pontos)")
    L.append(f"Cônica de referência: R = {g.conic_R_mm:.4f} mm, Q = {g.conic_Q:+.4f}  (K ceratométrico equivalente {g.k_mean_D_keratometric:.2f} D, só conferência)")
    L.append(f"Resíduo da reconstrução: RMS {g.residual_rms_mm*1000:.3f} µm, máx {g.residual_max_mm*1000:.3f} µm; condicionamento {g.condition_number:.2e}")
    L.append(f"Lacuna central {g.central_gap_mm:.2f} mm; raio externo medido {g.outer_radius_mm:.2f} mm; fração medida {g.measured_fraction:.0%}"
             + (f"; resíduo em pontos reservados {g.holdout_residual_rms_mm*1000:.3f} µm" if g.holdout_residual_rms_mm else ""))
    L.append(f"Pupila: {a.pupil.state}" + (f" — centro ({a.pupil.center_x:+.2f}, {a.pupil.center_y:+.2f}) mm, diâmetro {a.pupil.diameter if a.pupil.diameter else '—'} mm, referencial {a.pupil.frame}, método {a.pupil.method}" if a.pupil.state == "registrada" else ""))
    L.append(f"Referencial: {a.frame.axes}; origem {a.frame.origin}; altura {a.frame.height_reference}; ângulo inicial {a.frame.angle_start_deg:g}°, sentido {'anti-horário' if a.frame.angle_direction == 1 else 'horário'}; espelho x: {a.frame.mirror_x}; transformação verificada: {'sim' if a.frame.transform_verified else 'NÃO'}")
    for wf in a.wavefronts:
        p = wf.profile
        L.append(f"## Perfil {p.name} — {wf.validity.upper()}")
        L.append(f"Abertura: diâmetro {p.diameter_mm:g} mm no plano {p.aperture_plane}; centro {p.center_mode} ({_f2(wf.center_xy[0])}, {_f2(wf.center_xy[1])}) mm; índices {p.n_in}/{p.n_out}; objeto {p.object_conjugate}; referência {p.reference}; Zernike OSA ordem {p.zernike_order}")
        if wf.support:
            s = wf.support
            L.append(f"Suporte: medidos {s.measured_fraction:.0%}, utilizáveis {s.usable_fraction:.0%}, borda atingida em {s.edge_reach_fraction:.0%} dos meridianos, maior setor sem dados {s.max_missing_sector_deg:.0f}°, lacuna central {s.central_gap_mm:.2f} mm, dentro do domínio: {'sim' if s.inside_fit_domain else 'não'}")
        if wf.validity != "disponivel":
            for r in wf.block_reasons:
                L.append(f"  BLOQUEIO: {r}")
            L.append("  Números indisponíveis para este perfil (ausência, não zero).")
            continue
        m = wf.metrics
        L.append(f"  SA primária c(4,0) = {m['sa_c40']:+.4f} µm   SA secundária c(6,0) = {m.get('sa_secundaria_c60', float('nan')):+.4f} µm")
        L.append(f"  Coma c(3,-1) = {m['coma_c3m1']:+.4f}  c(3,+1) = {m['coma_c31']:+.4f}  |coma| = {m['coma_magnitude']:.4f} µm")
        L.append(f"  Trefoil c(3,-3) = {m['trefoil_c3m3']:+.4f}  c(3,+3) = {m['trefoil_c33']:+.4f}  |trefoil| = {m['trefoil_magnitude']:.4f} µm")
        L.append(f"  Astig. c(2,-2) = {m['astig_c2m2']:+.4f}  c(2,+2) = {m['astig_c22']:+.4f} µm  (cil. refrativo equiv. {m['cilindro_equivalente_D']:+.2f} D × {m['eixo_equivalente_deg']:.0f}°; não é ceratometria)")
        L.append(f"  Defocus c(2,0) = {m['defocus_c20']:+.4f} µm (referência paraxial; termo informado, removido do RMS HOA)")
        L.append(f"  RMS HOA (n≥3) = {m['rms_hoa_n3plus']:.4f} µm   RMS 3ª = {m['rms_n3']:.4f}   RMS 4ª = {m['rms_n4']:.4f}   RMS ≥5ª = {m['rms_n5plus']:.4f} µm")
        L.append(f"  Ajuste: resíduo RMS {wf.fit_residual_rms_um:.2e} µm, condicionamento {wf.condition_number:.2e}, raios válidos {wf.n_valid_rays}/{wf.n_rays}; termos removidos do HOA: {', '.join(p.hoa_removed_terms)}")
        ref = wf.reference
        L.append(f"  Referência: s' = {ref.get('s_imagem_mm', float('nan')):.3f} mm, potência equivalente {ref.get('potencia_equivalente_D', float('nan')):.2f} D (n={p.n_out}), curvaturas {ref.get('curvatura_max_1_mm', float('nan')):.4f}/{ref.get('curvatura_min_1_mm', float('nan')):.4f} mm⁻¹ @ {ref.get('eixo_curvatura_max_deg', float('nan')):.0f}°")
    L.append("## Qualidade e limitações")
    qc = a.quality_config
    L.append(f"Limites de disponibilidade (provisórios, não clínicos): medidos ≥{qc.min_measured_fraction:.0%}, utilizáveis ≥{qc.min_usable_fraction:.0%}, borda ≥{qc.min_edge_meridians_fraction:.0%} dos meridianos, setor ≤{qc.max_missing_sector_deg:.0f}°, lacuna central ≤{qc.max_central_gap_mm:.2f} mm, resíduo ≤{qc.max_surface_residual_rms_um:.1f} µm")
    for wmsg in a.warnings:
        L.append(f"AVISO: {wmsg}")
    L.append("Nenhuma faixa de normalidade é atribuída: limites clínicos exigem estudo próprio (plano §9).")
    L.append("## Proveniência")
    L.append(f"Identificador da análise: {a.analysis_id}   Executada em: {a.created_at}")
    L.append("Versões: " + ", ".join(f"{k}={v}" for k, v in a.versions.items()))
    L.append(f"Reconstrução: {dataclasses.asdict(a.reconstruction)}")
    for k, v in a.bundle_hashes.items():
        L.append(f"  {k}: sha256 {v[:16]}…{v[-8:]}")
    for d in a.import_diagnostics:
        L.append(f"  diagnóstico: {d}")
    return L


def _f2(v) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:+.2f}"


def export_pdf(exam: NormalizedExam, analysis: Analysis, path: str) -> str:
    with PdfPages(path) as pdf:
        title = f"Aberrometria corneana anterior calculada — relatório de pesquisa (tmsab {CORE_VERSION})"
        # página 1: resultado completo (topografia + frente de onda + tabelas) — pronta para impressão
        try:
            pdf.savefig(figure_resultado_completo(exam, analysis))
        except Exception as e:  # noqa: BLE001 — a página-resumo nunca impede o relatório completo
            pdf.savefig(_text_page([f"Página de resultado completo indisponível: {e}"], title))
        lines = _wrap(report_lines(analysis))
        # paginação simples
        per_page = 50
        for i in range(0, len(lines), per_page):
            pdf.savefig(_text_page(lines[i:i + per_page], title if i == 0 else title + " (cont.)"))
        # mapas de geometria
        fig = Figure(figsize=(11, 5.5))
        ax1, ax2 = fig.subplots(1, 2)
        plot_elevation(ax1, exam)
        aps = [(w.center_xy[0], w.center_xy[1], w.profile.radius_mm) for w in analysis.wavefronts if w.validity == "disponivel"]
        plot_coverage(ax2, exam, apertures=aps)
        fig.tight_layout()
        pdf.savefig(fig)
        # por perfil disponível
        for wf in analysis.wavefronts:
            if wf.validity != "disponivel" or wf.map_grid is None:
                continue
            fig = Figure(figsize=(11, 8.5))
            axs = fig.subplots(2, 2)
            plot_wavefront(axs[0, 0], wf, "W")
            plot_wavefront(axs[0, 1], wf, "W_fit")
            plot_wavefront(axs[1, 0], wf, "residual")
            plot_zernike_bars(axs[1, 1], wf)
            fig.suptitle(f"Perfil {wf.profile.name} — exame {analysis.identity.exam_id} {analysis.identity.eye or ''} — análise {analysis.analysis_id}", fontsize=9)
            fig.tight_layout()
            pdf.savefig(fig)
            # tabela de coeficientes
            rows = [f"  {c.j:>3}  n={c.n} m={c.m:+d}  {c.value_um:+.5f} µm  {c.name}" for c in wf.coefficients]
            pdf.savefig(_text_page(_wrap([f"## Tabela de Zernike (OSA, ordem {wf.profile.zernike_order}, µm) — perfil {wf.profile.name}"]) + rows,
                                   f"Coeficientes — análise {analysis.analysis_id}"))
        d = pdf.infodict()
        d["Title"] = "Aberrometria corneana anterior calculada (pesquisa)"
        d["Subject"] = f"análise {analysis.analysis_id}"
    return path


def report_stem(analysis: Analysis) -> str:
    """Nome-base dos relatórios: código de estudo (pseudônimo) + exame + olho + prefixo da identidade do cálculo.
    Só caracteres seguros para nome de arquivo; nunca dados nominais."""
    import re
    ident = analysis.identity
    parts = [ident.study_id if ident.study_id and ident.study_id != "sem_estudo" else "", ident.exam_id, ident.eye or "X", analysis.analysis_id[:8]]
    stem = "_".join(p for p in parts if p)
    return re.sub(r"[^A-Za-z0-9_.\-]+", "_", stem)


def export_all(exam: NormalizedExam, analysis: Analysis, out_dir: str, stem: Optional[str] = None) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    stem = stem or report_stem(analysis)
    paths = {
        "json": export_json(analysis, os.path.join(out_dir, stem + ".json")),
        "csv": export_csv(analysis, os.path.join(out_dir, stem + ".csv")),
        "pdf": export_pdf(exam, analysis, os.path.join(out_dir, stem + ".pdf")),
    }
    if any(w.map_grid is not None for w in analysis.wavefronts):
        paths["npz"] = export_maps_npz(analysis, os.path.join(out_dir, stem + "_mapas.npz"))
    return paths
