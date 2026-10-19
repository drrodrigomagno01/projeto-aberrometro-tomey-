"""Relatórios (plano §7): JSON canônico, CSV e PDF derivados do mesmo objeto de resultado.

- JSON: null para ausências, nunca números não finitos; mapas não entram (ficam em arquivos NPZ opcionais).
- CSV: coeficientes assinados com índices n, m, j, unidade, perfil e identificação da análise; métricas por perfil;
  ceratometria por zona (seção própria, com cabeçalho próprio, depois das métricas).
- PDF: identificação, finalidade, perfil, números, qualidade, proveniência; mapas; tabela de Zernike.
Um perfil bloqueado gera relatório com o motivo; números indisponíveis aparecem como ausência.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
from typing import Any, Optional

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from . import CORE_VERSION
from .keratometry import COLUNAS_TABELA, LEGENDA_TABELA, eixo_tabo, fmt_eixo, fmt_k, linhas_resumo
from .model import Analysis, KeratometryResult, KeratometryZone, NormalizedExam, WavefrontResult, to_jsonable
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
        # ceratometria por zona: seção nova DEPOIS das anteriores (que são lidas por posição), com cabeçalho próprio
        w.writerow([])
        w.writerow([KERATO_CSV_MARCADOR])
        w.writerow(list(KERATO_CSV_COLUNAS))
        ident = [analysis.analysis_id, analysis.identity.study_id, analysis.identity.exam_id, analysis.identity.eye or ""]
        kres = getattr(analysis, "keratometry", None)
        if kres is None:
            w.writerow(ident + ["", "", "", "indisponivel"] + [""] * 12 + ["ceratometria não calculada nesta análise"])
        for rec in keratometry_records(kres):
            w.writerow(ident + [rec["tipo"], rec["zona"]]
                       + [_num(rec["diametro_mm"], _KERATO_CASAS["diametro_mm"]), rec["validade"]]
                       + [_num(rec[c], _KERATO_CASAS[c]) for c in KERATO_CSV_COLUNAS[8:-1]] + [rec["motivo"]])
    return path


def _fmt(v) -> str:
    try:
        return "" if v is None or not np.isfinite(v) else f"{float(v):.6f}"
    except TypeError:
        return str(v)


# ---------------------------------------------------------------------------
# Ceratometria por zona — linhas do CSV (números crus; o texto formatado vem de keratometry.linhas_resumo)
# ---------------------------------------------------------------------------
KERATO_CSV_MARCADOR = "# ceratometria por zona (D e graus, K1 = plano, K2 = curvo, potência axial do TMS, n 1,3375)"
KERATO_CSV_COLUNAS = (
    "analysis_id", "study_id", "exam_id", "eye", "tipo", "zona", "diametro_mm", "validade",
    "k1_plano_D", "eixo_k1_deg", "k2_curvo_D", "eixo_k2_deg", "delta_k_D", "k_medio_D", "j0_D", "j45_D",
    "assimetria_D", "irregularidade_D", "fracao_cobertura", "fracao_interpolados", "motivo",
)
# casas decimais (ponto decimal; vazio = ausente): potências com 6, eixos TABO com 2
_KERATO_CASAS = {
    "diametro_mm": 4, "k1_plano_D": 6, "eixo_k1_deg": 2, "k2_curvo_D": 6, "eixo_k2_deg": 2, "delta_k_D": 6,
    "k_medio_D": 6, "j0_D": 6, "j45_D": 6, "assimetria_D": 6, "irregularidade_D": 6,
    "fracao_cobertura": 4, "fracao_interpolados": 4,
}
_KERATO_NUMEROS = ("k1_plano_D", "eixo_k1_deg", "k2_curvo_D", "eixo_k2_deg", "delta_k_D", "k_medio_D", "j0_D", "j45_D",
                   "assimetria_D", "irregularidade_D", "fracao_cobertura", "fracao_interpolados")


def _num(v, casas: int) -> str:
    """Número com ponto decimal e `casas` casas; vazio para ausente/não finito (sem '-0.000000')."""
    if v is None:
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    return f"{round(f, casas) + 0.0:.{casas}f}" if math.isfinite(f) else ""


def _eixo(v) -> Optional[float]:
    """Eixo em graus TABO (0, 180] (0° vira 180°); None se ausente."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return eixo_tabo(f) if math.isfinite(f) else None


def _registro_zona(z: KeratometryZone, zona: str) -> dict[str, Any]:
    rec: dict[str, Any] = {"tipo": z.tipo, "zona": zona, "diametro_mm": z.diametro_mm, "validade": z.validity,
                           "motivo": "; ".join(z.block_reasons)}
    for c in _KERATO_NUMEROS:
        rec[c] = getattr(z, c)
    if z.validity != "disponivel" and not z.fracao_cobertura:
        # zona sem nenhum dado (fora dos anéis, pupila não registrada, DIO ausente): frações ausentes, não "0%"
        rec["fracao_cobertura"] = rec["fracao_interpolados"] = None
    return rec


def _registro_vazio(tipo: str, zona: str, motivo: str) -> dict[str, Any]:
    rec: dict[str, Any] = {"tipo": tipo, "zona": zona, "diametro_mm": None, "validade": "indisponivel", "motivo": motivo}
    rec.update({c: None for c in _KERATO_NUMEROS})
    return rec


def _registro_simk(tipo: str, zona: str, k1, e1, k2, e2, dk) -> dict[str, Any]:
    rec = _registro_vazio(tipo, zona, "")
    rec.update({"validade": "disponivel", "k1_plano_D": k1, "eixo_k1_deg": _eixo(e1), "k2_curvo_D": k2,
                "eixo_k2_deg": _eixo(e2), "delta_k_D": dk, "k_medio_D": 0.5 * (k1 + k2)})
    return rec


def kerato_chave(z: KeratometryZone) -> str:
    """Chave estável (ASCII) de uma linha da ceratometria, usada no CSV e no resumo_longo.csv do `tmsab batch`:
    circunferência "3mm_vertice", zona 0–d "zona_0-3mm_vertice", "pupila", anel "anel_7"."""
    d = f"{z.diametro_mm:g}" if z.diametro_mm is not None else "?"
    if z.tipo == "circunferencia":
        return f"{d}mm_vertice"
    if z.tipo == "zona":
        return f"zona_0-{d}mm_vertice"
    if z.tipo == "pupila":
        return "pupila"
    if z.tipo == "anel":
        return f"anel_{z.anel}"
    return z.tipo


def keratometry_records(kres: Optional[KeratometryResult]) -> list[dict[str, Any]]:
    """Linhas da seção de ceratometria do CSV, na ordem: circunferências (Ø fixo, centro no vértice), zonas 0–d
    (média por área, centro no vértice), zona pupilar, SimK recalculado (regra do TMS), SimK do XREF e anéis
    medidos. Números crus (None = ausente).

    `tipo`: circunferencia | zona | pupila | simk_calculado | simk_tms | anel; `zona`: chave estável (kerato_chave).
    K1 = meridiano mais plano, K2 = mais curvo (no XREF, SimK1 é o curvo: aqui vira k2_curvo_D). Eixos em graus TABO
    (0, 180]. Nas linhas de SimK, Km = (K1 + K2)/2 e J0/J45, assimetria, irregularidade e frações ficam vazios (a regra
    do TMS não os define)."""
    if kres is None:
        return []
    out = [_registro_zona(z, kerato_chave(z)) for z in list(kres.zonas) + list(getattr(kres, "discos", None) or [])]
    if kres.pupila is not None:
        out.append(_registro_zona(kres.pupila, "pupila"))
    else:
        out.append(_registro_vazio("pupila", "pupila", "zona pupilar desligada na configuração (zona_pupilar = nenhuma)"))
    aneis = "_".join(str(a) for a in kres.config.simk_aneis)
    sc = kres.simk_calculado
    if sc and all(sc.get(c) is not None for c in ("k1_plano_D", "k2_curvo_D")):
        out.append(_registro_simk("simk_calculado", f"regra_tms_aneis_{aneis}", sc["k1_plano_D"], sc.get("eixo_k1_deg"),
                                  sc["k2_curvo_D"], sc.get("eixo_k2_deg"), sc.get("delta_k_D", sc["k2_curvo_D"] - sc["k1_plano_D"])))
    else:
        motivo = next((n.split(": ", 1)[-1] for n in kres.notes
                       if n.startswith(("ceratometria indisponível", "SimK recalculado indisponível"))),
                      f"regra do TMS sem meridianos completos nos anéis {aneis.replace('_', ', ')}")
        out.append(_registro_vazio("simk_calculado", f"regra_tms_aneis_{aneis}", motivo))
    t = kres.simk_tms or {}
    if {"SimK1", "SimK2"} <= set(t):
        # XREF do TMS: SimK1 = meridiano CURVO, SimK2 = plano
        out.append(_registro_simk("simk_tms", "xref_tms", t["SimK2"], t.get("SimK2 Ang."), t["SimK1"], t.get("SimK1 Ang."),
                                  t.get("CYL", t["SimK1"] - t["SimK2"])))
    else:
        out.append(_registro_vazio("simk_tms", "xref_tms", "XREF sem SimK1/SimK2 para este exame"))
    out.extend(_registro_zona(z, kerato_chave(z)) for z in kres.aneis)
    return out


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
        width = 96 if indent else 100      # texto corrido em DejaVu Sans 8,5 pt: 100 caracteres cabem até ~0,93 da folha
        # recuo da continuação: espaços comuns nas linhas monoespaçadas; no texto corrido, espaços não separáveis —
        # dois espaços comuns no início fariam _text_page usar a fonte monoespaçada (mais larga: a linha saía da folha)
        sub = "    " if indent else chr(0xA0) * 4
        parts = textwrap.wrap(ln.strip(), width=width, subsequent_indent=sub) or [""]
        out.extend(indent + p for p in parts)
    return out


_Y_TOPO, _Y_BASE, _H_LINHA, _H_TITULO = 0.91, 0.05, 0.0165, 0.024     # layout das páginas de texto (fração da folha)


def _text_page(lines: list[str], title: str) -> Figure:
    fig = Figure(figsize=(8.27, 11.69))
    fig.text(0.07, 0.95, title, fontsize=13, weight="bold", va="top")
    y = _Y_TOPO
    for ln in lines:
        if y < _Y_BASE:
            break
        size = 8.5
        weight = "bold" if ln.startswith("## ") else "normal"
        txt = ln[3:] if ln.startswith("## ") else ln
        fig.text(0.07, y, txt, fontsize=size, va="top", weight=weight, family="monospace" if ln.startswith("  ") else "sans-serif")
        y -= _H_LINHA if not ln.startswith("## ") else _H_TITULO
    return fig


def _paginar(lines: list[str]) -> list[list[str]]:
    """Divide as linhas em páginas pela altura ocupada (títulos são mais altos que as linhas comuns): nenhuma linha
    fica abaixo da margem (antes, 50 linhas fixas por página podiam perder a última com muitos títulos) e um título
    nunca fica sozinho no pé da página."""
    pages: list[list[str]] = []
    cur: list[str] = []
    y = _Y_TOPO
    for ln in lines:
        titulo = ln.startswith("## ")
        h = _H_TITULO if titulo else _H_LINHA
        precisa = h + (_H_LINHA if titulo else 0.0)
        if cur and y - precisa < _Y_BASE - 1e-9:
            pages.append(cur)
            cur, y = [], _Y_TOPO
        cur.append(ln)
        y -= h
    if cur:
        pages.append(cur)
    return pages


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
    L.append(f"Pupila: {a.pupil.state}" + (f" — centro ({a.pupil.center_x:+.2f}, {a.pupil.center_y:+.2f}) mm, diâmetro {_mm2(a.pupil.diameter)} mm, referencial {a.pupil.frame}, método {a.pupil.method}" if a.pupil.state == "registrada" else ""))
    L.append(f"Referencial: {a.frame.axes}; origem {a.frame.origin}; altura {a.frame.height_reference}; ângulo inicial {a.frame.angle_start_deg:g}°, sentido {'anti-horário' if a.frame.angle_direction == 1 else 'horário'}; espelho x: {a.frame.mirror_x}; transformação verificada: {'sim' if a.frame.transform_verified else 'NÃO'}")
    L.extend(keratometry_report_lines(getattr(a, "keratometry", None)))
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
    L.append(kerato_limits_text(qc))
    for wmsg in a.warnings:
        L.append(f"AVISO: {wmsg}")
    L.append("Nenhuma faixa de normalidade é atribuída: limites clínicos exigem estudo próprio (plano §9).")
    L.append("## Proveniência")
    L.append(f"Identificador da análise: {a.analysis_id}   Executada em: {a.created_at}")
    L.append("Versões: " + ", ".join(f"{k}={v}" for k, v in a.versions.items()))
    L.append(f"Reconstrução: {dataclasses.asdict(a.reconstruction)}")
    kres = getattr(a, "keratometry", None)
    L.append(f"Ceratometria: {dataclasses.asdict(kres.config)}" if kres is not None
             else "Ceratometria: não calculada nesta análise")
    for k, v in a.bundle_hashes.items():
        L.append(f"  {k}: sha256 {v[:16]}…{v[-8:]}")
    for d in a.import_diagnostics:
        L.append(f"  diagnóstico: {d}")
    return L


def _f2(v) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:+.2f}"


def _mm2(v) -> str:
    """Comprimento em mm com 2 casas ('—' se ausente ou zero)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    return f"{f:.2f}" if math.isfinite(f) and f > 0 else "—"


# ---------------------------------------------------------------------------
# Ceratometria por zona — seção do relatório em texto (PDF técnico, aba Técnico, "Relatório em texto" do HTML)
# ---------------------------------------------------------------------------
KERATO_TITULO = "Ceratometria por zona (potência axial do TMS, n 1,3375)"
_KZ, _KN, _KE = 34, 12, 7          # larguras: zona, potência, eixo (linha ≤ 96 caracteres: cabe no PDF sem quebrar)
_ESTADO_PT = {"disponivel": "disponível", "indisponivel": "INDISPONÍVEL", "invalida": "INVÁLIDA"}


def _linha_tabela(rotulo: str, cel: list[str]) -> str:
    k1, e1, k2, e2, dk, km = cel
    return (f"  {rotulo:<{_KZ}}{k1:>{_KN}}  {e1:>{_KE}}  {k2:>{_KN}}  {e2:>{_KE}}  {dk:>6}  {km:>6}").rstrip()


def _linhas_tabela(linhas: list[dict[str, Any]]) -> list[str]:
    out = []
    for ln in linhas:
        if ln["disponivel"]:
            out.append(_linha_tabela(ln["rotulo"], list(ln["texto"])))
        else:
            estado = _ESTADO_PT.get(ln["validity"], ln["validity"])
            out.append(f"  {ln['rotulo']:<{_KZ}}{estado}" + (f" — {ln['motivo']}" if ln["motivo"] else ""))
    return out


def _k_eixo(k, eixo) -> str:
    return f"{fmt_k(k)} D @ {fmt_eixo(eixo)}"


def keratometry_report_lines(kres: Optional[KeratometryResult]) -> list[str]:
    """Seção "## Ceratometria por zona": tabela completa (circunferências, zonas 0–d, zona pupilar, SimK do TMS),
    SimK recalculado, índices do XREF, conferência SimK/ACP recalculados × TMS, alertas, notas e os anéis medidos.
    Linhas com dois espaços à esquerda saem em fonte monoespaçada no PDF (colunas alinhadas)."""
    L = [f"## {KERATO_TITULO}"]
    if kres is None:
        L.append("Ceratometria não calculada nesta análise (resultado anterior à versão 0.4.0).")
        return L
    L.append(LEGENDA_TABELA)
    linhas = linhas_resumo(kres, incluir_aneis=True)
    L.append(_linha_tabela(COLUNAS_TABELA[0], list(COLUNAS_TABELA[1:])))
    L.extend(_linhas_tabela([ln for ln in linhas if ln["tipo"] != "anel"]))
    sc = kres.simk_calculado
    aneis = "–".join(str(a) for a in (kres.config.simk_aneis[0], kres.config.simk_aneis[-1])) if kres.config.simk_aneis else ""
    if sc:
        L.append(f"SimK recalculado (regra do TMS, anéis {aneis}): K1 plano {_k_eixo(sc.get('k1_plano_D'), sc.get('eixo_k1_deg'))}, "
                 f"K2 curvo {_k_eixo(sc.get('k2_curvo_D'), sc.get('eixo_k2_deg'))}, ΔK {fmt_k(sc.get('delta_k_D'))} D; "
                 f"MinK {_k_eixo(sc.get('mink_D'), sc.get('eixo_mink_deg'))}")
    else:
        mot = next((n.split(": ", 1)[-1] for n in kres.notes
                    if n.startswith(("ceratometria indisponível", "SimK recalculado indisponível"))), "")
        L.append(f"SimK recalculado (regra do TMS, anéis {aneis}): indisponível" + (f" — {mot}" if mot else ""))
    t = kres.simk_tms or {}
    extras = []
    if "MinK" in t:
        extras.append(f"MinK {_k_eixo(t['MinK'], t.get('MinK Ang.'))}")
    if "CYL" in t:
        extras.append(f"CYL {fmt_k(t['CYL'])} D")
    if "ACP" in t:
        # o TMS grava ACP = 0 quando não detecta a pupila: não é uma potência
        extras.append(f"ACP {fmt_k(t['ACP'])} D" if t["ACP"] > 0 else "ACP não informado (0 no XREF: pupila não detectada)")
    if "3mm Reg. Astig." in t:
        # "Reg. Astig." do XREF = amplitude do 2º harmônico (metade do cilindro); o eixo é o do meridiano curvo
        extras.append(f"Fourier 3 mm: astig. regular {fmt_k(2 * t['3mm Reg. Astig.'])} D (2 × amplitude "
                      f"{fmt_k(t['3mm Reg. Astig.'])}) @ {fmt_eixo(t.get('3mm Reg. Astig. Axis'))} (meridiano curvo)")
    L.append("Índices do TMS (XREF): " + ("; ".join(extras) if extras else "indisponíveis para este exame")
             + (" — no XREF, SimK1 é o meridiano curvo (K2 acima) e SimK2 o plano (K1)" if {"SimK1", "SimK2"} <= set(t) else ""))
    if kres.conferencia:
        L.extend(f"Conferência: {c}" for c in kres.conferencia)
    else:
        L.append("Conferência com o TMS: sem SimK no XREF para comparar")
    alertas = list(getattr(kres, "alertas", None) or [])
    L.extend(f"Alerta (informativo, não é recomendação): {a}" for a in alertas)
    L.extend(f"Nota: {n}" for n in kres.notes if n not in alertas)
    if kres.pupila is not None:
        L.extend(f"Nota (pupila): {n}" for n in kres.pupila.notes)
    aneis_linhas = [ln for ln in linhas if ln["tipo"] == "anel"]
    if aneis_linhas:
        L.append(f"  Anéis do Plácido medidos ({len(aneis_linhas)}), no diâmetro medido (2 × RAD médio dos pontos medidos):")
        L.extend(_linhas_tabela(aneis_linhas))
    return L


def kerato_limits_text(qc) -> str:
    """Limites provisórios da ceratometria (QualityConfig), para a seção de qualidade."""
    eixo_min = getattr(qc, "k_eixo_min_cyl_D", 0.75)
    txt = (f"Limites da ceratometria (provisórios): meridianos (ou área da zona) com dados ≥{qc.k_min_meridian_fraction:.0%}, "
           f"maior setor sem dados ≤{qc.k_max_missing_sector_deg:.0f}°; marca † com ≥{qc.k_interp_mark_fraction:.0%} dos meridianos "
           f"(ou da área) usando pontos interpolados pelo aparelho; * eixo pouco definido com ΔK < {qc.k_axis_min_cyl_D:.2f} D; "
           f"conferência SimK × XREF ±{qc.k_simk_tol_D:.2f} D e ±{qc.k_simk_axis_tol_deg:.0f}° (eixo só com CYL ≥ {eixo_min:.2f} D)")
    if hasattr(qc, "k_zona_alerta_dif_D"):
        txt += (f"; alerta de astigmatismo dependente da zona (0–3 mm × pupila) com diferença vetorial > {qc.k_zona_alerta_dif_D:.2f} D "
                f"ou de eixo > {getattr(qc, 'k_zona_alerta_eixo_deg', 10.0):.0f}° (ΔK ≥ {eixo_min:.2f} D nas duas)")
    return txt


def export_pdf(exam: NormalizedExam, analysis: Analysis, path: str, patient_name: Optional[str] = None) -> str:
    """PDF: página 1 = resultado completo (com nome do paciente e data, se `patient_name` for dado — opção do usuário);
    demais páginas = relatório técnico só com o código do estudo."""
    with PdfPages(path) as pdf:
        title = f"Aberrometria corneana anterior calculada — relatório de pesquisa (tmsab {CORE_VERSION})"
        # página 1: resultado completo (topografia + frente de onda + tabelas) — pronta para impressão
        try:
            pdf.savefig(figure_resultado_completo(exam, analysis, patient_name=patient_name))
        except Exception as e:  # noqa: BLE001 — a página-resumo nunca impede o relatório completo
            pdf.savefig(_text_page([f"Página de resultado completo indisponível: {e}"], title))
        lines = _wrap(report_lines(analysis))
        for i, page in enumerate(_paginar(lines)):
            pdf.savefig(_text_page(page, title if i == 0 else title + " (cont.)"))
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


def export_all(exam: NormalizedExam, analysis: Analysis, out_dir: str, stem: Optional[str] = None,
               patient_name: Optional[str] = None) -> dict[str, str]:
    """JSON e CSV nunca levam o nome; o PDF leva-o apenas na página 1 quando `patient_name` é informado."""
    os.makedirs(out_dir, exist_ok=True)
    stem = stem or report_stem(analysis)
    paths = {
        "json": export_json(analysis, os.path.join(out_dir, stem + ".json")),
        "csv": export_csv(analysis, os.path.join(out_dir, stem + ".csv")),
        "pdf": export_pdf(exam, analysis, os.path.join(out_dir, stem + ".pdf"), patient_name=patient_name),
    }
    if any(w.map_grid is not None for w in analysis.wavefronts):
        paths["npz"] = export_maps_npz(analysis, os.path.join(out_dir, stem + "_mapas.npz"))
    return paths
