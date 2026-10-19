"""Versão HTML (navegador): funções chamadas pelo arquivo AberrometriaTMS.html via Pyodide.

Tudo roda dentro do navegador (WebAssembly) — sem servidor, sem banco de dados, nenhum dado sai do computador.
Este módulo é Python puro (NumPy/SciPy/Matplotlib) e também roda em CPython, o que permite testá-lo na suíte.

Contrato (JSON-ável em ambos os sentidos):
- list_folder(directory)                -> {"exams": [...], "patients": [...], "warnings": [...]}
- analyze_exam_html(directory, exam_id, options) -> {"summary": {...}, "html": "...", "json": "...", "csv": "...",
                                                     "pdf_b64": "...", "stem": "..."}
"""
from __future__ import annotations

import base64
import html
import io
import os
import tempfile
from typing import Any, Optional

import matplotlib
import numpy as np
from matplotlib.figure import Figure

from . import CORE_VERSION
from .adapters import AdapterError, import_tms_bundle, list_exams, patient_display_name
from .keratometry import COLUNAS_TABELA, LEGENDA_TABELA, eixo_tabo, linhas_resumo
from .model import Analysis, NormalizedExam, QualityConfig, ReconstructionConfig, to_jsonable
from .pupila import fmt_chord_mu
from .pipeline import analyze_exam, default_profiles
from .plots import figure_resultado_completo, plot_coverage, plot_elevation, plot_wavefront, plot_zernike_bars
from .reporting import (
    KERATO_TITULO,
    canonical_result,
    export_csv,
    export_csv_ceratometria,
    export_json,
    export_pdf,
    kerato_limits_text,
    report_lines,
    report_stem,
)

matplotlib.use("Agg")

DISCLAIMER = ("Ferramenta de pesquisa — aberrometria corneana ANTERIOR calculada a partir das Data Tables do Tomey TMS. "
              "Não é aberrometria ocular total, não é alegação diagnóstica, não serve para seleção de LIO nem planejamento "
              "terapêutico. Resultados vinculados a abertura, centro, modelo óptico e versões registrados.")


def _e(v: Any) -> str:
    return html.escape("" if v is None else str(v))


def _fmt(v: Any, nd: int = 3, sign: bool = False) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return _e(v)
    if not np.isfinite(f):
        return "—"
    return f"{f:+.{nd}f}" if sign else f"{f:.{nd}f}"


def _png(fig: Figure, dpi: int = 110) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# listagem
# ---------------------------------------------------------------------------
def list_folder(directory: str) -> dict[str, Any]:
    """Exames de uma pasta exportada (ou de um pacote único)."""
    try:
        exams = list_exams(directory)
    except Exception as e:  # noqa: BLE001
        return {"exams": [], "patients": [], "warnings": [f"não foi possível listar a pasta: {e}"]}
    out = []
    for e in exams:
        out.append({"exam_id": e["exam_id"], "eye": e.get("eye"), "acquired_at": e.get("acquired_at"),
                    "capture_index": e.get("capture_index"), "patient_key": e.get("patient_key"),
                    "n_files": len(e.get("files", [])),
                    "simk": {k: e.get("indices", {}).get(k) for k in ("SimK1", "SimK1 Ang.", "SimK2", "SimK2 Ang.", "CYL")}})
    patients = sorted({e["patient_key"] for e in out if e.get("patient_key")})
    warnings = []
    if len(patients) > 1:
        warnings.append(f"a pasta contém exames de {len(patients)} pacientes diferentes (o TMS acrescenta linhas ao XREF.dat "
                        "ao exportar de novo para a mesma pasta); cada exame recebe o código do seu paciente")
    if not out and os.path.isdir(directory) and any(n.upper().startswith(("HIT", "RAD")) for n in os.listdir(directory)):
        out.append({"exam_id": None, "eye": None, "acquired_at": None, "capture_index": None, "patient_key": None, "n_files": 0, "simk": {}})
    return {"exams": out, "patients": patients, "warnings": warnings}


# ---------------------------------------------------------------------------
# relatório HTML (imprimível)
# ---------------------------------------------------------------------------
def _metrics_table(an: Analysis) -> str:
    rows = []
    for wf in an.wavefronts:
        p = wf.profile
        label = f"{p.center_mode} {p.diameter_mm:g} mm"
        if wf.validity != "disponivel":
            reasons = "; ".join(wf.block_reasons) or wf.validity
            rows.append(f"<tr><td>{_e(label)}</td><td class='estado-{_e(wf.validity)}'>{_e(wf.validity.upper())}</td>"
                        f"<td colspan='7' class='motivo'>{_e(reasons)} — números indisponíveis (ausência, não zero)</td></tr>")
            continue
        m = wf.metrics
        cil = m.get("cilindro_equivalente_D")
        eixo = m.get("eixo_equivalente_deg")
        cil_txt = f"{cil:+.2f} D × {eixo:.0f}°" if cil is not None and eixo is not None and np.isfinite(cil) else "—"
        rows.append(
            "<tr>"
            f"<td>{_e(label)}</td><td class='estado-disponivel'>DISPONÍVEL</td>"
            f"<td>{_fmt(m.get('sa_c40'), 3, True)}</td>"
            f"<td>{_fmt(m.get('coma_magnitude'), 3)}</td>"
            f"<td>{_fmt(m.get('trefoil_magnitude'), 3)}</td>"
            f"<td>{_fmt(m.get('rms_hoa_n3plus'), 3)}</td>"
            f"<td>{_fmt(m.get('sa_secundaria_c60'), 3, True)}</td>"
            f"<td>{_e(cil_txt)}</td>"
            f"<td>{wf.n_valid_rays}/{wf.n_rays}</td>"
            "</tr>"
        )
    return ("<table class='metricas'><thead><tr><th>Perfil (centro, abertura)</th><th>Estado</th><th>SA c(4,0) µm</th>"
            "<th>|Coma| µm</th><th>|Trefoil| µm</th><th>RMS HOA (n≥3) µm</th><th>SA 2ª c(6,0) µm</th>"
            "<th>Astig. equiv. (refrativo)</th><th>Raios</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def _zernike_table(wf) -> str:
    cells = [f"<tr><td>{c.j}</td><td>{c.n}</td><td>{c.m:+d}</td><td>{c.value_um:+.4f}</td><td>{_e(c.name)}</td></tr>"
             for c in wf.coefficients]
    return ("<table class='zernike'><thead><tr><th>j</th><th>n</th><th>m</th><th>µm</th><th>termo</th></tr></thead><tbody>"
            + "".join(cells) + "</tbody></table>")


_KERATO_CLASSE = {"pupila": "kerato-pupila", "referencia": "kerato-ref", "zona": "kerato-zona"}
_ESTADO_PT = {"indisponivel": "INDISPONÍVEL", "invalida": "INVÁLIDA"}


def _kerato_rows(linhas: list[dict[str, Any]]) -> str:
    rows = []
    for ln in linhas:
        cls = _KERATO_CLASSE.get(ln.get("destaque") or "", "")
        attr = f" class='{cls}'" if cls else ""
        if ln["disponivel"]:
            rows.append(f"<tr{attr}><td>{_e(ln['rotulo'])}</td>" + "".join(f"<td>{_e(t)}</td>" for t in ln["texto"]) + "</tr>")
        else:
            estado = _ESTADO_PT.get(ln["validity"], str(ln["validity"]).upper())
            rows.append(f"<tr{attr}><td>{_e(ln['rotulo'])}</td><td colspan='6' class='motivo'>{_e(estado)}"
                        + (f" — {_e(ln['motivo'])}" if ln["motivo"] else "") + "</td></tr>")
    return "".join(rows)


def _kerato_table(an: Analysis) -> str:
    """Tabela da ceratometria por zona (classe 'metricas'): circunferências, zonas 0–d, zona pupilar (linha destacada),
    SimK do TMS (cinza); anéis medidos num <details> recolhível; conferência, alertas e notas. Sem nome do paciente."""
    k = getattr(an, "keratometry", None)
    if k is None:
        return "<p class='aviso'>Ceratometria não calculada nesta análise.</p>"
    linhas = linhas_resumo(k, incluir_aneis=True)
    head = "<thead><tr>" + "".join(f"<th>{_e(c)}</th>" for c in COLUNAS_TABELA) + "</tr></thead>"
    principal = [ln for ln in linhas if ln["tipo"] != "anel"]
    aneis = [ln for ln in linhas if ln["tipo"] == "anel"]
    html_ = f"<table class='metricas kerato'>{head}<tbody>{_kerato_rows(principal)}</tbody></table>"
    html_ += f"<p class='legenda'>{_e(LEGENDA_TABELA)}</p>"
    if aneis:
        html_ += (f"<details class='kerato-aneis'><summary>Anéis do Plácido medidos ({len(aneis)}), no diâmetro medido "
                  f"(2 × RAD médio)</summary><table class='metricas kerato'>{head}<tbody>{_kerato_rows(aneis)}</tbody></table></details>")
    alertas = list(getattr(k, "alertas", None) or [])
    itens = [f"<li class='aviso'>Alerta (informativo, não é recomendação): {_e(a)}</li>" for a in alertas]
    itens += [f"<li>Conferência: {_e(c)}</li>" for c in k.conferencia]
    itens += [f"<li>{_e(n)}</li>" for n in k.notes if n not in alertas]
    if k.pupila is not None:
        itens += [f"<li>{_e(n)}</li>" for n in k.pupila.notes]
    if itens:
        html_ += f"<ul class='kerato-notas'>{''.join(itens)}</ul>"
    return html_


def _num(v: Any) -> Optional[float]:
    """Número finito (float) ou None — o JSON da página não aceita NaN."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _kerato_summary(an: Analysis) -> Optional[dict[str, Any]]:
    """Ceratometria resumida para o JSON da página: linhas da folha A4 (números; eixos TABO 1–180), conferência e
    alertas. K1 = plano, K2 = curvo (no XREF, SimK1 é o curvo)."""
    k = getattr(an, "keratometry", None)
    if k is None:
        return None
    linhas = []
    for ln in linhas_resumo(k):
        e1, e2 = _num(ln["eixo_k1"]), _num(ln["eixo_k2"])
        linhas.append({
            "rotulo": ln["rotulo"], "tipo": ln["tipo"], "destaque": ln["destaque"], "validity": ln["validity"],
            "motivo": ln["motivo"], "diametro_mm": _num(ln["diametro_mm"]),
            "k1_plano_D": _num(ln["k1"]), "eixo_k1_deg": eixo_tabo(e1) if e1 is not None else None,
            "k2_curvo_D": _num(ln["k2"]), "eixo_k2_deg": eixo_tabo(e2) if e2 is not None else None,
            "delta_k_D": _num(ln["dk"]), "k_medio_D": _num(ln["km"]),
        })
    return {"linhas": linhas, "conferencia": list(k.conferencia), "alertas": list(getattr(k, "alertas", None) or [])}


def report_html(exam: NormalizedExam, an: Analysis, with_maps: bool = True, patient_name: Optional[str] = None) -> str:
    """Fragmento HTML de UM exame (seção <article>), com figuras PNG embutidas; imprimível.
    `patient_name` (opção do usuário) aparece no cabeçalho e na página de resultado completo; nunca no JSON/CSV."""
    from .plots import _data_pt
    ident = an.identity
    g = an.geometry
    cap = f"captura {ident.capture_index}" if ident.capture_index is not None else "captura —"
    head = (
        f"<header><h2>{(_e(patient_name) + ' — ') if patient_name else ''}Exame {_e(ident.exam_id)} — olho {_e(ident.eye or '?')} — {_e(cap)}</h2>"
        f"<div class='ident'><span><b>Código do estudo:</b> {_e(ident.study_id)}</span>"
        f"<span><b>Data do exame:</b> {_e(_data_pt(ident.acquired_at))}</span>"
        f"<span><b>Equipamento:</b> {_e(an.device.model)} / sw {_e(an.device.software_version)}</span>"
        f"<span><b>Análise:</b> <code>{_e(an.analysis_id)}</code> em {_e(an.created_at)}</span>"
        f"<span><b>Núcleo:</b> tmsab {_e(CORE_VERSION)} (navegador)</span></div>"
        + ("<p class='sintetico'>*** EXAME SINTÉTICO — gerado por software; não corresponde a um paciente. ***</p>" if an.synthetic else "")
        + "</header>"
    )
    geo = (
        "<section><h3>Geometria reconstruída</h3><ul>"
        f"<li>Método {_e(g.method)} (ordem {g.surface_order}, raio de ajuste {g.fit_radius_mm:.2f} mm, {g.n_points_used} pontos)</li>"
        f"<li>Cônica de referência: R = {g.conic_R_mm:.4f} mm, Q = {g.conic_Q:+.4f} (K equivalente {g.k_mean_D_keratometric:.2f} D, só conferência)</li>"
        f"<li>Resíduo da reconstrução: RMS {g.residual_rms_mm*1000:.3f} µm, máx {g.residual_max_mm*1000:.3f} µm; lacuna central {g.central_gap_mm:.2f} mm; "
        f"raio externo {g.outer_radius_mm:.2f} mm; fração medida {g.measured_fraction:.0%}</li>"
        f"<li>Pupila: {_e(an.pupil.state)}"
        + (f" — centro ({an.pupil.center_x:+.2f}, {an.pupil.center_y:+.2f}) mm, diâmetro {an.pupil.diameter:.2f} mm ({_e(an.pupil.method)})"
           if an.pupil.state == "registrada" and an.pupil.diameter else (f" — {_e(an.pupil.method)}" if an.pupil.method else ""))
        + (f"; chord mu aparente {_e(fmt_chord_mu(an.chord_mu))}" if getattr(an, "chord_mu", None) is not None else "")
        + "</li></ul></section>"
    )
    metrics = "<section><h3>Resultados por perfil</h3>" + _metrics_table(an) + "</section>"
    kerato = f"<section class='kerato'><h3>{_e(KERATO_TITULO)}</h3>{_kerato_table(an)}</section>"

    completo = ""
    if with_maps:
        try:
            completo = (f"<figure class='completo'><img src='{_png(figure_resultado_completo(exam, an, patient_name=patient_name), dpi=150)}' "
                        "alt='resultado completo (uma página)'></figure>")
        except Exception as e:  # noqa: BLE001
            completo = f"<p class='aviso'>Página de resultado completo indisponível: {_e(e)}</p>"

    figs = ""
    if with_maps:
        fig = Figure(figsize=(9, 4.2))
        ax1, ax2 = fig.subplots(1, 2)
        plot_elevation(ax1, exam)
        aps = [(w.center_xy[0], w.center_xy[1], w.profile.radius_mm) for w in an.wavefronts if w.validity == "disponivel"]
        plot_coverage(ax2, exam, apertures=aps)
        fig.tight_layout()
        figs += f"<figure class='geo'><img src='{_png(fig)}' alt='elevação e cobertura'><figcaption>Elevação residual e cobertura dos pontos (aberturas dos perfis disponíveis)</figcaption></figure>"
        for wf in an.wavefronts:
            if wf.validity != "disponivel" or wf.map_grid is None:
                continue
            fig = Figure(figsize=(9, 7))
            axs = fig.subplots(2, 2)
            plot_wavefront(axs[0, 0], wf, "W")
            plot_wavefront(axs[0, 1], wf, "W_fit")
            plot_wavefront(axs[1, 0], wf, "residual")
            plot_zernike_bars(axs[1, 1], wf)
            fig.tight_layout()
            p = wf.profile
            figs += (f"<figure class='perfil'><img src='{_png(fig)}' alt='mapas {_e(p.name)}'>"
                     f"<figcaption>Perfil {_e(p.name)} — centro {p.center_mode} ({wf.center_xy[0]:+.2f}, {wf.center_xy[1]:+.2f}) mm, "
                     f"abertura {p.diameter_mm:g} mm, Zernike OSA ordem {p.zernike_order}; ajuste RMS {wf.fit_residual_rms_um:.2e} µm</figcaption>"
                     f"<details class='zern'><summary>Coeficientes de Zernike (µm)</summary>{_zernike_table(wf)}</details></figure>")
    figs = f"<section class='figuras'>{figs}</section>" if figs else ""

    qc = an.quality_config
    av = "".join(f"<li class='aviso'>{_e(w)}</li>" for w in an.warnings)
    diag = "".join(f"<li>{_e(d)}</li>" for d in an.import_diagnostics)
    hashes = "".join(f"<li><code>{_e(k)}</code>: sha256 {_e(v[:16])}…{_e(v[-8:])}</li>" for k, v in an.bundle_hashes.items())
    qual = (
        "<section class='qualidade'><h3>Qualidade, limitações e proveniência</h3>"
        f"<p>Limites de disponibilidade (provisórios, não clínicos): medidos ≥{qc.min_measured_fraction:.0%}, utilizáveis ≥{qc.min_usable_fraction:.0%}, "
        f"borda ≥{qc.min_edge_meridians_fraction:.0%} dos meridianos, setor ≤{qc.max_missing_sector_deg:.0f}°, lacuna central ≤{qc.max_central_gap_mm:.2f} mm, "
        f"resíduo ≤{qc.max_surface_residual_rms_um:.1f} µm. Nenhuma faixa de normalidade é atribuída.</p>"
        + f"<p>{_e(kerato_limits_text(qc))}.</p>"
        + (f"<ul>{av}</ul>" if av else "")
        + f"<p>Versões: {_e(', '.join(f'{k}={v}' for k, v in an.versions.items()))}</p>"
        + f"<details><summary>Arquivos de origem e diagnósticos da importação</summary><ul>{hashes}{diag}</ul></details>"
        + "</section>"
    )
    full_text = "<details class='texto'><summary>Relatório em texto (idêntico ao PDF)</summary><pre>" + _e("\n".join(report_lines(an))) + "</pre></details>"
    return (f"<article class='exame' data-exam='{_e(ident.exam_id)}' data-eye='{_e(ident.eye or '')}'>"
            + head + completo + "<div class='detalhes'>" + metrics + geo + kerato + figs + qual + full_text
            + f"<p class='rodape'>{_e(DISCLAIMER)}</p></div></article>")


def report_document(articles: list[str], title: str = "Aberrometria corneana anterior calculada — relatório de pesquisa") -> str:
    """Documento HTML completo e autônomo (para salvar/imprimir) com os artigos gerados."""
    return ("<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'><title>" + _e(title) + "</title><style>"
            + REPORT_CSS + "</style></head><body><main class='relatorio'><h1>" + _e(title) + "</h1>"
            + "".join(articles) + "</main></body></html>")


REPORT_CSS = """
.relatorio { font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color: #111; max-width: 1100px; margin: 0 auto; padding: 12px; font-size: 13px; }
.relatorio h1 { font-size: 18px; margin: 6px 0 12px; }
.exame { border: 1px solid #ccd; border-radius: 8px; padding: 12px 16px; margin: 14px 0; background: #fff; }
.exame h2 { font-size: 16px; margin: 0 0 6px; }
.exame h3 { font-size: 13.5px; margin: 12px 0 4px; border-bottom: 1px solid #dde; padding-bottom: 2px; }
.ident { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 12px; color: #333; }
.ident code { font-size: 11px; }
.sintetico { color: #a00; font-weight: bold; }
table.metricas, table.zernike { border-collapse: collapse; width: 100%; font-size: 12px; }
table.metricas th, table.metricas td, table.zernike th, table.zernike td { border: 1px solid #ccd; padding: 3px 6px; text-align: center; }
table.metricas th { background: #eef1f7; }
table.metricas td:first-child { text-align: left; }
.estado-disponivel { color: #1b5e20; font-weight: bold; }
.estado-indisponivel, .estado-invalida { color: #8B0000; font-weight: bold; }
.motivo { text-align: left !important; color: #555; }
table.kerato tr.kerato-pupila td { background: #fff3cd; font-weight: bold; }
table.kerato tr.kerato-pupila td.motivo { font-weight: normal; }
table.kerato tr.kerato-zona td { background: #eef6ee; }
table.kerato tr.kerato-ref td { color: #777; }
.kerato p.legenda, ul.kerato-notas li { font-size: 11px; color: #444; }
ul.kerato-notas { margin: 4px 0; padding-left: 18px; }
details.kerato-aneis { font-size: 12px; margin: 4px 0; }
details.kerato-aneis > summary { cursor: pointer; color: #1a4f8a; }
.figuras { display: flex; flex-wrap: wrap; gap: 10px; }
figure { margin: 6px 0; }
figure img { max-width: 100%; height: auto; border: 1px solid #eee; }
figure.completo { margin: 8px 0 12px; text-align: center; }
figure.completo img { max-width: 100%; width: 900px; border: 1px solid #ccd; }
figure.geo img { max-width: 720px; }
figure.perfil img { max-width: 720px; }
figcaption { font-size: 11px; color: #444; }
details.zern { font-size: 11px; }
details.zern table { width: auto; }
.qualidade p, .qualidade li { font-size: 11.5px; color: #333; }
li.aviso { color: #8a6d00; }
details.texto pre { font-size: 10.5px; white-space: pre-wrap; background: #f7f7f9; padding: 8px; border: 1px solid #e3e3e8; }
.rodape { font-size: 10.5px; color: #555; border-top: 1px solid #dde; padding-top: 6px; margin-top: 10px; }
@media print {
  @page { size: A4; margin: 10mm; }
  body { background: #fff; }
  .relatorio { max-width: none; padding: 0; font-size: 11px; }
  .exame { border: none; padding: 0; margin: 0; page-break-after: always; break-after: page; }
  .exame:last-child { page-break-after: auto; break-after: auto; }
  figure { break-inside: avoid; page-break-inside: avoid; }
  figure.completo { margin: 0; page-break-after: always; break-after: page; }
  figure.completo img { width: auto; max-width: 186mm; max-height: 268mm; border: none; }
  body.so-completo .detalhes, body.so-completo h1, body.so-completo article.exame > header { display: none !important; }
  body.so-completo figure.completo { page-break-after: always; }
  figure.geo img { max-width: 170mm; }
  figure.perfil img { max-width: 170mm; }
  details.texto { display: none; }
  details.zern { display: block; }
  details.zern[open], details.zern { }
  details.zern > summary { display: none; }
  details.zern table { display: table; columns: 2; }
  .no-print { display: none !important; }
}
"""


# ---------------------------------------------------------------------------
# análise de um exame → pacote de resultados
# ---------------------------------------------------------------------------
def _options(options: Optional[dict[str, Any]]) -> tuple[list, ReconstructionConfig, bool, bool]:
    """include_name (padrão True): nome do paciente do XREF no cabeçalho/página impressa (nunca em JSON/CSV)."""
    o = dict(options or {})
    diam = [float(d) for d in o.get("diameters", [4.0, 6.0])]
    centers = tuple(o.get("centers", ["vertice", "pupila"]))
    order = int(o.get("zernike_order", 6))
    profiles = default_profiles(diam, centers, order)
    rec = ReconstructionConfig(method=str(o.get("method", "conica_zernike")), surface_order=int(o.get("surface_order", 10)))
    return profiles, rec, bool(o.get("maps", True)), bool(o.get("pdf", True))


def analyze_exam_html(directory: str, exam_id: Optional[str], options: Optional[dict[str, Any]] = None,
                      progress=None) -> dict[str, Any]:
    """Importa + analisa um exame e devolve o relatório HTML, o JSON/CSV canônicos e o PDF (base64)."""
    profiles, rec, with_maps, with_pdf = _options(options)
    try:
        exam = import_tms_bundle(directory, None, exam_id=exam_id or None)
    except AdapterError as e:
        return {"error": f"importação inválida: {e}", "exam_id": exam_id}
    an, _ = analyze_exam(exam, profiles, rec, QualityConfig(), with_maps=with_maps, progress=progress)
    name = patient_display_name(directory, an.identity.exam_id) if dict(options or {}).get("include_name", True) else None
    out: dict[str, Any] = {"exam_id": an.identity.exam_id, "eye": an.identity.eye, "study_id": an.identity.study_id,
                           "capture_index": an.identity.capture_index, "acquired_at": an.identity.acquired_at,
                           "analysis_id": an.analysis_id, "stem": report_stem(an), "warnings": list(an.warnings)}
    out["summary"] = {
        "R_mm": an.geometry.conic_R_mm, "Q": an.geometry.conic_Q, "residual_um": an.geometry.residual_rms_mm * 1000.0,
        "profiles": [{"name": w.profile.name, "validity": w.validity, "block_reasons": w.block_reasons,
                      "metrics": {k: (None if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in w.metrics.items()}}
                     for w in an.wavefronts],
        "keratometry": _kerato_summary(an),
        "chord_mu": to_jsonable(an.chord_mu) if getattr(an, "chord_mu", None) is not None else None,
    }
    out["html"] = report_html(exam, an, with_maps=with_maps, patient_name=name)
    tmp = tempfile.mkdtemp(prefix="tmsab_web_")
    out["json"] = open(export_json(an, os.path.join(tmp, "a.json")), encoding="utf-8").read()
    out["csv"] = open(export_csv(an, os.path.join(tmp, "a.csv")), encoding="utf-8").read()
    out["csv_ceratometria"] = open(export_csv_ceratometria(an, os.path.join(tmp, "k.csv")), encoding="utf-8").read()
    if with_pdf:
        with open(export_pdf(exam, an, os.path.join(tmp, "a.pdf"), patient_name=name), "rb") as fh:
            out["pdf_b64"] = base64.b64encode(fh.read()).decode("ascii")
    return out
