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
import json
import os
import tempfile
from typing import Any, Optional

import matplotlib
import numpy as np
from matplotlib.figure import Figure

from . import CORE_VERSION
from .adapters import AdapterError, import_tms_bundle, list_exams, patient_display_name
from .keratometry import (
    COLUNAS_TABELA,
    CONTROLE_LEGENDA,
    LEGENDA_TABELA,
    dados_do_controle,
    eixo_tabo,
    faixa_do_controle,
    linhas_resumo,
)
from .model import Analysis, NormalizedExam, QualityConfig, ReconstructionConfig, to_jsonable
from .pupila import fmt_chord_mu
from .pipeline import DEFAULT_DIAMETERS, analyze_exam, profiles_for_exam
from .plots import (
    figure_resultado_completo,
    plot_coverage,
    plot_elevation,
    plot_topo_map,
    plot_wavefront,
    plot_zernike_bars,
)
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


# ---------------------------------------------------------------------------
# Controle de diâmetro da ceratometria (mapa axial + controle deslizante; o desenho é feito por CONTROLE_JS)
# ---------------------------------------------------------------------------
_KC_FIGURA_POL = (4.8, 4.0)          # figura do mapa (polegadas); a 110 dpi o PNG tem exatamente 528 × 440 px
_KC_QUADRO_POL = (0.42, 0.32, 3.3)   # esquerda, base e lado do quadro do mapa (polegadas): QUADRADO também em pixels
_KC_DPI = 110


def _png_mapa_controle(exam: NormalizedExam) -> tuple[str, dict[str, Any]]:
    """Mapa de potência axial (D) do controle de diâmetro e a geometria da sobreposição SVG.

    Sem bbox_inches="tight" (o recorte deslocaria o quadro): o PNG tem figsize × dpi pixels e `caixa` =
    [esquerda, topo, largura, altura] do quadro do mapa em frações da imagem, a partir do canto superior esquerdo, lida
    depois do desenho; `L` = meia largura do quadro em mm (limites simétricos e iguais em x e y, aspecto igual)."""
    fw, fh = _KC_FIGURA_POL
    esq, base, lado = _KC_QUADRO_POL
    fig = Figure(figsize=(fw, fh))
    ax = fig.add_axes((esq / fw, base / fh, lado / fw, lado / fh))
    cax = fig.add_axes(((esq + lado + 0.12) / fw, base / fh, 0.14 / fw, lado / fh))
    if plot_topo_map(ax, exam, "DIO", title="Potência axial (D)", zones=(), pupil=True, cax=cax, fontsize=9) is None:
        # sem DIO o quadro mostra só o motivo; continua em mm para o círculo do diâmetro escolhido
        cax.set_visible(False)
        lim = 0.52 * faixa_do_controle(exam)[1]
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    else:
        ax.tick_params(labelsize=7.5)
        cax.tick_params(labelsize=7.5)
        cax.yaxis.label.set_size(8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_KC_DPI)
    pos = ax.get_position()                      # quadro final, depois do desenho (aspecto igual já aplicado)
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    if not (x1 > 0 and abs(x0 + x1) < 1e-9 and abs(y0 + y1) < 1e-9 and abs(x1 - y1) < 1e-9):
        raise ValueError(f"limites do mapa não simétricos: x {x0:g}…{x1:g}, y {y0:g}…{y1:g}")
    geo = {"L": float(x1), "caixa": [round(float(v), 6) for v in (pos.x0, 1.0 - pos.y1, pos.width, pos.height)]}
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii"), geo


def _json_no_script(obj: Any) -> str:
    """JSON para um bloco <script type="application/json">: sem NaN e sem "<" literal (nem "</script" nem "<!--")."""
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")


def _controle_html(exam: NormalizedExam, an: Analysis) -> str:
    """Controle de diâmetro (topo da seção de ceratometria, só com mapas): mapa axial com a sobreposição SVG, controle
    deslizante e as linhas da circunferência Ø d e da zona 0–d. As linhas do diâmetro inicial já vêm preenchidas
    (leitura e impressão sem JavaScript); a grade completa vai num bloco JSON que CONTROLE_JS (kcBind) lê. Os números
    vêm de dados_do_controle com a configuração da própria análise: no Ø 3 mm são os da tabela. Sem nome do paciente."""
    k = an.keratometry
    dados = dados_do_controle(exam, k.config, an.quality_config)
    try:
        src, geo = _png_mapa_controle(exam)
        mapa = (f"<div class='kc-mapa'><img class='kc-img' src='{src}' width='{round(_KC_FIGURA_POL[0] * _KC_DPI)}' "
                f"height='{round(_KC_FIGURA_POL[1] * _KC_DPI)}' alt='mapa de potência axial com o diâmetro escolhido'>"
                f"<svg class='kc-sobre' viewBox='{-geo['L']:.6f} {-geo['L']:.6f} {2 * geo['L']:.6f} {2 * geo['L']:.6f}' "
                f"style='left:{100 * geo['caixa'][0]:.4f}%;top:{100 * geo['caixa'][1]:.4f}%;"
                f"width:{100 * geo['caixa'][2]:.4f}%;height:{100 * geo['caixa'][3]:.4f}%' "
                "aria-hidden='true' focusable='false'></svg></div>")
    except Exception as e:  # noqa: BLE001
        geo = None
        mapa = f"<p class='aviso'>Mapa do controle indisponível: {_e(e)}</p>"
    p0 = next(p for p in dados["pontos"] if round(p["d"] * 10) == round(dados["inicial"] * 10))
    # as linhas de _kerato_rows (CONTROLE_JS repete a estrutura, inclusive INDISPONÍVEL × INVÁLIDA)
    linhas = _kerato_rows([{**p0["circ"], "destaque": ""}, {**p0["zona"], "destaque": "zona"}])
    head = "<thead><tr>" + "".join(f"<th>{_e(c)}</th>" for c in COLUNAS_TABELA) + "</tr></thead>"
    legenda = ("Clique ou arraste no mapa (ou use o controle) para escolher o diâmetro. " + CONTROLE_LEGENDA
               + " No mapa: círculo = diâmetro escolhido (cinza tracejado quando indisponível); linhas = meridianos da "
               "circunferência, K2 curvo (laranja) e K1 plano (azul), tracejadas com eixo pouco definido; + = vértice; "
               "contorno tracejado fino = pupila. † pontos interpolados pelo aparelho; * eixo pouco definido (ΔK baixo).")
    return (
        "<div class='kc'>" + mapa
        + "<div class='kc-painel'><div class='kc-controle'><label>Diâmetro "
        f"<input class='kc-d' type='range' min='{dados['dmin']:.1f}' max='{dados['dmax']:.1f}' step='{dados['passo']:.1f}' "
        f"value='{dados['inicial']:.1f}'></label><output class='kc-valor' aria-live='polite'>{dados['inicial']:.1f} mm</output></div>"
        f"<div class='kc-rolagem'><table class='metricas kerato kc-tabela'>{head}<tbody>{linhas}</tbody></table></div>"
        f"<p class='legenda'>{_e(legenda)}</p></div>"
        f"<script type='application/json' class='kc-dados'>{_json_no_script({**dados, 'mapa': geo})}</script></div>"
    )


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
    controle = ""
    if with_maps and getattr(an, "keratometry", None) is not None:
        try:
            controle = _controle_html(exam, an)
        except Exception as e:  # noqa: BLE001
            controle = f"<p class='aviso'>Controle de diâmetro indisponível: {_e(e)}</p>"
    kerato = f"<section class='kerato'><h3>{_e(KERATO_TITULO)}</h3>{controle}{_kerato_table(an)}</section>"

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
    """Documento HTML completo e autônomo (para salvar/imprimir) com os artigos gerados; o controle de diâmetro da
    ceratometria funciona nele (CONTROLE_JS no fim do corpo)."""
    return ("<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'><title>" + _e(title) + "</title><style>"
            + REPORT_CSS + "</style></head><body><main class='relatorio'><h1>" + _e(title) + "</h1>"
            + "".join(articles) + "</main><script>" + CONTROLE_JS + "\nkcBind(document);</script></body></html>")


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
.kc { display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: flex-start; margin: 6px 0 12px; }
.kc-mapa { position: relative; flex: 0 1 440px; max-width: 100%; }
.kc-img { display: block; width: 100%; height: auto; }
.kc-sobre { position: absolute; display: block; overflow: hidden; touch-action: none; cursor: crosshair; -webkit-user-select: none; user-select: none; -webkit-tap-highlight-color: transparent; }
.kc-painel { flex: 1 1 300px; min-width: 0; }
.kc-controle { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 14px; margin: 2px 0 8px; }
.kc-controle label { display: flex; align-items: center; gap: 8px; flex: 1 1 180px; font-size: 13px; font-weight: bold; }
.kc-d { flex: 1 1 auto; min-width: 120px; margin: 0; accent-color: #1a4f8a; }
.kc-valor { font-size: 16px; font-weight: bold; font-variant-numeric: tabular-nums; min-width: 4.6em; }
.kc-rolagem { overflow-x: auto; }
@media (max-width: 520px) {
  table.kc-tabela { font-size: 11px; }
  table.kc-tabela th, table.kc-tabela td { padding: 2px 3px; }
}
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
  .kc { break-inside: avoid; page-break-inside: avoid; }
  .kc-mapa { flex: 0 0 88mm; }
  .kc-painel { flex: 1 1 80mm; }
  .kc-d { display: none; }
  .kc-rolagem { overflow: visible; }
}
"""

# Controle de diâmetro da ceratometria no navegador. Uma única cópia: a página (marcador {{CONTROLE_JS}} em
# web/template.html), o relatório salvo pela página e report_document(). Sem "</script" (vai dentro de <script>).
CONTROLE_JS = r"""
/* Controle de diâmetro da ceratometria (tmsab.web.CONTROLE_JS). O controle deslizante ou um clique/arraste no mapa
   axial escolhe o diâmetro d (décimos de mm); a sobreposição SVG (em mm, y para cima como o mapa, vista do examinador)
   desenha o círculo Ø d e os meridianos K2 (curvo) e K1 (plano) da circunferência; as duas linhas (circunferência Ø d
   e zona 0–d) vêm da grade calculada pelo núcleo (bloco JSON .kc-dados). Só mostra números: não recalcula nada. */
var kcBind = window.kcBind || (function () {
  "use strict";
  const SVG_NS = "http://www.w3.org/2000/svg";
  const COR_K1 = "#2a78d6", COR_K2 = "#eb6834";   // as cores do programa (K_COR_K1 plano, K_COR_K2 curvo)
  const ligados = new WeakSet();                   // sem atributo no DOM: cópias (relatório salvo) são religadas

  function svgNo(pai, tag, attrs) {
    const el = document.createElementNS(SVG_NS, tag);
    for (const k of Object.keys(attrs)) if (attrs[k] !== null && attrs[k] !== undefined) el.setAttribute(k, String(attrs[k]));
    pai.appendChild(el);
    return el;
  }

  function celula(tr, texto) {
    const td = document.createElement("td");
    td.textContent = texto;
    tr.appendChild(td);
    return td;
  }

  // a mesma estrutura de _kerato_rows (tmsab/web.py): rótulo + 6 colunas, ou "INDISPONÍVEL — motivo" nas 6 colunas
  // ("INVÁLIDA" quando o ajuste ficou sem amostras, como na tabela)
  function linhaTabela(ln, classe) {
    const tr = document.createElement("tr");
    if (classe) tr.className = classe;
    celula(tr, ln.rotulo);
    if (ln.disponivel) {
      for (const t of ln.texto) celula(tr, t);
    } else {
      const estado = ln.validity === "invalida" ? "INVÁLIDA" : "INDISPONÍVEL";
      const td = celula(tr, estado + (ln.motivo ? " — " + ln.motivo : ""));
      td.colSpan = 6;
      td.className = "motivo";
    }
    return tr;
  }

  function desenhar(svg, L, p) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const larg = svg.getBoundingClientRect().width;
    const u = 2 * L / (larg > 0 ? larg : 300);       // mm por pixel: traços e rótulos com tamanho fixo na tela
    const c = p.circ, r = p.d / 2;
    const ok = !!(c.disponivel && c.eixo_k1 !== null && c.eixo_k2 !== null);
    const g = svgNo(svg, "g", { transform: "scale(1,-1)" });   // y para cima, como o mapa
    const traco = (x1, y1, x2, y2, cor, px, tracos, classe) => svgNo(g, "line", {
      x1: x1, y1: y1, x2: x2, y2: y2, stroke: cor, "stroke-width": px, "stroke-linecap": "round",
      "stroke-dasharray": tracos, "vector-effect": "non-scaling-stroke", class: classe });
    svgNo(g, "circle", { cx: 0, cy: 0, r: r, fill: "none", stroke: "#fff", "stroke-opacity": 0.9, "stroke-width": 4.5,
                         "vector-effect": "non-scaling-stroke" });
    svgNo(g, "circle", { class: "kc-circulo", cx: 0, cy: 0, r: r, fill: "none", stroke: ok ? "#111" : "#666",
                         "stroke-width": ok ? 2 : 1.6, "stroke-dasharray": ok ? null : "6 4",
                         "vector-effect": "non-scaling-stroke" });
    // meridiano no eixo TABO a: reta pelo vértice na direção (cos a, sen a); estende-se até 1,12 × o raio
    const eixos = !ok ? [] : [
      { nome: "k2", graus: c.eixo_k2, cor: COR_K2, texto: "K2 " + c.texto[3] },
      { nome: "k1", graus: c.eixo_k1, cor: COR_K1, texto: "K1 " + c.texto[1] },
    ];
    const ext = 1.12 * r;
    for (const e of eixos) {
      e.ux = Math.cos(e.graus * Math.PI / 180);
      e.uy = Math.sin(e.graus * Math.PI / 180);
      traco(-ext * e.ux, -ext * e.uy, ext * e.ux, ext * e.uy, "#fff", 5.5);
    }
    for (const e of eixos) {
      traco(-ext * e.ux, -ext * e.uy, ext * e.ux, ext * e.uy, e.cor, 2.4, c.pouco_definido ? "7 5" : null, "kc-" + e.nome);
    }
    const t = 6 * u;                                 // + no vértice
    traco(-t, 0, t, 0, "#fff", 4); traco(0, -t, 0, t, "#fff", 4);
    traco(-t, 0, t, 0, "#111", 1.5); traco(0, -t, 0, t, "#111", 1.5);
    // rótulos fora do grupo invertido (texto em pé), na ponta de cima do meridiano (eixo 1–180°), dentro do quadro
    const fs = 12.5 * u, m = 3 * u;
    for (const e of eixos) {
      const ancora = e.ux > 0.35 ? "start" : (e.ux < -0.35 ? "end" : "middle");
      const dist = ext + 5 * u + 0.55 * fs * Math.abs(e.uy);
      const w = 0.62 * fs * e.texto.length, h = fs;           // caixa estimada do texto
      const desloc = ancora === "start" ? 0 : (ancora === "end" ? w : w / 2);
      const x0 = Math.min(Math.max(dist * e.ux - desloc, -L + m), L - m - w);
      const y = Math.min(Math.max(dist * e.uy, -L + m + h / 2), L - m - h / 2);
      svgNo(svg, "text", { class: "kc-rotulo kc-rotulo-" + e.nome, x: x0 + desloc, y: -y, dy: "0.35em",
        "text-anchor": ancora, "font-size": fs, "font-weight": "bold", fill: e.cor, stroke: "#fff",
        "stroke-width": 3 * u, "stroke-linejoin": "round", "paint-order": "stroke" }).textContent = e.texto;
    }
  }

  function ligar(raiz) {
    const dados = JSON.parse(raiz.querySelector("script.kc-dados").textContent);
    const inp = raiz.querySelector("input.kc-d"), saida = raiz.querySelector("output.kc-valor");
    const corpo = raiz.querySelector("table.kc-tabela tbody"), svg = dados.mapa ? raiz.querySelector("svg.kc-sobre") : null;
    const n0 = Math.round(dados.dmin * 10), n1 = Math.round(dados.dmax * 10);
    const porDecimo = new Map(dados.pontos.map((p) => [Math.round(p.d * 10), p]));
    inp.min = dados.dmin.toFixed(1);
    inp.max = dados.dmax.toFixed(1);
    inp.step = String(dados.passo);
    let atual = null;
    function mostrar(d, forcar) {
      if (!isFinite(d)) return;
      const n = Math.min(n1, Math.max(n0, Math.round(d * 10)));   // décimos exatos, dentro da faixa
      const p = porDecimo.get(n);
      if (!p) return;
      const txt = (n / 10).toFixed(1);
      inp.value = txt;
      if (n === atual && !forcar) return;
      atual = n;
      inp.setAttribute("value", txt);                // a cópia salva do relatório reabre neste diâmetro
      inp.setAttribute("aria-valuetext", txt + " mm");
      saida.textContent = txt + " mm";
      while (corpo.firstChild) corpo.removeChild(corpo.firstChild);
      corpo.appendChild(linhaTabela(p.circ, ""));
      corpo.appendChild(linhaTabela(p.zona, "kerato-zona"));
      if (svg) desenhar(svg, dados.mapa.L, p);
    }
    inp.addEventListener("input", () => mostrar(parseFloat(inp.value)));
    inp.addEventListener("change", () => mostrar(parseFloat(inp.value)));
    if (svg) {
      let arrastando = false;
      const peloPonteiro = (ev) => {
        const ctm = svg.getScreenCTM();
        if (!ctm) return;
        const i = ctm.inverse();                     // tela → viewBox (mm, y para baixo)
        const x = i.a * ev.clientX + i.c * ev.clientY + i.e;
        const y = -(i.b * ev.clientX + i.d * ev.clientY + i.f);
        mostrar(Math.round(2 * Math.hypot(x, y) * 10) / 10);
      };
      svg.addEventListener("pointerdown", (ev) => {
        if (ev.pointerType === "mouse" && ev.button !== 0) return;
        arrastando = true;
        try { svg.setPointerCapture(ev.pointerId); } catch (e) { /* sem captura: segue pelo pointermove */ }
        ev.preventDefault();
        peloPonteiro(ev);
        try { inp.focus({ preventScroll: true }); } catch (e) { /* foco opcional (setas do teclado) */ }
      });
      svg.addEventListener("pointermove", (ev) => { if (arrastando) peloPonteiro(ev); });
      for (const tipo of ["pointerup", "pointercancel", "lostpointercapture"]) {
        svg.addEventListener(tipo, () => { arrastando = false; });
      }
      // os traços e rótulos têm tamanho fixo em pixels: redesenha quando o mapa muda de tamanho
      if (typeof ResizeObserver === "function") {
        new ResizeObserver(() => { if (atual !== null) desenhar(svg, dados.mapa.L, porDecimo.get(atual)); }).observe(svg);
      }
    }
    const d0 = parseFloat(inp.getAttribute("value"));
    mostrar(isFinite(d0) ? d0 : dados.inicial, true);
  }

  function kcBind(raiz) {
    raiz = raiz || document;
    const blocos = Array.from(raiz.querySelectorAll(".kc"));
    if (raiz.matches && raiz.matches(".kc")) blocos.unshift(raiz);
    for (const el of blocos) {
      if (ligados.has(el)) continue;
      ligados.add(el);
      try { ligar(el); } catch (e) { if (window.console) console.warn("controle de diâmetro da ceratometria:", e); }
    }
  }
  return kcBind;
})();
"""


# ---------------------------------------------------------------------------
# análise de um exame → pacote de resultados
# ---------------------------------------------------------------------------
def _options(options: Optional[dict[str, Any]], exam: Optional[NormalizedExam] = None) -> tuple[list, ReconstructionConfig, bool, bool]:
    """include_name (padrão True): nome do paciente do XREF no cabeçalho/página impressa (nunca em JSON/CSV).
    diameters (padrão 3/4/5/6 mm), centers, zernike_order, measured_pupil (padrão True: perfil com a pupila medida)."""
    o = dict(options or {})
    diam = [float(d) for d in o.get("diameters", list(DEFAULT_DIAMETERS))]
    centers = tuple(o.get("centers", ["vertice", "pupila"]))
    order = int(o.get("zernike_order", 6))
    if exam is not None:
        profiles = profiles_for_exam(exam, diam, centers, order, measured_pupil=bool(o.get("measured_pupil", True)))
    else:
        from .pipeline import default_profiles
        profiles = default_profiles(diam, centers, order)
    rec = ReconstructionConfig(method=str(o.get("method", "conica_zernike")), surface_order=int(o.get("surface_order", 10)))
    return profiles, rec, bool(o.get("maps", True)), bool(o.get("pdf", True))


_CACHE: dict[tuple[str, str], tuple[NormalizedExam, Optional[Analysis]]] = {}     # (pasta, exame) → último cálculo


def _finite(obj: Any) -> Any:
    """NaN/inf → None recursivamente (JSON estrito para o navegador)."""
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def _view_from(view: Optional[dict[str, Any]]):
    from .topo import ViewOptions
    v = dict(view or {})
    return ViewOptions(kind=str(v.get("kind", "axial")), scale=str(v.get("scale", "tms")),
                       zone_mm=(float(v["zone_mm"]) if v.get("zone_mm") else None),
                       pupil_mm=(float(v["pupil_mm"]) if v.get("pupil_mm") else None),
                       center=str(v.get("center", "vertice")))


def topo_png(directory: str, exam_id: Optional[str], view: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Uma visualização no estilo do TMS (os "botões": mapa, escala, zona, diâmetro pupilar, centro) → PNG + Astig Stats."""
    from .topo import ZONES_MM, figure_tms_view, select_profile, zone_axis_stats
    key = (os.path.abspath(directory), str(exam_id or ""))
    if key in _CACHE:
        exam, an = _CACHE[key]
    else:
        try:
            exam = import_tms_bundle(directory, None, exam_id=exam_id or None)
        except AdapterError as e:
            return {"error": f"importação inválida: {e}"}
        an = None
        _CACHE[key] = (exam, an)
    vo = _view_from(view)
    fig = figure_tms_view(exam, an, vo, width_in=6.2)
    role = {"axial": "DIO", "tangencial": "ROC", "refrativa": "POW"}.get(vo.kind, "DIO")
    stats = []
    for z in ZONES_MM:
        zs = zone_axis_stats(exam, z, role)
        stats.append({"zone_mm": z, "steep_deg": zs.steep_deg, "flat_deg": zs.flat_deg, "steep_power": zs.steep_power,
                      "flat_power": zs.flat_power, "rows": [[p, a] for p, a in zs.rows]})
    prof = None
    if an is not None:
        wf = select_profile(an, vo)
        if wf is not None:
            m = wf.metrics
            prof = {"name": wf.profile.name, "diameter_mm": wf.profile.diameter_mm, "center": wf.profile.center_mode,
                    "sa_c40": m.get("sa_c40"), "coma": m.get("coma_magnitude"), "trefoil": m.get("trefoil_magnitude"),
                    "rms_hoa": m.get("rms_hoa_n3plus"), "cyl_D": m.get("cilindro_equivalente_D"), "axis_deg": m.get("eixo_equivalente_deg")}
    return _finite({"png": _png(fig, dpi=120), "stats": stats, "profile": prof, "view": dict(view or {})})


def completo_png(directory: str, exam_id: Optional[str], view: Optional[dict[str, Any]] = None, include_name: bool = True) -> dict[str, Any]:
    """Página "Resultado completo" com os filtros escolhidos (para a impressão no navegador)."""
    key = (os.path.abspath(directory), str(exam_id or ""))
    if key not in _CACHE or _CACHE[key][1] is None:
        return {"error": "exame ainda não calculado"}
    exam, an = _CACHE[key]
    name = patient_display_name(directory, an.identity.exam_id) if include_name else None
    fig = figure_resultado_completo(exam, an, patient_name=name, view=_view_from(view))
    return {"png": _png(fig, dpi=150)}


def analyze_exam_html(directory: str, exam_id: Optional[str], options: Optional[dict[str, Any]] = None,
                      progress=None) -> dict[str, Any]:
    """Importa + analisa um exame e devolve o relatório HTML, o JSON/CSV canônicos e o PDF (base64)."""
    try:
        exam = import_tms_bundle(directory, None, exam_id=exam_id or None)
    except AdapterError as e:
        return {"error": f"importação inválida: {e}", "exam_id": exam_id}
    profiles, rec, with_maps, with_pdf = _options(options, exam)
    an, _ = analyze_exam(exam, profiles, rec, QualityConfig(), with_maps=with_maps, progress=progress)
    _CACHE[(os.path.abspath(directory), str(an.identity.exam_id))] = (exam, an)
    if len(_CACHE) > 24:
        _CACHE.pop(next(iter(_CACHE)))
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
