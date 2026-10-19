"""Interface de linha de comando (plano §6): processa pacotes e executa verificações em lote.

  tmsab synth --out DIR                      gera pacotes sintéticos identificados
  tmsab manifest DIR --eye OD --model TMS-4N …  cria manifest.json para um pacote real
  tmsab import DIR                           valida/importa e mostra diagnóstico
  tmsab analyze PASTA [--out OUT]            todos os exames da pasta exportada → JSON/CSV/PDF em PASTA/analises
  tmsab batch ROOT --out OUT                 processa todos os subdiretórios + resumo CSV (formato longo)
  tmsab verify                               suíte científica rápida (casos analíticos e sintéticos)
  tmsab stats TABELA.csv --out OUT           repetibilidade e concordância do piloto
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np

from . import CORE_VERSION


def _profiles_from_args(args):
    from .pipeline import default_profiles
    diam = [float(v) for v in args.diameters.split(",")]
    cent = [v.strip() for v in args.centers.split(",")]
    profs = default_profiles(diam, cent, args.order)
    for p in profs:
        p.reference = args.reference
    return profs


def _recon_from_args(args):
    from .model import ReconstructionConfig
    return ReconstructionConfig(method=args.method, surface_order=args.surface_order,
                                fit_radius_mm=args.fit_radius, holdout_rings=tuple(int(v) for v in args.holdout.split(",") if v.strip()))


def cmd_synth(args):
    from .synthetic import SyntheticSpec, standard_cases, write_bundle
    specs = standard_cases()
    if args.spec:
        with open(args.spec, "r", encoding="utf-8") as fh:
            specs = [SyntheticSpec(**d) for d in json.load(fh)]
    for s in specs:
        d = write_bundle(s, args.out)
        print(f"sintético: {d}")
    print(f"{len(specs)} pacotes em {args.out}")


def cmd_manifest(args):
    path = os.path.join(args.dir, "manifest.json")
    if os.path.exists(path) and not args.force:
        print(f"manifest.json já existe ({path}); use --force para sobrescrever")
        return 1
    manifest = {
        "synthetic": False,
        "study_id": args.study_id,
        "exam_id": args.exam_id or os.path.basename(os.path.abspath(args.dir)),
        "eye": args.eye,
        "acquired_at": args.date,
        "date_precision": "dia" if args.date else "desconhecida",
        "capture_index": args.capture,
        "device": {"model": args.model, "software_version": args.software, "cone": args.cone, "calibration_state": args.calibration},
        "export": {"orientation": args.orientation, "decimals": args.decimals, "dialog_screenshot": args.screenshot},
        "notes": args.notes,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(f"manifesto escrito: {path}")
    return 0


def _profile_arg(args):
    return None if getattr(args, "profile", "auto") in (None, "", "auto") else args.profile


def _exams_in(directory, args):
    """Exames de uma pasta: formato real (vários exames) ou pacote único."""
    from .adapters import list_exams
    try:
        found = list_exams(directory, _profile_arg(args))
    except Exception:  # noqa: BLE001
        found = []
    return [e["exam_id"] for e in found] if len(found) > 1 else [found[0]["exam_id"] if found else None]


def cmd_import(args):
    from .adapters import AdapterError, import_tms_bundle
    rc = 0
    for exam_id in _exams_in(args.dir, args):
        try:
            exam = import_tms_bundle(args.dir, _profile_arg(args), exam_id=exam_id)
        except AdapterError as e:
            print(f"IMPORTAÇÃO INVÁLIDA ({exam_id or args.dir}): {e}")
            rc = 2
            continue
        print(f"exame {exam.identity.exam_id} olho {exam.identity.eye} captura {exam.identity.capture_index} data {exam.identity.acquired_at} "
              f"código {exam.identity.study_id} — perfil {exam.bundle.adapter_profile} (sintético: {exam.bundle.synthetic})")
        for d in exam.diagnostics:
            print("  " + d)
        for w in exam.warnings:
            print("  AVISO: " + w)
        if args.json:
            from .model import to_jsonable
            print(json.dumps(to_jsonable({"identity": exam.identity, "device": exam.device, "frame": exam.frame, "pupil": exam.pupil,
                                          "files": exam.bundle.files}), ensure_ascii=False, indent=2))
    return rc


def _print_analysis(an):
    cap = f" c{an.identity.capture_index}" if an.identity.capture_index else ""
    print(f"análise {an.analysis_id} — {an.identity.study_id} exame {an.identity.exam_id} {an.identity.eye or ''}{cap} — "
          f"R={an.geometry.conic_R_mm:.4f} mm Q={an.geometry.conic_Q:+.4f} resíduo={an.geometry.residual_rms_mm*1000:.3f} µm")
    for w in an.wavefronts:
        if w.validity != "disponivel":
            print(f"  {w.profile.name}: {w.validity.upper()} — " + "; ".join(w.block_reasons))
            continue
        m = w.metrics
        astig = ""
        if m.get("cilindro_equivalente_D") is not None:
            astig = (f" | cil {m['cilindro_equivalente_D']:+.2f} D eixo {m['eixo_equivalente_deg']:.0f}° "
                     f"(meridiano mais curvo {(m['eixo_equivalente_deg'] - 90) % 180:.0f}°)")
        print(f"  {w.profile.name}: SA c40={m['sa_c40']:+.4f} µm | coma |{m['coma_magnitude']:.4f}| (c3-1 {m['coma_c3m1']:+.4f}, c31 {m['coma_c31']:+.4f}) | trefoil |{m['trefoil_magnitude']:.4f}| | HOA RMS={m['rms_hoa_n3plus']:.4f} µm | astig c22={m['astig_c22']:+.4f} c2-2={m['astig_c2m2']:+.4f}{astig}")
    _print_keratometry(an)


def _print_keratometry(an):
    """Resumo curto da ceratometria por zona: Ø 3 mm, zona 0–3 mm, zona pupilar e SimK do TMS (+ conferência)."""
    from .keratometry import linhas_resumo
    k = getattr(an, "keratometry", None)
    if k is None:
        return
    linhas = linhas_resumo(k)

    def _achar(tipo, d=None):
        return next((ln for ln in linhas if ln["tipo"] == tipo
                     and (d is None or (ln["diametro_mm"] is not None and abs(ln["diametro_mm"] - d) < 1e-6))), None)

    escolhidas = [_achar("circunferencia", 3.0), _achar("zona", 3.0), _achar("pupila"), _achar("simk_tms")]
    print("  ceratometria (potência axial do TMS, n 1,3375; K1 = plano, K2 = curvo; eixos TABO):")
    for ln in (x for x in escolhidas if x is not None):
        if ln["disponivel"]:
            k1, e1, k2, e2, dk, km = ln["texto"]
            print(f"    {ln['rotulo']}: K1 {k1} D @ {e1} / K2 {k2} D @ {e2} (ΔK {dk} D, Km {km} D)")
        elif "não registrada" in ln["rotulo"]:
            print(f"    {ln['rotulo']}")
        else:
            print(f"    {ln['rotulo']}: {ln['validity'].upper()} — {ln['motivo']}")
    for c in k.conferencia:
        print(f"    conferência: {c}")
    for a in getattr(k, "alertas", None) or []:
        print(f"    alerta (informativo): {a}")


def cmd_analyze(args):
    """Analisa uma pasta exportada do TMS (todos os exames) ou um pacote único; relatórios em --out
    (padrão: <pasta>/analises)."""
    from .adapters import AdapterError, import_tms_bundle, patient_display_name
    from .pipeline import analyze_exam
    from .reporting import export_all
    t0 = time.time()
    out = args.out or os.path.join(os.path.abspath(args.dir), "analises")
    rc = 0
    for exam_id in _exams_in(args.dir, args):
        try:
            exam = import_tms_bundle(args.dir, _profile_arg(args), exam_id=exam_id)
        except AdapterError as e:
            print(f"IMPORTAÇÃO INVÁLIDA ({exam_id or args.dir}): {e}")
            rc = 2
            continue
        an, _ = analyze_exam(exam, _profiles_from_args(args), _recon_from_args(args), with_maps=not args.no_pdf)
        _print_analysis(an)
        name = None if (args.sem_nome or exam.bundle.synthetic) else patient_display_name(args.dir, an.identity.exam_id)
        paths = export_all(exam, an, out, patient_name=name)
        for k, p in paths.items():
            print(f"  {k}: {p}")
    print(f"tempo total {time.time()-t0:.1f} s")
    return rc


def _long_rows(an):
    rows = []
    for w in an.wavefronts:
        if w.validity != "disponivel":
            rows.append({"participante": an.identity.study_id, "olho": an.identity.eye or "", "metodo": "TMS_calc",
                         "captura": an.identity.capture_index if an.identity.capture_index is not None else "", "exame": an.identity.exam_id,
                         "perfil": w.profile.name, "metrica": "validade", "valor": w.validity, "analise": an.analysis_id})
            continue
        for k, v in w.metrics.items():
            rows.append({"participante": an.identity.study_id, "olho": an.identity.eye or "", "metodo": "TMS_calc",
                         "captura": an.identity.capture_index if an.identity.capture_index is not None else "", "exame": an.identity.exam_id,
                         "perfil": w.profile.name, "metrica": f"{k}@{w.profile.diameter_mm:g}mm_{w.profile.center_mode}", "valor": f"{v:.6f}", "analise": an.analysis_id})
    rows.extend(_kerato_long_rows(an))
    return rows


# métricas da ceratometria que entram na estatística: potências e vetor de potência (J0/J45), nunca o eixo cru
_KERATO_LONGO = ("k1_plano_D", "k2_curvo_D", "delta_k_D", "k_medio_D", "j0_D", "j45_D")


def _kerato_long_rows(an):
    """Ceratometria no resumo_longo.csv: K1, K2, ΔK, Km, J0 e J45 por circunferência (@{d}mm_vertice), zona 0–d
    (@zona_0-{d}mm_vertice) e zona pupilar (@pupila), mais diametro_pupila_mm. Sem eixo: média e desvio-padrão
    lineares de ângulo não fazem sentido (o eixo entra pelo J0/J45). Linhas só com códigos do exame (nunca nomes)."""
    from .reporting import kerato_chave
    k = getattr(an, "keratometry", None)
    if k is None:
        return []
    ident = {"participante": an.identity.study_id, "olho": an.identity.eye or "", "metodo": "TMS_calc",
             "captura": an.identity.capture_index if an.identity.capture_index is not None else "",
             "exame": an.identity.exam_id, "perfil": k.config.versao, "analise": an.analysis_id}
    zonas = list(k.zonas) + list(getattr(k, "discos", None) or []) + ([k.pupila] if k.pupila is not None else [])
    rows = []
    for z in zonas:
        chave = kerato_chave(z)
        if z.validity != "disponivel":
            rows.append({**ident, "metrica": f"validade@{chave}", "valor": z.validity})
            continue
        for m in _KERATO_LONGO:
            v = getattr(z, m)
            if v is not None and np.isfinite(v):
                rows.append({**ident, "metrica": f"{m}@{chave}", "valor": f"{round(float(v), 6) + 0.0:.6f}"})
    if an.pupil.state == "registrada" and an.pupil.diameter and an.pupil.diameter > 0:
        rows.append({**ident, "metrica": "diametro_pupila_mm", "valor": f"{an.pupil.diameter:.4f}"})
    return rows


def cmd_batch(args):
    import csv
    from .adapters import AdapterError, import_tms_bundle
    from .pipeline import analyze_exam
    from .reporting import export_all
    os.makedirs(args.out, exist_ok=True)
    rows = []
    n_ok = n_fail = 0
    for name in sorted(os.listdir(args.root)):
        d = os.path.join(args.root, name)
        if not os.path.isdir(d):
            continue
        for exam_id in _exams_in(d, args):
            tag = name + (f"_{exam_id}" if exam_id else "")
            try:
                exam = import_tms_bundle(d, _profile_arg(args), exam_id=exam_id)
                an, _ = analyze_exam(exam, _profiles_from_args(args), _recon_from_args(args), with_maps=not args.no_pdf)
                export_all(exam, an, os.path.join(args.out, tag))
                rows.extend(_long_rows(an))
                n_ok += 1
                print(f"OK   {tag}: {an.analysis_id}")
            except AdapterError as e:
                n_fail += 1
                rows.append({"participante": "", "olho": "", "metodo": "TMS_calc", "captura": "", "exame": tag, "perfil": "", "metrica": "importacao", "valor": f"INVALIDA: {e}", "analise": ""})
                print(f"FALHA {tag}: {e}")
    path = os.path.join(args.out, "resumo_longo.csv")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["participante", "olho", "metodo", "captura", "exame", "perfil", "metrica", "valor", "analise"], delimiter=";")
        w.writeheader(); w.writerows(rows)
    print(f"{n_ok} processados, {n_fail} falhas; resumo: {path}")
    return 0 if n_fail == 0 else 1


def cmd_verify(args):
    """Verificação científica rápida (sem pytest): casos analíticos e sintéticos com tolerâncias registradas."""
    from .geometry import ConicSurface
    from .model import OpticalProfile
    from .optics import fit_wavefront, trace_wavefront
    from .reference.meridional import meridional_wavefront_conic
    from .synthetic import SyntheticSpec, build_surface, write_bundle
    from .adapters import import_tms_bundle
    from .pipeline import analyze_exam
    import tempfile
    fails = 0
    def check(name, ok, detail):
        nonlocal fails
        fails += (0 if ok else 1)
        print(f"[{'OK ' if ok else 'FALHA'}] {name}: {detail}")
    n_out = 1.376
    # 1) elipsoide aplanático: W ≡ 0
    s = ConicSurface(7.8, -1.0 / n_out**2)
    tr = trace_wavefront(s, OpticalProfile(diameter_mm=6.0))
    check("oval de Descartes Q=-1/n² (W≡0)", np.nanmax(np.abs(tr.W_um)) < 1e-3, f"max|W|={np.nanmax(np.abs(tr.W_um)):.2e} µm")
    # 2) esfera: SA primária vs Seidel em 3 mm
    s = ConicSurface(7.8, 0.0)
    fr = fit_wavefront(trace_wavefront(s, OpticalProfile(diameter_mm=3.0)), 1.5, 6)
    c40 = fr.coeffs[fr.terms.index((4, 0))]
    seidel = (n_out - 1) * (1 / n_out**2) / (8 * 7.8**3) * 1.5**4 / (6 * np.sqrt(5)) * 1000
    check("esfera: c40 ≈ Seidel (3 mm)", abs(c40 - seidel) / seidel < 0.05 and c40 > 0, f"c40={c40:+.5f} µm, Seidel={seidel:+.5f} µm")
    # 3) traçado 3D vs referência meridional independente
    for Q in (0.0, -0.3):
        s = ConicSurface(7.7, Q)
        tr = trace_wavefront(s, OpticalProfile(diameter_mm=6.0))
        m = np.abs(tr.y) < 1e-9
        Wref = meridional_wavefront_conic(7.7, Q, n_out, tr.x[m])
        check(f"3D vs meridional (Q={Q})", np.nanmax(np.abs(tr.W_um[m] - Wref)) < 1e-4, f"max diff={np.nanmax(np.abs(tr.W_um[m]-Wref)):.2e} µm")
    # 4) sintético completo: recuperação de SA, coma e sinais
    with tempfile.TemporaryDirectory() as td:
        for spec, key, tol in ((SyntheticSpec(exam_id="V1", R=7.8, Q=-0.2, decimals=6), (4, 0), 0.002),
                               (SyntheticSpec(exam_id="V2", R=7.8, Q=-0.2, zernike_um={"3,1": 3.0}, decimals=6), (3, 1), 0.002),
                               (SyntheticSpec(exam_id="V3", R=7.8, Q=-0.2, zernike_um={"3,-3": 2.0}, decimals=6), (3, -3), 0.002)):
            d = write_bundle(spec, td)
            an, _ = analyze_exam(import_tms_bundle(d), with_maps=False)
            w = an.wavefronts[1]
            got = next(c.value_um for c in w.coefficients if (c.n, c.m) == key)
            trt = fit_wavefront(trace_wavefront(build_surface(spec), w.profile), 3.0, 6)
            exp = trt.coeffs[trt.terms.index(key)]
            check(f"sintético {spec.exam_id} c{key}", abs(got - exp) < tol and np.sign(got) == np.sign(exp), f"obtido {got:+.4f}, esperado {exp:+.4f} µm")
    print("verificação:", "TODOS OS ENSAIOS PASSARAM" if fails == 0 else f"{fails} FALHA(S)")
    if args.pytest:
        import subprocess
        return subprocess.call([sys.executable, "-m", "pytest", "-q"])
    return 0 if fails == 0 else 1


def cmd_stats(args):
    from .stats import LongTable, agreement_repeated, bland_altman_figure, repeatability, write_results_csv
    tb = LongTable.read_csv(args.table)
    os.makedirs(args.out, exist_ok=True)
    metrics = [m for m in tb.metrics() if (not args.metrics or m in args.metrics.split(","))]
    reps, agrs = [], []
    for met in metrics:
        for method in tb.methods():
            try:
                r = repeatability(tb.select(method, met, args.eye), method, met)
                reps.append(r)
                print(f"repetibilidade {method} {met}: Sw={r.sw:.4f} (IC95 {r.sw_ic95[0]:.4f}–{r.sw_ic95[1]:.4f}), CR={r.coef_repetibilidade:.4f}, ICC={r.icc:.3f}, n={r.n_participantes}")
            except ValueError as e:
                print(f"repetibilidade {method} {met}: {e}")
        if args.compare:
            a, b = args.compare.split(",")
            try:
                g = agreement_repeated(tb.select(a, met, args.eye), tb.select(b, met, args.eye), a, b, met)
                agrs.append(g)
                print(f"concordância {a}−{b} {met}: viés={g.vies:+.4f} (IC95 {g.vies_ic95[0]:+.4f}/{g.vies_ic95[1]:+.4f}), LoA={g.loa[0]:+.4f}/{g.loa[1]:+.4f}, σw={g.sigma_w:.4f}, σb={g.sigma_b:.4f}, p_heteroc={g.heterocedasticidade_p:.3f}, n={g.n_participantes}")
                fig = bland_altman_figure(g)
                fig.savefig(os.path.join(args.out, f"bland_altman_{met}.png".replace("/", "_")), dpi=150)
            except ValueError as e:
                print(f"concordância {met}: {e}")
    path = write_results_csv(os.path.join(args.out, "estatistica_piloto.csv"), reps, agrs)
    print(f"resultados: {path}")
    return 0


def cmd_audit(args):
    """Auditoria LGPD local: procura os nomes do XREF.dat (lidos e descartados em memória) nos relatórios gerados
    (JSON, CSV, TXT e PDF quando houver extrator). Imprime apenas contagens — nunca os nomes."""
    from .adapters.tms4 import audit_reports_for_names
    res = audit_reports_for_names(args.dir, args.out)
    print(f"tokens nominais considerados: {res['n_tokens']} | arquivos verificados: {res['n_files']} "
          f"(PDF {'lidos' if res['pdf_checked'] else 'NÃO lidos — instale pypdf'}) | ocorrências: {res['n_hits']}")
    for f in res["files_with_hits"]:
        print(f"  NOME ENCONTRADO em {f}")
    if res.get("n_hits_pagina_completa"):
        print(f"  (nome presente na 1ª página de {res['n_hits_pagina_completa']} PDF(s) — página de resultado completo, por opção do usuário; permitido)")
    if res["n_hits"] == 0:
        print("OK — nenhum nome do XREF aparece em JSON/CSV nem nas páginas técnicas dos PDFs.")
    return 0 if res["n_hits"] == 0 else 3


def build_parser():
    p = argparse.ArgumentParser(prog="tmsab", description=f"Aberrometria corneana anterior calculada (TMS) — v{CORE_VERSION}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="gera pacotes sintéticos identificados")
    s.add_argument("--out", required=True)
    s.add_argument("--spec", help="JSON com lista de SyntheticSpec")
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("manifest", help="cria manifest.json para um pacote real exportado")
    s.add_argument("dir")
    s.add_argument("--study-id", default="sem_estudo")
    s.add_argument("--exam-id")
    s.add_argument("--eye", choices=["OD", "OS"])
    s.add_argument("--date", help="ISO-8601 (AAAA-MM-DD)")
    s.add_argument("--capture", type=int)
    s.add_argument("--model", default="desconhecido")
    s.add_argument("--software", default="desconhecida")
    s.add_argument("--cone")
    s.add_argument("--calibration", default="nao_informado")
    s.add_argument("--orientation", default="desconhecida")
    s.add_argument("--decimals", type=int)
    s.add_argument("--screenshot", help="arquivo de imagem da tela de configuração da exportação")
    s.add_argument("--notes", default="")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_manifest)

    def common(sp):
        sp.add_argument("--profile", default="auto", help="perfil do adaptador (auto = detectar pelo formato; nome ou caminho JSON)")
        sp.add_argument("--diameters", default="4,6")
        sp.add_argument("--centers", default="vertice,pupila")
        sp.add_argument("--order", type=int, default=6)
        sp.add_argument("--reference", default="paraxial_media_principais", choices=["paraxial_media_principais", "refoco_rms"])
        sp.add_argument("--method", default="conica_zernike", choices=["conica_zernike", "rbf_tps"])
        sp.add_argument("--surface-order", type=int, default=10)
        sp.add_argument("--fit-radius", type=float)
        sp.add_argument("--holdout", default="", help="anéis reservados (0-based), ex.: 5,15")
        sp.add_argument("--no-pdf", action="store_true", help="não gera mapas/PDF (mais rápido)")

    s = sub.add_parser("import", help="valida e importa um pacote")
    s.add_argument("dir"); s.add_argument("--profile", default="auto"); s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_import)

    s = sub.add_parser("analyze", help="analisa uma pasta exportada do TMS (todos os exames) e salva relatórios")
    s.add_argument("--sem-nome", action="store_true", help="não pôr o nome do paciente (XREF) na página de resultado completo do PDF")
    s.add_argument("dir"); s.add_argument("--out", help="pasta dos relatórios (padrão: <dir>/analises)"); common(s)
    s.set_defaults(func=cmd_analyze)

    s = sub.add_parser("batch", help="processa todos os subdiretórios de ROOT")
    s.add_argument("root"); s.add_argument("--out", required=True); common(s)
    s.set_defaults(func=cmd_batch)

    s = sub.add_parser("verify", help="verificação científica rápida")
    s.add_argument("--pytest", action="store_true", help="roda também a suíte pytest completa")
    s.set_defaults(func=cmd_verify)

    s = sub.add_parser("audit", help="auditoria LGPD: confirma que nenhum nome do XREF aparece nos relatórios (só contagens)")
    s.add_argument("dir", help="pasta exportada do TMS (com XREF.dat)")
    s.add_argument("--out", help="pasta dos relatórios (padrão: <dir>/analises)")
    s.set_defaults(func=cmd_audit)

    s = sub.add_parser("stats", help="estatística do piloto (tabela longa)")
    s.add_argument("table"); s.add_argument("--out", required=True)
    s.add_argument("--compare", help="dois métodos separados por vírgula, ex.: TMS_calc,OPD_corneal")
    s.add_argument("--metrics", help="métricas separadas por vírgula (padrão: todas)")
    s.add_argument("--eye", help="filtrar um olho (OD/OS)")
    s.set_defaults(func=cmd_stats)
    return p


def main(argv=None):
    from . import utf8_console
    utf8_console()
    args = build_parser().parse_args(argv)
    rc = args.func(args)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
