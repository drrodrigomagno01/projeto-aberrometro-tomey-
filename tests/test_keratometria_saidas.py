"""Saídas da ceratometria por zona: CSV (seção própria), relatório em texto (PDF técnico / aba Técnico / HTML),
tabela do HTML, JSON da página, resumo da linha de comando e resumo_longo.csv do `tmsab batch`.

Exportação v1 sintética (formato real do TMS-4N): tórica com pupila registrada e esférica sem pupila. Os números das
saídas são conferidos contra o objeto KeratometryResult (a verdade óptica está em test_keratometria.py).
"""
import csv
import dataclasses
import io
import json
import math

import pytest

from tmsab.adapters import import_tms_bundle
from tmsab.cli import _long_rows, _print_analysis
from tmsab.keratometry import diferenca_eixo, linhas_resumo
from tmsab.pipeline import analyze_exam
from tmsab.reporting import (
    KERATO_CSV_COLUNAS,
    PUPILA_CSV_PERFIL,
    _H_LINHA,
    _H_TITULO,
    _Y_BASE,
    _Y_TOPO,
    _fmt,
    _paginar,
    _wrap,
    canonical_result,
    export_csv,
    export_csv_ceratometria,
    keratometry_records,
    report_lines,
)
from tmsab.synthetic import SyntheticSpec, write_export_folder_v1
from tmsab.web import analyze_exam_html, report_html

TITULO = "## Ceratometria por zona (potência axial do TMS, n 1,3375)"
COLUNAS = ["analysis_id", "study_id", "exam_id", "eye", "tipo", "zona", "diametro_mm", "validade",
           "k1_plano_D", "eixo_k1_deg", "k2_curvo_D", "eixo_k2_deg", "delta_k_D", "k_medio_D", "j0_D", "j45_D",
           "assimetria_D", "irregularidade_D", "fracao_cobertura", "fracao_interpolados", "motivo"]
POTENCIAS = ("k1_plano_D", "k2_curvo_D", "delta_k_D", "k_medio_D", "j0_D", "j45_D", "assimetria_D", "irregularidade_D")
NOMES = ("SINTETICO", "PACIENTE")          # nome gravado no XREF sintético: só pode aparecer na página impressa


@pytest.fixture(scope="module")
def exames(tmp_path_factory):
    d = tmp_path_factory.mktemp("ksaidas")
    write_export_folder_v1([
        SyntheticSpec(exam_id="K", R=7.8, Q=-0.2, cyl_D=2.0, axis_deg=30.0, eye="OD", decimals=4,
                      pupil_diameter=5.0, pupil_center=(0.3, -0.2)),
        SyntheticSpec(exam_id="K", R=7.8, Q=-0.2, eye="OS", decimals=4, pupil_registered=False),
    ], str(d))
    out = {}
    for eid in ("90001", "90002"):
        ex = import_tms_bundle(str(d), exam_id=eid)
        an, _ = analyze_exam(ex, with_maps=False)
        out[eid] = (ex, an)
    return {"dir": str(d), "exames": out}


def _csv_texto(an, tmp_path) -> str:
    p = tmp_path / f"{an.identity.exam_id}_{an.analysis_id[:6]}.csv"
    export_csv(an, str(p))
    with open(p, encoding="utf-8", newline="") as fh:       # preserva o \r\n do módulo csv
        return fh.read()


def _csv_kerato(an, tmp_path) -> str:
    p = tmp_path / f"{an.identity.exam_id}_{an.analysis_id[:6]}_ceratometria.csv"
    export_csv_ceratometria(an, str(p))
    with open(p, encoding="utf-8", newline="") as fh:
        return fh.read()


def _secao(txt: str):
    """CSV da ceratometria (arquivo próprio): cabeçalho na 1ª linha, uma linha por zona/anel."""
    rows = list(csv.reader(io.StringIO(txt), delimiter=";"))
    return rows, 0, rows[0], [r for r in rows[1:] if r]


def _csv_antigo(an) -> str:
    """Cópia congelada do export_csv da 0.3.1 (coeficientes + métricas): as seções lidas por posição não mudam."""
    buf = io.StringIO(newline="")
    w = csv.writer(buf, delimiter=";")
    w.writerow(["analysis_id", "study_id", "exam_id", "eye", "perfil", "centro", "centro_x_mm", "centro_y_mm",
                "diametro_mm", "plano_abertura", "ordem_max", "normalizacao", "validade", "n", "m", "j", "nome", "valor_um"])
    for wf in an.wavefronts:
        p = wf.profile
        base = [an.analysis_id, an.identity.study_id, an.identity.exam_id, an.identity.eye or "",
                p.name, p.center_mode, _fmt(wf.center_xy[0]), _fmt(wf.center_xy[1]), p.diameter_mm, p.aperture_plane,
                p.zernike_order, p.normalization, wf.validity]
        if wf.validity != "disponivel" or not wf.coefficients:
            w.writerow(base + ["", "", "", "BLOQUEADO: " + "; ".join(wf.block_reasons), ""])
            continue
        for c in wf.coefficients:
            w.writerow(base + [c.n, c.m, c.j, c.name, f"{c.value_um:.6f}"])
    w.writerow([])
    w.writerow(["# metricas (um) por perfil"])
    for wf in an.wavefronts:
        if wf.validity != "disponivel":
            continue
        for k, v in wf.metrics.items():
            w.writerow([an.analysis_id, "", "", "", wf.profile.name, "", "", "", "", "", "", "", "", "", "", "", k, _fmt(v)])
    return buf.getvalue()


def _f(s: str):
    return None if s == "" else float(s)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
def test_csv_secao_da_ceratometria_bate_com_o_objeto(exames, tmp_path):
    ex, an = exames["exames"]["90001"]
    k = an.keratometry
    rows, i, head, dados = _secao(_csv_kerato(an, tmp_path))
    assert head == COLUNAS and list(KERATO_CSV_COLUNAS) == COLUNAS and len(rows) == len(dados) + 1
    zonas = list(k.zonas) + list(k.discos) + [k.pupila]
    assert len(dados) == len(zonas) + 2 + len(k.aneis) and len(k.aneis) == 25
    assert all(len(r) == len(COLUNAS) for r in dados)
    assert all(r[:4] == [an.analysis_id, an.identity.study_id, "90001", "OD"] for r in dados)
    assert [r[4] for r in dados] == (["circunferencia"] * len(k.zonas) + ["zona"] * len(k.discos) + ["pupila"]
                                     + ["simk_calculado", "simk_tms"] + ["anel"] * 25)
    por_zona = {r[5]: dict(zip(COLUNAS, r)) for r in dados}
    assert len(por_zona) == len(dados)                                     # chaves estáveis e únicas
    assert {"3mm_vertice", "2.4mm_vertice", "zona_0-3mm_vertice", "pupila", "anel_1", "anel_25"} <= set(por_zona)
    for z, chave in zip(zonas + list(k.aneis), [r[5] for r in dados[:len(zonas)]] + [f"anel_{n}" for n in range(1, 26)]):
        r = por_zona[chave]
        assert r["validade"] == z.validity == "disponivel", (chave, z.block_reasons)
        for c in POTENCIAS:
            assert abs(_f(r[c]) - getattr(z, c)) <= 1e-6, (chave, c)
        assert abs(_f(r["eixo_k1_deg"]) - z.eixo_k1_deg) <= 0.005 and abs(_f(r["eixo_k2_deg"]) - z.eixo_k2_deg) <= 0.005
        assert abs(_f(r["diametro_mm"]) - z.diametro_mm) <= 5e-5
        assert abs(_f(r["fracao_cobertura"]) - z.fracao_cobertura) <= 5e-5 and r["motivo"] == ""
    assert por_zona["3mm_vertice"]["diametro_mm"] == "3.0000" and len(por_zona["3mm_vertice"]["k1_plano_D"].split(".")[1]) == 6
    assert len(por_zona["3mm_vertice"]["eixo_k1_deg"].split(".")[1]) == 2
    # SimK recalculado (regra do TMS) e SimK do XREF: no XREF, SimK1 é o CURVO → k2_curvo_D
    sc, t = k.simk_calculado, k.simk_tms
    rc, rt = por_zona["regra_tms_aneis_6_7_8"], por_zona["xref_tms"]
    assert abs(_f(rc["k1_plano_D"]) - sc["k1_plano_D"]) <= 1e-6 and abs(_f(rc["k2_curvo_D"]) - sc["k2_curvo_D"]) <= 1e-6
    assert diferenca_eixo(_f(rc["eixo_k2_deg"]), sc["eixo_k2_deg"]) <= 0.005
    assert abs(_f(rt["k1_plano_D"]) - t["SimK2"]) <= 1e-6 and abs(_f(rt["k2_curvo_D"]) - t["SimK1"]) <= 1e-6
    assert abs(_f(rt["delta_k_D"]) - t["CYL"]) <= 1e-6 and abs(_f(rt["k_medio_D"]) - (t["SimK1"] + t["SimK2"]) / 2) <= 1e-6
    assert diferenca_eixo(_f(rt["eixo_k2_deg"]), t["SimK1 Ang."]) <= 0.005 and 0.0 < _f(rt["eixo_k2_deg"]) <= 180.0
    assert rt["diametro_mm"] == "" and rt["j0_D"] == "" and rc["fracao_cobertura"] == ""
    assert keratometry_records(k)[len(zonas)]["tipo"] == "simk_calculado"


def test_csv_pupila_ausente_fica_vazia_com_motivo(exames, tmp_path):
    _, an = exames["exames"]["90002"]
    _, _, _, dados = _secao(_csv_kerato(an, tmp_path))
    rp = dict(zip(COLUNAS, next(r for r in dados if r[4] == "pupila")))
    assert rp["validade"] == "indisponivel" and "não registrada" in rp["motivo"]
    assert all(rp[c] == "" for c in COLUNAS[8:20])                         # ausência, não zero
    rt = dict(zip(COLUNAS, next(r for r in dados if r[4] == "simk_tms")))
    assert rt["eixo_k2_deg"] == "180.00"                                   # XREF grava 0°: TABO registra 180°
    assert "-0.000000" not in "".join(";".join(r) for r in dados)


def test_csv_secoes_anteriores_inalteradas(exames, tmp_path):
    for eid in ("90001", "90002"):
        _, an = exames["exames"][eid]
        txt = _csv_texto(an, tmp_path)
        antigo = _csv_antigo(an)
        assert txt.startswith(antigo)                                       # as seções da 0.3.1 não mudam
        rows = list(csv.reader(io.StringIO(txt), delimiter=";"))
        assert ";".join(rows[0]).startswith("analysis_id;study_id") and len(rows[0]) == 18
        # retangular: nenhuma linha passa das 18 colunas do cabeçalho (pandas/R leem sem tratar seções)
        assert max(len(r) for r in rows) == 18
        novas = list(csv.reader(io.StringIO(txt[len(antigo):]), delimiter=";"))
        assert novas and all(len(r) == 18 and r[4] == PUPILA_CSV_PERFIL for r in novas)
        k = list(csv.reader(io.StringIO(_csv_kerato(an, tmp_path)), delimiter=";"))
        assert {len(r) for r in k} == {len(COLUNAS)}


# ---------------------------------------------------------------------------
# Relatório em texto (páginas técnicas do PDF, aba Técnico, "Relatório em texto" do HTML)
# ---------------------------------------------------------------------------
def test_report_lines_secao_tabela_conferencia_e_configuracao(exames):
    _, an = exames["exames"]["90001"]
    L = report_lines(an)
    i = L.index(TITULO)
    assert L.index("## Geometria reconstruída") < i < next(j for j, ln in enumerate(L) if ln.startswith("## Perfil"))
    sec = L[i:next(j for j in range(i + 1, len(L)) if L[j].startswith("## "))]
    assert any("K1 plano (D)" in ln and "K2 curvo (D)" in ln for ln in sec)
    for ln in linhas_resumo(an.keratometry, incluir_aneis=True):          # tabela completa, inclusive os 25 anéis
        linha = next(s for s in sec if s.startswith("  " + ln["rotulo"] + " "))
        assert all(t in linha for t in ln["texto"]) and len(linha) <= 96, linha
    assert any(s.startswith("  pupila Ø 5.0 mm") for s in sec) and any(s.startswith("  anel 25 ·") for s in sec)
    assert any(s.startswith("  SimK do TMS") for s in sec) and any(s.startswith("SimK recalculado") for s in sec)
    conf = [s for s in sec if s.startswith("Conferência: ")]
    assert conf and "SimK recalculado" in conf[0] and conf[0].endswith("confere") and any("ACP" in c for c in conf)
    assert any("Pupila: registrada" in ln and "diâmetro 5.00 mm" in ln for ln in L)
    assert any(ln.startswith("Limites da ceratometria (provisórios)") for ln in L[L.index("## Qualidade e limitações"):])
    prov = L[L.index("## Proveniência"):]
    assert any(ln.startswith("Ceratometria: {") and "'versao': 'ceratometria_v2'" in ln and "'fonte': 'DIO'" in ln for ln in prov)


def test_report_lines_sem_pupila_e_acp_zero(exames):
    _, an = exames["exames"]["90002"]
    sec = report_lines(an)
    lp = next(s for s in sec if s.startswith("  pupila — não registrada"))
    assert "INDISPONÍVEL" in lp and "não registrada no exame" in lp
    assert any("ACP não informado" in s for s in sec)                      # o TMS grava ACP = 0 sem pupila


def test_texto_corrido_quebrado_nao_vira_monoespacado_e_paginas_nao_perdem_linha():
    longa = "palavra " * 60
    partes = _wrap([longa, "  " + longa])
    prosa = [p for p in partes if p.strip() and not p.startswith("  ")]
    assert len(prosa) >= 3 and all(len(p) <= 118 for p in prosa)          # continuação com espaço não separável
    mono = [p for p in partes if p.startswith("  ")]
    assert len(mono) >= 4 and all(len(p) <= 98 for p in mono)
    linhas = []
    for n in range(40):                                                    # muitos títulos: a paginação fixa perdia linhas
        linhas += [f"## título {n}", f"linha {n}a", f"linha {n}b"]
    pags = _paginar(linhas)
    assert [ln for p in pags for ln in p] == linhas
    for p in pags:
        y = _Y_TOPO
        for ln in p:
            assert y >= _Y_BASE                                            # _text_page desenha a linha
            y -= _H_TITULO if ln.startswith("## ") else _H_LINHA
        assert not p[-1].startswith("## ")                                 # título nunca sozinho no pé da página


# ---------------------------------------------------------------------------
# HTML e JSON da página
# ---------------------------------------------------------------------------
def test_report_html_tabela_com_linha_da_pupila_sem_nome(exames):
    r = analyze_exam_html(exames["dir"], "90001", {"pdf": False, "maps": False, "include_name": True})
    h = r["html"]
    assert "PACIENTE SINTETICO" in h[: h.index("class='detalhes'")]      # nome só no cabeçalho (opção do usuário)
    i = h.index("<section class='kerato'>")
    sec = h[i:h.index("</section>", i)]
    assert h.index("class='detalhes'") < i and "Ceratometria por zona" in sec and "<table class='metricas kerato'>" in sec
    assert "<tr class='kerato-pupila'><td>pupila Ø 5.0 mm</td>" in sec
    assert "<tr class='kerato-ref'><td>SimK do TMS" in sec and "<tr class='kerato-zona'><td>zona 0–3 mm" in sec
    assert "<details class='kerato-aneis'>" in sec and "anel 25 · Ø" in sec[sec.index("<details class='kerato-aneis'>"):]
    assert "Conferência: SimK recalculado" in sec and not any(n in sec for n in NOMES)
    ex2, an2 = exames["exames"]["90002"]
    h2 = report_html(ex2, an2, with_maps=False, patient_name=None)
    assert "<tr class='kerato-pupila'><td>pupila — não registrada</td><td colspan='6' class='motivo'>INDISPONÍVEL" in h2
    assert not any(n in h2 for n in NOMES)


def test_summary_keratometry_no_json_da_pagina(exames):
    _, an = exames["exames"]["90001"]
    r = analyze_exam_html(exames["dir"], "90001", {"pdf": False, "maps": False, "include_name": False})
    json.dumps(r["summary"], allow_nan=False)                              # a página faz JSON.parse: sem NaN
    s = r["summary"]["keratometry"]
    k = an.keratometry
    assert [ln["tipo"] for ln in s["linhas"]] == [ln["tipo"] for ln in linhas_resumo(k)]
    l3 = next(ln for ln in s["linhas"] if ln["tipo"] == "circunferencia" and ln["diametro_mm"] == 3.0)
    z3 = next(z for z in k.zonas if z.diametro_mm == 3.0)
    assert abs(l3["k1_plano_D"] - z3.k1_plano_D) < 1e-9 and abs(l3["k2_curvo_D"] - z3.k2_curvo_D) < 1e-9
    assert abs(l3["eixo_k2_deg"] - z3.eixo_k2_deg) < 1e-9 and abs(l3["delta_k_D"] - z3.delta_k_D) < 1e-9
    lp = next(ln for ln in s["linhas"] if ln["tipo"] == "pupila")
    assert lp["destaque"] == "pupila" and abs(lp["k_medio_D"] - k.pupila.k_medio_D) < 1e-9
    lt = next(ln for ln in s["linhas"] if ln["tipo"] == "simk_tms")
    assert lt["k1_plano_D"] == k.simk_tms["SimK2"] and lt["k2_curvo_D"] == k.simk_tms["SimK1"]
    assert s["conferencia"] == k.conferencia and s["alertas"] == list(k.alertas)
    s2 = analyze_exam_html(exames["dir"], "90002", {"pdf": False, "maps": False})["summary"]["keratometry"]
    lp2 = next(ln for ln in s2["linhas"] if ln["tipo"] == "pupila")
    assert lp2["validity"] == "indisponivel" and lp2["k1_plano_D"] is None and "não registrada" in lp2["motivo"]


# ---------------------------------------------------------------------------
# Linha de comando
# ---------------------------------------------------------------------------
def test_long_rows_tem_j0_j45_e_nunca_eixo_cru(exames):
    _, an = exames["exames"]["90001"]
    rows = _long_rows(an)
    met = {r["metrica"]: r for r in rows}
    for chave in ("3mm_vertice", "pupila", "zona_0-3mm_vertice"):
        for m in ("k1_plano_D", "k2_curvo_D", "delta_k_D", "k_medio_D", "j0_D", "j45_D"):
            assert f"{m}@{chave}" in met, (m, chave)
    assert not any(r["metrica"].startswith("eixo_k") or "eixo_k" in r["metrica"] for r in rows)
    z3 = next(z for z in an.keratometry.zonas if z.diametro_mm == 3.0)
    r3 = met["j0_D@3mm_vertice"]
    assert abs(float(r3["valor"]) - z3.j0_D) <= 1e-6 and abs(float(met["k1_plano_D@3mm_vertice"]["valor"]) - z3.k1_plano_D) <= 1e-6
    assert r3["exame"] == "90001" and r3["participante"] == an.identity.study_id and r3["perfil"] == "ceratometria_v2"
    assert r3["metodo"] == "TMS_calc" and r3["olho"] == "OD" and r3["analise"] == an.analysis_id
    assert abs(float(met["diametro_pupila_mm"]["valor"]) - an.pupil.diameter) <= 1e-4
    assert abs(math.hypot(float(met["j0_D@pupila"]["valor"]), float(met["j45_D@pupila"]["valor"])) - an.keratometry.pupila.delta_k_D / 2) < 1e-5
    _, an2 = exames["exames"]["90002"]
    met2 = {r["metrica"]: r["valor"] for r in _long_rows(an2)}
    assert met2["validade@pupila"] == "indisponivel" and "k1_plano_D@pupila" not in met2 and "diametro_pupila_mm" not in met2
    assert not any(n in json.dumps(rows + _long_rows(an2), ensure_ascii=False) for n in NOMES)


def test_print_analysis_resumo_curto_da_ceratometria(exames, capsys):
    an = exames["exames"]["90001"][1]
    _print_analysis(an)
    out = capsys.readouterr().out
    t = next(ln["texto"] for ln in linhas_resumo(an.keratometry) if ln["rotulo"] == "Ø 3 mm")
    assert "ceratometria (potência axial do TMS" in out and f"Ø 3 mm: K1 {t[0]} D @ {t[1]} / K2 {t[2]} D @ {t[3]}" in out
    assert t[1] == "30°" and t[3] == "120°"                               # K1 plano no eixo do gerador
    assert "zona 0–3 mm: K1" in out and "pupila Ø 5.0 mm: K1" in out and "SimK do TMS" in out
    assert "conferência: SimK recalculado" in out and "anel" not in out    # resumo curto: sem os 25 anéis
    _print_analysis(exames["exames"]["90002"][1])
    assert "pupila — não registrada" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# LGPD e compatibilidade
# ---------------------------------------------------------------------------
def test_nenhum_nome_nas_saidas_tecnicas(exames, tmp_path):
    from pypdf import PdfReader
    from tmsab.adapters.tms4 import audit_reports_for_names
    from tmsab.reporting import export_all
    ex, an = exames["exames"]["90001"]
    out = tmp_path / "analises"
    paths = export_all(ex, an, str(out), patient_name=None)
    for k in ("json", "csv", "csv_ceratometria"):
        txt = open(paths[k], encoding="utf-8").read()
        assert not any(n in txt.upper() for n in NOMES), k
    assert not any(n in json.dumps(canonical_result(an), ensure_ascii=False) for n in NOMES)
    assert not any(n in "\n".join(report_lines(an)).upper() for n in NOMES)
    paginas = [(p.extract_text() or "") for p in PdfReader(paths["pdf"]).pages[1:]]
    assert any("Ceratometria por zona" in t for t in paginas) and any("anel 25" in t for t in paginas)
    res = audit_reports_for_names(exames["dir"], str(out))
    assert res["n_hits"] == 0 and res["n_hits_pagina_completa"] == 0 and res["n_files"] == 4     # JSON, 2 CSV, PDF


def test_analise_sem_ceratometria_nao_quebra_as_saidas(exames, tmp_path, capsys):
    ex, an = exames["exames"]["90001"]
    velho = dataclasses.replace(an, keratometry=None)                      # resultado anterior à 0.4.0
    _, _, head, dados = _secao(_csv_kerato(velho, tmp_path))
    assert head == COLUNAS and len(dados) == 1 and dados[0][7] == "indisponivel" and "não calculada" in dados[0][-1]
    L = report_lines(velho)
    assert TITULO in L and any("não calculada" in ln for ln in L)
    assert "Ceratometria não calculada" in report_html(ex, velho, with_maps=False)
    assert not any("@" in r["metrica"] and r["perfil"] == "ceratometria_v2" for r in _long_rows(velho))
    _print_analysis(velho)
    assert "ceratometria" not in capsys.readouterr().out
