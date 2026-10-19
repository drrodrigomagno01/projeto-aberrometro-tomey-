"""Controle de diâmetro da ceratometria na versão HTML (tmsab.web): bloco .kc no topo da seção de ceratometria, mapa PNG
com a caixa do quadro para a sobreposição SVG, grade JSON, linhas já preenchidas no diâmetro inicial e o código
CONTROLE_JS (página, relatório salvo e report_document).

Só exportações sintéticas no formato real (v1): o repositório é público. O comportamento no navegador (clique, arraste,
teclado, relatório salvo) é conferido à parte com Playwright no Chrome e no WebKit.
"""
import base64
import copy
import html
import io
import json
import math
import os
import re
import shutil
import subprocess

import matplotlib.image as mpimg
import numpy as np
import pytest
from matplotlib.figure import Figure

from tmsab.adapters import import_tms_bundle
from tmsab.keratometry import ceratometria_no_diametro, dados_do_controle, linha_da_zona
from tmsab.pipeline import analyze_exam
from tmsab.plots import plot_topo_map
from tmsab.synthetic import SyntheticSpec, write_export_folder_v1
from tmsab.web import CONTROLE_JS, _controle_html, analyze_exam_html, report_document, report_html

NOMES = ("SINTETICO", "PACIENTE")          # nome gravado no XREF sintético: só pode aparecer no cabeçalho


@pytest.fixture(scope="module")
def exames(tmp_path_factory):
    d = tmp_path_factory.mktemp("kc_web")
    write_export_folder_v1([
        SyntheticSpec(exam_id="W", R=7.8, Q=-0.2, cyl_D=2.0, axis_deg=30.0, eye="OD", decimals=4,
                      pupil_diameter=5.0, pupil_center=(0.3, -0.2)),
        SyntheticSpec(exam_id="W", R=7.8, Q=-0.2, eye="OS", decimals=4, pupil_registered=False),
    ], str(d))
    out = {"dir": str(d)}
    for eid in ("90001", "90002"):
        ex = import_tms_bundle(str(d), exam_id=eid)
        an, _ = analyze_exam(ex, with_maps=False)
        h = analyze_exam_html(str(d), eid, {"pdf": False, "include_name": True})["html"]
        out[eid] = {"ex": ex, "an": an, "html": h}
    return out


def _blocos(h: str) -> list[str]:
    return re.findall(r"<div class='kc'>.*?</script></div>", h, flags=re.S)


def _partes(blk: str) -> dict:
    raw = re.search(r"<script type='application/json' class='kc-dados'>(.*?)</script>", blk, flags=re.S).group(1)
    corpo = re.search(r"<table class='metricas kerato kc-tabela'>.*?<tbody>(.*?)</tbody>", blk, flags=re.S).group(1)
    return {
        "raw": raw, "dados": json.loads(raw),
        "png": base64.b64decode(re.search(r"<img class='kc-img' src='data:image/png;base64,([^']+)'", blk).group(1)),
        "img_wh": tuple(int(v) for v in re.search(r"<img class='kc-img' [^>]*width='(\d+)' height='(\d+)'", blk).groups()),
        "svg": re.search(r"<svg class='kc-sobre' viewBox='([^']+)' style='([^']+)'", blk).groups(),
        "range": re.search(r"<input class='kc-d' type='range' min='([^']+)' max='([^']+)' step='([^']+)' value='([^']+)'>",
                           blk).groups(),
        "saida": re.search(r"<output class='kc-valor' aria-live='polite'>([^<]*)</output>", blk).group(1),
        "linhas_html": re.findall(r"<tr[^>]*>.*?</tr>", corpo),
        "linhas": [[html.unescape(c) for c in re.findall(r"<td[^>]*>(.*?)</td>", tr)]
                   for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", corpo)],
    }


@pytest.mark.parametrize("eid", ["90001", "90002"])
def test_um_bloco_no_topo_da_secao_de_ceratometria(exames, eid):
    h = exames[eid]["html"]
    assert len(_blocos(h)) == 1 and h.count("<div class='kc'>") == 1
    i = h.index("<section class='kerato'>")
    fim = h.index("</section>", i)
    assert i < h.index("<div class='kc'>") < h.index("<table class='metricas kerato'>", i) < fim


def test_mapa_png_com_quadro_quadrado_alinhado_e_meia_largura_do_mapa(exames):
    ex = exames["90001"]["ex"]
    p = _partes(_blocos(exames["90001"]["html"])[0])
    geo = p["dados"]["mapa"]
    img = mpimg.imread(io.BytesIO(p["png"]))
    H, W = img.shape[:2]
    assert (W, H) == p["img_wh"] == (528, 440)
    esq, topo, larg, alt = geo["caixa"]
    x0, y0, w, h = esq * W, topo * H, larg * W, alt * H
    assert abs(w - h) <= 1.0                                         # quadro quadrado em pixels
    assert x0 >= 0 and y0 >= 0 and x0 + w <= W and y0 + h <= H      # dentro da imagem
    # a caixa é o quadro desenhado: as bordas (eixos pretos) caem nas 4 arestas da caixa
    lum = img[..., :3].mean(axis=2)
    linhas, colunas = np.arange(int(y0) + 6, int(y0 + h) - 6), np.arange(int(x0) + 6, int(x0 + w) - 6)
    for borda in (x0, x0 + w):
        faixa = lum[linhas][:, int(round(borda)) - 1:int(round(borda)) + 2]
        assert (faixa.min(axis=1) < 0.35).mean() > 0.95, borda
    for borda in (y0, y0 + h):
        faixa = lum[int(round(borda)) - 1:int(round(borda)) + 2][:, colunas]
        assert (faixa.min(axis=0) < 0.35).mean() > 0.95, borda
    # L = meia largura do mapa de plot_topo_map (independe do tamanho da figura)
    ax = Figure().add_subplot()
    plot_topo_map(ax, ex, "DIO", zones=())
    assert geo["L"] == ax.get_xlim()[1] == ax.get_ylim()[1] and ax.get_xlim()[0] == -geo["L"]
    L = geo["L"]
    assert p["svg"][0] == f"{-L:.6f} {-L:.6f} {2 * L:.6f} {2 * L:.6f}"
    estilo = dict(kv.split(":") for kv in p["svg"][1].split(";"))
    for k, v in zip(("left", "top", "width", "height"), geo["caixa"]):
        assert abs(float(estilo[k].rstrip("%")) - 100 * v) < 1e-3, k
    # posição em mm → pixel: o disco colorido termina no último anel (meridianos 0° e 90°), a ±3 px
    sat = img[..., :3].max(axis=2) - img[..., :3].min(axis=2)
    cx, cy, px_mm = x0 + w / 2, y0 + h / 2, w / (2 * L)
    th = np.asarray(ex.samples.theta_deg)
    r_ult = np.asarray(ex.samples.r)[-1]
    j0, j90 = int(np.argmin(np.abs(th - 0.0))), int(np.argmin(np.abs(th - 90.0)))
    linha_c = sat[int(round(cy))]
    borda_x = max(i for i in range(int(cx), int(x0 + w)) if linha_c[i] > 0.15)
    assert abs((borda_x + 0.5) - (cx + r_ult[j0] * px_mm)) <= 3.0, (borda_x, cx + r_ult[j0] * px_mm)
    coluna_c = sat[:, int(round(cx))]
    borda_y = min(i for i in range(int(y0), int(cy)) if coluna_c[i] > 0.15)
    assert abs((borda_y + 0.5) - (cy - r_ult[j90] * px_mm)) <= 3.0, (borda_y, cy - r_ult[j90] * px_mm)


@pytest.mark.parametrize("eid", ["90001", "90002"])
def test_controle_e_grade_json_iguais_ao_nucleo(exames, eid):
    ex, an = exames[eid]["ex"], exames[eid]["an"]
    p = _partes(_blocos(exames[eid]["html"])[0])
    dados = dados_do_controle(ex, an.keratometry.config, an.quality_config)
    assert {k: v for k, v in p["dados"].items() if k != "mapa"} == dados
    assert set(p["dados"]) == set(dados) | {"mapa"} and set(p["dados"]["mapa"]) == {"L", "caixa"}
    mn, mx, passo, valor = (float(v) for v in p["range"])
    assert (mn, mx, passo, valor) == (dados["dmin"], dados["dmax"], 0.1, 3.0) == (dados["dmin"], dados["dmax"],
                                                                                   dados["passo"], dados["inicial"])
    assert p["saida"] == "3.0 mm"
    # bloco de dados seguro dentro de <script>: nenhum "<" literal (motivos como "… 80.0% < 85%" voltam pelo JSON.parse)
    assert "<" not in p["raw"] and "</" not in p["raw"]
    com_menor = [pt["d"] for pt in dados["pontos"] if "<" in pt["circ"]["motivo"] + pt["zona"]["motivo"]]
    assert ("\\u003c" in p["raw"]) == bool(com_menor)
    if eid == "90001":                                               # tórica: borda externa parcial (cobertura < 85%)
        assert com_menor


@pytest.mark.parametrize("eid", ["90001", "90002"])
def test_linhas_preenchidas_no_diametro_inicial_sao_as_da_tabela(exames, eid):
    ex, an, h = exames[eid]["ex"], exames[eid]["an"], exames[eid]["html"]
    p = _partes(_blocos(h)[0])
    zc, zd = ceratometria_no_diametro(ex, 3.0, an.keratometry.config, an.quality_config)
    esperado = [[ln["rotulo"], *ln["texto"]] for ln in (linha_da_zona(zc), linha_da_zona(zd, "zona"))]
    assert p["linhas"] == esperado
    assert p["linhas"][0][0].startswith("Ø 3 mm") and p["linhas"][1][0].startswith("zona 0–3 mm")
    # a mesma linha (mesmo HTML) da tabela completa da seção
    i = h.index("<table class='metricas kerato'>")
    tabela = h[i:h.index("</table>", i)]
    assert p["linhas_html"][0] in tabela and p["linhas_html"][1] in tabela
    assert p["linhas_html"][1].startswith("<tr class='kerato-zona'>")
    if eid == "90002":                                               # sem astigmatismo: eixo pouco definido (*)
        p3 = next(pt for pt in p["dados"]["pontos"] if pt["d"] == 3.0)
        assert p["linhas"][0][0] == "Ø 3 mm *" and p3["circ"]["pouco_definido"]


def test_sem_mapas_nao_ha_controle(exames):
    r = analyze_exam_html(exames["dir"], "90001", {"pdf": False, "maps": False})
    assert "class='kc" not in r["html"] and "kc-dados" not in r["html"]
    assert "class='kc" not in report_html(exames["90001"]["ex"], exames["90001"]["an"], with_maps=False)


def test_nome_do_paciente_nunca_no_controle(exames):
    h = exames["90001"]["html"]
    assert "PACIENTE SINTETICO" in h[: h.index("class='detalhes'")]    # opção ligada: só no cabeçalho
    blk = _blocos(h)[0]
    assert not any(n in blk.upper() for n in NOMES)
    assert not any(n in json.dumps(_partes(blk)["dados"], ensure_ascii=False).upper() for n in NOMES)


def test_sem_dio_o_bloco_mostra_o_motivo(exames):
    ex, an = exames["90001"]["ex"], exames["90001"]["an"]
    sem_dio = copy.deepcopy(ex)
    del sem_dio.samples.fields["DIO"]
    blk = _controle_html(sem_dio, an)
    p = _partes(blk)
    assert p["dados"]["mapa"]["L"] > 0 and mpimg.imread(io.BytesIO(p["png"])).shape[:2] == (440, 528)
    assert all(not pt["circ"]["disponivel"] and not pt["zona"]["disponivel"] for pt in p["dados"]["pontos"])
    assert p["linhas"][0] == ["Ø 3 mm", "INDISPONÍVEL — tabela DIO ausente na exportação"]
    assert p["linhas"][1] == ["zona 0–3 mm", "INDISPONÍVEL — tabela DIO ausente na exportação"]


def test_codigo_do_controle_uma_vez_no_documento_e_sem_fechar_o_script(exames):
    js = CONTROLE_JS.lower()
    assert "</script" not in js and "<!--" not in js and "<script" not in js
    assert CONTROLE_JS.count("function kcBind(") == 1
    doc = report_document([exames["90001"]["html"], exames["90002"]["html"]])
    assert doc.count(CONTROLE_JS) == 1 and doc.count("function kcBind(") == 1
    assert doc.index("kcBind(document)") > doc.rindex("<div class='kc'>")
    assert doc.count("<div class='kc'>") == 2


def test_pagina_gerada_liga_o_controle_e_o_relatorio_salvo(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_html", os.path.join(os.path.dirname(__file__), "..", "web", "build_html.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    txt = open(mod.build(str(tmp_path)), encoding="utf-8").read()
    assert not re.search(r"\{\{[A-Z_]+\}\}", txt)
    assert txt.count(CONTROLE_JS) == 1 and txt.count("function kcBind(") == 1
    assert f'<script id="kc-codigo">{CONTROLE_JS}</script>' in txt
    assert "kcBind(art)" in txt                                       # cada exame, ao entrar na página
    i = txt.index('$("btnSaveHtml").onclick')                        # relatório salvo: código + kcBind(document)
    salvar = txt[i:txt.index("};", i)]
    assert '$("kc-codigo").textContent' in salvar and "kcBind(document)" in salvar


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js ausente")
def test_controle_js_e_javascript_valido(tmp_path):
    f = tmp_path / "controle.js"
    f.write_text(CONTROLE_JS, encoding="utf-8")
    r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr


# ---------------------------------------------------------------------------
# Comportamento de CONTROLE_JS sem navegador: o código roda no Node sobre um DOM falso mínimo
# (tests/controle_js_dom_falso.js) e o que ele desenha é conferido contra a grade calculada pelo núcleo.
# ---------------------------------------------------------------------------
COR_K1, COR_K2 = "#2a78d6", "#eb6834"       # K1 plano azul, K2 curvo laranja: as cores do programa e da legenda
DOM_FALSO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "controle_js_dom_falso.js")
sem_node = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js ausente")


def _roda_js(tmp_path, raw: str, passos: list) -> dict:
    (tmp_path / "controle.js").write_text(CONTROLE_JS, encoding="utf-8")
    (tmp_path / "dados.json").write_text(raw, encoding="utf-8")
    (tmp_path / "passos.json").write_text(json.dumps(passos), encoding="utf-8")
    r = subprocess.run(["node", DOM_FALSO, *(str(tmp_path / f) for f in ("controle.js", "dados.json", "passos.json"))],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def _linha_esperada(ln: dict) -> list[str]:
    if ln["disponivel"]:
        return [ln["rotulo"], *ln["texto"]]
    estado = "INVÁLIDA" if ln["validity"] == "invalida" else "INDISPONÍVEL"
    return [ln["rotulo"], estado + (f" — {ln['motivo']}" if ln["motivo"] else "")]


def _confere_desenho(est: dict, p: dict) -> None:
    """Círculo r = d/2; meridianos pelo vértice nos eixos TABO da circunferência (anti-horário na tela a partir das
    3 h), com 1,12·d/2 de cada lado, nas cores K1 azul / K2 laranja; rótulos e as duas linhas da tabela."""
    d, c = p["d"], p["circ"]
    assert est["saida"] == f"{d:.1f} mm" and est["valor"] == est["valorAttr"] == f"{d:.1f}"
    assert est["r_circulos"] == pytest.approx([d / 2, d / 2])
    assert est["linhas"] == [_linha_esperada(c), _linha_esperada(p["zona"])] and est["classes"] == ["", "kerato-zona"]
    sy = float(re.fullmatch(r"scale\(1,\s*(-?1)\)", est["transform"]).group(1))
    assert sy == -1.0                                   # grupo com y para cima, como o mapa (vista do examinador)
    if not c["disponivel"]:
        assert est["circulo"] == {"cor": "#666", "tracejado": "6 4"}
        assert est["k1"] is est["k2"] is est["rot_k1"] is est["rot_k2"] is None
        return
    assert est["circulo"] == {"cor": "#111", "tracejado": None}
    for nome, cor in (("k1", COR_K1), ("k2", COR_K2)):
        x1, y1, x2, y2 = est[nome]["xy"]
        ang_tela = math.degrees(math.atan2(-sy * (y2 - y1), x2 - x1))
        assert abs((ang_tela - c["eixo_" + nome] + 90.0) % 180.0 - 90.0) < 0.01, (d, nome, ang_tela)
        assert math.hypot(x2 - x1, y2 - y1) == pytest.approx(1.12 * d) and (x1, y1) == pytest.approx((-x2, -y2))
        assert est[nome]["cor"] == cor and est["rot_" + nome]["cor"] == cor
        assert est[nome]["tracejado"] == ("7 5" if c["pouco_definido"] else None)
    assert est["rot_k2"]["texto"] == "K2 " + c["texto"][3] and est["rot_k1"]["texto"] == "K1 " + c["texto"][1]


@sem_node
def test_controle_js_desenha_o_que_o_nucleo_calculou(exames, tmp_path):
    raw = _partes(_blocos(exames["90001"]["html"])[0])["raw"]
    dados = json.loads(raw)
    por_d = {round(p["d"] * 10): p for p in dados["pontos"]}
    indisp = [p for p in dados["pontos"] if not p["circ"]["disponivel"]]
    assert indisp, "a grade do exame tórico deveria ter diâmetros sem cobertura"
    ds = [2.4, 3.0, 4.0, 6.0, indisp[0]["d"], indisp[-1]["d"]]
    res = _roda_js(tmp_path, raw, [{"d": f"{d:.1f}"} for d in ds])
    _confere_desenho(res["inicial"], por_d[30])
    for d, est in zip(ds, res["passos"]):
        _confere_desenho(est, por_d[round(d * 10)])
    # o exame do gerador: K1 plano no meridiano de 30°, K2 curvo a 120°
    assert res["passos"][1]["rot_k1"]["texto"] == "K1 30°" and res["passos"][1]["rot_k2"]["texto"] == "K2 120°"


@sem_node
def test_controle_js_eixo_pouco_definido_e_linha_invalida(exames, tmp_path):
    raw = _partes(_blocos(exames["90002"]["html"])[0])["raw"]      # esfera: eixo pouco definido (ΔK baixo)
    dados = json.loads(raw)
    p3 = next(p for p in dados["pontos"] if p["d"] == 3.0)
    assert p3["circ"]["disponivel"] and p3["circ"]["pouco_definido"]
    res = _roda_js(tmp_path, raw, [{"d": "3.0"}])
    _confere_desenho(res["passos"][0], p3)
    # ajuste de Fourier sem amostras: "INVÁLIDA", como na tabela (não "INDISPONÍVEL")
    p3["circ"] = {**p3["circ"], "rotulo": "Ø 3 mm", "disponivel": False, "validity": "invalida", "texto": ["—"] * 6,
                  "motivo": "ajuste de Fourier sem amostras suficientes", "eixo_k1": None, "eixo_k2": None,
                  "pouco_definido": False}
    res = _roda_js(tmp_path, json.dumps(dados), [{"d": "3.0"}])
    assert res["passos"][0]["linhas"][0] == ["Ø 3 mm", "INVÁLIDA — ajuste de Fourier sem amostras suficientes"]
    _confere_desenho(res["passos"][0], p3)


@sem_node
def test_controle_js_clique_e_arrasto_no_mapa(exames, tmp_path):
    raw = _partes(_blocos(exames["90001"]["html"])[0])["raw"]
    dados = json.loads(raw)
    L = dados["mapa"]["L"]

    def tela(x, y):                                     # mm do mapa (y para cima) → pixel da sobreposição (440 px)
        return (x + L) / (2 * L) * 440, (L - y) / (2 * L) * 440

    def ponteiro(*eventos):
        return {"ponteiro": [[tipo, *tela(x, y)] for tipo, x, y in eventos]}

    passos = [
        ponteiro(("pointerdown", 1.5, 0.0), ("pointerup", 1.5, 0.0)),            # Ø = 2·r
        ponteiro(("pointerdown", 0.0, 2.0), ("pointerup", 0.0, 2.0)),
        ponteiro(("pointerdown", -1.2, -1.6), ("pointerup", -1.2, -1.6)),
        ponteiro(("pointerdown", 1.0, 0.0), ("pointermove", 0.0, 2.5)),          # arrasto: segue o ponteiro
        ponteiro(("pointerup", 0.0, 2.5), ("pointermove", 0.0, 1.0)),            # depois de soltar: não muda
        ponteiro(("pointerdown", L, L), ("pointerup", L, L)),                    # canto do mapa: limita ao máximo
    ]
    res = _roda_js(tmp_path, raw, passos)
    assert [e["saida"] for e in res["passos"]] == ["3.0 mm", "4.0 mm", "4.0 mm", "5.0 mm", "5.0 mm",
                                                  f"{dados['dmax']:.1f} mm"]
    por_d = {round(p["d"] * 10): p for p in dados["pontos"]}
    for est in res["passos"]:
        _confere_desenho(est, por_d[round(float(est["valor"]) * 10)])


def test_cores_do_controle_do_site_sao_as_do_programa():
    assert re.search(r'const COR_K1 = "(#[0-9a-f]{6})", COR_K2 = "(#[0-9a-f]{6})"', CONTROLE_JS).groups() == (COR_K1, COR_K2)
