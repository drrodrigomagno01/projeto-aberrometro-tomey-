"""Ceratometria por zona (tmsab/keratometry.py): K1 plano, K2 curvo e eixos por diâmetro.

Verdade conhecida em três níveis:
1. harmônicos puros (o ajuste de Fourier e a convenção de eixos);
2. tórica com Q = 0 no formato real v1 (potência axial constante ao longo de cada meridiano principal);
3. tórica asférica (Q = −0,2): a referência é a potência AXIAL analítica da própria superfície, amostrada na
   circunferência e ajustada à parte — independente da interpolação entre anéis feita pelo módulo.
O gerador v1 grava DIO = potência axial (como o TMS-4N; antes gravava a tangencial).
Os exames reais nunca entram aqui (repositório público): o teste com eles só roda localmente, com
TMSAB_REAL_DATA apontando para a pasta (ver docs/validacao.md).
"""
import json
import math
import os

import numpy as np
import pytest

from tmsab.adapters import import_tms_bundle
from tmsab.adapters.tms4 import list_exams
from tmsab.keratometry import (
    alerta_dependencia_zona,
    compute_keratometry,
    diferenca_eixo,
    eixo_tabo,
    fmt_eixo,
    fourier_fit,
    linhas_resumo,
    simk_regra_tms,
)
from tmsab.model import KeratometryConfig, KeratometryResult, KeratometryZone, PointState, QualityConfig, to_jsonable
from tmsab.pipeline import analyze_exam
from tmsab.synthetic import SyntheticSpec, build_surface, write_export_folder_v1


# ---------------------------------------------------------------------------
# 1. Ajuste de Fourier e convenção de eixos
# ---------------------------------------------------------------------------
def _harmonico(km, dk, eixo_curvo_deg, n=256, quant=4):
    th = np.radians(np.arange(n) * 360.0 / n)
    p = km + (dk / 2.0) * np.cos(2.0 * (th - math.radians(eixo_curvo_deg)))
    return th, np.round(p, quant)


@pytest.mark.parametrize("eixo", [0.0, 30.0, 90.0, 135.0, 179.5])
def test_fourier_recupera_k1_k2_e_eixos(eixo):
    th, p = _harmonico(44.0, 1.5, eixo)
    f = fourier_fit(th, p)
    assert abs(f["a0"] - 44.0) < 1e-4 and abs(2 * f["h2"] - 1.5) < 1e-4
    assert diferenca_eixo(f["eixo_k2_deg"], eixo) < 0.01
    assert 0.0 < f["eixo_k2_deg"] <= 180.0


def test_convencao_tabo_e_formatacao():
    assert eixo_tabo(0.0) == 180.0 and eixo_tabo(180.0) == 180.0 and eixo_tabo(-30.0) == 150.0
    assert fmt_eixo(0.4) == "180°" and fmt_eixo(179.6) == "180°" and fmt_eixo(0.6) == "1°" and fmt_eixo(None) == "—"
    assert diferenca_eixo(179.0, 2.0) == 3.0 and diferenca_eixo(10.0, 100.0) == 90.0


def test_fourier_com_poucas_amostras_devolve_none():
    th, p = _harmonico(44.0, 1.0, 10.0, n=8)
    assert fourier_fit(th, p) is None


def test_simk_regra_tms_em_potencia_construida():
    # anéis 6–8 com astigmatismo puro: curvo no meridiano de 60°, plano a 150°
    na, nr = 256, 25
    th_deg = np.arange(na) * 360.0 / na
    dio = np.tile(44.0 + 0.75 * np.cos(2 * np.radians(th_deg - 60.0)), (nr, 1))
    st = np.zeros((nr, na), dtype=np.int8)
    s = simk_regra_tms(dio, st, th_deg)
    assert abs(s["k2_curvo_D"] - 44.75) < 1e-3 and abs(s["k1_plano_D"] - 43.25) < 1e-3
    assert diferenca_eixo(s["eixo_k2_deg"], 60.0) <= 0.71 and diferenca_eixo(s["eixo_k1_deg"], 150.0) <= 0.71


# ---------------------------------------------------------------------------
# 2–3. Formato real v1 com superfície conhecida
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def toricas(tmp_path_factory):
    d = tmp_path_factory.mktemp("kerato_v1")
    casos = [
        SyntheticSpec(exam_id="K", R=7.8, Q=0.0, cyl_D=2.0, axis_deg=30.0, eye="OD", decimals=4, pupil_diameter=5.0, pupil_center=(0.3, -0.2)),
        SyntheticSpec(exam_id="K", R=7.8, Q=0.0, cyl_D=0.5, axis_deg=172.0, eye="OS", decimals=4),
        SyntheticSpec(exam_id="K", R=7.6, Q=0.0, cyl_D=1.25, axis_deg=0.0, eye="OD", decimals=4),
        SyntheticSpec(exam_id="K", R=7.8, Q=0.0, cyl_D=2.0, axis_deg=90.0, eye="OS", decimals=4),
        SyntheticSpec(exam_id="K", R=7.8, Q=-0.2, cyl_D=2.0, axis_deg=30.0, eye="OD", decimals=4),
        SyntheticSpec(exam_id="K", R=7.8, Q=-0.2, eye="OD", decimals=4, pupil_registered=False),
        SyntheticSpec(exam_id="K", R=7.8, Q=-0.2, cyl_D=1.0, axis_deg=10.0, eye="OD", decimals=4,
                      lid_gap_from_ring=16, lid_gap_half_angle_deg=30.0),
        SyntheticSpec(exam_id="K", R=7.8, Q=-0.2, cyl_D=1.0, axis_deg=10.0, eye="OD", decimals=4,
                      lid_gap_from_ring=12, lid_gap_half_angle_deg=35.0, lid_gap_as_interpolated=False),
        SyntheticSpec(exam_id="K", R=7.7, Q=-0.15, cyl_D=1.5, axis_deg=100.0, eye="OS", decimals=4,
                      pupil_diameter=4.0, pupil_center=(0.0, 0.0)),
    ]
    write_export_folder_v1(casos, str(d))
    exames = [import_tms_bundle(str(d), exam_id=str(90001 + i)) for i in range(len(casos))]
    return {"dir": str(d), "specs": casos, "exams": exames}


@pytest.mark.parametrize("i", [0, 1, 2, 3])
def test_torica_q0_k1_plano_k2_curvo_e_eixos(toricas, i):
    spec, ex = toricas["specs"][i], toricas["exams"][i]
    k = compute_keratometry(ex)
    km = 337.5 / spec.R
    for z in k.zonas:
        if not 2.0 <= z.diametro_mm <= 7.0:
            continue
        assert z.validity == "disponivel", z.block_reasons
        assert abs(z.k1_plano_D - (km - spec.cyl_D / 2)) < 0.03 and abs(z.k2_curvo_D - (km + spec.cyl_D / 2)) < 0.03
        # o eixo do gerador é o meridiano mais PLANO; K2 fica a 90° dele (OD e OS sem espelhamento)
        assert diferenca_eixo(z.eixo_k1_deg, spec.axis_deg) < 0.5
        assert diferenca_eixo(z.eixo_k2_deg, spec.axis_deg + 90.0) < 0.5
        assert abs(diferenca_eixo(z.eixo_k1_deg, z.eixo_k2_deg) - 90.0) < 1e-9
        assert abs(z.delta_k_D - spec.cyl_D) < 0.03 and abs(z.k_medio_D - km) < 0.03
        # vetor de potência (positivo = a favor da regra: K2 vertical)
        assert abs(math.hypot(z.j0_D, z.j45_D) - spec.cyl_D / 2) < 0.02
        assert not z.eixo_pouco_definido


def _axial_analitica(spec, d, n=256):
    """K1/K2/eixo da potência axial da superfície analítica na circunferência de diâmetro d (verdade independente)."""
    surf = build_surface(spec)
    th = np.radians(np.arange(n) * 360.0 / n)
    r = d / 2.0
    x, y = r * np.cos(th), r * np.sin(th)
    gx, gy = surf.grad(x, y)
    s = np.asarray(gx).ravel() * np.cos(th) + np.asarray(gy).ravel() * np.sin(th)
    p = 337.5 * s / (r * np.sqrt(1.0 + s**2))
    f = fourier_fit(th, p)
    return f["a0"] - f["h2"], f["a0"] + f["h2"], f["eixo_k2_deg"]


def test_torica_asferica_contra_potencia_axial_analitica(toricas):
    spec, ex = toricas["specs"][4], toricas["exams"][4]
    k = compute_keratometry(ex)
    for z in k.zonas:
        if not 2.0 <= z.diametro_mm <= 7.0:
            continue
        k1, k2, e2 = _axial_analitica(spec, z.diametro_mm)
        assert abs(z.k1_plano_D - k1) < 0.03 and abs(z.k2_curvo_D - k2) < 0.03, (z.rotulo, z.k1_plano_D, k1, z.k2_curvo_D, k2)
        assert diferenca_eixo(z.eixo_k2_deg, e2) < 0.5
    # a asfericidade prolata (Q < 0) aplana a periferia: K médio cai do centro para a borda
    km = [z.k_medio_D for z in k.zonas if z.validity == "disponivel"]
    assert all(a > b for a, b in zip(km, km[1:]))


def test_aneis_medidos_tem_diametro_do_ave_e_conferem_com_o_simk(toricas):
    ex = toricas["exams"][4]
    k = compute_keratometry(ex)
    assert len(k.aneis) == ex.samples.n_rings
    diam = [z.diametro_mm for z in k.aneis]
    assert all(b > a for a, b in zip(diam, diam[1:]))
    # 2·média do RAD medido por anel (a mesma definição da 2ª coluna do AVE do TMS)
    st = ex.samples.fields["DIO"].states
    r = ex.samples.r
    for kk in (0, 7, 15):
        med = st[kk] == PointState.MEASURED
        assert abs(k.aneis[kk].diametro_mm - 2 * np.mean(r[kk][med])) < 1e-9
    assert k.conferencia and "confere" in k.conferencia[0] and "NÃO" not in k.conferencia[0]


def test_zona_pupilar_media_por_area_igual_ao_acp(toricas):
    ex = toricas["exams"][0]
    k = compute_keratometry(ex)
    z = k.pupila
    assert z.validity == "disponivel" and abs(z.diametro_mm - 5.0) < 0.05
    assert z.centro_xy is not None and abs(z.centro_xy[0] - 0.3) < 0.02 and abs(z.centro_xy[1] + 0.2) < 0.02
    assert abs(z.k_medio_D - k.simk_tms["ACP"]) < 0.011            # ACP do XREF sintético (simulador: regra v1, amostra inteira)
    assert any("ACP" in c for c in k.conferencia)
    assert diferenca_eixo(z.eixo_k1_deg, 30.0) < 1.0 and abs(z.delta_k_D - 2.0) < 0.05


def test_pupila_ausente_fica_indisponivel_sem_cair_para_o_vertice(toricas):
    ex = toricas["exams"][5]
    assert ex.pupil.state != "registrada"
    k = compute_keratometry(ex)
    assert k.pupila.validity == "indisponivel" and "não registrada" in k.pupila.block_reasons[0]
    assert k.pupila.k1_plano_D is None
    linha = next(l for l in linhas_resumo(k) if l["tipo"] == "pupila")
    assert linha["rotulo"] == "pupila — não registrada" and linha["texto"] == ["—"] * 6
    # sem astigmatismo: o eixo é marcado como pouco definido (nota, não bloqueio)
    z3 = next(z for z in k.zonas if z.diametro_mm == 3.0)
    assert z3.validity == "disponivel" and z3.eixo_pouco_definido and z3.delta_k_D < 0.05


def test_palpebra_interpolada_marca_e_ausente_bloqueia(toricas):
    k_int = compute_keratometry(toricas["exams"][6])
    k_mis = compute_keratometry(toricas["exams"][7])
    d16 = toricas["exams"][6].samples.r[16].mean() * 2
    perif = [z for z in k_int.zonas if z.diametro_mm >= d16 + 0.3]
    assert perif and all(z.validity == "disponivel" and z.fracao_interpolados > 0.10 and z.marca_interpolados for z in perif)
    assert all(not z.marca_interpolados for z in k_int.zonas if z.diametro_mm <= 3.0)
    d12 = toricas["exams"][7].samples.r[12].mean() * 2
    for z in k_mis.zonas:
        if z.diametro_mm <= 3.0:
            assert z.validity == "disponivel"
        elif z.diametro_mm >= d12 + 0.3:
            assert z.validity == "indisponivel" and any("setor" in m for m in z.block_reasons), (z.rotulo, z.block_reasons)


def test_zonas_fora_da_faixa_dos_aneis(toricas):
    ex = toricas["exams"][2]
    k = compute_keratometry(ex, KeratometryConfig(diametros_mm=(0.3, 3.0, 12.0)))
    z0, z3, z12 = k.zonas
    assert z0.validity == "indisponivel" and "lacuna central" in z0.block_reasons[0]
    assert z3.validity == "disponivel"
    assert z12.validity == "indisponivel" and "fora da cobertura" in z12.block_reasons[0]


def test_tabela_sem_dio_fica_indisponivel_com_motivo(toricas):
    ex = toricas["exams"][2]
    campos = dict(ex.samples.fields)
    ex.samples.fields = {k: v for k, v in campos.items() if k != "DIO"}
    try:
        k = compute_keratometry(ex)
    finally:
        ex.samples.fields = campos
    assert all(z.validity == "indisponivel" and "DIO ausente" in z.block_reasons[0] for z in k.zonas)
    assert len(k.discos) == 2 and all(z.validity == "indisponivel" and "DIO ausente" in z.block_reasons[0] for z in k.discos)


def test_resultado_no_json_sem_nan_e_identidade_inclui_a_configuracao(toricas):
    ex = toricas["exams"][0]
    a1, _ = analyze_exam(ex, with_maps=False)
    a2, _ = analyze_exam(ex, with_maps=False)
    a3, _ = analyze_exam(ex, with_maps=False, kcfg=KeratometryConfig(diametros_mm=(3.0, 6.0)))
    assert a1.analysis_id == a2.analysis_id and a1.analysis_id != a3.analysis_id
    d = to_jsonable(a1)
    s = json.dumps(d, allow_nan=False)
    k = d["keratometry"]
    assert [z["rotulo"] for z in k["zonas"]] == [f"Ø {x:g} mm" for x in KeratometryConfig().diametros_mm]
    assert len(k["aneis"]) == 25 and k["pupila"]["tipo"] == "pupila"
    z3 = next(z for z in k["zonas"] if z["diametro_mm"] == 3.0)
    obj = next(z for z in a1.keratometry.zonas if z.diametro_mm == 3.0)
    assert z3["k1_plano_D"] == obj.k1_plano_D and z3["eixo_k2_deg"] == obj.eixo_k2_deg
    assert "SINTETICO" not in s and "PACIENTE" not in s
    assert set(k["simk_tms"]) <= {"SimK1", "SimK1 Ang.", "SimK2", "SimK2 Ang.", "MinK", "MinK Ang.", "CYL", "ACP",
                                  "3mm Reg. Astig.", "3mm Reg. Astig. Axis"}


def test_limites_novos_entram_no_relatorio_de_qualidade():
    qc = QualityConfig()
    assert qc.k_min_meridian_fraction == 0.85 and qc.k_axis_min_cyl_D == 0.50 and qc.provisional
    assert qc.k_eixo_min_cyl_D == 0.75 and qc.k_zona_alerta_dif_D == 0.50 and qc.k_zona_alerta_eixo_deg == 10.0


# ---------------------------------------------------------------------------
# 4. Zonas 0–d (média por área), linhas de exibição e alerta de dependência da zona
# ---------------------------------------------------------------------------
def test_zona_0_d_no_vertice_igual_a_zona_pupilar_centrada(toricas):
    # com a pupila centrada no vértice, a zona 0–4 mm e a zona pupilar de 4 mm são a mesma área
    ex = toricas["exams"][8]
    k = compute_keratometry(ex, KeratometryConfig(zonas_disco_mm=(4.0,)))
    z4, zp = k.discos[0], k.pupila
    assert z4.validity == zp.validity == "disponivel" and z4.tipo == "zona" and z4.rotulo == "zona 0–4 mm"
    assert abs(z4.k_medio_D - zp.k_medio_D) < 1e-6 and abs(z4.delta_k_D - zp.delta_k_D) < 1e-6
    assert diferenca_eixo(z4.eixo_k2_deg, zp.eixo_k2_deg) < 1e-6
    assert diferenca_eixo(z4.eixo_k1_deg, 100.0) < 1.0 and abs(z4.delta_k_D - 1.5) < 0.05


def test_zonas_0_d_padrao_e_fora_da_cobertura(toricas):
    ex = toricas["exams"][0]
    k = compute_keratometry(ex)
    assert [z.rotulo for z in k.discos] == ["zona 0–3 mm", "zona 0–6 mm"]
    z3 = k.discos[0]
    assert all(z.validity == "disponivel" for z in k.discos)
    assert diferenca_eixo(z3.eixo_k1_deg, 30.0) < 1.0 and abs(z3.delta_k_D - 2.0) < 0.05
    # média por área (peso r·Δr, regra do ACP): perto do K médio dos anéis de dentro; o peso r é maior nos meridianos
    # planos (anéis mais afastados do vértice), o que puxa a média ~0,01 D para baixo com ΔK de 2 D
    dentro = [a.k_medio_D for a in k.aneis if a.validity == "disponivel" and a.diametro_mm <= 3.0]
    assert abs(z3.k_medio_D - float(np.mean(dentro))) < 0.03
    fora = compute_keratometry(ex, KeratometryConfig(zonas_disco_mm=(12.0,))).discos[0]
    assert fora.validity == "indisponivel" and "além da cobertura" in fora.block_reasons[0]


def _sem_marca(rotulo):
    return rotulo.rstrip(" †*")


def test_linhas_resumo_folha_a4_e_tabela_completa(toricas):
    k = compute_keratometry(toricas["exams"][0])
    curto = linhas_resumo(k)
    assert [_sem_marca(l["rotulo"]) for l in curto[:9]] == [f"Ø {d:g} mm" for d in (1, 2, 3, 4, 5, 6, 7, 8)] + ["zona 0–3 mm"]
    assert curto[8]["destaque"] == "zona"
    assert [l["tipo"] for l in curto[9:]] == ["pupila", "simk_tms"]
    longo = linhas_resumo(k, incluir_aneis=True)
    rot = [_sem_marca(l["rotulo"]) for l in longo]
    assert rot[:10] == [f"Ø {d:g} mm" for d in KeratometryConfig().diametros_mm]
    assert rot[10:12] == ["zona 0–3 mm", "zona 0–6 mm"]
    assert [l["tipo"] for l in longo[12:14]] == ["pupila", "simk_tms"]
    assert sum(l["tipo"] == "anel" for l in longo) == 25 and len(longo) == 14 + 25
    l24 = next(l for l in longo if l["rotulo"].startswith("Ø 2.4 mm"))
    z24 = next(z for z in k.zonas if z.diametro_mm == 2.4)
    assert l24["k1"] == z24.k1_plano_D and l24["texto"][1] == fmt_eixo(z24.eixo_k1_deg)


def _zona_ast(rotulo, tipo, d, dk, eixo_k2, validity="disponivel"):
    c, s2 = math.cos(math.radians(2 * eixo_k2)), math.sin(math.radians(2 * eixo_k2))
    return KeratometryZone(rotulo=rotulo, tipo=tipo, diametro_mm=d, validity=validity, delta_k_D=dk,
                           eixo_k2_deg=eixo_k2, k_medio_D=44.0, j0_D=-(dk / 2) * c, j45_D=-(dk / 2) * s2)


@pytest.mark.parametrize("z3, zp, alerta", [
    ((1.75, 89.0), (1.15, 89.0), True),      # mesmo eixo, 0,60 D de diferença (como num olho real)
    ((1.00, 90.0), (1.00, 102.0), True),     # 0,42 D vetorial, mas 12° com ΔK ≥ 0,75 D
    ((0.49, 90.0), (0.49, 120.0), False),    # 0,49 D vetorial (abaixo de 0,50) e ΔK baixo: eixo não conta
    ((2.10, 20.0), (1.92, 23.0), False),     # zonas concordantes
])
def test_alerta_astigmatismo_dependente_da_zona(z3, zp, alerta):
    res = KeratometryResult(config=KeratometryConfig(), discos=[_zona_ast("zona 0–3 mm", "zona", 3.0, *z3)],
                            pupila=_zona_ast("pupila Ø 6.0 mm", "pupila", 6.0, *zp))
    txt = alerta_dependencia_zona(res, QualityConfig())
    assert (txt is not None) == alerta
    if alerta:
        assert txt.startswith("astigmatismo dependente da zona") and "pupila Ø 6.0 mm" in txt


def test_alerta_nao_aparece_sem_pupila():
    res = KeratometryResult(config=KeratometryConfig(), discos=[_zona_ast("zona 0–3 mm", "zona", 3.0, 1.75, 89.0)],
                            pupila=_zona_ast("pupila", "pupila", None, 0.0, 90.0, validity="indisponivel"))
    assert alerta_dependencia_zona(res, QualityConfig()) is None


# ---------------------------------------------------------------------------
# Exames reais (só local: o repositório é público e os exames são dados pessoais)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not os.environ.get("TMSAB_REAL_DATA"), reason="defina TMSAB_REAL_DATA com a pasta das exportações reais")
def test_exames_reais_reproduzem_simk_e_acp_do_tms():
    import glob
    raiz = os.environ["TMSAB_REAL_DATA"]
    pastas = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(raiz, "**", "XREF.dat"), recursive=True)})
    vistos, n = set(), 0
    for p in pastas:
        for e in list_exams(p):
            if e["exam_id"] in vistos:
                continue
            vistos.add(e["exam_id"])
            k = compute_keratometry(import_tms_bundle(p, exam_id=e["exam_id"]))
            assert k.conferencia and "NÃO" not in k.conferencia[0], e["exam_id"]
            c, t = k.simk_calculado, k.simk_tms
            assert abs(c["k2_curvo_D"] - t["SimK1"]) <= 0.02 and abs(c["k1_plano_D"] - t["SimK2"]) <= 0.05
            if k.pupila is not None and k.pupila.validity == "disponivel":
                assert abs(k.pupila.k_medio_D - t["ACP"]) <= 0.03
            assert all(z.validity == "disponivel" for z in k.discos), e["exam_id"]
            # diâmetro de cada anel = 2ª coluna do AVE gravado pelo próprio TMS (média do RAD dos pontos medidos)
            with open(os.path.join(p, f"AVE_1_{e['exam_id']}.dat"), encoding="latin-1") as fh:
                ave = [ln.split(",") for ln in fh.read().splitlines() if ln.strip()]
            for j, a in enumerate(k.aneis):
                if a.diametro_mm and float(ave[j][1]) > 0:
                    assert abs(a.diametro_mm - float(ave[j][1])) <= 1e-3, (e["exam_id"], j, a.diametro_mm, ave[j][1])
            # Fourier 3 mm do XREF = amplitude do 2º harmônico (metade do cilindro), eixo no meridiano curvo
            if "3mm Reg. Astig." in t:
                z3 = k.discos[0]
                assert abs(z3.delta_k_D - 2 * t["3mm Reg. Astig."]) <= 0.15, e["exam_id"]
                if z3.delta_k_D >= 0.75:
                    assert diferenca_eixo(z3.eixo_k2_deg, t["3mm Reg. Astig. Axis"]) <= 3.0, e["exam_id"]
            n += 1
    assert n > 0
