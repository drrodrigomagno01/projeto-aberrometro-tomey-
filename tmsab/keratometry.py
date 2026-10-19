"""Ceratometria por zona: K1 (plano), K2 (curvo) e eixos por diâmetro, a partir da potência axial do TMS.

Definições (docs/convencoes.md, "Ceratometria por zona"; decisão D23 em docs/decisoes.md):

- Fonte: DIO das Data Tables = potência AXIAL (sagital) do TMS na convenção ceratométrica
  337,5/R_axial (índice 1,3375), verificada nos exames reais (DIO·R_axial/337,5 = 1,000 ± 0,0005).
  Não é a curvatura local naquele diâmetro (essa é a tangencial, ROC) nem a potência refrativa da
  frente de onda (n = 1,376; o cilindro de Zernike é ≈ 1,114 × o ceratométrico).
- Ajuste em θ por mínimos quadrados: P(θ) = a0 + a1·cosθ + b1·senθ + a2·cos2θ + b2·sen2θ.
  K médio = a0; h2 = √(a2² + b2²); K2 (curvo) = a0 + h2; K1 (plano) = a0 − h2; ΔK = 2·h2;
  eixo do K2 = ½·atan2(b2, a2) (meridiano mais curvo); eixo do K1 = eixo do K2 + 90°.
  É a decomposição de Fourier do astigmatismo regular usada pela Tomey; K1 e K2 são ortogonais por
  construção. O 1º harmônico (assimetria/descentração) e o resíduo (irregularidade de ordem ≥ 3)
  saem do mesmo ajuste.
- Eixos em graus TABO (0° às 3 h, anti-horário na vista do examinador, nos dois olhos; é o referencial
  canônico verificado no G0), registrados em (0, 180]: 0° vira 180°.
- Circunferência de diâmetro d (centro no vértice): em cada meridiano, DIO interpolado linearmente em
  r = d/2 entre os dois anéis vizinhos, com o RAD daquele ângulo (os anéis se deslocam com a curvatura,
  então "3 mm" não é um anel fixo).
- Anel medido: os valores do próprio anel; diâmetro = 2·média do RAD medido (= 2ª coluna do AVE).
- Zona pupilar: amostras dentro do contorno pupilar do PUP (no mesmo meridiano), ponderadas pela área
  (r·Δr); a média corresponde ao ACP do TMS. Centro implícito no próprio contorno da pupila.
- SimK pela regra do TMS-4N (reproduzida nos exames reais com erro médio de 0,004 D): média dos anéis
  6–8 em cada meridiano completo (θ e θ + 180°); curvo = máximo, plano = meridiano a +90°, MinK = mínimo
  em qualquer ângulo. No XREF do TMS, SimK1 é o CURVO e SimK2 o plano — o contrário da biometria.
- J0/J45: vetor de potência de Thibos do astigmatismo ceratométrico, positivo = a favor da regra:
  J0 = −(ΔK/2)·cos(2·eixo K2) = −a2 e J45 = −(ΔK/2)·sen(2·eixo K2) = −b2. Estatística usa J0/J45 e
  K médio, nunca o eixo cru.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from .model import (
    KeratometryConfig,
    KeratometryResult,
    KeratometryZone,
    NormalizedExam,
    PointState,
    QualityConfig,
)

# Colunas numéricas do XREF copiadas para o resultado (lista de PERMITIDAS: nunca nome, ID ou caminho).
SIMK_TMS_COLUNAS = (
    "SimK1", "SimK1 Ang.", "SimK2", "SimK2 Ang.", "MinK", "MinK Ang.", "CYL", "ACP",
    "3mm Reg. Astig.", "3mm Reg. Astig. Axis",
)

_MIN_AMOSTRAS_AJUSTE = 9
_ASSIMETRIA_MIN_D = 5e-3       # abaixo disto o eixo da assimetria (1º harmônico) não é informado


# ---------------------------------------------------------------------------
# Utilidades de ângulo e formatação (usadas por todas as saídas)
# ---------------------------------------------------------------------------
_EIXO_TOL = 5e-3        # 0,005° (metade da menor casa impressa): 0° e 180° são o mesmo meridiano
PERFIL_VERIFICADO = "tms4_datatables_v1"     # único formato em que o DIO foi conferido como potência axial


def eixo_tabo(deg: float) -> float:
    """Meridiano em (0, 180]: 0° vira 180° (notação TABO de eixo), também a menos de 0,005° de 0°/180°."""
    a = float(deg) % 180.0
    return 180.0 if a < _EIXO_TOL or a > 180.0 - _EIXO_TOL else a


def fmt_eixo(deg: Optional[float]) -> str:
    """Eixo inteiro de 1 a 180 (arredondado), '—' se ausente."""
    if deg is None or not math.isfinite(deg):
        return "—"
    v = int(round(float(deg))) % 180
    return f"{180 if v == 0 else v}°"


def fmt_k(v: Optional[float], casas: int = 2) -> str:
    if v is None or not math.isfinite(v):
        return "—"
    return f"{v:.{casas}f}"


def _assinado(v: float) -> str:
    """Diferença com sinal e 2 casas, sem "-0.00"."""
    return f"{round(float(v), 2) + 0.0:+.2f}"


def diferenca_eixo(a: float, b: float) -> float:
    """Diferença entre dois meridianos, em graus (0–90)."""
    d = abs(float(a) - float(b)) % 180.0
    return min(d, 180.0 - d)


# ---------------------------------------------------------------------------
# Ajuste de Fourier em θ
# ---------------------------------------------------------------------------
def fourier_fit(theta_rad: np.ndarray, values: np.ndarray, weights: Optional[np.ndarray] = None) -> Optional[dict[str, float]]:
    """Mínimos quadrados de P(θ) = a0 + a1·cosθ + b1·senθ + a2·cos2θ + b2·sen2θ.

    Devolve None com menos de 9 amostras válidas. `irregularidade` = RMS (ponderado) do resíduo.
    """
    t = np.asarray(theta_rad, dtype=float).ravel()
    v = np.asarray(values, dtype=float).ravel()
    m = np.isfinite(t) & np.isfinite(v)
    w = None
    if weights is not None:
        w = np.asarray(weights, dtype=float).ravel()
        m &= np.isfinite(w) & (w > 0)
    if int(m.sum()) < _MIN_AMOSTRAS_AJUSTE:
        return None
    t, v = t[m], v[m]
    A = np.column_stack([np.ones_like(t), np.cos(t), np.sin(t), np.cos(2 * t), np.sin(2 * t)])
    if w is not None:
        ww = w[m]
        sw = np.sqrt(ww)
        coef, *_ = np.linalg.lstsq(A * sw[:, None], v * sw, rcond=None)
    else:
        ww = None
        coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    a0, a1, b1, a2, b2 = (float(c) for c in coef)
    res = v - A @ coef
    irr = float(np.sqrt(np.sum(ww * res**2) / np.sum(ww))) if ww is not None else float(np.sqrt(np.mean(res**2)))
    h1, h2 = math.hypot(a1, b1), math.hypot(a2, b2)
    return {
        "a0": a0, "a1": a1, "b1": b1, "a2": a2, "b2": b2, "h1": h1, "h2": h2,
        "eixo_k2_deg": eixo_tabo(0.5 * math.degrees(math.atan2(b2, a2))),
        # com amplitude abaixo de 0,005 D o ângulo é ruído numérico (difere entre plataformas): sem eixo
        "eixo_assimetria_deg": math.degrees(math.atan2(b1, a1)) % 360.0 if h1 >= _ASSIMETRIA_MIN_D else None,
        "irregularidade": irr, "n": int(m.sum()),
    }


def _zona_do_ajuste(z: KeratometryZone, fit: dict[str, float], qc: QualityConfig) -> KeratometryZone:
    a0, h2 = fit["a0"], fit["h2"]
    e2 = fit["eixo_k2_deg"]
    z.k_medio_D = a0
    z.k2_curvo_D = a0 + h2
    z.k1_plano_D = a0 - h2
    z.delta_k_D = 2.0 * h2
    z.eixo_k2_deg = e2
    z.eixo_k1_deg = eixo_tabo(e2 + 90.0)
    z.j0_D = -fit["a2"]
    z.j45_D = -fit["b2"]
    z.assimetria_D = fit["h1"]
    z.eixo_assimetria_deg = fit["eixo_assimetria_deg"]
    z.irregularidade_D = fit["irregularidade"]
    z.n_amostras = fit["n"]
    if round(z.delta_k_D, 2) < qc.k_axis_min_cyl_D:      # compara o valor como ele é exibido (2 casas)
        z.eixo_pouco_definido = True
        z.notes.append(
            f"eixo pouco definido: ΔK {z.delta_k_D:.2f} D < {qc.k_axis_min_cyl_D:.2f} D (o meridiano varia com a zona e entre capturas)"
        )
    return z


# ---------------------------------------------------------------------------
# Amostragem
# ---------------------------------------------------------------------------
def _maior_setor(falta: np.ndarray) -> float:
    """Maior setor angular contíguo (circular) marcado em `falta`, em graus."""
    n = falta.size
    if n == 0 or falta.all():
        return 360.0
    if not falta.any():
        return 0.0
    dobro = np.concatenate([falta, falta])
    melhor = atual = 0
    for v in dobro:
        atual = atual + 1 if v else 0
        melhor = max(melhor, atual)
    return min(360.0, melhor * 360.0 / n)


def _valores_na_circunferencia(dio: np.ndarray, st: np.ndarray, r: np.ndarray, R: float):
    """Por meridiano: DIO interpolado linearmente em r = R entre anéis consecutivos utilizáveis.

    Devolve (valores[na], usa_interpolado[na], disponivel[na]).
    """
    nr, na = r.shape
    usable = ((st == PointState.MEASURED) | (st == PointState.INTERPOLATED)) & np.isfinite(r) & np.isfinite(dio)
    vals = np.full(na, np.nan)
    interp = np.zeros(na, dtype=bool)
    ok = np.zeros(na, dtype=bool)
    for j in range(na):
        rj = r[:, j]
        uj = usable[:, j]
        if uj.sum() < 2:
            continue
        idx = np.nonzero(uj)[0]
        rr = rj[idx]
        if np.any(np.diff(rr) <= 0) or R < rr[0] or R > rr[-1]:
            continue
        k = int(np.searchsorted(rr, R))
        k = min(max(k, 1), idx.size - 1)
        i0, i1 = idx[k - 1], idx[k]
        if i1 - i0 != 1:                     # um anel ausente no meio: não interpolar através da lacuna
            continue
        t = (R - rj[i0]) / (rj[i1] - rj[i0])
        vals[j] = dio[i0, j] + t * (dio[i1, j] - dio[i0, j])
        ok[j] = True
        interp[j] = bool(st[i0, j] == PointState.INTERPOLATED or st[i1, j] == PointState.INTERPOLATED)
    return vals, interp, ok


def _disponibilidade_meridianos(z: KeratometryZone, ok: np.ndarray, interp: np.ndarray, qc: QualityConfig) -> None:
    z.fracao_cobertura = float(ok.mean()) if ok.size else 0.0
    z.fracao_interpolados = float(interp[ok].mean()) if ok.any() else 0.0
    z.maior_setor_sem_dados_deg = _maior_setor(~ok)
    if z.fracao_cobertura < qc.k_min_meridian_fraction:
        z.block_reasons.append(f"meridianos com dados {z.fracao_cobertura:.1%} < {qc.k_min_meridian_fraction:.0%}")
    if z.maior_setor_sem_dados_deg > qc.k_max_missing_sector_deg:
        z.block_reasons.append(
            f"maior setor sem dados {z.maior_setor_sem_dados_deg:.0f}° > {qc.k_max_missing_sector_deg:.0f}°"
        )
    if z.fracao_interpolados > 0:
        z.notes.append(f"{z.fracao_interpolados:.0%} dos meridianos usam pontos interpolados pelo aparelho (pálpebra)")
        z.marca_interpolados = z.fracao_interpolados >= qc.k_interp_mark_fraction


def _media_por_anel(r: np.ndarray, mediana: bool = False) -> np.ndarray:
    """Média (ou mediana) de r por anel só nos anéis com algum valor finito; NaN nos demais (sem RuntimeWarning)."""
    out = np.full(r.shape[0], np.nan)
    tem = np.isfinite(r).any(axis=1)
    if tem.any():
        out[tem] = (np.nanmedian if mediana else np.nanmean)(r[tem], axis=1)
    return out


def _circunferencia(d: float, dio, st, r, theta, qc: QualityConfig) -> KeratometryZone:
    z = KeratometryZone(rotulo=f"Ø {d:g} mm", tipo="circunferencia", diametro_mm=float(d), validity="indisponivel",
                        centro_xy=(0.0, 0.0))
    vals, interp, ok = _valores_na_circunferencia(dio, st, r, d / 2.0)
    if not ok.any():
        r_med = _media_por_anel(r, mediana=True)
        primeiro = float(r_med[np.isfinite(r_med)][0]) if np.isfinite(r_med).any() else float("nan")
        ultimo = float(r_med[np.isfinite(r_med)][-1]) if np.isfinite(r_med).any() else float("nan")
        if math.isfinite(primeiro) and d / 2.0 < primeiro:
            z.block_reasons.append(f"Ø {d:g} mm dentro da lacuna central do Plácido (1º anel em Ø ≈ {2 * primeiro:.2f} mm)")
        elif math.isfinite(ultimo) and d / 2.0 > ultimo:
            z.block_reasons.append(f"Ø {d:g} mm fora da cobertura dos anéis (último anel em Ø ≈ {2 * ultimo:.2f} mm)")
        else:
            z.block_reasons.append(f"nenhum meridiano com dados no Ø {d:g} mm")
        return z
    _disponibilidade_meridianos(z, ok, interp, qc)
    if z.block_reasons:
        return z
    fit = fourier_fit(theta, vals)
    if fit is None:
        z.validity = "invalida"
        z.block_reasons.append("ajuste de Fourier sem amostras suficientes")
        return z
    z.validity = "disponivel"
    return _zona_do_ajuste(z, fit, qc)


def _anel(k: int, dio, st, r, theta, qc: QualityConfig) -> KeratometryZone:
    meas = st[k] == PointState.MEASURED
    usable = ((st[k] == PointState.MEASURED) | (st[k] == PointState.INTERPOLATED)) & np.isfinite(dio[k]) & np.isfinite(r[k])
    diam = float(2.0 * np.mean(r[k][meas & np.isfinite(r[k])])) if (meas & np.isfinite(r[k])).any() else (
        float(2.0 * np.nanmean(r[k])) if np.isfinite(r[k]).any() else None)
    z = KeratometryZone(rotulo=f"anel {k + 1}", tipo="anel", diametro_mm=diam, validity="indisponivel", anel=k + 1,
                        centro_xy=(0.0, 0.0))
    interp = usable & (st[k] == PointState.INTERPOLATED)
    _disponibilidade_meridianos(z, usable, interp, qc)
    if z.block_reasons:
        return z
    fit = fourier_fit(theta, np.where(usable, dio[k], np.nan))
    if fit is None:
        z.validity = "invalida"
        z.block_reasons.append("ajuste de Fourier sem amostras suficientes")
        return z
    z.validity = "disponivel"
    return _zona_do_ajuste(z, fit, qc)


def _borda_pupilar(exam: NormalizedExam, theta: np.ndarray) -> Optional[np.ndarray]:
    """Raio da borda pupilar por meridiano, a partir do vértice (mm). PUP do TMS-4N se houver; senão círculo."""
    p = exam.pupil
    na = theta.size
    if p.edge_r is not None and np.asarray(p.edge_r).size == na:
        e = np.asarray(p.edge_r, dtype=float).copy()
        e[~(np.isfinite(e) & (e > 0))] = np.nan
        return e
    if p.center_x is None or p.center_y is None or not p.diameter:
        return None
    cx, cy, R = float(p.center_x), float(p.center_y), float(p.diameter) / 2.0
    b = cx * np.cos(theta) + cy * np.sin(theta)
    disc = R**2 - (cx**2 + cy**2) + b**2
    return np.where(disc >= 0, b + np.sqrt(np.maximum(disc, 0.0)), np.nan)


def _zona_por_area(z: KeratometryZone, borda: np.ndarray, dio, st, r, theta, qc: QualityConfig,
                   nome: str = "zona") -> KeratometryZone:
    """Preenche `z` com a ceratometria da área dentro de `borda` (raio por meridiano a partir do vértice).

    Regra do ACP do TMS (reproduzida nos exames reais): amostras com r ≤ borda no mesmo meridiano, peso de área
    r·Δr com Δr do raio nominal do anel; Km = média ponderada; K1/K2 = Km ∓ h2 do ajuste de Fourier ponderado.
    """
    nr, na = r.shape
    usable = ((st == PointState.MEASURED) | (st == PointState.INTERPOLATED)) & np.isfinite(r) & np.isfinite(dio)
    rn = _media_por_anel(r)
    idx = np.arange(nr)
    good = np.isfinite(rn)
    rn_full = np.interp(idx, idx[good], rn[good]) if good.sum() >= 2 else np.where(good, rn, 0.0)
    dr = np.abs(np.gradient(rn_full)) if nr > 1 else np.ones(nr)
    dentro = np.isfinite(r) & (r <= borda[None, :])
    sel = dentro & usable
    peso = np.where(np.isfinite(r), r, 0.0) * dr[:, None]
    # cobertura: meridianos em que os anéis utilizáveis alcançam a borda (meio anel de folga) e não há ponto
    # ausente dentro dela
    folga = 0.5 * float(np.nanmedian(dr)) if np.isfinite(dr).any() else 0.0
    alcance = np.where(usable, r, -np.inf).max(axis=0)
    # ponto sem dado dentro da zona, pela posição nominal do anel (no formato real o RAD do ponto ausente é NaN), com a
    # mesma folga de meio anel do alcance: no astigmatismo o RAD de um anel varia ~0,1–0,25 mm com o meridiano
    ausente_dentro = np.any((rn_full[:, None] <= borda[None, :] - folga) & ~usable, axis=0)
    valida = np.isfinite(borda)
    meridianos_ok = valida & (alcance >= borda - folga) & ~ausente_dentro
    z.fracao_cobertura = float(meridianos_ok.mean())          # meridiano sem borda conta como não coberto
    if (~valida).any():
        z.notes.append(f"{int((~valida).sum())} meridianos sem borda no contorno (fora da área)")
    soma = float(np.sum(peso[sel]))
    interp = sel & (st == PointState.INTERPOLATED)
    z.fracao_interpolados = float(np.sum(peso[interp]) / soma) if soma > 0 else 0.0
    z.maior_setor_sem_dados_deg = _maior_setor(~meridianos_ok)
    if z.fracao_cobertura < qc.k_min_meridian_fraction:
        z.block_reasons.append(f"meridianos da {nome} com dados até a borda {z.fracao_cobertura:.1%} < {qc.k_min_meridian_fraction:.0%}")
    if z.maior_setor_sem_dados_deg > qc.k_max_missing_sector_deg:
        z.block_reasons.append(f"maior setor da {nome} sem dados {z.maior_setor_sem_dados_deg:.0f}° > {qc.k_max_missing_sector_deg:.0f}°")
    if z.fracao_interpolados > 0:
        z.notes.append(f"{z.fracao_interpolados:.0%} da área usa pontos interpolados pelo aparelho (pálpebra)")
        z.marca_interpolados = z.fracao_interpolados >= qc.k_interp_mark_fraction
    if not z.block_reasons and soma <= 0:
        z.block_reasons.append(f"nenhuma amostra dentro da {nome}")
    if z.block_reasons:
        return z
    th = np.broadcast_to(theta[None, :], r.shape)
    fit = fourier_fit(th[sel], dio[sel], peso[sel])
    if fit is None:
        z.validity = "invalida"
        z.block_reasons.append("ajuste de Fourier sem amostras suficientes")
        return z
    # Km = média ponderada pela área (definição do ACP do TMS). Numa zona descentrada o termo constante do ajuste
    # difere dela em alguns centésimos; K1/K2 = Km ∓ h2 (h2 do ajuste).
    fit["a0"] = float(np.sum(peso[sel] * dio[sel]) / soma)
    z.validity = "disponivel"
    return _zona_do_ajuste(z, fit, qc)


def _zona_pupilar(exam: NormalizedExam, dio, st, r, theta, qc: QualityConfig) -> KeratometryZone:
    p = exam.pupil
    diam = float(p.diameter) if p.diameter else None
    rot = f"pupila Ø {diam:.1f} mm" if diam else "pupila"
    z = KeratometryZone(rotulo=rot, tipo="pupila", diametro_mm=diam, validity="indisponivel",
                        centro_xy=(float(p.center_x), float(p.center_y)) if p.center_x is not None else None)
    if p.state != "registrada":
        motivo = "pupila não registrada no exame (a zona pupilar não é substituída pelo vértice)"
        if p.method not in (None, "", "desconhecido"):
            motivo += f" — {p.method}"
        z.block_reasons.append(motivo)
        return z
    borda = _borda_pupilar(exam, theta)
    if borda is None or not np.isfinite(borda).any():
        z.block_reasons.append("contorno pupilar indisponível")
        return z
    _zona_por_area(z, borda, dio, st, r, theta, qc, nome="pupila")
    if z.validity == "disponivel":
        z.notes.append(
            "zona = área dentro do contorno pupilar do TMS (pupila de entrada, iluminação do Plácido nesta captura), "
            f"centro pupilar ({_assinado(p.center_x)}, {_assinado(p.center_y)}) mm do vértice; média ponderada pela área"
        )
    return z


def _disco(d: float, dio, st, r, theta, qc: QualityConfig) -> KeratometryZone:
    """Zona 0–d centrada no vértice (média ponderada pela área), como as 'zonas' do Pentacam/Atlas."""
    z = KeratometryZone(rotulo=f"zona 0–{d:g} mm", tipo="zona", diametro_mm=float(d), validity="indisponivel",
                        centro_xy=(0.0, 0.0))
    r_med = _media_por_anel(r, mediana=True)
    fin = r_med[np.isfinite(r_med)]
    if fin.size and d / 2.0 < float(fin[0]):
        z.block_reasons.append(f"zona 0–{d:g} mm dentro da lacuna central do Plácido (1º anel em Ø ≈ {2 * float(fin[0]):.2f} mm)")
        return z
    if fin.size and d / 2.0 > float(fin[-1]):
        z.block_reasons.append(f"zona 0–{d:g} mm além da cobertura dos anéis (último anel em Ø ≈ {2 * float(fin[-1]):.2f} mm)")
        return z
    borda = np.full(r.shape[1], d / 2.0)
    return _zona_por_area(z, borda, dio, st, r, theta, qc, nome="zona")


# ---------------------------------------------------------------------------
# SimK pela regra do TMS-4N e conferência com o XREF
# ---------------------------------------------------------------------------
def simk_regra_tms(dio: np.ndarray, st: np.ndarray, theta_deg: np.ndarray, aneis=(6, 7, 8)) -> Optional[dict[str, float]]:
    """Reproduz o SimK do TMS-4N: média dos anéis em cada meridiano completo (θ e θ+180°).

    Devolve chaves na convenção da biometria (k1 = plano, k2 = curvo) mais MinK. None se impossível.
    """
    nr, na = dio.shape
    if na % 4 != 0:
        return None
    ks = [k - 1 for k in aneis if 0 <= k - 1 < nr]
    if not ks:
        return None
    usable = ((st == PointState.MEASURED) | (st == PointState.INTERPOLATED)) & np.isfinite(dio)
    sub = np.where(usable[ks], dio[ks], np.nan)
    if np.isnan(sub).any():
        media = np.full(na, np.nan)
        completos = ~np.isnan(sub).any(axis=0)
        media[completos] = sub[:, completos].mean(axis=0)
    else:
        media = sub.mean(axis=0)
    meia = na // 2
    M = 0.5 * (media + np.roll(media, -meia))
    if not np.isfinite(M[:meia]).any():
        return None
    j = int(np.nanargmax(M[:meia]))
    j90 = (j + na // 4) % na
    if not np.isfinite(M[j90]):
        return None            # meridiano plano (90° do curvo) sem os anéis completos: a regra do TMS não se aplica
    jmin = int(np.nanargmin(M))
    out = {
        "k2_curvo_D": float(M[j]), "eixo_k2_deg": eixo_tabo(theta_deg[j]),
        "k1_plano_D": float(M[j90]), "eixo_k1_deg": eixo_tabo(theta_deg[j90]),
        "mink_D": float(M[jmin]), "eixo_mink_deg": eixo_tabo(theta_deg[jmin]),
    }
    out["delta_k_D"] = out["k2_curvo_D"] - out["k1_plano_D"]
    return out


def simk_do_xref(exam: NormalizedExam) -> Optional[dict[str, float]]:
    """Colunas permitidas do XREF para este exame (números do aparelho; nunca nome/ID)."""
    idx = (exam.bundle.export_settings or {}).get("tms_indices") or {}
    out = {k: float(idx[k]) for k in SIMK_TMS_COLUNAS if k in idx and idx[k] is not None and math.isfinite(float(idx[k]))}
    return out or None


def conferir_simk(calc: Optional[dict[str, float]], tms: Optional[dict[str, float]], qc: QualityConfig) -> list[str]:
    if not calc or not tms or not {"SimK1", "SimK2"} <= set(tms):
        return []
    d_curvo = calc["k2_curvo_D"] - tms["SimK1"]
    d_plano = calc["k1_plano_D"] - tms["SimK2"]
    partes = [f"ΔK curvo {_assinado(d_curvo)} D", f"ΔK plano {_assinado(d_plano)} D"]
    confere = abs(d_curvo) <= qc.k_simk_tol_D and abs(d_plano) <= qc.k_simk_tol_D
    cyl = tms.get("CYL", calc["delta_k_D"])
    if "SimK1 Ang." in tms:
        de = diferenca_eixo(calc["eixo_k2_deg"], tms["SimK1 Ang."])
        partes.append(f"Δeixo {de:.0f}°")
        if cyl >= qc.k_eixo_min_cyl_D and de > qc.k_simk_axis_tol_deg:
            confere = False
    estado = "confere" if confere else "NÃO confere — verificar leitura da exportação"
    return [f"SimK recalculado (regra do TMS, anéis 6–8) × XREF: {', '.join(partes)} — {estado}"]


def alerta_dependencia_zona(res: KeratometryResult, qc: QualityConfig) -> Optional[str]:
    """Compara o astigmatismo da zona 0–3 mm com o da zona pupilar (diferença vetorial, em D ceratométricas).

    Informativo, não é recomendação: na literatura de LIO tórica, a zona de 3 mm no vértice previu melhor o
    astigmatismo (Meng 2021); com pupila grande, uma diferença acima de `k_zona_alerta_dif_D` (vetorial) ou de
    `k_zona_alerta_eixo_deg` (eixo, só com ΔK ≥ `k_eixo_min_cyl_D` nas duas zonas) entre as duas zonas merece atenção.
    """
    z3 = next((z for z in res.discos if z.diametro_mm and abs(z.diametro_mm - 3.0) < 1e-6), None)
    zp = res.pupila
    if not (z3 and zp and z3.validity == "disponivel" and zp.validity == "disponivel"):
        return None
    dif = 2.0 * math.hypot(zp.j0_D - z3.j0_D, zp.j45_D - z3.j45_D)
    de = diferenca_eixo(zp.eixo_k2_deg, z3.eixo_k2_deg)
    eixo_conta = min(zp.delta_k_D, z3.delta_k_D) >= qc.k_eixo_min_cyl_D
    if dif > qc.k_zona_alerta_dif_D or (eixo_conta and de > qc.k_zona_alerta_eixo_deg):
        return (f"astigmatismo dependente da zona: zona 0–3 mm ΔK {z3.delta_k_D:.2f} D @ {fmt_eixo(z3.eixo_k2_deg)} × "
                f"{zp.rotulo} ΔK {zp.delta_k_D:.2f} D @ {fmt_eixo(zp.eixo_k2_deg)} "
                f"(diferença vetorial {dif:.2f} D, eixo {de:.0f}°)")
    return None


# ---------------------------------------------------------------------------
# Entrada principal
# ---------------------------------------------------------------------------
def compute_keratometry(exam: NormalizedExam, cfg: Optional[KeratometryConfig] = None,
                        qc: Optional[QualityConfig] = None) -> KeratometryResult:
    """Calcula a ceratometria por zona de um exame normalizado (nunca lança por falta de dados)."""
    cfg = cfg or KeratometryConfig()
    qc = qc or QualityConfig()
    res = KeratometryResult(config=cfg)
    s = exam.samples
    tab = s.fields.get(cfg.fonte) if s.fields else None
    motivo = None
    if tab is None:
        motivo = f"tabela {cfg.fonte} ausente na exportação"
    elif str(tab.unit).strip().upper() != "D":
        motivo = f"tabela {cfg.fonte} em '{tab.unit}' (esperado D, potência axial)"
    elif s.r is None:
        motivo = "posições radiais (RAD) indisponíveis"
    if motivo:
        res.notes.append(f"ceratometria indisponível: {motivo}")
        res.zonas = [KeratometryZone(rotulo=f"Ø {d:g} mm", tipo="circunferencia", diametro_mm=float(d),
                                     validity="indisponivel", block_reasons=[motivo]) for d in cfg.diametros_mm]
        res.discos = [KeratometryZone(rotulo=f"zona 0–{d:g} mm", tipo="zona", diametro_mm=float(d),
                                      validity="indisponivel", block_reasons=[motivo]) for d in cfg.zonas_disco_mm]
        if cfg.zona_pupilar != "nenhuma":
            res.pupila = KeratometryZone(rotulo="pupila", tipo="pupila", diametro_mm=exam.pupil.diameter,
                                         validity="indisponivel", block_reasons=[motivo])
        res.simk_tms = simk_do_xref(exam)
        if res.simk_tms and {"SimK1", "SimK2"} <= set(res.simk_tms):
            res.conferencia = [f"SimK × XREF não conferido: SimK recalculado indisponível ({motivo})"]
        return res

    dio = np.asarray(tab.values, dtype=float)
    st = np.asarray(tab.states)
    r = np.asarray(s.r, dtype=float)
    theta_deg = np.asarray(s.theta_deg, dtype=float)
    theta = np.radians(theta_deg)

    res.zonas = [_circunferencia(float(d), dio, st, r, theta, qc) for d in cfg.diametros_mm]
    res.discos = [_disco(float(d), dio, st, r, theta, qc) for d in cfg.zonas_disco_mm]
    if cfg.aneis_medidos:
        res.aneis = [_anel(k, dio, st, r, theta, qc) for k in range(dio.shape[0])]
    if cfg.zona_pupilar != "nenhuma":
        res.pupila = _zona_pupilar(exam, dio, st, r, theta, qc)

    res.simk_calculado = simk_regra_tms(dio, st, theta_deg, cfg.simk_aneis)
    res.simk_tms = simk_do_xref(exam)
    res.conferencia = conferir_simk(res.simk_calculado, res.simk_tms, qc)
    if res.simk_calculado is None:
        a = f"{cfg.simk_aneis[0]}–{cfg.simk_aneis[-1]}" if cfg.simk_aneis else ""
        res.notes.append(f"SimK recalculado indisponível (regra do TMS, anéis {a}): o meridiano curvo ou o plano "
                         f"não tem os anéis completos")
        if res.simk_tms and {"SimK1", "SimK2"} <= set(res.simk_tms):
            res.conferencia.append(f"SimK × XREF não conferido: SimK recalculado indisponível (anéis {a} incompletos)")
    if res.pupila is not None and res.pupila.validity == "disponivel" and res.simk_tms and "ACP" in res.simk_tms:
        dk = res.pupila.k_medio_D - res.simk_tms["ACP"]
        res.conferencia.append(f"K médio na zona pupilar × ACP do TMS: Δ {_assinado(dk)} D")
    alerta = alerta_dependencia_zona(res, qc)
    if alerta:
        res.alertas.append(alerta)
    res.notes.append(
        f"fonte {cfg.fonte} (potência axial do TMS, índice {cfg.indice_ceratometrico}); Fourier por meridiano "
        f"(harmônicos 0–2); centro no vértice; K1 = plano, K2 = curvo; eixos TABO 1–180"
    )
    perfil = getattr(getattr(exam, "bundle", None), "adapter_profile", None)
    if perfil != PERFIL_VERIFICADO or not exam.frame.transform_verified:
        res.notes.append(f"perfil de exportação {perfil or 'desconhecido'} (presumido): a semântica do DIO como potência "
                         f"axial só foi verificada no formato real ({PERFIL_VERIFICADO})")
    return res


# ---------------------------------------------------------------------------
# Linhas prontas para exibição (página A4, programa, HTML, relatório em texto)
# ---------------------------------------------------------------------------
def linhas_resumo(kres: Optional[KeratometryResult], incluir_aneis: bool = False) -> list[dict[str, Any]]:
    """Linhas da tabela de ceratometria na ordem de exibição.

    Padrão (folha A4): as circunferências de `diametros_resumo_mm`, as zonas de `zonas_disco_resumo_mm`, a zona
    pupilar e o SimK do TMS. Com `incluir_aneis=True` (tabelas completas do programa, HTML e relatório): todas as
    circunferências e zonas, a pupila, o SimK e os 25 anéis medidos. Cada linha traz os números e o texto já
    formatado (`destaque`: "zona", "pupila", "referencia" ou "")."""
    if kres is None:
        return []
    out: list[dict[str, Any]] = []
    cfg = kres.config

    def _no_resumo(z: KeratometryZone, diametros) -> bool:
        return z.diametro_mm is not None and any(abs(z.diametro_mm - float(d)) < 1e-6 for d in diametros)

    def _linha(z: KeratometryZone, destaque: str = "") -> dict[str, Any]:
        disp = z.validity == "disponivel"
        marca = ("†" if z.marca_interpolados else "") + ("*" if z.eixo_pouco_definido else "")
        rot = z.rotulo
        if z.tipo == "anel" and z.diametro_mm:
            rot = f"anel {z.anel} · Ø {z.diametro_mm:.2f} mm"
        return {
            "rotulo": rot + (f" {marca}" if marca and disp else ""), "tipo": z.tipo, "destaque": destaque,
            "disponivel": disp, "validity": z.validity, "motivo": "; ".join(z.block_reasons),
            "k1": z.k1_plano_D, "eixo_k1": z.eixo_k1_deg, "k2": z.k2_curvo_D, "eixo_k2": z.eixo_k2_deg,
            "dk": z.delta_k_D, "km": z.k_medio_D, "diametro_mm": z.diametro_mm,
            "texto": [fmt_k(z.k1_plano_D), fmt_eixo(z.eixo_k1_deg), fmt_k(z.k2_curvo_D), fmt_eixo(z.eixo_k2_deg),
                      fmt_k(z.delta_k_D), fmt_k(z.k_medio_D)] if disp else ["—"] * 6,
        }

    circ = kres.zonas if incluir_aneis else [z for z in kres.zonas if _no_resumo(z, cfg.diametros_resumo_mm)]
    discos = kres.discos if incluir_aneis else [z for z in kres.discos if _no_resumo(z, cfg.zonas_disco_resumo_mm)]
    for z in circ:
        out.append(_linha(z))
    for z in discos:
        out.append(_linha(z, destaque="zona"))
    if kres.pupila is not None:
        lp = _linha(kres.pupila, destaque="pupila")
        if not lp["disponivel"]:
            lp["rotulo"] = "pupila — não registrada" if "não registrada" in lp["motivo"] else "pupila — indisponível"
        out.append(lp)
    t = kres.simk_tms
    if t and {"SimK1", "SimK2"} <= set(t):
        out.append({
            "rotulo": "SimK do TMS (anéis 6–8, Ø≈2.5 mm)", "tipo": "simk_tms", "destaque": "referencia",
            "disponivel": True, "validity": "disponivel", "motivo": "",
            "k1": t["SimK2"], "eixo_k1": t.get("SimK2 Ang."), "k2": t["SimK1"], "eixo_k2": t.get("SimK1 Ang."),
            "dk": t.get("CYL", t["SimK1"] - t["SimK2"]), "km": 0.5 * (t["SimK1"] + t["SimK2"]), "diametro_mm": None,
            "texto": [fmt_k(t["SimK2"]), fmt_eixo(t.get("SimK2 Ang.")), fmt_k(t["SimK1"]), fmt_eixo(t.get("SimK1 Ang.")),
                      fmt_k(t.get("CYL", t["SimK1"] - t["SimK2"])), fmt_k(0.5 * (t["SimK1"] + t["SimK2"]))],
        })
    if incluir_aneis:
        for z in kres.aneis:
            out.append(_linha(z))
    return out


COLUNAS_TABELA = ("Zona", "K1 plano (D)", "eixo K1", "K2 curvo (D)", "eixo K2", "ΔK (D)", "Km (D)")
LEGENDA_TABELA = (
    "Potência axial do TMS (DIO, n 1,3375), 2º harmônico de Fourier por meridiano; circunferências, zonas 0–d e anéis "
    "centrados no vértice, zona pupilar dentro do contorno da pupila. "
    "K1 = meridiano mais plano, K2 = mais curvo (no XREF do TMS, SimK1 é o curvo). Eixos TABO. "
    "† pontos interpolados pelo aparelho; * eixo pouco definido (ΔK baixo)."
)
