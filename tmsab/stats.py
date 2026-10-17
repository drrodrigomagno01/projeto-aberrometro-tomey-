"""Estatística do piloto (plano §9): repetibilidade e concordância com observações repetidas.

Entrada: tabela longa (CSV) com colunas
    participante, olho, metodo, captura, metrica, valor
(uma linha por captura por métrica). "metodo" distingue, por exemplo, "TMS_calc" e "OPD_corneal".

- Repetibilidade (por método e métrica): desvio-padrão intrassujeito Sw (ANOVA de um fator),
  coeficiente de repetibilidade 2,77·Sw (ISO 5725 / Bland & Altman 1996) e IC aproximado.
- Concordância entre dois métodos com múltiplas observações por indivíduo
  (Bland & Altman 2007, caso de valor verdadeiro constante): diferença por captura pareada
  pela ordem da captura, componentes de variância entre e dentro de indivíduos, viés com IC
  e limites de concordância com IC aproximado. Heterocedasticidade por regressão da
  diferença absoluta na média.

A unidade de análise é o participante (um olho por participante por padrão). Se ambos os
olhos forem incluídos, o agrupamento por participante deve ser tratado (não implementado
aqui: filtrar um olho antes).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats as sps


@dataclass
class LongTable:
    rows: list[dict[str, str]] = field(default_factory=list)

    @classmethod
    def read_csv(cls, path: str) -> "LongTable":
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            sample = fh.read(4096); fh.seek(0)
            dialect = csv.Sniffer().sniff(sample, delimiters=";,\t")
            rd = csv.DictReader(fh, dialect=dialect)
            rows = [{k.strip().lower(): (v or "").strip() for k, v in r.items()} for r in rd]
        need = {"participante", "metodo", "captura", "metrica", "valor"}
        if rows and not need.issubset(rows[0].keys()):
            raise ValueError(f"colunas obrigatórias: {sorted(need)}; encontradas: {sorted(rows[0].keys())}")
        return cls(rows)

    def select(self, metodo: str, metrica: str, olho: Optional[str] = None) -> dict[str, dict[int, float]]:
        """-> {participante: {captura: valor}}"""
        out: dict[str, dict[int, float]] = {}
        for r in self.rows:
            if r["metodo"] != metodo or r["metrica"] != metrica:
                continue
            if olho and r.get("olho", "") != olho:
                continue
            try:
                v = float(r["valor"].replace(",", "."))
            except ValueError:
                continue
            out.setdefault(r["participante"], {})[int(float(r["captura"]))] = v
        return out

    def methods(self) -> list[str]:
        return sorted({r["metodo"] for r in self.rows})

    def metrics(self) -> list[str]:
        return sorted({r["metrica"] for r in self.rows})


@dataclass
class RepeatabilityResult:
    metodo: str
    metrica: str
    n_participantes: int
    n_total: int
    media: float
    sw: float                  # desvio-padrão intrassujeito
    sw_ic95: tuple[float, float]
    coef_repetibilidade: float # 2.77 * Sw
    sb: float                  # desvio-padrão entre indivíduos (componente)
    icc: float                 # ICC(1,1) aproximado


def repeatability(data: dict[str, dict[int, float]], metodo: str, metrica: str) -> RepeatabilityResult:
    groups = [np.array(list(d.values()), float) for d in data.values() if len(d) >= 2]
    if len(groups) < 2:
        raise ValueError("repetibilidade exige ≥2 participantes com ≥2 capturas")
    n = len(groups)
    m = np.array([len(g) for g in groups], float)
    N = m.sum()
    grand = np.concatenate(groups).mean()
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    dfw, dfb = N - n, n - 1
    msw, msb = ssw / dfw, ssb / dfb
    sw = float(np.sqrt(msw))
    m0 = (N - (m**2).sum() / N) / (n - 1)
    sb2 = max(0.0, (msb - msw) / m0)
    icc = sb2 / (sb2 + msw) if (sb2 + msw) > 0 else float("nan")
    lo = sw * np.sqrt(dfw / sps.chi2.ppf(0.975, dfw))
    hi = sw * np.sqrt(dfw / sps.chi2.ppf(0.025, dfw))
    return RepeatabilityResult(metodo, metrica, n, int(N), float(grand), sw, (float(lo), float(hi)),
                               2.77 * sw, float(np.sqrt(sb2)), float(icc))


@dataclass
class AgreementResult:
    metodo_a: str
    metodo_b: str
    metrica: str
    n_participantes: int
    n_pares: int
    vies: float
    vies_ic95: tuple[float, float]
    sd_dif_total: float        # sqrt(σ_b² + σ_w²) da diferença
    loa: tuple[float, float]
    loa_ic95: tuple[tuple[float, float], tuple[float, float]]
    sigma_w: float
    sigma_b: float
    heterocedasticidade_p: float
    correlacao_pearson: float  # complemento; alta correlação não demonstra concordância
    medias: np.ndarray
    diferencas: np.ndarray
    participantes: list[str]


def agreement_repeated(da: dict[str, dict[int, float]], db: dict[str, dict[int, float]], metodo_a: str, metodo_b: str,
                       metrica: str) -> AgreementResult:
    """Bland–Altman com múltiplas observações por indivíduo (valor verdadeiro constante), pareando capturas pela ordem."""
    diffs: list[np.ndarray] = []
    means: list[np.ndarray] = []
    parts: list[str] = []
    for p in sorted(set(da) & set(db)):
        ka = sorted(da[p]); kb = sorted(db[p])
        k = min(len(ka), len(kb))
        if k == 0:
            continue
        a = np.array([da[p][key] for key in ka[:k]]); b = np.array([db[p][key] for key in kb[:k]])
        diffs.append(a - b); means.append((a + b) / 2); parts.append(p)
    if len(diffs) < 2:
        raise ValueError("concordância exige ≥2 participantes com capturas em ambos os métodos")
    n = len(diffs)
    m = np.array([len(d) for d in diffs], float); N = m.sum()
    alld = np.concatenate(diffs); allm = np.concatenate(means)
    bias = float(alld.mean())
    # ANOVA de um fator das diferenças por participante
    ssw = sum(((d - d.mean()) ** 2).sum() for d in diffs)
    ssb = sum(len(d) * (d.mean() - bias) ** 2 for d in diffs)
    dfw, dfb = N - n, n - 1
    msw = ssw / dfw if dfw > 0 else 0.0
    msb = ssb / dfb
    m0 = (N - (m**2).sum() / N) / (n - 1)
    sw2 = msw
    sb2 = max(0.0, (msb - msw) / m0)
    sd_total = float(np.sqrt(sb2 + sw2))
    loa = (bias - 1.96 * sd_total, bias + 1.96 * sd_total)
    # IC do viés: erro padrão com base na variância entre médias dos participantes (conservador)
    pm = np.array([d.mean() for d in diffs])
    se_bias = pm.std(ddof=1) / np.sqrt(n)
    t = sps.t.ppf(0.975, n - 1)
    bias_ci = (bias - t * se_bias, bias + t * se_bias)
    # IC aproximado dos LoA (Bland & Altman 1986: var(LoA) ≈ (1/n + 1.96²/(2(n−1))) sd²)
    se_loa = sd_total * np.sqrt(1.0 / n + 1.96**2 / (2 * (n - 1)))
    loa_ci = ((loa[0] - t * se_loa, loa[0] + t * se_loa), (loa[1] - t * se_loa, loa[1] + t * se_loa))
    # heterocedasticidade: regressão |dif − viés| ~ média
    slope, intercept, r, p, se = sps.linregress(allm, np.abs(alld - bias))
    corr = float(np.corrcoef(np.concatenate([d + mm for d, mm in zip(diffs, means)]), np.concatenate([mm - d / 2 for d, mm in zip(diffs, means)]))[0, 1])
    return AgreementResult(metodo_a, metodo_b, metrica, n, int(N), bias, (float(bias_ci[0]), float(bias_ci[1])), sd_total,
                           (float(loa[0]), float(loa[1])), tuple(tuple(float(x) for x in ci) for ci in loa_ci),
                           float(np.sqrt(sw2)), float(np.sqrt(sb2)), float(p), corr, allm, alld, parts)


def bland_altman_figure(res: AgreementResult):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(7, 5))
    ax = fig.subplots()
    ax.scatter(res.medias, res.diferencas, s=14, color="#2E6F9E", alpha=0.8)
    ax.axhline(res.vies, color="black", lw=1, label=f"viés {res.vies:+.4f}")
    ax.axhline(res.loa[0], color="#C0504D", lw=1, ls="--", label=f"LoA {res.loa[0]:+.4f} / {res.loa[1]:+.4f}")
    ax.axhline(res.loa[1], color="#C0504D", lw=1, ls="--")
    ax.set_xlabel(f"média dos métodos — {res.metrica}")
    ax.set_ylabel(f"{res.metodo_a} − {res.metodo_b}")
    ax.set_title(f"Bland–Altman (observações repetidas) — {res.metrica}; n={res.n_participantes} participantes, {res.n_pares} pares")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def write_results_csv(path: str, reps: list[RepeatabilityResult], agrs: list[AgreementResult]) -> str:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["tipo", "metodo", "metodo_b", "metrica", "n_participantes", "n_obs", "media_ou_vies", "ic95_inf", "ic95_sup",
                    "sw", "coef_repetibilidade_2.77sw", "sb", "icc", "sd_dif_total", "loa_inf", "loa_sup", "loa_inf_ic", "loa_sup_ic", "p_heteroc", "r_pearson"])
        for r in reps:
            w.writerow(["repetibilidade", r.metodo, "", r.metrica, r.n_participantes, r.n_total, f"{r.media:.6f}", f"{r.sw_ic95[0]:.6f}", f"{r.sw_ic95[1]:.6f}",
                        f"{r.sw:.6f}", f"{r.coef_repetibilidade:.6f}", f"{r.sb:.6f}", f"{r.icc:.4f}", "", "", "", "", "", "", ""])
        for a in agrs:
            w.writerow(["concordancia", a.metodo_a, a.metodo_b, a.metrica, a.n_participantes, a.n_pares, f"{a.vies:.6f}", f"{a.vies_ic95[0]:.6f}", f"{a.vies_ic95[1]:.6f}",
                        f"{a.sigma_w:.6f}", "", f"{a.sigma_b:.6f}", "", f"{a.sd_dif_total:.6f}", f"{a.loa[0]:.6f}", f"{a.loa[1]:.6f}",
                        f"{a.loa_ic95[0][0]:.6f}/{a.loa_ic95[0][1]:.6f}", f"{a.loa_ic95[1][0]:.6f}/{a.loa_ic95[1][1]:.6f}", f"{a.heterocedasticidade_p:.4f}", f"{a.correlacao_pearson:.4f}"])
    return path
