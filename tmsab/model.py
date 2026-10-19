"""Contrato de dados interno (plano §4).

O esquema é uma proposta da aplicação, não o formato literal do fabricante.
Objetivo: desacoplar o núcleo óptico dos nomes e detalhes de cada exportação.

Convenções:
- Unidades geométricas em milímetros; frente de onda em micrômetros (conversão em um único ponto, ver optics.py).
- Ausências internas usam máscara própria (PointState); nunca zero.
- Sistema canônico (vista do observador): x para a direita da imagem, y para cima,
  ângulo anti-horário a partir das 3 horas, z positivo para dentro do olho (posterior).
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Estados de ponto / campo
# ---------------------------------------------------------------------------
class PointState(IntEnum):
    """Estados mínimos de um dado (plano §4)."""

    MEASURED = 0        # medido pelo dispositivo
    INTERPOLATED = 1    # interpolado pelo dispositivo
    MISSING = 2         # ausente
    INVALID = 3         # inválido na leitura
    EXCLUDED = 4        # excluído por qualidade
    GENERATED = 5       # gerado pelo processamento


STATE_LABELS_PT = {
    PointState.MEASURED: "medido",
    PointState.INTERPOLATED: "interpolado",
    PointState.MISSING: "ausente",
    PointState.INVALID: "invalido_leitura",
    PointState.EXCLUDED: "excluido_qualidade",
    PointState.GENERATED: "gerado_processamento",
}


# ---------------------------------------------------------------------------
# Pacote de origem
# ---------------------------------------------------------------------------
@dataclass
class SourceFile:
    path: str
    role: str                      # HIT, RAD, ROC, DIO, PUP, RNG, MANIFEST, OUTRO
    sha256: str
    size_bytes: int
    n_rows: Optional[int] = None
    n_cols: Optional[int] = None
    delimiter: Optional[str] = None
    decimal: Optional[str] = None
    orientation: Optional[str] = None   # "angle_rows" | "ring_rows"


@dataclass
class SourceBundle:
    directory: str
    files: list[SourceFile]
    adapter_profile: str
    adapter_version: str
    export_settings: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    synthetic: bool = False

    def hashes(self) -> dict[str, str]:
        import os
        return {f.role + ":" + os.path.basename(f.path): f.sha256 for f in self.files}


# ---------------------------------------------------------------------------
# Exame / equipamento / geometria
# ---------------------------------------------------------------------------
@dataclass
class ExamIdentity:
    study_id: str                      # código de estudo (pseudonimizado)
    exam_id: str                       # identificador técnico do exame
    eye: Optional[str]                 # "OD" | "OS" | None
    acquired_at: Optional[str] = None  # ISO-8601 ou None
    date_precision: str = "desconhecida"   # "segundo" | "dia" | "desconhecida"
    capture_index: Optional[int] = None    # repetição dentro da sessão


@dataclass
class DeviceInfo:
    model: str = "desconhecido"          # "TMS-4" | "TMS-4N" | "TMS-5" | "sintetico"
    software_version: str = "desconhecida"
    cone: Optional[str] = None
    calibration_state: str = "nao_informado"


@dataclass
class GeometryFrame:
    surface_type: str = "anterior"
    origin: str = "vertice_topografico"
    axes: str = "x_direita_y_cima_vista_observador"
    unit: str = "mm"
    height_reference: str = "plano_apical_z_positivo_posterior"
    angle_start_deg: float = 0.0        # primeira posição angular (3 horas = 0°)
    angle_direction: int = 1            # +1 anti-horário (vista do observador), -1 horário
    mirror_x: bool = False
    transform_verified: bool = False    # só True após confirmação documental e experimental
    notes: str = ""


# ---------------------------------------------------------------------------
# Amostras
# ---------------------------------------------------------------------------
@dataclass
class FieldTable:
    name: str
    unit: str
    values: np.ndarray        # float64 [n_rings, n_angles], NaN onde não há valor
    states: np.ndarray        # int8   [n_rings, n_angles] (PointState)
    raw: Optional[np.ndarray] = None   # tokens originais (object), opcional

    def measured_mask(self) -> np.ndarray:
        return self.states == PointState.MEASURED

    def usable_mask(self) -> np.ndarray:
        return (self.states == PointState.MEASURED) | (self.states == PointState.INTERPOLATED)


@dataclass
class Samples:
    n_rings: int
    n_angles: int
    theta_deg: np.ndarray                  # [n_angles] no sistema canônico
    fields: dict[str, FieldTable]
    # coordenadas físicas canônicas derivadas (NaN onde indisponível)
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    z: Optional[np.ndarray] = None
    r: Optional[np.ndarray] = None
    point_state: Optional[np.ndarray] = None   # estado combinado da geometria

    def geometry_points(self, usable: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Retorna (x, y, z, state) achatados dos pontos com geometria disponível."""
        assert self.x is not None and self.y is not None and self.z is not None
        st = self.point_state
        if usable:
            m = (st == PointState.MEASURED) | (st == PointState.INTERPOLATED)
        else:
            m = st == PointState.MEASURED
        return self.x[m], self.y[m], self.z[m], st[m]


@dataclass
class PupilData:
    state: str = "ausente"                 # "registrada" | "ausente" | "nao_verificada"
    center_x: Optional[float] = None       # mm, sistema canônico, relativo ao vértice
    center_y: Optional[float] = None
    diameter: Optional[float] = None       # mm
    frame: str = "desconhecido"
    method: str = "desconhecido"
    edge_r: Optional[np.ndarray] = None    # raios da borda por ângulo, se existirem


@dataclass
class NormalizedExam:
    bundle: SourceBundle
    identity: ExamIdentity
    device: DeviceInfo
    frame: GeometryFrame
    samples: Samples
    pupil: PupilData
    diagnostics: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------
@dataclass
class ReconstructionConfig:
    method: str = "conica_zernike"      # "conica_zernike" | "rbf_tps"
    surface_order: int = 10             # ordem radial do resíduo de Zernike da superfície
    fit_radius_mm: Optional[float] = None   # None -> automático (raio máximo medido)
    weights: str = "area"               # "area" | "uniforme"
    holdout_rings: tuple[int, ...] = ()     # anéis reservados para resíduo de verificação (0-based)
    rbf_smoothing: float = 1e-4
    use_interpolated: bool = True       # usar pontos interpolados pelo dispositivo na reconstrução


@dataclass
class OpticalProfile:
    name: str = "anterior_v1"
    n_in: float = 1.0                   # ar
    n_out: float = 1.376                # córnea (assunção do modelo)
    object_conjugate: str = "axial_infinito"
    aperture_plane: str = "corneano"
    center_mode: str = "vertice"        # "vertice" | "pupila"
    diameter_mm: float = 6.0
    reference: str = "paraxial_media_principais"   # | "refoco_rms"
    zernike_order: int = 6
    normalization: str = "OSA_ortonormal"
    dispersion: str = "nenhuma"
    ray_grid: int = 97                  # pontos por lado da grade cartesiana na abertura
    hoa_removed_terms: tuple[str, ...] = ("piston", "tilt", "defocus", "astigmatismo")

    @property
    def radius_mm(self) -> float:
        return self.diameter_mm / 2.0


@dataclass
class QualityConfig:
    """Limites provisórios de disponibilidade (plano §8: derivados do orçamento de erro, não clínicos)."""

    min_measured_fraction: float = 0.85      # fração de pontos medidos dentro da abertura
    min_usable_fraction: float = 0.95        # medidos + interpolados
    max_missing_sector_deg: float = 20.0     # maior setor angular sem dados medidos na borda
    edge_margin: float = 0.98                # cobertura radial mínima (fração do raio) por meridiano
    min_edge_meridians_fraction: float = 0.95
    max_central_gap_mm: float = 0.45         # raio do primeiro ponto medido (lacuna central do Plácido)
    max_surface_residual_rms_um: float = 1.0
    max_condition_number: float = 1e6
    max_invalid_ray_fraction: float = 0.01
    # ceratometria por zona (tmsab/keratometry.py)
    k_min_meridian_fraction: float = 0.85    # fração de meridianos (ou da área da pupila) com dados utilizáveis
    k_max_missing_sector_deg: float = 20.0   # maior setor angular contíguo sem dados na circunferência/anel
    k_axis_min_cyl_D: float = 0.50           # abaixo disto o eixo é marcado "pouco definido" (nota, não bloqueio)
    k_interp_mark_fraction: float = 0.10     # fração de meridianos com pontos interpolados pelo aparelho que gera a marca †
    k_simk_tol_D: float = 0.05               # conferência SimK recalculado × XREF (potência): pega erro de leitura grosseiro
    k_simk_axis_tol_deg: float = 5.0         # idem (eixo), aplicada só com cilindro ≥ k_eixo_min_cyl_D
    k_eixo_min_cyl_D: float = 0.75           # eixos só são comparados (SimK × XREF, zona × pupila) acima deste ΔK
    k_zona_alerta_dif_D: float = 0.50        # alerta "astigmatismo dependente da zona": diferença vetorial 0–3 mm × pupila
    k_zona_alerta_eixo_deg: float = 10.0     # idem, diferença de eixo (com ΔK ≥ k_eixo_min_cyl_D nas duas zonas)
    provisional: bool = True


@dataclass
class KeratometryConfig:
    """Ceratometria por zona a partir da potência axial do TMS (definições em tmsab/keratometry.py e
    docs/convencoes.md). Entra no identificador do cálculo: mudar zonas ou método gera nova análise."""

    fonte: str = "DIO"                           # potência axial (sagital) do TMS, convenção 337,5/R_axial
    indice_ceratometrico: float = 1.3375
    metodo: str = "fourier_harmonicos_0a2"       # P(θ) = a0 + 1º e 2º harmônicos, mínimos quadrados
    centro: str = "vertice"                      # circunferências, zonas e anéis centrados no vértice topográfico
    # circunferências ("anel" de diâmetro d, como um ceratômetro): 2,4 mm ≈ anel dos biômetros ópticos (IOLMaster
    # 2,3–2,5; Lenstar 2,3; AL-Scan 2,4) e do SimK; 3,3 mm = anel externo do AL-Scan, que previu melhor o astigmatismo
    # residual de LIO tórica que o de 2,4 mm (Chang 2022)
    diametros_mm: tuple[float, ...] = (1.0, 2.0, 2.4, 3.0, 3.3, 4.0, 5.0, 6.0, 7.0, 8.0)
    diametros_resumo_mm: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)   # os que vão à folha A4
    # zonas (disco 0–d, média ponderada pela área, como Pentacam/Atlas): 0–3 mm centrada no vértice teve o menor
    # erro de astigmatismo em tórica (Meng 2021); 0–6 mm ≈ zona óptica da refrativa
    zonas_disco_mm: tuple[float, ...] = (3.0, 6.0)
    zonas_disco_resumo_mm: tuple[float, ...] = (3.0,)
    aneis_medidos: bool = True                   # uma linha por anel do Plácido, no diâmetro medido (2·RAD médio)
    zona_pupilar: str = "area_contorno_PUP"      # "area_contorno_PUP" | "nenhuma"
    simk_aneis: tuple[int, ...] = (6, 7, 8)      # regra do SimK do TMS-4N (anéis 1-based, ≈ Ø 2,5 mm)
    versao: str = "ceratometria_v1"


# ---------------------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------------------
@dataclass
class ZernikeCoefficient:
    n: int
    m: int
    j: int
    value_um: float
    name: str = ""


@dataclass
class SupportMetrics:
    measured_fraction: float
    usable_fraction: float
    central_gap_mm: float
    edge_reach_fraction: float            # fração de meridianos cuja cobertura medida atinge a borda
    max_missing_sector_deg: float
    n_points_inside: int
    n_measured_inside: int
    inside_fit_domain: bool


@dataclass
class WavefrontResult:
    profile: OpticalProfile
    center_xy: tuple[float, float]
    validity: str                                   # "disponivel" | "indisponivel" | "invalida"
    block_reasons: list[str]
    support: Optional[SupportMetrics]
    coefficients: list[ZernikeCoefficient] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    fit_residual_rms_um: Optional[float] = None
    condition_number: Optional[float] = None
    reference: dict[str, float] = field(default_factory=dict)   # s' (mm), potência equivalente, etc.
    n_rays: int = 0
    n_valid_rays: int = 0
    map_grid: Optional[dict[str, np.ndarray]] = None            # x, y, W, residual (µm), mask
    notes: list[str] = field(default_factory=list)


@dataclass
class KeratometryZone:
    """Uma linha da ceratometria: circunferência de diâmetro fixo, anel medido ou zona pupilar.

    K1 = meridiano mais PLANO e K2 = mais CURVO (convenção da biometria). Atenção: no XREF do TMS,
    SimK1 é o curvo. Eixos em graus TABO (0° às 3 h, anti-horário na vista do examinador), 1–180;
    eixo do K1 = eixo do K2 + 90° (2º harmônico de Fourier).
    """

    rotulo: str
    tipo: str                                    # "circunferencia" | "zona" (disco 0–d) | "anel" | "pupila"
    diametro_mm: Optional[float]
    validity: str                                # "disponivel" | "indisponivel" | "invalida"
    block_reasons: list[str] = field(default_factory=list)
    k1_plano_D: Optional[float] = None
    eixo_k1_deg: Optional[float] = None
    k2_curvo_D: Optional[float] = None
    eixo_k2_deg: Optional[float] = None
    delta_k_D: Optional[float] = None            # K2 − K1 (cilindro ceratométrico)
    k_medio_D: Optional[float] = None            # a0 do ajuste (média em θ)
    j0_D: Optional[float] = None                 # vetor de potência (Thibos), positivo = a favor da regra
    j45_D: Optional[float] = None
    assimetria_D: Optional[float] = None         # amplitude do 1º harmônico (assimetria/descentração)
    eixo_assimetria_deg: Optional[float] = None  # meridiano (0–360°) do lado mais curvo do 1º harmônico
    irregularidade_D: Optional[float] = None     # RMS do resíduo após os harmônicos 0–2
    anel: Optional[int] = None                   # anel do Plácido (1-based), quando tipo == "anel"
    centro_xy: Optional[tuple[float, float]] = None
    n_amostras: int = 0
    fracao_cobertura: float = 0.0                # meridianos com dados (zonas e pupila: com dados até a borda)
    fracao_interpolados: float = 0.0             # parte da cobertura que usa pontos interpolados pelo aparelho
    maior_setor_sem_dados_deg: float = 0.0
    eixo_pouco_definido: bool = False
    marca_interpolados: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class KeratometryResult:
    config: KeratometryConfig
    zonas: list[KeratometryZone] = field(default_factory=list)      # circunferências de diâmetro fixo
    discos: list[KeratometryZone] = field(default_factory=list)     # zonas 0–d (média ponderada pela área)
    aneis: list[KeratometryZone] = field(default_factory=list)      # anéis medidos (diâmetro medido)
    pupila: Optional[KeratometryZone] = None                        # área dentro do contorno pupilar
    simk_calculado: Optional[dict[str, float]] = None               # regra do TMS-4N recalculada (K1 = plano)
    simk_tms: Optional[dict[str, float]] = None                     # valores do XREF (lista de colunas permitidas)
    conferencia: list[str] = field(default_factory=list)            # SimK/ACP recalculados × TMS
    alertas: list[str] = field(default_factory=list)                # informativos (ex.: astigmatismo dependente da zona)
    notes: list[str] = field(default_factory=list)


@dataclass
class GeometrySummary:
    method: str
    fit_radius_mm: float
    surface_order: int
    n_points_used: int
    conic_R_mm: float
    conic_Q: float
    k_mean_D_keratometric: float          # (1.3375-1)/R*1000 — convenção ceratométrica, só para conferência
    residual_rms_mm: float
    residual_max_mm: float
    residual_by_ring_rms_mm: list[float]
    holdout_residual_rms_mm: Optional[float]
    condition_number: float
    central_gap_mm: float
    outer_radius_mm: float
    measured_fraction: float


@dataclass
class ApparentChordMu:
    """Chord mu aparente (tmsab/pupila.py): centro da pupila de entrada desta captura em relação ao vértice do TMS."""
    distancia_mm: float
    x_mm: float                                  # referencial do exame: vista do examinador, +x à direita da imagem
    y_mm: float                                  # +y superior
    nasal_mm: Optional[float]                    # + nasal, − temporal (None se o olho não for conhecido)
    superior_mm: float                           # + superior, − inferior
    diametro_pupila_mm: Optional[float]          # pupila desta captura (iluminação do Plácido)


@dataclass
class Analysis:
    analysis_id: str
    created_at: str
    versions: dict[str, str]
    identity: ExamIdentity
    device: DeviceInfo
    frame: GeometryFrame
    bundle_hashes: dict[str, str]
    synthetic: bool
    reconstruction: ReconstructionConfig
    geometry: GeometrySummary
    pupil: PupilData
    wavefronts: list[WavefrontResult]
    quality_config: QualityConfig
    import_diagnostics: list[str]
    warnings: list[str]
    purpose: str = "Aberrometria corneana anterior calculada — estágio de pesquisa"
    keratometry: Optional[KeratometryResult] = None    # a partir da 0.4.0; ausente em JSON antigo = não calculado
    chord_mu: Optional[ApparentChordMu] = None         # a partir da 0.4.0; None = pupila não detectada


# ---------------------------------------------------------------------------
# Utilidades de serialização
# ---------------------------------------------------------------------------
def _to_jsonable(obj: Any) -> Any:
    """Converte dataclasses/ndarrays para estruturas JSON; NaN/inf -> None."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, IntEnum):
        return int(obj)
    return obj


def to_jsonable(obj: Any) -> Any:
    return _to_jsonable(obj)


def stable_hash(payload: Any) -> str:
    """Hash SHA-256 de uma estrutura JSON canônica (chaves ordenadas)."""
    s = json.dumps(_to_jsonable(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
