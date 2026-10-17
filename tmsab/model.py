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
    provisional: bool = True


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
