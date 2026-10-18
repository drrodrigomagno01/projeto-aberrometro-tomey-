"""Gerador de exames sintéticos identificados como tais (plano §2, §8).

Simula a amostragem de um videoceratoscópio de Plácido sobre uma superfície analítica
conhecida e escreve um pacote no formato PRESUMIDO das Data Tables do TMS-4
(perfil tms4_datatables_v0), com manifesto `synthetic: true` e parâmetros de verdade.

Modelo de amostragem: anel j reflete-se onde a inclinação meridional da superfície
atinge o ângulo α_j; para uma esfera de raio R, r_j = R·sin(α_j). Para superfícies
astigmáticas, r_j(θ) ≈ R_local(θ)·sin(α_j), o que torna RAD dependente do ângulo
(como no aparelho real). Lacunas de pálpebra, ruído e quantização são opcionais.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from .geometry import BiconicSurface, CompositeSurface, ConicSurface, Surface, ZernikeDeformation


@dataclass
class SyntheticSpec:
    exam_id: str = "SYN0001"
    eye: str = "OD"
    R: float = 7.8                       # raio apical (mm) — para tórica: raio médio
    Q: float = -0.26
    cyl_D: float = 0.0                   # astigmatismo corneano (D, convenção ceratométrica 337.5/R) — eixo da meridiana mais plana
    axis_deg: float = 0.0
    zernike_um: dict[str, float] = field(default_factory=dict)   # perturbação de elevação {"3,1": 2.0} em µm, raio 4.5 mm
    zernike_radius: float = 4.5
    n_rings: int = 25
    n_angles: int = 256
    r_first: float = 0.35                # raio do primeiro anel (mm)
    r_last: float = 5.2
    pupil_center: tuple[float, float] = (0.0, 0.0)
    pupil_diameter: float = 4.0
    pupil_registered: bool = True
    lid_gap_from_ring: Optional[int] = None    # anel (0-based) a partir do qual há oclusão superior
    lid_gap_half_angle_deg: float = 25.0
    lid_gap_as_interpolated: bool = True      # True: marca como interpolado (negativo); False: ausente (zero)
    noise_um: float = 0.0                     # ruído gaussiano em HIT (µm)
    decimals: int = 4                         # precisão da exportação (casas decimais em mm)
    orientation: str = "angle_rows"           # "angle_rows" | "ring_rows"
    delimiter: str = ","
    seed: int = 0
    study_id: str = "ESTUDO_SINTETICO"
    model: str = "sintetico"


def build_surface(spec: SyntheticSpec) -> Surface:
    parts = []
    if abs(spec.cyl_D) > 1e-9:
        # meridiano mais plano no eixo (axis_deg) com raio R_flat; mais curvo perpendicular
        P_flat = 337.5 / spec.R - abs(spec.cyl_D) / 2.0
        P_steep = 337.5 / spec.R + abs(spec.cyl_D) / 2.0
        R_flat, R_steep = 337.5 / P_flat, 337.5 / P_steep
        parts.append(BiconicSurface(Rx=R_flat, Ry=R_steep, Qx=spec.Q, Qy=spec.Q, axis_deg=spec.axis_deg))
    else:
        parts.append(ConicSurface(spec.R, spec.Q))
    if spec.zernike_um:
        coeffs = {}
        for key, val in spec.zernike_um.items():
            n, m = (int(v) for v in key.split(","))
            coeffs[(n, m)] = val * 1e-3
        order = max(n for n, _ in coeffs)
        parts.append(ZernikeDeformation(spec.zernike_radius, order, coeffs))
    return CompositeSurface(parts) if len(parts) > 1 else parts[0]


def local_vertex_radius(surface: Surface, theta: np.ndarray, h: float = 0.05) -> np.ndarray:
    """Raio de curvatura meridional no vértice por diferença finita ao longo de cada meridiano."""
    x = h * np.cos(theta); y = h * np.sin(theta)
    z1 = np.asarray(surface.z(x, y)).ravel()
    z2 = np.asarray(surface.z(-x, -y)).ravel()
    z0 = float(np.asarray(surface.z(np.array([0.0]), np.array([0.0]))).ravel()[0])
    curv = (z1 + z2 - 2 * z0) / h**2
    return 1.0 / curv


def sample_placido(spec: SyntheticSpec, surface: Surface):
    """Retorna dicionário com arrays [n_rings, n_angles]: r, z, roc, dio, estado (0 medido, 1 interpolado, 2 ausente)."""
    rng = np.random.default_rng(spec.seed)
    k = np.arange(spec.n_angles)
    theta = np.radians(k * 360.0 / spec.n_angles)          # 0° às 3 h, anti-horário
    R_loc = local_vertex_radius(surface, theta)
    alpha = np.linspace(np.arcsin(spec.r_first / spec.R), np.arcsin(spec.r_last / spec.R), spec.n_rings)
    r = np.outer(np.sin(alpha), R_loc)                      # [n_rings, n_angles]
    x = r * np.cos(theta)[None, :]
    y = r * np.sin(theta)[None, :]
    z = np.asarray(surface.z(x, y)).reshape(r.shape)
    if spec.noise_um > 0:
        z = z + rng.normal(0.0, spec.noise_um * 1e-3, size=z.shape)
    # ROC meridional instantâneo (numérico) e potência ceratométrica
    gx, gy = surface.grad(x, y)
    gx = np.asarray(gx).reshape(r.shape); gy = np.asarray(gy).reshape(r.shape)
    dr = 1e-3
    x2 = (r + dr) * np.cos(theta)[None, :]; y2 = (r + dr) * np.sin(theta)[None, :]
    gx2, gy2 = surface.grad(x2, y2)
    gx2 = np.asarray(gx2).reshape(r.shape); gy2 = np.asarray(gy2).reshape(r.shape)
    zr = gx * np.cos(theta)[None, :] + gy * np.sin(theta)[None, :]
    zr2 = gx2 * np.cos(theta)[None, :] + gy2 * np.sin(theta)[None, :]
    zrr = (zr2 - zr) / dr
    roc = (1 + zr**2) ** 1.5 / zrr
    dio = 337.5 / roc
    state = np.zeros(r.shape, dtype=np.int8)
    if spec.lid_gap_from_ring is not None:
        ang = np.degrees(theta)
        sup = np.abs(((ang - 90.0 + 180.0) % 360.0) - 180.0) <= spec.lid_gap_half_angle_deg
        for j in range(spec.lid_gap_from_ring, spec.n_rings):
            state[j, sup] = 1 if spec.lid_gap_as_interpolated else 2
    return {"theta_deg": np.degrees(theta), "r": r, "z": z, "roc": roc, "dio": dio, "state": state}


def _fmt(v: float, decimals: int) -> str:
    return f"{v:.{decimals}f}"


def _write_table(path: str, values: np.ndarray, state: np.ndarray, spec: SyntheticSpec, role: str) -> None:
    """Codifica estado: medido -> valor; interpolado -> valor negativo; ausente -> 0 (formato presumido)."""
    enc = np.where(state == 1, -np.abs(values), values)
    enc = np.where(state == 2, 0.0, enc)
    table = enc.T if spec.orientation == "angle_rows" else enc     # angle_rows: 256 linhas x n_rings colunas
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# SINTETICO — Data Table presumida TMS-4, papel={role}, orientacao={spec.orientation}, unidade={'D' if role=='DIO' else 'mm'}\n")
        for row in table:
            fh.write(spec.delimiter.join(_fmt(v, spec.decimals) for v in row) + "\n")


def write_bundle(spec: SyntheticSpec, out_dir: str, with_roles=("HIT", "RAD", "ROC", "DIO")) -> str:
    """Escreve o pacote sintético em out_dir/<exam_id>_<eye>/ e devolve o caminho."""
    surface = build_surface(spec)
    s = sample_placido(spec, surface)
    d = os.path.join(out_dir, f"{spec.exam_id}_{spec.eye}")
    os.makedirs(d, exist_ok=True)
    base = f"{spec.exam_id}_{spec.eye}"
    if "HIT" in with_roles:
        _write_table(os.path.join(d, f"{base}_HIT.csv"), s["z"], s["state"], spec, "HIT")
    if "RAD" in with_roles:
        _write_table(os.path.join(d, f"{base}_RAD.csv"), s["r"], s["state"], spec, "RAD")
    if "ROC" in with_roles:
        _write_table(os.path.join(d, f"{base}_ROC.csv"), s["roc"], s["state"], spec, "ROC")
    if "DIO" in with_roles:
        _write_table(os.path.join(d, f"{base}_DIO.csv"), s["dio"], s["state"], spec, "DIO")
    if spec.pupil_registered:
        with open(os.path.join(d, f"{base}_PUP.txt"), "w", encoding="utf-8") as fh:
            fh.write("# SINTETICO — pupila digitalizada, referencial video_apex (x direita, y cima), mm\n")
            fh.write(f"center_x_mm={spec.pupil_center[0]:.4f}\ncenter_y_mm={spec.pupil_center[1]:.4f}\n")
            fh.write(f"diameter_mm={spec.pupil_diameter:.3f}\nframe=video_apex_x_direita_y_cima\nmethod=sintetico\n")
    manifest = {
        "synthetic": True,
        "study_id": spec.study_id,
        "exam_id": spec.exam_id,
        "eye": spec.eye,
        "acquired_at": None,
        "device": {"model": spec.model, "software_version": "sintetico-0.1", "calibration_state": "nao_aplicavel"},
        "export": {"orientation": spec.orientation, "decimals": spec.decimals, "delimiter": spec.delimiter},
        "truth": {k: v for k, v in asdict(spec).items()},
    }
    with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    return d


def standard_cases() -> list[SyntheticSpec]:
    """Conjunto padrão de casos sintéticos identificados (ver docs/validacao.md)."""
    return [
        SyntheticSpec(exam_id="SYN-ESF", R=7.8, Q=0.0, eye="OD"),
        SyntheticSpec(exam_id="SYN-CON", R=7.8, Q=-0.26, eye="OD"),
        SyntheticSpec(exam_id="SYN-APL", R=7.8, Q=-1.0 / 1.376**2, eye="OD"),
        SyntheticSpec(exam_id="SYN-TOR", R=7.8, Q=-0.2, cyl_D=2.0, axis_deg=0.0, eye="OD"),
        SyntheticSpec(exam_id="SYN-TOR", R=7.8, Q=-0.2, cyl_D=2.0, axis_deg=0.0, eye="OS"),
        SyntheticSpec(exam_id="SYN-COMA", R=7.8, Q=-0.2, zernike_um={"3,1": 3.0}, eye="OD"),
        SyntheticSpec(exam_id="SYN-TREF", R=7.8, Q=-0.2, zernike_um={"3,-3": 2.0}, eye="OD"),
        SyntheticSpec(exam_id="SYN-PUP", R=7.8, Q=-0.2, pupil_center=(0.3, -0.2), zernike_um={"4,0": 1.0}, eye="OD"),
        SyntheticSpec(exam_id="SYN-GAP", R=7.8, Q=-0.2, lid_gap_from_ring=16, lid_gap_half_angle_deg=30, eye="OD"),
        SyntheticSpec(exam_id="SYN-GAPM", R=7.8, Q=-0.2, lid_gap_from_ring=12, lid_gap_half_angle_deg=35, lid_gap_as_interpolated=False, eye="OD"),
        SyntheticSpec(exam_id="SYN-RUIDO", R=7.8, Q=-0.2, noise_um=0.5, decimals=3, eye="OD", seed=7),
        SyntheticSpec(exam_id="SYN-RING", R=7.8, Q=-0.2, orientation="ring_rows", delimiter=";", eye="OD"),
        SyntheticSpec(exam_id="SYN-IRREG", R=7.6, Q=-0.5, zernike_um={"3,1": 4.0, "3,-3": 2.5, "4,0": 1.5, "5,1": 0.8}, eye="OD"),
    ]


# ---------------------------------------------------------------------------
# Formato REAL do TMS-4N (perfil v1): HIT_1_<exame>.dat etc., 256 linhas x 34 colunas, XREF.dat
# ---------------------------------------------------------------------------
def write_export_folder_v1(specs: list[SyntheticSpec], out_dir: str, first_exam: int = 90001,
                           patient_name: tuple[str, str] = ("SINTETICO", "PACIENTE"), append: bool = False,
                           date_str: str = "17/06/2026") -> str:
    """Escreve uma pasta no formato observado na estação real (TmsSw, cone de 25 anéis):
    uma linha por ângulo (256), 34 colunas (anéis; as restantes zero), 4 casas decimais, negativo =
    interpolado, zero = ausente; PUP com 256 raios da borda pupilar; AVE com potência e diâmetro médios por
    anel; XREF.dat com cabeçalho de grupos, nome fictício (que o leitor deve ignorar), data/hora, olho e
    caminho nativo. Com `append=True` reproduz o comportamento observado do TmsSw ao exportar de novo para
    uma pasta que já tem XREF.dat: as linhas novas são acrescentadas ao índice existente (sem novo
    cabeçalho) — é assim que dois pacientes acabam na mesma pasta. Devolve a pasta."""
    os.makedirs(out_dir, exist_ok=True)
    xref_rows = []
    for i, spec in enumerate(specs):
        exam = first_exam + i
        surface = build_surface(spec)
        s = sample_placido(spec, surface)
        ncols = 34
        nr = spec.n_rings

        def table(values, state):
            enc = np.where(state == 1, -np.abs(values), values)
            enc = np.where(state == 2, 0.0, enc)
            full = np.zeros((spec.n_angles, ncols))
            full[:, :nr] = enc.T
            lines = [",".join(f"{v:9.4f}" for v in row) for row in full]
            return "\r\n".join(lines) + "\r\n"

        for role, arr in (("HIT", s["z"]), ("RAD", s["r"]), ("DIO", s["dio"]), ("ROC", s["dio"] * 0.98), ("POW", s["dio"] * 1.02),
                          ("RNG", s["r"] * 44.0)):
            with open(os.path.join(out_dir, f"{role}_1_{exam}.dat"), "w", encoding="ascii", newline="") as fh:
                fh.write(table(arr, s["state"]))
        # PUP: raio da borda por ângulo, centro deslocado (pupil_center) e alguns pontos interpolados (negativos)
        th = np.radians(s["theta_deg"])
        cx, cy = spec.pupil_center
        R = spec.pupil_diameter / 2.0
        # raio da borda a partir do vértice para um círculo de centro (cx, cy) e raio R
        b = cx * np.cos(th) + cy * np.sin(th)
        r_edge = b + np.sqrt(np.maximum(R**2 - (cx**2 + cy**2) + b**2, 0.0))
        r_edge = np.round(r_edge, 2)
        r_edge[100:106] *= -1
        with open(os.path.join(out_dir, f"PUP_1_{exam}.dat"), "w", encoding="ascii", newline="") as fh:
            fh.write("\r\n".join(f"{v:.2f}" for v in r_edge) + "\r\n")
        # AVE: potência média e diâmetro médio por anel (34 linhas)
        ave = []
        for j in range(ncols):
            if j < nr:
                ave.append(f"{np.nanmean(s['dio'][j]):9.4f},{2*np.nanmean(s['r'][j]):9.4f}")
            else:
                ave.append(f"{0.0:9.4f},{0.0:9.4f}")
        with open(os.path.join(out_dir, f"AVE_1_{exam}.dat"), "w", encoding="ascii", newline="") as fh:
            fh.write("\r\n".join(ave) + "\r\n")
        hh = 9
        mm = 30 + i
        xref_rows.append(
            f"1,{patient_name[0]} ,{patient_name[1]} ,0,,{date_str} {hh:02d}:{mm:02d}:00,{spec.eye},C:\\\\Tomey\\\\Tms\\\\NewExams\\\\{exam}.tms,"
            f"44.70,{(spec.axis_deg + 90) % 180:.0f},44.30,{spec.axis_deg % 180:.0f},44.20,165,{abs(spec.cyl_D):.2f},44.44,0.06,0.22,0.33,0.39,1.53,0.80,0.09,0.00,0.34,0.98,0.00,0.00,"
            f"44.54,0.20,57.30,0.16,172.84,0.13,44.45,0.21,57.61,0.24,166.56,0.15, 0.29, 0.31,{spec.Q:.2f}, 0.11, 0.89"
        )
    header1 = ",,,Patient Record,,,,,,,,,,,,,,,Corneal Statistics,,,,,,,,,,,,,,,Fourier Index Standard,,,,,,,Eccentricity,"
    header2 = ("File REF.,Last Name,First Name,-----,ID,Date&Time,Eye,Filename,SimK1,SimK1 Ang.,SimK2,SimK2 Ang.,MinK,MinK Ang.,CYL,ACP,"
               "SRI,SAI,CEI,SDP,DSI,OSI,CSI,KCI,IAI,AA,EDP,EDD,3mm Spherical Equ.,3mm Reg. Astig.,3mm Reg. Astig. Axis,3mm Asymmetry,"
               "3mm Asymmetry Axis,3mm Higher order irregularity,6mm Spherical Equ.,6mm Reg. Astig.,6mm Reg. Astig. Axis,6mm Asymmetry,"
               "6mm Asymmetry Axis,6mm Higher order irregularity,Es,Em,Q,E,p")
    xref_path = os.path.join(out_dir, "XREF.dat")
    if append and os.path.isfile(xref_path):
        with open(xref_path, "a", encoding="ascii", newline="") as fh:
            fh.write("\r\n".join(xref_rows) + "\r\n")
    else:
        with open(xref_path, "w", encoding="ascii", newline="") as fh:
            fh.write("\r\n".join([header1, header2] + xref_rows) + "\r\n")
    return out_dir
