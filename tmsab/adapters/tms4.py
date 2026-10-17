"""Adaptador TMS-4 / TMS-4N — exportação "Data Tables" (ASCII), perfil parametrizado (plano §3).

Regras obrigatórias do importador:
- Conserva os arquivos recebidos e seus hashes; trabalha em cópias em memória; registra o perfil usado.
- Valida estrutura completa: papéis dos arquivos, dimensões, separador, formato decimal e vínculo
  ao mesmo exame/olho. Nome plausível não basta para parear: formas devem coincidir.
- Decodificador específico por campo: preserva token original, valor decodificado e estado.
- Jamais aplica valor absoluto ao conjunto; o sinal só é interpretado pela regra do campo.
- Erros identificáveis (AdapterError) com causa e requisito; nenhuma importação parcial segue
  silenciosamente para o cálculo.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from importlib import resources
from typing import Any, Optional

import numpy as np

from ..model import (
    DeviceInfo,
    ExamIdentity,
    FieldTable,
    GeometryFrame,
    NormalizedExam,
    PointState,
    PupilData,
    Samples,
    SourceBundle,
    SourceFile,
    sha256_file,
)

MAX_FILE_BYTES = 50 * 1024 * 1024


class AdapterError(Exception):
    """Erro identificável de importação: mensagem com causa e requisito para corrigir a entrada."""


# ---------------------------------------------------------------------------
# Perfis
# ---------------------------------------------------------------------------
def list_profiles() -> list[str]:
    pkg = resources.files("tmsab.adapters.profiles")
    return sorted(p.name[:-5] for p in pkg.iterdir() if p.name.endswith(".json"))


def load_profile(name_or_path: str) -> dict[str, Any]:
    if os.path.isfile(name_or_path):
        with open(name_or_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    pkg = resources.files("tmsab.adapters.profiles")
    cand = pkg / f"{name_or_path}.json"
    if not cand.is_file():
        raise AdapterError(f"perfil de adaptador desconhecido: {name_or_path}. Disponíveis: {list_profiles()}")
    return json.loads(cand.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Leitura lexical
# ---------------------------------------------------------------------------
@dataclass
class RawTable:
    rows: list[list[str]]
    delimiter: str
    decimal: str
    header_lines: int
    had_index_column: bool


def _detect_delimiter(line: str, candidates: list[str]) -> str:
    """Prioridade: tabulação > ';' > ',' > espaço (vírgula só é separador quando não há ';' nem tabulação,
    pois pode ser separador decimal)."""
    priority = ["\t", ";", ",", " "]
    present = [d for d in priority if d in candidates and line.count(d) > 0]
    if not present:
        raise AdapterError("separador não identificado na primeira linha de dados; requisito: tabela delimitada por vírgula, ponto-e-vírgula, tabulação ou espaço")
    return present[0]


def read_table_text(path: str, profile: dict[str, Any]) -> RawTable:
    size = os.path.getsize(path)
    if size > MAX_FILE_BYTES:
        raise AdapterError(f"{os.path.basename(path)}: arquivo maior que o limite ({size} bytes > {MAX_FILE_BYTES}); requisito: exportação de tabela única")
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        lines = fh.read().splitlines()
    data_lines = []
    header = 0
    started = False
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith("//"):
            if not started:
                header += 1
            continue
        # linha de dados precisa começar com algo numérico (ou índice)
        first = re.split(r"[,;\t ]+", s)[0]
        if not re.match(r"^[-+]?(\d+([.,]\d*)?|[.,]\d+)([eE][-+]?\d+)?$|^[*]$|^NaN$|^nan$|^-$", first):
            if not started:
                header += 1
                continue
            raise AdapterError(f"{os.path.basename(path)}: linha não numérica após o início dos dados: '{s[:40]}'; requisito: tabela retangular sem texto intercalado")
        started = True
        data_lines.append(s)
    if not data_lines:
        raise AdapterError(f"{os.path.basename(path)}: nenhuma linha de dados encontrada")
    delim = _detect_delimiter(data_lines[0], profile.get("delimiters", [",", ";", "\t", " "]))
    splitter = re.compile(r"[ \t]+") if delim in (" ", "\t") else re.compile(re.escape(delim))
    rows = [[t.strip() for t in splitter.split(l)] for l in data_lines]
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise AdapterError(f"{os.path.basename(path)}: larguras de linha inconsistentes {sorted(widths)}; requisito: tabela retangular (arquivo truncado ou linha corrompida)")
    # decimal
    decimal = profile.get("decimal", "auto")
    if decimal == "auto":
        sample = " ".join(rows[0][:8])
        decimal = "," if ("," in sample and delim != ",") else "."
    # coluna de índice: primeira coluna inteira crescente 0..N-1 ou 1..N
    had_index = False
    first_col = [r[0] for r in rows]
    if all(re.match(r"^\d+$", v) for v in first_col):
        vals = [int(v) for v in first_col]
        if vals == list(range(len(vals))) or vals == list(range(1, len(vals) + 1)):
            had_index = True
            rows = [r[1:] for r in rows]
    return RawTable(rows, delim, decimal, header, had_index)


def _decode_token(tok: str, rule: dict[str, Any], decimal: str) -> tuple[float, int]:
    """Decodifica um token conforme a regra do campo. Retorna (valor, estado)."""
    if tok in rule.get("missing_tokens", ["", "*", "NaN", "nan", "-"]):
        return np.nan, int(PointState.MISSING)
    t = tok.replace(decimal, ".") if decimal != "." else tok
    try:
        v = float(t)
    except ValueError:
        return np.nan, int(PointState.INVALID)
    if not np.isfinite(v):
        return np.nan, int(PointState.INVALID)
    if v == 0.0 and rule.get("zero_is_missing", False):
        return np.nan, int(PointState.MISSING)
    if v < 0.0 and rule.get("negative_means_interpolated", False):
        return -v, int(PointState.INTERPOLATED)
    return v, int(PointState.MEASURED)


def decode_field(raw: RawTable, role: str, rule: dict[str, Any], n_angles: int, n_rings_allowed: list[int],
                 orientation: str = "auto") -> tuple[FieldTable, str]:
    """Converte a tabela crua em FieldTable [n_rings, n_angles]; retorna também a orientação detectada."""
    nrow, ncol = len(raw.rows), len(raw.rows[0])
    if orientation == "auto":
        if nrow == n_angles and ncol in n_rings_allowed:
            orientation = "angle_rows"
        elif ncol == n_angles and nrow in n_rings_allowed:
            orientation = "ring_rows"
        elif nrow == n_angles and ncol == n_angles:
            raise AdapterError(f"{role}: tabela {nrow}x{ncol} ambígua; requisito: informar 'orientation' no perfil")
        else:
            raise AdapterError(
                f"{role}: dimensões {nrow}x{ncol} incompatíveis com {n_angles} ângulos e anéis {n_rings_allowed}; "
                "requisito: exportação completa (Angle ou Ring) do mesmo exame"
            )
    tokens = np.array(raw.rows, dtype=object)
    if orientation == "angle_rows":
        tokens = tokens.T  # -> [n_rings, n_angles]
    elif orientation != "ring_rows":
        raise AdapterError(f"orientação desconhecida no perfil: {orientation}")
    nr, na = tokens.shape
    if na != n_angles or nr not in n_rings_allowed:
        raise AdapterError(f"{role}: após orientação, dimensões {nr}x{na} inválidas")
    values = np.empty((nr, na), dtype=float)
    states = np.empty((nr, na), dtype=np.int8)
    for i in range(nr):
        for k in range(na):
            v, s = _decode_token(str(tokens[i, k]), rule, raw.decimal)
            values[i, k] = v
            states[i, k] = s
    n_invalid = int((states == PointState.INVALID).sum())
    if n_invalid:
        raise AdapterError(f"{role}: {n_invalid} tokens não numéricos/não finitos; requisito: exportação íntegra (sem células corrompidas)")
    return FieldTable(name=role, unit=rule.get("unit", ""), values=values, states=states, raw=tokens), orientation


# ---------------------------------------------------------------------------
# PUP (chave=valor) — formato presumido
# ---------------------------------------------------------------------------
def read_pupil_file(path: str, rule: dict[str, Any]) -> PupilData:
    kv: dict[str, str] = {}
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        for ln in fh:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*[=:,;\t]\s*(.+)$", s)
            if m:
                kv[m.group(1).lower()] = m.group(2).strip()
    def num(key):
        if key not in kv:
            return None
        try:
            return float(kv[key].replace(",", "."))
        except ValueError:
            raise AdapterError(f"PUP: valor inválido para {key}: '{kv[key]}'")
    cx, cy, dia = num("center_x_mm"), num("center_y_mm"), num("diameter_mm")
    if cx is None or cy is None:
        return PupilData(state="ausente", frame=rule.get("frame", "desconhecido"), method=kv.get("method", "desconhecido"))
    return PupilData(
        state="registrada",
        center_x=rule.get("x_sign", 1) * cx,
        center_y=rule.get("y_sign", 1) * cy,
        diameter=dia,
        frame=kv.get("frame", rule.get("frame", "desconhecido")),
        method=kv.get("method", "desconhecido"),
    )


# ---------------------------------------------------------------------------
# Pacote
# ---------------------------------------------------------------------------
def _match_role(filename: str, roles: dict[str, Any]) -> Optional[str]:
    hits = [r for r, rule in roles.items() if any(re.search(p, filename) for p in rule.get("patterns", []))]
    if len(hits) > 1:
        raise AdapterError(f"{filename}: nome corresponde a mais de um papel {hits}; requisito: um arquivo por tabela")
    return hits[0] if hits else None


def _match_eye(filename: str, eye_patterns: dict[str, list[str]]) -> Optional[str]:
    hits = [e for e, pats in eye_patterns.items() if any(re.search(p, filename) for p in pats)]
    if len(hits) > 1:
        raise AdapterError(f"{filename}: lateralidade ambígua no nome {hits}")
    return hits[0] if hits else None


def import_tms_bundle(directory: str, profile: str | dict[str, Any] = "tms4_datatables_v0",
                      manifest_override: Optional[dict[str, Any]] = None) -> NormalizedExam:
    """Importa um pacote (diretório) exportado do TMS e devolve o exame normalizado.

    Lança AdapterError com causa e requisito em qualquer inconsistência essencial.
    """
    prof = load_profile(profile) if isinstance(profile, str) else profile
    if not os.path.isdir(directory):
        raise AdapterError(f"diretório não encontrado: {directory}")
    roles_cfg = prof["roles"]
    n_angles = int(prof["n_angles"])
    n_rings_allowed = [int(v) for v in prof["n_rings_allowed"]]

    # manifesto (opcional)
    manifest: dict[str, Any] = {}
    mpath = os.path.join(directory, prof.get("manifest_filename", "manifest.json"))
    if os.path.isfile(mpath):
        with open(mpath, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    if manifest_override:
        manifest.update(manifest_override)

    files: list[SourceFile] = []
    tables: dict[str, tuple[RawTable, str]] = {}
    pupil = PupilData(state="ausente")
    eyes: set[str] = set()
    exam_ids: set[str] = set()
    diagnostics: list[str] = []
    warnings: list[str] = []

    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        if os.path.islink(path) or ".." in name or os.sep in name:
            raise AdapterError(f"{name}: caminho indevido dentro do pacote")
        if name == os.path.basename(mpath):
            files.append(SourceFile(path=path, role="MANIFEST", sha256=sha256_file(path), size_bytes=os.path.getsize(path)))
            continue
        role = _match_role(name, roles_cfg)
        if role is None:
            files.append(SourceFile(path=path, role="OUTRO", sha256=sha256_file(path), size_bytes=os.path.getsize(path)))
            diagnostics.append(f"arquivo ignorado (papel não reconhecido): {name}")
            continue
        if role in tables or (role == "PUP" and pupil.state != "ausente"):
            raise AdapterError(f"{name}: papel {role} duplicado no pacote; requisito: um arquivo por tabela e por exame")
        eye = _match_eye(name, prof.get("eye_patterns", {}))
        if eye:
            eyes.add(eye)
        m = re.match(prof.get("exam_id_regex", "^(?P<exam>[^_]+)"), name)
        if m:
            exam_ids.add(m.group("exam"))
        sf = SourceFile(path=path, role=role, sha256=sha256_file(path), size_bytes=os.path.getsize(path))
        if role == "PUP":
            pupil = read_pupil_file(path, roles_cfg["PUP"])
            files.append(sf)
            continue
        raw = read_table_text(path, prof)
        sf.n_rows, sf.n_cols, sf.delimiter, sf.decimal = len(raw.rows), len(raw.rows[0]), raw.delimiter, raw.decimal
        tables[role] = (raw, name)
        files.append(sf)

    # papéis obrigatórios
    for role, rule in roles_cfg.items():
        if rule.get("required") and role not in tables:
            raise AdapterError(f"tabela obrigatória ausente: {role}; requisito: exportar {role} (Data Tables) do mesmo exame")

    # lateralidade e exame: manifesto tem precedência, mas conflito bloqueia
    eye_manifest = manifest.get("eye")
    if eye_manifest and eyes and eye_manifest not in eyes:
        raise AdapterError(f"conflito de lateralidade: manifesto {eye_manifest} vs nomes {sorted(eyes)}")
    if len(eyes) > 1:
        raise AdapterError(f"mistura de olhos no pacote: {sorted(eyes)}; requisito: um olho por pacote")
    if len(exam_ids) > 1:
        raise AdapterError(f"mistura de exames no pacote: {sorted(exam_ids)}; requisito: um exame por pacote")
    eye = eye_manifest or (next(iter(eyes)) if eyes else None)
    if eye == "OE":
        eye = "OS"
    exam_id = manifest.get("exam_id") or (next(iter(exam_ids)) if exam_ids else os.path.basename(os.path.abspath(directory)))
    if manifest.get("exam_id") and exam_ids and manifest["exam_id"] not in exam_ids:
        raise AdapterError(f"conflito de identificador de exame: manifesto {manifest['exam_id']} vs nomes {sorted(exam_ids)}")
    if eye is None:
        warnings.append("lateralidade não informada (nem manifesto nem nome de arquivo)")

    # decodificação
    fields: dict[str, FieldTable] = {}
    orientations: dict[str, str] = {}
    shapes: set[tuple[int, int]] = set()
    for role, (raw, name) in tables.items():
        ft, orient = decode_field(raw, role, roles_cfg[role], n_angles, n_rings_allowed, prof.get("orientation", "auto"))
        fields[role] = ft
        orientations[role] = orient
        shapes.add(ft.values.shape)
        for sf in files:
            if sf.path.endswith(name):
                sf.orientation = orient
    if len(shapes) != 1:
        raise AdapterError(f"tabelas com dimensões diferentes {sorted(shapes)}; requisito: todas as tabelas do mesmo exame e configuração")
    n_rings, _ = next(iter(shapes))

    # coordenadas canônicas
    frame = GeometryFrame(
        angle_start_deg=float(prof.get("angle_start_deg", 0.0)),
        angle_direction=int(prof.get("angle_direction", 1)),
        mirror_x=bool(prof.get("mirror_x", False)),
        transform_verified=(prof.get("status") == "verificado") or bool(manifest.get("synthetic")),
        notes=prof.get("angle_note", ""),
    )
    k = np.arange(n_angles)
    theta = frame.angle_start_deg + frame.angle_direction * k * (360.0 / n_angles)
    if frame.mirror_x:
        theta = 180.0 - theta
    theta = np.mod(theta, 360.0)

    if prof.get("rad_meaning", "radial_plano") != "radial_plano":
        raise AdapterError("rad_meaning diferente de 'radial_plano' ainda não suportado (tarefa de investigação do formato)")
    hcfg = prof.get("height", {})
    if hcfg.get("reference", "plano_apical") != "plano_apical":
        raise AdapterError("referência de altura diferente de 'plano_apical' ainda não suportada; requisito: raio da esfera de referência documentado")
    hit, rad = fields["HIT"], fields["RAD"]
    r = rad.values
    z = hcfg.get("sign", 1) * hit.values
    th = np.radians(theta)[None, :]
    x = r * np.cos(th)
    y = r * np.sin(th)
    st = np.full((n_rings, n_angles), int(PointState.MEASURED), dtype=np.int8)
    interp = (hit.states == PointState.INTERPOLATED) | (rad.states == PointState.INTERPOLATED)
    missing = (hit.states == PointState.MISSING) | (rad.states == PointState.MISSING) | ~np.isfinite(r) | ~np.isfinite(z)
    st[interp] = int(PointState.INTERPOLATED)
    st[missing] = int(PointState.MISSING)
    x = np.where(missing, np.nan, x); y = np.where(missing, np.nan, y); z = np.where(missing, np.nan, z); r = np.where(missing, np.nan, r)

    # sanidade física mínima
    meas = st == PointState.MEASURED
    if meas.sum() < 0.3 * st.size:
        raise AdapterError(f"apenas {int(meas.sum())} de {st.size} pontos medidos; requisito: exame com cobertura suficiente")
    if np.nanmax(r) > 8.0 or np.nanmin(r) < 0.05:
        raise AdapterError(f"RAD fora da faixa física esperada (min {np.nanmin(r):.3f}, max {np.nanmax(r):.3f} mm); requisito: confirmar unidade (mm) e significado de RAD")
    if np.nanmax(z) > 3.0 or np.nanmin(z) < -0.5:
        raise AdapterError(f"HIT fora da faixa física esperada (min {np.nanmin(z):.3f}, max {np.nanmax(z):.3f} mm); requisito: confirmar unidade, sinal e referência de altura")
    # monotonicidade radial média por anel
    rn = np.nanmean(r, axis=1)
    if not np.all(np.diff(rn[np.isfinite(rn)]) > 0):
        warnings.append("raio médio por anel não é monotônico crescente — verificar orientação Angle/Ring e semântica de RAD")

    samples = Samples(n_rings=n_rings, n_angles=n_angles, theta_deg=theta, fields=fields, x=x, y=y, z=z, r=r, point_state=st)

    device = DeviceInfo(
        model=str(manifest.get("device", {}).get("model", "desconhecido")),
        software_version=str(manifest.get("device", {}).get("software_version", "desconhecida")),
        cone=manifest.get("device", {}).get("cone"),
        calibration_state=str(manifest.get("device", {}).get("calibration_state", "nao_informado")),
    )
    identity = ExamIdentity(
        study_id=str(manifest.get("study_id", "sem_estudo")),
        exam_id=str(exam_id),
        eye=eye,
        acquired_at=manifest.get("acquired_at"),
        date_precision=str(manifest.get("date_precision", "desconhecida" if not manifest.get("acquired_at") else "dia")),
        capture_index=manifest.get("capture_index"),
    )
    bundle = SourceBundle(
        directory=os.path.abspath(directory), files=files, adapter_profile=prof["profile_id"],
        adapter_version=prof.get("adapter_version", "0"), export_settings={"orientations": orientations, **manifest.get("export", {})},
        manifest=manifest, synthetic=bool(manifest.get("synthetic", False)),
    )
    if prof.get("status") != "verificado" and not bundle.synthetic:
        warnings.append(f"perfil de exportação '{prof['profile_id']}' NÃO verificado com arquivos reais — compatibilidade não declarada")
    if not manifest:
        warnings.append("manifesto ausente — modelo, versão de software e data não informados")
    diagnostics.append(f"tabelas: {', '.join(f'{k} ({orientations[k]})' for k in fields)}; anéis={n_rings}, ângulos={n_angles}")
    diagnostics.append(f"pontos medidos {int(meas.sum())}, interpolados {int((st == PointState.INTERPOLATED).sum())}, ausentes {int((st == PointState.MISSING).sum())}")
    diagnostics.append(f"pupila: {pupil.state}" + (f" centro ({pupil.center_x:+.2f}, {pupil.center_y:+.2f}) mm, referencial {pupil.frame}" if pupil.state == "registrada" else ""))
    return NormalizedExam(bundle=bundle, identity=identity, device=device, frame=frame, samples=samples, pupil=pupil,
                          diagnostics=diagnostics, warnings=warnings)
