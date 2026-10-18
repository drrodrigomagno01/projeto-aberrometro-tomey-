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
                 orientation: str = "auto", trim_padding: bool = False) -> tuple[FieldTable, str]:
    """Converte a tabela crua em FieldTable [n_rings, n_angles]; retorna também a orientação detectada.

    trim_padding: a exportação real do TMS-4N tem 34 colunas por ângulo, das quais só os anéis do cone
    (25 ou 31) têm dados; as colunas finais são zeros. Elas são removidas antes da validação.
    """
    nrow, ncol = len(raw.rows), len(raw.rows[0])
    max_rings = max(n_rings_allowed)
    if orientation == "auto":
        if nrow == n_angles and (ncol in n_rings_allowed or (trim_padding and min(n_rings_allowed) <= ncol <= 64)):
            orientation = "angle_rows"
        elif ncol == n_angles and (nrow in n_rings_allowed or (trim_padding and min(n_rings_allowed) <= nrow <= 64)):
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
    if na != n_angles:
        raise AdapterError(f"{role}: após orientação, dimensões {nr}x{na} inválidas ({n_angles} ângulos esperados)")
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
    if trim_padding:
        has_data = (states != PointState.MISSING).any(axis=1)
        last = int(np.max(np.where(has_data)[0])) + 1 if has_data.any() else 0
        if last < nr:
            values, states, tokens = values[:last], states[:last], tokens[:last]
            nr = last
    if nr not in n_rings_allowed or nr > max_rings:
        raise AdapterError(f"{role}: {nr} anéis com dados; esperado {n_rings_allowed}; requisito: confirmar cone/exportação")
    return FieldTable(name=role, unit=rule.get("unit", ""), values=values, states=states, raw=tokens), orientation


# ---------------------------------------------------------------------------
# PUP no formato real (256 raios por ângulo) e XREF (índice do TMS, sem dados nominais)
# ---------------------------------------------------------------------------
def read_pupil_radii(path: str, rule: dict[str, Any], n_angles: int, theta_deg: np.ndarray) -> PupilData:
    """PUP do TMS-4N: uma linha por ângulo com o raio da borda pupilar (mm) referido ao vértice; negativo = interpolado.

    O centro pupilar é obtido por ajuste de círculo (mínimos quadrados) aos pontos da borda, no sistema canônico.
    """
    vals: list[float] = []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        for ln in fh:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            try:
                vals.append(float(s.replace(",", ".")))
            except ValueError:
                raise AdapterError(f"PUP: linha não numérica '{s[:20]}'")
    if len(vals) != n_angles:
        raise AdapterError(f"PUP: {len(vals)} linhas; esperado {n_angles} raios (um por ângulo)")
    r = np.array(vals, dtype=float)
    interp = r < 0
    if rule.get("negative_means_interpolated", True):
        r = np.abs(r)
    ok = np.isfinite(r) & (r > 0)
    if ok.sum() == 0:
        # observado em exame real: PUP com 256 zeros = o TMS não detectou a borda pupilar nessa captura
        return PupilData(state="ausente", frame=rule.get("frame", "desconhecido"),
                         method="PUP com todos os raios = 0 (TMS não detectou a borda pupilar nesta captura)")
    if ok.sum() < n_angles * 0.5:
        return PupilData(state="ausente", frame=rule.get("frame", "desconhecido"),
                         method=f"PUP com apenas {int(ok.sum())} de {n_angles} raios válidos (< 50%)")
    th = np.radians(theta_deg)
    x, y = r * np.cos(th), r * np.sin(th)
    # ajuste de círculo: x² + y² + D x + E y + F = 0
    A = np.column_stack([x[ok], y[ok], np.ones(ok.sum())])
    b = -(x[ok] ** 2 + y[ok] ** 2)
    (D, E, F), *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = -D / 2.0, -E / 2.0
    R = float(np.sqrt(max(cx**2 + cy**2 - F, 0.0)))
    return PupilData(
        state="registrada", center_x=float(cx), center_y=float(cy), diameter=2.0 * R,
        frame=rule.get("frame", "vertice_topografico_polar"),
        method=f"ajuste de círculo a {int(ok.sum())} pontos da borda PUP ({int(interp.sum())} interpolados)",
        edge_r=r,
    )


XREF_EXCLUDED_COLUMNS = ("Last Name", "First Name", "ID", "-----")
XREF_IDENTITY_COLUMNS = ("Last Name", "First Name", "ID")


def _patient_key_from_row(row: dict[str, Any]) -> Optional[str]:
    """Código de estudo pseudonimizado a partir das colunas de identificação do XREF.

    Nome e ID são usados SOMENTE para calcular um SHA-256 (8 hex) e descartados em seguida — o código
    é o mesmo para o mesmo paciente em qualquer pasta/exportação, e nenhum nome fica em estrutura alguma.
    Normalização: maiúsculas, espaços internos colapsados, acentos preservados.
    """
    import hashlib
    parts = []
    for col in XREF_IDENTITY_COLUMNS:
        v = row.get(col)
        parts.append(" ".join(str(v if v is not None else "").upper().split()))
    if not any(parts):
        return None
    return "P-" + hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:8].upper()


def read_xref(path: str) -> dict[str, dict[str, Any]]:
    """Lê o XREF.dat (ou xref_anon.csv) exportado pelo TMS e devolve
    {exame: {eye, acquired_at, indices, patient_key, native_path_hint}}.

    As colunas com identificação direta (nome, ID) NUNCA são copiadas para a estrutura devolvida: servem
    apenas para derivar `patient_key` (hash), que separa pacientes quando uma pasta recebeu mais de uma
    exportação (o TMS acrescenta linhas a um XREF.dat existente).
    """
    import csv
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    # primeira linha pode ser cabeçalho de grupos (",,,Patient Record,...")
    start = 0
    for i, ln in enumerate(lines[:3]):
        if "Eye" in ln and "Filename" in ln:
            start = i
            break
    rows = list(csv.DictReader(lines[start:]))
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        fn = (r.get("Filename") or "").strip()
        m = re.search(r"(\d+)\.tms$", fn, flags=re.IGNORECASE)
        if not m:
            continue
        exam = m.group(1)
        patient_key = _patient_key_from_row(r)
        eye = (r.get("Eye") or "").strip().upper()
        eye = {"OE": "OS"}.get(eye, eye) if eye in ("OD", "OS", "OE") else None
        dt = (r.get("Date&Time") or "").strip()
        acquired = None
        m2 = re.match(r"(\d{2})/(\d{2})/(\d{4})\s+(\d{2}):(\d{2}):(\d{2})", dt)
        if m2:
            d, mo, y, hh, mi, ss = m2.groups()
            acquired = f"{y}-{mo}-{d}T{hh}:{mi}:{ss}"
        indices: dict[str, Any] = {}
        for k, v in r.items():
            if k is None or k in XREF_EXCLUDED_COLUMNS or k in ("Filename", "Date&Time", "Eye", "File REF."):
                continue
            try:
                indices[k.strip()] = float(str(v).strip())
            except (TypeError, ValueError):
                pass
        out[exam] = {"eye": eye, "acquired_at": acquired, "indices": indices, "patient_key": patient_key,
                     "native_path_hint": os.path.dirname(fn)}
    return out


def patients_in_xref(xref: dict[str, dict[str, Any]]) -> list[str]:
    """Códigos de paciente (pseudônimos) distintos presentes num índice lido por read_xref."""
    return sorted({v["patient_key"] for v in xref.values() if v.get("patient_key")})


def detect_profile(directory: str) -> str:
    """Escolhe o perfil pelo padrão dos nomes: formato real do TMS-4N (HIT_1_53703.dat) ou pacote v0/sintético."""
    names = os.listdir(directory) if os.path.isdir(directory) else []
    if any(re.match(r"^(HIT|RAD)_\d+_\d+\.dat$", n, flags=re.IGNORECASE) for n in names):
        return "tms4_datatables_v1"
    return "tms4_datatables_v0"


def list_exams(directory: str, profile: str | dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Lista os exames presentes numa pasta exportada do TMS (normalmente um paciente, vários exames/olhos/capturas).

    Retorna dicionários {exam_id, eye, acquired_at, capture_index, patient_key, files, indices} ordenados por
    data/número. `patient_key` é o código pseudonimizado do XREF (None sem XREF); o número da captura conta
    por paciente e olho, de modo que uma pasta com duas exportações (dois pacientes) não mistura as contagens.
    """
    prof_name = profile or detect_profile(directory)
    prof = load_profile(prof_name) if isinstance(prof_name, str) else prof_name
    rx = re.compile(prof.get("exam_id_regex", "^(?P<exam>[^_]+)"))
    groups: dict[str, list[str]] = {}
    for name in sorted(os.listdir(directory)):
        m = rx.match(name)
        if m and os.path.isfile(os.path.join(directory, name)):
            groups.setdefault(m.group("exam"), []).append(name)
    xref: dict[str, dict[str, Any]] = {}
    xcfg = prof.get("xref", {})
    for cand in (xcfg.get("filename", "XREF.dat"), "xref_anon.csv"):
        p = os.path.join(directory, cand)
        if os.path.isfile(p):
            try:
                xref.update(read_xref(p))
            except Exception:  # noqa: BLE001 — índice opcional
                pass
    exams = []
    for exam, files in groups.items():
        info = xref.get(exam, {})
        exams.append({"exam_id": exam, "eye": info.get("eye"), "acquired_at": info.get("acquired_at"),
                      "patient_key": info.get("patient_key"), "files": files, "indices": info.get("indices", {})})
    exams.sort(key=lambda e: (e["acquired_at"] or "", int(e["exam_id"]) if e["exam_id"].isdigit() else 0))
    counters: dict[tuple[str, str], int] = {}
    for e in exams:
        k = (e["patient_key"] or "", e["eye"] or "?")
        counters[k] = counters.get(k, 0) + 1
        e["capture_index"] = counters[k]
    return exams


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


def _pseudonym(directory: str) -> str:
    """Código de estudo derivado do nome da pasta por hash (a pasta pode ter o nome do paciente; o código não)."""
    import hashlib
    base = os.path.basename(os.path.abspath(directory).rstrip("/\\"))
    return "P-" + hashlib.sha256(base.encode("utf-8")).hexdigest()[:8].upper()


def import_tms_bundle(directory: str, profile: str | dict[str, Any] | None = None,
                      manifest_override: Optional[dict[str, Any]] = None, exam_id: Optional[str] = None) -> NormalizedExam:
    """Importa um exame de uma pasta exportada do TMS e devolve o exame normalizado.

    - profile None → detecção automática (formato real do TMS-4N ou pacote v0/sintético).
    - exam_id → seleciona um exame numa pasta com vários (formato real: HIT_1_53703.dat, ...).
    - XREF.dat (formato real) fornece olho, data/hora e índices do TMS; nome e ID do paciente nunca são lidos.
    Lança AdapterError com causa e requisito em qualquer inconsistência essencial.
    """
    if not os.path.isdir(directory):
        raise AdapterError(f"diretório não encontrado: {directory}")
    prof_sel = profile if profile is not None else detect_profile(directory)
    prof = load_profile(prof_sel) if isinstance(prof_sel, str) else prof_sel
    roles_cfg = prof["roles"]
    n_angles = int(prof["n_angles"])
    n_rings_allowed = [int(v) for v in prof["n_rings_allowed"]]
    trim_padding = bool(prof.get("trim_padding_columns", False))
    rx_exam = re.compile(prof.get("exam_id_regex", "^(?P<exam>[^_]+)"))
    xref_name = prof.get("xref", {}).get("filename", "XREF.dat")

    # manifesto (opcional)
    manifest: dict[str, Any] = {}
    mpath = os.path.join(directory, prof.get("manifest_filename", "manifest.json"))
    if os.path.isfile(mpath):
        with open(mpath, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    if manifest_override:
        manifest.update(manifest_override)

    # vários exames na mesma pasta: exigir seleção explícita
    present_exams: set[str] = set()
    for name in os.listdir(directory):
        m = rx_exam.match(name)
        if m and os.path.isfile(os.path.join(directory, name)):
            present_exams.add(m.group("exam"))
    if exam_id is None and len(present_exams) > 1:
        raise AdapterError(f"a pasta contém {len(present_exams)} exames {sorted(present_exams)}; requisito: informar exam_id (use list_exams)")
    if exam_id is not None and present_exams and exam_id not in present_exams:
        raise AdapterError(f"exame {exam_id} não encontrado na pasta; presentes: {sorted(present_exams)}")

    # índice XREF (sem dados nominais)
    xref_info: dict[str, Any] = {}
    xref_path = None
    xref_patients: list[str] = []
    for cand in (xref_name, "xref_anon.csv"):
        p = os.path.join(directory, cand)
        if os.path.isfile(p):
            xref_path = p
            try:
                allx = read_xref(p)
            except Exception as e:  # noqa: BLE001
                raise AdapterError(f"{cand}: índice ilegível ({e}); requisito: exportação íntegra")
            sel = exam_id or (next(iter(present_exams)) if len(present_exams) == 1 else None)
            if sel and sel in allx:
                xref_info = allx[sel]
            xref_patients = patients_in_xref({k: v for k, v in allx.items() if k in present_exams} or allx)
            break

    files: list[SourceFile] = []
    tables: dict[str, tuple[RawTable, str]] = {}
    pupil = PupilData(state="ausente")
    pup_path: Optional[str] = None
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
        if xref_path and os.path.abspath(path) == os.path.abspath(xref_path):
            files.append(SourceFile(path=path, role="XREF", sha256=sha256_file(path), size_bytes=os.path.getsize(path)))
            continue
        m = rx_exam.match(name)
        if exam_id is not None and (not m or m.group("exam") != exam_id):
            continue      # arquivo de outro exame da mesma pasta
        role = _match_role(name, roles_cfg)
        if role is None:
            if m is None:
                files.append(SourceFile(path=path, role="OUTRO", sha256=sha256_file(path), size_bytes=os.path.getsize(path)))
                diagnostics.append(f"arquivo ignorado (papel não reconhecido): {name}")
            continue
        if role in tables or (role == "PUP" and pup_path is not None):
            raise AdapterError(f"{name}: papel {role} duplicado no pacote; requisito: um arquivo por tabela e por exame")
        eye = _match_eye(name, prof.get("eye_patterns", {}))
        if eye:
            eyes.add(eye)
        if m:
            exam_ids.add(m.group("exam"))
        sf = SourceFile(path=path, role=role, sha256=sha256_file(path), size_bytes=os.path.getsize(path))
        fmt = roles_cfg[role].get("format")
        if role == "PUP":
            pup_path = path
            if fmt != "radii_per_angle":
                pupil = read_pupil_file(path, roles_cfg["PUP"])
            files.append(sf)
            continue
        if fmt == "ring_summary":
            files.append(sf)      # AVE: só proveniência
            continue
        raw = read_table_text(path, prof)
        sf.n_rows, sf.n_cols, sf.delimiter, sf.decimal = len(raw.rows), len(raw.rows[0]), raw.delimiter, raw.decimal
        tables[role] = (raw, name)
        files.append(sf)

    # papéis obrigatórios
    for role, rule in roles_cfg.items():
        if rule.get("required") and role not in tables:
            raise AdapterError(f"tabela obrigatória ausente: {role}; requisito: exportar {role} (Data Tables) do mesmo exame")

    # lateralidade e exame: manifesto > XREF > nome de arquivo; conflito bloqueia
    eye_manifest = manifest.get("eye") or xref_info.get("eye")
    if eye_manifest and eyes and eye_manifest not in eyes:
        raise AdapterError(f"conflito de lateralidade: manifesto/XREF {eye_manifest} vs nomes {sorted(eyes)}")
    if len(eyes) > 1:
        raise AdapterError(f"mistura de olhos no pacote: {sorted(eyes)}; requisito: um olho por pacote")
    if len(exam_ids) > 1:
        raise AdapterError(f"mistura de exames no pacote: {sorted(exam_ids)}; requisito: um exame por pacote")
    eye = eye_manifest or (next(iter(eyes)) if eyes else None)
    if eye == "OE":
        eye = "OS"
    exam_id_final = manifest.get("exam_id") or exam_id or (next(iter(exam_ids)) if exam_ids else os.path.basename(os.path.abspath(directory)))
    if manifest.get("exam_id") and exam_ids and manifest["exam_id"] not in exam_ids:
        raise AdapterError(f"conflito de identificador de exame: manifesto {manifest['exam_id']} vs nomes {sorted(exam_ids)}")
    if eye is None:
        warnings.append("lateralidade não informada (nem manifesto, nem XREF, nem nome de arquivo)")

    # decodificação
    fields: dict[str, FieldTable] = {}
    orientations: dict[str, str] = {}
    shapes: set[tuple[int, int]] = set()
    for role, (raw, name) in tables.items():
        ft, orient = decode_field(raw, role, roles_cfg[role], n_angles, n_rings_allowed, prof.get("orientation", "auto"), trim_padding)
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
        transform_verified=(prof.get("status", "").startswith("verificado")) or bool(manifest.get("synthetic")),
        notes=prof.get("angle_note", ""),
    )
    k = np.arange(n_angles)
    theta = frame.angle_start_deg + frame.angle_direction * k * (360.0 / n_angles)
    if frame.mirror_x:
        theta = 180.0 - theta
    theta = np.mod(theta, 360.0)
    if pup_path is not None and roles_cfg["PUP"].get("format") == "radii_per_angle":
        pupil = read_pupil_radii(pup_path, roles_cfg["PUP"], n_angles, theta)

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
    acquired = manifest.get("acquired_at") or xref_info.get("acquired_at")
    if manifest.get("acquired_at"):
        precision = str(manifest.get("date_precision", "dia"))
    elif xref_info.get("acquired_at"):
        precision = "segundo"
    else:
        precision = "desconhecida"
    capture_index = manifest.get("capture_index")
    if capture_index is None and xref_info and exam_id is not None:
        try:
            for e in list_exams(directory, prof):
                if e["exam_id"] == exam_id:
                    capture_index = e["capture_index"]
        except Exception:  # noqa: BLE001
            capture_index = None
    if xref_info and str(manifest.get("device", {}).get("model", "")) in ("", "desconhecido"):
        # o XREF não traz modelo; o cone de 25 anéis e o formato identificam a família TMS-4/4N
        device.model = manifest.get("device", {}).get("model", "TMS-4N (presumido pelo formato)")
        device.cone = device.cone or f"{n_rings} anéis"
    if manifest.get("study_id"):
        study_id, study_source = str(manifest["study_id"]), "manifesto"
    elif xref_info.get("patient_key"):
        study_id, study_source = str(xref_info["patient_key"]), "XREF (hash do nome/ID; o nome não é armazenado)"
    elif xref_info or exam_id:
        study_id, study_source = _pseudonym(directory), "hash do nome da pasta"
    else:
        study_id, study_source = "sem_estudo", "nenhuma"
    identity = ExamIdentity(
        study_id=study_id,
        exam_id=str(exam_id_final),
        eye=eye,
        acquired_at=acquired,
        date_precision=precision,
        capture_index=capture_index,
    )
    export_settings = {"orientations": orientations, **manifest.get("export", {})}
    if xref_info:
        export_settings["tms_indices"] = xref_info.get("indices", {})
        export_settings["tms_native_folder"] = xref_info.get("native_path_hint")
    if xref_patients:
        export_settings["patients_in_folder"] = len(xref_patients)
        if len(xref_patients) > 1:
            warnings.append(
                f"a pasta contém exames de {len(xref_patients)} pacientes diferentes (XREF): este exame recebeu o código "
                f"{study_id}; o TMS acrescenta linhas ao XREF.dat quando se exporta de novo para a mesma pasta — "
                "recomenda-se uma pasta nova por paciente"
            )
    bundle = SourceBundle(
        directory=os.path.abspath(directory), files=files, adapter_profile=prof["profile_id"],
        adapter_version=prof.get("adapter_version", "0"), export_settings=export_settings,
        manifest=manifest, synthetic=bool(manifest.get("synthetic", False)),
    )
    status = str(prof.get("status", ""))
    if not status.startswith("verificado") and not bundle.synthetic:
        warnings.append(f"perfil de exportação '{prof['profile_id']}' NÃO verificado com arquivos reais — compatibilidade não declarada")
    elif status == "verificado_estrutura" and not bundle.synthetic:
        warnings.append("perfil verificado na estrutura (unidades, marcação, precisão); orientação angular verificada por ensaio de eixo do SimK")
    if not manifest and not xref_info:
        warnings.append("manifesto e XREF ausentes — modelo, versão de software e data não informados")
    if xref_info and xref_info.get("indices"):
        ix = xref_info["indices"]
        diagnostics.append(
            f"XREF do TMS: SimK1 {ix.get('SimK1', float('nan')):.2f} D @ {ix.get('SimK1 Ang.', float('nan')):.0f}°, "
            f"SimK2 {ix.get('SimK2', float('nan')):.2f} D @ {ix.get('SimK2 Ang.', float('nan')):.0f}°, CYL {ix.get('CYL', float('nan')):.2f} D, "
            f"SRI {ix.get('SRI', float('nan')):.2f}, SAI {ix.get('SAI', float('nan')):.2f}, Q {ix.get('Q', float('nan')):+.2f}"
        )
    diagnostics.append(f"código de estudo {study_id} — origem: {study_source}")
    diagnostics.append(f"tabelas: {', '.join(f'{k} ({orientations[k]})' for k in fields)}; anéis={n_rings}, ângulos={n_angles}")
    diagnostics.append(f"pontos medidos {int(meas.sum())}, interpolados {int((st == PointState.INTERPOLATED).sum())}, ausentes {int((st == PointState.MISSING).sum())}")
    diagnostics.append(f"pupila: {pupil.state}" + (f" centro ({pupil.center_x:+.2f}, {pupil.center_y:+.2f}) mm, referencial {pupil.frame}" if pupil.state == "registrada" else ""))
    return NormalizedExam(bundle=bundle, identity=identity, device=device, frame=frame, samples=samples, pupil=pupil,
                          diagnostics=diagnostics, warnings=warnings)
