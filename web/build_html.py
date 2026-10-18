"""Gera o arquivo único AberrometriaTMS-<versão>.html (versão navegador).

Uso (na raiz do repositório):  python web/build_html.py [--out dist]
O HTML embute o pacote `tmsab` (zip em base64, sem a interface Qt) e carrega o Pyodide + NumPy/SciPy/Matplotlib
do CDN jsdelivr na primeira abertura (depois fica em cache do navegador). Nenhum dado sai do computador.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYODIDE_VERSION = "0.29.5"            # Python 3.13, numpy 2.2, scipy 1.14, matplotlib 3.8
PYODIDE_URL = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full/"


def package_zip() -> bytes:
    """Zip do pacote tmsab sem a GUI (PySide6) e sem caches."""
    buf = io.BytesIO()
    pkg = os.path.join(ROOT, "tmsab")
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for dp, dns, fns in os.walk(pkg):
            dns[:] = [d for d in dns if d not in ("__pycache__", "gui")]
            for fn in sorted(fns):
                if fn.endswith((".py", ".json")) and fn != "cli.py":
                    full = os.path.join(dp, fn)
                    z.write(full, os.path.relpath(full, ROOT))
    return buf.getvalue()


def build(out_dir: str) -> str:
    sys.path.insert(0, ROOT)
    from tmsab import __version__
    from tmsab.web import REPORT_CSS
    tpl = open(os.path.join(ROOT, "web", "template.html"), encoding="utf-8").read()
    zb = package_zip()
    html = (tpl.replace("{{VERSION}}", __version__)
               .replace("{{PYODIDE_URL}}", PYODIDE_URL)
               .replace("{{REPORT_CSS_JSON}}", json.dumps(REPORT_CSS))
               .replace("{{REPORT_CSS}}", REPORT_CSS)
               .replace("{{TMSAB_ZIP_B64}}", base64.b64encode(zb).decode("ascii")))
    assert not re.search(r"\{\{[A-Z_]+\}\}", html), "marcador não substituído"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"AberrometriaTMS-{__version__}.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"{path}: {os.path.getsize(path)/1024:.0f} KB (pacote tmsab {len(zb)/1024:.0f} KB; Pyodide {PYODIDE_VERSION} via CDN)")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "dist"))
    a = ap.parse_args()
    build(a.out)
