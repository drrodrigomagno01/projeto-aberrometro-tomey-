#!/usr/bin/env bash
# Build de validação (Linux/macOS): gera dist/AberrometriaTMS com os mesmos artefatos do Windows.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m pip install --break-system-packages -q -e ".[gui,dev]"
python3 -m tmsab.cli verify
QT_QPA_PLATFORM=offscreen python3 -m tmsab.gui.app --smoke
pyinstaller packaging/AberrometriaTMS.spec --noconfirm --clean
echo "Pronto: dist/AberrometriaTMS/"
