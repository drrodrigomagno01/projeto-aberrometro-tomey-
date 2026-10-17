# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller — AberrometriaTMS (GUI) + tmsab (CLI) em uma única pasta de distribuição.

Uso (na raiz do repositório):  pyinstaller packaging/AberrometriaTMS.spec --noconfirm
Resultado: dist/AberrometriaTMS/  (AberrometriaTMS.exe, tmsab.exe e dependências)
"""
import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
sys.path.insert(0, ROOT)

datas = collect_data_files("tmsab", includes=["adapters/profiles/*.json"])
datas += collect_data_files("matplotlib", includes=["mpl-data/**"])

hiddenimports = (
    ["matplotlib.backends.backend_qtagg", "matplotlib.backends.backend_pdf", "matplotlib.backends.backend_agg",
     "scipy.special._cdflib", "scipy._lib.messagestream", "PySide6.QtSvg"]
    + collect_submodules("tmsab")
    + collect_submodules("scipy.interpolate")
    + collect_submodules("scipy.optimize")
    + collect_submodules("scipy.stats")
)

excludes = [
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick", "PySide6.QtQml", "PySide6.QtQuick",
    "PySide6.QtQuick3D", "PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets", "PySide6.QtCharts", "PySide6.QtDataVisualization",
    "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.QtBluetooth", "PySide6.QtNfc", "PySide6.QtPositioning", "PySide6.QtLocation",
    "PySide6.QtSensors", "PySide6.QtSerialPort", "PySide6.QtRemoteObjects", "PySide6.QtPdf", "PySide6.QtPdfWidgets",
    "PySide6.QtTest", "PySide6.QtDesigner", "PySide6.QtHelp", "PySide6.QtSql", "PySide6.QtNetworkAuth", "PySide6.QtWebSockets",
    "PySide6.QtWebChannel", "PySide6.QtOpenGL", "PySide6.QtOpenGLWidgets", "PySide6.QtSpatialAudio", "PySide6.QtTextToSpeech",
    "tkinter", "IPython", "jupyter", "notebook", "pytest", "PyQt5", "PyQt6", "pandas", "pyarrow", "numba", "sympy", "PIL.ImageQt",
]

block_cipher = None

a_gui = Analysis(
    [os.path.join(SPECPATH, "launch_gui.py")],
    pathex=[ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
a_cli = Analysis(
    [os.path.join(SPECPATH, "launch_cli.py")],
    pathex=[ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
MERGE((a_gui, "AberrometriaTMS", "AberrometriaTMS"), (a_cli, "tmsab", "tmsab"))

pyz_gui = PYZ(a_gui.pure, a_gui.zipped_data, cipher=block_cipher)
exe_gui = EXE(
    pyz_gui, a_gui.scripts, [],
    exclude_binaries=True, name="AberrometriaTMS", debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
    console=False, icon=os.path.join(SPECPATH, "icon.ico") if os.path.exists(os.path.join(SPECPATH, "icon.ico")) else None,
)
pyz_cli = PYZ(a_cli.pure, a_cli.zipped_data, cipher=block_cipher)
exe_cli = EXE(
    pyz_cli, a_cli.scripts, [],
    exclude_binaries=True, name="tmsab", debug=False, bootloader_ignore_signals=False, strip=False, upx=False, console=True,
)
coll = COLLECT(
    exe_gui, a_gui.binaries, a_gui.zipfiles, a_gui.datas,
    exe_cli, a_cli.binaries, a_cli.zipfiles, a_cli.datas,
    strip=False, upx=False, name="AberrometriaTMS",
)
