@echo off
REM Compila o programa para Windows (pasta dist\AberrometriaTMS com AberrometriaTMS.exe e tmsab.exe)
REM Requisitos: Python 3.11+ instalado (python.org), com "py" no PATH. Rodar na raiz do repositório.
setlocal
cd /d "%~dp0\.."
if not exist .venv (
  py -3 -m venv .venv || goto :erro
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip wheel >nul
python -m pip install -e ".[gui,dev]" || goto :erro
echo === Testes rapidos ===
python -m tmsab.cli verify || goto :erro
echo === Smoke test da interface ===
set QT_QPA_PLATFORM=offscreen
python -m tmsab.gui.app --smoke || goto :erro
set QT_QPA_PLATFORM=
echo === PyInstaller ===
pyinstaller packaging\AberrometriaTMS.spec --noconfirm --clean || goto :erro
echo.
echo Pronto: dist\AberrometriaTMS\AberrometriaTMS.exe
echo Para instalar: copie a pasta dist\AberrometriaTMS para "C:\Program Files\AberrometriaTMS" e crie um atalho para AberrometriaTMS.exe
endlocal
exit /b 0
:erro
echo FALHA na compilacao (veja as mensagens acima)
endlocal
exit /b 1
