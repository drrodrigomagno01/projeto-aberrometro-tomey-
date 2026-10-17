"""Ponto de entrada do executável gráfico (PyInstaller)."""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from tmsab.gui.app import main
    sys.exit(main())
