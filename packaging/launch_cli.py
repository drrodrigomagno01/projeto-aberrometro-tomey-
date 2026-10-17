"""Ponto de entrada do executável de linha de comando (PyInstaller)."""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from tmsab.cli import main
    sys.exit(main())
