"""Ponto de entrada do programa desktop.

  tmsab-gui                       abre a janela
  tmsab-gui --open DIR            abre já com um pacote carregado
  tmsab-gui --smoke [--synth-dir] teste de fumaça (CI): carrega sintéticos, calcula, popula abas, exporta, sai com 0
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import traceback

os.environ.setdefault("QT_API", "pyside6")

from PySide6.QtWidgets import QApplication  # noqa: E402

SMOKE_CASES = ("SYN-CON_OD", "SYN-PUP_OD")


def _ensure_smoke_packages(synth_dir: str | None) -> list[str]:
    if synth_dir and all(os.path.isdir(os.path.join(synth_dir, c)) for c in SMOKE_CASES):
        return [os.path.join(synth_dir, c) for c in SMOKE_CASES]
    from ..synthetic import standard_cases, write_bundle
    out = tempfile.mkdtemp(prefix="tmsab_smoke_")
    dirs = []
    for spec in standard_cases():
        if f"{spec.exam_id}_{spec.eye}" in SMOKE_CASES:
            dirs.append(write_bundle(spec, out))
    return dirs


def run_smoke(app: QApplication, win, synth_dir: str | None, screenshot: str | None) -> int:
    from ..pipeline import default_profiles
    try:
        dirs = _ensure_smoke_packages(synth_dir)
        keys = [win.load_package(d) for d in dirs]
        app.processEvents()
        win.compute_sync(keys, default_profiles())
        app.processEvents()
        for key in keys:
            st = win.packages[key]
            assert st.analysis is not None, f"análise ausente para {key}: {st.message}"
        # selecionar o primeiro e percorrer todas as abas/perfis
        win.list.setCurrentRow(0)
        app.processEvents()
        st0 = win.packages[keys[0]]
        assert win.profile_combo.count() == len(st0.analysis.wavefronts)
        for i in range(win.profile_combo.count()):
            win.profile_combo.setCurrentIndex(i)
            app.processEvents()
        assert win.z_table.rowCount() > 0, "tabela de Zernike vazia"
        assert "analysis_id" in win.tech_text.toPlainText()
        # comparação A × B no primeiro perfil disponível
        win.refresh_compare_controls()
        win.cmp_a.setCurrentIndex(win.cmp_a.findData(keys[0]))
        win.cmp_b.setCurrentIndex(win.cmp_b.findData(keys[1]))
        win.cmp_profile.setCurrentIndex(0)
        win.run_compare()
        app.processEvents()
        assert win.cmp_table.rowCount() > 0, f"comparação sem tabela: {win.cmp_msg.text()}"
        # exportação
        out = tempfile.mkdtemp(prefix="tmsab_smoke_out_")
        paths = win.export_package(st0, out)
        for p in paths.values():
            assert os.path.isfile(p) and os.path.getsize(p) > 0, f"exportação vazia: {p}"
        # página "Resultado completo": canvas preenchido e salvamento em PDF
        win.list.setCurrentRow(0); app.processEvents()
        assert len(win.cv_summary.figure.axes) >= 6, "página de resultado completo não montada"
        sp = win.save_summary()
        assert sp and os.path.isfile(sp) and os.path.getsize(sp) > 10_000, "resultado completo não salvo"
        # parâmetros alterados -> desatualizado, sem números antigos
        win.order_spin.setValue(7)
        app.processEvents()
        assert win.packages[keys[0]].analysis is None and win.packages[keys[0]].status == "desatualizado"
        assert win.bn["sa"].val.text() == "—"
        win.order_spin.setValue(6)
        # caminho com thread (worker + sinais), com espera limitada
        from PySide6.QtCore import QEventLoop, QTimer
        loop = QEventLoop()
        done = {"cancelled": None}

        def _finished(cancelled):
            done["cancelled"] = cancelled
            loop.quit()

        win.start_compute()
        assert win.worker is not None, "worker não iniciado"
        win.worker.finished_all.connect(_finished)
        QTimer.singleShot(300_000, loop.quit)
        loop.exec()
        assert done["cancelled"] is False, "worker não concluiu no tempo limite"
        app.processEvents()
        for key in keys:
            assert win.packages[key].status == "calculado", f"{key}: {win.packages[key].status} {win.packages[key].message}"
        win.list.setCurrentRow(0)
        win.tabs.setCurrentIndex(2)
        win.profile_combo.setCurrentIndex(min(1, win.profile_combo.count() - 1))
        app.processEvents()
        win.resize(1366, 768)
        win.show()
        app.processEvents()
        if screenshot:
            win.grab().save(screenshot)
            print(f"screenshot: {screenshot}")
            base, ext = os.path.splitext(screenshot)
            for i in range(win.tabs.count()):
                win.tabs.setCurrentIndex(i)
                app.processEvents()
                win.grab().save(f"{base}_tab{i}{ext}")
            win.tabs.setCurrentIndex(2)
            app.processEvents()
        print(f"smoke OK — {len(keys)} pacotes, {len(st0.analysis.wavefronts)} perfis, exportação em {out}")
        return 0
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tmsab-gui", description="Aberrometria corneana anterior calculada — programa desktop")
    parser.add_argument("--open", help="pacote a abrir ao iniciar")
    parser.add_argument("--profile", default="auto", help="perfil do adaptador (auto = detectar pelo formato)")
    parser.add_argument("--smoke", action="store_true", help="teste de fumaça (sem interação)")
    parser.add_argument("--synth-dir", help="pasta com pacotes sintéticos para o teste de fumaça")
    parser.add_argument("--screenshot", help="salva uma captura da janela no teste de fumaça")
    args = parser.parse_args(argv)
    from .. import utf8_console
    utf8_console()

    app = QApplication.instance() or QApplication(sys.argv[:1])
    from .main_window import MainWindow
    win = MainWindow(adapter_profile=args.profile, interactive=not args.smoke)
    if args.smoke:
        rc = run_smoke(app, win, args.synth_dir, args.screenshot)
        win.close()
        return rc
    if args.open:
        win.open_export_folder(args.open)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
