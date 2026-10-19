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
        # abas "Topografia (TMS)" (2ª) e "Ceratometria" (3ª): tabela por zona de analysis.keratometry; pacote v0 sem XREF -> sem SimK do TMS
        nomes = [win.tabs.tabText(i) for i in range(win.tabs.count())]
        assert nomes[1] == "Topografia (TMS)" and nomes[2] == "Ceratometria", f"abas fora de ordem: {nomes}"
        assert len(win.cv_topo.figure.axes) >= 1, "aba de topografia não desenhada"
        win.vg_kind.buttons()[1].click(); app.processEvents()          # botão "Tangencial" redesenha
        assert "Astig Stats" in win.topo_text.toPlainText()
        kres = st0.analysis.keratometry
        assert kres is not None, "ceratometria ausente na análise"
        rotulos = [win.k_table.item(r, 0).text() for r in range(win.k_table.rowCount()) if win.k_table.item(r, 0) is not None]
        assert win.k_table.rowCount() >= len(kres.zonas) + len(kres.aneis), f"tabela de ceratometria com {win.k_table.rowCount()} linhas"
        assert any(r.startswith("Ø 3 mm") for r in rotulos) and any(r.startswith("anel 1 ") for r in rotulos), rotulos
        assert win.k_bn["pupila"].val.text() != "—", f"K na zona pupilar vazio: {win.k_bn['pupila'].sub.text()}"
        tsv = win.keratometry_tsv()
        assert tsv.splitlines()[1].split("\t")[:3] == ["Zona", "K1 plano (D)", "eixo K1"], tsv[:200]
        if app.platformName() == "offscreen":   # área de transferência só sem tela (não mexe na do usuário)
            assert win.copy_keratometry_tsv() == tsv and QApplication.clipboard().text() == tsv, "cópia TSV falhou"
        # controle de diâmetro (sub-aba do mapa axial): no Ø escolhido, os mesmos textos da linha da tabela
        subabas = [win.k_subtabs.tabText(i) for i in range(win.k_subtabs.count())]
        assert subabas == ["Mapa axial — diâmetro", "Gráficos por diâmetro"], subabas
        assert win.k_slider.isEnabled() and win.k_spin.isEnabled(), "controle de diâmetro desabilitado"
        from PySide6.QtGui import QTextDocument

        def _leitura() -> str:
            doc = QTextDocument()
            doc.setHtml(win.k_leitura.text())
            return doc.toPlainText().replace("\xa0", " ")

        def _textos_da_linha(rotulo: str) -> list[str]:          # K1, eixo K1, K2, eixo K2, ΔK, Km na tela
            for r in range(win.k_table.rowCount()):
                it = win.k_table.item(r, 0)
                if it is not None and it.text().rstrip(" †*") == rotulo:
                    return [win.k_table.item(r, c).text() for c in range(1, 7)]
            raise AssertionError(f"linha {rotulo} ausente na tabela")

        for ajustar, rotulo in ((lambda: win.k_slider.setValue(30), "Ø 3 mm"), (lambda: win.k_spin.setValue(2.4), "Ø 2.4 mm")):
            ajustar()
            app.processEvents()
            t = _textos_da_linha(rotulo)
            esperado = f"K1 plano {t[0]} D @ {t[1]} · K2 curvo {t[2]} D @ {t[3]} · ΔK {t[4]} D · Km {t[5]} D"
            primeira = _leitura().splitlines()[0]
            assert primeira.startswith(rotulo) and esperado in primeira, f"{rotulo}: {primeira!r} ≠ tabela {t}"
        assert abs(win.k_spin.value() - 2.4) < 1e-9 and win.k_slider.value() == 24
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
        assert win.k_table.rowCount() == 0 and win.k_bn["pupila"].val.text() == "—", "ceratometria antiga ainda na tela"
        assert not win.k_slider.isEnabled() and not win.k_spin.isEnabled() and win.k_leitura.text() == "", \
            "controle de diâmetro com números antigos"
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
        aba_wf = next(i for i in range(win.tabs.count()) if win.tabs.tabText(i) == "Frente de onda")   # pelo nome: a ordem das abas muda
        win.tabs.setCurrentIndex(aba_wf)
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
            win.tabs.setCurrentIndex(aba_wf)
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
