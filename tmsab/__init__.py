"""tmsab — Aberrometria corneana anterior calculada a partir de exportações do Tomey TMS.

Ferramenta de pesquisa. Não realiza aquisição, não controla o instrumento e não
emite alegações diagnósticas. Ver docs/limitacoes.md.
"""

__version__ = "0.2.1"
CORE_VERSION = __version__

def utf8_console() -> None:
    """Força UTF-8 (com substituição) em stdout/stderr — consoles Windows usam cp1252 e quebram em 'µ', '≡', '—'."""
    import sys
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                pass
