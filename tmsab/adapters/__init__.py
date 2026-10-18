"""Adaptadores de origem (plano §3): leitura estrita por perfil, pareamento e normalização."""
from .tms4 import (  # noqa: F401
    AdapterError,
    detect_profile,
    import_tms_bundle,
    list_exams,
    list_profiles,
    load_profile,
    patients_in_xref,
    read_xref,
)
