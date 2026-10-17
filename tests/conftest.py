import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tmsab.synthetic import SyntheticSpec, write_bundle  # noqa: E402


@pytest.fixture(scope="session")
def synth_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    specs = {
        "con": SyntheticSpec(exam_id="T-CON", R=7.8, Q=-0.2, decimals=6),
        "coma": SyntheticSpec(exam_id="T-COMA", R=7.8, Q=-0.2, zernike_um={"3,1": 3.0}, decimals=6),
        "tref": SyntheticSpec(exam_id="T-TREF", R=7.8, Q=-0.2, zernike_um={"3,-3": 2.0}, decimals=6),
        "tor": SyntheticSpec(exam_id="T-TOR", R=7.8, Q=-0.2, cyl_D=2.0, axis_deg=0.0, decimals=6),
        "pup": SyntheticSpec(exam_id="T-PUP", R=7.8, Q=-0.2, pupil_center=(0.3, -0.2), decimals=6),
        "nopup": SyntheticSpec(exam_id="T-NOPUP", R=7.8, Q=-0.2, pupil_registered=False, decimals=6),
        "ring": SyntheticSpec(exam_id="T-RING", R=7.8, Q=-0.2, orientation="ring_rows", delimiter=";", decimals=6),
        "gapmiss": SyntheticSpec(exam_id="T-GAPM", R=7.8, Q=-0.2, lid_gap_from_ring=10, lid_gap_half_angle_deg=40, lid_gap_as_interpolated=False, decimals=6),
        "gapint": SyntheticSpec(exam_id="T-GAPI", R=7.8, Q=-0.2, lid_gap_from_ring=16, lid_gap_half_angle_deg=30, decimals=6),
        "coarse": SyntheticSpec(exam_id="T-COARSE", R=7.8, Q=-0.2, decimals=2),
        "os": SyntheticSpec(exam_id="T-TOR", R=7.8, Q=-0.2, cyl_D=2.0, axis_deg=0.0, eye="OS", decimals=6),
    }
    paths = {k: write_bundle(s, str(root)) for k, s in specs.items()}
    return {"root": str(root), "paths": paths, "specs": specs}
