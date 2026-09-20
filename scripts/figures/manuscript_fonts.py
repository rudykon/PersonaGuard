"""Use the ACM manuscript's Libertine faces in selected figure renderers.

Resolve the same OpenType files used by TeX Live rather than substituting a
similar system font. The context covers both text and math through export,
and restores the caller's settings when the figure is finished.
"""

from __future__ import annotations

from functools import lru_cache, wraps
from pathlib import Path
import subprocess

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font


FONT_FAMILY = "Linux Libertine O"
FONT_FILES = {
    "regular": "LinLibertine_R.otf",
    "bold": "LinLibertine_RB.otf",
    "italic": "LinLibertine_RI.otf",
    "bold_italic": "LinLibertine_RBI.otf",
}


@lru_cache(maxsize=1)
def font_paths() -> dict[str, Path]:
    """Locate and register the manuscript's four real font faces."""
    paths = {}
    for variant, filename in FONT_FILES.items():
        try:
            resolved = subprocess.check_output(
                ["kpsewhich", filename], text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                f"Install TeX Live's Libertine fonts to render this figure: {filename}"
            ) from exc
        path = Path(resolved)
        if not resolved or not path.is_file():
            raise RuntimeError(f"Required manuscript font unavailable: {filename}")
        if FT2Font(str(path)).family_name != FONT_FAMILY:
            raise RuntimeError(f"Unexpected manuscript font family: {filename}")
        font_manager.fontManager.addfont(str(path))
        paths[variant] = path
    return paths


def font_settings() -> dict[str, object]:
    """Keep axes, legends, annotations, and math labels in the body family."""
    font_paths()
    return {
        "font.family": [FONT_FAMILY],
        "font.serif": [FONT_FAMILY],
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT_FAMILY,
        "mathtext.it": f"{FONT_FAMILY}:style=italic",
        "mathtext.bf": f"{FONT_FAMILY}:weight=bold",
        "mathtext.bfit": f"{FONT_FAMILY}:style=italic:weight=bold",
        "mathtext.sf": FONT_FAMILY,
        "mathtext.tt": FONT_FAMILY,
        "mathtext.cal": FONT_FAMILY,
        "mathtext.fallback": None,
    }


def with_manuscript_fonts(render):
    """Apply body typography only during this figure's creation and export."""
    @wraps(render)
    def wrapped(*args, **kwargs):
        with mpl.rc_context(font_settings()):
            return render(*args, **kwargs)
    return wrapped
