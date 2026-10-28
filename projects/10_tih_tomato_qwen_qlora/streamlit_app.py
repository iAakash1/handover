"""PlantDx — leaf-disease inference demo (Streamlit entry point).

Run with:

    streamlit run streamlit_app.py

A presentation-only layer over the trained PlantDx adapters. It reuses the
existing ``plantdx`` inference/evaluation code unchanged and never trains,
evaluates, or regenerates anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make the `plantdx` package importable even without an editable install.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

st.set_page_config(
    page_title="PlantDx — Leaf Disease Demo",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    """Wire the page together. All heavy lifting lives in ``app/``."""
    from app.components import header, inject_css
    from app.storage import ensure_dirs
    from app.ui import render_main, render_sidebar

    inject_css()
    ensure_dirs()
    header()

    try:
        settings = render_sidebar()
        render_main(settings)
    except (KeyboardInterrupt, SystemExit):  # never swallow these
        raise
    except BaseException as exc:  # last-resort guard: the app must never hard-crash
        # A broken numba/NumPy C extension in the launching interpreter can raise
        # errors that a plain `except Exception` misses; catch everything here so
        # the user always sees a friendly page, never a raw traceback.
        st.error(f"Something went wrong: {exc}", icon="😵")
        st.caption("This is a friendly catch-all — check the terminal for details.")


main()
