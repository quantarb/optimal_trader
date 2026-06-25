from __future__ import annotations

import os
from pathlib import Path
import runpy


os.environ["MOE_STREAMLIT_EMBEDDED_PAGE"] = "1"
runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "moe" / "streamlit_moe_paper_trading.py"),
    run_name="__main__",
)
