"""
streamlit_app.py — Top-level entrypoint alias redirecting to app.py
Ensures compatibility whether Streamlit Cloud is configured with 'streamlit_app.py' or 'app.py'.
"""
import os
import sys
from pathlib import Path

# On Windows, ensure PyTorch's bundled DLLs are located
if sys.platform == "win32":
    try:
        import site
        for sp in site.getsitepackages():
            torch_lib = os.path.join(sp, "torch", "lib")
            if os.path.exists(torch_lib):
                os.add_dll_directory(torch_lib)
    except Exception:
        pass
    try:
        import torch
    except Exception:
        pass

import runpy

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

runpy.run_path(str(_DIR / "app.py"), run_name="__main__")
