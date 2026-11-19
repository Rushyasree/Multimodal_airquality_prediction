"""
streamlit_app.py — Top-level entrypoint alias redirecting to app.py
Ensures compatibility whether Streamlit Cloud is configured with 'streamlit_app.py' or 'app.py'.
"""
import runpy
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

runpy.run_path(str(_DIR / "app.py"), run_name="__main__")
