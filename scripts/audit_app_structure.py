"""
scripts/audit_app_structure.py
Inspects existing pages, services, models, and dependencies in app.py.
"""

from pathlib import Path
import re

app_path = Path("app.py")
text = app_path.read_text(encoding="utf-8")

print(f"Total lines in app.py: {len(text.splitlines())}")

# Find page routes
for i, line in enumerate(text.splitlines()):
    if 'st.session_state["nav_tab"]' in line and ("==" in line or "in" in line):
        print(f"Line {i+1}: {line.strip()[:100]}")

print("\n--- Imports in app.py related to prediction or services ---")
for line in text.splitlines():
    if "import" in line and any(k in line for k in ["prediction", "model", "service", "estimator", "forecaster"]):
        print(line.strip())
