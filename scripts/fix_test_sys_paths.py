"""
scripts/fix_test_sys_paths.py
"""
from pathlib import Path

files = [
    "tests/test_forecast_table.py",
    "tests/test_freshness_and_stale_data.py",
    "tests/test_geolocation_and_live_stations.py",
    "tests/test_greeting_and_aqi_card.py",
]

prefix = """import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
"""

for fpath in files:
    p = Path(fpath)
    content = p.read_text(encoding="utf-8")
    if "ROOT_DIR" not in content:
        # insert after initial docstring
        if '"""' in content:
            second_quote = content.find('"""', content.find('"""') + 3)
            if second_quote != -1:
                insert_pos = second_quote + 3
                new_content = content[:insert_pos] + "\n\n" + prefix + content[insert_pos:]
                p.write_text(new_content, encoding="utf-8")
                print(f"Added sys.path to {fpath}")
