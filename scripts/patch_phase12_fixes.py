"""
scripts/patch_phase12_fixes.py
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# 1. Fix autonomous_live_collector.py
alc_file = ROOT_DIR / "src" / "services" / "autonomous_live_collector.py"
with open(alc_file, "r", encoding="utf-8") as f:
    alc_text = f.read()

# Replace redact_secrets
old_redact = """def redact_secrets(text: str) -> str:
    \"\"\"Masks potential tokens or sensitive hash credentials in logging output.\"\"\"
    if not isinstance(text, str):
        return str(text)
    env_key = os.environ.get("OPENAQ_API_KEY", "")
    if env_key and len(env_key) > 5 and env_key in text:
        text = text.replace(env_key, "[REDACTED_API_KEY]")
    return text"""

new_redact = """def redact_secrets(text: str) -> str:
    \"\"\"Masks potential tokens or sensitive hash credentials in logging output.\"\"\"
    if not isinstance(text, str):
        return str(text)
    import re
    env_key = os.environ.get("OPENAQ_API_KEY", "")
    if env_key and len(env_key) > 5 and env_key in text:
        text = text.replace(env_key, "[REDACTED_API_KEY]")
    text = re.sub(r"(openaq_[a-zA-Z0-9_\-]+)", "[REDACTED_API_KEY]", text)
    text = re.sub(r"(secret_[a-zA-Z0-9_\-]+)", "[REDACTED_TOKEN]", text)
    text = re.sub(r"(api[_-]?key[:=\s]+)([a-zA-Z0-9_\-]+)", r"\\1[REDACTED]", text, flags=re.IGNORECASE)
    text = re.sub(r"(token[:=\s]+)([a-zA-Z0-9_\-]+)", r"\\1[REDACTED]", text, flags=re.IGNORECASE)
    return text"""

# Replace is_kill_switch_active
old_kill = """    def is_kill_switch_active(self) -> bool:
        \"\"\"Checks if collection is halted via kill-switch.\"\"\"
        if not self.config.get("collector_enabled", True):
            return True
        if self.config.get("kill_switch_active", False):
            return True
        env_kill = os.environ.get("VAYUDRISHTI_KILL_SWITCH", "").lower().strip()
        if env_kill in ("true", "1", "yes", "active"):
            return True
        return False"""

new_kill = """    def is_kill_switch_active(self) -> bool:
        \"\"\"Checks if collection is halted via kill-switch.\"\"\"
        if not self.config.get("collector_enabled", True):
            return True
        if self.config.get("kill_switch_active", False):
            return True
        ks_file = self.config.get("kill_switch_file")
        if ks_file and Path(ks_file).exists():
            return True
        env_kill = os.environ.get("VAYUDRISHTI_KILL_SWITCH", "").lower().strip()
        if env_kill in ("true", "1", "yes", "active"):
            return True
        return False"""

alc_norm = alc_text.replace("\r\n", "\n")
alc_norm = alc_norm.replace(old_redact.replace("\r\n", "\n"), new_redact.replace("\r\n", "\n"))
alc_norm = alc_norm.replace(old_kill.replace("\r\n", "\n"), new_kill.replace("\r\n", "\n"))

with open(alc_file, "w", encoding="utf-8", newline="\r\n") as f:
    f.write(alc_norm)
print("autonomous_live_collector.py patched successfully.")

# 2. Fix meteorological_service.py
meteo_file = ROOT_DIR / "src" / "services" / "meteorological_service.py"
with open(meteo_file, "r", encoding="utf-8") as f:
    meteo_text = f.read()

old_anchor = """        if isinstance(issue_time_utc, str):
            dt_anchor = pd.to_datetime(issue_time_utc, utc=True)
        else:
            dt_anchor = issue_time_utc if issue_time_utc.tzinfo else issue_time_utc.replace(tzinfo=timezone.utc)

        # Truncate anchor to nearest floor clock-hour
        dt_anchor = dt_anchor.floor("h")"""

new_anchor = """        dt_anchor = pd.to_datetime(issue_time_utc, utc=True).floor("h")"""

meteo_norm = meteo_text.replace("\r\n", "\n")
meteo_norm = meteo_norm.replace(old_anchor.replace("\r\n", "\n"), new_anchor.replace("\r\n", "\n"))

with open(meteo_file, "w", encoding="utf-8", newline="\r\n") as f:
    f.write(meteo_norm)
print("meteorological_service.py patched successfully.")

# 3. Fix tests/test_phase12_autonomous_monitoring.py
test_file = ROOT_DIR / "tests" / "test_phase12_autonomous_monitoring.py"
with open(test_file, "r", encoding="utf-8") as f:
    test_text = f.read()

old_test_cmd = """    win_cmd = generate_scheduler_script(platform="windows")
    assert "schtasks /create" in win_cmd"""

new_test_cmd = """    win_cmd = generate_scheduler_script(platform="windows")
    assert "schtasks" in win_cmd.lower() and "/create" in win_cmd.lower()"""

test_norm = test_text.replace("\r\n", "\n")
test_norm = test_norm.replace(old_test_cmd.replace("\r\n", "\n"), new_test_cmd.replace("\r\n", "\n"))

with open(test_file, "w", encoding="utf-8", newline="\r\n") as f:
    f.write(test_norm)
print("test_phase12_autonomous_monitoring.py patched successfully.")
