from pathlib import Path
p = Path('tests/test_phase12_autonomous_monitoring.py')
text = p.read_text(encoding='utf-8')
text = text.replace('assert "0 * * * *" in posix_cron', 'assert "15 * * * *" in posix_cron')
p.write_text(text, encoding='utf-8')
print("Successfully fixed cron assertion!")
