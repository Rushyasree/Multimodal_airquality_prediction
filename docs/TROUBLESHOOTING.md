# VayuDrishti — Troubleshooting & Diagnostics Guide

This document catalogs common runtime anomalies, architectural safety gates, diagnostic commands, and verified resolutions.

---

## 1. PyTorch Windows DLL Initialization Failure (`[WinError 1114]`)

### Symptom:
```text
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed.
Error loading "...\torch\lib\c10.dll" or one of its dependencies.
```

### Root Cause:
On Windows 10/11, Python 3.8+ restricts default DLL search paths. When PyTorch or Torchvision loads C++ native libraries (`c10.dll`, `torch_cpu.dll`), dependent runtime libraries are not located if the `torch/lib` directory has not been registered via Windows API `AddDllDirectory`.

### Resolution:
VayuDrishti automatically registers the directory inside [`src/__init__.py`](file:///E:/Documents/multimodal_air_quality_vscode_starter/src/__init__.py). If executing custom scripts outside `src/`, include:
```python
import os, importlib.util
spec = importlib.util.find_spec("torch")
if spec and spec.origin:
    lib_path = os.path.join(os.path.dirname(spec.origin), "lib")
    if os.path.isdir(lib_path):
        os.add_dll_directory(lib_path)
```

---

## 2. OpenAQ API DNS Resolution Failure (`[Errno 11001] getaddrinfo failed`)

### Symptom:
```text
WARNING [VayuDrishti] OpenAQ API locations failed near (16.5151,80.5182):
HTTPSConnectionPool(host='api.openaq.org', port=443): Max retries exceeded
(Caused by NameResolutionError: Failed to resolve 'api.openaq.org' ([Errno 11001] getaddrinfo failed))
```

### Root Cause:
The environment lacks external internet access (e.g. sandboxed test runner, offline demo environment, corporate firewall).

### Behavior & Resolution:
1. **Zero Crash Guarantee**: `CurrentAirQualityService` catches connection exceptions gracefully.
2. **Automatic Offline Fallback**: The service automatically returns cached or verified baseline telemetry while tagging the status with `is_live = False` and `data_source = "Fallback/Simulated Telemetry"`.
3. To test live connectivity when internet access is restored:
   ```bash
   pytest tests/test_live_openaq_connectivity.py -v
   ```

---

## 3. Windows Terminal UnicodeEncodeError (`charmap codec can't encode character`)

### Symptom:
```text
UnicodeEncodeError: 'charmap' codec can't encode character '\u2713' in position 0: character maps to <undefined>
```

### Root Cause:
Windows command prompt (cmd.exe) and default PowerShell sessions often use legacy code pages (e.g., `cp1252`), which cannot encode certain Unicode symbols (such as checkmarks `✓` or arrows `→`).

### Resolution:
1. Set the terminal encoding to UTF-8 before running scripts:
   ```powershell
   chcp 65001
   $env:PYTHONIOENCODING = "utf-8"
   ```
2. In codebase scripts, VayuDrishti standardizes on ASCII indicators (e.g., `[OK]`, `[PASS]`, `[FAIL]`) to guarantee deterministic execution across all shells.

---

## 4. Streamlit Port Conflicts & Headless Execution

### Symptom:
```text
AxiosError: Request failed with status code 403 / Port 8501 is already in use
```

### Resolution:
Specify an alternative port or explicitly enable headless execution:
```bash
streamlit run app.py --server.port 8502 --server.headless true
```

To terminate any hanging Streamlit processes:
```powershell
# Windows
Get-Process python | Where-Object { $_.CommandLine -like "*streamlit*" } | Stop-Process -Force

# Linux / macOS
pkill -f "streamlit run"
```

---

## 5. Stale Data Warning in Autonomous Collector / Buffer

### Symptom:
```text
"reason": "stale_data", "detail": "Telemetry is 33.4 hours old (threshold is 3.0h)."
```

### Root Cause:
A station's last reported observation from OpenAQ exceeds the configured `max_staleness_hours` (default: 3.0 hours). This occurs during station maintenance, network outages, or when testing historical timestamps.

### Resolution:
- For **live operations**: This is an intended safety gate preventing stale data from masquerading as current observations. The buffer correctly excludes stale telemetry from live next-hour forecasts.
- For **historical/unit tests**: Pass `current_system_time=anchor_time_utc` to evaluate historical rolling sequences without tripping real-time staleness filters.
- Adjust threshold in `.env` if necessary:
  ```env
  VAYUDRISHTI_STALENESS_HOURS=6.0
  ```

---

## 6. Checkpoint Integrity & Hash Mismatch

### Symptom:
Model loading fails with checksum warning or `FileNotFoundError`.

### Resolution:
Verify all canonical checkpoint paths and hashes:
```bash
python -c "
from pathlib import Path
import hashlib
checkpoints = [
    'models/lstm_baseline/best_lstm_model.pt',
    'experiments/recent_data_audit/checkpoints/adapted_gbdt_pm25.joblib',
    'experiments/recent_data_audit/checkpoints/adapted_gbdt_pm10.joblib',
    'experiments/unified_hybrid/checkpoints/best_unified_hybrid_model.pt',
    'experiments/unified_hybrid/accuracy_optimization/checkpoints/best_optimized_hybrid_model.pt',
    'models/pm25vision_cnn/best_model.pt'
]
for p in checkpoints:
    path = Path(p)
    if path.exists():
        h = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        print(f'[OK] {path.name:35s} SHA256: {h} ({path.stat().st_size:,} bytes)')
    else:
        print(f'[FAIL] MISSING: {path}')
"
```
