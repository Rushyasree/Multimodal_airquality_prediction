# VayuDrishti — Environment Setup & Installation Guide

This document provides instructions for configuring the VayuDrishti runtime environment on Windows, Linux, and macOS.

---

## 1. Prerequisites

- **Python**: Version `3.10` or `3.11` (Tested on Python `3.10.11`).
- **Operating System**: Windows 10/11, Ubuntu 20.04/22.04 LTS, or macOS 12+ (Apple Silicon or Intel).
- **RAM**: Minimum 8 GB (16 GB recommended for multi-station spatial simulations).
- **Disk Space**: ~3 GB free disk space (includes checkpoints, sample images, and dependencies).

---

## 2. Virtual Environment Creation

Create a clean virtual environment in the project root:

### Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Dependency Installation

Upgrade pip and install the hardened production dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Core Dependency Breakdown:
- **Deep Learning**: `torch>=2.0.0`, `torchvision>=0.15.0`
- **Machine Learning & Descriptors**: `scikit-learn>=1.3.0`, `joblib>=1.3.0`, `scipy>=1.11,<2`
- **Data Engineering**: `pandas>=2.0.0`, `numpy>=1.24.0`, `pytz>=2023.3`
- **Telemetry & APIs**: `requests>=2.31,<3`
- **Web Dashboard & Visualization**: `streamlit>=1.28.0`, `pydeck>=0.8,<1`, `plotly>=5.17.0`, `matplotlib>=3.7.0`
- **System Monitoring**: `psutil>=5.9,<7`
- **Testing**: `pytest>=7.4.0`

---

## 4. Platform-Specific Setup: Windows DLLs

On Windows platforms, PyTorch's native C++ runtime (`c10.dll`, `torch_cpu.dll`) requires dependent C++ runtime libraries to be explicitly resolved in user-mode applications.

VayuDrishti includes an automated DLL handler in [`src/__init__.py`](file:///E:/Documents/multimodal_air_quality_vscode_starter/src/__init__.py):

```python
import sys, os
if sys.platform == "win32":
    import importlib.util
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec and torch_spec.origin:
        torch_lib = os.path.join(os.path.dirname(torch_spec.origin), "lib")
        if os.path.isdir(torch_lib):
            os.add_dll_directory(torch_lib)
```

This ensures zero `[WinError 1114] A dynamic link library (DLL) initialization routine failed` exceptions occur when loading PyTorch or torchvision modules.

---

## 5. Configuration & API Credentials

VayuDrishti connects to OpenAQ API v3 for live CAAQMS air quality telemetry.

Create a `.env` file in the project root (a template `.env.example` is provided):

```env
# OpenAQ API v3 Configuration
OPENAQ_API_KEY=your_openaq_api_key_here
OPENAQ_BASE_URL=https://api.openaq.org/v3

# Operational Parameters
VAYUDRISHTI_STALENESS_HOURS=3.0
VAYUDRISHTI_SEARCH_RADIUS_KM=50.0
VAYUDRISHTI_LIVE_TRIAL_ENABLED=false
```

### Security Commitment:
- The system never exposes raw API keys in web interfaces, console logs, or exported JSON/CSV artifacts.
- When live credentials or external network access are unavailable, the platform automatically engages validated fallback mechanisms without crashing.

---

## 6. Verifying Clean Installation

Run the automated environment sanity check:

```bash
python -c "
import torch, torchvision, sklearn, streamlit, pydeck, requests, scipy
print('[OK] All required libraries successfully loaded.')
print(f'PyTorch: {torch.__version__} (CUDA available: {torch.cuda.is_available()})')
print(f'Torchvision: {torchvision.__version__}')
print(f'Scikit-Learn: {sklearn.__version__}')
print(f'Streamlit: {streamlit.__version__}')
"
```
