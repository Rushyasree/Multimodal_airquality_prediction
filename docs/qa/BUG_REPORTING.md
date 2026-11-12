# VayuDrishti Bug Reporting & Issue Management Guide

This document standardizes the process for discovering, reporting, triaging, and resolving defects in VayuDrishti.

---

## 1. Severity & Priority Classification

Every discovered issue must be tagged with a Severity level and an operational Priority:

| Severity | Operational Impact | Target Resolution | Example |
| :--- | :--- | :--- | :--- |
| **S1 — Blocker / Critical** | Service crash, infinite loop/deadlock, secret leakage, or invalid data presented as genuine telemetry. | Immediate (<24h) | Thread lock deadlock causing page freeze; unhandled exception on default route. |
| **S2 — Major** | Core forecasting feature failure, validation gate miscalculation, fallback failure. | High (<48h) | Persistence baseline computation error; GBDT 49-feature schema misalignment. |
| **S3 — Moderate** | Visual layout glitch, non-breaking latency degradation, warning message spam. | Normal | Streamlit container width warning; minor chart legend clipping. |
| **S4 — Minor** | Typo in documentation, non-functional styling inconsistency, minor cosmetic flaw. | Low (Next Sprint) | Outdated link in markdown guide; minor badge color mismatch. |

---

## 2. Standard Bug Report Format

When reporting an issue, use the following standardized structure:

```markdown
### Bug Description
A clear and concise summary of what the bug is.

### Environment Details
- **OS**: Windows 11 / Ubuntu 22.04 / macOS
- **Python Version**: 3.10.11
- **Streamlit Version**: 1.28.0+
- **Commit Hash / Branch**: e.g., `04cc5f2` on `main`
- **Execution Mode**: Local / Docker / Streamlit Community Cloud

### Steps to Reproduce
1. Start the application: `streamlit run app.py`
2. Navigate to tab: 'Unified Hybrid Forecast'
3. Select station: 'AP001 - Amaravati'
4. Observe the forecast card.

### Expected Behavior
A valid forecast is rendered with conformal uncertainty bounds and explicit data provenance.

### Actual Behavior
An unhandled exception occurs or incorrect fallback status is displayed.

### Sanitized Logs & Error Trace
```text
[Paste error log here - ensure NO API tokens or private paths are included]
```

### Reproducibility
- [ ] Always (100%)
- [ ] Intermittent
- [ ] Only when offline
- [ ] Only with specific station/inputs
```

---

## 3. Bug Fix Verification & Regression Policy

Before any bug fix is merged into `main`:
1. **Root Cause Analysis (RCA)**: Explain why the bug occurred (e.g., non-reentrant lock, timezone mismatch).
2. **Dedicated Regression Test**: Write an automated pytest in `tests/` that reproduces the issue and verifies that it is solved.
3. **Audit Suite Execution**: Run the relevant test suite (e.g., `tests/test_operational_monitoring_and_reliability.py`) to confirm zero regressions.
4. **Secret Scrubbing Check**: Confirm that no API tokens, sensitive environment variables, or private paths appear in the fix or logs.
