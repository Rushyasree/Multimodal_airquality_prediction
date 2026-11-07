"""
scripts/run_full_audit_tests.py

Comprehensive Test Suite Orchestrator for Phase 15 Readiness Audit.
Executes all 24 test files in tests/, captures execution metrics,
handles network isolation gracefully, and exports readiness_test_results.csv.
"""

import sys
import os
import time
import subprocess
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
TESTS_DIR = ROOT_DIR / "tests"
OUTPUT_CSV = ROOT_DIR / "readiness_test_results.csv"

# Collect all test files
test_files = sorted([f.name for f in TESTS_DIR.glob("test_*.py")])

results = []

print("=" * 80)
print(f"VAYUDRISHTI PHASE 15: AUDITING ALL {len(test_files)} TEST SUITES")
print("=" * 80)

total_passed = 0
total_failed = 0
total_skipped = 0
total_errors = 0
start_all = time.time()

for idx, test_file in enumerate(test_files, 1):
    test_path = TESTS_DIR / test_file
    print(f"[{idx:02d}/{len(test_files):02d}] Running {test_file}...", end=" ", flush=True)
    t0 = time.time()
    
    # Run pytest on the individual test file without langsmith plugin to avoid network hanging
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-p", "no:langsmith",
        str(test_path),
        "-q",
        "--tb=short"
    ]
    
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(ROOT_DIR)
        )
        duration = round(time.time() - t0, 2)
        stdout = proc.stdout + "\n" + proc.stderr
        
        # Parse pytest summary line
        # e.g., "5 passed in 2.34s" or "3 passed, 1 failed in 4.56s"
        passed = 0
        failed = 0
        skipped = 0
        errors = 0
        
        for line in stdout.splitlines():
            line_str = line.strip()
            if "passed" in line_str or "failed" in line_str or "error" in line_str or "skipped" in line_str:
                import re
                m_p = re.search(r"(\d+)\s+passed", line_str)
                m_f = re.search(r"(\d+)\s+failed", line_str)
                m_s = re.search(r"(\d+)\s+skipped", line_str)
                m_e = re.search(r"(\d+)\s+error", line_str)
                if m_p: passed = int(m_p.group(1))
                if m_f: failed = int(m_f.group(1))
                if m_s: skipped = int(m_s.group(1))
                if m_e: errors = int(m_e.group(1))
        
        # If exit code was 0 but passed was 0, check if it was a standalone script or passed without pytest summary
        if proc.returncode == 0 and passed == 0:
            # Check if there were warnings or collected items
            if "collected" in stdout:
                import re
                m_c = re.search(r"collected\s+(\d+)\s+items", stdout)
                if m_c:
                    passed = int(m_c.group(1))
        
        status = "PASSED" if proc.returncode == 0 and failed == 0 and errors == 0 else "FAILED"
        if proc.returncode != 0 and failed == 0 and errors == 0:
            status = "ERROR"
            errors = 1
            
        print(f"{status} ({passed} passed, {failed} failed, {errors} errors, {duration}s)")
        if status != "PASSED":
            print(f"    Details:\n{stdout[:400]}")
            
    except subprocess.TimeoutExpired:
        duration = round(time.time() - t0, 2)
        status = "TIMEOUT"
        passed, failed, skipped, errors = 0, 0, 0, 1
        stdout = "Test suite timed out after 120 seconds."
        print(f"TIMEOUT ({duration}s)")
    except Exception as exc:
        duration = round(time.time() - t0, 2)
        status = "EXCEPTION"
        passed, failed, skipped, errors = 0, 0, 0, 1
        stdout = str(exc)
        print(f"EXCEPTION ({duration}s): {exc}")
        
    total_passed += passed
    total_failed += failed
    total_skipped += skipped
    total_errors += errors
    
    results.append({
        "test_file": test_file,
        "status": status,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "errors": errors,
        "duration_sec": duration,
        "notes": stdout[-200:].replace("\n", " ") if status != "PASSED" else "Clean execution"
    })

df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
total_duration = round(time.time() - start_all, 2)

print("=" * 80)
print(f"TEST SUITE SUMMARY: {len(test_files)} Suites Executed in {total_duration}s")
print(f"Total Passed:  {total_passed}")
print(f"Total Failed:  {total_failed}")
print(f"Total Skipped: {total_skipped}")
print(f"Total Errors:  {total_errors}")
print(f"Results exported to: {OUTPUT_CSV}")
print("=" * 80)
