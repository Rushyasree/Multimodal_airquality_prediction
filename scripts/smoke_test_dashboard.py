import subprocess
import time
import urllib.request
import sys

print("Launching Streamlit dashboard in headless mode...")
cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true", "--server.port", "8501"]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

try:
    print("Waiting 6 seconds for Streamlit server initialization...")
    time.sleep(6)
    url = "http://localhost:8501"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=5) as response:
        code = response.getcode()
        print(f"Streamlit HTTP Response Code: {code}")
        assert code == 200, f"Expected 200, got {code}"
        print("[SUCCESS] Streamlit dashboard successfully launched, bound to port 8501, and returned HTTP 200 OK!")
finally:
    print("Terminating test Streamlit server...")
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("Dashboard test process terminated cleanly.")
