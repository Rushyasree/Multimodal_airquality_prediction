from streamlit.testing.v1 import AppTest
import os

print("Testing Streamlit app execution via AppTest (simulated browser session)...")
at = AppTest.from_file("app.py", default_timeout=45)
at.run()

if at.exception:
    print(f"Exception encountered during app run: {at.exception}")
    for exc in at.exception:
        print(f"  {exc}")
else:
    print(f"App rendered successfully! Title/Headers count: {len(at.title) + len(at.header)}")
    print(f"Total markdown blocks: {len(at.markdown)}")
    print(f"Total metrics rendered: {len(at.metric)}")
    print(f"Total tabs rendered: {len(at.tabs)}")
    print("[SUCCESS] app.py executed and rendered without exceptions!")
