"""
src/__init__.py

VayuDrishti core package initialization.
Ensures Windows DLL path resolution for PyTorch C-extensions.
"""
import os
import sys

if sys.platform == "win32":
    try:
        import site
        for p in site.getsitepackages():
            torch_lib = os.path.join(p, "torch", "lib")
            if os.path.isdir(torch_lib):
                try:
                    os.add_dll_directory(torch_lib)
                except (AttributeError, OSError):
                    pass
    except Exception:
        pass
