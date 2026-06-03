import os
import sys

# Ensure project root is on the path so `backend` package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.main import app  # noqa: F401  — Vercel detects the ASGI app by name
