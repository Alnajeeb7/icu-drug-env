"""
Root app.py — HuggingFace Spaces entry point.
HF Spaces looks for app.py in root by default.
This just imports and re-exports the FastAPI app from server/.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.app import app  # noqa: F401 — re-export for uvicorn

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
