"""
CSV upload handling: save uploaded file to a temp path and return path.
"""
import os
import tempfile
import uuid
from pathlib import Path

from fastapi import UploadFile


def save_uploaded_csv(file: UploadFile) -> str | None:
    """
    Save uploaded CSV to a temporary file. Returns the file path or None on failure.
    """
    try:
        suffix = Path(file.filename or "upload.csv").suffix or ".csv"
        fd, path = tempfile.mkstemp(suffix=suffix, prefix="analyst_")
        os.close(fd)
        content = file.file.read()
        with open(path, "wb") as f:
            f.write(content)
        return path
    except Exception:
        return None
