from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from src.rag.pipeline import ingest_documents
import tempfile
import os

router = APIRouter()

@router.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(
            status_code=400, 
            detail="No files provided"
        )

    file_paths = []
    try:
        for file in files:
            suffix = ".pdf" if file.content_type == "application/pdf" else ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await file.read()
                tmp.write(content)
                file_paths.append((tmp.name, file.filename))

        result = ingest_documents(file_paths)
        return {"status": "success", **result}

    finally:
        for path, _ in file_paths:
            os.unlink(path)