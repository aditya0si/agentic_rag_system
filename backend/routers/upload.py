"""
Document upload endpoint — accepts PDF/DOCX/TXT files, validates, and triggers ingestion.
"""

import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, status
from fastapi.responses import JSONResponse

from .. import config
from ..core.ingestion import extract_text, chunk_document
from ..core.vector_store import VectorStoreManager
from ..core.memory import add_document_to_session
from ..core.security import validate_filename
from ..models.schemas import UploadResponse

router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    """
    Accepts a document upload, validates the extension and size,
    extracts/chunks/embeds text, stores in ChromaDB, and registers with session.
    """
    filename = file.filename or "unknown"

    # 1. Validate filename (extension + path-traversal guard)
    try:
        filename = validate_filename(filename)
    except ValueError as e:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "unsupported_file_type",
                "message": str(e)
            }
        )

    suffix = Path(filename).suffix.lower()
        
    # 2. Write file to temp directory while checking size
    temp_dir = Path(__file__).parent.parent / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    doc_id = f"doc_{uuid.uuid4().hex[:6]}"
    temp_file_path = temp_dir / f"{doc_id}{suffix}"
    
    try:
        size = 0
        with open(temp_file_path, "wb") as buffer:
            while chunk := await file.read(8192):
                size += len(chunk)
                if size > config.MAX_UPLOAD_SIZE:
                    buffer.close()
                    if temp_file_path.exists():
                        temp_file_path.unlink()
                    return JSONResponse(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        content={
                            "error": "file_too_large",
                            "message": f"File exceeds the maximum size limit of {config.MAX_UPLOAD_SIZE // (1024*1024)}MB."
                        }
                    )
                buffer.write(chunk)
                
        # 3. Extract text
        try:
            pages = extract_text(temp_file_path)
        except Exception as e:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "error": "parsing_failed",
                    "message": f"Failed to parse document: {str(e)}"
                }
            )
            
        # 4. Chunk document
        chunks = chunk_document(
            pages=pages,
            doc_id=doc_id,
            doc_name=filename,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        
        # 5. Embed and save to ChromaDB
        if chunks:
            manager = VectorStoreManager()
            manager.add_chunks(chunks)
            
        # 6. Associate with session in memory
        add_document_to_session(session_id, doc_id)
        
        return UploadResponse(
            doc_id=doc_id,
            filename=filename,
            chunks_created=len(chunks),
            status="indexed"
        )
        
    finally:
        # Clean up temp file
        if temp_file_path.exists():
            temp_file_path.unlink()
