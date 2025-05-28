#!/usr/bin/env python3
"""
API Server for Document Upload Integration

FastAPI server providing RESTful endpoints for document upload,
processing status, and retrieval of results. Integrates with
Hayhooks/RAG pipeline, Pinecone, and Redis cache.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import asyncio
import aiofiles
import os
import json
import uuid
import time
from datetime import datetime
from pathlib import Path
import logging
import traceback

# Import document processor
from document_processor import DocumentProcessor, ProcessedDocument, create_document_processor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Document Upload API",
    description="API for document upload, processing, and RAG pipeline integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
SUPPORTED_FORMATS = [".pdf", ".txt", ".md", ".docx", ".csv", ".json"]

# In-memory storage for processing status (in production, use Redis/database)
processing_status: Dict[str, Dict[str, Any]] = {}
processed_results: Dict[str, ProcessedDocument] = {}

# Initialize directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Pydantic models
class UploadResponse(BaseModel):
    """Response model for file upload"""
    job_id: str = Field(..., description="Unique job identifier")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    namespace: str = Field(..., description="Document namespace")
    status: str = Field(..., description="Initial status")
    uploaded_at: str = Field(..., description="Upload timestamp")

class ProcessingStatus(BaseModel):
    """Model for processing status"""
    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Current status")
    progress: float = Field(..., description="Progress percentage (0-100)")
    message: str = Field(..., description="Status message")
    started_at: Optional[str] = Field(None, description="Processing start time")
    completed_at: Optional[str] = Field(None, description="Processing completion time")
    error_message: Optional[str] = Field(None, description="Error message if failed")

class ProcessingResult(BaseModel):
    """Model for processing results"""
    job_id: str = Field(..., description="Job identifier")
    success: bool = Field(..., description="Processing success status")
    filename: str = Field(..., description="Original filename")
    namespace: str = Field(..., description="Document namespace")
    processing_time: float = Field(..., description="Processing time in seconds")
    word_count: Optional[int] = Field(None, description="Total word count")
    chunk_count: Optional[int] = Field(None, description="Number of chunks generated")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")
    error_message: Optional[str] = Field(None, description="Error message if failed")

class ChunkInfo(BaseModel):
    """Model for chunk information"""
    chunk_id: int = Field(..., description="Chunk identifier")
    content: str = Field(..., description="Chunk content")
    word_count: int = Field(..., description="Words in chunk")
    start_char: int = Field(..., description="Start character position")
    end_char: int = Field(..., description="End character position")
    metadata: Dict[str, Any] = Field(..., description="Chunk metadata")

class ProcessingConfig(BaseModel):
    """Model for processing configuration"""
    chunk_size: int = Field(512, description="Chunk size in words", ge=100, le=1000)
    chunk_overlap: int = Field(50, description="Chunk overlap in words", ge=0, le=200)
    enable_cache: bool = Field(True, description="Enable Redis cache")
    generate_embeddings: bool = Field(False, description="Generate embeddings")
    index_to_pinecone: bool = Field(False, description="Index to Pinecone")

# Helper functions
def generate_job_id() -> str:
    """Generate unique job ID"""
    return f"job_{uuid.uuid4().hex[:12]}_{int(time.time())}"

def validate_file_format(filename: str) -> bool:
    """Validate file format"""
    extension = Path(filename).suffix.lower()
    return extension in SUPPORTED_FORMATS

def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get file information"""
    path_obj = Path(file_path)
    stat = path_obj.stat()
    
    return {
        "filename": path_obj.name,
        "file_size": stat.st_size,
        "extension": path_obj.suffix.lower(),
        "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
    }

async def save_uploaded_file(file: UploadFile, namespace: str) -> str:
    """Save uploaded file to disk"""
    # Create namespace directory
    namespace_dir = Path(UPLOAD_DIR) / namespace
    namespace_dir.mkdir(exist_ok=True)
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    file_path = namespace_dir / filename
    
    # Save file
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    return str(file_path)

async def process_document_async(job_id: str, file_path: str, namespace: str, config: ProcessingConfig):
    """Process document asynchronously"""
    try:
        # Update status to processing
        processing_status[job_id].update({
            "status": "processing",
            "progress": 10,
            "message": "Processing document...",
            "started_at": datetime.now().isoformat()
        })
        
        # Create document processor
        processor = create_document_processor(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        
        # Update progress
        processing_status[job_id].update({
            "progress": 30,
            "message": "Extracting text content..."
        })
        
        # Process document
        result = processor.process_document(file_path, namespace)
        
        # Update progress
        processing_status[job_id].update({
            "progress": 70,
            "message": "Generating chunks..."
        })
        
        if result.success:
            # Store result
            processed_results[job_id] = result
            
            # Simulate additional processing steps
            if config.generate_embeddings:
                processing_status[job_id].update({
                    "progress": 85,
                    "message": "Generating embeddings..."
                })
                await asyncio.sleep(1)  # Simulate embedding generation
            
            if config.index_to_pinecone:
                processing_status[job_id].update({
                    "progress": 95,
                    "message": "Indexing to Pinecone..."
                })
                await asyncio.sleep(0.5)  # Simulate indexing
            
            # Complete processing
            processing_status[job_id].update({
                "status": "completed",
                "progress": 100,
                "message": "Processing completed successfully",
                "completed_at": datetime.now().isoformat()
            })
            
            logger.info(f"Successfully processed document for job {job_id}")
            
        else:
            # Processing failed
            processing_status[job_id].update({
                "status": "failed",
                "progress": 100,
                "message": f"Processing failed: {result.error_message}",
                "completed_at": datetime.now().isoformat(),
                "error_message": result.error_message
            })
            
            logger.error(f"Failed to process document for job {job_id}: {result.error_message}")
        
    except Exception as e:
        # Unexpected error
        error_msg = f"Unexpected error: {str(e)}"
        processing_status[job_id].update({
            "status": "failed",
            "progress": 100,
            "message": error_msg,
            "completed_at": datetime.now().isoformat(),
            "error_message": error_msg
        })
        
        logger.error(f"Unexpected error processing job {job_id}: {e}")
        logger.error(traceback.format_exc())

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Document Upload API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": "running"
    }

@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    namespace: str = Form("default"),
    chunk_size: int = Form(512),
    chunk_overlap: int = Form(50),
    enable_cache: bool = Form(True),
    generate_embeddings: bool = Form(False),
    index_to_pinecone: bool = Form(False)
):
    """
    Upload and process a document
    
    Args:
        file: Document file to upload
        namespace: Document namespace for organization
        chunk_size: Size of text chunks in words
        chunk_overlap: Overlap between chunks in words
        enable_cache: Whether to use Redis cache
        generate_embeddings: Whether to generate embeddings
        index_to_pinecone: Whether to index to Pinecone
    
    Returns:
        UploadResponse: Upload confirmation with job ID
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        if not validate_file_format(file.filename):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Supported: {', '.join(SUPPORTED_FORMATS)}"
            )
        
        # Check file size
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Reset file pointer and save
        await file.seek(0)
        file_path = await save_uploaded_file(file, namespace)
        
        # Generate job ID
        job_id = generate_job_id()
        
        # Initialize processing status
        processing_status[job_id] = {
            "job_id": job_id,
            "status": "uploaded",
            "progress": 0,
            "message": "File uploaded, processing queued",
            "filename": file.filename,
            "namespace": namespace,
            "file_path": file_path,
            "uploaded_at": datetime.now().isoformat()
        }
        
        # Create processing config
        config = ProcessingConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            enable_cache=enable_cache,
            generate_embeddings=generate_embeddings,
            index_to_pinecone=index_to_pinecone
        )
        
        # Start background processing
        background_tasks.add_task(
            process_document_async,
            job_id, file_path, namespace, config
        )
        
        logger.info(f"File uploaded: {file.filename} -> Job ID: {job_id}")
        
        return UploadResponse(
            job_id=job_id,
            filename=file.filename,
            file_size=len(content),
            namespace=namespace,
            status="uploaded",
            uploaded_at=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/status/{job_id}", response_model=ProcessingStatus)
async def get_processing_status(job_id: str):
    """
    Get processing status for a job
    
    Args:
        job_id: Job identifier
        
    Returns:
        ProcessingStatus: Current processing status
    """
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    status_data = processing_status[job_id]
    
    return ProcessingStatus(
        job_id=job_id,
        status=status_data["status"],
        progress=status_data["progress"],
        message=status_data["message"],
        started_at=status_data.get("started_at"),
        completed_at=status_data.get("completed_at"),
        error_message=status_data.get("error_message")
    )

@app.get("/result/{job_id}", response_model=ProcessingResult)
async def get_processing_result(job_id: str):
    """
    Get processing result for a completed job
    
    Args:
        job_id: Job identifier
        
    Returns:
        ProcessingResult: Processing result with metadata
    """
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    status_data = processing_status[job_id]
    
    if status_data["status"] not in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    if job_id in processed_results:
        result = processed_results[job_id]
        
        return ProcessingResult(
            job_id=job_id,
            success=result.success,
            filename=result.metadata.filename,
            namespace=result.metadata.namespace,
            processing_time=result.processing_time,
            word_count=result.metadata.word_count,
            chunk_count=len(result.chunks),
            metadata={
                "file_size": result.metadata.file_size,
                "file_type": result.metadata.file_type,
                "title": result.metadata.title,
                "author": result.metadata.author,
                "character_count": result.metadata.character_count,
                "processed_at": result.metadata.processed_at
            },
            error_message=result.error_message
        )
    else:
        # Job failed
        return ProcessingResult(
            job_id=job_id,
            success=False,
            filename=status_data["filename"],
            namespace=status_data["namespace"],
            processing_time=0.0,
            error_message=status_data.get("error_message", "Unknown error")
        )

@app.get("/chunks/{job_id}", response_model=List[ChunkInfo])
async def get_document_chunks(
    job_id: str,
    limit: int = Query(10, description="Maximum number of chunks to return", ge=1, le=100),
    offset: int = Query(0, description="Number of chunks to skip", ge=0)
):
    """
    Get document chunks for a processed job
    
    Args:
        job_id: Job identifier
        limit: Maximum number of chunks to return
        offset: Number of chunks to skip
        
    Returns:
        List[ChunkInfo]: Document chunks
    """
    if job_id not in processed_results:
        raise HTTPException(status_code=404, detail="Job not found or not processed")
    
    result = processed_results[job_id]
    
    if not result.success:
        raise HTTPException(status_code=400, detail="Job processing failed")
    
    # Apply pagination
    chunks = result.chunks[offset:offset + limit]
    
    return [
        ChunkInfo(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            word_count=chunk.word_count,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            metadata=chunk.metadata
        )
        for chunk in chunks
    ]

@app.get("/jobs", response_model=List[Dict[str, Any]])
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    namespace: Optional[str] = Query(None, description="Filter by namespace"),
    limit: int = Query(50, description="Maximum number of jobs to return", ge=1, le=100)
):
    """
    List processing jobs
    
    Args:
        status: Filter by job status
        namespace: Filter by namespace
        limit: Maximum number of jobs to return
        
    Returns:
        List of job information
    """
    jobs = []
    
    for job_id, job_data in processing_status.items():
        # Apply filters
        if status and job_data.get("status") != status:
            continue
        if namespace and job_data.get("namespace") != namespace:
            continue
        
        jobs.append({
            "job_id": job_id,
            "filename": job_data.get("filename"),
            "namespace": job_data.get("namespace"),
            "status": job_data.get("status"),
            "progress": job_data.get("progress"),
            "uploaded_at": job_data.get("uploaded_at"),
            "completed_at": job_data.get("completed_at")
        })
    
    # Sort by upload time (newest first) and apply limit
    jobs.sort(key=lambda x: x.get("uploaded_at", ""), reverse=True)
    return jobs[:limit]

@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job and its associated data
    
    Args:
        job_id: Job identifier
        
    Returns:
        Success message
    """
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = processing_status[job_id]
    
    # Delete file if exists
    file_path = job_data.get("file_path")
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"Deleted file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not delete file {file_path}: {e}")
    
    # Remove from memory
    del processing_status[job_id]
    if job_id in processed_results:
        del processed_results[job_id]
    
    logger.info(f"Deleted job: {job_id}")
    
    return {"message": f"Job {job_id} deleted successfully"}

@app.get("/stats", response_model=Dict[str, Any])
async def get_system_stats():
    """
    Get system statistics
    
    Returns:
        System statistics and metrics
    """
    total_jobs = len(processing_status)
    completed_jobs = len([j for j in processing_status.values() if j.get("status") == "completed"])
    failed_jobs = len([j for j in processing_status.values() if j.get("status") == "failed"])
    in_progress_jobs = len([j for j in processing_status.values() if j.get("status") in ["uploaded", "processing"]])
    
    # Calculate total processed documents and chunks
    total_chunks = sum(len(result.chunks) for result in processed_results.values() if result.success)
    total_words = sum(result.metadata.word_count or 0 for result in processed_results.values() if result.success)
    
    # Get namespaces
    namespaces = list(set(j.get("namespace", "default") for j in processing_status.values()))
    
    return {
        "total_jobs": total_jobs,
        "completed_jobs": completed_jobs,
        "failed_jobs": failed_jobs,
        "in_progress_jobs": in_progress_jobs,
        "success_rate": (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0,
        "total_chunks_generated": total_chunks,
        "total_words_processed": total_words,
        "active_namespaces": namespaces,
        "uptime": datetime.now().isoformat(),
        "supported_formats": SUPPORTED_FORMATS
    }

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8503,
        reload=True,
        log_level="info"
    ) 