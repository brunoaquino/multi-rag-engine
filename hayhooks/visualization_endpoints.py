"""
Visualization Endpoints for Haystack Pipelines

This module provides FastAPI endpoints for pipeline visualization,
allowing external tools and interfaces to generate pipeline diagrams.
"""

import logging
import os
from typing import Dict, Any, Optional, List
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from pipeline_visualizer import PipelineVisualizer, create_visualizer

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/visualize", tags=["visualization"])

# Initialize visualizer
visualizer = create_visualizer(local_server=True)


class VisualizationRequest(BaseModel):
    """Request model for pipeline visualization"""
    pipeline_name: str
    format: str = "mermaid-image"
    super_component_expansion: bool = True
    use_local_server: bool = True


class PipelineInfo(BaseModel):
    """Response model for pipeline information"""
    pipeline_name: str
    total_components: int
    components: List[Dict[str, str]]
    connections: List[Dict[str, str]]


@router.get("/health")
async def health_check():
    """Health check endpoint for visualization service"""
    try:
        # Check if mermaid server is accessible
        server_available = visualizer._check_local_server()
        
        return {
            "status": "healthy",
            "mermaid_server": "available" if server_available else "unavailable",
            "local_server_url": visualizer.local_mermaid_url
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@router.get("/pipelines")
async def list_available_pipelines():
    """List all available pipelines for visualization"""
    try:
        # Import pipeline creators
        from pipelines.rag_pipeline import create_rag_pipeline
        from pipelines.chat_pipeline import create_chat_pipeline
        
        pipelines = [
            {
                "name": "rag_pipeline",
                "description": "Retrieval-Augmented Generation Pipeline",
                "components": ["DocumentRetriever", "ModelManagerGenerator", "AnswerBuilder"],
                "type": "RAG"
            },
            {
                "name": "chat_pipeline", 
                "description": "Direct Chat Pipeline with ModelManager",
                "components": ["ModelManagerGenerator"],
                "type": "Chat"
            }
        ]
        
        return {
            "available_pipelines": pipelines,
            "total_count": len(pipelines)
        }
        
    except Exception as e:
        logger.error(f"Error listing pipelines: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list pipelines: {str(e)}")


@router.get("/pipeline/{pipeline_name}/info")
async def get_pipeline_info(pipeline_name: str) -> PipelineInfo:
    """Get detailed information about a specific pipeline"""
    try:
        pipeline = _load_pipeline(pipeline_name)
        if not pipeline:
            raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_name}' not found")
        
        info = visualizer.get_pipeline_info(pipeline)
        
        return PipelineInfo(
            pipeline_name=pipeline_name,
            total_components=info.get("total_components", 0),
            components=info.get("components", []),
            connections=info.get("connections", [])
        )
        
    except Exception as e:
        logger.error(f"Error getting pipeline info for {pipeline_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline info: {str(e)}")


@router.post("/pipeline/{pipeline_name}/visualize")
async def visualize_pipeline(
    pipeline_name: str,
    request: VisualizationRequest
):
    """Generate visualization for a specific pipeline"""
    try:
        pipeline = _load_pipeline(pipeline_name)
        if not pipeline:
            raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_name}' not found")
        
        # Create output directory
        output_dir = Path("visualizations")
        output_dir.mkdir(exist_ok=True)
        
        # Generate filename
        extension = "png" if request.format == "mermaid-image" else "mmd"
        filename = f"{pipeline_name}_{request.format}.{extension}"
        output_path = output_dir / filename
        
        # Generate visualization
        result = visualizer.visualize_pipeline(
            pipeline=pipeline,
            output_path=str(output_path),
            format=request.format,
            use_local_server=request.use_local_server,
            super_component_expansion=request.super_component_expansion
        )
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to generate visualization")
        
        # Return file info
        return {
            "pipeline_name": pipeline_name,
            "format": request.format,
            "output_path": str(output_path),
            "download_url": f"/visualize/download/{filename}",
            "file_size": output_path.stat().st_size if output_path.exists() else 0
        }
        
    except Exception as e:
        logger.error(f"Error visualizing pipeline {pipeline_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to visualize pipeline: {str(e)}")


@router.get("/download/{filename}")
async def download_visualization(filename: str):
    """Download a generated visualization file"""
    try:
        file_path = Path("visualizations") / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Visualization file not found")
        
        # Determine media type
        media_type = "image/png" if filename.endswith(".png") else "text/plain"
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type=media_type
        )
        
    except Exception as e:
        logger.error(f"Error downloading file {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download file: {str(e)}")


@router.get("/pipeline/{pipeline_name}/mermaid")
async def get_pipeline_mermaid(
    pipeline_name: str,
    super_expansion: bool = Query(True, description="Expand SuperComponents")
):
    """Get the Mermaid diagram text for a pipeline"""
    try:
        pipeline = _load_pipeline(pipeline_name)
        if not pipeline:
            raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_name}' not found")
        
        # Generate mermaid text
        output_dir = Path("visualizations")
        output_dir.mkdir(exist_ok=True)
        
        mermaid_file = output_dir / f"{pipeline_name}_mermaid.mmd"
        
        result = visualizer.visualize_pipeline(
            pipeline=pipeline,
            output_path=str(mermaid_file),
            format="mermaid-text",
            super_component_expansion=super_expansion
        )
        
        if not result or not mermaid_file.exists():
            raise HTTPException(status_code=500, detail="Failed to generate Mermaid diagram")
        
        # Read and return mermaid content
        mermaid_content = mermaid_file.read_text()
        
        return Response(
            content=mermaid_content,
            media_type="text/plain",
            headers={"Content-Disposition": f"inline; filename={pipeline_name}.mmd"}
        )
        
    except Exception as e:
        logger.error(f"Error generating Mermaid for {pipeline_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate Mermaid: {str(e)}")


@router.post("/mermaid/start-server")
async def start_mermaid_server(port: int = Query(3000, description="Port for Mermaid server")):
    """Start the local Mermaid.ink server"""
    try:
        success = visualizer.start_local_mermaid_server(port=port)
        
        if success:
            return {
                "status": "started",
                "server_url": f"http://localhost:{port}",
                "message": "Mermaid server started successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to start Mermaid server")
            
    except Exception as e:
        logger.error(f"Error starting Mermaid server: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start server: {str(e)}")


@router.post("/mermaid/stop-server")
async def stop_mermaid_server():
    """Stop the local Mermaid.ink server"""
    try:
        success = visualizer.stop_local_mermaid_server()
        
        if success:
            return {
                "status": "stopped",
                "message": "Mermaid server stopped successfully"
            }
        else:
            return {
                "status": "warning",
                "message": "Server may not have been running"
            }
            
    except Exception as e:
        logger.error(f"Error stopping Mermaid server: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop server: {str(e)}")


@router.get("/batch/all-pipelines")
async def visualize_all_pipelines(
    format: str = Query("mermaid-image", description="Output format"),
    super_expansion: bool = Query(True, description="Expand SuperComponents")
):
    """Generate visualizations for all available pipelines"""
    try:
        results = []
        pipeline_names = ["rag_pipeline", "chat_pipeline"]
        
        for pipeline_name in pipeline_names:
            try:
                pipeline = _load_pipeline(pipeline_name)
                if not pipeline:
                    results.append({
                        "pipeline_name": pipeline_name,
                        "status": "error",
                        "message": "Pipeline not found"
                    })
                    continue
                
                # Create output directory
                output_dir = Path("visualizations")
                output_dir.mkdir(exist_ok=True)
                
                # Generate filename
                extension = "png" if format == "mermaid-image" else "mmd"
                filename = f"{pipeline_name}_batch.{extension}"
                output_path = output_dir / filename
                
                # Generate visualization
                result = visualizer.visualize_pipeline(
                    pipeline=pipeline,
                    output_path=str(output_path),
                    format=format,
                    super_component_expansion=super_expansion
                )
                
                if result:
                    results.append({
                        "pipeline_name": pipeline_name,
                        "status": "success",
                        "output_path": str(output_path),
                        "download_url": f"/visualize/download/{filename}",
                        "file_size": output_path.stat().st_size if output_path.exists() else 0
                    })
                else:
                    results.append({
                        "pipeline_name": pipeline_name,
                        "status": "error",
                        "message": "Visualization generation failed"
                    })
                    
            except Exception as e:
                results.append({
                    "pipeline_name": pipeline_name,
                    "status": "error",
                    "message": str(e)
                })
        
        return {
            "batch_results": results,
            "total_processed": len(results),
            "successful": len([r for r in results if r["status"] == "success"]),
            "failed": len([r for r in results if r["status"] == "error"])
        }
        
    except Exception as e:
        logger.error(f"Error in batch visualization: {e}")
        raise HTTPException(status_code=500, detail=f"Batch visualization failed: {str(e)}")


@router.get("/interface", response_class=FileResponse)
async def get_visualization_interface():
    """
    Serve the HTML interface for pipeline visualization.
    
    Returns:
        FileResponse: The HTML interface file
    """
    try:
        interface_path = Path(__file__).parent / "static" / "pipeline-visualizer.html"
        
        if not interface_path.exists():
            raise HTTPException(
                status_code=404, 
                detail="Visualization interface not found"
            )
        
        return FileResponse(
            path=str(interface_path),
            media_type="text/html",
            filename="pipeline-visualizer.html"
        )
    except Exception as e:
        logger.error(f"Error serving visualization interface: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to serve interface: {str(e)}"
        )


def _load_pipeline(pipeline_name: str):
    """Helper function to load a pipeline by name"""
    try:
        if pipeline_name == "rag_pipeline":
            from pipelines.rag_pipeline import create_rag_pipeline
            return create_rag_pipeline()
        elif pipeline_name == "chat_pipeline":
            from pipelines.chat_pipeline import create_chat_pipeline
            return create_chat_pipeline()
        else:
            return None
    except Exception as e:
        logger.error(f"Error loading pipeline {pipeline_name}: {e}")
        return None


# Export router for inclusion in main FastAPI app
__all__ = ["router"] 