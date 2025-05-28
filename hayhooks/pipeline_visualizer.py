"""
Pipeline Visualizer for Haystack RAG System

Provides visualization capabilities for Haystack pipelines using both
native Haystack methods and local mermaid.ink server integration.
"""

import logging
import subprocess
import time
from typing import Optional, Dict, Any
from pathlib import Path

from haystack import Pipeline

logger = logging.getLogger(__name__)


class PipelineVisualizer:
    """
    Handles visualization of Haystack pipelines with multiple rendering options.
    """
    
    def __init__(self, local_mermaid_url: Optional[str] = None):
        """
        Initialize the pipeline visualizer.
        
        Args:
            local_mermaid_url: URL of local mermaid.ink server (e.g., "http://localhost:3000")
        """
        self.local_mermaid_url = local_mermaid_url or "http://localhost:3000"
        self.mermaid_container_name = "haystack-mermaid"
        
    def start_local_mermaid_server(self, port: int = 3000) -> bool:
        """
        Start a local mermaid.ink server using Docker.
        
        Args:
            port: Port to expose the mermaid server
            
        Returns:
            bool: True if server started successfully
        """
        try:
            # Check if container is already running
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={self.mermaid_container_name}", "--format", "{{.Names}}"],
                capture_output=True,
                text=True
            )
            
            if self.mermaid_container_name in result.stdout:
                logger.info(f"Mermaid server already running on port {port}")
                return True
            
            # Start new container
            cmd = [
                "docker", "run", "-d",
                "--name", self.mermaid_container_name,
                "--platform", "linux/amd64",
                "--publish", f"{port}:3000",
                "--cap-add=SYS_ADMIN",
                "ghcr.io/jihchi/mermaid.ink"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Wait for server to be ready
                time.sleep(3)
                logger.info(f"Mermaid server started successfully on port {port}")
                return True
            else:
                logger.error(f"Failed to start mermaid server: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting mermaid server: {e}")
            return False
    
    def stop_local_mermaid_server(self) -> bool:
        """
        Stop the local mermaid.ink server.
        
        Returns:
            bool: True if server stopped successfully
        """
        try:
            result = subprocess.run(
                ["docker", "stop", self.mermaid_container_name],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Remove container
                subprocess.run(["docker", "rm", self.mermaid_container_name], capture_output=True)
                logger.info("Mermaid server stopped successfully")
                return True
            else:
                logger.warning(f"Failed to stop mermaid server: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error stopping mermaid server: {e}")
            return False
    
    def visualize_pipeline(
        self,
        pipeline: Pipeline,
        output_path: Optional[str] = None,
        format: str = "mermaid-image",
        use_local_server: bool = True,
        super_component_expansion: bool = True
    ) -> Optional[str]:
        """
        Visualize a Haystack pipeline.
        
        Args:
            pipeline: The Haystack pipeline to visualize
            output_path: Path to save the visualization (optional)
            format: Output format ("mermaid-image" or "mermaid-text")
            use_local_server: Use local mermaid server instead of remote
            super_component_expansion: Show internal structure of SuperComponents
            
        Returns:
            str: Path to generated file or None if failed
        """
        try:
            server_url = self.local_mermaid_url if use_local_server else None
            
            if use_local_server and not self._check_local_server():
                logger.warning("Local mermaid server not available, starting it...")
                if not self.start_local_mermaid_server():
                    logger.warning("Failed to start local server, using remote mermaid.ink")
                    server_url = None
            
            if output_path:
                # Save to file
                pipeline.draw(
                    path=output_path,
                    format=format,
                    server_url=server_url,
                    super_component_expansion=super_component_expansion
                )
                logger.info(f"Pipeline visualization saved to: {output_path}")
                return output_path
            else:
                # Display inline (for Jupyter notebooks)
                pipeline.show(
                    server_url=server_url,
                    super_component_expansion=super_component_expansion
                )
                logger.info("Pipeline visualization displayed inline")
                return None
                
        except Exception as e:
            logger.error(f"Error visualizing pipeline: {e}")
            return None
    
    def _check_local_server(self) -> bool:
        """
        Check if local mermaid server is running.
        
        Returns:
            bool: True if server is accessible
        """
        try:
            import requests
            response = requests.get(self.local_mermaid_url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_pipeline_info(self, pipeline: Pipeline) -> Dict[str, Any]:
        """
        Get information about a pipeline structure.
        
        Args:
            pipeline: The Haystack pipeline to analyze
            
        Returns:
            dict: Pipeline information including components and connections
        """
        try:
            graph = pipeline.graph
            
            info = {
                "total_components": len(graph.nodes()),
                "components": [],
                "connections": [],
                "pipeline_name": getattr(pipeline, 'name', 'Unnamed Pipeline')
            }
            
            # Get component information
            for node_id in graph.nodes():
                component = graph.nodes[node_id]['instance']
                info["components"].append({
                    "id": node_id,
                    "type": type(component).__name__,
                    "class": f"{type(component).__module__}.{type(component).__name__}"
                })
            
            # Get connection information
            for edge in graph.edges():
                source, target = edge
                info["connections"].append({
                    "from": source,
                    "to": target
                })
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting pipeline info: {e}")
            return {"error": str(e)}


def create_visualizer(local_server: bool = True) -> PipelineVisualizer:
    """
    Factory function to create a pipeline visualizer.
    
    Args:
        local_server: Whether to use local mermaid server
        
    Returns:
        PipelineVisualizer: Configured visualizer instance
    """
    return PipelineVisualizer() if local_server else PipelineVisualizer(local_mermaid_url=None)


def visualize_rag_pipeline(output_dir: str = "visualizations") -> None:
    """
    Generate visualizations for our RAG pipeline.
    
    Args:
        output_dir: Directory to save visualizations
    """
    try:
        from pipelines.rag_pipeline import create_rag_pipeline
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Initialize visualizer
        visualizer = create_visualizer()
        
        # Create and visualize RAG pipeline
        rag_pipeline = create_rag_pipeline()
        
        # Generate different visualizations
        visualizations = [
            {
                "filename": f"{output_dir}/rag_pipeline.png",
                "format": "mermaid-image",
                "super_expansion": True,
                "description": "Complete RAG Pipeline with SuperComponent expansion"
            },
            {
                "filename": f"{output_dir}/rag_pipeline_simple.png", 
                "format": "mermaid-image",
                "super_expansion": False,
                "description": "Simplified RAG Pipeline view"
            },
            {
                "filename": f"{output_dir}/rag_pipeline.mmd",
                "format": "mermaid-text",
                "super_expansion": True,
                "description": "RAG Pipeline as Mermaid text"
            }
        ]
        
        for viz in visualizations:
            result = visualizer.visualize_pipeline(
                pipeline=rag_pipeline,
                output_path=viz["filename"],
                format=viz["format"],
                super_component_expansion=viz["super_expansion"]
            )
            
            if result:
                logger.info(f"✅ Generated: {viz['description']} -> {result}")
            else:
                logger.error(f"❌ Failed: {viz['description']}")
        
        # Generate pipeline info
        info = visualizer.get_pipeline_info(rag_pipeline)
        
        info_file = f"{output_dir}/rag_pipeline_info.json"
        import json
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"✅ Pipeline info saved to: {info_file}")
        
    except Exception as e:
        logger.error(f"Error generating RAG pipeline visualizations: {e}")


if __name__ == "__main__":
    # Demo: Generate visualizations for RAG pipeline
    visualize_rag_pipeline() 