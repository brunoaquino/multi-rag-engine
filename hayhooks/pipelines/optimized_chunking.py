"""
Optimized Chunking Strategies for Document Processing

This module provides intelligent chunking strategies that adapt based on
content type and structure to improve retrieval accuracy and processing efficiency.
"""

import asyncio
import logging
import os
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import concurrent.futures
from haystack import Document
from haystack.components.preprocessors import DocumentSplitter

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Content type classifications for adaptive chunking"""
    TEXT = "text"
    CODE = "code"
    TECHNICAL_DOC = "technical"
    NARRATIVE = "narrative"
    FAQ = "faq"
    LIST = "list"
    TABLE = "table"
    MIXED = "mixed"


@dataclass
class ChunkingConfig:
    """Configuration for content-specific chunking strategies"""
    content_type: ContentType
    chunk_size: int
    chunk_overlap: int
    split_by: str
    preserve_structure: bool = True
    min_chunk_size: int = 100
    max_chunk_size: int = 1000
    
    # Content-specific parameters
    preserve_code_blocks: bool = False
    preserve_lists: bool = False
    preserve_tables: bool = False
    sentence_boundary_preference: bool = True


class ContentTypeDetector:
    """Detects content type for optimal chunking strategy"""
    
    def __init__(self):
        self.patterns = {
            ContentType.CODE: [
                r'```[\s\S]*?```',  # Code blocks
                r'def\s+\w+\s*\(',  # Python functions
                r'function\s+\w+\s*\(',  # JavaScript functions
                r'class\s+\w+\s*[{:]',  # Class definitions
                r'import\s+\w+',  # Import statements
                r'#include\s*<',  # C/C++ includes
            ],
            ContentType.FAQ: [
                r'Q:\s*.*?\nA:\s*',  # Q&A format
                r'Question:\s*.*?\nAnswer:\s*',  # Question/Answer format
                r'P:\s*.*?\nR:\s*',  # Portuguese Q&A
                r'Pergunta:\s*.*?\nResposta:\s*',  # Portuguese Question/Answer
            ],
            ContentType.LIST: [
                r'^\s*[-*•]\s+',  # Bullet points
                r'^\s*\d+\.\s+',  # Numbered lists
                r'^\s*[a-zA-Z]\.\s+',  # Lettered lists
            ],
            ContentType.TABLE: [
                r'\|.*\|.*\|',  # Markdown tables
                r'\+[-=]+\+',  # ASCII tables
                r'^\s*\w+\s*\t\s*\w+',  # Tab-separated data
            ],
            ContentType.TECHNICAL_DOC: [
                r'API\s+reference',
                r'configuration\s+parameters',
                r'installation\s+guide',
                r'error\s+codes?:',
                r'troubleshooting',
                r'specifications?:',
            ],
        }
    
    def detect_content_type(self, text: str) -> ContentType:
        """Detect the primary content type of a document"""
        text_lower = text.lower()
        scores = {content_type: 0 for content_type in ContentType}
        
        for content_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
                scores[content_type] += len(matches)
        
        # Calculate content type scores
        total_length = len(text)
        
        # Boost scores based on content characteristics
        if total_length > 0:
            # Code detection
            code_indicators = text.count('{') + text.count('}') + text.count(';')
            scores[ContentType.CODE] += code_indicators / total_length * 1000
            
            # List detection
            list_lines = len([line for line in text.split('\n') if re.match(r'^\s*[-*•]\s+|^\s*\d+\.\s+', line)])
            scores[ContentType.LIST] += list_lines / len(text.split('\n')) * 100
            
            # Technical document detection
            tech_words = ['api', 'configuration', 'parameter', 'installation', 'error', 'specification']
            tech_count = sum(text_lower.count(word) for word in tech_words)
            scores[ContentType.TECHNICAL_DOC] += tech_count / len(text.split()) * 100
        
        # Determine the highest scoring content type
        max_score = max(scores.values())
        if max_score == 0:
            return ContentType.TEXT  # Default fallback
        
        best_type = max(scores, key=scores.get)
        
        # If scores are close, consider it mixed content
        second_best_score = sorted(scores.values(), reverse=True)[1]
        if max_score > 0 and second_best_score / max_score > 0.7:
            return ContentType.MIXED
        
        return best_type


class OptimizedChunker:
    """Advanced document chunker with content-type specific strategies"""
    
    def __init__(self):
        self.detector = ContentTypeDetector()
        self.configs = self._get_default_configs()
    
    def _get_default_configs(self) -> Dict[ContentType, ChunkingConfig]:
        """Get optimized chunking configurations for each content type"""
        return {
            ContentType.TEXT: ChunkingConfig(
                content_type=ContentType.TEXT,
                chunk_size=512,
                chunk_overlap=50,
                split_by="sentence",
                sentence_boundary_preference=True
            ),
            ContentType.CODE: ChunkingConfig(
                content_type=ContentType.CODE,
                chunk_size=800,  # Larger chunks for code
                chunk_overlap=100,
                split_by="word",
                preserve_code_blocks=True,
                sentence_boundary_preference=False
            ),
            ContentType.TECHNICAL_DOC: ChunkingConfig(
                content_type=ContentType.TECHNICAL_DOC,
                chunk_size=600,
                chunk_overlap=80,
                split_by="sentence",
                preserve_structure=True,
                preserve_lists=True
            ),
            ContentType.NARRATIVE: ChunkingConfig(
                content_type=ContentType.NARRATIVE,
                chunk_size=400,  # Smaller chunks for better context
                chunk_overlap=40,
                split_by="sentence",
                sentence_boundary_preference=True
            ),
            ContentType.FAQ: ChunkingConfig(
                content_type=ContentType.FAQ,
                chunk_size=300,  # Keep Q&A pairs together
                chunk_overlap=30,
                split_by="sentence",
                preserve_structure=True
            ),
            ContentType.LIST: ChunkingConfig(
                content_type=ContentType.LIST,
                chunk_size=600,
                chunk_overlap=60,
                split_by="word",
                preserve_lists=True,
                preserve_structure=True
            ),
            ContentType.TABLE: ChunkingConfig(
                content_type=ContentType.TABLE,
                chunk_size=1000,  # Larger chunks to preserve table structure
                chunk_overlap=0,  # No overlap for tables
                split_by="word",
                preserve_tables=True,
                preserve_structure=True
            ),
            ContentType.MIXED: ChunkingConfig(
                content_type=ContentType.MIXED,
                chunk_size=512,
                chunk_overlap=50,
                split_by="sentence",
                preserve_structure=True,
                sentence_boundary_preference=True
            ),
        }
    
    def update_config(self, content_type: ContentType, config: ChunkingConfig):
        """Update chunking configuration for a specific content type"""
        self.configs[content_type] = config
        logger.info(f"Updated chunking config for {content_type.value}")
    
    def _preserve_special_structures(self, text: str, config: ChunkingConfig) -> List[Tuple[int, int, str]]:
        """Identify special structures that should be preserved"""
        preserved_ranges = []
        
        if config.preserve_code_blocks:
            # Find code blocks
            code_pattern = r'```[\s\S]*?```'
            for match in re.finditer(code_pattern, text):
                preserved_ranges.append((match.start(), match.end(), "code_block"))
        
        if config.preserve_lists:
            # Find list structures
            lines = text.split('\n')
            in_list = False
            list_start = 0
            
            for i, line in enumerate(lines):
                if re.match(r'^\s*[-*•]\s+|^\s*\d+\.\s+', line):
                    if not in_list:
                        list_start = text.find(line)
                        in_list = True
                elif in_list and line.strip() == '':
                    continue
                elif in_list:
                    # End of list
                    list_end = text.find(lines[i-1]) + len(lines[i-1])
                    preserved_ranges.append((list_start, list_end, "list"))
                    in_list = False
        
        if config.preserve_tables:
            # Find table structures
            table_pattern = r'\|.*\|.*\|'
            lines = text.split('\n')
            in_table = False
            table_start = 0
            
            for i, line in enumerate(lines):
                if re.match(table_pattern, line):
                    if not in_table:
                        table_start = text.find(line)
                        in_table = True
                elif in_table and not re.match(table_pattern, line) and line.strip() != '':
                    # End of table
                    table_end = text.find(lines[i-1]) + len(lines[i-1])
                    preserved_ranges.append((table_start, table_end, "table"))
                    in_table = False
        
        return preserved_ranges
    
    def _smart_split(self, text: str, config: ChunkingConfig) -> List[str]:
        """Perform intelligent splitting based on content type and configuration"""
        # First, identify structures to preserve
        preserved_ranges = self._preserve_special_structures(text, config)
        
        # Use standard Haystack splitter as base
        splitter = DocumentSplitter(
            split_by=config.split_by,
            split_length=config.chunk_size,
            split_overlap=config.chunk_overlap
        )
        
        # Create temporary document for splitting
        temp_doc = Document(content=text)
        result = splitter.run(documents=[temp_doc])
        
        chunks = []
        for doc in result["documents"]:
            chunk_text = doc.content
            
            # Apply content-specific optimizations
            if config.sentence_boundary_preference and config.split_by == "sentence":
                # Ensure chunks end at sentence boundaries when possible
                chunk_text = self._adjust_to_sentence_boundary(chunk_text)
            
            # Ensure minimum chunk size
            if len(chunk_text.split()) >= config.min_chunk_size // 4:  # Rough word count
                chunks.append(chunk_text)
            else:
                # Merge small chunks with previous chunk if possible
                if chunks:
                    chunks[-1] += " " + chunk_text
                else:
                    chunks.append(chunk_text)
        
        return chunks
    
    def _adjust_to_sentence_boundary(self, text: str) -> str:
        """Adjust chunk boundaries to sentence endings when possible"""
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) > 1:
            # Remove the last incomplete sentence
            complete_text = '.'.join(sentences[:-1]) + '.'
            return complete_text.strip()
        return text
    
    def chunk_document(self, document: Document) -> List[Document]:
        """Chunk a document using optimal strategy based on content type"""
        try:
            # Detect content type
            content_type = self.detector.detect_content_type(document.content)
            config = self.configs[content_type]
            
            logger.debug(f"Detected content type: {content_type.value} for document")
            
            # Perform optimized chunking
            chunks = self._smart_split(document.content, config)
            
            # Create Document objects for each chunk
            chunked_documents = []
            for i, chunk in enumerate(chunks):
                # Preserve original metadata and add chunking info
                chunk_meta = document.meta.copy() if document.meta else {}
                chunk_meta.update({
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "content_type": content_type.value,
                    "chunk_size": len(chunk),
                    "chunking_strategy": config.split_by
                })
                
                chunked_documents.append(Document(
                    content=chunk,
                    meta=chunk_meta
                ))
            
            logger.debug(f"Created {len(chunked_documents)} optimized chunks")
            return chunked_documents
            
        except Exception as e:
            logger.error(f"Error during optimized chunking: {e}")
            # Fallback to simple chunking
            return self._fallback_chunking(document)
    
    def _fallback_chunking(self, document: Document) -> List[Document]:
        """Fallback chunking strategy if optimization fails"""
        logger.warning("Using fallback chunking strategy")
        
        splitter = DocumentSplitter(
            split_by="word",
            split_length=512,
            split_overlap=50
        )
        
        result = splitter.run(documents=[document])
        return result["documents"]


class ParallelDocumentProcessor:
    """Parallel document processing for improved indexing performance"""
    
    def __init__(self, max_workers: int = None, batch_size: int = 50):
        self.max_workers = max_workers or min(4, (os.cpu_count() or 1) + 1)
        self.batch_size = batch_size
        self.chunker = OptimizedChunker()
    
    async def process_documents_async(self, documents: List[Document]) -> List[Document]:
        """Process documents asynchronously using thread pool"""
        try:
            loop = asyncio.get_event_loop()
            
            # Split documents into batches
            batches = [
                documents[i:i + self.batch_size] 
                for i in range(0, len(documents), self.batch_size)
            ]
            
            # Process batches in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    loop.run_in_executor(executor, self._process_batch, batch)
                    for batch in batches
                ]
                
                results = await asyncio.gather(*futures)
            
            # Flatten results
            processed_documents = []
            for batch_result in results:
                processed_documents.extend(batch_result)
            
            logger.info(f"Processed {len(processed_documents)} documents in {len(batches)} parallel batches")
            return processed_documents
            
        except Exception as e:
            logger.error(f"Error in parallel processing: {e}")
            # Fallback to sequential processing
            return self._process_sequential(documents)
    
    def _process_batch(self, documents: List[Document]) -> List[Document]:
        """Process a batch of documents (runs in thread pool)"""
        processed = []
        for doc in documents:
            try:
                chunked_docs = self.chunker.chunk_document(doc)
                processed.extend(chunked_docs)
            except Exception as e:
                logger.error(f"Error processing document in batch: {e}")
                processed.append(doc)  # Include original if chunking fails
        
        return processed
    
    def _process_sequential(self, documents: List[Document]) -> List[Document]:
        """Fallback sequential processing"""
        logger.warning("Using sequential processing fallback")
        processed = []
        for doc in documents:
            try:
                chunked_docs = self.chunker.chunk_document(doc)
                processed.extend(chunked_docs)
            except Exception as e:
                logger.error(f"Error processing document: {e}")
                processed.append(doc)
        
        return processed
    
    def process_documents_sync(self, documents: List[Document]) -> List[Document]:
        """Synchronous wrapper for async processing"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.process_documents_async(documents))
        except Exception as e:
            logger.error(f"Error in sync processing wrapper: {e}")
            return self._process_sequential(documents)
        finally:
            loop.close()


# Performance metrics collection
class ChunkingMetrics:
    """Collect and analyze chunking performance metrics"""
    
    def __init__(self):
        self.metrics = {
            "total_documents": 0,
            "total_chunks": 0,
            "avg_chunk_size": 0,
            "content_type_distribution": {},
            "processing_time": 0,
            "chunking_efficiency": 0
        }
    
    def record_chunking(self, documents: List[Document], chunks: List[Document], processing_time: float):
        """Record metrics from a chunking operation"""
        self.metrics["total_documents"] += len(documents)
        self.metrics["total_chunks"] += len(chunks)
        self.metrics["processing_time"] += processing_time
        
        # Calculate average chunk size
        if chunks:
            total_size = sum(len(chunk.content) for chunk in chunks)
            self.metrics["avg_chunk_size"] = total_size / len(chunks)
        
        # Track content type distribution
        for chunk in chunks:
            content_type = chunk.meta.get("content_type", "unknown")
            self.metrics["content_type_distribution"][content_type] = (
                self.metrics["content_type_distribution"].get(content_type, 0) + 1
            )
        
        # Calculate chunking efficiency (chunks per document)
        if self.metrics["total_documents"] > 0:
            self.metrics["chunking_efficiency"] = self.metrics["total_chunks"] / self.metrics["total_documents"]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return self.metrics.copy()
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics = {
            "total_documents": 0,
            "total_chunks": 0,
            "avg_chunk_size": 0,
            "content_type_distribution": {},
            "processing_time": 0,
            "chunking_efficiency": 0
        }


# Global metrics instance
chunking_metrics = ChunkingMetrics()

# Factory functions for easy integration
def create_optimized_chunker() -> OptimizedChunker:
    """Create an optimized chunker with default configurations"""
    return OptimizedChunker()

def create_parallel_processor(max_workers: int = None, batch_size: int = 50) -> ParallelDocumentProcessor:
    """Create a parallel document processor"""
    return ParallelDocumentProcessor(max_workers=max_workers, batch_size=batch_size)

# Example usage
if __name__ == "__main__":
    import time
    
    # Test content type detection
    detector = ContentTypeDetector()
    
    test_texts = {
        "code": "def hello_world():\n    print('Hello, World!')\n    return True",
        "faq": "Q: What is Python?\nA: Python is a programming language.",
        "list": "• Item 1\n• Item 2\n• Item 3",
        "text": "This is a regular text document with some sentences."
    }
    
    for content_type, text in test_texts.items():
        detected = detector.detect_content_type(text)
        print(f"Text type '{content_type}' detected as: {detected.value}")
    
    # Test optimized chunking
    chunker = OptimizedChunker()
    test_doc = Document(content="This is a test document. It has multiple sentences. We want to see how it gets chunked.")
    
    start_time = time.time()
    chunks = chunker.chunk_document(test_doc)
    end_time = time.time()
    
    print(f"\nChunked document into {len(chunks)} chunks in {end_time - start_time:.4f}s")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {chunk.content[:50]}...") 