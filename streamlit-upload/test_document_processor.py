#!/usr/bin/env python3
"""
Test Document Processor

Test script to validate multi-format document processing functionality.
"""

import os
import json
import csv
from pathlib import Path
from document_processor import DocumentProcessor, create_document_processor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_files():
    """Create test files for each supported format"""
    test_dir = Path("test_files")
    test_dir.mkdir(exist_ok=True)
    
    # Create TXT file
    txt_file = test_dir / "sample.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("""Document Processing Test
        
This is a sample text document for testing the document processor.
It contains multiple paragraphs with various types of content.

The processor should be able to extract this text and create appropriate chunks
for further processing in the RAG pipeline.

This document serves as a basic test case for TXT format processing.""")
    
    # Create Markdown file
    md_file = test_dir / "sample.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("""# Markdown Test Document

This is a **markdown** document for testing purposes.

## Features

- Text extraction
- Metadata parsing
- Chunk generation

### Code Example

```python
def hello_world():
    print("Hello, World!")
```

This document tests markdown processing capabilities.""")
    
    # Create JSON file
    json_file = test_dir / "sample.json"
    data = {
        "title": "JSON Test Document",
        "description": "Sample JSON data for testing",
        "items": [
            {"id": 1, "name": "Item One", "value": 100},
            {"id": 2, "name": "Item Two", "value": 200},
            {"id": 3, "name": "Item Three", "value": 300}
        ],
        "metadata": {
            "created": "2024-01-01",
            "version": "1.0",
            "author": "Test Author"
        }
    }
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Create CSV file
    csv_file = test_dir / "sample.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Name', 'Age', 'City', 'Occupation'])
        writer.writerow([1, 'Alice Johnson', 28, 'New York', 'Engineer'])
        writer.writerow([2, 'Bob Smith', 34, 'San Francisco', 'Designer'])
        writer.writerow([3, 'Carol Davis', 29, 'Chicago', 'Manager'])
        writer.writerow([4, 'David Wilson', 31, 'Boston', 'Developer'])
        writer.writerow([5, 'Eva Brown', 26, 'Seattle', 'Analyst'])
    
    return {
        'txt': txt_file,
        'md': md_file,
        'json': json_file,
        'csv': csv_file
    }

def test_format_processing(processor, test_files):
    """Test processing for each format"""
    results = {}
    
    for format_name, file_path in test_files.items():
        print(f"\n🔍 Testing {format_name.upper()} processing...")
        
        try:
            result = processor.process_document(str(file_path), namespace="test")
            
            if result.success:
                print(f"✅ Successfully processed {file_path.name}")
                print(f"   📊 Words: {result.metadata.word_count}")
                print(f"   📄 Characters: {result.metadata.character_count}")
                print(f"   🧩 Chunks: {len(result.chunks)}")
                print(f"   ⏱️ Time: {result.processing_time:.3f}s")
                
                if result.metadata.title:
                    print(f"   📑 Title: {result.metadata.title}")
                
                # Show first chunk preview
                if result.chunks:
                    first_chunk = result.chunks[0]
                    preview = first_chunk.content[:100] + "..." if len(first_chunk.content) > 100 else first_chunk.content
                    print(f"   📝 First chunk preview: {preview}")
                
                results[format_name] = {
                    'success': True,
                    'word_count': result.metadata.word_count,
                    'chunk_count': len(result.chunks),
                    'processing_time': result.processing_time
                }
            else:
                print(f"❌ Failed to process {file_path.name}: {result.error_message}")
                results[format_name] = {'success': False, 'error': result.error_message}
                
        except Exception as e:
            print(f"❌ Exception processing {file_path.name}: {str(e)}")
            results[format_name] = {'success': False, 'error': str(e)}
    
    return results

def test_chunking_parameters():
    """Test different chunking parameters"""
    print("\n🧩 Testing chunking parameters...")
    
    # Create a longer text file
    test_dir = Path("test_files")
    long_text_file = test_dir / "long_text.txt"
    
    # Generate longer text
    long_text = " ".join([f"This is sentence number {i} in a longer document." for i in range(1, 201)])
    with open(long_text_file, 'w') as f:
        f.write(long_text)
    
    # Test different chunk sizes
    chunk_configs = [
        {'chunk_size': 50, 'chunk_overlap': 10},
        {'chunk_size': 100, 'chunk_overlap': 20},
        {'chunk_size': 200, 'chunk_overlap': 50}
    ]
    
    for config in chunk_configs:
        processor = DocumentProcessor(
            chunk_size=config['chunk_size'],
            chunk_overlap=config['chunk_overlap']
        )
        
        result = processor.process_document(str(long_text_file), namespace="test")
        
        if result.success:
            print(f"   📊 Chunk size {config['chunk_size']}, overlap {config['chunk_overlap']}: {len(result.chunks)} chunks")
            
            # Show chunk word counts
            chunk_words = [chunk.word_count for chunk in result.chunks]
            avg_words = sum(chunk_words) / len(chunk_words)
            print(f"      Average words per chunk: {avg_words:.1f}")
        else:
            print(f"   ❌ Failed with config {config}: {result.error_message}")

def test_error_handling():
    """Test error handling for invalid files"""
    print("\n🚨 Testing error handling...")
    
    processor = DocumentProcessor()
    
    # Test non-existent file
    result = processor.process_document("non_existent_file.txt", namespace="test")
    assert not result.success, "Should fail for non-existent file"
    print("✅ Correctly handled non-existent file")
    
    # Test unsupported format
    test_dir = Path("test_files")
    unsupported_file = test_dir / "test.xyz"
    unsupported_file.write_text("test content")
    
    result = processor.process_document(str(unsupported_file), namespace="test")
    assert not result.success, "Should fail for unsupported format"
    print("✅ Correctly handled unsupported format")
    
    # Clean up
    unsupported_file.unlink()

def test_processor_stats():
    """Test processor statistics"""
    print("\n📊 Testing processor statistics...")
    
    processor = DocumentProcessor(chunk_size=256, chunk_overlap=32)
    stats = processor.get_processing_stats()
    
    print(f"   Supported formats: {stats['supported_formats']}")
    print(f"   Chunk size: {stats['chunk_size']}")
    print(f"   Chunk overlap: {stats['chunk_overlap']}")
    print(f"   Dependencies: {stats['dependencies']}")

def main():
    """Run all tests"""
    print("🧪 Document Processor Test Suite")
    print("=" * 50)
    
    # Create test files
    print("📁 Creating test files...")
    test_files = create_test_files()
    print(f"✅ Created {len(test_files)} test files")
    
    # Create processor
    processor = create_document_processor(chunk_size=100, chunk_overlap=20)
    
    # Test format processing
    results = test_format_processing(processor, test_files)
    
    # Test chunking parameters
    test_chunking_parameters()
    
    # Test error handling
    test_error_handling()
    
    # Test processor stats
    test_processor_stats()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 30)
    successful_formats = [fmt for fmt, result in results.items() if result.get('success', False)]
    failed_formats = [fmt for fmt, result in results.items() if not result.get('success', False)]
    
    print(f"✅ Successful formats: {successful_formats}")
    if failed_formats:
        print(f"❌ Failed formats: {failed_formats}")
    
    total_chunks = sum(result.get('chunk_count', 0) for result in results.values() if result.get('success', False))
    print(f"📊 Total chunks generated: {total_chunks}")
    
    print("\n🎉 Test suite completed!")

if __name__ == "__main__":
    main() 