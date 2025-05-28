#!/usr/bin/env python3
"""
Test Preview and Validation Functionality

Test script to validate file preview and validation features.
"""

import os
import json
import csv
from pathlib import Path
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_files():
    """Create test files for validation and preview testing"""
    test_dir = Path("test_preview_files")
    test_dir.mkdir(exist_ok=True)
    
    files_created = {}
    
    # Valid TXT file
    txt_file = test_dir / "valid.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("This is a valid text file.\nIt has multiple lines.\nAnd some content for testing.")
    files_created['valid_txt'] = txt_file
    
    # Invalid TXT file (binary content)
    invalid_txt = test_dir / "invalid.txt"
    with open(invalid_txt, 'wb') as f:
        f.write(b'\x00\x01\x02\x03\x04\x05')  # Binary content
    files_created['invalid_txt'] = invalid_txt
    
    # Valid JSON file
    json_file = test_dir / "valid.json"
    data = {
        "name": "Test Document",
        "type": "validation_test",
        "items": [1, 2, 3, 4, 5],
        "metadata": {"created": "2024-01-01", "version": "1.0"}
    }
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    files_created['valid_json'] = json_file
    
    # Invalid JSON file
    invalid_json = test_dir / "invalid.json"
    with open(invalid_json, 'w') as f:
        f.write('{"invalid": json, "missing": quotes}')
    files_created['invalid_json'] = invalid_json
    
    # Valid CSV file
    csv_file = test_dir / "valid.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Age', 'City'])
        writer.writerow(['Alice', 25, 'New York'])
        writer.writerow(['Bob', 30, 'San Francisco'])
    files_created['valid_csv'] = csv_file
    
    # Empty file
    empty_file = test_dir / "empty.txt"
    empty_file.touch()
    files_created['empty_file'] = empty_file
    
    # Large file (exceeding size limit for testing)
    large_file = test_dir / "large.txt"
    with open(large_file, 'w') as f:
        # Write 1MB of text
        content = "This is a large file for testing size limits. " * 20000
        f.write(content)
    files_created['large_file'] = large_file
    
    # Valid Markdown file
    md_file = test_dir / "valid.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("""# Test Markdown Document

This is a **test** markdown file with various elements:

## Features
- Lists
- **Bold text**
- *Italic text*
- [Links](https://example.com)

### Code Block
```python
def hello():
    print("Hello, World!")
```

This tests markdown preview functionality.""")
    files_created['valid_md'] = md_file
    
    return files_created

def test_file_validation():
    """Test file validation functionality"""
    print("🧪 Testing File Validation")
    print("=" * 40)
    
    # Import validation function
    import sys
    sys.path.append('.')
    
    # Create a mock uploaded file class for testing
    class MockUploadedFile:
        def __init__(self, name, size, content=None, file_type=None):
            self.name = name
            self.size = size
            self.type = file_type
            self._content = content
            self._position = 0
        
        def read(self, size=-1):
            if self._content is None:
                return b""
            if isinstance(self._content, str):
                content = self._content.encode('utf-8')
            else:
                content = self._content
            
            if size == -1:
                result = content[self._position:]
                self._position = len(content)
            else:
                result = content[self._position:self._position + size]
                self._position += len(result)
            return result
        
        def seek(self, position):
            self._position = position
    
    # Test cases
    test_cases = [
        # Valid files
        ("valid.txt", 1000, "Valid text content", "text/plain", True),
        ("valid.json", 500, '{"valid": "json"}', "application/json", True),
        ("valid.csv", 300, "name,age\nAlice,25", "text/csv", True),
        ("valid.md", 400, "# Valid Markdown", "text/markdown", True),
        
        # Invalid files
        (None, 0, None, None, False),  # No file
        ("empty.txt", 0, "", "text/plain", False),  # Empty file
        ("large.txt", 60*1024*1024, "Large content", "text/plain", False),  # Too large
        ("noext", 100, "Content", "text/plain", False),  # No extension
        ("invalid.xyz", 100, "Content", "application/octet-stream", False),  # Unsupported format
        ("invalid.json", 100, '{"invalid": json}', "application/json", False),  # Invalid JSON
    ]
    
    # Import validation function (would need to be adapted for actual testing)
    try:
        from app import validate_file
        
        for name, size, content, file_type, expected_valid in test_cases:
            if name is None:
                mock_file = None
            else:
                mock_file = MockUploadedFile(name, size, content, file_type)
            
            try:
                is_valid, error_msg = validate_file(mock_file)
                
                if is_valid == expected_valid:
                    status = "✅ PASS"
                else:
                    status = "❌ FAIL"
                
                print(f"{status} {name or 'None'}: {is_valid} ({'Expected' if is_valid == expected_valid else 'Unexpected'})")
                if not is_valid and error_msg:
                    print(f"    Error: {error_msg}")
                    
            except Exception as e:
                print(f"❌ ERROR {name or 'None'}: {str(e)}")
    
    except ImportError as e:
        print(f"⚠️ Could not import validation function: {e}")
        print("This test requires the Streamlit app to be available")

def test_file_preview_content():
    """Test file preview content generation"""
    print("\n🔍 Testing File Preview Content")
    print("=" * 40)
    
    test_files = create_test_files()
    
    # Test each file type
    for file_type, file_path in test_files.items():
        print(f"\n📄 Testing {file_type}: {file_path.name}")
        
        try:
            file_size = file_path.stat().st_size
            print(f"   Size: {file_size} bytes")
            
            # Test basic file reading
            if file_path.suffix.lower() in ['.txt', '.md']:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    print(f"   ✅ Text content readable ({len(content)} chars)")
                    if len(content) > 100:
                        print(f"   Preview: {content[:100]}...")
                    else:
                        print(f"   Content: {content}")
                except Exception as e:
                    print(f"   ❌ Text reading failed: {e}")
            
            elif file_path.suffix.lower() == '.json':
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    print(f"   ✅ JSON valid ({type(data).__name__})")
                    if isinstance(data, dict):
                        print(f"   Keys: {list(data.keys())}")
                except Exception as e:
                    print(f"   ❌ JSON invalid: {e}")
            
            elif file_path.suffix.lower() == '.csv':
                try:
                    import pandas as pd
                    df = pd.read_csv(file_path)
                    print(f"   ✅ CSV valid ({df.shape[0]} rows, {df.shape[1]} cols)")
                    print(f"   Columns: {list(df.columns)}")
                except Exception as e:
                    print(f"   ❌ CSV reading failed: {e}")
            
        except Exception as e:
            print(f"   ❌ File access failed: {e}")

def test_content_validation():
    """Test content validation for different file types"""
    print("\n🔍 Testing Content Validation")
    print("=" * 40)
    
    test_files = create_test_files()
    
    validation_results = {}
    
    for file_type, file_path in test_files.items():
        print(f"\n📄 Validating {file_type}: {file_path.name}")
        
        try:
            # Simulate content validation
            if 'invalid' in file_type or 'empty' in file_type or 'large' in file_type:
                expected_valid = False
            else:
                expected_valid = True
            
            # Basic validation checks
            file_size = file_path.stat().st_size
            
            # Size check
            if file_size == 0:
                validation_results[file_type] = (False, "Empty file")
                print("   ❌ Empty file")
                continue
            elif file_size > 50 * 1024 * 1024:  # 50MB limit
                validation_results[file_type] = (False, "File too large")
                print("   ❌ File too large")
                continue
            
            # Extension check
            if '.' not in file_path.name:
                validation_results[file_type] = (False, "No extension")
                print("   ❌ No extension")
                continue
            
            # Content validation
            extension = file_path.suffix.lower()
            
            if extension in ['.txt', '.md']:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    validation_results[file_type] = (True, "Valid text")
                    print("   ✅ Valid text file")
                except UnicodeDecodeError:
                    validation_results[file_type] = (False, "Invalid encoding")
                    print("   ❌ Invalid text encoding")
            
            elif extension == '.json':
                try:
                    with open(file_path, 'r') as f:
                        json.load(f)
                    validation_results[file_type] = (True, "Valid JSON")
                    print("   ✅ Valid JSON")
                except json.JSONDecodeError:
                    validation_results[file_type] = (False, "Invalid JSON")
                    print("   ❌ Invalid JSON format")
            
            elif extension == '.csv':
                try:
                    import pandas as pd
                    df = pd.read_csv(file_path)
                    if df.empty:
                        validation_results[file_type] = (False, "Empty CSV")
                        print("   ❌ Empty CSV data")
                    else:
                        validation_results[file_type] = (True, "Valid CSV")
                        print("   ✅ Valid CSV")
                except Exception:
                    validation_results[file_type] = (False, "Invalid CSV")
                    print("   ❌ Invalid CSV format")
            
            else:
                validation_results[file_type] = (True, "Unknown format")
                print("   ⚠️ Unknown format, assuming valid")
                
        except Exception as e:
            validation_results[file_type] = (False, f"Error: {str(e)}")
            print(f"   ❌ Validation error: {e}")
    
    # Summary
    print(f"\n📊 Validation Summary:")
    valid_count = sum(1 for is_valid, _ in validation_results.values() if is_valid)
    total_count = len(validation_results)
    print(f"Valid files: {valid_count}/{total_count}")
    
    return validation_results

def cleanup_test_files():
    """Clean up test files"""
    test_dir = Path("test_preview_files")
    if test_dir.exists():
        import shutil
        shutil.rmtree(test_dir)
        print("🧹 Cleaned up test files")

def main():
    """Run all tests"""
    print("🧪 File Preview and Validation Test Suite")
    print("=" * 50)
    
    try:
        # Create test files
        print("📁 Creating test files...")
        test_files = create_test_files()
        print(f"✅ Created {len(test_files)} test files")
        
        # Test file validation
        test_file_validation()
        
        # Test file preview content
        test_file_preview_content()
        
        # Test content validation
        validation_results = test_content_validation()
        
        print("\n🎉 Test suite completed!")
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
    
    finally:
        # Cleanup
        cleanup_test_files()

if __name__ == "__main__":
    main() 