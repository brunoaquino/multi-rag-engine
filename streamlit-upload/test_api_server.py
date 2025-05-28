#!/usr/bin/env python3
"""
Test API Server

Test script to validate all API endpoints and functionality.
"""

import requests
import json
import time
import os
from pathlib import Path
import tempfile

# API Configuration
API_BASE_URL = "http://localhost:8503"
TEST_FILES_DIR = "test_api_files"

def create_test_files():
    """Create test files for API testing"""
    test_dir = Path(TEST_FILES_DIR)
    test_dir.mkdir(exist_ok=True)
    
    files_created = {}
    
    # Create TXT file
    txt_file = test_dir / "test_document.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("API Test Document\n\nThis is a sample text document for testing the API endpoints.\n")
        f.write("It contains multiple lines and paragraphs.\n\n")
        f.write("The document should be processed correctly by the API.")
    files_created['txt'] = txt_file
    
    # Create JSON file
    json_file = test_dir / "test_data.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            "title": "Test Document",
            "content": "This is JSON content for API testing",
            "metadata": {
                "author": "API Test",
                "version": "1.0"
            }
        }, f, indent=2)
    files_created['json'] = json_file
    
    # Create Markdown file
    md_file = test_dir / "test_readme.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# API Test Document\n\n")
        f.write("This is a **markdown** document for testing.\n\n")
        f.write("## Features\n")
        f.write("- Document processing\n")
        f.write("- API endpoints\n")
        f.write("- Status tracking\n")
    files_created['md'] = md_file
    
    return files_created

def test_health_endpoint():
    """Test health check endpoint"""
    print("🔍 Testing health endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        
        print("✅ Health endpoint working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Health endpoint failed: {e}")
        return False

def test_root_endpoint():
    """Test root endpoint"""
    print("🔍 Testing root endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        
        print("✅ Root endpoint working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
        return False

def test_stats_endpoint():
    """Test stats endpoint"""
    print("🔍 Testing stats endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_jobs" in data
        assert "supported_formats" in data
        
        print("✅ Stats endpoint working correctly")
        print(f"   Supported formats: {data['supported_formats']}")
        return True
        
    except Exception as e:
        print(f"❌ Stats endpoint failed: {e}")
        return False

def test_upload_endpoint(test_files):
    """Test document upload endpoint"""
    print("🔍 Testing upload endpoint...")
    
    job_ids = []
    
    for file_type, file_path in test_files.items():
        try:
            print(f"   Uploading {file_type.upper()} file: {file_path.name}")
            
            with open(file_path, 'rb') as f:
                files = {'file': (file_path.name, f, 'text/plain')}
                data = {
                    'namespace': f'test_{file_type}',
                    'chunk_size': 256,
                    'chunk_overlap': 50,
                    'enable_cache': True
                }
                
                response = requests.post(f"{API_BASE_URL}/upload", files=files, data=data)
                assert response.status_code == 200
                
                result = response.json()
                assert "job_id" in result
                assert "filename" in result
                assert result["filename"] == file_path.name
                
                job_ids.append(result["job_id"])
                print(f"   ✅ Uploaded successfully: Job ID {result['job_id']}")
                
        except Exception as e:
            print(f"   ❌ Upload failed for {file_type}: {e}")
    
    print(f"✅ Upload endpoint working - {len(job_ids)} files uploaded")
    return job_ids

def test_status_endpoint(job_ids):
    """Test status tracking endpoint"""
    print("🔍 Testing status endpoint...")
    
    completed_jobs = []
    
    for job_id in job_ids:
        try:
            # Wait for processing to complete
            max_attempts = 30
            attempts = 0
            
            while attempts < max_attempts:
                response = requests.get(f"{API_BASE_URL}/status/{job_id}")
                assert response.status_code == 200
                
                status_data = response.json()
                assert "status" in status_data
                assert "progress" in status_data
                
                print(f"   Job {job_id}: {status_data['status']} ({status_data['progress']}%)")
                
                if status_data["status"] in ["completed", "failed"]:
                    if status_data["status"] == "completed":
                        completed_jobs.append(job_id)
                    break
                
                time.sleep(1)
                attempts += 1
            
            if attempts >= max_attempts:
                print(f"   ⚠️  Job {job_id} did not complete within timeout")
                
        except Exception as e:
            print(f"   ❌ Status check failed for {job_id}: {e}")
    
    print(f"✅ Status endpoint working - {len(completed_jobs)} jobs completed")
    return completed_jobs

def test_result_endpoint(completed_jobs):
    """Test result retrieval endpoint"""
    print("🔍 Testing result endpoint...")
    
    successful_results = 0
    
    for job_id in completed_jobs:
        try:
            response = requests.get(f"{API_BASE_URL}/result/{job_id}")
            assert response.status_code == 200
            
            result_data = response.json()
            assert "success" in result_data
            assert "filename" in result_data
            assert "chunk_count" in result_data
            
            if result_data["success"]:
                successful_results += 1
                print(f"   ✅ Job {job_id}: {result_data['chunk_count']} chunks generated")
            else:
                print(f"   ❌ Job {job_id} failed: {result_data.get('error_message', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ Result retrieval failed for {job_id}: {e}")
    
    print(f"✅ Result endpoint working - {successful_results} successful results")
    return successful_results

def test_chunks_endpoint(completed_jobs):
    """Test chunks retrieval endpoint"""
    print("🔍 Testing chunks endpoint...")
    
    chunks_retrieved = 0
    
    for job_id in completed_jobs:
        try:
            response = requests.get(f"{API_BASE_URL}/chunks/{job_id}?limit=5")
            assert response.status_code == 200
            
            chunks_data = response.json()
            assert isinstance(chunks_data, list)
            
            chunks_retrieved += len(chunks_data)
            print(f"   ✅ Job {job_id}: Retrieved {len(chunks_data)} chunks")
            
            if chunks_data:
                chunk = chunks_data[0]
                assert "chunk_id" in chunk
                assert "content" in chunk
                assert "word_count" in chunk
                print(f"      Sample chunk: {chunk['content'][:50]}...")
                
        except Exception as e:
            print(f"   ❌ Chunks retrieval failed for {job_id}: {e}")
    
    print(f"✅ Chunks endpoint working - {chunks_retrieved} chunks retrieved")
    return chunks_retrieved

def test_jobs_endpoint():
    """Test jobs listing endpoint"""
    print("🔍 Testing jobs listing endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/jobs")
        assert response.status_code == 200
        
        jobs_data = response.json()
        assert isinstance(jobs_data, list)
        
        print(f"✅ Jobs endpoint working - {len(jobs_data)} jobs listed")
        
        if jobs_data:
            job = jobs_data[0]
            assert "job_id" in job
            assert "filename" in job
            assert "status" in job
            print(f"   Sample job: {job['filename']} ({job['status']})")
        
        return True
        
    except Exception as e:
        print(f"❌ Jobs listing failed: {e}")
        return False

def test_api_integration():
    """Run comprehensive API integration test"""
    print("🚀 Starting API Integration Tests")
    print("=" * 50)
    
    # Create test files
    print("📁 Creating test files...")
    test_files = create_test_files()
    print(f"   Created {len(test_files)} test files")
    
    # Test basic endpoints
    health_ok = test_health_endpoint()
    root_ok = test_root_endpoint()
    stats_ok = test_stats_endpoint()
    
    if not all([health_ok, root_ok, stats_ok]):
        print("❌ Basic endpoints failed - stopping tests")
        return False
    
    # Test upload and processing workflow
    job_ids = test_upload_endpoint(test_files)
    if not job_ids:
        print("❌ Upload failed - stopping tests")
        return False
    
    completed_jobs = test_status_endpoint(job_ids)
    if not completed_jobs:
        print("❌ No jobs completed - stopping tests")
        return False
    
    successful_results = test_result_endpoint(completed_jobs)
    chunks_retrieved = test_chunks_endpoint(completed_jobs)
    jobs_listed = test_jobs_endpoint()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"   Files uploaded: {len(job_ids)}")
    print(f"   Jobs completed: {len(completed_jobs)}")
    print(f"   Successful results: {successful_results}")
    print(f"   Chunks retrieved: {chunks_retrieved}")
    print(f"   Jobs listing: {'✅' if jobs_listed else '❌'}")
    
    # Final verdict
    all_tests_passed = (
        len(job_ids) > 0 and
        len(completed_jobs) > 0 and
        successful_results > 0 and
        chunks_retrieved > 0 and
        jobs_listed
    )
    
    if all_tests_passed:
        print("\n🎉 All API tests passed successfully!")
        return True
    else:
        print("\n❌ Some API tests failed")
        return False

def cleanup_test_files():
    """Clean up test files"""
    test_dir = Path(TEST_FILES_DIR)
    if test_dir.exists():
        import shutil
        shutil.rmtree(test_dir)
        print(f"🧹 Cleaned up test files from {TEST_FILES_DIR}")

if __name__ == "__main__":
    try:
        # Wait for server to be ready
        print("⏳ Waiting for API server to be ready...")
        max_wait = 30
        wait_count = 0
        
        while wait_count < max_wait:
            try:
                response = requests.get(f"{API_BASE_URL}/health", timeout=2)
                if response.status_code == 200:
                    print("✅ API server is ready!")
                    break
            except:
                pass
            
            time.sleep(1)
            wait_count += 1
        
        if wait_count >= max_wait:
            print("❌ API server is not responding - make sure it's running on port 8503")
            exit(1)
        
        # Run tests
        success = test_api_integration()
        
        # Cleanup
        cleanup_test_files()
        
        exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        cleanup_test_files()
        exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        cleanup_test_files()
        exit(1) 