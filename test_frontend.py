#!/usr/bin/env python
"""
Test script for frontend container
"""

import subprocess
import sys
import time
import requests
from pathlib import Path

def test_frontend_container():
    """Test the frontend container build and run"""
    
    print("🧪 Testing Frontend Container")
    print("=" * 50)
    
    # Test 1: Build the frontend container
    print("\n1️⃣ Building frontend container...")
    try:
        result = subprocess.run([
            "docker", "build", "-f", "Dockerfile.frontend", "-t", "potato-frontend-test", "."
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("✅ Container build successful")
        else:
            print("❌ Container build failed")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Build error: {e}")
        return False
    
    # Test 2: Run the container
    print("\n2️⃣ Running frontend container...")
    try:
        container_process = subprocess.Popen([
            "docker", "run", "--rm", "-p", "8502:8501", 
            "-v", f"{Path.cwd()}/models:/app/models",
            "-v", f"{Path.cwd()}/data:/app/data",
            "potato-frontend-test"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for container to start
        print("⏳ Waiting for container to start...")
        time.sleep(30)
        
        # Test 3: Check if frontend is accessible
        print("\n3️⃣ Testing frontend accessibility...")
        try:
            response = requests.get("http://localhost:8502/_stcore/health", timeout=10)
            if response.status_code == 200:
                print("✅ Frontend is accessible")
                success = True
            else:
                print(f"❌ Frontend not accessible (status: {response.status_code})")
                success = False
        except requests.exceptions.RequestException as e:
            print(f"❌ Frontend connection failed: {e}")
            success = False
        
        # Stop container
        container_process.terminate()
        container_process.wait()
        
        return success
        
    except Exception as e:
        print(f"❌ Container run error: {e}")
        return False

def test_docker_compose():
    """Test docker-compose frontend service"""
    
    print("\n🐳 Testing Docker Compose Frontend")
    print("=" * 50)
    
    try:
        # Start only frontend service
        print("Starting frontend service...")
        subprocess.run([
            "docker-compose", "up", "-d", "frontend"
        ], check=True, cwd=Path(__file__).parent)
        
        # Wait for service to be ready
        print("⏳ Waiting for service to be ready...")
        time.sleep(30)
        
        # Test accessibility
        try:
            response = requests.get("http://localhost:8501/_stcore/health", timeout=10)
            if response.status_code == 200:
                print("✅ Docker Compose frontend is accessible")
                success = True
            else:
                print(f"❌ Frontend not accessible (status: {response.status_code})")
                success = False
        except requests.exceptions.RequestException as e:
            print(f"❌ Frontend connection failed: {e}")
            success = False
        
        # Stop service
        subprocess.run([
            "docker-compose", "down"
        ], cwd=Path(__file__).parent)
        
        return success
        
    except Exception as e:
        print(f"❌ Docker Compose error: {e}")
        return False

if __name__ == "__main__":
    print("🥔 Potato Disease Classification - Frontend Container Test")
    print("=" * 60)
    
    # Test individual container
    container_success = test_frontend_container()
    
    # Test docker-compose
    compose_success = test_docker_compose()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)
    print(f"Individual Container: {'✅ PASS' if container_success else '❌ FAIL'}")
    print(f"Docker Compose:       {'✅ PASS' if compose_success else '❌ FAIL'}")
    
    if container_success and compose_success:
        print("\n🎉 All tests passed! Frontend container is working properly.")
        print("\nTo run the frontend:")
        print("docker-compose up frontend")
        print("Then visit: http://localhost:8501")
    else:
        print("\n❌ Some tests failed. Check the logs above for details.")
        sys.exit(1)
