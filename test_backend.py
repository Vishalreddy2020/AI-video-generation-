"""
Quick test script to check if backend is running and accessible.
"""
import requests
import sys

def test_backend():
    url = "http://localhost:8000"
    
    print("Testing backend connection...")
    print(f"URL: {url}")
    print("-" * 50)
    
    try:
        # Test root endpoint
        print("1. Testing root endpoint (/):")
        response = requests.get(f"{url}/")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        print("   ✓ Root endpoint working\n")
        
        # Test health endpoint
        print("2. Testing health endpoint (/api/health):")
        response = requests.get(f"{url}/api/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        print("   ✓ Health endpoint working\n")
        
        print("=" * 50)
        print("✓ Backend is running and accessible!")
        print("=" * 50)
        return True
        
    except requests.exceptions.ConnectionError:
        print("✗ ERROR: Cannot connect to backend!")
        print("\nPossible issues:")
        print("  1. Backend server is not running")
        print("  2. Backend is running on a different port")
        print("  3. Firewall is blocking the connection")
        print("\nTo start the backend:")
        print("  cd backend")
        print("  venv\\Scripts\\activate")
        print("  python main.py")
        return False
        
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_backend()
    sys.exit(0 if success else 1)

