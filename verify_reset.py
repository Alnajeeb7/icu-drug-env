
from fastapi.testclient import TestClient
import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.app import app

client = TestClient(app)

def test_reset_no_body():
    print("Testing POST /reset with NO body...")
    # Send a POST request with no body at all
    response = client.post("/reset", content=None)
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {response.json()}")
    
    if response.status_code == 200:
        print("SUCCESS: Reset endpoint handled missing body correctly.")
    else:
        print(f"FAILURE: Expected 200, got {response.status_code}")
        sys.exit(1)

def test_reset_empty_dict():
    print("\nTesting POST /reset with empty dict body {}...")
    response = client.post("/reset", json={})
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {response.json()}")
    
    if response.status_code == 200:
        print("SUCCESS: Reset endpoint handled empty {} correctly.")
    else:
        print(f"FAILURE: Expected 200, got {response.status_code}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        test_reset_no_body()
        test_reset_empty_dict()
        print("\nAll verification tests PASSED ✓")
    except Exception as e:
        print(f"\nVerification FAILED with error: {e}")
        sys.exit(1)
