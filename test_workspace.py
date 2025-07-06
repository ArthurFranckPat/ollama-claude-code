#!/usr/bin/env python3
"""
Test script for workspace detection in ollama-claude-code proxy
This script demonstrates different ways to set and detect workspaces
"""

import requests
import json
import sys

PROXY_URL = "http://localhost:11435"

def test_endpoint(name, method, endpoint, headers=None, data=None):
    """Test an endpoint and display results"""
    print(f"\n{'='*50}")
    print(f"Test: {name}")
    print(f"{'='*50}")
    
    url = f"{PROXY_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers or {})
        else:
            response = requests.post(url, headers=headers or {}, json=data or {})
        
        print(f"Status: {response.status_code}")
        
        if response.headers.get('content-type', '').startswith('application/json'):
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
        else:
            print(f"Response: {response.text[:200]}...")
            
    except Exception as e:
        print(f"Error: {e}")

def main():
    print("🚀 Testing Ollama-Claude Proxy Workspace Detection")
    
    # Test 1: Check server info
    test_endpoint(
        "Server Information", 
        "GET", 
        "/"
    )
    
    # Test 2: Set workspace explicitly
    test_workspace = "/Users/arthur/Desktop"
    test_endpoint(
        "Set Workspace Explicitly",
        "POST",
        "/api/set-working-directory",
        data={"working_dir": test_workspace}
    )
    
    # Test 3: Get current workspace
    test_endpoint(
        "Get Current Workspace",
        "GET",
        "/api/get-working-directory"
    )
    
    # Test 4: Chat with x-project-path header
    test_endpoint(
        "Chat with x-project-path header",
        "POST",
        "/api/chat",
        headers={
            "Content-Type": "application/json",
            "x-project-path": test_workspace,
            "User-Agent": "ZedWorkspaceTest/1.0"
        },
        data={
            "model": "claude-sonnet-4",
            "messages": [
                {"role": "user", "content": "Quel est mon dossier de travail actuel ? Liste 3 fichiers/dossiers."}
            ]
        }
    )
    
    # Test 5: Chat with multiple workspace headers
    test_endpoint(
        "Chat with Multiple Workspace Headers",
        "POST", 
        "/api/chat",
        headers={
            "Content-Type": "application/json",
            "x-workspace": test_workspace,
            "x-working-directory": test_workspace,
            "workspace-root": test_workspace,
            "User-Agent": "Zed/workspace=/Users/arthur/Desktop"
        },
        data={
            "model": "claude-sonnet-4", 
            "messages": [
                {"role": "user", "content": "Confirme que tu travailles dans le bon dossier."}
            ]
        }
    )
    
    # Test 6: List all sessions
    test_endpoint(
        "List All Sessions",
        "GET",
        "/api/sessions"
    )
    
    print(f"\n{'='*50}")
    print("✅ All tests completed!")
    print("💡 You can now use any of these methods in Zed:")
    print("   - Send x-project-path header")
    print("   - Use /api/set-working-directory endpoint")
    print("   - Include workspace path in User-Agent")
    print("   - Mention working directory in prompts")
    print(f"{'='*50}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Test failed: {e}")
        sys.exit(1)