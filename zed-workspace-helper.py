#!/usr/bin/env python3
"""
Zed Workspace Helper for Ollama-Claude Proxy
This script helps detect and set the current workspace for Zed integration
"""

import os
import sys
import json
import time
import requests
import subprocess
from pathlib import Path

PROXY_URL = "http://localhost:11435"

def find_zed_workspaces():
    """Find potential Zed workspaces by looking for .zed directories and git repos"""
    workspaces = []
    
    # Common places to look for workspaces
    search_paths = [
        os.path.expanduser("~/Desktop"),
        os.path.expanduser("~/Documents"),
        os.path.expanduser("~/Projects"),
        os.path.expanduser("~/Code"),
        os.path.expanduser("~/Developer"),
        os.path.expanduser("~/src"),
    ]
    
    for base_path in search_paths:
        if not os.path.exists(base_path):
            continue
            
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path):
                # Check if it looks like a workspace
                workspace_markers = ['.git', '.zed', 'package.json', 'Cargo.toml', 
                                   'pyproject.toml', '.vscode', 'pom.xml', 'go.mod']
                
                for marker in workspace_markers:
                    if os.path.exists(os.path.join(item_path, marker)):
                        workspaces.append({
                            'path': item_path,
                            'name': item,
                            'marker': marker
                        })
                        break
    
    return workspaces

def get_current_workspace():
    """Try to detect the current active workspace"""
    
    # Method 1: Check if we're already in a git repository
    try:
        result = subprocess.run(['git', 'rev-parse', '--show-toplevel'], 
                              capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            git_root = result.stdout.strip()
            if os.path.exists(git_root):
                return git_root
    except:
        pass
    
    # Method 2: Look for workspace markers in current directory and parents
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        workspace_markers = ['.git', '.zed', 'package.json', 'Cargo.toml', 
                           'pyproject.toml', '.vscode']
        for marker in workspace_markers:
            if (parent / marker).exists():
                return str(parent)
    
    # Method 3: Use current directory
    return str(Path.cwd())

def set_workspace(workspace_path):
    """Set the workspace in the proxy"""
    try:
        response = requests.post(
            f"{PROXY_URL}/api/set-working-directory",
            json={"working_dir": workspace_path},
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Workspace set: {result['working_dir']}")
            print(f"   Session ID: {result['session_id']}")
            return True
        else:
            print(f"❌ Failed to set workspace: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error setting workspace: {e}")
        return False

def monitor_zed_workspace():
    """Monitor for Zed workspace changes (simplified version)"""
    print("🔍 Monitoring Zed workspace changes...")
    print("   (This is a basic version - press Ctrl+C to stop)")
    
    last_workspace = None
    
    try:
        while True:
            current = get_current_workspace()
            
            if current != last_workspace:
                print(f"\n📁 Workspace changed: {current}")
                if set_workspace(current):
                    last_workspace = current
                else:
                    print("   Failed to update proxy")
            
            time.sleep(2)  # Check every 2 seconds
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")

def main():
    if len(sys.argv) < 2:
        print("🚀 Zed Workspace Helper for Ollama-Claude Proxy")
        print("\nUsage:")
        print("  python3 zed-workspace-helper.py <command>")
        print("\nCommands:")
        print("  detect     - Detect current workspace and set it")
        print("  list       - List all potential workspaces")
        print("  set <path> - Set specific workspace path")
        print("  monitor    - Monitor workspace changes (basic)")
        print("  status     - Check proxy status")
        return
    
    command = sys.argv[1]
    
    if command == "detect":
        print("🔍 Detecting current workspace...")
        workspace = get_current_workspace()
        print(f"📁 Found workspace: {workspace}")
        
        if set_workspace(workspace):
            print("✅ Ready to use with Zed!")
        else:
            print("❌ Failed to configure proxy")
    
    elif command == "list":
        print("📋 Scanning for potential workspaces...")
        workspaces = find_zed_workspaces()
        
        if workspaces:
            print(f"\n✅ Found {len(workspaces)} potential workspaces:")
            for i, ws in enumerate(workspaces, 1):
                print(f"   {i}. {ws['name']} ({ws['marker']})")
                print(f"      {ws['path']}")
        else:
            print("❌ No workspaces found")
    
    elif command == "set":
        if len(sys.argv) < 3:
            print("❌ Please provide workspace path: set <path>")
            return
            
        workspace_path = sys.argv[2]
        if not os.path.exists(workspace_path):
            print(f"❌ Path does not exist: {workspace_path}")
            return
            
        print(f"📁 Setting workspace: {workspace_path}")
        set_workspace(workspace_path)
    
    elif command == "monitor":
        monitor_zed_workspace()
    
    elif command == "status":
        try:
            response = requests.get(f"{PROXY_URL}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Proxy server is running")
                
                # Get current sessions
                sessions_response = requests.get(f"{PROXY_URL}/api/sessions", timeout=5)
                if sessions_response.status_code == 200:
                    sessions = sessions_response.json()['sessions']
                    if sessions:
                        print(f"📊 Active sessions: {len(sessions)}")
                        for session_id, info in sessions.items():
                            wd = info.get('working_dir', 'Unknown')
                            print(f"   {session_id}: {wd}")
                    else:
                        print("📊 No active sessions")
            else:
                print(f"❌ Proxy server error: {response.status_code}")
        except:
            print("❌ Proxy server not reachable - make sure it's running on port 11435")
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Run without arguments to see usage")

if __name__ == "__main__":
    main()