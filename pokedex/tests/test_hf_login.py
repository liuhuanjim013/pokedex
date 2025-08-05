#!/usr/bin/env python3
"""
Test different methods of Hugging Face login to find the most reliable one.
"""

import os
import subprocess
from huggingface_hub import login, HfApi, whoami
from datasets import load_dataset

def validate_token(token: str) -> bool:
    """Validate Hugging Face token format."""
    if not token:
        return False
    token = token.strip()
    # Check basic format (should be "hf_..." and about 31-40 chars)
    if not token.startswith("hf_") or len(token) < 31 or len(token) > 40:
        return False
    return True

def test_internet_connection() -> bool:
    """Test internet connectivity to Hugging Face."""
    import socket
    try:
        # Try to connect to huggingface.co
        socket.create_connection(("huggingface.co", 443), timeout=5)
        print("✅ Internet connection available")
        return True
    except (socket.timeout, socket.gaierror):
        print("❌ No internet connection to Hugging Face")
        print("ℹ️ Please check your internet connection")
        return False

def test_login_methods():
    """Test different Hugging Face login methods."""
    # First check internet
    if not test_internet_connection():
        return False

    # Try different environment variables
    token = None
    for var in ["HF_TOKEN", "HUGGINGFACE_TOKEN"]:
        if var in os.environ:
            token = os.environ[var].strip()
            if validate_token(token):
                print(f"✅ Found valid token in {var}")
                break
            else:
                print(f"⚠️ Invalid token format in {var}")
    
    if not token:
        print("❌ No valid token found in environment")
        print("ℹ️ Token should:")
        print("  • Start with 'hf_'")
        print("  • Be about 35 characters long")
        print("  • Example format: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        return False

    # Show token info
    print(f"\n🔍 Testing Hugging Face login methods...")
    print(f"• Token length: {len(token)} characters")
    print(f"• Token format: {token[:6]}...{token[-4:]}")
    
    # Clear any existing token
    if "HF_TOKEN" in os.environ:
        del os.environ["HF_TOKEN"]
    token_file = os.path.expanduser("~/.huggingface/token")
    if os.path.exists(token_file):
        os.remove(token_file)
    
    # Method 1: Using CLI login
    print("\n1️⃣ Testing CLI login...")
    try:
        # Clear any existing tokens
        if os.path.exists(os.path.expanduser("~/.huggingface/token")):
            os.remove(os.path.expanduser("~/.huggingface/token"))
        if "HF_TOKEN" in os.environ:
            del os.environ["HF_TOKEN"]
            
        # Try CLI login with git credentials
        subprocess.run(["hf", "auth", "login", "--token", token, "--add-to-git-credential"], check=True)
        print("✅ CLI login successful")
        
        # Verify login worked
        api = HfApi()
        user = api.whoami()
        print(f"✅ Method 1 works! Logged in as: {user['name']} ({user['fullname']})")
    except subprocess.CalledProcessError as e:
        print(f"❌ CLI login failed: {e}")
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")

    # Method 2: Using HfApi directly
    print("\n2️⃣ Testing HfApi direct token...")
    try:
        # Clear any existing tokens
        if os.path.exists(os.path.expanduser("~/.huggingface/token")):
            os.remove(os.path.expanduser("~/.huggingface/token"))
        if "HF_TOKEN" in os.environ:
            del os.environ["HF_TOKEN"]
            
        # Try direct token
        api = HfApi(token=token, endpoint="https://huggingface.co")
        user = api.whoami()
        print(f"✅ Method 2 works! Logged in as: {user['name']} ({user['fullname']})")
    except Exception as e:
        print(f"❌ Method 2 failed: {e}")

    # Method 3: Using token file
    print("\n3️⃣ Testing token file...")
    try:
        # Clear any existing token
        if "HF_TOKEN" in os.environ:
            del os.environ["HF_TOKEN"]
            
        # Create token file
        token_dir = os.path.expanduser("~/.huggingface")
        token_file = os.path.join(token_dir, "token")
        os.makedirs(token_dir, exist_ok=True)
        
        # Write token to file
        with open(token_file, 'w') as f:
            f.write(token)
        print("✅ Token file created")
        
        # Try to verify token
        api = HfApi()  # Should use token file
        user = api.whoami()
        print(f"✅ Method 3 works! Token file is valid")
    except Exception as e:
        print(f"❌ Method 3 failed: {e}")
    finally:
        # Clean up token file
        if os.path.exists(token_file):
            os.remove(token_file)

    # Method 4: Using HF_TOKEN environment variable
    print("\n4️⃣ Testing HF_TOKEN variable...")
    try:
        # Clear any existing token file
        if os.path.exists(os.path.expanduser("~/.huggingface/token")):
            os.remove(os.path.expanduser("~/.huggingface/token"))
            
        # Set environment variable
        os.environ["HF_TOKEN"] = token
        
        # Try to verify token
        api = HfApi()  # Should use HF_TOKEN automatically
        user = api.whoami()
        print(f"✅ Method 4 works! HF_TOKEN is valid")
    except Exception as e:
        print(f"❌ Method 4 failed: {e}")
    finally:
        # Clean up environment
        if "HF_TOKEN" in os.environ:
            del os.environ["HF_TOKEN"]

    # Test actual dataset access
    print("\n🐉 Testing Pokemon dataset access...")
    try:
        # First verify we can list the dataset
        api = HfApi(token=token)
        
        # Try to access dataset directly (no datasets/ prefix)
        dataset = load_dataset("liuhuanjim013/pokemon-yolo-1025", token=token)
        print("✅ Successfully loaded Pokemon dataset!")
        print(f"• Training examples: {len(dataset['train'])}")
        
        # Try to list files in the dataset repo
        try:
            files = api.list_repo_files(
                repo_id="liuhuanjim013/pokemon-yolo-1025",
                repo_type="dataset",
                token=token
            )
            print("\n📂 Files in dataset:")
            for f in sorted(files)[:5]:  # Show first 5 files
                print(f"  • {f}")
            if len(files) > 5:
                print(f"  • ... and {len(files)-5} more files")
        except Exception as e:
            print(f"⚠️ Could not list files: {e}")
            
    except Exception as e:
        print(f"❌ Pokemon dataset access failed: {e}")
        print("\nℹ️ Common issues:")
        print("1. Token might be expired - generate a new one")
        print("2. Dataset might be private - check permissions")
        print("3. Network connectivity issues")
        
    # Print git credential helper status
    print("\n🔑 Git Credential Helper Status:")
    try:
        result = subprocess.run(
            ["git", "config", "--global", "credential.helper"],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            print(f"✅ Git credential helper is set to: {result.stdout.strip()}")
        else:
            print("⚠️ No git credential helper configured")
            print("ℹ️ To set up credential storage, run:")
            print("   git config --global credential.helper store")
    except Exception as e:
        print(f"❌ Could not check git config: {e}")

    print("\n📋 Summary of findings:")
    print("1. Direct login: Most reliable, but needs explicit call")
    print("2. HfApi: Good for API operations")
    print("3. Token file: Works for persistent storage")
    print("4. HF_TOKEN: Alternative environment variable")
    print("\nRecommended approach:")
    print("""
    import os
    from huggingface_hub import login
    
    def validate_hf_token(token: str) -> bool:
        '''Validate Hugging Face token format.'''
        if not token:
            return False
        token = token.strip()
        # Check basic format (should be "hf_..." and about 31-40 chars)
        if not token.startswith("hf_") or len(token) < 31 or len(token) > 40:
            return False
        return True
    
    def setup_hf_auth() -> bool:
        '''Set up Hugging Face authentication.'''
        # Try different environment variables
        token = None
        for var in ["HF_TOKEN", "HUGGINGFACE_TOKEN"]:
            if var in os.environ:
                token = os.environ[var].strip()
                if validate_hf_token(token):
                    break
        
        if not token:
            print("❌ No valid token found in environment")
            print("ℹ️ Token should:")
            print("  • Start with 'hf_'")
            print("  • Be about 35 characters long")
            print("  • Example: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            return False
            
        try:
            # Clear any existing token
            if "HF_TOKEN" in os.environ:
                del os.environ["HF_TOKEN"]
            token_file = os.path.expanduser("~/.huggingface/token")
            if os.path.exists(token_file):
                os.remove(token_file)
            
            # 1. Explicit login first
            login(token=token)
            
            # 2. Save to token file
            token_dir = os.path.expanduser("~/.huggingface")
            os.makedirs(token_dir, exist_ok=True)
            with open(os.path.join(token_dir, "token"), "w") as f:
                f.write(token)
            
            # 3. Set HF_TOKEN last (since it takes precedence)
            os.environ["HF_TOKEN"] = token
            
            return True
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            return False
    """)

if __name__ == "__main__":
    test_login_methods()