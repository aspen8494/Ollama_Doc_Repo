#!/usr/bin/env python3
"""
Setup script for Ollama Document Repository.
Installs dependencies and verifies Ollama connection.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✓ {description}")
            return True
        else:
            print(f"  ✗ {description}")
            print(f"    Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ✗ {description}")
        print(f"    Exception: {e}")
        return False

def check_python_version():
    """Check Python version is 3.9+."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"✗ Python 3.9+ required, found {version.major}.{version.minor}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_ollama():
    """Check if Ollama is running and accessible."""
    import urllib.request
    try:
        response = urllib.request.urlopen("http://localhost:11435/api/tags", timeout=5)
        if response.status == 200:
            print("✓ Ollama is running on port 11435")
            return True
    except Exception:
        pass
    print("✗ Ollama not accessible on http://localhost:11435")
    print("  Make sure Ollama is running: ollama serve")
    return False

def install_dependencies():
    """Install Python dependencies."""
    req_file = Path(__file__).parent / "requirements.txt"
    return run_command(
        f"{sys.executable} -m pip install -r {req_file}",
        "Installing Python dependencies"
    )

def pull_models():
    """Pull required Ollama models."""
    models = ["nomic-embed-text", "llama3.2"]
    all_ok = True
    for model in models:
        if not run_command(f"ollama pull {model}", f"Pulling model: {model}"):
            all_ok = False
    return all_ok

def create_directories():
    """Create necessary directories."""
    dirs = [
        Path.home() / "Documents" / "Ollama_Doc_Repo" / "documents",
        Path.home() / "Documents" / "Ollama_Doc_Repo" / "chroma_db"
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print("✓ Created directories")
    return True

def main():
    print("=" * 50)
    print("Ollama Document Repository - Setup")
    print("=" * 50)
    
    checks = [
        ("Python version", check_python_version),
        ("Ollama connection", check_ollama),
        ("Create directories", create_directories),
        ("Install dependencies", install_dependencies),
        ("Pull Ollama models", pull_models),
    ]
    
    all_passed = True
    for name, check in checks:
        if not check():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ Setup complete!")
        print("\nNext steps:")
        print("  1. Add documents to ~/Documents/Ollama_Doc_Repo/documents/")
        print("  2. Run: python -m src.main ingest")
        print("  3. Ask questions: python -m src.main ask 'your question'")
    else:
        print("✗ Setup incomplete - see errors above")
        sys.exit(1)

if __name__ == '__main__':
    main()