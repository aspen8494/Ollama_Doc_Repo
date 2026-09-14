#!/usr/bin/env python3
"""
Setup script for Ollama Document Repository.
Installs dependencies and verifies the Ollama connection.
"""

import subprocess
import sys
from pathlib import Path

# The Ollama endpoint to target. Override with OLLAMA_HOST in the environment.
import os
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://192.168.68.59:11435")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gemma3:4b")


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   OK  {description}")
            return True
        else:
            print(f"   X  {description}")
            print(f"    Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"   X  {description}")
        print(f"    Exception: {e}")
        return False


def check_python_version():
    """Check Python version is 3.9+."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"X  Python 3.9+ required, found {version.major}.{version.minor}")
        return False
    print(f"OK  Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_ollama():
    """Check if Ollama is running and reachable at the configured host."""
    import urllib.request
    try:
        response = urllib.request.urlopen(
            f"{OLLAMA_HOST}/api/tags", timeout=5)
        if response.status == 200:
            print(f"OK  Ollama reachable at {OLLAMA_HOST}")
            return True
    except Exception:
        pass
    print(f"X  Ollama not reachable at {OLLAMA_HOST}")
    print("  Make sure Ollama is running on the target host (ollama serve). "
        "Override the endpoint with OLLAMA_HOST=http://host:11435.")
    return False


def install_dependencies():
    """Install Python dependencies."""
    req_file = Path(__file__).parent / "requirements.txt"
    return run_command(
        f"{sys.executable} -m pip install -r {req_file}",
        "Installing Python dependencies",
    )


def pull_models():
    """Pull the Ollama models the app uses.

    Skipped when Ollama is unreachable or when the models are already present,
    so this does not block a local-only install.
    """
    all_ok = True
    for model in (EMBEDDING_MODEL, CHAT_MODEL):
        # Best-effort: only pull if the model is missing on the remote.
        try:
            import urllib.request, json
            base = OLLAMA_HOST.rstrip("/")
            resp = urllib.request.urlopen(f"{base}/api/tags", timeout=5)
            data = json.loads(resp.read())
            have = {m.get("model", "") for m in data.get("models", [])}
            want = model.replace(":latest", ":latest")
            if any(h.startswith(model.split(':')[0]) for h in have):
                print(f"OK  Model already present: {model}")
                continue
        except Exception:
            pass
        if not run_command(f"ollama pull {model}", f"Pulling model: {model}"):
            all_ok = False
    return all_ok


def create_directories():
    """Create necessary directories."""
    base = Path(__file__).parent
    dirs = [
        base / "documents",
        base / "chroma_db",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print("OK  Created directories")
    return True


def make_launcher_executable():
    """Make the launcher script executable."""
    launcher = Path(__file__).parent / "ollama-docs"
    if launcher.exists():
        launcher.chmod(0o755)
        print("OK  Made ollama-docs executable")
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
        ("Make launcher executable", make_launcher_executable),
    ]

    all_passed = True
    for name, check in checks:
        if not check():
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("OK  Setup complete!")
        print("\nNext steps:")
        print("   1. Add documents to ./documents/")
        print("   2. Ingest:  ./ollama-docs ingest")
        print("   3. Ask:     ./ollama-docs ask 'your question'")
        print("   4. TUI:     ./ollama-docs          (no subcommand)")
    else:
        print("X  Setup incomplete - see errors above")
        sys.exit(1)


if __name__ == '__main__':
    main()
