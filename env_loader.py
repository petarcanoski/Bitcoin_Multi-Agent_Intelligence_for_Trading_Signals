"""env_loader.py

Small helper to explicitly load the repository root `.env` into the process
environment using python-dotenv. Importing this module is safe (it silently
does nothing if python-dotenv is not installed).

Usage: `import env_loader` near the top of your entrypoint scripts.
"""
from pathlib import Path


def _load():
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    root = Path(__file__).resolve().parent
    env_path = root / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path), override=False)


_load()

