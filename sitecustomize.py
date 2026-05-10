
from pathlib import Path

def _load_root_dotenv():
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    root = Path(__file__).resolve().parent
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path), override=False)


_load_root_dotenv()


