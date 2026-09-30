"""Runtime configuration, read from environment variables (or a .env file)."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env = BASE_DIR / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        val = val.split(" #", 1)[0]  # inline comment
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv()

# QWEN_BACKEND:
#   "openai"       -> any OpenAI-compatible server hosting Qwen3 (Ollama, vLLM, LM Studio,
#                     OpenRouter, Alibaba DashScope). Set QWEN_BASE_URL / QWEN_API_KEY.
#   "transformers" -> run Qwen3 locally with Hugging Face transformers (CPU or GPU).
#   "none"         -> rules only (no LLM). Useful for tests / offline demo.
QWEN_BACKEND = os.getenv("QWEN_BACKEND", "transformers").lower()

# Text model (Hinglish understanding + replies)
QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen3-1.7B")
# Vision model (burn photo assessment). Empty string disables image analysis.
QWEN_VL_MODEL = os.getenv("QWEN_VL_MODEL", "Qwen/Qwen3-VL-2B-Instruct")

QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "http://localhost:11434/v1")
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "ollama")

# How far to trust Qwen's extraction when the keyword rules found nothing for a field:
#   "pending" -> only for the question just asked, or when it raises safety (default; right for 0.6B-1.7B)
#   "full"    -> accept everything the model extracts (use with Qwen3-4B/8B or larger)
LLM_TRUST = os.getenv("LLM_TRUST", "pending").lower()

# Reply in Devanagari when the caller writes in Devanagari (Qwen converts our Roman-Hinglish templates).
#   "auto" -> on for the openai backend or a GPU; off on CPU, where converting a long reply takes minutes
#   "on" / "off"
LOCALIZE_REPLIES = os.getenv("LOCALIZE_REPLIES", "auto").lower()

LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))

DB_PATH = Path(os.getenv("DB_PATH", str(BASE_DIR / "data" / "triage.db")))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "data" / "uploads")))
PHC_DIRECTORY = BASE_DIR / "data" / "phc_directory.json"

EMERGENCY_NUMBER = os.getenv("EMERGENCY_NUMBER", "108")
