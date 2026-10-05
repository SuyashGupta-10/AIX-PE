# AIX-PE: Hinglish child-care triage (PR 4)

**Burn Sahayak** is a conversational assistant that acts as the first point of contact for child burns in rural India.
It understands Hinglish, Devanagari Hindi and messy code-mixed text, voice and photos. It gives immediate WHO
first aid, asks only the questions that are still unanswered, and triages each case **RED / YELLOW / GREEN**.
RED cases are routed to a PHC doctor (a human handoff), and follow-up calls are scheduled.

Language understanding uses **Qwen3** for text and **Qwen3-VL** for photos. Severity is decided by fixed rules, not by the model.

## Quick start

```bash
cd burn_triage
pip install -r requirements.txt
cp .env.example .env      # choose a backend: transformers (local), openai (Ollama/vLLM), or none (rules only)
uvicorn app.main:app --port 8000
```

- Chat: http://localhost:8000
- Staff dashboard (escalations, follow-ups): http://localhost:8000/staff
- Tests: `python tests/test_burn_triage.py`

> On a CPU, Qwen3 replies take about 20–60 s. Use Ollama or a GPU for speed, or `QWEN_BACKEND=none` for an instant demo.

## Repo layout

```
burn_triage/
  app/          FastAPI server, dialogue engine, Qwen3 client, triage protocol, web UI
  tests/        unit tests + Qwen evaluation
  data/         sample PHC directory
  README.md     full documentation: design, safety choices, API, results
```

See [burn_triage/README.md](burn_triage/README.md) for details.

> ⚠️ Course prototype. The medical content needs clinician review before any real use. The PHC data is sample data.
