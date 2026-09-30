# Burn Sahayak: Hinglish triage assistant for child burns

Problem statement **PR 4**: *Automated mixed-code (Hinglish) triaging for childcare in rural India*.
This first version covers **burns and scalds**. It is a conversational first point of contact
(text, voice or photo) that understands Hinglish, Devanagari Hindi and messy code-mixed input. It gives
immediate WHO first-aid steps, asks only for the information that's still missing, triages
**RED / YELLOW / GREEN**, routes RED cases to a PHC doctor (human handoff) and schedules follow-up calls.

Language understanding uses **Qwen3** (text) and **Qwen3-VL** (photos).

> ⚠️ Prototype for a course project. The medical content in `app/protocol.py` follows WHO burn first-aid
> guidance and standard paediatric referral criteria, but it must be reviewed by a clinician before any real use.
> The PHC directory in `data/phc_directory.json` is sample data.

## Flow (matches the design flowcharts)

```
caller ──► voice (browser ASR hi-IN) / text / photo
             │
             ▼
   2. Understand & classify ── Qwen3 JSON extraction  ─┐
                               keyword rules (Hinglish) ┴─► safety merge ─► case
             │
   3. Emergency? (breathing, unconscious, electrical, chemical, deep, >10%, face+flame, ...)
       ├─ yes ─► 108 / PHC now + first aid + escalation queue (RED) ─► details ─► follow-up in 2 h
       └─ no  ─► cooling advice first ─► 4. targeted questions (only missing fields, max 2 at a time)
                    ─► 5/6. triage (deterministic) + first aid + when to seek care
                    ─► 7. name / phone / village (saved; returning callers recognised)
                    ─► 8. follow-up scheduled (YELLOW: next day, GREEN: 2 days)
   9-11. follow-up call: pain / blister trend, fever / pus / spreading redness, movement
         ─► worse → "PHC aaj hi" (+ escalate) · better → continue care · healed → close
```

Other conditions (snake bite, dog bite, diarrhoea, fever, fall) get safe first advice and a handoff
message, since this build only triages burns.

## Why the LLM doesn't decide severity

Small Qwen3 models (0.6B–1.7B) sometimes invent details. In testing, Qwen3-1.7B turned
*"मेरे बच्चे को जल गया है"* into "scald on the hand, superficial". A made-up "superficial" could produce a
false GREEN, so the design is:

| Layer | Job |
|---|---|
| `nlu_rules.py` | Hinglish/Devanagari keyword extractor with negation ("saans mein dikkat **nahi**"), word boundaries (`jaldi` ≠ `jal`, `sirf` ≠ `sir`), ages ("dedh saal", "6 mahine", "३ साल"), palm-rule sizes |
| `nlu.py` | Qwen3 JSON extraction, schema-validated. **Keyword evidence wins on conflict.** Qwen-only values are accepted for the question just asked, or when they raise safety (`LLM_TRUST=pending`). Numbers not present in the text are dropped. Red flags from either source are OR-ed |
| `protocol.py` | Deterministic triage + all medical text. The LLM never writes medical advice |
| Vision | Qwen3-VL can only **raise** severity (blisters / deep burn / infection). A reassuring photo reading doesn't skip the questions |

Qwen3 is also used for intent classification, answering free-form questions grounded in the protocol text
("kya haldi laga sakte hain?"), and turning replies into Devanagari for Devanagari callers.

## Run

```bash
cd burn_triage
pip install -r requirements.txt
cp .env.example .env          # pick a backend (see below)
uvicorn app.main:app --port 8000
```

Open http://localhost:8000 for the caller chat and http://localhost:8000/staff for the PHC/ANM dashboard
(escalation queue, follow-ups due, transcripts).

**Backends** (`QWEN_BACKEND`):
- `transformers` (default): downloads `Qwen/Qwen3-1.7B` and `Qwen/Qwen3-VL-2B-Instruct` from Hugging Face.
  Works on CPU (~10–20 s per turn on a laptop CPU); use `Qwen/Qwen3-4B`/`8B` with `LLM_TRUST=full` on a GPU.
- `openai`: any OpenAI-compatible server with Qwen3, e.g. Ollama (`ollama pull qwen3:4b`), vLLM, DashScope.
- `none`: rules only. Fully offline and instant; used by the tests.

If the LLM is unreachable, the app keeps working on rules alone (logged, and shown in `/api/health`).

Voice uses the browser's Web Speech API (`hi-IN` handles Hinglish speech in Chrome/Edge) for speech-to-text
and read-aloud. For IVR/WhatsApp voice, plug a server-side ASR (e.g. AI4Bharat IndicConformer / Whisper) into
`/api/chat`.

## Test

```bash
python tests/test_burn_triage.py               # 15 deterministic tests, rules-only, <1 s (pytest also works)
python tests/eval_qwen.py                      # Qwen vs rules on 14 Hinglish messages (needs the model)
```

`eval_qwen.py` results on this laptop's CPU (no GPU), 14 Hinglish/Devanagari messages:

| | Qwen alone | Rules alone | Merged (what the app uses) | Time per call |
|---|---|---|---|---|
| Qwen3-1.7B | 7/14 (invents details on vague messages) | 14/14 | 14/14 | ~10–20 s |
| Qwen3-4B | 13/14 (no invented details) | 14/14 | 14/14 | ~35–55 s |

Note: the rules scoring 14/14 is partly because the lexicon was tuned on these messages; unseen phrasing is
where Qwen earns its place. On a GPU or through Ollama, use Qwen3-4B/8B with `LLM_TRUST=full`.

The tests cover the brief's three burn prompts (Devanagari / Roman / messy code-mixed), negation and
false-positive traps, every red flag → RED + 108, GREEN/YELLOW cases, "pata nahi" answers not blocking the
flow, full intake → follow-up (better / worse / infection), handoff for non-burn conditions, and the rule that
photos can't lower severity.

## API

| Method | Path | Body |
|---|---|---|
| POST | `/api/session` | `{"mode": "intake" \| "followup", "phone": "98…"}` |
| POST | `/api/chat` | `{"session_id", "message"}` → `reply`, `triage`, `case`, `pending`, `debug` |
| POST | `/api/image` | multipart `session_id`, `file`, optional `message` |
| GET | `/api/dashboard` | escalations, follow-ups, recent cases |
| POST | `/api/escalations/{id}/ack` | staff acknowledges a RED case |
| GET | `/api/health` | backend / model / last LLM error |

## Files

```
app/protocol.py    triage rules, question bank, first-aid text (clinical content lives here only)
app/nlu_rules.py   Hinglish keyword extractor
app/nlu.py         Qwen3 prompts, validation, safety merge, FAQ, Devanagari, photo assessment
app/llm.py         Qwen3 / Qwen3-VL client (transformers or OpenAI-compatible), thinking disabled
app/dialogue.py    conversation state machine (intake + follow-up)
app/db.py          SQLite: users, cases, transcripts, escalations, follow-ups; PHC lookup
app/main.py        FastAPI
app/static/        caller chat UI + staff dashboard
tests/             unit tests + Qwen evaluation
```

## Limitations / next steps
- Sessions live in memory (a server restart ends open chats; saved cases stay in SQLite).
- Real PHC directory + telephony (Exotel/Twilio IVR, WhatsApp Business API) for the 3-way call and outbound follow-ups.
- Fine-tune or evaluate larger Qwen3 on real (de-identified) helpline transcripts; validate Qwen3-VL on a
  clinical burn-image set before trusting photo depth estimates.
- Add snake bite, dog bite and diarrhoea protocols using the same `protocol.py` pattern.
