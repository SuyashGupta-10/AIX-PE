"""FastAPI server: chat API, image upload, staff dashboard, static web UI.

Run:  uvicorn app.main:app --port 8000   (from the burn_triage folder)
"""
from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, db, dialogue
from .llm import get_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = FastAPI(title="Burn Sahayak - Hinglish child burn triage")
STATIC = config.BASE_DIR / "app" / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")

MAX_IMAGE_BYTES = 8 * 1024 * 1024
ALLOWED_IMAGE = {"image/jpeg", "image/png", "image/webp"}


class StartReq(BaseModel):
    mode: str = "intake"  # intake | followup
    phone: str | None = None


class ChatReq(BaseModel):
    session_id: str
    message: str


def _session(sid: str) -> dialogue.Session:
    s = dialogue.SESSIONS.get(sid)
    if not s:
        raise HTTPException(404, "Session not found or expired. Start a new session.")
    return s


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/staff")
def staff():
    return FileResponse(STATIC / "staff.html")


@app.get("/api/health")
def health():
    c = get_client()
    return {"backend": c.backend_name, "model": config.QWEN_MODEL if c.enabled else None,
            "vision_model": config.QWEN_VL_MODEL if c.vision_enabled else None, "last_error": c.last_error}


@app.post("/api/session")
def start(req: StartReq):
    if req.mode not in ("intake", "followup"):
        raise HTTPException(400, "mode must be intake or followup")
    s, reply = dialogue.start_session(req.mode, req.phone)
    return {"reply": reply, **dialogue.public_state(s)}


@app.post("/api/chat")
def chat(req: ChatReq):
    _session(req.session_id)
    if not req.message.strip():
        raise HTTPException(400, "Empty message")
    return dialogue.handle(req.session_id, req.message[:2000])


@app.post("/api/image")
async def image(session_id: str = Form(...), file: UploadFile = File(...), message: str = Form("")):
    _session(session_id)
    if file.content_type not in ALLOWED_IMAGE:
        raise HTTPException(400, "Please upload a JPG, PNG or WEBP photo")
    data = await file.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(400, "Photo too large (max 8 MB)")
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ext = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}[file.content_type]
    (config.UPLOAD_DIR / f"{session_id}_{uuid.uuid4().hex[:6]}{ext}").write_bytes(data)
    return dialogue.handle(session_id, message[:2000], image=(data, file.content_type))


@app.get("/api/session/{sid}")
def get_session(sid: str):
    s = _session(sid)
    return {**dialogue.public_state(s), "transcript": db.transcript(s.case_id)}


@app.get("/api/dashboard")
def dashboard():
    return db.dashboard()


@app.post("/api/escalations/{esc_id}/ack")
def ack(esc_id: int):
    db.ack_escalation(esc_id)
    return {"ok": True}


@app.get("/api/case/{case_id}/transcript")
def case_transcript(case_id: int):
    c = db.get_case(case_id)
    if not c:
        raise HTTPException(404, "Case not found")
    return {"case": {k: v for k, v in c.items() if k != "state_json"}, "transcript": db.transcript(case_id)}
