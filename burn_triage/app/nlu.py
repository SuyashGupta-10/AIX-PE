"""Qwen3-powered understanding, merged with the rule layer.

  understand()   message -> partial case update (LLM JSON + rules, safety-merged)
  classify()     intent: burn / other condition / greeting / question / info
  localize()     Roman-Hinglish template -> caller's script (Devanagari if they wrote in it)
  answer_faq()   free-form burn-care questions, grounded in protocol text
  assess_image() Qwen3-VL look at a burn photo
"""
from __future__ import annotations

import json
import logging

from . import config, nlu_rules, protocol
from .llm import get_client, parse_json

log = logging.getLogger("nlu")

EXTRACT_SYSTEM = """You are the language-understanding module of a child burn-care helpline in rural India.
Callers write in Hinglish (Roman Hindi+English), Devanagari Hindi, or messy code-mixed text.
Read the caller's latest message and return ONLY a JSON object with the fields you can fill from it.
Include a field ONLY if the message itself states it. Do not copy anything from the examples.
If the message only says the child got burnt, output just {"intent":"burn"}.

Fields:
- intent: "burn" | "snake_bite" | "dog_bite" | "diarrhoea" | "fever" | "fall" | "greeting" | "question" | "other"
- mechanism: "scald" (hot water/tea/milk/dal/steam) | "hot_oil" | "flame" (fire, chulha flame, lamp, kerosene) | "contact" (tawa, iron, hot utensil, bike silencer) | "electrical" | "chemical" (acid, lime/chuna)
- body_parts: list from ["hand","fingers","arm","foot","thigh","face","eye","neck","head","chest","abdomen","back","genitals","joint","buttocks"]
- depth: "superficial" (only red) | "partial" (blisters/chhale/phafole, skin peeling) | "full" (white, black, leathery, painless)
- size: "coin" | "palm" (about one child palm) | "few_palms" (2-9 palms, whole hand) | "large" (whole arm/leg, many areas)
- tbsa_percent: number, only if a percentage is stated
- child_age_years: number (convert months to years, e.g. 6 mahine -> 0.5)
- pain: "none" | "mild" | "moderate" | "severe"
- breathing_difficulty, face_neck, unconscious, circumferential, other_injuries, clothes_on_fire: true/false
- cooled_with_water: true/false
- harmful_remedy: list, e.g. ["toothpaste","ghee","haldi","ice","gobar"]
- patient_name, village: string ; caller_phone: 10 digit string
- follow-up only: pain_trend "better"|"same"|"worse" ; blister_trend "smaller"|"same"|"bigger" ; fever, pus_or_smell, redness_spreading, movement_difficulty: true/false

A short reply like "nahi"/"haan" answers the pending question: set every pending field to false/true.
If the reply names one thing ("haan, saans mein dikkat hai"), set only that field.
"gir gaya" about hot liquid means it spilled, not a fall injury."""

# Small Qwen3 models copy facts from examples, so the first example teaches "output only what is said".
FEW_SHOT = [
    ("pending: []\nmessage: bachche ko jal gaya, jaldi batao kya karein",
     '{"intent":"burn"}'),
    ("pending: []\nmessage: bitiya ki taang pe kukar ki bhaap lag gayi, laal ho gaya hai",
     '{"intent":"burn","mechanism":"scald","body_parts":["foot"],"depth":"superficial"}'),
    ("pending: [\"breathing_difficulty\",\"face_neck\",\"unconscious\"]\nmessage: Nahi, sirf haath pe hua hai.",
     '{"breathing_difficulty":false,"face_neck":false,"unconscious":false,"body_parts":["hand"]}'),
]


def _llm_extract(message: str, pending: list[str], case: dict) -> dict | None:
    client = get_client()
    if not client.enabled:
        return None
    known = {k: v for k, v in case.items() if v not in (None, [], "") and k != "image_findings"}
    msgs = [{"role": "system", "content": EXTRACT_SYSTEM}]
    for u, a in FEW_SHOT:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": f"already known: {json.dumps(known, ensure_ascii=False)}\n"
                                            f"pending: {json.dumps(pending)}\nmessage: {message}"})
    out = client.chat(msgs, max_tokens=256, temperature=0.0)
    data = parse_json(out) if out else None
    if data is None and out is not None:
        log.warning("Qwen returned non-JSON: %r", out[:200])
    return _validate(data) if isinstance(data, dict) else None


def _validate(d: dict) -> dict:
    """Drop anything outside the schema so a hallucinated value can't enter the case."""
    enums = {"mechanism": protocol.MECHANISMS, "depth": protocol.DEPTHS, "size": protocol.SIZES, "pain": protocol.PAIN,
             "pain_trend": ["better", "same", "worse"], "blister_trend": ["smaller", "same", "bigger"]}
    bools = ["breathing_difficulty", "face_neck", "unconscious", "circumferential", "other_injuries", "cooled_with_water",
             "clothes_on_fire",
             "fever", "pus_or_smell", "redness_spreading", "movement_difficulty"]
    parts = set(nlu_rules.PARTS)
    out = {}
    for k, v in d.items():
        if k in enums and v in enums[k] and v != "unknown":
            out[k] = v
        elif k in bools and isinstance(v, bool):
            out[k] = v
        elif k == "body_parts" and isinstance(v, list):
            ps = [p for p in v if p in parts]
            if ps:
                out[k] = ps
        elif k in ("child_age_years", "tbsa_percent") and isinstance(v, (int, float)) and 0 <= v < 100:
            out[k] = float(v)
        elif k == "harmful_remedy" and isinstance(v, list):
            out[k] = [str(x) for x in v][:5]
        elif k in ("patient_name", "village") and isinstance(v, str) and 0 < len(v) < 60:
            out[k] = v.strip()
        elif k == "caller_phone" and isinstance(v, (str, int)):
            ph = nlu_rules.extract_phone(str(v))
            if ph:
                out[k] = ph
        elif k == "intent" and isinstance(v, str):
            out[k] = v
    if out.get("child_age_years", 0) >= 18:
        out.pop("child_age_years")
    return out


GATED = {"mechanism", "body_parts", "depth", "size", "pain", "child_age_years", "tbsa_percent", "face_neck",
         "breathing_difficulty", "unconscious", "circumferential", "other_injuries", "pain_trend", "blister_trend",
         "fever", "pus_or_smell", "redness_spreading", "movement_difficulty", "patient_name", "village", "caller_phone"}


def _raises_safety(k: str, v) -> bool:
    return (k in SAFETY_BOOLS and v is True) or (k == "mechanism" and v in SAFETY_MECH) \
        or (k == "depth" and v == "full")


# Fields where a "yes" from either source wins: a missed emergency is worse than a false alarm.
SAFETY_BOOLS = ["breathing_difficulty", "unconscious", "circumferential", "other_injuries", "face_neck",
                "fever", "pus_or_smell", "redness_spreading", "movement_difficulty"]
SAFETY_MECH = {"electrical", "chemical"}


def understand(message: str, pending: list[str], case: dict) -> tuple[dict, dict]:
    """Returns (update, meta). meta says which sources contributed."""
    rules = nlu_rules.extract(message, pending)
    llm = _llm_extract(message, pending, case)
    meta = {"rules": rules, "llm": llm, "llm_used": llm is not None}
    if llm is None:
        return rules, meta

    # Keyword hits are explicit evidence in the text, so they win on conflict; Qwen fills the gaps
    # (paraphrases, spelling variants, context the keyword lists don't cover).
    merged = {k: v for k, v in llm.items() if k != "intent"}
    for k in list(merged):
        if k in rules:
            continue
        if k in ("child_age_years", "tbsa_percent", "caller_phone") and not nlu_rules.has_number(message):
            merged.pop(k)  # a number that isn't in the message is a hallucination
        elif config.LLM_TRUST != "full" and k in GATED and k not in pending and not _raises_safety(k, merged[k]):
            merged.pop(k)  # small models invent details; only trust them for the question just asked
    meta["llm_dropped"] = sorted(set(llm) - set(merged) - {"intent"})
    trusted = dict(merged)
    merged.update(rules)
    for k in SAFETY_BOOLS:
        if rules.get(k) is True or llm.get(k) is True:
            merged[k] = True
    if llm.get("mechanism") in SAFETY_MECH and rules.get("mechanism") not in SAFETY_MECH:
        merged["mechanism"] = llm["mechanism"]
    # deeper estimate wins (a missed blister/deep burn is the costly error); for size, an explicit
    # phrase the caller used ("hatheli se aadha") beats the model's guess, so rules already won above
    order = protocol.DEPTHS[:3]
    cands = [x for x in (rules.get("depth"), trusted.get("depth")) if x in order]
    if cands:
        merged["depth"] = max(cands, key=order.index)
    if llm.get("intent") == "burn":
        merged["is_burn"] = True
    return merged, meta


def classify(message: str, update: dict, meta: dict) -> str:
    llm_intent = (meta.get("llm") or {}).get("intent")
    if update.get("is_burn") or llm_intent == "burn":
        return "burn"
    other = nlu_rules.detect_other_condition(message)
    if other:
        return other
    if llm_intent in ("snake_bite", "dog_bite", "diarrhoea", "fever", "fall"):
        return llm_intent
    if nlu_rules.is_greeting(message) or llm_intent == "greeting":
        return "greeting"
    if nlu_rules.is_question(message) or llm_intent == "question":
        return "question"
    return "info"


# ---------------------------------------------------------------------------
LOCALIZE_SYSTEM = (
    "Convert the given Roman-script Hinglish text into Hindi written in Devanagari script. "
    "Keep the exact meaning, numbering, line breaks and the number 108. Keep common English medical words "
    "(ORS, PHC, paracetamol, tetanus) as they are. Output only the converted text."
)


def _localize_enabled() -> bool:
    client = get_client()
    if config.LOCALIZE_REPLIES == "auto":
        return client.enabled and (client.backend_name == "openai" or client.on_gpu)
    return config.LOCALIZE_REPLIES == "on" and client.enabled


def localize(text: str, script: str) -> str:
    """Our templates are Roman Hinglish. If the caller wrote in Devanagari, reply in Devanagari."""
    if script != "devanagari" or not _localize_enabled():
        return text
    out = get_client().chat([{"role": "system", "content": LOCALIZE_SYSTEM}, {"role": "user", "content": text}],
                            max_tokens=900, temperature=0.0)
    # Accept only if it actually came back in Devanagari and kept the emergency number
    if out and nlu_rules.script_of(out) == "devanagari" and ("108" in out or "108" not in text):
        return out
    return text


QA_SYSTEM = """You are "Burn Sahayak", a child burn first-aid helper for rural Indian families.
Answer the caller's question in short, simple Hinglish (Roman script), 2-4 sentences, warm and calm.
Use ONLY the reference guidance below. Do not prescribe medicine doses or new treatments.
If the question is outside burn first aid or you are unsure, say to contact the nearest PHC/ASHA or call 108.

Reference guidance:
{kb}"""


def _kb() -> str:
    lines = [protocol.immediate_cooling({}), protocol.DONTS, protocol.WHEN_TO_SEEK_CARE, protocol.EMERGENCY_TRANSPORT]
    lines += [a for _, a in protocol.FAQ]
    return "\n".join("- " + x for x in lines)


def answer_faq(question: str, case: dict) -> str | None:
    t = question.lower()
    for keys, ans in protocol.FAQ:
        if any(k in t for k in keys):
            rule_answer = ans
            break
    else:
        rule_answer = None
    out = get_client().chat([{"role": "system", "content": QA_SYSTEM.format(kb=_kb())},
                             {"role": "user", "content": question}], max_tokens=220, temperature=0.3)
    return out or rule_answer


# ---------------------------------------------------------------------------
VISION_PROMPT = """You are helping a health worker triage a CHILD's possible burn from a photo taken by a parent.
Look carefully and reply ONLY with JSON:
{"is_burn_likely": true/false,
 "body_part": one of ["hand","fingers","arm","foot","thigh","face","eye","neck","head","chest","abdomen","back","genitals","joint","buttocks","unclear"],
 "depth_guess": "superficial" (red, dry, no blisters) | "partial" (blisters, wet/shiny, peeling) | "full" (white, waxy, brown/black, leathery) | "unclear",
 "size_guess": "coin" | "palm" | "few_palms" | "large" | "unclear",
 "infection_signs": true/false (pus, yellow crust, spreading redness),
 "image_quality": "good" | "poor",
 "description": one short plain-English sentence of what you see}
If the image is not a skin injury, set is_burn_likely false."""


def assess_image(image_bytes: bytes, mime: str) -> dict | None:
    raw = get_client().vision(image_bytes, mime, VISION_PROMPT)
    data = parse_json(raw) if raw else None
    if not isinstance(data, dict):
        return None
    ok = lambda v, allowed: v if v in allowed else "unclear"  # noqa: E731
    return {
        "is_burn_likely": bool(data.get("is_burn_likely")),
        "body_part": ok(data.get("body_part"), list(nlu_rules.PARTS)),
        "depth_guess": ok(data.get("depth_guess"), ["superficial", "partial", "full"]),
        "size_guess": ok(data.get("size_guess"), ["coin", "palm", "few_palms", "large"]),
        "infection_signs": bool(data.get("infection_signs")),
        "image_quality": data.get("image_quality") if data.get("image_quality") in ("good", "poor") else "unclear",
        "description": str(data.get("description", ""))[:240],
    }
