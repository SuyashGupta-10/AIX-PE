"""Conversation engine implementing the burn-care flowchart.

 1 User input  ->  2 Understand & classify (Qwen3 + rules)
 3 Emergency?  --yes-->  immediate emergency response + escalate to PHC (RED)
               --no --> 4 targeted WHO-based questions (only what is missing)
 5 first aid   ->  6 triage decision (GREEN / YELLOW / RED, deterministic)
 7 personal details + save case  ->  8 schedule follow-up
 9-11 follow-up call: compare with previous answers, reassess, repeat or close
"""
from __future__ import annotations

import json
import re
import uuid

from . import db, nlu, nlu_rules, protocol

LEVEL_TEXT = {"RED": "RED (Emergency / Urgent)", "YELLOW": "YELLOW (Dhyaan dena zaroori)", "GREEN": "GREEN (Ghar par dekhbhaal)"}

OTHER_CONDITION_REPLY = {
    "snake_bite": "Saanp ke kaatne par turant {em} par call karein aur bachche ko nazdeeki CHC/hospital le jaayein jahan "
                  "anti-venom ho. Bachche ko shaant aur bina hilaaye rakhein, kaati hui jagah ko dil se neeche rakhein. "
                  "Kaatna, choosna, tight rassi baandhna ya jhaad-phoonk mat karein.",
    "dog_bite": "Kutte ke kaatne par ghaav ko turant 15 minute saabun aur behte paani se dhoyein. Aaj hi PHC jaakar "
                "anti-rabies tika (aur zaroorat ho toh tetanus) lagvaayein. Ghaav par mirchi, haldi ya tel mat lagaiye.",
    "diarrhoea": "Dast mein bachche ko ORS ka ghol aur zinc ki goli (ASHA/PHC se) dein, maa ka doodh aur khaana jaari rakhein. "
                 "Agar bachcha bahut sust ho, peena band kare, potty mein khoon ho, ya aankhein dhansi hon toh turant PHC jaayein.",
    "fever": "Bukhaar ke liye bachche ko halke kapde pehnaayein, paani pilate rahein. Bukhaar 2 din se zyada, jhatke, ya "
             "bachcha bahut sust ho toh turant PHC/ASHA se sampark karein.",
    "fall": "Girne ki chot mein agar bachcha behosh hua ho, ulti ho rahi ho, ya haddi tooti lag rahi ho toh turant {em} "
            "par call karein. Chot ko hilaaye bina PHC le jaayein.",
}

WELCOME = ("Namaste! Main Burn Sahayak hoon, bachchon ke jalne (burn) par turant madad ke liye. "
           "Bataiye kya hua hai? Aap bol kar, likh kar ya photo bhej kar bata sakte hain.")


class Session:
    def __init__(self, mode: str = "intake", phone: str | None = None):
        self.id = uuid.uuid4().hex[:12]
        self.mode = mode
        self.phase = "start" if mode == "intake" else "fu_identify"
        self.case = protocol.empty_case()
        self.asked: dict[str, int] = {}
        self.pending: list[str] = []  # case fields the last question asked for
        self.pending_keys: list[str] = []
        self.triage: dict | None = None
        self.script = "roman"
        self.cooling_given = False
        self.emergency_sent = False
        self.escalation_id: int | None = None
        self.followup_due: str | None = None
        self.parent_case_id: int | None = None
        self.previous: dict | None = None
        self.phone = phone
        self.case_id = db.create_case(self.id, kind=mode, phone=phone)
        self.log: list[dict] = []  # debug: what each module extracted per turn

    def state(self) -> dict:
        return {k: getattr(self, k) for k in ("id", "mode", "phase", "case", "asked", "pending", "pending_keys",
                                              "triage", "script", "cooling_given", "emergency_sent",
                                              "followup_due", "parent_case_id", "phone")}


SESSIONS: dict[str, Session] = {}


# ---------------------------------------------------------------------------
def start_session(mode: str = "intake", phone: str | None = None) -> tuple[Session, str]:
    phone = nlu_rules.extract_phone(phone or "") if phone else None
    s = Session(mode, phone)
    SESSIONS[s.id] = s
    if mode == "followup":
        reply = _start_followup(s)
    else:
        reply = WELCOME
        user = db.get_user(phone) if phone else None
        if user:  # returning caller: reuse what we know (Stage 1, step 3)
            s.case.update({"caller_phone": phone, "patient_name": user.get("name"), "village": user.get("village"),
                           "child_age_years": user.get("child_age")})
            past = db.cases_for(phone)
            name = user.get("name") or "aapke bachche"
            reply = (f"Namaste! Aap pehle bhi humse baat kar chuke hain ({len(past)} purane record). "
                     f"Kya {name} ke liye call hai? Bataiye abhi kya hua hai?")
        elif phone:
            s.case["caller_phone"] = phone
    _say(s, "assistant", reply)
    return s, reply


def _say(s: Session, role: str, text: str) -> None:
    db.add_message(s.case_id, role, text)


def _ask(s: Session, questions: list[protocol.Question]) -> str:
    s.pending = [f for q in questions for f in q.fields]
    s.pending_keys = [q.key for q in questions]
    for q in questions:
        s.asked[q.key] = s.asked.get(q.key, 0) + 1
    if len(questions) == 1:
        return questions[0].text
    return "\n".join(f"({i}) {q.text}" for i, q in enumerate(questions, 1))


def _merge(case: dict, update: dict) -> list[str]:
    changed = []
    for k, v in update.items():
        if k not in case or v in (None, [], ""):
            continue
        if k == "body_parts":
            v = sorted(set(case["body_parts"]) | set(v))
        elif k == "harmful_remedy":
            v = sorted(set(case["harmful_remedy"]) | set(v))
        elif k == "depth" and case["depth"] in protocol.DEPTHS[:3] and v in protocol.DEPTHS[:3]:
            v = max(case["depth"], v, key=protocol.DEPTHS.index)  # never downgrade depth
        elif k == "size" and case["size"] in protocol.SIZES[:4] and v in protocol.SIZES[:4]:
            v = max(case["size"], v, key=protocol.SIZES.index)
        elif k == "mechanism" and case["mechanism"] and v not in ("electrical", "chemical"):
            continue  # the first stated cause stands; a later mention (e.g. "thanda paani") is not a new burn
        if case[k] != v:
            case[k] = v
            changed.append(k)
    return changed


def _summary(case: dict, level: str | None) -> str:
    bits = []
    if case.get("child_age_years") is not None:
        a = case["child_age_years"]
        bits.append(f"{a:g} saal" if a >= 1 else f"{round(a * 12)} mahine")
    if case.get("mechanism"):
        bits.append(protocol.MECHANISM_LABEL[case["mechanism"]])
    if case.get("body_parts"):
        bits.append("/".join(case["body_parts"]))
    if case.get("depth"):
        bits.append(protocol.DEPTH_LABEL[case["depth"]])
    if case.get("size"):
        bits.append(protocol.SIZE_LABEL[case["size"]])
    flags = [f for f in ("breathing_difficulty", "unconscious", "face_neck", "circumferential") if case.get(f)]
    if flags:
        bits.append("flags: " + ",".join(flags))
    return (f"[{level}] " if level else "") + "Burn: " + "; ".join(bits)


def _persist(s: Session, **extra) -> None:
    fields = {"triage_level": s.triage["level"] if s.triage else None,
              "summary": _summary(s.case, s.triage["level"] if s.triage else None),
              "phone": s.case.get("caller_phone") or s.phone}
    fields.update(extra)
    db.save_case(s.case_id, s.state(), **fields)


def _escalate(s: Session) -> dict:
    phc = db.find_phc(s.case.get("village"))
    s.escalation_id = db.add_escalation(s.case_id, "RED", phc, _summary(s.case, "RED"))
    return phc


# ---------------------------------------------------------------------------
def handle(session_id: str, message: str, image: tuple[bytes, str] | None = None) -> dict:
    s = SESSIONS[session_id]
    message = (message or "").strip()
    if message:
        _say(s, "user", message)
        if sum(c.isalpha() for c in message) >= 4:
            s.script = nlu_rules.script_of(message)
    prefix: list[str] = []
    meta: dict = {}

    if image:
        prefix += _handle_image(s, *image)

    if s.mode == "followup":
        parts = _followup_turn(s, message) if message or not image else _followup_next(s)
    else:
        parts = _intake_turn(s, message, meta, from_image=bool(image))

    reply = "\n\n".join(p for p in prefix + parts if p)
    reply = nlu.localize(reply, s.script)
    _say(s, "assistant", reply)
    _persist(s)
    s.log.append({"message": message, "extracted": meta})
    return {"reply": reply, **public_state(s), "debug": meta}


def public_state(s: Session) -> dict:
    return {"session_id": s.id, "mode": s.mode, "phase": s.phase, "case": s.case, "triage": s.triage,
            "pending": s.pending, "script": s.script, "escalation_id": s.escalation_id,
            "followup_due": s.followup_due, "case_id": s.case_id}


def _handle_image(s: Session, data: bytes, mime: str) -> list[str]:
    findings = nlu.assess_image(data, mime)
    if findings is None:
        return ["Photo mil gayi, dhanyavaad. Abhi photo ki automatic jaanch uplabdh nahi hai, "
                "isliye kripya shabdon mein bhi bataiye ki jagah kaisi dikh rahi hai."]
    s.case["image_findings"] = findings
    if not findings["is_burn_likely"]:
        return ["Photo mein saaf jalne ka nishaan nahi dikh raha (ya photo saaf nahi hai). "
                "Kripya jali jagah ki paas se, achhi roshni mein photo bhejein, ya shabdon mein bataiye."]
    s.case["is_burn"] = True
    upd = {}
    if findings["body_part"] != "unclear" and not s.case["body_parts"]:
        upd["body_parts"] = [findings["body_part"]]
    # A photo may only RAISE severity. A reassuring reading ("just red", "small") from a small vision
    # model is not trusted on its own; those fields are still asked of the caller.
    if findings["depth_guess"] in ("partial", "full"):
        upd["depth"] = findings["depth_guess"]  # _merge keeps the deeper of user/image
    if findings["size_guess"] in ("few_palms", "large") and s.case["size"] in (None, "coin", "palm"):
        upd["size"] = findings["size_guess"]
    if findings["infection_signs"] and s.mode == "followup":
        upd["pus_or_smell"] = True
    _merge(s.case, upd)
    seen = []
    if findings["depth_guess"] != "unclear":
        seen.append(protocol.DEPTH_LABEL[findings["depth_guess"]])
    if findings["size_guess"] != "unclear":
        seen.append("size lagbhag " + protocol.SIZE_LABEL[findings["size_guess"]])
    if findings["infection_signs"]:
        seen.append("infection ke kuch nishaan")
    txt = "Photo dekh kar andaza: " + (", ".join(seen) if seen else "jalne ka nishaan dikh raha hai")
    return [txt + ". (Yeh sirf photo se andaza hai, doctor ki jaanch ki jagah nahi. Kripya neeche ke sawaalon "
                  "ka jawab bhi dein.)"]


# ---------------------------------------------------------------------------
def _intake_turn(s: Session, message: str, meta: dict, from_image: bool = False) -> list[str]:
    out: list[str] = []
    if message:
        update, m = nlu.understand(message, s.pending, s.case)
        meta.update(m)
        intent = nlu.classify(message, update, m)
        meta["intent"] = intent
    else:
        update, intent = {}, "info"

    # Personal details (name / village) are free text; take the answer as-is if the model didn't
    if s.phase == "details" and message:
        _fill_details_fallback(s, message, update)

    burn_signal = update.get("is_burn") or any(update.get(k) for k in ("mechanism", "depth")) or s.case["is_burn"]
    if not burn_signal and s.phase in ("start", "handoff"):
        if intent in OTHER_CONDITION_REPLY:
            s.phase = "handoff"
            return [OTHER_CONDITION_REPLY[intent].format(em=protocol.EMERGENCY_NUMBER),
                    "Abhi yeh seva khaas taur par jalne (burn) ke liye hai. Aapko ASHA / nazdeeki PHC se jod rahe hain. "
                    "Kya bachche ko kahin jala bhi hai?"]
        if intent == "question" and message:
            ans = nlu.answer_faq(message, s.case)
            return [ans or "", "Kya bachche ko jala hai? Bataiye kya hua."]
        if not from_image:
            return [WELCOME if intent == "greeting" else
                    "Main samajh nahi paaya. Kya bachche ko kisi garam cheez, aag, bijli ya chemical se jala hai? Kya hua bataiye."]

    s.case["is_burn"] = True
    was_questioned = list(s.pending)
    _merge(s.case, update)
    if s.phase in ("start", "handoff"):
        s.phase = "assess"

    # Question answered with nothing usable -> the counter in s.asked lets us move on after 2 tries
    s.pending, s.pending_keys = [], []

    # Step 3: emergency check (runs every turn: new info can upgrade the case at any time)
    reds = protocol.red_flags(s.case)
    if reds and not s.emergency_sent:
        s.emergency_sent = True
        s.cooling_given = True
        s.triage = protocol.triage(s.case).to_dict()
        phc = _escalate(s)
        out.append("⚠️ YEH EMERGENCY HAI: " + "; ".join(reds) + ".")
        out.append(f"Abhi turant {protocol.EMERGENCY_NUMBER} par call karke ambulance bulaiye ya bachche ko nazdeeki "
                   f"PHC/CHC/hospital le jaaiye. Main aapka case {phc['name']} ke doctor ko bhej raha hoon "
                   f"(phone {phc['phone']}), woh aapko call karenge.")
        out.append("Tab tak:\n" + protocol.immediate_cooling(s.case) + "\n" + protocol.EMERGENCY_TRANSPORT + "\n" + protocol.DONTS)
        s.phase = "details"
        q = protocol.missing_personal(s.case, s.asked)[:2]
        if q:
            out.append("Doctor tak jaankari bhejne ke liye: " + _ask(s, q))
        return out

    if s.phase == "assess":
        if not s.cooling_given:
            s.cooling_given = True
            out.append("Samajh gaya, bachche ko jala hai. Sabse pehle: " + protocol.immediate_cooling(s.case))
            if s.case["harmful_remedy"]:
                out.append("Jo " + ", ".join(s.case["harmful_remedy"]) + " lagaya hai use saaf paani se halke se dho dijiye. "
                           + protocol.DONTS)
        elif intent == "question" and message and not any(k in update for k in ("mechanism", "depth", "size", "body_parts")):
            ans = nlu.answer_faq(message, s.case)
            if ans:
                out.append(ans)
        missing = protocol.missing_fields(s.case, s.asked)
        if missing:
            # red-flag screen goes alone; other questions in pairs
            batch = missing[:1] if missing[0].key == "red_flags" else missing[:2]
            out.append(_ask(s, batch))
            return out
        return out + _deliver_triage(s)

    if s.phase == "details":
        # A late red flag after triage still escalates (handled above). Otherwise re-check triage level.
        if s.triage and s.triage["level"] != "RED":
            new = protocol.triage(s.case).to_dict()
            if _rank(new["level"]) > _rank(s.triage["level"]):
                s.triage = new
                out.append(f"Nayi jaankari ke hisaab se triage badal kar {LEVEL_TEXT[new['level']]} hai. {new['route']}")
        if intent == "question" and message and not update:
            ans = nlu.answer_faq(message, s.case)
            if ans:
                out.append(ans)
        q = protocol.missing_personal(s.case, s.asked)
        if q:
            out.append(_ask(s, q[:1]))
            return out
        return out + _close_intake(s)

    # closed: answer questions, watch for worsening
    if s.phase == "closed":
        new = protocol.triage(s.case).to_dict()
        if s.triage and _rank(new["level"]) > _rank(s.triage["level"]):
            s.triage = new
            out.append(f"Dhyaan dein: nayi jaankari ke hisaab se yeh {LEVEL_TEXT[new['level']]} hai. {new['route']}")
            if new["level"] == "RED":
                _escalate(s)
        if message:
            ans = nlu.answer_faq(message, s.case) if intent == "question" else None
            out.append(ans or "Aapka case save hai. Koi aur sawaal ho toh poochiye. Haalat bigde toh turant "
                              f"{protocol.EMERGENCY_NUMBER} par call karein.")
    return out


def _rank(level: str) -> int:
    return {"GREEN": 0, "YELLOW": 1, "RED": 2}[level]


def _deliver_triage(s: Session) -> list[str]:
    t = protocol.triage(s.case)
    s.triage = t.to_dict()
    out = [f"Aapki jaankari ke hisaab se yeh {LEVEL_TEXT[t.level]} case hai.\nKaaran: " + "; ".join(t.reasons) + ".",
           t.route,
           "First aid:\n" + protocol.first_aid_advice(s.case, t.level),
           protocol.WHEN_TO_SEEK_CARE]
    if t.level == "RED":
        phc = _escalate(s)
        out.insert(2, f"Case {phc['name']} (phone {phc['phone']}) ko bhej diya gaya hai.")
    s.phase = "details"
    q = protocol.missing_personal(s.case, s.asked)
    if q:
        out.append("Follow-up ke liye kuch jaankari chahiye. " + _ask(s, q[:1]))
    else:
        out += _close_intake(s)
    return out


def _fill_details_fallback(s: Session, message: str, update: dict) -> None:
    if "patient_name" in s.pending and not update.get("patient_name"):
        name = re.sub(r"(?i)\b(mera|meri|mere|uska|uski|bachche|bachchi|beta|beti|ka|ki|naam|name|hai|is|my|child|son|daughter)\b",
                      " ", message)
        name = " ".join(re.sub(r"[^\w\sऀ-ॿ]", " ", name).split())
        if name and len(name) <= 40 and nlu_rules.short_yes_no(message) is not False:
            update["patient_name"] = name.title()
    if "village" in s.pending and not update.get("village"):
        v = re.sub(r"(?i)\b(main|hum|gaon|gaanv|village|se|hoon|hu|hain|hai|mera|hamara|zila|district|block)\b", " ", message)
        v = " ".join(re.sub(r"[^\w\sऀ-ॿ]", " ", v).split())
        if v and len(v) <= 60:
            update["village"] = v.title()
    _merge(s.case, {k: update[k] for k in ("patient_name", "village", "caller_phone") if update.get(k)})


def _close_intake(s: Session) -> list[str]:
    s.phase = "closed"
    phone = s.case.get("caller_phone") or s.phone
    if phone:
        db.upsert_user(phone, name=s.case.get("patient_name"), child_age=s.case.get("child_age_years"),
                       village=s.case.get("village"), language=s.script)
        s.phone = phone
    level = s.triage["level"] if s.triage else "YELLOW"
    hours = protocol.triage(s.case).followup_hours if level != "RED" else 2
    s.followup_due = db.schedule_followup(s.case_id, phone, hours)
    if s.escalation_id:  # refresh PHC now that we may know the village
        _escalate(s)
    _persist(s)
    when = {"RED": "2 ghante mein", "YELLOW": "kal", "GREEN": "2 din baad"}[level]
    name = s.case.get("patient_name") or "bachche"
    routed = ""
    if s.escalation_id:
        phc = db.find_phc(s.case.get("village"))
        routed = f" Case {phc['name']} (phone {phc['phone']}) ke doctor ko bhej diya gaya hai."
    return [f"Dhanyavaad. {name} ka case save ho gaya hai ({_summary(s.case, level)}).{routed}",
            f"Hum {when} follow-up ke liye call karenge" + (f" ({phone} par)." if phone else ".") +
            f" Agar beech mein haalat bigde, toh turant {protocol.EMERGENCY_NUMBER} par call karein ya PHC jaayein."]


# ---------------------------------------------------------------------------
# Follow-up call (steps 9-11)
def _start_followup(s: Session) -> str:
    prev = db.latest_case_for(s.phone) if s.phone else None
    if not prev:
        s.mode, s.phase = "intake", "start"
        return "Is number par koi purana burn case nahi mila. " + WELCOME
    prev_state = json.loads(prev["state_json"])
    s.previous = prev_state
    s.parent_case_id = prev["id"]
    db.save_case(s.case_id, s.state(), parent_case_id=prev["id"])
    carried = {k: prev_state["case"].get(k) for k in ("mechanism", "body_parts", "depth", "size", "child_age_years",
                                                       "patient_name", "caller_phone", "village", "face_neck")}
    s.case.update(carried)
    s.case["is_burn"] = True
    s.triage = prev_state.get("triage")
    s.phase = "fu_questions"
    name = carried.get("patient_name") or "aapke bachche"
    part = "/".join(carried.get("body_parts") or []) or "shareer"
    mech = protocol.MECHANISM_LABEL.get(carried.get("mechanism") or "unknown")
    q = _ask(s, protocol.FOLLOWUP_QUESTIONS[:1])
    return (f"Namaste! Main Burn Sahayak se follow-up ke liye baat kar raha hoon. Pichhli baar {name} ka {part} "
            f"({mech}) se jala tha, triage {prev['triage_level']} tha. Ab ghaav kaisa hai? {q}")


def _followup_turn(s: Session, message: str) -> list[str]:
    if s.phase == "fu_identify":
        phone = nlu_rules.extract_phone(message)
        if not phone:
            return ["Kripya woh mobile number bataiye jis par pichhli baar case darj hua tha."]
        s.phone = phone
        return [_start_followup(s)]
    update, meta = nlu.understand(message, s.pending, s.case)
    s.log.append({"followup_extract": meta})
    _merge(s.case, update)
    if _healed(message):
        s.phase = "closed"
        db.complete_followups(s.parent_case_id)
        db.save_case(s.parent_case_id, s.previous, status="closed")
        return ["Bahut achhi baat hai ki ghaav bhar gaya hai! Case band kar rahe hain. Dhoop mein nayi chamdi ko dhak kar rakhein. "
                "Dobara zaroorat ho toh call karein."]
    return _followup_next(s, update)


def _healed(message: str) -> bool:
    t = message.lower()
    return any(w in t for w in ("bhar gaya", "theek ho gaya", "thik ho gaya", "healed", "puri tarah theek", "ठीक हो गया", "भर गया"))


def _followup_next(s: Session, update: dict | None = None) -> list[str]:
    new_danger = any((update or {}).get(f) is True for f in ("fever", "pus_or_smell", "redness_spreading", "movement_difficulty"))
    if s.phase == "closed" and new_danger:
        s.phase = "fu_questions"  # re-evaluate with the new sign
    if s.phase != "fu_questions":
        return ["Aapka follow-up poora ho chuka hai. Koi aur sawaal ho toh poochiye. Haalat bigde toh turant "
                f"{protocol.EMERGENCY_NUMBER} par call karein ya PHC jaayein."]
    remaining = [q for q in protocol.FOLLOWUP_QUESTIONS
                 if all(s.case.get(f) is None for f in q.fields) and s.asked.get(q.key, 0) < 2]
    danger = any(s.case.get(f) for f in ("fever", "pus_or_smell", "redness_spreading", "movement_difficulty")) \
        or s.case.get("pain_trend") == "worse"
    if remaining and not danger:  # a danger sign ends the questioning and goes straight to advice
        return [_ask(s, remaining[:1])]

    s.pending, s.pending_keys = [], []
    t = protocol.triage(s.case)
    s.triage = t.to_dict()
    s.phase = "closed"
    db.complete_followups(s.parent_case_id)
    worse = [r for r in t.reasons if "infection" in r or "bigad" in r or "hilane" in r]
    reds = protocol.red_flags(s.case)
    if reds or worse:
        if reds:
            _escalate(s)
        s.followup_due = db.schedule_followup(s.case_id, s.phone, 24)
        return ["Dhyaan dein: " + "; ".join(reds + worse) + ".",
                f"Bachche ko aaj hi nazdeeki PHC / doctor ko dikhaiye. Zaroorat ho toh {protocol.EMERGENCY_NUMBER} par call karein. "
                "Ghaav ko saaf kapde se dhak kar rakhein, khud se koi cream ya ghar ka nuskha mat lagaiye.",
                "Hum kal phir call karenge."]
    s.followup_due = db.schedule_followup(s.case_id, s.phone, 48)
    note = ("Chhala pehle 1-2 din thoda bada ho sakta hai, yeh aam hai. Use phodiye mat. "
            if s.case.get("blister_trend") == "bigger" else "")
    return [note + "Achhi baat hai, koi khatre ka lakshan nahi hai. Abhi wali dekhbhaal jaari rakhiye: ghaav saaf aur dhaka rakhein, "
            "chhale mat phodiye, paani/ORS pilate rahein.",
            protocol.WHEN_TO_SEEK_CARE,
            "Hum 2 din baad phir follow-up ke liye call karenge. Dhanyavaad!"]
