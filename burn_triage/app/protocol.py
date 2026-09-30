"""Burn-care clinical protocol for children.

Every piece of medical content the assistant gives comes from this file. The LLM
is used to understand the caller and to phrase text, never to decide severity
or invent treatment. Content follows WHO burn first-aid guidance and standard
paediatric burn referral criteria (IAP / ANZBA-style). It needs sign-off from a
clinician before any real deployment.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import EMERGENCY_NUMBER

# ---------------------------------------------------------------------------
# Case schema
# ---------------------------------------------------------------------------
MECHANISMS = ["scald", "hot_oil", "flame", "contact", "electrical", "chemical", "unknown"]
DEPTHS = ["superficial", "partial", "full", "unknown"]
SIZES = ["coin", "palm", "few_palms", "large", "unknown"]
PAIN = ["none", "mild", "moderate", "severe"]
SPECIAL_AREAS = {"face", "neck", "hand", "fingers", "foot", "genitals", "joint", "eye"}

MECHANISM_LABEL = {
    "scald": "garam paani / chai / doodh (scald)",
    "hot_oil": "garam tel",
    "flame": "aag / chulha ki lau",
    "contact": "garam bartan / tawa / istri se chhoona",
    "electrical": "bijli ka current",
    "chemical": "tezaab / chemical",
    "unknown": "pata nahi",
}
DEPTH_LABEL = {
    "superficial": "sirf laali (superficial)",
    "partial": "chhale / phafole (partial thickness)",
    "full": "safed / kaali / sakht chamdi (deep, full thickness)",
    "unknown": "pata nahi",
}
SIZE_LABEL = {
    "coin": "sikke jitna (<1%)",
    "palm": "ek hatheli jitna (~1%)",
    "few_palms": "2-9 hatheli jitna (2-9%)",
    "large": "bahut bada / poora haath-pair (10%+)",
    "unknown": "pata nahi",
}


def empty_case() -> dict:
    return {
        "is_burn": None,
        "mechanism": None,
        "body_parts": [],
        "depth": None,
        "size": None,
        "tbsa_percent": None,
        "child_age_years": None,
        "pain": None,
        # red flags (None = not asked yet)
        "breathing_difficulty": None,
        "face_neck": None,
        "unconscious": None,
        "circumferential": None,
        "other_injuries": None,
        "clothes_on_fire": None,
        # first aid status
        "cooled_with_water": None,
        "harmful_remedy": [],
        # follow-up signs
        "fever": None,
        "pus_or_smell": None,
        "redness_spreading": None,
        "pain_trend": None,  # better / same / worse
        "blister_trend": None,  # smaller / same / bigger
        "movement_difficulty": None,
        # personal details
        "patient_name": None,
        "caller_phone": None,
        "village": None,
        "gender": None,
        # image assessment (kept separate, never lowers severity)
        "image_findings": None,
    }


# ---------------------------------------------------------------------------
# Question bank (Roman Hinglish). Keys map to case fields the answer fills.
# ---------------------------------------------------------------------------
@dataclass
class Question:
    key: str
    fields: list[str]
    text: str
    required: bool = True


QUESTIONS: list[Question] = [
    Question(
        "red_flags",
        ["breathing_difficulty", "face_neck", "unconscious"],
        "Kya bachche ko saans lene mein dikkat hai, awaaz bhaari hai, ya woh behosh/bahut sust hai? "
        "Kya chehra ya gardan bhi jala hai?",
    ),
    Question(
        "mechanism",
        ["mechanism"],
        "Bachcha kaise jala? (garam paani/chai/doodh, garam tel, aag/chulha, garam tawa/istri, "
        "bijli ka current, ya tezaab/chemical?)",
    ),
    Question(
        "body_parts",
        ["body_parts"],
        "Shareer ka kaunsa hissa jala hai? (haath, ungliyan, pair, chehra, chhati, pet, peeth...)",
    ),
    Question(
        "depth",
        ["depth"],
        "Jali hui jagah kaisi dikh rahi hai? Sirf laal hai, chhale (phafole) pad gaye hain, "
        "ya chamdi safed/kaali/sakht ho gayi hai? (Photo bhi bhej sakte hain.)",
    ),
    Question(
        "size",
        ["size"],
        "Jali hui jagah kitni badi hai? Bachche ki apni hatheli (ungliyon samet) se milaiye: "
        "sikke jitni, ek hatheli jitni, 2-3 hatheli jitni, ya poora haath/pair?",
    ),
    Question("age", ["child_age_years"], "Bachche ki umar kitni hai?"),
]
QUESTION_BY_KEY = {q.key: q for q in QUESTIONS}

PERSONAL_QUESTIONS: list[Question] = [
    Question("patient_name", ["patient_name"], "Bachche ka naam kya hai?"),
    Question("caller_phone", ["caller_phone"], "Aapka mobile number kya hai? (follow-up call ke liye)"),
    Question("village", ["village"], "Aap kis gaon / block / zile se hain? (nazdeeki PHC dhoondhne ke liye)"),
]

FOLLOWUP_QUESTIONS: list[Question] = [
    Question("pain_trend", ["pain_trend"], "Dard kal ke mukable kam hai, utna hi hai, ya zyada hai?"),
    Question("blister_trend", ["blister_trend"], "Chhale/ghaav chhote hue, waise hi hain, ya bade ho gaye hain?"),
    Question(
        "infection",
        ["fever", "pus_or_smell", "redness_spreading"],
        "Kya bukhaar aaya hai, ghaav se peela paani/pus ya badboo aa rahi hai, ya laali aas-paas phail rahi hai?",
    ),
    Question("movement", ["movement_difficulty"], "Kya bachche ko haath/pair hilane mein dikkat ho rahi hai?"),
]


def missing_fields(case: dict, asked: dict) -> list[Question]:
    """Required triage questions still unanswered. A question asked twice with no
    usable answer is treated as 'unknown' so the flow never gets stuck."""
    out = []
    for q in QUESTIONS:
        if asked.get(q.key, 0) >= 2:
            continue
        vals = [case.get(f) for f in q.fields]
        if q.key == "body_parts":
            if not case.get("body_parts"):
                out.append(q)
        elif q.key == "red_flags":
            if case.get("breathing_difficulty") is None and case.get("unconscious") is None:
                out.append(q)
        elif all(v is None for v in vals):
            out.append(q)
    return out


def missing_personal(case: dict, asked: dict) -> list[Question]:
    return [q for q in PERSONAL_QUESTIONS if case.get(q.fields[0]) in (None, "") and asked.get(q.key, 0) < 2]


# ---------------------------------------------------------------------------
# Triage
# ---------------------------------------------------------------------------
@dataclass
class Triage:
    level: str  # RED / YELLOW / GREEN
    reasons: list[str] = field(default_factory=list)
    route: str = ""
    followup_hours: int = 24

    def to_dict(self) -> dict:
        return {"level": self.level, "reasons": self.reasons, "route": self.route, "followup_hours": self.followup_hours}


def red_flags(case: dict) -> list[str]:
    """Signs that need emergency care immediately, regardless of other details."""
    r = []
    if case.get("breathing_difficulty"):
        r.append("saans lene mein dikkat / dhuaan (airway burn ka khatra)")
    if case.get("unconscious"):
        r.append("bachcha behosh ya bahut sust hai")
    if case.get("mechanism") == "electrical":
        r.append("bijli ka jalna (andar ki chot aur dil ki dhadkan par asar ho sakta hai)")
    if case.get("mechanism") == "chemical":
        r.append("chemical / tezaab se jalna")
    if case.get("depth") == "full":
        r.append("gehra jalna (chamdi safed/kaali/sakht)")
    if case.get("size") == "large" or (case.get("tbsa_percent") or 0) >= 10:
        r.append("bada hissa jala hai (10% se zyada)")
    if case.get("circumferential"):
        r.append("haath/pair/chhati ke chaaron taraf jala hai")
    if case.get("other_injuries"):
        r.append("saath mein koi aur gambhir chot")
    parts = set(case.get("body_parts") or [])
    if "eye" in parts:
        r.append("aankh jali hai")
    if (case.get("face_neck") or parts & {"face", "neck"}) and (
        case.get("mechanism") == "flame" or case.get("depth") in ("partial", "full")
    ):
        r.append("chehre/gardan par aag se ya chhale wala jalna (saans ki nali ka khatra)")
    if "genitals" in parts and case.get("depth") in ("partial", "full"):
        r.append("gupt ang par chhale wala jalna")
    age = case.get("child_age_years")
    if age is not None and age < 1 and case.get("depth") in ("partial", "full") and case.get("size") not in ("coin", None):
        r.append("1 saal se chhota bachcha, chhale wala jalna")
    return r


def triage(case: dict) -> Triage:
    reds = red_flags(case)
    if reds:
        return Triage(
            "RED",
            reds,
            f"Turant {EMERGENCY_NUMBER} ambulance / nazdeeki PHC-CHC. Case PHC doctor ko transfer kiya ja raha hai.",
            followup_hours=2,
        )

    yellow = []
    parts = set(case.get("body_parts") or [])
    depth = case.get("depth")
    size = case.get("size")
    age = case.get("child_age_years")
    if depth == "partial" and size not in ("coin",):
        yellow.append("chhale wala jalna jo sikke se bada hai")
    if depth == "partial" and parts & SPECIAL_AREAS:
        yellow.append("chhale haath/pair/chehra/jod jaisi khaas jagah par hain")
    if parts & {"face", "neck", "genitals"}:
        yellow.append("chehra/gardan/gupt ang jaisi naazuk jagah")
    if size in ("palm", "few_palms"):
        yellow.append("jalne ka hissa hatheli ya usse bada hai")
    if age is not None and age < 5 and depth == "partial":
        yellow.append("5 saal se chhota bachcha, chhale ke saath")
    if age is not None and age < 1:
        yellow.append("1 saal se chhota bachcha")
    if case.get("mechanism") == "flame":
        yellow.append("aag se jalna aksar gehra hota hai")
    if case.get("pain") == "severe":
        yellow.append("bahut tez dard")
    if depth in (None, "unknown") or size in (None, "unknown"):
        yellow.append("jalne ki gehrai/size pakka pata nahi (savdhani ke liye)")
    # follow-up worsening signs
    if case.get("fever") or case.get("pus_or_smell") or case.get("redness_spreading"):
        yellow.append("infection ke lakshan (bukhaar / pus / laali phailna)")
    # a blister growing a little in the first 1-2 days is common; only escalate with rising pain
    if case.get("pain_trend") == "worse":
        yellow.append("haalat bigad rahi hai (dard badh raha hai)")
    if case.get("movement_difficulty"):
        yellow.append("haath/pair hilane mein dikkat")
    img = case.get("image_findings") or {}
    if img.get("depth_guess") in ("partial", "full") and depth == "superficial":
        yellow.append("photo mein chhale/gehra jalna dikh raha hai")
    if img.get("infection_signs"):
        yellow.append("photo mein infection ke nishaan")

    if yellow:
        return Triage(
            "YELLOW",
            yellow,
            "Aaj hi (24 ghante ke andar) nazdeeki PHC / ANM / doctor ko dikhayein. Ghar par first aid jaari rakhein.",
            followup_hours=24,
        )
    return Triage(
        "GREEN",
        ["halka, chhota jalna (sirf laali ya sikke jitna chhala), khaas jagah par nahi"],
        "Ghar par dekhbhaal kaafi hai. Neeche diye lakshan dikhein toh PHC jaayein.",
        followup_hours=48,
    )


# ---------------------------------------------------------------------------
# Advice text (WHO burn first aid)
# ---------------------------------------------------------------------------
def immediate_cooling(case: dict) -> str:
    mech = case.get("mechanism")
    if mech == "electrical":
        return (
            "PEHLE: bijli ka main switch band karein. Bachche ko sookhi lakdi se hi alag karein, "
            "nange haath se mat chhuiye. Phir jali jagah par 20 minute saaf thanda (barf nahi) paani daalein."
        )
    if mech == "chemical":
        return (
            "Chemical wale kapde turant utaarein (dastane/plastic se). Sookha powder (jaise chuna) pehle jhaad dein, "
            "phir jagah ko kam se kam 20 minute behte saaf paani se dhoyein. Aankh mein gaya ho toh aankh khol kar "
            "lagatar paani daalein."
        )
    lead = ("Agar kapdon mein aag hai: bachche ko zameen par letaakar ghumaayein ya kambal se lapet kar aag bujhaayein "
            "(bhaagne na dein). " if case.get("clothes_on_fire") else "")
    return (
        lead + "Jali jagah ko turant 20 minute tak saaf, thande (barf-thande nahi) behte paani ke neeche rakhein. "
        "Anguthi, choodi, tight kapde nikaal dein (jo chamdi se chipke ho unhe mat kheenchein). "
        "Baaki shareer ko kambal se garam rakhein."
    )


DONTS = (
    "MAT lagaiye: toothpaste, ghee, makhan, tel, haldi, gobar, mitti, syahi ya barf. "
    "Chhale mat phodiye. Ruyi (cotton) seedha ghaav par mat rakhiye."
)


def first_aid_advice(case: dict, level: str) -> str:
    steps = [immediate_cooling(case)]
    steps.append(
        "Thanda karne ke baad jagah ko saaf, dhule hue sooti kapde ya clean plastic wrap (cling film) se "
        "dheele se dhakein."
    )
    steps.append(DONTS)
    if level != "RED":
        steps.append(
            "Dard ke liye paracetamol syrup de sakte hain, dose bachche ki umar/wazan ke hisaab se "
            "(ASHA/ANM ya dawai ke dabbe se confirm karein)."
        )
        steps.append("Bachche ko paani / ORS / maa ka doodh pilate rahein. Tetanus ka tika laga hai ya nahi, check karein.")
        steps.append("Ghaav ko saaf aur sookha rakhein, roz dekhein.")
    return "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))


EMERGENCY_TRANSPORT = (
    "Hospital le jaate waqt: jagah ko saaf kapde se dhakein, bachche ko kambal mein garam rakhein, "
    "khaana-peena mat dijiye jab tak doctor na kahe (saans ki dikkat ho toh bilkul nahi)."
)

WHEN_TO_SEEK_CARE = (
    "Turant doctor/PHC jaayein agar: bukhaar aaye, ghaav se pus ya badboo aaye, laali phailne lage, "
    "dard badhta jaaye, bachcha peena band kare ya sust ho, ya 10-14 din mein ghaav na bhare."
)

# Short FAQ used when no LLM is available (and as grounding for the LLM).
FAQ = [
    (("haldi", "turmeric", "हल्दी"), "Haldi mat lagaiye. Isse ghaav mein infection aur gandagi ho sakti hai, aur doctor ko ghaav dekhne mein dikkat hoti hai."),
    (("toothpaste", "colgate", "paste", "टूथपेस्ट"), "Toothpaste mat lagaiye. Yeh garmi andar hi rok leta hai aur infection ka khatra badhata hai. Sirf thanda behta paani 20 minute."),
    (("ghee", "makhan", "butter", "tel", "oil", "घी", "मक्खन"), "Ghee, makhan ya tel mat lagaiye. Isse jalan andar tak jaati hai aur infection hota hai."),
    (("barf", "ice", "बर्फ"), "Barf mat lagaiye. Barf se chamdi aur kharab hoti hai aur chhote bachche thande pad sakte hain. Saadha thanda paani use karein."),
    (("phod", "phodna", "burst", "फोड़"), "Chhale mat phodiye. Chhala ek prakritik dhakkan hai jo infection se bachata hai. Apne aap phoot jaaye toh saaf rakhein aur PHC par dressing karvaayein."),
    (("gobar", "mitti", "गोबर", "मिट्टी"), "Gobar ya mitti bilkul mat lagaiye. Isse tetanus aur gambhir infection ho sakta hai."),
    (("dawai", "medicine", "cream", "silver", "burnol", "दवा", "क्रीम"), "Koi bhi cream doctor/ANM se poochh kar hi lagaiye. PHC par silver sulfadiazine jaisi dawa di ja sakti hai. Tab tak ghaav saaf kapde se dhak kar rakhein."),
    (("nahla", "nahana", "bath", "नहा"), "Bachche ko nahla sakte hain, ghaav ko saaf paani se halka dhoyein aur ragadiye nahi. Phir saaf kapde se dhakein."),
]
