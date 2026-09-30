"""Keyword-based Hinglish / Hindi / English extractor.

Qwen3 does the main understanding. This module runs on every message anyway:
  * it is the fallback when the LLM is unavailable or returns bad JSON, and
  * its red-flag hits are OR-ed with the LLM's, so a model mistake cannot hide
    an emergency.
Handles Roman Hinglish, Devanagari and messy code-mixed text, with simple
clause-level negation ("saans mein dikkat nahi hai").
"""
from __future__ import annotations

import re

DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def script_of(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "roman"
    dev = sum(1 for c in letters if DEVANAGARI.match(c))
    return "devanagari" if dev / len(letters) > 0.3 else "roman"


def _norm(text: str) -> str:
    t = text.lower()
    t = t.replace("‍", "").replace("‌", "")
    return t


NEG = r"(nahi|nahin|nahee|nai|nhi|no|not|mat|na|नहीं|नही|ना|मत|kuch nahi)"


def _clauses(t: str) -> list[str]:
    return [c.strip() for c in re.split(r"[,.;!?\n|।]|\b(?:aur|lekin|par|but|and)\b|और|लेकिन", t) if c and c.strip()]


_B = r"(?<![a-zऀ-ॿ])"
_E = r"(?![a-zऀ-ॿ])"


def _pat(w: str) -> str:
    # short words must match as whole words ("sir" != "sirf", "jal" != "jaldi");
    # longer stems may take suffixes ("haath" -> "haathon")
    return _B + re.escape(w) + (_E if len(w) <= 4 else "")


def _has(t: str, words) -> bool:
    return any(re.search(_pat(w), t) for w in words)


def _find(t: str, words) -> tuple[bool, bool]:
    """(mentioned, negated) looking clause by clause."""
    hit, neg = False, False
    for c in _clauses(t):
        if _has(c, words):
            hit = True
            if re.search(_B + NEG + _E, c):
                neg = True
            else:
                return True, False
    return hit, neg


BURN_WORDS = ["jal gay", "jal gai", "jal ga", "jala", "jali", "jalne", "jalna", "jalan", "jal", "jhulas", "burn", "jala", "jali", "jhulsa", "garam", "hot", "chhala", "chhale", "phafol", "fafol",
              "scald", "जल गय", "जल ग", "जला", "जली", "जलने", "जलन", "झुलस", "छाल", "फफोल", "गरम", "गर्म", "tezaab", "acid", "current", "करंट", "तेजाब"]

MECH = {
    "electrical": ["current", "bijli", "electric", "shock", "wire", "taar", "करंट", "बिजली", "तार"],
    "chemical": ["acid", "tezaab", "tezab", "chemical", "chuna", "bleach", "harpic", "phenyl", "तेजाब", "तेज़ाब", "एसिड", "चूना", "केमिकल"],
    "hot_oil": ["tel", "oil", "ghee", "kadhai", "तेल", "घी", "कढ़ाई"],
    "flame": ["aag", "fire", "flame", "lau", "diya se", "diye se", "diya jal", "deepak", "dhibri", "lamp", "kerosene", "mitti ka tel", "patakh", "cracker",
              "आग", "लौ", "दीये", "दीपक", "पटाख", "कपड़ों में आग"],
    # bare "paani" is NOT a scald: "thanda paani daala" is first aid
    "scald": ["garam paani", "garam pani", "garm paani", "hot water", "chai", "tea", "doodh", "milk", "daal", "dal",
              "steam", "bhaap", "cooker", "ubal", "boil", "khaulta", "soup", "गरम पानी", "गर्म पानी", "चाय", "दूध", "दाल",
              "भाप", "उबल", "खौल", "कुकर"],
    "contact": ["tawa", "istri", "press", "iron", "silencer", "bartan", "pateela", "handle", "chimta", "तवा", "इस्त्री",
                "प्रेस", "साइलेंसर", "बर्तन"],
}
# "chulha/stove" alone is ambiguous: could be flame or a hot pan. Default to flame
# (more conservative) only if nothing more specific is found.
CLOTHES = ["kapde", "kapdon", "kapda", "clothes", "dupatt", "saree", "sari", "frock", "kurta", "कपड़", "दुपट्टा", "साड़ी"]
STOVE = ["chulha", "chulhe", "stove", "gas", "angithi", "चूल्हा", "चूल्हे", "स्टोव", "अंगीठी"]

PARTS = {
    "fingers": ["ungli", "ungliyan", "ungliyon", "finger", "उंगली", "उँगली"],
    "hand": ["haath", "hath", "hand", "hands", "hatheli", "palm", "हाथ", "हथेली"],
    "arm": ["baanh", "banh", "bazu", "arm", "kalai", "wrist", "बांह", "बाँह", "कलाई", "बाजू"],
    "foot": ["pair", "pairon", "paon", "paanv", "pao", "foot", "feet", "leg", "legs", "taang", "tang", "talwa", "पैर", "पांव", "टांग", "तलवा"],
    "thigh": ["jaangh", "jangh", "thigh", "जांघ"],
    "face": ["chehra", "chehre", "muh", "munh", "face", "gaal", "cheek", "honth", "lip", "naak", "nose", "चेहरा", "मुंह", "मुँह", "गाल", "होंठ", "नाक"],
    "eye": ["aankh", "ankh", "eye", "आंख", "आँख"],
    "neck": ["gardan", "neck", "gala", "गर्दन", "गला"],
    "head": ["sir", "sar", "head", "scalp", "सिर", "सर"],
    "chest": ["chhati", "chati", "seena", "chest", "छाती", "सीना"],
    "abdomen": ["pet", "stomach", "tummy", "belly", "पेट"],
    "back": ["peeth", "pith", "back", "kamar", "पीठ", "कमर"],
    "genitals": ["private", "gupt", "genital", "nunu", "susu", "potty ki jagah", "गुप्तांग", "गुप्त"],
    "joint": ["kohni", "elbow", "ghutna", "ghutne", "knee", "joint", "kandha", "shoulder", "कोहनी", "घुटन", "कंधा"],
    "buttocks": ["kulha", "hips", "buttock", "chootad", "कूल्हे"],
}

DEPTH_FULL = ["safed", "white", "kaala", "kala", "kaali", "black", "charred", "sakht", "hard", "leathery", "sunn",
              "numb", "dard nahi", "dard nahin", "सफेद", "काला", "काली", "सख्त", "सुन्न"]
DEPTH_PARTIAL = ["chhala", "chhale", "chala", "chale", "chhaale", "phafola", "phafole", "fafola", "fafole", "blister",
                 "chamdi utar", "khaal utar", "skin peel", "geela", "paani bhar", "bubble", "paani wale", "pani wale",
                 "paani ke daane", "pani ke dane", "daane", "dane nikal", "paani ki thaili",
                 "छाला", "छाले", "फफोला", "फफोले", "खाल उतर", "चमड़ी उतर"]
DEPTH_SUPERFICIAL = ["laal", "lal", "red", "redness", "laali", "lali", "lalai", "lalima", "sirf laal", "लाल", "लाली", "ललाई"]

# Checked in this order; first hit wins.
SIZES_ORDERED = [
    ("large", ["poora shareer", "pura shareer", "pura sharir", "whole body", "poori baanh", "puri baanh", "poora pair",
               "pura pair", "poori taang", "puri tang", "whole arm", "whole leg", "whole limb", "entire", "bahut bada",
               "bahut badi", "bahut zyada jagah", "kai jagah", "multiple", "पूरा शरीर", "पूरा पैर", "बहुत बड़ा", "कई जगह"]),
    ("few_palms", ["2 hatheli", "3 hatheli", "do hatheli", "teen hatheli", "2-3 hatheli", "2 palm", "3 palm", "few palm",
                   "kaafi bada", "bada hissa", "badi jagah", "bada area", "large area", "poora", "pura", "puri", "poori",
                   "whole", "पूरा", "पूरी", "बड़ा हिस्सा", "बड़ी जगह"]),
    ("palm", ["hatheli jitna", "hatheli jitni", "hatheli jaisa", "hatheli ke barabar", "ek hatheli", "1 hatheli",
              "palm size", "palm jitna", "हथेली जितना", "हथेली जितनी", "हथेली के बराबर"]),
    ("coin", ["aadhi hatheli", "hatheli se aadha", "hatheli se aadhi", "hatheli se chhota", "hatheli se chhoti",
              "half palm", "sikka", "sikke", "coin", "chhota sa", "chhoti si", "chota sa", "choti si", "chhota area", "chota area",
              "chhoti jagah", "choti jagah", "small area", "pore jitna", "pore jitni", "nakhun jitna", "nakhun jitni", "small burn", "thoda sa", "thodi si", "tiny", "zara sa",
              "सिक्के", "छोटा सा", "छोटी सी", "जरा सा", "थोड़ा सा"]),
]

BREATH = ["saans", "sans", "saas", "breath", "breathing", "dam ghut", "dhuaan", "dhuan", "dhuwa", "smoke",
          "awaaz bhaari", "awaz bhari", "hoarse", "khansi", "खांसी", "सांस", "साँस", "धुआं", "धुआँ", "दम घुट"]
UNCONSCIOUS = ["behosh", "unconscious", "hosh nahi", "bol nahi raha", "sust", "drowsy", "faint", "jhatke",
               "fits", "बेहोश", "सुस्त", "होश नहीं", "झटके"]
BREATH_PROBLEM = ["dikkat", "dikat", "takleef", "taklif", "problem", "pareshani", "tez", "ruk", "nahi le", "ghut",
                  "difficulty", "trouble", "hard", "phool", "fool", "दिक्कत", "तकलीफ", "परेशानी", "रुक", "घुट", "फूल"]
BREATH_ALONE = ["dhuaan", "dhuan", "dhuwa", "smoke", "awaaz bhaari", "awaz bhari", "hoarse", "dam ghut", "धुआं", "धुआँ", "दम घुट"]
BREATH_OK = ["theek", "thik", "normal", "sahi", "fine", "ok", "ठीक", "सही"]
CIRCUMFERENTIAL = ["chaaron taraf", "charo taraf", "chaaro or", "gol ghera", "all around", "around the", "चारों तरफ"]
# note: "gir gaya" usually means the hot liquid spilled, so it is NOT an injury signal
OTHER_INJ = ["fracture", "haddi toot", "khoon beh", "bleeding", "sir mein chot", "sar pe chot", "upar se gir",
             "chhat se", "हड्डी टूट", "खून बह"]

HARMFUL = {
    "toothpaste": ["toothpaste", "colgate", "paste", "टूथपेस्ट"],
    "ghee/makhan/tel": ["ghee laga", "makhan", "butter", "tel laga", "oil laga", "घी लगा", "मक्खन"],
    "haldi": ["haldi", "turmeric", "हल्दी"],
    "barf": ["barf", "ice", "बर्फ"],
    "gobar/mitti": ["gobar", "mitti laga", "गोबर", "मिट्टी लगा"],
    "syahi": ["ink", "syahi", "स्याही"],
}
COOLED = ["thanda paani", "thande paani", "cold water", "paani daala", "paani dala", "nal ke neeche", "tap water",
          "ठंडा पानी", "ठंडे पानी", "पानी डाला"]

PAIN_SEVERE = ["bahut dard", "bohot dard", "zyada dard", "jyada dard", "tez dard", "severe", "ro raha", "ro rahi",
               "cheekh", "बहुत दर्द", "तेज दर्द", "रो रहा", "रो रही"]
PAIN_MILD = ["halka dard", "thoda dard", "mild", "kam dard", "हल्का दर्द", "थोड़ा दर्द"]
PAIN_ANY = ["dard", "pain", "painful", "jalan", "दर्द", "जलन"]

HINDI_NUM = {"ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5, "chhe": 6, "chhah": 6,
             "saat": 7, "aath": 8, "nau": 9, "das": 10, "gyarah": 11, "barah": 12, "dedh": 1.5, "dhai": 2.5,
             "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "छह": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10,
             "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_NUM = r"(\d+(?:\.\d+)?|" + "|".join(sorted(HINDI_NUM, key=len, reverse=True)) + r")"
DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

YES = ["haan", "han", "ha", "haa", "yes", "ji haan", "hanji", "ji", "bilkul", "sahi", "हाँ", "हां", "जी"]
NO = ["nahi", "nahin", "nai", "nhi", "no", "na", "nope", "bilkul nahi", "नहीं", "नही", "ना"]


def _num(s: str) -> float:
    s = s.translate(DEVA_DIGITS)
    return float(s) if re.match(r"\d", s) else float(HINDI_NUM[s])


def extract_age(t: str):
    t = t.translate(DEVA_DIGITS)
    m = re.search(_NUM + r"\s*(mahine|mahina|maheene|month|months|mah|महीन|महीने)", t)
    if m:
        return round(_num(m.group(1)) / 12, 2)
    m = re.search(_NUM + r"\s*(saal|sal|year|years|yr|yrs|varsh|baras|साल|वर्ष|बरस)", t)
    if m:
        return _num(m.group(1))
    m = re.search(r"(umar|umr|age|उम्र|उमर)\D{0,12}(\d+(?:\.\d+)?)", t)
    if m:
        return float(m.group(2))
    return None


def has_number(t: str) -> bool:
    return bool(re.search(r"\d|[०-९]|" + _B + "(" + "|".join(HINDI_NUM) + ")" + _E, _norm(t)))


def extract_phone(t: str):
    t = t.translate(DEVA_DIGITS)
    m = re.search(r"(?<!\d)(?:\+?91[\s-]?)?([6-9]\d{4}[\s-]?\d{5})(?!\d)", t)
    return re.sub(r"\D", "", m.group(1)) if m else None


def short_yes_no(text: str):
    t = _norm(text).strip(" .!?।")
    if not t:
        return None
    first = re.split(r"[\s,]+", t)[0] if t else ""
    if first in NO:
        return False
    if first in YES:
        return True
    return None


def extract(text: str, pending: list[str] | None = None) -> dict:
    """Return only the fields found in this message (partial case update)."""
    t = _norm(text)
    out: dict = {}
    pending = pending or []

    if _has(t, BURN_WORDS):
        out["is_burn"] = True

    # mechanism: most specific first
    for mech in ["electrical", "chemical", "hot_oil", "flame", "contact", "scald"]:
        hit, neg = _find(t, MECH[mech])
        if hit and not neg:
            if mech == "hot_oil" and _has(t, ["mitti ka tel", "kerosene", "tel laga", "oil laga"]):
                continue
            out["mechanism"] = mech
            break
    if "mechanism" not in out and _has(t, STOVE):
        out["mechanism"] = "flame"
    if _has(t, CLOTHES) and _has(t, ["aag", "fire", "flame", "lau", "आग"]):
        out["clothes_on_fire"] = True
        out["mechanism"] = "flame"

    # "hatheli jitna / hatheli se aadha" is the palm-size ruler, not the burnt body part
    t_parts = re.sub(r"(aadhi |ek |do |teen |\d )?(hatheli|palm|हथेली)(yon|on)?\s*(se|jitn|jais|ke barabar|size|के बराबर|जितन|से)",
                     " ", t)
    parts = []
    for part, words in PARTS.items():
        hit, neg = _find(t_parts, words)
        if hit and not neg:
            parts.append(part)
    if "fingers" in parts and "hand" not in parts:
        parts.append("hand")
    if parts:
        out["body_parts"] = parts
    if {"face", "neck"} & set(parts):
        out["face_neck"] = True

    for depth, words in (("full", DEPTH_FULL), ("partial", DEPTH_PARTIAL), ("superficial", DEPTH_SUPERFICIAL)):
        hit, neg = _find(t, words)
        if hit and not neg:
            out["depth"] = depth
            break
        if depth == "partial" and hit and neg and "depth" not in out:
            out["depth"] = "superficial"  # "chhale nahi hain" -> only redness

    m = re.search(r"(\d+(?:\.\d+)?)\s*(%|percent|pratishat|प्रतिशत)", t)
    if m:
        pct = float(m.group(1))
        out["tbsa_percent"] = pct
        out["size"] = "coin" if pct < 1 else "palm" if pct < 2 else "few_palms" if pct < 10 else "large"
    else:
        for size, words in SIZES_ORDERED:
            if _has(t, words):
                out["size"] = size
                break

    for c in _clauses(t):
        if _has(c, BREATH_ALONE) or (_has(c, BREATH) and _has(c, BREATH_PROBLEM)):
            neg = bool(re.search(_B + NEG + _E, c))
            out["breathing_difficulty"] = not neg
            if not neg:
                break
        elif _has(c, BREATH) and _has(c, BREATH_OK):
            out.setdefault("breathing_difficulty", False)
    for key, words in (("unconscious", UNCONSCIOUS),
                       ("circumferential", CIRCUMFERENTIAL), ("other_injuries", OTHER_INJ)):
        hit, neg = _find(t, words)
        if hit:
            out[key] = not neg

    harmful = [name for name, words in HARMFUL.items() if _find(t, words) == (True, False)]
    if harmful and _has(t, ["lagaya", "lagayi", "lagaai", "laga diya", "laga di", "laga hai", "lagaa", "daala", "dala", "rakha", "लगाया", "लगाई", "लगा दिया", "लगा दी", "डाला"]):
        out["harmful_remedy"] = harmful
    if _has(t, COOLED):
        out["cooled_with_water"] = True

    if _has(t, PAIN_SEVERE):
        out["pain"] = "severe"
    elif _has(t, PAIN_MILD):
        out["pain"] = "mild"
    elif _find(t, PAIN_ANY) == (True, False) and "dard nahi" not in t:
        out["pain"] = "moderate"

    age = extract_age(t)
    if age is not None and age < 18:
        out["child_age_years"] = age
    elif "child_age_years" in pending:
        m = re.fullmatch(r"\s*" + _NUM + r"\s*", t.translate(DEVA_DIGITS))
        if m:
            out["child_age_years"] = _num(m.group(1))

    phone = extract_phone(t)
    if phone:
        out["caller_phone"] = phone

    if set(pending) & FOLLOWUP_FIELDS:
        _followup(t, pending, out)

    # Bare yes/no answer to the question we just asked
    # to a multi-part question: "haan, saans mein dikkat hai" is a yes to breathing only, while a bare
    # "haan" is a yes to everything asked (the safe reading). A "nahi" covers whatever wasn't stated.
    yn = short_yes_no(text)
    if yn is not None and pending:
        named = [f for f in pending if f in YES_NO_FIELDS and out.get(f) is not None]
        for f in pending:
            if f in YES_NO_FIELDS and not (yn and named):
                out.setdefault(f, yn)

    return out


FOLLOWUP_FIELDS = {"pain_trend", "blister_trend", "fever", "pus_or_smell", "redness_spreading", "movement_difficulty"}
YES_NO_FIELDS = {"breathing_difficulty", "face_neck", "unconscious", "fever", "pus_or_smell", "redness_spreading",
                 "movement_difficulty", "circumferential", "other_injuries"}


BLISTER_WORDS = ["chhal", "chhale", "chala", "chale", "ghaav", "ghav", "zakhm", "blister", "wound", "phafol", "छाल", "घाव", "फफोल"]
REDNESS_WORDS = ["laali", "lali", "redness", "laal", "लाली", "लाल"]
TREND = [
    ("bigger", ["bada ho", "bade ho", "badi ho", "badh", "zyada", "jyada", "zyaada", "worse", "bigger", "badtar",
                "बड़ा हो", "बड़े हो", "बढ़", "ज्यादा", "ज़्यादा"]),
    ("smaller", ["kam", "chhota ho", "chhote ho", "sookh", "sukh", "bhar raha", "better", "behtar", "theek", "aaram",
                 "smaller", "कम", "छोटे हो", "सूख", "आराम", "बेहतर", "ठीक"]),
    ("same", ["same", "waise hi", "vaise hi", "utna hi", "wahi", "barabar", "वैसे ही", "उतना ही", "वही"]),
]
PAIN_MAP = {"bigger": "worse", "smaller": "better", "same": "same"}


def _followup(t: str, pending: list[str], out: dict) -> None:
    for c in _clauses(t):
        trend = next((name for name, words in TREND if _has(c, words)), None)
        if trend is None or _has(c, REDNESS_WORDS):
            continue
        if _has(c, BLISTER_WORDS):
            out["blister_trend"] = trend
        elif _has(c, PAIN_ANY) or pending == ["pain_trend"]:
            out["pain_trend"] = PAIN_MAP[trend]
        elif pending == ["blister_trend"]:
            out["blister_trend"] = trend
    for key, words in (("fever", ["bukhar", "bukhaar", "fever", "taap", "बुखार"]),
                       ("pus_or_smell", ["pus", "peep", "mawaad", "badboo", "smell", "peela paani", "पस", "मवाद", "बदबू"]),
                       ("redness_spreading", ["phail", "spread", "फैल"]),
                       ("movement_difficulty", ["hila nahi", "hilane mein", "move nahi", "mod nahi", "हिला नहीं", "हिलाने में"])):
        hit, neg = _find(t, words)
        if hit:
            out[key] = not neg


OTHER_CONDITIONS = {
    "snake_bite": ["saanp", "sanp", "snake", "सांप", "साँप", "naag"],
    "dog_bite": ["kutta", "kutte", "kutiya", "dog", "कुत्ते", "कुत्ता", "rabies"],
    "diarrhoea": ["dast", "loose motion", "diarrh", "potty baar", "दस्त", "ulti", "vomit"],
    "fever": ["bukhar", "bukhaar", "fever", "बुखार"],
    "fall": ["gir gaya", "gir gayi", "fall", "गिर"],
}


def detect_other_condition(text: str):
    t = _norm(text)
    for cond, words in OTHER_CONDITIONS.items():
        if _has(t, words):
            return cond
    return None


GREETINGS = ["hi", "hello", "namaste", "namaskar", "hey", "helo", "नमस्ते", "नमस्कार", "pranam", "ram ram"]


def is_greeting(text: str) -> bool:
    t = _norm(text).strip(" .!?")
    return len(t.split()) <= 3 and _has(t, GREETINGS)


def is_question(text: str) -> bool:
    t = _norm(text)
    return "?" in t or _has(t, ["kya ", "kaise", "kyun", "kyon", "kab ", "should", "can i", "sakte", "क्या", "कैसे", "कब"])
