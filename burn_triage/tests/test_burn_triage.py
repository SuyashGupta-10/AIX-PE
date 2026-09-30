"""Deterministic tests (rules only, no LLM). Run:  python -m pytest tests  (or python tests/test_burn_triage.py)"""
import os
import sys
import tempfile
from pathlib import Path

os.environ["QWEN_BACKEND"] = "none"
os.environ["DB_PATH"] = str(Path(tempfile.mkdtemp()) / "test.db")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import dialogue, protocol  # noqa: E402
from app.nlu_rules import extract, script_of  # noqa: E402

# The three burn prompts from the project brief (Devanagari / Roman Hinglish / messy code-mixed)
BRIEF_PROMPTS = [
    ("Devanagari Hindi", "मेरे बच्चे को जल गया है, क्या करूं?"),
    ("Roman Hinglish", "Mera bacche ko jal gaya hai, kya karu?"),
    ("Messy code-mixed", "baby ka hand jal gaya stove se, kitna serious hai ye? urgent hai kya"),
]


def talk(*msgs, **kw):
    s, _ = dialogue.start_session(**kw)
    out = None
    for m in msgs:
        out = dialogue.handle(s.id, m)
    return s, out


def test_brief_prompts_detected_as_burn():
    for style, p in BRIEF_PROMPTS:
        s, out = talk(p)
        assert s.case["is_burn"], style
        assert "20 minute" in out["reply"], style  # immediate cooling advice comes first
        assert out["pending"] == ["breathing_difficulty", "face_neck", "unconscious"], style  # red-flag screen next


def test_script_detection():
    assert script_of(BRIEF_PROMPTS[0][1]) == "devanagari"
    assert script_of(BRIEF_PROMPTS[1][1]) == "roman"


def test_messy_extraction():
    e = extract(BRIEF_PROMPTS[2][1])
    assert e["body_parts"] == ["hand"] and e["mechanism"] == "flame"


def test_negation_and_false_positives():
    assert extract("saans lene mein dikkat nahi hai")["breathing_difficulty"] is False
    assert extract("saans theek hai").get("breathing_difficulty") is False
    assert "is_burn" not in extract("jaldi batao")          # "jaldi" is not "jal"
    assert "head" not in extract("sirf laal hai").get("body_parts", [])  # "sirf" is not "sir"
    assert "other_injuries" not in extract("garam chai gir gayi haath par")  # spill, not a fall
    assert extract("chhale nahi hain sirf laal hai")["depth"] == "superficial"
    assert "harmful_remedy" not in extract("kya haldi laga sakte hain?")
    assert extract("haldi lagayi thi")["harmful_remedy"] == ["haldi"]
    assert "mechanism" not in extract("maine thanda paani daala")  # first aid, not a scald
    assert extract("garam paani gir gaya")["mechanism"] == "scald"


def test_stop_drop_roll_only_for_clothes_fire():
    assert "ghumaayein" not in protocol.immediate_cooling(extract("hand jal gaya stove se"))
    assert "ghumaayein" in protocol.immediate_cooling(extract("dupatte mein aag lag gayi"))


def test_yes_to_multipart_question():
    P = ["breathing_difficulty", "face_neck", "unconscious"]
    e = extract("Haan, saans lene mein dikkat hai", P)
    assert e["breathing_difficulty"] is True and "unconscious" not in e and "face_neck" not in e
    assert all(extract("haan", P)[f] for f in P)  # bare yes: assume all (safe side)


def test_age_parsing():
    assert extract("6 mahine ka hai")["child_age_years"] == 0.5
    assert extract("dedh saal ki beti")["child_age_years"] == 1.5
    assert extract("३ साल का बच्चा")["child_age_years"] == 3


def test_red_flags_escalate_immediately():
    for msg in ["bachche ko current laga, haath jal gaya",
                "tezaab gir gaya pair par",
                "kapdon mein aag lagi, saans lene mein dikkat ho rahi hai",
                "पूरा पैर गरम पानी से जल गया, छाले हैं"]:
        s, out = talk(msg)
        assert out["triage"]["level"] == "RED", msg
        assert "108" in out["reply"], msg
        assert out["escalation_id"], msg


def test_green_case():
    s, out = talk("beti ka pair press se jal gaya, sirf laal hai, sikke jitna", "8 saal", "nahi")
    assert out["triage"]["level"] == "GREEN"


def test_yellow_blister_on_hand_small_child():
    s, out = talk("garam tel haath pe gir gaya", "nahi", "chhale hain, chhota sa area", "3 saal")
    assert out["triage"]["level"] == "YELLOW"
    assert out["phase"] == "details"


def test_unknowns_do_not_block_and_default_to_yellow():
    s, out = talk("bachcha jal gaya", "nahi", "pata nahi", "pata nahi", "pata nahi", "pata nahi", "5 saal", "pata nahi")
    assert out["triage"] and out["triage"]["level"] in ("YELLOW", "RED")


def test_full_flow_then_followup():
    s, out = talk("garam doodh gir gaya haath pe", "nahi", "chhale pad gaye, hatheli jitna", "2 saal",
                  "Aarav", "9811122233", "Sanganer")
    assert out["phase"] == "closed" and out["followup_due"]
    fs, fout = talk("dard kam hai", "chhale waise hi hain", "nahi", "nahi", mode="followup", phone="9811122233")
    assert fout["phase"] == "closed"
    assert "Achhi baat" in fout["reply"]
    fs, fout = talk("dard zyada hai", mode="followup", phone="9811122233")
    assert "aaj hi" in fout["reply"]  # worsening pain ends questioning immediately
    fs, fout = talk("dard kam hai", "waise hi", "haan bukhaar hai aur pus bhi", mode="followup", phone="9811122233")
    assert "aaj hi" in fout["reply"] and fout["case"]["fever"] and fout["case"]["pus_or_smell"]


def test_other_condition_routed():
    s, out = talk("kutte ne bacche ko kaat liya")
    assert out["phase"] == "handoff" and "rabies" in out["reply"]


def test_triage_never_lowered_by_image():
    case = protocol.empty_case()
    case.update(depth="partial", size="palm", body_parts=["hand"], child_age_years=4)
    before = protocol.triage(case).level
    case["image_findings"] = {"depth_guess": "superficial"}
    assert protocol.triage(case).level == before


def test_reassuring_photo_does_not_skip_questions():
    from app import nlu

    orig = nlu.assess_image
    nlu.assess_image = lambda *_: {"is_burn_likely": True, "body_part": "hand", "depth_guess": "superficial",
                                   "size_guess": "coin", "infection_signs": False, "image_quality": "good",
                                   "description": ""}
    try:
        s, _ = dialogue.start_session()
        out = dialogue.handle(s.id, "haath jal gaya", image=(b"x", "image/jpeg"))
        assert out["case"]["depth"] is None and out["case"]["size"] is None
        nlu.assess_image = lambda *_: {"is_burn_likely": True, "body_part": "hand", "depth_guess": "full",
                                       "size_guess": "unclear", "infection_signs": False, "image_quality": "good",
                                       "description": ""}
        out = dialogue.handle(s.id, "", image=(b"x", "image/jpeg"))
        assert out["case"]["depth"] == "full" and out["triage"]["level"] == "RED"
    finally:
        nlu.assess_image = orig


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("PASS", name)
            except AssertionError as e:
                fails += 1
                print("FAIL", name, e)
    sys.exit(1 if fails else 0)
