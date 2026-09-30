"""Compare Qwen3 extraction vs the keyword rules on Hinglish burn messages.

    python tests/eval_qwen.py                 # uses QWEN_BACKEND / QWEN_MODEL from env or .env
Prints per-message fields from each source and latency.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, nlu, nlu_rules  # noqa: E402
from app.llm import get_client  # noqa: E402

CASES = [
    # (message, pending fields, expected subset)
    ("मेरे बच्चे को जल गया है, क्या करूं?", [], {"intent": "burn", "_absent": ["depth", "body_parts", "mechanism"]}),
    ("Mera bacche ko jal gaya hai, kya karu?", [], {"intent": "burn", "_absent": ["depth", "body_parts", "mechanism"]}),
    ("baby ka hand jal gaya stove se, kitna serious hai ye? urgent hai kya", [], {"body_parts": ["hand"]}),
    ("Jab main cooking kar rahi thi garam tel mere haath pe gir gaya.", [], {"mechanism": "hot_oil", "body_parts": ["hand"]}),
    ("Nahi, sirf haath pe hua hai.", ["breathing_difficulty", "face_neck", "unconscious"],
     {"breathing_difficulty": False, "unconscious": False}),
    ("Pan se gir gaya. Right hand hai. Red hai aur chhote chhote chhale hain. Sirf haath ke upar chhota area hai.",
     ["depth", "size"], {"depth": "partial"}),
    ("बेटी के ऊपर उबलती चाय गिर गई, पेट पर फफोले हैं, 2 साल की है", [],
     {"mechanism": "scald", "depth": "partial", "child_age_years": 2}),
    ("bijli ke taar se chipak gaya tha, ungliyan kaali ho gayi hain", [], {"mechanism": "electrical", "depth": "full"}),
    ("ladke ke kapdon mein diya se aag lag gayi, chest aur gardan jal gaye, awaaz bhaari ho gayi", [],
     {"mechanism": "flame", "breathing_difficulty": True}),
    ("6 mahine ka hai", ["child_age_years"], {"child_age_years": 0.5}),
    ("usne toothpaste laga diya tha", [], {"harmful_remedy": ["toothpaste"]}),
    # answers the keyword lists do not cover: this is where Qwen adds value
    ("bas thodi si lalai hai, aur kuch nahi", ["depth", "size"], {"depth": "superficial"}),
    ("ungli ke pore jitna hi hai", ["size"], {"size": "coin"}),
    ("do-teen jagah pani wale dane nikal aaye", ["depth"], {"depth": "partial"}),
]


def matches(got: dict, exp: dict) -> bool:
    for k, v in exp.items():
        if k == "_absent":  # nothing in the message supports these: filling them is a hallucination
            if any(got.get(f) for f in v):
                return False
            continue
        g = got.get(k)
        if isinstance(v, list):
            if not g or not set(v) <= set(g):
                return False
        elif isinstance(v, float) or isinstance(v, int) and not isinstance(v, bool):
            if g is None or abs(float(g) - v) > 0.01:
                return False
        elif g != v:
            return False
    return True


def main():
    client = get_client()
    print(f"backend={client.backend_name} model={config.QWEN_MODEL}\n")
    score = {"rules": 0, "qwen": 0, "merged": 0}
    for msg, pending, exp in CASES:
        rules = nlu_rules.extract(msg, pending)
        if "intent" in exp:
            rules = {**rules, "intent": "burn" if rules.get("is_burn") else None}
        t0 = time.time()
        merged, meta = nlu.understand(msg, pending, {})
        dt = time.time() - t0
        qwen = meta["llm"] or {}
        merged = {**merged, "intent": "burn" if merged.get("is_burn") or qwen.get("intent") == "burn" else None}
        r, q, m = matches(rules, exp), matches(qwen, exp), matches(merged, exp)
        score["rules"] += r
        score["qwen"] += q
        score["merged"] += m
        print(f"MSG: {msg}\n  expected: {exp}\n  rules  {'OK ' if r else 'MISS'} {json.dumps(rules, ensure_ascii=False)}"
              f"\n  qwen   {'OK ' if q else 'MISS'} {json.dumps(qwen, ensure_ascii=False)}  ({dt:.1f}s)"
              f"\n  merged {'OK ' if m else 'MISS'} {json.dumps(merged, ensure_ascii=False)}"
              f"  (dropped from qwen: {meta.get('llm_dropped')})\n")
    n = len(CASES)
    print(f"Accuracy on {n} cases: rules {score['rules']}/{n}, qwen {score['qwen']}/{n}, merged {score['merged']}/{n}")
    if client.last_error:
        print("last LLM error:", client.last_error)


if __name__ == "__main__":
    main()
