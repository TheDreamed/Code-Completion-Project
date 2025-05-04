import os, random, ast, logging, time
from flask import Flask, request, jsonify, render_template, make_response
import anthropic
from dotenv import load_dotenv
from anthropic import RateLimitError           # ←  NEW

###############################################################################
# Configuration
###############################################################################
load_dotenv()
API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    raise RuntimeError("Set ANTHROPIC_API_KEY in .env or your shell.")
client = anthropic.Anthropic(api_key=API_KEY)

app = Flask(__name__)
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

###############################################################################
# Anthropic helpers (unchanged)
###############################################################################
def _block_to_text(b) -> str:
    if isinstance(b, dict) and b.get("type") == "text":
        return b.get("text", "")
    return getattr(b, "text", "") or ""

def _join_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_block_to_text(b) for b in content)
    return ""

# ────────────────────────────────────────────────────────────────────────────
# 1. MINI‑GA ADD‑ON  (pop ≤ 5, 1 generation)
# ────────────────────────────────────────────────────────────────────────────
def _is_valid_python(txt: str) -> bool:
    try:
        ast.parse(txt, mode="exec")
        return True
    except SyntaxError:
        return False

def _fitness(candidate: str, need_space: bool) -> float:
    if not candidate:                                   # empty → worst
        return 0
    if need_space and not candidate.startswith((" ", "\n")):
        return 0                                        # violates spacing rule
    score = 1 if _is_valid_python(candidate) else 0     # syntax bonus
    score += max(0, 8 - len(candidate))                 # brevity bonus
    return score

def _sample_completion(prefix: str,
                       temperature: float,
                       max_tokens: int = 16) -> str:
    """Single call to Claude; returns '' on rate‑limit errors."""
    try:
        resp = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            system=SYSTEM_PROMPT,        # ← keeps your prompt intact
            messages=[{"role": "user", "content": prefix}],
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=[]
        )
        return _join_content(resp.content).rstrip('\n')
    except RateLimitError:
        time.sleep(0.3)                  # brief back‑off, then skip
        return ""
    except Exception as exc:
        app.logger.exception("Anthropic error during sampling: %s", exc)
        return ""

def ga_best_completion(prefix: str,
                       pop_size: int = 5,
                       base_temp: float = 0.7,
                       sigma: float = 0.2):
    need_space = not prefix.endswith(("\n", " ", "\t"))

    def legal_temp():
        return min(1.0, max(0.1, random.gauss(base_temp, sigma)))

    # Initial samples
    population = [_sample_completion(prefix, legal_temp()) for _ in range(pop_size)]

    # ► NEW: filter out rate‑limited blanks; if all blank, just return ''
    population = [c for c in population if c]
    if not population:
        return ""

    ranked = sorted(population, key=lambda c: _fitness(c, need_space), reverse=True)

    # ► NEW: if top fitness is 0, still return the first non‑empty candidate
    best = ranked[0]
    if _fitness(best, need_space) == 0:
        return best
    app.logger.debug("GA picked: %r", best) 
    return best



###############################################################################
# Original SYSTEM_PROMPT, next_tokens, apply_edit …  unchanged
###############################################################################
SYSTEM_PROMPT = """
You are an advanced *Python* coding‑assistant that returns **only** the next code
tokens that naturally follow the user‑supplied snippet.

✦ General rules
• Output raw code *exactly* as it should be inserted—no wrapping in markdown,
  no commentary.
• Never repeat any part of the snippet that was already provided.
• Absolutely no explanations or apologies.

✦ Whitespace rules
• If the continuation is **on the same line** as the cursor, **begin with a
  single space** before the first non‑space character (e.g. "` return x`").
• If the continuation starts **on a new line**, start with a newline followed
  by the correct indentation (spaces) for that scope.
• Preserve Python indentation levels exactly; do not trim leading spaces.

Samples of bad suggestion based on the current code:
Current code:
class Solution(object):
    def twoSum(self, nums, target):
    
        prevMap = {} # val : index

        for i, n in enumerate(nums):
            diff = target -n
            if diff in prevMap:
                return[prevMap[diff], i]
            prevMap[n] = i

Code Suggestion:
return[] (the indentation is wrong here and the spacing is incorrect too. This should be printed below the previous line)
"""
   # (same block you provided)

def next_tokens(code:str, *, max_tokens:int=12) -> str:
    try:
        resp = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            system=SYSTEM_PROMPT,
            messages=[{"role":"user","content":code}],
            max_tokens=max_tokens,
            temperature=0.1,
            stop_sequences=[]
        )
        return _join_content(resp.content).rstrip('\n')
    except Exception as exc:
        app.logger.exception("Anthropic error: %s", exc)
        return ""

def apply_edit(code:str, instruction:str) -> tuple[str,str]:
    ...  # (unchanged)

###############################################################################
# Routes (only /autocomplete modified)
###############################################################################
@app.route("/")
def index():
    return render_template("index.html")

@app.post("/autocomplete")
def autocomplete():
    snippet = request.json.get("snippet", "")
    suggestion = ga_best_completion(snippet)      # ←  now GA‑powered
    resp = make_response(jsonify({"suggestion": suggestion}))
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp

@app.post("/chat")
def chat():
    data = request.json or {}
    code = data.get("code", "")
    instruction = data.get("instruction", "")
    new_code, reply = apply_edit(code, instruction)
    resp = make_response(jsonify({"code": new_code, "reply": reply}))
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp

###############################################################################
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
