import os, random, ast, logging, time
from flask import Flask, request, jsonify, render_template, make_response
import anthropic
from dotenv import load_dotenv
from anthropic import RateLimitError     # for graceful handling

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
# Anthropic helpers
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

###############################################################################
# Prompt (UNCHANGED)
###############################################################################
SYSTEM_PROMPT = """
You are an advanced *Python* coding‑assistant that returns **only** the next code
tokens that naturally follow the user‑supplied snippet. DO NOT REPEAT THE CODE
ALREADY SUPPLIED. You are not allowed to add any comments, explanations, docstrings or anything except the code itself. Do not include any comments like with quotations marks or even #.

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
"""

###############################################################################
# Mini‑Genetic‑Algorithm for picking best completion (≤ 5 calls)
###############################################################################
def _is_valid_python(txt: str) -> bool:
    try:
        ast.parse(txt, mode="exec")
        return True
    except SyntaxError:
        return False

def _fitness(candidate: str, need_space: bool) -> float:
    if not candidate:
        return 0
    if need_space and not candidate.startswith((" ", "\n")):
        return 0
    score = 1 if _is_valid_python(candidate) else 0
    score += max(0, 8 - len(candidate))           # shorter is better
    return score

def _sample_completion(prefix: str, temperature: float, max_tokens: int = 16) -> str:
    try:
        resp = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prefix}],
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=[]
        )
        return _join_content(resp.content).rstrip("\n")
    except RateLimitError:
        app.logger.warning("Rate‑limited sample — returning empty string")
        time.sleep(0.3)
        return ""
    except Exception as exc:
        app.logger.exception("Anthropic error during sampling: %s", exc)
        return ""

def ga_best_completion(prefix: str,
                       pop_size: int = 5,
                       base_temp: float = 0.2,
                       sigma: float = 0.2) -> str:
    need_space = not prefix.endswith(("\n", " ", "\t"))

    def legal_temp():
        return min(1.0, max(0.1, random.gauss(base_temp, sigma)))

    population = [_sample_completion(prefix, legal_temp()) for _ in range(pop_size)]
    population = [c for c in population if c]          # drop blanks
    if not population:
        return ""

    ranked = sorted(population,
                    key=lambda c: _fitness(c, need_space),
                    reverse=True)
    best = ranked[0]
    if _fitness(best, need_space) == 0:
        return best
    return best

###############################################################################
# Chat‑level edit helper (with graceful error / rate‑limit handling)
###############################################################################
def apply_edit(code: str, instruction: str) -> tuple[str, str]:
    try:
        resp = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            system=(
                "You are an AI pair‑programmer. "
                "The user will send the current code and a request. "
                "Respond with the FULL REVISED file that satisfies the request. "
                "After the code, append a brief explanation prefixed with >>> "
                "(one short paragraph)."
            ),
            messages=[{
                "role": "user",
                "content": (
                    f"### Current code:\n```python\n{code}\n```\n\n"
                    f"### Request:\n{instruction}\n"
                )
            }],
            max_tokens=4096,
            temperature=0
        )
        raw = _join_content(resp.content)
        if "```" in raw:
            parts = raw.split("```")
            new_code = parts[1] if len(parts) > 1 else code
            explanation = "".join(parts[2:]).lstrip(">\n ")
        else:
            new_code = raw
            explanation = ""
        return new_code, explanation or "Done."
    except RateLimitError:
        return code, "⚠️  Claude is rate‑limited right now – please try again."
    except Exception as exc:
        app.logger.exception("Anthropic error during edit: %s", exc)
        return code, f"❌ Server error: {type(exc).__name__}"

###############################################################################
# Routes
###############################################################################
@app.route("/")
def index():
    return render_template("index.html")

@app.post("/autocomplete")
def autocomplete():
    snippet = request.json.get("snippet", "")
    suggestion = ga_best_completion(snippet)
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
