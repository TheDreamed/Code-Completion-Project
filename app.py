import os, logging
from flask import Flask, request, jsonify, render_template, make_response
import anthropic
from dotenv import load_dotenv

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

def next_tokens(code:str, *, max_tokens:int=12) -> str:
    """Return only the next few code tokens."""
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
    """
    Ask Claude to rewrite the file according to the user's instruction.
    Return (new_code, assistant_reply).
    """
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
                "role":"user",
                "content":(
                    f"### Current code:\n```python\n{code}\n```\n\n"
                    f"### Request:\n{instruction}\n"
                )
            }],
            max_tokens=4096,
            temperature=0
        )
        raw=_join_content(resp.content)
        if "```" in raw:
            parts=raw.split("```")
            new_code=parts[1] if len(parts)>1 else code
            explanation="".join(parts[2:]).lstrip(">\n ")
        else:
            new_code=raw
            explanation=""
        return new_code, explanation or "Done."
    except Exception as exc:
        app.logger.exception("Anthropic error: %s", exc)
        return code, "I hit an error while processing that request."

###############################################################################
# Routes
###############################################################################
@app.route("/")
def index():
    return render_template("index.html")

@app.post("/autocomplete")
def autocomplete():
    snippet = request.json.get("snippet", "")
    suggestion = next_tokens(snippet)
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
