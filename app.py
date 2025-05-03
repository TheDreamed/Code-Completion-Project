import os
import logging
from flask import Flask, request, jsonify, render_template, make_response
import anthropic
from dotenv import load_dotenv

###############################################################################
# Configuration
###############################################################################

load_dotenv()  # reads .env in current working directory

API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    raise RuntimeError("Set ANTHROPIC_API_KEY in .env or your shell.")

client = anthropic.Anthropic(api_key=API_KEY)

app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

###############################################################################
# Anthropic helpers
###############################################################################

def _block_to_text(block) -> str:
    if isinstance(block, dict) and block.get("type") == "text":
        return block.get("text", "")
    return getattr(block, "text", "") or ""

def _join_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_block_to_text(b) for b in content)
    return ""

def next_tokens(code: str, *, max_tokens: int = 12) -> str:
    """
    Ask Claude for the next few tokens — no explanations, no duplicates.
    """
    try:
        resp = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            system=(
                "You are an advanced Python coding assistant. "
                "Return ONLY the next code tokens that naturally follow the snippet. "
                "Do not explain."
            ),
            messages=[{"role": "user", "content": code}],
            max_tokens=max_tokens,
            temperature=0.1,
            stop_sequences=[],
        )
        return _join_content(resp.content).strip()
    except Exception as exc:
        app.logger.exception("Anthropic error: %s", exc)
        return ""

###############################################################################
# Routes
###############################################################################

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/autocomplete", methods=["POST"])
def autocomplete():
    snippet = request.json.get("snippet", "")
    suggestion = next_tokens(snippet)
    resp = make_response(jsonify({"suggestion": suggestion}))
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp

###############################################################################
# Entrypoint
###############################################################################

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
