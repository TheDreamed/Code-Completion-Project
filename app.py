import anthropic
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Replace with your real Anthropic API key
client = anthropic.Anthropic(api_key="")

def block_to_text(block) -> str:
    """
    Gracefully handle either a dict-based block or an object with .text.
    """
    # Case 1: Possibly a dict (older style)
    if isinstance(block, dict):
        # If it’s a text-type block, return block["text"]
        if block.get("type") == "text":
            return block.get("text", "")
        return ""

    # Case 2: Possibly an object with .text
    text_attr = getattr(block, "text", "")
    if not text_attr:
        return ""
    return text_attr

def join_anthropic_content(content) -> str:
    """
    content may be a string or a list of block objects/dicts.
    Collect any text we find.
    """
    if isinstance(content, str):
        return content  # just a plain string

    if isinstance(content, list):
        parts = []
        for block in content:
            parts.append(block_to_text(block))
        return "".join(parts)

    return ""  # fallback if unexpected

def generate_completion_anthropic(snippet: str, max_tokens: int = 10) -> str:
    """
    Ask Anthropic for only the minimal next snippet of code.
    The refined prompt ensures no explanation is provided.
    """
    model_name = "claude-3-7-sonnet-20250219"  # or "claude-2", "claude-instant-1", etc.
    try:
        response = client.messages.create(
            model=model_name,
            # The refined system instructions:
            system=(
                "You are an advanced Python coding assistant. The user is writing code and you need to predict what line of code or characters of code he will write next based on the code's context. "
                "You should ONLY respond with the next few tokens of code that naturally "
                "continue from the snippet. Absolutely no explanations, comments, or repeated code. "
                "Keep it minimal and just provide the next chunk of code to continue the snippet."
            ),
            messages=[{"role": "user", "content": snippet}],
            max_tokens=max_tokens,  # e.g., 10 tokens to keep the snippet short
            temperature=0.1,
            stop_sequences=[]
        )
        short_snippet = join_anthropic_content(response.content)
        return short_snippet.strip()
    except Exception as e:
        print("Error generating completion:", str(e))
        return ""

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/autocomplete", methods=["POST"])
def autocomplete():
    data = request.get_json()
    snippet = data.get("snippet", "")
    suggestion = generate_completion_anthropic(snippet, max_tokens=10)
    return jsonify({"suggestion": suggestion})

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
