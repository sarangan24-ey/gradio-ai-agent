import os
import re
import json
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv

import gradio as gr

# ----------------------------
# Config
# ----------------------------
load_dotenv()
DATA_DIR = os.path.join(os.getcwd(), "data")
PROVIDER = os.getenv("PROVIDER", "local").lower()

SYSTEM_PROMPT = (
    "You are a helpful AI agent. Prefer calling tools when they improve accuracy. "
    "Clearly present final answers. If tools are used, integrate their results."
)

# ----------------------------
# Tools
# ----------------------------


def calculator(expr: str) -> Dict[str, Any]:
    """Evaluate a simple math expression safely."""
    allowed = set("0123456789+-*/(). ")
    if not set(expr) <= allowed:
        return {"error": "Unsupported characters in expression."}
    try:
        value = eval(expr, {"__builtins__": {}}, {})
        return {"result": float(value)}
    except Exception as e:
        return {"error": f"Invalid expression: {e}"}


def web_search(query: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Stubbed web search. In production, integrate Bing or enterprise search.
    """
    results = [
        f"https://example.com/search?q={query}&rank={i}" for i in range(1, top_k + 1)
    ]
    return {"results": results}


def _load_text_files() -> List[Tuple[str, str]]:
    texts = []
    if not os.path.isdir(DATA_DIR):
        return texts
    for fname in os.listdir(DATA_DIR):
        if fname.lower().endswith((".txt", ".md")):
            path = os.path.join(DATA_DIR, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    texts.append((fname, f.read()))
            except Exception:
                pass
    return texts


def file_rag(question: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Keyword-based file lookup. For robust RAG, replace with embeddings/vector DB.
    """
    words = [w.lower() for w in re.findall(r"\w+", question) if len(w) > 2]
    files = _load_text_files()
    if not files:
        return {"results": [], "note": "No files found in ./data"}
    scored = []
    for name, text in files:
        t = text.lower()
        score = sum(t.count(w) for w in words)
        if score > 0:
            idx = max(0, min(len(text) - 1, t.find(words[0]) if words else 0))
            snippet = text[idx : idx + 400].replace("", " ")
            scored.append((score, name, snippet))
    scored.sort(reverse=True)
    top = [{"file": n, "score": s, "snippet": sn} for s, n, sn in scored[:top_k]]
    return {"results": top}


TOOL_REGISTRY = {
    "calculator": {"fn": calculator, "signature": {"expr": str}},
    "web_search": {"fn": web_search, "signature": {"query": str, "top_k": int}},
    "file_rag": {"fn": file_rag, "signature": {"question": str, "top_k": int}},
}

# ----------------------------
# Simple local planner
# ----------------------------


def plan_tool_calls(user_text: str) -> List[Dict[str, Any]]:
    """
    Very simple heuristics for demo:
      - calculator if arithmetic appears
      - web_search if 'search/find/google' or ends with '?'
      - file_rag if mentions 'file/doc/data'
    """
    calls = []
    # calculator
    if re.search(r"[0-9][0-9\+\-\*\/\.\(\) ]+", user_text):
        exprs = re.findall(r"[0-9\+\-\*\/\.\(\) ]+", user_text)
        if exprs:
            calls.append(
                {"name": "calculator", "arguments": {"expr": exprs[-1].strip()}}
            )
    # web_search
    if re.search(r"(search|find|google)|\?$", user_text, re.IGNORECASE):
        q = user_text.strip()
        calls.append({"name": "web_search", "arguments": {"query": q, "top_k": 3}})
    # file_rag
    if re.search(r"(file|doc|data)", user_text, re.IGNORECASE):
        calls.append(
            {"name": "file_rag", "arguments": {"question": user_text, "top_k": 3}}
        )
    return calls


# ----------------------------
# Agent loop
# ----------------------------


def run_agent(
    user_text: str, history: List[Dict[str, str]] | None = None
) -> Dict[str, Any]:
    """
    Returns a dict suitable for Gradio:
    {
      "reply": str,
      "tool_calls": [{name, arguments, output}],
      "provider": "local"
    }
    """
    history = history or []
    executed = []
    for call in plan_tool_calls(user_text):
        name = call["name"]
        args = call["arguments"]
        fn = TOOL_REGISTRY.get(name, {}).get("fn")
        if fn is None:
            out = {"error": f"Unknown tool: {name}"}
        else:
            try:
                out = fn(**args)
            except TypeError as te:
                out = {"error": f"Bad arguments for {name}: {te}"}
            except Exception as e:
                out = {"error": f"{name} failed: {e}"}
        executed.append({"name": name, "arguments": args, "output": out})

    # Compose a reply for the chat
    reply_lines = [f"**You asked:** {user_text}"]
    if executed:
        reply_lines.append("I used the following tools:")
        for e in executed:
            reply_lines.append(f"- {e['name']}: {json.dumps(e['output'])}")
    else:
        reply_lines.append(
            "No tools were needed. Try arithmetic, 'search …', or mention 'file/data'."
        )
    return {"reply": "".join(reply_lines), "tool_calls": executed, "provider": "local"}


# ----------------------------
# Gradio UI
# ----------------------------


def gradio_chat_fn(message: str, chat_history: List[Dict[str, str]]):
    """
    Gradio ChatInterface handler:
      - message: latest user input
      - chat_history: list of {'role': 'user'|'assistant', 'content': str}
    Returns: assistant text (string).
    """
    history = [{"role": h["role"], "content": h["content"]} for h in chat_history]
    result = run_agent(message, history)
    tool_log = (
        "**Tool calls:**\n```json\n"
        + json.dumps(result["tool_calls"], indent=2)
        + "\n```"
    )
    return result["reply"] + tool_log


demo = gr.ChatInterface(
    fn=gradio_chat_fn,
    title="AI Agent (Gradio Demo)",
    description="Ask questions that may require tools: arithmetic (calculator), 'search ...' (web), or 'file/data' (local file snippets).",
)

if __name__ == "__main__":
    demo.launch()
