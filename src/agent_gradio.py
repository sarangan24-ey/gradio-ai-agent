import os
import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()
import re
import json
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv

import gradio as gr
from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam
from tavily import TavilyClient

# ----------------------------
# Config
# ----------------------------
# load_dotenv()
load_dotenv(".env.local", override=True)
DATA_DIR = os.path.join(os.getcwd(), "data")
PROVIDER = os.getenv("PROVIDER", "local").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if PROVIDER == "openai" and not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY environment variable is required when PROVIDER=openai"
    )

client = OpenAI(api_key=OPENAI_API_KEY) if PROVIDER == "openai" else None
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

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
    Web search using Tavily API. Falls back to stub if API key not provided.
    """
    if not tavily_client:
        # Fallback to stub results
        results = [
            f"https://example.com/search?q={query}&rank={i}"
            for i in range(1, top_k + 1)
        ]
        return {"results": results, "source": "stub"}

    try:
        response = tavily_client.search(
            query=query, max_results=top_k, include_answer=True
        )
        results = []
        for result in response.get("results", []):
            results.append(
                {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("snippet", ""),
                    "source": "tavily",
                }
            )
        return {
            "results": results,
            "answer": response.get("answer", ""),
            "source": "tavily",
        }
    except Exception as e:
        return {"error": f"Tavily search failed: {str(e)}", "source": "tavily"}


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


def get_openai_tools() -> list[ChatCompletionToolParam]:
    """Convert tool registry to OpenAI function calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate a simple math expression safely.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expr": {
                            "type": "string",
                            "description": "Math expression to evaluate",
                        }
                    },
                    "required": ["expr"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return",
                            "default": 3,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "file_rag",
                "description": "Search local files for relevant information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Question to search files for",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return",
                            "default": 3,
                        },
                    },
                    "required": ["question"],
                },
            },
        },
    ]


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


def execute_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool by name with given arguments."""
    fn = TOOL_REGISTRY.get(name, {}).get("fn")
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        return fn(**arguments)
    except TypeError as te:
        return {"error": f"Bad arguments for {name}: {te}"}
    except Exception as e:
        return {"error": f"{name} failed: {e}"}


def run_agent_openai(
    user_text: str, history: List[Dict[str, str]] | None = None
) -> Dict[str, Any]:
    """Run agent using OpenAI API with function calling."""
    history = history or []
    messages = [{"role": "user", "content": user_text}]
    executed = []

    try:
        # Call OpenAI with tools
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=get_openai_tools(),
            tool_choice="auto",
        )

        # Process tool calls if any
        while response.choices[0].message.tool_calls:
            tool_calls = response.choices[0].message.tool_calls

            for tool_call in tool_calls:
                name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                output = execute_tool(name, arguments)
                executed.append(
                    {"name": name, "arguments": arguments, "output": output}
                )

            # Add assistant response and tool results to messages for next iteration
            messages.append(response.choices[0].message)
            for tool_call in tool_calls:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(
                            execute_tool(
                                tool_call.function.name,
                                json.loads(tool_call.function.arguments),
                            )
                        ),
                    }
                )

            # Get next response
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                tools=get_openai_tools(),
                tool_choice="auto",
            )

        # Extract final reply
        reply = response.choices[0].message.content or "No response generated."
        return {"reply": reply, "tool_calls": executed, "provider": "openai"}

    except Exception as e:
        print(e)
        return {
            "reply": f"Error with OpenAI API: {str(e)}",
            "tool_calls": executed,
            "provider": "openai",
        }


def run_agent_local(
    user_text: str, history: List[Dict[str, str]] | None = None
) -> Dict[str, Any]:
    """Run agent using local heuristics (original implementation)."""
    history = history or []
    executed = []
    for call in plan_tool_calls(user_text):
        name = call["name"]
        args = call["arguments"]
        output = execute_tool(name, args)
        executed.append({"name": name, "arguments": args, "output": output})

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


def run_agent(
    user_text: str, history: List[Dict[str, str]] | None = None
) -> Dict[str, Any]:
    """Route to appropriate agent implementation based on PROVIDER."""
    if PROVIDER == "openai":
        return run_agent_openai(user_text, history)
    else:
        return run_agent_local(user_text, history)


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
