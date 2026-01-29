# Gradio AI Agent (Ready-to-Run)

A minimal, ready-to-run **Gradio** web UI that wraps a simple **AI agent orchestration** with tool-calling:

- Calculator (safe arithmetic evaluation)
- Web search (powered by **Tavily API** or stub fallback)
- File RAG (keyword-based search in `data/` directory)

## Features

- **Dual Provider Modes:**
  - `PROVIDER=local` — Simple heuristic-based tool planner (no LLM required)
  - `PROVIDER=openai` — OpenAI function calling with GPT models
- **OpenAI Integration:** Uses OpenAI API with function calling for intelligent tool selection
- **Tavily Web Search:** Real-time web search with answer extraction (requires API key)
- **File RAG:** Keyword-based retrieval from local text/markdown files
- **Calculator Tool:** Safe expression evaluation for arithmetic queries

## Quickstart

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -U pip
pip install -r requirements.txt
# Or install in editable mode:
# pip install -e .

python src/agent_gradio.py
```

Open the URL shown by Gradio (e.g., http://127.0.0.1:7860).

## VS Code Debug

Use the provided **.vscode/launch.json** to run the app directly from VS Code.

## Configuration

Create a `.env.local` file (or `.env`) with the following variables:

```bash
# Provider mode: 'local' (heuristic) or 'openai' (GPT function calling)
PROVIDER=local

# OpenAI API Configuration (required when PROVIDER=openai)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini

# Tavily API Configuration (optional, enables real web search)
TAVILY_API_KEY=your_tavily_api_key_here
```

### Provider Modes

**Local Mode (`PROVIDER=local`)**

- Uses simple heuristic rules to select tools
- No API keys required
- Good for testing and offline use
- Web search returns stub results without Tavily API key

**OpenAI Mode (`PROVIDER=openai`)**

- Requires `OPENAI_API_KEY`
- Uses GPT models with function calling for intelligent tool selection
- Better at understanding complex queries
- Recommended models: `gpt-4o-mini`, `gpt-4o`, `gpt-4-turbo`

### API Keys

- **OpenAI API Key:** Get from [platform.openai.com](https://platform.openai.com/api-keys)
- **Tavily API Key:** Get from [tavily.com](https://tavily.com) (optional but recommended for web search)

## Usage Examples

**Calculator:**

- "What is 25 \* 48 + 100?"
- "Calculate 2^10"

**Web Search (with Tavily API key):**

- "Search for latest Python releases"
- "Find information about AI agents"

**File RAG:**

- Put `.txt` or `.md` files into the `data/` directory
- Ask: "What's in my files about [topic]?"

## Notes

- **File RAG:** Place `.txt` or `.md` files in `data/` directory for keyword-based retrieval
- **Web Search:** Set `TAVILY_API_KEY` for real web search; otherwise uses stub results
- **SSL Certificates:** The app uses `certifi` for SSL verification
- **Environment:** See [ENVIRONMENT.md](ENVIRONMENT.md) for detailed setup and troubleshooting
