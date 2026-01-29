# Gradio AI Agent (Ready-to-Run)

A minimal, ready-to-run **Gradio** web UI that wraps a simple **AI agent orchestration** with tool-calling:
- Calculator (safe arithmetic)
- Web search (stubbed)
- File RAG (reads snippets from `data/`)

## Quickstart

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -U pip
pip install -e .

python src/agent_gradio.py
```
Open the URL shown by Gradio (e.g., http://127.0.0.1:7860).

## VS Code Debug
Use the provided **.vscode/launch.json** to run the app directly from VS Code.

## Configuration
Edit `.env`:
```
PROVIDER=local
```

## Notes
- Put `.txt` files into `data/` to test the File RAG tool.
- Replace the stubbed web search with a real API for production.
