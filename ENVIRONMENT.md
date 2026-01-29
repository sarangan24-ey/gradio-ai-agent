# Environment & Troubleshooting

This file documents the local environment setup, recently taken steps, and common fixes for running the Gradio AI Agent.

## Virtual environment (Windows PowerShell)

Create and activate the project's virtual environment before installing packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## Install dependencies

Install the core runtime dependencies used while developing and debugging this repo:

```powershell
pip install -U pip
pip install gradio pytz tavily
# or install the project in editable mode if the repo is packaged
pip install -e .
```

## Generate `requirements.txt` (optional)

```powershell
pip freeze > requirements.txt
```

## Run the app

```powershell
python src/agent_gradio.py
```

## VS Code debugging tips

- Ensure the Python interpreter is set to the project's `.venv` (choose the interpreter from the status bar or configure `launch.json`).
- If the debugger shows `ModuleNotFoundError`, confirm the venv is activated and the package is installed into that venv.

## Common issues & fixes

- `ModuleNotFoundError: No module named 'gradio'` — install `gradio` into the active virtualenv or switch the debugger to use `.venv`.
- `ModuleNotFoundError: No module named 'pytz'` — install `pytz`.
- `ModuleNotFoundError: No module named 'tavilly'` — the correct PyPI package name is `tavily` (single "l"); run `pip install tavily`.

## Recent actions taken in this workspace

- Installed `gradio`, `pytz`, and `tavily` into the active `.venv` to resolve import errors encountered while debugging.

## Next steps I can take for you

- Create or update `requirements.txt` with current pinned versions.
- Patch `README.md` to link to this file or embed these instructions.
- Add a `launch.json` snippet that pins the interpreter path for VS Code.
