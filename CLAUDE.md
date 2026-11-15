# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

This is a new project with no source code yet. The Python environment is set up; the application itself has not been written. Update this file once source files, a dependency manifest, and tests exist.

## Environment

- Python 3.13, virtual environment at `.venv/`. Activate it with `source .venv/bin/activate` before running anything.
- Installed dependencies: `openai`, `numpy`, `python-dotenv`. There is no `requirements.txt` yet; generate one with `pip freeze > requirements.txt` when adding dependencies.
- Secrets are stored in `.env` (git-ignored) and loaded with `python-dotenv`. The `openai` client reads `OPENAI_API_KEY` from the environment.

## Repository

- This folder sits inside a parent git repo at `/Users/justin/Code`. It has not yet been made its own repo or pushed to GitHub.
