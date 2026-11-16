# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

A learning project: building RAG (retrieval-augmented generation) over a code repo, step by step. Phase 1 (steps 1–3) is done; phase 2 is evaluation and improvements.

- `step1_chat.py`: a single chat completion call; prints the answer and token usage.
- `step2_embeddings.py`: embeds example snippets and ranks them by cosine similarity to a question.
- `step3_rag.py`: full RAG pipeline. It loads source files, splits them into 50-line chunks with 10-line overlap, embeds the chunks in batches of 100, caches the index, retrieves the top k=5 chunks by cosine similarity, and answers in an `Ask:` loop. Run it with `python step3_rag.py <repo_path>`.
- `eval/phase1_results.md`: benchmark questions, graded results, and failure analysis. Add new eval runs here.

No tests or `requirements.txt` yet.

## Environment

- Python 3.13, virtual environment at `.venv/`. Activate it with `source .venv/bin/activate` before running anything.
- Installed dependencies: `openai`, `numpy`, `python-dotenv`. Generate `requirements.txt` with `pip freeze > requirements.txt` when adding dependencies.
- Secrets are stored in `.env` (git-ignored) and loaded with `python-dotenv`. The `openai` client reads `OPENAI_API_KEY` from the environment.
- Models: chat `gpt-5.4-nano`, embeddings `text-embedding-3-small` (1536 dimensions).

## Caching

- `step3_rag.py` caches the index in `.rag_cache/<repo-folder-name>/` (`chunks.json` + `vectors.npy`; git-ignored).
- The cache does not detect file changes. Delete the repo's cache folder after editing the indexed repo.

## Repository

- Own git repo, pushed to GitHub at `JuustinJoseph/repomind` (branch `main`).
