# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

A learning project: building RAG (retrieval-augmented generation) over a code repo, step by step. Phase 1 (steps 1–3) is done. Phase 2 is evaluation and improvements: step 4 (Postgres + pgvector) is done, and step 5 (Tree-sitter chunking) is next.

- `step1_chat.py`: a single chat completion call; prints the answer and token usage.
- `step2_embeddings.py`: embeds example snippets and ranks them by cosine similarity to a question.
- `step3_rag.py`: full RAG pipeline with numpy. It loads source files, splits them into 50-line chunks with 10-line overlap, embeds the chunks in batches of 100, caches the index to disk, retrieves the top k=5 chunks by cosine similarity, and answers in an `Ask:` loop. Run it with `python step3_rag.py <repo_path>`. Kept unchanged as the numpy baseline.
- `step4_pg.py`: the same pipeline, but chunks and vectors are stored in Postgres and search is a pgvector SQL query (`<=>` cosine distance, score = `1 - distance`). Run it with `python step4_pg.py <repo_path> [--reindex]`. It also skips the `eval/` folder (see Evaluation).
- `schema.sql`: creates the `vector` extension and the `chunks` table (`repo, path, start_line, end_line, content, embedding VECTOR(1536)`). It is idempotent, and `step4_pg.py` runs it on every startup.
- `eval/phase1_results.md`: benchmark questions, graded results, and failure analysis. Add new eval runs here.

No tests yet.

## Environment

- Python 3.13, virtual environment at `.venv/`. Activate it with `source .venv/bin/activate` before running anything.
- Dependencies are listed in `requirements.txt`. Regenerate it with `pip freeze > requirements.txt` when adding dependencies.
- Secrets and config are stored in `.env` (git-ignored) and loaded with `python-dotenv`: `OPENAI_API_KEY`, and `DATABASE_URL` (`postgresql://postgres:<password>@localhost:5432/repomind`).
- Models: chat `gpt-5.4-nano`, embeddings `text-embedding-3-small` (1536 dimensions, which must match `VECTOR(1536)`).

## Database

- Postgres runs in Docker as container `repomind-db` (image `pgvector/pgvector:pg17`, port 5432, named volume `repomind_pgdata`). The database is named `repomind`.
- Start it with `docker start repomind-db`. Open a shell with `docker exec -it repomind-db psql -U postgres -d repomind`.
- After connecting, run `schema.sql` before `register_vector(conn)`, because it looks up the `vector` type.
- Rows are keyed by `repo` = the indexed folder's name. If a repo already has rows, `step4_pg.py` reuses them. The index does not detect file changes, so pass `--reindex` (which deletes and rebuilds that repo's rows) after editing the indexed repo.
- There is no HNSW index yet. Search is exact, which is fine at this size.

## Evaluation

- `step4_pg.py` skips `eval/` because it contains the expected answers. Indexing it leaked answers into retrieval. `step3_rag.py` still indexes it.
- When indexing repomind itself, `step4_pg.py` duplicates most of `step3_rag.py`, so near-duplicate chunks fill the top 5. Keep this in mind when comparing eval runs.

## Caching (step 3 only)

- `step3_rag.py` caches the index in `.rag_cache/<repo-folder-name>/` (`chunks.json` + `vectors.npy`; git-ignored). Delete the folder after editing the indexed repo.

## Repository

- Own git repo, pushed to GitHub at `JuustinJoseph/repomind` (branch `main`).
