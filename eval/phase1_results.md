# Phase 1 evaluation: step3_rag.py on repomind

Date: 2026-11-15
Repo indexed: `repomind` (4 files, 8 chunks)
Setup: 50-line chunks, 10-line overlap, k=5, `text-embedding-3-small`, `gpt-5.4-nano`

Note: the cache was built before later edits to step2/step3, so some chunk contents were stale during this run.

## Questions and expected answers

| # | Question | Expected answer | Expected chunk |
|---|---|---|---|
| 1 | What chat model does this project use? | `gpt-5.4-nano` | `step1_chat.py:1-27` / `step3_rag.py:1-50` |
| 2 | How big are the chunks and how much do they overlap? | 50 lines, 10 overlap | `step3_rag.py:1-50` |
| 3 | Where are embeddings cached and in what format? | `.rag_cache/<repo>/`, `chunks.json` + `vectors.npy` | `step3_rag.py:81-130` |
| 4 | Which folders are skipped when loading files? | `.git`, `node_modules`, `.venv`, `venv`, `bin`, `obj`, `dist`, `build`, `__pycache__` | `step3_rag.py:1-50` |
| 5 | How does the project handle user login? (trap) | No real login; only example snippets in step 2 | `step2_embeddings.py:1-50` |

## Results

| # | Answer correct | Right chunk retrieved (rank, score) | Citation correct | Failure type |
|---|---|---|---|---|
| 1 | Yes | Yes (#1, 0.523) | Yes | None |
| 2 | Yes | Weak (#5, 0.212) | Yes | Weak retrieval |
| 3 | Yes | Weak (#4, 0.334); step2 chunks ranked #1-2 | No (invented `41-68`, `69-90`) | Weak retrieval + generation |
| 4 | Yes | Yes (#1, 0.342) | No (invented `18-39`) | Generation |
| 5 | Yes (said no real login) | Yes | No (invented `1-44`) | Generation |

Score: 5/5 answers correct, 2/5 citations correct, 3/5 with the right chunk ranked first.

## Findings

1. **Invented citation ranges (generation).** The model cites narrower line ranges than the chunk labels because it can't see real line numbers. Seen 4 times, including an earlier run (`step3_rag.py:99-104`).
   - Ideas to test: prefix each line with its line number, or tell the prompt to cite only the given chunk labels.
2. **Keyword-heavy chunks beat implementation chunks (retrieval).** For "where are embeddings cached", step2 chunks full of the word "embeddings" outranked `build_index`. With k=3, Q2 and Q3 would have lost the right chunk.
   - Ideas to test: function-boundary chunking, smaller chunks, keyword plus vector hybrid search.
3. **Trap question passed.** The "use ONLY the code provided" instruction prevented an invented answer.

## Next

- Rebuild the cache (`rm -rf .rag_cache/repomind`) and re-run.
- Repeat on a real project (under 50 files) with 5 questions, including a trap question.
