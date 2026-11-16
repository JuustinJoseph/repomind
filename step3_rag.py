import json
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

CHAT_MODEL = "gpt-5.4-nano"  # same as step 1
EMBED_MODEL = "text-embedding-3-small"

SOURCE_EXTS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".cs",
    ".md",
    ".java",
    ".go",
    ".rb",
    ".html",
    ".css",
}
SKIP_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "bin",
    "obj",
    "dist",
    "build",
    "__pycache__",
}

CHUNK_LINES = 50
OVERLAP = 10
BATCH_SIZE = 100
CACHE_DIR = Path(".rag_cache")

load_dotenv()
client = OpenAI()


# 3a. Load files
def load_files(repo_path):
    repo = Path(repo_path)
    files = []
    for path in sorted(repo.rglob("*")):
        rel = path.relative_to(repo)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if path.is_file() and path.suffix in SOURCE_EXTS:
            try:
                files.append((str(rel), path.read_text(encoding="utf-8")))
            except UnicodeDecodeError:
                pass  # skip binary or oddly-encoded files
    return files


# 3b. Chunk
def chunk_file(path, text):
    lines = text.splitlines()
    chunks = []
    step = CHUNK_LINES - OVERLAP
    for start in range(0, max(len(lines), 1), step):
        end = min(start + CHUNK_LINES, len(lines))
        body = "\n".join(lines[start:end])
        chunks.append(
            {
                "path": path,
                "start_line": start + 1,
                "end_line": end,
                "text": f"# File: {path}\n{body}",
            }
        )
        if end >= len(lines):
            break
    return chunks


# 3c. Embed and cache
def embed(texts):
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        result = client.embeddings.create(
            model=EMBED_MODEL, input=texts[i : i + BATCH_SIZE]
        )
        vectors.extend(item.embedding for item in result.data)
    return np.array(vectors)


def build_index(repo_path):
    cache = CACHE_DIR / Path(repo_path).resolve().name
    chunks_file, vectors_file = cache / "chunks.json", cache / "vectors.npy"
    if chunks_file.exists() and vectors_file.exists():
        print(f"Loading cached index from {cache}")
        return json.loads(chunks_file.read_text()), np.load(vectors_file)

    files = load_files(repo_path)
    print(f"Found {len(files)} files:")
    for path, _ in files:
        print(f"  {path}")

    chunks = [c for path, text in files for c in chunk_file(path, text)]
    print(f"Created {len(chunks)} chunks. Embedding...")
    vectors = embed([c["text"] for c in chunks])

    cache.mkdir(parents=True, exist_ok=True)
    chunks_file.write_text(json.dumps(chunks))
    np.save(vectors_file, vectors)
    return chunks, vectors


# 3d. Retrieve
def search(question, chunks, vectors, k=5):
    q = embed([question])[0]
    # One matrix op: (N, d) @ (d,) -> (N,), divided by each row's norm times the query norm.
    scores = vectors @ q / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q))
    top = np.argsort(scores)[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in top]


# 3e. Generate
SYSTEM_PROMPT = (
    "You answer questions about a codebase using ONLY the code provided. "
    "Cite sources as path:start-end. "
    "If the code provided doesn't contain the answer, say so."
)


def answer(question, results):
    context = "\n\n".join(
        f"--- {c['path']}:{c['start_line']}-{c['end_line']} ---\n{c['text']}"
        for c, _ in results
    )
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{context}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content, response.usage


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python3 step3_rag.py <repo_path>")
    chunks, vectors = build_index(sys.argv[1])

    while True:
        question = input("\nAsk: ").strip()
        if question in {"", "quit", "exit"}:
            break
        results = search(question, chunks, vectors)
        # Always look at retrieval before trusting the answer.
        print("\nRetrieved:")
        for c, score in results:
            print(f"  {score:.3f}  {c['path']}:{c['start_line']}-{c['end_line']}")
        text, usage = answer(question, results)
        print(f"\n{text}\n\n({usage.total_tokens} tokens)")


if __name__ == "__main__":
    main()
