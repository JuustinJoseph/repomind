import os
import sys
from pathlib import Path

import numpy as np
import psycopg
from dotenv import load_dotenv
from openai import OpenAI
from pgvector.psycopg import register_vector

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
    "eval",  # holds benchmark questions + expected answers; indexing it leaks them
}

CHUNK_LINES = 50
OVERLAP = 10
BATCH_SIZE = 100
SCHEMA_FILE = Path(__file__).parent / "schema.sql"  # next to this script, not cwd

load_dotenv()
client = OpenAI()


# 4c. Connect to Postgres.
# Order matters: the schema creates the `vector` extension, and register_vector
# looks that type up in the database, so the schema must run first.
def connect():
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    conn.execute(SCHEMA_FILE.read_text())  # idempotent: IF NOT EXISTS everywhere
    conn.commit()
    register_vector(conn)  # numpy arrays <-> VECTOR columns
    return conn


# 3a. Load files (unchanged from step 3)
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


# 3b. Chunk (unchanged from step 3)
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


# 3c. Embed (unchanged from step 3)
def embed(texts):
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        result = client.embeddings.create(
            model=EMBED_MODEL, input=texts[i : i + BATCH_SIZE]
        )
        vectors.extend(item.embedding for item in result.data)
    return np.array(vectors)


# 4d. Index into Postgres instead of the .rag_cache files.
# Returns the repo name, which is the key used to filter rows in search().
def build_index(conn, repo_path, reindex=False):
    # Same name step 3 used for its cache folder, so both versions line up.
    repo = Path(repo_path).resolve().name

    # Replaces "does the cache folder exist?": does this repo already have rows?
    count = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE repo = %s", (repo,)
    ).fetchone()[0]
    if count and not reindex:
        print(f"Using existing index for '{repo}' ({count} chunks)")
        return repo
    if count:
        # Re-index: wipe this repo's rows (and only this repo's) and start fresh.
        conn.execute("DELETE FROM chunks WHERE repo = %s", (repo,))

    files = load_files(repo_path)
    print(f"Found {len(files)} files:")
    for path, _ in files:
        print(f"  {path}")

    chunks = [c for path, text in files for c in chunk_file(path, text)]
    print(f"Created {len(chunks)} chunks. Embedding...")
    vectors = embed([c["text"] for c in chunks])

    # One tuple per row, in the same order as the INSERT's column list.
    # content stores c["text"] (with the "# File:" header): the exact string embedded.
    rows = [
        (repo, c["path"], c["start_line"], c["end_line"], c["text"], v)
        for c, v in zip(chunks, vectors)
    ]
    # %s placeholders: psycopg sends values separately from the SQL, so nothing
    # in a file's content can break or inject into the query.
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (repo, path, start_line, end_line, content, embedding) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            rows,
        )
    conn.commit()  # without this, the DELETE and INSERTs are rolled back on exit
    print(f"Inserted {len(rows)} chunks for '{repo}'")
    return repo


# 4e. Retrieve with SQL instead of numpy.
# <=> is pgvector's cosine DISTANCE (0 = identical), so 1 - distance gives the
# same cosine similarity score as step 3. ORDER BY distance ascending = most similar first.
def search(conn, repo, question, k=5):
    q = embed([question])[0]  # numpy array: register_vector sends it as a vector
    rows = conn.execute(
        """
        SELECT path, start_line, end_line, content,
               1 - (embedding <=> %s) AS score
        FROM chunks
        WHERE repo = %s
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (q, repo, q, k),
    ).fetchall()
    # Rebuild step 3's (chunk_dict, score) shape so answer() and main() don't change.
    return [
        (
            {"path": path, "start_line": start, "end_line": end, "text": content},
            float(score),
        )
        for path, start, end, content, score in rows
    ]


# 3e. Generate (unchanged from step 3)
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
    args = sys.argv[1:]
    reindex = "--reindex" in args
    args = [a for a in args if a != "--reindex"]
    if len(args) != 1:
        sys.exit("Usage: python3 step4_pg.py <repo_path> [--reindex]")

    with connect() as conn:  # closes the connection when the loop ends
        repo = build_index(conn, args[0], reindex=reindex)

        while True:
            question = input("\nAsk: ").strip()
            if question in {"", "quit", "exit"}:
                break
            results = search(conn, repo, question)
            # Always look at retrieval before trusting the answer.
            print("\nRetrieved:")
            for c, score in results:
                print(f"  {score:.3f}  {c['path']}:{c['start_line']}-{c['end_line']}")
            text, usage = answer(question, results)
            print(f"\n{text}\n\n({usage.total_tokens} tokens)")


if __name__ == "__main__":
    main()
