CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id         BIGSERIAL PRIMARY KEY,
    repo       TEXT NOT NULL,
    path       TEXT NOT NULL,
    start_line INT  NOT NULL,
    end_line   INT  NOT NULL,
    content    TEXT NOT NULL,
    embedding  VECTOR(1536) NOT NULL
);
