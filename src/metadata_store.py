"""SQLITE store: documents + chunk metadata (text,page,source filename). """
import logging 
import os
import sqlite3

logger = logging.getLogger(__name__)

_SCHEMA ="""
CREATE TABLE IF NOT EXISTS documents(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    file_hash TEXT NOT NULL UNIQUE,
    page_count INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS chunks(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    page INTEGER NOT NULL,
    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""

class MetadataStore:
    def __init__(self,db_path:str):
        self.db_path =db_path
        os.makedirs(os.path.dirname(db_path),exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory =sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def find_document_by_hash(self,file_hash:str):
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE file_hash = ?",(file_hash,)).fetchone()
            return dict(row) if row else None

    def add_document(self,filename:str,file_hash:str,page_count:int) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO documents (filename,file_hash,page_count) VALUES (?,?,?)",(filename,file_hash,page_count),)
            
            row_id = cur.lastrowid
            if row_id is None:
                raise RuntimeError("Failed to insert document")
            return row_id

    def add_chunks(self, rows: list[tuple]):
            
            """rows: (doc_id, chunk_index, text, page). Insertion order == vector order."""
            with self._connect() as conn:

                conn.executemany(
                "INSERT INTO chunks (doc_id, chunk_index, text, page) VALUES (?, ?, ?, ?)", rows
            )


    def get_chunk_ids_for_doc(self,doc_id:int) -> list[int]:
        with self._connect()  as conn:
            rows = conn.execute(
                "SELECT id FROM chunks WHERE doc_id = ? ORDER BY id",(doc_id,)
            ).fetchall()
            return [r["id"] for r in rows]

    def get_chunk_with_source(self,chunk_id:int):
        with self._connect() as conn:
            row = conn.execute(
                """SELECT c.id, c.text, c.page, d.filename, d.id AS doc_id
                FROM chunks c JOIN documents d on d.id = c.doc_id
                WHERE c.id =?""",
                (chunk_id,),
            ).fetchone()
            return dict(row) if row else None

    def stats(self) -> dict:
        with self._connect() as conn:
            docs = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
            chunks = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
            return {"documents":docs,"chunks":chunks}

    def clear_all(self):
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM documents") 

    def list_documents(self) -> list[dict]:
     
     
     """
        Returns all indexed documents with chunk counts.
     """
     with self._connect() as conn:
             rows = conn.execute(
            """
            SELECT
                d.id,
                d.filename,
                d.file_hash,
                d.page_count,
                COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON c.doc_id = d.id
            GROUP BY d.id, d.filename, d.file_hash, d.page_count
            ORDER BY d.created_at DESC, d.id DESC
            """
        ).fetchall()

     return [dict(row) for row in rows]

    def delete_document(self, doc_id: int) -> None:

        """
        Delete one document and all of its chunks.
        """
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM chunks WHERE doc_id = ?",
                (doc_id,),
            )

            conn.execute(
                "DELETE FROM documents WHERE id = ?",
                (doc_id,),
            ) 