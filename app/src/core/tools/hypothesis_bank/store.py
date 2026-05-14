"""SQLite metadata store + FAISS vector index for hypothesis retrieval."""

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import faiss
import numpy as np

from app.src.llm_gateway.providers.factory import get_embedding_provider
from app.src.utils.logger import get_logger
from app.src.utils.settings import get_settings


logger = get_logger(__name__)

_LOCK = threading.Lock()

# Matches nomic-embed-text output dimension.
EMBED_DIM = 768

# Embedding model name as configured in litellm/config.yml.
_EMBED_MODEL = "ollama/nomic-embed-text"

_CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS hypotheses (
        id               TEXT    PRIMARY KEY,
        disease          TEXT    NOT NULL,
        phase_origin     TEXT    NOT NULL,
        text             TEXT    NOT NULL,
        keywords         TEXT    NOT NULL DEFAULT '',
        efficacy_score   REAL    NOT NULL DEFAULT 0.0,
        safety_score     REAL    NOT NULL DEFAULT 0.0,
        notes            TEXT    NOT NULL DEFAULT '',
        created_at       TEXT    NOT NULL,
        updated_at       TEXT    NOT NULL,
        retrieval_count  INTEGER NOT NULL DEFAULT 0
    )
"""


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _build_openai_embed_fn() -> Callable[[list[str]], np.ndarray]:
    """Return an embed function backed by the litellm OpenAI-compatible proxy."""
    client = get_embedding_provider(_EMBED_MODEL)

    def embed(texts: list[str]) -> np.ndarray:
        vecs = client.embed_documents(texts)
        arr = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-9)

    return embed


# ---------------------------------------------------------------------------
# HypothesisStore
# ---------------------------------------------------------------------------

class HypothesisStore:
    """
    Hypothesis store with:
      - SQLite at metadata_path for all fields except the embedding vector.
      - FAISS IndexIDMap(IndexFlatIP) at index_path for vector similarity search.

    Vectors are L2-normalized before insertion so inner product == cosine similarity.
    All public methods are thread-safe via a module-level lock.
    """

    def __init__(
        self,
        metadata_path: Optional[Path] = None,
        index_path: Optional[Path] = None,
        embed_fn: Optional[Callable[[list[str]], np.ndarray]] = None,
    ) -> None:
        self._metadata_path = metadata_path
        self._index_path = index_path
        # Defer building the default embed_fn until first use so Settings()
        # is not called at import time when running tests without .env.
        self._embed_fn: Optional[Callable] = embed_fn
        self._embed_fn_initialized = embed_fn is not None

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _sql_path(self) -> Path:
        if self._metadata_path is not None:
            self._metadata_path.parent.mkdir(parents=True, exist_ok=True)
            return self._metadata_path
        settings = get_settings()
        db_dir = Path(settings.cache_prefix) / "hypothesis_bank"
        db_dir.mkdir(parents=True, exist_ok=True)
        return db_dir / "metadata.db"

    def _faiss_path(self) -> Path:
        if self._index_path is not None:
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            return self._index_path
        settings = get_settings()
        db_dir = Path(settings.cache_prefix) / "hypothesis_bank"
        db_dir.mkdir(parents=True, exist_ok=True)
        return db_dir / "embeddings.db"

    # ------------------------------------------------------------------
    # SQLite helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._sql_path()), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute(_CREATE_TABLE)
        conn.commit()
        return conn

    # ------------------------------------------------------------------
    # FAISS helpers
    # ------------------------------------------------------------------

    def _load_index(self) -> faiss.IndexIDMap:
        path = self._faiss_path()
        if path.exists():
            idx = faiss.read_index(str(path))
            if idx.d != EMBED_DIM:
                logger.warning(
                    "FAISS index dimension %d != expected %d; reinitializing.",
                    idx.d, EMBED_DIM,
                )
                return self._new_index()
            return idx
        return self._new_index()

    @staticmethod
    def _new_index() -> faiss.IndexIDMap:
        return faiss.IndexIDMap(faiss.IndexFlatIP(EMBED_DIM))

    def _save_index(self, index: faiss.IndexIDMap) -> None:
        faiss.write_index(index, str(self._faiss_path()))

    # ------------------------------------------------------------------
    # Embedding helper (lazy init of default embed_fn)
    # ------------------------------------------------------------------

    def _embed(self, texts: list[str]) -> np.ndarray:
        if not self._embed_fn_initialized:
            self._embed_fn = _build_openai_embed_fn()
            self._embed_fn_initialized = True
        return self._embed_fn(texts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        hypothesis_id: str,
        disease: str,
        phase_origin: str,
        text: str,
        keywords: str,
        efficacy_score: float,
        safety_score: float,
        notes: str,
    ) -> None:
        vec = self._embed([f"{text} {keywords}"])  # shape (1, EMBED_DIM)
        now = datetime.now(timezone.utc).isoformat()

        with _LOCK:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """INSERT INTO hypotheses
                           (id, disease, phase_origin, text, keywords,
                            efficacy_score, safety_score, notes,
                            created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        hypothesis_id, disease, phase_origin, text, keywords,
                        efficacy_score, safety_score, notes,
                        now, now,
                    ),
                )
                conn.commit()
                # Use the SQLite rowid as the FAISS vector ID — stable, unique,
                # and works with any hypothesis_id string format.
                row_id = cur.lastrowid
            finally:
                conn.close()

            index = self._load_index()
            index.add_with_ids(vec, np.array([row_id], dtype=np.int64))
            self._save_index(index)

    def search(
        self,
        query: str,
        disease_filter: Optional[str],
        top_k: int,
    ) -> list[dict]:
        query_vec = self._embed([query])  # shape (1, EMBED_DIM)

        with _LOCK:
            index = self._load_index()
            total = index.ntotal
            if total == 0:
                return []

            # Fetch more candidates when filtering so we still get top_k after pruning.
            k = min(total, top_k * 10 if disease_filter else top_k)
            scores, faiss_ids = index.search(query_vec, k)

            conn = self._connect()
            try:
                results: list[dict] = []
                for score, fid in zip(scores[0], faiss_ids[0]):
                    if fid < 0:
                        continue
                    # Look up by SQLite rowid, which is the ID stored in FAISS.
                    if disease_filter:
                        row = conn.execute(
                            "SELECT * FROM hypotheses WHERE rowid = ? AND disease = ?",
                            (int(fid), disease_filter),
                        ).fetchone()
                    else:
                        row = conn.execute(
                            "SELECT * FROM hypotheses WHERE rowid = ?",
                            (int(fid),),
                        ).fetchone()
                    if row is None:
                        continue
                    r = dict(row)
                    r["similarity_score"] = round(float(score), 4)
                    results.append(r)
                    if len(results) >= top_k:
                        break

                for r in results:
                    conn.execute(
                        "UPDATE hypotheses SET retrieval_count = retrieval_count + 1 WHERE id = ?",
                        (r["id"],),
                    )
                conn.commit()
                return results
            finally:
                conn.close()

    def get_by_id(self, hypothesis_id: str) -> Optional[dict]:
        with _LOCK:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM hypotheses WHERE id = ?", (hypothesis_id,)
                ).fetchone()
                if row is None:
                    return None
                result = dict(row)
                conn.execute(
                    "UPDATE hypotheses SET retrieval_count = retrieval_count + 1 WHERE id = ?",
                    (hypothesis_id,),
                )
                conn.commit()
                return result
            finally:
                conn.close()

    def list_all(
        self,
        disease: Optional[str],
        phase_filter: Optional[str],
        limit: int,
    ) -> list[dict]:
        conditions, params = [], []
        if disease:
            conditions.append("disease = ?")
            params.append(disease)
        if phase_filter:
            conditions.append("phase_origin = ?")
            params.append(phase_filter)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        with _LOCK:
            conn = self._connect()
            try:
                rows = conn.execute(
                    f"""SELECT id, disease, phase_origin, keywords,
                               efficacy_score, safety_score,
                               created_at, retrieval_count
                        FROM hypotheses {where}
                        ORDER BY created_at DESC LIMIT ?""",
                    params,
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def update_scores(
        self,
        hypothesis_id: str,
        new_efficacy_score: float,
        new_safety_score: float,
        update_reason: str,
    ) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        with _LOCK:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """UPDATE hypotheses
                       SET efficacy_score = ?,
                           safety_score   = ?,
                           notes          = COALESCE(notes, '') || ?,
                           updated_at     = ?
                       WHERE id = ?""",
                    (
                        new_efficacy_score, new_safety_score,
                        f"\n[{now}] {update_reason}",
                        now, hypothesis_id,
                    ),
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()
