"""Tests for the hypothesis bank store (FAISS + SQLite)."""

import threading
from hashlib import md5 as _md5
from pathlib import Path

import numpy as np
import pytest

from app.src.core.tools.hypothesis_bank.store import EMBED_DIM, HypothesisStore


# ---------------------------------------------------------------------------
# Deterministic embed function for tests (no litellm proxy required).
# Each unique word gets a fixed random unit-direction vector derived from
# its MD5 hash, so texts sharing words naturally have higher cosine similarity.
# ---------------------------------------------------------------------------

def _word_vec(word: str) -> np.ndarray:
    seed = int(_md5(word.encode()).hexdigest(), 16) % (2**31)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _test_embed_fn(texts: list[str]) -> np.ndarray:
    """Word-overlap-based embedding: shared words -> higher cosine similarity."""
    result = []
    for text in texts:
        vec = np.zeros(EMBED_DIM, dtype=np.float32)
        for word in text.lower().split():
            vec += _word_vec(word)
        norm = np.linalg.norm(vec)
        result.append(vec / max(norm, 1e-9))
    return np.array(result, dtype=np.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path: Path) -> HypothesisStore:
    return HypothesisStore(
        metadata_path=tmp_path / "metadata.db",
        index_path=tmp_path / "embeddings.db",
        embed_fn=_test_embed_fn,
    )


@pytest.fixture
def populated_store(store: HypothesisStore) -> HypothesisStore:
    store.add(
        hypothesis_id="aaaa0001",
        disease="alzheimer",
        phase_origin="hypothesize",
        text="Inhibiting tau aggregation may slow neurodegeneration in alzheimer patients",
        keywords="tau aggregation neurodegeneration alzheimer",
        efficacy_score=0.72,
        safety_score=0.85,
        notes="Initial hypothesis from literature scan",
    )
    store.add(
        hypothesis_id="bbbb0002",
        disease="diabetes",
        phase_origin="hypothesize",
        text="SGLT2 inhibitors reduce glucose reabsorption in the kidney tubules",
        keywords="SGLT2 glucose insulin kidney",
        efficacy_score=0.80,
        safety_score=0.88,
        notes="",
    )
    store.add(
        hypothesis_id="cccc0003",
        disease="alzheimer",
        phase_origin="test",
        text="Beta-amyloid clearance via passive immunotherapy improves cognition",
        keywords="amyloid immunotherapy cognition antibody",
        efficacy_score=0.65,
        safety_score=0.78,
        notes="Derived from in-silico trial",
    )
    return store


# ---------------------------------------------------------------------------
# Add and retrieve
# ---------------------------------------------------------------------------

class TestAddAndRetrieve:
    def test_add_then_get_by_id_returns_full_record(self, store: HypothesisStore) -> None:
        store.add(
            hypothesis_id="test0001",
            disease="cancer",
            phase_origin="hypothesize",
            text="PD-L1 checkpoint inhibition activates T-cell response",
            keywords="PD-L1 checkpoint T-cell",
            efficacy_score=0.75,
            safety_score=0.80,
            notes="checkpoint inhibitor hypothesis",
        )
        result = store.get_by_id("test0001")

        assert result is not None
        assert result["id"] == "test0001"
        assert result["disease"] == "cancer"
        assert result["phase_origin"] == "hypothesize"
        assert result["text"] == "PD-L1 checkpoint inhibition activates T-cell response"
        assert result["efficacy_score"] == pytest.approx(0.75)
        assert result["safety_score"] == pytest.approx(0.80)

    def test_get_by_id_increments_retrieval_count(self, store: HypothesisStore) -> None:
        store.add(
            hypothesis_id="test0002",
            disease="cancer",
            phase_origin="hypothesize",
            text="KRAS inhibition reduces tumor proliferation",
            keywords="KRAS tumor",
            efficacy_score=0.6,
            safety_score=0.7,
            notes="",
        )
        store.get_by_id("test0002")
        store.get_by_id("test0002")
        result = store.get_by_id("test0002")

        # Snapshot is captured before the counter increment, so after 3 calls
        # the 3rd result shows count=2 (the 2 prior fetches).
        assert result["retrieval_count"] == 2

    def test_get_by_id_not_found_returns_none(self, store: HypothesisStore) -> None:
        assert store.get_by_id("doesnotexist") is None


# ---------------------------------------------------------------------------
# Semantic search via FAISS
# ---------------------------------------------------------------------------

class TestSemanticSearch:
    def test_search_returns_most_relevant_result_first(
        self, populated_store: HypothesisStore
    ) -> None:
        # Query shares key words with aaaa0001 (tau, aggregation, neurodegeneration).
        results = populated_store.search(
            query="tau protein aggregation neurodegeneration alzheimer",
            disease_filter=None,
            top_k=5,
        )

        assert len(results) > 0
        assert results[0]["id"] == "aaaa0001"
        assert "similarity_score" in results[0]

    def test_search_with_disease_filter_restricts_results(
        self, populated_store: HypothesisStore
    ) -> None:
        results = populated_store.search(
            query="glucose insulin kidney SGLT2",
            disease_filter="diabetes",
            top_k=5,
        )

        assert len(results) == 1
        assert results[0]["disease"] == "diabetes"

    def test_search_disease_filter_excludes_other_diseases(
        self, populated_store: HypothesisStore
    ) -> None:
        results = populated_store.search(
            query="tau aggregation",
            disease_filter="diabetes",
            top_k=5,
        )

        assert all(r["disease"] == "diabetes" for r in results)

    def test_search_top_k_limits_result_count(
        self, populated_store: HypothesisStore
    ) -> None:
        results = populated_store.search(query="disease", disease_filter=None, top_k=1)

        assert len(results) == 1

    def test_search_similarity_scores_are_in_range(
        self, populated_store: HypothesisStore
    ) -> None:
        results = populated_store.search(
            query="inhibitor treatment",
            disease_filter=None,
            top_k=5,
        )

        for r in results:
            assert -1.0 <= r["similarity_score"] <= 1.0

    def test_search_increments_retrieval_count(
        self, populated_store: HypothesisStore
    ) -> None:
        populated_store.search(query="tau", disease_filter="alzheimer", top_k=5)
        result = populated_store.get_by_id("aaaa0001")

        assert result["retrieval_count"] >= 1


# ---------------------------------------------------------------------------
# Empty database
# ---------------------------------------------------------------------------

class TestEmptyDatabase:
    def test_search_on_empty_db_returns_empty_list(self, store: HypothesisStore) -> None:
        assert store.search(query="anything", disease_filter=None, top_k=5) == []

    def test_list_on_empty_db_returns_empty_list(self, store: HypothesisStore) -> None:
        assert store.list_all(disease=None, phase_filter=None, limit=20) == []

    def test_get_by_id_on_empty_db_returns_none(self, store: HypothesisStore) -> None:
        assert store.get_by_id("xyz") is None


# ---------------------------------------------------------------------------
# List hypotheses
# ---------------------------------------------------------------------------

class TestListHypotheses:
    def test_list_all_returns_all_records(self, populated_store: HypothesisStore) -> None:
        results = populated_store.list_all(disease=None, phase_filter=None, limit=20)

        assert len(results) == 3

    def test_list_filter_by_disease(self, populated_store: HypothesisStore) -> None:
        results = populated_store.list_all(disease="alzheimer", phase_filter=None, limit=20)

        assert len(results) == 2
        assert all(r["disease"] == "alzheimer" for r in results)

    def test_list_filter_by_phase(self, populated_store: HypothesisStore) -> None:
        results = populated_store.list_all(disease=None, phase_filter="test", limit=20)

        assert len(results) == 1
        assert results[0]["id"] == "cccc0003"

    def test_list_combined_filters(self, populated_store: HypothesisStore) -> None:
        results = populated_store.list_all(
            disease="alzheimer", phase_filter="hypothesize", limit=20
        )

        assert len(results) == 1
        assert results[0]["id"] == "aaaa0001"

    def test_list_limit_truncates_results(self, tmp_path: Path) -> None:
        store = HypothesisStore(
            metadata_path=tmp_path / "limit_meta.db",
            index_path=tmp_path / "limit_emb.db",
            embed_fn=_test_embed_fn,
        )
        for i in range(6):
            store.add(
                hypothesis_id=f"lim0000{i}",
                disease="cancer",
                phase_origin="hypothesize",
                text=f"hypothesis number {i}",
                keywords="test",
                efficacy_score=0.5,
                safety_score=0.5,
                notes="",
            )

        results = store.list_all(disease=None, phase_filter=None, limit=3)

        assert len(results) == 3

    def test_list_summary_excludes_full_text_and_embedding(
        self, populated_store: HypothesisStore
    ) -> None:
        results = populated_store.list_all(disease=None, phase_filter=None, limit=20)

        for r in results:
            assert "text" not in r
            assert "id" in r
            assert "efficacy_score" in r


# ---------------------------------------------------------------------------
# Update scores
# ---------------------------------------------------------------------------

class TestUpdateScores:
    def test_update_scores_changes_values(self, populated_store: HypothesisStore) -> None:
        found = populated_store.update_scores(
            hypothesis_id="aaaa0001",
            new_efficacy_score=0.91,
            new_safety_score=0.96,
            update_reason="Post in-silico trial",
        )
        result = populated_store.get_by_id("aaaa0001")

        assert found is True
        assert result["efficacy_score"] == pytest.approx(0.91)
        assert result["safety_score"] == pytest.approx(0.96)

    def test_update_scores_appends_reason_to_notes(
        self, populated_store: HypothesisStore
    ) -> None:
        populated_store.update_scores(
            hypothesis_id="aaaa0001",
            new_efficacy_score=0.91,
            new_safety_score=0.96,
            update_reason="Post in-silico trial",
        )
        result = populated_store.get_by_id("aaaa0001")

        assert "Post in-silico trial" in result["notes"]

    def test_update_scores_on_empty_notes_field(
        self, populated_store: HypothesisStore
    ) -> None:
        found = populated_store.update_scores(
            hypothesis_id="bbbb0002",
            new_efficacy_score=0.82,
            new_safety_score=0.90,
            update_reason="Safety screen passed",
        )
        result = populated_store.get_by_id("bbbb0002")

        assert found is True
        assert "Safety screen passed" in result["notes"]

    def test_update_nonexistent_hypothesis_returns_false(
        self, store: HypothesisStore
    ) -> None:
        found = store.update_scores(
            hypothesis_id="nope1234",
            new_efficacy_score=0.5,
            new_safety_score=0.5,
            update_reason="test",
        )

        assert found is False


# ---------------------------------------------------------------------------
# FAISS index persistence
# ---------------------------------------------------------------------------

class TestFAISSPersistence:
    def test_index_survives_store_reload(self, tmp_path: Path) -> None:
        meta = tmp_path / "meta.db"
        idx = tmp_path / "embeddings.db"

        store1 = HypothesisStore(metadata_path=meta, index_path=idx, embed_fn=_test_embed_fn)
        store1.add(
            hypothesis_id="pers0001",
            disease="cancer",
            phase_origin="hypothesize",
            text="mTOR pathway inhibition reduces tumor growth",
            keywords="mTOR tumor growth",
            efficacy_score=0.7,
            safety_score=0.8,
            notes="",
        )

        # A second store instance pointing at the same files should find the vector.
        store2 = HypothesisStore(metadata_path=meta, index_path=idx, embed_fn=_test_embed_fn)
        results = store2.search(query="mTOR tumor", disease_filter=None, top_k=5)

        assert len(results) == 1
        assert results[0]["id"] == "pers0001"


# ---------------------------------------------------------------------------
# Concurrent writes
# ---------------------------------------------------------------------------

class TestConcurrentWrites:
    def test_concurrent_adds_all_succeed(self, tmp_path: Path) -> None:
        store = HypothesisStore(
            metadata_path=tmp_path / "concurrent_meta.db",
            index_path=tmp_path / "concurrent_emb.db",
            embed_fn=_test_embed_fn,
        )
        errors: list[Exception] = []

        def add_one(i: int) -> None:
            try:
                store.add(
                    hypothesis_id=f"con0{i:04x}",
                    disease="diabetes",
                    phase_origin="hypothesize",
                    text=f"concurrent hypothesis {i} about glucose metabolism pathways",
                    keywords="glucose insulin metabolism",
                    efficacy_score=0.5,
                    safety_score=0.5,
                    notes="",
                )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=add_one, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent writes raised: {errors}"
        results = store.list_all(disease=None, phase_filter=None, limit=100)
        assert len(results) == 20
