"""Hypothesis bank tools available across all research phases."""

import json
import uuid

from langchain_core.tools import tool

from app.src.core.tools.hypothesis_bank.store import HypothesisStore
from app.src.utils.logger import get_logger


logger = get_logger(__name__)
_store = HypothesisStore()


def _ok(**kwargs) -> str:
    return json.dumps({"status": "ok", **kwargs}, indent=2)


def _err(msg: str) -> str:
    return json.dumps({"status": "error", "message": msg})


@tool
def add_to_hypothesis_bank(
    hypothesis_text: str,
    disease: str,
    phase_origin: str,
    keywords: str,
    efficacy_score: float,
    safety_score: float,
    notes: str = "",
) -> str:
    """Store a hypothesis in the shared hypothesis bank for cross-phase retrieval.

    Embeds and indexes the hypothesis for later semantic search.
    Returns confirmation JSON with hypothesis_id.

    Args:
        hypothesis_text: Full text of the hypothesis (str).
        disease: Target disease in lowercase, e.g. alzheimer (str).
        phase_origin: Phase that generated this, e.g. hypothesize (str).
        keywords: Comma-separated relevant keywords (str).
        efficacy_score: Estimated efficacy 0.0-1.0 (float).
        safety_score: Estimated safety 0.0-1.0 (float).
        notes: Optional extra notes or context (str).
    """
    hypothesis_id = uuid.uuid4().hex[:8]
    try:
        _store.add(
            hypothesis_id=hypothesis_id,
            disease=disease.strip().lower(),
            phase_origin=phase_origin.strip().lower(),
            text=hypothesis_text,
            keywords=keywords,
            efficacy_score=efficacy_score,
            safety_score=safety_score,
            notes=notes,
        )
        logger.info("Hypothesis stored: id=%s disease=%s", hypothesis_id, disease)
        return _ok(
            hypothesis_id=hypothesis_id,
            disease=disease,
            message="Hypothesis stored successfully.",
        )
    except Exception as e:
        logger.error("add_to_hypothesis_bank failed: %s", e)
        return _err(str(e))


@tool
def search_hypothesis_bank(
    query: str,
    disease_filter: str = "",
    top_k: int = 5,
) -> str:
    """Semantic search against stored hypotheses using cosine similarity.

    Returns JSON list of top_k matches with similarity scores and metadata.

    Args:
        query: Natural language search query (str).
        disease_filter: Restrict results to this disease; empty means all (str).
        top_k: Max number of results to return, default 5 (int).
    """
    try:
        results = _store.search(
            query=query,
            disease_filter=disease_filter.strip().lower() or None,
            top_k=max(1, top_k),
        )
        if not results:
            return _ok(count=0, results=[], message="No hypotheses found.")
        return _ok(count=len(results), results=results)
    except Exception as e:
        logger.error("search_hypothesis_bank failed: %s", e)
        return _err(str(e))


@tool
def get_hypothesis_by_id(hypothesis_id: str) -> str:
    """Retrieve a specific hypothesis by its unique ID.

    Returns the full hypothesis record as JSON.

    Args:
        hypothesis_id: The 8-char hypothesis ID returned by add_to_hypothesis_bank (str).
    """
    try:
        result = _store.get_by_id(hypothesis_id.strip())
        if result is None:
            return _err(f"Hypothesis '{hypothesis_id}' not found.")
        return _ok(hypothesis=result)
    except Exception as e:
        logger.error("get_hypothesis_by_id failed: %s", e)
        return _err(str(e))


@tool
def list_hypotheses(
    disease: str = "",
    phase_filter: str = "",
    limit: int = 20,
) -> str:
    """List stored hypotheses with optional filters, ordered by most recent.

    Returns JSON summary list with id, disease, phase_origin, keywords, scores.

    Args:
        disease: Filter by disease name; empty means all (str).
        phase_filter: Filter by originating phase; empty means all (str).
        limit: Max results to return, default 20 (int).
    """
    try:
        results = _store.list_all(
            disease=disease.strip().lower() or None,
            phase_filter=phase_filter.strip().lower() or None,
            limit=max(1, limit),
        )
        if not results:
            return _ok(count=0, results=[], message="No hypotheses found.")
        return _ok(count=len(results), results=results)
    except Exception as e:
        logger.error("list_hypotheses failed: %s", e)
        return _err(str(e))


@tool
def update_hypothesis_scores(
    hypothesis_id: str,
    new_efficacy_score: float,
    new_safety_score: float,
    update_reason: str,
) -> str:
    """Update efficacy and safety scores for a hypothesis after test phase results.

    Appends the update reason to the hypothesis notes with a timestamp.
    Returns confirmation JSON.

    Args:
        hypothesis_id: The unique hypothesis ID to update (str).
        new_efficacy_score: Revised efficacy score 0.0-1.0 (float).
        new_safety_score: Revised safety score 0.0-1.0 (float).
        update_reason: Reason for the update, e.g. in-silico trial results (str).
    """
    try:
        found = _store.update_scores(
            hypothesis_id=hypothesis_id.strip(),
            new_efficacy_score=new_efficacy_score,
            new_safety_score=new_safety_score,
            update_reason=update_reason,
        )
        if not found:
            return _err(f"Hypothesis '{hypothesis_id}' not found.")
        logger.info(
            "Scores updated: id=%s efficacy=%.2f safety=%.2f",
            hypothesis_id, new_efficacy_score, new_safety_score,
        )
        return _ok(
            hypothesis_id=hypothesis_id,
            efficacy_score=new_efficacy_score,
            safety_score=new_safety_score,
            message="Scores updated successfully.",
        )
    except Exception as e:
        logger.error("update_hypothesis_scores failed: %s", e)
        return _err(str(e))


hypothesis_bank_tools = [
    add_to_hypothesis_bank,
    search_hypothesis_bank,
    get_hypothesis_by_id,
    list_hypotheses,
    update_hypothesis_scores,
]
