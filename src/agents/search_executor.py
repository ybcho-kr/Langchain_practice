import threading
import time
from typing import Dict, Any

from src.utils.logger import get_logger

_rag_system_instance = None
_rag_system_lock = threading.Lock()


def _get_rag_system():
    logger = get_logger()
    try:
        from src.api.main import get_rag_system  # type: ignore
        return get_rag_system()
    except Exception:
        # Lazy fallback creation to avoid circular import issues at import time
        global _rag_system_instance
        if _rag_system_instance is not None:
            logger.debug("Reusing cached fallback RAGSystem instance.")
            return _rag_system_instance

        with _rag_system_lock:
            if _rag_system_instance is None:
                from src.modules.rag_system import RAGSystem  # type: ignore
                logger.debug("Creating fallback RAGSystem instance.")
                _rag_system_instance = RAGSystem()
            else:
                logger.debug("Fallback RAGSystem instance already created by another thread.")

        return _rag_system_instance


async def search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Search node that adapts existing RAGSystem.query_async() (비동기)."""
    logger = get_logger()
    start_time = time.time()
    rag = _get_rag_system()
    logger.debug(f"[search] Using RAG system instance id: {id(rag)}")
    question: str = state.get("question", "") or ""
    logger.debug(f"[search] Executing search for: {question[:50]}...")

    # Optional parameters passthrough
    params: Dict[str, Any] = state.get("parameters", {}) or {}
    max_sources = params.get("max_sources")
    score_threshold = params.get("score_threshold")
    model_name = params.get("model")
    retrievers = None
    if any(k in params for k in ("use_qdrant", "use_faiss", "use_bm25", "use_reranker", "reranker_alpha", "reranker_top_k")):
        retrievers = {}
        if "use_qdrant" in params:
            retrievers["use_qdrant"] = params["use_qdrant"]
        if "use_faiss" in params:
            retrievers["use_faiss"] = params["use_faiss"]
        if "use_bm25" in params:
            retrievers["use_bm25"] = params["use_bm25"]
        if "use_reranker" in params:
            retrievers["use_reranker"] = params["use_reranker"]
        if "reranker_alpha" in params:
            retrievers["reranker_alpha"] = params["reranker_alpha"]
        if "reranker_top_k" in params:
            retrievers["reranker_top_k"] = params["reranker_top_k"]

    response = await rag.query_async(
        question=question,
        max_sources=max_sources,
        score_threshold=score_threshold,
        model_name=model_name,
        retrievers=retrievers,
    )

    elapsed = time.time() - start_time
    logger.debug(
        f"[search] Completed in {elapsed:.2f}s | "
        f"Sources: {len(response.sources)} | "
        f"Confidence: {response.confidence:.2f}"
    )
    return {
        "search_results": response.sources,
        "answer": response.answer,
        "confidence": response.confidence,
        "processing_time": response.processing_time,
        "model_used": response.model_used,
    }


