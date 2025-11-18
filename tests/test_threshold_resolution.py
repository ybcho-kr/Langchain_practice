from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import sys
import types as py_types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

sys.modules.setdefault("loguru", py_types.SimpleNamespace(logger=MagicMock()))
sys.modules.setdefault("yaml", py_types.SimpleNamespace(safe_load=lambda *_, **__: {}, SafeLoader=None))

dummy_config_module = py_types.ModuleType("src.utils.config")
dummy_config_module.get_config = lambda: SimpleNamespace(model={'llm': SimpleNamespace(name='stub-llm')}, qdrant=SimpleNamespace(), reranker=None)
dummy_config_module.get_rag_config = lambda: SimpleNamespace(
    score_threshold=0.55,
    default_max_sources=4,
    default_max_sources_table=4,
    low_score_general_threshold=0.2,
)
sys.modules.setdefault("src.utils.config", dummy_config_module)

dummy_logger_module = py_types.ModuleType("src.utils.logger")
dummy_logger_module.get_logger = lambda: MagicMock()
sys.modules.setdefault("src.utils.logger", dummy_logger_module)

dummy_doc_module = py_types.ModuleType("src.modules.document_processor")
dummy_doc_module.DocumentProcessor = MagicMock()
dummy_doc_module.DocumentChunk = MagicMock()
sys.modules.setdefault("src.modules.document_processor", dummy_doc_module)

dummy_embedding_module = py_types.ModuleType("src.modules.embedding_module")
dummy_embedding_module.EmbeddingManager = MagicMock()
sys.modules.setdefault("src.modules.embedding_module", dummy_embedding_module)

dummy_vector_store_module = py_types.ModuleType("src.modules.vector_store")
dummy_vector_store_module.VectorStoreManager = MagicMock()
sys.modules.setdefault("src.modules.vector_store", dummy_vector_store_module)

dummy_llm_module = py_types.ModuleType("src.models.llm_models")
dummy_llm_module.OllamaLLMClient = MagicMock()
sys.modules.setdefault("src.models.llm_models", dummy_llm_module)

dummy_reranker_module = py_types.ModuleType("src.modules.reranker_module")
dummy_reranker_module.CrossEncoderReranker = MagicMock()
sys.modules.setdefault("src.modules.reranker_module", dummy_reranker_module)

import asyncio
import pytest

from src.modules import rag_system
from src.modules.rag_system import RAGSystem


def build_stub_rag_system(monkeypatch) -> RAGSystem:
    """RAGSystem 동작 중 threshold 결정 경로만 검증하기 위한 최소 더블"""

    system = RAGSystem.__new__(RAGSystem)
    system.logger = MagicMock()
    system.config = SimpleNamespace(
        model={'llm': SimpleNamespace(base_url='http://localhost:11434', max_tokens=1000, temperature=0.1, top_p=0.9)}
    )
    system.rag_config = SimpleNamespace(
        score_threshold=0.55,
        default_max_sources=4,
        default_max_sources_table=4,
        low_score_general_threshold=0.2,
    )
    system.llm_client = MagicMock()
    system.llm_client.model_name = "test-llm"
    system.llm_client.generate_answer_async = AsyncMock(
        return_value=SimpleNamespace(text="ans", is_general=False, has_rag_context=True)
    )
    system.llm_client.generate_answer_with_metadata = MagicMock(
        return_value=SimpleNamespace(text="ans", is_general=False, has_rag_context=True)
    )
    system._build_context = MagicMock(return_value="ctx")
    system._calculate_confidence = MagicMock(return_value=0.8)
    system._format_sources = MagicMock(return_value=[{"source": "s"}])
    system.reranker = None
    system.embedding_manager = MagicMock()

    qdrant_store = SimpleNamespace(
        search_similar_async=AsyncMock(
            return_value=[{'content': 'c', 'score': 0.7, 'metadata': {}}]
        )
    )
    system.vector_store_manager = SimpleNamespace(
        store=qdrant_store,
        langchain_retrieval_manager=None,
        search_with_table_filter=MagicMock(return_value=[{'content': 'c', 'score': 0.7, 'metadata': {}}]),
        search_by_table_title=MagicMock(return_value=[{'content': 'c', 'score': 0.7, 'metadata': {}}]),
    )

    # 검색 경로로 진입하도록 일반 질문 판별을 무효화
    monkeypatch.setattr(rag_system, "is_general_question", lambda _: False)
    return system


def test_threshold_alignment_across_entrypoints(monkeypatch):
    rag = build_stub_rag_system(monkeypatch)
    question = "동일한 질문으로 threshold 비교"

    expected_decision = rag._build_threshold_decision(
        question=question,
        max_sources=rag.rag_config.default_max_sources,
        requested_threshold=None,
    )

    asyncio.run(rag.query_async(question, retrievers={'use_qdrant': True}))
    table_response = rag.query_with_table_filter(question, table_title="표", is_table_data=True)

    assert rag.vector_store_manager.store.search_similar_async.call_args.kwargs["score_threshold"] == pytest.approx(
        expected_decision.final_threshold
    )
    assert rag.vector_store_manager.search_with_table_filter.call_args.kwargs["score_threshold"] == pytest.approx(
        expected_decision.final_threshold
    )
    assert table_response.sources  # ensure flow executed
