from typing import Dict, Any
import time
import uuid

try:
    from typing_extensions import TypedDict
except Exception:  # pragma: no cover
    TypedDict = dict  # type: ignore

from src.utils.logger import get_logger


class RAGState(TypedDict, total=False):
    question: str
    analysis: Dict[str, Any]
    search_results: list
    answer: str
    session_id: str | None
    parameters: Dict[str, Any]
    confidence: float
    processing_time: float
    model_used: str


def _build_graph():
    try:
        from langgraph.graph import StateGraph  # type: ignore
    except Exception:
        return None

    from src.agents.question_analyzer import analyze_node
    from src.agents.search_executor import search_node
    from src.agents.answer_generator import generate_node

    graph = StateGraph(state_schema=RAGState)
    graph.add_node("analyze", analyze_node)
    graph.add_node("search", search_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "search")
    graph.add_edge("search", "generate")

    return graph.compile()


compiled_graph = _build_graph()


def run_basic_graph(question: str, session_id: str | None = None, parameters: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Run the basic LangGraph workflow.
    
    Returns:
        State dict with answer, sources, confidence, etc.
    """
    if compiled_graph is None:
        raise RuntimeError("LangGraph is not available or not installed.")
    
    logger = get_logger()
    graph_run_id = f"basic-{uuid.uuid4().hex[:8]}"
    start_time = time.time()
    
    logger.info(f"[Graph {graph_run_id}] Starting workflow for question: {question[:50]}...")
    
    input_state: RAGState = {
        "question": question,
        "session_id": session_id,
        "parameters": parameters or {},
    }
    
    try:
        result = compiled_graph.invoke(input_state)
        elapsed = time.time() - start_time
        logger.info(
            f"[Graph {graph_run_id}] Completed in {elapsed:.2f}s | "
            f"Answer length: {len(result.get('answer', ''))} | "
            f"Sources: {len(result.get('search_results', []))}"
        )
        result["graph_run_id"] = graph_run_id
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[Graph {graph_run_id}] Failed after {elapsed:.2f}s: {str(e)}")
        raise


