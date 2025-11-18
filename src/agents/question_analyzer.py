from typing import Dict, Any
from src.utils.logger import get_logger


def analyze_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Lightweight question analysis node.

    Produces simple intent metadata to guide downstream nodes.
    """
    logger = get_logger()
    question: str = state.get("question", "") or ""
    intent = "qa"
    if any(k in question for k in ["표", "table", "제목"]):
        intent = "qa-table"
    logger.debug(f"[analyze] Intent: {intent}")
    return {"analysis": {"intent": intent}}


