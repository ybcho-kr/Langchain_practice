from typing import Dict, Any
from src.utils.logger import get_logger


def generate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Answer node.

    For the basic pipeline, search_node already populated the final answer.
    This node acts as a pass-through, but can be extended for formatting.
    """
    logger = get_logger()
    answer = state.get("answer", "")
    logger.debug(f"[generate] Final answer length: {len(answer)}")
    return {}


