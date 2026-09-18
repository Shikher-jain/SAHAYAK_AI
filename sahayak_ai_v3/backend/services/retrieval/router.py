import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class QueryRouter:
    """
    Multimodal Query Router.
    Determines which retrieval channels (Dense, BM25, Graph, specific Modalities)
    should be activated based on the query characteristics and intent.
    """
    
    def route_query(self, query: str, context_modality: str = "text") -> Dict[str, bool]:
        """
        Heuristic-based (and potentially LLM-based) routing.
        For Render free-tier, we rely on fast heuristics to avoid extra LLM latency.
        """
        query_lower = query.lower()
        
        # Default strategy
        strategy = {
            "dense": True,
            "bm25": True,
            "graph": False,
            "image": False,
            "audio": False,
            "video": False,
            "table": False,
            "code": False
        }
        
        # 1. Detect Graph Intent (Relationships, Multi-hop, Dependencies)
        graph_keywords = [
            "who is connected", "depends on", "uses", "related to", 
            "how is", "relationship", "developed by", "owned by"
        ]
        if any(kw in query_lower for kw in graph_keywords):
            strategy["graph"] = True
            
        # 2. Detect Modality Intent
        if "diagram" in query_lower or "image" in query_lower or "picture" in query_lower or context_modality == "image":
            strategy["image"] = True
            
        if "audio" in query_lower or "speaker" in query_lower or "transcript" in query_lower or "said" in query_lower or context_modality == "audio":
            strategy["audio"] = True
            
        if "video" in query_lower or "timestamp" in query_lower or "frame" in query_lower or context_modality == "video":
            strategy["video"] = True
            # Video implies we also want audio transcript
            strategy["audio"] = True
            
        if "table" in query_lower or "column" in query_lower or "row" in query_lower or "numerical" in query_lower:
            strategy["table"] = True
            
        if "code" in query_lower or "function" in query_lower or "class " in query_lower or "file" in query_lower or "def " in query_lower:
            strategy["code"] = True
            
        logger.info(f"Query Router decision for '{query}': {strategy}")
        return strategy
