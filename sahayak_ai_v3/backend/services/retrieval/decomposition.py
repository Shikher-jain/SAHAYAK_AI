import logging
from typing import List

logger = logging.getLogger(__name__)

class QueryDecomposer:
    """
    Decomposes complex queries into multiple sub-queries to improve recall.
    Uses local heuristics/rules to avoid LLM latency on the free tier, 
    but falls back to LLM if available and fast.
    """
    
    def __init__(self):
        pass
        
    def decompose(self, query: str) -> List[str]:
        """
        Takes a complex query and returns a list of sub-queries.
        Always includes the original query.
        """
        sub_queries = [query]
        
        query_lower = query.lower()
        
        # Simple heuristic: split by "and" if it joins two clauses
        if " and " in query_lower:
            parts = [p.strip() for p in query_lower.split(" and ") if len(p.strip()) > 10]
            if len(parts) > 1:
                sub_queries.extend(parts)
                
        # Simple heuristic: split by "vs" or "versus" or "compared to"
        for split_token in [" vs ", " versus ", " compared to "]:
            if split_token in query_lower:
                parts = [p.strip() for p in query_lower.split(split_token) if len(p.strip()) > 3]
                if len(parts) == 2:
                    sub_queries.append(f"details about {parts[0]}")
                    sub_queries.append(f"details about {parts[1]}")
                    break
                    
        # Optional: LLM-based decomposition using Groq (commented out for latency/rate limits, 
        # but could be enabled conditionally)
        # try:
        #     from backend.common.groq_client import groq_complete
        #     # LLM call here...
        # except Exception:
        #     pass
            
        # Deduplicate
        seen = set()
        unique_queries = []
        for q in sub_queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)
                
        if len(unique_queries) > 1:
            logger.info(f"Decomposed query into: {unique_queries}")
            
        return unique_queries
