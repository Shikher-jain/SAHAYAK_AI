import os
import time
import logging
import json
from typing import Optional, Dict, Any
from upstash_vector import AsyncIndex

logger = logging.getLogger(__name__)

UPSTASH_VECTOR_REST_URL = os.getenv("UPSTASH_VECTOR_REST_URL")
UPSTASH_VECTOR_REST_TOKEN = os.getenv("UPSTASH_VECTOR_REST_TOKEN")
SEMANTIC_CACHE_SIMILARITY_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_SIMILARITY_THRESHOLD", "0.92"))
SEMANTIC_CACHE_TTL_SECONDS = int(os.getenv("SEMANTIC_CACHE_TTL_SECONDS", "172800"))

class SemanticCache:
    def __init__(self):
        if UPSTASH_VECTOR_REST_URL and UPSTASH_VECTOR_REST_TOKEN:
            # Using Upstash Vector's built-in embedding capabilities by passing raw text as `data`
            self.index = AsyncIndex(url=UPSTASH_VECTOR_REST_URL, token=UPSTASH_VECTOR_REST_TOKEN)
            self.enabled = True
        else:
            self.enabled = False
            logger.warning("Upstash Vector credentials missing. Semantic Caching disabled.")

    async def get_cached_response(self, query: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Queries Upstash Vector for a semantically identical question."""
        if not self.enabled:
            return None

        # Check bypass rules
        if self._should_bypass(query):
            return None

        try:
            # Upstash automatically embeds the 'data' string using the index's configured model
            results = await self.index.query(
                data=query,
                top_k=1,
                include_metadata=True
            )
            if not results:
                return None

            best_match = results[0]
            if best_match.score >= SEMANTIC_CACHE_SIMILARITY_THRESHOLD:
                metadata = best_match.metadata or {}
                timestamp = metadata.get("timestamp", 0)
                
                # Check TTL
                if time.time() - timestamp > SEMANTIC_CACHE_TTL_SECONDS:
                    return None
                    
                logger.info(f"Semantic Cache HIT for query: '{query[:30]}...' (Score: {best_match.score})")
                return {
                    "response": metadata.get("response", ""),
                    "routed_agent": metadata.get("agent_type", "cached"),
                    "citations": json.loads(metadata.get("citations", "[]")),
                    "cached": True
                }
        except Exception as e:
            logger.error(f"Semantic Cache Lookup Error: {e}")
            
        return None

    async def set_cached_response(self, query: str, response_data: Dict[str, Any]):
        """Asynchronously stores the response in Upstash Vector."""
        if not self.enabled or self._should_bypass(query):
            return
            
        # We also bypass caching for certain agent responses
        agent_type = response_data.get("routed_agent", "")
        if agent_type in ["counseling_agent", "unknown"]:
            return

        try:
            # Generate a unique ID (hash of query + timestamp)
            doc_id = f"cache_{abs(hash(query))}_{int(time.time())}"
            metadata = {
                "query": query,
                "response": response_data.get("response", ""),
                "agent_type": agent_type,
                "citations": json.dumps(response_data.get("citations", [])),
                "timestamp": int(time.time())
            }
            
            await self.index.upsert(
                vectors=[
                    {"id": doc_id, "data": query, "metadata": metadata}
                ]
            )
            logger.info(f"Semantic Cache SET for query: '{query[:30]}...'")
        except Exception as e:
            logger.error(f"Semantic Cache Storage Error: {e}")

    def _should_bypass(self, query: str) -> bool:
        """Bypass rules for sensitive or highly dynamic queries."""
        query_lower = query.lower()
        
        # Rule 1: Crisis / Counseling
        crisis_keywords = ["depressed", "suicide", "help me", "sad", "anxious", "lonely", "kill myself", "hurt", "counsel"]
        if any(k in query_lower for k in crisis_keywords):
            return True
            
        # Rule 2: Session-Specific / Personal Pronouns
        personal_keywords = ["my", "i", "me", "mine", "uploaded", "file", "minute", "ago", "today", "now"]
        words = query_lower.split()
        if any(k in words for k in personal_keywords):
            return True
            
        return False

semantic_cache = SemanticCache()
