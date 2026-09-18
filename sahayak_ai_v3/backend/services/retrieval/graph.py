import logging
from typing import List, Dict, Optional, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

from sahayak_ai_v3.backend.services.retrieval.interfaces import GraphRetriever
from sahayak_ai_v3.backend.core.models import RetrievedDocument

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=4)

class Neo4jGraphRetriever(GraphRetriever):
    """
    Implements GraphRAG logic. Connects to local SQLite Knowledge Graph 
    (or Neo4j if configured) to extract schemas and traverse relationships.
    """
    def __init__(self):
        # Local SQLite backend
        pass

    def _sync_retrieve(self, query: str, top_k: int, filters: Optional[Dict] = None) -> List[RetrievedDocument]:
        try:
            from backend.services.knowledge_graph import extract_from_text, get_entity, query_path
            
            user_id = filters.get("user_id") if filters else None
            
            # 1. Extract entities from the query
            extracted = extract_from_text(query, user_id=user_id)
            entities = extracted.get("entities", [])
            
            if not entities:
                return []
                
            docs = []
            seen_relationships = set()
            
            # 2. Extract 1-hop relationships for each entity
            for entity_name in entities:
                entity_data = get_entity(entity_name, user_id=user_id)
                if not entity_data:
                    continue
                    
                for rel in entity_data.get("relationships", []):
                    rel_sig = f"{rel['source']}-{rel['relation']}-{rel['target']}"
                    if rel_sig not in seen_relationships:
                        seen_relationships.add(rel_sig)
                        docs.append(RetrievedDocument(
                            id=f"graph_{len(docs)}",
                            modality="text",
                            content=f"{rel['source']} is {rel['relation']} {rel['target']}.",
                            source="local_graph",
                            graph_score=1.0,
                            metadata={"type": "relationship"}
                        ))
                        
            # 3. Find paths between multiple entities (Multi-hop)
            if len(entities) > 1:
                for i in range(len(entities)):
                    for j in range(i + 1, min(i + 3, len(entities))):
                        path = query_path(entities[i], entities[j], user_id=user_id)
                        if path and len(path) > 2:
                            path_str = " -> ".join(path)
                            docs.append(RetrievedDocument(
                                id=f"graph_{len(docs)}",
                                modality="text",
                                content=f"Multi-hop connection: {path_str}",
                                source="local_graph",
                                graph_score=1.5, # Boost multi-hop paths
                                metadata={"type": "path"}
                            ))
                            
            # Sort by score descending and return top_k
            docs.sort(key=lambda d: d.graph_score or 0.0, reverse=True)
            return docs[:top_k]
        except Exception as e:
            logger.error(f"Graph retrieval failed: {e}")
            return []

    async def retrieve(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[Dict] = None
    ) -> List[RetrievedDocument]:
        
        logger.info(f"GraphRAG: Initiating schema extraction and relationship traversal for query: {query}")
        
        loop = asyncio.get_running_loop()
        docs = await loop.run_in_executor(_executor, self._sync_retrieve, query, top_k, filters)
        
        if docs:
            logger.info(f"GraphRAG retrieved {len(docs)} results.")
        else:
            logger.warning("GraphRAG retrieval yielded no results.")
            
        return docs
