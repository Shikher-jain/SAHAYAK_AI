from typing import List, Dict, Set
import logging
from sahayak_ai_v3.backend.core.models import RetrievedDocument

logger = logging.getLogger(__name__)

class ContextEngine:
    """
    Handles context deduplication, formatting, and compression before 
    passing it to the LLM for grounded generation.
    """
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens

    def deduplicate(self, documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """
        Removes identical text snippets to maximize context window usage.
        """
        seen_texts: Set[str] = set()
        unique_docs: List[RetrievedDocument] = []
        
        for doc in documents:
            text_hash = hash(doc.text.strip().lower())
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                unique_docs.append(doc)
            else:
                logger.debug(f"Deduplicated doc: {doc.id}")
                
        return unique_docs

    def compress_context(self, documents: List[RetrievedDocument]) -> str:
        """
        Formats and optionally compresses the final context payload.
        (Future: Use LLMLingua or similar if memory allows; for now we use simple truncation).
        """
        unique_docs = self.deduplicate(documents)
        
        context_parts = []
        estimated_length = 0
        
        for idx, doc in enumerate(unique_docs):
            # Very rough token estimation (approx 4 chars per token)
            doc_len = len(doc.text) // 4 
            if estimated_length + doc_len > self.max_tokens:
                logger.warning("Context max tokens reached. Truncating remaining docs.")
                break
                
            snippet = f"--- Document {idx+1} [Source: {doc.metadata.get('source', 'Unknown')}] ---\n{doc.text}\n"
            context_parts.append(snippet)
            estimated_length += doc_len
            
        return "\n".join(context_parts)
