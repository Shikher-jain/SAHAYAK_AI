import aiohttp
import logging
from typing import List, Dict, Any
from sahayak_ai_v3.backend.core.config import settings

logger = logging.getLogger(__name__)

class HuggingFaceInferenceClient:
    """
    Lightweight Async HTTP client for Hugging Face Serverless Inference API.
    Zero local GPU/RAM footprint.
    """
    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/models/"
        self.headers = {
            "Authorization": f"Bearer {settings.HF_TOKEN}",
            "Content-Type": "application/json"
        }

    async def rerank(self, query: str, documents: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Reranks documents using BAAI/bge-reranker-base via HF API.
        """
        model_id = "BAAI/bge-reranker-base"
        url = f"{self.api_url}{model_id}"
        
        payload = {
            "inputs": {
                "source_sentence": query,
                "sentences": documents
            }
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, headers=self.headers, json=payload) as response:
                    response.raise_for_status()
                    # The response is typically a list of scores corresponding to the sentences
                    scores = await response.json()
                    
                    if not isinstance(scores, list):
                        logger.error(f"Unexpected response from HF API: {scores}")
                        return []
                        
                    # Pair documents with their scores and sort descending
                    scored_docs = list(zip(documents, scores))
                    scored_docs.sort(key=lambda x: x[1], reverse=True)
                    
                    # Return Top K
                    top_results = [{"document": doc, "score": score} for doc, score in scored_docs[:top_k]]
                    return top_results
                    
            except aiohttp.ClientError as e:
                logger.error(f"HF Inference API Error during reranking: {str(e)}")
                # Fallback: return original order if reranking fails to maintain resilience
                return [{"document": doc, "score": 0.0} for doc in documents[:top_k]]

hf_client = HuggingFaceInferenceClient()
