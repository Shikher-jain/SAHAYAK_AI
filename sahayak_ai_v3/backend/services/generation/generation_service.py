import logging
import os
from typing import List, Optional

# Reusing the existing LiteLLM or direct integrations
# We will construct a minimal interface for the CRAG pipeline
from backend.rag.generator import Generator

logger = logging.getLogger(__name__)

class GroundedGenerationService:
    """
    Handles final answer generation securely grounded in provided context.
    Optionally evaluates hallucination/faithfulness.
    """
    def __init__(self):
        # We reuse the legacy generator's model chain
        self.generator = Generator()

    async def generate_answer(self, query: str, compressed_context: str) -> str:
        """
        Constructs the prompt and streams or returns the response.
        """
        system_prompt = (
            "You are Sahayak AI, a highly accurate and helpful AI assistant. "
            "Your task is to answer the user's question using ONLY the provided context.\n"
            "If the context does not contain the answer, say 'I do not have enough information to answer that.'\n"
            "Do NOT make up information or use outside knowledge."
        )
        
        user_prompt = f"Context:\n{compressed_context}\n\nQuestion: {query}"
        
        try:
            # We call the existing synchronous generator wrapper 
            # In production, we'd wrap this in run_in_executor or use async client
            response = self.generator.generate_answer(
                context=compressed_context,
                question=query
            )
            return response.get("answer", "")
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "I apologize, but I encountered an error while generating the answer."

    async def check_faithfulness(self, query: str, context: str, generated_answer: str) -> bool:
        """
        A lightweight LLM call to verify if the answer is faithful to the context.
        """
        prompt = (
            f"Context: {context}\n"
            f"Question: {query}\n"
            f"Answer: {generated_answer}\n"
            "Is the answer strictly supported by the context? Answer only 'yes' or 'no'."
        )
        
        response = self.generator.generate_answer(context=context, question=prompt)
        answer = response.get("answer", "")
        if answer and 'yes' in answer.lower():
            return True
        return False
