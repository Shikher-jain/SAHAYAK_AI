from typing import Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from backend.auth import api_key_auth
from backend.services import dataset_service

router = APIRouter(prefix="/finetune", tags=["fine-tuning"], dependencies=[Depends(api_key_auth)])


class TrainingExample(BaseModel):
    prompt: str = Field(..., description="Instruction or question")
    completion: str = Field(..., description="Desired assistant response")
    metadata: Optional[Dict[str, str]] = Field(default_factory=dict)


@router.post("/examples")
def add_training_example(example: TrainingExample):
    dataset_service.append_example(example.prompt, example.completion, example.metadata)
    return {"status": "ok"}


@router.get("/examples")
def list_examples(limit: int = 20) -> List[Dict[str, str]]:
    return dataset_service.load_examples(limit=limit)


@router.get("/stats")
def dataset_statistics():
    return dataset_service.dataset_stats()
