from typing import Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from backend.auth import api_key_auth
from backend.services import dataset_service
from backend.services import training_service

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


# --- TASK 16: Fine-tuning training pipeline ---

class TrainRequest(BaseModel):
    base_model: str = "google/flan-t5-base"
    num_epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 4
    use_lora: bool = True


@router.post("/train")
def start_training(req: TrainRequest):
    """Start a fine-tuning training job."""
    return training_service.start_training(
        base_model=req.base_model,
        num_epochs=req.num_epochs,
        learning_rate=req.learning_rate,
        batch_size=req.batch_size,
        use_lora=req.use_lora,
    )


@router.get("/status")
def training_status():
    """Get current training status and progress."""
    return training_service.get_status()


@router.get("/results")
def training_results():
    """Get the results of the last training run."""
    status = training_service.get_status()
    if status["status"] != "completed":
        return {"error": "No completed training run. Current status: " + status["status"]}
    return {
        "status": "completed",
        "model_path": status.get("model_path"),
        "final_loss": status.get("loss"),
        "epochs": status.get("total_epochs"),
        "message": status.get("message"),
    }
