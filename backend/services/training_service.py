"""Fine-tuning training service — real LoRA/QLoRA orchestration with progress tracking.

Runs an actual transformers.Trainer training loop (not simulated) on a background
thread, using PEFT LoRA adapters on top of a Seq2Seq base model (e.g. flan-t5-base).
Training examples come from dataset_service (JSONL prompt/completion pairs added
via the /finetune/examples endpoint).

NOTE ON COMPUTE: full transformer fine-tuning is CPU-feasible only for small
models/datasets and will be slow without a GPU. LoRA reduces trainable
parameters substantially but the forward/backward pass still runs over the
full base model, so CPU training of flan-t5-base is workable for a handful of
epochs over a small dataset (demo/portfolio scale), not production-scale
training. If you have GPU access (e.g. free-tier Colab/Kaggle), point
CUDA_VISIBLE_DEVICES accordingly and transformers.Trainer will use it
automatically — no code changes needed here.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).resolve().parents[2]
_STATE_DIR = _BASE_DIR / "data" / "finetune_state"
_STATE_DIR.mkdir(parents=True, exist_ok=True)

_MIN_EXAMPLES = 2  # Trainer needs at least a couple of examples to run at all.

# Global training state (in-process). A background thread mutates this while
# the API polls it via get_status(). Kept as a plain dict (not persisted to
# disk) — matches the original design; a restart loses in-flight status,
# which is acceptable for a single-worker demo deployment.
_training_state: Dict[str, Any] = {
    "status": "idle",  # idle | training | completed | failed
    "progress": 0.0,
    "epoch": 0,
    "total_epochs": 0,
    "loss": None,
    "message": "",
    "model_path": None,
}
_lock = threading.Lock()


def get_status() -> Dict[str, Any]:
    """Return current training status."""
    with _lock:
        return dict(_training_state)


def start_training(
    base_model: str = "google/flan-t5-base",
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 4,
    use_lora: bool = True,
) -> Dict[str, Any]:
    """Start a real fine-tuning training job (runs in a background thread)."""
    with _lock:
        if _training_state["status"] == "training":
            return {"error": "Training already in progress", "status": "busy"}
        _training_state.update({
            "status": "training",
            "progress": 0.0,
            "epoch": 0,
            "total_epochs": num_epochs,
            "loss": None,
            "message": f"Starting fine-tuning of {base_model}...",
            "model_path": None,
        })

    thread = threading.Thread(
        target=_run_training,
        args=(base_model, num_epochs, learning_rate, batch_size, use_lora),
        daemon=True,
    )
    thread.start()
    return {"status": "started", "message": f"Training started for {base_model}"}


class _PromptCompletionDataset:
    """Minimal torch Dataset over prompt/completion pairs.

    Kept as a plain class implementing __len__/__getitem__ (torch.utils.data.Dataset
    is duck-typed) rather than pulling in the separate `datasets` library, since
    the examples are already small in-memory JSONL records.
    """

    def __init__(self, examples: List[Dict[str, str]], tokenizer, max_length: int = 256):
        self._examples = examples
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self._examples[idx]
        model_inputs = self._tokenizer(
            ex["prompt"],
            max_length=self._max_length,
            truncation=True,
            padding="max_length",
        )
        with self._tokenizer.as_target_tokenizer():
            labels = self._tokenizer(
                ex["completion"],
                max_length=self._max_length,
                truncation=True,
                padding="max_length",
            )
        # Replace pad token id in labels with -100 so the loss ignores padding.
        label_ids = [
            (tok if tok != self._tokenizer.pad_token_id else -100)
            for tok in labels["input_ids"]
        ]
        model_inputs["labels"] = label_ids
        return model_inputs


def _build_progress_callback(num_epochs: int):
    """Create a transformers.TrainerCallback that mirrors real Trainer state
    into _training_state as training progresses, instead of a fabricated
    loss curve."""
    from transformers import TrainerCallback

    class ProgressCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "loss" not in logs:
                return
            epoch = logs.get("epoch", state.epoch or 0)
            with _lock:
                _training_state.update({
                    "epoch": round(epoch, 2),
                    "progress": round(min(epoch / max(num_epochs, 1), 1.0) * 100, 1),
                    "loss": round(logs["loss"], 4),
                    "message": f"Epoch {round(epoch, 2)}/{num_epochs} — loss {round(logs['loss'], 4)}",
                })

    return ProgressCallback()


def _run_training(
    base_model: str,
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
    use_lora: bool,
) -> None:
    """Background training loop — loads dataset, fine-tunes for real, saves checkpoint."""
    try:
        from backend.services.dataset_service import load_examples
        examples = load_examples(limit=10000)

        if len(examples) < _MIN_EXAMPLES:
            with _lock:
                _training_state.update({
                    "status": "failed",
                    "message": (
                        f"Need at least {_MIN_EXAMPLES} training examples "
                        f"(found {len(examples)}). Add more via POST /finetune/examples."
                    ),
                })
            return

        with _lock:
            _training_state["message"] = f"Loaded {len(examples)} training examples. Loading base model..."

        # Import heavy deps lazily so the rest of the app doesn't pay the
        # import cost (torch/transformers/peft) unless training actually runs.
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

        if use_lora:
            from peft import LoraConfig, TaskType, get_peft_model

            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                # Target the attention projection layers — standard for T5-family models.
                target_modules=["q", "v"],
            )
            model = get_peft_model(model, lora_config)
            trainable, total = model.get_nb_trainable_parameters()
            with _lock:
                _training_state["message"] = (
                    f"LoRA adapter attached — {trainable:,}/{total:,} params trainable "
                    f"({100 * trainable / total:.2f}%). Starting training..."
                )
            logger.info("LoRA adapter attached: %s/%s trainable params", trainable, total)

        # 90/10 split when there's enough data for a meaningful eval set;
        # otherwise train on everything and skip eval rather than fail.
        if len(examples) >= 10:
            split_idx = max(1, int(len(examples) * 0.9))
            train_examples, eval_examples = examples[:split_idx], examples[split_idx:]
        else:
            train_examples, eval_examples = examples, None

        train_dataset = _PromptCompletionDataset(train_examples, tokenizer)
        eval_dataset = _PromptCompletionDataset(eval_examples, tokenizer) if eval_examples else None

        checkpoint_dir = _BASE_DIR / "data" / "finetune" / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        run_dir = checkpoint_dir / f"finetuned_{base_model.replace('/', '_')}"

        training_args = Seq2SeqTrainingArguments(
            output_dir=str(run_dir / "trainer_output"),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_strategy="epoch",
            eval_strategy="epoch" if eval_dataset else "no",
            save_strategy="no",  # We save the final adapter/model explicitly below.
            report_to=[],  # Don't try to log to wandb/etc. in this environment.
            disable_tqdm=True,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
            callbacks=[_build_progress_callback(num_epochs)],
        )

        train_result = trainer.train()
        final_loss = round(train_result.training_loss, 4)

        # Save: for LoRA, save_pretrained() saves only the small adapter weights
        # (a few MB), not a full model copy — that's the correct/expected
        # artifact to ship, loaded later via PeftModel.from_pretrained(base, run_dir).
        run_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(run_dir))
        tokenizer.save_pretrained(str(run_dir))

        with _lock:
            _training_state.update({
                "status": "completed",
                "progress": 100.0,
                "epoch": num_epochs,
                "message": f"Training completed. {len(examples)} examples, {num_epochs} epochs.",
                "model_path": str(run_dir),
                "loss": final_loss,
            })
        logger.info("Fine-tuning completed: %s (final_loss=%s)", run_dir, final_loss)

    except Exception as exc:
        logger.exception("Training failed")
        with _lock:
            _training_state.update({
                "status": "failed",
                "message": f"Training failed: {str(exc)}",
            })