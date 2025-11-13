from .base_llm import BaseLLM
from .sft import test_model


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    # Ensure compatibility by passing string path
    llm.model = PeftModel.from_pretrained(
        llm.model, str(model_path)).to(llm.device)
    llm.model.eval()

    return llm


def train_model(
    output_dir: str,
    **kwargs,
):
    # Reuse much of the SFT code here
    from pathlib import Path
    import json
    import torch
    from transformers import TrainingArguments, Trainer, default_data_collator
    from peft import LoraConfig, get_peft_model

    from .sft import TokenizedDataset

    # Load RFT dataset (question, correct_answer, reasoning)
    data_dir = Path(__file__).parent.parent / "data"
    rft_path = Path(kwargs.get("input_json", data_dir / "rft.json"))
    with rft_path.open() as f:
        raw = json.load(f)

    # Convert to (question, reasoning) pairs
    rft_pairs: list[tuple[str, str]] = [
        (q, reasoning) for q, _ans, reasoning in raw]

    class RFTRaw:
        def __init__(self, pairs):
            self.items = pairs

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    def format_rft(question: str, reasoning: str) -> dict[str, str]:
        # Match SFT: prepend the same instruction prefix used at inference
        instructed_question = (
            f"{question}\n"
            "Answer with only one number inside <answer>...</answer> and nothing after."
        )
        # Supervise the full reasoning (should end with <answer>...</answer>)
        return {"question": instructed_question, "answer": reasoning}

    # Build base model and attach LoRA
    from .base_llm import BaseLLM

    llm = BaseLLM()
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        r=8,
        lora_alpha=48,
        lora_dropout=0.05,
        bias="none",
    )
    llm.model = get_peft_model(llm.model, lora_config).to(llm.device)

    if torch.cuda.is_available():
        llm.model.enable_input_require_grads()

    train_ds = TokenizedDataset(llm.tokenizer, RFTRaw(rft_pairs), format_rft)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        logging_dir=str(out_dir),
        report_to=["tensorboard"],
        learning_rate=1e-4,
        gradient_checkpointing=True,
        per_device_train_batch_size=32,
        num_train_epochs=10,
        save_strategy="epoch",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=default_data_collator,
    )

    trainer.train()

    # Save the adapter as rft_model for loader compatibility
    final_out = Path(__file__).parent / "rft_model"
    final_out.mkdir(parents=True, exist_ok=True)
    llm.model.save_pretrained(final_out)
    llm.model.save_pretrained(out_dir)


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
