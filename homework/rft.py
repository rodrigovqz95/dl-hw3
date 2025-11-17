from .base_llm import BaseLLM
from .sft import test_model, TokenizedDataset


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def train_model(
    output_dir: str | None = None,
    input_json: str | None = None,
    start_from_sft: bool = True,
    **kwargs,
):
    from pathlib import Path
    import json
    import torch
    from transformers import TrainingArguments, Trainer, default_data_collator
    from peft import LoraConfig, get_peft_model
    from peft import PeftModel

    data_dir = Path(__file__).parent.parent / "data"
    rft_path = Path(input_json) if input_json is not None else (
        data_dir / "rft.json")
    with rft_path.open() as f:
        raw = json.load(f)

    # Convert to (question, reasoning) pairs
    rft_pairs: list[tuple[str, str]] = [
        (q, reasoning) for q, _ans, reasoning in raw]

    class RFTRaw:
        def __init__(self, pairs: list[tuple[str, str]]):
            self.items = pairs

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    def format_example(question: str, reasoning: str) -> dict[str, str]:
        instructed_question = (
            f"{question}\n"
            "Answer with only one number inside <answer>...</answer> and nothing after."
        )
        return {"question": instructed_question, "answer": reasoning}

    llm = BaseLLM()

    sft_dir = Path(__file__).parent / "sft_model"
    if start_from_sft and sft_dir.exists():
        llm.model = PeftModel.from_pretrained(
            llm.model, str(sft_dir)).to(llm.device)
    else:
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

    train_ds = TokenizedDataset(
        llm.tokenizer, RFTRaw(rft_pairs), format_example)

    out_path = Path(output_dir) if output_dir else (
        Path(__file__).parent / "rft_runs")
    out_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_path),
        logging_dir=str(out_path),
        report_to=["tensorboard"],
        learning_rate=1e-4,
        gradient_checkpointing=True,
        per_device_train_batch_size=32,
        num_train_epochs=5,
        warmup_steps=300,
        lr_scheduler_type="cosine",
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        logging_steps=50,
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=default_data_collator,
    )

    trainer.train()

    final_out = Path(__file__).parent / "rft_model"
    final_out.mkdir(parents=True, exist_ok=True)
    llm.model.save_pretrained(final_out)
    llm.model.save_pretrained(out_path)


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
