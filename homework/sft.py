from .base_llm import BaseLLM
from .data import Dataset, benchmark


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(
        full_text,
        padding="max_length",
        truncation=True,
        max_length=256,
        add_special_tokens=False,
    )

    input_ids = full["input_ids"]

    question_ids = tokenizer(
        f"{question} ", add_special_tokens=False)["input_ids"]
    answer_start = min(len(question_ids), len(input_ids))

    # Create labels: mask out the prompt part
    labels = [-100] * len(input_ids)
    for i in range(answer_start, len(input_ids)):
        labels[i] = input_ids[i]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_example(prompt: str, answer: str) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    # Prepend the same instruction used at inference without instantiating a model/tokenizer
    instructed_question = (
        f"{prompt}\n"
        ".Answer following this exact format: answer>answer</answer>, where answer is the ground truth `float` answer. Do not write anything more, do not write anything less. Only exactly the float answer inside the format mentioned"
    )
    answer_text = f"<answer>{answer}</answer>"

    return {
        "question": instructed_question,
        "answer": answer_text,
    }


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)


def train_model(
    output_dir: str | None = None,
    **kwargs,
):
    from pathlib import Path

    import torch
    from transformers import Trainer, TrainingArguments, default_data_collator
    from peft import LoraConfig, get_peft_model

    llm = BaseLLM()

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        r=8,
        lora_alpha=48,
        lora_dropout=0.02,
        bias="none",
    )
    llm.model = get_peft_model(llm.model, lora_config).to(llm.device)

    if torch.cuda.is_available():
        llm.model.enable_input_require_grads()

    train_ds = TokenizedDataset(
        llm.tokenizer,
        Dataset("train"),
        format_example,
    )

    if not output_dir:
        output_path = Path(__file__).parent / "sft_runs"
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_path),
        logging_dir=str(output_path),
        report_to=["tensorboard"],
        learning_rate=1e-4,
        gradient_checkpointing=True,
        per_device_train_batch_size=32,
        num_train_epochs=5,
    )

    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=default_data_collator,
    )

    trainer.train()

    final_ckpt = Path(__file__).parent / "sft_model"
    final_ckpt.mkdir(parents=True, exist_ok=True)
    llm.model.save_pretrained(final_ckpt)
    llm.model.save_pretrained(output_path)

    output_dir = str(final_ckpt)
    test_model(output_dir)


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()

    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
