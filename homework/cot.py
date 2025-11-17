from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise unit conversion assistant. Be concise. "
                    "Show one short calculation line with the correct conversion factor(s). "
                    "Then, on a new line, output ONLY the final numeric value inside <answer>...</answer>. "
                    "Do not include units or any words inside the tags, and write nothing after </answer>. "
                    "Round to 3 decimals if needed. Use base-10 data units (1 kB = 1000 B)."
                ),
            },
            {
                "role": "user",
                "content": "Please convert 10 mi/h to in/s.",
            },
            {
                "role": "assistant",
                "content": (
                    "1 mi = 63,360 in and 1 h = 3600 s. "
                    "10 × 63,360 / 3600 =\n<answer>176</answer>"
                ),
            },
            {
                "role": "user",
                "content": "How many gram are there per 6 kg?",
            },
            {
                "role": "assistant",
                "content": "1 kg = 1000 g. 6 × 1000 =\n<answer>6000</answer>",
            },
            {
                "role": "user",
                "content": "Convert 3 L to mL.",
            },
            {
                "role": "assistant",
                "content": "1 L = 1000 mL. 3 × 1000 =\n<answer>3000</answer>",
            },
            {
                "role": "user",
                "content": "What is 2 kB in B?",
            },
            {
                "role": "assistant",
                "content": "1 kB = 1000 B. 2 × 1000 =\n<answer>2000</answer>",
            },
            {
                "role": "user",
                "content": "Convert 12 in to cm.",
            },
            {
                "role": "assistant",
                "content": "1 in = 2.54 cm. 12 × 2.54 =\n<answer>30.48</answer>",
            },
            {
                "role": "user",
                "content": question.strip(),
            },
        ]

        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
