def generate_dataset(output_json: str, oversample: int = 20, temperature: float = 0.8):
    import json
    from pathlib import Path
    from tqdm import tqdm
    from .cot import CoTModel
    from .data import Dataset, is_answer_valid

    model = CoTModel()
    train = Dataset("train")

    results: list[list[object]] = []

    for q, correct in tqdm(train, desc="Generating RFT dataset"):
        prompt = model.format_prompt(q)
        generations_nested = model.batched_generate(
            [prompt], num_return_sequences=oversample, temperature=temperature)
        generations = generations_nested[0] if isinstance(
            generations_nested, list) and len(generations_nested) > 0 else []

        chosen_reasoning = None
        for g in generations:
            pred = model.parse_answer(g)
            if pred == pred and is_answer_valid(pred, correct):
                chosen_reasoning = g
                break

        if chosen_reasoning is not None:
            results.append([q, float(correct), chosen_reasoning])

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    from fire import Fire

    Fire({"generate_dataset": generate_dataset})
