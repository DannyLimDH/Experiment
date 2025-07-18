import json
import math
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from tqdm.auto import tqdm

MODEL_NAME = "google/gemma-3-4b-it"
BATCH_SIZE = 4
TEMPLATE = """
Role: You are a conversational AI fulfilling the role of a professional empathetic counselor.
Quickly detect the user’s emotion and provide a natural, meaningful response in 1–2 sentences with the appropriate tone (empathy, curiosity, or encouragement).

Internal processing (keep hidden):
1. Identify the primary emotion in the user’s message (sadness, anger, anxiety, neutral, or joy).
2. Map that emotion to a response style:
   • Empathy       → sadness, anger, anxiety
   • Curiosity     → neutral
   • Encouragement → joy
3. Follow these guidelines to generate your reply:
   - Reflect at least one key word or emotion from the input
   - Keep it concise (1–2 sentences)
   - Avoid clichés (“Oh wow,” “That’s beautiful”)
   - If asking a question, start with “why,” “how,” or “what”

Final output:
- Output only the pure response text.
- Do not include any labels, examples, tags, code fences, or internal analysis.

User: "{input}"
"""


def main():
    data_path = Path("train.json")
    with data_path.open() as f:
        data = json.load(f)

    total = len(data)
    quarter = math.ceil(total / 4)
    save_steps = [quarter, quarter * 2, quarter * 3]
    save_names = [f"Save{i+1}.json" for i in range(3)]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )

    results = []
    processed = 0
    save_idx = 0

    for i in tqdm(range(0, total, BATCH_SIZE)):
        batch = data[i : i + BATCH_SIZE]
        prompts = [TEMPLATE.format(input=item["input"]) for item in batch]
        outputs = gen(
            prompts,
            max_new_tokens=64,
            num_return_sequences=3,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False,
        )

        for item, resp in zip(batch, outputs):
            cands = [r["generated_text"].strip() for r in resp]
            results.append({"input": item["input"], "candidates": cands})
            processed += 1
            if save_idx < len(save_steps) and processed == save_steps[save_idx]:
                with open(save_names[save_idx], "w") as sf:
                    json.dump(results, sf, ensure_ascii=False, indent=2)
                save_idx += 1

    with open("Cand.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
