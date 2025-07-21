import json
import math
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from tqdm.auto import tqdm

# 1. Torch‑Dynamo JIT 캐싱 설정
try:
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 256
except Exception:
    pass

# 2. TF32(Matmul) 가속 활성화 (Ampere 계열 GPU용)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

MODEL_NAME = "google/gemma-3-4b-it"
BATCH_SIZE = 4

TEMPLATE = """
Role: You are a conversational AI fulfilling the role of a professional empathetic counselor.
Quickly detect the user’s emotion and provide a natural, meaningful reply with the appropriate tone (empathy, curiosity, or encouragement).

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
    # 데이터 로드
    data_path = Path("train.json")
    with data_path.open(encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    quarter = math.ceil(total / 4)
    save_steps = [quarter, quarter * 2, quarter * 3]
    save_names = [f"Save{i+1}.json" for i in range(3)]

    # 토크나이저와 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
    )
    # 3. 모델을 명시적 단일 GPU에 올리기
    model.to("cuda:0")

    # 3. pipeline도 device=0으로 고정
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0,
        return_full_text=False,
    )

    results = []
    processed = 0
    save_idx = 0

    # 4. 명시적 배치 처리
    for i in tqdm(range(0, total, BATCH_SIZE), desc="Generating"):
        batch = data[i : i + BATCH_SIZE]
        prompts = [TEMPLATE.format(input=item["input"]) for item in batch]

        # 5. max_new_tokens를 40으로 줄여 디코딩 비용 절감
        outputs = gen(
            prompts,
            max_new_tokens=40,
            num_return_sequences=3,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            batch_size=BATCH_SIZE,
            pad_token_id=tokenizer.eos_token_id,
        )

        for item, resp in zip(batch, outputs):
            cands = []
            for r in resp:
                text = r["generated_text"].strip()
                if "Final output:" in text:
                    text = text.split("Final output:", 1)[1].strip()
                if text.startswith("Response:"):
                    text = text[len("Response:"):].strip()
                cands.append(text)
            results.append({"input": item["input"], "candidates": cands})
            processed += 1

            # 중간 저장
            if save_idx < len(save_steps) and processed == save_steps[save_idx]:
                with open(save_names[save_idx], "w", encoding="utf-8") as sf:
                    json.dump(results, sf, ensure_ascii=False, indent=2)
                save_idx += 1

    # 최종 저장
    with open("Cand.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
