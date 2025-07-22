import json
import math
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from tqdm.auto import tqdm

# GPU 최적화 설정
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

MODEL_NAME = "google/gemma-3-4b-it"
BATCH_SIZE = 32
NUM_CANDIDATES = 3
MAX_NEW_TOKENS = 40
SAMPLING_PARAMS = {
    "do_sample": True,
    "top_p": 0.9,
    "temperature": 0.7,
    "pad_token_id": None  # to be set after tokenizer
}

# 프롬프트: 감정 및 의도를 반영한 공감 응답 생성 지시
PROMPT_TEMPLATE = (
    "System: You are an experienced, professional empathetic counselor."
    " When a user message is provided, first identify the user's primary emotion and intent,"  
    " then generate three distinct, concise (1–2 sentences) empathetic responses that directly address the user's intent and reflect their emotion."
    " Avoid generic phrases or clichés; if posing a question, begin with 'Why', 'How', or 'What'."
    " Only output the responses without any additional labels."
    "\nUser: {input}\n"
    "Assistant:"
)

def clean_response(raw: str) -> str:
    """
    'Assistant:' 이후 텍스트만 추출하여, 프롬프트 잔여물을 제거합니다.
    """
    raw = raw.strip()
    if "Assistant:" in raw:
        return raw.split("Assistant:", 1)[1].strip()
    return raw


def generate_candidates(gen, prompt: str, missing: int) -> list[str]:
    """
    주어진 프롬프트로 부족한 후보 수 만큼 추가 생성합니다.
    """
    extra_outputs = gen(
        [prompt] * missing,
        max_new_tokens=MAX_NEW_TOKENS,
        num_return_sequences=missing,
        **SAMPLING_PARAMS
    )
    return [clean_response(out.get("generated_text", "")) for out in extra_outputs]


def main():
    # train.json 로드 (gold-standard inputs)
    data = json.load(Path("train.json").open(encoding="utf-8"))
    total = len(data)
    
    # 중간 저장 포인트
    quarter = math.ceil(total / 4)
    save_steps = [quarter, quarter * 2, quarter * 3]
    save_names = [f"Save{i+1}.json" for i in range(3)]
    
    # 토크나이저 & 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
    ).to("cuda:0")
    SAMPLING_PARAMS["pad_token_id"] = tokenizer.eos_token_id
    
    # 텍스트 생성 파이프라인 설정
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

    # 배치 단위 프롬프트 생성 및 후보 추출
    for i in tqdm(range(0, total, BATCH_SIZE), desc="Generating"):
        batch = data[i : i + BATCH_SIZE]
        prompts = [PROMPT_TEMPLATE.format(input=item["input"]) for item in batch]
        
        # 초기 후보 생성
        outputs = gen(
            prompts,
            max_new_tokens=MAX_NEW_TOKENS,
            num_return_sequences=NUM_CANDIDATES,
            **SAMPLING_PARAMS
        )
        
        # 각 입력(prompt)별로 후보 리스트 분리 및 정제
        for item, cand_group in zip(batch, outputs):
            texts = [clean_response(cand.get("generated_text", "")) for cand in cand_group]

            # 부족한 후보에 대해 추가 생성
            if len(texts) < NUM_CANDIDATES:
                missing = NUM_CANDIDATES - len(texts)
                texts.extend(generate_candidates(gen, PROMPT_TEMPLATE.format(input=item["input"]), missing))

            # 과도한 경우 잘라냄
            texts = texts[:NUM_CANDIDATES]

            results.append({
                "input": item["input"],
                "candidates": texts
            })
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
