import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig
from datasets import Dataset
from peft import LoraConfig, get_peft_model
import json

# 환경 변수 설정 (병렬 처리 경고 비활성화)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 데이터 로드 함수
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    def generate_prompts(example):
        prompt_list = []
        for key, example in dataset.items():
            prompt_list.append(
                f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>다음 글을 요약해주세요:
{example['content']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{example['answer']}<|eot_id|>"""
            )
        return prompt_list

    prompts = generate_prompts(dataset)
    prompts_dataset = Dataset.from_list([{"input_text": prompt} for prompt in prompts])
    return prompts_dataset

# 모델 설정 함수 (PyTorch 양자화 + LoRA 사용)
def setup_model():
    BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
    
    # 토크나이저 로드
    # config = BitsAndBytesConfig(load_in_4bit = True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, device_map="cuda", token=os.environ['TOKEN_2'])
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})  # pad_token 설정

    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="cuda",  # 두 번째 GPU로 할당
        # quantization_config=config,
        token=os.environ['TOKEN_2']
    )

    # PEFT 설정: LoRA 어댑터 추가
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # LoRA 어댑터를 모델에 적용
    model = get_peft_model(model, lora_config)

    return model, tokenizer

# 데이터셋 토크나이즈 함수
def tokenize_function(examples, tokenizer):
    tokenized = tokenizer(
        examples["input_text"], padding="max_length", truncation=True, max_length=512, 
    )
    tokenized["labels"] = tokenized["input_ids"]  # labels 필드를 input_ids와 동일하게 설정
    return tokenized

# 학습 함수
def train_model():
    # 데이터 로드
    prompts_dataset = load_data("dataFromAIHUB.json")

    # 모델과 토크나이저 준비
    model, tokenizer = setup_model()

    # 데이터셋 토큰화
    tokenized_prompts = prompts_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # 데이터셋 텐서 형식으로 설정
    tokenized_prompts.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # DataCollator 정의
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 학습 설정 (FP16을 활용한 메모리 최적화)
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1,  # 배치 크기 조정
        num_train_epochs=3,
        logging_steps=10,
        fp16=True,  # Mixed precision training (FP16)
        save_steps = 500,
        save_strategy="steps",
        gradient_accumulation_steps=8,  # 배치 크기 누적
    )

    # Trainer 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_prompts,
        data_collator=data_collator  # DataCollatorForLanguageModeling 사용
    )

    # 학습 전에 GPU 메모리 비우기
    torch.cuda.empty_cache()

    # 학습 시작
    print(f"Model is on device: {model.device}")
    print("Starting training...")
    # trainer.train()
    trainer.train(resume_from_checkpoint="./results/checkpoint-18000")
    print("Training completed!")

if __name__ == "__main__":
    train_model()
