import json
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

with open("dataFromAIHUB.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)
    
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

# LoRA 설정 : 양자화된 모델에서 Adaptor를 붙여서 학습할 파라미터만 따로 구성함
lora_config = LoraConfig(
    r=8,
    lora_alpha = 8,
    lora_dropout = 0.05,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

# 4bit 양자화 설정 - QLoRA로 해야 함
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 모델 로드 (양자화)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL,
                                             quantization_config=bnb_config,
                                             device_map="auto"
                                            )

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, add_special_tokens=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

# LoRA 적용
model = get_peft_model(model, lora_config)

def generate_prompts(example):
    prompt_list = []
    for i in range(len(example['document'])):
        prompt_list.append(
f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>다음 글을 요약해주세요:
{example['content'][i]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{example['answer'][i]}<|eot_id|>"""
        )
    return prompt_list

# 프롬프트 리스트 생성
prompts = generate_prompts(dataset)

print(prompts[0])

from transformers import Trainer, TrainingArguments

# 학습 설정
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    fp16=True,  # 16-bit로 계산하여 메모리 절약
)

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=prompts,  # 프롬프트 데이터셋 사용
    tokenizer=tokenizer,
)

# 학습 시작
trainer.train()