from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_path):
    # 저장된 모델과 토크나이저 로드
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def summarize_text(text, model, tokenizer):
    # 텍스트를 토큰화
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    
    # 모델을 이용해 요약 생성
    summary_ids = model.generate(inputs['input_ids'], max_length=150, num_beams=4, early_stopping=True)
    
    # 요약된 텍스트 디코딩
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def main():
    # 저장된 모델과 토크나이저 로드
    model, tokenizer = load_model("./trained_model")

    # 예시 텍스트
    text = "여기에 요약할 텍스트를 입력하세요."

    # 요약 생성
    summary = summarize_text(text, model, tokenizer)
    print("Generated Summary:", summary)

if __name__ == "__main__":
    main()
