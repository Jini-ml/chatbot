import csv
from pathlib import Path
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def load_questions(path: str) -> list[str]:
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(path)
    
    with file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        questions = [row["question"] for row in reader]

    return questions

def save_classification_results(results: list[dict], path: str) -> None:
    file = Path(path)
    with file.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question", "classification"]
        )
        writer.writeheader()
        
        for row in results:
            writer.writerow({
                "question": row["question"],
                "classification": row["classification"]
            })


def main():
    llm = OllamaLLM(
        model="llama3:8b-instruct-q4_K_M",
        temperature=0
    )
    prompt = PromptTemplate.from_template(
    """
    너는 질문 분류 전용 AI다.

    아래 질문이 세금, 신고, 납부, 환급, 소득세, 법인세, 부가가치세, 증여세, 상속세, 세무조사 등과 
    직접적으로 관련되어 있으면 TAX, 
    그렇지 않고 그 외의 모든 경우는 NON_TAX 로만 답해라.

    규칙:
    - 반드시 TAX 또는 NON_TAX 중 하나만 출력
    - 다른 설명, 문장, 기호, 이모티콘 절대 출력 금지

    질문: {question}
    출력:
    """
    )
    chain = prompt | llm | StrOutputParser()
    print("도메인 분류기")
    
    questions = load_questions("modules/taxQA_GPT.csv")
    
    results = []
    
    for q in questions:
        result = chain.invoke({'question': q})
        print(f"질문: {q}\n분류기: {result}\n")
        
        if result not in ("TAX", "NON_TAX"):
            result = "UNKNOWN"

        results.append({
            "question": q,
            "classification": result
        })
        
    output_path = "modules/domain_classification_results.csv"
    save_classification_results(results, output_path)


    print(f"분류 결과가 {output_path}에 저장되었습니다.")
    print("도메인 분류기 종료")

if __name__ == "__main__":
    main()