import single_model
import two_llama_model
import bert_llama_model

def run_single():
    import single_model
    single_model.main()

def run_two_llama():
    import two_llama_model
    two_llama_model.main()

def run_bert_llama():
    import bert_llama_model
    bert_llama_model.main()

MENU = {
    "1": ("단일 모델 챗봇 (single_model.py)", run_single),
    "2": ("2모델 분리 챗봇 (two_llama_model.py)", run_two_llama),
    "3": ("BERT+LLM 챗봇 (bert_llama_model.py)", run_bert_llama),
}

def main():
    while True:
        print("\n=== 챗봇 선택 ===")
        for k, (desc, _) in MENU.items():
            print(f"{k}. {desc}")
        print("0. 종료")

        choice = input("선택: ").strip()
        if choice == "0":
            print("종료합니다.")
            return

        item = MENU.get(choice)
        if not item:
            print("잘못된 입력입니다.")
            continue

        desc, fn = item
        print(f"\n--- 실행: {desc} ---\n")

        try:
            fn()
        except KeyboardInterrupt:
            print("\n(CTRL+C) 중단됨. 메뉴로 돌아갑니다.")
        except Exception as e:
            print(f"\n에러: {e}\n메뉴로 돌아갑니다.")

if __name__ == "__main__":
    main()