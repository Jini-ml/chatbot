from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def chatbot():
    llm = Ollama(
        model="llama3",   # LLaMA 3 8B
        temperature=0.7
    )
    prompt = PromptTemplate.from_template(
    """
    너는 친절한 한국어 세무사 AI 챗봇이야.
    질문에 대해 이모티콘과 특수 기호 없이 질문에 정확하게 답해줘.

    질문: {question}
    답변:
    """
    )
    chain = prompt | llm | StrOutputParser()
    
    print("세무사 AI Chatbot")

    while True:
        user_input = input("상담자: ")
        if user_input.lower() in ["종료", "끝"]:
            print("상담을 종료합니다.")
            break

        response = chain.invoke({'question': user_input})
        print("AI 세무사:", response)
        print()

if __name__ == "__main__":
    chatbot()