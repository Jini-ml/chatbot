from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def main():
    llm = ChatOllama(
        model="bllossom-tax",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """
        너는 한국어 세무사 AI 챗봇이다.
        
        규칙:
        1) 세무(부가세, 종합소득세, 법인세, 원천세, 연말정산, 사업자등록, 세금계산서, 현금영수증 등) 관련 질문에만 답변한다.
        2) 세무와 무관한 질문에는 절대 답변하지 않는다.
        3) 세무와 무관한 질문이면 아래 문장만 출력한다.
        세무 관련 질문에만 답변할 수 있습니다.
        4) 이모티콘과 불필요한 특수기호를 사용하지 않는다.
        5) 확정적인 신고 및 법률 판단은 세무사 또는 전문가 확인이 필요하다고 안내한다.
        
        """),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])

    chain = prompt | llm

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    print("세무사 AI Chatbot")
    print("종료하려면 '종료' 또는 '끝' 입력")
    print()

    session_id = "user-1"

    while True:
        user_input = input("상담자: ").strip()

        if not user_input:
            print("AI 세무사: 질문을 입력해주세요.\n")
            continue

        if user_input.lower() in ["종료", "끝"]:
            print("상담을 종료합니다.")
            break
   
        response = chain_with_history.invoke(
            {"question": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        print("AI 세무사:", response.content)
        print()



if __name__ == "__main__":
    main()
