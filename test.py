from langchain_ollama import ChatOllama


llm = ChatOllama(
            base_url="http://localhost:11434",
            model="gpt-oss:20b",
            thinking=True,
            temperature=0
        )


print(llm.invoke("Hi"))