from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Chat model that writes the answer from the retrieved chunks.
CHAT_MODEL = "gpt-5-mini"


def get_answer(question: str) -> str:
    documents = get_rag_documents(question)
    print(f"Retrieved {len(documents)} documents for the question.")

    for document in documents:
        print("-----")
        print(f"Document content: {document.page_content}")
        print(f"Document metadata: {document.metadata}")
        print("-----")

    context = "\n\n".join(document.page_content for document in documents)
    prompt = (
        "Answer the question using the context below. If the context does not "
        "contain the answer, say you do not know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )

    chat = ChatOpenAI(model=CHAT_MODEL)
    response = chat.invoke(prompt)
    return response.text


def get_rag_documents(query: str, top_k: int = 5) -> list[Document]:
    from python_rag.retreive.retreive_chunks import get_documents
    return get_documents(query, top_k=top_k)
