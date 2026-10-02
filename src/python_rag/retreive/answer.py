from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pathlib import Path

from python_rag.vector_store import ENV_FILE

# ChatOpenAI reads OPENAI_API_KEY from the environment.
load_dotenv(ENV_FILE)

# Chat model that writes the answer from the retrieved chunks.
CHAT_MODEL = "gpt-5-mini"


def get_answer(question: str) -> str:
    answer, documents = answer_with_context(question)
    print(f"Retrieved {len(documents)} documents for the question.")

    for document in documents:
        print("-----")
        print(f"Document content: {document.page_content}")
        print(f"Document metadata: {document.metadata}")
        print("-----")

    return answer


# Returns the retrieved chunks alongside the answer so evals can judge both
# retrieval quality and whether the answer is grounded in those chunks.
def answer_with_context(question: str, top_k: int = 5) -> tuple[str, list[Document]]:
    documents = get_rag_documents(question, top_k=top_k)
    chat = ChatOpenAI(model=CHAT_MODEL)
    response = chat.invoke(build_prompt(question, documents))
    return response.text, documents


def build_prompt(question: str, documents: list[Document]) -> str:
    context = "\n\n".join(document.page_content for document in documents)
    return (
        "Answer the question using the context below. If the context does not "
        "contain the answer, say you do not know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )


def get_rag_documents(query: str, top_k: int = 5) -> list[Document]:
    from python_rag.retreive.retreive_chunks import get_documents
    return get_documents(query, top_k=top_k)
