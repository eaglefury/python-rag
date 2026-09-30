def main():
    from pathlib import Path
    from .ingest.ingest_document import ingest_document
    from python_rag.retreive.answer import get_answer
    question = input("what do you want to do ingest/question? : ")

    if question and question.lower() == "ingest":
        document_path = input(
            "What do you want to ingest. give me the path to the document? : ")
        path = Path(document_path)
        if not path.exists():
            print(f"The path {path} does not exist.")
            return
        else:
            ingest_document(str(path))
    else:
        question = input(f"What is your question? :")
        answer = get_answer(question)
        print(f"Answer: {answer}")
