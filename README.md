# Python RAG

This repository contains code for reading, chunking, and ingesting documents into a vector store using LangChain and Google Generative AI embeddings.

## Features

- Read documents from various file formats.
- Chunk documents into smaller pieces for efficient processing.
- Ingest documents into a vector store.
- Use LangChain and Google Generative AI embeddings for vector representation.
- Store and retrieve document vectors efficiently using Chroma.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/python-rag.git
   cd python-rag
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your environment variables in a `.env` file, including `GOOGLE_API_KEY`.

## Usage

1. Ingest a document into the vector store:

   ```python
   from python_rag.ingest.ingest_document import ingest_document

   ingest_document("path/to/your/document.pdf")
   ```

2. Retrieve documents from the vector store (example):

   ```python
   from python_rag.vector_store import get_vector_store

   chroma = get_vector_store()
   query = "Your search query here"
   results = chroma.similarity_search(query)
   for result in results:
       print(result)
   ```
