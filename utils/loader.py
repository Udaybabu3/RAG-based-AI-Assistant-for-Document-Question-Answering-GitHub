from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

def load_pdf_documents(pdf_paths: List[str]) -> List[Document]:
    documents = []

    for path in pdf_paths:
        if not Path(path).exists():
            print(f"File not found: {path}")
            continue

        loader = PyPDFLoader(path)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = path

        documents.extend(docs)

    return documents