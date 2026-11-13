import chromadb
from chromadb.utils import embedding_functions


def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping character chunks."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap
    return [text[i:i + chunk_size] for i in range(0, len(text), step)]


def ingest_documents(documents, collection_name="document_store", persist_directory="./my_vectordb"):
    client = chromadb.PersistentClient(path=persist_directory)
    existing_names = {collection.name for collection in client.list_collections()}
    if collection_name in existing_names:
        client.delete_collection(collection_name)

    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_func,
    )

    for doc_id, text in enumerate(documents):
        chunks = chunk_text(text)
        collection.add(
            documents=chunks,
            metadatas=[{"source": f"doc_{doc_id}"} for _ in chunks],
            ids=[f"doc_{doc_id}_chunk_{i}" for i in range(len(chunks))],
        )

    print(f"Successfully ingested {len(documents)} documents into {persist_directory}.")


if __name__ == "__main__":
    with open("sample_data.txt", "r", encoding="utf-8") as file:
        ingest_documents([file.read()])