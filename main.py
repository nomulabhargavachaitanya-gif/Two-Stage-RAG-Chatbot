import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
from engine import client


def run_rag_chatbot():
    # 1. Setup ChromaDB
    chroma_client = chromadb.PersistentClient(path="./my_vectordb")
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = chroma_client.get_or_create_collection(
        name="document_store",
        embedding_function=embedding_func,
    )

    if collection.count() == 0:
        print("Error: Database is empty. Please run evaluate.py or ingestion script first.")
        return

    # 2. Setup the Re-ranker (This downloads a small, specialized scoring model)
    print("Loading Re-ranker model...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # 3. Get User Input
    user_question = input("\nAsk a question about the document: ")
    # 4. Phase 1 Search (The Net: Get top 5 from ChromaDB)
    print("Fetching top 5 candidates from ChromaDB...")
    results = collection.query(
        query_texts=[user_question],
        n_results=5
    )
    candidate_chunks = results["documents"][0]

    # 5. Phase 2 Search (The Scalpel: Re-rank the candidates)
    print("Re-ranking candidates...")

    # Create the pairs: [[question, chunk1], [question, chunk2], ...]
    cross_input = [[user_question, chunk] for chunk in candidate_chunks]

    # Score them all
    scores = reranker.predict(cross_input)

    # Pair the scores with their chunks and sort them from highest to lowest
    scored_chunks = list(zip(scores, candidate_chunks))
    scored_chunks.sort(key=lambda item: item[0], reverse=True)

    # Grab the text of the absolute best chunk
    best_match = scored_chunks[0][1]

    # 6. Generate Answer
    system_prompt = f"""
    Answer the user using ONLY this context. If not found, say "I don't know."
    CONTEXT: {best_match}
    """

    print("Generating final answer...")
    interaction = client.interactions.create(
        model="gemini-3.8-flash",
        input=system_prompt + "\n\nQUESTION: " + user_question,
    )

    print("\n--- AI ANSWER ---")
    print(interaction.output_text)


if __name__ == "__main__":
    run_rag_chatbot()