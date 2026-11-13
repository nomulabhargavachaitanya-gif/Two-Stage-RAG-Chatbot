import os
from dotenv import load_dotenv
from google import genai

# Initialize environment and Gemini client
load_dotenv()
client = genai.Client()


def chunk_text_by_words(text: str, chunk_size: int = 30, overlap: int = 5) -> list[str]:
    """Slices a string into overlapping word chunks."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i : i + chunk_size]))
        if i + chunk_size >= len(words):
            break

    return chunks


def get_embedding(text: str) -> list[float]:
    """Generates a 3072-dimensional vector embedding for the input text."""
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return response.embeddings[0].values