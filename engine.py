import os
import json
import time
from dotenv import load_dotenv
from google import genai

# Initialize environment and Gemini client
load_dotenv()
client = genai.Client()

_FLASH_LITE_MODEL = None
CACHE_FILE = os.path.join(os.path.dirname(__file__), "prompt_cache.json")
EMBEDDING_CACHE_FILE = os.path.join(os.path.dirname(__file__), ".embedding_cache.json")


def get_flash_lite_model() -> str:
    """Discovers available flash-lite models from client.models.list() without guessing."""
    global _FLASH_LITE_MODEL
    if _FLASH_LITE_MODEL is not None:
        return _FLASH_LITE_MODEL

    models = [m.name for m in client.models.list()]
    # Filter for flash-lite text generation models
    candidates = [
        m.replace("models/", "")
        for m in models
        if "flash-lite" in m and "image" not in m
    ]

    # Pick the latest available candidate
    for preferred in ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-2.5-flash-lite"]:
        if preferred in candidates:
            _FLASH_LITE_MODEL = preferred
            return _FLASH_LITE_MODEL

    if candidates:
        _FLASH_LITE_MODEL = candidates[0]
        return _FLASH_LITE_MODEL

    raise RuntimeError("No flash-lite model found in client.models.list()")


def _load_prompt_cache() -> dict[str, str]:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_prompt_cache(cache: dict[str, str]):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Failed to save prompt cache: {e}")


def generate_with_backoff_and_cache(
    prompt: str,
    model: str = None,
    max_retries: int = 8,
    base_delay: float = 2.0
) -> str:
    """Generates a text completion with exponential backoff on HTTP 429 and disk caching keyed by prompt."""
    if model is None:
        model = get_flash_lite_model()

    cache = _load_prompt_cache()
    if prompt in cache:
        return cache[prompt]

    delay = base_delay
    last_err = None
    for attempt in range(max_retries):
        try:
            interaction = client.interactions.create(
                model=model,
                input=prompt,
            )
            output = interaction.output_text.strip()
            cache[prompt] = output
            _save_prompt_cache(cache)
            return output
        except Exception as e:
            err_msg = str(e).lower()
            last_err = e
            if "429" in err_msg or "rate limit" in err_msg or "quota" in err_msg or "resourceexhausted" in err_msg or "503" in err_msg:
                print(f"[Retry {attempt+1}/{max_retries}] Rate limit / server error ({e}). Sleeping {delay:.1f}s...")
                time.sleep(delay)
                delay *= 2.0
            else:
                raise e

    raise RuntimeError(f"Exceeded max retries for model {model}: {last_err}")


# Alias for convenience
generate_text = generate_with_backoff_and_cache


def _load_embedding_cache() -> dict[str, list[float]]:
    if os.path.exists(EMBEDDING_CACHE_FILE):
        try:
            with open(EMBEDDING_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_embedding_cache(cache: dict[str, list[float]]):
    try:
        with open(EMBEDDING_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception as e:
        print(f"Warning: Failed to save embedding cache: {e}")


def get_embedding(text: str) -> list[float]:
    """Generates a vector embedding for the input text using gemini-embedding-001 with disk caching."""
    cache = _load_embedding_cache()
    if text in cache:
        return cache[text]

    delay = 2.0
    for attempt in range(6):
        try:
            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=text
            )
            vec = response.embeddings[0].values
            cache[text] = vec
            _save_embedding_cache(cache)
            return vec
        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "quota" in err_msg or "503" in err_msg:
                time.sleep(delay)
                delay *= 2.0
            else:
                raise e

    raise RuntimeError("Failed to compute embedding after retries.")


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Batch computes embeddings for multiple texts with disk caching."""
    cache = _load_embedding_cache()
    missing_texts = [t for t in texts if t not in cache]

    if missing_texts:
        # Embed in chunks of up to 20 to avoid payload limits
        batch_size = 20
        for i in range(0, len(missing_texts), batch_size):
            batch = missing_texts[i : i + batch_size]
            delay = 2.0
            for attempt in range(6):
                try:
                    response = client.models.embed_content(
                        model="gemini-embedding-001",
                        contents=batch
                    )
                    for text, emb in zip(batch, response.embeddings):
                        cache[text] = emb.values
                    break
                except Exception as e:
                    err_msg = str(e).lower()
                    if "429" in err_msg or "quota" in err_msg or "503" in err_msg:
                        time.sleep(delay)
                        delay *= 2.0
                    else:
                        raise e
        _save_embedding_cache(cache)

    return [cache[t] for t in texts]


def chunk_text_by_words(text: str, chunk_size: int = 30, overlap: int = 5) -> list[str]:
    """Slices a string into overlapping word chunks."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("overlap must be less than chunk_size")

    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i : i + chunk_size]))
        if i + chunk_size >= len(words):
            break

    return chunks
