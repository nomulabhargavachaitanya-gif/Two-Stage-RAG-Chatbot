import os
import json
import glob
from engine import generate_with_backoff_and_cache, get_flash_lite_model


def build_golden_dataset():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    doc_files = sorted(glob.glob(os.path.join(data_dir, "doc*.txt")))

    if not doc_files:
        raise FileNotFoundError(f"No document files found in {data_dir}")

    model_name = get_flash_lite_model()
    print(f"Discovered model for benchmark generation: {model_name}")
    print(f"Found {len(doc_files)} source documents in {data_dir}...")

    all_test_cases = []

    for idx, doc_path in enumerate(doc_files, 1):
        doc_name = os.path.basename(doc_path)
        with open(doc_path, "r", encoding="utf-8") as f:
            doc_text = f.read().strip()

        print(f"Generating 10 factual benchmark Q&A pairs for {doc_name} ({idx}/{len(doc_files)})...")

        prompt = f"""
You are an expert automated evaluation benchmark generator for a RAG system.
Analyze the following source document and generate exactly 10 distinct, factual Question and Answer pairs based strictly on the text.

Rules:
1. Every question must be fully answerable using ONLY the provided document.
2. The expected answer must be concise, factual, and strictly verifiable against the text.
3. For each pair, include 'source_evidence' which is the exact verbatim sentence from the document containing the answer.
4. Do not include markdown code block fencing (like ```json), just the raw valid JSON array.

Format the output strictly as a JSON array of objects with keys:
- "input": the question string
- "expected_output": the factual ground truth answer string
- "source_evidence": the exact sentence from the text verifying the answer

DOCUMENT:
{doc_text}
"""
        response_text = generate_with_backoff_and_cache(prompt, model=model_name)

        # Clean potential markdown fencing
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"JSON decode failed for {doc_name}: {e}. Retrying with strict parsing...")
            # Fallback retry without cache if malformed
            raise e

        for item in parsed:
            item["document_id"] = f"doc_{idx}"
            item["document_file"] = doc_name
            all_test_cases.append(item)

        print(f"  -> Generated {len(parsed)} questions for {doc_name}")

    out_file = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_test_cases, f, indent=4, ensure_ascii=False)

    print(f"\n Successfully generated and saved {len(all_test_cases)} test cases to {out_file}!")


if __name__ == "__main__":
    build_golden_dataset()
