import json
import random
import os
import chromadb
from sentence_transformers import CrossEncoder
from engine import (
    chunk_text_by_words,
    get_embeddings_batch,
    get_embedding,
    generate_with_backoff_and_cache,
    get_flash_lite_model,
)

# Set random seeds for reproducibility
random.seed(42)

def is_hit(retrieved_chunk: str, test_case: dict) -> bool:
    """Checks whether the retrieved chunk contains the expected answer or supporting evidence."""
    chunk_lower = retrieved_chunk.lower()
    evidence = test_case.get("source_evidence", "").lower().strip()
    expected = test_case.get("expected_output", "").lower().strip()

    # Exact expected answer match
    if expected and expected in chunk_lower:
        return True

    # Check evidence sentence match
    if evidence:
        if evidence in chunk_lower:
            return True
        ev_words = evidence.split()
        if len(ev_words) >= 5:
            # Check sliding window of 5 consecutive words from evidence
            for start in range(len(ev_words) - 4):
                sub = " ".join(ev_words[start : start + 5])
                if sub in chunk_lower:
                    return True
        elif all(w in chunk_lower for w in ev_words if len(w) > 3):
            return True

    return False


def generate_answer(question: str, retrieved_context: str) -> str:
    """Generates an answer using the discovered Flash-Lite model with disk caching."""
    prompt = (
        f"Context:\n{retrieved_context}\n\n"
        f"Answer the question using only the context: {question}"
    )
    return generate_with_backoff_and_cache(prompt)


def check_generation_correctness(question: str, expected_answer: str, generated_answer: str) -> bool:
    """Evaluates factual correctness using Flash-Lite with disk caching."""
    prompt = f"""You are an expert evaluator. Compare the generated answer to the expected ground-truth answer.

Question: {question}
Expected Answer: {expected_answer}
Generated Answer: {generated_answer}

Is the generated answer factually correct and equivalent to the expected answer?
Answer strictly with YES or NO."""
    resp = generate_with_backoff_and_cache(prompt)
    return resp.strip().upper().startswith("YES")


def evaluate_split(
    corpus_text: str,
    test_cases: list[dict],
    chunk_size: int,
    overlap: int,
    stage1_top_k: int,
    stage2_top_k: int,
    reranker: CrossEncoder,
    cached_collections: dict = None,
):
    """Evaluates Stage 1 (ChromaDB) and Stage 2 (Cross-Encoder) retrieval on a set of test cases."""
    cache_key = (chunk_size, overlap)
    if cached_collections is not None and cache_key in cached_collections:
        collection, chunks = cached_collections[cache_key]
    else:
        temp_client = chromadb.EphemeralClient()
        collection = temp_client.create_collection(
            name=f"test_coll_{chunk_size}_{overlap}_{random.randint(1000, 9999)}"
        )
        chunks = chunk_text_by_words(corpus_text, chunk_size=chunk_size, overlap=overlap)
        embeddings = get_embeddings_batch(chunks)
        ids = [f"c_{i}" for i in range(len(chunks))]
        collection.add(documents=chunks, embeddings=embeddings, ids=ids)
        if cached_collections is not None:
            cached_collections[cache_key] = (collection, chunks)

    q_texts = [item["input"] for item in test_cases]
    q_embeddings = get_embeddings_batch(q_texts)

    # Retrieval metrics accumulation
    s1_hit1, s1_hit3, s1_hit5 = 0, 0, 0
    s1_rr_sum = 0.0

    s2_hit1, s2_hit3, s2_hit5 = 0, 0, 0
    s2_rr_sum = 0.0

    retrieval_details = []

    for i, item in enumerate(test_cases):
        q_vec = q_embeddings[i]
        # Query ChromaDB for max(stage1_top_k, 5) to accurately compute hit@1, hit@3, hit@5
        query_k = max(stage1_top_k, 5)
        results = collection.query(query_embeddings=[q_vec], n_results=min(query_k, len(chunks)))
        retrieved_all = results["documents"][0]

        # Stage 1 candidate pool for reranking is stage1_top_k
        stage1_candidates = retrieved_all[:stage1_top_k]

        # Stage 1 metrics
        s1_rank = 0
        for rank, chunk in enumerate(stage1_candidates, 1):
            if is_hit(chunk, item):
                s1_rank = rank
                break

        if s1_rank > 0:
            if s1_rank <= 1:
                s1_hit1 += 1
            if s1_rank <= 3:
                s1_hit3 += 1
            if s1_rank <= 5:
                s1_hit5 += 1
            s1_rr_sum += 1.0 / s1_rank

        # Stage 2: Cross-Encoder Re-ranking of Stage 1 candidates
        cross_pairs = [[item["input"], chunk] for chunk in stage1_candidates]
        scores = reranker.predict(cross_pairs)
        ranked_candidates = [
            chunk for _, chunk in sorted(zip(scores, stage1_candidates), key=lambda x: x[0], reverse=True)
        ]

        # Stage 2 metrics
        s2_rank = 0
        for rank, chunk in enumerate(ranked_candidates, 1):
            if is_hit(chunk, item):
                s2_rank = rank
                break

        if s2_rank > 0:
            if s2_rank <= 1:
                s2_hit1 += 1
            if s2_rank <= 3:
                s2_hit3 += 1
            if s2_rank <= 5:
                s2_hit5 += 1
            s2_rr_sum += 1.0 / s2_rank

        retrieval_details.append({
            "test_case": item,
            "stage1_candidates": stage1_candidates,
            "stage2_candidates": ranked_candidates[:stage2_top_k],
        })

    n = len(test_cases)
    metrics = {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "overlap_pct": round((overlap / chunk_size) * 100, 1),
        "total_chunks": len(chunks),
        "stage1_top_k": stage1_top_k,
        "stage2_top_k": stage2_top_k,
        "n_samples": n,
        "stage1": {
            "hit@1": (s1_hit1 / n) * 100,
            "hit@3": (s1_hit3 / n) * 100,
            "hit@5": (s1_hit5 / n) * 100,
            "mrr": s1_rr_sum / n,
            "counts": {"hit@1": s1_hit1, "hit@3": s1_hit3, "hit@5": s1_hit5},
        },
        "stage2": {
            "hit@1": (s2_hit1 / n) * 100,
            "hit@3": (s2_hit3 / n) * 100,
            "hit@5": (s2_hit5 / n) * 100,
            "mrr": s2_rr_sum / n,
            "counts": {"hit@1": s2_hit1, "hit@3": s2_hit3, "hit@5": s2_hit5},
        },
        "details": retrieval_details,
    }
    return metrics


def run_generation_evaluation(retrieval_results: list[dict]) -> tuple[float, int, int]:
    """Generates answers and checks factual correctness against ground truth using Flash-Lite."""
    correct = 0
    total = len(retrieval_results)

    for item in retrieval_results:
        test_case = item["test_case"]
        top_chunks = item["stage2_candidates"]
        context_str = "\n\n".join(top_chunks)

        gen_answer = generate_answer(test_case["input"], context_str)
        is_ok = check_generation_correctness(
            test_case["input"],
            test_case["expected_output"],
            gen_answer
        )
        if is_ok:
            correct += 1

    accuracy = (correct / total) * 100 if total > 0 else 0.0
    return accuracy, correct, total


def main():
    model_name = get_flash_lite_model()
    print("=" * 70)
    print(f"Two-Stage RAG Chatbot Comprehensive Evaluation")
    print(f"Flash-Lite Generative Model: {model_name}")
    print("=" * 70)

    # 1. Load dataset & documents
    with open("golden_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    with open("sample_data.txt", "r", encoding="utf-8") as f:
        corpus_text = f.read()

    print(f"Total benchmark questions in golden_dataset.json: {len(dataset)}")
    print(f"Corpus total words: {len(corpus_text.split())}")

    # 2. Split 50/50 with seed 42
    rng = random.Random(42)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    split_idx = len(dataset) // 2
    dev_set = [dataset[i] for i in indices[:split_idx]]
    test_set = [dataset[i] for i in indices[split_idx:]]

    print(f"Dev set size: {len(dev_set)}, Test set size: {len(test_set)}")

    # 3. Load Cross-Encoder reranker
    print("Loading Cross-Encoder model (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # 4. Hyperparameter tuning on DEV set
    print("\n" + "=" * 70)
    print("PHASE 1: Hyperparameter Tuning on Dev Split (Seed=42)")
    print("=" * 70)

    chunk_configs = [
        # (size, overlap, description)
        (30, 5, "Baseline (16.7% overlap)"),
        (30, 3, "10% overlap"),
        (30, 6, "20% overlap"),
        (50, 5, "10% overlap"),
        (50, 10, "20% overlap"),
        (80, 8, "10% overlap"),
        (80, 16, "20% overlap"),
    ]

    s1_k_options = [5, 10, 20]
    s2_k_options = [1, 3]

    cached_collections = {}
    tuning_records = []

    for size, overlap, desc in chunk_configs:
        for s1_k in s1_k_options:
            for s2_k in s2_k_options:
                res = evaluate_split(
                    corpus_text=corpus_text,
                    test_cases=dev_set,
                    chunk_size=size,
                    overlap=overlap,
                    stage1_top_k=s1_k,
                    stage2_top_k=s2_k,
                    reranker=reranker,
                    cached_collections=cached_collections,
                )
                tuning_records.append(res)
                print(
                    f"Config: Size={size:>2}, Overlap={overlap:>2} ({desc:<22}) | "
                    f"S1 top-k={s1_k:>2}, S2 top-k={s2_k} | Chunks={res['total_chunks']:>2} | "
                    f"S1 Hit@1: {res['stage1']['hit@1']:>5.1f}% MRR: {res['stage1']['mrr']:>5.3f} | "
                    f"S2 Hit@1: {res['stage2']['hit@1']:>5.1f}% MRR: {res['stage2']['mrr']:>5.3f}"
                )

    # Find Baseline on Dev
    baseline_dev = next(
        r for r in tuning_records
        if r["chunk_size"] == 30 and r["overlap"] == 5 and r["stage1_top_k"] == 5 and r["stage2_top_k"] == 1
    )

    # Select Best Config on Dev: maximize Stage 2 MRR, then Stage 2 Hit@1, then Stage 1 MRR
    best_dev = max(
        tuning_records,
        key=lambda r: (r["stage2"]["mrr"], r["stage2"]["hit@1"], r["stage1"]["mrr"], r["stage1"]["hit@1"])
    )

    print("\n" + "-" * 70)
    print("DEV TUNING SUMMARY:")
    print(
        f"Baseline Config (Size 30, Ov 5, S1 k=5, S2 k=1):\n"
        f"  Stage 1: Hit@1={baseline_dev['stage1']['hit@1']:.1f}%, Hit@3={baseline_dev['stage1']['hit@3']:.1f}%, Hit@5={baseline_dev['stage1']['hit@5']:.1f}%, MRR={baseline_dev['stage1']['mrr']:.4f}\n"
        f"  Stage 2: Hit@1={baseline_dev['stage2']['hit@1']:.1f}%, Hit@3={baseline_dev['stage2']['hit@3']:.1f}%, Hit@5={baseline_dev['stage2']['hit@5']:.1f}%, MRR={baseline_dev['stage2']['mrr']:.4f}"
    )
    print(
        f"Best Config on Dev (Size {best_dev['chunk_size']}, Ov {best_dev['overlap']}, S1 k={best_dev['stage1_top_k']}, S2 k={best_dev['stage2_top_k']}):\n"
        f"  Stage 1: Hit@1={best_dev['stage1']['hit@1']:.1f}%, Hit@3={best_dev['stage1']['hit@3']:.1f}%, Hit@5={best_dev['stage1']['hit@5']:.1f}%, MRR={best_dev['stage1']['mrr']:.4f}\n"
        f"  Stage 2: Hit@1={best_dev['stage2']['hit@1']:.1f}%, Hit@3={best_dev['stage2']['hit@3']:.1f}%, Hit@5={best_dev['stage2']['hit@5']:.1f}%, MRR={best_dev['stage2']['mrr']:.4f}"
    )

    # 5. FINAL EVALUATION ON TEST SET
    print("\n" + "=" * 70)
    print("PHASE 2: Final Evaluation on Test Split (Seed=42)")
    print("=" * 70)

    # Evaluate Baseline on Test
    test_baseline = evaluate_split(
        corpus_text=corpus_text,
        test_cases=test_set,
        chunk_size=30,
        overlap=5,
        stage1_top_k=5,
        stage2_top_k=1,
        reranker=reranker,
        cached_collections=cached_collections,
    )
    print("Running Generation Correctness on Test Split (Baseline)...")
    base_acc, base_corr, base_tot = run_generation_evaluation(test_baseline["details"])

    # Evaluate Best Config on Test
    test_best = evaluate_split(
        corpus_text=corpus_text,
        test_cases=test_set,
        chunk_size=best_dev["chunk_size"],
        overlap=best_dev["overlap"],
        stage1_top_k=best_dev["stage1_top_k"],
        stage2_top_k=best_dev["stage2_top_k"],
        reranker=reranker,
        cached_collections=cached_collections,
    )
    print("Running Generation Correctness on Test Split (Best Config)...")
    best_acc, best_corr, best_tot = run_generation_evaluation(test_best["details"])

    print("\n" + "=" * 70)
    print("FINAL TEST SPLIT RESULTS:")
    print("=" * 70)
    print(f"Baseline (Size 30, Overlap 5, S1 k=5, S2 k=1):")
    print(f"  Stage 1 (ChromaDB):      Hit@1: {test_baseline['stage1']['hit@1']:>5.1f}% | Hit@3: {test_baseline['stage1']['hit@3']:>5.1f}% | Hit@5: {test_baseline['stage1']['hit@5']:>5.1f}% | MRR: {test_baseline['stage1']['mrr']:.4f}")
    print(f"  Stage 2 (Cross-Encoder): Hit@1: {test_baseline['stage2']['hit@1']:>5.1f}% | Hit@3: {test_baseline['stage2']['hit@3']:>5.1f}% | Hit@5: {test_baseline['stage2']['hit@5']:>5.1f}% | MRR: {test_baseline['stage2']['mrr']:.4f}")
    print(f"  Generation Correctness:  {base_acc:>5.1f}% ({base_corr}/{base_tot})")

    print(f"\nBest Config (Size {best_dev['chunk_size']}, Overlap {best_dev['overlap']}, S1 k={best_dev['stage1_top_k']}, S2 k={best_dev['stage2_top_k']}):")
    print(f"  Stage 1 (ChromaDB):      Hit@1: {test_best['stage1']['hit@1']:>5.1f}% | Hit@3: {test_best['stage1']['hit@3']:>5.1f}% | Hit@5: {test_best['stage1']['hit@5']:>5.1f}% | MRR: {test_best['stage1']['mrr']:.4f}")
    print(f"  Stage 2 (Cross-Encoder): Hit@1: {test_best['stage2']['hit@1']:>5.1f}% | Hit@3: {test_best['stage2']['hit@3']:>5.1f}% | Hit@5: {test_best['stage2']['hit@5']:>5.1f}% | MRR: {test_best['stage2']['mrr']:.4f}")
    print(f"  Generation Correctness:  {best_acc:>5.1f}% ({best_corr}/{best_tot})")

    # Save complete evaluation output dictionary as JSON for transparent logging
    results_payload = {
        "model_generative": model_name,
        "model_embedding": "gemini-embedding-001",
        "model_reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "corpus_words": len(corpus_text.split()),
        "total_questions": len(dataset),
        "dev_questions": len(dev_set),
        "test_questions": len(test_set),
        "tuning_dev_records": [
            {
                "chunk_size": r["chunk_size"],
                "overlap": r["overlap"],
                "overlap_pct": r["overlap_pct"],
                "stage1_top_k": r["stage1_top_k"],
                "stage2_top_k": r["stage2_top_k"],
                "total_chunks": r["total_chunks"],
                "stage1_hit1": r["stage1"]["hit@1"],
                "stage1_hit3": r["stage1"]["hit@3"],
                "stage1_hit5": r["stage1"]["hit@5"],
                "stage1_mrr": r["stage1"]["mrr"],
                "stage2_hit1": r["stage2"]["hit@1"],
                "stage2_hit3": r["stage2"]["hit@3"],
                "stage2_hit5": r["stage2"]["hit@5"],
                "stage2_mrr": r["stage2"]["mrr"],
            }
            for r in tuning_records
        ],
        "dev_baseline": {
            "stage1": baseline_dev["stage1"],
            "stage2": baseline_dev["stage2"],
        },
        "dev_best": {
            "config": {
                "chunk_size": best_dev["chunk_size"],
                "overlap": best_dev["overlap"],
                "stage1_top_k": best_dev["stage1_top_k"],
                "stage2_top_k": best_dev["stage2_top_k"],
            },
            "stage1": best_dev["stage1"],
            "stage2": best_dev["stage2"],
        },
        "test_baseline": {
            "stage1": test_baseline["stage1"],
            "stage2": test_baseline["stage2"],
            "generation_correctness": {
                "accuracy": base_acc,
                "correct": base_corr,
                "total": base_tot,
            },
        },
        "test_best": {
            "config": {
                "chunk_size": best_dev["chunk_size"],
                "overlap": best_dev["overlap"],
                "stage1_top_k": best_dev["stage1_top_k"],
                "stage2_top_k": best_dev["stage2_top_k"],
            },
            "stage1": test_best["stage1"],
            "stage2": test_best["stage2"],
            "generation_correctness": {
                "accuracy": best_acc,
                "correct": best_corr,
                "total": best_tot,
            },
        },
    }

    with open("eval_results_summary.json", "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print("\nSaved eval_results_summary.json successfully.")


if __name__ == "__main__":
    main()
