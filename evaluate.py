import json
import chromadb
from engine import chunk_text_by_words, get_embedding, client


def generate_answer(question, retrieved_context):
    prompt = (
        f"Context: {retrieved_context}\n\n"
        f"Answer the question using only the context: {question}"
    )
    interaction = client.interactions.create(
        model="gemini-3.8-flash",
        input=prompt,
    )
    return interaction.output_text.strip()


def check_generation_correctness(question, expected_answer, generated_answer):
    prompt = f"""
    You are an expert evaluator. Compare the generated answer to the expected ground-truth answer.

    Question: {question}
    Expected Answer: {expected_answer}
    Generated Answer: {generated_answer}

    Is the generated answer factually correct and equivalent to the expected answer?
    Answer strictly with YES or NO.
    """
    interaction = client.interactions.create(
        model="gemini-3.8-flash",
        input=prompt,
    )
    return interaction.output_text.strip().upper().startswith("YES")

# --- PHASE 1: Load DeepEval Synthetic Dataset ---
print("Loading synthetic dataset...")
with open("golden_dataset.json", "r", encoding="utf-8") as file:
    raw_dataset = json.load(file)

golden_dataset = []
for item in raw_dataset:
    # DeepEval uses "input" for the question and either spelling for the answer.
    question = item.get("input", "")
    ground_truth = item.get("expected_output", item.get("expectedOutput", ""))

    if question and ground_truth:
        golden_dataset.append({
            "question": question,
            "ground_truth": ground_truth
        })

# --- PHASE 2: Compare Strategies ---
chunk_sizes_to_test = [15, 30, 50]
overlap = 5

with open("sample_data.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()

print(f"Starting Strategy Comparison across {len(golden_dataset)} synthetic questions...\n")

for size in chunk_sizes_to_test:
    temp_client = chromadb.EphemeralClient()
    collection = temp_client.create_collection(name=f"test_collection_{size}")

    chunks = chunk_text_by_words(raw_text, chunk_size=size, overlap=overlap)
    embeddings = [get_embedding(chunk) for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(documents=chunks, embeddings=embeddings, ids=ids)

    hits = 0
    correct_generations = 0
    for test in golden_dataset:
        q_vec = get_embedding(test["question"])
        results = collection.query(query_embeddings=[q_vec], n_results=1)
        retrieved_chunk = results["documents"][0][0]

        # Use a short snippet to account for longer DeepEval answers.
        truth_words = test["ground_truth"].lower().split()[:5]
        truth_snippet = " ".join(truth_words)

        if truth_snippet in retrieved_chunk.lower():
            hits += 1

        generated_answer = generate_answer(test["question"], retrieved_chunk)
        if check_generation_correctness(
            test["question"], test["ground_truth"], generated_answer
        ):
            correct_generations += 1

    hit_rate = (hits / len(golden_dataset)) * 100 if golden_dataset else 0
    generation_rate = (
        (correct_generations / len(golden_dataset)) * 100
        if golden_dataset
        else 0
    )
    print(
        f"Chunk Size {size:>2} | Chunks Created: {len(chunks):>3} | "
        f"Retrieval Hit Rate: {hit_rate:>6.2f}% ({hits}/{len(golden_dataset)}) | "
        f"Generation Correctness: {generation_rate:>6.2f}% "
        f"({correct_generations}/{len(golden_dataset)})"
    )

print("\nEvaluation Complete!")