import json
from engine import client

def build_golden_dataset():
    print("1/3: Loading sample_data.txt...")
    with open("sample_data.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("2/3: Prompting Gemini to synthesize test cases...")
    prompt = f"""
    You are an automated evaluation benchmark generator.
    Analyze the following source document and generate 10 distinct, factual Question and Answer pairs based strictly on the text.
    
    Format the output strictly as a JSON array of objects with keys "input" (the question) and "expected_output" (the factual ground truth answer).
    Do not include markdown code block formatting (like ```json), just the raw valid JSON array.

    DOCUMENT:
    {raw_text}
    """

    response = client.interactions.create(
        model="gemini-3.8-flash",
        input=prompt
    )

    # Clean any potential markdown fencing
    raw_response = response.output_text.strip()
    if raw_response.startswith("```"):
        raw_response = raw_response.strip("`")
        if raw_response.startswith("json"):
            raw_response = raw_response[4:].strip()

    print("3/3: Parsing and saving golden_dataset.json...")
    dataset = json.loads(raw_response)

    with open("golden_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)

    print(f"✅ Successfully generated and saved {len(dataset)} synthetic test cases to golden_dataset.json!")


if __name__ == "__main__":
    build_golden_dataset()
