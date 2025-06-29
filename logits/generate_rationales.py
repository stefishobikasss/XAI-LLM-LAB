import json
import requests
import os

HUGGINGFACE_API_KEY = os.getenv("HF_TOKEN") or input("Enter your HuggingFace API key: ")
HF_MODEL = "HuggingFaceH4/zephyr-7b-beta"

def generate_rationale_with_phi(token, context):
    prompt = f"What does the token '{token}' mean in the context: \"{context}\"?"
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}"
    }
    json_data = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 50}
    }

    response = requests.post(
        f"https://api-inference.huggingface.co/models/{HF_MODEL}",
        headers=headers,
        json=json_data
    )

    if response.status_code == 200:
        result = response.json()
        return result[0]["generated_text"].replace(prompt, "").strip()
    else:
        return f"[Error: {response.status_code}]"

def create_explanation(auto_mode=True):
    prompt = input("\nEnter your prompt: ").strip()
    tokens = prompt.split()
    rationale = []

    if auto_mode:
        print("\n🧠 Generating rationales using Phi-mini...")
        for token in tokens:
            explanation = generate_rationale_with_phi(token, prompt)
            print(f"  {token}: {explanation}")
            rationale.append(explanation)
    else:
        print("\n✍️ Enter rationale for each token:")
        for token in tokens:
            explanation = input(f"  {token}: ")
            rationale.append(explanation)

    data = {
        "prompt": prompt,
        "tokens": tokens,
        "rationale": rationale
    }

    with open("rationales.jsonl", "a") as f:
        f.write(json.dumps(data) + "\n")

    print("\n✅ Saved to rationales.jsonl!\n")

# Main loop
counter = 1
while True:
    mode = input("\nUse Phi-mini for rationale generation? (y/n): ").lower()
    auto = mode == 'y'
    create_explanation(auto_mode=auto)
    again = input("Add another? (y/n): ")
    if again.lower() != 'y':
        break