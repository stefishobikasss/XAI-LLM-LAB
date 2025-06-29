# generate_rationales.py

import streamlit as st
import requests
import json

HF_MODEL = "microsoft/phi-2"

# Ask for Hugging Face API key in the sidebar
with st.sidebar:
    api_key = st.text_input("Hugging Face API Key", type="password")

st.title("🧠 Auto Token Rationale Generator (Phi-mini)")

prompt = st.text_area("Enter a sentence to analyze:", height=100)
auto_mode = st.checkbox("Generate rationale automatically with Phi-mini", value=True)

if st.button("Generate Rationales"):
    if not api_key:
        st.error("❌ Please enter your Hugging Face API key in the sidebar.")
    elif not prompt.strip():
        st.warning("⚠️ Enter a sentence first.")
    else:
        tokens = prompt.strip().split()
        rationales = []

        with st.spinner("Generating rationales..."):
            for token in tokens:
                if auto_mode:
                    response = requests.post(
                        f"https://api-inference.huggingface.co/models/{HF_MODEL}",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json={
                            "inputs": f"What does the token '{token}' mean in the context: \"{prompt}\"?",
                            "parameters": {"max_new_tokens": 60}
                        }
                    )
                    if response.status_code == 200:
                        text = response.json()[0]["generated_text"]
                        rationale = text.split("context:")[-1].strip()
                    else:
                        rationale = f"[Error {response.status_code}]"
                else:
                    rationale = st.text_input(f"✍️ Enter rationale for `{token}`:", key=token)
                rationales.append({"token": token, "rationale": rationale})

        st.subheader("💡 Rationales")
        for r in rationales:
            st.markdown(f"**{r['token']}** → {r['rationale']}")

        # Save to rationales.jsonl
        with open("rationales.jsonl", "a") as f:
            f.write(json.dumps({
                "prompt": prompt,
                "tokens": [r["token"] for r in rationales],
                "rationale": [r["rationale"] for r in rationales]
            }) + "\n")

        st.success("✅ Rationales saved to `rationales.jsonl`")
