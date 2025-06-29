import streamlit as st
import pandas as pd
import plotly.express as px
import json
import numpy as np

st.set_page_config(page_title="LLM Neuron Explainer", layout="wide")
st.title("🧠 LLM Neuron Attribution Dashboard")

# --- Prompt Input ---
prompt = st.text_input("Enter your prompt:")
submit = st.button("Submit")

if submit and prompt:
    st.subheader("🎯 Token Attention Graph")

    try:
        # Load activation data in chunks
        chunk_iter = pd.read_csv("logits_activations.csv", chunksize=1000)
        tokens = prompt.strip().split()
        filtered_chunks = []

        for chunk in chunk_iter:
            if "token" in chunk.columns:
                match = chunk[chunk["token"].isin(tokens)]
                if not match.empty:
                    filtered_chunks.append(match)
            else:
                filtered_chunks.append(chunk.iloc[:50])  # fallback if no token col

        if filtered_chunks:
            filtered_df = pd.concat(filtered_chunks)
        else:
            filtered_df = pd.DataFrame()

        # Plot filtered data
        if not filtered_df.empty:
            activation_matrix = filtered_df.drop(columns=["token"], errors="ignore").values

            # Compute vmin and vmax for better contrast
            vmin, vmax = np.percentile(activation_matrix, [1, 99])

            fig = px.imshow(
                activation_matrix,
                labels=dict(x="Tokens", y="Layer", color="Activation"),
                x=filtered_df["token"] if "token" in filtered_df.columns else list(range(activation_matrix.shape[1])),
                y=[f"Layer {i}" for i in range(activation_matrix.shape[0])],
                color_continuous_scale="Bluered_r",
                zmin=vmin,
                zmax=vmax
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No matching tokens found in activation data.")

    except Exception as e:
        st.error(f"Couldn't load or filter logits_activations.csv: {e}")

    # --- Token-wise Rationale ---
    st.subheader("📚 Token-wise Rationale")
    found = False

    try:
        with open("rationales.jsonl") as f:
            for line in f:
                data = json.loads(line)
                if data.get("prompt", "").strip() == prompt.strip():
                    found = True
                    for token, rationale in zip(data.get("tokens", []), data.get("rationale", [])):
                        st.markdown(f"**{token}**: {rationale}")
                    break

        if not found:
            st.warning("No rationale found for this prompt.")
    except FileNotFoundError:
        st.error("rationales.jsonl not found.")
