# XAI-LLM-LAB
# 🧠 XAI-LLM-LAB: Interpreting Language Models through Activation Graphs

This project explores the internal decision-making of language models by visualizing neuron-level and attention head activations. We apply interpretability tools on small GGUF-based models (like Phi-2 or TinyLlama) running via `llama.cpp`, and generate rationale-based explanations to match TransformerLens outputs.

## ✨ Features

- 🔍 Activation extraction from modified `llama.cpp`
- 📊 Compare and visualize token-level logits and neuron paths
- 📈 TransformerLens-style rationale generation
- 📂 Integrated tokenizer and TTS tools
- 📦 Compatible with GGUF CPU-friendly models (e.g., Phi-2, TinyLlama)
  
## 📁 Project Structure

<pre lang="md"> ## 📁 Project Structure ``` XAI-LLM-LAB/ ├── llama.cpp/ │ └── src/ │ └── llama-context.cpp # Modified llama.cpp with hooks and tools ├── logits/ │ ├── generate_rationales.py # Script to trace and save activations │ ├── compare_logits.py # Compare GGUF vs TransformerLens │ ├── ration.py # Prompt-to-rationale conversion │ ├── rationales.jsonl # Output rationales │ ├── report.html / report.pdf # Final visual report │ └── logits_activations.csv # [External download below 🔗] ``` </pre>

## ⚙️ Setup Instructions

**1. Clone this repo:**
   ```bash
   git clone https://github.com/stefishobikasss/XAI-LLM-LAB.git
   cd XAI-LLM-LAB
   
**2.To Access the modified hooks and tools**
  cd llama.cpp
  cd build
  cmake ..
  make -j

**3.To generate rationales**
cd logits
python generate_rationales.py

**4.To access the dashboard**
cd logits
streamlit run dashboard.py

---

### 📌 5. External File Links

```md
## 📂 Large Files

Due to GitHub’s 100MB file limit, this file is hosted externally:

- [`logits_activations.csv`](https://drive.google.com/file/d/1Zw9jhiKfDmSY0J2_CPaGDen-fOm4SfVX/view?usp=sharing)

## 🙏 Acknowledgements

- This project is part of an internship at **Karunya Innovation & Design Studio**.
- Based on `llama.cpp`, `TransformerLens`, and HuggingFace tools.

