# ⌖ Interpretation Gap Engine {AI Perception Gap}

A multi-agent AI system that demonstrates how the same text can be interpreted
radically differently depending on an agent's worldview, goals, and priors.

\

## What It Does

1. You enter any statement (e.g. *"The machine replaced the teacher."*)
2. Three agents interpret it simultaneously, each with a distinct personality (can be modified):

   * **Techno-Optimist** - sees progress and empowerment
   * **Humanist Philosopher** - sees ethics and lost meaning
   * **Systems Economist** - sees market forces and equilibria
3. The engine extracts:

   * Interpretation text
   * Core framing label
   * Hidden assumptions
   * Emotional tone
   * Keywords + confidence score
4. Sentence-transformer embeddings measure semantic divergence
5. Plotly visualizations show:

   * Cosine similarity heatmap
   * Divergence gauge (0–100%)
   * Semantic dimension radar
   * PCA embedding scatter

\

## Setup

### 1\. Clone / download this folder

```bash
cd interpretation\_gap\_engine
```

### 2\. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\\Scripts\\activate         # Windows
```

### 3\. Install dependencies

```bash
pip install -r requirements.txt
```

> \*\*Note:\*\* `sentence-transformers` will download `all-MiniLM-L6-v2` on first run.

### 4\. Run the app

```bash
streamlit run app.py
```

### 5\. Enter your API key

Paste your API key in the sidebar. It stays in your browser session only.

\

## Project Structure

```
interpretation\_gap\_engine/
├── app.py               # Streamlit UI - layout, input, results rendering
├── agents.py            # Agent definitions (system prompts + metadata)
├── engine.py            # API calls, embeddings, similarity, divergence score
├── visualizations.py    # All Plotly charts (heatmap, gauge, radar, scatter)
├── requirements.txt
└── README.md
```

\

## Tech Stack

|Layer|Library|
|-|-|
|LLM|`Gemini, Ollama, Groq`|
|Embeddings|`sentence-transformers` (all-MiniLM-L6-v2)|
|Similarity|`sklearn` cosine\_similarity|
|UI|`streamlit`|
|Charts|`plotly`|
|Parallelism|`concurrent.futures.ThreadPoolExecutor`|

\

## Phase 2 Roadmap

* \[ ] **Memory** - agents remember past prompts via ChromaDB/FAISS
* \[ ] **Debate mode** - agents challenge each other's hidden assumptions
* \[ ] **NLI contradiction detection** - find logical conflicts using DeBERTa
* \[ ] **Interpretation trees** - multi-layer output (primary/secondary/moral)
* \[ ] **Semantic graphs** - NetworkX node graphs of concept relationships
* \[ ] **Belief evolution** - track how interpretations shift over a conversation

\

## Why This Project Is Strong

* Touches **AI alignment** and **interpretability** (not just a chatbot)
* Demonstrates **multi-agent orchestration**
* Uses **real NLP** (embeddings, cosine similarity, PCA)
* Visually impressive without needing frontier compute
* Connects to philosophy of language and cognitive science
* Portfolio-friendly: *"I explored how model priors alter semantic interpretation"*

