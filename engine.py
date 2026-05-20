"""
engine.py - Core analysis engine
=================================
Handles:
  - Per-provider API calls (Gemini, Groq, Ollama)
  - Parallel execution via ThreadPoolExecutor
  - Response parsing + validation
  - Sentence-transformer embeddings (all-MiniLM-L6-v2)
  - Cosine similarity matrix
  - Divergence score

Notes for reference (token limits are an issue here)
--------------
  Gemini  → google-generativeai SDK, free tier (no card)
  Groq    → openai-compatible REST API, free tier (no card)
  Ollama  → local REST API at http://localhost:11434, no key needed
"""

import json
import re
import concurrent.futures
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import google.generativeai as genai   # pip install google-generativeai
from openai import OpenAI             # pip install openai  (used for Groq too)

from agents import AGENTS


# ─── Embedding model (lazy-loaded once) ──────────────────────────────────────
_EMBED_MODEL = None

def get_embed_model() -> SentenceTransformer:
    """Load all-MiniLM-L6-v2 once; reuse across calls."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED_MODEL


# ─── JSON parser (shared) ─────────────────────────────────────────────────────
def _parse_json(raw: str) -> dict:
    """Strip markdown fences and parse JSON. Raises json.JSONDecodeError on failure."""
    cleaned = re.sub(r"```json|```", "", raw).strip()
    return json.loads(cleaned)


# ─── Provider-specific callers ───────────────────────────────────────────────

def _call_gemini(agent: dict, user_input: str, api_key: str) -> dict:
    """
    Called Google Gemini via the google-generativeai SDK.
    key: https://aistudio.google.com/app/apikey
    """
    genai.configure(api_key=api_key)

    # Gemini uses a single 'contents' field; I prepended the system prompt
    # as a user turn since the free SDK doesn't support system_instruction
    # on all models - this works universally.
    model = genai.GenerativeModel(
        model_name=agent["model"],
        system_instruction=agent["system"],   # supported in gemini-2.0-flash
        # gemini-1.5-flash was not responding to query request - reason unknown
    )

    response = model.generate_content(user_input)
    return _parse_json(response.text)


def _call_groq(agent: dict, user_input: str, api_key: str) -> dict:
    """
    Call Groq via its OpenAI-compatible API.
    key: https://console.groq.com/keys
    """
    # Groq uses the OpenAI SDK pointed at a different base URL
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    response = client.chat.completions.create(
        model=agent["model"],
        max_tokens=1024,
        messages=[
            {"role": "system", "content": agent["system"]},
            {"role": "user",   "content": user_input},
        ],
    )

    raw = response.choices[0].message.content
    return _parse_json(raw)


def _call_ollama(agent: dict, user_input: str, _api_key: str = None) -> dict:
    """
    Called Ollama's local REST API (no API key needed).
    Installed: https://ollama.com
    Then ran: ollama pull mistral
    Ollama must be running locally on port 11434. (keep the terminal on)
    """
    payload = {
        "model":  agent["model"],
        "prompt": agent["system"] + "\n\nUser input: " + user_input,
        "stream": False,   # to get the full response at once
    }

    resp = requests.post(
        "http://localhost:11434/api/generate",
        json=payload,
        timeout=300,   # local models can be slower on first call
                       # however it might be a system limitation too
    )
    resp.raise_for_status()

    raw = resp.json().get("response", "")
    return _parse_json(raw)


# ─── Dispatcher ───────────────────────────────────────────────────────────────

# Maps provider string → caller function
_CALLERS = {
    "gemini": _call_gemini,
    "groq":   _call_groq,
    "ollama": _call_ollama,
}

def call_agent(agent: dict, user_input: str, api_keys: dict) -> dict:
    """
    Route the call to the correct provider based on agent["provider"].

    api_keys = {
        "gemini": "AIza...",
        "groq":   "gsk_...",
        "ollama": None,      # no key needed
    }
    """
    provider = agent["provider"]
    caller   = _CALLERS.get(provider)

    if caller is None:
        raise ValueError(f"Unknown provider: '{provider}'")

    key = api_keys.get(provider)  # None is fine for Ollama
    return caller(agent, user_input, key)


# ─── Parallel execution ───────────────────────────────────────────────────────

def run_agents(user_input: str, api_keys: dict, progress_callback=None):
    """
    Fire all three agent calls concurrently.

    Args:
        user_input      — the prompt string
        api_keys        — dict of { "gemini": key, "groq": key, "ollama": None }
        progress_callback — optional Streamlit progress bar object

    Returns:
        results — list of 3 parsed dicts (in AGENTS order)
        errors  — list of 3 error strings or None
    """
    results = [None] * len(AGENTS)
    errors  = [None] * len(AGENTS)
    done    = [0]

    def task(index: int, agent: dict):
        try:
            results[index] = call_agent(agent, user_input, api_keys)
        except Exception as exc:
            # Store a placeholder so downstream code always has 3 items
            results[index] = {
                "interpretation": f"[{agent['provider']} error] {exc}",
                "core_frame":     "error",
                "hidden_assumptions": [],
                "emotional_tone": "neutral",
                "keywords":       [],
                "confidence":     0.0,
            }
            errors[index] = str(exc)
        finally:
            done[0] += 1
            if progress_callback:
                progress_callback.progress(
                    done[0] / len(AGENTS),
                    text=f"{AGENTS[index]['label']} ({AGENTS[index]['provider']}) done — {done[0]}/{len(AGENTS)}",
                )

    # I/O-bound: threads work well here
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(task, i, agent) for i, agent in enumerate(AGENTS)]
        concurrent.futures.wait(futures)

    return results, errors


# ─── Embeddings ───────────────────────────────────────────────────────────────

def embed_results(results: list) -> np.ndarray:
    """
    Encode each agent's interpretation + keywords into a dense vector.
    Returns shape (n_agents, 384) for all-MiniLM-L6-v2.
    """
    model = get_embed_model()
    texts = [
        result.get("interpretation", "") + " " + " ".join(result.get("keywords", []))
        for result in results
    ]
    return model.encode(texts, normalize_embeddings=True)


# ─── Similarity matrix ────────────────────────────────────────────────────────

def compute_similarity_matrix(results: list) -> np.ndarray:
    """
    Pairwise cosine similarity between agent embeddings.
    Returns a (3, 3) float array, values clipped to [0, 1].
    """
    embeddings = embed_results(results)
    matrix = cosine_similarity(embeddings)
    return np.clip(matrix, 0, 1).round(3)


# ─── Divergence score ─────────────────────────────────────────────────────────

def divergence_score(matrix: np.ndarray) -> float:
    """
    Mean off-diagonal dissimilarity: average of (1 - sim) for all i≠j pairs.
    0.0 = perfect consensus, 1.0 = maximally different.
    """
    n = matrix.shape[0]
    vals = [1 - matrix[i][j] for i in range(n) for j in range(n) if i != j]
    return round(sum(vals) / len(vals), 3) if vals else 0.0


# ─── Phase 2: NLI contradiction detection (stub) ─────────────────────────────

def detect_contradictions(results: list) -> list:
    """
    INCOMPLETE SECTION

    Detect logical contradictions between agent interpretations using NLI.
    Returns list of (agent_i, agent_j, contradiction_score) tuples.

    To be uncommented when ready for Phase 2 - requires:
        pip install transformers torch
    Model: cross-encoder/nli-deberta-v3-small
    """
    # from transformers import pipeline
    # nli = pipeline("zero-shot-classification",
    #                model="cross-encoder/nli-deberta-v3-small")
    # pairs = []
    # for i in range(len(results)):
    #     for j in range(i + 1, len(results)):
    #         out = nli(
    #             results[j]["interpretation"],
    #             candidate_labels=["entailment", "neutral", "contradiction"],
    #         )
    #         score = out["scores"][out["labels"].index("contradiction")]
    #         pairs.append((i, j, round(score, 3)))
    # return pairs
    return []
