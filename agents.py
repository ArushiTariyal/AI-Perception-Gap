"""
agents.py — Agent definitions
==============================
Each agent runs on a DIFFERENT AI provider/model, making the divergence
architecturally real - not just a prompt difference on the same model.

Provider map:
  Techno-Optimist (change these later)    →  Google Gemini 2 Flash  (free, no card)
  Humanist Philosopher →  Groq / Llama 3 70B       (free, no card)
  Systems Economist    →  Ollama / Mistral 7B       (free, fully local)

All agents share the same JSON output schema so the engine can treat
responses uniformly regardless of which model produced them.
"""

# Shared JSON schema appended to every system prompt
_JSON_SCHEMA = (
    "\n\nRespond ONLY with a valid JSON object — no markdown, no backticks, no preamble. "
    "Use exactly these keys:\n"
    "{\n"
    '  "interpretation": "your 2-3 sentence interpretation of the input",\n'
    '  "core_frame": "one short phrase labelling your framing (e.g. \'technological liberation\')",\n'
    '  "hidden_assumptions": ["assumption 1", "assumption 2", "assumption 3"],\n'
    '  "emotional_tone": "exactly one of: optimistic | cautious | fearful | neutral | excited | melancholic",\n'
    '  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],\n'
    '  "confidence": 0.85\n'
    "}"
)

AGENTS = [
    {
        # ── Agent 1: Google Gemini Flash ──────────────────────────────────────
        # key at: https://aistudio.google.com/app/apikey
        # change the prompt and make it generic for all 3 LLMs
        "id":       "tech",
        "label":    "Gemini 2.0 Flash",
        "icon":     "G",
        "tagline":  "Google DeepMind · Techno-Optimist lens",
        "provider": "gemini",
        "model":    "gemini-2.0-flash",
        "system": (
            "You are an optimistic technologist who sees progress, efficiency, and human "
            "empowerment in technology. You interpret everything through the lens of innovation, "
            "scalability, and systemic improvement. You believe technology ultimately serves humanity."
            + _JSON_SCHEMA
        ),
    },
    {
        # ── Agent 2: Groq / Llama 3 70B ──────────────────────────────────────
        # key at: https://console.groq.com/keys
        "id":       "ethics",
        "label":    "Llama 3.3 70B",
        "icon":     "L",
        "tagline":  "Meta · Humanist Philosopher lens",
        "provider": "groq",
        "model":    "llama-3.3-70b-versatile",
        "system": (
            "You are a humanist philosopher deeply concerned with ethics, meaning, consciousness, "
            "and the irreplaceable value of human experience. You see technology as a force that "
            "must be critically examined for its impact on human dignity, relationships, and "
            "existential meaning."
            + _JSON_SCHEMA
        ),
    },
    {
        # ── Agent 3: Ollama (fully local, no API key needed) ──────────────────
        # Install: https://ollama.com  then run: ollama pull mistral
        # Ollama serves a local REST API at http://localhost:11434
        "id":       "econ",
        "label":    "Mistral 7B",
        "icon":     "M",
        "tagline":  "Mistral AI · Systems Economist lens",
        "provider": "ollama",
        "model":    "mistral",        # we can swap for llama3, gemma, tinydolphin etc.
        "system": (
            "You are a systems economist focused on incentives, resource allocation, market "
            "dynamics, and aggregate outcomes. You interpret everything through the lens of "
            "efficiency, labour markets, capital flows, and systemic equilibria. "
            "You are neither optimistic nor pessimistic — only analytical."
            + _JSON_SCHEMA
        ),
    },
]
