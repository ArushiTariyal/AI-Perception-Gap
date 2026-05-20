"""
Interpretation Gap Engine - Streamlit UI
=========================================
Run:  streamlit run app.py
"""

import streamlit as st
import json
import time
from engine import run_agents, compute_similarity_matrix, divergence_score
from agents import AGENTS
from visualizations import (
    plot_heatmap,
    plot_divergence_gauge,
    plot_semantic_radar,
    plot_embedding_scatter,
)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Interpretation Gap Engine",
    page_icon="⌖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark background */
    .stApp { background-color: #0a0a0f; color: #d4d4e8; }
    .block-container { padding-top: 2rem; }

    /* Hide default streamlit header */
    header[data-testid="stHeader"] { background: transparent; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #12121a;
        border-right: 1px solid #1e1e2e;
    }

    /* Cards */
    .agent-card {
        background: #12121a;
        border-radius: 12px;
        padding: 18px;
        border-left: 3px solid;
        margin-bottom: 12px;
    }

    /* Tag chips */
    .chip {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
        margin: 2px;
    }

    /* Assumption bullets */
    .assumption {
        padding: 5px 10px;
        border-left: 2px solid;
        margin: 4px 0;
        border-radius: 0 6px 6px 0;
        font-size: 13px;
    }

    /* Metric tiles */
    div[data-testid="metric-container"] {
        background: #12121a;
        border: 1px solid #1e1e2e;
        border-radius: 10px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⌖ Interpretation Gap Engine")
    st.markdown("*Multi-agent semantic divergence analysis*")
    st.divider()

    st.markdown("### API Keys")
    st.caption("Keys stay in your browser session only.")

    gemini_key = st.text_input(
        "⚙ Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Free - get at aistudio.google.com/app/apikey",
    )
    groq_key = st.text_input(
        "φ Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Free - get at console.groq.com/keys",
    )
    st.markdown(
        "∑ **Ollama** - no key needed.  \n"
        "<span style=\'color:#6b6b8a;font-size:12px\'>"
        "Install ollama.com → run: <code>ollama pull mistral</code>"
        "</span>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("### Agents & Models")
    model_info = [
        ("G", "Gemini 2.0 Flash",   "Google DeepMind · via AI Studio",  "Free · no card"),
        ("L", "Llama 3.3 70B",      "Meta · via Groq",                   "Free · no card"),
        ("M", "Mistral 7B",         "Mistral AI · via Ollama (local)",   "Free · offline"),
    ]
    for icon, label, model, tagline in model_info:
        st.markdown(
            f"**{icon} {label}**  \n"
            f"<span style=\'color:#e8d5a3;font-size:11px\'>{model}</span>  \n"
            f"<span style=\'color:#6b6b8a;font-size:11px\'>{tagline}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("")

    st.divider()
    st.markdown("### Presets")
    presets = [
        "The machine replaced the teacher.",
        "AI will replace artists.",
        "Progress always has a cost.",
        "Silence is the loudest form of protest.",
        "We optimised our way out of meaning.",
        "The city never sleeps.",
    ]
    for p in presets:
        if st.button(p, use_container_width=True):
            st.session_state["preset"] = p

    st.divider()
    st.caption("Phase 1 · Prototype")

# ─── Main area ────────────────────────────────────────────────────────────────
st.markdown("# ⌖ Interpretation Gap Engine")
st.markdown(
    "Enter any statement. Three AI agents with different worldviews will interpret it - "
    "then we measure how far apart their meanings really are."
)
st.divider()

# Input box (pre-fill from sidebar preset)
default_input = st.session_state.pop("preset", "")
user_input = st.text_area(
    "Input Prompt",
    value=default_input,
    placeholder="e.g. The machine replaced the teacher.",
    height=80,
)

col_run, col_clear = st.columns([1, 5])
with col_run:
    run_btn = st.button("▶ Run Engine", type="primary", use_container_width=True)

# ─── Run ──────────────────────────────────────────────────────────────────────
if run_btn:
    # Validate at least Gemini + Groq keys are present (Ollama needs none)
    if not gemini_key:
        st.error("Please enter your Gemini API key in the sidebar (free at aistudio.google.com).")
        st.stop()
    if not groq_key:
        st.error("Please enter your Groq API key in the sidebar (free at console.groq.com).")
        st.stop()
    if not user_input.strip():
        st.warning("Please enter a prompt.")
        st.stop()

    # Bundle keys for the engine dispatcher
    api_keys = {
        "gemini": gemini_key,
        "groq":   groq_key,
        "ollama": None,   # Ollama is local - no key needed
    }

    # Progress bar while agents work
    progress = st.progress(0, text="Dispatching agents…")
    results, errors = run_agents(user_input.strip(), api_keys, progress_callback=progress)
    progress.empty()

    # Store in session
    st.session_state["results"] = results
    st.session_state["errors"] = errors
    st.session_state["input"] = user_input.strip()

    # Compute analysis
    matrix = compute_similarity_matrix(results)
    div = divergence_score(matrix)
    st.session_state["matrix"] = matrix
    st.session_state["divergence"] = div

    # Append to history
    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.session_state["history"].append({
        "input": user_input.strip(),
        "divergence": div,
        "results": results,
    })

# ─── Results ──────────────────────────────────────────────────────────────────
if "results" in st.session_state:
    results  = st.session_state["results"]
    errors   = st.session_state["errors"]
    matrix   = st.session_state["matrix"]
    div      = st.session_state["divergence"]
    prompt   = st.session_state["input"]

    st.markdown(f"> **Prompt:** *{prompt}*")
    st.divider()

    # ── Top metrics ──
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Divergence Score", f"{round(div * 100)}%")

    # Find max conflict pair
    min_sim = 1.0
    conflict_pair = (0, 1)
    for i in range(3):
        for j in range(i + 1, 3):
            if matrix[i][j] < min_sim:
                min_sim = matrix[i][j]
                conflict_pair = (i, j)
    a_idx, b_idx = conflict_pair
    m2.metric("Max Conflict", f"{AGENTS[a_idx]['label']} vs {AGENTS[b_idx]['label']}")
    m3.metric("Min Similarity", f"{min_sim:.2f}")

    tones = [r.get("emotional_tone", "neutral") for r in results]
    unique_tones = len(set(tones))
    m4.metric("Emotional Variance", f"{unique_tones}/3 unique tones")

    st.divider()

    # ── Agent cards ──
    st.markdown("### Agent Interpretations")
    cols = st.columns(3)
    colors = ["#7eb8f7", "#f79e7e", "#9ef7c8"]
    emotion_colors = {
        "optimistic": "#7ef7a0", "excited": "#f7e97e", "neutral": "#a0a0c0",
        "cautious": "#f7c97e", "melancholic": "#9eb8f7", "fearful": "#f77e9e",
    }

    for i, (agent, result, col) in enumerate(zip(AGENTS, results, cols)):
        with col:
            ec = emotion_colors.get(result.get("emotional_tone", "neutral"), "#a0a0c0")
            st.markdown(
                f"""<div class="agent-card" style="border-color:{colors[i]}">
                    <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
                        <span style="font-size:22px;color:{colors[i]}">{agent['icon']}</span>
                        <div>
                            <strong style="color:{colors[i]}">{agent['label']}</strong><br/>
                            <span style="font-size:11px;color:#6b6b8a">
                                Confidence: {round(result.get('confidence', 0.8) * 100)}%
                            </span>
                        </div>
                        <span class="chip" style="margin-left:auto;background:{ec}22;color:{ec};border:1px solid {ec}55">
                            {result.get('emotional_tone','neutral')}
                        </span>
                    </div>
                    <div style="font-size:11px;color:#a89060;margin-bottom:6px">
                        Frame: <strong style="color:#e8d5a3">{result.get('core_frame','—')}</strong>
                    </div>
                    <p style="font-size:13px;line-height:1.7;color:#d4d4e8">
                        {result.get('interpretation','—')}
                    </p>
                </div>""",
                unsafe_allow_html=True,
            )

            # Hidden assumptions
            st.markdown("**Hidden Assumptions**")
            for assumption in result.get("hidden_assumptions", []):
                st.markdown(
                    f'<div class="assumption" style="border-color:{colors[i]}88;background:{colors[i]}08">{assumption}</div>',
                    unsafe_allow_html=True,
                )

            # Keywords
            st.markdown("**Keywords**")
            chips_html = "".join(
                f'<span class="chip" style="background:{colors[i]}18;color:{colors[i]};border:1px solid {colors[i]}33">{kw}</span>'
                for kw in result.get("keywords", [])
            )
            st.markdown(chips_html, unsafe_allow_html=True)

            if errors[i]:
                st.error(f"Error: {errors[i]}")

    st.divider()

    # ── Visualizations ──
    st.markdown("### Semantic Analysis")
    v1, v2 = st.columns(2)

    with v1:
        st.markdown("**Similarity Heatmap**")
        fig_heatmap = plot_heatmap(matrix, AGENTS)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with v2:
        st.markdown("**Divergence Gauge**")
        fig_gauge = plot_divergence_gauge(div)
        st.plotly_chart(fig_gauge, use_container_width=True)

    v3, v4 = st.columns(2)

    with v3:
        st.markdown("**Semantic Radar**")
        fig_radar = plot_semantic_radar(results, AGENTS)
        st.plotly_chart(fig_radar, use_container_width=True)

    with v4:
        st.markdown("**Embedding Scatter (PCA)**")
        fig_scatter = plot_embedding_scatter(results, AGENTS)
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()

    # ── Raw JSON expander ──
    with st.expander("📄 Raw JSON output"):
        for agent, result in zip(AGENTS, results):
            st.markdown(f"**{agent['label']}**")
            st.json(result)

# ─── History ──────────────────────────────────────────────────────────────────
if st.session_state.get("history") and len(st.session_state["history"]) > 1:
    st.divider()
    st.markdown("### Run History")
    for i, h in enumerate(reversed(st.session_state["history"])):
        pct = round(h["divergence"] * 100)
        hue = max(0, 120 - int(h["divergence"] * 120))
        color = f"hsl({hue}, 70%, 55%)"
        st.markdown(
            f'<div style="display:flex;gap:12px;align-items:center;padding:8px 12px;'
            f'background:#12121a;border-radius:8px;margin-bottom:6px;border:1px solid #1e1e2e">'
            f'<span style="color:#6b6b8a;font-family:monospace">{str(i+1).zfill(2)}</span>'
            f'<span style="flex:1;color:#d4d4e8">{h["input"]}</span>'
            f'<span style="color:{color};font-family:monospace">Δ {pct}%</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
