"""
visualizations.py - Plotly chart builders
==========================================
All charts use a dark theme consistent with the Streamlit app.
"""

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# ─── Shared colour palette ────────────────────────────────────────────────────
AGENT_COLORS = ["#7eb8f7", "#f79e7e", "#9ef7c8"]
BG = "#0a0a0f"
SURFACE = "#12121a"
GRID = "#1e1e2e"
TEXT = "#d4d4e8"
MUTED = "#6b6b8a"
GOLD = "#e8d5a3"

DARK_LAYOUT = dict(
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(color=TEXT, family="Georgia, serif"),
    margin=dict(l=20, r=20, t=40, b=20),
)

def _hex_to_rgba(hex_color: str, alpha: float = 0.16) -> str:
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ─── 1. Similarity Heatmap ───────────────────────────────────────────────────
def plot_heatmap(matrix: np.ndarray, agents: list) -> go.Figure:
    """
    3×3 colour-coded cosine similarity heatmap.
    Diagonal = 1.0 (grey), off-diagonal = blue (low) → warm (high).
    """
    labels = [a["label"].split()[0] for a in agents]

    # Build annotation text
    annotations = []
    for i in range(3):
        for j in range(3):
            val = "—" if i == j else f"{matrix[i][j]:.2f}"
            annotations.append(
                dict(x=j, y=i, text=val, showarrow=False,
                     font=dict(color=TEXT, size=14, family="monospace"))
            )

    # Mask diagonal with NaN so it renders differently
    display = matrix.astype(float).copy()
    np.fill_diagonal(display, np.nan)

    fig = go.Figure(
        go.Heatmap(
            z=display,
            x=labels,
            y=labels,
            colorscale=[
                [0.0, "#1a3a6b"],   # low similarity → dark blue
                [0.5, "#5a3a1a"],   # mid
                [1.0, "#c87941"],   # high similarity → warm amber
            ],
            zmin=0,
            zmax=1,
            showscale=True,
            colorbar=dict(
                tickfont=dict(color=TEXT),
                outlinecolor=GRID,
                title=dict(text="Sim", font=dict(color=MUTED)),
            ),
            hoverongaps=False,
        )
    )

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="Cosine Similarity", font=dict(color=GOLD), x=0.5),
        annotations=annotations,
        xaxis=dict(tickfont=dict(color=TEXT), gridcolor=GRID),
        yaxis=dict(tickfont=dict(color=TEXT), gridcolor=GRID, autorange="reversed"),
    )
    return fig


# ─── 2. Divergence Gauge ─────────────────────────────────────────────────────
def plot_divergence_gauge(score: float) -> go.Figure:
    """
    Radial gauge from 0% (consensus) to 100% (max divergence).
    Colour shifts green → amber → red.
    """
    pct = round(score * 100)
    hue_g = max(0, 120 - int(score * 120))
    bar_color = f"hsl({hue_g}, 70%, 45%)"

    label = "High consensus" if pct < 30 else "Moderate divergence" if pct < 60 else "Strong conflict"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=pct,
            number=dict(suffix="%", font=dict(color=TEXT, size=36)),
            title=dict(text=f"{label}", font=dict(color=MUTED, size=13)),
            gauge=dict(
                axis=dict(
                    range=[0, 100],
                    tickwidth=1,
                    tickcolor=MUTED,
                    tickfont=dict(color=MUTED),
                ),
                bar=dict(color=bar_color, thickness=0.25),
                bgcolor=GRID,
                borderwidth=0,
                steps=[
                    dict(range=[0,  30], color="#0f2a0f"),
                    dict(range=[30, 60], color="#2a1f0a"),
                    dict(range=[60, 100], color="#2a0a0a"),
                ],
                threshold=dict(
                    line=dict(color=GOLD, width=2),
                    thickness=0.75,
                    value=pct,
                ),
            ),
        )
    )

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="Interpretation Divergence", font=dict(color=GOLD), x=0.5),
        height=280,
    )
    return fig


# ─── 3. Semantic Radar ───────────────────────────────────────────────────────
# projecting each agent's response onto 5 semantic dimensions by counting
# keyword overlap with curated word lists.

DIMENSIONS = {
    "Technology":  ["machine", "ai", "algorithm", "automation", "system", "digital", "software", "data"],
    "Humanity":    ["human", "people", "emotion", "meaning", "dignity", "care", "soul", "identity"],
    "Economics":   ["labour", "market", "cost", "efficiency", "capital", "wage", "profit", "resource"],
    "Ethics":      ["moral", "right", "wrong", "value", "justice", "harm", "responsibility", "equity"],
    "Uncertainty": ["risk", "unknown", "uncertain", "unclear", "complex", "ambiguous", "potential", "maybe"],
}

def _dimension_scores(result: dict) -> list:
    """Score a result on each semantic dimension (0–1)."""
    text = (
        result.get("interpretation", "") + " " +
        result.get("core_frame", "") + " " +
        " ".join(result.get("keywords", []))
    ).lower()
    scores = []
    for words in DIMENSIONS.values():
        hits = sum(1 for w in words if w in text)
        scores.append(min(hits / 3, 1.0))  # normalise to ~[0,1]
    return scores


def plot_semantic_radar(results: list, agents: list) -> go.Figure:
    """
    Radar chart: each agent plotted on 5 semantic dimensions.
    """
    categories = list(DIMENSIONS.keys())
    fig = go.Figure()

    for i, (agent, result) in enumerate(zip(agents, results)):
        scores = _dimension_scores(result)
        # Close the polygon
        fig.add_trace(
            go.Scatterpolar(
                r=scores + [scores[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=agent["label"],
                line=dict(color=AGENT_COLORS[i], width=2),
                fillcolor=_hex_to_rgba(AGENT_COLORS[i]),
                opacity=0.85,
            )
        )

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="Semantic Dimension Radar", font=dict(color=GOLD), x=0.5),
        polar=dict(
            bgcolor=SURFACE,
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickfont=dict(color=MUTED, size=10),
                gridcolor=GRID,
                linecolor=GRID,
            ),
            angularaxis=dict(
                tickfont=dict(color=TEXT, size=11),
                gridcolor=GRID,
                linecolor=GRID,
            ),
        ),
        legend=dict(font=dict(color=TEXT), bgcolor=SURFACE, bordercolor=GRID),
        height=320,
    )
    return fig


# ─── 4. Embedding Scatter (PCA) ──────────────────────────────────────────────
def plot_embedding_scatter(results: list, agents: list) -> go.Figure:
    """
    Project the 3 agent embeddings to 2D with PCA and plot as a scatter.
    Because we only have 3 points, this is mainly illustrative -
    but it becomes genuinely informative once we accumulate history runs.
    """
    from engine import embed_results

    embeddings = embed_results(results)  # shape (3, dim)

    if embeddings.shape[0] < 2:
        fig = go.Figure()
        fig.update_layout(
            **DARK_LAYOUT,
            title=dict(text="Not enough data for PCA", font=dict(color=GOLD), x=0.5),
        )
        return fig

    # PCA to 2D
    n_components = min(2, embeddings.shape[0] - 1)
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(embeddings)

    # Pad to 2D if needed
    if coords.shape[1] == 1:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 1))])

    fig = go.Figure()

    for i, agent in enumerate(agents):
        fig.add_trace(
            go.Scatter(
                x=[coords[i, 0]],
                y=[coords[i, 1]],
                mode="markers+text",
                name=agent["label"],
                text=[agent["icon"] + " " + agent["label"].split()[0]],
                textposition="top center",
                textfont=dict(color=AGENT_COLORS[i], size=12),
                marker=dict(
                    size=18,
                    color=AGENT_COLORS[i],
                    line=dict(color=SURFACE, width=2),
                ),
            )
        )

    # Draw lines between all pairs
    for i in range(3):
        for j in range(i + 1, 3):
            fig.add_trace(
                go.Scatter(
                    x=[coords[i, 0], coords[j, 0]],
                    y=[coords[i, 1], coords[j, 1]],
                    mode="lines",
                    line=dict(color=GRID, width=1, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    var_explained = pca.explained_variance_ratio_ * 100
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="Embedding Space (PCA 2D)", font=dict(color=GOLD), x=0.5),
        xaxis=dict(
            title=dict(text=f"PC1 ({var_explained[0]:.1f}%)", font=dict(color=MUTED)),
            tickfont=dict(color=MUTED),
            gridcolor=GRID,
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text=f"PC2 ({var_explained[1]:.1f}% if available)", font=dict(color=MUTED)),
            tickfont=dict(color=MUTED),
            gridcolor=GRID,
            zeroline=False,
        ),
        legend=dict(font=dict(color=TEXT), bgcolor=SURFACE, bordercolor=GRID),
        height=320,
    )
    return fig
