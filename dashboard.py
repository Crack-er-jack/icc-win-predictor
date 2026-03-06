"""
dashboard.py — Real-Time Cricket Win Prediction Dashboard

Streamlit + Plotly dashboard that displays:
    - Win probability gauges (IND vs NZ)
    - Win probability trend over time
    - Projected final score
    - Momentum indicator
    - Player impact chart
    - "What If?" analysis panel
    - Ball-by-ball outcome probabilities
    - Live commentary ticker

Auto-refreshes every 10 seconds.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator import MatchSimulator

# ============================================================================
# Page Configuration
# ============================================================================

# Move set_page_config inside a function to avoid being called multiply during imports
def setup_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="🏏 ICC Win Predictor — IND vs NZ",
        page_icon="🏏",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# ============================================================================
# Custom CSS for Premium Dark Theme
# ============================================================================

def apply_custom_styles():
    """Inject custom CSS for the dark theme with mobile responsiveness."""
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Dark Theme */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1117 50%, #0a0a1a 100%);
        color: #e6edf3;
        font-family: 'Inter', sans-serif;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Section headers */
    .section-header {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 1.1rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #8b949e;
        margin-bottom: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(88, 166, 255, 0.15);
    }

    /* Match score banner */
    .score-banner {
        background: linear-gradient(145deg, #161b22 0%, #1c2333 100%);
        border: 1px solid rgba(88, 166, 255, 0.2);
        border-radius: 16px;
        padding: 1.2rem 2rem;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255,255,255,0.05);
    }

    .score-banner .score {
        font-family: 'JetBrains Mono', monospace;
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ff6b00, #ff9500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }

    .score-banner .info {
        font-size: 1rem;
        color: #8b949e;
        margin-top: 0.3rem;
    }

    .score-banner .target {
        font-size: 0.95rem;
        color: #58a6ff;
        font-weight: 600;
    }

    /* Win probability big numbers */
    .win-prob-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 3rem;
        margin: 1rem 0;
    }

    .win-prob-card {
        text-align: center;
        padding: 1.5rem 2rem;
        border-radius: 16px;
        min-width: 200px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: transform 0.3s ease;
    }

    .win-prob-card:hover {
        transform: translateY(-4px);
    }

    .win-prob-india {
        background: linear-gradient(145deg, rgba(255, 107, 0, 0.12), rgba(255, 149, 0, 0.06));
        border: 1px solid rgba(255, 107, 0, 0.3);
    }

    .win-prob-nz {
        background: linear-gradient(145deg, rgba(88, 166, 255, 0.12), rgba(56, 132, 244, 0.06));
        border: 1px solid rgba(88, 166, 255, 0.3);
    }

    .win-prob-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 3rem;
        font-weight: 800;
        line-height: 1;
    }

    .win-prob-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 0.4rem;
        opacity: 0.8;
    }

    /* Stat card */
    .stat-card {
        background: linear-gradient(145deg, #161b22, #1c2333);
        border: 1px solid rgba(88, 166, 255, 0.1);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }

    .stat-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: #e6edf3;
    }

    .stat-label {
        font-size: 0.75rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.2rem;
    }

    /* Commentary ticker */
    .commentary-item {
        background: rgba(22, 27, 34, 0.8);
        border-left: 3px solid rgba(88, 166, 255, 0.4);
        padding: 0.6rem 1rem;
        margin-bottom: 0.4rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
        transition: background 0.2s;
    }

    .commentary-item:hover {
        background: rgba(22, 27, 34, 1.0);
    }

    .commentary-wicket {
        border-left-color: #f85149;
        background: rgba(248, 81, 73, 0.08);
    }

    .commentary-boundary {
        border-left-color: #3fb950;
        background: rgba(63, 185, 80, 0.08);
    }

    .commentary-six {
        border-left-color: #d2a8ff;
        background: rgba(210, 168, 255, 0.08);
    }

    /* What If cards */
    .whatif-card {
        background: linear-gradient(145deg, #161b22, #1c2333);
        border: 1px solid rgba(136, 136, 136, 0.15);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        transition: border-color 0.2s;
    }

    .whatif-card:hover {
        border-color: rgba(88, 166, 255, 0.4);
    }

    .whatif-positive { border-left: 3px solid #3fb950; }
    .whatif-negative { border-left: 3px solid #f85149; }

    /* Demo badge */
    .demo-badge {
        display: inline-block;
        background: linear-gradient(135deg, #ff6b00, #ff9500);
        color: #000;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    /* Live badge */
    .live-badge {
        display: inline-block;
        background: linear-gradient(135deg, #f85149, #ff6b6b);
        color: #fff;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        animation: pulse 1.5s infinite;
    }

    /* Momentum indicator */
    .momentum-up { color: #3fb950; }
    .momentum-down { color: #f85149; }
    .momentum-neutral { color: #8b949e; }

    /* Override Streamlit default backgrounds */
    .stMetric {
        background: transparent !important;
    }

    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    }

    /* MOBILE RESPONSIVENESS */
    @media (max-width: 768px) {
        .score-banner .score {
            font-size: 2.2rem;
        }
        .score-banner {
            padding: 1rem;
        }
        .win-prob-container {
            flex-direction: column;
            gap: 1rem;
        }
        .win-prob-card {
            width: 100%;
            min-width: unset;
            padding: 1rem;
        }
        .win-prob-value {
            font-size: 2.2rem;
        }
        .stat-value {
            font-size: 1.3rem;
        }
        .section-header {
            font-size: 0.9rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Session State Initialization
# ============================================================================

def init_session_state():
    """Initialize Streamlit session state on first load."""
    if "simulator" not in st.session_state:
        st.session_state.simulator = MatchSimulator(
            demo_mode=True, target=268, n_simulations=10000
        )
    if "dashboard_data" not in st.session_state:
        st.session_state.dashboard_data = st.session_state.simulator.refresh()
    if "refresh_count" not in st.session_state:
        st.session_state.refresh_count = 0
    if "auto_refresh" not in st.session_state:
        st.session_state.auto_refresh = True

# Removed top-level call to init_session_state()


# ============================================================================
# Chart Builders
# ============================================================================

def create_win_probability_gauge(india_prob: float, nz_prob: float) -> go.Figure:
    """Create a dual-gauge win probability meter."""
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}]],
        horizontal_spacing=0.15
    )

    # India gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=india_prob * 100,
        number={"suffix": "%", "font": {"size": 36, "color": "#ff9500", "family": "JetBrains Mono"}},
        title={"text": "🇮🇳 INDIA", "font": {"size": 16, "color": "#e6edf3"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#333", "tickwidth": 1},
            "bar": {"color": "#ff6b00", "thickness": 0.7},
            "bgcolor": "rgba(30,30,50,0.3)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "rgba(248, 81, 73, 0.15)"},
                {"range": [30, 60], "color": "rgba(210, 168, 255, 0.1)"},
                {"range": [60, 100], "color": "rgba(63, 185, 80, 0.15)"}
            ],
            "threshold": {
                "line": {"color": "#ff9500", "width": 3},
                "thickness": 0.8,
                "value": india_prob * 100
            }
        }
    ), row=1, col=1)

    # NZ gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=nz_prob * 100,
        number={"suffix": "%", "font": {"size": 36, "color": "#58a6ff", "family": "JetBrains Mono"}},
        title={"text": "🇳🇿 NEW ZEALAND", "font": {"size": 16, "color": "#e6edf3"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#333", "tickwidth": 1},
            "bar": {"color": "#3884f4", "thickness": 0.7},
            "bgcolor": "rgba(30,30,50,0.3)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "rgba(248, 81, 73, 0.15)"},
                {"range": [30, 60], "color": "rgba(210, 168, 255, 0.1)"},
                {"range": [60, 100], "color": "rgba(63, 185, 80, 0.15)"}
            ],
            "threshold": {
                "line": {"color": "#58a6ff", "width": 3},
                "thickness": 0.8,
                "value": nz_prob * 100
            }
        }
    ), row=1, col=2)

    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"}
    )
    return fig


def create_win_prob_timeline(history: dict) -> go.Figure:
    """Create a win probability timeline chart."""
    fig = go.Figure()

    if history["overs"]:
        # India probability area
        fig.add_trace(go.Scatter(
            x=history["overs"],
            y=[p * 100 for p in history["india_win"]],
            name="🇮🇳 India",
            fill="tozeroy",
            fillcolor="rgba(255, 107, 0, 0.15)",
            line=dict(color="#ff9500", width=3, shape="spline"),
            mode="lines+markers",
            marker=dict(size=4, color="#ff9500"),
            hovertemplate="Over %{x}<br>India: %{y:.1f}%<extra></extra>"
        ))

        # NZ probability area
        fig.add_trace(go.Scatter(
            x=history["overs"],
            y=[p * 100 for p in history["nz_win"]],
            name="🇳🇿 New Zealand",
            fill="tozeroy",
            fillcolor="rgba(88, 166, 255, 0.15)",
            line=dict(color="#58a6ff", width=3, shape="spline"),
            mode="lines+markers",
            marker=dict(size=4, color="#58a6ff"),
            hovertemplate="Over %{x}<br>NZ: %{y:.1f}%<extra></extra>"
        ))

        # 50% reference line
        fig.add_hline(y=50, line=dict(color="rgba(139, 148, 158, 0.3)",
                      width=1, dash="dash"))

    fig.update_layout(
        title=None,
        height=340,
        margin=dict(l=40, r=20, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title={"text": "Overs", "font": {"color": "#8b949e"}},
            gridcolor="rgba(139, 148, 158, 0.1)",
            tickfont=dict(color="#8b949e"),
            range=[0, max(history["overs"][-1] + 5, 50) if history["overs"] else 50]
        ),
        yaxis=dict(
            title={"text": "Win Probability (%)", "font": {"color": "#8b949e"}},
            gridcolor="rgba(139, 148, 158, 0.1)",
            tickfont=dict(color="#8b949e"),
            range=[0, 100]
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(color="#e6edf3")
        ),
        hoverlabel=dict(bgcolor="#161b22", font_size=13, font_family="JetBrains Mono")
    )
    return fig


def create_momentum_chart(over_by_over: list, crr: float, rrr: float) -> go.Figure:
    """Create a bar chart showing runs per over with overlaid run rates."""
    fig = go.Figure()

    if over_by_over:
        overs = list(range(1, len(over_by_over) + 1))
        colors = []
        for runs in over_by_over:
            if runs >= 10:
                colors.append("#3fb950")   # excellent
            elif runs >= 7:
                colors.append("#58a6ff")   # good
            elif runs >= 4:
                colors.append("#d2a8ff")   # average
            else:
                colors.append("#f85149")   # poor

        fig.add_trace(go.Bar(
            x=overs, y=over_by_over,
            marker=dict(
                color=colors,
                line=dict(width=0),
                opacity=0.85
            ),
            text=over_by_over,
            textposition="outside",
            textfont=dict(color="#e6edf3", size=10, family="JetBrains Mono"),
            hovertemplate="Over %{x}: %{y} runs<extra></extra>",
            name="Runs"
        ))

        # CRR line
        fig.add_hline(y=crr, line=dict(color="#ff9500", width=2, dash="dot"),
                      annotation_text=f"CRR {crr}", annotation_font=dict(color="#ff9500", size=11))

        # RRR line
        fig.add_hline(y=rrr, line=dict(color="#f85149", width=2, dash="dot"),
                      annotation_text=f"RRR {rrr}", annotation_font=dict(color="#f85149", size=11))

    fig.update_layout(
        height=260,
        margin=dict(l=40, r=20, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title={"text": "Over", "font": {"color": "#8b949e"}},
            tickfont=dict(color="#8b949e"),
            gridcolor="rgba(139, 148, 158, 0.05)"
        ),
        yaxis=dict(
            title={"text": "Runs", "font": {"color": "#8b949e"}},
            tickfont=dict(color="#8b949e"),
            gridcolor="rgba(139, 148, 158, 0.1)"),
        showlegend=False,
        hoverlabel=dict(bgcolor="#161b22", font_size=13)
    )
    return fig


def create_player_impact_chart(player_impact: dict) -> go.Figure:
    """Create a horizontal bar chart showing player contributions."""
    if not player_impact:
        fig = go.Figure()
        fig.update_layout(height=200, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        return fig

    # Top 6 contributors
    players = list(player_impact.keys())[:6]
    impacts = [player_impact[p]["impact_score"] for p in players]
    runs = [player_impact[p]["runs"] for p in players]
    srs = [player_impact[p]["sr"] for p in players]

    # Color gradient based on impact
    colors = [f"rgba(255, {max(100, 200 - i*25)}, 0, 0.8)" for i in range(len(players))]

    fig = go.Figure(go.Bar(
        y=players[::-1],
        x=impacts[::-1],
        orientation="h",
        marker=dict(
            color=colors[::-1],
            line=dict(width=0)
        ),
        text=[f"{r} ({sr})" for r, sr in zip(runs[::-1], srs[::-1])],
        textposition="inside",
        textfont=dict(color="#fff", size=12, family="JetBrains Mono"),
        hovertemplate="%{y}: %{x:.1f}% impact<extra></extra>"
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=120, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title={"text": "Impact %", "font": {"color": "#8b949e"}},
            tickfont=dict(color="#8b949e"),
            gridcolor="rgba(139, 148, 158, 0.1)"
        ),
        yaxis=dict(tickfont=dict(color="#e6edf3", size=12)),
        showlegend=False
    )
    return fig


def create_outcome_probs_chart(outcome_data: dict) -> go.Figure:
    """Create a pie/donut chart of GNN-predicted ball outcome probabilities."""
    labels = outcome_data.get("labels", [])
    values = outcome_data.get("values", [])

    if not values:
        fig = go.Figure()
        fig.update_layout(height=200, paper_bgcolor="rgba(0,0,0,0)")
        return fig

    colors = ["#8b949e", "#58a6ff", "#3fb950", "#d2a8ff",
              "#ff9500", "#ff6b6b", "#f85149"]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=[v * 100 for v in values],
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#0d1117", width=2)),
        textinfo="label+percent",
        textfont=dict(size=11, color="#e6edf3", family="Inter"),
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>"
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        annotations=[dict(
            text="Next<br>Ball",
            x=0.5, y=0.5,
            font=dict(size=16, color="#8b949e", family="Inter"),
            showarrow=False
        )]
    )
    return fig


# ============================================================================
# Dashboard Layout
# ============================================================================

def render_dashboard():
    """Render the complete dashboard."""
    data = st.session_state.dashboard_data

    # ---- HEADER ----
    mode = data.get("mode", "demo")
    badge = '<span class="demo-badge">⚡ DEMO MODE</span>' if mode == "demo" \
        else '<span class="live-badge">🔴 LIVE</span>'

    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 0.5rem;">
        <span style="font-family: 'Inter'; font-weight: 900; font-size: 1.6rem;
              letter-spacing: -0.5px; color: #e6edf3;">
            🏏 ICC Win Predictor
        </span>
        &nbsp;&nbsp;{badge}
    </div>
    """, unsafe_allow_html=True)

    ms = data.get("match_state", {})
    pred = data.get("prediction", {})

    # ---- SCORE BANNER ----
    st.markdown(f"""
    <div class="score-banner">
        <div style="font-size: 0.85rem; color: #8b949e; font-weight: 600;
             letter-spacing: 1px; text-transform: uppercase;">
            {ms.get('batting_team', 'India')} vs {ms.get('bowling_team', 'New Zealand')}
              &nbsp;•&nbsp; 2nd Innings
        </div>
        <div class="score">
            {ms.get('score', 0)}/{ms.get('wickets', 0)}
        </div>
        <div class="info">
            <span style="font-family: 'JetBrains Mono'; font-size: 1.1rem;">
                ({ms.get('overs', '0.0')} overs)
            </span>
        </div>
        <div class="target" style="margin-top: 0.3rem;">
            Need {ms.get('runs_remaining', 0)} off {ms.get('balls_remaining', 0)} balls
            &nbsp;•&nbsp; RRR: {ms.get('required_run_rate', 0)}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- WIN PROBABILITY SECTION ----
    india_wp = pred.get("india_win_prob", 0.5)
    nz_wp = pred.get("nz_win_prob", 0.5)

    st.markdown('<div class="section-header">⚡ Win Probability</div>',
                unsafe_allow_html=True)

    # Big number display
    st.markdown(f"""
    <div class="win-prob-container">
        <div class="win-prob-card win-prob-india">
            <div class="win-prob-value" style="color: #ff9500;">{india_wp:.1%}</div>
            <div class="win-prob-label" style="color: #ff9500;">🇮🇳 India</div>
        </div>
        <div style="font-size: 1.5rem; color: #333; font-weight: 700;">VS</div>
        <div class="win-prob-card win-prob-nz">
            <div class="win-prob-value" style="color: #58a6ff;">{nz_wp:.1%}</div>
            <div class="win-prob-label" style="color: #58a6ff;">🇳🇿 New Zealand</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Gauge chart
    gauge_fig = create_win_probability_gauge(india_wp, nz_wp)
    st.plotly_chart(gauge_fig, width="stretch", key="gauge")

    # ---- STATS ROW ----
    stat_cols = st.columns(5)
    with stat_cols[0]:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-value">{ms.get('current_run_rate', 0)}</div>
            <div class="stat-label">Current RR</div>
        </div>""", unsafe_allow_html=True)
    with stat_cols[1]:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-value">{ms.get('required_run_rate', 0)}</div>
            <div class="stat-label">Required RR</div>
        </div>""", unsafe_allow_html=True)
    with stat_cols[2]:
        projected = pred.get("projected_total", 0)
        st.markdown(f"""<div class="stat-card">
            <div class="stat-value">~{projected}</div>
            <div class="stat-label">Projected Score</div>
        </div>""", unsafe_allow_html=True)
    with stat_cols[3]:
        momentum = data.get("momentum", 0)
        m_class = "momentum-up" if momentum > 0 else "momentum-down" if momentum < 0 else "momentum-neutral"
        m_icon = "↗" if momentum > 0 else "↘" if momentum < 0 else "→"
        st.markdown(f"""<div class="stat-card">
            <div class="stat-value {m_class}">{m_icon} {abs(momentum):.1f}</div>
            <div class="stat-label">Momentum</div>
        </div>""", unsafe_allow_html=True)
    with stat_cols[4]:
        sim_time = pred.get("simulation_time", 0)
        st.markdown(f"""<div class="stat-card">
            <div class="stat-value">{sim_time:.3f}s</div>
            <div class="stat-label">Sim Time</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- MAIN CONTENT: Two columns ----
    left_col, right_col = st.columns([3, 2])

    with left_col:
        # Win Probability Timeline
        st.markdown('<div class="section-header">📈 Win Probability Over Time</div>',
                    unsafe_allow_html=True)
        history = data.get("history", {"overs": [], "india_win": [], "nz_win": []})
        timeline_fig = create_win_prob_timeline(history)
        st.plotly_chart(timeline_fig, width="stretch", key="timeline")

        # Momentum chart (runs per over)
        st.markdown('<div class="section-header">📊 Runs Per Over</div>',
                    unsafe_allow_html=True)
        over_runs = data.get("over_by_over", [])
        mom_fig = create_momentum_chart(
            over_runs,
            ms.get("current_run_rate", 0),
            ms.get("required_run_rate", 0)
        )
        st.plotly_chart(mom_fig, width="stretch", key="momentum")

    with right_col:
        # GNN Outcome Probabilities
        st.markdown('<div class="section-header">🧠 GNN Next Ball Prediction</div>',
                    unsafe_allow_html=True)
        outcome_data = data.get("outcome_probs", {})
        outcome_fig = create_outcome_probs_chart(outcome_data)
        st.plotly_chart(outcome_fig, width="stretch", key="outcomes")

        # Player Impact
        st.markdown('<div class="section-header">👤 Player Impact</div>',
                    unsafe_allow_html=True)
        player_impact = data.get("player_impact", {})
        impact_fig = create_player_impact_chart(player_impact)
        st.plotly_chart(impact_fig, width="stretch", key="impact")

    # ---- COMMENTARY TICKER ----
    st.markdown('<div class="section-header">💬 Live Commentary</div>',
                unsafe_allow_html=True)
    recent_balls = data.get("recent_balls", [])
    for ball in reversed(recent_balls):
        css_class = "commentary-item"
        if ball.get("is_wicket"):
            css_class += " commentary-wicket"
        elif ball.get("is_six"):
            css_class += " commentary-six"
        elif ball.get("is_boundary"):
            css_class += " commentary-boundary"

        over_str = f"{ball.get('over', 0):.1f}"
        st.markdown(f"""
        <div class="{css_class}">
            <span style="font-family: 'JetBrains Mono'; color: #58a6ff; font-weight: 600;">
                {over_str}
            </span>
            &nbsp;&nbsp;{ball.get('commentary', '')}
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# Sidebar — What If? Analysis + Controls
# ============================================================================

def render_sidebar():
    """Render the sidebar with What If analysis and controls."""
    data = st.session_state.dashboard_data

    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <span style="font-size: 1.3rem; font-weight: 800; color: #e6edf3;">
            🔮 What If?
        </span>
    </div>
    """, unsafe_allow_html=True)

    what_if_scenarios = data.get("what_if", [])
    for scenario in what_if_scenarios:
        impact_class = "whatif-positive" if scenario.get("impact") == "positive" \
            else "whatif-negative"
        india_pct = scenario.get("india_win_prob", 0) * 100
        nz_pct = scenario.get("nz_win_prob", 0) * 100

        st.sidebar.markdown(f"""
        <div class="whatif-card {impact_class}">
            <div style="font-weight: 700; font-size: 0.95rem; color: #e6edf3;">
                {scenario.get('scenario', '')}
            </div>
            <div style="font-size: 0.8rem; color: #8b949e; margin: 0.3rem 0;">
                {scenario.get('description', '')}
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                <span style="color: #ff9500; font-family: 'JetBrains Mono';
                       font-weight: 700;">
                    🇮🇳 {india_pct:.1f}%
                </span>
                <span style="color: #58a6ff; font-family: 'JetBrains Mono';
                       font-weight: 700;">
                    🇳🇿 {nz_pct:.1f}%
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---- CONTROLS ----
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="font-weight: 700; font-size: 0.9rem; color: #8b949e;
         letter-spacing: 1px; text-transform: uppercase; margin-bottom: 0.5rem;">
        ⚙️ Controls
    </div>
    """, unsafe_allow_html=True)

    # Auto-refresh toggle
    st.session_state.auto_refresh = st.sidebar.toggle(
        "Auto-Refresh (10s)", value=st.session_state.auto_refresh
    )

    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now", width="stretch"):
        st.session_state.dashboard_data = st.session_state.simulator.refresh()
        st.session_state.refresh_count += 1
        st.rerun()

    # Match info
    st.sidebar.markdown("---")
    ms = data.get("match_state", {})
    st.sidebar.markdown(f"""
    <div style="font-size: 0.8rem; color: #8b949e; text-align: center;">
        Phase: <span style="color: #58a6ff; font-weight: 600;">{ms.get('phase', '—')}</span>
        <br>Striker: <span style="color: #ff9500; font-weight: 600;">{ms.get('striker', '—')}</span>
        <br>Bowler: <span style="color: #3fb950; font-weight: 600;">{ms.get('bowler', '—')}</span>
        <br><br>
        Refreshes: {st.session_state.refresh_count}
        <br>10,000 Monte Carlo sims/cycle
    </div>
    """, unsafe_allow_html=True)

    # Prediction confidence
    pred = data.get("prediction", {})
    pcts = pred.get("percentiles", {})
    if pcts:
        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        <div style="font-weight: 700; font-size: 0.9rem; color: #8b949e;
             letter-spacing: 1px; text-transform: uppercase; margin-bottom: 0.5rem;">
            📊 Score Confidence Interval
        </div>
        """, unsafe_allow_html=True)

        current_score = ms.get("score", 0)
        st.sidebar.markdown(f"""
        <div style="font-family: 'JetBrains Mono'; font-size: 0.85rem; color: #e6edf3;">
            10th pct: {current_score + pcts.get('p10', 0):.0f}<br>
            25th pct: {current_score + pcts.get('p25', 0):.0f}<br>
            <span style="color: #ff9500; font-weight: 700;">
            Median: {current_score + pcts.get('p50', 0):.0f}</span><br>
            75th pct: {current_score + pcts.get('p75', 0):.0f}<br>
            90th pct: {current_score + pcts.get('p90', 0):.0f}
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# Main Render
# ============================================================================

def main():
    """Main function to render the dashboard."""
    setup_page()
    apply_custom_styles()
    init_session_state()
    
    render_dashboard()
    render_sidebar()

    # Auto-refresh mechanism
    if st.session_state.auto_refresh:
        time.sleep(10)
        st.session_state.dashboard_data = st.session_state.simulator.refresh()
        st.session_state.refresh_count += 1
        st.rerun()

if __name__ == "__main__":
    main()
