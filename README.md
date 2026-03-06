# 🏏 ICC Win Predictor — Real-Time Cricket Match Prediction

AI-powered real-time win probability predictor for **IND vs NZ** cricket matches, combining **Graph Neural Networks**, **Monte Carlo simulation**, and an interactive **Streamlit dashboard**.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=flat-square&logo=streamlit)

---

## ✨ Features

- **🧠 GNN Prediction** — Graph Attention Network models batter vs bowler matchups to predict next-ball outcomes
- **🎲 Monte Carlo Simulation** — 10,000 parallel match simulations in under 1 second (NumPy vectorized)
- **📊 Real-Time Dashboard** — Premium dark-themed Streamlit dashboard with Plotly charts
- **📱 Mobile Responsive** — Dynamic CSS layout optimized for perfect viewing on smartphones
- **🔮 "What If?" Analysis** — Instant scenario testing ("What if next ball is a wicket?")
- **📡 Live Scraping** — ESPN Cricinfo ball-by-ball commentary scraper
- **🏏 T20 Calibrated** — Specifically tuned for T20 dynamics (20 overs, ~185 par score)
- **⏰ Auto-Switch** — Moves from Demo to Live Mode automatically at **7:00 PM on March 8th, 2026**
- **🎬 Demo Mode** — Realistic simulated match for showcasing before match day

---

## 🗂️ Architecture

```
icc-win-predictor/
├── data_scraper.py    # ESPN Cricinfo scraper + demo match simulator
├── match_state.py     # Match state management & derived statistics
├── gnn_model.py       # Graph Attention Network (PyTorch Geometric)
├── monte_carlo.py     # Vectorized Monte Carlo simulator (NumPy)
├── simulator.py       # Orchestrator + "What If?" analysis engine
├── dashboard.py       # Streamlit + Plotly real-time dashboard
├── main.py            # CLI entry point for the pipeline
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

### Data Flow

```
ESPN Cricinfo / Demo  →  Match State  →  GNN Predictor  →  Monte Carlo  →  Dashboard
     ↓                      ↓                ↓                  ↓              ↓
  Ball events         Score, RR, etc    Outcome probs      Win probs     Plotly charts
```

---

## 🚀 Quick Start

### 1. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note on PyTorch Geometric**: If `torch-geometric` fails to install, the system automatically falls back to an MLP model. The GNN is not required for the system to work.
>
> To install PyTorch Geometric properly:
> ```bash
> pip install torch-geometric
> # If issues, install from wheels:
> # https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
> ```

### 3. Run the Dashboard

```bash
streamlit run dashboard.py
```

This launches the full real-time dashboard at `http://localhost:8501`.

### 4. Run CLI-Only Pipeline

```bash
python main.py --interval 5 --max-refreshes 20
```

### 5. Run Monte Carlo Benchmark

```bash
python monte_carlo.py
```

---

## 📋 CLI Options

```bash
python main.py --help

Options:
  --match-id TEXT      ESPN Cricinfo match ID for live scraping
  --demo               Run in demo mode (default: on)
  --target INT         Target score for the chase (default: 268)
  --interval INT       Refresh interval in seconds (default: 10)
  --max-refreshes INT  Max number of refresh cycles
  --dashboard          Launch Streamlit dashboard instead
```

---

## 🎬 Demo Mode

The system includes a **full demo mode** that simulates a realistic IND vs NZ chase. This is perfect for:

- 📹 Recording reels before match day
- 🧪 Testing and development
- 🎓 Learning and demonstrations

Demo mode features:
- Realistic scoring patterns (powerplay, middle, death overs)
- Phase-aware probability distributions
- Actual player names and realistic commentary
- Wickets cause batting order progression
- Run rate pressure dynamics

---

## 🧠 How It Works

### Graph Neural Network

Players are modeled as **graph nodes** with features:
- Batting average, strike rate, recent form
- Economy rate, bowling average

**Batter ↔ bowler edges** encode historical matchup data:
- Dismissal probability
- Matchup difficulty score

The **GAT (Graph Attention Network)** learns which matchups matter most, outputting a 7-class probability distribution:
`[dot, single, double, triple, four, six, wicket]`

### Monte Carlo Simulation

Using GNN probabilities, the engine:
1. Pre-generates all random outcomes (NumPy vectorized)
2. Processes 10,000 × remaining balls in parallel
3. Tracks wickets, scores, and win conditions
4. Returns win probabilities and projected scores

**Performance**: 10,000 simulations typically complete in **< 100ms**.

### "What If?" Scenarios

| Scenario | Description |
|----------|-------------|
| 🔴 Next Ball: Wicket | Removes a wicket and recomputes |
| 🟢 Big Over (12 runs) | Adds 12 runs in 6 balls |
| 🟢 Big Over (18 runs) | Adds 18 runs in 6 balls |
| 🔵 Maiden Over | 6 dot balls, pure pressure |
| 🟡 Boundary Streak | 3 consecutive fours |

---

## 📡 Live Mode (Match Day)

When a live match is on, provide the ESPN Cricinfo match ID:

```bash
streamlit run dashboard.py
# Then modify the simulator init in dashboard.py with your match ID
```

Or via CLI:
```bash
python main.py --match-id 1234567 --no-demo
```

The scraper will attempt to fetch live commentary. If it fails or if it is before match day, it falls back to demo mode automatically.

> [!TIP]
> **Finding the Match ID**: Go to the ESPN Cricinfo match page. The ID is the number in the URL (e.g., `https://www.espncricinfo.com/series/.../match-name-1234567`). Pass `1234567` as the `--match-id`.

---

## 🌐 Deployment (Hosting)

### Why not GitHub Pages?
**GitHub Pages** is a static hosting service. It only supports HTML, CSS, and JavaScript. Our system requires a **Python runtime** to run the Graph Neural Network, Monte Carlo simulations, and the Scraper. Therefore, it cannot be hosted directly on GitHub Pages.

### Recommended Hosting (Free & Easy)

#### 1. Streamlit Cloud (Best)
1. **GitHub Upload**: Create a new repository on GitHub and upload all files from `icc-win-predictor/`.
2. **Sign Up**: Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.
3. **Deploy**:
   - Click **"New app"**.
   - Select your repository and the `main` branch.
   - Set **Main file path** to `streamlit_app.py`.
   - Click **"Deploy!"**.
4. **Environment**: Streamlit Cloud will automatically detect `requirements.txt` and install all dependencies (including PyTorch).

---

## ⏰ Auto-Switch Logic

The system is hard-coded to switch from **Demo Mode** to **Live Mode** at **2026-03-08 19:00:00**.
- **Before Match**: Shows the realistic IND vs NZ demo chase (perfect for reels).
- **After Match Starts**: Automatically attempts to fetch live score via the ESPNCricinfo API.

> [!IMPORTANT]
> To ensure the auto-switch works, make sure your deployment server time is correct. Streamlit Cloud usually uses UTC, so the code handles the comparison based on the target date.

---

## 🎨 Dashboard Features

| Component | Description |
|-----------|-------------|
| Win Probability Gauges | Dual gauge meters for IND/NZ |
| Win Probability Timeline | Line chart tracking changes over time |
| Score Banner | Live score, target, required rate |
| Momentum Indicator | Run rate delta over last 2 overs |
| Runs Per Over | Color-coded bar chart |
| Player Impact | Horizontal bars showing top contributors |
| GNN Prediction | Donut chart of next-ball probabilities |
| Commentary Ticker | Color-coded ball-by-ball feed |
| What If Panel | Sidebar with scenario analysis |
| Score Confidence | Percentile-based score projections |

---

## 📄 License

This project is for educational and demonstration purposes.

---

*Built with ❤️ for cricket and AI*
