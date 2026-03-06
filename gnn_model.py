"""
gnn_model.py — Graph Neural Network for Cricket Ball Outcome Prediction

Uses PyTorch Geometric to model player interactions as a graph:
- Nodes = players (batters + bowlers) with statistical features
- Edges = batter vs bowler matchup relationships
- Output = probability distribution over ball outcomes (0,1,2,3,4,6,wicket)

The model uses Graph Attention Networks (GAT) to learn which matchup
interactions matter most for predicting the next delivery outcome.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

# Try importing PyTorch Geometric — fall back gracefully if not installed
try:
    from torch_geometric.nn import GATConv, global_mean_pool
    from torch_geometric.data import Data
    HAS_PYGEOMETRIC = True
except ImportError:
    HAS_PYGEOMETRIC = False
    print("[GNN] PyTorch Geometric not found. Using fallback MLP model.")


# ============================================================================
# Player Database — Synthetic but Realistic Stats for IND & NZ
# ============================================================================

class PlayerDatabase:
    """
    Database of player statistics used as node features in the GNN.
    Stats are synthetic but calibrated to real-world ODI performance ranges.

    Node features per player (5 features):
        [batting_avg, strike_rate, recent_form, economy_rate, bowling_avg]

    For batters: economy_rate and bowling_avg are set to neutral values.
    For bowlers: batting_avg and strike_rate reflect their lower-order ability.
    """

    def __init__(self):
        # India squad — batting stats calibrated to ODI averages
        self.india_players: Dict[str, Dict] = {
            "Rohit Sharma": {
                "batting_avg": 49.2, "strike_rate": 92.5, "recent_form": 0.82,
                "economy_rate": 6.5, "bowling_avg": 65.0, "role": "batter",
                "historical_matchups": {"Trent Boult": 0.08, "Tim Southee": 0.10,
                                         "Matt Henry": 0.07, "Mitchell Santner": 0.06,
                                         "Glenn Phillips": 0.05, "Rachin Ravindra": 0.04}
            },
            "Shubman Gill": {
                "batting_avg": 55.8, "strike_rate": 100.3, "recent_form": 0.88,
                "economy_rate": 7.0, "bowling_avg": 70.0, "role": "batter",
                "historical_matchups": {"Trent Boult": 0.09, "Tim Southee": 0.07,
                                         "Matt Henry": 0.08, "Mitchell Santner": 0.05,
                                         "Glenn Phillips": 0.06, "Rachin Ravindra": 0.05}
            },
            "Virat Kohli": {
                "batting_avg": 58.7, "strike_rate": 94.2, "recent_form": 0.90,
                "economy_rate": 7.0, "bowling_avg": 75.0, "role": "batter",
                "historical_matchups": {"Trent Boult": 0.06, "Tim Southee": 0.05,
                                         "Matt Henry": 0.07, "Mitchell Santner": 0.04,
                                         "Glenn Phillips": 0.03, "Rachin Ravindra": 0.04}
            },
            "Shreyas Iyer": {
                "batting_avg": 47.5, "strike_rate": 97.8, "recent_form": 0.75,
                "economy_rate": 7.5, "bowling_avg": 80.0, "role": "batter",
                "historical_matchups": {"Trent Boult": 0.10, "Tim Southee": 0.09,
                                         "Matt Henry": 0.08, "Mitchell Santner": 0.11,
                                         "Glenn Phillips": 0.07, "Rachin Ravindra": 0.06}
            },
            "KL Rahul": {
                "batting_avg": 50.1, "strike_rate": 88.5, "recent_form": 0.78,
                "economy_rate": 7.0, "bowling_avg": 70.0, "role": "batter",
                "historical_matchups": {"Trent Boult": 0.07, "Tim Southee": 0.08,
                                         "Matt Henry": 0.09, "Mitchell Santner": 0.06,
                                         "Glenn Phillips": 0.05, "Rachin Ravindra": 0.05}
            },
            "Hardik Pandya": {
                "batting_avg": 32.4, "strike_rate": 117.5, "recent_form": 0.72,
                "economy_rate": 5.8, "bowling_avg": 35.0, "role": "allrounder",
                "historical_matchups": {"Trent Boult": 0.11, "Tim Southee": 0.10,
                                         "Matt Henry": 0.09, "Mitchell Santner": 0.07,
                                         "Glenn Phillips": 0.08, "Rachin Ravindra": 0.06}
            },
            "Ravindra Jadeja": {
                "batting_avg": 33.8, "strike_rate": 87.3, "recent_form": 0.70,
                "economy_rate": 4.9, "bowling_avg": 36.5, "role": "allrounder",
                "historical_matchups": {"Trent Boult": 0.09, "Tim Southee": 0.08,
                                         "Matt Henry": 0.10, "Mitchell Santner": 0.06,
                                         "Glenn Phillips": 0.07, "Rachin Ravindra": 0.05}
            },
            "Ravichandran Ashwin": {
                "batting_avg": 16.5, "strike_rate": 72.3, "recent_form": 0.55,
                "economy_rate": 4.5, "bowling_avg": 30.2, "role": "bowler",
                "historical_matchups": {"Trent Boult": 0.15, "Tim Southee": 0.14,
                                         "Matt Henry": 0.13, "Mitchell Santner": 0.12,
                                         "Glenn Phillips": 0.10, "Rachin Ravindra": 0.09}
            },
            "Kuldeep Yadav": {
                "batting_avg": 8.5, "strike_rate": 65.0, "recent_form": 0.40,
                "economy_rate": 5.1, "bowling_avg": 27.8, "role": "bowler",
                "historical_matchups": {"Trent Boult": 0.20, "Tim Southee": 0.18,
                                         "Matt Henry": 0.17, "Mitchell Santner": 0.15,
                                         "Glenn Phillips": 0.14, "Rachin Ravindra": 0.12}
            },
            "Jasprit Bumrah": {
                "batting_avg": 4.2, "strike_rate": 55.0, "recent_form": 0.30,
                "economy_rate": 4.2, "bowling_avg": 22.5, "role": "bowler",
                "historical_matchups": {"Trent Boult": 0.25, "Tim Southee": 0.22,
                                         "Matt Henry": 0.20, "Mitchell Santner": 0.18,
                                         "Glenn Phillips": 0.16, "Rachin Ravindra": 0.15}
            },
            "Mohammed Siraj": {
                "batting_avg": 3.5, "strike_rate": 50.0, "recent_form": 0.25,
                "economy_rate": 5.4, "bowling_avg": 28.5, "role": "bowler",
                "historical_matchups": {"Trent Boult": 0.28, "Tim Southee": 0.25,
                                         "Matt Henry": 0.22, "Mitchell Santner": 0.20,
                                         "Glenn Phillips": 0.18, "Rachin Ravindra": 0.16}
            }
        }

        # New Zealand bowlers
        self.nz_players: Dict[str, Dict] = {
            "Trent Boult": {
                "batting_avg": 15.0, "strike_rate": 70.0, "recent_form": 0.82,
                "economy_rate": 4.8, "bowling_avg": 27.1, "role": "bowler"
            },
            "Tim Southee": {
                "batting_avg": 17.0, "strike_rate": 82.0, "recent_form": 0.75,
                "economy_rate": 5.3, "bowling_avg": 30.5, "role": "bowler"
            },
            "Matt Henry": {
                "batting_avg": 10.0, "strike_rate": 60.0, "recent_form": 0.78,
                "economy_rate": 5.1, "bowling_avg": 26.8, "role": "bowler"
            },
            "Mitchell Santner": {
                "batting_avg": 22.0, "strike_rate": 85.0, "recent_form": 0.70,
                "economy_rate": 4.7, "bowling_avg": 35.2, "role": "bowler"
            },
            "Glenn Phillips": {
                "batting_avg": 35.0, "strike_rate": 100.0, "recent_form": 0.68,
                "economy_rate": 5.9, "bowling_avg": 42.0, "role": "allrounder"
            },
            "Rachin Ravindra": {
                "batting_avg": 42.0, "strike_rate": 95.0, "recent_form": 0.80,
                "economy_rate": 5.5, "bowling_avg": 38.0, "role": "allrounder"
            }
        }

        # Build player index (name → integer ID)
        all_players = list(self.india_players.keys()) + list(self.nz_players.keys())
        self.player_index: Dict[str, int] = {
            name: i for i, name in enumerate(all_players)
        }
        self.index_to_player: Dict[int, str] = {
            i: name for name, i in self.player_index.items()
        }
        self.num_players = len(all_players)

    def get_player_features(self, name: str) -> np.ndarray:
        """Get normalized feature vector for a player."""
        stats = self.india_players.get(name, self.nz_players.get(name, None))
        if stats is None:
            # Unknown player — return average features
            return np.array([30.0/60, 80.0/150, 0.5, 5.5/8, 35.0/50], dtype=np.float32)

        return np.array([
            stats["batting_avg"] / 60.0,       # normalize to [0, 1]
            stats["strike_rate"] / 150.0,       # normalize to [0, 1]
            stats["recent_form"],                # already [0, 1]
            stats["economy_rate"] / 8.0,         # normalize to [0, 1]
            stats["bowling_avg"] / 50.0          # normalize to [0, 1]
        ], dtype=np.float32)

    def get_edge_features(self, batter: str, bowler: str) -> np.ndarray:
        """
        Get edge features for a batter-bowler matchup.
        Returns: [dismissal_probability, historical_matchup_difficulty, balls_faced_norm]
        """
        batter_stats = self.india_players.get(batter, {})
        matchups = batter_stats.get("historical_matchups", {})

        dismissal_prob = matchups.get(bowler, 0.10)  # default 10% dismissal rate
        # Synthetic matchup difficulty based on bowler economy vs batter SR
        bowler_stats = self.nz_players.get(bowler, {})
        bowler_eco = bowler_stats.get("economy_rate", 5.5)
        batter_sr = batter_stats.get("strike_rate", 85.0)

        difficulty = bowler_eco / max(batter_sr / 100.0 * 6, 1.0)  # normalized
        balls_faced = np.random.uniform(0.1, 0.8)  # synthetic

        return np.array([dismissal_prob, difficulty, balls_faced], dtype=np.float32)

    def get_player_id(self, name: str) -> int:
        """Get integer ID for a player name."""
        return self.player_index.get(name, 0)


# ============================================================================
# GNN Model using PyTorch Geometric
# ============================================================================

if HAS_PYGEOMETRIC:
    class CricketGNN(nn.Module):
        """
        Graph Attention Network for predicting ball outcome probabilities.

        Architecture:
            Input → GATConv(5→32) → ReLU → GATConv(32→32) → ReLU
            → Global Mean Pool → Concat with match features
            → MLP(42→64→32→7) → Softmax

        The GAT layers learn to weight batter-bowler matchup importance.
        The global pool aggregates player-level features into a match-level
        representation, which is concatenated with match state features
        before the final prediction MLP.
        """

        def __init__(self, node_features: int = 5, edge_features: int = 3,
                     hidden_dim: int = 32, match_features: int = 10,
                     num_outcomes: int = 7):
            super(CricketGNN, self).__init__()

            # Graph Attention layers — learn matchup importance
            self.gat1 = GATConv(node_features, hidden_dim, heads=2,
                                concat=False, edge_dim=edge_features)
            self.gat2 = GATConv(hidden_dim, hidden_dim, heads=2,
                                concat=False, edge_dim=edge_features)

            # Match state encoder
            self.match_encoder = nn.Sequential(
                nn.Linear(match_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )

            # Final prediction MLP
            self.predictor = nn.Sequential(
                nn.Linear(hidden_dim + hidden_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_outcomes)
            )

            # Initialize with smart defaults for cricket
            self._initialize_weights()

        def _initialize_weights(self):
            """
            Initialize the final layer bias to produce realistic
            cricket outcome probabilities before any training.
            Default: ~30% dot, 25% single, 12% double, 3% triple,
                     12% four, 8% six, 10% wicket
            """
            with torch.no_grad():
                final_layer = self.predictor[-1]
                # Log-prob initialization for T20 defaults (Par: ~9 RPO)
                # T20 avg: dot(30%), 1(35%), 4(12%), 6(8%), W(5%)
                default_probs = torch.tensor([0.30, 0.35, 0.08, 0.02, 0.12, 0.08, 0.05])
                final_layer.bias.copy_(torch.log(default_probs + 1e-8))

        def forward(self, data: Data, match_features: torch.Tensor) -> torch.Tensor:
            """
            Forward pass through the GNN.

            Args:
                data: PyG Data object with x (node features), edge_index, edge_attr
                match_features: Tensor of shape (1, match_features) with match state

            Returns:
                Tensor of shape (1, 7) — outcome probabilities
            """
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            batch = data.batch if hasattr(data, 'batch') and data.batch is not None \
                else torch.zeros(x.size(0), dtype=torch.long)

            # Graph attention layers
            x = F.relu(self.gat1(x, edge_index, edge_attr))
            x = F.dropout(x, p=0.2, training=self.training)
            x = F.relu(self.gat2(x, edge_index, edge_attr))

            # Global mean pooling — aggregate player features
            x = global_mean_pool(x, batch)  # (1, hidden_dim)

            # Encode match state
            match_enc = self.match_encoder(match_features)  # (1, hidden_dim)

            # Concatenate and predict
            combined = torch.cat([x, match_enc], dim=1)  # (1, 2*hidden_dim)
            logits = self.predictor(combined)              # (1, 7)

            # Softmax for probability distribution
            probs = F.softmax(logits, dim=1)
            return probs


# ============================================================================
# Fallback MLP Model (when PyTorch Geometric is not available)
# ============================================================================

class FallbackMLP(nn.Module):
    """
    Simple MLP fallback when PyTorch Geometric isn't installed.
    Takes concatenated player + match features and predicts outcomes.
    """

    def __init__(self, input_dim: int = 25, num_outcomes: int = 7):
        super(FallbackMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_outcomes)
        )
        # Initialize with cricket-realistic defaults (as in GNN)
        with torch.no_grad():
            default_probs = torch.tensor([0.45, 0.35, 0.07, 0.01, 0.08, 0.02, 0.02])
            self.net[-1].bias.copy_(torch.log(default_probs + 1e-8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return F.softmax(logits, dim=1)


# ============================================================================
# Outcome Predictor — Unified Interface
# ============================================================================

class OutcomePredictor:
    """
    High-level interface for predicting ball outcomes.
    Automatically uses GNN if PyTorch Geometric is available, otherwise MLP.

    Usage:
        predictor = OutcomePredictor()
        probs = predictor.predict(match_state)
        # probs = [p_0, p_1, p_2, p_3, p_4, p_6, p_wicket]
    """

    OUTCOMES = [0, 1, 2, 3, 4, 6, "wicket"]
    OUTCOME_LABELS = ["Dot Ball", "Single", "Double", "Triple", "Four", "Six", "Wicket"]

    def __init__(self):
        self.db = PlayerDatabase()
        self.use_gnn = HAS_PYGEOMETRIC

        if self.use_gnn:
            self.model = CricketGNN()
            self.model.eval()
            print("[GNN] Cricket Graph Attention Network initialized ✓")
        else:
            self.model = FallbackMLP()
            self.model.eval()
            print("[GNN] Fallback MLP model initialized ✓")

    def predict(self, match_state) -> np.ndarray:
        """
        Predict outcome probabilities for the next ball.

        Args:
            match_state: MatchState object with current game situation

        Returns:
            numpy array of shape (7,) — probabilities for [0,1,2,3,4,6,wicket]
        """
        with torch.no_grad():
            if self.use_gnn:
                return self._predict_gnn(match_state)
            else:
                return self._predict_mlp(match_state)

    def _predict_gnn(self, match_state) -> np.ndarray:
        """Generate prediction using the full GNN model."""
        # Build graph from current match state
        graph_data = self._build_graph(match_state)
        match_features = torch.tensor(
            match_state.get_feature_vector(), dtype=torch.float32
        ).unsqueeze(0)

        probs = self.model(graph_data, match_features)
        probs = probs.squeeze().numpy()

        # Phase-based adjustment — make predictions context-aware
        probs = self._apply_match_context(probs, match_state)
        return probs

    def _predict_mlp(self, match_state) -> np.ndarray:
        """Generate prediction using fallback MLP."""
        # Concatenate striker, bowler, and match features
        striker_feats = self.db.get_player_features(match_state.striker)
        bowler_feats = self.db.get_player_features(match_state.bowler)
        match_feats = match_state.get_feature_vector()

        combined = np.concatenate([striker_feats, bowler_feats, match_feats])
        x = torch.tensor(combined, dtype=torch.float32).unsqueeze(0)

        probs = self.model(x).squeeze().numpy()
        probs = self._apply_match_context(probs, match_state)
        return probs

    def _build_graph(self, match_state) -> 'Data':
        """
        Build a PyTorch Geometric graph from the current match state.

        Nodes: active batter (striker), non-striker, and current bowler.
        We also include 2 additional context nodes (other bowlers) for richer graph.
        Edges: batter ↔ bowler matchup relationships (bidirectional).
        """
        # Collect active players
        active_players = [
            match_state.striker,
            match_state.non_striker,
            match_state.bowler
        ]

        # Add a couple more NZ bowlers for graph richness
        nz_bowlers = list(self.db.nz_players.keys())
        for b in nz_bowlers:
            if b != match_state.bowler and len(active_players) < 5:
                active_players.append(b)

        num_nodes = len(active_players)

        # Node features
        node_features = []
        for player in active_players:
            node_features.append(self.db.get_player_features(player))
        x = torch.tensor(np.array(node_features), dtype=torch.float32)

        # Edges: connect batters to all bowlers (bidirectional)
        edge_sources = []
        edge_targets = []
        edge_features = []

        batter_indices = [0, 1]  # striker, non-striker
        bowler_indices = list(range(2, num_nodes))  # bowlers

        for bi in batter_indices:
            for bwi in bowler_indices:
                batter_name = active_players[bi]
                bowler_name = active_players[bwi]

                # Forward edge: batter → bowler
                edge_sources.append(bi)
                edge_targets.append(bwi)
                edge_features.append(self.db.get_edge_features(batter_name, bowler_name))

                # Backward edge: bowler → batter
                edge_sources.append(bwi)
                edge_targets.append(bi)
                edge_features.append(self.db.get_edge_features(batter_name, bowler_name))

        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_features), dtype=torch.float32)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def _apply_match_context(self, probs: np.ndarray, match_state) -> np.ndarray:
        """
        Apply match-phase adjustments to make predictions more realistic.
        This accounts for game situations that the untrained model can't capture.
        """
        overs = len(match_state.ball_history) / 6.0
        wickets = match_state.wickets
        rrr = match_state.required_run_rate
        crr = match_state.current_run_rate

        # Death overs adjustment — more boundaries and wickets
        if overs > 40:
            probs[4] *= 1.3   # more fours
            probs[5] *= 1.5   # more sixes
            probs[6] *= 1.2   # more wickets (risky shots)
            probs[0] *= 0.7   # fewer dots

        # Pressure adjustment — high required rate
        if rrr > crr * 1.3 and overs > 20:
            probs[4] *= 1.2
            probs[5] *= 1.4
            probs[6] *= 1.3  # desperation wickets
            probs[0] *= 0.8

        # New batter adjustment — cautious start
        striker_stats = match_state.batter_stats.get(match_state.striker)
        if striker_stats and striker_stats.balls_faced < 10:
            probs[0] *= 1.3  # more dots
            probs[1] *= 1.2  # more singles
            probs[5] *= 0.5  # fewer sixes
            probs[6] *= 1.2  # vulnerable early

        # Tail-ender adjustment
        if wickets >= 7:
            probs[0] *= 1.3
            probs[6] *= 1.5
            probs[5] *= 0.6

        # Re-normalize
        probs = np.clip(probs, 0.01, None)  # floor at 1%
        probs = probs / probs.sum()

        return probs

    def get_prediction_summary(self, probs: np.ndarray) -> Dict:
        """Format prediction probabilities with labels."""
        return {
            label: float(np.round(p, 4))
            for label, p in zip(self.OUTCOME_LABELS, probs)
        }
