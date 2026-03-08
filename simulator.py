"""
simulator.py — Orchestrator & "What If?" Analysis Engine

Ties together all components:
    Data Scraper → Match State → GNN Predictor → Monte Carlo Simulator

Also implements the "What If?" analysis module for scenario testing:
    - "What if next ball is a wicket?"
    - "What if next over scores 12 runs?"
    - Custom probability adjustments

Tracks prediction history for trend visualization.
"""

import numpy as np
import time
import copy
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from data_scraper import DataManager, BallEvent
from match_state import MatchState
from gnn_model import OutcomePredictor
from monte_carlo import MonteCarloSimulator


@dataclass
class PredictionSnapshot:
    """A single prediction at a point in time."""
    timestamp: float
    over: float
    india_win_prob: float
    nz_win_prob: float
    projected_score: float
    projected_score_std: float
    score: int
    wickets: int
    momentum: float


class WhatIfAnalyzer:
    """
    "What If?" scenario analysis engine.

    Allows instant recomputation of win probabilities under hypothetical
    scenarios. Used for interactive exploration in the dashboard.
    """

    def __init__(self, mc_simulator: MonteCarloSimulator,
                 predictor: OutcomePredictor):
        self.mc = mc_simulator
        self.predictor = predictor

    def what_if_wicket(self, match_state: MatchState) -> Dict:
        """
        What if the next ball is a wicket?
        Recomputes win probability with one fewer wicket.
        """
        modified = copy.deepcopy(match_state)
        modified.wickets += 1
        modified.wickets_left = max(modified.wickets_left - 1, 0)
        modified.balls_remaining = max(modified.balls_remaining - 1, 0)

        # Recalculate with adjusted state
        probs = self.predictor.predict(modified)
        result = self.mc.simulate(
            probs,
            balls_remaining=modified.balls_remaining,
            wickets_remaining=modified.wickets_left,
            runs_remaining=modified.runs_remaining,
            target=modified.target
        )

        return {
            "scenario": "Next Ball: WICKET 🔴",
            "description": f"India {modified.score}/{modified.wickets} — "
                          f"Lost a key wicket, {modified.wickets_left} left",
            "india_win_prob": result["batting_team_win_prob"],
            "nz_win_prob": result["bowling_team_win_prob"],
            "projected_score": result["projected_score_mean"],
            "impact": "negative"
        }

    def what_if_big_over(self, match_state: MatchState,
                         runs_in_over: int = 12) -> Dict:
        """
        What if the next over scores N runs (default 12)?
        Adjusts score and remaining stats accordingly.
        """
        modified = copy.deepcopy(match_state)
        modified.score += runs_in_over
        modified.runs_remaining = max(modified.target - modified.score, 0)
        modified.balls_remaining = max(modified.balls_remaining - 6, 0)
        modified._compute_derived_stats()

        probs = self.predictor.predict(modified)
        result = self.mc.simulate(
            probs,
            balls_remaining=modified.balls_remaining,
            wickets_remaining=modified.wickets_left,
            runs_remaining=modified.runs_remaining,
            target=modified.target
        )

        return {
            "scenario": f"Next Over: {runs_in_over} RUNS 🟢",
            "description": f"India {modified.score}/{modified.wickets} — "
                          f"Big over! {modified.runs_remaining} needed off "
                          f"{modified.balls_remaining} balls",
            "india_win_prob": result["batting_team_win_prob"],
            "nz_win_prob": result["bowling_team_win_prob"],
            "projected_score": result["projected_score_mean"],
            "impact": "positive"
        }

    def what_if_dot_over(self, match_state: MatchState) -> Dict:
        """What if the next over is a maiden (0 runs)?"""
        modified = copy.deepcopy(match_state)
        modified.balls_remaining = max(modified.balls_remaining - 6, 0)
        modified._compute_derived_stats()

        probs = self.predictor.predict(modified)
        result = self.mc.simulate(
            probs,
            balls_remaining=modified.balls_remaining,
            wickets_remaining=modified.wickets_left,
            runs_remaining=modified.runs_remaining,
            target=modified.target
        )

        return {
            "scenario": "Next Over: MAIDEN 🔵",
            "description": f"India {modified.score}/{modified.wickets} — "
                          f"Pressure builds! {modified.runs_remaining} needed off "
                          f"{modified.balls_remaining} balls",
            "india_win_prob": result["batting_team_win_prob"],
            "nz_win_prob": result["bowling_team_win_prob"],
            "projected_score": result["projected_score_mean"],
            "impact": "negative"
        }

    def what_if_boundary_streak(self, match_state: MatchState,
                                 n_boundaries: int = 3) -> Dict:
        """What if the next 3 balls are all boundaries (4s)?"""
        modified = copy.deepcopy(match_state)
        modified.score += 4 * n_boundaries
        modified.runs_remaining = max(modified.target - modified.score, 0)
        modified.balls_remaining = max(modified.balls_remaining - n_boundaries, 0)
        modified._compute_derived_stats()

        probs = self.predictor.predict(modified)
        result = self.mc.simulate(
            probs,
            balls_remaining=modified.balls_remaining,
            wickets_remaining=modified.wickets_left,
            runs_remaining=modified.runs_remaining,
            target=modified.target
        )

        return {
            "scenario": f"Next {n_boundaries} Balls: ALL FOURS 🟡",
            "description": f"India {modified.score}/{modified.wickets} — "
                          f"Boundary fest! {modified.runs_remaining} needed",
            "india_win_prob": result["batting_team_win_prob"],
            "nz_win_prob": result["bowling_team_win_prob"],
            "projected_score": result["projected_score_mean"],
            "impact": "positive"
        }

    def run_all_scenarios(self, match_state: MatchState) -> List[Dict]:
        """Run all What If scenarios and return results."""
        return [
            self.what_if_wicket(match_state),
            self.what_if_big_over(match_state, runs_in_over=12),
            self.what_if_big_over(match_state, runs_in_over=18),
            self.what_if_dot_over(match_state),
            self.what_if_boundary_streak(match_state, n_boundaries=3),
        ]


class MatchSimulator:
    """
    Main orchestrator that runs the full prediction pipeline.

    Pipeline per refresh cycle:
        1. Scrape/generate latest ball events
        2. Update match state
        3. Run GNN to predict outcome probabilities
        4. Run Monte Carlo simulation
        5. Store prediction snapshot for history
        6. Run "What If?" analysis
    """

    def __init__(self, match_id: Optional[str] = None, demo_mode: bool = True,
                 target: int = 185, n_simulations: int = 10000):
        """
        Initialize all components of the prediction system.

        Args:
            match_id: ESPN Cricinfo match ID (None for demo mode)
            demo_mode: Whether to use simulated match data
            target: Target score for the chase
            n_simulations: Number of Monte Carlo simulations
        """
        print("=" * 60)
        print("🏏  ICC Win Predictor — Initializing Components")
        print("=" * 60)

        # Data layer
        self.data_manager = DataManager(
            match_id=match_id, demo_mode=demo_mode, target=target
        )
        print(f"  ✓ Data Manager ({self.data_manager.mode} mode)")

        # Match state
        self.match_state = MatchState(target=target)
        print(f"  ✓ Match State (target: {target})")

        # GNN predictor
        self.predictor = OutcomePredictor()

        # Monte Carlo simulator
        self.mc_simulator = MonteCarloSimulator(n_simulations=n_simulations)
        print(f"  ✓ Monte Carlo Simulator ({n_simulations:,} simulations)")

        # What If analyzer
        self.what_if = WhatIfAnalyzer(self.mc_simulator, self.predictor)
        print(f"  ✓ What If? Analyzer")

        # Prediction history
        self.history: List[PredictionSnapshot] = []
        self.current_prediction: Optional[Dict] = None
        self.current_probs: Optional[np.ndarray] = None
        self.what_if_results: List[Dict] = []

        # Initialize with existing data
        events = self.data_manager.get_all_events()
        if events:
            self.match_state.update_from_ball_events(events)
            self._run_prediction()

        print(f"\n  🏁 System ready! Mode: {self.data_manager.mode.upper()}")
        print("=" * 60)

    def refresh(self) -> Dict:
        """
        Execute one complete refresh cycle of the prediction pipeline.
        Called every 10 seconds by the main loop or dashboard.

        Returns:
            Dictionary with complete prediction state for the dashboard.
        """
        # Step 1: Get latest ball events
        events = self.data_manager.refresh()

        # Step 2: Update match state
        self.match_state.update_from_ball_events(events)

        # Check if innings is over
        if self.match_state.is_innings_complete():
            result = self.match_state.get_result()
            return {
                "status": "completed",
                "result": result,
                "match_state": self.match_state.to_dict(),
                "history": self._get_history_data(),
                "final_prediction": self.current_prediction
            }

        # Step 3-5: Run prediction pipeline
        self._run_prediction()

        # Step 6: Run What If analysis
        self.what_if_results = self.what_if.run_all_scenarios(self.match_state)

        return self._compile_dashboard_data()

    def _run_prediction(self):
        """Run GNN + Monte Carlo prediction pipeline."""
        # GNN prediction
        self.current_probs = self.predictor.predict(self.match_state)

        # Monte Carlo simulation
        result = self.mc_simulator.simulate(
            self.current_probs,
            balls_remaining=self.match_state.balls_remaining,
            wickets_remaining=self.match_state.wickets_left,
            runs_remaining=self.match_state.runs_remaining,
            target=self.match_state.target
        )

        # Apply starting bias for realism (User request: base case of past matches)
        # If match has just started, bias towards India (slight favorites)
        overs = len(self.match_state.ball_history) / 6.0
        if overs < 2.0:
            # India pre-match favoritism (approx 52%)
            india_base = 0.52
            weight = max(0, (2.0 - overs) / 2.0)
            result["batting_team_win_prob"] = (
                result["batting_team_win_prob"] * (1 - weight) +
                india_base * weight
            )
            result["bowling_team_win_prob"] = 1.0 - result["batting_team_win_prob"]

        self.current_prediction = result

        # Record snapshot
        overs = len(self.match_state.ball_history) / 6.0
        snapshot = PredictionSnapshot(
            timestamp=time.time(),
            over=round(overs, 1),
            india_win_prob=self.current_prediction["batting_team_win_prob"],
            nz_win_prob=self.current_prediction["bowling_team_win_prob"],
            projected_score=self.current_prediction["projected_score_mean"],
            projected_score_std=self.current_prediction["projected_score_std"],
            score=self.match_state.score,
            wickets=self.match_state.wickets,
            momentum=self.match_state.get_momentum()
        )
        self.history.append(snapshot)

    def _compile_dashboard_data(self) -> Dict:
        """Compile all data needed by the dashboard into a single dictionary."""
        # Player impact: batter contributions normalized
        player_impact = {}
        for name, stats in self.match_state.batter_stats.items():
            if stats.runs > 0:
                player_impact[name] = {
                    "runs": stats.runs,
                    "balls": stats.balls_faced,
                    "sr": round(stats.strike_rate, 1),
                    "fours": stats.fours,
                    "sixes": stats.sixes,
                    "impact_score": round(
                        (stats.runs / max(self.match_state.score, 1)) * 100, 1
                    )
                }

        # Sort by impact
        player_impact = dict(
            sorted(player_impact.items(),
                   key=lambda x: x[1]["impact_score"], reverse=True)
        )

        return {
            "status": "live",
            "mode": self.data_manager.mode,
            "match_state": self.match_state.to_dict(),
            "prediction": {
                "india_win_prob": self.current_prediction["batting_team_win_prob"],
                "nz_win_prob": self.current_prediction["bowling_team_win_prob"],
                "projected_score_add": self.current_prediction["projected_score_mean"],
                "projected_score_std": self.current_prediction["projected_score_std"],
                "projected_total": round(
                    self.match_state.score +
                    self.current_prediction["projected_score_mean"]
                ),
                "percentiles": self.current_prediction.get("percentiles", {}),
                "score_distribution": self.current_prediction.get("score_distribution", []).tolist() if isinstance(self.current_prediction.get("score_distribution"), np.ndarray) else [],
                "simulation_time": self.current_prediction.get("simulation_time", 0),
                "all_out_pct": self.current_prediction.get(
                    "wicket_distribution", {}
                ).get("all_out_pct", 0)
            },
            "outcome_probs": {
                "labels": ["Dot", "Single", "Double", "Triple", "Four", "Six", "Wicket"],
                "values": self.current_probs.tolist() if self.current_probs is not None else []
            },
            "history": self._get_history_data(),
            "momentum": self.match_state.get_momentum(),
            "over_by_over": self.match_state.get_over_by_over_runs(),
            "player_impact": player_impact,
            "what_if": self.what_if_results,
            "recent_balls": [
                {
                    "over": b["over"],
                    "runs": b["runs"],
                    "is_wicket": b["is_wicket"],
                    "is_boundary": b["is_boundary"],
                    "commentary": b["commentary"]
                }
                for b in self.match_state.ball_history[-6:]
            ]
        }

    def _get_history_data(self) -> Dict:
        """Format history data for charting."""
        return {
            "overs": [s.over for s in self.history],
            "india_win": [s.india_win_prob for s in self.history],
            "nz_win": [s.nz_win_prob for s in self.history],
            "projected": [s.projected_score for s in self.history],
            "scores": [s.score for s in self.history],
            "wickets": [s.wickets for s in self.history],
            "momentum": [s.momentum for s in self.history]
        }


def run_cli_demo():
    """Run a quick CLI demo of the T20 prediction system."""
    print("\n🏏 ICC Win Predictor — CLI T20 Demo\n")

    sim = MatchSimulator(demo_mode=True, target=185)

    for i in range(5):
        print(f"\n--- Refresh #{i+1} ---")
        data = sim.refresh()

        if data["status"] == "completed":
            print(f"🏆 MATCH RESULT: {data['result']}")
            break

        ms = data["match_state"]
        pred = data["prediction"]

        print(f"📊 {ms['batting_team']}: {ms['score']}/{ms['wickets']} "
              f"({ms['overs']} ov)")
        print(f"   Need {ms['runs_remaining']} off {ms['balls_remaining']} balls")
        print(f"   CRR: {ms['current_run_rate']} | RRR: {ms['required_run_rate']}")
        print(f"\n🎯 Win Probability:")
        print(f"   🇮🇳 India:       {pred['india_win_prob']:.1%}")
        print(f"   🇳🇿 New Zealand: {pred['nz_win_prob']:.1%}")
        print(f"   📈 Projected:    +{pred['projected_score_add']:.0f} runs "
              f"(total ~{pred['projected_total']})")

        if data.get("what_if"):
            print(f"\n🔮 What If?")
            for scenario in data["what_if"][:3]:
                print(f"   {scenario['scenario']}: "
                      f"IND {scenario['india_win_prob']:.1%} | "
                      f"NZ {scenario['nz_win_prob']:.1%}")

        time.sleep(1)

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    run_cli_demo()
