"""
monte_carlo.py — Vectorized Monte Carlo Match Simulator

Runs 10,000 simulations of the remaining innings using NumPy vectorization.
Uses GNN-predicted outcome probabilities to sample ball-by-ball results.

Performance target: < 1 second for 10,000 simulations.
"""

import numpy as np
from typing import Tuple, Dict, Optional
import time


class MonteCarloSimulator:
    """
    Vectorized Monte Carlo engine for cricket match simulation.

    Given outcome probabilities and current match state, simulates
    the remainder of the innings 10,000 times in parallel using
    NumPy's vectorized operations for sub-second performance.
    """

    # Outcome values: runs scored per outcome category
    OUTCOME_VALUES = np.array([0, 1, 2, 3, 4, 6, 0])  # last is wicket (0 runs)
    WICKET_INDEX = 6

    def __init__(self, n_simulations: int = 10000, seed: Optional[int] = None):
        """
        Args:
            n_simulations: Number of parallel simulations to run
            seed: Random seed for reproducibility (None for random)
        """
        self.n_sims = n_simulations
        self.rng = np.random.default_rng(seed)

    def simulate(self, outcome_probs: np.ndarray, balls_remaining: int,
                 wickets_remaining: int, runs_remaining: int,
                 target: Optional[int] = None) -> Dict:
        """
        Run full Monte Carlo simulation of the remaining innings.

        Args:
            outcome_probs: shape (7,) — probabilities for [0,1,2,3,4,6,wicket]
            balls_remaining: balls left in the innings
            wickets_remaining: wickets in hand
            runs_remaining: runs needed to win (for chase scenarios)
            target: target score (if chasing)

        Returns:
            Dictionary with:
                - batting_team_win_prob: probability of batting team winning
                - bowling_team_win_prob: probability of bowling team winning
                - draw_prob: probability of a tie
                - projected_score_mean: mean final score addition
                - projected_score_std: std dev of final score addition
                - projected_total_mean: mean total (current + additions)
                - score_distribution: histogram of final scores
                - percentiles: 10th, 25th, 50th, 75th, 90th percentiles
        """
        start_time = time.perf_counter()

        # Validate and normalize probabilities
        probs = np.clip(outcome_probs, 0.001, None)
        probs = probs / probs.sum()

        # Ensure we have enough balls and wickets to simulate
        if balls_remaining <= 0 or wickets_remaining <= 0:
            # Innings already over
            win_prob = 1.0 if runs_remaining <= 0 else 0.0
            return self._make_result(
                win_prob=win_prob,
                scores=np.zeros(self.n_sims),
                runs_remaining=runs_remaining,
                elapsed=time.perf_counter() - start_time
            )

        # ================================================================
        # VECTORIZED SIMULATION
        # All n_sims simulations run in parallel using NumPy
        # ================================================================

        # Pre-generate all random outcomes at once for maximum speed
        # Shape: (n_sims, balls_remaining)
        max_balls = min(balls_remaining, 300)

        # Generate random outcomes for all simulations × all possible balls
        random_outcomes = self._sample_outcomes(probs, self.n_sims, max_balls)

        # Track scores and wickets across simulations (vectorized)
        total_runs = np.zeros(self.n_sims, dtype=np.int32)
        total_wickets = np.zeros(self.n_sims, dtype=np.int32)
        active = np.ones(self.n_sims, dtype=bool)  # which sims are still active

        # Process ball by ball (but all sims in parallel)
        for ball in range(max_balls):
            if not active.any():
                break

            outcomes = random_outcomes[:, ball]  # shape: (n_sims,)

            # Determine which outcomes are wickets
            is_wicket = (outcomes == self.WICKET_INDEX) & active

            # Add runs for non-wicket outcomes
            runs_scored = self.OUTCOME_VALUES[outcomes] * active.astype(np.int32)
            total_runs += runs_scored

            # Process wickets
            total_wickets += is_wicket.astype(np.int32)

            # Deactivate simulations where all wickets fell
            all_out = total_wickets >= wickets_remaining
            active &= ~all_out

            # Deactivate simulations where target is reached (chase won)
            if target is not None:
                target_reached = total_runs >= runs_remaining
                active &= ~target_reached

        # ================================================================
        # COMPUTE RESULTS
        # ================================================================

        elapsed = time.perf_counter() - start_time
        return self._make_result(
            scores=total_runs,
            runs_remaining=runs_remaining,
            elapsed=elapsed,
            total_wickets=total_wickets,
            wickets_remaining=wickets_remaining
        )

    def _sample_outcomes(self, probs: np.ndarray, n_sims: int,
                         n_balls: int) -> np.ndarray:
        """
        Generate random outcomes using cumulative probability sampling.
        This is much faster than np.random.choice for large arrays.

        Returns: shape (n_sims, n_balls) array of outcome indices (0-6)
        """
        # Cumulative probabilities for vectorized sampling
        cum_probs = np.cumsum(probs)

        # Generate uniform random numbers
        randoms = self.rng.random((n_sims, n_balls))

        # Convert to outcome indices via searchsorted (vectorized)
        outcomes = np.searchsorted(cum_probs, randoms).astype(np.int32)

        # Clip to valid range
        outcomes = np.clip(outcomes, 0, 6)

        return outcomes

    def _make_result(self, scores: np.ndarray, runs_remaining: int,
                     elapsed: float, win_prob: Optional[float] = None,
                     total_wickets: Optional[np.ndarray] = None,
                     wickets_remaining: int = 10) -> Dict:
        """Compile simulation results into a structured dictionary."""

        if win_prob is not None:
            # Short-circuit for trivial cases
            return {
                "batting_team_win_prob": win_prob,
                "bowling_team_win_prob": 1.0 - win_prob,
                "draw_prob": 0.0,
                "projected_score_mean": 0,
                "projected_score_std": 0,
                "projected_total_mean": 0,
                "score_distribution": scores,
                "percentiles": {},
                "simulation_time": elapsed,
                "n_simulations": self.n_sims
            }

        # Chase analysis: how many simulations resulted in a win?
        wins = (scores >= runs_remaining).sum()
        ties = (scores == runs_remaining - 1).sum()  # exact tie (rare)

        win_prob = wins / self.n_sims
        tie_prob = ties / self.n_sims
        loss_prob = 1.0 - win_prob - tie_prob

        # Score statistics
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))

        # Percentiles for confidence intervals
        percentiles = {
            "p10": float(np.percentile(scores, 10)),
            "p25": float(np.percentile(scores, 25)),
            "p50": float(np.percentile(scores, 50)),
            "p75": float(np.percentile(scores, 75)),
            "p90": float(np.percentile(scores, 90))
        }

        return {
            "batting_team_win_prob": float(round(win_prob, 4)),
            "bowling_team_win_prob": float(round(loss_prob, 4)),
            "draw_prob": float(round(tie_prob, 4)),
            "projected_score_mean": float(round(mean_score, 1)),
            "projected_score_std": float(round(std_score, 1)),
            "score_distribution": scores,
            "percentiles": percentiles,
            "simulation_time": float(round(elapsed, 4)),
            "n_simulations": self.n_sims,
            "wicket_distribution": {
                "mean_wickets_lost": float(np.mean(total_wickets)) if total_wickets is not None else 0,
                "all_out_pct": float((total_wickets >= wickets_remaining).mean() * 100) if total_wickets is not None else 0
            }
        }

    def simulate_with_dynamic_probs(self, get_probs_fn, balls_remaining: int,
                                     wickets_remaining: int,
                                     runs_remaining: int) -> Dict:
        """
        Advanced simulation where probabilities change based on wickets lost.
        Uses a callback function to get updated probabilities after each wicket.

        This is more realistic but slower. Used for smaller simulation counts.

        Args:
            get_probs_fn: callable(wickets_lost) → np.ndarray of shape (7,)
            balls_remaining: balls left
            wickets_remaining: wickets in hand
            runs_remaining: runs to win
        """
        n_sims = min(self.n_sims, 2000)  # fewer sims for accuracy

        total_runs = np.zeros(n_sims, dtype=np.int32)
        total_wickets = np.zeros(n_sims, dtype=np.int32)
        active = np.ones(n_sims, dtype=bool)

        for ball in range(balls_remaining):
            if not active.any():
                break

            # Get probabilities (could depend on wickets lost in each sim)
            # For simplicity, use the most common wicket count
            median_wickets = int(np.median(total_wickets[active]))
            probs = get_probs_fn(median_wickets)
            probs = np.clip(probs, 0.001, None)
            probs = probs / probs.sum()

            # Sample outcomes for active simulations
            cum_probs = np.cumsum(probs)
            randoms = self.rng.random(n_sims)
            outcomes = np.searchsorted(cum_probs, randoms).astype(np.int32)
            outcomes = np.clip(outcomes, 0, 6)

            is_wicket = (outcomes == self.WICKET_INDEX) & active
            runs_scored = self.OUTCOME_VALUES[outcomes] * active.astype(np.int32)
            total_runs += runs_scored
            total_wickets += is_wicket.astype(np.int32)

            all_out = total_wickets >= wickets_remaining
            target_reached = total_runs >= runs_remaining
            active &= ~all_out & ~target_reached

        elapsed = 0.0  # not timed here
        return self._make_result(
            scores=total_runs,
            runs_remaining=runs_remaining,
            elapsed=elapsed,
            total_wickets=total_wickets,
            wickets_remaining=wickets_remaining
        )


def benchmark():
    """Quick performance benchmark to verify < 1 second requirement."""
    mc = MonteCarloSimulator(n_simulations=10000, seed=42)
    probs = np.array([0.30, 0.25, 0.12, 0.03, 0.12, 0.08, 0.10])

    start = time.perf_counter()
    result = mc.simulate(probs, balls_remaining=180, wickets_remaining=7, runs_remaining=150)
    elapsed = time.perf_counter() - start

    print(f"╔══════════════════════════════════════════╗")
    print(f"║  Monte Carlo Benchmark Results           ║")
    print(f"╠══════════════════════════════════════════╣")
    print(f"║  Simulations:  {result['n_simulations']:>10,}              ║")
    print(f"║  Time:         {elapsed:>10.4f}s              ║")
    print(f"║  Batting Win:  {result['batting_team_win_prob']:>10.1%}              ║")
    print(f"║  Bowling Win:  {result['bowling_team_win_prob']:>10.1%}              ║")
    print(f"║  Avg Score +:  {result['projected_score_mean']:>10.1f}              ║")
    print(f"║  Score Std:    {result['projected_score_std']:>10.1f}              ║")
    print(f"╚══════════════════════════════════════════╝")

    assert elapsed < 1.0, f"Performance requirement failed! Took {elapsed:.3f}s"
    print("✅ Performance requirement met: < 1 second")


if __name__ == "__main__":
    benchmark()
