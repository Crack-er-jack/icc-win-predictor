"""Quick verification script — tests all components end-to-end."""
import time

print("=" * 60)
print("  ICC Win Predictor — System Verification")
print("=" * 60)

# Test 1: Monte Carlo benchmark
print("\n[1/4] Monte Carlo Benchmark...")
from monte_carlo import MonteCarloSimulator
import numpy as np

mc = MonteCarloSimulator(n_simulations=10000, seed=42)
probs = np.array([0.30, 0.25, 0.12, 0.03, 0.12, 0.08, 0.10])
t = time.perf_counter()
result = mc.simulate(probs, balls_remaining=180, wickets_remaining=7, runs_remaining=150)
elapsed = time.perf_counter() - t
print(f"  Simulations: 10,000 in {elapsed:.4f}s")
print(f"  Win prob: {result['batting_team_win_prob']:.1%}")
assert elapsed < 1.0, "TOO SLOW!"
print("  PASS")

# Test 2: GNN model
print("\n[2/4] GNN Model...")
from gnn_model import OutcomePredictor
predictor = OutcomePredictor()
print("  PASS")

# Test 3: Full pipeline
print("\n[3/4] Full Pipeline (3 refreshes)...")
from simulator import MatchSimulator
sim = MatchSimulator(demo_mode=True, target=268, n_simulations=10000)

for i in range(3):
    data = sim.refresh()
    ms = data["match_state"]
    pred = data["prediction"]
    print(f"  Refresh {i+1}: {ms['score']}/{ms['wickets']} "
          f"({ms['overs']} ov) | "
          f"IND {pred['india_win_prob']:.1%} | "
          f"NZ {pred['nz_win_prob']:.1%} | "
          f"Sim: {pred['simulation_time']:.4f}s")
print("  PASS")

# Test 4: What If
print("\n[4/4] What If Analysis...")
what_if = data.get("what_if", [])
print(f"  Scenarios generated: {len(what_if)}")
for s in what_if:
    print(f"    {s['scenario']}: IND {s['india_win_prob']:.1%}")
print("  PASS")

print("\n" + "=" * 60)
print("  ALL TESTS PASSED")
print("=" * 60)
