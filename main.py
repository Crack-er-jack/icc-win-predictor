"""
main.py — Real-Time Cricket Win Prediction Pipeline

Entry point for the ICC Win Predictor system.
Runs the prediction pipeline in a continuous loop, updating every 10 seconds.

Usage:
    # Run the Streamlit dashboard (recommended):
    streamlit run dashboard.py

    # Run CLI-only pipeline:
    python main.py

    # Run Monte Carlo benchmark:
    python monte_carlo.py
"""

import sys
import time
import argparse
from simulator import MatchSimulator


def run_realtime_pipeline(match_id=None, demo_mode=True, target=185,
                          refresh_interval=10, max_refreshes=None):
    """
    Run the real-time prediction pipeline.

    This is the main loop that:
        1. Scrapes/generates latest ball events
        2. Updates match state
        3. Runs GNN prediction
        4. Runs Monte Carlo simulation
        5. Displays results in CLI

    Args:
        match_id: ESPN Cricinfo match ID (None for demo)
        demo_mode: Use demo simulation instead of live scraping
        target: Target score for the chase
        refresh_interval: Seconds between refreshes
        max_refreshes: Limit number of cycles (None = infinite)
    """
    print("\n" + "═" * 60)
    print("  🏏  ICC Win Predictor — Real-Time Pipeline")
    print("═" * 60)
    print(f"  Mode:     {'DEMO' if demo_mode else 'LIVE'}")
    print(f"  Target:   {target}")
    print(f"  Refresh:  Every {refresh_interval}s")
    print(f"  Match ID: {match_id or 'N/A (demo mode)'}")
    print("═" * 60 + "\n")

    # Initialize the simulator
    sim = MatchSimulator(
        match_id=match_id,
        demo_mode=demo_mode,
        target=target,
        n_simulations=10000
    )

    refresh_count = 0

    try:
        while True:
            refresh_count += 1

            if max_refreshes and refresh_count > max_refreshes:
                print("\n🏁 Maximum refreshes reached. Exiting.")
                break

            # Run one prediction cycle
            data = sim.refresh()

            # Clear screen for clean output
            print(f"\n{'─' * 60}")
            print(f"  Refresh #{refresh_count}")
            print(f"{'─' * 60}")

            ms = data.get("match_state", {})
            pred = data.get("prediction", {})

            # Match status check
            if data.get("status") == "completed":
                print(f"\n  🏆 MATCH RESULT: {data.get('result', 'Unknown')}")
                print(f"\n  Final Score: {ms.get('batting_team')} "
                      f"{ms.get('score')}/{ms.get('wickets')} "
                      f"({ms.get('overs')} overs)")
                break

            # Display match state
            print(f"\n  📋 {ms.get('batting_team', 'India')} vs "
                  f"{ms.get('bowling_team', 'New Zealand')}")
            print(f"     Score: {ms.get('score', 0)}/{ms.get('wickets', 0)} "
                  f"({ms.get('overs', '0.0')} overs)")
            print(f"     Need:  {ms.get('runs_remaining', 0)} off "
                  f"{ms.get('balls_remaining', 0)} balls")
            print(f"     CRR:   {ms.get('current_run_rate', 0)} | "
                  f"RRR: {ms.get('required_run_rate', 0)}")
            print(f"     Phase: {ms.get('phase', 'Unknown')}")

            # Display players
            print(f"\n  🏏 Striker:     {ms.get('striker', '—')}")
            print(f"  🏏 Non-Striker: {ms.get('non_striker', '—')}")
            print(f"  ⚾ Bowler:      {ms.get('bowler', '—')}")

            # Win probabilities
            india_wp = pred.get("india_win_prob", 0)
            nz_wp = pred.get("nz_win_prob", 0)
            print(f"\n  ⚡ Win Probability:")
            print(f"     🇮🇳 India:       {india_wp:>6.1%}  "
                  f"{'█' * int(india_wp * 30)}")
            print(f"     🇳🇿 New Zealand: {nz_wp:>6.1%}  "
                  f"{'█' * int(nz_wp * 30)}")

            # Projected score
            projected = pred.get("projected_total", 0)
            proj_std = pred.get("projected_score_std", 0)
            print(f"\n  📈 Projected Total: ~{projected} (±{proj_std:.0f})")
            print(f"  ⏱️  Sim Time:       {pred.get('simulation_time', 0):.4f}s")

            # Momentum
            momentum = data.get("momentum", 0)
            m_icon = "📈" if momentum > 0 else "📉" if momentum < 0 else "➡️"
            print(f"  {m_icon} Momentum:      {momentum:+.1f}")

            # GNN predictions
            outcome_probs = data.get("outcome_probs", {})
            if outcome_probs.get("values"):
                print(f"\n  🧠 GNN Next Ball Prediction:")
                for label, val in zip(outcome_probs["labels"],
                                       outcome_probs["values"]):
                    bar = "▓" * int(val * 40)
                    print(f"     {label:<8} {val:>5.1%}  {bar}")

            # What If scenarios (top 3)
            what_if = data.get("what_if", [])
            if what_if:
                print(f"\n  🔮 What If?")
                for scenario in what_if[:3]:
                    print(f"     {scenario['scenario']}: "
                          f"IND {scenario['india_win_prob']:.1%} | "
                          f"NZ {scenario['nz_win_prob']:.1%}")

            # Recent commentary
            recent = data.get("recent_balls", [])
            if recent:
                print(f"\n  💬 Last {len(recent)} balls:")
                for ball in recent[-3:]:
                    icon = "🔴" if ball.get("is_wicket") else \
                           "🟣" if ball.get("is_six") else \
                           "🟢" if ball.get("is_boundary") else "⚪"
                    print(f"     {icon} {ball.get('commentary', '')}")

            # Wait for next refresh
            if refresh_interval > 0:
                print(f"\n  ⏳ Next refresh in {refresh_interval}s...")
                time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print("\n\n  👋 Pipeline stopped by user. Goodbye!")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="🏏 ICC Win Predictor — Real-Time Cricket Match Prediction"
    )
    parser.add_argument(
        "--match-id", type=str, default=None,
        help="ESPN Cricinfo match ID (e.g. 1412547). If provided, system will auto-switch to Live Mode at 7 PM on March 8th."
    )
    parser.add_argument(
        "--demo", action="store_true", default=True,
        help="Run in demo mode initially (default: True). Will auto-switch to Live if it's match time."
    )
    parser.add_argument(
        "--target", type=int, default=185,
        help="Target score for the chase (default: 185 for T20)"
    )
    parser.add_argument(
        "--interval", type=int, default=10,
        help="Refresh interval in seconds (default: 10)"
    )
    parser.add_argument(
        "--max-refreshes", type=int, default=None,
        help="Maximum number of refresh cycles (default: unlimited)"
    )
    parser.add_argument(
        "--dashboard", action="store_true",
        help="Launch the Streamlit dashboard instead of CLI"
    )

    args = parser.parse_args()

    if args.dashboard:
        import subprocess
        print("🚀 Launching Streamlit dashboard...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"])
    else:
        run_realtime_pipeline(
            match_id=args.match_id,
            demo_mode=args.demo,
            target=args.target,
            refresh_interval=args.interval,
            max_refreshes=args.max_refreshes
        )


if __name__ == "__main__":
    main()
