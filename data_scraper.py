"""
data_scraper.py — ESPN Cricinfo Ball-by-Ball Scraper + Demo Mode

Scrapes live commentary from ESPN Cricinfo for ball-by-ball data.
Falls back to a realistic demo simulation when no live match is available,
perfect for showcasing the system before the actual match day.
"""

import requests
from bs4 import BeautifulSoup
import time
import random
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime
import streamlit as st


@dataclass
class BallEvent:
    """Represents a single ball delivery in a cricket match."""
    over: float          # e.g., 12.3 means 3rd ball of 13th over
    runs: int            # runs scored off the bat
    extras: int          # extras (wides, no-balls, etc.)
    is_wicket: bool      # whether a wicket fell
    is_boundary: bool    # 4 or 6
    is_six: bool         # specifically a six
    batter: str          # striker's name
    bowler: str          # bowler's name
    non_striker: str     # non-striker's name
    commentary: str      # text description
    total_score: int     # team total after this ball
    total_wickets: int   # total wickets fallen


class CricAPIScraper:
    """
    Fetches live data from CricAPI instead of scraping ESPN.
    Uses Streamlit secrets for API key management.
    """
    def __init__(self, match_id: Optional[str] = None):
        self.API_KEY = st.secrets.get("CRICAPI_KEY", "")
        if not self.API_KEY:
            print("[CricAPI] WARNING: CRICAPI_KEY missing from secrets!")
            
        self.session = requests.Session()
        self.cached_score = 0
        self.cached_wickets = 0
        self.cached_overs = 0.0
        
        # Validate or find a live match ID
        self.match_id = match_id
        try:
            r = self.session.get(f"https://api.cricapi.com/v1/currentMatches?apikey={self.API_KEY}", timeout=10)
            if r.status_code == 200:
                matches = r.json().get("data", [])
                
                # If we have a match ID, ensure it's valid
                if self.match_id:
                    valid = any(m["id"] == self.match_id for m in matches)
                    if not valid:
                        self.match_id = None
                
                # If no match ID or it was invalid, pick the first one with a live score
                if not self.match_id:
                    with_score = [m for m in matches if m.get("score")]
                    if with_score:
                        # Prioritize IND vs NZ match
                        ind_nz = [m for m in with_score if "India" in m.get("name", "") and "New Zealand" in m.get("name", "")]
                        if ind_nz:
                            self.match_id = ind_nz[0]["id"]
                        else:
                            self.match_id = with_score[0]["id"]
        except Exception as e:
            print(f"[CricAPI] Error finding match: {e}")

    def fetch_live_commentary(self) -> List[BallEvent]:
        """Fetch the latest score from CricAPI and create a synthetic event for the app."""
        if not self.match_id:
            return []

        try:
            url = f"https://api.cricapi.com/v1/match_info?apikey={self.API_KEY}&offset=0&id={self.match_id}"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                score_data = data.get("data", {}).get("score", [])
                if not score_data:
                    return []
                
                # Try to get India's innings score, fallback to the latest
                india_score = next((s for s in score_data if "India" in s.get("inning", "")), None)
                if not india_score:
                    india_score = score_data[0]
                
                runs = india_score.get("r", 0)
                wickets = india_score.get("w", 0)
                overs = float(india_score.get("o", 0.0))
                
                # Only return an event if the score or over advanced
                if runs > self.cached_score or wickets > self.cached_wickets or overs > self.cached_overs:
                    run_diff = max(0, runs - self.cached_score)
                    is_wicket = wickets > self.cached_wickets
                    
                    self.cached_score = runs
                    self.cached_wickets = wickets
                    self.cached_overs = overs
                    
                    commentary = f"Live Update: Score pushes to {runs}/{wickets} after {overs} overs."
                    if is_wicket:
                        commentary = f"WICKET! Big moment. Score is now {runs}/{wickets}."
                    elif run_diff > 0:
                        commentary = f"{run_diff} runs added to the total. Score: {runs}/{wickets}."

                    event = BallEvent(
                        over=overs,
                        runs=run_diff,
                        extras=0,
                        is_wicket=is_wicket,
                        is_boundary=run_diff in [4, 6],
                        is_six=(run_diff == 6),
                        batter="Batter",
                        bowler="Bowler",
                        non_striker="?",
                        commentary=commentary,
                        total_score=runs,
                        total_wickets=wickets
                    )
                    return [event]
                    
        except Exception as e:
            print(f"[CricAPI] Live fetch failed: {e}")

        return []


class DemoMatchSimulator:
    """
    Simulates a realistic IND vs NZ 2nd innings chase for demo purposes.
    Generates ball-by-ball data that looks like a real tense chase.

    Perfect for recording demo reels before the actual match!
    """

    # India batting lineup for the chase
    IND_BATTERS = [
        "Rohit Sharma", "Shubman Gill", "Virat Kohli", "Shreyas Iyer",
        "KL Rahul", "Hardik Pandya", "Ravindra Jadeja", "Ravichandran Ashwin",
        "Kuldeep Yadav", "Jasprit Bumrah", "Mohammed Siraj"
    ]

    # NZ bowling attack
    NZ_BOWLERS = [
        "Trent Boult", "Tim Southee", "Matt Henry",
        "Mitchell Santner", "Glenn Phillips", "Rachin Ravindra"
    ]

    def __init__(self, target: int = 268, speed_factor: float = 1.0):
        """
        Args:
            target: Target score for India to chase (1st innings NZ score + 1)
            speed_factor: 1.0 = real-time pacing, 0.1 = 10x faster for demo
        """
        self.target = target if target != 268 else 185  # Default T20 target
        self.speed_factor = speed_factor

        # Match state
        self.score = 0
        self.wickets = 0
        self.balls_bowled = 0
        self.current_over = 0
        self.ball_in_over = 0
        self.batter_idx = 0
        self.non_striker_idx = 1
        self.next_batter_idx = 2
        self.current_bowler_idx = 0

        # Per-batter stats for realism
        self.batter_runs: Dict[str, int] = {}
        self.batter_balls: Dict[str, int] = {}

        # Match phases affect scoring patterns
        self.events_generated: List[BallEvent] = []

        # Pre-generate some early overs for instant display
        self._pregame_simulation()

    def _pregame_simulation(self):
        """Pre-generate first few overs so the dashboard has data immediately."""
        # Simulate 5 overs of realistic cricket
        for _ in range(30):
            self._generate_next_ball()

    def _get_phase_probabilities(self) -> List[float]:
        """
        Return T20 outcome probabilities based on match phase.
        Phases: Powerplay (0-6), Middle (6-16), Death (16-20)
        """
        overs = self.balls_bowled / 6

        if self.wickets >= 7:
            # Tail-enders: more dots and wickets
            #              0     1     2     3     4     6     W
            return [0.35, 0.25, 0.10, 0.02, 0.08, 0.05, 0.15]

        if overs < 6:
            # Powerplay: High risk, high reward
            return [0.22, 0.30, 0.10, 0.02, 0.20, 0.10, 0.06]
        elif overs < 16:
            # Middle overs: Rotation + boundaries
            return [0.28, 0.35, 0.12, 0.02, 0.12, 0.05, 0.06]
        else:
            # Death overs: Full slog mode
            run_rate = self.score / max(overs, 1)
            required_rr = (self.target - self.score) / max((20 - overs) , 0.1)

            if required_rr > run_rate * 1.5:
                # Under pressure: more risky shots
                return [0.20, 0.20, 0.10, 0.03, 0.18, 0.14, 0.15]
            else:
                # Comfortable: controlled aggression
                return [0.22, 0.28, 0.14, 0.04, 0.16, 0.08, 0.08]

    def _generate_next_ball(self) -> Optional[BallEvent]:
        """Generate the next ball event with realistic T20 dynamics."""
        if self.balls_bowled >= 120 or self.wickets >= 10:
            return None  # T20 innings over (120 balls)

        if self.score >= self.target:
            return None  # India wins

        probs = self._get_phase_probabilities()
        outcomes = [0, 1, 2, 3, 4, 6, -1]  # -1 = wicket
        result = random.choices(outcomes, weights=probs, k=1)[0]

        striker = self.IND_BATTERS[self.batter_idx]
        non_striker = self.IND_BATTERS[self.non_striker_idx]
        bowler = self.NZ_BOWLERS[self.current_bowler_idx % len(self.NZ_BOWLERS)]

        is_wicket = (result == -1)
        runs = max(result, 0)

        if is_wicket:
            self.wickets += 1
            commentary = f"OUT! {bowler} gets {striker}! {self._random_dismissal()} India {self.score}/{self.wickets}"
            if self.next_batter_idx < len(self.IND_BATTERS):
                self.batter_idx = self.next_batter_idx
                self.next_batter_idx += 1
        else:
            self.score += runs
            self.batter_runs[striker] = self.batter_runs.get(striker, 0) + runs
            self.batter_balls[striker] = self.batter_balls.get(striker, 0) + 1
            commentary = self._generate_commentary(striker, bowler, runs)

            # Rotate strike on odd runs
            if runs % 2 == 1:
                self.batter_idx, self.non_striker_idx = self.non_striker_idx, self.batter_idx

        self.balls_bowled += 1
        self.ball_in_over += 1

        over_display = self.current_over + (self.ball_in_over / 10.0)

        # End of over: rotate strike and change bowler
        if self.ball_in_over >= 6:
            self.ball_in_over = 0
            self.current_over += 1
            self.batter_idx, self.non_striker_idx = self.non_striker_idx, self.batter_idx
            self.current_bowler_idx += 1

        event = BallEvent(
            over=over_display,
            runs=runs,
            extras=0,
            is_wicket=is_wicket,
            is_boundary=(runs in [4, 6]),
            is_six=(runs == 6),
            batter=striker,
            bowler=bowler,
            non_striker=non_striker,
            commentary=commentary,
            total_score=self.score,
            total_wickets=self.wickets
        )
        self.events_generated.append(event)
        return event

    def _random_dismissal(self) -> str:
        """Generate a random realistic dismissal type."""
        dismissals = [
            "Caught at mid-off!", "Clean bowled!", "LBW!",
            "Caught behind!", "Caught at deep square leg!",
            "Run out! Terrible mix-up!", "Caught at cover!",
            "Edged and gone! Slip takes it!", "Bowled through the gate!",
            "Stumped! Down the track and missed!"
        ]
        return random.choice(dismissals)

    def _generate_commentary(self, batter: str, bowler: str, runs: int) -> str:
        """Generate realistic ball commentary."""
        if runs == 0:
            dots = [
                f"Dot ball! {bowler} keeps it tight to {batter}",
                f"Defended solidly by {batter}",
                f"Good length, {batter} leaves it alone",
                f"Beaten! {bowler} gets one past the edge",
                f"Played to the fielder, no run"
            ]
            return random.choice(dots)
        elif runs == 1:
            return random.choice([
                f"Single taken by {batter}, rotates the strike",
                f"Pushed to mid-on for a quick single",
                f"Dabbed to third man, easy single"
            ])
        elif runs == 2:
            return random.choice([
                f"Two runs! {batter} works it through midwicket",
                f"Good running between the wickets, two taken",
                f"Placed into the gap, they come back for two"
            ])
        elif runs == 3:
            return random.choice([
                f"Three runs! Misfield at the boundary",
                f"Driven hard, overthrow gives them three"
            ])
        elif runs == 4:
            return random.choice([
                f"FOUR! {batter} drives it through the covers! Magnificent!",
                f"FOUR! Cut shot races to the boundary!",
                f"FOUR! Pulled powerfully by {batter}!",
                f"FOUR! Edges past the keeper to the fence!"
            ])
        elif runs == 6:
            return random.choice([
                f"SIX! {batter} launches it into the stands! 🚀",
                f"SIX! Massive hit by {batter} over long-on!",
                f"SIX! Scooped over fine leg! Audacious!",
                f"SIX! {batter} goes downtown! What a shot! 💥"
            ])
        return f"{runs} run(s) scored by {batter}"

    def get_next_ball(self) -> Optional[BallEvent]:
        """Get the next ball event (call this every refresh cycle)."""
        return self._generate_next_ball()

    def get_all_events(self) -> List[BallEvent]:
        """Get all ball events generated so far."""
        return self.events_generated.copy()

    def get_match_summary(self) -> dict:
        """Get current match summary."""
        overs = self.current_over + (self.ball_in_over / 6.0)
        return {
            "score": self.score,
            "wickets": self.wickets,
            "overs": round(overs, 1),
            "target": self.target,
            "runs_remaining": self.target - self.score,
            "balls_remaining": 300 - self.balls_bowled,
            "current_run_rate": round(self.score / max(overs, 0.1), 2),
            "required_run_rate": round(
                (self.target - self.score) / max((20 - overs) / 6, 0.1) * 6, 2
            ) if overs < 20 else 0,
            "batting_team": "India",
            "bowling_team": "New Zealand"
        }

    def is_match_time(self) -> bool:
        """Check if it's currently past the match start time (March 8th, 7 PM)."""
        match_start = datetime(2026, 3, 8, 19, 0, 0)
        return datetime.now() >= match_start


class DataManager:
    """
    Unified data interface — tries live scraping first, falls back to demo mode.
    This makes the system work seamlessly whether there's a live match or not.
    """

    def __init__(self, match_id: Optional[str] = None, demo_mode: bool = True,
                 target: int = 185):
        # Auto-switch: if it's match time, force live mode (unless match_id is missing)
        self.match_start_time = datetime(2026, 3, 8, 19, 0, 0)
        
        # Determine initial mode: if match_id is provided AND (not demo_mode OR it's match time)
        if match_id and (not demo_mode or datetime.now() >= self.match_start_time):
            self.mode = "live"
        else:
            self.mode = "demo"

        self.match_id = match_id
        self.demo_mode = demo_mode
        self.target = target

        self.scraper = CricAPIScraper(match_id) if match_id else None
        self.demo = DemoMatchSimulator(target=target) if self.mode == "demo" else None

        self.events: List[BallEvent] = []

        # Initialize with current events
        if self.mode == "live" and self.scraper:
            if not self.scraper.match_id:
                 # API didn't return any live matches at all
                 self.mode = "demo"
                 self.demo = DemoMatchSimulator(target=self.target)
                 self.events = self.demo.get_all_events()
            else:
                 live_events = self.scraper.fetch_live_commentary()
                 if live_events:
                     self.events = live_events
                 elif self.scraper.cached_score == 0 and self.scraper.cached_wickets == 0:
                     # Live fetch returned nothing and score is 0, fallback to demo
                     self.mode = "demo"
                     self.demo = DemoMatchSimulator(target=self.target)
                     self.events = self.demo.get_all_events()
                 else:
                     self.events = []
        
        if self.mode == "demo" and self.demo:
            self.events = self.demo.get_all_events()

    def refresh(self) -> List[BallEvent]:
        """Refresh data — check for auto-switch and fetch new balls."""
        # Check for auto-switch from demo to live
        if self.mode == "demo" and self.match_id and datetime.now() >= self.match_start_time:
            print("[DataManager] Auto-switching to Live Mode! Match has started.")
            self.mode = "live"
            self.scraper = CricAPIScraper(self.match_id)
            # Fetch live events
            live_events = self.scraper.fetch_live_commentary()
            if live_events:
                self.events = live_events
                return self.events
            # If live fails, we'll hit the fallback below
        if self.mode == "live" and self.scraper:
            new_events = self.scraper.fetch_live_commentary()
            if new_events:
                self.events = new_events
                return self.events

        if self.demo:
            # Generate 1-3 new balls per refresh (simulates realistic pacing)
            num_new = random.randint(1, 3)
            for _ in range(num_new):
                ball = self.demo.get_next_ball()
                if ball is None:
                    break
            self.events = self.demo.get_all_events()

        return self.events

    def get_latest_events(self, n: int = 10) -> List[BallEvent]:
        """Get the last N ball events."""
        return self.events[-n:]

    def get_all_events(self) -> List[BallEvent]:
        """Get all events."""
        return self.events

    def is_demo(self) -> bool:
        return self.mode == "demo"
