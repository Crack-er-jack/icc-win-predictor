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
        self.api_outage = False
        
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
                
                # Detect which innings is active
                current_inning = score_data[-1] 
                inning_name = current_inning.get("inning", "")
                
                # Detect target during 2nd innings chase
                is_second_innings = len(score_data) > 1 or "2nd Inning" in inning_name
                detected_target = 0
                if is_second_innings and len(score_data) > 1:
                    first_inn = score_data[0]
                    detected_target = int(first_inn.get("r", 0)) + 1
                
                runs = current_inning.get("r", 0)
                wickets = current_inning.get("w", 0)
                overs = float(current_inning.get("o", 0.0))
                
                # Only return an event if the score or over advanced
                if runs > self.cached_score or wickets > self.cached_wickets or overs > self.cached_overs or (is_second_innings and self.cached_score == 0):
                    run_diff = max(0, runs - self.cached_score) if not (is_second_innings and self.cached_score == 0) else runs
                    is_wicket = wickets > self.cached_wickets
                    
                    self.cached_score = runs
                    self.cached_wickets = wickets
                    self.cached_overs = overs
                    
                    commentary = f"Live Update: {inning_name} - {runs}/{wickets} ({overs} ov)"
                    if detected_target > 0:
                        commentary += f" [TARGET: {detected_target}]"
                    if is_wicket:
                        commentary = f"WICKET! {inning_name} score is now {runs}/{wickets}."
                    
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
                else:
                    self.api_outage = False # We got data, but no new balls
            else:
                self.api_outage = True
        except Exception as e:
            print(f"[CricAPI] Live fetch failed: {e}")
            self.api_outage = True

        return []


class DemoMatchSimulator:
    """
    Simulates a realistic IND vs NZ 2nd innings chase for demo purposes.
    Generates ball-by-ball data that looks like a real tense chase.
    """

    # New Zealand batting lineup for the chase
    NZ_BATTERS = [
        "Devon Conway", "Rachin Ravindra", "Kane Williamson", "Daryl Mitchell",
        "Glenn Phillips", "Mark Chapman", "Mitchell Santner", "Tom Latham",
        "Kyle Jamieson", "Tim Southee", "Trent Boult"
    ]

    # India bowling attack
    IND_BOWLERS = [
        "Jasprit Bumrah", "Mohammed Siraj", "Mohammed Shami",
        "Ravindra Jadeja", "Kuldeep Yadav", "Hardik Pandya"
    ]

    def __init__(self, target: int = 256, speed_factor: float = 1.0, start_2nd_innings: bool = True):
        """
        Args:
            target: Target score for India to chase (1st innings score + 1)
            speed_factor: 1.0 = real-time pacing, 0.1 = 10x faster for demo
            start_2nd_innings: If True, starts the simulation from the 2nd innings chase.
        """
        self.target = target
        self.speed_factor = speed_factor
        self.start_2nd_innings = start_2nd_innings

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
        num_balls = 30 if not self.start_2nd_innings else 6
        for _ in range(num_balls):
            self._generate_next_ball()

    def _get_phase_probabilities(self) -> List[float]:
        """T20 outcome probabilities based on match phase."""
        overs = self.balls_bowled / 6
        if self.wickets >= 7:
            return [0.35, 0.25, 0.10, 0.02, 0.08, 0.05, 0.15]
        if overs < 6:
            return [0.22, 0.30, 0.10, 0.02, 0.20, 0.10, 0.06]
        elif overs < 16:
            return [0.28, 0.35, 0.12, 0.02, 0.12, 0.05, 0.06]
        else:
            run_rate = self.score / max(overs, 1)
            required_rr = (self.target - self.score) / max((20 - overs) , 0.1)
            if required_rr > run_rate * 1.5:
                return [0.20, 0.20, 0.10, 0.03, 0.18, 0.14, 0.15]
            else:
                return [0.22, 0.28, 0.14, 0.04, 0.16, 0.08, 0.08]

    def _generate_next_ball(self) -> Optional[BallEvent]:
        """Generate the next ball event."""
        if self.balls_bowled >= 120 or self.wickets >= 10:
            return None
        if self.score >= self.target:
            return None

        probs = self._get_phase_probabilities()
        outcomes = [0, 1, 2, 3, 4, 6, -1]
        result = random.choices(outcomes, weights=probs, k=1)[0]

        striker = self.NZ_BATTERS[self.batter_idx]
        non_striker = self.NZ_BATTERS[self.non_striker_idx]
        bowler = self.IND_BOWLERS[self.current_bowler_idx % len(self.IND_BOWLERS)]

        is_wicket = (result == -1)
        runs = max(result, 0)

        if is_wicket:
            self.wickets += 1
            commentary = f"OUT! {bowler} gets {striker}! {self._random_dismissal()} NZ {self.score}/{self.wickets}"
            if self.next_batter_idx < len(self.NZ_BATTERS):
                self.batter_idx = self.next_batter_idx
                self.next_batter_idx += 1
        else:
            self.score += runs
            self.batter_runs[striker] = self.batter_runs.get(striker, 0) + runs
            self.batter_balls[striker] = self.batter_balls.get(striker, 0) + 1
            commentary = self._generate_commentary(striker, bowler, runs)
            if runs % 2 == 1:
                self.batter_idx, self.non_striker_idx = self.non_striker_idx, self.batter_idx

        self.balls_bowled += 1
        self.ball_in_over += 1
        over_display = self.current_over + (self.ball_in_over / 10.0)

        if self.ball_in_over >= 6:
            self.ball_in_over = 0
            self.current_over += 1
            self.batter_idx, self.non_striker_idx = self.non_striker_idx, self.batter_idx
            self.current_bowler_idx += 1

        event = BallEvent(
            over=over_display, runs=runs, extras=0, is_wicket=is_wicket,
            is_boundary=(runs in [4, 6]), is_six=(runs == 6),
            batter=striker, bowler=bowler, non_striker=non_striker,
            commentary=commentary, total_score=self.score, total_wickets=self.wickets
        )
        self.events_generated.append(event)
        return event

    def _random_dismissal(self) -> str:
        dismissals = ["Caught at mid-off!", "Clean bowled!", "LBW!", "Caught behind!", "Caught deep!", "Run out!", "Caught cover!"]
        return random.choice(dismissals)

    def _generate_commentary(self, batter: str, bowler: str, runs: int) -> str:
        if runs == 0: return f"Dot ball by {bowler} to {batter}"
        elif runs == 4: return f"FOUR! {batter} hits a boundary!"
        elif runs == 6: return f"SIX! {batter} clears the ropes!"
        return f"{runs} run(s) scored by {batter}"

    def get_next_ball(self) -> Optional[BallEvent]:
        return self._generate_next_ball()

    def get_all_events(self) -> List[BallEvent]:
        return self.events_generated.copy()


class DataManager:
    def __init__(self, match_id: Optional[str] = None, demo_mode: bool = True, target: int = 256):
        self.match_start_time = datetime(2026, 3, 8, 19, 0, 0)
        self.match_id = match_id
        self.demo_mode = demo_mode
        self.target = target
        self.scraper = CricAPIScraper(match_id) if match_id else None
        
        # Determine mode: prioritize live if match_id or demo_mode=False
        if (match_id or not demo_mode) and datetime.now() >= self.match_start_time:
            self.mode = "live"
        else:
            self.mode = "demo"
        
        self.demo = DemoMatchSimulator(target=target) if self.mode == "demo" else None
        self.events: List[BallEvent] = []

        if self.mode == "live" and self.scraper:
            # Try to get data once, if it fails but demo_mode=False, we STAY in live mode 
            # and just show an outage warning instead of reverting to demo.
            live_events = self.scraper.fetch_live_commentary()
            if live_events: 
                self.events = live_events
            elif not self.scraper.match_id and demo_mode: # Only revert if demo allowed
                 self.mode = "demo"
                 self.demo = DemoMatchSimulator(target=self.target)
                 self.events = self.demo.get_all_events()
        
        if self.mode == "demo" and self.demo:
            self.events = self.demo.get_all_events()

    def refresh(self) -> List[BallEvent]:
        if self.mode == "demo" and self.match_id and datetime.now() >= self.match_start_time:
            self.mode = "live"
            self.scraper = CricAPIScraper(self.match_id)
            live_events = self.scraper.fetch_live_commentary()
            if live_events:
                self.events = live_events
                return self.events

        if self.mode == "live" and self.scraper:
            new_events = self.scraper.fetch_live_commentary()
            if new_events:
                # IMPORTANT: Append to history, don't overwrite!
                self.events.extend(new_events)
                self.scraper.api_outage = False
                return self.events
            elif self.scraper.api_outage:
                return self.events

        if self.demo:
            num_new = random.randint(1, 3)
            for _ in range(num_new):
                ball = self.demo.get_next_ball()
                if ball is None: break
            self.events = self.demo.get_all_events()

        return self.events

    def get_all_events(self) -> List[BallEvent]:
        return self.events

    def is_demo(self) -> bool:
        return self.mode == "demo"
