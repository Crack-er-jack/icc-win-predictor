"""
match_state.py — Cricket Match State Management

Maintains the complete state of a cricket match, computes derived statistics,
and provides the interface between raw ball events and the ML/simulation layers.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np


@dataclass
class PlayerStats:
    """Individual player statistics for the current innings."""
    name: str
    runs: int = 0
    balls_faced: int = 0
    fours: int = 0
    sixes: int = 0
    is_out: bool = False
    dismissal: str = ""

    @property
    def strike_rate(self) -> float:
        return (self.runs / self.balls_faced * 100) if self.balls_faced > 0 else 0.0


@dataclass
class BowlerStats:
    """Individual bowler statistics for the current innings."""
    name: str
    overs: float = 0.0
    runs_conceded: int = 0
    wickets: int = 0
    maidens: int = 0
    dots: int = 0

    @property
    def economy(self) -> float:
        return (self.runs_conceded / self.overs) if self.overs > 0 else 0.0


@dataclass
class MatchState:
    """
    Complete state of a cricket match at any given point.
    This is the central data structure consumed by GNN and Monte Carlo modules.
    """

    # Core match info
    batting_team: str = "India"
    bowling_team: str = "New Zealand"
    innings: int = 1  # 1st innings
    target: int = 256   # Target for NZ chase
    first_innings_score: int = 255 # India's final score

    # Current score state
    score: int = 0
    wickets: int = 0
    overs_completed: int = 0
    balls_in_current_over: int = 0

    def __post_init__(self):
        """Force 2nd innings state if target is provided (e.g., NZ Chase)."""
        # Hardcode India vs NZ 2nd Innings state
        if self.target == 256:
            self.innings = 2
            self.batting_team = "New Zealand"
            self.bowling_team = "India"
            self.first_innings_score = 255
            # User requested 70/4 for now
            self.score = 70
            self.wickets = 4
            self.overs_completed = 10 # Estimate 10 overs for 70 runs
            self.balls_in_current_over = 0
            
        self._compute_derived_stats()

    # Derived chase metrics
    runs_remaining: int = 0
    balls_remaining: int = 120  # 20 overs × 6 balls for T20
    wickets_left: int = 10

    # Run rates
    current_run_rate: float = 0.0
    required_run_rate: float = 0.0

    # Current players on field
    striker: str = "Rohit Sharma"
    non_striker: str = "Yashasvi Jaiswal"
    bowler: str = "Trent Boult"

    # Ball-by-ball history
    ball_history: List[dict] = field(default_factory=list)

    # Recent over tracking for momentum
    recent_overs: List[int] = field(default_factory=list)  # runs per over

    # Partnership tracking
    current_partnership_runs: int = 0
    current_partnership_balls: int = 0

    # Player stats
    batter_stats: Dict[str, PlayerStats] = field(default_factory=dict)
    bowler_stats: Dict[str, BowlerStats] = field(default_factory=dict)

    @property
    def total_balls_bowled(self) -> int:
        """Get the true number of balls bowled, accounting for synthetic live events."""
        if not self.ball_history:
            return 0
        
        last_event = self.ball_history[-1]
        last_over = last_event.get("over", 0.0)
        completed_overs = int(last_over)
        balls_in_over = int(round((last_over - completed_overs) * 10))
        
        # Use the highest of raw history length or the over-based calculation
        return max(len(self.ball_history), completed_overs * 6 + balls_in_over)

    def update_from_ball_events(self, events: list):
        """
        Update the match state from a list of BallEvent objects.
        Recomputes all derived metrics.
        """
        if not events:
            return

        # Reset and recompute from all events
        self.ball_history = []
        self.batter_stats = {}
        self.bowler_stats = {}
        self.recent_overs = []
        current_over_runs = 0
        current_over_balls = 0
        last_over_num = -1

        for event in events:
            # Detect innings transition (over reset)
            if self.ball_history and event.over < self.ball_history[-1]["over"]:
                if self.innings == 1:
                    # Capture 1st innings state
                    self.first_innings_score = self.score
                    self.target = self.score + 1
                    self.innings = 2
                    # Swap teams
                    self.batting_team, self.bowling_team = self.bowling_team, self.batting_team
                    # Reset current innings stats for the chase
                    self.ball_history = []
                    self.score = 0
                    self.wickets = 0
                    self.batter_stats = {}
                    self.bowler_stats = {}
            
            # Additional check: If target is not set but commentary says so
            if "[TARGET:" in event.commentary and self.target == 0:
                try:
                    parts = event.commentary.split("[TARGET:")
                    self.target = int(parts[1].split("]")[0].strip())
                    self.innings = 2
                    self.batting_team, self.bowling_team = self.bowling_team, self.batting_team
                except:
                    pass
            
            # Track ball history
            self.ball_history.append({
                "over": event.over,
                "runs": event.runs,
                "is_wicket": event.is_wicket,
                "batter": event.batter,
                "bowler": event.bowler,
                "is_boundary": event.is_boundary,
                "is_six": event.is_six,
                "commentary": event.commentary
            })

            # Update batter stats
            if event.batter not in self.batter_stats:
                self.batter_stats[event.batter] = PlayerStats(name=event.batter)
            bs = self.batter_stats[event.batter]
            bs.balls_faced += 1
            if not event.is_wicket:
                bs.runs += event.runs
                if event.runs == 4:
                    bs.fours += 1
                elif event.runs == 6:
                    bs.sixes += 1
            else:
                bs.is_out = True

            # Update bowler stats
            if event.bowler not in self.bowler_stats:
                self.bowler_stats[event.bowler] = BowlerStats(name=event.bowler)

            # Track overs for momentum
            over_num = int(event.over)
            if over_num != last_over_num:
                if last_over_num >= 0:
                    self.recent_overs.append(current_over_runs)
                current_over_runs = 0
                current_over_balls = 0
                last_over_num = over_num

            current_over_runs += event.runs
            current_over_balls += 1

        # Add the current (incomplete) over
        if current_over_balls > 0:
            self.recent_overs.append(current_over_runs)

        # Update core state from the last event
        last = events[-1]
        self.score = last.total_score
        self.wickets = last.total_wickets
        self.striker = last.batter if not last.is_wicket else last.non_striker
        self.non_striker = last.non_striker if not last.is_wicket else "Next Batter"
        self.bowler = last.bowler

        # Compute derived metrics
        self._compute_derived_stats()

    def _compute_derived_stats(self):
        """Recompute all derived statistics from current state."""
        effective_balls = self.total_balls_bowled
        
        self.overs_completed = effective_balls // 6
        self.balls_in_current_over = effective_balls % 6

        overs_decimal = effective_balls / 6.0

        if self.innings == 2 and self.target > 0:
            self.runs_remaining = max(self.target - self.score, 0)
            remaining_overs = max(120 - effective_balls, 0) / 6.0
            self.required_run_rate = (
                round(self.runs_remaining / remaining_overs, 2)
                if remaining_overs > 0 else 999.99
            )
        else:
            self.runs_remaining = 0
            self.required_run_rate = 0.0

        self.balls_remaining = max(120 - effective_balls, 0)
        self.wickets_left = max(10 - self.wickets, 0)

        # Current run rate
        self.current_run_rate = (
            round(self.score / overs_decimal, 2) if overs_decimal > 0 else 0.0
        )

    def get_momentum(self, last_n_overs: int = 2) -> float:
        """
        Calculate momentum based on recent overs.
        Positive = batting team accelerating, Negative = slowing down.
        Returns the difference between recent RR and overall RR.
        """
        if len(self.recent_overs) < last_n_overs:
            return 0.0

        recent = self.recent_overs[-last_n_overs:]
        recent_rr = sum(recent) / last_n_overs
        return round(recent_rr - self.current_run_rate, 2)

    def get_over_by_over_runs(self) -> List[int]:
        """Get runs scored in each completed over."""
        if len(self.recent_overs) > 1:
            return self.recent_overs[:-1]  # exclude current incomplete over
        return self.recent_overs

    def get_feature_vector(self) -> np.ndarray:
        """
        Convert match state to a feature vector for ML models.
        Returns normalized features suitable for neural network input.
        """
        features = np.array([
            self.score / 350.0,                    # normalized score
            self.wickets / 10.0,                   # normalized wickets
            self.runs_remaining / 350.0,           # normalized runs remaining
            self.balls_remaining / 300.0,           # normalized balls remaining
            self.current_run_rate / 12.0,           # normalized CRR
            self.required_run_rate / 15.0,          # normalized RRR
            self.wickets_left / 10.0,               # normalized wickets left
            self.get_momentum() / 6.0,             # normalized momentum
            self.current_partnership_runs / 100.0,  # partnership progress
            1.0 if self.balls_remaining < 60 else 0.0  # death overs flag
        ], dtype=np.float32)

        return features

    def get_match_phase(self) -> str:
        """Determine current match phase."""
        overs = self.total_balls_bowled / 6.0
        if overs < 6:
            return "Powerplay"
        elif overs < 15:
            return "Middle Overs"
        else:
            return "Death Overs"

    def is_innings_complete(self) -> bool:
        """Check if the innings is over."""
        if self.innings == 2 and self.target > 0 and self.score >= self.target:
            return True
        return (
            self.wickets >= 10 or
            self.balls_remaining <= 0
        )

    def get_result(self) -> Optional[str]:
        """Get match result if innings is complete."""
        if not self.is_innings_complete():
            return None
        if self.innings == 1:
            return f"Innings Break. Target for {self.bowling_team} is {self.score + 1}"
        if self.score >= self.target:
            return f"{self.batting_team} win by {self.wickets_left} wickets!"
        elif self.wickets >= 10 or self.balls_remaining <= 0:
            return f"{self.bowling_team} win by {self.runs_remaining} runs!"
        return None

    def to_dict(self) -> dict:
        """Serialize match state to dictionary."""
        return {
            "batting_team": self.batting_team,
            "bowling_team": self.bowling_team,
            "innings": self.innings,
            "score": self.score,
            "wickets": self.wickets,
            "overs": f"{self.overs_completed}.{self.balls_in_current_over}",
            "target": self.target,
            "first_innings_score": self.first_innings_score,
            "runs_remaining": self.runs_remaining,
            "balls_remaining": self.balls_remaining,
            "wickets_left": self.wickets_left,
            "current_run_rate": self.current_run_rate,
            "required_run_rate": self.required_run_rate,
            "striker": self.striker,
            "non_striker": self.non_striker,
            "bowler": self.bowler,
            "momentum": self.get_momentum(),
            "phase": self.get_match_phase()
        }
