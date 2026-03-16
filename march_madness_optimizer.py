#!/usr/bin/env python3
"""
==============================================================================
  MARCH MADNESS BRACKET OPTIMIZER
  Expected-Value Maximization via Monte Carlo + Game Theory
==============================================================================

  Usage  : python march_madness_optimizer.py --csv teams.csv
           python march_madness_optimizer.py --csv teams.csv --sims 50000
           python march_madness_optimizer.py --demo          # run with mock data
           python march_madness_optimizer.py --scrape        # live Sports-Reference + Vegas
           python march_madness_optimizer.py --scrape --odds-key YOUR_KEY

  See --help for all options.
==============================================================================
"""

import argparse
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _load_env(path: str = ".env") -> Dict[str, str]:
    """Parse a simple KEY=VALUE or KEY = VALUE .env file."""
    env: Dict[str, str] = {}
    env_path = Path(path)
    if not env_path.exists():
        return env
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()
    return env


_ENV = _load_env(os.path.join(os.path.dirname(__file__), ".env"))
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================================
# 0.  CONSTANTS & CONFIGURATION
# ============================================================================

NUM_TEAMS = 64
NUM_GAMES = 63
ROUNDS = 6  # R64, R32, S16, E8, F4, NCG
ROUND_NAMES = [
    "Round of 64",
    "Round of 32",
    "Sweet 16",
    "Elite 8",
    "Final Four",
    "National Championship",
]

# ESPN-standard scoring: 10-20-40-80-160-320
DEFAULT_SCORING = {0: 10, 1: 20, 2: 40, 3: 80, 4: 160, 5: 320}

# Seeds in standard bracket order for each region (1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15)
SEED_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
REGIONS = ["East", "West", "South", "Midwest"]

# Game-theory thresholds
PUBLIC_PICK_PENALTY_THRESHOLD = 0.40   # If a 1-seed is picked by >40% public
PUBLIC_PICK_PENALTY = 0.05             # 5% penalty applied
UNDERVALUED_THRESHOLD = 0.10           # Strong 2/3/4 seed picked by <10% public
UNDERVALUED_BOOST = 1.15               # 15% EV multiplier

# KenPom championship constraint
KENPOM_TOP_N = 25  # Must be top-25 in BOTH AdjO and AdjD to win title
KENPOM_CHAMP_PENALTY = 0.70  # Multiply championship prob by this if violated

# Upset quota minimums (first round)
UPSET_12_OVER_5 = 1
UPSET_11_OVER_6 = 1


# ============================================================================
# 1.  DATA STRUCTURES
# ============================================================================

@dataclass
class Team:
    """Represents a single tournament team with all features."""
    name: str
    seed: int
    region: str
    adj_o: float          # Adjusted Offensive Efficiency (pts/100 poss)
    adj_d: float          # Adjusted Defensive Efficiency (pts/100 poss allowed)
    efg_pct_off: float    # Effective FG% (offense)
    to_pct_off: float     # Turnover % (offense — lower is better)
    orb_pct: float        # Offensive Rebound %
    ftr_off: float        # Free-Throw Rate (offense)
    efg_pct_def: float    # Effective FG% allowed (defense — lower is better)
    to_pct_def: float     # Turnover % forced (defense — higher is better)
    drb_pct: float        # Defensive Rebound % (= 1 - opponent ORB%)
    ftr_def: float        # Free-Throw Rate allowed (defense — lower is better)
    last10_wins: int      # Wins in last 10 games
    public_pick_pct: float  # Public pick % to win tournament (0-1 scale)
    adj_t: float = 68.5   # Adjusted Tempo (possessions per 40 min); default = national avg
    # Derived
    net_efficiency: float = 0.0
    momentum: float = 1.0
    adj_o_rank: int = 0
    adj_d_rank: int = 0
    bracket_slot: int = 0  # 0-63 position in the bracket


# ============================================================================
# 2.  CSV LOADING & VALIDATION
# ============================================================================

REQUIRED_COLUMNS = [
    "team", "seed", "region",
    "adj_o", "adj_d",
    "efg_pct_off", "to_pct_off", "orb_pct", "ftr_off",
    "efg_pct_def", "to_pct_def", "drb_pct", "ftr_def",
    "last10_wins", "public_pick_pct",
]


def load_teams_from_csv(path: str) -> List[Team]:
    """Load and validate team data from CSV."""
    print(f"\n📂  Loading data from: {path}")
    df = pd.read_csv(path)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"\n❌  ERROR: Missing required columns: {missing}")
        print(f"   Required columns: {REQUIRED_COLUMNS}")
        sys.exit(1)

    if len(df) != NUM_TEAMS:
        print(f"\n⚠️   WARNING: Expected {NUM_TEAMS} teams, found {len(df)}.")
        if len(df) < NUM_TEAMS:
            print("   The bracket requires exactly 64 teams. Exiting.")
            sys.exit(1)
        else:
            print(f"   Using first {NUM_TEAMS} rows.")
            df = df.head(NUM_TEAMS)

    teams = []
    for _, row in df.iterrows():
        t = Team(
            name=str(row["team"]).strip(),
            seed=int(row["seed"]),
            region=str(row["region"]).strip(),
            adj_o=float(row["adj_o"]),
            adj_d=float(row["adj_d"]),
            efg_pct_off=float(row["efg_pct_off"]),
            to_pct_off=float(row["to_pct_off"]),
            orb_pct=float(row["orb_pct"]),
            ftr_off=float(row["ftr_off"]),
            efg_pct_def=float(row["efg_pct_def"]),
            to_pct_def=float(row["to_pct_def"]),
            drb_pct=float(row["drb_pct"]),
            ftr_def=float(row["ftr_def"]),
            last10_wins=int(row["last10_wins"]),
            public_pick_pct=float(row["public_pick_pct"]),
            adj_t=float(row["adj_t"]) if "adj_t" in df.columns else 68.5,
        )
        teams.append(t)

    print(f"   ✅  Loaded {len(teams)} teams across {len(set(t.region for t in teams))} regions.")
    return teams


# ============================================================================
# 3.  MOCK DATA GENERATOR (for --demo mode)
# ============================================================================

def generate_mock_teams() -> List[Team]:
    """Generate realistic mock data for 64 tournament teams."""
    np.random.seed(42)
    mock_names = {
        1: ["UConn", "Houston", "Purdue", "North Carolina"],
        2: ["Tennessee", "Marquette", "Iowa State", "Arizona"],
        3: ["Illinois", "Creighton", "Baylor", "Kentucky"],
        4: ["Auburn", "Kansas", "Duke", "Gonzaga"],
        5: ["San Diego State", "Wisconsin", "BYU", "Clemson"],
        6: ["Texas Tech", "South Carolina", "Michigan State", "TCU"],
        7: ["Florida", "Dayton", "Nevada", "Texas"],
        8: ["Mississippi State", "Northwestern", "Utah State", "Nebraska"],
        9: ["Michigan", "Memphis", "Missouri", "Oregon"],
        10: ["Colorado State", "Drake", "Virginia", "New Mexico"],
        11: ["NC State", "Oregon State", "Duquesne", "Pitt"],
        12: ["Grand Canyon", "McNeese", "James Madison", "UAB"],
        13: ["Yale", "Vermont", "Oakland", "Samford"],
        14: ["Morehead State", "Colgate", "Grambling", "Montana St"],
        15: ["Longwood", "Wagner", "South Dakota St", "Kennesaw St"],
        16: ["Stetson", "Howard", "Long Island", "MSVU"],
    }

    teams = []
    for seed in range(1, 17):
        for i, region in enumerate(REGIONS):
            name = mock_names[seed][i]
            # Higher seeds → better efficiency (lower seed number = better)
            base_o = 120 - seed * 2.5 + np.random.normal(0, 2)
            base_d = 88 + seed * 2.0 + np.random.normal(0, 2)
            efg_off = 0.56 - seed * 0.008 + np.random.normal(0, 0.01)
            to_off = 0.15 + seed * 0.003 + np.random.normal(0, 0.008)
            orb = 0.34 - seed * 0.005 + np.random.normal(0, 0.01)
            ftr_off = 0.38 - seed * 0.005 + np.random.normal(0, 0.02)
            efg_def = 0.44 + seed * 0.006 + np.random.normal(0, 0.01)
            to_def = 0.22 - seed * 0.003 + np.random.normal(0, 0.008)
            drb = 0.75 - seed * 0.005 + np.random.normal(0, 0.01)
            ftr_def = 0.27 + seed * 0.004 + np.random.normal(0, 0.02)
            last10 = max(3, min(10, int(11 - seed * 0.5 + np.random.normal(0, 1))))

            # Public pick pct: 1-seeds get most, drops fast
            if seed == 1:
                pub = np.random.uniform(0.15, 0.50)
            elif seed <= 4:
                pub = np.random.uniform(0.02, 0.12)
            elif seed <= 8:
                pub = np.random.uniform(0.005, 0.03)
            else:
                pub = np.random.uniform(0.0, 0.005)

            adj_t = round(max(60.0, min(78.0, 68.5 + np.random.normal(0, 3))), 1)
            t = Team(
                name=name, seed=seed, region=region,
                adj_o=round(base_o, 1), adj_d=round(base_d, 1),
                efg_pct_off=round(efg_off, 3), to_pct_off=round(to_off, 3),
                orb_pct=round(orb, 3), ftr_off=round(ftr_off, 3),
                efg_pct_def=round(efg_def, 3), to_pct_def=round(to_def, 3),
                drb_pct=round(drb, 3), ftr_def=round(ftr_def, 3),
                last10_wins=last10, public_pick_pct=round(pub, 4),
                adj_t=adj_t,
            )
            teams.append(t)
    return teams


# ============================================================================
# 4.  FEATURE ENGINEERING
# ============================================================================

def engineer_features(teams: List[Team]) -> List[Team]:
    """
    Compute derived features: net efficiency, momentum, rankings.
    Assigns bracket slots based on region + seed.
    """
    # Net efficiency
    for t in teams:
        t.net_efficiency = t.adj_o - t.adj_d

    # Momentum: bonus for >75% win rate in last 10
    for t in teams:
        if t.last10_wins >= 8:  # 8/10 = 80%, above 75% threshold
            t.momentum = 1.0 + (t.last10_wins - 7) * 0.03  # up to 1.09 for 10-0
        else:
            t.momentum = 1.0

    # Rank AdjO (descending) and AdjD (ascending = lower is better)
    sorted_by_o = sorted(teams, key=lambda x: x.adj_o, reverse=True)
    sorted_by_d = sorted(teams, key=lambda x: x.adj_d, reverse=False)
    for rank, t in enumerate(sorted_by_o, 1):
        t.adj_o_rank = rank
    for rank, t in enumerate(sorted_by_d, 1):
        t.adj_d_rank = rank

    # Assign bracket slots: 16 per region, ordered by SEED_ORDER
    region_teams = {r: [] for r in REGIONS}
    for t in teams:
        region_teams[t.region].append(t)

    for r_idx, region in enumerate(REGIONS):
        r_teams = sorted(region_teams[region], key=lambda x: SEED_ORDER.index(x.seed))
        for slot_in_region, t in enumerate(r_teams):
            t.bracket_slot = r_idx * 16 + slot_in_region

    return teams


# ============================================================================
# 5.  PROBABILITY MODEL (Logistic Regression on efficiency features)
# ============================================================================

class MatchupModel:
    """
    Trains a Logistic Regression on synthetic historical matchup outcomes
    derived from team efficiency differentials, then predicts P(Team A wins).
    """

    def __init__(self, teams: List[Team]):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=2000, C=1.0)
        self.teams = teams
        self._train(teams)

    def _build_feature_vector(self, a: Team, b: Team) -> np.ndarray:
        """Build a difference-based feature vector for A vs B."""
        return np.array([
            a.adj_o - b.adj_o,
            b.adj_d - a.adj_d,          # lower def eff = better for A's opponent
            a.efg_pct_off - b.efg_pct_off,
            b.to_pct_off - a.to_pct_off,  # higher opp TO = better for A
            a.orb_pct - b.orb_pct,
            a.ftr_off - b.ftr_off,
            b.efg_pct_def - a.efg_pct_def,  # higher opp def eFG = worse defense for B
            a.to_pct_def - b.to_pct_def,
            a.drb_pct - b.drb_pct,
            b.ftr_def - a.ftr_def,
            a.net_efficiency - b.net_efficiency,
            a.momentum - b.momentum,
            float(a.seed - b.seed),  # seed difference (negative = A is better seed)
        ])

    def _train(self, teams: List[Team]):
        """
        Generate synthetic training matchups.  For each pair, the 'true'
        outcome is derived from a log5-style formula on net efficiency so
        the LR learns the mapping from feature deltas → win probability.
        """
        X_rows, y_rows = [], []
        n = len(teams)

        for i in range(n):
            for j in range(i + 1, n):
                a, b = teams[i], teams[j]
                fv = self._build_feature_vector(a, b)

                # Log5-style "true" probability using net efficiency
                diff = (a.net_efficiency * a.momentum) - (b.net_efficiency * b.momentum)
                p_a = 1.0 / (1.0 + 10 ** (-diff / 12.0))  # 12-pt spread ≈ 90%

                # Sample a binary outcome from this probability (synthetic label)
                outcome = 1 if np.random.random() < p_a else 0
                X_rows.append(fv)
                y_rows.append(outcome)

                # Also add the mirror (B vs A)
                X_rows.append(-fv)
                y_rows.append(1 - outcome)

        X = np.array(X_rows)
        y = np.array(y_rows)

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        print(f"   🧠  Model trained on {len(X)} synthetic matchups (accuracy on training set: "
              f"{self.model.score(X_scaled, y):.3f})")

    def predict(self, a: Team, b: Team) -> float:
        """Return P(a beats b), a float in [0, 1]."""
        fv = self._build_feature_vector(a, b).reshape(1, -1)
        fv_scaled = self.scaler.transform(fv)
        prob = self.model.predict_proba(fv_scaled)[0][1]

        # Apply momentum as a light multiplier (shifts probability slightly)
        adj = prob * a.momentum / (prob * a.momentum + (1 - prob) * b.momentum)
        return np.clip(adj, 0.01, 0.99)

    def predict_score(self, a: Team, b: Team) -> Tuple[float, float]:
        """
        Predict final score (score_a, score_b) using the KenPom-style formula:

          possessions  = avg(AdjT_a, AdjT_b)
          score_a      = AdjO_a * AdjD_b * possessions / 10000
          score_b      = AdjO_b * AdjD_a * possessions / 10000

        AdjO and AdjD are in pts/100 poss relative to a national avg of 100,
        so dividing by 100 twice (i.e. /10000) converts to pts per possession.
        """
        possessions = (a.adj_t + b.adj_t) / 2.0
        score_a = a.adj_o * b.adj_d * possessions / 10000.0
        score_b = b.adj_o * a.adj_d * possessions / 10000.0
        return round(score_a, 1), round(score_b, 1)


# ============================================================================
# 6.  GAME THEORY / EV ADJUSTMENTS
# ============================================================================

def apply_game_theory_adjustment(
    prob: float,
    team: Team,
    _round_idx: int,
    is_championship: bool = False,
) -> Tuple[float, float]:
    """
    Returns (adjusted_prob, ev_multiplier).

    - Penalize over-picked 1-seeds for tournament winner
    - Boost under-picked strong 2/3/4 seeds
    - KenPom championship constraint
    """
    ev_mult = 1.0

    # ---- Contrarian penalty for over-picked 1-seeds ----
    if team.seed == 1 and team.public_pick_pct > PUBLIC_PICK_PENALTY_THRESHOLD:
        prob = prob * (1.0 - PUBLIC_PICK_PENALTY)

    # ---- Boost under-picked strong mid-seeds ----
    if team.seed in (2, 3, 4) and team.public_pick_pct < UNDERVALUED_THRESHOLD:
        # Only boost if they're actually good (top-half net efficiency)
        if team.adj_o_rank <= 32 and team.adj_d_rank <= 40:
            ev_mult = UNDERVALUED_BOOST

    # ---- KenPom championship penalty ----
    if is_championship:
        if team.adj_o_rank > KENPOM_TOP_N or team.adj_d_rank > KENPOM_TOP_N:
            prob = prob * KENPOM_CHAMP_PENALTY

    return (np.clip(prob, 0.01, 0.99), ev_mult)


# ============================================================================
# 7.  BRACKET STRUCTURE & SIMULATION
# ============================================================================

def build_initial_bracket(teams: List[Team]) -> List[Team]:
    """
    Return the 64 teams ordered by bracket slot (0-63).
    Slots 0-15 = East, 16-31 = West, 32-47 = South, 48-63 = Midwest.
    Within each region the seed order is:
        1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
    """
    return sorted(teams, key=lambda t: t.bracket_slot)


def simulate_tournament_once(
    bracket: List[Team],
    model: MatchupModel,
    scoring: Dict[int, int],
    rng: np.random.Generator,
) -> Tuple[List[int], float]:
    """
    Simulate one full tournament.

    Returns
    -------
    winners : list[int]
        Length-63 list where winners[g] is the bracket_slot of the game-g winner.
    total_ev : float
        Weighted EV score for this bracket run.
    """
    # Current round's teams (indices into `bracket`)
    current = list(range(NUM_TEAMS))
    winners = []
    total_ev = 0.0

    for round_idx in range(ROUNDS):
        next_round = []
        is_final = (round_idx == ROUNDS - 1)
        pts = scoring.get(round_idx, 10)

        for g in range(0, len(current), 2):
            a = bracket[current[g]]
            b = bracket[current[g + 1]]

            prob_a = model.predict(a, b)
            adj_a, ev_mult_a = apply_game_theory_adjustment(
                prob_a, a, round_idx, is_championship=is_final
            )
            adj_b, ev_mult_b = apply_game_theory_adjustment(
                1.0 - prob_a, b, round_idx, is_championship=is_final
            )

            # Renormalize
            total = adj_a + adj_b
            p_a = adj_a / total

            if rng.random() < p_a:
                winner = current[g]
                ev_contrib = pts * p_a * ev_mult_a
            else:
                winner = current[g + 1]
                ev_contrib = pts * (1 - p_a) * ev_mult_b

            winners.append(winner)
            next_round.append(winner)
            total_ev += ev_contrib

        current = next_round

    return winners, total_ev


def enforce_upset_quota(
    bracket: List[Team],
    _model: "MatchupModel",
    sim_results: np.ndarray,
    n_sims: int,
) -> None:
    """
    Check that the aggregate bracket satisfies the upset quota.
    If not, we flag which games need an override.
    This is applied in post-processing of the optimal bracket.
    """
    # Identify first-round matchups (games 0-31)
    twelve_over_five = []
    eleven_over_six = []

    for g in range(0, 32, 2):
        slot_a = g
        slot_b = g + 1
        a = bracket[slot_a]
        b = bracket[slot_b]

        seeds = sorted([(a.seed, slot_a, a.name), (b.seed, slot_b, b.name)])
        low_seed, high_seed = seeds[0], seeds[1]  # low number = better

        if low_seed[0] == 5 and high_seed[0] == 12:
            # Count how often the 12-seed won across sims
            twelve_wins = np.sum(sim_results[:, g // 2] == high_seed[1])
            twelve_over_five.append((g // 2, high_seed[2], twelve_wins / n_sims))
        elif low_seed[0] == 6 and high_seed[0] == 11:
            eleven_wins = np.sum(sim_results[:, g // 2] == high_seed[1])
            eleven_over_six.append((g // 2, high_seed[2], eleven_wins / n_sims))

    return twelve_over_five, eleven_over_six


# ============================================================================
# 8.  MONTE CARLO ENGINE
# ============================================================================

def run_monte_carlo(
    teams: List[Team],
    model: MatchupModel,
    n_sims: int = 10000,
    scoring: Optional[Dict[int, int]] = None,
    seed: int = 2024,
) -> Tuple[List[int], float, Dict]:
    """
    Run `n_sims` tournament simulations and select the single bracket
    path that maximizes expected value.

    Returns
    -------
    best_bracket : list of team names for each of the 63 games
    best_ev      : the EV of that bracket
    stats        : dict of summary statistics
    """
    if scoring is None:
        scoring = DEFAULT_SCORING

    bracket = build_initial_bracket(teams)
    rng = np.random.default_rng(seed)

    print(f"\n🎲  Running {n_sims:,} Monte Carlo simulations...")

    # Track how often each team wins each game slot
    # Game slots: 32 (R64) + 16 (R32) + 8 (S16) + 4 (E8) + 2 (F4) + 1 (NCG) = 63
    # We'll accumulate: for each game g ∈ [0,63), which bracket_slot wins most often
    # and the EV associated with picking that team.

    # Strategy: for each game, count wins by bracket_slot, weighted by EV
    game_winner_ev = {}  # game_idx -> {bracket_slot: cumulative_ev}
    game_winner_cnt = {}  # game_idx -> {bracket_slot: count}
    champion_count = {}
    total_evs = []

    for sim in range(n_sims):
        current = list(range(NUM_TEAMS))
        sim_ev = 0.0
        game_idx = 0

        for round_idx in range(ROUNDS):
            next_round = []
            is_final = (round_idx == ROUNDS - 1)
            pts = scoring.get(round_idx, 10)

            for g in range(0, len(current), 2):
                a = bracket[current[g]]
                b = bracket[current[g + 1]]

                prob_a = model.predict(a, b)
                adj_a, ev_mult_a = apply_game_theory_adjustment(
                    prob_a, a, round_idx, is_championship=is_final
                )
                adj_b, ev_mult_b = apply_game_theory_adjustment(
                    1.0 - prob_a, b, round_idx, is_championship=is_final
                )
                total_p = adj_a + adj_b
                p_a = adj_a / total_p

                if rng.random() < p_a:
                    winner_slot = current[g]
                    ev_pick = pts * ev_mult_a
                else:
                    winner_slot = current[g + 1]
                    ev_pick = pts * ev_mult_b

                # Accumulate
                if game_idx not in game_winner_ev:
                    game_winner_ev[game_idx] = {}
                    game_winner_cnt[game_idx] = {}

                game_winner_ev[game_idx][winner_slot] = (
                    game_winner_ev[game_idx].get(winner_slot, 0.0) + ev_pick
                )
                game_winner_cnt[game_idx][winner_slot] = (
                    game_winner_cnt[game_idx].get(winner_slot, 0) + 1
                )

                next_round.append(winner_slot)
                sim_ev += ev_pick
                game_idx += 1

            current = next_round

        # Track champion
        champ_slot = current[0]
        champion_count[champ_slot] = champion_count.get(champ_slot, 0) + 1
        total_evs.append(sim_ev)

        if (sim + 1) % 2500 == 0:
            print(f"   ... {sim+1:,}/{n_sims:,} simulations complete")

    # ---- Build optimal bracket: pick the team with highest EV per game ----
    # We must ensure consistency: if Team X wins game g, they must have won
    # all prior games leading to game g.

    print("\n🏗️   Building optimal bracket with consistency enforcement...")
    optimal = _build_consistent_bracket(bracket, game_winner_ev, game_winner_cnt, n_sims, scoring)

    # ---- Enforce upset quota ----
    optimal = _enforce_upsets(bracket, optimal, game_winner_cnt, n_sims)

    # ---- Compute EV of optimal bracket ----
    opt_ev = _compute_bracket_ev(bracket, optimal, model, scoring)

    # ---- Summary stats ----
    stats = {
        "mean_sim_ev": np.mean(total_evs),
        "std_sim_ev": np.std(total_evs),
        "champion_distribution": {
            bracket[s].name: c / n_sims
            for s, c in sorted(champion_count.items(), key=lambda x: -x[1])[:10]
        },
    }

    return optimal, opt_ev, stats, bracket


def _build_consistent_bracket(
    _bracket: List[Team],
    game_winner_ev: Dict,
    _game_winner_cnt: Dict,
    _n_sims: int,
    _scoring: Dict,
) -> List[int]:
    """
    Build a 63-game bracket that is internally consistent.
    For each game, pick the team with the highest EV that is eligible
    (i.e., won its prior matchup in our bracket).
    """
    optimal = [None] * NUM_GAMES

    # Round boundaries
    round_starts = []
    g = 0
    n = NUM_TEAMS
    for r in range(ROUNDS):
        round_starts.append(g)
        g += n // 2
        n //= 2

    # Round of 64: games 0-31, any team in its slot is eligible
    for game_idx in range(round_starts[0], round_starts[0] + 32):
        ev_map = game_winner_ev.get(game_idx, {})
        if ev_map:
            best_slot = max(ev_map, key=lambda s: ev_map[s])
        else:
            best_slot = game_idx * 2  # fallback to higher seed
        optimal[game_idx] = best_slot

    # Subsequent rounds: winner must have won prior game
    for r in range(1, ROUNDS):
        start = round_starts[r]
        n_games = NUM_TEAMS // (2 ** (r + 1))
        prev_start = round_starts[r - 1]

        for g_offset in range(n_games):
            game_idx = start + g_offset
            # The two feeder games
            feeder_a = prev_start + g_offset * 2
            feeder_b = prev_start + g_offset * 2 + 1

            eligible = [optimal[feeder_a], optimal[feeder_b]]
            ev_map = game_winner_ev.get(game_idx, {})

            # Pick the eligible team with higher EV
            best_slot = None
            best_ev = -1
            for slot in eligible:
                if slot is not None and ev_map.get(slot, 0) > best_ev:
                    best_ev = ev_map.get(slot, 0)
                    best_slot = slot

            if best_slot is None:
                best_slot = eligible[0]  # fallback

            optimal[game_idx] = best_slot

    return optimal


def _enforce_upsets(
    bracket: List[Team],
    optimal: List[int],
    game_winner_cnt: Dict,
    n_sims: int,
) -> List[int]:
    """
    Ensure at least one 12>5 and one 11>6 upset in R64.
    If none exist, flip the most likely one.
    """
    has_12_over_5 = False
    has_11_over_6 = False
    best_12_game = None
    best_12_prob = 0
    best_11_game = None
    best_11_prob = 0

    for game_idx in range(32):
        slot_a = game_idx * 2
        slot_b = game_idx * 2 + 1

        if slot_a >= len(bracket) or slot_b >= len(bracket):
            continue

        a = bracket[slot_a]
        b = bracket[slot_b]
        winner_slot = optimal[game_idx]
        winner = bracket[winner_slot] if winner_slot is not None else a

        seeds = {a.seed: (a, slot_a), b.seed: (b, slot_b)}

        # Check 5v12
        if set([a.seed, b.seed]) == {5, 12}:
            if winner.seed == 12:
                has_12_over_5 = True
            else:
                # Check upset probability from sims
                twelve_team = seeds[12]
                cnt = game_winner_cnt.get(game_idx, {})
                prob = cnt.get(twelve_team[1], 0) / max(n_sims, 1)
                if prob > best_12_prob:
                    best_12_prob = prob
                    best_12_game = (game_idx, twelve_team[1])

        # Check 6v11
        if set([a.seed, b.seed]) == {6, 11}:
            if winner.seed == 11:
                has_11_over_6 = True
            else:
                eleven_team = seeds[11]
                cnt = game_winner_cnt.get(game_idx, {})
                prob = cnt.get(eleven_team[1], 0) / max(n_sims, 1)
                if prob > best_11_prob:
                    best_11_prob = prob
                    best_11_game = (game_idx, eleven_team[1])

    if not has_12_over_5 and best_12_game is not None:
        g, slot = best_12_game
        print(f"   🔄  Enforcing upset: {bracket[slot].name} (12) over 5-seed in game {g}")
        optimal[g] = slot

    if not has_11_over_6 and best_11_game is not None:
        g, slot = best_11_game
        print(f"   🔄  Enforcing upset: {bracket[slot].name} (11) over 6-seed in game {g}")
        optimal[g] = slot

    return optimal


def _compute_bracket_ev(
    bracket: List[Team],
    optimal: List[int],
    model: MatchupModel,
    scoring: Dict,
) -> float:
    """Compute the total EV of the selected bracket path."""
    total_ev = 0.0
    game_idx = 0
    current_slots = list(range(NUM_TEAMS))

    for round_idx in range(ROUNDS):
        n_games = len(current_slots) // 2
        pts = scoring.get(round_idx, 10)
        next_slots = []

        for g in range(n_games):
            slot_a = current_slots[g * 2]
            slot_b = current_slots[g * 2 + 1]
            a = bracket[slot_a]
            b = bracket[slot_b]

            picked_slot = optimal[game_idx]
            picked = bracket[picked_slot]

            prob_a = model.predict(a, b)
            if picked_slot == slot_a:
                win_prob = prob_a
            else:
                win_prob = 1.0 - prob_a

            adj_prob, ev_mult = apply_game_theory_adjustment(
                win_prob, picked, round_idx,
                is_championship=(round_idx == ROUNDS - 1),
            )
            total_ev += pts * adj_prob * ev_mult

            next_slots.append(picked_slot)
            game_idx += 1

        current_slots = next_slots

    return total_ev


# ============================================================================
# 9.  OUTPUT DISPLAY
# ============================================================================

def print_bracket(
    optimal: List[int],
    bracket: List[Team],
    ev: float,
    stats: Dict,
    model: "MatchupModel",
    vegas_odds: Optional["pd.DataFrame"] = None,
):
    """Pretty-print the optimal bracket with score predictions and Vegas lines."""
    sep = "=" * 90
    print(f"\n{sep}")
    print("  🏆  OPTIMAL MARCH MADNESS BRACKET  🏆")
    print(f"{sep}\n")

    game_idx = 0
    current_slots = list(range(NUM_TEAMS))

    for round_idx in range(ROUNDS):
        n_games = len(current_slots) // 2
        print(f"  ── {ROUND_NAMES[round_idx]} ({DEFAULT_SCORING[round_idx]} pts each) "
              f"{'─' * (62 - len(ROUND_NAMES[round_idx]))}")
        next_slots = []

        for g in range(n_games):
            slot_a = current_slots[g * 2]
            slot_b = current_slots[g * 2 + 1]
            a = bracket[slot_a]
            b = bracket[slot_b]
            winner_slot = optimal[game_idx]
            winner = bracket[winner_slot]

            loser = a if winner_slot != slot_a else b
            upset_marker = " ⚡" if winner.seed > loser.seed else ""

            # Predicted score
            score_a, score_b = model.predict_score(a, b)
            if winner_slot == slot_a:
                score_str = f"  ~{score_a:.0f}-{score_b:.0f}"
            else:
                score_str = f"  ~{score_b:.0f}-{score_a:.0f}"

            # Vegas spread (if available)
            vegas_str = ""
            if vegas_odds is not None and not vegas_odds.empty:
                spread = _lookup_vegas_spread(a.name, b.name, vegas_odds)
                if spread is not None:
                    vegas_str = f"  [Vegas: {spread:+.1f}]"

            print(f"    ({a.seed:>2}) {a.name:<22} vs ({b.seed:>2}) {b.name:<22} "
                  f"→  ({winner.seed:>2}) {winner.name:<20}{upset_marker}"
                  f"{score_str}{vegas_str}")

            next_slots.append(winner_slot)
            game_idx += 1

        current_slots = next_slots
        print()

    # Champion
    champ = bracket[optimal[-1]]
    print(f"  🏆  CHAMPION:  ({champ.seed}) {champ.name}")
    print(f"  📊  Bracket Expected Value: {ev:.1f} points\n")

    print(f"  ── Simulation Statistics {'─' * 62}")
    print(f"    Mean EV per sim : {stats['mean_sim_ev']:.1f} ± {stats['std_sim_ev']:.1f}")
    print(f"\n  ── Top 10 Championship Probabilities {'─' * 50}")
    for name, prob in list(stats["champion_distribution"].items())[:10]:
        bar = "█" * int(prob * 100)
        print(f"    {name:<22} {prob*100:5.1f}%  {bar}")
    print(f"\n{sep}\n")


def _lookup_vegas_spread(
    name_a: str,
    name_b: str,
    odds_df: "pd.DataFrame",
) -> Optional[float]:
    """
    Find the Vegas spread for a matchup (positive = name_a is underdog).
    Returns None if not found.
    """
    if odds_df is None or odds_df.empty:
        return None
    a_lower = name_a.lower()
    b_lower = name_b.lower()
    for _, row in odds_df.iterrows():
        h = str(row.get("home_team", "")).lower()
        aw = str(row.get("away_team", "")).lower()
        if (a_lower in h or h in a_lower) and (b_lower in aw or aw in b_lower):
            return row.get("spread_home")
        if (b_lower in h or h in b_lower) and (a_lower in aw or aw in a_lower):
            s = row.get("spread_home")
            return -s if s is not None else None
    return None


# ============================================================================
# 10. CSV TEMPLATE GENERATOR
# ============================================================================

def export_csv_template(path: str = "teams_template.csv"):
    """Export a blank CSV template with all required columns."""
    cols = REQUIRED_COLUMNS
    df = pd.DataFrame(columns=cols)
    # Add one example row
    df.loc[0] = [
        "Example State", 1, "East",
        118.5, 89.2,
        0.545, 0.155, 0.340, 0.380,
        0.440, 0.220, 0.750, 0.270,
        8, 0.25
    ]
    df.to_csv(path, index=False)
    print(f"\n📝  Template saved to: {path}")
    print(f"   Fill in all {NUM_TEAMS} teams and re-run with --csv {path}\n")


# ============================================================================
# 11. DATAFRAME → TEAM LIST HELPER (used by --scrape)
# ============================================================================

def _teams_from_dataframe(df: pd.DataFrame) -> List[Team]:
    """Convert a scraped DataFrame to a list of Team objects."""
    teams = []
    for _, row in df.iterrows():
        t = Team(
            name=str(row["team"]).strip(),
            seed=int(row["seed"]),
            region=str(row["region"]).strip(),
            adj_o=float(row["adj_o"]),
            adj_d=float(row["adj_d"]),
            efg_pct_off=float(row["efg_pct_off"]),
            to_pct_off=float(row["to_pct_off"]),
            orb_pct=float(row["orb_pct"]),
            ftr_off=float(row["ftr_off"]),
            efg_pct_def=float(row["efg_pct_def"]),
            to_pct_def=float(row["to_pct_def"]),
            drb_pct=float(row["drb_pct"]),
            ftr_def=float(row["ftr_def"]),
            last10_wins=int(row["last10_wins"]),
            public_pick_pct=float(row["public_pick_pct"]),
            adj_t=float(row["adj_t"]) if "adj_t" in df.columns else 68.5,
        )
        teams.append(t)
    return teams


# ============================================================================
# 12. MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="March Madness Bracket Optimizer — EV-maximizing Monte Carlo engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  python march_madness_optimizer.py --demo                       # Run with mock data
  python march_madness_optimizer.py --csv my_teams.csv           # Run with your data
  python march_madness_optimizer.py --scrape                     # Live Barttorvik scrape
  python march_madness_optimizer.py --scrape --odds-key YOUR_KEY # + Vegas odds
  python march_madness_optimizer.py --scrape --save-csv scraped.csv --sims 50000
  python march_madness_optimizer.py --template                   # Export blank CSV

CSV FORMAT (14 required + 1 optional column):
  team, seed, region, adj_o, adj_d,
  efg_pct_off, to_pct_off, orb_pct, ftr_off,
  efg_pct_def, to_pct_def, drb_pct, ftr_def,
  last10_wins, public_pick_pct
  [adj_t]  ← optional tempo column; defaults to 68.5 if absent
        """,
    )
    parser.add_argument("--csv",      type=str, help="Path to your 64-team CSV file")
    parser.add_argument("--demo",     action="store_true", help="Run with mock data")
    parser.add_argument("--scrape",       action="store_true",
                        help="Scrape live stats from Sports-Reference + Vegas odds")
    parser.add_argument("--year",         type=int, default=2025,
                        help="SR season year (2025=2024-25, 2026=2025-26; default: 2025)")
    parser.add_argument("--bracket-csv",  type=str, default="",
                        help="CSV with team/seed/region columns to use as bracket source "
                             "(required when the SR bracket page isn't posted yet)")
    parser.add_argument("--odds-key",     type=str, default="",
                        help="The Odds API key for Vegas spreads (free at the-odds-api.com)")
    parser.add_argument("--save-csv",     type=str, default="",
                        help="Save scraped team data to this CSV path")
    parser.add_argument("--last10",       action="store_true",
                        help="(unused — kept for compatibility)")
    parser.add_argument("--template", action="store_true", help="Export blank CSV template")
    parser.add_argument("--sims",     type=int, default=10000,
                        help="Monte Carlo simulations (default: 10000)")
    parser.add_argument("--seed",     type=int, default=2024,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    print(sep := "=" * 90)
    print("  MARCH MADNESS BRACKET OPTIMIZER v2.0")
    print("  Monte Carlo + Game Theory + ML Probability Model + Score Predictions")
    print(sep)

    if args.template:
        export_csv_template()
        return

    vegas_odds = None   # populated when --scrape + --odds-key

    # ── Load or generate teams ───────────────────────────────────────────────
    if args.scrape:
        try:
            from scraper import build_tournament_dataframe, get_scraped_odds
        except ImportError:
            print("\n❌  scraper.py not found. Make sure it is in the same directory.")
            sys.exit(1)

        # Resolve odds key: CLI flag → .env → empty
        odds_key = args.odds_key or _ENV.get("odds-key", "")
        if odds_key:
            print(f"  🔑  Odds API key loaded {'from --odds-key' if args.odds_key else 'from .env'}")

        # Optional bracket override from --bracket-csv
        bracket_df = None
        if args.bracket_csv:
            print(f"  📋  Loading bracket seeds/regions from: {args.bracket_csv}")
            bracket_df = pd.read_csv(args.bracket_csv)
            bracket_df.columns = bracket_df.columns.str.strip().str.lower()

        print(f"\n🌐  Scraping live data for {args.year} season...")
        scraped_df = build_tournament_dataframe(
            year=args.year,
            odds_api_key=odds_key,
            bracket_df=bracket_df,
        )

        if scraped_df.empty:
            print("\n❌  Scrape returned no data.")
            sys.exit(1)

        if args.save_csv:
            scraped_df.to_csv(args.save_csv, index=False)
            print(f"\n💾  Scraped data saved to: {args.save_csv}")

        # Convert DataFrame → Team list
        teams = _teams_from_dataframe(scraped_df)
        vegas_odds = get_scraped_odds()

    elif args.csv:
        teams = load_teams_from_csv(args.csv)
    elif args.demo:
        print("\n🎮  Running in DEMO mode with mock data...")
        teams = generate_mock_teams()
    else:
        print("\n❌  Specify --scrape, --csv <file>, or --demo. Use --help for details.")
        sys.exit(1)

    # ── Feature engineering ──────────────────────────────────────────────────
    print("\n⚙️   Engineering features...")
    teams = engineer_features(teams)

    # Print team summary
    print(f"\n📋  Team Summary (sorted by Net Efficiency):")
    print(f"   {'Team':<22} {'Seed':>4} {'Region':<8} {'AdjO':>6} {'AdjD':>6} "
          f"{'AdjT':>5} {'NetEff':>7} {'Mom':>5} {'Pub%':>6}")
    print(f"   {'─'*80}")
    for t in sorted(teams, key=lambda x: x.net_efficiency, reverse=True)[:15]:
        print(f"   {t.name:<22} {t.seed:>4} {t.region:<8} {t.adj_o:>6.1f} {t.adj_d:>6.1f} "
              f"{t.adj_t:>5.1f} {t.net_efficiency:>7.1f} {t.momentum:>5.2f} "
              f"{t.public_pick_pct*100:>5.1f}%")
    print(f"   ... and {len(teams)-15} more teams\n")

    # ── Train model ──────────────────────────────────────────────────────────
    print("🧠  Training probability model...")
    model = MatchupModel(teams)

    # ── Run Monte Carlo ──────────────────────────────────────────────────────
    optimal, ev, stats, bracket = run_monte_carlo(
        teams, model, n_sims=args.sims, seed=args.seed
    )

    # ── Print results ────────────────────────────────────────────────────────
    print_bracket(optimal, bracket, ev, stats, model, vegas_odds=vegas_odds)


if __name__ == "__main__":
    main()
