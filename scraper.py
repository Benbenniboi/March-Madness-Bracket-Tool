#!/usr/bin/env python3
"""
scraper.py — Live data scrapers for March Madness Optimizer

Sources:
  1. Barttorvik (barttorvik.com)  — T-Rank efficiency stats + tempo
  2. The Odds API (the-odds-api.com) — Vegas moneylines / spreads
       Free tier: 500 requests/month. Get a free key at https://the-odds-api.com
  3. ESPN public API (fallback)   — tournament bracket / seeds / regions

Usage:
  python scraper.py                            # print scraped data summary
  python scraper.py --year 2025               # specific year
  python scraper.py --odds-key YOUR_API_KEY   # include Vegas odds
  python scraper.py --out teams_scraped.csv   # save to CSV
"""

import argparse
import json
import sys
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# HTTP session with polite headers
# ---------------------------------------------------------------------------
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "en-US,en;q=0.9",
})
REQUEST_DELAY = 0.5   # seconds between requests (be polite)

# ---------------------------------------------------------------------------
# Barttorvik
# ---------------------------------------------------------------------------
BART_API = "https://barttorvik.com/getteams.php"
BART_GAME_API = "https://barttorvik.com/getgames.php"

# Column positions in barttorvik getteams.php response array
# (these are stable for recent seasons; verified against 2023-2025 data)
BART_IDX = {
    "team":         1,
    "conf":         2,
    # index 3 = games played, 4 = record string "W-L"
    "adj_o":        5,
    "adj_d":        6,
    # index 7 = barthag
    "efg_pct_off":  8,
    "efg_pct_def":  9,
    "to_pct_off":   10,
    "to_pct_def":   11,   # steal/turnover rate forced
    "orb_pct":      12,
    "_opp_orb":     13,   # opponent ORB% — we compute drb_pct = 1 - this
    "ftr_off":      14,
    "ftr_def":      15,
    # 16 = 2P% off, 17 = 2P% def, 18 = 3P% off, 19 = 3P% def
    "adj_t":        20,   # tempo (possessions per 40 min)
    # 21 = WAB (Wins Above Bubble)
}


def fetch_barttorvik_stats(year: int = 2025) -> pd.DataFrame:
    """
    Fetch full-season T-Rank stats from barttorvik.com.

    Returns a DataFrame with one row per team including:
      team, conf, adj_o, adj_d, adj_t, efg_pct_off, efg_pct_def,
      to_pct_off, to_pct_def, orb_pct, drb_pct, ftr_off, ftr_def, win_pct
    """
    print(f"  📡  Fetching Barttorvik stats for {year}...")
    params = {
        "year":      year,
        "conlimit":  "All",
        "top":       0,
        "start":     f"{year-1}1101",   # start of season (Nov 1)
        "end":       f"{year}0401",      # end (Apr 1)
        "type":      "All",
        "team":      "",
        "ot":        1,
        "venue":     "All",
        "oppType":   "All",
    }

    try:
        resp = SESSION.get(BART_API, params=params, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
    except requests.RequestException as e:
        print(f"  ❌  Barttorvik request failed: {e}")
        return pd.DataFrame()
    except ValueError:
        print("  ❌  Barttorvik returned non-JSON response")
        return pd.DataFrame()

    if not raw:
        print("  ⚠️   Barttorvik returned empty data")
        return pd.DataFrame()

    rows = []
    for entry in raw:
        # entry can be a list (array API) or dict; handle both
        if isinstance(entry, dict):
            # dict-style response
            row = {
                "team":         str(entry.get("team", "")).strip(),
                "conf":         str(entry.get("conf", "")).strip(),
                "adj_o":        _safe_float(entry.get("adjoe")),
                "adj_d":        _safe_float(entry.get("adjde")),
                "adj_t":        _safe_float(entry.get("adjt", 68.5)),
                "efg_pct_off":  _safe_float(entry.get("efg", 0.5)) / 100,
                "efg_pct_def":  _safe_float(entry.get("efgd", 0.5)) / 100,
                "to_pct_off":   _safe_float(entry.get("tord", 0.18)) / 100,
                "to_pct_def":   _safe_float(entry.get("tord_d", 0.18)) / 100,
                "orb_pct":      _safe_float(entry.get("orb", 0.3)) / 100,
                "drb_pct":      1.0 - _safe_float(entry.get("drbp", 0.27)) / 100,
                "ftr_off":      _safe_float(entry.get("ftr", 0.35)) / 100,
                "ftr_def":      _safe_float(entry.get("ftrd", 0.35)) / 100,
                "record":       str(entry.get("record", "0-0")),
            }
        elif isinstance(entry, list) and len(entry) > 20:
            # array-style response — use index map
            efg_off = _safe_float(entry[BART_IDX["efg_pct_off"]])
            efg_def = _safe_float(entry[BART_IDX["efg_pct_def"]])
            to_off  = _safe_float(entry[BART_IDX["to_pct_off"]])
            to_def  = _safe_float(entry[BART_IDX["to_pct_def"]])
            orb     = _safe_float(entry[BART_IDX["orb_pct"]])
            opp_orb = _safe_float(entry[BART_IDX["_opp_orb"]])
            ftr_off = _safe_float(entry[BART_IDX["ftr_off"]])
            ftr_def = _safe_float(entry[BART_IDX["ftr_def"]])

            # barttorvik returns percentages as 0-100; normalise to 0-1
            def pct(v):
                return v / 100.0 if v > 1 else v

            row = {
                "team":        str(entry[BART_IDX["team"]]).strip(),
                "conf":        str(entry[BART_IDX["conf"]]).strip(),
                "adj_o":       _safe_float(entry[BART_IDX["adj_o"]]),
                "adj_d":       _safe_float(entry[BART_IDX["adj_d"]]),
                "adj_t":       _safe_float(entry[BART_IDX["adj_t"]]),
                "efg_pct_off": pct(efg_off),
                "efg_pct_def": pct(efg_def),
                "to_pct_off":  pct(to_off),
                "to_pct_def":  pct(to_def),
                "orb_pct":     pct(orb),
                "drb_pct":     1.0 - pct(opp_orb),
                "ftr_off":     pct(ftr_off),
                "ftr_def":     pct(ftr_def),
                "record":      str(entry[4]) if len(entry) > 4 else "0-0",
            }
        else:
            continue

        # Parse win% from "W-L" record
        row["win_pct"] = _parse_win_pct(row.get("record", "0-0"))
        rows.append(row)

    df = pd.DataFrame(rows)
    # Normalize team names (title-case, strip extra whitespace)
    if "team" in df.columns:
        df["team"] = df["team"].str.strip()
    print(f"  ✅  Barttorvik: {len(df)} teams loaded")
    return df


def fetch_barttorvik_last10(team_name: str, year: int = 2025) -> int:
    """
    Fetch last-10-game wins for a specific team from barttorvik game data.
    Returns integer wins (0-10). Falls back to 7 on error.
    """
    params = {
        "team": team_name,
        "year": year,
        "opp":  "",
        "type": "All",
        "venue": "All",
        "ot":   1,
    }
    try:
        resp = SESSION.get(BART_GAME_API, params=params, timeout=10)
        resp.raise_for_status()
        games = resp.json()
        if not games or not isinstance(games, list):
            return 7
        # Each game entry is typically: [date, team, opp, result, ...]
        # result column is usually at index 3: "W" or "L"
        # Take last 10 games in chronological order
        last_10 = games[-10:] if len(games) >= 10 else games
        wins = 0
        for g in last_10:
            if isinstance(g, list) and len(g) > 3:
                result = str(g[3]).strip().upper()
                if result.startswith("W"):
                    wins += 1
            elif isinstance(g, dict):
                result = str(g.get("result", g.get("wl", ""))).upper()
                if result.startswith("W"):
                    wins += 1
        return wins
    except Exception:
        return 7


# ---------------------------------------------------------------------------
# NCAA tournament bracket  (seeds + regions)
# ---------------------------------------------------------------------------
ESPN_BRACKET_API = (
    "https://site.api.espn.com/apis/v2/sports/basketball"
    "/mens-college-basketball/tournaments/22/bracket"
)
ESPN_SCOREBOARD_API = (
    "https://site.api.espn.com/apis/v2/sports/basketball"
    "/mens-college-basketball/scoreboard"
)


def fetch_tournament_field(year: int = 2025) -> Dict[str, Dict]:
    """
    Fetch tournament bracket data from ESPN's public API.

    Returns dict: {team_name: {"seed": int, "region": str}}
    Falls back to an empty dict on failure.
    """
    print(f"  📡  Fetching {year} tournament bracket from ESPN...")
    try:
        resp = SESSION.get(
            ESPN_BRACKET_API,
            params={"seasontype": 3, "season": year},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  ⚠️   ESPN bracket API failed ({e}), trying scoreboard fallback...")
        return _fetch_bracket_from_scoreboard(year)

    field = {}
    try:
        # ESPN bracket JSON nests teams under groups → competitors
        for group in data.get("bracket", {}).get("groups", []):
            region_name = group.get("name", "Unknown")
            for game in group.get("games", []):
                for comp in game.get("competitors", []):
                    tname = comp.get("team", {}).get("displayName", "")
                    seed  = comp.get("seed", 0)
                    if tname:
                        field[tname] = {"seed": int(seed), "region": region_name}
    except Exception:
        pass

    if not field:
        print("  ⚠️   ESPN bracket parse failed, trying scoreboard fallback...")
        return _fetch_bracket_from_scoreboard(year)

    print(f"  ✅  ESPN bracket: {len(field)} teams found")
    return field


def _fetch_bracket_from_scoreboard(year: int) -> Dict[str, Dict]:
    """Fallback: pull tournament teams from ESPN scoreboard for March/April."""
    field = {}
    # Fetch games across tournament dates
    import datetime
    start = datetime.date(year, 3, 15)
    for offset in range(0, 25):
        d = start + datetime.timedelta(days=offset)
        date_str = d.strftime("%Y%m%d")
        try:
            resp = SESSION.get(
                ESPN_SCOREBOARD_API,
                params={"groups": 50, "limit": 25, "dates": date_str},
                timeout=10,
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            for event in data.get("events", []):
                for comp in event.get("competitions", [{}])[0].get("competitors", []):
                    tname = comp.get("team", {}).get("displayName", "")
                    seed  = comp.get("curatedRank", {}).get("current", 0)
                    # Region from note or group
                    note = event.get("notes", [{}])[0].get("headline", "") if event.get("notes") else ""
                    region = _parse_region_from_note(note)
                    if tname and seed:
                        field[tname] = {"seed": int(seed), "region": region}
            time.sleep(REQUEST_DELAY)
        except Exception:
            continue

    if field:
        print(f"  ✅  ESPN scoreboard fallback: {len(field)} teams found")
    else:
        print("  ⚠️   Could not determine tournament field automatically.")
    return field


REGION_KEYWORDS = {
    "East": ["East"],
    "West": ["West"],
    "South": ["South"],
    "Midwest": ["Midwest", "Mid"],
}

def _parse_region_from_note(note: str) -> str:
    for region, keywords in REGION_KEYWORDS.items():
        if any(kw.lower() in note.lower() for kw in keywords):
            return region
    return "Unknown"


# ---------------------------------------------------------------------------
# Vegas odds  (The Odds API)
# ---------------------------------------------------------------------------
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds/"
ODDS_FUTURES_URL = "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds/"


def fetch_vegas_odds(api_key: str) -> pd.DataFrame:
    """
    Fetch current Vegas moneylines + point spreads for NCAAB games
    using The Odds API (https://the-odds-api.com — free tier available).

    Returns a DataFrame with columns:
      home_team, away_team, home_ml, away_ml, spread_home, spread_away,
      implied_home_prob, implied_away_prob

    Pass an empty string to skip (returns empty DataFrame).
    """
    if not api_key:
        print("  ℹ️   No odds API key provided — skipping Vegas odds scrape.")
        print("       Get a free key at https://the-odds-api.com (500 req/month)")
        return pd.DataFrame()

    print("  📡  Fetching Vegas odds from The Odds API...")
    params = {
        "apiKey":    api_key,
        "regions":   "us",
        "markets":   "h2h,spreads",
        "oddsFormat": "american",
        "sport":     "basketball_ncaab",
    }
    try:
        resp = SESSION.get(ODDS_API_URL, params=params, timeout=15)
        if resp.status_code == 401:
            print("  ❌  Odds API: invalid API key")
            return pd.DataFrame()
        resp.raise_for_status()
        games = resp.json()
    except Exception as e:
        print(f"  ❌  Odds API request failed: {e}")
        return pd.DataFrame()

    rows = []
    for game in games:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        home_ml = away_ml = None
        spread_home = spread_away = None

        for bookmaker in game.get("bookmakers", [])[:1]:   # use first bookmaker
            for market in bookmaker.get("markets", []):
                if market["key"] == "h2h":
                    for outcome in market.get("outcomes", []):
                        if outcome["name"] == home:
                            home_ml = outcome["price"]
                        elif outcome["name"] == away:
                            away_ml = outcome["price"]
                elif market["key"] == "spreads":
                    for outcome in market.get("outcomes", []):
                        if outcome["name"] == home:
                            spread_home = outcome.get("point", 0)
                        elif outcome["name"] == away:
                            spread_away = outcome.get("point", 0)

        if home_ml is not None and away_ml is not None:
            rows.append({
                "home_team":          home,
                "away_team":          away,
                "home_ml":            home_ml,
                "away_ml":            away_ml,
                "spread_home":        spread_home,
                "spread_away":        spread_away,
                "implied_home_prob":  _ml_to_prob(home_ml),
                "implied_away_prob":  _ml_to_prob(away_ml),
            })

    df = pd.DataFrame(rows)
    remaining = resp.headers.get("x-requests-remaining", "?")
    print(f"  ✅  Vegas odds: {len(df)} games loaded "
          f"(API requests remaining: {remaining})")
    return df


def fetch_championship_futures(api_key: str) -> Dict[str, float]:
    """
    Fetch tournament winner futures odds and convert to implied probabilities.
    Returns dict: {team_name: implied_win_prob}
    """
    if not api_key:
        return {}

    print("  📡  Fetching championship futures odds...")
    params = {
        "apiKey":     api_key,
        "regions":    "us",
        "markets":    "outrights",
        "oddsFormat": "american",
    }
    try:
        url = "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds/"
        resp = SESSION.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  ⚠️   Futures odds failed: {e}")
        return {}

    probs: Dict[str, float] = {}
    for event in data:
        for bk in event.get("bookmakers", [])[:1]:
            for market in bk.get("markets", []):
                if market.get("key") == "outrights":
                    for outcome in market.get("outcomes", []):
                        name = outcome["name"]
                        probs[name] = _ml_to_prob(outcome["price"])

    # Normalize so they sum to 1
    total = sum(probs.values()) or 1.0
    return {k: v / total for k, v in probs.items()}


# ---------------------------------------------------------------------------
# Main assembler: combine all sources into a teams DataFrame
# ---------------------------------------------------------------------------

def build_tournament_dataframe(
    year: int = 2025,
    odds_api_key: str = "",
    fetch_last10: bool = False,
) -> pd.DataFrame:
    """
    Scrape all sources and return a DataFrame ready for the optimizer.

    Columns match REQUIRED_COLUMNS in march_madness_optimizer.py plus 'adj_t'.

    Parameters
    ----------
    year          : Season year (2025 = 2024-25 season).
    odds_api_key  : Optional The Odds API key for live Vegas odds.
    fetch_last10  : If True, make per-team API calls to compute last-10 wins
                    (slow — ~64 requests). If False, estimates from record.
    """
    # 1. Barttorvik efficiency stats
    bart_df = fetch_barttorvik_stats(year)
    if bart_df.empty:
        print("\n❌  Could not load Barttorvik stats — cannot continue scrape.")
        sys.exit(1)

    # 2. Tournament field (seeds + regions)
    field = fetch_tournament_field(year)

    # 3. Vegas odds (championship futures → public_pick_pct proxy)
    time.sleep(REQUEST_DELAY)
    futures = fetch_championship_futures(odds_api_key) if odds_api_key else {}

    # 4. Game-level Vegas odds (for score prediction display)
    time.sleep(REQUEST_DELAY)
    odds_df = fetch_vegas_odds(odds_api_key) if odds_api_key else pd.DataFrame()

    # 5. Merge
    # Normalize names for fuzzy matching
    bart_df["_key"] = bart_df["team"].str.lower().str.strip()
    field_lower = {k.lower().strip(): v for k, v in field.items()}
    futures_lower = {k.lower().strip(): v for k, v in futures.items()}

    rows = []
    for _, bart_row in bart_df.iterrows():
        key = bart_row["_key"]
        bracket_info = field_lower.get(key)

        # Only include tournament teams (if bracket data available)
        if field and bracket_info is None:
            # Try partial match
            bracket_info = _fuzzy_match(key, field_lower)
            if bracket_info is None:
                continue    # not a tournament team

        seed   = bracket_info["seed"]   if bracket_info else 8
        region = bracket_info["region"] if bracket_info else "Unknown"

        # public_pick_pct: use futures implied prob if available, else seed-based default
        if futures_lower:
            pub_pct = futures_lower.get(key, _default_pub_pct(seed))
        else:
            pub_pct = _default_pub_pct(seed)

        # last10_wins
        if fetch_last10:
            time.sleep(REQUEST_DELAY)
            last10 = fetch_barttorvik_last10(bart_row["team"], year)
        else:
            last10 = _estimate_last10(bart_row.get("win_pct", 0.7))

        rows.append({
            "team":         bart_row["team"],
            "seed":         seed,
            "region":       region,
            "adj_o":        round(float(bart_row["adj_o"]), 1),
            "adj_d":        round(float(bart_row["adj_d"]), 1),
            "adj_t":        round(float(bart_row.get("adj_t", 68.5)), 1),
            "efg_pct_off":  round(float(bart_row["efg_pct_off"]), 3),
            "to_pct_off":   round(float(bart_row["to_pct_off"]), 3),
            "orb_pct":      round(float(bart_row["orb_pct"]), 3),
            "ftr_off":      round(float(bart_row["ftr_off"]), 3),
            "efg_pct_def":  round(float(bart_row["efg_pct_def"]), 3),
            "to_pct_def":   round(float(bart_row["to_pct_def"]), 3),
            "drb_pct":      round(float(bart_row["drb_pct"]), 3),
            "ftr_def":      round(float(bart_row["ftr_def"]), 3),
            "last10_wins":  last10,
            "public_pick_pct": round(pub_pct, 4),
        })

    df = pd.DataFrame(rows)

    if df.empty:
        print("\n⚠️   No tournament teams matched. "
              "Barttorvik teams may not match ESPN bracket names.")
        print("     Try running with --out to save all Barttorvik stats, "
              "then manually set seed/region in the CSV.")
        # Fall back: return all barttorvik teams with placeholder seeds
        return _fallback_full_bart_df(bart_df)

    # Sort by seed then team
    df = df.sort_values(["seed", "team"]).reset_index(drop=True)
    print(f"\n  ✅  Assembled {len(df)} tournament teams.")

    # Attach odds DataFrame as a module-level variable for downstream use
    global _SCRAPED_ODDS
    _SCRAPED_ODDS = odds_df

    return df


# Module-level cache for scraped odds (accessed by print_score_predictions)
_SCRAPED_ODDS: pd.DataFrame = pd.DataFrame()


def get_scraped_odds() -> pd.DataFrame:
    return _SCRAPED_ODDS


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _parse_win_pct(record: str) -> float:
    """Parse 'W-L' string to win percentage."""
    try:
        parts = record.split("-")
        w, l = int(parts[0]), int(parts[1])
        return w / (w + l) if (w + l) > 0 else 0.5
    except Exception:
        return 0.5


def _estimate_last10(win_pct: float) -> int:
    """Estimate last-10 wins from season win percentage."""
    return max(1, min(10, round(win_pct * 10)))


def _default_pub_pct(seed: int) -> float:
    """Seed-based default public pick percentage (used when futures unavailable)."""
    defaults = {
        1: 0.20, 2: 0.07, 3: 0.04, 4: 0.025,
        5: 0.015, 6: 0.01, 7: 0.008, 8: 0.005,
    }
    return defaults.get(seed, 0.002)


def _ml_to_prob(ml: int) -> float:
    """Convert American moneyline to implied probability (vig-inclusive)."""
    if ml > 0:
        return 100.0 / (ml + 100.0)
    else:
        return abs(ml) / (abs(ml) + 100.0)


def _fuzzy_match(key: str, field_lower: Dict) -> Optional[Dict]:
    """Try partial substring matching for team name differences."""
    # Check if any known bracket name is a substring of the barttorvik name or vice versa
    for bracket_key, info in field_lower.items():
        bk_parts = bracket_key.split()
        key_parts = key.split()
        # Match if the last word (usually school name) matches
        if bk_parts and key_parts and bk_parts[-1] == key_parts[-1]:
            return info
        # Or if one is contained in the other (handles "NC State" vs "NC St")
        if bracket_key in key or key in bracket_key:
            return info
    return None


def _fallback_full_bart_df(bart_df: pd.DataFrame) -> pd.DataFrame:
    """When bracket matching fails, return all teams with placeholder data."""
    rows = []
    for _, r in bart_df.iterrows():
        rows.append({
            "team":         r["team"],
            "seed":         8,
            "region":       "East",
            "adj_o":        round(float(r["adj_o"]), 1),
            "adj_d":        round(float(r["adj_d"]), 1),
            "adj_t":        round(float(r.get("adj_t", 68.5)), 1),
            "efg_pct_off":  round(float(r["efg_pct_off"]), 3),
            "to_pct_off":   round(float(r["to_pct_off"]), 3),
            "orb_pct":      round(float(r["orb_pct"]), 3),
            "ftr_off":      round(float(r["ftr_off"]), 3),
            "efg_pct_def":  round(float(r["efg_pct_def"]), 3),
            "to_pct_def":   round(float(r["to_pct_def"]), 3),
            "drb_pct":      round(float(r["drb_pct"]), 3),
            "ftr_def":      round(float(r["ftr_def"]), 3),
            "last10_wins":  _estimate_last10(r.get("win_pct", 0.7)),
            "public_pick_pct": _default_pub_pct(8),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI (standalone usage)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="March Madness data scraper — Barttorvik + Vegas odds"
    )
    parser.add_argument("--year",      type=int, default=2025,
                        help="Season year (default: 2025)")
    parser.add_argument("--odds-key",  type=str, default="",
                        help="The Odds API key (free at the-odds-api.com)")
    parser.add_argument("--out",       type=str, default="",
                        help="Save result to CSV path")
    parser.add_argument("--last10",    action="store_true",
                        help="Fetch per-team last-10 wins (slow, ~64 extra requests)")
    args = parser.parse_args()

    print("=" * 60)
    print("  MARCH MADNESS DATA SCRAPER")
    print("=" * 60)

    df = build_tournament_dataframe(
        year=args.year,
        odds_api_key=args.odds_key,
        fetch_last10=args.last10,
    )

    if df.empty:
        print("No data scraped.")
        sys.exit(1)

    print(f"\n{'Team':<25} {'Seed':>4} {'Region':<8} {'AdjO':>6} {'AdjD':>6} "
          f"{'AdjT':>5} {'Pub%':>6}")
    print("-" * 65)
    for _, row in df.head(20).iterrows():
        print(f"{row['team']:<25} {row['seed']:>4} {row['region']:<8} "
              f"{row['adj_o']:>6.1f} {row['adj_d']:>6.1f} "
              f"{row.get('adj_t', 68.5):>5.1f} "
              f"{row['public_pick_pct']*100:>5.1f}%")

    if len(df) > 20:
        print(f"  ... and {len(df) - 20} more rows")

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\n💾  Saved {len(df)} teams to: {args.out}")


if __name__ == "__main__":
    main()
