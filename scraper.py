#!/usr/bin/env python3
"""
scraper.py — Live data scrapers for March Madness Optimizer

Sources:
  1. Sports-Reference CBB (sports-reference.com)
       - Team efficiency stats + tempo  (adv school/opponent stats pages)
       - Tournament bracket seeds + regions  (postseason NCAA bracket page)
       NOTE: Barttorvik blocks Python with a JS challenge; SR is the replacement.
  2. The Odds API (the-odds-api.com) — Vegas moneylines / spreads
       Free tier: 500 requests/month. Get a free key at https://the-odds-api.com
  3. ESPN public API — scoreboard fallback for bracket info

Usage:
  python scraper.py                            # print scraped data summary
  python scraper.py --year 2025               # specific year
  python scraper.py --odds-key YOUR_API_KEY   # include Vegas odds
  python scraper.py --out teams_scraped.csv   # save to CSV
"""

import argparse
import sys
import time
import datetime
import re
from typing import Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/json,*/*",
    "Accept-Language": "en-US,en;q=0.9",
})
REQUEST_DELAY = 1.5   # seconds between SR requests — they rate-limit aggressively


# ---------------------------------------------------------------------------
# Sports-Reference CBB — efficiency stats
# ---------------------------------------------------------------------------
SR_SCHOOL_URL = "https://www.sports-reference.com/cbb/seasons/men/{year}-advanced-school-stats.html"
SR_OPP_URL    = "https://www.sports-reference.com/cbb/seasons/men/{year}-advanced-opponent-stats.html"

# data-stat keys → our field names
SR_OFF_COLS = {
    "school_name": "team",
    "wins":        "_wins",
    "losses":      "_losses",
    "pace":        "adj_t",       # tempo (possessions/40 min)
    "off_rtg":     "adj_o",       # offensive rating pts/100 poss
    "efg_pct":     "efg_pct_off", # eFG% off — already 0-1
    "tov_pct":     "to_pct_off",  # TOV% off — SR stores as 0-100
    "orb_pct":     "orb_pct",     # ORB% off — SR stores as 0-100
    "ft_rate":     "ftr_off",     # FTA/FGA off — already 0-1
}
SR_DEF_COLS = {
    "school_name":  "team",
    "opp_off_rtg":  "adj_d",       # opp off-rating = our def efficiency
    "opp_efg_pct":  "efg_pct_def", # eFG% allowed — already 0-1
    "opp_tov_pct":  "to_pct_def",  # TOV% forced  — SR stores as 0-100
    "opp_orb_pct":  "_opp_orb",    # opp ORB% → 1 - drb_pct — SR 0-100
    "opp_ft_rate":  "ftr_def",     # FTA/FGA allowed — already 0-1
}


def fetch_sports_reference_stats(year: int = 2025) -> pd.DataFrame:
    """
    Scrape offensive + defensive advanced stats from Sports-Reference CBB.

    SR year convention: 2025 = 2024-25 season, 2026 = 2025-26 season.

    Returns a DataFrame with all Four Factors, tempo, ratings, and win%.
    """
    print(f"  📡  Fetching Sports-Reference CBB stats for {year}...")

    off_df = _fetch_sr_table(SR_SCHOOL_URL.format(year=year), "adv_school_stats",  SR_OFF_COLS)
    if off_df.empty:
        print("  ❌  Could not load offensive stats from Sports-Reference.")
        return pd.DataFrame()

    time.sleep(REQUEST_DELAY)

    def_df = _fetch_sr_table(SR_OPP_URL.format(year=year), "adv_opp_stats", SR_DEF_COLS)
    if def_df.empty:
        print("  ❌  Could not load defensive stats from Sports-Reference.")
        return pd.DataFrame()

    df = off_df.merge(def_df, on="team", how="inner")

    df["drb_pct"] = 1.0 - df["_opp_orb"]
    df["win_pct"] = df["_wins"] / (df["_wins"] + df["_losses"]).clip(lower=1)
    df.drop(columns=["_opp_orb", "_wins", "_losses"], inplace=True, errors="ignore")

    print(f"  ✅  Sports-Reference stats: {len(df)} teams loaded")
    return df


def _fetch_sr_table(url: str, table_id: str, col_map: Dict[str, str]) -> pd.DataFrame:
    """Fetch one Sports-Reference HTML table and extract the requested columns."""
    try:
        resp = SESSION.get(url, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  ❌  SR request failed ({url[-50:]}): {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", {"id": table_id})
    if table is None:
        print(f"  ❌  Table '{table_id}' not found at {url[-50:]}")
        return pd.DataFrame()

    rows = []
    for tr in table.select("tbody tr:not(.thead)"):
        cells = {td.get("data-stat"): td.get_text(strip=True) for td in tr.find_all("td")}
        if not cells.get("school_name"):
            continue

        row: Dict = {}
        for stat, field in col_map.items():
            raw = cells.get(stat, "")
            if field == "team":
                # Strip tournament markers e.g. "Duke\xa0NCAA" → "Duke"
                row["team"] = re.sub(r"\s*(NCAA|NIT|CBI|CIT)$", "", raw.replace("\xa0", " ")).strip()
            else:
                row[field] = _safe_float(raw)

        # SR stores TOV%, ORB%, opp-ORB% as 0-100 — normalise to 0-1
        for field in ("to_pct_off", "to_pct_def", "orb_pct", "_opp_orb"):
            if field in row and row[field] > 1.0:
                row[field] = row[field] / 100.0

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sports-Reference CBB — tournament bracket (seeds + regions)
# ---------------------------------------------------------------------------
SR_BRACKET_URL = "https://www.sports-reference.com/cbb/postseason/men/{year}-ncaa.html"
SR_REGIONS     = ["east", "midwest", "south", "west"]


def fetch_sr_bracket(year: int = 2025) -> Dict[str, Dict]:
    """
    Scrape seeds and regions from the Sports-Reference NCAA bracket page.

    Returns dict: {team_name: {"seed": int, "region": str}}
    """
    url = SR_BRACKET_URL.format(year=year)
    print(f"  📡  Fetching tournament bracket from Sports-Reference ({year})...")
    try:
        resp = SESSION.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  ⚠️   SR bracket request failed: {e}")
        return {}

    soup = BeautifulSoup(resp.text, "lxml")
    field: Dict[str, Dict] = {}

    for region_id in SR_REGIONS:
        region_div = soup.find("div", id=region_id)
        if not region_div:
            continue
        region_name = region_id.capitalize()

        # Each first-round matchup in the bracket div looks like:
        # "1Duke ...  16Mount St. Mary's ..."
        # We extract seed+team pairs by scanning the text of the first bracket div
        bracket_div = region_div.find("div", id="bracket")
        if not bracket_div:
            bracket_div = region_div

        # Find all team links — each has a title="seed TeamName"
        for a in bracket_div.find_all("a", href=re.compile(r"/cbb/schools/")):
            parent_text = a.parent.get_text(" ", strip=True) if a.parent else ""
            team_name = a.get_text(strip=True)
            # Seed is the number immediately before the team name in the parent text
            # e.g. "1 Duke" or just the preceding sibling text node
            seed = _extract_seed_before_link(a)
            if team_name and seed:
                field[team_name] = {"seed": seed, "region": region_name}

    if field:
        print(f"  ✅  SR bracket: {len(field)} teams found")
    else:
        print(f"  ⚠️   SR bracket page found but no teams parsed (tournament may not be posted yet for {year})")

    return field


def _extract_seed_before_link(a_tag) -> Optional[int]:
    """Find the seed number that appears immediately before a team link in the bracket HTML."""
    # Walk backwards through previous siblings looking for a number
    for sibling in reversed(list(a_tag.previous_siblings)):
        text = sibling.get_text(strip=True) if hasattr(sibling, "get_text") else str(sibling).strip()
        if text.isdigit():
            seed = int(text)
            if 1 <= seed <= 16:
                return seed
    # Also check parent text before the link
    if a_tag.parent:
        full = a_tag.parent.get_text(" ", strip=True)
        team = a_tag.get_text(strip=True)
        # Find "N TeamName" pattern
        m = re.search(r"\b(\d{1,2})\s+" + re.escape(team), full)
        if m:
            seed = int(m.group(1))
            if 1 <= seed <= 16:
                return seed
    return None


# ---------------------------------------------------------------------------
# ESPN scoreboard fallback (for when SR bracket page isn't up yet)
# ---------------------------------------------------------------------------
ESPN_SCOREBOARD_API = (
    "https://site.api.espn.com/apis/v2/sports/basketball"
    "/mens-college-basketball/scoreboard"
)
REGION_KEYWORDS = {
    "East":    ["East"],
    "West":    ["West"],
    "South":   ["South"],
    "Midwest": ["Midwest", "Mid"],
}


def _fetch_bracket_from_espn_scoreboard(year: int) -> Dict[str, Dict]:
    """Scan ESPN scoreboard for tournament game dates to extract seeds + regions."""
    field: Dict[str, Dict] = {}
    start = datetime.date(year, 3, 15)
    for offset in range(25):
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
                note = (event.get("notes") or [{}])[0].get("headline", "")
                region = _parse_region(note)
                for comp in event.get("competitions", [{}])[0].get("competitors", []):
                    tname = comp.get("team", {}).get("displayName", "")
                    seed  = comp.get("seed") or comp.get("curatedRank", {}).get("current", 0)
                    if tname and seed:
                        field[tname] = {"seed": int(seed), "region": region}
            time.sleep(REQUEST_DELAY)
        except Exception:
            continue
    return field


def _parse_region(note: str) -> str:
    for region, keywords in REGION_KEYWORDS.items():
        if any(kw.lower() in note.lower() for kw in keywords):
            return region
    return "Unknown"


def fetch_tournament_field(year: int = 2025) -> Dict[str, Dict]:
    """
    Try SR bracket page first, then ESPN scoreboard fallback.

    Returns dict: {team_name: {"seed": int, "region": str}}
    """
    # 1. Sports-Reference bracket page (most reliable when posted)
    field = fetch_sr_bracket(year)
    if field:
        return field

    # 2. ESPN scoreboard scan
    print(f"  ⚠️   Trying ESPN scoreboard fallback for bracket data...")
    field = _fetch_bracket_from_espn_scoreboard(year)
    if field:
        print(f"  ✅  ESPN scoreboard: {len(field)} teams found")
        return field

    print("  ⚠️   Could not fetch bracket automatically.")
    print("        Use --bracket-csv to provide seeds/regions from your own CSV.")
    return {}


# ---------------------------------------------------------------------------
# Vegas odds  (The Odds API)
# ---------------------------------------------------------------------------
ODDS_BASE_URL    = "https://api.the-odds-api.com/v4/sports/{sport}/odds/"
NCAAB_SPORT      = "basketball_ncaab"
NCAAB_FUTURES_SPORT = "basketball_ncaab_championship_winner"


def fetch_vegas_odds(api_key: str) -> pd.DataFrame:
    """
    Fetch current Vegas moneylines + point spreads for NCAAB games.

    Returns DataFrame with columns:
      home_team, away_team, home_ml, away_ml, spread_home, implied_home_prob, implied_away_prob
    """
    if not api_key:
        print("  ℹ️   No odds API key — skipping game odds.")
        return pd.DataFrame()

    print("  📡  Fetching Vegas game odds from The Odds API...")
    params = {
        "apiKey":     api_key,
        "regions":    "us",
        "markets":    "h2h,spreads",
        "oddsFormat": "american",
    }
    try:
        resp = SESSION.get(ODDS_BASE_URL.format(sport=NCAAB_SPORT), params=params, timeout=15)
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
        spread_home = None

        for bookmaker in game.get("bookmakers", [])[:1]:
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
                            spread_home = outcome.get("point")

        if home_ml is not None and away_ml is not None:
            rows.append({
                "home_team":         home,
                "away_team":         away,
                "home_ml":           home_ml,
                "away_ml":           away_ml,
                "spread_home":       spread_home,
                "implied_home_prob": _ml_to_prob(home_ml),
                "implied_away_prob": _ml_to_prob(away_ml),
            })

    df = pd.DataFrame(rows)
    remaining = resp.headers.get("x-requests-remaining", "?")
    print(f"  ✅  Vegas odds: {len(df)} games loaded (requests remaining: {remaining})")
    return df


def fetch_championship_futures(api_key: str) -> Dict[str, float]:
    """
    Fetch tournament winner futures and convert to implied probabilities.
    Returns dict: {team_name: implied_win_prob}  (normalised to sum to 1)
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
        resp = SESSION.get(
            ODDS_BASE_URL.format(sport=NCAAB_FUTURES_SPORT),
            params=params,
            timeout=15,
        )
        if resp.status_code == 401:
            print("  ❌  Odds API: invalid API key")
            return {}
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
                        probs[outcome["name"]] = _ml_to_prob(outcome["price"])

    if not probs:
        return {}

    total = sum(probs.values()) or 1.0
    normalised = {k: v / total for k, v in probs.items()}
    print(f"  ✅  Futures: {len(normalised)} teams with odds")
    return normalised


# ---------------------------------------------------------------------------
# Main assembler
# ---------------------------------------------------------------------------

def build_tournament_dataframe(
    year: int = 2025,
    odds_api_key: str = "",
    fetch_last10: bool = False,
    bracket_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Scrape all sources and return a DataFrame ready for the optimizer.

    Parameters
    ----------
    year          : SR season year (2025 = 2024-25 season).
    odds_api_key  : The Odds API key for live Vegas data.
    fetch_last10  : Unused (kept for API compatibility).
    bracket_df    : Optional DataFrame with columns [team, seed, region] to
                    override the scraped bracket (use when SR page isn't up yet).
    """
    # 1. Efficiency stats from Sports-Reference
    stats_df = fetch_sports_reference_stats(year)
    if stats_df.empty:
        print("\n❌  Could not load team stats.")
        sys.exit(1)

    # 2. Tournament bracket — use provided CSV override, else scrape
    if bracket_df is not None and not bracket_df.empty:
        print("  📋  Using bracket data from --bracket-csv")
        field = {
            str(row["team"]).strip(): {
                "seed":   int(row["seed"]),
                "region": str(row["region"]).strip(),
            }
            for _, row in bracket_df.iterrows()
        }
    else:
        time.sleep(REQUEST_DELAY)
        field = fetch_tournament_field(year)

    # 3. Championship futures → public_pick_pct
    time.sleep(REQUEST_DELAY)
    futures = fetch_championship_futures(odds_api_key) if odds_api_key else {}

    # 4. Live game odds
    time.sleep(REQUEST_DELAY)
    odds_df = fetch_vegas_odds(odds_api_key) if odds_api_key else pd.DataFrame()

    # 5. Merge: iterate over bracket teams → find best SR stat match (1-to-1)
    # Build a lookup from normalised SR name → row
    stats_df["_key"] = stats_df["team"].str.lower().str.strip()
    stats_lookup = {row["_key"]: row for _, row in stats_df.iterrows()}
    futures_lower = {k.lower().strip(): v for k, v in futures.items()}

    # Build a lookup from the bracket CSV's own stats (fallback for unmatched teams)
    bracket_stats_lookup: Dict = {}
    if bracket_df is not None and not bracket_df.empty:
        stat_cols = ["adj_o", "adj_d", "efg_pct_off", "to_pct_off", "orb_pct",
                     "ftr_off", "efg_pct_def", "to_pct_def", "drb_pct", "ftr_def",
                     "last10_wins", "public_pick_pct"]
        if all(c in bracket_df.columns for c in stat_cols):
            bracket_stats_lookup = {
                str(row["team"]).strip().lower(): row
                for _, row in bracket_df.iterrows()
            }

    rows = []
    unmatched = []

    if field:
        for bracket_name, bracket_info in field.items():
            key = bracket_name.lower().strip()

            # Try SR lookup: exact → alias → fuzzy
            r = stats_lookup.get(key)
            if r is None:
                alias = _TEAM_ALIASES.get(key)
                r = stats_lookup.get(alias) if alias else None
            if r is None:
                fk = _fuzzy_key(key, stats_lookup)
                r = stats_lookup.get(fk) if fk else None

            seed   = bracket_info["seed"]
            region = bracket_info["region"]
            pub_pct = futures_lower.get(key)
            if pub_pct is None:
                fk = _fuzzy_key(key, futures_lower)
                pub_pct = futures_lower.get(fk) if fk else None
            if pub_pct is None:
                pub_pct = _default_pub_pct(seed)

            if r is not None:
                rows.append(_build_row(r, seed, region, pub_pct))
            elif key in bracket_stats_lookup:
                # Fall back to bracket CSV's own stats
                br = bracket_stats_lookup[key]
                rows.append({
                    "team":            bracket_name,
                    "seed":            seed,
                    "region":          region,
                    "adj_o":           round(float(br["adj_o"]),       1),
                    "adj_d":           round(float(br["adj_d"]),       1),
                    "adj_t":           round(float(br.get("adj_t", 68.5)), 1),
                    "efg_pct_off":     round(float(br["efg_pct_off"]), 3),
                    "to_pct_off":      round(float(br["to_pct_off"]),  3),
                    "orb_pct":         round(float(br["orb_pct"]),     3),
                    "ftr_off":         round(float(br["ftr_off"]),     3),
                    "efg_pct_def":     round(float(br["efg_pct_def"]), 3),
                    "to_pct_def":      round(float(br["to_pct_def"]),  3),
                    "drb_pct":         round(float(br["drb_pct"]),     3),
                    "ftr_def":         round(float(br["ftr_def"]),     3),
                    "last10_wins":     int(br["last10_wins"]),
                    "public_pick_pct": round(pub_pct, 4),
                })
            else:
                unmatched.append(bracket_name)

        if unmatched:
            print(f"  ⚠️   {len(unmatched)} bracket teams could not be matched: "
                  f"{', '.join(unmatched)}")
    else:
        # No bracket data — include all SR teams with placeholder seed/region
        for _, r in stats_df.iterrows():
            rows.append(_build_row(r, 8, "East", _default_pub_pct(8)))

    df = pd.DataFrame(rows)

    if df.empty:
        print("\n⚠️   No teams matched. Returning all SR teams.")
        print("     Re-run with --bracket-csv your_bracket.csv to map seeds/regions.")
        df = _fallback_full_df(stats_df)

    df = df.sort_values(["seed", "team"]).reset_index(drop=True)
    print(f"\n  ✅  Assembled {len(df)} tournament teams.")

    global _SCRAPED_ODDS
    _SCRAPED_ODDS = odds_df

    return df


_SCRAPED_ODDS: pd.DataFrame = pd.DataFrame()


def get_scraped_odds() -> pd.DataFrame:
    return _SCRAPED_ODDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Common name differences between bracket CSVs and Sports-Reference
# Keys are lowercase normalised bracket names; values are SR names (lowercase)
_TEAM_ALIASES: Dict[str, str] = {
    "uconn":                  "connecticut",
    "byu":                    "brigham young",
    "miami fl":               "miami (fl)",
    "miami (fl)":             "miami (fl)",
    "smu":                    "southern methodist",
    "vcu":                    "virginia commonwealth",
    "nc state":               "nc state",
    "lsu":                    "lsu",
    "uab":                    "uab",
    "unlv":                   "unlv",
    "utep":                   "utep",
    "tcu":                    "tcu",
    "ucf":                    "ucf",
    "usc":                    "usc",
    "utsa":                   "utsa",
    "fiu":                    "fiu",
    "fau":                    "florida atlantic",
    "saint mary's":           "saint mary's (ca)",
    "st. mary's":             "saint mary's (ca)",
    "mcneese":                "mcneese state",
    "unc":                    "north carolina",
    "pitt":                   "pittsburgh",
    "ucsb":                   "uc santa barbara",
    "siue":                   "siu edwardsville",
    "siu-edwardsville":       "siu edwardsville",
    "app state":              "appalachian state",
    "appalachian st.":        "appalachian state",
    "long island":            "long island university",
    "liu":                    "long island university",
    "prairie view a&m":       "prairie view",
    "prairie view":           "prairie view",
    "queens (nc)":            "queens",
    "queens nc":              "queens",
    "north dakota st.":       "north dakota state",
    "south dakota st.":       "south dakota state",
    "wright st.":             "wright state",
    "kennesaw st.":           "kennesaw state",
    "morehead st.":           "morehead state",
    "montana st.":            "montana state",
    "cal baptist":            "california baptist",
    "saint louis":            "saint louis",
    "st. john's":             "st. john's (ny)",
    "miami hurricanes":       "miami (fl)",
}


def _build_row(r, seed: int, region: str, pub_pct: float) -> Dict:
    """Build a single output row from an SR stats row + bracket info."""
    return {
        "team":            r["team"],
        "seed":            seed,
        "region":          region,
        "adj_o":           round(float(r["adj_o"]),       1),
        "adj_d":           round(float(r["adj_d"]),       1),
        "adj_t":           round(float(r.get("adj_t", 68.5)), 1),
        "efg_pct_off":     round(float(r["efg_pct_off"]), 3),
        "to_pct_off":      round(float(r["to_pct_off"]),  3),
        "orb_pct":         round(float(r["orb_pct"]),     3),
        "ftr_off":         round(float(r["ftr_off"]),     3),
        "efg_pct_def":     round(float(r["efg_pct_def"]), 3),
        "to_pct_def":      round(float(r["to_pct_def"]),  3),
        "drb_pct":         round(float(r["drb_pct"]),     3),
        "ftr_def":         round(float(r["ftr_def"]),     3),
        "last10_wins":     _estimate_last10(float(r.get("win_pct", 0.7))),
        "public_pick_pct": round(pub_pct, 4),
    }


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(str(v).replace("%", "").strip())
    except (TypeError, ValueError):
        return default


def _estimate_last10(win_pct: float) -> int:
    return max(1, min(10, round(win_pct * 10)))


def _default_pub_pct(seed: int) -> float:
    defaults = {1: 0.20, 2: 0.07, 3: 0.04, 4: 0.025,
                5: 0.015, 6: 0.01, 7: 0.008, 8: 0.005}
    return defaults.get(seed, 0.002)


def _ml_to_prob(ml: int) -> float:
    if ml > 0:
        return 100.0 / (ml + 100.0)
    return abs(ml) / (abs(ml) + 100.0)


def _fuzzy_key(key: str, d: Dict) -> Optional[str]:
    """
    Find the best matching key in dict d for the given normalised team name.

    Rules (in priority order):
      1. Exact match (already handled by caller)
      2. All words in the shorter name appear in the longer name AND the
         longer name has at most 2 extra words  (e.g. "gonzaga" → "gonzaga bulldogs")
      3. Last significant word matches and names share ≥ 60% of words
         (handles "st. john's" → "st. john's (ny)")
    """
    key_parts = set(key.split())
    key_list  = key.split()

    best: Optional[str] = None
    best_score = 0

    for dk in d:
        dk_parts = set(dk.split())
        dk_list  = dk.split()

        # Rule 1: contained — shorter is fully inside longer with ≤2 extra words
        shorter, longer = (key_parts, dk_parts) if len(key_parts) <= len(dk_parts) else (dk_parts, key_parts)
        if shorter and shorter.issubset(longer) and len(longer) - len(shorter) <= 2:
            extra = len(longer) - len(shorter)
            score = 100 - extra * 10
            if score > best_score:
                best_score = score
                best = dk
            continue

        # Rule 2: last word matches + Jaccard similarity ≥ 0.5
        if key_list and dk_list and key_list[-1] == dk_list[-1]:
            union = key_parts | dk_parts
            jaccard = len(key_parts & dk_parts) / len(union) if union else 0
            if jaccard >= 0.5:
                score = int(jaccard * 80)
                if score > best_score:
                    best_score = score
                    best = dk

    return best


def _fallback_full_df(stats_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    regions = ["East", "West", "South", "Midwest"]
    for i, (_, r) in enumerate(stats_df.iterrows()):
        rows.append({
            "team": r["team"], "seed": 8, "region": regions[i % 4],
            "adj_o": round(float(r["adj_o"]), 1),
            "adj_d": round(float(r["adj_d"]), 1),
            "adj_t": round(float(r.get("adj_t", 68.5)), 1),
            "efg_pct_off": round(float(r["efg_pct_off"]), 3),
            "to_pct_off":  round(float(r["to_pct_off"]),  3),
            "orb_pct":     round(float(r["orb_pct"]),     3),
            "ftr_off":     round(float(r["ftr_off"]),     3),
            "efg_pct_def": round(float(r["efg_pct_def"]), 3),
            "to_pct_def":  round(float(r["to_pct_def"]),  3),
            "drb_pct":     round(float(r["drb_pct"]),     3),
            "ftr_def":     round(float(r["ftr_def"]),     3),
            "last10_wins": _estimate_last10(float(r.get("win_pct", 0.7))),
            "public_pick_pct": _default_pub_pct(8),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="March Madness data scraper — Sports-Reference + Vegas odds"
    )
    parser.add_argument("--year",     type=int, default=2025)
    parser.add_argument("--odds-key", type=str, default="")
    parser.add_argument("--out",      type=str, default="")
    args = parser.parse_args()

    print("=" * 60)
    print("  MARCH MADNESS DATA SCRAPER")
    print("=" * 60)

    df = build_tournament_dataframe(year=args.year, odds_api_key=args.odds_key)

    if df.empty:
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
