# March Madness Bracket Optimizer

An expected-value maximizing bracket engine that combines **Monte Carlo simulation**, **machine learning win probabilities**, **game theory adjustments**, **live data scraping** from Sports-Reference and Vegas odds, and **tempo-adjusted score predictions**.

---

## How It Works

### 1. Win Probability Model

Each matchup probability is computed by a **Logistic Regression** trained on synthetic matchup pairs generated from the team pool. The feature vector for Team A vs Team B is built from **differential Four Factors**:

| Feature | Description |
|---|---|
| `adj_o_a - adj_o_b` | Offensive efficiency differential |
| `adj_d_b - adj_d_a` | Defensive efficiency differential (lower = better) |
| `efg_pct_off_a - efg_pct_off_b` | Effective FG% differential |
| `to_pct_off_b - to_pct_off_a` | Turnover rate differential (higher opp TO = better) |
| `orb_pct_a - orb_pct_b` | Offensive rebound % differential |
| `ftr_off_a - ftr_off_b` | Free throw rate differential |
| `efg_pct_def_b - efg_pct_def_a` | Opponent eFG% allowed differential |
| `to_pct_def_a - to_pct_def_b` | Steals/turnovers forced differential |
| `drb_pct_a - drb_pct_b` | Defensive rebound % differential |
| `ftr_def_b - ftr_def_a` | Free throw rate allowed differential |
| `net_eff_a - net_eff_b` | Net efficiency (AdjO − AdjD) differential |
| `momentum_a - momentum_b` | Recent form multiplier |
| `seed_a - seed_b` | Seed differential |

Training labels are derived from a **log5-style formula** on net efficiency, giving the model a statistically grounded baseline before it learns feature weights:

```
P(A wins) = 1 / (1 + 10^(-(netEff_A - netEff_B) / 12))
```

### 2. Momentum

Teams with ≥ 8 wins in their last 10 games receive a momentum multiplier of `1.0 + (last10_wins - 7) × 0.03`, up to `1.09` for a perfect 10-0 run. This shifts the logistic probability slightly in their favor.

### 3. Game Theory Adjustments

Three contrarian adjustments shift expected value (EV) away from the public consensus:

| Condition | Adjustment |
|---|---|
| 1-seed picked by > 40% of public | Win probability penalized −5% |
| 2/3/4-seed picked by < 10% of public AND top-32 AdjO + top-40 AdjD | EV multiplied × 1.15 |
| Championship pick not in top-25 both AdjO and AdjD | Championship probability × 0.70 |

### 4. Score Prediction

Predicted scores use the standard **KenPom tempo-adjusted formula**:

```
possessions  = (AdjT_A + AdjT_B) / 2
score_A      = AdjO_A × AdjD_B × possessions / 10,000
score_B      = AdjO_B × AdjD_A × possessions / 10,000
```

`AdjO` and `AdjD` are points per 100 possessions normalized to a national average of 100, so dividing by 10,000 converts to actual points. `AdjT` defaults to `68.5` (national average) if not provided; the scraper pulls real per-team tempo from Sports-Reference.

### 5. Monte Carlo Engine

The optimizer runs N full-tournament simulations (default 10,000). For each game in each simulation, it:

1. Computes `P(A wins)` via the ML model
2. Applies game theory adjustments and renormalizes
3. Samples a winner stochastically

After all simulations, it builds an **optimal bracket** by selecting the team with the highest cumulative EV per game slot, enforcing bracket consistency (a team can only appear in a game if they won all prior games in the bracket).

### 6. Upset Quota

The final bracket is checked for mandatory first-round upsets:
- At least **one 12-over-5 upset** must be picked
- At least **one 11-over-6 upset** must be picked

If either is missing, the highest-probability upset candidate from the simulations is flipped in.

---

## Installation

```bash
cd march-madness
pip install numpy pandas scikit-learn requests beautifulsoup4 lxml
```

**Python 3.9+** required.

---

## Quick Start

```bash
# Run with built-in mock data (no setup needed)
python march_madness_optimizer.py --demo

# Run with your own CSV
python march_madness_optimizer.py --csv teams_2026_official_bracket.csv

# Scrape live stats + use your bracket CSV for seeds/regions
python march_madness_optimizer.py --scrape --year 2026 --bracket-csv teams_2026_official_bracket.csv

# Same as above + Vegas spreads loaded automatically from .env
python march_madness_optimizer.py --scrape --year 2026 --bracket-csv teams_2026_official_bracket.csv

# Save the scraped data out to a CSV for reuse
python march_madness_optimizer.py --scrape --year 2026 --bracket-csv teams_2026_official_bracket.csv --save-csv live_stats.csv

# More simulations for higher accuracy
python march_madness_optimizer.py --csv teams.csv --sims 50000
```

---

## All CLI Options

| Flag | Default | Description |
|---|---|---|
| `--csv PATH` | — | Path to a 64-team CSV file |
| `--demo` | — | Run with generated mock data |
| `--scrape` | — | Scrape live efficiency stats from Sports-Reference |
| `--year YEAR` | `2025` | Sports-Reference season year (2025 = 2024-25, 2026 = 2025-26) |
| `--bracket-csv PATH` | — | CSV with `team/seed/region` columns to supply seeds and regions when the SR bracket page isn't posted yet |
| `--save-csv PATH` | — | Save scraped team data to this CSV path |
| `--odds-key KEY` | *(from .env)* | The Odds API key — auto-loaded from `.env` if present |
| `--template` | — | Export a blank CSV template and exit |
| `--sims N` | `10000` | Number of Monte Carlo simulations |
| `--seed N` | `2024` | Random seed for reproducibility |

---

## CSV Format

Provide exactly **64 teams**. Column names are case-insensitive and spaces are normalized.

### Required columns

| Column | Type | Description |
|---|---|---|
| `team` | string | Team name |
| `seed` | int | Tournament seed (1–16) |
| `region` | string | `East`, `West`, `South`, or `Midwest` |
| `adj_o` | float | Adjusted Offensive Efficiency (pts/100 poss) |
| `adj_d` | float | Adjusted Defensive Efficiency (pts/100 poss allowed) |
| `efg_pct_off` | float | Effective FG% offense (0–1 scale) |
| `to_pct_off` | float | Turnover rate offense — lower is better (0–1 scale) |
| `orb_pct` | float | Offensive rebound % (0–1 scale) |
| `ftr_off` | float | Free throw rate offense (FTA/FGA, 0–1 scale) |
| `efg_pct_def` | float | Effective FG% allowed — lower is better (0–1 scale) |
| `to_pct_def` | float | Turnover rate forced — higher is better (0–1 scale) |
| `drb_pct` | float | Defensive rebound % (0–1 scale) |
| `ftr_def` | float | Free throw rate allowed — lower is better (0–1 scale) |
| `last10_wins` | int | Wins in last 10 games (0–10) |
| `public_pick_pct` | float | Public pick % to win tournament (0–1 scale, e.g. `0.22` = 22%) |

### Optional column

| Column | Type | Default | Description |
|---|---|---|---|
| `adj_t` | float | `68.5` | Adjusted Tempo (possessions per 40 min) — used for score predictions |

### Example row

```csv
team,seed,region,adj_o,adj_d,efg_pct_off,to_pct_off,orb_pct,ftr_off,efg_pct_def,to_pct_def,drb_pct,ftr_def,last10_wins,public_pick_pct
Duke,1,East,128.1,90.8,0.568,0.157,0.381,0.378,0.462,0.181,0.752,0.237,9,0.22
```

Generate a blank template:

```bash
python march_madness_optimizer.py --template
```

---

## Live Data Scraping

Running `--scrape` pulls from three sources automatically.

### Sports-Reference CBB (`sports-reference.com`)

Scrapes two pages to get all Four Factors, offensive/defensive ratings, and tempo:

- `/{year}-advanced-school-stats.html` — offensive stats, pace, win%
- `/{year}-advanced-opponent-stats.html` — opponent (defensive) stats

> **Note:** Barttorvik was the original source but now blocks all Python HTTP clients with a JavaScript browser challenge. Sports-Reference provides equivalent data and is reliably accessible.

**Year convention:** SR uses the ending year of the season — `2025` = 2024-25 season, `2026` = 2025-26 season.

### Tournament Bracket — Seeds & Regions

The scraper tries two sources in order:

1. **Sports-Reference bracket page** (`/cbb/postseason/men/{year}-ncaa.html`) — parses seeds and regions directly from the HTML bracket. This is the primary source and works once SR posts the bracket after Selection Sunday.
2. **ESPN scoreboard fallback** — scans ESPN game data across March dates to extract seeds and regions from live game records.

**Before the SR bracket page is posted** (before or shortly after Selection Sunday), use `--bracket-csv` to supply seeds and regions from your own CSV while still pulling fresh efficiency stats from SR:

```bash
python march_madness_optimizer.py --scrape --year 2026 --bracket-csv teams_2026_official_bracket.csv
```

The `--bracket-csv` file only needs `team`, `seed`, and `region` columns — but if it also contains stat columns (as the included CSVs do), those are used as a fallback for any team that SR can't match.

**Team name matching:** SR names sometimes differ from bracket CSV names (e.g. `Connecticut` vs `UConn`, `Brigham Young` vs `BYU`). The scraper resolves these through a built-in alias table, fuzzy word-overlap matching, and a final fallback to the bracket CSV's own stats.

### The Odds API (`the-odds-api.com`)

Fetches two types of data with a single API key:

| Data | Sport key used | How it's used |
|---|---|---|
| Live moneylines + point spreads | `basketball_ncaab` | Displayed inline next to each game in the bracket output |
| Championship futures | `basketball_ncaab_championship_winner` | Implied win probabilities become each team's `public_pick_pct` input |

Free tier: **500 requests/month**. Get a key at **https://the-odds-api.com**

#### API Key Setup

Create a `.env` file in the project directory — the optimizer loads it automatically:

```
odds-key = YOUR_API_KEY_HERE
```

You can still override it per-run with `--odds-key YOUR_KEY`. The CLI flag takes priority over `.env`.

---

## Output

```
  ── Round of 64 (10 pts each) ──────────────────────────────────────────────────
    ( 1) Duke              vs (16) Siena            →  ( 1) Duke           ~82-68  [Vegas: -29.5]
    ( 8) Ohio State        vs ( 9) TCU              →  ( 8) Ohio State     ~83-82  [Vegas: -2.5]
    ( 5) St. John's        vs (12) Northern Iowa    →  ( 5) St. John's     ~74-72
    ( 4) Kansas            vs (13) Cal Baptist      →  (13) Cal Baptist   ⚡  ~74-74
    ...

  🏆  CHAMPION:  (1) Duke
  📊  Bracket Expected Value: 1399.0 points

  ── Top 10 Championship Probabilities ──────────────────────────────────────────
    Duke                    32.2%  ████████████████████████████████
    Gonzaga                 20.8%  ████████████████████
    High Point              13.4%  █████████████
```

| Marker | Meaning |
|---|---|
| `⚡` | Upset pick — winner has a higher seed number than the loser |
| `~82-68` | Model's tempo-adjusted predicted final score |
| `[Vegas: -7.5]` | Point spread (negative = that team is favored); only shown when odds data is available |

---

## Scoring System

Default ESPN Tournament Challenge scoring:

| Round | Points per correct pick |
|---|---|
| Round of 64 | 10 |
| Round of 32 | 20 |
| Sweet 16 | 40 |
| Elite 8 | 80 |
| Final Four | 160 |
| National Championship | 320 |

---

## Project Files

| File | Description |
|---|---|
| `march_madness_optimizer.py` | Main optimizer — ML model, Monte Carlo engine, score predictions, bracket output |
| `scraper.py` | Live data scrapers — Sports-Reference stats, SR/ESPN bracket, The Odds API |
| `teams_2026_official_bracket.csv` | 2025-26 tournament field with full Four Factors data |
| `teams_2026_final.csv` | Alternate 2025-26 team dataset |
| `.env` | Local secrets (odds API key) — do not commit to version control |

---

## Data Sources

| Source | Data provided | Notes |
|---|---|---|
| Sports-Reference CBB | Offensive/defensive ratings, Four Factors, tempo, win% | Primary stats source; no API key needed |
| Sports-Reference bracket page | Tournament seeds and regions | Available after Selection Sunday |
| ESPN scoreboard API | Tournament seeds and regions (fallback) | Used when SR bracket page isn't posted yet |
| The Odds API | Vegas moneylines, spreads, championship futures | Free tier: 500 req/month |
| Your bracket CSV (`--bracket-csv`) | Seeds and regions override | Use before SR bracket page is live; stat columns used as fallback for unmatched teams |
