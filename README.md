# March Madness Bracket Optimizer

An expected-value maximizing bracket engine that combines **Monte Carlo simulation**, **machine learning win probabilities**, **game theory adjustments**, and **live data scraping** from Barttorvik and Vegas odds.

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
| 2/3/4-seed picked by < 10% of public AND top-32 in AdjO + top-40 in AdjD | EV multiplied × 1.15 |
| Championship pick not in KenPom top-25 AdjO **and** AdjD | Championship probability × 0.70 |

### 4. Score Prediction

Predicted scores use the standard **KenPom tempo-adjusted formula**:

```
possessions  = (AdjT_A + AdjT_B) / 2
score_A      = AdjO_A × AdjD_B × possessions / 10,000
score_B      = AdjO_B × AdjD_A × possessions / 10,000
```

`AdjO` and `AdjD` are points per 100 possessions normalized to a national average of 100, so dividing by 10,000 converts to actual points. `AdjT` defaults to `68.5` (national average) if not provided.

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
# Clone / navigate to the project directory
cd march-madness

# Install dependencies
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

# Scrape live Barttorvik + ESPN bracket data automatically
python march_madness_optimizer.py --scrape

# Scrape + include Vegas spreads and championship odds
python march_madness_optimizer.py --scrape --odds-key YOUR_KEY

# More simulations for higher accuracy
python march_madness_optimizer.py --csv teams.csv --sims 50000
```

---

## All CLI Options

| Flag | Default | Description |
|---|---|---|
| `--csv PATH` | — | Path to a 64-team CSV file |
| `--demo` | — | Run with generated mock data |
| `--scrape` | — | Scrape live data from Barttorvik + ESPN |
| `--year YEAR` | `2025` | Season year for the scraper |
| `--odds-key KEY` | *(from .env)* | The Odds API key for Vegas spreads |
| `--save-csv PATH` | — | Save scraped data to a CSV after scraping |
| `--last10` | off | Fetch per-team last-10 wins (slower, ~64 extra requests) |
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

Running `--scrape` pulls from three sources automatically:

### Barttorvik (`barttorvik.com`)

Fetches full-season T-Rank stats via the `getteams.php` JSON endpoint, including all Four Factors and tempo. No API key required.

### ESPN Bracket API

Pulls the current tournament bracket (seeds + regions) from ESPN's public API. Falls back to scraping the ESPN scoreboard across March dates if the bracket endpoint is unavailable.

### The Odds API (`the-odds-api.com`)

Fetches live Vegas moneylines, point spreads, and championship futures odds. Requires a free API key (500 requests/month on the free tier).

- **Point spreads** are displayed inline in the bracket output next to each game
- **Championship futures** implied probabilities are used as the `public_pick_pct` input

#### API Key Setup

Create a `.env` file in the project directory so you never need to pass the key manually:

```
odds-key = YOUR_API_KEY_HERE
```

The optimizer reads this automatically. You can still override it per-run with `--odds-key`.

Get a free key at **https://the-odds-api.com**

---

## Output

```
  ── Round of 64 (10 pts each) ─────────────────────────────────────────────
    ( 1) Duke              vs (16) Siena            →  ( 1) Duke          ~96-67
    ( 8) Ohio State        vs ( 9) TCU              →  ( 8) Ohio State    ~85-80
    ( 5) St. John's        vs (12) Northern Iowa    →  ( 5) St. John's    ~81-71  [Vegas: -7.5]
    ...

  🏆  CHAMPION:  (1) Duke
  📊  Bracket Expected Value: 1447.2 points

  ── Top 10 Championship Probabilities ──────────────────────────────────────
    Duke                    31.1%  ███████████████████████████████
    Michigan                28.4%  ████████████████████████████
    Arizona                 21.8%  █████████████████████
```

- `⚡` marks upset picks (winner has a higher seed number than the loser)
- `~79-72` is the model's predicted final score
- `[Vegas: -7.5]` is the point spread when odds data is available (negative = favorite)

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
| `march_madness_optimizer.py` | Main optimizer — ML model, Monte Carlo engine, bracket output |
| `scraper.py` | Live data scrapers for Barttorvik, ESPN, and The Odds API |
| `teams_2026_official_bracket.csv` | 2026 tournament field with full Four Factors data |
| `teams_2026_final.csv` | Alternate 2026 team dataset |
| `.env` | Local secrets (odds API key) — not committed to version control |

---

## Data Sources

| Source | Data | URL |
|---|---|---|
| Barttorvik T-Rank | Adjusted efficiency, Four Factors, tempo | barttorvik.com |
| ESPN | Tournament bracket, seeds, regions | site.api.espn.com |
| The Odds API | Vegas moneylines, spreads, futures | the-odds-api.com |
| KenPom (manual) | Can be used to fill CSV columns | kenpom.com |
