# BetEdge

Football match predictor: Poisson / Monte Carlo simulation with recency-weighted
team ratings and shot-quality (xG) data built from real shot x/y coordinates.
Runs entirely on free data and free hosting (GitHub Pages + GitHub Actions).

**Live app:** enable GitHub Pages (Settings → Pages → deploy from branch → `main`, root).

## How the model works

1. **Attack & defense strength** per team = weighted average goals scored/conceded
   (home and away separately) divided by the league average.
2. **Recency weighting** — every match is weighted `0.5^(age / 3 years)`, so a
   match from 3 years ago counts half as much as one today.
3. **Shrinkage** — teams with few matches are regressed toward the league mean.
   Confidence uses the **raw match count**, never the recency-weighted sum
   (using the weighted sum makes every team collapse to the league average — a
   real bug we hit once and fixed).
4. **Shot quality, two layers:**
   - *Club leagues:* shots + shots-on-target from football-data.co.uk are turned
     into a goals proxy (`a·SOT + b·off-target`, coefficients fitted per league by
     weighted least squares) and blended into the ratings. Lower variance than
     goals alone → less sensitive to lucky bounces.
   - *National teams:* a custom xG model (logistic regression on shot **x/y
     coordinates** → distance, goal angle, header) trained on StatsBomb open-data
     from WC 2018/2022, Euro 2020/2024 and Copa América 2024. Team xG for/against
     is blended into ratings with the same shrinkage logic (more matches with
     shot data → more trust). The model is validated by correlation against
     StatsBomb's own xG values (reported in `data/xg_model.json` and in the UI).
5. **Neutral venue switch** removes home advantage (e.g. World Cup 2026).
6. **Simulation** — expected goals for both teams → 10,000 Poisson-simulated
   matches → win/draw/loss (always sums to exactly 100%), most likely scores,
   over/under 2.5, BTTS, and a full scoreline grid.
7. **Cross-league mode** (Champions League) uses static league-strength factors —
   a rough, documented approximation.
8. **Value bets** — fetches real odds from [The Odds API](https://the-odds-api.com)
   (bring your own free key, 500 requests/month, stored only in your browser),
   removes the bookmaker vig to get market fair-value probabilities, compares them
   with the model, and shows EV (`model prob × best odds − 1`) plus a capped
   quarter-Kelly stake suggestion. Big visible disclaimers: bookmakers are usually
   right, and a positive EV is more likely model error than free money.

## Data sources (all free)

| Source | Used for |
|---|---|
| [football-data.co.uk](https://www.football-data.co.uk) | Club results + shots/SOT, 9 leagues, 8 seasons |
| [martj42/international_results](https://github.com/martj42/international_results) | National team results incl. neutral-venue flag |
| [StatsBomb open-data](https://github.com/statsbomb/open-data) | Shot x/y coordinates for the xG model |
| [The Odds API](https://the-odds-api.com) | Live odds for the value-bet scanner (own free key) |

## Repository layout

```
index.html                  the whole app (vanilla JS, no build step)
data/*.json                 precomputed ratings (served to the browser)
pipeline/fetch_statsbomb.py downloads shot events (resumable)
pipeline/build_data.py      computes ratings, trains xG model, writes data/
.github/workflows/          refreshes data twice a week automatically
```

## Updating data manually

```
python3 pipeline/fetch_statsbomb.py   # once; cached afterwards
python3 pipeline/build_data.py
```

No dependencies beyond Python 3 standard library.

## Disclaimer

For entertainment only. The model knows nothing about injuries, suspensions,
lineups or motivation. Bookmakers are usually right; nothing here is a guarantee
and this is not betting advice.
