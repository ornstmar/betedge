#!/usr/bin/env python3
"""BetEdge data pipeline.

Builds team ratings from free sources:
  * football-data.co.uk        - club league results + shots/shots-on-target
  * martj42/international_results - national team results since 1872
  * StatsBomb open-data        - shot x/y coordinates -> custom xG model,
                                 shot-quality ratings for national teams

Core model (same logic as the original app):
  * attack/defense strength = team weighted goal averages vs league average,
    home and away computed separately
  * recency weighting with a 3-year half-life
  * shrinkage toward the league mean using RAW match counts
    (never the recency-weighted sums - see README, "the shrinkage bug")
  * shot-based ratings blended in with the same shrinkage logic

Run:  python3 build_data.py          (writes ../data/*.json)
"""
import csv
import io
import json
import math
import os
import re
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
CACHE = os.path.join(HERE, "cache")

TODAY = date.today()
HALF_LIFE_DAYS = 3 * 365.25          # recency half-life
K_SHRINK = 12                        # shrinkage: matches for 50/50 trust
K_SHOTS = 20                         # shots-proxy blend confidence scale
BETA_SHOTS = 0.45                    # max weight of shots-proxy rating
K_SB = 8                             # StatsBomb blend confidence scale
BETA_SB = 0.40                       # max weight of shot-quality rating
PENALTY_XG = 0.76
FRIENDLY_WEIGHT = 0.6
CLUB_SEASONS = ["1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526", "2627"]
INTL_SINCE = date(2012, 1, 1)
RECENT_SEASONS_KEPT = 2              # a club team appears if seen in last N seasons

LEAGUES = [
    ("E0", "Premier League", "England"),
    ("E1", "Championship", "England"),
    ("SP1", "La Liga", "Spain"),
    ("D1", "Bundesliga", "Germany"),
    ("I1", "Serie A", "Italy"),
    ("F1", "Ligue 1", "France"),
    ("N1", "Eredivisie", "Netherlands"),
    ("P1", "Primeira Liga", "Portugal"),
    ("BRA", "Brasileirao Serie A", "Brazil"),
]

# Rough relative league strength for cross-league (Champions League) matchups.
# Static, documented approximation based on UEFA coefficients / club Elo levels.
LEAGUE_STRENGTH = {
    "E0": 1.00, "SP1": 0.94, "D1": 0.90, "I1": 0.90, "F1": 0.85,
    "P1": 0.74, "N1": 0.72, "BRA": 0.76, "E1": 0.62, "INT": None,
}

# StatsBomb -> martj42 national team name aliases
TEAM_ALIASES = {
    "Türkiye": "Turkey", "USA": "United States", "Korea Republic": "South Korea",
    "IR Iran": "Iran", "Czechia": "Czech Republic", "Côte d'Ivoire": "Ivory Coast",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
}


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "betedge"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def recency_weight(d, importance=1.0):
    age = (TODAY - d).days
    if age < 0:
        age = 0
    return importance * 0.5 ** (age / HALF_LIFE_DAYS)


def shrink(rating, n_raw, k=K_SHRINK):
    """Regress toward 1.0 (league mean). n_raw MUST be the raw match count."""
    return (n_raw * rating + k * 1.0) / (n_raw + k)


def parse_date(s):
    s = s.strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


# --------------------------------------------------------------------------
# 1. Load matches
# --------------------------------------------------------------------------

def load_club_league(code):
    """Return list of match dicts for one football-data.co.uk league."""
    matches = []
    if code == "BRA":
        try:
            text = fetch("https://www.football-data.co.uk/new/BRA.csv")
        except Exception as e:  # noqa: BLE001
            print(f"  BRA fetch failed: {e}")
            return matches
        for row in csv.DictReader(io.StringIO(text)):
            d = parse_date(row.get("Date", ""))
            if not d or not row.get("HG") or not row.get("AG"):
                continue
            matches.append({
                "date": d, "home": row["Home"].strip(), "away": row["Away"].strip(),
                "hg": int(float(row["HG"])), "ag": int(float(row["AG"])),
                "hs": None, "as": None, "hst": None, "ast": None,
                "season": row.get("Season", ""),
            })
        return matches

    for season in CLUB_SEASONS:
        url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
        try:
            text = fetch(url)
        except Exception:  # noqa: BLE001 - future seasons 404, that's fine
            continue
        for row in csv.DictReader(io.StringIO(text)):
            if not row.get("Date") or not row.get("FTHG"):
                continue
            d = parse_date(row["Date"])
            if not d:
                continue

            def num(key):
                v = (row.get(key) or "").strip()
                return int(float(v)) if v else None

            matches.append({
                "date": d, "home": row["HomeTeam"].strip(), "away": row["AwayTeam"].strip(),
                "hg": num("FTHG"), "ag": num("FTAG"),
                "hs": num("HS"), "as": num("AS"), "hst": num("HST"), "ast": num("AST"),
                "season": season,
            })
    return matches


def load_internationals():
    text = fetch("https://raw.githubusercontent.com/martj42/international_results/master/results.csv")
    matches = []
    for row in csv.DictReader(io.StringIO(text)):
        d = parse_date(row["date"])
        if not d or d < INTL_SINCE or d > TODAY:
            continue
        try:
            hg, ag = int(float(row["home_score"])), int(float(row["away_score"]))
        except (ValueError, TypeError):
            continue  # unplayed / 'NA' rows
        matches.append({
            "date": d, "home": row["home_team"].strip(), "away": row["away_team"].strip(),
            "hg": hg, "ag": ag,
            "neutral": row["neutral"].strip().upper() == "TRUE",
            "friendly": row["tournament"].strip() == "Friendly",
            "tournament": row["tournament"].strip(),
        })
    return matches


# Teams must have played in (or qualified for) a major tournament to appear.
MAJOR_TOURNAMENT_RE = re.compile(
    r"World Cup|UEFA Euro|Copa Am|Nations League|Gold Cup|Asian Cup|"
    r"African Cup|Africa Cup|Confederations", re.IGNORECASE)


# --------------------------------------------------------------------------
# 2. Goals-based ratings
# --------------------------------------------------------------------------

class Acc:
    """Weighted accumulator: goals for/against + raw count."""

    __slots__ = ("w", "gf", "ga", "n")

    def __init__(self):
        self.w = self.gf = self.ga = 0.0
        self.n = 0

    def add(self, w, gf, ga):
        self.w += w
        self.gf += w * gf
        self.ga += w * ga
        self.n += 1

    def avg_for(self):
        return self.gf / self.w if self.w > 0 else None

    def avg_against(self):
        return self.ga / self.w if self.w > 0 else None


def compute_club_ratings(matches):
    """Home/away attack & defense vs league mean, with shots-proxy blend."""
    sw = sh = sa = 0.0
    for m in matches:
        w = recency_weight(m["date"])
        sw += w
        sh += w * m["hg"]
        sa += w * m["ag"]
    mu_h, mu_a = sh / sw, sa / sw

    # Shots-proxy conversion rates: weighted least squares
    # goals ~ a*SOT + b*(shots off target), fit on all team-matches with data.
    s11 = s12 = s22 = t1 = t2 = 0.0
    for m in matches:
        if m["hst"] is None or m["hs"] is None:
            continue
        w = recency_weight(m["date"])
        for sot, off, g in ((m["hst"], m["hs"] - m["hst"], m["hg"]),
                            (m["ast"], m["as"] - m["ast"], m["ag"])):
            if sot is None or off is None or off < 0:
                continue
            s11 += w * sot * sot
            s12 += w * sot * off
            s22 += w * off * off
            t1 += w * sot * g
            t2 += w * off * g
    det = s11 * s22 - s12 * s12
    if det > 1e-9:
        conv_on = (t1 * s22 - t2 * s12) / det
        conv_off = (t2 * s11 - t1 * s12) / det
        conv_on = max(conv_on, 0.05)
        conv_off = max(conv_off, 0.0)
    else:
        conv_on, conv_off = 0.30, 0.03

    def proxy(sot, off):
        return conv_on * sot + conv_off * off

    home_acc = defaultdict(Acc)
    away_acc = defaultdict(Acc)
    ph_acc = defaultdict(Acc)   # shots-proxy accumulators
    pa_acc = defaultdict(Acc)
    psw = psh = psa = 0.0       # league proxy means
    seasons_seen = defaultdict(set)

    for m in matches:
        w = recency_weight(m["date"])
        home_acc[m["home"]].add(w, m["hg"], m["ag"])
        away_acc[m["away"]].add(w, m["ag"], m["hg"])
        seasons_seen[m["home"]].add(m["season"])
        seasons_seen[m["away"]].add(m["season"])
        if m["hst"] is not None and m["hs"] is not None and m["ast"] is not None:
            pxh = proxy(m["hst"], max(m["hs"] - m["hst"], 0))
            pxa = proxy(m["ast"], max(m["as"] - m["ast"], 0))
            ph_acc[m["home"]].add(w, pxh, pxa)
            pa_acc[m["away"]].add(w, pxa, pxh)
            psw += w
            psh += w * pxh
            psa += w * pxa
    pmu_h = psh / psw if psw > 0 else None
    pmu_a = psa / psw if psw > 0 else None

    all_seasons = sorted({m["season"] for m in matches})
    recent = set(all_seasons[-RECENT_SEASONS_KEPT:])

    teams = {}
    for t in set(home_acc) | set(away_acc):
        if not (seasons_seen[t] & recent):
            continue
        h, a = home_acc[t], away_acc[t]
        if h.n + a.n < 5:
            continue

        def rate(acc, avg_fn, mu):
            v = avg_fn(acc)
            return shrink(v / mu, acc.n) if v is not None and mu else 1.0

        att_h = rate(h, Acc.avg_for, mu_h)
        def_h = rate(h, Acc.avg_against, mu_a)
        att_a = rate(a, Acc.avg_for, mu_a)
        def_a = rate(a, Acc.avg_against, mu_h)

        # Blend in shots-proxy ratings (lower variance than goals alone)
        n_shot = ph_acc[t].n + pa_acc[t].n
        if n_shot > 0 and pmu_h and pmu_a:
            beta = BETA_SHOTS * n_shot / (n_shot + K_SHOTS)
            p_att_h = rate(ph_acc[t], Acc.avg_for, pmu_h)
            p_def_h = rate(ph_acc[t], Acc.avg_against, pmu_a)
            p_att_a = rate(pa_acc[t], Acc.avg_for, pmu_a)
            p_def_a = rate(pa_acc[t], Acc.avg_against, pmu_h)
            att_h = (1 - beta) * att_h + beta * p_att_h
            def_h = (1 - beta) * def_h + beta * p_def_h
            att_a = (1 - beta) * att_a + beta * p_att_a
            def_a = (1 - beta) * def_a + beta * p_def_a

        teams[t] = {
            "att_h": round(att_h, 4), "def_h": round(def_h, 4),
            "att_a": round(att_a, 4), "def_a": round(def_a, 4),
            "att_n": round((att_h + att_a) / 2, 4),
            "def_n": round((def_h + def_a) / 2, 4),
            "n_home": h.n, "n_away": a.n, "n_shot": n_shot, "n_sb": 0,
        }
    return {
        "mu_h": round(mu_h, 4), "mu_a": round(mu_a, 4),
        "conv_on": round(conv_on, 4), "conv_off": round(conv_off, 4),
        "teams": teams,
    }


def compute_intl_ratings(matches, sq=None):
    """International ratings. Home/away from non-neutral games, overall from all.

    sq: optional {team: (sq_att, sq_def, n_matches)} StatsBomb shot-quality
    ratings, blended with shrinkage on the raw StatsBomb match count.
    """
    sq = sq or {}
    sw = sh = sa = 0.0            # non-neutral means
    swn = sgn = 0.0               # neutral: goals per team per match
    for m in matches:
        w = recency_weight(m["date"], FRIENDLY_WEIGHT if m["friendly"] else 1.0)
        if m["neutral"]:
            swn += w
            sgn += w * (m["hg"] + m["ag"]) / 2.0
        else:
            sw += w
            sh += w * m["hg"]
            sa += w * m["ag"]
    mu_h, mu_a = sh / sw, sa / sw
    mu_n = sgn / swn if swn > 0 else (mu_h + mu_a) / 2

    home_acc = defaultdict(Acc)
    away_acc = defaultdict(Acc)
    all_acc = defaultdict(Acc)
    last_played = {}
    for m in matches:
        w = recency_weight(m["date"], FRIENDLY_WEIGHT if m["friendly"] else 1.0)
        if not m["neutral"]:
            home_acc[m["home"]].add(w, m["hg"], m["ag"])
            away_acc[m["away"]].add(w, m["ag"], m["hg"])
        all_acc[m["home"]].add(w, m["hg"], m["ag"])
        all_acc[m["away"]].add(w, m["ag"], m["hg"])
        for t in (m["home"], m["away"]):
            last_played[t] = max(last_played.get(t, m["date"]), m["date"])

    mu_all = (mu_h + mu_a) / 2

    # eligibility: must have real major-tournament history + be active
    major_n = defaultdict(int)
    for m in matches:
        if MAJOR_TOURNAMENT_RE.search(m["tournament"]):
            major_n[m["home"]] += 1
            major_n[m["away"]] += 1
    eligible = {t for t, acc in all_acc.items()
                if acc.n >= 10 and major_n[t] >= 8
                and (TODAY - last_played[t]).days <= 730}

    # Opponent-adjusted attack/defense (fixes strength-of-schedule bias:
    # goals vs weak confederations should count less). Iterative fit of
    # gf ~ mu * att(team) * def(opponent), recency weighted.
    att = {t: 1.0 for t in all_acc}
    deff = {t: 1.0 for t in all_acc}
    rows = []
    for m in matches:
        w = recency_weight(m["date"], FRIENDLY_WEIGHT if m["friendly"] else 1.0)
        mu_home, mu_away = (mu_n, mu_n) if m["neutral"] else (mu_h, mu_a)
        rows.append((m["home"], m["away"], w, m["hg"], mu_home))
        rows.append((m["away"], m["home"], w, m["ag"], mu_away))
    for _ in range(12):
        num_a = defaultdict(float); den_a = defaultdict(float)
        num_d = defaultdict(float); den_d = defaultdict(float)
        for team, opp, w, gf, mu in rows:
            num_a[team] += w * gf
            den_a[team] += w * mu * deff[opp]
            num_d[opp] += w * gf
            den_d[opp] += w * mu * att[team]
        for t in all_acc:
            if den_a[t] > 0:
                att[t] = num_a[t] / den_a[t]
            if den_d[t] > 0:
                deff[t] = num_d[t] / den_d[t]
        # renormalise so the weighted means stay at 1.0
        tw = sum(a.w for a in all_acc.values())
        ma = sum(all_acc[t].w * att[t] for t in all_acc) / tw
        md = sum(all_acc[t].w * deff[t] for t in all_acc) / tw
        for t in all_acc:
            att[t] /= ma
            deff[t] /= md

    teams = {}
    for t in eligible:
        acc = all_acc[t]
        att_n = shrink(att[t], acc.n)
        def_n = shrink(deff[t], acc.n)
        raw_att = acc.avg_for() / mu_all if acc.avg_for() else 1.0
        raw_def = acc.avg_against() / mu_all if acc.avg_against() else 1.0
        adj_f_att = att_n / shrink(raw_att, acc.n)
        adj_f_def = def_n / shrink(raw_def, acc.n)
        h, a = home_acc[t], away_acc[t]
        # home/away splits, with the same opponent adjustment applied
        att_h = shrink(h.avg_for() / mu_h, h.n) * adj_f_att if h.w > 0 else att_n
        def_h = shrink(h.avg_against() / mu_a, h.n) * adj_f_def if h.w > 0 else def_n
        att_a = shrink(a.avg_for() / mu_a, a.n) * adj_f_att if a.w > 0 else att_n
        def_a = shrink(a.avg_against() / mu_h, a.n) * adj_f_def if a.w > 0 else def_n

        n_sb = 0
        if t in sq:
            sq_att, sq_def, n_sb = sq[t]
            beta = BETA_SB * n_sb / (n_sb + K_SB)
            f_att = ((1 - beta) * att_n + beta * sq_att) / att_n if att_n > 0 else 1.0
            f_def = ((1 - beta) * def_n + beta * sq_def) / def_n if def_n > 0 else 1.0
            att_n *= f_att
            def_n *= f_def
            att_h *= f_att
            def_h *= f_def
            att_a *= f_att
            def_a *= f_def

        teams[t] = {
            "att_h": round(att_h, 4), "def_h": round(def_h, 4),
            "att_a": round(att_a, 4), "def_a": round(def_a, 4),
            "att_n": round(att_n, 4), "def_n": round(def_n, 4),
            "n_home": h.n, "n_away": a.n, "n_shot": 0, "n_sb": n_sb,
        }
    return {
        "mu_h": round(mu_h, 4), "mu_a": round(mu_a, 4), "mu_n": round(mu_n, 4),
        "teams": teams,
    }


# --------------------------------------------------------------------------
# 3. StatsBomb: xG from shot x/y coordinates -> team shot quality
# --------------------------------------------------------------------------

def shot_features(s):
    """Pitch is 120x80, goal centre at (120, 40), posts at y=36..44."""
    x, y = s["x"], s["y"]
    if x is None or y is None:
        return None
    dx = 120.0 - x
    dist = math.hypot(dx, y - 40.0)
    # angle subtended by the two goalposts (Hewitt's shot angle)
    a1 = math.atan2(y - 36.0, dx) if dx > 0 else math.pi / 2
    a2 = math.atan2(y - 44.0, dx) if dx > 0 else -math.pi / 2
    angle = abs(a1 - a2)
    header = 1.0 if s.get("body_part") == "Head" else 0.0
    return [1.0, dist, angle, header]


def train_xg_model(shots):
    """Logistic regression (pure python gradient descent) on shot x/y features."""
    rows, ys = [], []
    for s in shots:
        if s.get("shot_type") == "Penalty":
            continue
        f = shot_features(s)
        if f is None:
            continue
        rows.append(f)
        ys.append(1.0 if s["outcome"] == "Goal" else 0.0)
    if len(rows) < 500:
        return None
    # standardize dist & angle for stable GD
    def col(i):
        return [r[i] for r in rows]
    stats = {}
    for i in (1, 2):
        c = col(i)
        mean = sum(c) / len(c)
        var = sum((v - mean) ** 2 for v in c) / len(c)
        stats[i] = (mean, math.sqrt(var) or 1.0)
        for r in rows:
            r[i] = (r[i] - stats[i][0]) / stats[i][1]
    w = [0.0, 0.0, 0.0, 0.0]
    lr, n = 0.5, len(rows)
    for _ in range(400):
        grad = [0.0] * 4
        for r, y in zip(rows, ys):
            z = sum(wi * xi for wi, xi in zip(w, r))
            p = 1.0 / (1.0 + math.exp(-max(min(z, 30), -30)))
            e = p - y
            for i in range(4):
                grad[i] += e * r[i]
        for i in range(4):
            w[i] -= lr * grad[i] / n
    return {"w": w, "std": {str(k): v for k, v in stats.items()}}


def xg_of(model, s):
    if s.get("shot_type") == "Penalty":
        return PENALTY_XG
    f = shot_features(s)
    if f is None:
        return None
    for i in (1, 2):
        mean, sd = model["std"][str(i)]
        f[i] = (f[i] - mean) / sd
    z = sum(wi * xi for wi, xi in zip(model["w"], f))
    return 1.0 / (1.0 + math.exp(-max(min(z, 30), -30)))


def compute_shot_quality():
    """Return ({team: (sq_att, sq_def, n)}, model, validation) or ({}, None, None)."""
    shots_path = os.path.join(CACHE, "sb_shots.jsonl")
    matches_path = os.path.join(CACHE, "sb_matches.json")
    if not (os.path.exists(shots_path) and os.path.exists(matches_path)):
        print("  no StatsBomb cache - skipping shot quality")
        return {}, None, None
    with open(matches_path) as f:
        sb_matches = {m["match_id"]: m for m in json.load(f)}
    shots = []
    with open(shots_path) as f:
        for line in f:
            shots.append(json.loads(line))

    model = train_xg_model(shots)
    if model is None:
        return {}, None, None

    # validation: correlation of our xG with StatsBomb's own xG
    pairs = [(xg_of(model, s), s["sb_xg"]) for s in shots
             if s.get("sb_xg") is not None and shot_features(s)]
    n = len(pairs)
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    cov = sum((a - mx) * (b - my) for a, b in pairs)
    vx = sum((a - mx) ** 2 for a, _ in pairs)
    vy = sum((b - my) ** 2 for _, b in pairs)
    corr = cov / math.sqrt(vx * vy) if vx * vy > 0 else 0.0
    validation = {"n_shots": n, "corr_vs_statsbomb_xg": round(corr, 3)}

    # per-team xG for/against per match, recency weighted
    per_match = defaultdict(lambda: defaultdict(float))
    for s in shots:
        v = xg_of(model, s)
        if v is None or s["match_id"] not in sb_matches:
            continue
        per_match[s["match_id"]][s["team"]] += v

    acc = defaultdict(lambda: [0.0, 0.0, 0.0, 0])  # w, xg_for, xg_against, n
    tot_w = tot_xg = 0.0
    for mid, m in sb_matches.items():
        if mid not in per_match:
            continue
        d = parse_date(m["date"])
        w = recency_weight(d) if d else 1.0
        for team, opp in ((m["home"], m["away"]), (m["away"], m["home"])):
            xf = per_match[mid].get(team, 0.0)
            xa = per_match[mid].get(opp, 0.0)
            a = acc[team]
            a[0] += w
            a[1] += w * xf
            a[2] += w * xa
            a[3] += 1
            tot_w += w
            tot_xg += w * xf
    mu = tot_xg / tot_w
    sq = {}
    for team, (w, xf, xa, n_m) in acc.items():
        name = TEAM_ALIASES.get(team, team)
        sq[name] = (
            shrink(xf / w / mu, n_m, K_SB),
            shrink(xa / w / mu, n_m, K_SB),
            n_m,
        )
    return sq, model, validation


# --------------------------------------------------------------------------
# 4. Main
# --------------------------------------------------------------------------

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    index = {"updated": now, "league_strength": LEAGUE_STRENGTH, "leagues": []}

    for code, name, country in LEAGUES:
        print(f"League {code} ({name})...", flush=True)
        matches = load_club_league(code)
        if not matches:
            print("  no matches, skipped")
            continue
        out = compute_club_ratings(matches)
        out.update({"id": code, "name": name, "country": country, "updated": now,
                    "n_matches": len(matches)})
        with open(os.path.join(DATA_DIR, f"{code}.json"), "w") as f:
            json.dump(out, f)
        index["leagues"].append({"id": code, "name": name, "country": country,
                                 "teams": len(out["teams"])})
        print(f"  {len(matches)} matches, {len(out['teams'])} teams")

    print("Internationals...", flush=True)
    sq, model, validation = compute_shot_quality()
    if validation:
        print(f"  xG model: corr vs StatsBomb xG = {validation['corr_vs_statsbomb_xg']} "
              f"on {validation['n_shots']} shots")
    intl = load_internationals()
    out = compute_intl_ratings(intl, sq)
    out.update({"id": "INT", "name": "Internationals (World Cup / Euro / Copa)",
                "country": "International", "updated": now, "n_matches": len(intl),
                "xg_validation": validation})
    with open(os.path.join(DATA_DIR, "INT.json"), "w") as f:
        json.dump(out, f)
    index["leagues"].append({"id": "INT", "name": out["name"],
                             "country": "International", "teams": len(out["teams"])})
    print(f"  {len(intl)} matches, {len(out['teams'])} teams, "
          f"{sum(1 for t in out['teams'].values() if t['n_sb'])} with shot data")

    if model:
        with open(os.path.join(DATA_DIR, "xg_model.json"), "w") as f:
            json.dump({"model": model, "validation": validation}, f)

    with open(os.path.join(DATA_DIR, "index.json"), "w") as f:
        json.dump(index, f)
    print("Wrote", DATA_DIR)


if __name__ == "__main__":
    main()
