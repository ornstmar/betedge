#!/usr/bin/env python3
"""Fetch shot events (with x/y pitch coordinates) from StatsBomb open-data.

Downloads events for recent international tournaments and extracts only shots,
writing a compact shots.jsonl used both to train the xG model and to compute
team shot-quality ratings. Free data, no API key.
"""
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
OUT = os.path.join(os.path.dirname(__file__), "cache")

# (competition_id, competition_name) -> only seasons from 2018 onward
TARGET_COMPS = {43: "World Cup", 55: "UEFA Euro", 223: "Copa America"}
MIN_YEAR = 2018


def get_json(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "betedge"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def season_year(name):
    try:
        return int(str(name).split("/")[0])
    except ValueError:
        return 0


def list_matches():
    comps = get_json(f"{BASE}/competitions.json")
    targets = [
        (c["competition_id"], c["season_id"], c["season_name"])
        for c in comps
        if c["competition_id"] in TARGET_COMPS
        and season_year(c["season_name"]) >= MIN_YEAR
    ]
    matches = []
    for cid, sid, sname in targets:
        for m in get_json(f"{BASE}/matches/{cid}/{sid}.json"):
            matches.append(
                {
                    "match_id": m["match_id"],
                    "comp": TARGET_COMPS[cid],
                    "season": sname,
                    "date": m["match_date"],
                    "home": m["home_team"]["home_team_name"],
                    "away": m["away_team"]["away_team_name"],
                }
            )
    return matches


def fetch_shots(match):
    events = get_json(f"{BASE}/events/{match['match_id']}.json")
    shots = []
    for e in events:
        if e.get("type", {}).get("name") != "Shot":
            continue
        s = e.get("shot", {})
        loc = e.get("location") or [None, None]
        shots.append(
            {
                "match_id": match["match_id"],
                "team": e.get("team", {}).get("name"),
                "x": loc[0],
                "y": loc[1],
                "outcome": s.get("outcome", {}).get("name"),
                "body_part": s.get("body_part", {}).get("name"),
                "shot_type": s.get("type", {}).get("name"),
                "sb_xg": s.get("statsbomb_xg"),
            }
        )
    return shots


def main():
    os.makedirs(OUT, exist_ok=True)
    matches = list_matches()
    print(f"{len(matches)} matches to fetch", flush=True)
    with open(os.path.join(OUT, "sb_matches.json"), "w") as f:
        json.dump(matches, f)

    done_path = os.path.join(OUT, "sb_shots.jsonl")
    have = set()
    if os.path.exists(done_path):
        with open(done_path) as f:
            for line in f:
                have.add(json.loads(line)["match_id"])
    todo = [m for m in matches if m["match_id"] not in have]
    print(f"{len(todo)} remaining", flush=True)

    with open(done_path, "a") as out, ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(fetch_shots, m): m for m in todo}
        n = 0
        for fut in as_completed(futs):
            m = futs[fut]
            try:
                for s in fut.result():
                    out.write(json.dumps(s) + "\n")
                out.flush()
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL {m['match_id']}: {exc}", flush=True)
            n += 1
            if n % 20 == 0:
                print(f"{n}/{len(todo)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
