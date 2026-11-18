"""Download NHL play-by-play and extract every unblocked shot attempt (shots on goal, misses, goals).

Needs data/processed/team_games.csv.gz from collect_data.py (for the list of game ids).
Each game is parsed into a compact JSON in data/raw/pbp_parsed/, so the run is resumable.
Output: data/processed/shots.csv.gz   (one row per unblocked shot attempt)

Run:  python collect_pbp.py            (~10,000 games; expect roughly an hour)
"""
import json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
from collect_data import BASE, SESSION

CACHE = Path("data/raw/pbp_parsed")
SHOT_EVENTS = {"shot-on-goal", "missed-shot", "goal"}   # blocked shots excluded (unblocked / "Fenwick" attempts)


def _sec(p):
    mm, ss = p["timeInPeriod"].split(":")
    return (p["periodDescriptor"]["number"] - 1) * 1200 + int(mm) * 60 + int(ss)


def _attack_sign(p, x, zone, owner_is_home):
    """+1 if the shooting team attacks the goal at x=+89, else -1."""
    side = p.get("homeTeamDefendingSide")
    if side in ("left", "right"):
        return 1 if (side == "left") == owner_is_home else -1
    if zone == "D":                     # older seasons: infer direction from zone
        return -1 if x > 0 else 1
    return 1 if x > 0 else -1


def _rel_zone(prev, owner_id):
    d = prev.get("details", {})
    z = d.get("zoneCode")
    if z is None:
        return None
    return z if d.get("eventOwnerTeamId") == owner_id else {"O": "D", "D": "O", "N": "N"}.get(z)


def parse_game(pbp):
    home, away = pbp["homeTeam"], pbp["awayTeam"]
    abbr = {home["id"]: home["abbrev"], away["id"]: away["abbrev"]}
    season = int(pbp["season"]) // 10000
    shots, prev = [], None
    for p in pbp.get("plays", []):
        t, d = p.get("typeDescKey"), p.get("details", {})
        if p["periodDescriptor"].get("periodType") == "SO":
            prev = p
            continue
        if t in SHOT_EVENTS and "xCoord" in d and "yCoord" in d:
            shooter = d.get("scoringPlayerId") if t == "goal" else d.get("shootingPlayerId")
            owner = d.get("eventOwnerTeamId")
            if shooter is not None and owner in abbr:
                is_home = owner == home["id"]
                x, y, zone = d["xCoord"], d["yCoord"], d.get("zoneCode")
                x_att = x * _attack_sign(p, x, zone, is_home)
                sc, diff, empty = p.get("situationCode"), 0, 0
                if sc and len(sc) == 4:
                    f, a = (int(sc[2]), int(sc[1])) if is_home else (int(sc[1]), int(sc[2]))
                    goalie_against = int(sc[0]) if is_home else int(sc[3])
                    diff, empty = max(-2, min(2, f - a)), int(goalie_against == 0)
                dt, rebound, rush = 30.0, 0, 0
                if prev is not None and prev["periodDescriptor"]["number"] == p["periodDescriptor"]["number"]:
                    dt = float(min(30, _sec(p) - _sec(prev)))
                    pd_ = prev.get("details", {})
                    rebound = int(prev.get("typeDescKey") == "shot-on-goal" and pd_.get("eventOwnerTeamId") == owner and dt <= 3)
                    rush = int(dt <= 4 and _rel_zone(prev, owner) in ("N", "D"))
                shots.append(dict(game_id=pbp["id"], season=season, period=p["periodDescriptor"]["number"],
                                  game_sec=_sec(p), team=abbr[owner], shooter_id=shooter, event=t, goal=int(t == "goal"),
                                  x_att=x_att, y=y, shot_type=d.get("shotType", "unknown"), skater_diff=diff,
                                  empty_net=empty, is_rebound=rebound, is_rush=rush, dt_prev=dt))
        prev = p
    return shots


def fetch(gid):
    path = CACHE / f"{gid}.json"
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(4):
        try:
            r = SESSION.get(f"{BASE}/gamecenter/{gid}/play-by-play", timeout=30)
            if r.status_code == 404:
                return []
            r.raise_for_status()
            shots = parse_game(r.json())
            path.write_text(json.dumps(shots))
            time.sleep(0.1)
            return shots
        except Exception as e:
            print(f"  retry {attempt + 1} for game {gid}: {e}")
            time.sleep(2 * (attempt + 1))
    return None


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    ids = sorted(pd.read_csv("data/processed/team_games.csv.gz")["game_id"].unique())
    print(f"{len(ids):,} games to fetch (cached ones are skipped)")
    rows, failed = [], []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, (gid, shots) in enumerate(zip(ids, ex.map(fetch, ids)), 1):
            if shots is None:
                failed.append(gid)
            else:
                rows += shots
            if i % 500 == 0:
                print(f"  {i:,}/{len(ids):,} games")
    df = pd.DataFrame(rows)
    df.to_csv("data/processed/shots.csv.gz", index=False)
    print(f"Saved {len(df):,} shot attempts from {df.game_id.nunique():,} games; {len(failed)} games failed (re-run to retry)")


if __name__ == "__main__":
    main()
