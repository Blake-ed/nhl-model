"""Download NHL forward game logs + team results from the public NHL web API.

Outputs (in data/processed/):
  player_games.csv.gz  one row per forward per regular-season game
  team_games.csv.gz    one row per team per regular-season game (goals for/against)

Raw API responses are cached in data/raw/, so re-running is fast and resumable.
Run:  python collect_data.py --start 2018 --end 2025
(season 2018 = 2018-19 ... season 2025 = 2025-26)
"""
import argparse, datetime, json, time
from pathlib import Path
import pandas as pd
import requests

BASE = "https://api-web.nhle.com/v1"
RAW = Path("data/raw")
OUT = Path("data/processed")
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "nhl-goal-model/1.0 (personal ML project)"


def get_json(url, cache_name, sleep=0.15):
    path = RAW / cache_name
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(4):
        try:
            r = SESSION.get(url, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            data = r.json()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
            time.sleep(sleep)
            return data
        except requests.RequestException as e:
            print(f"  retry {attempt + 1} for {url}: {e}")
            time.sleep(2 * (attempt + 1))
    return None


def toi_to_min(s):
    try:
        m, sec = str(s).split(":")
        return int(m) + int(sec) / 60
    except Exception:
        return float("nan")


def season_teams(season):
    today = datetime.date.today().isoformat()
    date = min(f"{season + 1}-04-01", today)   # in-progress season: use today
    data = get_json(f"{BASE}/standings/{date}", f"standings_{season}.json")
    teams = sorted({row["teamAbbrev"]["default"] for row in (data or {}).get("standings", [])})
    if not teams:
        (RAW / f"standings_{season}.json").unlink(missing_ok=True)   # never keep an empty answer in the cache
        if date == today:                                            # in-progress season: ask for the current table
            r = SESSION.get(f"{BASE}/standings/now", timeout=30)
            if r.ok:
                teams = sorted({row["teamAbbrev"]["default"] for row in r.json().get("standings", [])})
    return teams


def collect(start, end):
    player_rows, team_rows = [], []
    for season in range(start, end + 1):
        sid = f"{season}{season + 1}"
        teams = season_teams(season)
        print(f"Season {sid}: {len(teams)} teams")
        seen_players = {}
        for team in teams:
            # --- team results ---
            sched = get_json(f"{BASE}/club-schedule-season/{team}/{sid}", f"sched_{team}_{sid}.json")
            for g in (sched or {}).get("games", []):
                if g.get("gameType") != 2 or g.get("gameState") not in ("OFF", "FINAL"):
                    continue
                home, away = g["homeTeam"], g["awayTeam"]
                is_home = home["abbrev"] == team
                me, opp = (home, away) if is_home else (away, home)
                team_rows.append(dict(game_id=g["id"], game_date=g["gameDate"], season=season,
                                      team=team, opponent=opp["abbrev"], is_home=int(is_home),
                                      gf=me.get("score"), ga=opp.get("score")))
            # --- roster (forwards only) ---
            stats = get_json(f"{BASE}/club-stats/{team}/{sid}/2", f"clubstats_{team}_{sid}.json")
            for p in (stats or {}).get("skaters", []):
                if p.get("positionCode") in ("C", "L", "R"):
                    seen_players[p["playerId"]] = (
                        f'{p["firstName"]["default"]} {p["lastName"]["default"]}', p["positionCode"])
        print(f"  {len(seen_players)} forwards; fetching game logs...")
        for i, (pid, (name, pos)) in enumerate(seen_players.items(), 1):
            log = get_json(f"{BASE}/player/{pid}/game-log/{sid}/2", f"gamelog_{pid}_{sid}.json")
            for g in (log or {}).get("gameLog", []):
                player_rows.append(dict(
                    player_id=pid, name=name, position=pos, season=season, game_id=g["gameId"],
                    game_date=g["gameDate"], team=g["teamAbbrev"], opponent=g["opponentAbbrev"],
                    is_home=int(g["homeRoadFlag"] == "H"), goals=g["goals"], assists=g["assists"],
                    shots=g.get("shots", 0), toi_min=toi_to_min(g.get("toi")),
                    pp_goals=g.get("powerPlayGoals", 0), pp_points=g.get("powerPlayPoints", 0),
                    shifts=g.get("shifts"), pim=g.get("pim", 0), plus_minus=g.get("plusMinus", 0)))
            if i % 100 == 0:
                print(f"    {i}/{len(seen_players)}")
    OUT.mkdir(parents=True, exist_ok=True)
    pg = pd.DataFrame(player_rows).drop_duplicates(["player_id", "game_id"])
    tg = pd.DataFrame(team_rows).drop_duplicates(["team", "game_id"])
    pg.to_csv(OUT / "player_games.csv.gz", index=False)
    tg.to_csv(OUT / "team_games.csv.gz", index=False)
    print(f"Saved {len(pg):,} player-games and {len(tg):,} team-games to {OUT}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2018)
    ap.add_argument("--end", type=int, default=2025)
    a = ap.parse_args()
    collect(a.start, a.end)
