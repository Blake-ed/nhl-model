"""Generate fake data in the same schema as collect_data.py, for smoke-testing the pipeline offline."""
import numpy as np, pandas as pd
rng = np.random.default_rng(0)
teams = [f"T{i:02d}" for i in range(16)]
rows_p, rows_t = [], []
players = [(i, f"Player {i}", rng.choice(list("CLR")), teams[i % 16], rng.gamma(4, 0.05)) for i in range(160)]
gid = 0
for season in range(2018, 2026):
    start = pd.Timestamp(f"{season}-10-05")
    for day in range(0, 170, 2):
        order = rng.permutation(teams)
        for h, aw in zip(order[::2], order[1::2]):
            gid += 1
            date = (start + pd.Timedelta(days=day)).strftime("%Y-%m-%d")
            gh, ga = rng.poisson(3.0), rng.poisson(3.0)
            rows_t += [dict(game_id=gid, game_date=date, season=season, team=h, opponent=aw, is_home=1, gf=gh, ga=ga),
                       dict(game_id=gid, game_date=date, season=season, team=aw, opponent=h, is_home=0, gf=ga, ga=gh)]
            for pid, nm, pos, tm, skill in players:
                if tm in (h, aw) and rng.random() < 0.95:
                    toi = float(np.clip(rng.normal(16, 2.5), 8, 24)); shots = rng.poisson(skill * 14 * toi / 16)
                    goals = rng.binomial(shots, 0.105 * (0.8 + skill))
                    rows_p.append(dict(player_id=pid, name=nm, position=pos, season=season, game_id=gid, game_date=date, team=tm,
                                       opponent=aw if tm == h else h, is_home=int(tm == h), goals=goals, assists=0, shots=shots,
                                       toi_min=toi, pp_goals=0, pp_points=0, shifts=20, pim=0, plus_minus=0))
pd.DataFrame(rows_p).to_csv("data/processed/player_games.csv.gz", index=False)
pd.DataFrame(rows_t).to_csv("data/processed/team_games.csv.gz", index=False)
print(len(rows_p), "player-games (synthetic)")
