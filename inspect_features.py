import pandas as pd

df = pd.read_csv('data/processed/features.csv.gz')
print(df.columns.tolist()[:20])
print(df[['season','games_prior','goals','g_r5','g_career']].head(10).to_string(index=False))
print('nrows', len(df))
print('unique seasons', sorted(df['season'].dropna().unique())[:10], sorted(df['season'].dropna().unique())[-10:])
print('null season', int(df['season'].isna().sum()))
print('games_prior min/max', df['games_prior'].min(), df['games_prior'].max())
print('goals null', int(df['goals'].isna().sum()))
print('filtered subset', df[(df['games_prior'] >= 10) & df['goals'].notna()].shape)
print('by season', df[(df['games_prior'] >= 10) & df['goals'].notna()].groupby('season').size().to_dict())
