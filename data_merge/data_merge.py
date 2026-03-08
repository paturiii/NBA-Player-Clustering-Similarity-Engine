import pandas as pd

KEYS = ['season', 'player', 'player_id', 'team', 'pos']

ADV_COLS = [ # Columns from Advanced.csv (Advanced Stats)
    "ts_percent", "x3p_ar", "f_tr", "orb_percent", "drb_percent", "trb_percent", 
    "ast_percent", "stl_percent", "blk_percent", "tov_percent", "usg_percent", 
    "obpm" ]
"""
Play by Play
    bad_pass_turnover_rate,              -> bad_pass_turnover / ast -> rate rather than count
    shooting_foul_committed_perent,      -> shooting_foul_committed / fouls
    offensive_foul_committed_percent,    -> offensive_foul_committed / fouls
    shooting_foul_drawn_percentage,      -> shooting_foul_drawn / fouls_d
    offensive_foul_drawn_percentage,     -> offensive_foul_drawn / foulds_drawn
fouls_drawn                              -> offensive_foul_drawn + shooting_foul_drawn
"""

PBP_COLS = [    # Columns from Player Play By Play.csv (Hidden Stats)
    'bad_pass_turnover','lost_ball_turnover', 'shooting_foul_committed', 
    'offensive_foul_committed', 'shooting_foul_drawn', 'offensive_foul_drawn', 
    'points_generated_by_assists', 'and1','fga_blocked'
]

"""
x3p_percent
3pct_useage  -> x3p_attepmts / fga +
x2p_percent
2pct_useage  -> x2p_attempts / fga +
ft_percent
e_fg_percent
"""

PG_COLS = [ # Columns from Player Per Game.csv  (Per Game stats mostly for creating new features)
    'g', 'fga_per_game', 'fg_percent','x3p_percent', 'fga_per_game', 'x3pa_per_game', 
    'x2pa_per_game', 'x2p_percent', 'e_fg_percent', 'ft_percent', 'orb_per_game',
]


"""
- Selects only requested columns/keys from the dataset
- removes players prior to 1997 (NBA stat counting reason)

"""
def filter(df, cols):
    
    final_cols = list(set(KEYS + cols)) # Select only the explicitly requested columns PLUS the core merge keys

    df_filtered = df[final_cols].copy()

    try:
        # Seasons After 1980 (For leauge purposes) will fix later
        df_filtered['year'] = df_filtered['season'].astype(str).str[:4].astype(int) 
        df_filtered = df_filtered[df_filtered['year'] >= 1997]
        df_filtered = df_filtered.drop(columns=['year'])

    except:
        pass

    df['3_pt_usage'] = df['x3pa_per_game'] / df['fga_per_game']
    df['2_pt_usage'] = df['x2pa_per_game'] / df['fga_per_game']


    return df_filtered

df_pg_filtered = filter(pd.read_csv('data/Player Per Game.csv'), PG_COLS)
df_adv_filtered = filter(pd.read_csv('data/Advanced.csv'), ADV_COLS)
df_pbp_filtered = filter(pd.read_csv('data/Player Play By Play.csv'), PBP_COLS)

df_combine = pd.merge(df_pg_filtered, df_adv_filtered, on=KEYS, how='outer')
df_final = pd.merge(df_combine, df_pbp_filtered, on=KEYS, how='outer')

starting_columns_existing = ['player', 'season', 'pos', 'player_id', 'g' ]

# Get the list of all remaining columns
all_columns = df_final.columns.tolist()
remaining_columns = [col for col in all_columns if col not in starting_columns_existing]

# Apply the new column order
new_column_order = starting_columns_existing + remaining_columns
df_final_reordered = df_final[new_column_order]

df_final_reordered['3pt_useage'] = df_final_reordered['x3pa_per_game'] / df_final_reordered['fga_per_game']
df_final_reordered['3pt_useage'] = df_final_reordered['x2pa_per_game'] / df_final_reordered['fga_per_game']

df_final_reordered.to_csv('Comparison Stats.csv', index=False)