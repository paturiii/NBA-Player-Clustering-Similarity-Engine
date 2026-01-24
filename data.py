import pandas as pd

"""
Not used
"""
KEYS = ['season', 'player', 'player_id', 'team', 'pos']
def general_information():

    ADV_COLS = [ # Columns from Advanced.csv (Advanced Stats)
        "ts_percent", "x3p_ar", "f_tr", "orb_percent", "drb_percent", "trb_percent", "ast_percent", 
        "stl_percent", "blk_percent", "tov_percent", "usg_percent", "obpm"
    ]

    PBP_COLS = [    # Columns from Player Play By Play.csv (Hidden Stats)
        'bad_pass_turnover','lost_ball_turnover', 'shooting_foul_committed', 
        'offensive_foul_committed', 'shooting_foul_drawn', 'offensive_foul_drawn', 
        'points_generated_by_assists', 'and1','fga_blocked'
    ]

    PG_COLS = [ # Columns from Player Per Game.csv 
        'season','player','age','team','pos','g','gs','mp_per_game','fg_per_game','fga_per_game',
        'fg_percent','x3p_per_game','x3pa_per_game','x3p_percent','x2p_per_game','x2pa_per_game','x2p_percent',
        'e_fg_percent','ft_per_game','fta_per_game','ft_percent','orb_per_game','drb_per_game','trb_per_game',
        'ast_per_game','stl_per_game','blk_per_game','tov_per_game','pf_per_game','pts_per_game',
    ]

    per_game = filter(pd.read_csv('data/Player Per Game.csv'), PG_COLS)
    advanced = filter(pd.read_csv('data/Advanced.csv'), ADV_COLS)
    play_by_play = filter(pd.read_csv('data/Player Play By Play.csv'), PBP_COLS)

    return per_game, advanced, play_by_play


def data_merge():

    per_game, advanced, play_by_play = general_information()

    data_combine = pd.merge(per_game, advanced, on=KEYS, how='outer')
    data = pd.merge(data_combine, play_by_play, on=KEYS, how='outer')

    data.to_csv()
