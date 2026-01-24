import pandas as pd
import unicodedata

def normalize(text):
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower().strip()


data = pd.read_csv("data/Player Per Game.csv")

data = data[['player', 'season', 'team', 'g', 'stl_per_game',
             'blk_per_game', 'ast_per_game', 'trb_per_game', 'x3p_percent',
             'fg_percent', 'pts_per_game', 'mp_per_game', 'ft_per_game', 'fta_per_game']]

data = data[data['season'] > 1997].fillna(0)

def team_order(team):
    return (team=='@2TM', team)

data["player_search"] = data["player"].apply(normalize)
data= data.sort_values(by=['season', 'team'], key=lambda col: col.map(lambda t: 1 if t=='2TM' else 0))


def grab_data(player_name): 
    hashed_data = {}

    search_name = normalize(player_name)
    filtered = data[data['player_search'] == search_name]

    if filtered.empty:
        return None

    for _, row in filtered.iterrows():
        player = row['player']
        season = str(row['season'])

        if player not in hashed_data:
            hashed_data[player] = {}

        if season not in hashed_data[player]:
            hashed_data[player][season] = []

        hashed_data[player][season].append({
            col: row[col] for col in filtered.columns
            if col not in ['player_search', 'player', 'season']
        })

    return hashed_data


def search_players(query, limit=10):
    q = normalize(query)
    matches = data[data["player_search"].str.contains(q, regex=False)]
    return matches["player"].drop_duplicates().head(limit).tolist()
