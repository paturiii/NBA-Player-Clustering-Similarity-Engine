"""
pipeline/query.py
=================
NBAQueryEngine — loads trained model artifacts and answers
similarity queries.  Instantiated once in app.py at startup.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

from pipeline.similarity import STAT_FEATURES, _parse_seasons

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


class NBAQueryEngine:
    """
    Loads:
        scaler.pkl                  StandardScaler fit on career vectors
        career_df.csv               one row per player (g-weighted career)
        season_df.csv               one row per player-season
        career_vectors_scaled.npy   scaled career feature matrix
        season_vectors_scaled.npy   scaled season feature matrix
        career_sim_matrix.npy       precomputed pairwise cosine (careers)
    """

    def __init__(self):
        self.scaler      = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        self.career_df   = pd.read_csv(os.path.join(MODEL_DIR, 'career_df.csv'))
        self.season_df   = pd.read_csv(os.path.join(MODEL_DIR, 'season_df.csv'))
        self.career_vecs = np.load(os.path.join(MODEL_DIR, 'career_vectors_scaled.npy'))
        self.season_vecs = np.load(os.path.join(MODEL_DIR, 'season_vectors_scaled.npy'))
        self.career_sim  = np.load(os.path.join(MODEL_DIR, 'career_sim_matrix.npy'))

        # O(1) name -> row-index maps
        self._career_idx = {p: i for i, p in enumerate(self.career_df['player'])}
        self._season_idx = {k: i for i, k in enumerate(self.season_df['player_season_key'])}

        print(f"[QueryEngine] {len(self._career_idx)} players | "
              f"{len(self._season_idx)} player-seasons ready")

    # ── Mode 1: All-Time ──────────────────────────────────────────
    def query_alltime(self, player: str, k: int = 10) -> list[dict]:
        """
        Compare a player's g-weighted career vector against all other
        career vectors using the precomputed similarity matrix.
        Returns the top-k most similar players.
        """
        if player not in self._career_idx:
            raise ValueError(f"Player '{player}' not found.")

        idx    = self._career_idx[player]
        scores = self.career_sim[idx]

        results = []
        for i in np.argsort(scores)[::-1]:
            if i == idx:
                continue
            row = self.career_df.iloc[i]
            if row['player'] == player:    # skip any entry for the same player
                continue
            results.append({
                'player':       row['player'],
                'pos':          row.get('pos', ''),
                'similarity':   round(float(scores[i]), 4),
                'seasons':      _parse_seasons(row['seasons']),
                'season_count': int(row['season_count']),
                'total_g':      int(row['total_g']),
            })
            if len(results) >= k:
                break
        return results

    # ── Mode 2: Single Season ─────────────────────────────────────
    def query_season(self, player: str, season: str | int, k: int = 10) -> list[dict]:
        """
        Compare one player-season vector against every season row in the
        dataset using on-the-fly cosine similarity (~17k comparisons, <100ms).
        Returns the top-k most similar player-seasons from any year.

        season: accepts int (2017) or string ('2017').
        """
        key = f"{player} | {season}"
        if key not in self._season_idx:
            raise ValueError(f"'{key}' not found. Check player name and season.")

        idx       = self._season_idx[key]
        query_vec = self.season_vecs[idx].reshape(1, -1)
        scores    = cosine_similarity(query_vec, self.season_vecs)[0]

        results = []
        for i in np.argsort(scores)[::-1]:
            if i == idx:
                continue
            row = self.season_df.iloc[i]
            if row['player'] == player:    # skip all seasons of the same player
                continue
            results.append({
                'player':     row['player'],
                'season':     int(row['season']),
                'key':        row['player_season_key'],
                'pos':        row['pos'],
                'similarity': round(float(scores[i]), 4),
                'g':          int(row['g']),
            })
            if len(results) >= k:
                break
        return results

    # ── Player page ───────────────────────────────────────────────
    def player_page(self, player: str) -> dict:
        """
        Returns all season rows + career weighted averages for a player.
        Powers the /player/<name> page.
        """
        seasons = self.season_df[self.season_df['player'] == player].copy()
        if seasons.empty:
            raise ValueError(f"Player '{player}' not found.")

        career = self.career_df[self.career_df['player'] == player].iloc[0]

        return {
            'player':       player,
            'pos':          seasons['pos'].mode()[0],
            'total_g':      int(career['total_g']),
            'season_count': int(career['season_count']),
            'seasons':      (
                seasons[['season', 'pos', 'g'] + STAT_FEATURES]
                .sort_values('season')
                .to_dict('records')
            ),
            'career_avg': {f: round(float(career[f]), 3) for f in STAT_FEATURES},
        }

    # ── Helpers ───────────────────────────────────────────────────
    def get_player_seasons(self, player: str) -> list[int]:
        """Sorted list of seasons a player appears in."""
        rows = self.season_df[self.season_df['player'] == player]['season']
        if rows.empty:
            raise ValueError(f"Player '{player}' not found.")
        return sorted(rows.astype(int).tolist())

    def search_players(self, query: str, limit: int = 15) -> list[str]:
        """Case-insensitive substring search — used for autocomplete."""
        q = query.lower().strip()
        if not q:
            return []
        return sorted(p for p in self._career_idx if q in p.lower())[:limit]

    def all_seasons(self) -> list[int]:
        """Sorted list of every season in the dataset."""
        return sorted(self.season_df['season'].astype(int).unique().tolist())