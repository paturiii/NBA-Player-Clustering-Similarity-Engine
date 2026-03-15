"""
pipeline/similarity.py
======================
Data loading, cleaning, and model training.
Run via train.py — not imported at Flask runtime.

The only things query.py imports from here are:
  STAT_FEATURES  — single source of truth for feature list
  _parse_seasons — small helper
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
COMBINED_TAGS = {'TOT', '2TM', '3TM', '4TM'}
MIN_GAMES = 20

# ── Rate stats: percentages / per-possession — use as-is ──────────
RATE_FEATURES = [
    'ts_percent', 'e_fg_percent', 'x3p_ar',    'f_tr',
    'x3p_percent', 'x2p_percent', 'ft_percent',
    'usg_percent', 'obpm',
    'ast_percent', 'tov_percent',
    'orb_percent', 'drb_percent', 'trb_percent',
    'stl_percent', 'blk_percent',
]

# ── PBP count stats: raw totals — normalized per game before use ───
# (raw correlations with games played range 0.59 – 0.83)
COUNT_FEATURES = [
    'bad_pass_turnover',
    'lost_ball_turnover',
    'shooting_foul_committed',
    'offensive_foul_committed',
    'shooting_foul_drawn',
    'and1',
    'fga_blocked',
    'points_generated_by_assists',
]
# offensive_foul_drawn excluded — 27.5 % nulls, too sparse

COUNT_FEATURES_PG = [f + '_pg' for f in COUNT_FEATURES]
STAT_FEATURES = RATE_FEATURES + COUNT_FEATURES_PG   # 24 features total


# ══════════════════════════════════════════════════════════════════
#  1. CLEANING
# ══════════════════════════════════════════════════════════════════

def deduplicate_traded_players(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basketball Reference writes one row per team plus one combined
    row (TOT / 2TM / 3TM) when a player is traded mid-season.
    Keep only the combined row for traded players; keep the single
    row as-is for players who stayed on one team all year.
    """
    counts = df.groupby(['player', 'season'])['team'].transform('count')
    mask = (counts == 1) | (df['team'].isin(COMBINED_TAGS))
    deduped = df[mask].copy()
    assert deduped.groupby(['player', 'season']).size().max() == 1, \
        "Duplicate player-seasons remain after dedup!"
    print(f"  Dedup: removed {len(df) - len(deduped)} team-split rows "
          f"-> {len(deduped)} unique player-seasons")
    return deduped.reset_index(drop=True)


def normalize_count_features(df: pd.DataFrame) -> pd.DataFrame:
    """Divide each PBP count stat by games played -> per-game rate."""
    df = df.copy()
    for col in COUNT_FEATURES:
        df[col + '_pg'] = df[col] / df['g'].replace(0, np.nan)
    return df


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load Comparison_Stats.csv, clean it, and return a ready-to-use
    DataFrame with STAT_FEATURES columns populated.

    Expected source columns:
        player, season, player_id, team, pos, g
        + RATE_FEATURES (already rate/percentage)
        + COUNT_FEATURES (raw PBP totals)
    """
    df = pd.read_csv(csv_path)

    df = deduplicate_traded_players(df)

    before = len(df)
    df = df[df['g'] >= MIN_GAMES].copy()
    print(f"  Games filter (>= {MIN_GAMES}): removed {before - len(df)} rows "
          f"-> {len(df)} remaining")

    df = normalize_count_features(df)

    # x3p_percent null = player never attempted a 3 -> genuine 0
    df[STAT_FEATURES] = df[STAT_FEATURES].fillna(0)

    df['player_season_key'] = df['player'] + ' | ' + df['season'].astype(str)

    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════
#  2. CAREER VECTORS  (Mode 1)
#     Games-weighted average across all seasons per player.
#     A 70-game season contributes more than a 25-game season.
# ══════════════════════════════════════════════════════════════════

def build_career_vectors(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for p, grp in df.groupby('player'):
        weights = grp['g'].values.astype(float)
        weights /= weights.sum()
        weighted_stats = (grp[STAT_FEATURES].values * weights[:, None]).sum(axis=0)
        records.append({
            'player': p,
            'pos': grp['pos'].mode()[0],
            'total_g': int(grp['g'].sum()),
            'seasons': sorted(grp['season'].astype(str).tolist()),
            'season_count': len(grp),
            **dict(zip(STAT_FEATURES, weighted_stats))
        })
    career_df = pd.DataFrame(records).reset_index(drop=True)
    print(f"  Built {len(career_df)} career vectors")
    return career_df


# ══════════════════════════════════════════════════════════════════
#  3. TRAINING
#     StandardScaler -> PCA -> KMeans (career vectors only)
#     The fitted scaler is then reused to transform season rows.
# ══════════════════════════════════════════════════════════════════

def train_pipeline(career_df: pd.DataFrame, n_clusters: int = 10):
    """
    Fit scaler, PCA, KMeans on career vectors.
    Precompute the career cosine-similarity matrix.
    Save all artifacts to models/.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    X = career_df[STAT_FEATURES].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=0.90, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"  PCA: {X.shape[1]} features -> {X_pca.shape[1]} components "
          f"({pca.explained_variance_ratio_.sum():.1%} variance)")

    # KMeans used internally only — cluster IDs never shown in the UI
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    career_df = career_df.copy()
    career_df['cluster'] = kmeans.fit_predict(X_pca)

    # Precompute career similarity on full-resolution scaled vectors (not PCA)
    # so cosine captures finer playstyle differences
    career_sim = cosine_similarity(X_scaled)

    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(pca, os.path.join(MODEL_DIR, 'pca.pkl'))
    joblib.dump(kmeans, os.path.join(MODEL_DIR, 'kmeans.pkl'))
    np.save(os.path.join(MODEL_DIR, 'career_vectors_scaled.npy'), X_scaled)
    np.save(os.path.join(MODEL_DIR, 'career_sim_matrix.npy'),     career_sim)
    career_df.to_csv(os.path.join(MODEL_DIR, 'career_df.csv'), index=False)

    print(f"  Saved: {len(career_df)} players | {n_clusters} clusters")
    return scaler, pca, kmeans, career_df


def build_season_index(df: pd.DataFrame, scaler: StandardScaler):
    """
    Transform every season row with the career-fitted scaler so season
    vectors live in the same feature space as career vectors.

    Season similarity is computed on-the-fly at query time.
    A precomputed 17k x 17k matrix would be ~2.3 GB — not worth it.
    """
    X_season = scaler.transform(df[STAT_FEATURES].values)
    np.save(os.path.join(MODEL_DIR, 'season_vectors_scaled.npy'), X_season)
    df.to_csv(os.path.join(MODEL_DIR, 'season_df.csv'), index=False)
    print(f"  Indexed {len(df)} season rows")
    return X_season, df


# ══════════════════════════════════════════════════════════════════
#  HELPERS  (also imported by query.py)
# ══════════════════════════════════════════════════════════════════

def _parse_seasons(val) -> list[str]:
    """Safely parse the seasons column (stored as string repr of list in CSV)."""
    if isinstance(val, list):
        return val
    try:
        return eval(val)
    except Exception:
        return [str(val)]