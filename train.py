"""
train.py
========
Run once to build all model artifacts before starting the Flask app.

Usage:
    python train.py --csv data/Comparison_Stats.csv
    python train.py --csv data/Comparison_Stats.csv --clusters 12
    python train.py --csv data/Comparison_Stats.csv --elbow      # plot elbow first
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from pipeline.similarity import (
    load_data,
    build_career_vectors,
    build_season_index,
    train_pipeline,
    STAT_FEATURES,
)


def elbow_plot(X_pca):
    inertias  = []
    K_range   = range(4, 20)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_pca)
        inertias.append(km.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(list(K_range), inertias, 'o-', color='#4f7cff')
    plt.xlabel('K (clusters)')
    plt.ylabel('Inertia')
    plt.title('Elbow — pick K where inertia stops dropping sharply')
    plt.tight_layout()
    plt.savefig('models/elbow.png', dpi=120)
    print("  Elbow plot saved -> models/elbow.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',      default='data/Comparison_Stats.csv')
    parser.add_argument('--clusters', type=int, default=10)
    parser.add_argument('--elbow',    action='store_true')
    args = parser.parse_args()

    print("\n── Step 1: Load & clean ────────────────────────────────")
    df = load_data(args.csv)
    print(f"  {len(df)} player-seasons | {df['player'].nunique()} players "
          f"| seasons {df['season'].min()}–{df['season'].max()}")

    print("\n── Step 2: Career vectors ──────────────────────────────")
    career_df = build_career_vectors(df)

    print("\n── Step 3: Train scaler → PCA → KMeans ────────────────")
    scaler, pca, kmeans, career_df = train_pipeline(career_df, n_clusters=args.clusters)

    if args.elbow:
        print("\n── Elbow plot ──────────────────────────────────────────")
        X_scaled = scaler.transform(career_df[STAT_FEATURES].values)
        X_pca    = pca.transform(X_scaled)
        elbow_plot(X_pca)

    print("\n── Step 4: Season index ────────────────────────────────")
    build_season_index(df, scaler)

    print("\n✅  Done. Run:  python app.py\n")


if __name__ == '__main__':
    main()