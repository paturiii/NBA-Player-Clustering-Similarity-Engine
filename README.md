# NBA Player Archetype

An Machine Learning-powered app that clusters NBA players by their playstyle, allowing people to explore stylistic similarities, discover player archetypes and interactively query player networks

---

## Features
- **Machine Learning Clustering**: Groups players using advanced efficiency and rate statistics to identify player archetypes
- **Playstyle Vectors**: Normalized vectors quantify player styles using **K-Means** and **cosine similarity** to measure similarity between players. 
- **Dimensionality Reduction**: **PCA** simplifies data visualization and highlights key stylistic trends. 
- **Interactive Flask Web App**: Explore player networks, search for specific players, and query stylistic similarities. 
- **Modular Pipeline**: Preprocessing, inference, and visualization components are modular and ready for **REST API integration** and **cloud deployment**. 

---

## Installation
1. Clone the REpo
``` bash
    git clone https://github.com/paturiii/nba-player-archetype.git
    cd nba-player-archetype
```

2. Install dependencies:
``` bash
    pip install -r requirements.txt
```

3. Run the app
``` bash
    python3 app.py
```
open your browser and navigate to 

``` bash
    http://127.0.0.1:5000
```