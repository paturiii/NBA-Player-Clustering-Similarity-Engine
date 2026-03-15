"""
app.py
======
Flask application — routes only.
All ML logic lives in pipeline/similarity.py and pipeline/query.py.
"""

from flask import Flask, jsonify, render_template, request
from pipeline.query import NBAQueryEngine

app = Flask(__name__)
engine = NBAQueryEngine()          # loaded once at startup


# ══════════════════════════════════════════════════════════════════
#  API
# ══════════════════════════════════════════════════════════════════

@app.route('/api/similar')
def api_similar():
    """
    Mode 1 — All-Time:
        GET /api/similar?player=Stephen Curry&k=10

    Mode 2 — Single Season:
        GET /api/similar?player=Stephen Curry&season=2017&k=10
    """
    player = request.args.get('player', '').strip()
    season = request.args.get('season', '').strip()
    k = min(int(request.args.get('k', 10)), 50)   # hard cap at 50

    if not player:
        return jsonify({'error': 'player param required'}), 400

    try:
        if season:
            results = engine.query_season(player, season, k=k)
            mode = 'season'
        else:
            results = engine.query_alltime(player, k=k)
            mode = 'alltime'

        return jsonify({
            'query':   {'player': player, 'season': season or None, 'mode': mode},
            'results': results,
        })

    except ValueError as e:
        return jsonify({'error': str(e)}), 404


@app.route('/api/player/<path:player_name>')
def api_player(player_name):
    """
    GET /api/player/Stephen Curry
    Returns all seasons + career averages for a player.
    """
    try:
        return jsonify(engine.player_page(player_name))
    except ValueError as e:
        return jsonify({'error': str(e)}), 404


@app.route('/api/seasons/<path:player_name>')
def api_seasons(player_name):
    """
    GET /api/seasons/Stephen Curry
    Returns a sorted list of seasons — used to populate the season dropdown.
    """
    try:
        return jsonify({
            'player': player_name,
            'seasons': engine.get_player_seasons(player_name),
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 404


@app.route('/api/search')
def api_search():
    """
    GET /api/search?q=curry&limit=10
    Autocomplete — returns matching player names.
    """
    q = request.args.get('q', '').strip()
    limit = int(request.args.get('limit', 10))
    return jsonify(engine.search_players(q, limit=limit))


# ══════════════════════════════════════════════════════════════════
#  PAGES
# ══════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html', seasons=engine.all_seasons())


@app.route('/player/<path:player_name>')
def player_page(player_name):
    season = request.args.get('season', '').strip() or None
    try:
        data = engine.player_page(player_name)
        player_seasons = engine.get_player_seasons(player_name)
    except ValueError:
        return render_template('404.html'), 404

    return render_template(
        'player.html',
        data=data,
        player_seasons=player_seasons,
        selected_season=season,
    )


# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    app.run(debug=True, port=5000)