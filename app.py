from flask import Flask, render_template, request, redirect, url_for, jsonify
from pandas_json import grab_data, search_players

app = Flask(__name__)
app.secret_key = "dev-secret-key"
app.config['WTF_CSRF_ENABLED'] = False


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("query")
    if query:
        return redirect(url_for("player", player_name=query))
    return redirect(url_for("index"))


@app.route("/autocomplete")
def autocomplete():
    q = request.args.get("q", "")
    results = search_players(q)
    return jsonify(results)


@app.route("/player/<player_name>")
def player(player_name):
    result = grab_data(player_name)

    if not result:
        return f"{player_name} does not exist"

    real_name = list(result.keys())[0]
    player_data = result[real_name]

    return render_template("player.html", player_name=real_name, data=player_data)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)
