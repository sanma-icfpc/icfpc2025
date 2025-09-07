import os
import csv
import json
import threading
import time
import requests
from datetime import datetime, timezone, timedelta
from flask import Flask, render_template, send_from_directory, jsonify

app = Flask(__name__)

API_URL = "https://31pwr5t6ij.execute-api.eu-west-2.amazonaws.com/leaderboard/global"
# Use absolute path for the scores directory (relative to this file)
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scores")
PORT = 19386

# JST timezone
JST = timezone(timedelta(hours=9))


def fetch_and_save():
    """Fetch leaderboard and save as timestamped CSV."""
    try:
        r = requests.get(API_URL, timeout=10)
        r.raise_for_status()
        data = r.json()

        ts = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
        os.makedirs(SAVE_DIR, exist_ok=True)
        filename = os.path.join(SAVE_DIR, f"{ts}.csv")

        # sort by score desc
        sorted_data = sorted(data, key=lambda x: x["score"], reverse=True)

        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "rank", "teamName", "teamPl", "score"])
            for i, entry in enumerate(sorted_data, start=1):
                writer.writerow([ts, i, entry["teamName"], entry["teamPl"], entry["score"]])

        print(f"[INFO] Saved leaderboard: {filename}")
    except Exception as e:
        print(f"[ERROR] fetch_and_save failed: {e}")


def periodic_fetch(interval=600):
    """Background thread: fetch leaderboard every `interval` seconds."""
    while True:
        fetch_and_save()
        time.sleep(interval)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    """Aggregate all CSVs into JSON for charting."""
    result = {}
    for fname in sorted(os.listdir(SAVE_DIR)):
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(SAVE_DIR, fname)
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = row["timestamp"]
                team = row["teamName"]
                score = int(row["score"])
                if team not in result:
                    result[team] = []
                result[team].append({"timestamp": ts, "score": score})
    return jsonify(result)


@app.route("/scores/<path:filename>")
def download_file(filename):
    return send_from_directory(SAVE_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    # Start background fetch thread
    if True:
        t = threading.Thread(target=periodic_fetch, args=(600,), daemon=True)
        t.start()

    # Run Flask
    app.run(host="0.0.0.0", port=PORT, debug=True)
