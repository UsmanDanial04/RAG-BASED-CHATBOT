"""
app.py - Auth backend for the face-recognition login branch.

Endpoints:
  POST /api/register/start-check  -> validate a live webcam frame during signup (real-time feedback)
  POST /api/register/complete     -> store username + averaged face encoding (send 3-5 captured frames)
  POST /api/login                 -> validate frame + match against a stored user's encoding(s)
  POST /api/logout
  GET  /api/session               -> check if logged in (used by the chatbot branch to gate access)

Storage: SQLite, encodings stored as serialized numpy arrays (BLOB).
Swap this for your real user DB later; the face_utils functions are storage-agnostic.
"""

import io
import pickle
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, session
from flask_cors import CORS

from face_utils import (
    decode_base64_image,
    find_and_validate_face,
    get_face_encoding,
    match_encoding,
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "users.db"
DB_PATH.parent.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = "change-this-to-a-real-secret-in-.env"  # load from env var in production
CORS(app, supports_credentials=True)  # supports_credentials so the session cookie works cross-origin in dev


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with closing(get_db()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                encoding BLOB NOT NULL
            )
            """
        )
        conn.commit()


init_db()


@app.post("/api/register/start-check")
def register_check_frame():
    """
    Called repeatedly (e.g. every 500ms) while the user positions their face
    during signup, purely for live UI feedback -- doesn't save anything.
    """
    data = request.get_json(force=True)
    frame = decode_base64_image(data["image"])
    result = find_and_validate_face(frame)
    return jsonify({"ok": result.ok, "message": result.message})


@app.post("/api/register/complete")
def register_complete():
    """
    Body: { "username": "...", "images": ["data:image/jpeg;base64,...", ...] }
    Expects 3-5 good frames captured client-side (different angles/expressions)
    for a more robust average encoding.
    """
    data = request.get_json(force=True)
    username = data.get("username", "").strip()
    images = data.get("images", [])

    if not username:
        return jsonify({"error": "Username is required."}), 400
    if len(images) < 3:
        return jsonify({"error": "Please capture at least 3 face samples."}), 400

    with closing(get_db()) as conn:
        existing = conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
        if existing:
            return jsonify({"error": "Username already taken."}), 409

    encodings = []
    for img_b64 in images:
        frame = decode_base64_image(img_b64)
        check = find_and_validate_face(frame)
        if not check.ok:
            return jsonify({"error": f"One of the captures failed: {check.message}"}), 400
        encoding = get_face_encoding(check.face_image)
        if encoding is None:
            return jsonify({"error": "Could not extract a face encoding from one of the captures."}), 400
        encodings.append(encoding)

    avg_encoding = np.mean(encodings, axis=0)

    with closing(get_db()) as conn:
        conn.execute(
            "INSERT INTO users (id, username, encoding) VALUES (?, ?, ?)",
            (str(uuid.uuid4()), username, pickle.dumps(avg_encoding)),
        )
        conn.commit()

    return jsonify({"success": True, "message": "Registered successfully."})


@app.post("/api/login")
def login():
    """
    Body: { "username": "...", "image": "data:image/jpeg;base64,..." }
    Username narrows the search to one person's stored encoding (1:1 verification,
    which is faster and more reliable than 1:N identification against every user).
    """
    data = request.get_json(force=True)
    username = data.get("username", "").strip()
    img_b64 = data.get("image")

    if not username or not img_b64:
        return jsonify({"error": "username and image are required."}), 400

    with closing(get_db()) as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

    if row is None:
        return jsonify({"error": "No such user. Please register first."}), 404

    frame = decode_base64_image(img_b64)
    check = find_and_validate_face(frame)
    if not check.ok:
        return jsonify({"error": check.message}), 400

    candidate_encoding = get_face_encoding(check.face_image)
    if candidate_encoding is None:
        return jsonify({"error": "Could not read your face clearly. Try again."}), 400

    stored_encoding = pickle.loads(row["encoding"])
    is_match, distance = match_encoding(candidate_encoding, [stored_encoding])

    if not is_match:
        return jsonify({"error": "Face did not match.", "distance": distance}), 401

    session["user_id"] = row["id"]
    session["username"] = row["username"]
    return jsonify({"success": True, "username": row["username"], "distance": distance})


@app.post("/api/logout")
def logout():
    session.clear()
    return jsonify({"success": True})


@app.get("/api/session")
def session_status():
    if "user_id" in session:
        return jsonify({"logged_in": True, "username": session["username"]})
    return jsonify({"logged_in": False})


if __name__ == "__main__":
    # debug=True for local dev only
    app.run(host="0.0.0.0", port=5000, debug=True)
