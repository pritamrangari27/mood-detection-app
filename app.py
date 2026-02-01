"""Mood Detection Flask Application with Spotify Integration and PostgreSQL."""
import os
import time
import base64
import secrets
import random
import logging

import numpy as np
import cv2
from flask import Flask, request, render_template, redirect, url_for, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, get_jwt_identity, 
    set_access_cookies, unset_jwt_cookies
)
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# Config / App init
# ---------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "emotion_model.h5")

app = Flask(__name__, static_folder="static", template_folder="templates")

# Environment detection
IS_PRODUCTION = os.environ.get("FLASK_ENV", "development") == "production"

# Security configuration
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)
app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY") or secrets.token_hex(32)
app.config["JWT_TOKEN_LOCATION"] = ["cookies"]
app.config["JWT_COOKIE_SECURE"] = IS_PRODUCTION  # True for HTTPS in production
app.config["JWT_ACCESS_COOKIE_PATH"] = "/"
app.config["JWT_COOKIE_SAMESITE"] = "Strict" if IS_PRODUCTION else "Lax"
app.config["JWT_COOKIE_CSRF_PROTECT"] = IS_PRODUCTION  # Enable CSRF in production

# Warn if using default secrets in production
if IS_PRODUCTION and not os.environ.get("FLASK_SECRET_KEY"):
    logger.warning("FLASK_SECRET_KEY not set! Using random key (sessions won't persist across restarts)")
if IS_PRODUCTION and not os.environ.get("JWT_SECRET_KEY"):
    logger.warning("JWT_SECRET_KEY not set! Using random key (tokens won't persist across restarts)")

jwt = JWTManager(app)

# ---------------------------
# Database Configuration
# ---------------------------
DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL:
    # Fix for Render PostgreSQL URL (postgres:// -> postgresql://)
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
    logger.info("Using PostgreSQL database")
else:
    # Fallback to SQLite for local development
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///mood_app.db"
    logger.info("Using SQLite database (local development)")

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# ---------------------------
# Database Models
# ---------------------------
class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)
    email = db.Column(db.String(120), default="")
    gender = db.Column(db.String(20), default="")
    age = db.Column(db.String(10), default="")
    created_at = db.Column(db.DateTime, default=db.func.now())

    def to_dict(self):
        return {
            "username": self.username,
            "email": self.email or "None",
            "gender": self.gender or "None",
            "age": self.age or "None"
        }

class Message(db.Model):
    __tablename__ = "messages"
    id = db.Column(db.Integer, primary_key=True)
    sender = db.Column(db.String(80), nullable=False)
    receiver = db.Column(db.String(80), nullable=False)
    text = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.Integer, nullable=False)

    def to_dict(self):
        return {
            "from": self.sender,
            "to": self.receiver,
            "text": self.text,
            "ts": self.timestamp
        }

# Create tables
with app.app_context():
    db.create_all()
    logger.info("Database tables created")

# ---------------------------
# Load ML model (lazy loading to save memory)
# ---------------------------
model = None
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def get_model():
    """Lazy load the TensorFlow model to reduce startup memory."""
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}.")
        # Limit TensorFlow memory growth
        import tensorflow as tf
        tf.config.set_soft_device_placement(True)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        model = load_model(MODEL_PATH)
        logger.info("ML model loaded successfully")
    return model

# ---------------------------
# Spotify Setup
# ---------------------------
SPOTIFY_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET")

# Initialize Spotify client (will be None if credentials missing)
sp = None
if SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET:
    try:
        auth_manager = SpotifyClientCredentials(
            client_id=SPOTIFY_CLIENT_ID, 
            client_secret=SPOTIFY_CLIENT_SECRET
        )
        sp = spotipy.Spotify(auth_manager=auth_manager)
        logger.info("Spotify client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Spotify client: {e}")
else:
    logger.warning("Spotify credentials not configured. Music features will use fallback tracks.")

# ---------------------------
# Mood → Spotify
# ---------------------------
def mood_to_query(mood):
    mapping = {
        "neutral":"chill","angry":"rock","surprise":"party",
        "happy":"upbeat","sad":"melancholy","fear":"intense","disgust":"grunge"
    }
    return mapping.get(mood.lower(),mood.lower())

FALLBACK_TRACKS = [
    {"name": "Chill Vibes", "artist": "Unknown", "url": "https://open.spotify.com/embed/track/6rqhFgbbKwnb9MLmUQDhG6"},
    {"name": "Relaxing Beats", "artist": "Unknown", "url": "https://open.spotify.com/embed/track/0VjIjW4GlUZAMYd2vXMi3b"},
    {"name": "Mood Lifter", "artist": "Unknown", "url": "https://open.spotify.com/embed/track/7qiZfU4dY1lWllzX7mPBI3"},
    {"name": "Feel Good", "artist": "Unknown", "url": "https://open.spotify.com/embed/track/0pqnGHJpmpxLKifKRmU6WP"},
    {"name": "Easy Listening", "artist": "Unknown", "url": "https://open.spotify.com/embed/track/1BxfuPKGuaTgP7aM0Bbdwr"},
]
_song_cache = {}
CACHE_TTL_SECONDS = 10 * 60

def get_songs_spotify(mood: str) -> list:
    """Return 5 unique Spotify tracks for a given mood."""
    mood_key = (mood or "chill").lower()
    now = time.time()
    
    # Return fallback if Spotify not configured
    if sp is None:
        return FALLBACK_TRACKS[:5]
    
    # Use cache if valid
    cached = _song_cache.get(mood_key)
    if cached and now - cached[0] < CACHE_TTL_SECONDS:
        tracks = cached[1]
    else:
        query = f"{mood_to_query(mood)} bollywood"
        tracks = []
        try:
            results = sp.search(q=query, type="track", limit=50)
            seen = set()
            for t in results.get("tracks", {}).get("items", []):
                tid = t.get("id")
                if not tid or tid in seen:
                    continue
                tracks.append({
                    "name": t.get("name", "Unknown"),
                    "artist": t.get("artists", [{}])[0].get("name", "Unknown"),
                    "url": f"https://open.spotify.com/embed/track/{tid}"
                })
                seen.add(tid)
            # Pad with fallback tracks if needed
            while len(tracks) < 5:
                tracks.append(FALLBACK_TRACKS[len(tracks) % len(FALLBACK_TRACKS)].copy())
        except Exception as e:
            logger.error(f"Spotify search error: {e}")
            tracks = [t.copy() for t in FALLBACK_TRACKS[:5]]
        _song_cache[mood_key] = (now, tracks)
    
    if len(tracks) <= 5:
        return tracks
    return random.sample(tracks, 5)

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def root(): return redirect(url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    """Handle user registration."""
    error = None
    success = None
    
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        email = request.form.get("email", "").strip()
        gender = request.form.get("gender", "").strip()
        age = request.form.get("age", "").strip()
        
        if not username or not password:
            error = "Username and password are required"
        elif len(password) < 6:
            error = "Password must be at least 6 characters"
        elif User.query.filter_by(username=username).first():
            error = "Username already exists!"
        else:
            new_user = User(
                username=username,
                password=generate_password_hash(password),
                email=email,
                gender=gender,
                age=age
            )
            db.session.add(new_user)
            db.session.commit()
            success = "Registration successful! Please login."
            logger.info(f"New user registered: {username}")
    
    return render_template("register.html", error=error, success=success)

@app.route("/login", methods=["GET", "POST"])
def login():
    """Handle user login."""
    error = None
    
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            access_token = create_access_token(identity=username)
            resp = make_response(redirect(url_for("dashboard")))
            set_access_cookies(resp, access_token)
            logger.info(f"User logged in: {username}")
            return resp
        else:
            error = "Invalid username or password"
    
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    resp=make_response(redirect(url_for("login")))
    unset_jwt_cookies(resp)
    return resp

@app.route("/dashboard")
@jwt_required()
def dashboard():
    username = get_jwt_identity()
    user = User.query.filter_by(username=username).first()
    if not user:
        return redirect(url_for("logout"))
    return render_template("dashboard.html", 
                           username=username, 
                           email=user.email or "None",
                           gender=user.gender or "None", 
                           age=user.age or "None")

@app.route("/update_profile", methods=["POST"])
@jwt_required()
def update_profile():
    username = get_jwt_identity()
    user = User.query.filter_by(username=username).first()
    if user:
        user.gender = request.form.get("gender", "").strip()
        user.age = request.form.get("age", "").strip()
        user.email = request.form.get("email", "").strip()
        db.session.commit()
    return redirect(url_for("dashboard"))

# Load face cascade once at startup
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

@app.route("/predict", methods=["POST"])
@jwt_required()
def predict():
    """Predict emotion from uploaded face image."""
    body = request.get_json(silent=True)
    if not body or "image" not in body:
        return jsonify({"error": "No image provided", "emotion": "None"}), 400
    
    try:
        data_url = body["image"]
        # Decode base64 image
        img_data = base64.b64decode(data_url.split(",", 1)[1])
        arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Invalid image data", "emotion": "None"}), 400
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return jsonify({"emotion": "No face detected"})
        
        # Process first detected face
        (x, y, w, h) = faces[0]
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48)).astype("float32") / 255.0
        face_resized = np.expand_dims(face_resized, axis=(0, -1))
        
        # Use lazy-loaded model
        ml_model = get_model()
        prediction = ml_model.predict(face_resized, verbose=0)
        emotion = emotion_labels[int(np.argmax(prediction))]
        
        return jsonify({"emotion": emotion})
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": "Failed to process image", "emotion": "None"}), 500

@app.route("/songs", methods=["POST"])
def songs():
    """Get song recommendations based on mood."""
    body = request.get_json(silent=True)
    mood = body.get("mood") if body else None
    
    # Normalize mood
    safe_mood = (mood or "chill").strip()
    if safe_mood.lower() in ("no face detected", "not detected", "none", ""):
        safe_mood = "chill"
    
    tracks = get_songs_spotify(safe_mood)
    return jsonify({"mood": safe_mood, "songs": tracks})

# ---------------------------
# Chat routes
# ---------------------------
@app.route("/chat/send", methods=["POST"])
@jwt_required()
def chat_send():
    body = request.get_json()
    if not body or "to" not in body or "text" not in body:
        return jsonify({"error": "Missing fields"}), 400
    
    username = get_jwt_identity()
    new_msg = Message(
        sender=username,
        receiver=body["to"],
        text=body["text"],
        timestamp=int(time.time())
    )
    db.session.add(new_msg)
    db.session.commit()
    return jsonify({"status": "ok"})


@app.route("/chat/fetch", methods=["GET"])
@jwt_required()
def chat_fetch():
    chat_with = request.args.get("user")
    if not chat_with:
        return jsonify({"error": "Missing user"}), 400
    
    username = get_jwt_identity()
    
    messages = Message.query.filter(
        db.or_(
            db.and_(Message.sender == username, Message.receiver == chat_with),
            db.and_(Message.sender == chat_with, Message.receiver == username)
        )
    ).order_by(Message.timestamp).all()
    
    return jsonify([m.to_dict() for m in messages])

@app.route("/users/list", methods=["GET"])
@jwt_required()
def users_list():
    username = get_jwt_identity()
    users = User.query.filter(User.username != username).all()
    return jsonify([u.username for u in users])



# ---------------------------
# Run app
# ---------------------------
if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=debug_mode, use_reloader=debug_mode)
