# backend/app.py
# Signify: Urdu Sign Language Translator Backend (Levenshtein-based autocorrection)

import os
import base64
import threading
import uuid
import time
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
from gtts import gTTS
import joblib
import traceback
import csv
import functools

# ---------- CONFIG ----------
# Project root directory: when `app.py` lives at the repository root, use its folder.
ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(ROOT, "model"))
MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "model.h5")
LABEL_ENCODER_FILENAME = os.environ.get("LABEL_ENCODER_FILENAME", "label_encoder.pkl")
SCALER_FILENAME = os.environ.get("SCALER_FILENAME", "scaler.pkl")
# Allow overriding full paths via environment variables (useful for cloud hosting)
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(MODEL_DIR, MODEL_FILENAME))
LABEL_ENCODER_PATH = os.environ.get("LABEL_ENCODER_PATH", os.path.join(MODEL_DIR, LABEL_ENCODER_FILENAME))
SCALER_PATH = os.environ.get("SCALER_PATH", os.path.join(MODEL_DIR, SCALER_FILENAME))

STABLE_FRAMES = 2  # Number of stable frames required for prediction
CONF_THRESHOLD = 0.50  # Confidence threshold for predictions
STATE_TTL_MINUTES = 60  # Session state time-to-live
AUDIO_FOLDER = os.environ.get("AUDIO_FOLDER", os.path.join(ROOT, "frontend", "static", "audio"))
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# ---------- FLASK APP ----------
app = Flask(
    __name__,
    template_folder=os.path.join(ROOT, "frontend"),
    static_folder=os.path.join(ROOT, "frontend", "static")
)
CORS(app)

# ---------- LOAD MODEL, ENCODERS & SCALER ----------
model = None
label_encoder = None
scaler = None

try:
    if os.path.exists(MODEL_PATH):
        model = keras.models.load_model(MODEL_PATH)
        print(f"Model loaded: {MODEL_PATH}")
    else:
        print("Warning: CNN model file not found at", MODEL_PATH)
        print("You can set the MODEL_PATH environment variable to point to a model location in Azure.")
except Exception as e:
    print("⚠️ Model load error:", e)
    traceback.print_exc()

try:
    if os.path.exists(LABEL_ENCODER_PATH):
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    print("Label encoder and scaler load attempted.")
except Exception as e:
    print("⚠️ Encoder/Scaler load error:", e)
    traceback.print_exc()

# ---------- URDU MAP & DICTIONARY ----------
urdu_map = {
    "Alif": "ا", "Bay": "ب", "Pay": "پ", "Taay": "ت", "Tay": "ٹ", "Say": "ث",
    "Chay": "چ", "Hay": "ح", "Khay": "خ", "Daal": "د", "Dal": "ڈ",
    "Zaal": "ذ", "Ray": "ر", "Zay": "ز", "Zaey": "ژ", "Seen": "س",
    "Sheen": "ش", "Suad": "ص", "Zuad": "ض", "Tuey": "ط", "Zuey": "ظ", "Ain": "ع",
    "Ghain": "غ", "Fay": "ف", "Kaf": "ق", "Kiaf": "ک", "Gaaf": "گ", "Lam": "ل",
    "Meem": "م", "Nuun": "ن", "Wao": "و", "Cyeh": "ی", "Byeh": "ے",
    "Hamza": "ء", "Nuungh": "ں", "Dochahay": "ھ"
}

urdu_dict = []
try:
    _df_path = os.path.join(os.path.dirname(__file__), 'urdu_words.csv')
    if os.path.exists(_df_path):
        with open(_df_path, encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
            if not rows:
                urdu_dict = []
            else:
                header = [h.strip() for h in rows[0]] if rows and len(rows[0]) > 1 else []
                if 'words' in header:
                    idx = header.index('words')
                    urdu_dict = [r[idx].strip() for r in rows[1:] if len(r) > idx and r[idx].strip()]
                elif len(rows[0]) == 1:
                    # single-column CSV (no header)
                    urdu_dict = [r[0].strip() for r in rows if r and r[0].strip()]
                else:
                    # multi-column CSV without 'words' header: use first column excluding header row
                    urdu_dict = [r[0].strip() for r in rows[1:] if r and r[0].strip()]
    print(f"Loaded urdu dictionary from {_df_path} ({len(urdu_dict)} entries)")
except Exception as e:
    print("⚠️ Failed to load urdu_words.csv for autocorrect:", e)
    traceback.print_exc()
    urdu_dict = []

# ---------- MEDIAPIPE HANDS PROCESSOR ----------
mp_hands = mp.solutions.hands
hands_processor = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------- SESSION STATE MANAGEMENT ----------
state_lock = threading.Lock()
states = {}

def new_session_id():
    return str(uuid.uuid4())

def get_session_from_request():
    return request.headers.get("X-Session-Id") or request.cookies.get("session_id")

def get_state(sid):
    with state_lock:
        if sid not in states:
            states[sid] = {
                "prev_label": None,
                "stable_count": 0,
                "current_word": "",
                "urdu_sentence": [],
                "last_seen": datetime.utcnow(),
                "audio_file": None,
                "suggested_word": ""
            }
        states[sid]["last_seen"] = datetime.utcnow()
        return states[sid]

def cleanup_states_and_audio():
    while True:
        with state_lock:
            cutoff = datetime.utcnow() - timedelta(minutes=STATE_TTL_MINUTES)
            remove = [sid for sid, s in states.items() if s["last_seen"] < cutoff]
            for sid in remove:
                af = states[sid].get("audio_file")
                if af:
                    path = os.path.join(AUDIO_FOLDER, af)
                    try: os.remove(path)
                    except: pass
                del states[sid]
        now_ts = time.time()
        for fname in os.listdir(AUDIO_FOLDER):
            path = os.path.join(AUDIO_FOLDER, fname)
            if os.path.isfile(path) and now_ts - os.path.getmtime(path) > STATE_TTL_MINUTES * 60:
                try: os.remove(path)
                except: pass
        time.sleep(300)

threading.Thread(target=cleanup_states_and_audio, daemon=True).start()

# -----------------------------
# Levenshtein-only autocorrect helpers
# -----------------------------
def levenshtein_distance(a: str, b: str) -> int:
    """Compute Levenshtein distance between two strings using an iterative DP (memory-optimized)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    # ensure lb is smaller for less memory
    if la < lb:
        a, b = b, a
        la, lb = lb, la
    previous = list(range(lb + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * lb
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + cost
            current[j] = insertion if insertion < deletion else deletion
            if substitution < current[j]:
                current[j] = substitution
        previous = current
    return previous[lb]


# Prefer a C-optimized distance function when available, otherwise use the pure-Python one
try:
    import Levenshtein as _lev_ext
    distance_func = _lev_ext.distance
    print("Using python-Levenshtein C extension for distance calculations")
except Exception:
    distance_func = levenshtein_distance


# BK-tree for fast fuzzy lookup on large dictionaries
class BKTree:
    def __init__(self, words, dist_func):
        self.dist = dist_func
        self.root = None
        if words:
            it = iter(words)
            try:
                first = next(it)
            except StopIteration:
                return
            self.root = (first, {})
            for w in it:
                self.add(w)

    def add(self, word):
        if self.root is None:
            self.root = (word, {})
            return
        node = self.root
        while True:
            nword, children = node
            d = self.dist(word, nword)
            if d in children:
                node = children[d]
            else:
                children[d] = (word, {})
                break

    def search(self, word, max_dist):
        if self.root is None:
            return []
        results = []
        stack = [self.root]
        while stack:
            nword, children = stack.pop()
            d = self.dist(word, nword)
            if d <= max_dist:
                results.append((d, nword))
            low = d - max_dist
            high = d + max_dist
            for k, child in children.items():
                if low <= k <= high:
                    stack.append(child)
        return results


# Build BK-tree from `urdu_dict` (may be empty)
try:
    bk_tree = BKTree(urdu_dict, distance_func) if urdu_dict else None
    if bk_tree:
        print(f"Built BK-tree for urdu_dict ({len(urdu_dict)} words)")
except Exception:
    bk_tree = None


@functools.lru_cache(maxsize=2048)
def get_top_candidates(word, max_cand=50, max_distance=2):
    """Return top candidate words from `urdu_dict` ordered by Levenshtein distance.
    Uses a BK-tree for faster lookup when available.
    Always returns the closest matches, even if all are beyond max_distance.
    Results are cached to avoid repeated expensive scans.
    """
    if not urdu_dict:
        return tuple([word])

    # Try BK-tree first for speed (searches only nearby branches)
    results = []
    if bk_tree is not None:
        try:
            # Search within threshold; if insufficient results, expand threshold
            results = bk_tree.search(word, max_distance)
            if not results and max_distance < 10:
                # If no results within threshold, expand search radius
                results = bk_tree.search(word, max_distance + 2)
        except Exception:
            results = []

    # If BK-tree found results, return them sorted by distance
    if results:
        results.sort(key=lambda x: x[0])
        return tuple([ref for _, ref in results])[:max_cand]

    # Fallback: linear scan (ensures we find closest matches even if dict is huge)
    distances = []
    for ref in urdu_dict:
        try:
            d = distance_func(word, ref)
            distances.append((d, ref))
        except Exception:
            continue
    
    if not distances:
        return tuple([word])
    
    # Sort by distance and return top candidates
    distances.sort(key=lambda x: x[0])
    out = [ref for _, ref in distances[:max_cand]]
    return tuple(out) if out else tuple([word])

def select_best_word(candidates, sentence_context=None):
    """Pick the best candidate. Candidates are expected ordered by proximity."""
    if not candidates:
        return ""
    return candidates[0]

def suggest_next_word(sentence):
    # N-gram / GRU based suggestions removed — not available in Levenshtein-only mode.
    return ""

# ---------- HELPERS: prediction from landmarks ----------
def get_prediction_from_landmarks(landmarks):
    try:
        if model is None:
            return None
        if label_encoder is None or scaler is None:
            return None
        if len(landmarks) != 63:
            return None
        X = np.array(landmarks).reshape(1, -1)
        X_scaled = scaler.transform(X)
        X_scaled = X_scaled.reshape(1, 21, 3)
        preds = model.predict(X_scaled, verbose=0)
        idx = int(np.argmax(preds))
        try:
            label = label_encoder.inverse_transform([idx])[0]
        except Exception:
            label = str(idx)
        confidence = float(np.max(preds))
        return label, confidence
    except Exception as e:
        print("Error during prediction:", e)
        traceback.print_exc()
        return None

def autocorrect_word(word, sentence=""):
    """Autocorrect a word by finding its closest match in urdu_dict via Levenshtein distance.
    Returns the word itself if it's already in the dictionary (exact match).
    Short words (<=1 char) are not corrected.
    """
    if not word or len(word) <= 1 or word in urdu_dict:
        return word
    # get_top_candidates always returns the closest matches, cached for speed
    candidates = list(get_top_candidates(word, max_cand=50, max_distance=3))
    # Return the closest match (first in sorted order by distance)
    return candidates[0] if candidates else word

def finalize_and_tts_to_file(text, sid):
    words = [w['word'] for w in text if isinstance(w, dict) and 'word' in w]
    if not words:
        return "", None
    final_text = " ".join(words)
    try:
        tts = gTTS(text=final_text, lang="ur")
        filename = f"{sid}_{int(time.time())}.mp3"
        path = os.path.join(AUDIO_FOLDER, filename)
        tts.save(path)
        return final_text, filename
    except Exception as e:
        print("TTS error:", e)
        traceback.print_exc()
        return final_text, None

# ---------- ROUTES ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/session", methods=["POST"])
def create_session():
    sid = new_session_id()
    resp = jsonify({"session_id": sid})
    resp.set_cookie("session_id", sid, max_age=60*60*24)
    return resp

@app.route("/predict_page")
def predict_page():
    # Predict page was consolidated into the main index — serve index to avoid missing template errors
    return render_template("index.html")

@app.route("/_debug_loads")
def _debug_loads():
    return jsonify({
        "model_loaded": model is not None,
        "autocorrect_enabled": True,
        "urdu_dict_size": len(urdu_dict),
        "bk_tree_enabled": bk_tree is not None
    })

@app.route("/status")
def status():
    try:
        status = {
            "model": {"loaded": model is not None, "path": MODEL_PATH},
            "label_encoder": {"loaded": label_encoder is not None, "path": LABEL_ENCODER_PATH},
            "scaler": {"loaded": scaler is not None, "path": SCALER_PATH},
        }
        return jsonify({"status": "ok", "components": status})
    except Exception as e:
        error_msg = f"Error checking component status: {str(e)}"
        app.logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return jsonify({"status": "error", "error": error_msg}), 500

@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        sid = get_session_from_request() or request.remote_addr or "anon"
        s = get_state(sid)
        data = request.get_json(force=True)
        img_b64 = data.get("image")
        if not img_b64:
            return jsonify({"error": "no image"}), 400
        if "," in img_b64:
            img_b64 = img_b64.split(",")[1]
        try:
            img_bytes = base64.b64decode(img_b64)
        except Exception:
            return jsonify({"error": "invalid base64"}), 400
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "invalid image decoded"}), 400
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands_processor.process(rgb)
        label_out = None
        conf_out = None
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            lm = []
            for pt in hand.landmark:
                lm.extend([pt.x, pt.y, pt.z])
            pred = get_prediction_from_landmarks(lm)
            if pred:
                label, confidence = pred
                label_out, conf_out = label, confidence
                if confidence >= CONF_THRESHOLD:
                    if label == s["prev_label"]:
                        s["stable_count"] += 1
                    else:
                        s["prev_label"] = label
                        s["stable_count"] = 1
                    if s["stable_count"] >= STABLE_FRAMES:
                        urdu_char = urdu_map.get(label)
                        if urdu_char:
                            if (not s["current_word"]) or (s["current_word"][-1] != urdu_char):
                                s["current_word"] += urdu_char
                        s["stable_count"] = 0
                        s["prev_label"] = None
                else:
                    s["prev_label"] = None
                    s["stable_count"] = 0
        return jsonify({
            "word": s["current_word"],
            "sentence": s["urdu_sentence"],
            "label": label_out,
            "confidence": conf_out
        })
    except Exception as e:
        print("Prediction API error:", e)
        traceback.print_exc()
        return jsonify({"error": "internal error"}), 500

@app.route("/space", methods=["POST"])
def space_api():
    sid = get_session_from_request() or request.remote_addr or "anon"
    s = get_state(sid)
    suggestions = None
    if s["current_word"]:
        is_correct = s["current_word"] in urdu_dict
        word_id = str(uuid.uuid4())
        word_obj = {
            "word": s["current_word"],
            "is_correct": is_correct,
            "id": word_id
        }
        s["urdu_sentence"].append(word_obj)

        if not is_correct:
            candidates = get_top_candidates(s["current_word"], max_cand=10, max_distance=3)
            suggestions = {
                "word_id": word_id,
                "options": candidates
            }

        s["current_word"] = ""

    # The suggest_next_word function is not used anymore
    # suggestion = suggest_next_word(s["urdu_sentence"])
    # s["suggested_word"] = suggestion
    
    return jsonify({
        "sentence": s["urdu_sentence"],
        "suggestions": suggestions
    })

@app.route("/replace_word", methods=["POST"])
def replace_word_api():
    sid = get_session_from_request() or request.remote_addr or "anon"
    s = get_state(sid)
    data = request.get_json(force=True)
    word_id = data.get("word_id")
    new_word = data.get("new_word")

    if not all([word_id, new_word]):
        return jsonify({"error": "Missing word_id or new_word"}), 400

    for word_obj in s["urdu_sentence"]:
        if word_obj.get("id") == word_id:
            word_obj["word"] = new_word
            word_obj["is_correct"] = True
            break
    
    return jsonify({"sentence": s["urdu_sentence"]})


@app.route("/finalize", methods=["POST"])
def finalize_api():
    sid = get_session_from_request() or request.remote_addr or "anon"
    s = get_state(sid)
    # Build a list of word dicts to pass to the TTS helper. Keep existing sentence tokens
    list_words = list(s.get("urdu_sentence") or [])
    if s.get("current_word"):
        list_words.append({"word": s["current_word"], "is_correct": s["current_word"] in urdu_dict, "id": str(uuid.uuid4())})
    final_text, audio_file = finalize_and_tts_to_file(list_words, sid)
    # Reset session state
    s["urdu_sentence"] = []
    s["current_word"] = ""
    s["prev_label"] = None
    s["stable_count"] = 0
    s["audio_file"] = audio_file
    if audio_file:
        path = os.path.join(AUDIO_FOLDER, audio_file)
        try:
            with open(path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")
            return jsonify({"final_sentence": final_text, "audio_file": audio_file, "audio_base64": audio_b64})
        except Exception as e:
            print("Error reading audio:", e)
            traceback.print_exc()
            return jsonify({"final_sentence": final_text, "audio_file": audio_file, "audio_base64": None})
    else:
        return jsonify({"final_sentence": final_text, "audio_file": None, "audio_base64": None})

@app.route("/audio/<filename>")
def audio_file(filename):
    return send_from_directory(AUDIO_FOLDER, filename)


@app.route('/speak', methods=['POST'])
def speak_api():
    """Generate TTS for provided text and return base64 audio and filename."""
    sid = get_session_from_request() or request.remote_addr or "anon"
    s = get_state(sid)
    try:
        data = request.get_json(force=True)
        text = data.get('text') if data else None
        if not text:
            return jsonify({'error': 'no text provided'}), 400
        # Generate TTS and save to AUDIO_FOLDER
        filename = f"{sid}_speak_{int(time.time())}.mp3"
        path = os.path.join(AUDIO_FOLDER, filename)
        try:
            tts = gTTS(text=text, lang='ur')
            tts.save(path)
        except Exception as e:
            app.logger.error(f"TTS generation failed: {e}")
            traceback.print_exc()
            return jsonify({'error': 'tts_failed'}), 500

        # Read and return base64 audio
        try:
            with open(path, 'rb') as f:
                audio_b64 = base64.b64encode(f.read()).decode('utf-8')
            # store in session state for possible cleanup or later access
            s['audio_file'] = filename
            return jsonify({'audio_base64': audio_b64, 'audio_file': filename})
        except Exception as e:
            app.logger.error(f"Failed to read TTS file: {e}")
            traceback.print_exc()
            return jsonify({'error': 'read_error'}), 500
    except Exception as e:
        app.logger.error(f"/speak error: {e}\n{traceback.format_exc()}")
        return jsonify({'error': 'internal_error'}), 500

@app.route("/clear", methods=["POST"])
def clear_api():
    sid = get_session_from_request() or request.remote_addr or "anon"
    with state_lock:
        if sid in states:
            af = states[sid].get("audio_file")
            if af:
                path = os.path.join(AUDIO_FOLDER, af)
                try: os.remove(path)
                except: pass
            del states[sid]
    return jsonify({"ok": True})

@app.route("/delete_last", methods=["POST"])
def delete_last_api():
    sid = get_session_from_request() or request.remote_addr or "anon"
    s = get_state(sid)
    if s["current_word"]:
        s["current_word"] = s["current_word"][:-1]
    return jsonify({"word": s["current_word"], "sentence": s["urdu_sentence"]})

# ---------- MAIN ----------
if __name__ == "__main__":
    # IMPORTANT:
    # Azure App Service runs this app using Gunicorn (via startup.sh),
    # so Flask's built-in server is only for local development.
    debug_mode = os.environ.get("DEBUG", "False").lower() in ("1", "true", "yes")
    port = int(os.environ.get("PORT", 8000))  # Azure requires port 8000
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
