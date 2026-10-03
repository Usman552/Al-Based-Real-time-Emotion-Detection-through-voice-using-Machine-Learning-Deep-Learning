import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pickle
import tempfile
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Number of fused acoustic features produced by extract_features().
# 80 MFCC + 80 delta + 80 delta-delta + 12 chroma + 20 mel + 7 contrast
# + 1 ZCR + 1 RMS + 3 spectral-shape = 284
N_FEATURES = 284

# Resolve model files relative to this script so `streamlit run` works from any directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# If an ensemble has been trained (train_ensemble.py --deploy) it is used in place of
# the single model: its members' softmax outputs are averaged, which is more accurate.
# Delete or rename the folder to fall back to the single model.
ENSEMBLE_DIR = os.path.join(BASE_DIR, "ensemble")


class Ensemble:
    """Soft-voting wrapper exposing the same predict/count_params surface as a model."""

    def __init__(self, models):
        self.models = models

    def predict(self, x, verbose=0):
        # For the single-clip batches this app sends, calling the model directly is far
        # cheaper than predict(), which rebuilds a tf.function graph on every call.
        if len(x) <= 32:
            return np.mean([np.asarray(m(x, training=False)) for m in self.models], axis=0)
        return np.mean([m.predict(x, verbose=0) for m in self.models], axis=0)

    def count_params(self):
        return sum(m.count_params() for m in self.models)

    def __len__(self):
        return len(self.models)


# ── Model Load ──
@st.cache_resource
def load_model():
    # compile=False everywhere: the models were trained with a custom focal loss,
    # which inference does not need and which cannot be deserialized without it.
    try:
        members = sorted(
            f for f in os.listdir(ENSEMBLE_DIR) if f.startswith("member_")
        ) if os.path.isdir(ENSEMBLE_DIR) else []
    except OSError:
        members = []

    if members:
        try:
            models = [tf.keras.models.load_model(
                os.path.join(ENSEMBLE_DIR, f), compile=False) for f in members]
            with open(os.path.join(ENSEMBLE_DIR, "label_encoder.pkl"), "rb") as f:
                return Ensemble(models), pickle.load(f)
        except Exception as e:
            st.warning(f"Ensemble failed to load ({e}); falling back to the single model.")

    try:
        model = tf.keras.models.load_model(
            os.path.join(BASE_DIR, "emotion_model.h5"), compile=False
        )
        with open(os.path.join(BASE_DIR, "label_encoder.pkl"), "rb") as f:
            le = pickle.load(f)
        return model, le
    except FileNotFoundError:
        st.error("❌ Model files not found! Please make sure emotion_model.h5, label_encoder.pkl are present.")
        st.stop()

@st.cache_resource
def load_scaler():
    # the scaler must match whichever model was loaded above
    ens = os.path.join(ENSEMBLE_DIR, "scaler.pkl")
    path = ens if os.path.exists(ens) else os.path.join(BASE_DIR, "scaler.pkl")
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("❌ scaler.pkl not found!")
        st.stop()

# ── Warm-up ──
@st.cache_resource
def warm_up(_model, _scaler):
    """Compile librosa's numba kernels and TensorFlow's graph during startup.

    Without this the FIRST analysis a user runs takes about six seconds while
    numba JIT-compiles the MFCC/chroma kernels, which looks like the app has
    frozen. Doing it here moves that cost into the initial load, so every real
    analysis afterwards takes roughly 0.3 s.
    """
    try:
        sr = 22050
        a = (0.3 * np.sin(2 * np.pi * 220 * np.linspace(0, 1.0, sr))).astype(np.float32)
        a, _ = librosa.effects.trim(a, top_db=25)
        a = librosa.util.normalize(a)
        mf = librosa.feature.mfcc(y=a, sr=sr, n_mfcc=40)
        librosa.feature.delta(mf)
        librosa.feature.delta(mf, order=2)
        librosa.feature.chroma_stft(y=a, sr=sr)
        librosa.feature.melspectrogram(y=a, sr=sr)
        librosa.feature.spectral_contrast(y=a, sr=sr)
        librosa.feature.zero_crossing_rate(y=a)
        librosa.feature.rms(y=a)
        librosa.feature.spectral_centroid(y=a, sr=sr)
        librosa.feature.spectral_bandwidth(y=a, sr=sr)
        librosa.feature.spectral_rolloff(y=a, sr=sr)
        _model.predict(_scaler.transform(np.zeros((1, N_FEATURES))), verbose=0)
    except Exception:
        pass          # a failed warm-up only costs speed, never correctness
    return True


# ── Feature Extraction ──
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=22050)
        if len(audio) == 0:
            st.error("Empty audio file"); return None, None, None

        audio_p, _ = librosa.effects.trim(audio, top_db=25)
        if len(audio_p) < sr // 2:
            audio_p = np.pad(audio_p, (0, sr // 2 - len(audio_p)))
        audio_p = librosa.util.normalize(audio_p)

        mfcc = librosa.feature.mfcc(y=audio_p, sr=sr, n_mfcc=40)
        d1   = librosa.feature.delta(mfcc)
        d2   = librosa.feature.delta(mfcc, order=2)
        def ms(x): return np.concatenate([x.mean(axis=1), x.std(axis=1)])

        mfcc_f   = ms(mfcc)                                                          # 80
        d1_f     = ms(d1)                                                            # 80
        d2_f     = ms(d2)                                                            # 80
        chroma   = librosa.feature.chroma_stft(y=audio_p, sr=sr).mean(axis=1)        # 12
        mel      = librosa.feature.melspectrogram(y=audio_p, sr=sr).mean(axis=1)[:20]# 20
        contrast = librosa.feature.spectral_contrast(y=audio_p, sr=sr).mean(axis=1)  # 7
        zcr = np.array([librosa.feature.zero_crossing_rate(y=audio_p).mean()])       # 1
        rms = np.array([librosa.feature.rms(y=audio_p).mean()])                      # 1
        cent = librosa.feature.spectral_centroid(y=audio_p, sr=sr).mean()
        bw   = librosa.feature.spectral_bandwidth(y=audio_p, sr=sr).mean()
        roll = librosa.feature.spectral_rolloff(y=audio_p, sr=sr).mean()
        spec = np.array([cent, bw, roll])                                            # 3

        features = np.concatenate([mfcc_f, d1_f, d2_f, chroma, mel, contrast,
                                   zcr, rms, spec])                                  # 284

        # Guard against silent drift between this routine and the one used for
        # training — a mismatched vector would produce meaningless predictions.
        if features.shape[0] != N_FEATURES:
            st.error(f"Feature vector has {features.shape[0]} values, expected {N_FEATURES}.")
            return None, None, None
        return features, audio, sr
    except Exception as e:
        st.error(f"Feature extraction error: {e}")
        return None, None, None



# ── Feedback storage ──
# Corrections confirmed by users are appended here as feature vectors (never audio,
# so nothing identifiable is retained). auto_train.py folds them into later training
# rounds, which is how the model keeps improving after deployment.
FEEDBACK_DIR = os.path.join(BASE_DIR, "feedback")
FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, "feedback.csv")


def save_feedback(features, label, predicted, confidence, filename):
    """Append one labelled example. Returns True on success."""
    try:
        os.makedirs(FEEDBACK_DIR, exist_ok=True)
        write_header = not os.path.exists(FEEDBACK_FILE)
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file": filename,
            "label": label,
            "predicted": predicted,
            "confidence": round(float(confidence), 2),
            "was_correct": int(label == predicted),
        }
        row.update({f"f{i}": float(v) for i, v in enumerate(features)})
        with open(FEEDBACK_FILE, "a", encoding="utf-8", newline="") as f:
            if write_header:
                f.write(",".join(row.keys()) + "\n")
            f.write(",".join(str(v) for v in row.values()) + "\n")
        return True
    except Exception as e:
        st.error(f"Could not save feedback: {e}")
        return False


def feedback_count():
    try:
        if not os.path.exists(FEEDBACK_FILE):
            return 0
        with open(FEEDBACK_FILE, encoding="utf-8") as f:
            return max(0, sum(1 for _ in f) - 1)
    except Exception:
        return 0


# ── Prediction ──
def predict_emotion(file_path, model, le):
    features, audio, sr = extract_features(file_path)
    if features is None:
        return None, None, None, None, None

    # Kept so the feedback control can store the exact vector that was classified.
    st.session_state.last_features = features

    scaler = load_scaler()
    features_scaled = scaler.transform(np.array([features]))
    prediction = model.predict(features_scaled, verbose=0)

    emotion_index = np.argmax(prediction)
    emotion = le.classes_[emotion_index]

    # surprise/surprised fix — normalize label
    if emotion.lower() in ["surprised", "pleasant_surprised", "pleasant_surprise"]:
        emotion = "surprise"

    confidence = prediction[0][emotion_index] * 100
    all_confidences = {}
    for i in range(len(le.classes_)):
        label = le.classes_[i]
        if label.lower() in ["surprised", "pleasant_surprised", "pleasant_surprise"]:
            label = "surprise"
        # merge surprise + surprised scores
        if label in all_confidences:
            all_confidences[label] += prediction[0][i] * 100
        else:
            all_confidences[label] = prediction[0][i] * 100

    return emotion, confidence, all_confidences, audio, sr

# ── Emoji Map ──
emotion_emojis = {
    "angry":    "😠",
    "disgust":  "🤢",
    "fear":     "😨",
    "happy":    "😊",
    "neutral":  "😐",
    "sad":      "😢",
    "surprise": "😲",
    "surprised":"😲",
    "pleasant_surprised": "😲",
}

# ── Show Results ──
def show_results(emotion, confidence, all_confidences, audio, sr, filename):
    emoji = emotion_emojis.get(emotion.lower(), "🎭")
    accent = emo_color(emotion)
    ranked = sorted(all_confidences.items(), key=lambda x: x[1], reverse=True)

    if confidence < 50:
        st.warning(
            f"Confidence is only {confidence:.1f}%. The clip may be too short, too quiet "
            "or too noisy — the result below is a best guess rather than a reliable reading."
        )

    left, right = st.columns([1, 1.35], gap="large")

    # ── predicted emotion card ──
    with left:
        band = ("#34D399", "Strong") if confidence >= 75 else \
               ("#FBBF24", "Moderate") if confidence >= 50 else ("#FB7185", "Low")
        st.markdown(f"""
        <div class="card" style="border-color:{accent}55;
             box-shadow:0 0 0 1px {accent}22, 0 12px 40px -12px {accent}44;">
          <h3>Detected Emotion</h3>
          <div class="emo">
            <div class="face" style="filter:drop-shadow(0 0 26px {accent}88);">{emoji}</div>
            <div class="name" style="color:{accent};
                 text-shadow:0 0 26px {accent}66;">{emotion}</div>
            <div class="conf">
              <span class="pill" style="color:{band[0]};">{band[1]} · {confidence:.1f}%</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        secs = len(audio) / sr if sr else 0
        margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Duration", f"{secs:.1f}s")
        c2.metric("Runner-up", str(ranked[1][0]).capitalize() if len(ranked) > 1 else "—")
        c3.metric("Margin", f"{margin:.0f} pts")

    # ── full confidence breakdown ──
    with right:
        rows = ""
        for em, conf in ranked:
            col = emo_color(em)
            strong = "font-weight:700;" if em == emotion else ""
            rows += (
                f'<div class="row">'
                f'  <div class="lbl" style="{strong}">'
                f'    {emotion_emojis.get(str(em).lower(), "🎭")} {str(em).capitalize()}</div>'
                f'  <div class="track"><div class="fill" '
                f'       style="width:{max(conf, 0.6):.1f}%;background:{col};"></div></div>'
                f'  <div class="val" style="{strong}">{conf:.1f}%</div>'
                f'</div>'
            )
        st.markdown(
            f'<div class="card"><h3>Confidence across all seven emotions</h3>{rows}</div>',
            unsafe_allow_html=True)

    # ── waveform ──
    # A real Streamlit container is used here rather than a raw HTML <div>: Streamlit
    # renders widgets in their own blocks, so a chart cannot be nested inside markup.
    with st.container(border=True):
        st.markdown(
            '<h3 style="margin:0 0 6px;font-size:.68rem;font-weight:700;color:#8A94A6;'
            'text-transform:uppercase;letter-spacing:1.4px;font-family:ui-monospace,'
            'Consolas,monospace;">Waveform</h3>',
            unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(11, 2.1))
        t = np.linspace(0, len(audio) / sr, len(audio))
        # stacked translucent strokes fake a neon glow around the trace
        for lw, a in ((5.0, 0.05), (3.0, 0.09), (1.6, 0.18)):
            ax.plot(t, audio, color=accent, linewidth=lw, alpha=a,
                    solid_capstyle="round")
        ax.plot(t, audio, color=accent, linewidth=0.55, alpha=0.95)
        ax.fill_between(t, audio, color=accent, alpha=0.10)
        ax.axhline(0, color="#2A3342", linewidth=0.8)
        ax.set_xlabel("seconds", fontsize=8, color="#8A94A6",
                      fontfamily="monospace", labelpad=6)
        ax.set_xlim(0, max(t[-1] if len(t) else 1, 0.1))
        ax.set_yticks([])
        ax.tick_params(colors="#8A94A6", labelsize=7.5)
        for lbl in ax.get_xticklabels():
            lbl.set_fontfamily("monospace")
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#262D3A")
        ax.grid(axis="x", color="#1C2330", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_facecolor("#0E121A")
        fig.patch.set_alpha(0)
        plt.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    # ── feedback: lets the model improve after deployment ──
    feedback_ui(emotion, confidence, filename)

    # ── session history (guarded so a rerun does not duplicate the entry) ──
    if "history" not in st.session_state:
        st.session_state.history = []
    entry = {
        "file": filename,
        "emotion": f"{emoji} {emotion}",
        "confidence": f"{confidence:.1f}%",
        "time": datetime.now().strftime("%H:%M:%S"),
    }
    token = f"{filename}|{emotion}|{confidence:.4f}"
    if st.session_state.get("last_logged") != token:
        st.session_state.history.append(entry)
        st.session_state.last_logged = token


def feedback_ui(emotion, confidence, filename):
    """Confirm or correct a prediction; stored samples feed the next training round."""
    features = st.session_state.get("last_features")
    if features is None:
        return

    token = f"{filename}|{emotion}|{confidence:.4f}"
    if st.session_state.get("feedback_done") == token:
        st.success("Thanks — your answer was saved and will be used to improve the model.")
        return

    with st.expander("Was this prediction correct?  (helps the model improve)", expanded=False):
        st.caption(
            "Only the 284 numeric features are stored — never your audio. "
            "Confirmed and corrected samples are folded into the next training round."
        )
        c1, c2 = st.columns([1, 2])
        with c1:
            if st.button("✅ Yes, correct", width="stretch"):
                if save_feedback(features, emotion, emotion, confidence, filename):
                    st.session_state.feedback_done = token
                    st.rerun()
        with c2:
            options = [str(c) for c in le.classes_]
            choice = st.selectbox(
                "No — the correct emotion was:", options,
                index=options.index(emotion) if emotion in options else 0,
                key=f"fb_{token}", label_visibility="collapsed",
                placeholder="Pick the correct emotion",
            )
            if st.button("Submit correction", width="stretch"):
                if save_feedback(features, choice, emotion, confidence, filename):
                    st.session_state.feedback_done = token
                    st.rerun()

# ════════════════════════════════
# Page Config
# ════════════════════════════════
st.set_page_config(
    page_title="Emotion Recognition from Speech",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  :root{
    --bg:#0B0D12; --panel:#151922; --panel-2:#1B2029; --line:#262D3A;
    --ink:#E8ECF4; --muted:#8A94A6; --accent:#22D3EE;
    --mono: ui-monospace, "SF Mono", "JetBrains Mono", Consolas, monospace;
  }
  .stApp{ background:
      radial-gradient(1200px 520px at 12% -8%, #16202E 0%, transparent 60%),
      radial-gradient(900px 420px at 92% 0%, #1A1526 0%, transparent 58%),
      var(--bg);
  }
  .block-container{ padding-top:1.4rem; max-width:1320px; }
  /* no sidebar content any more — reclaim the space for the content */
  section[data-testid="stSidebar"]{ display:none !important; }

  /* footer strip */
  .foot{ display:flex; gap:22px; flex-wrap:wrap; align-items:center;
         border-top:1px solid var(--line); margin-top:26px; padding-top:14px;
         font-family:var(--mono); font-size:.72rem; color:var(--muted); }

  /* ── header: a console strip, not a marketing banner ── */
  .hero{ border:1px solid var(--line); border-radius:14px; padding:18px 22px;
         margin-bottom:18px; background:linear-gradient(180deg,#141926 0%,#10141C 100%);
         position:relative; overflow:hidden; }
  .hero:before{ content:""; position:absolute; left:0; top:0; bottom:0; width:3px;
                background:linear-gradient(180deg,var(--accent),#A855F7); }
  .hero h1{ font-size:1.35rem; font-weight:700; margin:0 0 4px; letter-spacing:-.2px; }
  .hero p{ margin:0; font-size:.83rem; color:var(--muted); font-family:var(--mono); }
  .hero .dot{ display:inline-block; width:7px; height:7px; border-radius:50%;
              background:var(--accent); margin-right:7px; vertical-align:middle;
              box-shadow:0 0 8px var(--accent); }
  .hero-row{ display:flex; align-items:center; justify-content:space-between;
             gap:24px; flex-wrap:wrap; }
  .hero-stats{ display:flex; gap:26px; }
  .hero-stats > div{ display:flex; flex-direction:column; gap:3px; }
  .hero-stats .k{ font-size:.62rem; letter-spacing:1.4px; text-transform:uppercase;
                  color:var(--muted); font-family:var(--mono); }
  .hero-stats .v{ font-size:1.15rem; font-weight:700; font-family:var(--mono);
                  color:var(--accent); }
  .hero-stats .v.sm{ font-size:.82rem; font-weight:600; color:var(--ink); }

  /* ── navbar: the page's only radio, restyled as tabs ──
     Targeted directly rather than through a wrapper class: Streamlit renders each
     markdown block in its own container, so an injected <div> is never a sibling
     of the widget it is meant to scope. */
  div[role="radiogroup"]{ display:flex !important; gap:8px; flex-wrap:wrap;
                          margin:0 0 20px; }
  div[role="radiogroup"] > label{
      background:var(--panel); border:1px solid var(--line); border-radius:10px;
      padding:9px 18px 9px 14px; margin:0 !important; cursor:pointer;
      transition:border-color .15s, background .15s; }
  div[role="radiogroup"] > label:hover{ border-color:#3A4556; background:var(--panel-2); }
  /* hide the radio dot so each option reads as a tab */
  div[role="radiogroup"] > label > div[data-testid="stRadioButtonInner"],
  div[role="radiogroup"] > label > div:first-child{ display:none !important; }
  div[role="radiogroup"] > label p{ font-size:.87rem !important; margin:0 !important; }
  div[role="radiogroup"] > label:has(input:checked){
      background:#16222E; border-color:#22D3EE88;
      box-shadow:0 0 0 1px #22D3EE22, 0 8px 24px -12px #22D3EE99; }
  div[role="radiogroup"] > label:has(input:checked) p{
      color:var(--accent) !important; font-weight:600 !important; }

  /* ── panels ── */
  .card{ background:var(--panel); border:1px solid var(--line); border-radius:14px;
         padding:18px 20px; margin-bottom:14px; }
  .card h3{ margin:0 0 14px; font-size:.68rem; font-weight:700; color:var(--muted);
            text-transform:uppercase; letter-spacing:1.4px; font-family:var(--mono); }

  /* ── the emotion readout ── */
  .emo{ text-align:center; padding:14px 0 6px; }
  .emo .face{ font-size:4.2rem; line-height:1.05; }
  .emo .name{ font-size:2.0rem; font-weight:800; letter-spacing:1px; margin-top:6px;
              text-transform:uppercase; }
  .emo .conf{ font-size:.82rem; margin-top:10px; font-family:var(--mono); }

  /* ── level meters ── */
  .row{ display:flex; align-items:center; gap:12px; margin:10px 0; }
  .row .lbl{ width:108px; font-size:.85rem; text-transform:capitalize; }
  .row .track{ flex:1; height:8px; background:#0E121A; border-radius:2px;
               overflow:hidden; border:1px solid #222A36; }
  .row .fill{ height:100%; border-radius:1px; }
  .row .val{ width:56px; text-align:right; font-size:.8rem; font-family:var(--mono);
             font-variant-numeric:tabular-nums; }

  .pill{ display:inline-block; padding:4px 12px; border-radius:4px; font-size:.72rem;
         font-weight:700; letter-spacing:.6px; font-family:var(--mono);
         text-transform:uppercase; border:1px solid currentColor; }

  /* ── chrome ── */
  section[data-testid="stSidebar"]{ background:#0E121A; border-right:1px solid var(--line); }
  div[data-testid="stMetricValue"]{ font-size:1.25rem; font-family:var(--mono); }
  div[data-testid="stMetricLabel"] p{ font-size:.7rem !important; letter-spacing:1px;
                                      text-transform:uppercase; }
  [data-testid="stFileUploaderDropzone"], .stExpander details{
      background:var(--panel) !important; border:1px dashed var(--line) !important;
      border-radius:12px !important; }
  .stExpander details{ border-style:solid !important; }
  div[data-testid="stDataFrame"]{ border:1px solid var(--line); border-radius:10px; }
  hr{ border-color:var(--line) !important; }
  footer, #MainMenu{ visibility:hidden; }

  /* ── readability: panels are dark, so text must stay light even if a viewer
        switches Streamlit to its light theme ── */
  .stApp [data-testid="stMarkdownContainer"],
  .stApp [data-testid="stMarkdownContainer"] *,
  .stApp [data-testid="stWidgetLabel"], .stApp [data-testid="stWidgetLabel"] *,
  .stApp [data-testid="stMetricLabel"], .stApp [data-testid="stMetricLabel"] *,
  .stApp [data-testid="stMetricValue"],
  .stApp .stRadio label, .stApp .stRadio label *,
  .stApp .stSelectbox label, .stApp .stExpander summary,
  .stApp .stExpander summary *,
  section[data-testid="stSidebar"] *{ color:var(--ink); }

  /* elements that carry their own colour win that rule back */
  .card h3{ color:var(--muted) !important; }
  .hero p{ color:var(--muted) !important; }
  .row .val, .stApp [data-testid="stCaptionContainer"],
  .stApp [data-testid="stCaptionContainer"] *{ color:var(--muted) !important; }
  .stApp button[kind="primary"], .stApp button[kind="primary"] *{ color:#04121A !important; }
</style>
""", unsafe_allow_html=True)

with st.spinner("Loading model and warming up the audio pipeline…"):
    model, le = load_model()
    warm_up(model, load_scaler())

# Per-emotion accent colour. Tuned for a dark ground: saturated enough to glow,
# light enough to stay legible as text.
EMOTION_COLORS = {
    "angry": "#FF4D5E", "disgust": "#A3E635", "fear": "#A78BFA",
    "happy": "#FBBF24", "neutral": "#94A3B8", "sad": "#38BDF8",
    "surprise": "#F472B6",
}


def emo_color(name):
    return EMOTION_COLORS.get(str(name).lower(), "#22D3EE")

# ── Top navigation bar ──
# Duck-typed on purpose: Streamlit re-executes this module on every rerun, which
# rebinds the Ensemble class, so the cached instance would fail an isinstance check.
IS_ENSEMBLE = hasattr(model, "models")
ACC = "77.89%" if IS_ENSEMBLE else "76.73%"
F1 = "78.50%" if IS_ENSEMBLE else "77.16%"
MODEL_KIND = f"{len(model)}-member ensemble" if IS_ENSEMBLE else "single network"

st.markdown(f"""
<div class="hero">
  <div class="hero-row">
    <div>
      <h1>Emotion Recognition from Speech</h1>
      <p><span class="dot"></span>{N_FEATURES}-feature acoustic model &nbsp;·&nbsp;
         {len(le.classes_)} classes &nbsp;·&nbsp; local inference &nbsp;·&nbsp; FYP, USP Multan</p>
    </div>
    <div class="hero-stats">
      <div><span class="k">accuracy</span><span class="v">{ACC}</span></div>
      <div><span class="k">macro f1</span><span class="v">{F1}</span></div>
      <div><span class="k">model</span><span class="v sm">{MODEL_KIND}</span></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# A horizontal radio is used rather than tabs so that only the selected page's code
# runs; st.tabs would execute every page on every rerun. CSS turns it into a navbar.
page = st.radio("Go to", [
    "🎵 Upload Audio",
    "🎤 Live Recording",
    "📊 About Emotions",
    "ℹ️ About Project",
    "📋 Session History",
], horizontal=True, label_visibility="collapsed")

# ════════════════════════════════
# PAGE 1 — Upload Audio
# ════════════════════════════════
if page == "🎵 Upload Audio":
    st.markdown(
        '<h3 style="margin:2px 0 2px;font-size:.68rem;font-weight:700;color:#8A94A6;'
        'text-transform:uppercase;letter-spacing:1.4px;font-family:ui-monospace,'
        'Consolas,monospace;">Input · Audio File</h3>'
        '<p style="color:#8A94A6;font-size:.85rem;margin:0 0 12px;">WAV, MP3 or FLAC — '
        'a clear 2–10 second clip with a single speaker works best.</p>',
        unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an audio file", type=["wav", "mp3", "flac"], label_visibility="collapsed")

    if uploaded_file is None:
        chips = "".join(
            f'<div style="flex:1;min-width:96px;text-align:center;padding:14px 8px;'
            f'background:#151922;border:1px solid #262D3A;border-radius:12px;">'
            f'<div style="font-size:1.5rem;filter:drop-shadow(0 0 14px {emo_color(e)}77);">'
            f'{emotion_emojis.get(e,"🎭")}</div>'
            f'<div style="font-size:.7rem;letter-spacing:1px;text-transform:uppercase;'
            f'font-family:ui-monospace,Consolas,monospace;color:{emo_color(e)};'
            f'margin-top:6px;">{e}</div></div>'
            for e in ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"])
        st.markdown(
            '<div style="margin-top:18px;">'
            '<h3 style="font-size:.68rem;font-weight:700;color:#8A94A6;'
            'text-transform:uppercase;letter-spacing:1.4px;font-family:ui-monospace,'
            'Consolas,monospace;margin:0 0 10px;">Detectable classes</h3>'
            f'<div style="display:flex;gap:10px;flex-wrap:wrap;">{chips}</div></div>',
            unsafe_allow_html=True)

    if uploaded_file is not None:
        st.audio(uploaded_file)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("🔍 Analyzing emotion..."):
            emotion, confidence, all_confidences, audio, sr = predict_emotion(tmp_path, model, le)

        os.unlink(tmp_path)

        if emotion:
            show_results(emotion, confidence, all_confidences, audio, sr, uploaded_file.name)

# ════════════════════════════════
# PAGE 2 — Live Recording
# ════════════════════════════════
elif page == "🎤 Live Recording":
    # Session state initialize
    if "recording_key" not in st.session_state:
        st.session_state.recording_key = 0
    if "show_result" not in st.session_state:
        st.session_state.show_result = False

    # ── STEP 1: Recording UI ──
    if not st.session_state.show_result:
        st.markdown("""
        <div class="card" style="text-align:center;padding:30px 22px;
             border-color:#22D3EE44;box-shadow:0 0 0 1px #22D3EE18,0 14px 44px -16px #22D3EE55;">
          <div style="font-size:2.6rem;line-height:1;
               filter:drop-shadow(0 0 22px #22D3EE99);">🎙️</div>
          <div style="font-size:1.15rem;font-weight:700;margin-top:10px;letter-spacing:.5px;
               text-transform:uppercase;">Speak naturally</div>
          <div style="color:#8A94A6;font-size:.83rem;margin-top:6px;
               font-family:ui-monospace,Consolas,monospace;">
            allow mic &nbsp;→&nbsp; record &nbsp;→&nbsp; stop &nbsp;→&nbsp; analyse
          </div>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            audio_bytes = st.audio_input(
                "Press to start recording",
                key=f"recorder_{st.session_state.recording_key}",
                label_visibility="collapsed"
            )

        if audio_bytes is not None:
            st.markdown("##### Your recording")
            st.audio(audio_bytes, format="audio/wav")

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔍 Analyze Emotion", type="primary", width="stretch"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_bytes.getvalue())
                        tmp_path = tmp.name

                    with st.spinner("🔍 Analyzing your voice..."):
                        emotion, confidence, all_confidences, audio, sr = predict_emotion(
                            tmp_path, model, le
                        )
                    os.unlink(tmp_path)

                    if emotion:
                        # Result session mein save karo
                        st.session_state.last_emotion       = emotion
                        st.session_state.last_confidence    = confidence
                        st.session_state.last_all_conf      = all_confidences
                        st.session_state.last_audio         = audio
                        st.session_state.last_sr            = sr
                        st.session_state.show_result        = True
                        st.rerun()

    # ── STEP 2: Result UI ──
    else:
        show_results(
            st.session_state.last_emotion,
            st.session_state.last_confidence,
            st.session_state.last_all_conf,
            st.session_state.last_audio,
            st.session_state.last_sr,
            "live_recording.wav"
        )

        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🎙️ New Recording", type="primary", width="stretch"):
                st.session_state.recording_key += 1
                st.session_state.show_result = False
                st.rerun()

# ════════════════════════════════
# PAGE 3 — About Emotions
# ════════════════════════════════
elif page == "📊 About Emotions":
    st.header("Emotion Categories")

    emotions_info = {
        "😠 Angry":   "High energy, aggressive, tense speech patterns",
        "🤢 Disgust":  "Disapproving, contemptuous tone of voice",
        "😨 Fear":    "Anxious, shaky, frightened speech",
        "😊 Happy":   "Joyful, positive, energetic speech",
        "😐 Neutral": "Calm, flat speech with no strong emotion",
        "😲 Surprise":"Astonished, unexpected reaction in voice",
        "😢 Sad":     "Low energy, slow, sorrowful speech"
    }

    for emotion, desc in emotions_info.items():
        st.markdown(f"**{emotion}** — {desc}")

    st.markdown("---")
    st.markdown("### Datasets Used")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🗂️ TESS — Toronto Emotional Speech Set**")
        st.markdown("- 2 Actresses (Young and Old)\n- 2,800 audio samples\n- English (Canadian)")
        st.markdown("---")
        st.markdown("**🗂️ RAVDESS**")
        st.markdown("- 24 Actors (Male & Female)\n- 1,440 audio samples\n- English (N. American)")
    with col2:
        st.markdown("**🗂️ CREMA-D** *(subset used)*")
        st.markdown("- 32 Actors\n- 2,585 audio samples\n- English (diverse)")
        st.markdown("---")
        st.markdown("**🗂️ SAVEE**")
        st.markdown("- 4 Male Speakers\n- 480 audio samples\n- English (British)")

    st.markdown("---")
    st.success(
        "**Integrated corpus: 7,305 clips from 62 speakers** — "
        "5,844 training clips (23,376 after augmentation) and 1,461 held-out test clips."
    )

# ════════════════════════════════
# PAGE 4 — About Project
# ════════════════════════════════
elif page == "ℹ️ About Project":
    st.header("About This Project")
    st.markdown("""
    **Project Title:** Emotion Recognition from Speech Using Machine Learning

    **Technology Stack:**
    - Python 3.11
    - TensorFlow 2.x
    - Librosa
    - Streamlit
    - Scikit-learn

    **How it works:**
    1. Audio file is uploaded or recorded via microphone
    2. Audio is resampled to 22,050 Hz, silence-trimmed and amplitude-normalized
    3. A fused 284-dimensional feature vector is extracted
    4. Features are standardized using the StandardScaler fitted during training
    5. A deep neural network predicts the emotion with a confidence score

    **Feature Vector (284 dimensions):**
    - 80 — MFCC (40 coefficients, mean + std)
    - 80 — Delta MFCC (first-order temporal derivative)
    - 80 — Delta-Delta MFCC (second-order temporal derivative)
    - 12 — Chroma  |  20 — Mel spectrogram  |  7 — Spectral contrast
    - 1 — ZCR  |  1 — RMS energy  |  3 — Spectral centroid, bandwidth, roll-off

    **Model Architecture:**
    - Input Layer: 284 fused acoustic features
    - Dense Layer: 512 neurons + BatchNorm + Dropout 0.4
    - Dense Layer: 256 neurons + BatchNorm + Dropout 0.4
    - Dense Layer: 128 neurons + Dropout 0.3
    - Dense Layer: 64 neurons
    - Output Layer: 7 emotions (Softmax)
    - Trainable parameters: 321,927
    - Loss: class-balanced focal loss (γ = 2) | Optimizer: Adam (lr = 0.001)

    **Datasets Used:**
    - TESS Toronto Emotional Speech Set — 2,800 clips, 2 speakers
    - RAVDESS Emotional Speech — 1,440 clips, 24 speakers
    - CREMA-D (subset) — 2,585 clips, 32 speakers
    - SAVEE Database — 480 clips, 4 speakers
    - Integrated: 7,305 clips from 62 speakers
    - Training: 5,844 clips → 23,376 after augmentation | Test: 1,461 clips

    **Results (held-out test set of 1,461 unseen clips):**
    - Overall accuracy: 76.73% | Macro F1: 77.16%
    - 5-fold cross-validation: 74.99% ± 0.94%
    - Per-class ROC-AUC: 0.944 – 0.992 (micro-average 0.965)
    - Inference time: 3.27 ms per clip
    - Per-dataset: TESS 99.82% | RAVDESS 75.35% | CREMA-D 57.46% | SAVEE 56.38%

    The combined figure is reported rather than the TESS-only figure because it
    reflects performance on unfamiliar voices, which is what a real user experiences.

    **Privacy:** all processing happens locally on this machine. No audio,
    feature vector or prediction is ever sent to an external server.

    **Final Year Project — Department of Computer Science**
    **University of Southern Punjab, Multan**
    """)

# ════════════════════════════════
# PAGE 5 — Session History
# ════════════════════════════════
elif page == "📋 Session History":
    st.subheader("Session History")
    history = st.session_state.get("history", [])

    if not history:
        st.info("Nothing analysed yet. Upload a file or record your voice to get started.")
    else:
        counts = {}
        for entry in history:
            counts[entry["emotion"]] = counts.get(entry["emotion"], 0) + 1
        top = max(counts, key=counts.get)

        m1, m2, m3 = st.columns(3)
        m1.metric("Analyses", len(history))
        m2.metric("Distinct emotions", len(counts))
        m3.metric("Most frequent", top)

        st.markdown("##### Breakdown")
        rows = ""
        for em, n in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            pct = 100 * n / len(history)
            name = em.split(" ", 1)[-1] if " " in em else em
            rows += (
                f'<div class="row">'
                f'  <div class="lbl">{em}</div>'
                f'  <div class="track"><div class="fill" '
                f'       style="width:{pct:.1f}%;background:{emo_color(name)};"></div></div>'
                f'  <div class="val">{n}</div>'
                f'</div>'
            )
        st.markdown(f'<div class="card">{rows}</div>', unsafe_allow_html=True)

        st.markdown("##### Detailed log")
        st.dataframe(
            [{"#": len(history) - i, "Time": e["time"], "File": e["file"],
              "Emotion": e["emotion"], "Confidence": e["confidence"]}
             for i, e in enumerate(reversed(history))],
            width="stretch", hide_index=True,
        )

        if st.button("🗑️ Clear history"):
            st.session_state.history = []
            st.session_state.pop("last_logged", None)
            st.rerun()

# ════════════════════════════════
# Footer
# ════════════════════════════════
_fb = feedback_count()
st.markdown(
    '<div class="foot">'
    '<span>&#128274; all audio processed locally &mdash; nothing is uploaded</span>'
    f'<span>{model.count_params():,} parameters &middot; {N_FEATURES} features</span>'
    '<span>RAVDESS &middot; TESS &middot; SAVEE &middot; CREMA-D &mdash; 7,305 clips, 62 speakers</span>'
    + (f'<span>&#128451; {_fb} feedback sample(s) queued for retraining</span>' if _fb else '')
    + '</div>',
    unsafe_allow_html=True)
