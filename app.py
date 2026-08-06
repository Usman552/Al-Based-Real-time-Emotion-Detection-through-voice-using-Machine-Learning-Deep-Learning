import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pickle
import tempfile
import os
import matplotlib.pyplot as plt
from datetime import datetime

# ── Model Load ──
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("emotion_model.h5")
        with open("label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        return model, le
    except FileNotFoundError:
        st.error("❌ Model files not found! Please make sure emotion_model.h5, label_encoder.pkl are present.")
        st.stop()

@st.cache_resource
def load_scaler():
    try:
        with open("scaler.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("❌ scaler.pkl not found!")
        st.stop()

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
        return features, audio, sr
    except Exception as e:
        st.error(f"Feature extraction error: {e}")
        return None, None, None



# ── Prediction ──
def predict_emotion(file_path, model, le):
    features, audio, sr = extract_features(file_path)
    if features is None:
        return None, None, None, None, None

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
    st.markdown("---")
    emoji = emotion_emojis.get(emotion.lower(), "🎭")

    # Low confidence warning
    if confidence < 50:
        st.warning(f"⚠️ Model is not very confident ({confidence:.1f}%). Result may not be accurate.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Predicted Emotion")
        st.markdown(f"# {emoji} {emotion.upper()}")

        if confidence >= 75:
            st.success(f"Confidence: {confidence:.1f}%")
        elif confidence >= 50:
            st.warning(f"Confidence: {confidence:.1f}%")
        else:
            st.error(f"Confidence: {confidence:.1f}% (Low)")

        st.markdown("**Top 3 Predictions:**")
        sorted_conf = sorted(all_confidences.items(), key=lambda x: x[1], reverse=True)[:3]
        for em, conf in sorted_conf:
            bar = "█" * int(conf / 10) + "░" * (10 - int(conf / 10))
            st.markdown(f"{emotion_emojis.get(em.lower(), '🎭')} **{em}** `{bar}` {conf:.1f}%")

    with col2:
        st.markdown("### Confidence Chart")
        emotions_list = list(all_confidences.keys())
        scores_list   = list(all_confidences.values())
        fig, ax = plt.subplots(figsize=(6, 3))
        colors = ['#FF6B6B' if e == emotion else '#4ECDC4' for e in emotions_list]
        bars = ax.barh(emotions_list, scores_list, color=colors)
        ax.set_xlabel("Confidence (%)")
        ax.set_xlim(0, 100)
        for bar, score in zip(bars, scores_list):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{score:.1f}%', va='center', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)  # memory leak fix

    # Waveform
    st.markdown("### Audio Waveform")
    fig2, ax2 = plt.subplots(figsize=(10, 2))
    time_axis = np.linspace(0, len(audio) / sr, len(audio))
    ax2.plot(time_axis, audio, color='#4ECDC4', linewidth=0.5)
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title(f"Waveform — {filename}")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)  # memory leak fix

    # Session history
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({
        "file":       filename,
        "emotion":    f"{emoji} {emotion}",
        "confidence": f"{confidence:.1f}%",
        "time":       datetime.now().strftime("%H:%M:%S")
    })

# ════════════════════════════════
# Page Config
# ════════════════════════════════
st.set_page_config(
    page_title="Emotion Recognition from Speech",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Emotion Recognition from Speech")
st.markdown("### Using Machine Learning | Final Year Project")
st.markdown("---")

model, le = load_model()

# ── Sidebar ──
st.sidebar.title("🎙️ Navigation")
page = st.sidebar.radio("Go to", [
    "🎵 Upload Audio",
    "🎤 Live Recording",
    "📊 About Emotions",
    "ℹ️ About Project",
    "📋 Session History"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info**")
st.sidebar.info(f"Classes: {len(le.classes_)} emotions\nFeatures: 114")

# ════════════════════════════════
# PAGE 1 — Upload Audio
# ════════════════════════════════
if page == "🎵 Upload Audio":
    st.header("📂 Upload Audio File")
    st.info("Upload a WAV, MP3 or FLAC audio file to detect the emotion.")

    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac"])

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
    st.header("🎤 Live Microphone Recording")

    # Session state initialize
    if "recording_key" not in st.session_state:
        st.session_state.recording_key = 0
    if "show_result" not in st.session_state:
        st.session_state.show_result = False

    # ── STEP 1: Recording UI ──
    if not st.session_state.show_result:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            border: 1px solid #0f3460;
            border-radius: 16px;
            padding: 32px;
            text-align: center;
            margin-bottom: 24px;
        ">
            <h2 style="color:#e94560; margin:0 0 8px;">🎙️ Voice Emotion Analyzer</h2>
            <p style="color:#a0a0b0; margin:0;">Speak naturally — our AI will detect your emotion</p>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            audio_bytes = st.audio_input(
                "Press to start recording",
                key=f"recorder_{st.session_state.recording_key}",
                label_visibility="collapsed"
            )

        st.markdown("")
        st.markdown(
            "<p style='text-align:center; color:#888; font-size:13px;'>"
            "🎤 Allow mic access → Speak → Stop → Analyze"
            "</p>",
            unsafe_allow_html=True
        )

        if audio_bytes is not None:
            st.markdown("---")
            st.markdown("#### 🎧 Your Recording")
            st.audio(audio_bytes, format="audio/wav")
            st.markdown("")

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔍 Analyze Emotion", type="primary", use_container_width=True):
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
            if st.button("🎙️ New Recording", type="primary", use_container_width=True):
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
        st.markdown("- 2 Actresses (Young and Old)\n- 2,800 audio samples\n- Language: English")
        st.markdown("---")
        st.markdown("**🗂️ RAVDESS**")
        st.markdown("- 24 Actors (Male & Female)\n- 1,440 audio samples\n- Language: English")
    with col2:
        st.markdown("**🗂️ CREMA-D**")
        st.markdown("- 91 Actors\n- 7,442 audio samples\n- Language: English")
        st.markdown("---")
        st.markdown("**🗂️ SAVEE**")
        st.markdown("- 4 Male Speakers\n- 480 audio samples\n- Language: English")

    st.markdown("---")
    st.success("**Total Training Samples: ~12,000+ (with augmentation: ~24,000+)**")

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
    2. MFCC, Chroma, Mel Spectrogram, ZCR, RMS features are extracted
    3. Features are normalized using StandardScaler
    4. Neural Network predicts the emotion with confidence score

    **Model Architecture:**
    - Input Layer: 114 features (MFCC + Chroma + Mel + ZCR + RMS)
    - Dense Layer: 512 neurons + BatchNorm + Dropout 0.4
    - Dense Layer: 256 neurons + BatchNorm + Dropout 0.4
    - Dense Layer: 128 neurons + Dropout 0.3
    - Dense Layer: 64 neurons
    - Output Layer: 7 emotions (Softmax)

    **Datasets Used:**
    - TESS Toronto Emotional Speech Set (2,800 samples)
    - RAVDESS Emotional Speech (1,440 samples)
    - CREMA-D (7,442 samples)
    - SAVEE Database (480 samples)
    - Total: ~12,000 samples | After Augmentation: ~24,000+

    **Final Year Project — Department of Computer Science**
    **University of Southern Punjab, Multan**
    """)

# ════════════════════════════════
# PAGE 5 — Session History
# ════════════════════════════════
elif page == "📋 Session History":
    st.header("Session History")

    if "history" not in st.session_state or len(st.session_state.history) == 0:
        st.info("No analysis done yet. Upload or record audio first!")
    else:
        st.markdown(f"**Total Analyses: {len(st.session_state.history)}**")
        st.markdown("---")

        # Emotion count summary
        emotion_counts = {}
        for entry in st.session_state.history:
            em = entry["emotion"]
            emotion_counts[em] = emotion_counts.get(em, 0) + 1

        st.markdown("**Summary:**")
        for em, count in emotion_counts.items():
            st.markdown(f"- {em}: {count} time(s)")

        st.markdown("---")
        st.markdown("**Detailed History:**")
        for i, entry in enumerate(reversed(st.session_state.history)):
            st.markdown(
                f"**{len(st.session_state.history)-i}.** "
                f"`{entry['time']}` | "
                f"File: `{entry['file']}` | "
                f"Emotion: {entry['emotion']} | "
                f"Confidence: {entry['confidence']}"
            )

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.success("History cleared!")
            st.rerun()