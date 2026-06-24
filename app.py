import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pickle
import tempfile
import os
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io.wavfile import write as write_wav

# Model aur label encoder load karo
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("emotion_model.h5")
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le
# Scaler load karo
@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as f:
        return pickle.load(f)

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=22050)

    # MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    mfcc_std = np.std(mfcc.T, axis=0)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_mean = np.mean(mel.T, axis=0)[:20]

    # ZCR aur RMS
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
    rms = np.mean(librosa.feature.rms(y=audio))

    features = np.concatenate([
        mfcc_mean, mfcc_std, chroma_mean, mel_mean, [zcr, rms]
    ])
    return features, audio, sr

def predict_emotion(file_path, model, le):
    features, audio, sr = extract_features(file_path)
    scaler = load_scaler()
    features = scaler.transform(np.array([features]))
    prediction = model.predict(features, verbose=0)
    emotion_index = np.argmax(prediction)
    emotion = le.classes_[emotion_index]
    confidence = prediction[0][emotion_index] * 100
    all_confidences = {le.classes_[i]: prediction[0][i] * 100
                       for i in range(len(le.classes_))}
    return emotion, confidence, all_confidences, audio, sr

emotion_emojis = {
    "angry": "😠", "disgust": "🤢", "fear": "😨",
    "happy": "😊", "neutral": "😐", "sad": "😢",
    "pleasant_surprised": "😲", "ps": "😲"
}

def show_results(emotion, confidence, all_confidences, audio, sr, filename):
    st.markdown("---")
    emoji = emotion_emojis.get(emotion.lower(), "🎭")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Predicted Emotion")
        st.markdown(f"# {emoji} {emotion.upper()}")
        st.markdown(f"**Confidence: {confidence:.1f}%**")

        st.markdown("**Top 3 Predictions:**")
        sorted_conf = sorted(all_confidences.items(),
                             key=lambda x: x[1], reverse=True)[:3]
        for em, conf in sorted_conf:
            st.markdown(f"- {emotion_emojis.get(em.lower(), '🎭')} {em}: {conf:.1f}%")

    with col2:
        st.markdown("### Confidence Chart")
        emotions_list = list(all_confidences.keys())
        scores_list = list(all_confidences.values())
        fig, ax = plt.subplots(figsize=(6, 3))
        colors = ['#FF6B6B' if e == emotion else '#4ECDC4'
                  for e in emotions_list]
        ax.barh(emotions_list, scores_list, color=colors)
        ax.set_xlabel("Confidence (%)")
        ax.set_xlim(0, 100)
        plt.tight_layout()
        st.pyplot(fig)

    # Waveform
    st.markdown("### Audio Waveform")
    fig2, ax2 = plt.subplots(figsize=(10, 2))
    ax2.plot(np.linspace(0, len(audio) / sr, len(audio)),
             audio, color='#4ECDC4')
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Amplitude")
    plt.tight_layout()
    st.pyplot(fig2)

    # Session history
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({
        "file": filename,
        "emotion": f"{emoji} {emotion}",
        "confidence": f"{confidence:.1f}%"
    })

# ── Page Config ──
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
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "🎵 Upload Audio",
    "🎤 Live Recording",
    "📊 About Emotions",
    "ℹ️ About Project",
    "📋 Session History"
])

# ════════════════════════════════
# PAGE 1 — Upload Audio
# ════════════════════════════════
if page == "🎵 Upload Audio":
    st.header("Upload Audio File")
    st.info("Upload a WAV, MP3 or FLAC audio file to detect the emotion.")

    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "flac"]
    )

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.audio(uploaded_file)

        with st.spinner("Analyzing emotion..."):
            emotion, confidence, all_confidences, audio, sr = predict_emotion(
                tmp_path, model, le
            )
        os.unlink(tmp_path)
        show_results(emotion, confidence, all_confidences,
                     audio, sr, uploaded_file.name)

# ════════════════════════════════
# PAGE 2 — Live Recording
# ════════════════════════════════
elif page == "🎤 Live Recording":
    st.header("Live Microphone Recording")
    st.info("Record your voice and the system will detect your emotion!")

    duration = st.slider("Recording Duration (seconds)",
                         min_value=3, max_value=10, value=5)

    st.markdown("### Instructions:")
    st.markdown("- 🎤 Allow  Mic when open browser")
    st.markdown("- 🗣️ Speak clearity")
    st.markdown("- ⏱️ speak After Countdown")

    if st.button("🎤 Start Recording", type="primary"):
        import time

        # Countdown
        countdown_placeholder = st.empty()
        for i in range(3, 0, -1):
            countdown_placeholder.markdown(f"## ⏳ Recording starts in... {i}")
            time.sleep(1)
        countdown_placeholder.markdown("## 🔴 RECORDING... Speak Now!")

        # Record
        sample_rate = 22050
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )

        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i in range(duration):
            time.sleep(1)
            progress = (i + 1) / duration
            progress_bar.progress(progress)
            status_text.markdown(f"⏱️ {i+1}/{duration} seconds recorded...")

        sd.wait()
        countdown_placeholder.markdown("## ✅ Recording Complete!")
        status_text.empty()
        progress_bar.empty()

        st.success("Processing your voice...")

        # Save
        tmp_path = tempfile.mktemp(suffix=".wav")
        audio_data = recording.flatten()
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        write_wav(tmp_path, sample_rate, audio_data)

        # Playback
        st.markdown("**Your Recording:**")
        with open(tmp_path, "rb") as f:
            st.audio(f.read(), format="audio/wav")

        with st.spinner("Analyzing emotion..."):
            emotion, confidence, all_confidences, audio, sr = predict_emotion(
                tmp_path, model, le
            )
        os.unlink(tmp_path)
        show_results(emotion, confidence, all_confidences,
                     audio, sr, "live_recording.wav")

# ════════════════════════════════
# PAGE 3 — About Emotions
# ════════════════════════════════
elif page == "📊 About Emotions":
    st.header("Emotion Categories")

    emotions_info = {
        "😠 Angry": "High energy, aggressive, tense speech",
        "🤢 Disgust": "Disapproving, contemptuous tone",
        "😨 Fear": "Anxious, shaky, frightened speech",
        "😊 Happy": "Joyful, positive, energetic speech",
        "😐 Neutral": "Calm, flat speech with no strong emotion",
        "😲 Surprise": "Astonished, unexpected reaction",
        "😢 Sad": "Low energy, slow, sorrowful speech"
    }

    for emotion, desc in emotions_info.items():
        st.markdown(f"**{emotion}** — {desc}")

    st.markdown("---")
    st.markdown("### Datasets Used")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🗂️ TESS — Toronto Emotional Speech Set**")
        st.markdown("- 2 Actresses (Young and Old)")
        st.markdown("- 2,800 audio samples")
        st.markdown("- Language: English")

        st.markdown("---")
        st.markdown("**🗂️ RAVDESS**")
        st.markdown("- 24 Actors (Male & Female)")
        st.markdown("- 1,440 audio samples")
        st.markdown("- Language: English")

    with col2:
        st.markdown("**🗂️ CREMA-D**")
        st.markdown("- 91 Actors")
        st.markdown("- 7,442 audio samples")
        st.markdown("- Language: English")

        st.markdown("---")
        st.markdown("**🗂️ SAVEE**")
        st.markdown("- 4 Male Speakers")
        st.markdown("- 480 audio samples")
        st.markdown("- Language: English")

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
    2. MFCC features are extracted using Librosa
    3. Features are passed to trained Neural Network
    4. Model predicts the emotion with confidence score

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
    - Total: 12,000+ samples | After Augmentation: 24,000+

    **Final Year Project — Department of Computer Science**
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
        for i, entry in enumerate(st.session_state.history):
            st.markdown(f"**{i+1}.** File: `{entry['file']}` | "
                        f"Emotion: {entry['emotion']} | "
                        f"Confidence: {entry['confidence']}")

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.success("History cleared!")
            st.rerun()