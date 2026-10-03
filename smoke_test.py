"""
smoke_test.py — quick health check for the trained emotion-recognition pipeline.

Run this after cloning, after retraining, or before a demo:

    python smoke_test.py

It verifies that the model, label encoder and scaler load correctly, that the
feature extractor still produces exactly 284 dimensions, and that predictions on
a handful of known-label clips come out right. Exits with status 1 if anything
is wrong, so it can also be used in CI.
"""
import os
import sys
import glob
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import librosa
import pickle
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
N_FEATURES = 284

failures = []


def check(condition, message):
    if condition:
        print(f"  PASS  {message}")
    else:
        print(f"  FAIL  {message}")
        failures.append(message)
    return condition


def extract_features(file_path):
    """Must stay identical to extract_features() in app.py and train_model.py."""
    audio, sr = librosa.load(file_path, sr=22050)
    audio, _ = librosa.effects.trim(audio, top_db=25)
    if len(audio) < sr // 2:
        audio = np.pad(audio, (0, sr // 2 - len(audio)))
    audio = librosa.util.normalize(audio)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    ms = lambda x: np.concatenate([x.mean(axis=1), x.std(axis=1)])

    return np.concatenate([
        ms(mfcc), ms(d1), ms(d2),
        librosa.feature.chroma_stft(y=audio, sr=sr).mean(axis=1),
        librosa.feature.melspectrogram(y=audio, sr=sr).mean(axis=1)[:20],
        librosa.feature.spectral_contrast(y=audio, sr=sr).mean(axis=1),
        [librosa.feature.zero_crossing_rate(y=audio).mean()],
        [librosa.feature.rms(y=audio).mean()],
        [librosa.feature.spectral_centroid(y=audio, sr=sr).mean(),
         librosa.feature.spectral_bandwidth(y=audio, sr=sr).mean(),
         librosa.feature.spectral_rolloff(y=audio, sr=sr).mean()],
    ])


def sample_clips():
    """A few clips with labels recoverable from the file name."""
    clips = []
    cremad = {"HAP": "happy", "ANG": "angry", "SAD": "sad", "NEU": "neutral"}
    for code, label in cremad.items():
        hits = sorted(glob.glob(os.path.join(
            BASE_DIR, "CREMAD", "AudioWAV", f"*_DFA_{code}_XX.wav")))
        if hits:
            clips.append((hits[0], label))
    for folder, label in [("OAF_happy", "happy"), ("YAF_angry", "angry")]:
        hits = sorted(glob.glob(os.path.join(
            BASE_DIR, "TESS Toronto emotional speech set data", folder, "*.wav")))
        if hits:
            clips.append((hits[0], label))
    hits = sorted(glob.glob(os.path.join(BASE_DIR, "RAVDESS", "Actor_01", "*-05-*.wav")))
    if hits:
        clips.append((hits[0], "angry"))
    return clips


def main():
    print("\n=== 1. Artefacts present ===")
    for name in ["emotion_model.h5", "label_encoder.pkl", "scaler.pkl"]:
        check(os.path.exists(os.path.join(BASE_DIR, name)), f"{name} found")
    if failures:
        print("\nCannot continue without the model artefacts.")
        return 1

    print("\n=== 2. Artefacts load ===")
    # compile=False because the model was trained with a custom focal loss.
    model = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "emotion_model.h5"), compile=False)
    le = pickle.load(open(os.path.join(BASE_DIR, "label_encoder.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
    check(True, f"model loaded ({model.count_params():,} parameters)")
    check(model.input_shape[-1] == N_FEATURES,
          f"model input is {model.input_shape[-1]}-D (expected {N_FEATURES})")
    check(model.output_shape[-1] == len(le.classes_),
          f"model outputs {model.output_shape[-1]} classes, encoder has {len(le.classes_)}")
    check(scaler.n_features_in_ == N_FEATURES,
          f"scaler expects {scaler.n_features_in_} features (expected {N_FEATURES})")
    print(f"        classes: {', '.join(le.classes_)}")

    clips = sample_clips()
    if not clips:
        print("\n=== 3. Predictions ===")
        print("  SKIP  no dataset folders found — model artefacts are fine, "
              "but end-to-end prediction was not exercised.")
        print("\nRESULT:", "FAILED" if failures else "PASSED (artefacts only)")
        return 1 if failures else 0

    print("\n=== 3. Feature extraction ===")
    vectors = []
    for path, label in clips:
        f = extract_features(path)
        vectors.append((f, label, path))
        if not check(f.shape[0] == N_FEATURES,
                     f"{os.path.basename(path)} -> {f.shape[0]}-D"):
            break
    check(np.all(np.isfinite(np.array([v[0] for v in vectors]))),
          "no NaN or infinite values in any feature vector")

    print("\n=== 4. End-to-end predictions ===")
    correct, times = 0, []
    for f, expected, path in vectors:
        t0 = time.time()
        probs = model.predict(scaler.transform([f]), verbose=0)[0]
        times.append((time.time() - t0) * 1000)
        predicted = le.classes_[probs.argmax()]
        ok = predicted == expected
        correct += ok
        print(f"  {'OK  ' if ok else 'MISS'}  expected={expected:<8s} "
              f"predicted={predicted:<8s} conf={probs.max() * 100:5.1f}%   "
              f"{os.path.basename(path)}")
        if not np.isclose(probs.sum(), 1.0, atol=1e-4):
            failures.append(f"probabilities do not sum to 1 for {path}")

    print(f"\n        {correct}/{len(vectors)} correct, "
          f"mean inference {np.mean(times):.0f} ms/clip")
    check(correct >= len(vectors) * 0.6,
          f"at least 60% of sample clips classified correctly ({correct}/{len(vectors)})")

    print("\nRESULT:", "FAILED" if failures else "PASSED")
    if failures:
        for f in failures:
            print("  -", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
