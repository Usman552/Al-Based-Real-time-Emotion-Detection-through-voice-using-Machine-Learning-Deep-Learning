"""
Shared helpers: dataset index (with speaker IDs),
the 284-D handcrafted feature extractor (identical to train_model.py) and the
waveform augmentations.
"""
import os
import numpy as np
import pandas as pd
import librosa

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.environ.get("SER_CACHE", os.path.join(BASE_DIR, "cache_experiments"))
RESULTS = os.environ.get("SER_RESULTS", os.path.join(BASE_DIR, "results", "experiments"))
os.makedirs(CACHE, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

TESS_PATH = os.path.join(BASE_DIR, "TESS Toronto emotional speech set data")
RAVDESS_PATH = os.path.join(BASE_DIR, "RAVDESS")
CREMAD_PATH = os.path.join(BASE_DIR, "CREMAD", "AudioWAV")
SAVEE_PATH = os.path.join(BASE_DIR, "SAVEE", "AudioData")

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
CORPORA = ["RAVDESS", "TESS", "SAVEE", "CREMA-D"]

# Native label -> unified label, per corpus (reported as a table in the paper).
RAV_MAP = {'01': 'neutral', '02': 'neutral', '03': 'happy', '04': 'sad',
           '05': 'angry', '06': 'fear', '07': 'disgust', '08': 'surprise'}
CRE_MAP = {'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fear', 'HAP': 'happy',
           'NEU': 'neutral', 'SAD': 'sad'}
SAV_MAP = {'a': 'angry', 'd': 'disgust', 'f': 'fear', 'h': 'happy',
           'n': 'neutral', 'sa': 'sad', 'su': 'surprise'}


def _cremad_genders():
    """Actor sex from CREMA-D's VideoDemographics.csv when available."""
    for p in [os.path.join(BASE_DIR, "CREMAD", "VideoDemographics.csv"),
              os.path.join(BASE_DIR, "CREMAD", "VideoDemographics.CSV")]:
        if os.path.isfile(p):
            d = pd.read_csv(p)
            return {str(a): ("F" if str(s).lower().startswith("f") else "M")
                    for a, s in zip(d["ActorID"], d["Sex"])}
    return {}


def build_index():
    """One row per utterance: path, label, corpus, speaker, gender."""
    out = os.path.join(CACHE, "meta.csv")
    if os.path.isfile(out):
        return pd.read_csv(out)
    rows = []
    for folder in sorted(os.listdir(TESS_PATH)):
        fp = os.path.join(TESS_PATH, folder)
        if not os.path.isdir(fp) or "_" not in folder:
            continue
        spk, emo = folder.split("_")[0], folder.split("_")[-1].lower()
        if emo in ("surprise", "surprised"):
            emo = "surprise"
        if emo not in CLASSES:
            continue
        for f in sorted(os.listdir(fp)):
            if f.endswith(".wav"):
                rows.append((os.path.join(fp, f), emo, "TESS", f"TESS_{spk}", "F"))
    for a in sorted(os.listdir(RAVDESS_PATH)):
        ap = os.path.join(RAVDESS_PATH, a)
        if not os.path.isdir(ap):
            continue
        for f in sorted(os.listdir(ap)):
            p = f.replace(".wav", "").split("-")
            if f.endswith(".wav") and len(p) >= 7 and p[2] in RAV_MAP:
                actor = int(p[6])
                rows.append((os.path.join(ap, f), RAV_MAP[p[2]], "RAVDESS",
                             f"RAV_{actor:02d}", "M" if actor % 2 else "F"))
    cg = _cremad_genders()
    for f in sorted(os.listdir(CREMAD_PATH)):
        p = f.split("_")
        if f.endswith(".wav") and len(p) >= 3 and p[2] in CRE_MAP:
            rows.append((os.path.join(CREMAD_PATH, f), CRE_MAP[p[2]], "CREMA-D",
                         f"CRE_{p[0]}", cg.get(p[0], "?")))
    for sp in sorted(os.listdir(SAVEE_PATH)):
        spp = os.path.join(SAVEE_PATH, sp)
        if not os.path.isdir(spp):
            continue
        for f in sorted(os.listdir(spp)):
            if f.endswith(".wav"):
                nm = f.replace(".wav", "")
                emo = SAV_MAP.get(nm[:2]) or SAV_MAP.get(nm[:1])
                if emo:
                    rows.append((os.path.join(spp, f), emo, "SAVEE", f"SAV_{sp}", "M"))
    meta = pd.DataFrame(rows, columns=["path", "label", "corpus", "speaker", "gender"])
    meta["path"] = meta["path"].map(lambda p: os.path.relpath(p, BASE_DIR))
    meta.to_csv(out, index=False)
    return meta


def abspath(rel):
    return os.path.join(BASE_DIR, rel)


# ─── 284-D handcrafted features (same as train_model.py) ─────────────
def preprocess(audio, sr):
    audio, _ = librosa.effects.trim(audio, top_db=25)
    if len(audio) < sr // 2:
        audio = np.pad(audio, (0, sr // 2 - len(audio)))
    return librosa.util.normalize(audio)


def features_from_audio(audio, sr=22050):
    audio = preprocess(audio, sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)

    def ms(x):
        return np.concatenate([x.mean(axis=1), x.std(axis=1)])
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr).mean(axis=1)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr).mean(axis=1)[:20]
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr).mean(axis=1)
    zcr = np.array([librosa.feature.zero_crossing_rate(y=audio).mean()])
    rms = np.array([librosa.feature.rms(y=audio).mean()])
    spec = np.array([librosa.feature.spectral_centroid(y=audio, sr=sr).mean(),
                     librosa.feature.spectral_bandwidth(y=audio, sr=sr).mean(),
                     librosa.feature.spectral_rolloff(y=audio, sr=sr).mean()])
    return np.concatenate([ms(mfcc), ms(d1), ms(d2), chroma, mel, contrast, zcr, rms, spec])


# Column groups of the 284-D vector, used by the feature ablation.
HAND_GROUPS = {
    "mfcc": slice(0, 80), "delta": slice(80, 160), "delta2": slice(160, 240),
    "chroma": slice(240, 252), "mel": slice(252, 272), "contrast": slice(272, 279),
    "zcr": slice(279, 280), "rms": slice(280, 281), "shape": slice(281, 284),
}

AUGS = ["noise", "pitch", "stretch"]


def augment_audio(audio, sr, kind, rng):
    if kind == "noise":
        return audio + 0.05 * rng.standard_normal(len(audio))
    if kind == "pitch":
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=int(rng.choice([-2, -1, 1, 2])))
    if kind == "stretch":
        return librosa.effects.time_stretch(audio, rate=float(rng.uniform(0.9, 1.1)))
    return audio
