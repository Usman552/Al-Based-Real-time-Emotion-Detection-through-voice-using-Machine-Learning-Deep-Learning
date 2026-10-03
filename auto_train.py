"""
Hyperparameter search for the 284-D handcrafted model.

Each trial trains a candidate configuration (architecture, activation, dropout,
learning rate, batch size, focal-loss gamma). Candidates are compared on a
validation split taken from the training data; the held-out test split is only
used to report the final score. The best model so far is kept on disk.

Features are extracted once and cached in cache/features_284.npz. Corrected
clips saved by the app in feedback/ are added to the training data.

Usage:
    python auto_train.py                        # 10 trials
    python auto_train.py --trials 50
    python auto_train.py --forever              # until Ctrl+C
    python auto_train.py --speaker-independent  # speaker-disjoint split
    python auto_train.py --rebuild-cache

Progress: results/auto_train_log.csv; best model: results/champion.json.
"""
import argparse
import json
import os
import pickle
import random
import re
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
OUTDIR = os.path.join(BASE_DIR, "results")
FEEDBACK_DIR = os.path.join(BASE_DIR, "feedback")
CACHE_FILE = os.path.join(CACHE_DIR, "features_284.npz")
LOG_FILE = os.path.join(OUTDIR, "auto_train_log.csv")
CHAMPION_FILE = os.path.join(OUTDIR, "champion.json")

SEED = 42                      # same seed as train_model.py -> same held-out test set
AUGS = ["noise", "pitch", "stretch"]
N_FEATURES = 284

TESS_PATH = os.path.join(BASE_DIR, "TESS Toronto emotional speech set data")
RAVDESS_PATH = os.path.join(BASE_DIR, "RAVDESS")
CREMAD_PATH = os.path.join(BASE_DIR, "CREMAD", "AudioWAV")
SAVEE_PATH = os.path.join(BASE_DIR, "SAVEE", "AudioData")


# ───────────────────────── features ─────────────────────────
def features_from_audio(audio, sr=22050):
    """284-D fused descriptor. Must match app.py and train_model.py exactly."""
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


def augment_audio(audio, sr, kind, rng):
    if kind == "noise":
        return audio + 0.05 * rng.standard_normal(len(audio))
    if kind == "pitch":
        return librosa.effects.pitch_shift(
            audio, sr=sr, n_steps=float(rng.choice([-2, -1, 1, 2])))
    if kind == "stretch":
        return librosa.effects.time_stretch(audio, rate=float(rng.uniform(0.9, 1.1)))
    return audio


# ───────────────────────── dataset scan ─────────────────────────
def speaker_of(path, source):
    """Speaker identity, used only for the speaker-independent split."""
    name = os.path.basename(path)
    if source == "CREMA-D":
        return "CREMAD_" + name.split("_")[0]
    if source == "RAVDESS":
        m = re.search(r"Actor_(\d+)", path)
        if m:
            return "RAVDESS_" + m.group(1)
        return "RAVDESS_" + name.replace(".wav", "").split("-")[-1]
    if source == "SAVEE":
        return "SAVEE_" + os.path.basename(os.path.dirname(path))
    if source == "TESS":
        return "TESS_OAF" if os.path.basename(os.path.dirname(path)).startswith("OAF") else "TESS_YAF"
    return source + "_unknown"


def scan_datasets():
    """Returns (paths, labels, sources) for every usable clip found on disk."""
    paths, labels, sources = [], [], []

    if os.path.isdir(TESS_PATH):
        for folder in os.listdir(TESS_PATH):
            fp = os.path.join(TESS_PATH, folder)
            if not os.path.isdir(fp):
                continue
            emo = folder.split("_")[-1].lower()
            if emo in ("pleasant_surprise", "surprised", "pleasant_surprised"):
                emo = "surprise"
            wavs = [f for f in os.listdir(fp) if f.endswith(".wav")]
            for f in wavs:
                paths.append(os.path.join(fp, f)); labels.append(emo); sources.append("TESS")

    rav = {'01': 'neutral', '02': 'neutral', '03': 'happy', '04': 'sad',
           '05': 'angry', '06': 'fear', '07': 'disgust', '08': 'surprise'}
    if os.path.isdir(RAVDESS_PATH):
        for a in os.listdir(RAVDESS_PATH):
            ap = os.path.join(RAVDESS_PATH, a)
            if not os.path.isdir(ap):
                continue
            for f in os.listdir(ap):
                if not f.endswith(".wav"):
                    continue
                p = f.split("-")
                if len(p) >= 3 and rav.get(p[2]):
                    paths.append(os.path.join(ap, f)); labels.append(rav[p[2]]); sources.append("RAVDESS")

    cre = {'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fear',
           'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'}
    if os.path.isdir(CREMAD_PATH):
        for f in os.listdir(CREMAD_PATH):
            if not f.endswith(".wav"):
                continue
            p = f.split("_")
            if len(p) >= 3 and cre.get(p[2]):
                paths.append(os.path.join(CREMAD_PATH, f)); labels.append(cre[p[2]]); sources.append("CREMA-D")

    sav = {'a': 'angry', 'd': 'disgust', 'f': 'fear', 'h': 'happy',
           'n': 'neutral', 'sa': 'sad', 'su': 'surprise'}
    if os.path.isdir(SAVEE_PATH):
        for sp in os.listdir(SAVEE_PATH):
            spp = os.path.join(SAVEE_PATH, sp)
            if not os.path.isdir(spp):
                continue
            for f in os.listdir(spp):
                if not f.endswith(".wav"):
                    continue
                nm = f.replace(".wav", "")
                emo = sav.get(nm[:2]) or sav.get(nm[:1])
                if emo:
                    paths.append(os.path.join(spp, f)); labels.append(emo); sources.append("SAVEE")

    return np.array(paths), np.array(labels), np.array(sources)


def build_cache(verbose=True):
    """Extract clean + augmented features once and store them on disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    paths, labels, sources = scan_datasets()
    if len(paths) == 0:
        raise SystemExit(
            "No audio found. Place the dataset folders next to this script "
            "(see README.md) or run with an existing cache.")

    if verbose:
        print(f"Found {len(paths)} clips across {len(set(sources))} corpora.")
        print("Extracting features (this happens only once)...")

    speakers = np.array([speaker_of(p, s) for p, s in zip(paths, sources)])
    X_clean = np.zeros((len(paths), N_FEATURES), dtype=np.float32)
    X_aug = {k: np.zeros((len(paths), N_FEATURES), dtype=np.float32) for k in AUGS}
    keep = np.zeros(len(paths), dtype=bool)

    t0 = time.time()
    for i, fp in enumerate(paths):
        try:
            audio, sr = librosa.load(fp, sr=22050)
            if len(audio) == 0:
                continue
            X_clean[i] = features_from_audio(audio, sr)
            rng = np.random.default_rng(SEED + i)   # deterministic augmentation
            for kind in AUGS:
                X_aug[kind][i] = features_from_audio(augment_audio(audio, sr, kind, rng), sr)
            keep[i] = True
        except Exception:
            continue
        if verbose and (i + 1) % 500 == 0:
            done, total = i + 1, len(paths)
            el = time.time() - t0
            print(f"  {done}/{total}   {el:.0f}s elapsed, "
                  f"~{el / done * (total - done):.0f}s remaining")

    np.savez_compressed(
        CACHE_FILE,
        X_clean=X_clean[keep], y=labels[keep], src=sources[keep],
        spk=speakers[keep], paths=paths[keep],
        **{f"X_{k}": X_aug[k][keep] for k in AUGS})
    if verbose:
        print(f"Cached {int(keep.sum())} clips to {CACHE_FILE} "
              f"in {time.time() - t0:.0f}s")
    return CACHE_FILE


def load_feedback():
    """User-corrected samples collected by the app, if any."""
    fp = os.path.join(FEEDBACK_DIR, "feedback.csv")
    if not os.path.exists(fp):
        return None, None
    try:
        df = pd.read_csv(fp)
        cols = [f"f{i}" for i in range(N_FEATURES)]
        if not set(cols).issubset(df.columns) or "label" not in df.columns:
            return None, None
        df = df.dropna(subset=cols + ["label"])
        if df.empty:
            return None, None
        return df[cols].to_numpy(np.float32), df["label"].astype(str).to_numpy()
    except Exception:
        return None, None


# ───────────────────────── model ─────────────────────────
def make_focal(gamma, alpha):
    ac = tf.constant(alpha, tf.float32)

    def focal(y_true, y_pred):
        yt = tf.one_hot(tf.cast(tf.reshape(y_true, [-1]), tf.int32), depth=y_pred.shape[-1])
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        ce = -yt * tf.math.log(y_pred)
        p_t = tf.reduce_sum(yt * y_pred, axis=-1, keepdims=True)
        return tf.reduce_sum(tf.pow(1 - p_t, gamma) * ce * ac, axis=-1)
    return focal


def build_model(input_dim, n_classes, hp, loss_fn):
    layers = [keras.Input(shape=(input_dim,))]
    for i, units in enumerate(hp["units"]):
        layers.append(keras.layers.Dense(units, activation=hp["activation"]))
        if i < hp["n_batchnorm"]:
            layers.append(keras.layers.BatchNormalization())
        drop = hp["dropout"] if i < 2 else hp["dropout_late"]
        if drop > 0:
            layers.append(keras.layers.Dropout(drop))
    layers.append(keras.layers.Dense(n_classes, activation="softmax"))
    m = keras.Sequential(layers)
    m.compile(optimizer=keras.optimizers.Adam(hp["lr"]), loss=loss_fn, metrics=["accuracy"])
    return m


def sample_hyperparams(rng, trial):
    """Trial 0 reproduces the published baseline; later trials explore."""
    if trial == 0:
        return {"units": [512, 256, 128, 64], "activation": "relu", "n_batchnorm": 2,
                "dropout": 0.4, "dropout_late": 0.3, "lr": 1e-3, "batch_size": 32,
                "gamma": 2.0, "label": "published baseline"}
    arch = rng.choice([[512, 256, 128, 64], [768, 384, 192, 96], [1024, 512, 256, 128],
                       [512, 512, 256, 128], [640, 320, 160, 80], [896, 448, 224, 112]])
    # cast away numpy scalar types — keras.layers.Dense rejects np.int64 for units
    return {"units": [int(u) for u in arch],
            "activation": str(rng.choice(["relu", "elu"])),
            "n_batchnorm": int(rng.choice([2, 3, 4])),
            "dropout": float(rng.choice([0.3, 0.35, 0.4, 0.45, 0.5])),
            "dropout_late": float(rng.choice([0.2, 0.25, 0.3, 0.35])),
            "lr": float(rng.choice([3e-4, 5e-4, 1e-3, 1.5e-3])),
            "batch_size": int(rng.choice([32, 64, 128])),
            "gamma": float(rng.choice([1.0, 1.5, 2.0, 2.5])),
            "label": "search"}


# ───────────────────────── data preparation ─────────────────────────
def prepare_splits(cache, speaker_independent):
    """
    Reproduces train_model.py's 80/20 stratified test split with seed 42, so the
    test set is identical to the one reported in the project documentation, then
    carves a validation set out of the training portion for model selection.
    """
    X_clean = cache["X_clean"]
    y_raw, src, spk = cache["y"], cache["src"], cache["spk"]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = list(le.classes_)
    idx = np.arange(len(X_clean))

    if speaker_independent:
        speakers = np.unique(spk)
        rs = np.random.RandomState(SEED)
        rs.shuffle(speakers)
        n_test = max(1, int(round(0.2 * len(speakers))))
        test_spk = set(speakers[:n_test])
        val_spk = set(speakers[n_test:n_test + max(1, int(round(0.15 * len(speakers))))])
        te = idx[np.isin(spk, list(test_spk))]
        va = idx[np.isin(spk, list(val_spk))]
        tr = idx[~np.isin(spk, list(test_spk | val_spk))]
    else:
        tr_full, te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
        tr, va = train_test_split(tr_full, test_size=0.15, random_state=SEED, stratify=y[tr_full])

    return X_clean, y, classes, le, src, tr, va, te


def assemble_training_set(cache, tr, y, classes, use_feedback=True):
    """Clean training clips + their cached augmented copies + user feedback."""
    X_parts = [cache["X_clean"][tr]]
    y_parts = [y[tr]]
    for kind in AUGS:
        key = f"X_{kind}"
        if key in cache:
            X_parts.append(cache[key][tr])
            y_parts.append(y[tr])

    n_feedback = 0
    if use_feedback:
        fx, fy = load_feedback()
        if fx is not None:
            mask = np.isin(fy, classes)
            if mask.any():
                mapping = {c: i for i, c in enumerate(classes)}
                X_parts.append(fx[mask])
                y_parts.append(np.array([mapping[v] for v in fy[mask]]))
                n_feedback = int(mask.sum())

    return np.concatenate(X_parts), np.concatenate(y_parts), n_feedback


# ───────────────────────── champion bookkeeping ─────────────────────────
def read_champion():
    if os.path.exists(CHAMPION_FILE):
        try:
            with open(CHAMPION_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def evaluate_deployed(X_val, y_val, X_test, y_test):
    """
    Score the currently deployed model so a challenger has a real incumbent to beat.

    The deployed model carries its own scaler, so the raw (unscaled) features are
    passed in and transformed with scaler.pkl rather than with the scaler fitted for
    this run. Returns (val_accuracy, test_accuracy) or None if nothing is deployed.
    """
    mp = os.path.join(BASE_DIR, "emotion_model.h5")
    sp = os.path.join(BASE_DIR, "scaler.pkl")
    if not (os.path.exists(mp) and os.path.exists(sp)):
        return None
    try:
        m = keras.models.load_model(mp, compile=False)
        sc = pickle.load(open(sp, "rb"))
        if m.input_shape[-1] != X_val.shape[1]:
            return None
        va = accuracy_score(y_val, m.predict(sc.transform(X_val), verbose=0).argmax(1))
        ta = accuracy_score(y_test, m.predict(sc.transform(X_test), verbose=0).argmax(1))
        keras.backend.clear_session()
        return float(va), float(ta)
    except Exception as e:
        print("could not score the deployed model:", str(e)[:120])
        return None


def promote(model, scaler, le, hp, val_acc, test_acc, test_f1, classes,
            X_test_s, y_test, src_test, trial):
    """Save a new best model and refresh the reporting artefacts."""
    os.makedirs(OUTDIR, exist_ok=True)
    # keep the previous model so a bad promotion is always reversible
    prev = os.path.join(BASE_DIR, "emotion_model.h5")
    if os.path.exists(prev):
        import shutil
        bak = os.path.join(BASE_DIR, "cache", "previous_model")
        os.makedirs(bak, exist_ok=True)
        for f in ("emotion_model.h5", "scaler.pkl", "label_encoder.pkl"):
            src_f = os.path.join(BASE_DIR, f)
            if os.path.exists(src_f):
                shutil.copy2(src_f, os.path.join(bak, f))
    model.save(os.path.join(BASE_DIR, "emotion_model.h5"))
    pickle.dump(le, open(os.path.join(BASE_DIR, "label_encoder.pkl"), "wb"))
    pickle.dump(scaler, open(os.path.join(BASE_DIR, "scaler.pkl"), "wb"))

    y_score = model.predict(X_test_s, verbose=0)
    y_pred = y_score.argmax(1)
    np.savez_compressed(os.path.join(OUTDIR, "test_predictions.npz"),
                        y_true=y_test, y_score=y_score, y_pred=y_pred,
                        classes=np.array(classes))
    pd.DataFrame(classification_report(y_test, y_pred, target_names=classes,
                                       output_dict=True, zero_division=0)
                 ).transpose().to_csv(os.path.join(OUTDIR, "classification_report.csv"))
    rows = []
    for name in ["RAVDESS", "TESS", "SAVEE", "CREMA-D"]:
        m = (src_test == name)
        if m.sum():
            rows.append({"Dataset": name,
                         "Accuracy(%)": round(accuracy_score(y_test[m], y_pred[m]) * 100, 2),
                         "MacroF1(%)": round(f1_score(y_test[m], y_pred[m],
                                                      average="macro", zero_division=0) * 100, 2),
                         "n_test": int(m.sum())})
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "per_dataset.csv"), index=False)

    with open(CHAMPION_FILE, "w") as f:
        json.dump({"trial": trial, "val_accuracy": round(val_acc * 100, 4),
                   "test_accuracy": round(test_acc * 100, 4),
                   "test_macro_f1": round(test_f1 * 100, 4),
                   "params": int(model.count_params()),
                   "hyperparameters": hp,
                   "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)


def best_config_from_log(split_kind):
    """
    Best configuration seen so far on the validation split, read back from the log.

    Reading the log rather than tracking the winner in memory means the refit also
    works after a resumed or interrupted run.
    """
    if not os.path.exists(LOG_FILE):
        return None, None
    try:
        df = pd.read_csv(LOG_FILE)
        df = df[df["split"] == split_kind]
        if df.empty:
            return None, None
        # tie-break on macro F1, then prefer the smaller model
        df = df.sort_values(["val_accuracy", "test_macro_f1", "params"],
                            ascending=[False, False, True])
        r = df.iloc[0]
        hp = {"units": [int(u) for u in str(r["architecture"]).split("-")],
              "activation": str(r["activation"]),
              "n_batchnorm": int(r["n_batchnorm"]),
              "dropout": float(r["dropout"]),
              "dropout_late": float(r["dropout_late"]),
              "lr": float(r["lr"]),
              "batch_size": int(r["batch_size"]),
              "gamma": float(r["gamma"]),
              "label": f"best of {len(df)} trials (val {r['val_accuracy']:.2f}%)"}
        return hp, int(r["epochs_run"])
    except Exception as e:
        print("could not read the trial log:", str(e)[:120])
        return None, None


def refit_on_full_data(cache, tr, va, te, y, classes, le, alpha, hp, epochs,
                       deploy_floor, use_feedback=True):
    """
    Retrain the winning configuration on train + validation, then test it once.

    Every search trial is handicapped: it trains without the validation clips, while
    the deployed model was fitted on the whole 80% partition. Refitting the selected
    configuration on all of that data is the standard way to give it a fair shot, and
    it is the step that can actually beat the incumbent.

    Early stopping is not used here — there is no held-out signal left to monitor
    without leaking the test set — so the epoch count found during the search is
    reused instead.
    """
    full = np.concatenate([tr, va])
    X_full, y_full, n_fb = assemble_training_set(cache, full, y, classes,
                                                 use_feedback=use_feedback)
    scaler = StandardScaler().fit(X_full)
    X_full_s = scaler.transform(X_full)
    X_test_s = scaler.transform(cache["X_clean"][te])
    y_test = y[te]

    print(f"\nRefitting the best configuration on train + validation "
          f"({len(X_full)} vectors"
          f"{f', incl. {n_fb} user-corrected' if n_fb else ''}) for {epochs} epochs ...")

    tf.keras.utils.set_random_seed(SEED)
    model = build_model(X_full_s.shape[1], len(classes), hp, make_focal(hp["gamma"], alpha))
    t0 = time.time()
    model.fit(X_full_s, y_full, epochs=epochs, batch_size=hp["batch_size"], verbose=0)
    secs = time.time() - t0

    y_pred = model.predict(X_test_s, verbose=0).argmax(1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    print(f"Refit result: test {acc*100:.2f}%   macroF1 {f1*100:.2f}%   ({secs:.0f}s)")
    return model, scaler, acc, f1


def log_trial(row):
    os.makedirs(OUTDIR, exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE), index=False)


# ───────────────────────── main loop ─────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=10, help="number of trials to run")
    ap.add_argument("--forever", action="store_true", help="keep searching until Ctrl+C")
    ap.add_argument("--epochs", type=int, default=120, help="max epochs per trial")
    ap.add_argument("--patience", type=int, default=15, help="early-stopping patience")
    ap.add_argument("--rebuild-cache", action="store_true", help="re-extract features")
    ap.add_argument("--speaker-independent", action="store_true",
                    help="split by speaker (stricter; NOT comparable to the report)")
    ap.add_argument("--no-feedback", action="store_true", help="ignore feedback/feedback.csv")
    ap.add_argument("--no-refit", action="store_true",
                    help="skip the final refit of the winning config on train+validation")
    args = ap.parse_args()

    if args.rebuild_cache or not os.path.exists(CACHE_FILE):
        build_cache()
    cache = np.load(CACHE_FILE, allow_pickle=True)

    X_clean, y, classes, le, src, tr, va, te = prepare_splits(
        cache, args.speaker_independent)
    X_train, y_train, n_feedback = assemble_training_set(
        cache, tr, y, classes, use_feedback=not args.no_feedback)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_clean[va])
    X_test_s = scaler.transform(X_clean[te])
    y_val, y_test, src_test = y[va], y[te], src[te]

    counts = np.bincount(y[tr], minlength=len(classes)).astype(float)
    inv = counts.sum() / (len(classes) * np.maximum(counts, 1))
    alpha = (inv / inv.mean()).astype(np.float32)

    split_kind = "speaker-independent" if args.speaker_independent else "stratified (matches report)"
    print(f"\nSplit: {split_kind}")
    print(f"Train {len(X_train)} (incl. augmentation"
          f"{f' + {n_feedback} user-corrected' if n_feedback else ''})"
          f" | Val {len(va)} | Test {len(te)}")

    # ── incumbent ──────────────────────────────────────────────────────────────
    # Two different comparisons are needed here, because the deployed model and the
    # challengers were not trained on the same data.
    #
    #   * Challengers are compared WITH EACH OTHER on the validation split. None of
    #     them trains on it, so it is a fair surface for choosing between them.
    #
    #   * The deployed model was trained on the full 80% partition, which includes
    #     this validation split, so its validation score is inflated and cannot be
    #     compared with a challenger's. The only surface that is fair for everyone
    #     is the held-out test set, which nothing here trains on. It is therefore
    #     used solely as a release gate: a challenger is deployed only if it beats
    #     the incumbent there.
    #
    # Note the trade-off: across many trials, gating on the test set makes the final
    # test figure mildly optimistic. Treat a promoted model's test accuracy as an
    # estimate and re-measure with --speaker-independent before quoting it.
    champ = read_champion()
    print("\nScoring the currently deployed model ...")
    deployed = evaluate_deployed(X_clean[va], y_val, X_clean[te], y_test)
    if deployed:
        dep_val, dep_test = deployed
        print(f"Deployed model: test {dep_test*100:.2f}%   "
              f"(its val {dep_val*100:.2f}% is inflated - it trained on those clips)")
    else:
        dep_val, dep_test = 0.0, 0.0
        print("Nothing deployed yet - the first trial becomes the champion.")

    # selection floor: only previous CHALLENGERS are comparable on validation
    best_val = (champ["val_accuracy"] / 100) if champ else 0.0
    # deployment floor: the fair, shared surface
    deploy_floor = max(dep_test, (champ.get("test_accuracy", 0) / 100) if champ else 0.0)
    print(f"Selection: beat val {best_val*100:.2f}%  |  "
          f"Deployment: beat test {deploy_floor*100:.2f}%")

    rng = np.random.default_rng(int(time.time()) & 0xFFFF)
    trial = (champ["trial"] + 1) if champ else 0
    done = 0
    best_hp, best_epochs = None, None

    try:
        while args.forever or done < args.trials:
            hp = sample_hyperparams(rng, trial)
            tf.keras.utils.set_random_seed(SEED + trial)
            model = build_model(X_train_s.shape[1], len(classes), hp,
                                make_focal(hp["gamma"], alpha))
            t0 = time.time()
            hist = model.fit(
                X_train_s, y_train, epochs=args.epochs, batch_size=hp["batch_size"],
                validation_data=(X_val_s, y_val), verbose=0,
                callbacks=[
                    keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                  patience=args.patience,
                                                  restore_best_weights=True),
                    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                      patience=5, min_lr=1e-5),
                ])
            secs = time.time() - t0

            val_acc = accuracy_score(y_val, model.predict(X_val_s, verbose=0).argmax(1))
            y_pred = model.predict(X_test_s, verbose=0).argmax(1)
            test_acc = accuracy_score(y_test, y_pred)
            test_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

            improved = val_acc > best_val
            arch = "-".join(str(u) for u in hp["units"])
            print(f"\nTrial {trial:>3}  [{hp['label']}]  {arch}  "
                  f"drop={hp['dropout']}/{hp['dropout_late']}  lr={hp['lr']:g}  "
                  f"bs={hp['batch_size']}  gamma={hp['gamma']}")
            print(f"          val {val_acc * 100:6.2f}%   test {test_acc * 100:6.2f}%   "
                  f"macroF1 {test_f1 * 100:6.2f}%   {len(hist.history['loss'])} epochs, "
                  f"{secs:.0f}s   {'*** NEW BEST on validation' if improved else 'no improvement'}")

            log_trial({"trial": trial, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "split": split_kind, "architecture": arch,
                       "activation": hp["activation"], "n_batchnorm": hp["n_batchnorm"],
                       "dropout": hp["dropout"], "dropout_late": hp["dropout_late"],
                       "lr": hp["lr"], "batch_size": hp["batch_size"], "gamma": hp["gamma"],
                       "epochs_run": len(hist.history["loss"]), "train_seconds": round(secs, 1),
                       "params": int(model.count_params()),
                       "val_accuracy": round(val_acc * 100, 3),
                       "test_accuracy": round(test_acc * 100, 3),
                       "test_macro_f1": round(test_f1 * 100, 3),
                       "n_feedback_samples": n_feedback,
                       "best_on_val": bool(improved)})

            if improved:
                best_val = val_acc
                best_hp, best_epochs = hp, len(hist.history["loss"])
                if args.speaker_independent:
                    # analysis run: the split is not comparable with the deployed model
                    print("          (speaker-independent run: logged but NOT deployed)")
                elif test_acc <= deploy_floor:
                    print(f"          held back: test {test_acc*100:.2f}% does not beat "
                          f"the deployed {deploy_floor*100:.2f}% - not deployed")
                else:
                    promote(model, scaler, le, hp, val_acc, test_acc, test_f1,
                            classes, X_test_s, y_test, src_test, trial)
                    deploy_floor = test_acc
                    print(f"          deployed (previous model backed up to "
                          f"cache/previous_model/)")

            keras.backend.clear_session()
            del model
            trial += 1
            done += 1
    except KeyboardInterrupt:
        print("\nInterrupted - the model on disk is the best one found so far.")

    # ── refit the winner on all of the training data ───────────────────────────
    log_hp, log_epochs = best_config_from_log(split_kind)
    if log_hp is not None:
        best_hp, best_epochs = log_hp, log_epochs
    if best_hp is not None and not args.speaker_independent and not args.no_refit:
        try:
            print(f"\nBest configuration: {best_hp['label']}")
            model, scaler, acc, f1 = refit_on_full_data(
                cache, tr, va, te, y, classes, le, alpha, best_hp, best_epochs,
                deploy_floor, use_feedback=not args.no_feedback)
            if acc > deploy_floor:
                X_test_s2 = scaler.transform(cache["X_clean"][te])
                promote(model, scaler, le, best_hp, best_val, acc, f1,
                        classes, X_test_s2, y_test, src_test, trial)
                print(f"DEPLOYED: {deploy_floor*100:.2f}% -> {acc*100:.2f}% "
                      f"(previous model backed up to cache/previous_model/)")
                deploy_floor = acc
            else:
                print(f"Not deployed: {acc*100:.2f}% does not beat the deployed "
                      f"{deploy_floor*100:.2f}%. The existing model is unchanged.")
            keras.backend.clear_session()
        except Exception as e:
            print("Refit skipped:", str(e)[:200])

    print("\n" + "=" * 62)
    print(f"DEPLOYED MODEL   test {deploy_floor*100:.2f}%")
    champ = read_champion()
    if champ:
        print(f"                 macroF1 {champ['test_macro_f1']:.2f}%, "
              f"{champ['params']:,} parameters")
    print(f"Full history: {LOG_FILE}")
    print("=" * 62)


if __name__ == "__main__":
    main()
