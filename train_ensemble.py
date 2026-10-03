"""
Soft-voting ensemble of the best configurations found by auto_train.py.

The top configurations (ranked on the validation split) are refitted on the
training partition for the number of epochs each needed during the search,
their softmax outputs are averaged, and the ensemble is scored once on the
held-out test split.

Usage:
    python train_ensemble.py                 # 5 members
    python train_ensemble.py --members 7
    python train_ensemble.py --deploy        # save to ensemble/ if it beats the current model
"""
import argparse
import json
import os
import pickle
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

from auto_train import (BASE_DIR, CACHE_FILE, OUTDIR, LOG_FILE, SEED,
                        assemble_training_set, build_model, make_focal)

ENS_DIR = os.path.join(BASE_DIR, "ensemble")


def top_configs(n, split_kind="stratified (matches report)"):
    """Best n configurations by validation accuracy, preferring distinct architectures."""
    df = pd.read_csv(LOG_FILE)
    df = df[df["split"] == split_kind].sort_values(
        ["val_accuracy", "test_macro_f1"], ascending=False)
    picked, seen = [], set()
    for _, r in df.iterrows():                      # one per architecture first
        if r["architecture"] not in seen:
            picked.append(r); seen.add(r["architecture"])
        if len(picked) == n:
            break
    for _, r in df.iterrows():                      # then fill up with the next best
        if len(picked) == n:
            break
        if not any(r["trial"] == p["trial"] for p in picked):
            picked.append(r)
    out = []
    for r in picked[:n]:
        out.append(({"units": [int(u) for u in str(r["architecture"]).split("-")],
                     "activation": str(r["activation"]),
                     "n_batchnorm": int(r["n_batchnorm"]),
                     "dropout": float(r["dropout"]),
                     "dropout_late": float(r["dropout_late"]),
                     "lr": float(r["lr"]),
                     "batch_size": int(r["batch_size"]),
                     "gamma": float(r["gamma"]),
                     "label": f"trial {int(r['trial'])}"},
                    int(r["epochs_run"]), float(r["val_accuracy"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--deploy", action="store_true",
                    help="save the ensemble if it beats the deployed model on test")
    args = ap.parse_args()

    cache = np.load(CACHE_FILE, allow_pickle=True)
    X, y_raw, src = cache["X_clean"], cache["y"], cache["src"]
    le = LabelEncoder().fit(y_raw)
    y = le.transform(y_raw)
    classes = list(le.classes_)

    idx = np.arange(len(X))
    tr_full, te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
    y_test, src_test = y[te], src[te]

    X_full, y_full, n_fb = assemble_training_set(cache, tr_full, y, classes)
    scaler = StandardScaler().fit(X_full)
    X_full_s = scaler.transform(X_full)
    X_test_s = scaler.transform(X[te])

    counts = np.bincount(y[tr_full], minlength=len(classes)).astype(float)
    inv = counts.sum() / (len(classes) * np.maximum(counts, 1))
    alpha = (inv / inv.mean()).astype(np.float32)

    configs = top_configs(args.members)
    print(f"Training {len(configs)} members on {len(X_full):,} vectors "
          f"({len(te):,} test clips held out)\n")

    probs, members, models = [], [], []
    for i, (hp, epochs, val_acc) in enumerate(configs):
        arch = "-".join(str(u) for u in hp["units"])
        tf.keras.utils.set_random_seed(SEED + 100 + i)
        m = build_model(X_full_s.shape[1], len(classes), hp, make_focal(hp["gamma"], alpha))
        t0 = time.time()
        m.fit(X_full_s, y_full, epochs=epochs, batch_size=hp["batch_size"], verbose=0)
        p = m.predict(X_test_s, verbose=0)
        acc = accuracy_score(y_test, p.argmax(1)) * 100
        probs.append(p)
        models.append(m)
        members.append({"member": i, "source": hp["label"], "architecture": arch,
                        "epochs": epochs, "search_val_accuracy": round(val_acc, 2),
                        "solo_test_accuracy": round(acc, 2)})
        run = accuracy_score(y_test, np.mean(probs, axis=0).argmax(1)) * 100
        print(f"  member {i}  {arch:<18s} {epochs:>3d} ep  "
              f"solo {acc:5.2f}%   ensemble so far {run:5.2f}%   ({time.time()-t0:.0f}s)")

    mean_p = np.mean(probs, axis=0)
    y_pred = mean_p.argmax(1)
    acc = accuracy_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0) * 100
    solo = [m["solo_test_accuracy"] for m in members]

    print(f"\nbest single member : {max(solo):.2f}%")
    print(f"mean single member : {np.mean(solo):.2f}%")
    print(f"ENSEMBLE           : {acc:.2f}%   macro F1 {f1:.2f}%")

    print("\nper-dataset:")
    for name in ["RAVDESS", "TESS", "SAVEE", "CREMA-D"]:
        mk = (src_test == name)
        if mk.sum():
            print(f"  {name:<9s} {accuracy_score(y_test[mk], y_pred[mk])*100:6.2f}%  "
                  f"(n={int(mk.sum())})")

    os.makedirs(OUTDIR, exist_ok=True)
    pd.DataFrame(members).to_csv(os.path.join(OUTDIR, "ensemble_members.csv"), index=False)
    with open(os.path.join(OUTDIR, "ensemble_result.json"), "w") as f:
        json.dump({"members": len(members), "test_accuracy": round(acc, 3),
                   "test_macro_f1": round(f1, 3),
                   "best_single": round(max(solo), 3),
                   "mean_single": round(float(np.mean(solo)), 3),
                   "n_feedback_samples": n_fb,
                   "created": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

    if args.deploy:
        deployed = 0.0
        cf = os.path.join(OUTDIR, "champion.json")
        mp = os.path.join(BASE_DIR, "emotion_model.h5")
        if os.path.exists(mp):
            try:
                dm = keras.models.load_model(mp, compile=False)
                ds = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
                deployed = accuracy_score(
                    y_test, dm.predict(ds.transform(X[te]), verbose=0).argmax(1)) * 100
            except Exception:
                pass
        print(f"\ncurrently deployed: {deployed:.2f}%")
        if acc > deployed:
            os.makedirs(ENS_DIR, exist_ok=True)
            for i, m in enumerate(models):
                m.save(os.path.join(ENS_DIR, f"member_{i}.h5"))
            pickle.dump(scaler, open(os.path.join(ENS_DIR, "scaler.pkl"), "wb"))
            pickle.dump(le, open(os.path.join(ENS_DIR, "label_encoder.pkl"), "wb"))
            pd.DataFrame(classification_report(
                y_test, y_pred, target_names=classes, output_dict=True, zero_division=0)
            ).transpose().to_csv(os.path.join(OUTDIR, "ensemble_classification_report.csv"))
            print(f"saved {len(models)} members to ensemble/ "
                  f"({deployed:.2f}% -> {acc:.2f}%)")
            print("app.py must be pointed at ensemble/ to use it.")
        else:
            print("not saved: the ensemble does not beat the deployed model.")


if __name__ == "__main__":
    main()
