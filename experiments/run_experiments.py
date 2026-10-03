"""
Cross-validated experiments for the paper. Every run is cached in
results/experiments/runs, so the script can be stopped and resumed.

  python run_experiments.py main             # handcrafted / WavLM-Base+ models and ablations
  python run_experiments.py extra            # speaker-normalized variants
  python run_experiments.py large            # WavLM-Large variants (final model)
  python run_experiments.py baselines        # ML baselines + CRNN (SI and SD)
  python run_experiments.py baselines_sd     # SD baselines without the CRNN
  python run_experiments.py large_baselines  # LR / SVM on WavLM-Large embeddings
  python run_experiments.py corpus           # within-corpus (LOSO) + without CREMA-D
  python run_experiments.py tables           # summary CSV (results/experiments/all_runs.csv)
  python run_experiments.py folds            # file list with fold assignments (folds.csv)

Run extract.py first. Protocols
  SD : 5-fold stratified CV (speaker-dependent; same speakers may appear in train and test)
  SI : 5-fold StratifiedGroupKFold on speaker ID (speaker-independent; no speaker overlap)
Early stopping uses a validation slice taken from the training fold only
(speaker-disjoint for SI); the test fold is never seen during training or model selection.
"""
import os, sys, time, json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score
from common import build_index, CACHE, RESULTS, CLASSES, CORPORA, HAND_GROUPS

RUNS = os.path.join(RESULTS, "runs"); os.makedirs(RUNS, exist_ok=True)
SEED = 42
MAX_EPOCHS = int(os.environ.get("SER_EPOCHS", 120))
meta = build_index()
y_all = meta.label.map({c: i for i, c in enumerate(CLASSES)}).values
spk_all = meta.speaker.values
corp_all = meta.corpus.values
N, C = len(meta), len(CLASSES)

_hand = np.load(os.path.join(CACHE, "hand.npz"))
HAND, HAND_AUG = _hand["clean"], _hand["aug"]          # N x 284, N x 3 x 284
_wav = {}


def wavlm(tag="wavlm"):
    if tag not in _wav:
        _wav[tag] = np.load(os.path.join(CACHE, f"{tag}.npy"), mmap_mode="r")
    return _wav[tag]


# ─── speaker-level feature normalization ─────────────────────────────
# Each speaker's features are z-normalized with that speaker's own statistics,
# computed from *unlabeled* utterances only (no emotion labels are used).
# spknorm="all": statistics from all of the speaker's utterances;
# spknorm=K (int): statistics from K randomly chosen "enrollment" utterances,
# which is what a deployed system would collect from a new user.
def speaker_stats_index(K):
    rng = np.random.default_rng(SEED)
    sel = {}
    for s in np.unique(spk_all):
        i = np.where(spk_all == s)[0]
        sel[s] = i if K == "all" else np.sort(rng.choice(i, min(K, len(i)), replace=False))
    return sel


_SN = {}


def spk_normalized(name, K):
    key = (name, K)
    if key in _SN:
        return _SN[key]
    sel = speaker_stats_index(K)
    if name == "hand":
        X, A = HAND.astype(np.float32).copy(), HAND_AUG.astype(np.float32).copy()
        for s, i in sel.items():
            m = spk_all == s
            mu, sd = HAND[i].mean(0), HAND[i].std(0) + 1e-6
            X[m] = (HAND[m] - mu) / sd
            A[m] = (HAND_AUG[m] - mu) / sd
        _SN[key] = (X, A)
    else:
        W = wavlm(name)
        X = np.empty(W.shape, np.float32)
        for s, i in sel.items():
            m = np.where(spk_all == s)[0]
            Wi = np.asarray(W[i], np.float32)
            mu, sd = Wi.mean(0), Wi.std(0) + 1e-6
            X[m] = (np.asarray(W[m], np.float32) - mu) / sd
        _SN[key] = X
    return _SN[key]


def hand_arrays(spec):
    if spec.get("spknorm"):
        return spk_normalized("hand", spec["spknorm"])
    return HAND, HAND_AUG


def ssl_array(spec):
    tag = spec.get("ssl_tag", "wavlm")
    return spk_normalized(tag, spec["spknorm"]) if spec.get("spknorm") else wavlm(tag)


# ─── folds ───────────────────────────────────────────────────────────
def folds(protocol, idx=None, k=5):
    idx = np.arange(N) if idx is None else np.asarray(idx)
    y, g = y_all[idx], spk_all[idx]
    if protocol == "SD":
        sp = StratifiedKFold(k, shuffle=True, random_state=SEED).split(idx, y)
    elif protocol == "SI":
        if k >= len(np.unique(g)):   # leave-one-speaker-out
            from sklearn.model_selection import LeaveOneGroupOut
            sp = LeaveOneGroupOut().split(idx, y, g)
        else:
            sp = StratifiedGroupKFold(k, shuffle=True, random_state=SEED).split(idx, y, g)
    else:
        raise ValueError(protocol)
    return [(idx[a], idx[b]) for a, b in sp]


def val_split(tr, protocol, seed):
    """Hold out ~12% of the training fold for early stopping."""
    g = spk_all[tr]
    if protocol == "SI" and len(np.unique(g)) >= 4:
        sgk = StratifiedGroupKFold(8, shuffle=True, random_state=seed)
        a, b = next(sgk.split(tr, y_all[tr], g))
        return tr[a], tr[b]
    a, b = train_test_split(np.arange(len(tr)), test_size=0.12, random_state=seed, stratify=y_all[tr])
    return tr[a], tr[b]


# ─── model pieces ────────────────────────────────────────────────────
def make_focal(alpha, gamma=2.0):
    ac = tf.constant(alpha, tf.float32)

    def focal(y_true, y_pred):
        yt = tf.one_hot(tf.cast(tf.reshape(y_true, [-1]), tf.int32), depth=C)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        p_t = tf.reduce_sum(yt * y_pred, -1, keepdims=True)
        return tf.reduce_sum(ac * tf.pow(1 - p_t, gamma) * (-yt * tf.math.log(y_pred)), -1)
    return focal


def class_alpha(ytr):
    cnt = np.bincount(ytr, minlength=C).astype(float)
    inv = cnt.sum() / (C * np.maximum(cnt, 1))
    inv[cnt == 0] = 0
    return (inv / inv[cnt > 0].mean()).astype(np.float32)


@keras.utils.register_keras_serializable()
class LayerWeightedSum(keras.layers.Layer):
    """Softmax-weighted sum over the L hidden layers of a speech encoder."""
    def build(self, shape):
        self.w = self.add_weight(name="w", shape=(shape[1],), initializer="zeros")

    def call(self, x):
        return tf.einsum("bld,l->bd", x, tf.nn.softmax(self.w))


def dense_head(x):
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x); x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x); x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(128, activation="relu")(x); x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    return keras.layers.Dense(C, activation="softmax")(x)


def build_net(spec, shapes):
    ins, parts = [], []
    if "hand" in shapes:
        i = keras.Input(shapes["hand"]); ins.append(i); parts.append(i)
    if "ssl" in shapes:
        i = keras.Input(shapes["ssl"]); ins.append(i)
        parts.append(LayerWeightedSum()(i) if len(shapes["ssl"]) == 2 else i)
    x = parts[0] if len(parts) == 1 else keras.layers.Concatenate()(parts)
    return keras.Model(ins, dense_head(x))


# ─── inputs for one fold ─────────────────────────────────────────────
def hand_cols(spec):
    if spec.get("hand_groups") is None:
        return np.arange(284)
    return np.concatenate([np.arange(284)[HAND_GROUPS[g]] for g in spec["hand_groups"]])


def ssl_view(idx, spec):
    w = ssl_array(spec)
    if spec["ssl"] == "last":
        return np.asarray(w[idx, -1], np.float32)            # n x 1536
    if spec["ssl"] == "avg":
        return np.asarray(w[idx], np.float32).mean(1)         # n x 1536
    return np.asarray(w[idx], np.float32)                    # n x 13 x 1536


def make_inputs(spec, tr, others, augment):
    """Fit scalers on training clips only; return dict of arrays for tr and each of others."""
    out_tr, out_o, ytr = {}, [{} for _ in others], y_all[tr]
    if spec.get("hand", False):
        cols = hand_cols(spec)
        HC, HA = hand_arrays(spec)
        Xtr = HC[tr][:, cols]
        if augment:
            Xtr = np.concatenate([Xtr] + [HA[tr, k][:, cols] for k in range(3)])
            ytr = np.concatenate([ytr] * 4)
        sc = StandardScaler().fit(Xtr)
        out_tr["hand"] = sc.transform(Xtr).astype(np.float32)
        for o, idx in zip(out_o, others):
            o["hand"] = sc.transform(HC[idx][:, cols]).astype(np.float32)
    if spec.get("ssl"):
        Str = ssl_view(tr, spec)
        shp = Str.shape
        mu = Str.reshape(len(tr), -1).mean(0); sd = Str.reshape(len(tr), -1).std(0) + 1e-6
        norm = lambda a: ((a.reshape(len(a), -1) - mu) / sd).reshape((len(a),) + shp[1:]).astype(np.float32)
        out_tr["ssl"] = norm(Str)
        for o, idx in zip(out_o, others):
            o["ssl"] = norm(ssl_view(idx, spec))
    return out_tr, out_o, ytr


def order(d):
    return [d[k] for k in ("hand", "ssl") if k in d]


def fit_predict_dnn(spec, tr, te, protocol, seed):
    keras.utils.set_random_seed(seed)
    trn, val = val_split(tr, protocol, seed)
    augment = spec.get("augment", False)
    Xtr, (Xval, Xte), ytr = make_inputs(spec, trn, [val, te], augment)
    model = build_net(spec, {k: v.shape[1:] for k, v in Xtr.items()})
    loss = make_focal(class_alpha(ytr)) if spec.get("loss", "focal") == "focal" else "sparse_categorical_crossentropy"
    model.compile(keras.optimizers.Adam(1e-3), loss=loss, metrics=["accuracy"])
    cb = [keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True),
          keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5)]
    t0 = time.time()
    h = model.fit(order(Xtr), ytr, validation_data=(order(Xval), y_all[val]), epochs=MAX_EPOCHS,
                  batch_size=spec.get("batch", 32), callbacks=cb, verbose=0)
    info = {"epochs": len(h.history["loss"]), "train_s": time.time() - t0,
            "params": int(model.count_params()), "history": {k: [float(v) for v in vs] for k, vs in h.history.items()}}
    if "ssl" in Xtr and len(Xtr["ssl"].shape) == 3:
        lw = [l for l in model.layers if isinstance(l, LayerWeightedSum)][0]
        info["layer_weights"] = tf.nn.softmax(lw.w).numpy().tolist()
    return model.predict(order(Xte), verbose=0, batch_size=256), info


# ─── classical baselines ─────────────────────────────────────────────
def fit_predict_sklearn(spec, tr, te, protocol, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    Xtr, (Xte,), ytr = make_inputs(spec, tr, [te], spec.get("augment", False))
    Xtr, Xte = np.concatenate(order(Xtr), 1), np.concatenate(order(Xte), 1)
    clf = {"LR": lambda: LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced"),
           "SVM": lambda: SVC(C=10, kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
           "RF": lambda: RandomForestClassifier(500, n_jobs=-1, class_weight="balanced", random_state=seed),
           "HGB": lambda: HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, class_weight="balanced", random_state=seed),
           "kNN": lambda: KNeighborsClassifier(7, weights="distance")}[spec["clf"]]()
    t0 = time.time(); clf.fit(Xtr, ytr)
    p = np.zeros((len(te), C)); p[:, clf.classes_] = clf.predict_proba(Xte)
    return p, {"train_s": time.time() - t0}


def fit_predict_crnn(spec, tr, te, protocol, seed):
    keras.utils.set_random_seed(seed)
    M = np.load(os.path.join(CACHE, "logmel.npy"), mmap_mode="r")
    trn, val = val_split(tr, protocol, seed)
    _m = np.asarray(M[trn], np.float32); mu, sd = float(_m.mean(dtype=np.float64)), float(_m.std(dtype=np.float64)); del _m
    get = lambda i: ((np.asarray(M[i], np.float32) - mu) / sd)[..., None]
    inp = keras.Input(M.shape[1:] + (1,))
    x = inp
    for f in (32, 64, 96):
        x = keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPool2D(2)(x); x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Permute((2, 1, 3))(x)
    x = keras.layers.Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(64))(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(C, activation="softmax")(keras.layers.Dense(64, activation="relu")(x))
    model = keras.Model(inp, out)
    model.compile(keras.optimizers.Adam(1e-3), loss=make_focal(class_alpha(y_all[trn])), metrics=["accuracy"])
    cb = [keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
          keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)]
    t0 = time.time()
    h = model.fit(get(trn), y_all[trn], validation_data=(get(val), y_all[val]), epochs=min(50, MAX_EPOCHS),
                  batch_size=64, callbacks=cb, verbose=0)
    return model.predict(get(te), verbose=0, batch_size=128), {
        "epochs": len(h.history["loss"]), "train_s": time.time() - t0, "params": int(model.count_params())}


# ─── experiment runner ───────────────────────────────────────────────
def run(name, spec, protocol, idx=None, k=5, tag=None):
    tag = tag or f"{protocol}__{name}"
    path = os.path.join(RUNS, f"{tag}.npz")
    if os.path.isfile(path):
        return path
    fn = {"dnn": fit_predict_dnn, "sk": fit_predict_sklearn, "crnn": fit_predict_crnn}[spec.get("kind", "dnn")]
    fl = folds(protocol, idx, k)
    prob = np.full((N, C), np.nan, np.float32); fold_id = np.full(N, -1)
    infos = []
    for f, (tr, te) in enumerate(fl):
        p, info = fn(spec, tr, te, protocol, SEED + f)
        prob[te] = p; fold_id[te] = f
        info["acc"] = accuracy_score(y_all[te], p.argmax(1))
        infos.append(info)
        print(f"  [{tag}] fold {f+1}/{len(fl)} acc={info['acc']*100:.2f}%  ({info.get('train_s', 0):.0f}s)", flush=True)
    np.savez_compressed(path, prob=prob, fold=fold_id, info=json.dumps(infos))
    return path


def summarize(path, subset=None):
    d = np.load(path)
    prob, fold = d["prob"], d["fold"]
    m = fold >= 0 if subset is None else (fold >= 0) & subset
    yt, yp = y_all[m], prob[m].argmax(1)
    labels = np.unique(yt)
    per_fold = [accuracy_score(y_all[m & (fold == f)], prob[m & (fold == f)].argmax(1)) * 100
                for f in np.unique(fold[m])]
    per_fold_f1 = [f1_score(y_all[m & (fold == f)], prob[m & (fold == f)].argmax(1), labels=labels,
                            average="macro", zero_division=0) * 100 for f in np.unique(fold[m])]
    return {"acc": accuracy_score(yt, yp) * 100,
            "acc_mean": np.mean(per_fold), "acc_std": np.std(per_fold),
            "f1": f1_score(yt, yp, labels=labels, average="macro", zero_division=0) * 100,
            "f1_mean": np.mean(per_fold_f1), "f1_std": np.std(per_fold_f1),
            "uar": recall_score(yt, yp, labels=labels, average="macro", zero_division=0) * 100,
            "n": int(m.sum())}


# ─── experiment sets ─────────────────────────────────────────────────
HAND_ONLY = {"hand": True}
PROPOSED = {"hand": True, "ssl": "layers"}            # WavLM layer-weighted + 284-D fusion, focal

MAIN = {
    # previous (submitted) configuration, now evaluated without test-set early stopping
    "hand284_aug_focal":   {**HAND_ONLY, "augment": True},
    # feature ablation on the handcrafted pipeline
    "hand284_focal":       {**HAND_ONLY},
    "hand284_aug_ce":      {**HAND_ONLY, "augment": True, "loss": "ce"},
    "handMFCC_aug_focal":  {**HAND_ONLY, "augment": True, "hand_groups": ["mfcc"]},
    "handNoDelta_aug_focal": {**HAND_ONLY, "augment": True,
                              "hand_groups": ["mfcc", "chroma", "mel", "contrast", "zcr", "rms", "shape"]},
    # self-supervised representations
    "wavlm_last_focal":    {"ssl": "last"},
    "wavlm_avg_focal":     {"ssl": "avg"},
    "wavlm_layers_focal":  {"ssl": "layers"},
    # proposed fusion
    "fusion_ce":           {**PROPOSED, "loss": "ce"},
    "fusion_focal":        PROPOSED,
}

BASELINES = {
    **{f"hand284_{c}": {"kind": "sk", "hand": True, "clf": c} for c in ["LR", "SVM", "RF", "HGB", "kNN"]},
    **{f"wavlmAvg_{c}": {"kind": "sk", "ssl": "avg", "clf": c} for c in ["LR", "SVM"]},
    "logmel_CRNN": {"kind": "crnn"},
}


# Speaker-normalized variants.
FINAL = {"hand": True, "ssl": "layers", "spknorm": "all"}
EXTRA = {
    "hand284_aug_focal_spk": {**HAND_ONLY, "augment": True, "spknorm": "all"},
    "wavlm_layers_focal_spk": {"ssl": "layers", "spknorm": "all"},
    "fusion_focal_spk":      FINAL,
    "fusion_focal_spk20":    {**FINAL, "spknorm": 20},
    "fusion_ce_spk":         {**FINAL, "loss": "ce"},
    "wavlm_avg_focal_spk":   {"ssl": "avg", "spknorm": "all"},
    "fusion_avg_focal_spk":  {"hand": True, "ssl": "avg", "spknorm": "all"},
    "fusion_avg_focal_spk20": {"hand": True, "ssl": "avg", "spknorm": 20},
    "hand284_SVM_spk":       {"kind": "sk", "hand": True, "clf": "SVM", "spknorm": "all"},
    "wavlmAvg_SVM_spk":      {"kind": "sk", "ssl": "avg", "clf": "SVM", "spknorm": "all"},
    "wavlmAvg_LR_spk":       {"kind": "sk", "ssl": "avg", "clf": "LR", "spknorm": "all"},
}
LARGE = {
    "wavlmL_layers_focal":     {"ssl": "layers", "ssl_tag": "wavlm_large"},
    "wavlmL_layers_focal_spk": {"ssl": "layers", "ssl_tag": "wavlm_large", "spknorm": "all"},
    "fusionL_focal_spk":       {**FINAL, "ssl_tag": "wavlm_large"},
    "fusionL_focal_spk20":     {**FINAL, "ssl_tag": "wavlm_large", "spknorm": 20},
    "fusionL_avg_focal_spk":   {"hand": True, "ssl": "avg", "ssl_tag": "wavlm_large", "spknorm": "all"},
    "fusionL_avg_focal_spk20": {"hand": True, "ssl": "avg", "ssl_tag": "wavlm_large", "spknorm": 20},
    "wavlmLAvg_SVM_spk":       {"kind": "sk", "ssl": "avg", "ssl_tag": "wavlm_large", "clf": "SVM", "spknorm": "all"},
    "wavlmLAvg_LR_spk":        {"kind": "sk", "ssl": "avg", "ssl_tag": "wavlm_large", "clf": "LR", "spknorm": "all"},
}


# Final model (best speaker-independent result); also used for the within-corpus experiments.
CORPUS_MODEL = "fusionL_avg_focal_spk"


def do_extra():
    for name, spec in EXTRA.items():
        print(f"\n=== SI {name}"); run(name, spec, "SI")
    for name in ["hand284_aug_focal_spk", "fusion_focal_spk", "fusion_focal_spk20", "fusion_avg_focal_spk",
                 "fusion_avg_focal_spk20"]:
        print(f"\n=== SD {name}"); run(name, EXTRA[name], "SD")


def do_large():
    if not os.path.isfile(os.path.join(CACHE, "wavlm_large.npy")):
        print("wavlm_large.npy missing; skipping"); return
    for name, spec in LARGE.items():
        print(f"\n=== SI {name}"); run(name, spec, "SI")
    for name in ["fusionL_focal_spk", "fusionL_focal_spk20", "fusionL_avg_focal_spk", "fusionL_avg_focal_spk20"]:
        print(f"\n=== SD {name}"); run(name, LARGE[name], "SD")


def do_baselines_sd():
    """Speaker-dependent baselines without the CRNN (≈35 min per fold on CPU)."""
    for name, spec in BASELINES.items():
        if spec.get("kind") != "crnn":
            print(f"\n=== SD {name}"); run(name, spec, "SD")


def do_large_baselines():
    for name in ["wavlmLAvg_SVM_spk", "wavlmLAvg_LR_spk"]:
        for proto in ["SI", "SD"]:
            print(f"\n=== {proto} {name}"); run(name, LARGE[name], proto)


def do_main():
    for proto in ["SI", "SD"]:
        for name, spec in MAIN.items():
            if proto == "SD" and name not in ("hand284_aug_focal", "hand284_focal", "wavlm_layers_focal", "fusion_focal"):
                continue
            print(f"\n=== {proto} {name}"); run(name, spec, proto)


def do_baselines():
    for proto in ["SI", "SD"]:
        for name, spec in BASELINES.items():
            print(f"\n=== {proto} {name}"); run(name, spec, proto)


def do_corpus():
    # within-corpus, speaker-independent (LOSO for the small corpora)
    for c, k in [("TESS", 2), ("SAVEE", 4), ("RAVDESS", 6), ("CREMA-D", 5)]:
        idx = np.where(corp_all == c)[0]
        for name, spec in [("hand284_aug_focal", MAIN["hand284_aug_focal"]), (CORPUS_MODEL, {**EXTRA, **LARGE}[CORPUS_MODEL])]:
            run(name, spec, "SI", idx, k, tag=f"within_{c}__SI__{name}")
            run(name, spec, "SD", idx, 5, tag=f"within_{c}__SD__{name}")
    # merged model without CREMA-D
    idx = np.where(corp_all != "CREMA-D")[0]
    run(CORPUS_MODEL, {**EXTRA, **LARGE}[CORPUS_MODEL], "SI", idx, 5, tag=f"noCREMA__SI__{CORPUS_MODEL}")


def do_folds():
    """Write the file list with speaker-independent and speaker-dependent fold assignments."""
    out = meta[["path", "label", "corpus", "speaker"]].copy()
    for proto in ["SI", "SD"]:
        col = np.full(N, -1)
        for f, (_, te) in enumerate(folds(proto)):
            col[te] = f
        out[f"{proto.lower()}_fold"] = col
    out["path"] = out["path"].str.replace("\\", "/", regex=False)
    out.to_csv(os.path.join(RESULTS, "folds.csv"), index=False)
    print("wrote folds.csv")


def do_tables():
    rows = []
    for f in sorted(os.listdir(RUNS)):
        if not f.endswith(".npz"):
            continue
        tag = f[:-4]; p = os.path.join(RUNS, f)
        s = summarize(p); s["run"] = tag
        d = np.load(p); infos = json.loads(str(d["info"]))
        s["params"] = infos[0].get("params", ""); s["train_s_per_fold"] = np.mean([i.get("train_s", 0) for i in infos])
        s["epochs"] = np.mean([i.get("epochs", 0) for i in infos]) if "epochs" in infos[0] else ""
        rows.append(s)
        if tag.count("__") == 1:  # merged runs: per-corpus breakdown
            for c in CORPORA:
                sc = summarize(p, corp_all == c)
                rows.append({**sc, "run": f"{tag}@{c}"})
    df = pd.DataFrame(rows).set_index("run").round(2)
    df.to_csv(os.path.join(RESULTS, "all_runs.csv"))
    print(df[["acc_mean", "acc_std", "f1_mean", "f1_std", "uar", "n"]].to_string())


if __name__ == "__main__":
    for stage in sys.argv[1:]:
        {"main": do_main, "extra": do_extra, "large": do_large, "baselines": do_baselines,
         "baselines_sd": do_baselines_sd, "large_baselines": do_large_baselines,
         "corpus": do_corpus, "folds": do_folds, "tables": do_tables}[stage]()
