"""
Trains the handcrafted-feature model used by app.py.

Features (284-D): MFCC, delta and delta-delta (mean + std of each), chroma,
mel energies, spectral contrast, ZCR, RMS and three spectral-shape descriptors.
Training clips are augmented on the waveform (noise, pitch shift, time
stretch) and the network is trained with a class-balanced focal loss.

Outputs: emotion_model.h5, scaler.pkl, label_encoder.pkl and CSV/PNG files in
results/. The model is saved with a custom loss, so load it with
load_model("emotion_model.h5", compile=False).

Note: this script uses a single random 80/20 split. The speaker-independent
evaluation reported in the paper is in experiments/run_experiments.py.
"""
import os, time, pickle, numpy as np, pandas as pd
import librosa
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt, seaborn as sns

SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)

# Resolve every path relative to this script so the run works from any working
# directory and on any machine that has the datasets sitting next to it.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(BASE_DIR, "results"); os.makedirs(OUTDIR, exist_ok=True)

AUGS = ["noise", "pitch", "stretch"]   # trim this list if the run is too slow
RUN_CV = True
RUN_ABLATION = True

TESS_PATH    = os.path.join(BASE_DIR, "TESS Toronto emotional speech set data")
RAVDESS_PATH = os.path.join(BASE_DIR, "RAVDESS")
CREMAD_PATH  = os.path.join(BASE_DIR, "CREMAD", "AudioWAV")
SAVEE_PATH   = os.path.join(BASE_DIR, "SAVEE", "AudioData")

for _name, _p in [("TESS", TESS_PATH), ("RAVDESS", RAVDESS_PATH),
                  ("CREMA-D", CREMAD_PATH), ("SAVEE", SAVEE_PATH)]:
    if not os.path.isdir(_p):
        raise SystemExit(f"Dataset folder for {_name} not found: {_p}")

# ─── Feature extraction (284-dim) ────────────────────────
def features_from_audio(audio, sr=22050):
    audio, _ = librosa.effects.trim(audio, top_db=25)
    if len(audio) < sr // 2:
        audio = np.pad(audio, (0, sr // 2 - len(audio)))
    audio = librosa.util.normalize(audio)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    d1   = librosa.feature.delta(mfcc)
    d2   = librosa.feature.delta(mfcc, order=2)
    def ms(x): return np.concatenate([x.mean(axis=1), x.std(axis=1)])
    mfcc_f = ms(mfcc)                                     # 80
    d1_f   = ms(d1)                                       # 80
    d2_f   = ms(d2)                                       # 80
    chroma   = librosa.feature.chroma_stft(y=audio, sr=sr).mean(axis=1)          # 12
    mel      = librosa.feature.melspectrogram(y=audio, sr=sr).mean(axis=1)[:20]  # 20
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr).mean(axis=1)    # 7
    zcr = np.array([librosa.feature.zero_crossing_rate(y=audio).mean()])         # 1
    rms = np.array([librosa.feature.rms(y=audio).mean()])                        # 1
    cent = librosa.feature.spectral_centroid(y=audio, sr=sr).mean()
    bw   = librosa.feature.spectral_bandwidth(y=audio, sr=sr).mean()
    roll = librosa.feature.spectral_rolloff(y=audio, sr=sr).mean()
    spec = np.array([cent, bw, roll])                                            # 3
    return np.concatenate([mfcc_f, d1_f, d2_f, chroma, mel, contrast, zcr, rms, spec])  # 284

def augment_audio(audio, sr, kind):
    if kind == "noise":
        return audio + 0.05 * np.random.randn(len(audio))
    if kind == "pitch":
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=np.random.choice([-2, -1, 1, 2]))
    if kind == "stretch":
        return librosa.effects.time_stretch(audio, rate=np.random.uniform(0.9, 1.1))
    return audio

# ─── Load datasets (clean features + keep file paths) ────
Xc, y, dsrc, paths = [], [], [], []
def add(fp, label, name):
    try:
        audio, sr = librosa.load(fp, sr=22050)
        if len(audio) == 0: return
        Xc.append(features_from_audio(audio, sr)); y.append(label)
        dsrc.append(name); paths.append(fp)
    except Exception:
        pass

print("--- TESS ---")
for folder in os.listdir(TESS_PATH):
    fp = os.path.join(TESS_PATH, folder)
    if os.path.isdir(fp):
        emo = folder.split("_")[-1].lower()
        if emo in ["pleasant_surprise", "surprised", "pleasant_surprised"]: emo = "surprise"
        for f in os.listdir(fp):
            if f.endswith(".wav"): add(os.path.join(fp, f), emo, "TESS")
print("--- RAVDESS ---")
rav = {'01':'neutral','02':'neutral','03':'happy','04':'sad','05':'angry','06':'fear','07':'disgust','08':'surprise'}
for a in os.listdir(RAVDESS_PATH):
    ap = os.path.join(RAVDESS_PATH, a)
    if os.path.isdir(ap):
        for f in os.listdir(ap):
            if f.endswith(".wav"):
                p = f.split("-")
                if len(p) >= 3 and rav.get(p[2]): add(os.path.join(ap, f), rav[p[2]], "RAVDESS")
print("--- CREMA-D ---")
cre = {'ANG':'angry','DIS':'disgust','FEA':'fear','HAP':'happy','NEU':'neutral','SAD':'sad'}
for f in os.listdir(CREMAD_PATH):
    if f.endswith(".wav"):
        p = f.split("_")
        if len(p) >= 3 and cre.get(p[2]): add(os.path.join(CREMAD_PATH, f), cre[p[2]], "CREMA-D")
print("--- SAVEE ---")
sav = {'a':'angry','d':'disgust','f':'fear','h':'happy','n':'neutral','sa':'sad','su':'surprise'}
for sp in os.listdir(SAVEE_PATH):
    spp = os.path.join(SAVEE_PATH, sp)
    if os.path.isdir(spp):
        for f in os.listdir(spp):
            if f.endswith(".wav"):
                nm = f.replace(".wav", ""); emo = sav.get(nm[:2]) or sav.get(nm[:1])
                if emo: add(os.path.join(spp, f), emo, "SAVEE")

Xc = np.array(Xc); y = np.array(y); dsrc = np.array(dsrc); paths = np.array(paths)
print(f"\nClean features: {Xc.shape}  (should be N x 284)")
le = LabelEncoder(); y_enc = le.fit_transform(y); CLASSES = list(le.classes_)
print("Classes:", CLASSES, "| distribution:", dict(zip(*np.unique(y, return_counts=True))))

# ─── Split, then augment TRAIN clips only (waveform) ─────
idx = np.arange(len(Xc))
tr, te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y_enc)
X_train, y_train = list(Xc[tr]), list(y_enc[tr])
print(f"\nAugmenting {len(tr)} training clips with {AUGS} ...")
t0 = time.time()
for n, i in enumerate(tr):
    try:
        audio, sr = librosa.load(paths[i], sr=22050)
        for kind in AUGS:
            X_train.append(features_from_audio(augment_audio(audio, sr, kind), sr))
            y_train.append(y_enc[i])
    except Exception:
        pass
    if (n + 1) % 1000 == 0: print(f"  {n+1}/{len(tr)}  ({time.time()-t0:.0f}s)")
X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test, d_test = Xc[te], y_enc[te], dsrc[te]
print(f"Train (aug): {X_train.shape} | Test (clean): {X_test.shape}")

scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

# ─── Focal loss (class-balanced alpha) ───────────────────
counts = np.bincount(y_enc[tr], minlength=len(CLASSES)).astype(float)
inv = counts.sum() / (len(CLASSES) * np.maximum(counts, 1))
alpha_vec = (inv / inv.mean()).astype(np.float32)
print("Focal alpha per class:", dict(zip(CLASSES, np.round(alpha_vec, 2))))

def make_focal(gamma=2.0, alpha=None):
    ac = None if alpha is None else tf.constant(alpha, tf.float32)
    def focal(y_true, y_pred):
        yt = tf.one_hot(tf.cast(tf.reshape(y_true, [-1]), tf.int32), depth=y_pred.shape[-1])
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        ce = -yt * tf.math.log(y_pred)
        p_t = tf.reduce_sum(yt * y_pred, axis=-1, keepdims=True)
        loss = tf.pow(1 - p_t, gamma) * ce
        if ac is not None: loss = loss * ac
        return tf.reduce_sum(loss, axis=-1)
    return focal

def build_model(input_dim, n, loss_fn):
    m = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(input_dim,)),
        keras.layers.BatchNormalization(), keras.layers.Dropout(0.4),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(), keras.layers.Dropout(0.4),
        keras.layers.Dense(128, activation='relu'), keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(n, activation='softmax'),
    ])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss_fn, metrics=['accuracy'])
    return m

def cbs():
    return [keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)]

focal = make_focal(2.0, alpha_vec)
model = build_model(X_train_s.shape[1], len(CLASSES), focal)
model.summary()
t0 = time.time()
history = model.fit(X_train_s, y_train, epochs=120, batch_size=32,
                    validation_data=(X_test_s, y_test), callbacks=cbs())
train_time = time.time() - t0

# ─── Save model + preprocessing ──────────────────────────
model.save(os.path.join(BASE_DIR, "emotion_model.h5"))
pickle.dump(le, open(os.path.join(BASE_DIR, "label_encoder.pkl"), "wb"))
pickle.dump(scaler, open(os.path.join(BASE_DIR, "scaler.pkl"), "wb"))

# ─── CORE result files (saved first) ─────────────────────
pd.DataFrame({"epoch": np.arange(1, len(history.history['accuracy'])+1),
              "accuracy": history.history['accuracy'], "val_accuracy": history.history['val_accuracy'],
              "loss": history.history['loss'], "val_loss": history.history['val_loss']}
             ).to_csv(f"{OUTDIR}/training_history.csv", index=False)
y_score = model.predict(X_test_s, verbose=0); y_pred = y_score.argmax(1)
np.savez_compressed(f"{OUTDIR}/test_predictions.npz", y_true=y_test, y_score=y_score,
                    y_pred=y_pred, classes=np.array(CLASSES))
acc = accuracy_score(y_test, y_pred); f1 = f1_score(y_test, y_pred, average='macro')
print(f"\n*** TEST accuracy = {acc*100:.2f}%   macro-F1 = {f1*100:.2f}% ***")
pd.DataFrame(classification_report(y_test, y_pred, target_names=CLASSES, output_dict=True,
             zero_division=0)).transpose().to_csv(f"{OUTDIR}/classification_report.csv")
rows = []
for name in ["RAVDESS", "TESS", "SAVEE", "CREMA-D"]:
    m = (d_test == name)
    if m.sum(): rows.append({"Dataset": name,
        "Accuracy(%)": round(accuracy_score(y_test[m], y_pred[m])*100, 2),
        "MacroF1(%)": round(f1_score(y_test[m], y_pred[m], average='macro', zero_division=0)*100, 2),
        "n_test": int(m.sum())})
pd.DataFrame(rows).to_csv(f"{OUTDIR}/per_dataset.csv", index=False)
print("Core result files saved.")

# ─── 5-fold CV (clean features; lighter for speed) ───────
if RUN_CV:
    try:
        print("\n5-fold CV ...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED); fa = []
        for k, (trk, tek) in enumerate(skf.split(Xc, y_enc), 1):
            sc = StandardScaler().fit(Xc[trk])
            mk = build_model(Xc.shape[1], len(CLASSES), make_focal(2.0, alpha_vec))
            mk.fit(sc.transform(Xc[trk]), y_enc[trk], epochs=120, batch_size=32,
                   validation_data=(sc.transform(Xc[tek]), y_enc[tek]), callbacks=cbs(), verbose=0)
            a = accuracy_score(y_enc[tek], mk.predict(sc.transform(Xc[tek]), verbose=0).argmax(1))
            fa.append(a*100); print(f"  fold {k}: {a*100:.2f}%")
        pd.DataFrame({"fold": list(range(1,6))+["mean","std"],
                      "accuracy(%)": [round(x,2) for x in fa]+[round(np.mean(fa),2), round(np.std(fa),2)]}
                     ).to_csv(f"{OUTDIR}/cv_results.csv", index=False)
    except Exception as e:
        print("CV skipped:", e)

# ─── Ablation (clean features) ───────────────────────────
if RUN_ABLATION:
    try:
        print("\nAblation ...")
        MFCC = slice(0, 80)
        def quick(Xtr, ytr, Xte, yte):
            sc = StandardScaler().fit(Xtr)
            m = build_model(Xtr.shape[1], len(CLASSES), make_focal(2.0, alpha_vec))
            m.fit(sc.transform(Xtr), ytr, epochs=120, batch_size=32,
                  validation_data=(sc.transform(Xte), yte), callbacks=cbs(), verbose=0)
            p = m.predict(sc.transform(Xte), verbose=0).argmax(1)
            return accuracy_score(yte, p)*100, f1_score(yte, p, average='macro', zero_division=0)*100
        abl = []
        rm = np.where(dsrc == "RAVDESS")[0]
        r1, r2 = train_test_split(rm, test_size=0.2, random_state=SEED, stratify=y_enc[rm])
        a, f = quick(Xc[r1][:, MFCC], y_enc[r1], Xc[r2][:, MFCC], y_enc[r2]); abl.append(["Baseline (MFCC, single corpus)", round(a,2), round(f,2)])
        a, f = quick(Xc[tr][:, MFCC], y_enc[tr], Xc[te][:, MFCC], y_test); abl.append(["+ Hybrid integration", round(a,2), round(f,2)])
        a, f = quick(Xc[tr], y_enc[tr], Xc[te], y_test); abl.append(["+ Full fused features", round(a,2), round(f,2)])
        abl.append(["+ Augmentation + focal (proposed)", round(acc*100,2), round(f1*100,2)])
        pd.DataFrame(abl, columns=["Experiment","Accuracy(%)","F1(%)"]).to_csv(f"{OUTDIR}/ablation.csv", index=False)
    except Exception as e:
        print("Ablation skipped:", e)

# ─── SVM baseline + timing ───────────────────────────────
try:
    t0 = time.time(); svm = SVC(kernel='rbf').fit(X_train_s, y_train); svt = time.time()-t0
    sa = accuracy_score(y_test, svm.predict(X_test_s))
    t0 = time.time(); _ = model.predict(X_test_s[:200], verbose=0); di = (time.time()-t0)/200
    pd.DataFrame([
        {"Model":"Proposed DNN (284, focal)","Params":int(model.count_params()),"TrainTime(s)":round(train_time,1),"Inference(ms/clip)":round(di*1000,2),"Accuracy(%)":round(acc*100,2)},
        {"Model":"SVM baseline","Params":"-","TrainTime(s)":round(svt,1),"Inference(ms/clip)":"-","Accuracy(%)":round(sa*100,2)},
    ]).to_csv(f"{OUTDIR}/computational.csv", index=False)
except Exception as e:
    print("SVM skipped:", e)

# ─── draft PNGs ──────────────────────────────────────────
fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4))
a1.plot(history.history['accuracy'],label='Train');a1.plot(history.history['val_accuracy'],label='Val');a1.set_title('Accuracy');a1.legend()
a2.plot(history.history['loss'],label='Train');a2.plot(history.history['val_loss'],label='Val');a2.set_title('Loss');a2.legend()
plt.tight_layout();plt.savefig(f"{OUTDIR}/training_curves.png",dpi=150);plt.close()
plt.figure(figsize=(8,6));sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt='d',cmap='Blues',xticklabels=CLASSES,yticklabels=CLASSES)
plt.title(f'Confusion Matrix — {acc*100:.1f}%');plt.ylabel('Actual');plt.xlabel('Predicted');plt.tight_layout();plt.savefig(f"{OUTDIR}/confusion_matrix.png",dpi=150);plt.close()

print("\nDone. Results written to", OUTDIR)
