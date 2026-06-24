import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow import keras
import pickle

# ─── Dataset Paths ───────────────────────────────────────
TESS_PATH   = r"D:\web\FYP\Emotion_Recognition\TESS Toronto emotional speech set data"
RAVDESS_PATH = r"D:\web\FYP\Emotion_Recognition\RAVDESS"
CREMAD_PATH  = r"D:\web\FYP\Emotion_Recognition\CREMAD\AudioWAV"
SAVEE_PATH   = r"D:\web\FYP\Emotion_Recognition\SAVEE\AudioData"

# ─── Feature Extraction ──────────────────────────────────
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=22050, duration=3)
        if len(audio) == 0:
            return None

        # MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        mfcc_std  = np.std(mfcc.T, axis=0)

        # Chroma
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)

        # Mel
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_mean = np.mean(mel.T, axis=0)[:20]

        # ZCR + RMS
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
        rms = np.mean(librosa.feature.rms(y=audio))

        return np.concatenate([mfcc_mean, mfcc_std, chroma_mean, mel_mean, [zcr, rms]])
    except Exception as e:
        return None

X, y = [], []

# ─── 1. TESS ─────────────────────────────────────────────
print("\n--- Loading TESS ---")
for folder in os.listdir(TESS_PATH):
    folder_path = os.path.join(TESS_PATH, folder)
    if os.path.isdir(folder_path):
        emotion = folder.split("_")[-1].lower()
        if emotion == "pleasant_surprise":
            emotion = "surprise"
        for file in os.listdir(folder_path):
            if file.endswith(".wav"):
                features = extract_features(os.path.join(folder_path, file))
                if features is not None:
                    X.append(features)
                    y.append(emotion)
print(f"TESS loaded: {len(X)} samples")

# ─── 2. RAVDESS ──────────────────────────────────────────
print("\n--- Loading RAVDESS ---")
ravdess_emotions = {
    '01': 'neutral', '02': 'neutral', '03': 'happy',
    '04': 'sad',     '05': 'angry',   '06': 'fear',
    '07': 'disgust', '08': 'surprise'
}
count = 0
for actor_folder in os.listdir(RAVDESS_PATH):
    actor_path = os.path.join(RAVDESS_PATH, actor_folder)
    if os.path.isdir(actor_path):
        for file in os.listdir(actor_path):
            if file.endswith(".wav"):
                parts = file.split("-")
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    emotion = ravdess_emotions.get(emotion_code)
                    if emotion:
                        features = extract_features(os.path.join(actor_path, file))
                        if features is not None:
                            X.append(features)
                            y.append(emotion)
                            count += 1
print(f"RAVDESS loaded: {count} samples")

# ─── 3. CREMA-D ──────────────────────────────────────────
print("\n--- Loading CREMA-D ---")
cremad_emotions = {
    'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fear',
    'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'
}
count = 0
for file in os.listdir(CREMAD_PATH):
    if file.endswith(".wav"):
        parts = file.split("_")
        if len(parts) >= 3:
            emotion_code = parts[2]
            emotion = cremad_emotions.get(emotion_code)
            if emotion:
                features = extract_features(os.path.join(CREMAD_PATH, file))
                if features is not None:
                    X.append(features)
                    y.append(emotion)
                    count += 1
print(f"CREMA-D loaded: {count} samples")

# ─── 4. SAVEE ────────────────────────────────────────────
print("\n--- Loading SAVEE ---")
savee_emotions = {
    'a': 'angry', 'd': 'disgust', 'f': 'fear',
    'h': 'happy', 'n': 'neutral', 'sa': 'sad', 'su': 'surprise'
}
count = 0
for speaker in os.listdir(SAVEE_PATH):
    speaker_path = os.path.join(SAVEE_PATH, speaker)
    if os.path.isdir(speaker_path):
        for file in os.listdir(speaker_path):
            if file.endswith(".wav"):
                # filename like: a01.wav, sa01.wav, su01.wav
                name = file.replace(".wav", "")
                emotion = None
                # 2 letter pehle check karo
                if name[:2] in savee_emotions:
                    emotion = savee_emotions[name[:2]]
                elif name[:1] in savee_emotions:
                    emotion = savee_emotions[name[:1]]
                if emotion:
                    features = extract_features(os.path.join(speaker_path, file))
                    if features is not None:
                        X.append(features)
                        y.append(emotion)
                        count += 1
print(f"SAVEE loaded: {count} samples")

# ─── Summary ─────────────────────────────────────────────
print(f"\nTotal samples before augmentation: {len(X)}")

# ─── Augmentation ────────────────────────────────────────
print("Adding augmented samples...")
X_aug, y_aug = [], []
for features, label in zip(X, y):
    X_aug.append(features)
    y_aug.append(label)
    # Noise add karo
    noisy = features + np.random.normal(0, 0.05, features.shape)
    X_aug.append(noisy)
    y_aug.append(label)

X = np.array(X_aug)
y = np.array(y_aug)
print(f"Total samples after augmentation: {len(X)}")

# ─── Label Encoding ──────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\nEmotions found: {le.classes_}")

# ─── Scaling ─────────────────────────────────────────────
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ─── Train Test Split ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# ─── Model ───────────────────────────────────────────────
input_dim = X.shape[1]
model = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(input_dim,)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(len(le.classes_), activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=15, restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001
)

# ─── Training ────────────────────────────────────────────
print("\nTraining started...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr]
)

# ─── Save ────────────────────────────────────────────────
model.save("emotion_model.h5")

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n✅ Model saved!")
print(f"✅ Final Accuracy: {max(history.history['val_accuracy'])*100:.2f}%")