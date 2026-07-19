import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Dataset Paths ───────────────────────────────────────
TESS_PATH    = r"D:\web\FYP\Emotion_Recognition\TESS Toronto emotional speech set data"
RAVDESS_PATH = r"D:\web\FYP\Emotion_Recognition\RAVDESS"
CREMAD_PATH  = r"D:\web\FYP\Emotion_Recognition\CREMAD\AudioWAV"
SAVEE_PATH   = r"D:\web\FYP\Emotion_Recognition\SAVEE\AudioData"

# ─── Feature Extraction ──────────────────────────────────
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=22050, duration=3)
        if len(audio) == 0:
            return None

        mfcc        = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc_mean   = np.mean(mfcc.T, axis=0)
        mfcc_std    = np.std(mfcc.T, axis=0)

        chroma      = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)

        mel         = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_mean    = np.mean(mel.T, axis=0)[:20]

        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
        rms = np.mean(librosa.feature.rms(y=audio))

        return np.concatenate([mfcc_mean, mfcc_std, chroma_mean, mel_mean, [zcr, rms]])
    except Exception:
        return None

X, y = [], []

# ─── 1. TESS ─────────────────────────────────────────────
print("\n--- Loading TESS ---")
for folder in os.listdir(TESS_PATH):
    folder_path = os.path.join(TESS_PATH, folder)
    if os.path.isdir(folder_path):
        emotion = folder.split("_")[-1].lower()
        # FIXED: normalize all surprise variants
        if emotion in ["pleasant_surprise", "surprised", "pleasant_surprised"]:
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
                    emotion = ravdess_emotions.get(parts[2])
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
            emotion = cremad_emotions.get(parts[2])
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
    'a':  'angry',   'd':  'disgust', 'f': 'fear',
    'h':  'happy',   'n':  'neutral', 'sa': 'sad', 'su': 'surprise'
}
count = 0
for speaker in os.listdir(SAVEE_PATH):
    speaker_path = os.path.join(SAVEE_PATH, speaker)
    if os.path.isdir(speaker_path):
        for file in os.listdir(speaker_path):
            if file.endswith(".wav"):
                name = file.replace(".wav", "")
                emotion = None
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
print(f"Unique emotions: {set(y)}")

# ─── Augmentation (IMPROVED) ─────────────────────────────
print("Adding augmented samples...")
X_aug, y_aug = [], []
for features, label in zip(X, y):
    X_aug.append(features)
    y_aug.append(label)
    # Noise augmentation
    noisy = features + np.random.normal(0, 0.02, features.shape)
    X_aug.append(noisy)
    y_aug.append(label)
    # Slight scaling augmentation
    scaled = features * np.random.uniform(0.9, 1.1)
    X_aug.append(scaled)
    y_aug.append(label)

X = np.array(X_aug)
y = np.array(y_aug)
print(f"Total samples after augmentation: {len(X)}")

# ─── Label Encoding ──────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\nEmotions found: {le.classes_}")
print(f"Total classes: {len(le.classes_)}")

# ─── Class Distribution ──────────────────────────────────
unique, counts = np.unique(y, return_counts=True)
print("\nClass distribution:")
for u, c in zip(unique, counts):
    print(f"  {u}: {c} samples")

# ─── Scaling ─────────────────────────────────────────────
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ─── Train Test Split ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# ─── Class Weights (FIXED — handles imbalance) ───────────
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_encoded),
    y=y_encoded
)
class_weight_dict = dict(enumerate(class_weights))
print(f"\nClass weights: {class_weight_dict}")

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
    class_weight=class_weight_dict,   # FIXED: class imbalance handle
    callbacks=[early_stop, reduce_lr]
)

# ─── Save Model ──────────────────────────────────────────
model.save("emotion_model.h5")
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("\n✅ Model saved!")

# ─── Final Accuracy ──────────────────────────────────────
final_acc = max(history.history['val_accuracy']) * 100
print(f"✅ Final Accuracy: {final_acc:.2f}%")

# ─── Training Curves ─────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['accuracy'],     label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

ax2.plot(history.history['loss'],     label='Train Loss')
ax2.plot(history.history['val_loss'], label='Val Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.close()
print("✅ Training curves saved: training_curves.png")

# ─── Confusion Matrix ────────────────────────────────────
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Confusion Matrix — Accuracy: {final_acc:.1f}%')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("✅ Confusion matrix saved: confusion_matrix.png")