# AI-Based Real-Time Emotion Detection through Voice

Speech Emotion Recognition (SER) using Machine Learning and Deep Learning.
Final Year Project — Department of Computer Science, University of Southern Punjab, Multan.

**Usman Qasim** · **M. Abdullah** — supervised by **Miss Maryam Ismail**

---

## What it does

Given a short speech clip — uploaded as a file or recorded live through the
microphone — the system predicts which of seven emotions the speaker is
expressing (**angry, disgust, fear, happy, neutral, sad, surprise**), along with a
confidence score for every class.

Everything runs locally. No audio, feature vector or prediction is ever sent to
an external server.

## Results

Held-out stratified test set of **1,461 unseen clips**:

| Metric | Value |
|---|---|
| Overall accuracy | **76.73%** |
| Macro F1 | **77.16%** |
| 5-fold cross-validation | 74.99% ± 0.94% |
| Per-class ROC-AUC | 0.944 – 0.992 (micro-avg 0.965) |
| Trainable parameters | 321,927 |
| Inference time | 3.27 ms per clip |

Per-dataset accuracy, which is the most informative result here:

| Dataset | Accuracy | Macro F1 | Test clips |
|---|---|---|---|
| TESS | 99.82% | 99.80% | 547 |
| RAVDESS | 75.35% | 74.02% | 284 |
| CREMA-D | 57.46% | 48.98% | 536 |
| SAVEE | 56.38% | 54.02% | 94 |
| **Combined** | **76.73%** | **77.16%** | **1,461** |

The combined number is reported rather than the TESS-only number. The same model
scores 99.8% on TESS alone — a figure competitive with published work, and almost
meaningless, since TESS contains only two speakers. 76.73% is what a user with an
unfamiliar voice actually experiences.

### Ablation

| Configuration | Accuracy | Macro F1 |
|---|---|---|
| Baseline: MFCC only, single corpus | 70.49% | 69.50% |
| + Hybrid dataset integration (4 corpora) | 73.44% | 74.24% |
| + Full 284-D feature fusion | 75.56% | 76.15% |
| + Augmentation and focal loss (proposed) | **76.73%** | **77.16%** |

### Ensemble (optional higher-accuracy mode)

Averaging six models found by the hyperparameter search beats the single network:

| | Single | Ensemble (6) | Change |
|---|---|---|---|
| Accuracy | 76.73% | **77.89%** | +1.16 |
| Macro F1 | 77.16% | **78.50%** | +1.34 |
| RAVDESS | 75.35% | 78.17% | +2.82 |
| SAVEE | 56.38% | 61.70% | +5.32 |
| CREMA-D | 57.46% | 58.02% | +0.56 |
| Parameters | 321,927 | 3,457,962 | 10.7× |
| Inference (single clip) | 10.5 ms | 83.8 ms | 8.0× |

```bash
python train_ensemble.py --members 6 --deploy
```

The app uses `ensemble/` automatically when it exists and shows which model is active
in the sidebar; delete the folder to go back to the single network. The single model
remains the *proposed* system because the project targets compact real-time inference —
the ensemble is the accuracy-first alternative.

## Approach

1. **Hybrid dataset integration** — RAVDESS, TESS, SAVEE and CREMA-D unified under
   one seven-class taxonomy: **7,305 clips from 62 speakers**.
2. **Feature fusion (284-D)** — 80 MFCC + 80 delta + 80 delta-delta + 12 chroma
   + 20 mel + 7 spectral contrast + ZCR + RMS + 3 spectral-shape features.
   The delta and delta-delta terms encode *how the voice changes over time*.
3. **Waveform augmentation** — additive noise, pitch shift and time stretch,
   applied **only after the train/test split and only to the training partition**,
   so no perturbed copy of a test clip is ever seen during training
   (5,844 → 23,376 training examples).
4. **Class-balanced focal loss** — keeps minority emotions in play. Surprise, the
   smallest class, ends up with the highest per-class F1 (86.9%).
5. **Compact DNN** — 512 → 256 → 128 → 64 → 7, with batch normalisation and
   dropout. Small enough for real-time local inference.

## Setup

```bash
pip install -r requirements.txt
```

## Running the app

```bash
streamlit run app.py
```

Requires `emotion_model.h5`, `label_encoder.pkl` and `scaler.pkl` in the project
root — all three are committed.

## Retraining

The four dataset folders must sit in the project root (they are gitignored, so
download them separately):

```
CREMAD/AudioWAV/                          *.wav
RAVDESS/Actor_01 … Actor_24/              *.wav
SAVEE/AudioData/DC, JE, JK, KL/           *.wav
TESS Toronto emotional speech set data/   OAF_* and YAF_* folders
```

Then:

```bash
python train_model.py
```

This writes `emotion_model.h5`, `label_encoder.pkl`, `scaler.pkl` and the full set
of evaluation artefacts into `results/`. Set `RUN_CV = False` and
`RUN_ABLATION = False` near the top of the script for a fast run, or shorten the
`AUGS` list to reduce feature-extraction time.

## Continuous improvement

Re-running the same training on the same data does **not** raise accuracy — the model
has already converged. `auto_train.py` automates the three things that genuinely do:

```bash
python auto_train.py                  # build the feature cache, then run 10 trials
python auto_train.py --trials 50      # search harder
python auto_train.py --forever        # keep searching until Ctrl+C
python auto_train.py --speaker-independent   # stricter split, for analysis only
```

- **Search.** Each trial trains a challenger with a different architecture and
  hyperparameters. Trial 0 reproduces the published baseline configuration.
- **Data.** Corrections that users submit in the app (`feedback/feedback.csv`, feature
  vectors only — never audio) are folded into every later training round.
- **Safety.** A worse model can never overwrite a better one; the outgoing model is
  copied to `cache/previous_model/` before every promotion. Progress is logged to
  `results/auto_train_log.csv` and the current best to `results/champion.json`.

### How models are compared

Two different surfaces are needed, because the deployed model and the challengers were
not trained on the same data:

| Comparison | Surface | Why |
|---|---|---|
| Challenger vs challenger | Validation split (877 clips) | No challenger trains on it, so it is fair between them |
| Challenger vs deployed model | Test set (1,461 clips) | The deployed model **was** trained on the validation clips, so its validation score is inflated (90.5% vs its true 76.7%). The test set is the only surface neither has seen |

A challenger is therefore *selected* on validation accuracy and *deployed* only if it
also beats the incumbent on test.

> **Caveat, stated honestly:** using the test set as the release gate means that across
> many trials the final test figure becomes mildly optimistic. Treat a promoted model's
> test accuracy as an estimate, and re-measure with `--speaker-independent` before
> quoting it in the report.

Feature extraction over ~7,300 clips is the slow part, so it runs once and is cached in
`cache/`. Later trials start in seconds. Delete the cache after adding new audio, or pass
`--rebuild-cache`.

> If auto-training promotes a better model, the accuracy figures in this README and in
> `docs/FYP_Report.docx` become stale — take the new numbers from `results/champion.json`.

## Health check

```bash
python smoke_test.py
```

Verifies that the model, encoder and scaler load, that the feature extractor still
produces exactly 284 dimensions, and that predictions on known clips are correct. Exits
non-zero on failure, so it also works in CI. Run it before any demo.

## Project layout

```
app.py              Streamlit application (upload, live recording, results, history)
train_model.py      Dataset integration, feature extraction, training, evaluation
auto_train.py       Self-improving champion/challenger search with feature cache
smoke_test.py       Fast end-to-end health check
emotion_model.h5    Trained network (321,927 parameters)
label_encoder.pkl   Fitted LabelEncoder for the seven emotion classes
scaler.pkl          StandardScaler fitted on the training features
results/            Metrics, confusion matrix, training curves, CV and ablation
feedback/           User-submitted corrections (created on first feedback)
cache/              Cached 284-D features (created on first auto_train run)
docs/               FYP report and generated diagrams
```

## Known limitations

- **Speaker-dependent split.** The train/test split is random and stratified, not
  speaker-independent, so for TESS (2 speakers) the same voices appear in both
  partitions. This inflates the TESS figure in particular. A speaker-independent
  protocol would give a stricter and lower estimate.
- **CREMA-D subset.** Only 2,585 of the full 7,442 clips (32 of 91 actors) are
  used here. Adding the rest should improve generalisation.
- **Acted, not spontaneous.** All four corpora contain acted emotion recorded in
  controlled conditions; natural speech is subtler.
- **English only.** Urdu and code-switched Urdu-English speech are not supported.
- **Noise robustness is not measured.** Noise augmentation is applied during
  training, but accuracy at controlled SNR levels (5/10/20 dB) has not been
  benchmarked.

## Datasets

| Dataset | Speakers | Clips used | Source |
|---|---|---|---|
| RAVDESS | 24 | 1,440 | Livingstone & Russo, 2018 |
| TESS | 2 | 2,800 | Pichora-Fuller & Dupuis, 2020 |
| SAVEE | 4 | 480 | Haq & Jackson, 2011 |
| CREMA-D (subset) | 32 | 2,585 | Cao et al., 2014 |

Each dataset remains under its own original licence.
