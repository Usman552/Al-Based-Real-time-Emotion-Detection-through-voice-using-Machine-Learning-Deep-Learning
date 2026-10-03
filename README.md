# AI-Based Real-Time Emotion Detection through Voice

Speech emotion recognition (SER) on four English corpora — RAVDESS, TESS, SAVEE and a
2,585-clip subset of CREMA-D — unified into seven classes (angry, disgust, fear, happy,
neutral, sad, surprise): **7,305 utterances from 62 speakers**.

The repository contains

- `app.py` — a Streamlit app that classifies an uploaded file or a microphone recording;
- `train_model.py` and helpers — the 284-D handcrafted-feature model used by the app;
- `experiments/` — the cross-validated experiments reported in the paper, including a
  speaker-independent protocol, baselines, ablations and latency measurements.

## Results

Five-fold cross-validation, mean ± std over folds. **Speaker-independent (SI)**: no speaker
appears in both training and test folds. **Speaker-dependent (SD)**: random stratified folds.

| Model | SI accuracy | SI macro-F1 | SD accuracy |
|---|---|---|---|
| Handcrafted 284-D + DNN | 48.80 ± 3.52 | 47.98 ± 3.03 | 73.63 ± 0.76 |
| Best classical model on 284-D (gradient boosting) | 52.90 ± 2.60 | 51.71 ± 2.81 | 75.96 ± 0.67 |
| CRNN on log-mel spectrograms | 48.55 ± 1.65 | 47.40 ± 2.12 | — |
| SVM on WavLM-Large + speaker normalization | 84.83 ± 5.07 | 84.59 ± 5.49 | 90.95 ± 0.69 |
| **284-D + WavLM-Large fusion + speaker normalization (final)** | **84.87 ± 3.97** | **84.55 ± 4.28** | **91.12 ± 0.79** |
| Final model, speaker statistics from 20 utterances | 81.25 ± 4.18 | 80.96 ± 4.57 | 89.75 ± 0.87 |

What the experiments show:

- A random split overstates generalization: the handcrafted model drops from 73.6% (SD) to
  48.8% (SI). Within TESS (two speakers), SD accuracy is 99.9% but leave-one-speaker-out is 69.1%.
- Most of the SI accuracy comes from the frozen self-supervised encoder (WavLM) and from
  speaker-level feature normalization (+6–7 points). The classifier type and the handcrafted
  features matter much less once both are used.
- On a laptop CPU (Intel i5-8350U, no GPU) the final pipeline takes about 1.05 s for a 2.75 s
  clip (real-time factor 0.38); the handcrafted-only pipeline takes about 0.1 s.

All numbers are in `results/experiments/all_runs.csv`; per-fold predictions are in
`results/experiments/runs/` and the fold assignment of every file in
`results/experiments/folds.csv`.

## Setup

```bash
pip install -r requirements.txt
# for the experiments additionally:
pip install torch transformers joblib psutil
```

### Datasets

The audio is not included in this repository; download each corpus from its original source
and place it in the project root:

```
CREMAD/AudioWAV/*.wav                        (actors 1001–1032 were used)
RAVDESS/Actor_01 … Actor_24/*.wav
SAVEE/AudioData/DC, JE, JK, KL/*.wav
TESS Toronto emotional speech set data/OAF_* and YAF_* folders
```

## Running the app

```bash
streamlit run app.py
```

The app uses the handcrafted-feature model (`emotion_model.h5`, `scaler.pkl`,
`label_encoder.pkl`), which is fast but, as the table shows, much less accurate on new
speakers than the WavLM-based model.

## Reproducing the experiments

```bash
cd experiments
python extract.py hand logmel wavlm wavlm_large    # features, cached in cache_experiments/
python run_experiments.py main extra large         # proposed model and ablations
python run_experiments.py baselines large_baselines corpus
python run_experiments.py folds tables
python benchmark_latency.py
python make_figures.py diagrams
python make_figures.py results SI__fusionL_avg_focal_spk SI__fusionL_focal_spk
```

Feature extraction takes about 1 h (WavLM-Base+) and 3 h (WavLM-Large) on a laptop CPU.
Every run is cached, so the scripts can be interrupted and restarted.

## Project layout

```
app.py                     Streamlit app
train_model.py             trains the handcrafted model used by the app
auto_train.py              hyperparameter search for the handcrafted model
train_ensemble.py          soft-voting ensemble of the best handcrafted models
smoke_test.py              checks that the saved model loads and predicts
experiments/
  common.py                dataset index with speaker IDs, 284-D features, augmentation
  extract.py               handcrafted, log-mel and WavLM feature extraction
  run_experiments.py       SI / SD / within-corpus cross-validation, baselines, ablations
  benchmark_latency.py     CPU latency, memory and model size
  make_figures.py          figures
results/experiments/       all_runs.csv, folds.csv, latency, figures, per-run predictions
```

## Limitations

- Only 32 of the 91 CREMA-D actors were used.
- All corpora contain acted English speech recorded in quiet conditions.
- Speaker normalization needs some unlabeled speech from the user; with 20 utterances part
  of the gain is lost.
- The WavLM-Large encoder (315 M parameters) makes the full pipeline much larger and slower
  than the handcrafted model.

## Authors

Usman Qasim and M. Abdullah, Department of Computer Science, University of Southern Punjab,
Multan. Supervised by Maryam Ismail and co-supervised by Muhammad Imran Ali.
