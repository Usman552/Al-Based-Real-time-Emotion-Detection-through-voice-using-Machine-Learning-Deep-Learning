"""
Figures for the paper (written to results/experiments/figures).

  python make_figures.py diagrams   # Fig. 1, 2, 5, 6 (method diagrams, redrawn)
  python make_figures.py results    # Fig. 7-11 from the cached experiment runs
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
FIG = os.path.join(BASE, "results", "experiments", "figures"); os.makedirs(FIG, exist_ok=True)
RUNS = os.path.join(BASE, "results", "experiments", "runs")

GREEN, LIGHT, BLUE, ORANGE, EDGE, PURPLE = "#3d8b5c", "#d6e4f0", "#2e6da4", "#e07b39", "#1f4e79", "#6a51a3"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})


def box(ax, x, y, w, h, text, fc, tc="white", fs=11, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=EDGE, lw=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.3)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=2, mutation_scale=16))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ─── Fig. 1: overall framework ───────────────────────────────────────
# Final model: WavLM-Large, 25 hidden-state layers, mean+std pooling (2 x 1024), uniform layer average.
ENC, NL, DIM = "WavLM-Large", 25, 2048
FUSED = 284 + DIM


def fig1():
    fig, ax = plt.subplots(figsize=(15, 4.2)); ax.set_xlim(0, 15); ax.set_ylim(0, 4.2); ax.axis("off")
    box(ax, 0.1, 1.5, 1.9, 1.3, "Speech input\nfile / microphone", GREEN)
    box(ax, 2.5, 1.5, 1.9, 1.3, "Audio\npreprocessing", LIGHT, "black")
    box(ax, 5.0, 2.55, 2.4, 1.15, "Handcrafted\nfusion (284-D)", LIGHT, "black", 10.5)
    box(ax, 5.0, 0.55, 2.4, 1.15, f"{ENC}\n{NL} layers × {DIM}", LIGHT, "black", 10.5)
    box(ax, 8.0, 0.55, 2.0, 1.15, "Speaker norm.\n+ layer average", LIGHT, "black", 10.5)
    box(ax, 8.0, 2.55, 2.0, 1.15, "Speaker norm.\n+ z-score", LIGHT, "black", 10.5)
    box(ax, 10.6, 1.5, 2.0, 1.3, "Fusion DNN\n7-class softmax", BLUE)
    box(ax, 13.1, 1.5, 1.8, 1.3, "Emotion +\nconfidence", ORANGE)
    arrow(ax, 2.0, 2.15, 2.5, 2.15)
    arrow(ax, 4.4, 2.3, 5.0, 3.1); arrow(ax, 4.4, 2.0, 5.0, 1.1)
    arrow(ax, 7.4, 3.1, 8.0, 3.1); arrow(ax, 7.4, 1.1, 8.0, 1.1)
    arrow(ax, 10.0, 3.1, 10.6, 2.4); arrow(ax, 10.0, 1.1, 10.6, 1.9)
    arrow(ax, 12.6, 2.15, 13.1, 2.15)
    save(fig, "fig1_framework.png")


# ─── Fig. 2: data and evaluation pipeline ────────────────────────────
def fig2():
    fig, ax = plt.subplots(figsize=(13, 6.2)); ax.set_xlim(0, 13); ax.set_ylim(0, 6.2); ax.axis("off")
    for i, (n, s) in enumerate([("RAVDESS", "1,440 clips · 24 spk"), ("TESS", "2,800 clips · 2 spk"),
                                ("SAVEE", "480 clips · 4 spk"), ("CREMA-D subset", "2,585 clips · 32 spk")]):
        box(ax, 0.1, 4.9 - i * 1.35, 2.5, 1.05, f"{n}\n{s}", GREEN, fs=10)
        arrow(ax, 2.6, 5.42 - i * 1.35, 3.2, 3.5)
    box(ax, 3.2, 2.85, 2.3, 1.3, "Label unification\n7 classes\n7,305 clips · 62 spk", LIGHT, "black", 10)
    box(ax, 6.1, 4.1, 2.9, 1.3, "Speaker-independent\n5-fold split (by speaker)", BLUE, fs=10.5)
    box(ax, 6.1, 1.6, 2.9, 1.3, "Speaker-dependent\n5-fold stratified split", LIGHT, "black", 10.5)
    arrow(ax, 5.5, 3.6, 6.1, 4.6); arrow(ax, 5.5, 3.3, 6.1, 2.3)
    box(ax, 9.6, 4.6, 3.2, 1.2, "Training folds\n(augmentation, scaler fit,\nspeaker-disjoint validation)", LIGHT, "black", 9.5)
    box(ax, 9.6, 2.85, 3.2, 1.2, "Test fold\n(clean, unseen; used\nonly once for scoring)", ORANGE, fs=9.5)
    box(ax, 9.6, 0.6, 3.2, 1.55, "Proposed fusion model +\nML / DL baselines +\nablations on identical folds", BLUE, fs=9.5)
    arrow(ax, 9.0, 4.9, 9.6, 5.2); arrow(ax, 9.0, 4.6, 9.6, 3.45)
    arrow(ax, 9.0, 2.5, 9.6, 4.75); arrow(ax, 9.0, 2.2, 9.6, 3.1)
    arrow(ax, 11.2, 4.6, 11.2, 4.05); arrow(ax, 11.2, 2.85, 11.2, 2.15)
    save(fig, "fig2_pipeline.png")


# ─── Fig. 5: fusion strategy ─────────────────────────────────────────
def fig5():
    fig, ax = plt.subplots(figsize=(14, 7)); ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis("off")
    groups = [("MFCC mean+std", 80), ("Δ-MFCC mean+std", 80), ("ΔΔ-MFCC mean+std", 80), ("Chroma", 12),
              ("Mel-spectrogram", 20), ("Spectral contrast", 7), ("ZCR", 1), ("RMS", 1), ("Centroid/BW/Roll-off", 3)]
    for i, (n, d) in enumerate(groups):
        y = 6.35 - i * 0.48
        box(ax, 0.1, y, 3.0, 0.38, n, BLUE, fs=9, bold=False)
        ax.text(3.25, y + 0.19, str(d), va="center", fontsize=9.5)
        arrow(ax, 3.55, y + 0.19, 4.3, 4.65)
    box(ax, 4.3, 4.25, 1.9, 0.8, "Concatenate\n(⊕) → ℝ²⁸⁴", BLUE, fs=10)
    box(ax, 6.7, 4.25, 1.7, 0.8, "speaker norm.\n+ z-score", ORANGE, fs=10)
    arrow(ax, 6.2, 4.65, 6.7, 4.65)
    box(ax, 0.1, 0.6, 3.0, 1.2, f"{ENC} encoder\n(frozen, 16 kHz)", PURPLE, fs=10)
    box(ax, 3.6, 0.6, 2.6, 1.2, f"Per-layer mean+std\npooling: {NL} × {DIM}", BLUE, fs=10)
    box(ax, 6.7, 0.6, 1.7, 1.2, "speaker norm. +\nlayer average\n(1/L) Σ ẽₗ", ORANGE, fs=9.5)
    arrow(ax, 3.1, 1.2, 3.6, 1.2); arrow(ax, 6.2, 1.2, 6.7, 1.2)
    box(ax, 9.0, 2.45, 2.0, 1.2, "Concatenate\nℝ²⁸⁴ ⊕ ℝ²⁰⁴⁸\n= ℝ²³³²", GREEN, fs=10)
    arrow(ax, 8.4, 4.65, 9.0, 3.4); arrow(ax, 8.4, 1.2, 9.0, 2.7)
    box(ax, 11.6, 2.45, 2.2, 1.2, "Dense classifier\n(Fig. 6)", BLUE, fs=10)
    arrow(ax, 11.0, 3.05, 11.6, 3.05)
    save(fig, "fig5_fusion.png")


# ─── Fig. 6: classifier ──────────────────────────────────────────────
def fig6(params_text):
    fig, ax = plt.subplots(figsize=(15, 3.6)); ax.set_xlim(0, 15); ax.set_ylim(0, 3.6); ax.axis("off")
    labels = [(f"Input\n{FUSED} (284 + {DIM})", GREEN), ("Dense 512\nBN + Drop 0.4", BLUE), ("Dense 256\nBN + Drop 0.4", BLUE),
              ("Dense 128\nDrop 0.3", BLUE), ("Dense 64\nReLU", BLUE), ("Softmax\n7 classes", ORANGE)]
    for i, (t, c) in enumerate(labels):
        box(ax, 0.1 + i * 2.5, 1.2, 2.0, 1.3, t, c, fs=10.5)
        if i:
            arrow(ax, i * 2.5 - 0.4, 1.85, i * 2.5 + 0.1, 1.85)
    ax.text(7.5, 0.55, params_text, ha="center", fontsize=10, style="italic", color="#444")
    save(fig, "fig6_network.png")


def diagrams():
    fig1(); fig2(); fig5()
    try:
        sys.path.insert(0, HERE)
        from run_experiments import build_net
        n = build_net({"hand": True, "ssl": "avg"}, {"hand": (284,), "ssl": (DIM,)}).count_params()
        fig6(f"≈ {n/1e6:.2f} M trainable classifier parameters (+ frozen 315 M {ENC} encoder) · focal loss (γ = 2) · Adam")
    except Exception as e:
        print("fig6 param count failed:", e)
        fig6("focal loss (γ = 2) · Adam")


# ─── result figures ──────────────────────────────────────────────────
def load(tag):
    d = np.load(os.path.join(RUNS, tag + ".npz"))
    return d["prob"], d["fold"], json.loads(str(d["info"]))


def results(tag="SI__fusion_focal_spk", lw_tag="SI__fusion_focal_spk"):
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    sys.path.insert(0, HERE)
    from common import CLASSES, build_index
    meta = build_index()
    y = meta.label.map({c: i for i, c in enumerate(CLASSES)}).values
    prob, fold, infos = load(tag)
    m = fold >= 0; yt, P = y[m], prob[m]; yp = P.argmax(1)
    names = [c.capitalize() for c in CLASSES]

    # training curves of fold 1
    h = infos[0]["history"]; ep = np.arange(1, len(h["accuracy"]) + 1)
    for key, vkey, ylab, fn, title in [("accuracy", "val_accuracy", "Accuracy (%)", "fig7_accuracy.png", "Training and Validation Accuracy"),
                                       ("loss", "val_loss", "Loss (focal)", "fig8_loss.png", "Training and Validation Loss")]:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        s = 100 if key == "accuracy" else 1
        ax.plot(ep, np.array(h[key]) * s, color=EDGE, lw=2, label="Training")
        ax.plot(ep, np.array(h[vkey]) * s, "--", color=ORANGE, lw=2, label="Validation (held-out speakers)")
        best = int(np.argmax(h["val_accuracy"])) + 1
        ax.axvline(best, color="grey", ls=":", lw=1.5, label=f"Best epoch ({best}, weights restored)")
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylab); ax.set_title(title, fontweight="bold"); ax.grid(alpha=.3); ax.legend()
        save(fig, fn)

    # confusion matrix
    cm = confusion_matrix(yt, yp, labels=range(len(CLASSES)))
    cmn = cm / cm.sum(1, keepdims=True) * 100
    acc = (yt == yp).mean() * 100
    fig, ax = plt.subplots(figsize=(7.5, 6.3))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=100)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cmn[i, j]:.1f}\n({cm[i, j]})", ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > 55 else "black")
    ax.set_xticks(range(7)); ax.set_xticklabels(names, rotation=40, ha="right"); ax.set_yticks(range(7)); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_title(f"Speaker-independent confusion matrix (accuracy = {acc:.2f}%)", fontweight="bold", fontsize=11)
    fig.colorbar(im, ax=ax, label="Row-normalized (%)", fraction=0.046)
    save(fig, "fig9_confusion.png")

    # ROC
    fig, ax = plt.subplots(figsize=(7, 6))
    cols = plt.cm.tab10(np.arange(7))
    for c in range(7):
        fpr, tpr, _ = roc_curve(yt == c, P[:, c])
        ax.plot(fpr, tpr, color=cols[c], lw=1.6, label=f"{names[c]} (AUC = {auc(fpr, tpr):.3f})")
    Y = np.eye(7)[yt]; fpr, tpr, _ = roc_curve(Y.ravel(), P.ravel())
    ax.plot(fpr, tpr, "k:", lw=2.2, label=f"Micro-avg (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=.8)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest), speaker-independent", fontweight="bold"); ax.legend(loc="lower right", fontsize=8.5); ax.grid(alpha=.3)
    save(fig, "fig10_roc.png")

    # layer weights
    lw = np.mean([i["layer_weights"] for i in load(lw_tag)[2] if "layer_weights" in i], 0)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(range(len(lw)), lw, color=BLUE, edgecolor=EDGE)
    ax.set_xlabel("WavLM layer (0 = CNN feature encoder output)"); ax.set_ylabel("Learned weight")
    ax.set_title("Learned WavLM layer weights (mean over 5 SI folds)", fontweight="bold"); ax.set_xticks(range(len(lw))); ax.grid(axis="y", alpha=.3)
    save(fig, "fig11_layer_weights.png")
    print("result figures written to", FIG)


if __name__ == "__main__":
    if sys.argv[1] == "results" and len(sys.argv) > 2:
        results(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else sys.argv[2])
    else:
        for a in sys.argv[1:]:
            {"diagrams": diagrams, "results": results}[a]()
