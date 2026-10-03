"""
Real-time / deployment measurements: per-stage CPU latency, peak memory,
model sizes and real-time factor on 200 randomly chosen clips, for the handcrafted
pipeline and for the fusion pipeline with WavLM-Base+ and (if cached) WavLM-Large.
"""
import os, time, platform, json, subprocess
import numpy as np
import pandas as pd
import librosa
import psutil
import torch
from transformers import WavLMModel
from common import build_index, abspath, features_from_audio, preprocess, RESULTS, CACHE
from run_experiments import build_net

torch.set_num_threads(os.cpu_count())
proc = psutil.Process()
meta = build_index()
rng = np.random.default_rng(0)
sample = meta.path.values[rng.choice(len(meta), 200, replace=False)]

try:
    cpu = subprocess.check_output(["powershell", "-NoProfile", "-Command",
                                   "(Get-CimInstance Win32_Processor).Name"], text=True).strip()
except Exception:
    cpu = platform.processor()

encoders = [("WavLM-Base+", "microsoft/wavlm-base-plus")]
if os.path.isfile(os.path.join(CACHE, "wavlm_large.npy")):
    encoders.append(("WavLM-Large", "microsoft/wavlm-large"))

# audio + handcrafted timings (shared)
audio, t_load, t_hand, durs = [], [], [], []
for rel in sample:
    t0 = time.perf_counter(); a22, sr = librosa.load(abspath(rel), sr=22050)
    a16 = librosa.resample(a22, orig_sr=22050, target_sr=16000); t1 = time.perf_counter()
    f = features_from_audio(a22, sr)[None].astype(np.float32); t2 = time.perf_counter()
    audio.append((a16, f)); t_load.append(t1 - t0); t_hand.append(t2 - t1); durs.append(len(a22) / sr)

hand_head = build_net({"hand": True}, {"hand": (284,)})
hand_head(np.zeros((1, 284), np.float32), training=False)
t_hh = []
for _, f in audio:
    t0 = time.perf_counter(); hand_head(f, training=False); t_hh.append(time.perf_counter() - t0)

rows = [{"pipeline": "Handcrafted 284-D + DNN", "load_ms": np.mean(t_load) * 1e3, "features_ms": np.mean(t_hand) * 1e3,
         "encoder_ms": 0.0, "classifier_ms": np.mean(t_hh) * 1e3,
         "total_ms": (np.mean(t_load) + np.mean(t_hand) + np.mean(t_hh)) * 1e3,
         "p95_total_ms": np.percentile(np.array(t_load) + np.array(t_hand) + np.array(t_hh), 95) * 1e3,
         "encoder_params_M": 0.0, "classifier_params_M": hand_head.count_params() / 1e6}]

for name, hub in encoders:
    enc = WavLMModel.from_pretrained(hub).eval()
    L, H = enc.config.num_hidden_layers + 1, enc.config.hidden_size
    head = build_net({"hand": True, "ssl": "layers"}, {"hand": (284,), "ssl": (L, 2 * H)})
    head([np.zeros((1, 284), np.float32), np.zeros((1, L, 2 * H), np.float32)], training=False)
    t_enc, t_head = [], []
    with torch.inference_mode():
        for a16, f in audio:
            t0 = time.perf_counter()
            a = preprocess(a16, 16000)
            x = torch.from_numpy(((a - a.mean()) / (a.std() + 1e-7)).astype(np.float32))[None]
            hs = torch.stack(enc(x, output_hidden_states=True).hidden_states)[:, 0]
            e = torch.cat([hs.mean(1), hs.std(1)], -1).numpy()[None]
            t1 = time.perf_counter(); head([f, e], training=False); t2 = time.perf_counter()
            t_enc.append(t1 - t0); t_head.append(t2 - t1)
    tot = np.array(t_load) + np.array(t_hand) + np.array(t_enc) + np.array(t_head)
    rows.append({"pipeline": f"{name} + 284-D fusion DNN", "load_ms": np.mean(t_load) * 1e3, "features_ms": np.mean(t_hand) * 1e3,
                 "encoder_ms": np.mean(t_enc) * 1e3, "classifier_ms": np.mean(t_head) * 1e3,
                 "total_ms": tot.mean() * 1e3, "p95_total_ms": np.percentile(tot, 95) * 1e3,
                 "encoder_params_M": sum(p.numel() for p in enc.parameters()) / 1e6,
                 "classifier_params_M": head.count_params() / 1e6})
    del enc

df = pd.DataFrame(rows)
df["rtf"] = df["total_ms"] / (np.mean(durs) * 1e3)
df["throughput_clips_s"] = 1000 / df["total_ms"]
df["model_size_MB_fp32"] = (df["encoder_params_M"] + df["classifier_params_M"]) * 4
info = {"cpu": cpu, "logical_cores": os.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "gpu": "none (CPU only)", "os": platform.platform(), "mean_clip_s": float(np.mean(durs)),
        "peak_rss_mb": proc.memory_info().rss / 2**20, "n_clips": len(sample),
        "torch": torch.__version__, "note": "batch size 1, single process, consumer laptop CPU"}
df.round(2).to_csv(os.path.join(RESULTS, "latency.csv"), index=False)
json.dump(info, open(os.path.join(RESULTS, "latency_info.json"), "w"), indent=2)
print(df.round(2).to_string(index=False)); print(json.dumps(info, indent=2))
