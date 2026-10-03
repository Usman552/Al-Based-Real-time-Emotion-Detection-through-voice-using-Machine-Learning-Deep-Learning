"""
Feature extraction (run once; results are cached in cache_experiments/).

  python extract.py hand     # 284-D handcrafted, clean + 3 augmented copies per clip
  python extract.py logmel   # log-mel spectrograms for the CRNN baseline
  python extract.py wavlm    # WavLM-Base+ layer-wise mean/std pooled embeddings

Augmented copies are computed for every clip so that each experiment can pick
the copies of its own *training* clips only; test clips are always clean.
"""
import os, sys, time
import numpy as np
import librosa
from joblib import Parallel, delayed
from common import build_index, abspath, features_from_audio, augment_audio, preprocess, AUGS, CACHE

meta = build_index()
N = len(meta)
print(f"{N} clips, {meta.speaker.nunique()} speakers")


def _hand_one(i, rel):
    rng = np.random.default_rng(1000 + i)
    audio, sr = librosa.load(abspath(rel), sr=22050)
    clean = features_from_audio(audio, sr)
    aug = np.stack([features_from_audio(augment_audio(audio, sr, k, rng), sr) for k in AUGS])
    return clean, aug


def run_hand():
    out = os.path.join(CACHE, "hand.npz")
    if os.path.isfile(out):
        print("hand.npz exists"); return
    t0 = time.time()
    res = Parallel(n_jobs=7, verbose=5)(delayed(_hand_one)(i, p) for i, p in enumerate(meta.path))
    np.savez_compressed(out, clean=np.stack([r[0] for r in res]).astype(np.float32),
                        aug=np.stack([r[1] for r in res]).astype(np.float32))
    print(f"hand done in {time.time()-t0:.0f}s")


LOGMEL_FRAMES = 192  # ~4.5 s at sr=22050, hop=512


def _logmel_one(rel):
    audio, sr = librosa.load(abspath(rel), sr=22050)
    audio = preprocess(audio, sr)
    m = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64, hop_length=512))
    m = m[:, :LOGMEL_FRAMES]
    if m.shape[1] < LOGMEL_FRAMES:
        m = np.pad(m, ((0, 0), (0, LOGMEL_FRAMES - m.shape[1])), constant_values=m.min())
    return m.astype(np.float16)


def run_logmel():
    out = os.path.join(CACHE, "logmel.npy")
    if os.path.isfile(out):
        print("logmel.npy exists"); return
    res = Parallel(n_jobs=7, verbose=5)(delayed(_logmel_one)(p) for p in meta.path)
    np.save(out, np.stack(res))


def run_wavlm(name="microsoft/wavlm-base-plus", tag="wavlm", threads=8):
    import torch
    from transformers import WavLMModel
    torch.set_num_threads(threads)
    out = os.path.join(CACHE, f"{tag}.npy")
    part = os.path.join(CACHE, f"{tag}_partial.npy")
    model = WavLMModel.from_pretrained(name).eval()
    L = model.config.num_hidden_layers + 1
    H = model.config.hidden_size
    feats = np.load(part) if os.path.isfile(part) else np.full((N, L, 2 * H), np.nan, np.float16)
    todo = [i for i in range(N) if np.isnan(feats[i, 0, 0])]
    print(f"WavLM: {len(todo)} clips left")
    t0 = time.time()
    with torch.inference_mode():
        for n, i in enumerate(todo):
            audio, sr = librosa.load(abspath(meta.path[i]), sr=16000)
            audio = preprocess(audio, sr)
            x = torch.from_numpy(((audio - audio.mean()) / (audio.std() + 1e-7)).astype(np.float32))[None]
            hs = torch.stack(model(x, output_hidden_states=True).hidden_states)[:, 0]  # L x T x H
            feats[i] = torch.cat([hs.mean(1), hs.std(1)], -1).numpy().astype(np.float16)
            if (n + 1) % 250 == 0:
                np.save(part, feats)
                el = time.time() - t0
                print(f"  {n+1}/{len(todo)}  {el:.0f}s  eta {el/(n+1)*(len(todo)-n-1)/60:.0f} min", flush=True)
    np.save(out, feats)
    if os.path.isfile(part):
        os.remove(part)
    print("WavLM done")


if __name__ == "__main__":
    for stage in sys.argv[1:]:
        {"hand": run_hand, "logmel": run_logmel, "wavlm": run_wavlm,
         "wavlm_large": lambda: run_wavlm("microsoft/wavlm-large", "wavlm_large", threads=4)}[stage]()
