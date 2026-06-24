import os, sys, glob
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, ".")
sys.path.insert(0, "scripts/eval")
import multiprocessing as mp
import numpy as np

SRC = os.environ.get("SRC", "output/evaluation/mib_ms272_ikfix/gtctrl/repack272")
OUT = os.environ.get("OUT", "output/evaluation/mib_ms272_ikfix/gtctrl/gt272ref")


def _work(f):
    from motionstreamer_272_encoder import motion135_to_272
    o = os.path.join(OUT, os.path.basename(f))
    if os.path.exists(o):
        return "skip"
    try:
        m135 = np.load(f)["motion_135"]
        m272 = motion135_to_272(m135, rotation_space="local")
        np.savez(o, motion_272=m272.astype(np.float32))
        return "ok"
    except Exception as e:
        return f"fail:{os.path.basename(f)}:{str(e)[:60]}"


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fs = sorted(glob.glob(SRC + "/*.npz"))
    print("building", len(fs), "gtctrl-272 ref files", flush=True)
    ok = skip = fail = 0
    with mp.Pool(32) as p:
        for i, r in enumerate(p.imap_unordered(_work, fs, chunksize=8), 1):
            if r == "ok":
                ok += 1
            elif r == "skip":
                skip += 1
            else:
                fail += 1
                if fail <= 5:
                    print(" ", r, flush=True)
            if i % 300 == 0:
                print(f"  {i}/{len(fs)} ok={ok} skip={skip} fail={fail}", flush=True)
    print(f"DONE ok={ok} skip={skip} fail={fail} -> {OUT}", flush=True)
