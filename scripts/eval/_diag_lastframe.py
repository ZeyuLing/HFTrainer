"""Quantify the last-frame jump for ours vs corrupted/gt.

For each clip: step_med = median_t ||j[t]-j[t-1]|| (typical motion per frame),
jump_last = ||j[L-1]-j[L-2]||. A clean clip has jump_last ~ step_med; a
last-frame teleport shows jump_last >> step_med.
"""
import sys
import numpy as np

SM = "ref_repo/StableMotion/output"
ROLES = {
    "corrupted":  (f"{SM}/brokenamass_star_sm_enhanced/results.npy", "motion"),
    "ours_final": (f"{SM}/brokenamass_star_ours_final/results.npy", "motion_fix"),
    "gt":         (f"{SM}/brokenamass_star_clean_v2/results_collected.npy", "motion"),
}


def main():
    data = {r: np.load(p, allow_pickle=True).item() for r, (p, k) in ROLES.items()}
    lens = np.asarray(data["corrupted"]["lengths"]).reshape(-1)
    N = min(len(data[r][k]) for r, (p, k) in ROLES.items())
    print(f"N={N}")
    print(f"{'role':12s} {'jumpLast(mm)':>13s} {'stepMed(mm)':>12s} {'ratio':>7s} "
          f"{'#ratio>3':>9s} {'#ratio>10':>10s}")
    for r, (p, k) in ROLES.items():
        d = data[r][k]
        jl, sm_, ratios = [], [], []
        n3 = n10 = 0
        for i in range(N):
            L = int(min(lens[i], np.asarray(d[i]["joints"]).shape[0]))
            if L < 4:
                continue
            j = np.asarray(d[i]["joints"])[:L, :22]      # (L,22,3)
            step = np.linalg.norm(np.diff(j, axis=0), axis=-1).mean(-1)  # (L-1,)
            sm_med = np.median(step) * 1000
            jlast = step[-1] * 1000
            jl.append(jlast); sm_.append(sm_med)
            rt = jlast / (sm_med + 1e-6)
            ratios.append(rt)
            n3 += int(rt > 3); n10 += int(rt > 10)
        print(f"{r:12s} {np.mean(jl):13.2f} {np.mean(sm_):12.2f} "
              f"{np.mean(ratios):7.2f} {n3:9d} {n10:10d}")


if __name__ == "__main__":
    sys.exit(main())
