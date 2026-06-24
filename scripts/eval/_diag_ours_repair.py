"""Quick numeric diagnostic for the BrokenAMASS* repair results.

For each role (corrupted / gt / stablemotion / ours_strict / ours_combo) compute,
using the stored FK 'joints' (T,24,3) in each results entry:
  - jitter = mean L2 of 2nd-difference (acceleration) of joints  [lower=smoother]
  - MPJPE  = mean per-joint position error vs GT joints           [lower=closer]
This tells us objectively whether 'ours' is over-jittery / frozen / drifted.
"""
import sys
import numpy as np

SM = "ref_repo/StableMotion/output"
ROLES = {
    "corrupted":    (f"{SM}/brokenamass_star_sm_enhanced/results.npy", "motion"),
    "stablemotion": (f"{SM}/brokenamass_star_sm_enhanced/results.npy", "motion_fix"),
    "ours_strict":  (f"{SM}/brokenamass_star_ours_strict_sd/results.npy", "motion_fix"),
    "ours_combo":   (f"{SM}/brokenamass_star_ours_combo_all_self_t0_frame/results.npy", "motion_fix"),
    "ours_C":       (f"{SM}/brokenamass_star_ours_C_selfden/results.npy", "motion_fix"),
    "ours_final":   (f"{SM}/brokenamass_star_ours_final/results.npy", "motion_fix"),
    "gt":           (f"{SM}/brokenamass_star_clean_v2/results_collected.npy", "motion"),
}
# Optional extra roles from argv: name:path:key  (e.g. A:/tmp/ours_A/results.npy:motion_fix)
for a in sys.argv[1:]:
    if a.startswith("--max="):
        continue
    name, path, key = a.split(":")
    ROLES[name] = (path, key)
_MAX = 300
for a in sys.argv[1:]:
    if a.startswith("--max="):
        _MAX = int(a.split("=")[1])


def joints_of(entry, L):
    j = np.asarray(entry["joints"])[:L]          # (T,24,3)
    return j[:, :22]                              # SMPL-22


def jitter(j):                                    # (T,22,3)
    if j.shape[0] < 3:
        return np.nan
    acc = j[2:] - 2 * j[1:-1] + j[:-2]
    return float(np.linalg.norm(acc, axis=-1).mean())


def main():
    data = {}
    for r, (p, k) in ROLES.items():
        d = np.load(p, allow_pickle=True).item()
        data[r] = (d[k], np.asarray(d.get("lengths", [100] * len(d[k]))).reshape(-1))
    N = min(len(v[0]) for v in data.values())
    N = min(N, _MAX)
    print(f"N={N}")

    agg = {r: {"jit": [], "mpjpe": []} for r in ROLES}
    gt_motion = data["gt"][0]
    for i in range(N):
        L = int(min(data["corrupted"][1][i],
                    np.asarray(data["corrupted"][0][i]["joints"]).shape[0]))
        try:
            gj = joints_of(gt_motion[i], L)
        except Exception:
            continue
        for r in ROLES:
            try:
                rj = joints_of(data[r][0][i], L)
                T = min(rj.shape[0], gj.shape[0])
                agg[r]["jit"].append(jitter(rj))
                agg[r]["mpjpe"].append(
                    float(np.linalg.norm(rj[:T] - gj[:T], axis=-1).mean()) * 1000)
            except Exception as e:
                if i < 3:
                    print(f"  [{r}][{i}] {type(e).__name__}: {e}")
    print(f"\n{'role':14s} {'jit-mean':>9s} {'jit-p50':>9s} {'jit-p90':>9s} {'jit-max':>9s} {'MPJPE':>9s}")
    cj = np.array(agg["corrupted"]["jit"]) * 1000
    for r in ROLES:
        j = np.array(agg[r]["jit"]) * 1000
        mp = np.nanmean(agg[r]["mpjpe"])
        print(f"{r:14s} {np.nanmean(j):9.2f} {np.nanmedian(j):9.2f} "
              f"{np.nanpercentile(j,90):9.2f} {np.nanmax(j):9.2f} {mp:9.2f}")
    jf = np.array(agg["ours_final"]["jit"]) * 1000
    print(f"\nours_final vs corrupted per-clip jitter: "
          f"better in {(jf < cj).sum()}/{len(cj)} clips; "
          f"worse-by>2x in {(jf > 2*cj).sum()} clips")
    order = np.argsort(jf)
    good = order[:8]
    bad = order[::-1][:8]
    print("BEST (lowest jitter) cases:", [(int(i), round(float(jf[i]), 1)) for i in good])
    print("WORST (blow-up) cases:    ", [(int(i), round(float(jf[i]), 1)) for i in bad])


if __name__ == "__main__":
    sys.exit(main())
