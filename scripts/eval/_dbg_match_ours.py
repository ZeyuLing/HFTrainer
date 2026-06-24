import glob, os, multiprocessing as mp
import numpy as np

SRC = sorted(glob.glob('data/eval/h3d_editing/source_npz/*.npz'))
OURS = sorted(glob.glob('output/evaluation/ours_mib_full_cfg20/shard_*/smpl_caption_editfix_latest/E2_both_1f/npz/*.npz'))

def src_feat(p):
    z = np.load(p, allow_pickle=True)
    m = np.asarray(z['motion_135'], np.float32)
    sid = str(z['source_id'])
    return (sid, m.shape[0], m[0].copy(), m[-1].copy())

def ours_feat(p):
    z = np.load(p, allow_pickle=True)
    g = np.asarray(z['gt_motion_135'], np.float32)
    return (p, g.shape[0], g[0].copy(), g[-1].copy(), str(z['caption']))

if __name__ == '__main__':
    with mp.Pool(32) as pool:
        S = pool.map(src_feat, SRC, chunksize=16)
        O = pool.map(ours_feat, OURS, chunksize=16)
    sids = [s[0] for s in S]
    Ts = np.array([s[1] for s in S])
    S0 = np.stack([s[2] for s in S]); SL = np.stack([s[3] for s in S])
    print('built source table', S0.shape, flush=True)
    # nearest by first+last frame
    dists = []
    badT = 0
    rng = np.random.default_rng(0)
    sample = rng.choice(len(O), size=min(200, len(O)), replace=False)
    matched_T = 0
    for i in sample:
        p, T, o0, oL, cap = O[i]
        d = np.abs(S0 - o0[None]).mean(1) + np.abs(SL - oL[None]).mean(1)
        j = int(np.argmin(d))
        dists.append(float(d[j]))
        if Ts[j] == T: matched_T += 1
        else: badT += 1
    dists = np.array(dists)
    print(f'sample={len(sample)} nn-dist min={dists.min():.5f} med={np.median(dists):.5f} '
          f'p90={np.percentile(dists,90):.5f} max={dists.max():.5f}')
    print(f'length-consistent={matched_T} length-mismatch={badT}')
