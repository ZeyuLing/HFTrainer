#!/usr/bin/env python3
"""Analyze FBX orig vs cleaned pairs using multiprocessing.

Computes bone world-position differences at FBX level (before SMPL retargeting).
"""
import fbx
import numpy as np
from pathlib import Path
import json, time, sys
from multiprocessing import Pool, cpu_count

FBX_DIR = Path('data/lightai_data/CJGame_MB/raw')

# Skeleton bone names (standard humanoid chain). Only these matter for visual diff.
SKELETON_BONES = {
    'hips','spine','spine1','spine2','spine3',
    'neck','neck1','head','headend',
    'leftshoulder','leftarm','leftforearm','lefthand',
    'rightshoulder','rightarm','rightforearm','righthand',
    'leftupleg','leftleg','leftfoot','lefttoebase','lefttoebaseend',
    'rightupleg','rightleg','rightfoot','righttoebase','righttoebaseend',
}

FINGER_KW = ['thumb','index','middle','ring','pinky','finger','metacarpal','distal','proximal','intermediate']

def classify_bone(name):
    """Return 'skeleton', 'finger', or 'marker'."""
    low = name.lower()
    if low in SKELETON_BONES:
        return 'skeleton'
    if any(k in low for k in FINGER_KW):
        return 'finger'
    # Everything else (mocap markers, helpers, etc.) = marker
    return 'marker'

def analyze_one(base, fps=30):
    """Analyze a single pair. Each call creates its own FBX manager (process-safe)."""
    manager = fbx.FbxManager.Create()
    ios = fbx.FbxIOSettings.Create(manager, "IOSRoot")
    manager.SetIOSettings(ios)
    importer = fbx.FbxImporter.Create(manager, "")

    try:
        orig_path = FBX_DIR / f'{base}.fbx'
        clean_path = FBX_DIR / f'{base}_cleaned.fbx'

        all_pos = []
        all_names = []
        for path in [orig_path, clean_path]:
            if not importer.Initialize(str(path), -1, manager.GetIOSettings()):
                return None
            scene = fbx.FbxScene.Create(manager, "s")
            importer.Import(scene)

            stack = scene.GetCurrentAnimationStack()
            if not stack:
                scene.Destroy(); return None

            ts = stack.GetLocalTimeSpan()
            start = ts.GetStart().GetSecondDouble()
            stop = ts.GetStop().GetSecondDouble()
            n_frames = int(round((stop - start) * fps))

            bones = []; names = []
            def collect(node):
                name = node.GetName()
                short = name.split(':')[-1] if ':' in name else name
                if short and short != 'RootNode':
                    bones.append(node); names.append(short)
                for i in range(node.GetChildCount()):
                    collect(node.GetChild(i))
            collect(scene.GetRootNode())

            positions = np.zeros((n_frames, len(bones), 3), dtype=np.float64)
            for fi in range(n_frames):
                t = fbx.FbxTime()
                t.SetSecondDouble(start + fi / fps)
                for bi, bone in enumerate(bones):
                    gt = bone.EvaluateGlobalTransform(t)
                    tr = gt.GetT()
                    positions[fi, bi] = [tr[0], tr[1], tr[2]]

            all_pos.append(positions)
            all_names.append(names)
            scene.Destroy()

        pos_o, pos_c = all_pos
        names_o, names_c = all_names
        fo, fc = pos_o.shape[0], pos_c.shape[0]
        length_changed = fo != fc
        n = min(fo, fc)

        common = [nm for nm in names_o if nm in names_c]
        idx_o = [names_o.index(nm) for nm in common]
        idx_c = [names_c.index(nm) for nm in common]

        dist = np.sqrt(np.sum((pos_o[:n, idx_o] - pos_c[:n, idx_c])**2, axis=-1)) * 10  # mm

        # Classify bones
        bone_classes = [classify_bone(nm) for nm in common]
        skel_mask = np.array([c == 'skeleton' for c in bone_classes])
        finger_mask = np.array([c == 'finger' for c in bone_classes])

        skel_dist = dist[:, skel_mask] if skel_mask.any() else np.zeros((n, 1))
        finger_dist = dist[:, finger_mask] if finger_mask.any() else np.zeros((n, 1))

        # Use median of per-frame max skeleton distance as the main metric
        # Median is robust against spike segments (even 10-20% of frames can be outliers)
        per_frame_max_skel = np.max(skel_dist, axis=1)  # (n,)
        median_body = float(np.median(per_frame_max_skel))
        p95_body = float(np.percentile(per_frame_max_skel, 95))
        max_body = float(np.max(per_frame_max_skel))
        mean_body = float(np.mean(skel_dist))
        max_finger = float(np.max(finger_dist))

        skel_names = [nm for nm, c in zip(common, bone_classes) if c == 'skeleton']
        worst_bone = ''
        if skel_names:
            worst_bone = skel_names[int(np.argmax(np.percentile(skel_dist, 95, axis=0)))]

        NO_DIFF = 1.0     # mm
        BODY_THRESH = 5.0 # mm on median — 5mm median means most frames have visible diff

        if length_changed:
            cat = 'length_changed'
        elif median_body < NO_DIFF and max_finger < NO_DIFF:
            cat = 'no_diff'
        elif median_body < BODY_THRESH:
            cat = 'finger_only'
        else:
            cat = 'has_diff'

        return {
            'name': base,
            'category': cat,
            'frames_orig': fo,
            'frames_clean': fc,
            'median_body_mm': round(median_body, 1),
            'p95_body_mm': round(p95_body, 1),
            'max_body_mm': round(max_body, 1),
            'mean_body_mm': round(mean_body, 1),
            'max_finger_mm': round(max_finger, 1),
            'worst_bone': worst_bone,
            'num_skeleton_bones': int(skel_mask.sum()),
            'num_finger_bones': int(finger_mask.sum()),
            'num_marker_bones': int(sum(1 for c in bone_classes if c == 'marker')),
        }
    except Exception as e:
        return {'name': base, 'error': str(e)}
    finally:
        manager.Destroy()


def main():
    pairs = []
    for f in sorted(FBX_DIR.glob('*.fbx')):
        if f.name.endswith('_cleaned.fbx'):
            continue
        base = f.stem
        if (FBX_DIR / f'{base}_cleaned.fbx').exists():
            pairs.append(base)

    total = len(pairs)
    n_workers = min(cpu_count(), 16)
    print(f'Analyzing {total} FBX pairs with {n_workers} workers...')
    t0 = time.time()

    results = []
    done = 0
    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(analyze_one, pairs):
            results.append(result)
            done += 1
            if done % 50 == 0:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                print(f'  {done}/{total} done, {elapsed:.0f}s elapsed, ETA {eta:.0f}s')

    elapsed = time.time() - t0
    print(f'Done in {elapsed:.0f}s')

    # Filter out errors
    ok = [r for r in results if r and 'error' not in r]
    errors = [r for r in results if r and 'error' in r]
    if errors:
        print(f'Errors: {len(errors)}')
        for e in errors[:5]:
            print(f"  {e['name']}: {e['error']}")

    cats = {'no_diff': 0, 'finger_only': 0, 'has_diff': 0, 'length_changed': 0}
    for r in ok:
        cats[r['category']] += 1

    summary = {
        'total_pairs': len(ok),
        'analysis_method': 'FBX bone world position distance (mm)',
        'body_threshold_mm': 10.0,
        **cats,
    }

    report = {
        'summary': summary,
        'length_changed': [r for r in ok if r['category'] == 'length_changed'],
        'visible_diff': sorted([r for r in ok if r['category'] == 'has_diff'], key=lambda x: -x['median_body_mm']),
        'finger_only': sorted([r for r in ok if r['category'] == 'finger_only'], key=lambda x: -x['max_finger_mm']),
        'no_visible_diff': sorted([r for r in ok if r['category'] == 'no_diff'], key=lambda x: -x.get('median_body_mm', 0)),
    }

    out_path = 'docs/temp/cjgame_fbx_analysis.json'
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f'\nReport saved to {out_path}')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
