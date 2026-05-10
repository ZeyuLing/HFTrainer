#!/usr/bin/env python3
"""Check quality issues in original FBX files using multiprocessing.

Detects: jitter, joint jumps, frozen frames in the ORIGINAL (pre-repair) FBX data.
"""
import fbx
import numpy as np
from pathlib import Path
import json, time
from multiprocessing import Pool, cpu_count

FBX_DIR = Path('data/lightai_data/CJGame_MB/raw')

SKELETON_BONES = {
    'hips','spine','spine1','spine2','spine3','neck','neck1','head',
    'leftshoulder','leftarm','leftforearm','lefthand',
    'rightshoulder','rightarm','rightforearm','righthand',
    'leftupleg','leftleg','leftfoot','lefttoebase',
    'rightupleg','rightleg','rightfoot','righttoebase',
}

# Upper body bones (exclude feet/legs for acceleration check — they naturally move fast)
UPPER_BODY = {
    'hips','spine','spine1','spine2','spine3','neck','neck1','head',
    'leftshoulder','leftarm','leftforearm','lefthand',
    'rightshoulder','rightarm','rightforearm','righthand',
}

def analyze_one(base):
    """Analyze quality of original FBX."""
    manager = fbx.FbxManager.Create()
    ios = fbx.FbxIOSettings.Create(manager, "IOSRoot")
    manager.SetIOSettings(ios)
    importer = fbx.FbxImporter.Create(manager, "")

    try:
        path = FBX_DIR / f'{base}.fbx'
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
        fps = 30
        nf = int(round((stop - start) * fps))
        if nf < 10:
            scene.Destroy(); return {'name': base, 'frames': nf, 'issues': [], 'has_issues': False}

        # Collect skeleton bones
        bones = {}
        def collect(node):
            name = node.GetName()
            short = name.split(':')[-1] if ':' in name else name
            if short.lower() in SKELETON_BONES:
                bones[short] = node
            for i in range(node.GetChildCount()):
                collect(node.GetChild(i))
        collect(scene.GetRootNode())

        names = sorted(bones.keys())
        pos = np.zeros((nf, len(names), 3), dtype=np.float64)
        for fi in range(nf):
            t = fbx.FbxTime()
            t.SetSecondDouble(start + fi / fps)
            for bi, name in enumerate(names):
                gt = bones[name].EvaluateGlobalTransform(t)
                tr = gt.GetT()
                pos[fi, bi] = [tr[0], tr[1], tr[2]]

        scene.Destroy()

        issues = []

        # 1. Velocity (mm/frame) — detect sudden jumps
        vel = np.diff(pos, axis=0) * 10  # cm -> mm
        vel_mag = np.linalg.norm(vel, axis=-1)  # (nf-1, num_bones)

        # Joint jump: any skeleton bone velocity > 150mm/frame (= 4.5m/s, very fast)
        # Focus on upper body where jumps are more noticeable
        upper_idx = [i for i, n in enumerate(names) if n.lower() in UPPER_BODY]
        if upper_idx:
            upper_vel = vel_mag[:, upper_idx]
            per_frame_max_vel = np.max(upper_vel, axis=1)
            jump_frames = np.where(per_frame_max_vel > 150)[0]
            if len(jump_frames) > 0:
                worst_fi = jump_frames[np.argmax(per_frame_max_vel[jump_frames])]
                worst_bi = upper_idx[np.argmax(upper_vel[worst_fi])]
                issues.append({
                    'type': 'joint_jump',
                    'count': int(len(jump_frames)),
                    'worst_frame': int(worst_fi),
                    'worst_bone': names[worst_bi],
                    'worst_vel_mm': round(float(per_frame_max_vel[worst_fi]), 1),
                })

        # 2. Jitter: high-frequency oscillation
        # Compute acceleration, then check if there are rapid direction changes
        if nf > 3:
            acc = np.diff(vel, axis=0)  # (nf-2, num_bones, 3)
            acc_mag = np.linalg.norm(acc, axis=-1)  # (nf-2, num_bones)

            if upper_idx:
                upper_acc = acc_mag[:, upper_idx]
                # Jitter = sustained high acceleration (multiple consecutive frames)
                # Check: how many frames have upper body acc > 100mm/f²
                high_acc_frames = np.max(upper_acc, axis=1) > 100
                # Count consecutive high-acc segments
                jitter_segments = 0
                in_segment = False
                max_segment_len = 0
                cur_len = 0
                for flag in high_acc_frames:
                    if flag:
                        if not in_segment:
                            jitter_segments += 1
                            in_segment = True
                            cur_len = 1
                        else:
                            cur_len += 1
                            max_segment_len = max(max_segment_len, cur_len)
                    else:
                        in_segment = False
                        cur_len = 0

                # Jitter if many short high-acc segments (>5 segments or sustained >10 frames)
                total_high_acc = int(np.sum(high_acc_frames))
                if total_high_acc > nf * 0.1:  # >10% of frames have high acceleration
                    issues.append({
                        'type': 'jitter',
                        'high_acc_frames': total_high_acc,
                        'pct': round(total_high_acc / nf * 100, 1),
                        'segments': jitter_segments,
                    })

        # 3. Frozen frames (consecutive identical frames anywhere, not just trailing)
        flat = pos.reshape(nf, -1)
        max_frozen = 0
        frozen_start = -1
        cur_frozen = 0
        for i in range(1, nf):
            if np.max(np.abs(flat[i] - flat[i-1])) < 0.001:  # < 0.001cm = 0.01mm
                cur_frozen += 1
                if cur_frozen > max_frozen:
                    max_frozen = cur_frozen
                    frozen_start = i - cur_frozen
            else:
                cur_frozen = 0
        if max_frozen >= 10:
            issues.append({
                'type': 'frozen',
                'max_consecutive': max_frozen,
                'start_frame': frozen_start,
            })

        # 4. Foot sliding: check if foot moves laterally while close to ground
        foot_idx = [i for i, n in enumerate(names) if n.lower() in {'leftfoot', 'rightfoot', 'lefttoebase', 'righttoebase'}]
        if foot_idx:
            foot_pos = pos[:, foot_idx, :]  # (nf, num_feet, 3)
            # Ground contact: Y position close to minimum Y
            min_y = np.min(foot_pos[:, :, 1])
            ground_thresh = min_y + 5  # within 5cm of lowest point

            foot_vel_horiz = np.sqrt(
                np.diff(foot_pos[:, :, 0], axis=0)**2 +
                np.diff(foot_pos[:, :, 2], axis=0)**2
            ) * 10  # mm/frame

            foot_near_ground = foot_pos[:-1, :, 1] < ground_thresh
            sliding = foot_vel_horiz * foot_near_ground  # only count when near ground

            # Sliding if foot moves > 20mm/frame while on ground
            slide_frames = np.sum(np.max(sliding, axis=1) > 20)
            if slide_frames > nf * 0.05:  # >5% of frames
                issues.append({
                    'type': 'foot_sliding',
                    'slide_frames': int(slide_frames),
                    'pct': round(int(slide_frames) / nf * 100, 1),
                })

        return {
            'name': base,
            'frames': nf,
            'duration_s': round(nf / fps, 1),
            'issues': issues,
            'has_issues': len(issues) > 0,
            'issue_types': [i['type'] for i in issues],
        }

    except Exception as e:
        return {'name': base, 'error': str(e)}
    finally:
        manager.Destroy()


def main():
    # Get all original FBX files that have a cleaned pair
    pairs = []
    for f in sorted(FBX_DIR.glob('*.fbx')):
        if f.name.endswith('_cleaned.fbx'):
            continue
        base = f.stem
        if (FBX_DIR / f'{base}_cleaned.fbx').exists():
            pairs.append(base)

    total = len(pairs)
    n_workers = min(cpu_count(), 16)
    print(f'Checking quality of {total} original FBX files with {n_workers} workers...')
    t0 = time.time()

    results = []
    done = 0
    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(analyze_one, pairs):
            if result:
                results.append(result)
            done += 1
            if done % 50 == 0:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                print(f'  {done}/{total} done, {elapsed:.0f}s elapsed, ETA {eta:.0f}s')

    elapsed = time.time() - t0
    print(f'Done in {elapsed:.0f}s')

    # Filter errors
    ok = [r for r in results if 'error' not in r]
    errors = [r for r in results if 'error' in r]
    if errors:
        print(f'Errors: {len(errors)}')

    # Summary
    has_issues = [r for r in ok if r['has_issues']]
    no_issues = [r for r in ok if not r['has_issues']]

    from collections import Counter
    issue_type_counts = Counter()
    for r in has_issues:
        for it in r['issue_types']:
            issue_type_counts[it] += 1

    summary = {
        'total': len(ok),
        'has_issues': len(has_issues),
        'no_issues': len(no_issues),
        'issue_type_counts': dict(issue_type_counts),
    }

    report = {
        'summary': summary,
        'has_issues': sorted(has_issues, key=lambda x: -len(x['issues'])),
        'no_issues': no_issues,
    }

    out_path = 'docs/temp/cjgame_original_quality.json'
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f'\nReport: {out_path}')
    print(json.dumps(summary, indent=2))

    # Cross-reference with repair diff data
    with open('docs/temp/cjgame_fbx_analysis.json') as f:
        diff_data = json.load(f)

    # Build lookup
    diff_map = {}
    for lst in [diff_data['visible_diff'], diff_data['finger_only'], diff_data.get('no_visible_diff', [])]:
        for r in lst:
            diff_map[r['name']] = r

    # For the 688 "no visible change" cases, how many had original quality issues?
    no_change = [r for r in ok if r['name'] in diff_map and diff_map[r['name']]['p95_body_mm'] <= 30]
    no_change_with_issues = [r for r in no_change if r['has_issues']]
    no_change_no_issues = [r for r in no_change if not r['has_issues']]

    print(f'\n--- Cross-reference: "no visible change" cases ({len(no_change)}) ---')
    print(f'  Original had quality issues: {len(no_change_with_issues)} (needed repair but not fixed)')
    print(f'  Original had no issues: {len(no_change_no_issues)} (did not need repair)')

    # Issue type breakdown for no_change_with_issues
    nc_types = Counter()
    for r in no_change_with_issues:
        for it in r['issue_types']:
            nc_types[it] += 1
    print(f'  Issue types: {dict(nc_types)}')


if __name__ == '__main__':
    main()
