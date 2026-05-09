"""Scan all work_dirs / latest ckpt and report bundle-level Parameter / buffer storage state."""
import os
import sys
import torch


def main():
    out = []
    base = 'work_dirs'
    for d in sorted(os.listdir(base)):
        full = os.path.join(base, d)
        if not os.path.isdir(full):
            continue
        ckdirs = [x for x in os.listdir(full) if x.startswith('checkpoint-')]
        if not ckdirs:
            continue

        def _key(s):
            try:
                return int(s.split('_')[-1])
            except Exception:
                return 0
        latest = sorted(ckdirs, key=_key)[-1]
        mpt = os.path.join(full, latest, 'model.pt')
        if not os.path.exists(mpt):
            continue
        try:
            md = torch.load(mpt, map_location='cpu', weights_only=False)
        except Exception as e:
            out.append(f'{d:60s} | {latest}: ERROR {e}')
            continue
        bp = md.get('__bundle_params__', {}) if isinstance(md, dict) else {}
        if not bp:
            out.append(f'{d:60s} | {latest}: NO __bundle_params__')
            continue
        # Categorise: norms of orphan params/buffers
        parts = []
        for k, v in bp.items():
            if hasattr(v, 'shape'):
                norm = v.float().norm().item()
                parts.append(f'{k}={norm:.3f}')
        out.append(f'{d:60s} | {latest}: {" ".join(parts)}')

    for line in out:
        print(line)


if __name__ == '__main__':
    main()
