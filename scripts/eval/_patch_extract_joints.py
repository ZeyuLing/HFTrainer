"""One-shot: make StableMotion extract_joints_smpldata accept numpy OR torch
smpldata payloads (so cross-method results.npy from our pipeline evaluate)."""
p = 'ref_repo/StableMotion/data_loaders/amasstools/extract_joints.py'
s = open(p).read()
old = (
    '    smpldata["mocap_framerate"] = fps\n'
    '    poses = smpldata["poses"].to(device)\n'
    '    trans = smpldata["trans"].to(device)\n'
    '    joints = smpldata["joints"]'
)
new = (
    '    smpldata["mocap_framerate"] = fps\n'
    '    # robustness: accept numpy or torch smpldata payloads (cross-method)\n'
    '    def _ten(x):\n'
    '        import numpy as _np\n'
    '        return x if torch.is_tensor(x) else torch.as_tensor(_np.asarray(x))\n'
    '    poses = _ten(smpldata["poses"]).to(device)\n'
    '    trans = _ten(smpldata["trans"]).to(device)\n'
    '    joints = _ten(smpldata["joints"])'
)
if 'robustness: accept numpy' in s:
    print('already patched')
elif old in s:
    open(p, 'w').write(s.replace(old, new))
    print('patched OK')
else:
    i = s.find('mocap_framerate')
    print('PATTERN NOT FOUND; context:')
    print(repr(s[i - 30:i + 220]))
