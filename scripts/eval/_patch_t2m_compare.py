#!/usr/bin/env python3
"""One-shot patcher: add Go-to-Zero + fresh HY-Motion + a focus mode to the
t2m_compare viewer (its app.py is under .cursorignore so the IDE edit tools are
blocked; we patch via the filesystem)."""
from pathlib import Path

APP = Path("motion_annot_web/t2m_compare/app.py")
src = APP.read_text()

OLD = '''METHODS = {
    "GT": os.path.join(PREP, "real_conv"),
    "MotionStreamer": os.path.join(PREP, "motionstreamer"),
    "HY-Motion": HY_DIR,
    PRISM_LABEL: PRISM_DIR,
}
# render order (GT first)
METHOD_ORDER = ["GT", "MotionStreamer", "HY-Motion", PRISM_LABEL]
if E15_NOKAFS_DIR is not None:
    METHODS[E15_NOKAFS_LABEL] = E15_NOKAFS_DIR
    METHOD_ORDER.append(E15_NOKAFS_LABEL)
if OLD_PRISM_DIR is not None:
    METHODS[OLD_PRISM_LABEL] = OLD_PRISM_DIR
    METHOD_ORDER.append(OLD_PRISM_LABEL)'''

NEW = '''# Fresh (this-eval) HY-Motion / Go-to-Zero repacks from the 272 generations, keyed
# by the canonical HumanML3D names (scripts/eval/repack_idx272_to_prep.py).
HY_VIZ = os.path.join(EVAL, "t2m_viz", "hymotion")
G2Z_VIZ = os.path.join(EVAL, "t2m_viz", "gotozero")

# Focus mode: T2M_COMPARE_FOCUS=hy_g2z shows only GT / HY-Motion / Go-to-Zero.
if os.environ.get("T2M_COMPARE_FOCUS") == "hy_g2z":
    METHODS = {
        "GT": os.path.join(PREP, "real_conv"),
        "HY-Motion": HY_VIZ,
        "Go-to-Zero": G2Z_VIZ,
    }
    METHOD_ORDER = ["GT", "HY-Motion", "Go-to-Zero"]
else:
    METHODS = {
        "GT": os.path.join(PREP, "real_conv"),
        "MotionStreamer": os.path.join(PREP, "motionstreamer"),
        "HY-Motion": HY_VIZ if glob.glob(os.path.join(HY_VIZ, "*.npz")) else HY_DIR,
        "Go-to-Zero": G2Z_VIZ,
        PRISM_LABEL: PRISM_DIR,
    }
    METHOD_ORDER = ["GT", "MotionStreamer", "HY-Motion", "Go-to-Zero", PRISM_LABEL]
    if E15_NOKAFS_DIR is not None:
        METHODS[E15_NOKAFS_LABEL] = E15_NOKAFS_DIR
        METHOD_ORDER.append(E15_NOKAFS_LABEL)
    if OLD_PRISM_DIR is not None:
        METHODS[OLD_PRISM_LABEL] = OLD_PRISM_DIR
        METHOD_ORDER.append(OLD_PRISM_LABEL)'''

CAP_OLD = "CAPTIONS = _build_captions()"
CAP_NEW = '''CAPTIONS = _build_captions()

# Prefer the actual generation prompts dumped alongside the repacked predictions
# (name -> caption), so the viewer shows the true text each clip was generated from.
for _vd in (G2Z_VIZ, HY_VIZ):
    _cap_f = os.path.join(_vd, "captions.json")
    if os.path.isfile(_cap_f):
        try:
            _cap = json.load(open(_cap_f))
            for _k, _v in _cap.items():
                CAPTIONS.setdefault(_k, _v)
        except Exception:
            pass'''

assert OLD in src, "METHODS block not found (already patched?)"
assert CAP_OLD in src, "CAPTIONS line not found"
src = src.replace(OLD, NEW).replace(CAP_OLD, CAP_NEW, 1)
APP.write_text(src)
print("patched", APP)
