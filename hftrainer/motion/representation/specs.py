"""Single source of truth for motion representation layouts.

Every motion vector used in this repo (HumanML3D-263, MotionStreamer-272,
SMPL ``motion_135`` / ``138`` / ``198`` / ``201`` / ``147`` / ``151`` …) has a
specific channel layout, frame rate, body model and — crucially — a **rot6d
packing convention**. Mixing these up is the single most common source of
silent motion bugs in this codebase (collapsed shoulders, exploded FID,
foot skating after a "correct looking" conversion).

This module is deliberately **import-light** (pure Python, no numpy/torch) so it
can be imported anywhere just to look up "what is this 263/272/135 vector".

Key fact to memorize
--------------------
``rot6d`` is stored in two incompatible layouts (see
:class:`hftrainer.motion.representation.rotation.Rot6DConvention`):

- ``column`` = ``[R00,R10,R20, R01,R11,R21]`` (first two columns) — math default.
- ``row``    = ``[R00,R01, R10,R11, R20,R21]`` (first two rows).

Per-representation convention:

============  =========  ===================================================
repr          rot6d      who/where
============  =========  ===================================================
motion_135    row        HyMotion M2M (load_smplx reorders aa→rot6d to row)
motion_138    column     PRISM / MCM / VerMo (SMPLPoseProcessor rot_convert default)
motion_201    row        HyMotion T2M o6dp_1103: motion_135 head + RIC joints
MS272         row        joint rot block [140:272] and heading [2:8]
HML263        column     rot_data block [67:193] (cont6d, column-major)
============  =========  ===================================================

Use :func:`get_spec` to fetch a :class:`MotionRepr` by name or alias, and
:meth:`MotionRepr.slice` / :meth:`MotionRepr.describe` to access channels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


# rot6d convention literals (kept as plain strings to stay import-light; they
# match hftrainer.motion.representation.rotation.Rot6DConvention).
ROT6D_ROW = "row"
ROT6D_COLUMN = "column"


@dataclass(frozen=True)
class FieldSpec:
    """A contiguous channel block ``[start:end)`` inside a motion vector."""

    name: str
    start: int
    end: int
    desc: str = ""

    @property
    def size(self) -> int:
        return self.end - self.start

    def as_slice(self) -> slice:
        return slice(self.start, self.end)


@dataclass(frozen=True)
class MotionRepr:
    """Layout + metadata for one motion representation.

    Attributes:
        name: canonical name (e.g. ``"motion_135"``).
        dim: total channel count.
        fps: native frame rate of this representation.
        body_model: skeleton/body model this vector describes.
        num_joints: number of skeleton joints (rotational dofs / position joints).
        rot6d_convention: ``"row"``, ``"column"`` or ``None`` (no per-joint rot6d
            block, e.g. HML263 stores velocities/positions and a separate cont6d
            block that is column-major; MS272's joint block is row).
        transl_type: ``"abs"``, ``"rel"``, ``"abs_rel"`` or ``None``.
        fields: ordered channel blocks; their sizes must sum to ``dim``.
        norm_stats: where normalization mean/std for this repr lives, or ``None``.
        decode_via: short hint for how to turn this vector into 3D joints
            (e.g. ``"forward_kinematics"`` vs ``"recover_from_ric"``).
        notes: free-form caveats.
        aliases: alternative names accepted by :func:`get_spec`.
    """

    name: str
    dim: int
    fps: int
    body_model: str
    num_joints: int
    rot6d_convention: Optional[str]
    transl_type: Optional[str]
    fields: Tuple[FieldSpec, ...]
    norm_stats: Optional[str] = None
    decode_via: str = ""
    notes: str = ""
    aliases: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        total = sum(f.size for f in self.fields)
        if total != self.dim:
            raise ValueError(
                f"{self.name}: field sizes sum to {total} but dim={self.dim}"
            )

    def field(self, name: str) -> FieldSpec:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(f"{self.name} has no field {name!r}; have {[f.name for f in self.fields]}")

    def slice(self, name: str) -> slice:
        """Return the ``slice`` for a named channel block."""
        return self.field(name).as_slice()

    def describe(self) -> str:
        lines = [
            f"{self.name}: dim={self.dim}, fps={self.fps}, body={self.body_model}, "
            f"joints={self.num_joints}, rot6d={self.rot6d_convention}, "
            f"transl={self.transl_type}",
        ]
        if self.decode_via:
            lines.append(f"  decode_via: {self.decode_via}")
        if self.norm_stats:
            lines.append(f"  norm_stats: {self.norm_stats}")
        for f in self.fields:
            lines.append(f"  [{f.start:>3}:{f.end:<3}] {f.name:<16} {f.desc}")
        if self.notes:
            lines.append(f"  notes: {self.notes}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_SMPL22 = "smpl_22"
_SMPL33 = "smpl_33"
_HML_SKELETON = "humanml3d_canonical_22"


MOTION_135 = MotionRepr(
    name="motion_135",
    dim=135,
    fps=30,
    body_model=_SMPL22,
    num_joints=22,
    rot6d_convention=ROT6D_ROW,
    transl_type="abs",
    fields=(
        FieldSpec("transl", 0, 3, "absolute root translation (x,y,z)"),
        FieldSpec("rot6d", 3, 135, "22 joints x rot6d(6), ROW-major, local (parent-relative)"),
    ),
    norm_stats="data/hymotion_m2m_data/_stats/{Mean,Std}.npy",
    decode_via="forward_kinematics (hftrainer.motion.skeleton.fk)",
    notes="HyMotion M2M main representation. rot6d is ROW-major; reorder [0,2,4,1,3,5] "
    "to column before calling rotation_6d_to_axis_angle, or pass convention='row'.",
    aliases=("135", "smpl22_135", "hymotion_m2m"),
)

MOTION_138 = MotionRepr(
    name="motion_138",
    dim=138,
    fps=30,
    body_model=_SMPL22,
    num_joints=22,
    rot6d_convention=ROT6D_COLUMN,
    transl_type="abs_rel",
    fields=(
        FieldSpec("transl", 0, 6, "abs(3) + rel(3) translation (abs_rel)"),
        FieldSpec("rot6d", 6, 138, "22 joints x rot6d(6), COLUMN-major, local"),
    ),
    norm_stats="data/statistic/smplx55_stats_hymotion_aug.json",
    decode_via="forward_kinematics",
    notes="PRISM / MCM / VerMo. rot6d is COLUMN-major (SMPLPoseProcessor rot_convert "
    "default, NO [0,3,1,4,2,5] reorder). Do NOT reuse motion_135 (row) decoders here.",
    aliases=("138", "prism", "vermo", "mcm"),
)

MOTION_198 = MotionRepr(
    name="motion_198",
    dim=198,
    fps=30,
    body_model=_SMPL22,
    num_joints=22,
    rot6d_convention=ROT6D_ROW,
    transl_type="abs",
    fields=(
        FieldSpec("transl", 0, 3, "absolute root translation"),
        FieldSpec("rot6d", 3, 135, "22 joints x rot6d(6), ROW-major (same as motion_135)"),
        FieldSpec("joint_pos", 135, 198, "21 non-root joint positions (21x3), FK-relative (Scheme D)"),
    ),
    norm_stats="data/hymotion_m2m_data/_stats (135 head) + position stats",
    decode_via="forward_kinematics on [0:135]; positions [135:198] are auxiliary",
    notes="M2M 198-dim variant = motion_135 + FK joint positions for an FK-consistency "
    "loss. The first 135 dims are exactly motion_135. See compute_198dim.py.",
    aliases=("198",),
)

MOTION_147 = MotionRepr(
    name="motion_147",
    dim=147,
    fps=30,
    body_model=_SMPL22,
    num_joints=22,
    rot6d_convention=ROT6D_ROW,
    transl_type="abs",
    fields=(
        FieldSpec("transl", 0, 3, "absolute root translation"),
        FieldSpec("rot6d", 3, 135, "22 joints x rot6d(6), ROW-major"),
        FieldSpec("end_effectors", 135, 147, "4 end-effector positions (4x3): hands+feet"),
    ),
    decode_via="forward_kinematics on [0:135]",
    notes="motion_135 + end-effector positions. See compute_147dim.py.",
    aliases=("147",),
)

MOTION_151 = MotionRepr(
    name="motion_151",
    dim=151,
    fps=30,
    body_model=_SMPL22,
    num_joints=22,
    rot6d_convention=ROT6D_ROW,
    transl_type="abs",
    fields=(
        FieldSpec("transl", 0, 3, "absolute root translation"),
        FieldSpec("rot6d", 3, 135, "22 joints x rot6d(6), ROW-major"),
        FieldSpec("end_effectors", 135, 147, "4 end-effector positions (4x3)"),
        FieldSpec("foot_contact", 147, 151, "4 foot-contact binary flags"),
    ),
    decode_via="forward_kinematics on [0:135]",
    notes="motion_147 + foot contact. See compute_151dim_contact.py.",
    aliases=("151",),
)

MOTION_201 = MotionRepr(
    name="motion_201",
    dim=201,
    fps=30,
    body_model=_SMPL22,
    num_joints=22,
    rot6d_convention=ROT6D_ROW,
    transl_type="abs",
    fields=(
        FieldSpec("transl", 0, 3, "absolute root translation (same convention as motion_135)"),
        FieldSpec("root_rot6d", 3, 9, "root joint rot6d, ROW-major"),
        FieldSpec("body_rot6d", 9, 135, "21 body joints x rot6d(6), ROW-major, local"),
        FieldSpec("ric_joint_pos", 135, 201, "22 joints x 3 root-invariant joint positions"),
    ),
    norm_stats="checkpoints/HY-Motion-1.0/stats/{Mean,Std}.npy",
    decode_via="decode_motion_from_latent -> motion_135 head + optional RIC diagnostics",
    notes="HyMotion T2M official o6dp_1103 representation. After denormalization, "
    "[0:135] is motion_135 (abs transl + 22 ROW rot6d) and [135:201] is FK-derived "
    "root-invariant joint positions. Official checkpoints use rel_trans=False; do "
    "not cumsum transl and do not reinterpret [135:201] as extra rotations.",
    aliases=("201", "hymotion_t2m"),
)

# HumanML3D-263: NOT a simple per-joint global rot6d vector. It is a redundant
# kinematic feature (root velocities + RIC joint positions + a cont6d rot block +
# local velocities + foot contacts). Decode with recover_from_ric, NOT FK on a
# rot6d block.
HML263 = MotionRepr(
    name="hml263",
    dim=263,
    fps=20,
    body_model=_HML_SKELETON,
    num_joints=22,
    rot6d_convention=ROT6D_COLUMN,  # rot_data block [67:193] is cont6d column-major
    transl_type=None,
    fields=(
        FieldSpec("root_rot_vel", 0, 1, "root angular (yaw) velocity"),
        FieldSpec("root_lin_vel", 1, 3, "root linear velocity in ground plane (x,z)"),
        FieldSpec("root_y", 3, 4, "root height (y)"),
        FieldSpec("ric_data", 4, 67, "21 non-root joint positions (21x3), local/heading-removed"),
        FieldSpec("rot_data", 67, 193, "21 non-root joint cont6d (21x6), COLUMN-major"),
        FieldSpec("local_velocity", 193, 259, "22 joint velocities (22x3)"),
        FieldSpec("foot_contact", 259, 263, "4 foot-contact flags"),
    ),
    norm_stats="checkpoints/evaluators/humanml3d_263/{Mean,Std}.npy (eval); "
    "MoMask t2m meta/{mean,std}.npy (MDM)",
    decode_via="recover_from_ric -> 22 joint positions (hftrainer.motion.representation.humanml)",
    notes="HumanML3D / MoMask / MDM / FlowMDM output space, 20 fps. The rot_data "
    "block is column-major cont6d but is rarely used directly; standard decoding "
    "goes through recover_from_ric to joints, then IK to SMPL.",
    aliases=("263", "humanml263", "humanml3d_263", "h3d263"),
)

# MotionStreamer / GoToZero 272: heading-removed root + global joint pos/vel +
# local joint rot6d. 30 fps.
MS272 = MotionRepr(
    name="ms272",
    dim=272,
    fps=30,
    body_model=_SMPL22,
    num_joints=22,
    rot6d_convention=ROT6D_ROW,
    transl_type=None,
    fields=(
        FieldSpec("root_lin_vel", 0, 2, "root linear velocity in ground plane (x,z), heading-removed"),
        FieldSpec("heading_delta", 2, 8, "per-frame heading delta as rot6d (ROW-major)"),
        FieldSpec("joint_pos", 8, 74, "22 joint positions (22x3), heading-removed, root xz at origin"),
        FieldSpec("joint_vel", 74, 140, "22 joint velocities (22x3)"),
        FieldSpec("joint_rot6d", 140, 272, "22 joint LOCAL rot6d (22x6), ROW-major"),
    ),
    norm_stats="data/evaluators/humanml3d_272/{Mean,Std}.npy",
    decode_via="recover_272_stored_positions for native joints; "
    "recover_local_rotations_and_root for SMPL-like root+rot",
    notes="MotionStreamer-272 evaluator space, 30 fps. It stores heading-removed "
    "joint positions directly and also stores recoverable root translation + local "
    "rotations, but it does not store betas. Raw SMPL should use smpl85_to_272 "
    "(official SMPL-X FK with betas when available). motion135_to_272 is an "
    "approximate feature bridge using the fixed GT-272 canonical offsets "
    "(bone_offsets_canon272.npy), not SMPL-H.",
    aliases=("272", "motionstreamer_272", "humanml3d_272", "h3d272", "gotozero_272"),
)


# InterHuman / InterGen 262: per-person two-person T2M representation. Canonical
# joint positions + velocities + NON-root SMPL body_pose local rot6d + foot
# contacts. The rot6d block is ROW-major (component-interleaved), same packing as
# MS272, NOT column. Encode drops the last frame (output length T-1).
IH262 = MotionRepr(
    name="interhuman_262",
    dim=262,
    fps=30,
    body_model=_SMPL22,
    num_joints=22,
    rot6d_convention=ROT6D_ROW,  # body_rot6d block [132:258], component-interleaved
    transl_type=None,            # absolute translation baked into canonical positions
    fields=(
        FieldSpec("joint_pos", 0, 66, "22 joint positions (22x3), canonical (Y-up, floor=0, root xz origin, face +Z)"),
        FieldSpec("joint_vel", 66, 132, "22 joint velocities (22x3), forward difference"),
        FieldSpec("body_rot6d", 132, 258, "21 NON-root joint local rot6d (21x6), ROW-major"),
        FieldSpec("foot_contact", 258, 262, "4 foot-contact flags (L heel/toe, R heel/toe)"),
    ),
    norm_stats="InterGen/InterCLIP global_mean_std (262-dim); two-person concat for InterCLIP",
    decode_via="interhuman262_to_joints (positions stored directly in [0:66]) "
    "(hftrainer.motion.representation.interhuman262)",
    notes="InterHuman / InterGen two-person space, 30 fps. rot6d block is ROW-major "
    "(SMPL body_pose, NON-root 21 joints); COLUMN layout silently drops ~0.3 R@3. "
    "Encode = process_motion_np canonicalisation, output length T-1. Person2 is "
    "rigid_transform-aligned to person1's first-frame heading + xz.",
    aliases=("262", "interhuman262", "ih262", "intergen262", "interhuman_native_262"),
)


_ALL: Tuple[MotionRepr, ...] = (
    MOTION_135,
    MOTION_138,
    MOTION_198,
    MOTION_147,
    MOTION_151,
    MOTION_201,
    HML263,
    MS272,
    IH262,
)

REGISTRY: Dict[str, MotionRepr] = {}
for _spec in _ALL:
    REGISTRY[_spec.name] = _spec
    for _alias in _spec.aliases:
        REGISTRY[_alias] = _spec


def get_spec(name: str) -> MotionRepr:
    """Look up a :class:`MotionRepr` by canonical name or alias.

    Examples::

        get_spec("motion_135")  # canonical
        get_spec("272")          # alias for MS272
        get_spec("humanml263")   # alias for HML263
    """
    key = str(name).strip().lower()
    if key in REGISTRY:
        return REGISTRY[key]
    raise KeyError(
        f"Unknown motion representation {name!r}. Known: "
        f"{sorted({s.name for s in _ALL})} (+aliases)."
    )


def list_specs() -> Tuple[MotionRepr, ...]:
    """Return all canonical specs (one per representation, no alias duplicates)."""
    return _ALL


def infer_spec_from_dim(dim: int) -> MotionRepr:
    """Best-effort lookup by channel count. Raises if ambiguous/unknown.

    Note dims are unique across the current registry, but this is a convenience
    helper only — prefer :func:`get_spec` with an explicit name when known.
    """
    matches = [s for s in _ALL if s.dim == dim]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(f"No motion representation with dim={dim}.")
    raise KeyError(
        f"Ambiguous dim={dim}: matches {[s.name for s in matches]}; use get_spec(name)."
    )


__all__ = [
    "ROT6D_ROW",
    "ROT6D_COLUMN",
    "FieldSpec",
    "MotionRepr",
    "REGISTRY",
    "get_spec",
    "list_specs",
    "infer_spec_from_dim",
    "MOTION_135",
    "MOTION_138",
    "MOTION_198",
    "MOTION_147",
    "MOTION_151",
    "MOTION_201",
    "HML263",
    "MS272",
    "IH262",
]
