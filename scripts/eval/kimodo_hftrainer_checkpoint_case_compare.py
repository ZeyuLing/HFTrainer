#!/usr/bin/env python3
"""Run one KIMODO prompt through hftrainer artifacts and render native body meshes."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("HFTRAINER_SKIP_AUTOREGISTER", "1")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("KIMODO_TEXT_ENCODER_DEVICE", "cpu")

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import pyrender  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hftrainer.models.motion.kimodo.bundle import (  # noqa: E402
    KIMODOBundle,
    _ensure_kimodo_importable,
)
from hftrainer.pipelines.motion.kimodo_pipeline import KIMODOPipeline  # noqa: E402


PANEL_W = 520
PANEL_H = 560
LABEL_H = 48
BG = np.array([244, 245, 247], dtype=np.uint8)
MODEL_SPECS = [
    ("soma_rp", "SOMA-RP", "hftrainer_soma_rp", "#4F7CAC"),
    ("g1_rp", "G1-RP", "hftrainer_g1_rp", "#4D9078"),
    ("g1_seed", "G1-SEED", "hftrainer_g1_seed", "#B86B44"),
    ("smplx_rp", "SMPLX-RP", "hftrainer_smplx_rp", "#815AC0"),
]
DEFAULT_SMPLX_ASSET = (
    Path("/apdcephfs_cq11/share_1467498/home/zeyuling")
    / "versatilemotion/checkpoints/smpl_models/smplx/SMPLX_MALE.npz"
)
MUJOCO_TO_KIMODO = np.array(
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
    dtype=np.float64,
)
G1_MESH_JOINT_MAP = {
    "pelvis_skel": ["pelvis.STL", "pelvis_contour_link.STL"],
    "left_hip_pitch_skel": ["left_hip_pitch_link.STL"],
    "left_hip_roll_skel": ["left_hip_roll_link.STL"],
    "left_hip_yaw_skel": ["left_hip_yaw_link.STL"],
    "left_knee_skel": ["left_knee_link.STL"],
    "left_ankle_pitch_skel": ["left_ankle_pitch_link.STL"],
    "left_ankle_roll_skel": ["left_ankle_roll_link.STL"],
    "right_hip_pitch_skel": ["right_hip_pitch_link.STL"],
    "right_hip_roll_skel": ["right_hip_roll_link.STL"],
    "right_hip_yaw_skel": ["right_hip_yaw_link.STL"],
    "right_knee_skel": ["right_knee_link.STL"],
    "right_ankle_pitch_skel": ["right_ankle_pitch_link.STL"],
    "right_ankle_roll_skel": ["right_ankle_roll_link.STL"],
    "waist_yaw_skel": ["waist_yaw_link_rev_1_0.STL", "waist_yaw_link.STL"],
    "waist_roll_skel": ["waist_roll_link_rev_1_0.STL", "waist_roll_link.STL"],
    "waist_pitch_skel": [
        "torso_link_rev_1_0.STL",
        "torso_link.STL",
        "logo_link.STL",
        "head_link.STL",
    ],
    "left_shoulder_pitch_skel": ["left_shoulder_pitch_link.STL"],
    "left_shoulder_roll_skel": ["left_shoulder_roll_link.STL"],
    "left_shoulder_yaw_skel": ["left_shoulder_yaw_link.STL"],
    "left_elbow_skel": ["left_elbow_link.STL"],
    "left_wrist_roll_skel": ["left_wrist_roll_link.STL"],
    "left_wrist_pitch_skel": ["left_wrist_pitch_link.STL"],
    "left_wrist_yaw_skel": ["left_wrist_yaw_link.STL", "left_rubber_hand.STL"],
    "right_shoulder_pitch_skel": ["right_shoulder_pitch_link.STL"],
    "right_shoulder_roll_skel": ["right_shoulder_roll_link.STL"],
    "right_shoulder_yaw_skel": ["right_shoulder_yaw_link.STL"],
    "right_elbow_skel": ["right_elbow_link.STL"],
    "right_wrist_roll_skel": ["right_wrist_roll_link.STL"],
    "right_wrist_pitch_skel": ["right_wrist_pitch_link.STL"],
    "right_wrist_yaw_skel": ["right_wrist_yaw_link.STL", "right_rubber_hand.STL"],
}


def hex2rgb(value: str) -> list[int]:
    value = value.lstrip("#")
    return [int(value[i : i + 2], 16) for i in (0, 2, 4)]


def look_at(eye, target, up=(0, 1, 0)) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-9
    side = np.cross(forward, up)
    side /= np.linalg.norm(side) + 1e-9
    true_up = np.cross(side, forward)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, 0] = side
    mat[:3, 1] = true_up
    mat[:3, 2] = -forward
    mat[:3, 3] = eye
    return mat


def quat_wxyz_to_mat(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat]
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_vendor_module(alias: str, relative_path: str):
    _ensure_kimodo_importable()
    import kimodo

    module_path = Path(kimodo.__file__).resolve().parent / relative_path
    spec = importlib.util.spec_from_file_location(alias, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def load_g1_mesh_data(mesh_dir: Path, skeleton) -> list[dict[str, Any]]:
    xml_path = mesh_dir.parent.parent / "xml/g1.xml"
    mesh_file_transforms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if xml_path.exists():
        tree = ET.parse(xml_path)
        root = tree.getroot()
        mesh_file_to_mesh_name = {}
        for mesh in root.findall(".//asset/mesh"):
            mesh_name = mesh.get("name")
            mesh_file = mesh.get("file")
            if mesh_name and mesh_file:
                mesh_file_to_mesh_name[mesh_file] = mesh_name
        mesh_name_to_transform = {}
        for geom in root.findall(".//geom"):
            mesh_name = geom.get("mesh")
            if mesh_name is None:
                continue
            pos = geom.get("pos")
            quat = geom.get("quat")
            geom_pos = (
                np.array([float(x) for x in pos.split()], dtype=np.float64)
                if pos
                else np.zeros(3, dtype=np.float64)
            )
            geom_rot = quat_wxyz_to_mat(np.array([float(x) for x in quat.split()], dtype=np.float64)) if quat else np.eye(3)
            mesh_name_to_transform[mesh_name] = (geom_pos, geom_rot)
        for mesh_file, mesh_name in mesh_file_to_mesh_name.items():
            geom_pos, geom_rot = mesh_name_to_transform.get(
                mesh_name,
                (np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)),
            )
            mesh_file_transforms[mesh_file] = (
                MUJOCO_TO_KIMODO @ geom_pos,
                MUJOCO_TO_KIMODO @ geom_rot @ MUJOCO_TO_KIMODO.T,
            )
    data = []
    for joint_name, mesh_files in G1_MESH_JOINT_MAP.items():
        if joint_name not in skeleton.bone_index:
            continue
        joint_idx = int(skeleton.bone_index[joint_name])
        for mesh_file in mesh_files:
            mesh_path = mesh_dir / mesh_file
            if not mesh_path.exists():
                continue
            mesh = trimesh.load_mesh(mesh_path, process=True)
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(mesh.dump())
            vertices = np.asarray(mesh.vertices, dtype=np.float32) @ MUJOCO_TO_KIMODO.T
            faces = np.asarray(mesh.faces, dtype=np.int32)
            geom_pos, geom_rot = mesh_file_transforms.get(
                mesh_file,
                (np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)),
            )
            data.append(
                {
                    "mesh_file": mesh_file,
                    "vertices": vertices,
                    "faces": faces,
                    "joint_idx": joint_idx,
                    "geom_pos": geom_pos,
                    "geom_rot": geom_rot,
                }
            )
    return data


def ensure_smplx_asset(asset_path: Path | None) -> dict[str, Any]:
    _ensure_kimodo_importable()
    from kimodo.assets import skeleton_asset_path

    target = Path(skeleton_asset_path("smplx22", "SMPLX_NEUTRAL.npz"))
    info = {
        "target": str(target),
        "exists_before": target.exists(),
        "source": None,
        "created": False,
    }
    if target.exists():
        return info
    if asset_path is None:
        return info
    asset_path = Path(asset_path)
    if not asset_path.exists():
        return info
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(asset_path)
    except OSError:
        shutil.copy2(asset_path, target)
    info.update({"source": str(asset_path), "created": True, "exists_after": target.exists()})
    return info


def precompute_text_embedding(args) -> dict[str, Any]:
    _ensure_kimodo_importable()
    os.environ["TEXT_ENCODERS_DIR"] = str(args.text_encoders_dir)
    os.environ.setdefault("KIMODO_TEXT_ENCODER_DEVICE", "cpu")
    from kimodo.model.llm2vec.llm2vec_wrapper import LLM2VecEncoder

    print(f"[encode] loading LLM2Vec from {args.text_encoders_dir}")
    encoder = LLM2VecEncoder(
        base_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
        dtype="bfloat16",
        llm_dim=4096,
    ).to("cpu")
    print(f"[encode] prompt={args.prompt!r}")
    text_feat, lengths = encoder([args.prompt])
    payload = {
        "prompt": args.prompt,
        "text_feat": text_feat.detach().cpu(),
        "lengths": list(lengths),
        "text_encoders_dir": str(args.text_encoders_dir),
    }
    out_path = args.save_text_embedding_path or args.text_embedding_path
    if out_path is None:
        out_path = args.out_dir / "prompt_embedding.pt"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(f"[encode] saved {out_path}")
    return {"path": str(out_path), "shape": list(payload["text_feat"].shape), "lengths": payload["lengths"]}


def load_text_embedding(path: Path, expected_prompt: str) -> tuple[torch.Tensor, list[int], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    prompt = payload.get("prompt")
    if prompt != expected_prompt:
        raise ValueError(f"Text embedding prompt mismatch: expected={expected_prompt!r}, got={prompt!r}")
    text_feat = payload["text_feat"].detach().cpu()
    lengths = list(payload["lengths"])
    info = {
        "path": str(path),
        "shape": list(text_feat.shape),
        "lengths": lengths,
        "prompt": prompt,
        "text_encoders_dir": payload.get("text_encoders_dir"),
    }
    return text_feat, lengths, info


def scalar_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return scalar_jsonable(value.item())
        return value.tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def save_native_npz(path: Path, output: dict[str, Any], metadata: dict[str, Any]) -> None:
    payload: dict[str, Any] = {}
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            payload[key] = value
    for key, value in metadata.items():
        payload[f"meta_{key}"] = np.array(value, dtype=object)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def summarize_output(output: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            summary[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "finite": bool(np.isfinite(value).all()) if np.issubdtype(value.dtype, np.number) else None,
            }
    return summary


def skin_soma(output: dict[str, Any], skeleton) -> tuple[np.ndarray, np.ndarray]:
    SOMASkin = load_vendor_module("kimodo_soma_skin_direct", "viz/soma_skin.py").SOMASkin

    skin = SOMASkin(skeleton)
    device = skin.bind_vertices.device
    joints = torch.as_tensor(output["posed_joints"], dtype=torch.float32, device=device)
    rots = torch.as_tensor(output["global_rot_mats"], dtype=torch.float32, device=device)
    with torch.no_grad():
        verts = skin.skin(rots, joints, rot_is_global=True).detach().cpu().numpy().astype(np.float32)
    faces = skin.faces.detach().cpu().numpy().astype(np.int32)
    return verts, faces


def skin_smplx(output: dict[str, Any], skeleton) -> tuple[np.ndarray, np.ndarray]:
    SMPLXSkin = load_vendor_module("kimodo_smplx_skin_direct", "viz/smplx_skin.py").SMPLXSkin

    skin = SMPLXSkin(skeleton)
    device = skin.bind_vertices.device
    joints = torch.as_tensor(output["posed_joints"], dtype=torch.float32, device=device)
    rots = torch.as_tensor(output["global_rot_mats"], dtype=torch.float32, device=device)
    with torch.no_grad():
        verts = skin.skin(rots, joints, rot_is_global=True).detach().cpu().numpy().astype(np.float32)
    faces = skin.faces.detach().cpu().numpy().astype(np.int32)
    return verts, faces


@dataclass
class Panel:
    key: str
    label: str
    color: list[int]
    centers: np.ndarray
    half_h: float
    half_w: float
    radius: float
    y_center: float
    cur_center: np.ndarray

    @classmethod
    def from_sequence(cls, key: str, label: str, color: list[int], seq: np.ndarray):
        seq = np.asarray(seq, dtype=np.float64)
        if seq.ndim == 2:
            seq = seq[None]
        lo = seq.min(axis=1)
        hi = seq.max(axis=1)
        centers = (lo + hi) / 2.0
        extent = hi - lo
        half_h = float(np.maximum(extent[:, 1].max() / 2.0, 1e-3))
        half_w = float(np.maximum(np.maximum(extent[:, 0], extent[:, 2]).max() / 2.0, 1e-3))
        radius = float(np.hypot(half_h, half_w) + 1e-3)
        y_center = float(np.median(centers[:, 1]))
        cur_center = centers[0].copy()
        cur_center[1] = y_center
        return cls(key, label, color, centers, half_h, half_w, radius, y_center, cur_center)

    def set_frame_center(self, frame_idx: int) -> None:
        frame_idx = int(np.clip(frame_idx, 0, len(self.centers) - 1))
        center = self.centers[frame_idx].copy()
        center[1] = self.y_center
        self.cur_center = center

    def camera_pose(self):
        yfov = np.deg2rad(42.0)
        aspect = PANEL_W / PANEL_H
        half = max(self.half_h, self.half_w / aspect) * 1.18
        dist = half / np.tan(yfov / 2.0) + self.half_w
        eye = self.cur_center + np.array([0.0, 0.15 * self.radius, dist])
        return look_at(eye, self.cur_center), yfov


class MeshPanel:
    def __init__(self, key: str, label: str, color: list[int], verts: np.ndarray, faces: np.ndarray):
        self.base = Panel.from_sequence(key, label, color, verts)
        self.verts = np.asarray(verts, dtype=np.float32)
        self.faces = np.asarray(faces, dtype=np.int32)
        self.material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[*(np.array(color) / 255.0), 1.0],
            roughnessFactor=0.65,
            metallicFactor=0.0,
        )

    @property
    def n_frames(self) -> int:
        return int(self.verts.shape[0])

    def geoms(self, frame_idx: int):
        frame_idx = int(np.clip(frame_idx, 0, self.n_frames - 1))
        self.base.set_frame_center(frame_idx)
        mesh = trimesh.Trimesh(vertices=self.verts[frame_idx], faces=self.faces, process=False)
        return [pyrender.Mesh.from_trimesh(mesh, material=self.material, smooth=True)]


class G1Panel:
    def __init__(self, key: str, label: str, color: list[int], output: dict[str, Any], skeleton):
        joints = np.asarray(output["posed_joints"], dtype=np.float32)
        self.rots = np.asarray(output["global_rot_mats"], dtype=np.float32)
        self.joints = joints
        self.base = Panel.from_sequence(key, label, color, joints)
        mesh_dir = Path(skeleton.folder) / "meshes/g1"
        items = load_g1_mesh_data(mesh_dir, skeleton)
        self.items = []
        for item in items:
            self.items.append(
                {
                    "mesh": trimesh.Trimesh(
                        vertices=np.asarray(item["vertices"], dtype=np.float32),
                        faces=np.asarray(item["faces"], dtype=np.int32),
                        process=False,
                    ),
                    "joint_idx": int(item["joint_idx"]),
                    "geom_pos": np.asarray(item["geom_pos"], dtype=np.float64),
                    "geom_rot": np.asarray(item["geom_rot"], dtype=np.float64),
                }
            )
        self.material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[*(np.array(color) / 255.0), 1.0],
            roughnessFactor=0.52,
            metallicFactor=0.08,
        )

    @property
    def n_frames(self) -> int:
        return int(self.joints.shape[0])

    def geoms(self, frame_idx: int):
        frame_idx = int(np.clip(frame_idx, 0, self.n_frames - 1))
        self.base.set_frame_center(frame_idx)
        geoms = []
        joints = self.joints[frame_idx]
        rots = self.rots[frame_idx]
        for item in self.items:
            joint_idx = item["joint_idx"]
            joint_pos = joints[joint_idx].astype(np.float64)
            joint_rot = rots[joint_idx].astype(np.float64)
            mesh_rot = joint_rot @ item["geom_rot"]
            mesh_pos = joint_pos + joint_rot @ item["geom_pos"]
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = mesh_rot
            pose[:3, 3] = mesh_pos
            geoms.append(
                pyrender.Mesh.from_trimesh(
                    item["mesh"],
                    material=self.material,
                    smooth=False,
                    poses=pose[None],
                )
            )
        return geoms


class PrecomputedTextEncoder:
    def __init__(self, text_feat: torch.Tensor, lengths):
        self.text_feat = text_feat.detach().cpu()
        self.lengths = list(lengths) if isinstance(lengths, (list, tuple)) else [int(lengths)]

    def __call__(self, text):
        count = 1 if isinstance(text, str) else len(text)
        if count != self.text_feat.shape[0]:
            raise ValueError(
                "PrecomputedTextEncoder only supports the cached prompt batch: "
                f"requested={count}, cached={self.text_feat.shape[0]}"
            )
        return self.text_feat.clone(), list(self.lengths)

    def to(self, device):
        return self

    def eval(self):
        return self


def render_panel(renderer, panel, geoms) -> np.ndarray:
    scene = pyrender.Scene(bg_color=[*(BG / 255.0), 1.0], ambient_light=[0.30, 0.30, 0.32])
    for geom in geoms:
        scene.add(geom)
    center = panel.base.cur_center
    radius = max(panel.base.radius, 0.5)
    floor_y = float(panel.base.centers[:, 1].min() - panel.base.half_h - 0.03)
    ground = trimesh.creation.box(extents=(radius * 2.8, 0.01, radius * 2.8))
    ground_pose = np.eye(4)
    ground_pose[:3, 3] = [center[0], floor_y, center[2]]
    ground_mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.82, 0.84, 0.86, 1.0],
        roughnessFactor=0.9,
        metallicFactor=0.0,
    )
    scene.add(pyrender.Mesh.from_trimesh(ground, material=ground_mat, smooth=False), pose=ground_pose)
    cam_pose, yfov = panel.base.camera_pose()
    scene.add(pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=PANEL_W / PANEL_H), pose=cam_pose)
    for offset, intensity in [
        (np.array([-1.2, 1.4, 1.8]), 4.2),
        (np.array([1.3, 0.6, 1.2]), 1.7),
        (np.array([0.2, 1.0, -1.6]), 1.5),
    ]:
        scene.add(
            pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=intensity),
            pose=look_at(center + offset * radius, center),
        )
    color, _ = renderer.render(scene)
    return color


def label_strip(img: np.ndarray, text: str) -> np.ndarray:
    out = Image.new("RGB", (PANEL_W, PANEL_H + LABEL_H), tuple(int(x) for x in BG))
    out.paste(Image.fromarray(img), (0, LABEL_H))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text(
        ((PANEL_W - (bbox[2] - bbox[0])) / 2, (LABEL_H - (bbox[3] - bbox[1])) / 2 - bbox[1]),
        text,
        fill=(25, 29, 35),
        font=font,
    )
    return np.asarray(out)


def render_outputs(panels, out_dir: Path, fps: int, max_video_frames: int) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_frames = min(max(p.n_frames for p in panels), max_video_frames)
    for panel in panels:
        panel.base.cur_center = panel.base.centers[0].copy()
    renderer = pyrender.OffscreenRenderer(PANEL_W, PANEL_H)
    video_path = out_dir / "comparison.mp4"
    writer = imageio.get_writer(video_path, fps=fps, codec="libx264", quality=7, macro_block_size=8)
    preview_indices = sorted(set([0, n_frames // 3, (2 * n_frames) // 3, n_frames - 1]))
    preview_rows = []
    for frame_idx in range(n_frames):
        strips = []
        for panel in panels:
            img = render_panel(renderer, panel, panel.geoms(frame_idx))
            strips.append(label_strip(img, panel.base.label))
        frame = np.concatenate(strips, axis=1)
        writer.append_data(frame)
        if frame_idx in preview_indices:
            preview_rows.append(frame)
        if frame_idx % 30 == 0:
            print(f"[render] frame {frame_idx}/{n_frames}")
    writer.close()
    renderer.delete()
    preview_path = out_dir / "comparison_preview.png"
    Image.fromarray(np.concatenate(preview_rows, axis=0)).save(preview_path)
    return {"video": str(video_path), "preview": str(preview_path)}


def run_one(
    key: str,
    label: str,
    artifact_dir: Path,
    color: str,
    args,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    from kimodo.tools import seed_everything

    print(f"[load] {label}: {artifact_dir}")
    load_text_encoder = args.text_encoder
    cached_text_embedding = getattr(args, "_cached_text_embedding", None)
    if args.text_encoder == "llm2vec" and args.share_text_embedding and cached_text_embedding is not None:
        load_text_encoder = "dummy"
    bundle = KIMODOBundle.from_pretrained(
        str(artifact_dir),
        device=args.device,
        diffusion_steps=args.diffusion_steps,
        text_encoder_mode="local",
        text_encoder=load_text_encoder,
        text_encoders_dir=str(args.text_encoders_dir),
    )
    pipe = KIMODOPipeline(bundle)
    import kimodo

    kimodo_file = str(Path(kimodo.__file__).resolve())
    if "_vendor" not in kimodo_file:
        raise RuntimeError(f"KIMODO did not import from vendor path: {kimodo_file}")
    if args.text_encoder == "llm2vec" and args.share_text_embedding:
        if cached_text_embedding is None:
            print(f"[encode] {label}: computing shared LLM2Vec prompt embedding")
            text_feat, lengths = bundle.model.text_encoder([args.prompt])
            args._cached_text_embedding = (text_feat.detach().cpu(), list(lengths))
            cached_text_embedding = args._cached_text_embedding
        bundle.model.text_encoder = PrecomputedTextEncoder(*cached_text_embedding)
        gc.collect()
    seed_everything(args.seed, deterministic=False)
    print(f"[infer] {label}: prompt={args.prompt!r}, frames={args.num_frames}, seed={args.seed}")
    output = pipe.text_to_motion(
        args.prompt,
        num_frames=args.num_frames,
        cfg_weight=[args.text_cfg, args.constraint_cfg],
        progress_bar=(lambda x: x),
    )
    metadata = {
        "key": key,
        "label": label,
        "artifact_dir": str(artifact_dir),
        "prompt": args.prompt,
        "seed": args.seed,
        "num_frames": args.num_frames,
        "diffusion_steps": args.diffusion_steps,
        "text_cfg": args.text_cfg,
        "constraint_cfg": args.constraint_cfg,
        "text_encoder": args.text_encoder,
        "load_text_encoder": load_text_encoder,
        "share_text_embedding": bool(args.share_text_embedding),
        "kimodo_import": kimodo_file,
        "resolved_model_name": bundle.resolved_model_name,
        "model_name": bundle.model_name,
        "skeleton_type": type(pipe.skeleton).__name__,
    }
    npz_path = args.out_dir / "native_npz" / f"{key}.npz"
    save_native_npz(npz_path, output, metadata)
    metadata["native_npz"] = str(npz_path)
    metadata["output_summary"] = summarize_output(output)
    skeleton = pipe.skeleton
    del pipe, bundle
    torch.cuda.empty_cache()
    gc.collect()
    return skeleton, output, metadata


def build_panel(key: str, label: str, color: str, skeleton, output: dict[str, Any], out_dir: Path):
    color_rgb = hex2rgb(color)
    skeleton_type = type(skeleton).__name__
    if skeleton_type.startswith("SOMA"):
        verts, faces = skin_soma(output, skeleton)
        np.savez_compressed(out_dir / "mesh_npz" / f"{key}_mesh.npz", vertices=verts, faces=faces)
        return MeshPanel(key, label, color_rgb, verts, faces), {
            "mesh_type": "soma_skin",
            "vertices_shape": list(verts.shape),
            "faces_shape": list(faces.shape),
            "mesh_npz": str(out_dir / "mesh_npz" / f"{key}_mesh.npz"),
        }
    if skeleton_type == "G1Skeleton34":
        panel = G1Panel(key, label, color_rgb, output, skeleton)
        return panel, {
            "mesh_type": "g1_stl",
            "link_mesh_count": len(panel.items),
        }
    if skeleton_type == "SMPLXSkeleton22":
        verts, faces = skin_smplx(output, skeleton)
        np.savez_compressed(out_dir / "mesh_npz" / f"{key}_mesh.npz", vertices=verts, faces=faces)
        return MeshPanel(key, label, color_rgb, verts, faces), {
            "mesh_type": "smplx_skin",
            "vertices_shape": list(verts.shape),
            "faces_shape": list(faces.shape),
            "mesh_npz": str(out_dir / "mesh_npz" / f"{key}_mesh.npz"),
        }
    raise ValueError(f"Unsupported skeleton type for mesh rendering: {skeleton_type}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        default="A person walks diagonally to the left and waves at someone on their right.",
    )
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--num_frames", type=int, default=150)
    parser.add_argument("--diffusion_steps", type=int, default=100)
    parser.add_argument("--text_cfg", type=float, default=2.0)
    parser.add_argument("--constraint_cfg", type=float, default=2.0)
    parser.add_argument("--text_encoder", choices=["llm2vec", "dummy"], default="llm2vec")
    parser.add_argument("--no_share_text_embedding", action="store_true")
    parser.add_argument("--text_embedding_path", type=Path, default=None)
    parser.add_argument("--save_text_embedding_path", type=Path, default=None)
    parser.add_argument("--precompute_text_embedding_only", action="store_true")
    parser.add_argument("--only", nargs="*", choices=[spec[0] for spec in MODEL_SPECS], default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max_video_frames", type=int, default=150)
    parser.add_argument("--checkpoint_root", type=Path, default=REPO_ROOT / "checkpoints/kimodo")
    parser.add_argument(
        "--text_encoders_dir",
        type=Path,
        default=REPO_ROOT / "checkpoints/kimodo/hftrainer_soma_rp/text_encoders",
    )
    parser.add_argument("--smplx_asset", type=Path, default=DEFAULT_SMPLX_ASSET)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=REPO_ROOT / "output/kimodo_hftrainer_case_compare/walk_wave_seed44",
    )
    args = parser.parse_args()
    args.share_text_embedding = not args.no_share_text_embedding
    args._cached_text_embedding = None
    delattr(args, "no_share_text_embedding")
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "native_npz").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "mesh_npz").mkdir(parents=True, exist_ok=True)
    if args.precompute_text_embedding_only:
        precompute_text_embedding(args)
        return
    text_embedding_info = None
    if args.text_embedding_path is not None:
        text_feat, lengths, text_embedding_info = load_text_embedding(args.text_embedding_path, args.prompt)
        args._cached_text_embedding = (text_feat, lengths)
        args.share_text_embedding = True
    _ensure_kimodo_importable()
    import kimodo

    kimodo_file = str(Path(kimodo.__file__).resolve())
    if "_vendor" not in kimodo_file:
        raise RuntimeError(f"KIMODO did not import from vendor path: {kimodo_file}")
    smplx_asset_info = ensure_smplx_asset(args.smplx_asset)
    summary: dict[str, Any] = {
        "prompt": args.prompt,
        "seed": args.seed,
        "num_frames": args.num_frames,
        "diffusion_steps": args.diffusion_steps,
        "text_cfg": args.text_cfg,
        "constraint_cfg": args.constraint_cfg,
        "text_encoder": args.text_encoder,
        "share_text_embedding": bool(args.share_text_embedding),
        "text_embedding": text_embedding_info,
        "kimodo_import": kimodo_file,
        "text_encoders_dir": str(args.text_encoders_dir),
        "smplx_asset": smplx_asset_info,
        "runs": [],
    }
    panels = []
    for key, label, artifact_name, color in MODEL_SPECS:
        if args.only and key not in args.only:
            continue
        artifact_dir = args.checkpoint_root / artifact_name
        skeleton, output, metadata = run_one(key, label, artifact_dir, color, args)
        panel, mesh_info = build_panel(key, label, color, skeleton, output, args.out_dir)
        metadata.update(mesh_info)
        panels.append(panel)
        summary["runs"].append(metadata)
        del output, skeleton
        torch.cuda.empty_cache()
        gc.collect()
    if not panels:
        raise ValueError("No model panels were selected. Check --only.")
    render_info = render_outputs(panels, args.out_dir, args.fps, args.max_video_frames)
    summary["render"] = render_info
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=scalar_jsonable))
    print(f"[done] summary: {summary_path}")
    print(f"[done] preview: {render_info['preview']}")
    print(f"[done] video: {render_info['video']}")


if __name__ == "__main__":
    main()
