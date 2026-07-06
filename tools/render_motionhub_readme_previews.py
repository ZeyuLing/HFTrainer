#!/usr/bin/env python3
"""Render MotionHub README modality previews with a headless Three.js viewer.

The release README should show representative motion-language, music, speech,
and two-person interaction examples without requiring users to start the full
inspection app.  This script:

1. selects representative MotionHub annotations,
2. forwards the released SMPL-H NPZ files through the same SMPL-H convention,
3. writes temporary binary vertex streams consumed by a small Three.js page,
4. records the Three.js canvas with Playwright, and
5. encodes MP4 files for ``assets/readme_previews``.

Only the MP4 files and ``manifest.json`` are release artifacts.  Intermediate
vertex streams stay under ``outputs/temp``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import http.server
import json
import math
import os
import shutil
import socketserver
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import imageio_ffmpeg
import numpy as np
import torch
from playwright.sync_api import sync_playwright


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "motionhub"
DEFAULT_OUT_DIR = DEFAULT_DATA_ROOT / "assets" / "readme_previews"
DEFAULT_WORK_DIR = REPO_ROOT / "outputs" / "temp" / "motionhub_readme_previews"
THREE_SRC = REPO_ROOT / "motion_annot_web" / "score_m2m" / "static" / "three" / "three.module.js"

PREVIEWS: List[Dict[str, Any]] = [
    {
        "key": "text_motion",
        "title": "Text and Motion",
        "subset": "HumanML3D_AMASS",
        "split": "test",
        "description": "Text descriptions paired with SMPL-H motion",
        "color": "#68d391",
    },
    {
        "key": "music_dance",
        "title": "Music and Dance",
        "subset": "aist",
        "split": "test",
        "description": "Synchronized music and dance motion",
        "color": "#f6ad55",
        "audio_field": "music_path",
    },
    {
        "key": "speech_gesture",
        "title": "Speech and Gesture",
        "subset": "beat_v2.0.0",
        "split": "test",
        "description": "Speech audio or transcript paired with gesture motion",
        "color": "#9f7aea",
        "audio_field": "audio_path",
        "select_key": "beatv2_13_lu_0_73_73_1",
    },
    {
        "key": "two_person_interaction",
        "title": "Two-Person Interaction",
        "subset": "interx",
        "split": "test",
        "description": "Interaction text paired with two-person motion",
        "color": "#4fd1c5",
        "two_person": True,
        "use_2p_canonical": True,
        "select_key": "interx_G019T003A018R019_p1",
    },
]


VIEWER_HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MotionHub README Preview</title>
  <style>
    html, body { margin: 0; width: 100%; height: 100%; overflow: hidden; background: #11151c; font-family: Inter, Arial, sans-serif; }
    #stage { position: relative; width: 100vw; height: 100vh; background: #11151c; }
    canvas { display: block; width: 100%; height: 100%; }
    #hud {
      position: absolute; left: 28px; right: 28px; top: 22px;
      display: grid; grid-template-columns: auto 1fr; gap: 12px 18px; align-items: start;
      pointer-events: none;
    }
    #task {
      color: #f7fafc; font-weight: 760; font-size: 27px; line-height: 1.08; letter-spacing: 0;
      text-shadow: 0 2px 18px rgba(0,0,0,.48);
    }
    #desc {
      color: #d6e1ef; font-size: 17px; line-height: 1.3; max-width: 680px; text-shadow: 0 2px 18px rgba(0,0,0,.48);
      padding-top: 2px;
    }
    #caption {
      grid-column: 1 / -1; color: #f8fbff; font-size: 19px; line-height: 1.28; max-width: 1160px;
      text-shadow: 0 2px 18px rgba(0,0,0,.62);
      display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
    }
    #badge {
      position: absolute; right: 24px; bottom: 20px; color: #cbd5e0; font-size: 14px;
      padding: 7px 10px; border: 1px solid rgba(255,255,255,.18); background: rgba(12,16,22,.44);
    }
  </style>
  <script type="importmap">
    { "imports": { "three": "./static/three/three.module.js" } }
  </script>
</head>
<body>
  <div id="stage">
    <div id="hud">
      <div id="task"></div>
      <div id="desc"></div>
      <div id="caption"></div>
    </div>
    <div id="badge">MotionHub / SMPL-H</div>
  </div>
  <script type="module">
    import * as THREE from 'three';

    const params = new URLSearchParams(location.search);
    const key = params.get('case') || 'text_motion';
    const manifest = await (await fetch(`cases/${key}/manifest.json`)).json();

    document.querySelector('#task').textContent = manifest.title;
    document.querySelector('#desc').textContent = manifest.description;
    document.querySelector('#caption').textContent = manifest.caption || '';

    const stage = document.querySelector('#stage');
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, preserveDrawingBuffer: true });
    renderer.setPixelRatio(1);
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.shadowMap.enabled = true;
    stage.prepend(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x11151c);
    scene.add(new THREE.HemisphereLight(0xf8fbff, 0x273241, 1.16));
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.65);
    keyLight.position.set(3.4, 5.6, 4.2);
    keyLight.castShadow = true;
    scene.add(keyLight);
    const rimLight = new THREE.DirectionalLight(0xa7c6ff, 0.72);
    rimLight.position.set(-4.2, 2.4, -3.8);
    scene.add(rimLight);

    const floorSize = Math.max(8.5, manifest.camera.floor_size || 8.5);
    const floorCenter = manifest.camera.floor_center || [0, 0];
    const gridDivisions = Math.max(16, Math.ceil(floorSize * 2));
    const grid = new THREE.GridHelper(floorSize, gridDivisions, 0x465466, 0x28313d);
    grid.position.set(floorCenter[0], 0, floorCenter[1]);
    scene.add(grid);
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(floorSize, floorSize),
      new THREE.MeshStandardMaterial({ color: 0x151b24, roughness: 0.95, metalness: 0.0 })
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(floorCenter[0], -0.004, floorCenter[1]);
    floor.receiveShadow = true;
    scene.add(floor);

    const camera = new THREE.PerspectiveCamera(38, window.innerWidth / window.innerHeight, 0.01, 120);
    const facesBuffer = await (await fetch('faces.bin')).arrayBuffer();
    const faces = new Uint32Array(facesBuffer);
    const bodies = [];

    for (const body of manifest.bodies) {
      const buf = await (await fetch(`cases/${key}/${body.vertices}`)).arrayBuffer();
      const allVerts = new Float32Array(buf);
      const frameSize = body.vertex_count * 3;
      const frameVerts = new Float32Array(allVerts.slice(0, frameSize));
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(frameVerts, 3));
      geometry.setIndex(new THREE.BufferAttribute(faces, 1));
      geometry.computeVertexNormals();
      const material = new THREE.MeshStandardMaterial({
        color: new THREE.Color(body.color || manifest.color || '#68d391'),
        roughness: 0.62,
        metalness: 0.02,
        side: THREE.FrontSide
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.castShadow = true;
      mesh.receiveShadow = false;
      scene.add(mesh);
      bodies.push({ body, allVerts, frameSize, frameVerts, geometry, mesh });
    }

    const radius = Math.max(1.05, manifest.camera.radius);
    function setCamera() {
      const center = new THREE.Vector3(...manifest.camera.center);
      camera.position.set(center.x + radius * 0.18, center.y + radius * 0.54, center.z + radius * 2.55);
      camera.lookAt(center.x, center.y + radius * 0.12, center.z);
    }
    setCamera();

    function renderFrame(t) {
      const frame = Math.max(0, Math.min(manifest.frames - 1, Math.floor(t)));
      for (const item of bodies) {
        const off = frame * item.frameSize;
        item.frameVerts.set(item.allVerts.subarray(off, off + item.frameSize));
        item.geometry.attributes.position.needsUpdate = true;
        item.geometry.computeVertexNormals();
      }
      renderer.render(scene, camera);
    }

    window.renderFrame = renderFrame;
    window.NUM_FRAMES = manifest.frames;
    window.READY = true;
    renderFrame(0);

    window.addEventListener('resize', () => {
      renderer.setSize(window.innerWidth, window.innerHeight);
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderFrame(window.CURRENT_FRAME || 0);
    });
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--smpl-model-root", default=str(REPO_ROOT / "checkpoints" / "smpl_models"))
    parser.add_argument("--duration-sec", type=float, default=4.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--tasks", default="", help="Comma-separated task keys. Default: all.")
    parser.add_argument("--skip-record", action="store_true", help="Only prepare Three.js cache; do not record MP4.")
    parser.add_argument("--no-audio", action="store_true", help="Do not mux music/speech audio into preview videos.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def iter_items(split_obj: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    data = split_obj.get("data_list", split_obj)
    if isinstance(data, dict):
        yield from data.items()
    else:
        for idx, row in enumerate(data):
            yield str(idx), row


def load_split(data_root: Path, subset: str, split: str) -> Dict[str, Any]:
    path = data_root / subset / f"{split}.json"
    if not path.exists() and split == "test":
        path = data_root / subset / "train.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pick_item(data_root: Path, spec: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Dict[str, Dict[str, Any]]]:
    split = load_split(data_root, spec["subset"], spec.get("split", "test"))
    all_items = list(iter_items(split))
    by_key = {k: v for k, v in all_items}
    items = all_items
    where = spec.get("where") or {}
    if where:
        items = [
            (key, row) for key, row in items
            if all(str(row.get(field, "")).lower() == str(value).lower() for field, value in where.items())
        ]
        if not items:
            raise ValueError(f"{spec['key']}: no items match {where}")
    select_key = spec.get("select_key")
    if select_key:
        if select_key not in by_key:
            raise KeyError(f"{spec['key']}: select_key not found: {select_key}")
        row = by_key[select_key]
        return select_key, row, by_key
    offset = int(spec.get("item_offset", 0))
    if offset >= len(items):
        raise IndexError(f"{spec['key']}: item_offset={offset} >= {len(items)}")
    key, row = items[offset]
    return key, row, by_key


def read_text(path: Path, max_chars: int = 260) -> str:
    if not path.exists():
        return ""
    txt = path.read_text(encoding="utf-8", errors="replace").strip()
    txt = " ".join(txt.split())
    return txt[:max_chars].rstrip()


def caption_from_row(data_root: Path, row: Dict[str, Any], task: str) -> str:
    if task == "speech_gesture" and row.get("speech_script_path"):
        txt = read_text(data_root / row["speech_script_path"], max_chars=300)
        if txt:
            return txt
    cap_path = row.get("hierarchical_caption_path") or row.get("caption_path")
    if cap_path and (data_root / cap_path).exists():
        with (data_root / cap_path).open("r", encoding="utf-8") as f:
            cap = json.load(f)
        for field in ("macro", "meso", "micro"):
            value = cap.get(field)
            if isinstance(value, list) and value:
                return str(value[0])
            if isinstance(value, str) and value:
                return value
        for field in ("action", "caption", "text"):
            if cap.get(field):
                return str(cap[field])
    for field in ("caption", "text", "action"):
        if row.get(field):
            return str(row[field])
    return ""


def load_npz_components(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    poses = np.asarray(data["poses"], dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] != 156:
        raise ValueError(f"expected SMPL-H poses (T,156), got {poses.shape}: {path}")
    trans = np.asarray(data.get("trans", data.get("transl")), dtype=np.float32)
    betas = np.asarray(data.get("betas", np.zeros(10, dtype=np.float32)), dtype=np.float32).reshape(-1)
    betas = betas[:10]
    if betas.shape[0] < 10:
        betas = np.pad(betas, (0, 10 - betas.shape[0]))
    return {
        "global_orient": poses[:, 0:3],
        "body_pose": poses[:, 3:66],
        "left_hand_pose": poses[:, 66:111],
        "right_hand_pose": poses[:, 111:156],
        "transl": trans,
        "betas": betas.astype(np.float32),
    }


def sample_indices(frames: int, src_fps: float, duration_sec: float, out_fps: int) -> np.ndarray:
    usable = min(frames, max(1, int(round(duration_sec * src_fps))))
    count = max(1, int(round(duration_sec * out_fps)))
    return np.linspace(0, usable - 1, count).round().astype(np.int64)


def smplh_vertices(
    model: Any,
    npz_path: Path,
    indices: np.ndarray,
    chunk: int,
    device: torch.device,
) -> np.ndarray:
    comp = load_npz_components(npz_path)
    comp = {k: (v[indices] if k != "betas" else v) for k, v in comp.items()}
    verts: List[np.ndarray] = []
    for start in range(0, len(indices), chunk):
        sl = slice(start, start + chunk)
        n = len(comp["transl"][sl])
        kwargs = {
            "global_orient": torch.from_numpy(comp["global_orient"][sl]).to(device),
            "body_pose": torch.from_numpy(comp["body_pose"][sl]).to(device),
            "left_hand_pose": torch.from_numpy(comp["left_hand_pose"][sl]).to(device),
            "right_hand_pose": torch.from_numpy(comp["right_hand_pose"][sl]).to(device),
            "transl": torch.from_numpy(comp["transl"][sl]).to(device),
            "betas": torch.from_numpy(np.repeat(comp["betas"][None], n, axis=0)).to(device),
            "return_verts": True,
        }
        with torch.no_grad():
            out = model(**kwargs)
        verts.append(out.vertices.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(verts, axis=0)


def compute_camera(all_vertices: List[np.ndarray]) -> Dict[str, Any]:
    pts = np.concatenate([v.reshape(-1, 3) for v in all_vertices], axis=0)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    center = (lo + hi) / 2.0
    extent = hi - lo
    xz_extent = max(float(extent[0]), float(extent[2]))
    radius = float(max(xz_extent * 0.52, float(extent[1]) * 0.82, 1.25))
    center[1] = max(center[1], 0.85)
    floor_center = [float((lo[0] + hi[0]) * 0.5), float((lo[2] + hi[2]) * 0.5)]
    floor_size = float(max(10.0, xz_extent + 7.0))
    return {
        "center": [float(x) for x in center],
        "radius": radius,
        "floor_center": floor_center,
        "floor_size": floor_size,
    }


def two_person_motion_rows(row: Dict[str, Any], other: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any], str]]:
    rows: List[Tuple[str, Dict[str, Any], str]] = []
    for label, item, color in (("P1", row, "#4fd1c5"), ("P2", other, "#f6ad55")):
        motion_rel = item.get("smplh_path") or item.get("smplx_path") or item.get("motion_path")
        if not motion_rel:
            raise KeyError(f"missing motion path for {label}")
        motion_rel = motion_rel.replace("/smplh_52_1p/", "/smplh_52_2p/")
        clone = dict(item)
        clone["smplh_path"] = motion_rel
        clone["smplx_path"] = motion_rel
        rows.append((label, clone, color))
    return rows


def ensure_three(work_dir: Path) -> None:
    dst = work_dir / "static" / "three"
    dst.mkdir(parents=True, exist_ok=True)
    target = dst / "three.module.js"
    if target.exists():
        return
    if not THREE_SRC.exists():
        raise FileNotFoundError(f"missing local three.module.js: {THREE_SRC}")
    shutil.copy2(THREE_SRC, target)


def write_case_cache(
    work_dir: Path,
    spec: Dict[str, Any],
    row: Dict[str, Any],
    by_key: Dict[str, Dict[str, Any]],
    data_root: Path,
    model: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, Any], Optional[Path]]:
    case_dir = work_dir / "cases" / spec["key"]
    case_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Tuple[str, Dict[str, Any], str]] = [("P1" if spec.get("two_person") else "Motion", row, spec["color"])]
    if spec.get("two_person"):
        other_key = row.get("interactor_key")
        other = by_key.get(other_key) if other_key else None
        if other is None:
            raise KeyError(f"{spec['key']}: missing interactor_key={other_key!r}")
        if spec.get("use_2p_canonical"):
            rows = two_person_motion_rows(row, other)
        else:
            rows.append(("P2", other, "#f6ad55"))

    vertices: List[np.ndarray] = []
    bodies: List[Dict[str, Any]] = []
    frames = None
    src_fps = float(row.get("fps", 30) or 30)
    for idx, (label, item, color) in enumerate(rows):
        motion_rel = item.get("smplh_path") or item.get("smplx_path") or item.get("motion_path")
        if not motion_rel:
            raise KeyError(f"{spec['key']}: missing motion path")
        motion_path = data_root / motion_rel
        if not motion_path.exists():
            raise FileNotFoundError(motion_path)
        comp = load_npz_components(motion_path)
        inds = sample_indices(comp["transl"].shape[0], src_fps, args.duration_sec, args.fps)
        if frames is None:
            frames = len(inds)
        verts = smplh_vertices(model, motion_path, inds, args.chunk, device)
        out_name = f"body{idx}.verts.bin"
        verts.astype(np.float32).tofile(case_dir / out_name)
        vertices.append(verts)
        bodies.append(
            {
                "label": label,
                "color": color,
                "vertices": out_name,
                "vertex_count": int(verts.shape[1]),
                "motion_path": motion_rel,
            }
        )

    caption = caption_from_row(data_root, row, spec["key"])
    manifest = {
        "key": spec["key"],
        "title": spec["title"],
        "description": spec["description"],
        "subset": spec["subset"],
        "caption": caption,
        "frames": int(frames or 0),
        "fps": int(args.fps),
        "color": spec["color"],
        "bodies": bodies,
        "camera": compute_camera(vertices),
    }
    (case_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    audio_path = None
    audio_field = spec.get("audio_field")
    if audio_field and row.get(audio_field):
        candidate = data_root / row[audio_field]
        if candidate.exists():
            audio_path = candidate
    return manifest, audio_path


class QuietHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        return


@contextlib.contextmanager
def serve_directory(directory: Path) -> Iterable[str]:
    handler = lambda *a, **kw: QuietHTTPRequestHandler(*a, directory=str(directory), **kw)
    with socketserver.TCPServer(("127.0.0.1", 0), handler) as httpd:
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{port}"
        finally:
            httpd.shutdown()
            thread.join(timeout=5)


def encode_frames(frames_dir: Path, output: Path, fps: int) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def mux_audio(video: Path, audio: Path) -> None:
    tmp = video.with_name(video.stem + "_audio.mp4")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video),
        "-i",
        str(audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    subprocess.run(cmd, check=True)
    tmp.replace(video)


def record_case(base_url: str, key: str, out_path: Path, args: argparse.Namespace) -> None:
    frames_dir = out_path.with_suffix("").with_name(out_path.stem + "_frames")
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--use-gl=swiftshader",
                "--enable-webgl",
                "--ignore-gpu-blocklist",
                "--enable-unsafe-swiftshader",
            ],
        )
        page = browser.new_page(viewport={"width": args.width, "height": args.height}, device_scale_factor=1)
        page.goto(f"{base_url}/viewer.html?case={key}", wait_until="networkidle", timeout=120000)
        page.wait_for_function("window.READY === true", timeout=120000)
        n_frames = int(page.evaluate("window.NUM_FRAMES"))
        stage = page.locator("#stage")
        for i in range(n_frames):
            page.evaluate("(frame) => window.renderFrame(frame)", i)
            page.wait_for_timeout(15)
            stage.screenshot(path=str(frames_dir / f"frame_{i:05d}.png"))
            if (i + 1) % max(args.fps, 1) == 0:
                print(f"[record] {key}: {i + 1}/{n_frames}", flush=True)
        browser.close()
    encode_frames(frames_dir, out_path, args.fps)
    shutil.rmtree(frames_dir, ignore_errors=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    work_dir = Path(args.work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    ensure_three(work_dir)
    (work_dir / "viewer.html").write_text(VIEWER_HTML, encoding="utf-8")

    requested = {x.strip() for x in args.tasks.split(",") if x.strip()}
    specs = [s for s in PREVIEWS if not requested or s["key"] in requested]
    if requested:
        missing = requested - {s["key"] for s in specs}
        if missing:
            raise ValueError(f"unknown tasks: {sorted(missing)}")

    import smplx

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smplx.create(
        str(args.smpl_model_root),
        model_type="smplh",
        gender="neutral",
        use_pca=False,
        num_betas=10,
        batch_size=args.chunk,
    ).to(device)
    model.eval()
    faces = np.asarray(model.faces, dtype=np.uint32)
    faces.tofile(work_dir / "faces.bin")

    release_manifest: List[Dict[str, Any]] = []
    audio_by_key: Dict[str, Optional[Path]] = {}
    for spec in specs:
        print(f"[prepare] {spec['key']} from {spec['subset']}", flush=True)
        _, row, by_key = pick_item(data_root, spec)
        case_manifest, audio_path = write_case_cache(work_dir, spec, row, by_key, data_root, model, args, device)
        audio_by_key[spec["key"]] = audio_path
        release_manifest.append(
            {
                "key": spec["key"],
                "title": spec["title"],
                "description": spec["description"],
                "subset": spec["subset"],
                "video": f"{spec['key']}.mp4",
                "caption": case_manifest.get("caption", ""),
                "bodies": [
                    {"label": b["label"], "motion_path": b["motion_path"]}
                    for b in case_manifest["bodies"]
                ],
            }
        )

    if not args.skip_record:
        with serve_directory(work_dir) as base_url:
            for spec in specs:
                out_path = out_dir / f"{spec['key']}.mp4"
                if out_path.exists() and not args.force:
                    print(f"[skip] {out_path} exists; use --force to overwrite", flush=True)
                else:
                    print(f"[record] {spec['key']} -> {out_path}", flush=True)
                    record_case(base_url, spec["key"], out_path, args)
                    audio = audio_by_key.get(spec["key"])
                    if audio and not args.no_audio:
                        print(f"[audio] mux {audio}", flush=True)
                        mux_audio(out_path, audio)
                if out_path.exists():
                    for item in release_manifest:
                        if item["key"] == spec["key"]:
                            item["bytes"] = out_path.stat().st_size
                            item["sha256"] = sha256(out_path)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"previews": release_manifest}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
