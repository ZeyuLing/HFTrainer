"""Motion visualization protocols and diagnostics.

This module defines reusable *data contracts* for motion visualization. Web
viewers, offline renderers, and dataset/export scripts should consume these
protocol objects instead of hard-coding task semantics in a frontend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TaskVisualizationProtocol:
    """Task-level condition/generated contract for motion visualization."""

    key: str
    label: str
    group: str
    condition: str
    generated: str
    frame_mode: str
    note: str = ""
    condition_is_overlay: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PanelSpec:
    """Panel-level role contract.

    A panel is a source of motion data, not a visual widget. For example,
    ``reference`` means the motion is a condition source or GT reference, while
    ``generated`` means it is model output.
    """

    key: str
    label: str
    role: str
    role_label: str
    description: str
    required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrameSemantics:
    """Frame-level condition markers and generated ranges."""

    mode: str
    condition_frames: tuple[int, ...] = ()
    condition_ranges: tuple[tuple[int, int], ...] = ()
    generated_ranges: tuple[tuple[int, int], ...] = ()
    note: str = ""
    condition_is_overlay: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "condition_frames": list(self.condition_frames),
            "condition_ranges": [list(r) for r in self.condition_ranges],
            "generated_ranges": [list(r) for r in self.generated_ranges],
            "note": self.note,
            "condition_is_overlay": self.condition_is_overlay,
        }


def _ints(values: Any) -> tuple[int, ...]:
    if values is None:
        return ()
    try:
        arr = np.asarray(values).reshape(-1)
    except Exception:
        return ()
    out = []
    for value in arr:
        try:
            out.append(int(value))
        except Exception:
            continue
    return tuple(sorted(set(out)))


def _ranges(values: Any) -> tuple[tuple[int, int], ...]:
    if values is None:
        return ()
    out = []
    try:
        arr = np.asarray(values)
    except Exception:
        return ()
    if arr.ndim == 1 and arr.size == 2:
        arr = arr.reshape(1, 2)
    for row in arr:
        if len(row) < 2:
            continue
        try:
            start, end = int(row[0]), int(row[1])
        except Exception:
            continue
        if end < start:
            start, end = end, start
        out.append((start, end))
    return tuple(out)


def infer_frame_semantics(
    protocol: TaskVisualizationProtocol,
    num_frames: int,
    metadata: Mapping[str, Any] | None = None,
) -> FrameSemantics:
    """Infer frame semantics from a protocol plus optional motion metadata."""

    metadata = metadata or {}
    n = max(0, int(num_frames))
    last = max(0, n - 1)
    generated = ((0, last),) if n else ()
    mode = protocol.frame_mode

    if mode == "keyframes":
        frames = tuple(f for f in _ints(metadata.get("keyframe_indices")) if 0 <= f <= last)
        note = f"{len(frames)} keyframe condition markers" if frames else "keyframe metadata missing"
        return FrameSemantics(mode, frames, (), generated, note, protocol.condition_is_overlay)

    if mode == "endpoints":
        frames = (0, last) if n > 1 else ((0,) if n == 1 else ())
        return FrameSemantics(
            mode,
            frames,
            (),
            generated,
            "endpoint condition markers",
            protocol.condition_is_overlay,
        )

    if mode == "every_30":
        frames = tuple(range(0, n, 30))
        if n and last not in frames:
            frames = frames + (last,)
        return FrameSemantics(
            mode,
            frames,
            (),
            generated,
            "sparse constraint markers; exact target metadata was not exported",
            protocol.condition_is_overlay,
        )

    if mode == "continuous_control":
        ranges = _ranges(metadata.get("condition_ranges")) or (((0, last),) if n else ())
        return FrameSemantics(
            mode,
            (),
            ranges,
            generated,
            "condition is active as an overlay over generated motion",
            True,
        )

    if mode == "text_only":
        return FrameSemantics(
            mode,
            (),
            (),
            generated,
            "text prompt conditions the whole generated motion",
            True,
        )

    return FrameSemantics(
        mode or "metadata_missing",
        (),
        (),
        generated,
        "condition metadata is unavailable for this sample",
        protocol.condition_is_overlay,
    )


def panel_meta(panel_specs: Mapping[str, PanelSpec], present: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Return serializable panel metadata for present panels."""

    meta = {}
    for key in present:
        spec = panel_specs.get(key)
        if spec is None:
            meta[key] = {
                "key": key,
                "label": key,
                "role": key,
                "role_label": key,
                "description": "",
                "required": False,
            }
        else:
            meta[key] = spec.to_dict()
    return meta


def missing_panels(
    panel_specs: Mapping[str, PanelSpec],
    present: Sequence[str],
    reasons: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Describe absent panels, preserving explicit source-level reasons."""

    present_set = set(present)
    reasons = reasons or {}
    missing = {}
    for key, spec in panel_specs.items():
        if key in present_set:
            continue
        if key in reasons:
            missing[key] = reasons[key]
        elif spec.required:
            missing[key] = "required panel is missing"
    return missing


def continuity_stats(
    motion_135: np.ndarray,
    marker_frames: Sequence[int] = (),
) -> dict[str, Any]:
    """Compute simple adjacent-frame jump diagnostics for SMPL motion_135.

    This is intentionally representation-level and renderer-independent. It is
    used to distinguish a frontend color transition from a real motion
    discontinuity around condition markers.
    """

    motion = np.asarray(motion_135, dtype=np.float32)
    if motion.ndim != 2 or motion.shape[0] < 2:
        return {
            "num_edges": 0,
            "root_diff_p50": 0.0,
            "root_diff_max": 0.0,
            "pose_diff_p50": 0.0,
            "pose_diff_max": 0.0,
            "marker_edges": [],
        }
    root_diff = np.linalg.norm(np.diff(motion[:, :3], axis=0), axis=1)
    pose_diff = np.linalg.norm(np.diff(motion[:, 3:], axis=0), axis=1)
    marker_edges = []
    for frame in marker_frames:
        try:
            f = int(frame)
        except Exception:
            continue
        for edge in (f - 1, f):
            if 0 <= edge < root_diff.shape[0]:
                marker_edges.append({
                    "edge": int(edge),
                    "root_diff": float(root_diff[edge]),
                    "pose_diff": float(pose_diff[edge]),
                })
    return {
        "num_edges": int(root_diff.shape[0]),
        "root_diff_p50": float(np.percentile(root_diff, 50)),
        "root_diff_max": float(np.max(root_diff)),
        "pose_diff_p50": float(np.percentile(pose_diff, 50)),
        "pose_diff_max": float(np.max(pose_diff)),
        "marker_edges": marker_edges,
    }


def build_case_record(
    *,
    sid: str,
    task: str,
    caption: str,
    protocols: Mapping[str, TaskVisualizationProtocol],
    panels: Sequence[str],
    panel_specs: Mapping[str, PanelSpec],
    num_frames: int,
    metadata: Mapping[str, Any] | None = None,
    missing_reasons: Mapping[str, str] | None = None,
    source_paths: Mapping[str, str] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a serializable visualization manifest row."""

    protocol = protocols.get(task)
    if protocol is None:
        protocol = TaskVisualizationProtocol(
            key=task,
            label=task,
            group="Motion task",
            condition="task condition",
            generated="generated motion",
            frame_mode="metadata_missing",
        )
    semantics = infer_frame_semantics(protocol, num_frames, metadata)
    return {
        "sid": sid,
        "task": task,
        "caption": caption,
        "panels": list(panels),
        "task_meta": protocol.to_dict(),
        "panel_meta": panel_meta(panel_specs, panels),
        "missing_panels": missing_panels(panel_specs, panels, missing_reasons),
        "frame_semantics": semantics.to_dict(),
        "condition_summary": protocol.condition,
        "generated_summary": protocol.generated,
        "num_frames": int(num_frames),
        "source_paths": dict(source_paths or {}),
        "diagnostics": dict(diagnostics or {}),
    }
