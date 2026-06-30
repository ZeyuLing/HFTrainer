#!/usr/bin/env python3
"""Validate canonical leaderboard manifests and result paths."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ANNO = ROOT / (
    "outputs/evaluation/t2m/humanml3d_official_test/captions/"
    "gt_motionclip_selected_20260622/"
    "test_hml3d_official272_gtlen_motionclip_selected_caption.json"
)
FORBIDDEN_PARTS = ("/prep/", "/_suites/", "/_runs/", "/predictions/motion135/")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.iterdir() if p.is_file() and p.suffix in {".npy", ".npz"})


def _stems(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {p.stem for p in path.iterdir() if p.is_file() and p.suffix in {".npy", ".npz"}}


def _official_ids() -> set[str]:
    raw = _load(ANNO)
    data = raw.get("data_list")
    if not isinstance(data, dict):
        raise RuntimeError(f"bad annotation format: {ANNO}")
    return set(data)


def _path(rel_path: str) -> Path:
    return ROOT / rel_path


def _check_path_policy(rel_path: str, errors: list[str]) -> None:
    norm = "/" + rel_path.strip("/") + "/"
    for bad in FORBIDDEN_PARTS:
        if bad in norm:
            errors.append(f"forbidden path part {bad}: {rel_path}")
    if not rel_path.startswith("outputs/evaluation/"):
        errors.append(f"path outside outputs/evaluation: {rel_path}")


def _check_method_segment(rel_path: str, method: str, errors: list[str]) -> None:
    parts = rel_path.strip("/").split("/")
    if len(parts) < 6 or parts[:2] != ["outputs", "evaluation"]:
        return
    actual = parts[5]
    if actual != method:
        errors.append(
            f"method path segment mismatch for {rel_path}: expected {method}, got {actual}"
        )


def validate_tp2m(manifest: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
    official = _official_ids()
    for proto in manifest.get("protocols", []):
        expected = int(proto.get("expected_count", 4042))
        if expected != len(official):
            errors.append(f"{proto.get('test_dataset')}: expected_count={expected}, official={len(official)}")
        for method in proto.get("methods", []):
            status = method.get("status")
            for rep, info in method.get("representations", {}).items():
                rel_path = info.get("path")
                if not rel_path:
                    continue
                _check_path_policy(rel_path, errors)
                _check_method_segment(rel_path, str(method.get("method")), errors)
                path = _path(rel_path)
                count = _count_files(path)
                if count != int(info.get("count", -1)):
                    errors.append(f"{rel_path}: manifest count={info.get('count')} actual={count}")
                if status in {"complete", "incomplete"} and count:
                    stems = _stems(path)
                    extra = sorted(stems - official)[:5]
                    missing = sorted(official - stems)[:5]
                    if extra:
                        errors.append(f"{rel_path}: non-official ids present, e.g. {extra}")
                    if status == "complete" and missing:
                        errors.append(f"{rel_path}: complete row missing official ids, e.g. {missing}")
                    if status == "complete" and count != expected:
                        errors.append(f"{rel_path}: complete row count {count} != {expected}")
                    if status == "incomplete":
                        warnings.append(f"{method.get('method')} {rep} incomplete: {count}/{expected} at {rel_path}")
            for metric_path in method.get("metrics", {}).values():
                _check_path_policy(str(metric_path), errors)
                _check_method_segment(str(metric_path), str(method.get("method")), errors)
                if not _path(str(metric_path)).exists():
                    errors.append(f"missing metric file: {metric_path}")


def validate_babel(manifest: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
    expected = int(manifest.get("expected_episodes", 1295))
    protocol_manifest = manifest.get("protocol_manifest")
    if protocol_manifest:
        _check_path_policy(str(protocol_manifest), errors)
        if not _path(str(protocol_manifest)).exists():
            errors.append(f"missing protocol manifest: {protocol_manifest}")
    for method in manifest.get("methods", []):
        status = method.get("status")
        for rep, info in method.get("representations", {}).items():
            rel_path = info.get("path")
            if not rel_path:
                continue
            _check_path_policy(rel_path, errors)
            _check_method_segment(rel_path, str(method.get("method")), errors)
            count = _count_files(_path(rel_path))
            if count != int(info.get("count", -1)):
                errors.append(f"{rel_path}: manifest count={info.get('count')} actual={count}")
            if status == "complete" and count != expected:
                errors.append(f"{rel_path}: complete row count {count} != {expected}")
            if status == "incomplete":
                warnings.append(f"{method.get('method')} {rep} incomplete: {count}/{expected} at {rel_path}")
        for metric_path in method.get("metrics", {}).values():
            _check_path_policy(str(metric_path), errors)
            _check_method_segment(str(metric_path), str(method.get("method")), errors)
            path = _path(str(metric_path))
            if not path.exists():
                errors.append(f"missing metric file: {metric_path}")
                continue
            metric = _load(path)
            if "n_episodes" in metric and int(metric["n_episodes"]) != expected:
                errors.append(f"{metric_path}: n_episodes={metric['n_episodes']} expected={expected}")
            if "n_segments" in metric and int(metric["n_segments"]) != int(manifest.get("expected_segments", 8441)):
                errors.append(f"{metric_path}: n_segments={metric['n_segments']}")
            if "n_transitions" in metric and int(metric["n_transitions"]) != int(manifest.get("expected_transitions", 8114)):
                errors.append(f"{metric_path}: n_transitions={metric['n_transitions']}")


def validate_reconstruction(manifest: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
    official = _official_ids()
    expected = int(manifest.get("expected_count", 4042))
    if expected != len(official):
        errors.append(f"{manifest.get('test_dataset')}: expected_count={expected}, official={len(official)}")
    for method in manifest.get("methods", []):
        method_name = str(method.get("method"))
        status = method.get("status")
        reps = method.get("representations", {})
        if "ms272" not in reps:
            errors.append(f"{method_name}: missing canonical ms272 representation")
        for rep, info in reps.items():
            rel_path = info.get("path")
            if not rel_path:
                continue
            _check_path_policy(rel_path, errors)
            _check_method_segment(rel_path, method_name, errors)
            path = _path(rel_path)
            count = _count_files(path)
            if count != int(info.get("count", -1)):
                errors.append(f"{rel_path}: manifest count={info.get('count')} actual={count}")
            if count:
                stems = _stems(path)
                extra = sorted(stems - official)[:5]
                missing = sorted(official - stems)[:5]
                if extra:
                    errors.append(f"{rel_path}: non-official ids present, e.g. {extra}")
                if status == "complete" and rep == "ms272" and missing:
                    errors.append(f"{rel_path}: complete row missing official ids, e.g. {missing}")
            if status == "complete" and rep == "ms272" and count != expected:
                errors.append(f"{rel_path}: complete ms272 row count {count} != {expected}")
            if status == "pending" and rep == "ms272" and count:
                warnings.append(f"{method_name} pending ms272 has partial files: {count}/{expected} at {rel_path}")
        for metric_path in method.get("metrics", {}).values():
            _check_path_policy(str(metric_path), errors)
            _check_method_segment(str(metric_path), method_name, errors)
            path = _path(str(metric_path))
            if not path.exists():
                errors.append(f"missing metric file: {metric_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifests", nargs="*", default=[
        "docs/leaderboards/tp2m_humanml3d.json",
        "docs/leaderboards/babel_sequential_t2m.json",
        "docs/leaderboards/reconstruction_humanml3d.json",
    ])
    args = ap.parse_args()
    errors: list[str] = []
    warnings: list[str] = []
    for rel_manifest in args.manifests:
        path = _path(rel_manifest)
        manifest = _load(path)
        if manifest.get("leaderboard") == "tp2m_humanml3d":
            validate_tp2m(manifest, errors, warnings)
        elif manifest.get("leaderboard") == "babel_sequential_t2m":
            validate_babel(manifest, errors, warnings)
        elif manifest.get("leaderboard") == "reconstruction_humanml3d":
            validate_reconstruction(manifest, errors, warnings)
        else:
            errors.append(f"unknown leaderboard manifest: {rel_manifest}")
    for msg in warnings:
        print(f"[warn] {msg}")
    if errors:
        for msg in errors:
            print(f"[error] {msg}", file=sys.stderr)
        return 1
    print(f"[ok] validated {len(args.manifests)} leaderboard manifest(s), warnings={len(warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
