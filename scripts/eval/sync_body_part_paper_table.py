#!/usr/bin/env python3
"""Synchronize completed body-part leaderboard metrics into paper Table 11.

The leaderboard is the metric source of truth. Position-control macro calls
and rotation-control target macros are expanded to explicit rows so
independently completed baseline cells can be filled without inventing values
for unfinished cells.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_LEADERBOARD = (
    REPO / "docs/leaderboards/body_part_condition_humanml3d.json"
)
DEFAULT_TABLE = (
    REPO
    / "papers/HYMotionM2M_ICLR2027/depds/tab_spatial_completion.tex"
)
TARGET_SLUG = {
    "upper body": "upper",
    "lower body": "lower",
    "left wrist": "wrist_left",
    "right wrist": "wrist_right",
    "both wrists": "wrist_both",
    "left elbow": "elbow_left",
    "right elbow": "elbow_right",
    "both elbows": "elbow_both",
    "left foot": "foot_left",
    "right foot": "foot_right",
    "both feet": "foot_both",
    "left knee": "knee_left",
    "right knee": "knee_right",
    "both knees": "knee_both",
}
METHOD_LATEX = {
    "kimodo": r"\methodkimodo",
    "maskcontrol": r"\methodmask",
    "omnicontrol": r"\methodomni",
    "condmdi": r"\methodcondmdi",
    "motionlab": r"\methodmotionlab",
    "ours": r"\ours{}",
}
LATEX_METHOD = {value: key for key, value in METHOD_LATEX.items()}
PLACEHOLDER = " & ".join(["--"] * 9)
EXPLICIT_ROW = re.compile(
    r"^(?P<target>[^&]+?) & Pos & (?P<density>sparse|dense) & "
    r"(?P<axes>XZ|XYZ) & (?P<method>\\(?:method\w+|ours\{\})) & "
    r"(?P<payload>.*?) \\\\$"
)
ROT_EXPLICIT_ROW = re.compile(
    r"^(?P<target>[^&]+?) & Rot & (?P<density>sparse|dense) & -- & "
    r"(?P<method>\\(?:method\w+|ours\{\})) & "
    r"(?P<payload>.*?) \\\\$"
)
ROT_MACRO = re.compile(
    r"^\\(?P<name>"
    r"rotplaceholderrows|rotmeasuredsparse|rotmeasureddense|rotmeasuredrows"
    r")"
)
OURS_MACRO = re.compile(
    r"^\\posoursmode\{(?P<density>sparse|dense)\}"
    r"\{(?P<axes>XZ|XYZ)\}\{(?P<target>[^{}]+)\}\{(?P<ours>.*)\}$"
)
MEASURED_TAIL = re.compile(
    r"^\\posmeasuredtail\{(?P<density>sparse|dense)\}"
    r"\{(?P<axes>XZ|XYZ)\}\{(?P<target>[^{}]+)\}"
    r"\{(?P<cond>.*)\}\{(?P<ours>.*)\}$"
)


def setting_id(target: str, density: str, axes: str) -> str:
    return f"pos_{TARGET_SLUG[target.strip()]}_{density}_{axes.lower()}"


def rotation_setting_id(target: str, density: str) -> str:
    return f"rot_{TARGET_SLUG[target.strip()]}_{density}"


def macro_args(line: str, pattern: re.Pattern[str]) -> tuple[str, list[str]] | None:
    """Parse one single-line LaTeX macro call with balanced brace arguments."""
    match = pattern.match(line)
    if match is None:
        return None
    args: list[str] = []
    index = match.end()
    while index < len(line):
        while index < len(line) and line[index].isspace():
            index += 1
        if index >= len(line):
            break
        if line[index] != "{":
            return None
        depth = 1
        start = index + 1
        index += 1
        while index < len(line) and depth:
            if line[index] == "{":
                depth += 1
            elif line[index] == "}":
                depth -= 1
            index += 1
        if depth:
            return None
        args.append(line[start:index - 1])
    return match.group("name"), args


def rotation_macro_settings(line: str) -> list[str]:
    parsed = macro_args(line, ROT_MACRO)
    if parsed is None:
        return []
    _, args = parsed
    if not args or args[0].strip() not in TARGET_SLUG:
        return []
    return [
        rotation_setting_id(args[0], "sparse"),
        rotation_setting_id(args[0], "dense"),
    ]


def line_setting(line: str) -> str | None:
    match = EXPLICIT_ROW.match(line)
    if match:
        return setting_id(
            match.group("target"), match.group("density"), match.group("axes")
        )
    match = ROT_EXPLICIT_ROW.match(line)
    if match and match.group("target").strip() in TARGET_SLUG:
        return rotation_setting_id(
            match.group("target"), match.group("density")
        )
    for pattern in (OURS_MACRO, MEASURED_TAIL):
        match = pattern.match(line)
        if match and match.group("target").strip() in TARGET_SLUG:
            return setting_id(
                match.group("target"),
                match.group("density"),
                match.group("axes"),
            )
    return None


def is_complete(method: dict[str, Any]) -> bool:
    return (
        method["artifacts"]["count"] == 4012
        and all(value is not None for value in method["metrics"].values())
    )


def is_reportable(method: dict[str, Any]) -> bool:
    return method["protocol_status"] != "unsupported" and is_complete(method)


def metric_payload(method: dict[str, Any]) -> str:
    metrics = method["metrics"]
    values = (
        f"{metrics['fid']:.4f}",
        f"{metrics['r_precision_top1']:.4f}",
        f"{metrics['r_precision_top2']:.4f}",
        f"{metrics['r_precision_top3']:.4f}",
        f"{metrics['mm_dist']:.2f}",
        f"{metrics['diversity']:.2f}",
        f"{metrics['control_error']:.2f}",
        f"{metrics['foot_skating']:.3f}",
        f"{metrics['jitter']:.1f}",
    )
    return " & ".join(values)


def existing_payloads(
    lines: list[str],
) -> dict[tuple[str, str], str]:
    payloads: dict[tuple[str, str], str] = {}
    for line in lines:
        match = EXPLICIT_ROW.match(line)
        if match:
            method = LATEX_METHOD.get(match.group("method"))
            if method:
                key = setting_id(
                    match.group("target"),
                    match.group("density"),
                    match.group("axes"),
                )
                payloads[(key, method)] = match.group("payload")
            continue
        match = ROT_EXPLICIT_ROW.match(line)
        if match and match.group("target").strip() in TARGET_SLUG:
            method = LATEX_METHOD.get(match.group("method"))
            if method:
                key = rotation_setting_id(
                    match.group("target"), match.group("density")
                )
                payloads[(key, method)] = match.group("payload")
            continue
        parsed = macro_args(line, ROT_MACRO)
        if parsed is not None:
            name, args = parsed
            if args and args[0].strip() in TARGET_SLUG:
                sparse = rotation_setting_id(args[0], "sparse")
                dense = rotation_setting_id(args[0], "dense")
                if name == "rotmeasuredrows" and len(args) == 5:
                    payloads[(sparse, "kimodo")] = args[1]
                    payloads[(sparse, "ours")] = args[2]
                    payloads[(dense, "kimodo")] = args[3]
                    payloads[(dense, "ours")] = args[4]
                elif name == "rotmeasuredsparse" and len(args) == 3:
                    payloads[(sparse, "kimodo")] = args[1]
                    payloads[(sparse, "ours")] = args[2]
                elif name == "rotmeasureddense" and len(args) == 3:
                    payloads[(dense, "kimodo")] = args[1]
                    payloads[(dense, "ours")] = args[2]
            continue
        match = OURS_MACRO.match(line)
        if match and match.group("target").strip() in TARGET_SLUG:
            key = setting_id(
                match.group("target"),
                match.group("density"),
                match.group("axes"),
            )
            payloads[(key, "ours")] = match.group("ours")
            continue
        match = MEASURED_TAIL.match(line)
        if match and match.group("target").strip() in TARGET_SLUG:
            key = setting_id(
                match.group("target"),
                match.group("density"),
                match.group("axes"),
            )
            payloads[(key, "condmdi")] = match.group("cond")
            payloads[(key, "ours")] = match.group("ours")
    return payloads


def render_setting(
    row: dict[str, Any],
    payloads: dict[tuple[str, str], str],
) -> list[str]:
    key = row["id"]
    target = row["target_label"]
    density = row["density"]
    axes = "--" if row["type"] == "rotation" else row["axes"].upper()
    methods = {method["method_id"]: method for method in row["methods"]}
    if row["type"] == "rotation":
        order = ["kimodo", "ours"]
        control_type = "Rot"
    else:
        order = ["kimodo"]
        mask = methods.get("maskcontrol")
        if mask and mask["protocol_status"] == "native":
            order.append("maskcontrol")
        order.extend(["omnicontrol", "condmdi", "motionlab", "ours"])
        control_type = "Pos"

    rendered: list[str] = []
    for method_id in order:
        method = methods.get(method_id)
        if method_id == "ours" and (key, method_id) in payloads:
            payload = payloads[(key, method_id)]
        elif method is not None and is_reportable(method):
            payload = metric_payload(method)
        else:
            payload = payloads.get((key, method_id), PLACEHOLDER)
        if method_id == "ours":
            rendered.append(r"\rowcolor{tablehl}")
        rendered.append(
            f"{target} & {control_type} & {density} & {axes} & "
            f"{METHOD_LATEX[method_id]} & {payload} \\\\"
        )
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaderboard", type=Path, default=DEFAULT_LEADERBOARD)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    leaderboard = json.loads(args.leaderboard.read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in leaderboard["settings"]}
    source_lines = args.table.read_text(encoding="utf-8").splitlines()
    payloads = existing_payloads(source_lines)

    output: list[str] = []
    emitted: set[str] = set()
    in_table = False
    skipped_rowcolor = False
    for index, line in enumerate(source_lines):
        if r"\begin{longtable}" in line:
            in_table = True
        if not in_table:
            output.append(line)
            continue

        rotation_keys = rotation_macro_settings(line)
        if rotation_keys:
            for key in rotation_keys:
                if key not in rows:
                    raise KeyError(f"paper setting missing from leaderboard: {key}")
                if key not in emitted:
                    output.extend(render_setting(rows[key], payloads))
                    emitted.add(key)
            skipped_rowcolor = False
            continue

        if line == r"\rowcolor{tablehl}":
            next_key = (
                line_setting(source_lines[index + 1])
                if index + 1 < len(source_lines)
                else None
            )
            if next_key is not None:
                skipped_rowcolor = True
                continue

        key = line_setting(line)
        if key is None:
            output.append(line)
            skipped_rowcolor = False
            continue
        if key not in rows:
            raise KeyError(f"paper setting missing from leaderboard: {key}")
        if key not in emitted:
            output.extend(render_setting(rows[key], payloads))
            emitted.add(key)
        skipped_rowcolor = False

    missing = sorted(set(rows) - emitted)
    if missing:
        raise RuntimeError(f"paper table did not expose settings: {missing}")
    text = "\n".join(output) + "\n"
    if not args.dry_run:
        args.table.write_text(text, encoding="utf-8")
    completed = sum(
        is_reportable(method)
        for row in rows.values()
        for method in row["methods"]
        if method["method_id"] != "ours"
    )
    print(
        f"[body-part-paper-sync] settings={len(emitted)} "
        f"completed_baseline_cells={completed} table={args.table}"
    )


if __name__ == "__main__":
    main()
