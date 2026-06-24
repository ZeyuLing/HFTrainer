"""KIMODO visualization protocol registry.

The definitions here describe KIMODO task semantics independently of any web
viewer. Export scripts and renderers should use this registry when writing or
reading KIMODO visualization manifests.
"""

from __future__ import annotations

from hftrainer.motion.visualization.protocol import PanelSpec, TaskVisualizationProtocol


KIMODO_TASK_PROTOCOLS: dict[str, TaskVisualizationProtocol] = {
    "text_to_motion": TaskVisualizationProtocol(
        key="text_to_motion",
        label="Text-to-motion",
        group="T2M",
        condition="text prompt",
        generated="full generated motion",
        frame_mode="text_only",
    ),
    "fullbody_keyframe": TaskVisualizationProtocol(
        key="fullbody_keyframe",
        label="Full-body keyframes",
        group="Keyframe control",
        condition="specified full-body keyframes",
        generated="generated interpolation between keyframe constraints",
        frame_mode="keyframes",
        note="Keyframes are condition markers, not a separate generated stream.",
    ),
    "inbetween_endpoint_control": TaskVisualizationProtocol(
        key="inbetween_endpoint_control",
        label="Inbetween endpoints",
        group="Inbetweening",
        condition="start/end endpoint frames",
        generated="generated middle frames",
        frame_mode="endpoints",
    ),
    "first_frame_continuation": TaskVisualizationProtocol(
        key="first_frame_continuation",
        label="First-frame continuation",
        group="Continuation",
        condition="first pose frame",
        generated="motion generated after the first-frame condition",
        frame_mode="keyframes",
    ),
    "loop_animation": TaskVisualizationProtocol(
        key="loop_animation",
        label="Loop animation",
        group="Looping",
        condition="loop endpoint pose constraints",
        generated="generated motion closing the loop",
        frame_mode="endpoints",
    ),
    "end_effector_control": TaskVisualizationProtocol(
        key="end_effector_control",
        label="End-effector control",
        group="End-effector control",
        condition="hand/foot target frames",
        generated="full-body motion satisfying end-effector targets",
        frame_mode="keyframes",
    ),
    "root2d": TaskVisualizationProtocol(
        key="root2d",
        label="2D root path",
        group="Trajectory control",
        condition="root x/z path constraints",
        generated="full-body motion following the path",
        frame_mode="continuous_control",
    ),
    "foot_contact": TaskVisualizationProtocol(
        key="foot_contact",
        label="Foot contact",
        group="Contact control",
        condition="foot contact target ranges",
        generated="motion generated under foot contact constraints",
        frame_mode="continuous_control",
    ),
    "constraint_json": TaskVisualizationProtocol(
        key="constraint_json",
        label="Constraint JSON",
        group="Structured control",
        condition="saved KIMODO JSON constraints",
        generated="generated motion satisfying sparse constraints",
        frame_mode="every_30",
    ),
    "multi_prompt_or_edit": TaskVisualizationProtocol(
        key="multi_prompt_or_edit",
        label="Multi-prompt / local edit",
        group="Prompt/edit control",
        condition="segment prompts or local edit mask",
        generated="stitched or locally edited generated motion",
        frame_mode="continuous_control",
    ),
    "style_edit": TaskVisualizationProtocol(
        key="style_edit",
        label="Style edit",
        group="Prompt/edit control",
        condition="style instruction",
        generated="style-transferred generated motion",
        frame_mode="continuous_control",
    ),
    "bodypart_control": TaskVisualizationProtocol(
        key="bodypart_control",
        label="Body-part control",
        group="Body-part control",
        condition="upper/lower body constraint mask",
        generated="generated unconstrained body parts and transitions",
        frame_mode="continuous_control",
    ),
    "transition_stitching": TaskVisualizationProtocol(
        key="transition_stitching",
        label="Transition stitching",
        group="Transition",
        condition="motion A tail and motion B head ranges",
        generated="generated transition between the conditioned ranges",
        frame_mode="continuous_control",
    ),
    "prepend_start_pose": TaskVisualizationProtocol(
        key="prepend_start_pose",
        label="Prepend start pose",
        group="Transition",
        condition="target start pose and conditioned motion-A prefix",
        generated="generated prepend transition",
        frame_mode="continuous_control",
    ),
    "legacy_end_effector": TaskVisualizationProtocol(
        key="legacy_end_effector",
        label="End-effector control",
        group="Legacy sample",
        condition="hand/foot targets were not persisted in this legacy NPZ",
        generated="legacy generated motion only",
        frame_mode="metadata_missing",
        note="Regenerate E4 with target metadata for a complete condition-source visualization.",
    ),
}


KIMODO_PANEL_SPECS: dict[str, PanelSpec] = {
    "gt": PanelSpec(
        key="gt",
        label="GT / condition source",
        role="reference",
        role_label="reference",
        description="ground-truth motion used to derive constraints when available",
    ),
    "condition_smpl": PanelSpec(
        key="condition_smpl",
        label="Condition SMPL",
        role="condition_source",
        role_label="condition",
        description="condition-source motion shown only on frames/ranges where constraints are active",
    ),
    "kimodo_smpl": PanelSpec(
        key="kimodo_smpl",
        label="KIMODO generated SMPL",
        role="generated",
        role_label="generated",
        description="KIMODO output retargeted to SMPL for mesh inspection and evaluation",
        required=True,
    ),
    "kimodo_soma": PanelSpec(
        key="kimodo_soma",
        label="KIMODO generated SOMA",
        role="generated_native",
        role_label="generated native",
        description="native KIMODO SOMA mesh before SMPL retargeting",
    ),
}
