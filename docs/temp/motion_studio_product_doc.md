# HyMotion Studio — Web Animation Editor

## Product Document v2.0

> **Status**: Draft | **Date**: 2026-04-16
>
> **Design Philosophy**: This is first and foremost a **complete animation editing tool**.
> Even without any AI capability, it must be a usable, professional motion editor.
> HyMotion M2M V2 is the accelerator, not the foundation.

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [Core Editor Architecture](#2-core-editor-architecture)
3. [Editor Feature Specification](#3-editor-feature-specification)
4. [AI Acceleration Layer](#4-ai-acceleration-layer)
5. [Animator Workflow](#5-animator-workflow)
6. [UI/UX Design](#6-uiux-design)
7. [Data Architecture](#7-data-architecture)
8. [System Architecture](#8-system-architecture)
9. [API Specification](#9-api-specification)
10. [Implementation Plan](#10-implementation-plan)

---

## 1. Design Philosophy

### 1.1 Editor First, AI Second

The product exists in two layers:

```
Layer 2 (AI)   : HyMotion M2M V2 — accelerates filling, interpolation, repair
                 Can be disabled entirely; editor still functions 100%

Layer 1 (Core) : Complete animation editor — timeline, keyframes, poses,
                 curves, playback, import/export, undo/redo
                 THIS must work perfectly before any AI is introduced
```

The correct mental model: **Maya in a browser for SMPL body motion, with an AI copilot**.

### 1.2 What We Mean by "Complete Animation Editor"

A real animation editor must support the full manual workflow:

1. **Create a blank motion** — specify duration + FPS, get a T-pose sequence
2. **Set keyposes** — go to frame N, pose each joint, mark it as a keyframe
3. **Interpolate** — the editor fills frames between keyposes using cubic/linear interpolation
4. **Adjust curves** — fine-tune interpolation via a curve editor (easing, tangents)
5. **Scrub and preview** — play, scrub, loop, step frame-by-frame
6. **Edit continuously** — adjust any frame/joint, re-interpolate, refine
7. **Export** — save result in standard formats

This workflow must work with zero network connectivity. The AI layer is additive:

- Step 2 becomes faster: AI suggests natural poses from text/context
- Step 3 becomes smarter: AI fills gaps with physically plausible motion
- Step 6 becomes forgiving: AI can repair artifacts the animator doesn't want to fix by hand

### 1.3 Reference Points

| Tool | What We Take | What We Don't |
|------|-------------|---------------|
| **Maya** | Timeline architecture, Graph Editor concept, keyframe workflow, joint hierarchy panel, manipulator gizmos | Full plugin system, MEL/Python scripting, rendering pipeline |
| **MotionBuilder** | Character-centric workflow, story timeline (clip sequencing), pose controls | Real-time device input, full-body IK solving |
| **Blender (Pose Mode)** | Bone selection, pose library, NLA Editor for clip layering | Modifier stack, mesh editing, physics sim |
| **MotionEditor** (our baseline) | Three.js SMPL renderer, NPZ I/O, playback engine, joint click detection | URDF focus, limited to playback + slider editing |
| **Cascadeur** | AI-assisted posing, auto physics, interpolation between keyposes | Desktop-only, proprietary physics engine |

---

## 2. Core Editor Architecture

### 2.1 The Five Pillars

Every professional animation editor has five core subsystems. Our editor must implement all five
before any AI feature is considered.

```
 +---------------+     +---------------+     +------------------+
 |   Viewport    |     |   Timeline    |     |  Property Panel  |
 |  (3D Scene)   |     | (Time + Keys) |     | (Values + Curves)|
 +-------+-------+     +-------+-------+     +--------+---------+
         |                      |                      |
         +----------------------+----------------------+
                                |
                    +-----------+-----------+
                    |   Scene Data Model    |
                    |  (Clips, Keyframes,   |
                    |   Joints, Hierarchy)  |
                    +-----------+-----------+
                                |
                    +-----------+-----------+
                    |   Command System      |
                    |  (Undo/Redo, History) |
                    +-----------------------+
```

### 2.2 Scene Data Model

The central data structure that all UI modules read from and write to.

```typescript
/** A single animation project */
interface Project {
  name: string;
  fps: number;                        // Default 30
  duration: number;                   // In frames
  skeleton: SkeletonDefinition;       // SMPL 22-joint hierarchy
  tracks: Track[];                    // Multiple animation tracks
  activeTrackId: string;
  selection: SelectionState;
  playback: PlaybackState;
}

/** SMPL skeleton definition — fixed for this editor */
interface SkeletonDefinition {
  joints: JointDefinition[];          // 22 joints
  hierarchy: number[];                // Parent indices
  restPose: Float32Array;             // T-pose, 135 dims
}

interface JointDefinition {
  index: number;                      // 0-21
  name: string;                       // "pelvis", "left_hip", ...
  parent: number;                     // -1 for root
  group: JointGroup;                  // "torso", "left_arm", "right_leg", ...
  dimOffset: number;                  // Offset in 135-dim vector (0 for root transl, 3 for pelvis rot6d, ...)
  dimCount: number;                   // 3 for root translation, 6 for rot6d joints
}

type JointGroup = "root" | "torso" | "head" | "left_arm" | "right_arm" | "left_leg" | "right_leg";

/** A single animation clip on a track */
interface Track {
  id: string;
  name: string;
  visible: boolean;
  locked: boolean;
  clips: Clip[];
}

interface Clip {
  id: string;
  name: string;
  startFrame: number;                 // Position on timeline
  data: ClipData;                     // The actual motion data
  color: string;                      // Track color for visual distinction
}

/** Raw motion data for a clip */
interface ClipData {
  frameCount: number;
  fps: number;
  /** Dense per-frame data: Float32Array of shape [frameCount, 135] */
  frames: Float32Array;
  /** Sparse keyframe data — only keyframed frames have entries */
  keyframes: Map<number, Keyframe>;
  /** Interpolation settings per joint per keyframe pair */
  curves: CurveStore;
}

/** A keyframe stores the user-authored pose at a specific frame */
interface Keyframe {
  frame: number;
  /** Which joints are keyed at this frame (others inherit interpolation) */
  keyedJoints: Set<number>;           // Joint indices
  /** Tangent/easing data per joint */
  tangents: Map<number, TangentData>;
}

interface TangentData {
  inType: "auto" | "linear" | "step" | "smooth";
  outType: "auto" | "linear" | "step" | "smooth";
  inWeight: number;
  outWeight: number;
}

/** Selection state — what the user is currently working on */
interface SelectionState {
  currentFrame: number;
  selectedJoints: Set<number>;        // Joint indices
  selectedKeyframes: Set<string>;     // "frame:jointIndex" keys
  selectedFrameRange: [number, number] | null;
  hoveredJoint: number | null;
}

/** Playback state */
interface PlaybackState {
  isPlaying: boolean;
  isLooping: boolean;
  playbackRange: [number, number] | null;  // null = full range
  playbackSpeed: number;              // 1.0 = normal
}
```

### 2.3 Command System (Undo/Redo)

Every state mutation goes through a Command object. This is critical for professional editing.

```typescript
interface Command {
  id: string;
  label: string;                      // Human-readable description for history panel
  execute(): void;
  undo(): void;
}

/** Example commands: */
// SetJointValueCommand     — user drags a slider or gizmo
// InsertKeyframeCommand    — user presses K
// DeleteKeyframeCommand    — user presses Delete on a keyframe
// MoveKeyframeCommand      — user drags a keyframe on timeline
// PasteFramesCommand       — user pastes copied frames
// SetInterpolationCommand  — user changes curve type
// ImportMotionCommand      — user loads a file
// ApplyAIResultCommand     — user applies an AI operation result
// TrimClipCommand          — user trims clip edges
// DuplicateClipCommand     — user duplicates a clip
```

---

## 3. Editor Feature Specification

### 3.1 Viewport (3D Scene)

The viewport is the central visual workspace. It renders the SMPL mesh in real-time and provides
direct manipulation tools for posing.

#### 3.1.1 Rendering

| Feature | Description | Priority |
|---------|-------------|----------|
| SMPL Mesh | Render SMPL body mesh with per-vertex skinning, shadows, PBR lighting | P0 |
| Skeleton Overlay | Toggle bone visualization (lines connecting joints) | P0 |
| Ground Plane + Grid | Reference grid with scale markers, shadow receiving | P0 |
| Joint Markers | Clickable spheres at each joint position (when in pose mode) | P0 |
| Ghost Frames | Semi-transparent meshes showing frames before/after current (onion skinning) | P1 |
| Trajectory Path | Line rendering showing root translation path over time | P1 |
| Contact Markers | Visual indicators for foot-ground contact points | P2 |

#### 3.1.2 Camera

| Feature | Description | Priority |
|---------|-------------|----------|
| Orbit / Pan / Zoom | Standard Three.js OrbitControls | P0 |
| Frame Character | Press F to center camera on character | P0 |
| Root Lock Mode | Camera follows character root (existing in MotionEditor) | P0 |
| Camera Presets | Front / Back / Left / Right / Top / Perspective quick-switch (1-6 keys) | P1 |
| Camera Bookmarks | Save/restore custom camera angles | P2 |

#### 3.1.3 Manipulation Gizmos

The core tool for posing. This is what makes it an editor, not just a viewer.

| Feature | Description | Priority |
|---------|-------------|----------|
| Rotate Gizmo | Per-joint rotation rings (like Maya's rotate tool) for selected joint | P0 |
| Translate Gizmo | XYZ arrows for root joint translation | P0 |
| Joint Click Selection | Click on mesh/skeleton to select a joint | P0 |
| Multi-Joint Selection | Shift+Click to add joints, Ctrl+Click to toggle | P1 |
| Symmetry Mode | Edit left arm, right arm mirrors automatically | P1 |
| IK Handle | Drag end effector (hand/foot), solve chain via analytical IK | P2 |

**Implementation note**: Three.js `TransformControls` provides rotation/translation gizmos.
For joint rotation, we attach a gizmo to the selected bone in local space and convert
gizmo delta into rot6d updates on the ClipData.

### 3.2 Timeline

The timeline is the horizontal time axis. It shows clip duration, keyframe positions, and
supports scrubbing, selection, and keyframe manipulation.

#### 3.2.1 Timeline Structure

```
 Frame Numbers:     0    10    20    30    40    50    60    70    80    90
                    |     |     |     |     |     |     |     |     |     |
 Playhead:          |           ▼                                         |
                    |           |                                         |
 Summary Track:     |  ◆        ◆              ◆        ◆              ◆  |
 (all keyframes)    |                                                     |
                    |-----------------------------------------------------|
 Per-Joint Tracks:  (expandable, shown in curve editor mode)
   pelvis:          |  ◆                       ◆                       ◆  |
   left_hip:        |           ◆              ◆                          |
   right_arm:       |  ◆        ◆                        ◆              ◆  |
```

| Feature | Description | Priority |
|---------|-------------|----------|
| Frame Ruler | Numbered frame ruler at top, with tick marks | P0 |
| Playhead | Draggable vertical line showing current frame | P0 |
| Keyframe Diamonds | Diamond markers at keyframed frames (on summary track) | P0 |
| Scrubbing | Click/drag on ruler to scrub playhead | P0 |
| Timeline Zoom | Ctrl+Scroll to zoom in/out on time axis | P0 |
| Timeline Pan | Middle-drag to pan along time axis | P0 |
| Frame Range Selection | Click+drag to select a frame range (highlighted region) | P0 |
| Keyframe Selection | Click diamond to select, Shift+Click for multi-select | P0 |
| Keyframe Move | Drag selected keyframe diamonds to new frame positions | P1 |
| Keyframe Copy/Paste | Ctrl+C/V to copy/paste keyframe data | P1 |
| Clip Bar | Horizontal bar showing clip extent, draggable edges for trimming | P1 |
| Multi-Track View | Stacked tracks showing multiple clips | P2 |
| NLA-style Blending | Overlapping clips auto-blend with configurable blend mode | P2 |

#### 3.2.2 Playback Controls

| Feature | Description | Key | Priority |
|---------|-------------|-----|----------|
| Play / Pause | Toggle playback | Space | P0 |
| Stop (go to start) | Stop and return to frame 0 | Shift+Space | P0 |
| Step Forward | Advance one frame | Right Arrow | P0 |
| Step Backward | Back one frame | Left Arrow | P0 |
| Next Keyframe | Jump to next keyframe | Shift+Right | P0 |
| Prev Keyframe | Jump to previous keyframe | Shift+Left | P0 |
| Go to Start | Jump to frame 0 | Home | P0 |
| Go to End | Jump to last frame | End | P0 |
| Loop Toggle | Toggle looping playback | L | P0 |
| Speed Control | 0.25x / 0.5x / 1x / 2x / 4x | +/- | P1 |
| Ping-Pong | Play forward then backward | — | P2 |
| Play Range | Play only selected frame range | Alt+Space | P1 |

### 3.3 Property Panel

The right-side panel showing detailed values for the current frame and selected joints.

#### 3.3.1 Joint Value Editor

```
+------------------------------------------+
| Joint: left_shoulder                     |
| Group: left_arm                          |
+------------------------------------------+
| Rotation (axis-angle, degrees)           |
|   X:  [  45.2 ] ◆  (keyed)             |
|   Y:  [ -12.8 ]     (interpolated)      |
|   Z:  [   3.1 ] ◆  (keyed)             |
+------------------------------------------+
| Quick Pose:                              |
|   [T-Pose] [A-Pose] [Copy Mirror]       |
+------------------------------------------+
```

| Feature | Description | Priority |
|---------|-------------|----------|
| Joint Value Sliders | Numeric input + slider for each rotation axis | P0 |
| Root Translation | XYZ numeric inputs for root position | P0 |
| Keyed Indicator | Visual marker showing if value is keyframed at current frame | P0 |
| Value Scrubbing | Click+drag on numeric field to scrub value (like Maya) | P1 |
| Copy/Paste Pose | Copy all joint values at current frame, paste to another frame | P0 |
| Mirror Pose | Swap left/right joint values | P1 |
| Reset to T-Pose | Reset selected joints to rest pose | P0 |
| Reset to A-Pose | Reset to arms-down natural pose | P1 |
| Batch Key | Key all displayed values at current frame | P0 |

#### 3.3.2 Curve Editor

The Graph Editor equivalent. Shows interpolation curves between keyframes for selected joints.

```
+------------------------------------------+
| Curve Editor: left_shoulder              |
|                                          |
| Value ^                                  |
|  90   |         .--*                     |
|  45   |    *--''                         |
|   0   |  /                               |
| -45   | *                                |
|       +-----+-----+-----+------> Frame  |
|       0    30    60    90   120          |
|                                          |
| Interp: [Auto|Linear|Step|Smooth]        |
| Tangent: In[___] Out[___]                |
+------------------------------------------+
```

| Feature | Description | Priority |
|---------|-------------|----------|
| Curve Display | Plot interpolated values between keyframes | P1 |
| Interpolation Mode | Auto (cubic hermite), Linear, Step, Smooth per keyframe pair | P0 |
| Tangent Handles | Draggable Bezier handles on keyframe control points | P1 |
| Multi-Curve View | Overlay multiple joint curves with color coding | P2 |
| Value Snap | Snap to round values while dragging | P2 |

### 3.4 Joint Hierarchy Panel

Left-side panel showing the SMPL joint tree with selection, visibility, and lock controls.

```
+------------------------------------------+
| Hierarchy                        [Filter] |
+------------------------------------------+
| > Body                                   |
|   v Torso                                |
|     [eye][lock] pelvis               ◆   |
|     [eye][lock] spine1                   |
|     [eye][lock] spine2                   |
|     [eye][lock] spine3                   |
|   v Head                                 |
|     [eye][lock] neck                     |
|     [eye][lock] head                 ◆   |
|   v Left Arm                             |
|     [eye][lock] left_collar              |
|     [eye][lock] left_shoulder        ◆   |
|     [eye][lock] left_elbow               |
|     [eye][lock] left_wrist               |
|   v Right Arm                            |
|     ...                                  |
|   v Left Leg                             |
|     ...                                  |
|   v Right Leg                            |
|     ...                                  |
+------------------------------------------+
| Selection: 3 joints                      |
| [Select All] [Deselect] [Invert]        |
| [Select Group: Left Arm v]              |
+------------------------------------------+
```

| Feature | Description | Priority |
|---------|-------------|----------|
| Joint Tree | Hierarchical joint list grouped by body part | P0 |
| Click-to-Select | Click joint name to select it in viewport + property panel | P0 |
| Visibility Toggle | Eye icon to hide/show joint influence in viewport | P1 |
| Lock Toggle | Lock icon to prevent editing joint (useful during AI operations) | P1 |
| Group Select | Select entire joint group (e.g., "Left Arm") at once | P0 |
| Keyframe Indicator | Diamond icon in tree when joint is keyed at current frame | P0 |
| Search/Filter | Type to filter joint list | P2 |

### 3.5 Interpolation Engine

This is the critical non-AI system that makes the editor work as a standalone tool.
Between keyframes, values must be smoothly interpolated.

```typescript
/** Interpolation modes */
type InterpMode = "linear" | "cubic" | "step" | "smooth";

/**
 * Given keyframes K1 at frame f1 and K2 at frame f2,
 * compute the value at any frame f where f1 <= f <= f2.
 *
 * For rotation (rot6d), interpolation happens in axis-angle space:
 *   1. Keyframe stores axis-angle (3 values per joint)
 *   2. Interpolate axis-angle with selected curve
 *   3. Convert to rot6d for storage in ClipData.frames
 *
 * For translation (root), interpolation happens directly in XYZ space.
 */
function interpolateFrame(
  clip: ClipData,
  frame: number,
  joint: number,
  mode: InterpMode
): Float32Array;

/**
 * Recompute all interpolated frames between two keyframes.
 * Called after any keyframe insert/modify/delete.
 */
function rebakeInterpolation(clip: ClipData, fromFrame: number, toFrame: number): void;
```

**Interpolation modes**:
- **Linear**: Straight line between keyframe values. Simplest, often sufficient.
- **Cubic Hermite**: Smooth curve using tangent handles. Default "Auto" mode computes tangents from neighboring keyframes.
- **Step**: Hold previous keyframe value until next keyframe. Useful for snap transitions.
- **Smooth**: Catmull-Rom spline, guaranteed smooth through all keyframes.

**Rotation interpolation**:
- Store keyframes as axis-angle (3D, matches what the user edits in the property panel)
- Interpolate in axis-angle space (avoids gimbal lock for small rotations typical in body motion)
- Convert back to rot6d for the dense frame storage
- For large rotation differences (>180 degrees), use SLERP on quaternion representation

### 3.6 Import/Export

| Feature | Format | Direction | Priority |
|---------|--------|-----------|----------|
| NPZ (SMPL) | `.npz` with `poses`, `trans`, `betas` arrays | Import + Export | P0 |
| BVH | Standard `.bvh` with SMPL joint mapping | Import + Export | P1 |
| Project File | `.hmproj` JSON with embedded clip data | Save + Load | P0 |
| FBX | `.fbx` (server-side conversion) | Export only | P2 |
| Pose Library | `.hmpose` JSON with named poses | Import + Export | P1 |

NPZ import/export can be done entirely client-side (existing NumpyIO from MotionEditor).
BVH requires joint-order mapping. FBX requires server-side SDK.

### 3.7 Undo/Redo

| Feature | Description | Priority |
|---------|-------------|----------|
| Unlimited Undo | Every edit creates a Command, undo stack with no limit | P0 |
| Redo | Redo undone commands until new edit branches | P0 |
| History Panel | List of all commands with descriptions | P1 |
| Undo Grouping | Consecutive slider drags merge into single undo entry | P0 |

---

## 4. AI Acceleration Layer

AI features are presented as **tools in a toolbar**, not as the primary workflow.
They appear in a dedicated "AI Tools" panel and always follow the pattern:

```
1. User selects region/joints (or entire clip)
2. User chooses AI tool + parameters
3. AI generates result in a preview overlay
4. User accepts, modifies, or rejects
5. If accepted, result becomes editable keyframes in the normal editor
```

**Key principle**: AI results are always converted back to editable keyframes.
The animator never loses control.

### 4.1 AI Tool: Generate Pose

**What**: At the current frame, AI suggests a natural pose based on text or context.
**When**: Animator needs a starting pose but doesn't want to manually set 22 joints.
**How**: Uses T2M or M2M V2 to generate a single frame, which the animator can then tweak.

```
Trigger: AI Panel > "Generate Pose" button, or right-click frame > "AI: Suggest Pose"
Input:   Optional text ("standing with arms crossed"), optional context (neighboring keyframes)
Output:  Pose applied to current frame as a keyframe
Preview: Ghost overlay showing suggested pose in yellow
```

### 4.2 AI Tool: Fill Between Keyposes

**What**: Given two or more keyposes set by the animator, AI generates physically plausible
motion between them (instead of simple cubic interpolation).
**When**: Animator has set keyposes but cubic interpolation looks robotic/unnatural.
**How**: M2M V2 with M6 (keyframe sparse) mask — keyposes as inactive, gaps as masked.

```
Trigger: Select keyframe range > AI Panel > "AI Fill" button
Input:   Existing keyposes in the selected range
Output:  Dense frames overwriting the interpolated region
Preview: Side-by-side: left=cubic interp, right=AI fill
Accept:  AI result becomes dense keyframe data; user can still edit individual frames
```

**This is THE killer feature.** It replaces:
1. The animator manually posing 30+ intermediate frames
2. OR accepting bad cubic interpolation
3. WITH one-click natural motion that respects the keyposes exactly

### 4.3 AI Tool: Extend Motion

**What**: Extend a clip forward/backward with natural continuation.
**When**: Animator has a 2-second walk and needs 5 more seconds.
**How**: M2M V2 with M3 (temporal) mask — existing as prefix, extension as masked.

```
Trigger: Right-click clip end > "AI: Extend" or drag clip edge with Alt held
Input:   Existing clip + desired additional frames + optional text ("continue walking then stop")
Output:  New frames appended to clip
```

### 4.4 AI Tool: Motion from Text

**What**: Generate a full motion clip from a text description.
**When**: Starting from scratch, want a quick base motion to edit.
**How**: T2M Runtime + LLM rewriter (existing from completion_apps).

```
Trigger: AI Panel > "Text to Motion" or Ctrl+Shift+G
Input:   Text description + duration
Output:  New clip added to timeline, all frames as editable keyframes
```

### 4.5 AI Tool: Repair Region

**What**: Fix quality issues (jitter, foot skating, joint jumps) in a selected region.
**When**: After manual editing or concatenation, some frames look wrong.
**How**: M2M V2 with custom mask — damaged region as masked, surrounding as condition.

```
Trigger: Select frames > AI Panel > "AI Repair" or Ctrl+Shift+R
Input:   Selected frame range + optional joint selection (default: auto-detect)
Output:  Repaired frames replacing selected region
Preview: A/B comparison with quality metrics (foot skating, smoothness)
```

### 4.6 AI Tool: Smart Transition

**What**: Create a smooth transition between two clips on the timeline.
**When**: Animator has clip A and clip B, needs them to blend naturally.
**How**: M2M V2 with M3 mask — tail of A + head of B as condition, gap as masked.

```
Trigger: Place two clips on timeline with gap > select gap > "AI: Transition"
Input:   Clip A end (last K frames) + Clip B start (first K frames) + gap length
Output:  New clip filling the gap
```

### 4.7 AI Tool: Looping

**What**: Make a clip loop seamlessly (end matches start).
**When**: Game animation cycles need perfect loops.
**How**: M2M V2 with cyclic conditioning.

```
Trigger: Select clip > AI Panel > "AI: Make Loop"
Input:   Clip + transition frames count
Output:  Modified clip with matching start/end
```

### 4.8 AI Tool: Body Part Regenerate

**What**: Keep some joints fixed, regenerate others.
**When**: Legs are fine but arms look wrong; regenerate only the arms.
**How**: M2M V2 with M4 (joint contiguous) mask.

```
Trigger: Select joint group(s) in hierarchy panel > Lock others > "AI: Regenerate Selected"
Input:   Full clip + locked joints mask
Output:  New motion for selected joints while locked joints are unchanged
```

### 4.9 AI Feature Summary Table

| AI Tool | M2M V2 Mask | Trigger | Core Editor Prerequisite |
|---------|-------------|---------|--------------------------|
| Generate Pose | M5 (full, single frame) | AI Panel button | Frame navigation, joint editing |
| Fill Between | M6 (keyframe sparse) | Select range + button | Keyframe system, interpolation |
| Extend | M3 (temporal, tail) | Drag clip edge + Alt | Clip management, timeline |
| Text to Motion | M5 (full) + T2M | Ctrl+Shift+G | Import/export (to accept result) |
| Repair | M7 (scattered) / custom | Select region + button | Frame selection, joint selection |
| Transition | M3 (temporal, middle) | Select gap on timeline | Multi-clip timeline |
| Looping | Cyclic conditioning | Clip context menu | Clip playback |
| Body Part Regen | M4 (joint contiguous) | Lock joints + button | Joint lock, group selection |

---

## 5. Animator Workflow

### 5.1 Primary Workflow: "Keypose + AI Fill"

This is the expected dominant workflow for HyMotion Studio.

```
Step 1: Create Project
  - New Project → 30 FPS, 300 frames (10 sec)
  - Character appears in T-pose

Step 2: Block Out Keyposes
  - Frame 0:   Set feet shoulder-width, arms at sides        → Insert Keyframe [K]
  - Frame 30:  Rotate left_hip forward, right_arm back       → [K]  (start of walk step)
  - Frame 60:  Rotate right_hip forward, left_arm back       → [K]  (mid-stride)
  - Frame 90:  Mirror of frame 30                            → [K]
  - Frame 120: Mirror of frame 60                            → [K]

  At this point: cubic interpolation shows a rough walk cycle. Looks robotic.

Step 3: AI Fill
  - Select frame range 0-120
  - Click "AI: Fill Between Keyposes"
  - Preview shows AI-generated walk cycle that passes through exact keyposes
  - Result looks natural: has weight shift, secondary motion, foot contacts
  - Click "Accept"

Step 4: Refine
  - Scrub through. Frame 45: left foot clips ground slightly
  - Select frame 40-50, select left_ankle + left_foot joints
  - Click "AI: Repair Region" → foot skating fixed

  OR do it manually:
  - Go to frame 45, select left_ankle
  - Drag rotation gizmo to lift foot slightly → auto-keyframe
  - Neighboring frames re-interpolate smoothly

Step 5: Make Loop
  - Click "AI: Make Loop" with 15-frame transition
  - Start/end now match seamlessly

Step 6: Export
  - Export as NPZ for training data, or BVH for game engine import
```

### 5.2 Alternative Workflow: "Text Generate + Edit"

```
Step 1: Text to Motion
  - Type "person walks forward then sits down on a chair"
  - AI generates 180-frame motion

Step 2: Review + Edit
  - Scrub through, find the sitting transition is too fast
  - Select frame 120-140, drag timeline to stretch to frame 120-160
  - The stretch creates sparse keyframes — editor fills with cubic interp
  - Still looks off → "AI: Fill Between" on the stretched region

Step 3: Fix Artifacts
  - Hands clip through body at frame 95
  - Select frame 90-100, select left_arm + right_arm
  - Manually adjust arm poses at frame 90 and 100
  - AI Fill between 90-100 for the arms only

Step 4: Export
```

### 5.3 Alternative Workflow: "Import + Repair"

```
Step 1: Import
  - Drag .npz file onto editor (from mocap dataset)

Step 2: Identify Issues
  - Play: foot skating at frame 50-70, jitter at frame 120-130

Step 3: Auto Repair
  - Select frame 50-130 > "AI: Repair" (auto-detect mode)
  - Review result → foot skating gone, jitter smoothed

Step 4: Trim + Loop
  - Trim clip to frames 20-150
  - "AI: Make Loop" with 20-frame transition

Step 5: Export
```

---

## 6. UI/UX Design

### 6.1 Layout

```
+====================================================================+
| Menu: [Project v] [Edit v] [View v] [AI Tools v] [Help v]         |
| Toolbar: [Select|Rotate|Translate] [Key:K] [Play/Pause] [Ghost]   |
+========================+=========+=================================+
|                        |         |                                 |
|   Joint Hierarchy      |         |       Property Panel            |
|                        |         |                                 |
|  > Torso               |         |  Joint: left_shoulder           |
|    [e][l] pelvis     ◆ |         |  Rot X: [ 45.2] ◆              |
|    [e][l] spine1       |         |  Rot Y: [-12.8]                 |
|    [e][l] spine2       |         |  Rot Z: [  3.1] ◆              |
|    [e][l] spine3       |         |                                 |
|  > Head                |         |  [T-Pose] [Copy] [Paste]       |
|    [e][l] neck         |         |  [Key All] [Delete Key]        |
|    [e][l] head         |  3D     |                                 |
|  > Left Arm            |         +---------------------------------+
|    [e][l] l_collar     | View-   |                                 |
|    [e][l] l_shoulder ◆ | port    |       AI Tools Panel            |
|    [e][l] l_elbow      |         |                                 |
|    [e][l] l_wrist      |         |  [Generate Pose]                |
|  > Right Arm           |         |  [Fill Between Keyposes]        |
|    ...                 |         |  [Extend Motion]                |
|  > Left Leg            |         |  [Text to Motion]               |
|    ...                 |         |  [Repair Region]                |
|  > Right Leg           |         |  [Smart Transition]             |
|    ...                 |         |  [Make Loop]                    |
|                        |         |  [Body Part Regen]              |
|  Selection: 1 joint    |         |                                 |
|  [Group: Left Arm v]   |         |  Text: [________________]      |
|                        |         |  Model: [uncond_flow v]         |
|                        |         |  [Run] [Preview] [Apply]       |
+========================+=========+=================================+
| Timeline                                                           |
| [|<][<][>][>|]  Frame: 045/300  FPS: 30  Speed: 1.0x  Loop: ON   |
|                                                                    |
|  0    30    60    90    120   150   180   210   240   270   300    |
|  |     |     |     |     |     |     |     |     |     |     |    |
|  ◆-----------◆-----------◆-----------◆-----------◆               |
|         ▼ (playhead)                                               |
|  [======|========================|===============]  walk_cycle     |
|                                                                    |
| Curve Editor (toggled):                                            |
|  Value ^                                                           |
|   90  |         .--*                                               |
|   45  |    *--''                                                   |
|    0  |--*                                                         |
|       +-------+-------+-------+--------> Frame                    |
+====================================================================+
| Status: Ready | GPU: 12.3GB | Model: uncond_flow | Undo: 5 steps  |
+====================================================================+
```

### 6.2 Panel Sizing

- **Hierarchy Panel**: 220px fixed width, left side, collapsible
- **3D Viewport**: Fills remaining center space, minimum 600x400
- **Property Panel**: 280px fixed width, right side, collapsible
- **Timeline**: Full width bottom, 200px default height, resizable (drag top border)
- **Curve Editor**: Replaces timeline area when toggled (Tab to switch)
- All panels: draggable borders, remember last layout

### 6.3 Color Coding

| Element | Color | Hex | Usage |
|---------|-------|-----|-------|
| Keyframed value | Gold | #FFD700 | Diamond markers, keyed indicators |
| Selected joint | Cyan | #00E5FF | Viewport highlight, hierarchy highlight |
| Hovered joint | Light blue | #80D8FF | Viewport hover state |
| Locked joint | Red-orange | #FF6D00 | Lock icons, dimmed in viewport |
| AI preview | Yellow-green | #C6FF00 | Ghost mesh showing AI result |
| AI region (masked) | Soft red | #FF5252 (30% opacity) | Timeline region to be regenerated |
| AI region (condition) | Soft blue | #448AFF (30% opacity) | Timeline region used as condition |
| Playing | Green | #00E676 | Play button, timeline cursor |
| Timeline background | Dark gray | #1A1A2E | Standard dark theme |
| Grid | Muted blue | #26485C | Viewport grid lines |

### 6.4 Keyboard Shortcuts

| Key | Action | Category |
|-----|--------|----------|
| **Space** | Play / Pause | Playback |
| **Left / Right** | Step frame backward / forward | Playback |
| **Shift+Left / Right** | Jump to prev / next keyframe | Playback |
| **Home / End** | Go to first / last frame | Playback |
| **L** | Toggle loop | Playback |
| **K** | Insert keyframe (all selected joints at current frame) | Keyframe |
| **Shift+K** | Insert keyframe (all joints at current frame) | Keyframe |
| **Delete** | Delete selected keyframes | Keyframe |
| **Ctrl+C / Ctrl+V** | Copy / Paste pose (all joints at current frame) | Editing |
| **Ctrl+Shift+V** | Paste mirrored pose | Editing |
| **Ctrl+Z** | Undo | Editing |
| **Ctrl+Shift+Z** | Redo | Editing |
| **W** | Translate mode (root) | Gizmo |
| **E** | Rotate mode (joints) | Gizmo |
| **F** | Frame character (center camera) | Viewport |
| **1-6** | Camera presets | Viewport |
| **G** | Toggle ghost/onion skin | Viewport |
| **J** | Toggle skeleton overlay | Viewport |
| **Tab** | Toggle timeline / curve editor | Panel |
| **Ctrl+Shift+G** | AI: Text to Motion | AI Tools |
| **Ctrl+Shift+F** | AI: Fill Between Keyposes | AI Tools |
| **Ctrl+Shift+R** | AI: Repair Region | AI Tools |
| **Ctrl+S** | Save project | File |
| **Ctrl+O** | Open project / import file | File |
| **Ctrl+Shift+E** | Export motion | File |

---

## 7. Data Architecture

### 7.1 Motion Representation

Internal representation uses 135-dim per frame (same as M2M V2):

```
Frame[i] = [transl_x, transl_y, transl_z,    // 3 dims: root translation
            pelvis_r6d[0..5],                  // 6 dims: pelvis rotation (rot6d, row-major)
            left_hip_r6d[0..5],                // 6 dims
            right_hip_r6d[0..5],               // 6 dims
            spine1_r6d[0..5],                  // ...
            ...                                // 22 joints total
            right_wrist_r6d[0..5]]             // 6 dims

Total: 3 + 22*6 = 135 dimensions
```

SMPL joint order (22 joints, indices 0-21):
```
0:  pelvis        1:  left_hip       2:  right_hip      3:  spine1
4:  left_knee     5:  right_knee     6:  spine2         7:  left_ankle
8:  right_ankle   9:  spine3         10: left_foot      11: right_foot
12: neck          13: left_collar    14: right_collar   15: head
16: left_shoulder 17: right_shoulder 18: left_elbow     19: right_elbow
20: left_wrist    21: right_wrist
```

### 7.2 Editor Internal vs. Display Representations

```
Storage (ClipData.frames):  rot6d (row-major) — matches M2M V2 exactly
Keyframe editing:           axis-angle (3D) — intuitive for sliders/gizmos
3D Display:                 quaternion — Three.js bone.quaternion
Property Panel:             degrees (Euler or axis-angle magnitude) — human-readable

Conversion chain:
  axis-angle ←→ rot6d     (for keyframe ↔ dense frame storage)
  axis-angle → quaternion  (for 3D rendering; existing in SmplMotionPlayer)
  axis-angle → degrees     (for property panel display)
```

### 7.3 Project File Format

```json
{
  "version": "1.0",
  "name": "walk_cycle_v3",
  "fps": 30,
  "duration": 300,
  "tracks": [
    {
      "id": "track-1",
      "name": "Main",
      "clips": [
        {
          "id": "clip-1",
          "name": "walk_cycle",
          "startFrame": 0,
          "frameCount": 120,
          "fps": 30,
          "frames_base64": "... (Float32Array → base64)",
          "keyframes": [
            { "frame": 0, "keyedJoints": [0,1,2,...21], "tangents": {...} },
            { "frame": 30, "keyedJoints": [1,4,16,18], "tangents": {...} },
            ...
          ]
        }
      ]
    }
  ],
  "poseLibrary": [
    { "name": "T-Pose", "pose_base64": "..." },
    { "name": "A-Pose", "pose_base64": "..." }
  ]
}
```

### 7.4 Client-Side vs. Server-Side Split

```
CLIENT (browser, TypeScript, zero-latency):
  - All editing operations (set joint value, insert keyframe, move keyframe)
  - Interpolation engine (cubic, linear, step, smooth)
  - Undo/redo command stack
  - 3D rendering (Three.js SMPL)
  - Timeline rendering (Canvas 2D)
  - NPZ import/export (NumpyIO.ts)
  - Project save/load (JSON)
  - Rotation conversion (rot6d ↔ axis-angle ↔ quaternion)
  - Pose copy/paste/mirror

SERVER (Flask+Python, GPU-required):
  - AI operations only (M2M V2 inference)
  - BVH ↔ NPZ conversion (complex joint remapping)
  - FBX export (requires SDK)
  - T2M text processing (LLM rewriter)
```

**Important**: The editor works 100% offline without the server. AI features show "Server
unavailable" and are grayed out. All manual editing, interpolation, playback, and NPZ I/O
work locally in the browser.

---

## 8. System Architecture

### 8.1 Frontend

**Technology**: TypeScript + Vite + Three.js

```
motion_studio/
  frontend/
    index.html
    package.json                       # three, vite, typescript
    tsconfig.json
    vite.config.ts
    src/
      main.ts                          # Entry point

      # --- Core Data Layer ---
      core/
        Project.ts                     # Project data model
        ClipData.ts                    # Clip data + keyframe storage
        Skeleton.ts                    # SMPL 22-joint definition
        Selection.ts                   # Selection state management
        CommandStack.ts                # Undo/redo (Command pattern)

      # --- Math ---
      math/
        Rot6d.ts                       # rot6d ↔ axis-angle ↔ rotation matrix
        Quaternion.ts                  # Quaternion utilities
        Interpolation.ts              # Linear, cubic, step, smooth interpolation
        ForwardKinematics.ts          # Joint positions from rotations (for display)
        MirrorPose.ts                 # Left/right joint swapping

      # --- 3D Viewport ---
      viewport/
        ViewportController.ts         # Main viewport orchestrator
        SceneSetup.ts                 # Lights, grid, camera (from MotionEditor)
        SmplRenderer.ts               # SMPL mesh + skeleton rendering
        GizmoManager.ts               # TransformControls for joint manipulation
        GhostRenderer.ts              # Onion skin (semi-transparent past/future frames)
        JointMarkers.ts               # Clickable spheres at joint positions
        TrajectoryRenderer.ts         # Root path line
        CameraPresets.ts              # Front/Side/Top/Perspective

      # --- Timeline ---
      timeline/
        TimelineView.ts               # Canvas-based timeline rendering
        Playhead.ts                   # Current frame indicator + scrubbing
        KeyframeTrack.ts              # Keyframe diamond rendering + interaction
        FrameRuler.ts                 # Numbered tick marks
        RegionSelector.ts             # Frame range selection
        ClipBar.ts                    # Clip extent visualization
        CurveEditor.ts               # Graph editor for interpolation curves

      # --- Panels ---
      panels/
        HierarchyPanel.ts            # Joint tree with select/lock/eye
        PropertyPanel.ts             # Joint value sliders + keyframe buttons
        AIToolsPanel.ts              # AI operation buttons + parameters
        MenuBar.ts                   # Top menu bar
        Toolbar.ts                   # Tool selection bar
        StatusBar.ts                 # Bottom status: GPU, model, undo count

      # --- Playback ---
      playback/
        PlaybackEngine.ts            # Frame timing, play/pause/step/loop
        AudioSync.ts                 # Future: audio playback sync

      # --- I/O ---
      io/
        NpzIO.ts                     # NPZ import/export (from MotionEditor NumpyIO)
        BvhIO.ts                     # BVH import (client-side parsing)
        ProjectIO.ts                 # .hmproj save/load
        PoseLibrary.ts               # Named poses save/load

      # --- AI Client ---
      ai/
        AIClient.ts                  # REST API client for AI operations
        MaskPreview.ts               # Visualize AI mask on timeline (red/blue regions)
        AIResultPreview.ts           # A/B comparison overlay in viewport

      # --- Types ---
      types/
        project.ts                   # All TypeScript interfaces
        constants.ts                 # SMPL joint names, groups, hierarchy
```

### 8.2 Backend

**Technology**: Flask (Python), port 8100

```
motion_studio/
  backend/
    app.py                           # Flask main, CORS, static serving

    # --- AI Engine ---
    ai_engine.py                     # Unified AI operation dispatcher
    mask_builder.py                  # Build VACE masks for each operation type
    model_manager.py                 # GPU model loading/caching (from completion_apps)

    # --- Format Conversion ---
    converters/
      npz_utils.py                   # NPZ read/write helpers
      bvh_converter.py               # BVH ↔ 135-dim conversion
      fbx_converter.py               # FBX export via SDK

    # --- API Routes ---
    routes/
      ai_routes.py                   # /api/ai/* endpoints
      convert_routes.py              # /api/convert/* endpoints
      system_routes.py               # /api/system/* endpoints
```

### 8.3 Deployment

```
Development:
  Terminal 1: cd frontend && npm run dev        # Vite dev server (HMR) on :5173
  Terminal 2: cd backend && python app.py       # Flask on :8100
  Vite proxies /api/* to Flask

Production:
  npm run build → dist/                         # Static assets
  Flask serves dist/ as static + API
  Single port :8100
```

---

## 9. API Specification

### 9.1 AI Operation Endpoints

#### POST `/api/ai/run`

Synchronous AI operation (simpler than async polling for initial version).

**Request:**
```json
{
  "operation": "fill_between | extend | text_to_motion | repair | transition | loop | body_part_regen | generate_pose",
  "model_variant": "uncond_flow",
  "params": {
    // Operation-specific (see §4)
  },
  "motion": {
    "frames_base64": "...",
    "num_frames": 180,
    "fps": 30
  },
  "keyframes": [
    { "frame": 0, "keyed_joints": [0,1,2,...] },
    { "frame": 60, "keyed_joints": [0,1,2,...] }
  ],
  "text": "optional caption",
  "seed": 42
}
```

**Response:**
```json
{
  "success": true,
  "motion": {
    "frames_base64": "...",
    "num_frames": 180,
    "fps": 30
  },
  "metadata": {
    "operation": "fill_between",
    "elapsed_seconds": 8.3,
    "model_variant": "uncond_flow"
  }
}
```

#### POST `/api/ai/preview`

Same as `/run` but uses fewer ODE steps (10 instead of 50) for faster preview.

### 9.2 System Endpoints

#### GET `/api/system/status`

```json
{
  "gpu_available": true,
  "gpu_memory_used_gb": 12.3,
  "gpu_memory_total_gb": 16.0,
  "loaded_models": ["uncond_flow"],
  "available_models": ["uncond_flow", "uncond_jit", "caption_flow", "caption_jit"]
}
```

#### POST `/api/system/load_model`

```json
{ "model_variant": "caption_flow" }
```

### 9.3 Conversion Endpoints

#### POST `/api/convert/bvh_to_135`

Upload BVH, receive 135-dim frames (for complex retargeting that can't run client-side).

#### POST `/api/convert/135_to_fbx`

Upload 135-dim frames, receive FBX binary.

---

## 10. Implementation Plan

### Phase 1: Core Editor (3 weeks)

**Goal**: A functional standalone animation editor. No AI. No server dependency.

| Week | Module | Deliverables |
|------|--------|-------------|
| W1 | Viewport | Three.js scene, SMPL mesh rendering, skeleton overlay, orbit camera, ground grid. Joint markers with click-to-select. Rotation gizmo on selected joint. |
| W1 | Core Data | Project/Clip/Keyframe data model. Command stack (undo/redo). SMPL skeleton definition (22 joints, hierarchy, rest pose). |
| W1 | I/O | NPZ import (client-side, from MotionEditor's NumpyIO). NPZ export. Basic project save/load. |
| W2 | Timeline | Canvas-based timeline: frame ruler, playhead scrubbing, keyframe diamond markers. Playback engine (play/pause/step/loop). |
| W2 | Property Panel | Joint value sliders (axis-angle, degrees). Root translation XYZ. Insert/delete keyframe buttons. Copy/paste pose. |
| W2 | Interpolation | Linear + cubic hermite interpolation engine. Auto-rebake on keyframe changes. |
| W3 | Hierarchy | Joint hierarchy tree panel with group headers, click-to-select, multi-select. Group select buttons. |
| W3 | Polish | Keyboard shortcuts. Menu bar. Status bar. Layout with resizable panels. Dark theme CSS. |

**Milestone**: Import an NPZ, see it in 3D, scrub timeline, select joints, edit rotations via gizmo,
insert keyframes, interpolation fills gaps, undo/redo works, export NPZ. **All offline.**

### Phase 2: AI Integration (2 weeks)

**Goal**: Connect AI features to the editor.

| Week | Module | Deliverables |
|------|--------|-------------|
| W4 | Backend | Flask app with model_manager (reuse from completion_apps). AI engine with mask_builder. `/api/ai/run` endpoint for: fill_between, text_to_motion, repair. |
| W4 | AI Client | Frontend AI client (fetch + loading state). AI Tools panel with buttons + text input. Frame range selection for AI operations. |
| W5 | AI UX | AI result preview (yellow ghost mesh). Accept/Reject flow. AI result → keyframe conversion. A/B comparison toggle. Mask visualization on timeline (red/blue regions). |
| W5 | More AI | Extend, transition, loop, body part regen. Model variant selector. Seed control. |

**Milestone**: Full "Keypose + AI Fill" workflow works end-to-end. Repair and text-to-motion
also functional.

### Phase 3: Professional Polish (2 weeks)

**Goal**: Features that make it a genuinely useful tool.

| Week | Module | Deliverables |
|------|--------|-------------|
| W6 | Curve Editor | Graph editor view (Tab to toggle from timeline). Per-joint curves. Tangent handles. Interpolation mode selector (auto/linear/step/smooth). |
| W6 | Ghost/Onion | Onion skinning (N frames before/after current as transparent ghosts). Trajectory path visualization. |
| W6 | Advanced Edit | Symmetry editing mode. Pose library (save/load named poses). Mirror paste. Lock joints. |
| W7 | I/O | BVH import/export. Camera presets (1-6). Multi-clip timeline (track stacking). |
| W7 | Polish | Responsive panel layout. Loading indicators. Error handling. Tooltips. Help overlay. Performance optimization for long clips. |

**Milestone**: Professional-quality tool suitable for animation production.

### Dependencies & Reuse

| Existing Asset | Reuse Plan |
|----------------|-----------|
| `MotionEditor/src/io/motion/NumpyIO.ts` | Direct copy: NPZ parsing (parseNpzFile) |
| `MotionEditor/src/io/motion/SmplMotionService.ts` | Reference: SMPL model loading, skinning math |
| `MotionEditor/src/motion/SmplMotionPlayer.ts` | Reference: axis-angle → quaternion, frame application |
| `MotionEditor/src/viewer/SceneController.ts` | Reference: Three.js scene setup, lighting, grid, camera |
| `completion_apps/model_manager.py` | Direct reuse: ModelManager class, MODEL_REGISTRY |
| `completion_apps/inference_engine.py` | Reference: mask building, alignment, normalization |
| `scripts/m2m/hymotion_completion_pipeline.py` | Reference: VACE batch construction, ODE solving |

### Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Three.js TransformControls for bone rotation is non-trivial | P0 blocked | Prototype gizmo in W1 day 1; fallback to slider-only editing |
| Canvas timeline performance with many keyframes | Degraded UX | Virtual rendering (only draw visible region); tested up to 10K frames |
| AI inference latency (5-15s) interrupts editing flow | Frustrating UX | Preview mode (10-step, 1-2s), non-blocking UI, clear progress indicator |
| Rot6d ↔ axis-angle conversion edge cases (gimbal lock) | Visual artifacts | Unit test suite for rotation conversions; use quaternion intermediary for large rotations |
| SMPL model NPZ loading in browser (large file) | Slow first load | Pre-bundle a default SMPL neutral model; lazy load alternate genders |

---

## Appendix A: SMPL Joint Hierarchy

```
pelvis (0)
├── left_hip (1)
│   └── left_knee (4)
│       └── left_ankle (7)
│           └── left_foot (10)
├── right_hip (2)
│   └── right_knee (5)
│       └── right_ankle (8)
│           └── right_foot (11)
└── spine1 (3)
    └── spine2 (6)
        └── spine3 (9)
            ├── neck (12)
            │   └── head (15)
            ├── left_collar (13)
            │   └── left_shoulder (16)
            │       └── left_elbow (18)
            │           └── left_wrist (20)
            └── right_collar (14)
                └── right_shoulder (17)
                    └── right_elbow (19)
                        └── right_wrist (21)
```

## Appendix B: Joint Group to Dimension Mapping

```
Group       | Joints                                 | Dim Range
root        | transl(3) + pelvis(6)                  | [0, 9)
torso       | spine1(6) + spine2(6) + spine3(6)      | [21,27), [39,45), [57,63)
head        | neck(6) + head(6)                      | [75,81), [93,99)
left_arm    | l_collar(6)+l_shoulder(6)+l_elbow(6)+l_wrist(6) | [81,87),[99,105),[111,117),[123,129)
right_arm   | r_collar(6)+r_shoulder(6)+r_elbow(6)+r_wrist(6) | [87,93),[105,111),[117,123),[129,135)
left_leg    | l_hip(6)+l_knee(6)+l_ankle(6)+l_foot(6)         | [9,15),[27,33),[45,51),[63,69)
right_leg   | r_hip(6)+r_knee(6)+r_ankle(6)+r_foot(6)         | [15,21),[33,39),[51,57),[69,75)
```

## Appendix C: Why Editor-First Design Matters

| Scenario | AI-First Tool | Editor-First Tool |
|----------|--------------|-------------------|
| AI server down | Tool is broken | Tool works 100%, AI features grayed out |
| AI generates bad result | User is stuck, must re-generate | User edits the result manually |
| Specific pose needed | Must prompt-engineer until AI gets it | User directly poses it |
| Subtle timing fix | AI doesn't understand "slightly slower" | User drags keyframe 2 frames right |
| Loop needs exact 60 frames | AI picks its own length | User sets duration to 60, AI fills |
| Left hand 3cm too high | AI regeneration changes everything | User rotates left_elbow by 2 degrees |

The pattern: **AI generates approximate; human refines precise.** The editor must support
the "refine precise" part completely, because that's where the animator spends most time.
