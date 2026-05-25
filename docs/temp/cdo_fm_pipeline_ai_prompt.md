# AI Image Generation Prompt: CDO-FM Method Pipeline Overview

> Use this prompt with a text-to-image or diagram-generation AI (e.g., GPT-4o image generation, Midjourney, DALL·E 3, or a diagramming LLM).
> The Python draft at `cdo_fm_pipeline_figure.py` and its rendered PNG (`cdo_fm_pipeline_overview.png`) serve as the structural reference.

---

## Prompt (English — primary)

```
Create a publication-quality academic method pipeline diagram for "CDO-FM: Condition-Decoupled Orchestration Flow Matching". NO title in the figure (the title goes in the paper caption). The figure should be clean, vector-style, horizontal layout (~17:10.5 aspect ratio), white background, with rounded-rectangle boxes connected by directed arrows. Use schematic illustrations (stick figures, speech bubbles, motion sequences) wherever possible instead of text-only boxes. Use an academic pastel color palette throughout.

=== LAYOUT ===

Three-column arrangement:

LEFT COLUMN — Input encoders (top to bottom):
1. "Text Input" box (pale blue) with a SPEECH BUBBLE ICON on the left side containing 3 short horizontal lines (representing text). To the right of the icon: label "Text Input" and italic sub-text "a person walks..."
   ↓ orange arrow
2. "Text Encoder" box (light orange) with sub-text "Frozen Qwen3 + CLIP-L"
   Below it: italic dim annotation "c_txt (B,S,4096) + v_txt (B,1,768)"
   → curved arrow going RIGHT to the backbone's Input Projection

   [Gap]

3. "Motion Condition" box (pale blue) with STICK FIGURE KEYFRAMES illustration — 3 stick figures in a row showing different poses (arms/legs at different angles), connected by dotted lines between them. Below the figures: label "Motion Condition (keyframes / trajectory / dense)"
   ↓ blue arrow
4. "Structured Condition Sampler" box (light purple) with sub-text "Rank-K Boolean Tensor Prior"
   Below it: small math formula "M = ⊕(t_k ⊗ d_k)"
   ↓ blue arrow
5. "VACE Cond. Encoder" box (medium orange, taller than other boxes) containing:
   - Title: "VACE Cond. Encoder"
   - Subtitle: "Mask-Aware Noise (MAN)"
   - A 3-CHANNEL VISUAL DIAGRAM showing:
     Row 1 "x_t(MAN)": light blue strip — some cells filled solid (known=clean x₁), other cells show noise dots (unknown=noisy)
     Row 2 "reactive": light green strip — same known cells filled solid, rest empty
     Row 3 "mask": light yellow strip — cells contain binary "1" (to-generate) or "0" (known condition)
   - Below: "concat → (B, L, 594)"
   → long curved arrow going RIGHT up to the backbone's Input Projection

   [Bottom of left column]
6. "Motion Representation" box (light grey, thin grey border, understated)
   Sub-text: "198-dim: trans(3)+rot6d(132)+pos(63)"
   Dashed upward arrow connecting to the VACE encoder above it.
   This is NOT a core innovation — just an engineering detail about the data format.

CENTER COLUMN — MMDiT Backbone (enclosed in a large rounded green container):
Title at top: "MMDiT Backbone (reused, no new params)" in dark green bold.
Inside the container, stacked top to bottom:
  a. "Input Projection" (soft green) — "text_emb + motion_emb → hidden"
  b. "Dual-Stream Blocks (×N_double)" (medium green) — shows "Motion Stream ↔ Text Stream" with a bidirectional arrow and "Joint Attention" label below the bidirectional arrow
  c. "Single-Stream Blocks (×N_single)" (darker green) — "[motion; text] → Self-Attn → FFN"
  d. "Flow Velocity Prediction" (peach) — "v_θ (B, L, 198)"
  e. "Timestep t + AdaLN" (light grey, smaller, side element) — "t → sinusoidal → adaptive LN", with a dashed arrow up to the flow velocity block
  f. "Output Motion" box (pale yellow, gold border) with a MOTION SEQUENCE ILLUSTRATION — 5 small stick figures in a row with increasing opacity (ghosting/afterimage effect), a motion trail line below them, and a small arrow at the end indicating continuation. Label: "Output Motion (B, L, 198)"

All backbone elements connected by thick downward arrows.

RIGHT COLUMN — Two core innovation panels, each taking half the column height, enclosed in a dashed-border rounded rectangle:

Panel 1 (top half, light purple background, purple dashed border):
  Title: "[Training] Progressive Density Curriculum Training (PDCT)"
  Sub-header: "Zero extra parameters — schedule modification only"
  Bullet text:
    • Phase A: Low density (E[ρ]≈0.15) — text pathway established first
    • Phase B: Linear density ramp — model learns to fuse text + motion conditions
    • Phase C: Full distribution (E[ρ]≈0.55) — full conditional generation capability
  INSET PLOT (large, ~40% panel width): A line chart showing E[ρ] vs Training Step. The curve is flat at 0.15 (Phase A), ramps linearly (Phase B), then flat at 0.55 (Phase C). Vertical dashed lines separate the three phases. Labels "A", "B", "C" above each region. Purple fill under the curve.
  Dashed purple arrow going LEFT from this panel toward the Structured Condition Sampler, labeled "controls ρ distribution"

Panel 2 (bottom half, very light coral background, red dashed border):
  Title: "[Inference] Condition-Progressive ODE Sampling (CPOS)"
  Sub-header: "Zero extra parameters — inference schedule only"
  Bullet text:
    • w_text(t): Time-varying text CFG weight (bell-shaped, peak at t≈0.3)
    • α(t): Condition replacement schedule (sigmoid, onset at t≈0.4) — NOT CFG, pure post-processing
    • Only 2 forward passes (same cost as standard CFG)
    • ODE early: text CFG dominant → global semantics; ODE late: condition replacement → local precision
  INSET PLOT (large, ~40% panel width): Two curves over "ODE time t" (0 to 1). A red bell-shaped curve labeled "w_text(t) (CFG)" peaking near t≈0.3. A blue sigmoid curve labeled "α(t) (replace)" rising around t≈0.4. Left annotation "text CFG dominant" in red, right annotation "condition replacement" in blue.
  Dashed red arrow going LEFT toward the Flow Velocity Prediction box, labeled "schedule w_text(t), α(t)"

=== BOTTOM STRIP ===
A horizontal color legend with 6 small rounded squares: Input (pale blue), Encoder (light orange), Backbone (light green), Training/PDCT (light purple, dashed), Inference/CPOS (light coral, dashed), Output (pale yellow).

=== STYLE ===
- Clean vector illustration, no 3D effects, no gradients on boxes (flat fills only)
- Academic paper figure style (think NeurIPS / CVPR method overview)
- Rounded rectangle boxes with thin dark borders (1-2pt)
- Arrows: solid for data flow, dashed for innovation linkages
- Color palette: pastel academic tones — blues, greens, oranges, purples, corals. No neon or saturated colors.
- All text in a clean sans-serif font (like Helvetica, Arial, or Source Sans)
- White background
- High resolution, suitable for printing at A4 / letter size
- USE SCHEMATIC ILLUSTRATIONS: speech bubbles for text input, stick-figure keyframes for motion condition, motion sequence with ghosting for output, 3-channel strip diagram for MAN. These make the figure more visually informative than text-only boxes.
- The two right panels should be the visual highlight — they represent the two core contributions (PDCT for training, CPOS for inference)
- The overall feeling should be: organized, readable, and informative — a figure you'd see in a top-tier ML conference paper
- NO TITLE in the figure itself. The figure caption is provided separately in the paper.
```

---

## Prompt (中文 — 备选)

```
创建一张学术出版品质的方法流水线总览图（CDO-FM: Condition-Decoupled Orchestration Flow Matching）。图中不要标题（标题由论文caption提供）。图片风格为干净的矢量插图风格，横向布局（约17:10.5宽高比），白色背景，使用圆角矩形框和有向箭头连接。尽量使用示意图（火柴人、对话气泡、运动序列等）代替纯文字框。全图采用学术风格的柔和配色方案。

=== 布局 ===

三列排布：

左列 — 输入编码器（从上到下）：
1. "Text Input"框（浅蓝色），左侧有对话气泡图标（内含3条短横线代表文字），右侧文字标签"Text Input" + 斜体"a person walks..."
   ↓ 橙色箭头
2. "Text Encoder"框（浅橙色），子文字"Frozen Qwen3 + CLIP-L"
   下方灰色标注：c_txt (B,S,4096) + v_txt (B,1,768)
   → 曲线箭头向右连接到骨干网络的Input Projection

3. "Motion Condition"框（浅蓝色），内含火柴人关键帧示意图——3个不同姿态的火柴人排列，中间用虚线连接。下方标签"Motion Condition (keyframes / trajectory / dense)"
   ↓ 蓝色箭头
4. "Structured Condition Sampler"框（浅紫色），子文字"Rank-K Boolean Tensor Prior"
   下方公式：M = ⊕(t_k ⊗ d_k)
   ↓ 蓝色箭头
5. "VACE Cond. Encoder"框（中等橙色，比其他框更高），内含：
   - 标题："VACE Cond. Encoder"
   - 副标题："Mask-Aware Noise (MAN)"
   - MAN三通道可视化图：
     第1行 "x_t(MAN)"：浅蓝色条带——已知位置=实色填充（干净信号），未知位置=噪声点
     第2行 "reactive"：浅绿色条带——已知位置=实色填充，其余为空
     第3行 "mask"：浅黄色条带——格子内标注二进制"1"（待生成）或"0"（已知条件）
   - 底部标注："concat → (B, L, 594)"

6. 底部："Motion Representation"框（浅灰色，低调标注）
   "198-dim: trans(3)+rot6d(132)+pos(63)"

中列 — MMDiT骨干网络（包裹在大型圆角绿色容器中）：
顶部标题："MMDiT Backbone (reused, no new params)"
内部从上到下排列：
  a. "Input Projection" — text_emb + motion_emb → hidden
  b. "Dual-Stream Blocks (×N_double)" — 显示Motion Stream ↔ Text Stream双向箭头 + Joint Attention
  c. "Single-Stream Blocks (×N_single)" — [motion; text] → Self-Attn → FFN
  d. "Flow Velocity Prediction" — v_θ (B, L, 198)
  e. "Timestep t + AdaLN"（灰色小框，虚线箭头连接到flow velocity）
  f. "Output Motion"框（淡黄色，金色边框）内含运动序列示意图——5个由浅到深的火柴人排列（残影效果），下方运动轨迹线，末端有箭头。标签"Output Motion (B, L, 198)"

右列 — 两个核心创新面板，各占右列的一半高度，每个用虚线圆角矩形包围：

面板1（上半部分，浅紫色背景，紫色虚线边框）：
  标题："[Training] Progressive Density Curriculum Training (PDCT)"
  副标题："零额外参数 — 仅训练调度修改"
  要点：Phase A/B/C 的密度调度说明，含各阶段的训练目的
  内嵌图（较大）：E[ρ] vs Training Step曲线（三段式：平坦→线性攀升→平坦）
  虚线紫色箭头指向左列Structured Condition Sampler，标注"controls ρ distribution"

面板2（下半部分，浅珊瑚色背景，红色虚线边框）：
  标题："[Inference] Condition-Progressive ODE Sampling (CPOS)"
  副标题："零额外参数 — 仅推理调度修改"
  要点：w_text(t)为时变文本CFG权重（钟形），α(t)为条件替换调度（sigmoid，非CFG，纯后处理）
  内嵌图（较大）：w_text(t)钟形曲线标注"(CFG)" + α(t) sigmoid曲线标注"(replace)"
  虚线红色箭头指向Flow Velocity Prediction，标注"schedule w_text(t), α(t)"

=== 风格要求 ===
- 干净的矢量插图风格，无3D效果，无渐变填充（纯色平涂）
- 学术论文figure风格（参考NeurIPS/CVPR会议论文的method overview图）
- 柔和学术配色：蓝、绿、橙、紫、珊瑚色调，不要荧光或饱和色
- 所有文字使用干净的无衬线字体
- 白色背景，高分辨率，适合A4打印
- 使用示意图：文本输入用对话气泡、运动条件用火柴人关键帧、输出用残影运动序列、MAN用三通道条带图。比纯文字框更有信息量。
- 图中不要标题（标题在论文中单独提供）
- 右侧两个面板应该是视觉重点 — 代表两个核心贡献（PDCT训练策略 + CPOS推理策略）
```

---

## Changelog

- **v4 (2026-05-14)**: CPOS technical correction:
  1. **α(t) redefined as condition replacement schedule** — NOT a CFG weight. Motion-condition CFG would require a 3rd forward pass (model(x_t, text, null_motion)), which is too expensive. Instead, α(t) is a post-processing replacement/repaint schedule: after each ODE step, blend known-position predictions back to clean condition values at strength α(t). Zero extra compute.
  2. **Updated inset plot legends**: w_text(t) labeled "(CFG)", α(t) labeled "(replace)". Annotations changed from "text heavy / motion heavy" to "text CFG dominant / condition replacement".
- **v3 (2026-05-14)**: Four major updates:
  1. **Removed title** — pipeline figures in papers use captions, not standalone titles
  2. **Added schematic illustrations** — speech bubble for Text Input, stick-figure keyframes for Motion Condition, motion sequence with ghosting for Output Motion
  3. **Renamed "V3 Condition Sampler" → "Structured Condition Sampler"** — V3 was an internal codename, not suitable for publication
  4. **Added MAN 3-channel visual diagram** — x_t(MAN)/reactive/mask shown as colored strips with schematic content inside the VACE Cond. Encoder box
- **v2 (2026-05-13)**: Removed Dual Root panel (engineering setting, not core innovation); demoted to small grey annotation box
- **v1 (2026-05-12)**: Initial version with 3 right panels (PDCT, CPOS, Dual Root)

---

## Usage Notes

1. **Attach the Python-rendered draft** (`cdo_fm_pipeline_overview.png`) as a visual reference when sending the prompt to an AI image model. Most models perform significantly better when given both a text description and a spatial reference image.

2. **For GPT-4o / Claude image generation**: Use the English prompt directly. Attach the PNG. Ask it to "recreate this figure in a cleaner, publication-quality vector style, following the detailed description below."

3. **For Midjourney**: Condense the prompt to Midjourney's style keywords. Append: `--ar 17:10 --style raw --v 7`. Note: Midjourney is not ideal for diagrams with precise text; consider using it only for the overall layout/aesthetic and then overlaying text in Illustrator/Figma.

4. **For diagram-specific tools** (Mermaid, draw.io, Figma AI):
   - The structure can be adapted into Mermaid flowchart syntax, but inset plots and stick-figure illustrations are not supported.
   - For draw.io: use the prompt to guide manual construction; the Python script provides exact coordinates.

5. **Iterative refinement**: After the first AI-generated image, use follow-up prompts like:
   - "Make the two right panels taller and add more spacing between them"
   - "The arrows from left encoders to center backbone should be more visible — use thicker lines"
   - "The inset plots are too small — enlarge them by 30%"
   - "The MAN 3-channel diagram needs clearer distinction between known (solid) and unknown (noise) cells"
   - "Make the stick figures in Motion Condition more distinct in their poses"

6. **Font rendering**: If the AI model struggles with mathematical notation (subscripts, Greek letters), generate the figure without math text and add annotations manually in a vector editor afterward.

7. **Illustration tips**: If the AI model renders the stick figures poorly, generate the figure with placeholder circles/rectangles in those positions and draw the stick figures manually in a vector editor.
