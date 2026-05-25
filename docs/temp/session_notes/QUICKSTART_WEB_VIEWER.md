# 🚀 Quick Start: View Converted Motions in Web Viewer

This guide will get you viewing the converted SMPL mesh animations in your browser in **2 minutes**.

---

## Prerequisites
- Python 3.8+
- Flask (`pip install flask`)
- Working directory: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer`

---

## 🎬 Step 1: Start the Flask Server

```bash
cd motion_annot_web/embodied_viz
python3 app.py --port 8095
```

You should see output like:
```
 * Serving Flask app 'app'
 * Running on http://127.0.0.1:8095
 * Press CTRL+C to quit
```

---

## 🌐 Step 2: Open in Browser

Once the server starts, open your browser to:

```
http://localhost:8095
```

Or if connecting from a remote machine, replace `localhost` with the server IP address:
```
http://<your-server-ip>:8095
```

---

## 📺 Step 3: Browse Motions

The web viewer will automatically discover all 76 converted JSON files and present them in a list:

- **Motion List**: Left sidebar shows all available motions
  - Format: `{variant}_{id}_{description}_{type}.json`
  - Examples:
    - `pretrained_00_a_person_stands_still_raw`
    - `finetuned_05_a_person_walks_forward_slowly_rl`

- **3D Viewer**: Center panel shows the SMPL mesh animation
  - Frame-by-frame playback controls
  - Play/pause button
  - Frame counter

- **Metadata**: Right panel shows motion info
  - Frame count
  - FPS (30)
  - Gender (neutral)
  - File size

---

## 🎮 Viewer Controls

| Control | Action |
|---------|--------|
| **Click Motion Name** | Load and play motion |
| **Play/Pause Button** | Start/stop animation |
| **Frame Slider** | Jump to specific frame |
| **Speed Control** | Adjust playback speed |
| **Mouse Drag** | Rotate 3D view |
| **Mouse Scroll** | Zoom in/out |

---

## 🔍 Viewing Different Variants

### Pretrained vs Fine-tuned
- **Pretrained**: Base model weights (`pretrained_*`)
- **Fine-tuned**: Model after fine-tuning (`finetuned_*`)

### Raw vs RL
- **Raw**: Direct model output, typically longer (~120 frames)
- **RL**: Reinforcement learning variant, typically shorter (~40-60 frames)

### Example Comparisons
```
# Compare pretrained vs finetuned for same motion
1. Load: pretrained_00_a_person_stands_still_raw
2. Play, observe motion quality
3. Load: finetuned_00_a_person_stands_still_raw
4. Compare smoothness, realism
```

---

## 🛠️ Troubleshooting

### Port Already in Use
```bash
# Use a different port
python3 app.py --port 8096
# Then access http://localhost:8096
```

### Can't Connect to Server
1. Ensure Flask server is running (`python3 app.py --port 8095`)
2. Check firewall rules allow port 8095
3. If remote, use server IP: `http://<ip>:8095`

### No Motions Appear
1. Verify symlink exists:
   ```bash
   ls -la motion_annot_web/embodied_viz/data/smpl_mesh
   ```
2. Verify JSON files exist:
   ```bash
   ls output/physflow_v2_compare_iter1000/smpl_mesh/ | wc -l
   # Should show 76
   ```

---

## 📊 Available Dataset

All 76 motions are ready to view:

```
38 Pretrained Model Results:
  ├─ 19 × Raw (longer sequences, ~120 frames each)
  └─ 19 × RL (reinforcement learning, ~40-60 frames each)

38 Fine-tuned Model Results:
  ├─ 19 × Raw (longer sequences, ~120 frames each)
  └─ 19 × RL (reinforcement learning, ~40-60 frames each)
```

### Motion Categories
All 19 base motions, repeated with 2 variants each:
- Standing (still, relaxed pose)
- Locomotion (walk forward, walk in circle, walk slowly, walk with long strides)
- Gestures (wave hand, raise arms, clap hands, stretch arms)
- Actions (walk & stop, walk & turn, jog & walk, kick, squat, jump, jumping jack, high kick)

---

## 💾 Dataset Locations

| Component | Location | Notes |
|-----------|----------|-------|
| **Source NPZ** | `output/physflow_v2_compare_iter1000/npz/` | 76 original files |
| **Converted JSON** | `output/physflow_v2_compare_iter1000/smpl_mesh/` | 13 MB total |
| **Web Symlink** | `motion_annot_web/embodied_viz/data/smpl_mesh` | Points to JSON dir |
| **Flask App** | `motion_annot_web/embodied_viz/app.py` | Web server |

---

## 🔧 Advanced: Export Motion to Video

Once you've identified a motion you want to save, export frames as video (requires ffmpeg):

```bash
# Download each frame's screenshot and combine with ffmpeg
# (Feature available in future viewer version)
```

---

## 📚 Related Documentation

- **Conversion Details**: See `CONVERSION_COMPLETE.md`
- **NPZ Format**: See `NPZ_TO_SMPL_CONVERSION_ANALYSIS.md`
- **Flask App Code**: `motion_annot_web/embodied_viz/app.py`
- **Web Viewer Code**: `motion_annot_web/embodied_viz/templates/`

---

## ✅ What's Next?

After viewing the motions:

1. **Compare Variants**: Load multiple motions to visually compare model outputs
2. **Identify Quality Issues**: Note any artifacts, jittering, or unnatural movements
3. **Use for Evaluation**: Integrate with eval_dashboard for quantitative metrics
4. **Further Processing**: Extract motion features, compute metrics, use for training

---

**Enjoy exploring the converted motions! 🎬**

For questions or issues, refer to the Flask app logs or check the JSON file structure directly.

```bash
# View a raw JSON file
cat output/physflow_v2_compare_iter1000/smpl_mesh/pretrained_00_a_person_stands_still_raw.json | python3 -m json.tool | head -50
```
