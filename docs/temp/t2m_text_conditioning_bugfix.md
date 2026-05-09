# T2M Text Conditioning Bug Fix (2026-04-13)

## 发现

M2M v2 caption 模型 T2M 推理结果与输入文本完全不匹配（sit/walk/run 都生成几乎相同的动作）。

## 根因：3 个互相叠加的 Bug

### Bug 1 (P0): eval 脚本 key name 不匹配 [已修复]

**文件**: `tools/eval_m2m_v2_t2m.py` 第 183-185 行

```python
# ❌ WRONG
batch['text_vec_raw'] = text_out['vtxt_input'].to(device)   # KeyError!
batch['text_ctxt_raw'] = text_out['ctxt_input'].to(device)   # KeyError!
batch['text_ctxt_raw_length'] = text_out['ctxt_length'].to(device)  # KeyError!

# ✅ FIXED
batch['text_vec_raw'] = text_out['text_vec_raw'].to(device)
batch['text_ctxt_raw'] = text_out['text_ctxt_raw'].to(device)
batch['text_ctxt_raw_length'] = text_out['text_ctxt_raw_length'].to(device)
```

**影响**: `encode_text()` 返回 `{text_vec_raw, text_ctxt_raw, text_ctxt_raw_length}`，eval 脚本用了错误的 key → KeyError → `except: pass` 静默吞掉 → 模型收到 null text。

### Bug 2 (P0): CFG pipeline 两个 branch 用了相同 text [已修复]

**文件**: `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py` 第 172-173 行

```python
# ❌ WRONG: 两个 branch 用相同的 text
ctxt_input=ctxt_input.repeat(2, 1, 1),   # 同样的 text 给 cond 和 uncond
vtxt_input=vtxt_input.repeat(2, 1, 1),   # cond - uncond = 0!

# ✅ FIXED: uncond branch 用 null embedding
ctxt_input=torch.cat([null_ctxt, ctxt_input], dim=0),   # [uncond=null, cond=text]
vtxt_input=torch.cat([null_vtxt, vtxt_input], dim=0),   # CFG: uncond + scale*(cond-uncond)
```

**影响**: CFG 公式 `uncond + scale * (cond - uncond)` 中 cond=uncond → guidance signal=0 → text 完全无效。

### Bug 3 (P1): 训练 cond_mask_prob=0.3 偏高 [待调整]

**文件**: `configs/hymotion_m2m/hymotion_m2m_completion_caption_fm_046b.py`

`cond_mask_prob=0.3` 即 30% 的 batch 丢弃文本条件。标准做法是 10%（0.1）。30% 过高会削弱文本信号强度。

## 验证

修复后需重新跑 eval:
```bash
python tools/eval_m2m_v2_t2m.py
```

然后检查 sit/walk/run 的 pelvis_height 是否出现合理差异（sit ~0.5-0.7m，walk ~0.9m）。
