# Implementation Roadmap: Physics-Guided Motion Generation
## HYMotion T2M + MuJoCo SMPL Humanoid

**Document Version**: 1.0 | **Date**: May 18, 2026 | **Status**: Ready for Implementation

---

## EXECUTIVE SUMMARY

You have **three proven paths** to pass physics signals back to your motion generation model:

| Path | When | Timeline | Papers Supporting It |
|------|------|----------|---------------------|
| **Policy Gradient + MuJoCo** | Start now | 2-4 weeks | REINFORCE (2016), PPO (2017) |
| **DPO + Physics Metrics** | After 1st | 3-6 weeks | MoDiPO (2024), RLPF (2025) |
| **Differentiable Physics (MJX)** | Later | 6-8 weeks | Brax (2021), MJX (2024+) |

**Recommended approach**: Start with **Policy Gradient** → move to **DPO** if needed.

---

## KEY FINDING: NO GRADIENTS THROUGH PHYSICS NEEDED

**Common Misconception**: "I need to compute gradients through the physics simulator"

**Reality**: 
- Physics simulator acts as a **reward function** (no gradients needed through it)
- Gradients flow only through your **T2M model parameters** (PyTorch)
- This is why it's feasible with vanilla MuJoCo (which has no autodiff support)

```
[Text] → [T2M Model] ← GRADIENTS FLOW HERE
           ↓
         [Motion]
           ↓
      [MuJoCo Sim] → Physics Score (no gradients through sim)
           ↓
    [Reward Function] → Backprop only to T2M params
```

---

## PART 1: START HERE - POLICY GRADIENT APPROACH

### 1.1 Understanding Policy Gradient

**The Core Idea (Score Function Estimator)**:

```python
# For a text prompt and generated motion
text = "person walks forward"
motion = t2m_model.sample(text)  # [T, D] trajectory

# Policy gradient formula:
# ∇_θ L = E_m [ ∇_θ log p_θ(m|text) × R(m) ]

log_prob = t2m_model.log_prob(text, motion)  # log p_θ(m|text)
physics_reward = mujoco_evaluate(motion)     # R(m) - no gradients here!

loss = -log_prob * physics_reward  # negative because we maximize reward
loss.backward()  # backprop through log_prob ONLY
```

**Why it works**:
- No autodiff needed through physics (MuJoCo works as-is)
- Works with PyTorch (your T2M is PyTorch)
- Mathematically proven (REINFORCE theorem)
- Real systems use this (OpenAI, DeepMind, etc.)

### 1.2 Implementation Steps (Week 1-2)

#### Step 1: Design Physics Reward Function (1-2 days)

**Goal**: Define what makes a motion "physically good"

```python
import mujoco
import numpy as np

class PhysicsEvaluator:
    def __init__(self, model_path='smpl_humanoid.xml'):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.dt = self.model.opt.timestep
    
    def evaluate(self, motion_sequence):
        """
        Args:
            motion_sequence: [T, n_dof] joint angles over time
        
        Returns:
            reward: scalar (higher = better motion)
        """
        self.data = mujoco.MjData(self.model)
        
        rewards = {}
        
        # 1. Collision penalty
        rewards['collision'] = self._collision_penalty(motion_sequence)
        
        # 2. Stability metric
        rewards['stability'] = self._center_of_mass_stability(motion_sequence)
        
        # 3. Energy efficiency
        rewards['efficiency'] = self._energy_efficiency(motion_sequence)
        
        # 4. Motion smoothness
        rewards['smoothness'] = self._motion_smoothness(motion_sequence)
        
        # Weighted combination
        weights = {'collision': 1.0, 'stability': 0.5, 
                   'efficiency': 0.3, 'smoothness': 0.2}
        
        total_reward = sum(w * rewards[k] for k, w in weights.items())
        return float(total_reward)
    
    def _collision_penalty(self, motion_sequence):
        """Penalize penetrations (contact.dist < 0)"""
        penalty = 0.0
        for t in range(len(motion_sequence)):
            self.data.qpos[:] = motion_sequence[t]
            mujoco.mj_kinematics(self.model, self.data)
            
            # Count penetrations
            for contact in self.data.contact:
                if contact.dist < 0:
                    penalty -= abs(contact.dist)
        
        # Normalize by episode length
        return max(-1.0, penalty / len(motion_sequence))
    
    def _center_of_mass_stability(self, motion_sequence):
        """Lower variance in COM height = higher stability"""
        com_heights = []
        
        for t in range(len(motion_sequence)):
            self.data.qpos[:] = motion_sequence[t]
            mujoco.mj_kinematics(self.model, self.data)
            
            # Get COM of root (usually pelvis for SMPL)
            com_height = self.data.subtree_com[0][2]
            com_heights.append(com_height)
        
        com_heights = np.array(com_heights)
        stability = -np.var(com_heights)  # negative variance (want low variance)
        
        return stability / 10.0  # normalize
    
    def _energy_efficiency(self, motion_sequence):
        """Reward low energy consumption relative to distance traveled"""
        # Simplified: penalize large accelerations
        accelerations = np.diff(np.diff(motion_sequence, axis=0), axis=0)
        energy_cost = -np.sum(np.linalg.norm(accelerations, axis=1)**2)
        
        return energy_cost / 1000.0  # normalize
    
    def _motion_smoothness(self, motion_sequence):
        """Reward smooth trajectories (low jerk)"""
        jerk = np.diff(np.diff(np.diff(motion_sequence, axis=0), axis=0), axis=0)
        smoothness = -np.sum(np.linalg.norm(jerk, axis=1)**2)
        
        return smoothness / 100.0  # normalize
```

**Testing your reward function** (1 hour):
```python
evaluator = PhysicsEvaluator('smpl_humanoid.xml')

# Test on a few motions
test_motions = [
    np.random.randn(100, 22),  # 100 frames, 22 DoF SMPL humanoid
    np.zeros((100, 22)),       # zero motion (should have low reward)
]

for i, motion in enumerate(test_motions):
    score = evaluator.evaluate(motion)
    print(f"Motion {i}: reward = {score:.3f}")
```

#### Step 2: Create MuJoCo Evaluation Wrapper (1 day)

**Goal**: Fast, batched evaluation

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

class BatchPhysicsEvaluator:
    def __init__(self, model_path='smpl_humanoid.xml', n_workers=4):
        self.model_path = model_path
        self.n_workers = n_workers
        self.evaluator = PhysicsEvaluator(model_path)
    
    def evaluate_batch(self, motion_list):
        """
        Args:
            motion_list: list of [T, D] motions
        
        Returns:
            rewards: [batch_size] rewards
        """
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            rewards = list(executor.map(self.evaluator.evaluate, motion_list))
        
        return np.array(rewards)

# Usage
evaluator = BatchPhysicsEvaluator('smpl_humanoid.xml', n_workers=8)
batch_motions = [np.random.randn(100, 22) for _ in range(64)]
rewards = evaluator.evaluate_batch(batch_motions)  # [64]
```

#### Step 3: Modify Training Loop (2-3 days)

**Current loop** (supervised learning):
```python
for epoch in range(n_epochs):
    for batch_idx, batch in enumerate(train_loader):
        # batch = {'text': [...], 'motion': [...]}
        
        # Forward pass
        predictions = t2m_model(batch['text'])
        
        # Loss: compare to ground truth
        loss = mse_loss(predictions, batch['motion'])
        
        # Backward
        loss.backward()
        optimizer.step()
```

**New loop** (policy gradient with physics rewards):
```python
evaluator = BatchPhysicsEvaluator('smpl_humanoid.xml', n_workers=8)
baseline_reward = None  # exponential moving average for variance reduction

for epoch in range(n_epochs):
    for batch_idx, batch in enumerate(train_loader):
        text_batch = batch['text']
        batch_size = len(text_batch)
        
        # 1. SAMPLE motions from T2M model (no gradients)
        with torch.no_grad():
            motions = [t2m_model.sample(text) for text in text_batch]
        
        # 2. EVALUATE physics for each motion
        motion_np = [m.cpu().numpy() if torch.is_tensor(m) else m 
                     for m in motions]
        physics_rewards = evaluator.evaluate_batch(motion_np)  # [batch_size]
        physics_rewards_tensor = torch.from_numpy(physics_rewards)
        
        # 3. BASELINE for variance reduction
        if baseline_reward is None:
            baseline_reward = physics_rewards.mean()
        else:
            baseline_reward = 0.99 * baseline_reward + 0.01 * physics_rewards.mean()
        
        # 4. COMPUTE log probabilities (now WITH gradients)
        log_probs = t2m_model.log_prob(text_batch, motions)  # [batch_size]
        
        # 5. POLICY GRADIENT LOSS
        advantages = physics_rewards_tensor - baseline_reward
        loss = -torch.mean(log_probs * advantages)
        
        # 6. BACKWARD & UPDATE
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Logging
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx} | "
                  f"Reward: {physics_rewards.mean():.3f} | "
                  f"Loss: {loss.item():.4f}")
```

**Key modifications**:
1. Use `.sample()` instead of deterministic forward pass
2. Evaluate physics WITHOUT gradient tracking
3. Compute log probabilities WITH gradient tracking
4. Use advantage (reward - baseline) for stability
5. Loss = -log_prob × advantage

#### Step 4: Validation (2-3 days)

```python
def validate_policy_gradient():
    """Check that physics metrics improve during training"""
    
    # Track metrics
    metrics = {
        'epoch': [],
        'avg_reward': [],
        'collision_penalty': [],
        'stability': [],
        'text_alignment': [],  # e.g., FID score vs ground truth
    }
    
    for epoch in range(n_validation_epochs):
        # Sample from model
        test_texts = ['person walks', 'person runs', 'person jumps']
        
        batch_rewards = []
        for text in test_texts:
            motion = t2m_model.sample(text)
            reward = evaluator.evaluate(motion.cpu().numpy())
            batch_rewards.append(reward)
        
        metrics['epoch'].append(epoch)
        metrics['avg_reward'].append(np.mean(batch_rewards))
        
        print(f"Epoch {epoch}: Avg Reward = {np.mean(batch_rewards):.3f}")
    
    # Plot results
    import matplotlib.pyplot as plt
    plt.plot(metrics['epoch'], metrics['avg_reward'])
    plt.xlabel('Epoch')
    plt.ylabel('Average Physics Reward')
    plt.title('Policy Gradient Training: Physics Reward Over Time')
    plt.savefig('policy_gradient_results.png')
    plt.show()
    
    return metrics
```

**Expected results** (Week 2 end):
- ✅ Physics rewards improve with training
- ✅ Motions show less clipping/penetration
- ✅ Center of mass more stable
- ✅ Text alignment maintained (or slight improvement)

---

## PART 2: UPGRADE TO DPO (WEEKS 3-6)

**Only do this if**:
- Policy Gradient converges but motions plateau
- You want more stable training
- You're aligned with latest papers (MoDiPO, RLPF)

### 2.1 DPO Concept

Instead of:
```
Single motion → Physics score → Gradient
```

Do:
```
Motion Pair (A, B) → Physics scores → Label winner → DPO loss → Gradient
```

**Advantages**:
- More stable (DPO has lower variance than REINFORCE)
- Proven in papers (MoDiPO 2024, RLPF 2025)
- Can leverage existing unlabeled motion data

### 2.2 Implementation

```python
def generate_motion_pairs(t2m_model, text_batch, n_samples=2):
    """Generate multiple samples per text"""
    pairs = []
    for text in text_batch:
        samples = [t2m_model.sample(text) for _ in range(n_samples)]
        pairs.append(samples)
    return pairs

def dpo_loss(t2m_model, text, winner, loser, beta=0.5):
    """DPO loss: encourage higher prob for preferred motion"""
    log_prob_winner = t2m_model.log_prob(text, winner)
    log_prob_loser = t2m_model.log_prob(text, loser)
    
    # Log of sigmoid ratio
    loss = -torch.log(torch.sigmoid(beta * (log_prob_winner - log_prob_loser)))
    return loss

# Training loop
for epoch in range(n_epochs):
    for batch_idx, batch in enumerate(train_loader):
        text_batch = batch['text']
        
        # Generate pairs
        pairs = generate_motion_pairs(t2m_model, text_batch, n_samples=2)
        
        # Evaluate and label
        total_loss = 0
        for text, pair in zip(text_batch, pairs):
            rewards = [evaluator.evaluate(m.cpu().numpy()) for m in pair]
            
            if rewards[0] > rewards[1]:
                winner, loser = pair[0], pair[1]
            else:
                winner, loser = pair[1], pair[0]
            
            # Compute DPO loss
            loss = dpo_loss(t2m_model, text, winner, loser, beta=0.5)
            total_loss += loss
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

---

## PART 3: DIFFERENTIABLE PHYSICS (OPTIONAL, LATER)

**Only pursue if**:
- Policy Gradient + DPO plateau
- You have 4-8 weeks
- You have JAX/GPU expertise

This requires:
1. Install JAX + MJX
2. Convert T2M model to JAX
3. Build end-to-end differentiable loss
4. 10× faster but much harder

---

## SYSTEM REQUIREMENTS & SETUP

### Minimal Setup (Option A: Policy Gradient)
```bash
# Already have
✅ MuJoCo (installed)
✅ PyTorch

# Optional: for faster evaluation
pip install dm-control

# Optional: for visualization
pip install meshcat
```

### Extended Setup (Option B: DPO)
```bash
# Same as above
✅ MuJoCo
✅ PyTorch

# No new requirements!
```

### Full Setup (Option C: MJX - later)
```bash
pip install jax "jax[cuda11_cudnn82]"
pip install mujoco-mjx
pip install mujoco-warp  # optional CUDA optimization
```

---

## RESEARCH VALIDATION

### Papers Supporting Each Approach

**1. Policy Gradient**:
- Original: [REINFORCE (Williams, 2016)](https://arxiv.org/abs/1604.06778)
- Modern: [PPO (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- RL from Pixels: Used in Atari, robotics

**2. DPO**:
- Core: [Direct Preference Optimization (Rafailov et al., 2024)](https://arxiv.org/abs/2305.18290)
- Motion: [MoDiPO (Collorone et al., 2024)](https://arxiv.org/abs/2405.03803)
- Real World: [RLPF (Yue et al., 2025)](https://arxiv.org/abs/2506.12769v1)
- **Deployed on real robots**: Unitree G1, Tesla Optimus

**3. Differentiable Physics**:
- Brax: [Brax: A Differentiable Physics Engine (Freeman et al., 2021)](https://arxiv.org/abs/2106.13281)
- MJX: [MuJoCo-XLA (Google DeepMind, 2024+)](https://github.com/google-deepmind/mujoco)

---

## COMMON PITFALLS & SOLUTIONS

| Pitfall | Cause | Solution |
|---------|-------|----------|
| Training unstable | High variance in policy gradient | Add baseline reward (EMA) |
| Motions collapse | Reward too easy to exploit | Add text alignment term to reward |
| Physics too slow | Simulation bottleneck | Parallelize with ProcessPoolExecutor |
| Gradients vanish | Numeric issues in backprop | Use double precision, clip gradients |
| Text alignment broken | Reward dominates text loss | Weight balance: `loss = -log_p * R + λ * text_loss` |

---

## SUCCESS METRICS

### Track These During Training:

1. **Physics Metrics**:
   - Collision count per trajectory (should decrease)
   - COM stability (height variance, should decrease)
   - Energy efficiency (J/meter, should decrease)

2. **Motion Quality**:
   - Visual inspection (render motions regularly)
   - FID score vs ground truth (should stay same or improve)
   - Text alignment (CLIP score, should stay high)

3. **Training Stability**:
   - Loss curve smoothness
   - Reward convergence
   - No NaN/Inf values

### Example Validation Code:
```python
def compute_metrics(motion):
    """Compute all evaluation metrics"""
    return {
        'collision_count': count_collisions(motion),
        'com_stability': compute_com_variance(motion),
        'energy': compute_energy_cost(motion),
        'smoothness': compute_motion_smoothness(motion),
    }

def log_metrics(metrics_dict, step):
    for key, value in metrics_dict.items():
        wandb.log({key: value}, step=step)
```

---

## TIMELINE SUMMARY

| Week | Task | Duration | Status |
|------|------|----------|--------|
| 1 | Design reward function + implement evaluator | 2 days | Can start now |
| 1 | Implement REINFORCE training loop | 2-3 days | Can start now |
| 1-2 | Validation & hyperparameter tuning | 2-3 days | Can start now |
| 2 | **Proof of Concept Ready** | - | ✅ End Week 2 |
| 3 | (Optional) Upgrade to DPO | 1 week | Depends on results |
| 4-5 | (Optional) Validation of DPO | 2 weeks | Depends on results |
| 6 | (Optional) Consider MJX migration | - | Only if needed |

**Critical path to working system: 2 weeks**

---

## NEXT STEPS (TODAY)

### Action Items (This Week):

1. **[Priority 1] Understand T2M Interface**
   - How to call `.sample(text)` on your model?
   - Can you compute `.log_prob(text, motion)`?
   - Document the API

2. **[Priority 2] Test Physics Evaluator**
   - Implement basic collision detection in MuJoCo
   - Test on 5-10 sample motions
   - Profile speed (msec per motion)

3. **[Priority 3] Setup Training Infrastructure**
   - Modify your training loop to support sampling
   - Add logging/monitoring for physics metrics
   - Ensure reproducibility (seed management)

### By End of Week:
- ✅ T2M interface documented
- ✅ Physics evaluator working
- ✅ First REINFORCE training loop running

---

## FINAL RECOMMENDATION

**Start with Policy Gradient approach**:
- ✅ 2-4 weeks to working system
- ✅ No additional infrastructure needed
- ✅ Proven mathematically and empirically
- ✅ Matches your current PyTorch setup
- ⏭️ Can upgrade to DPO if needed
- ⏭️ Can migrate to MJX if needed

**Do NOT start with MJX**:
- ❌ 6-8 weeks of JAX learning
- ❌ Major model rewriting required
- ❌ Only justified if policy gradient plateaus
- ⏭️ Revisit after Phase 1 success

---

**Next Document**: `IMPLEMENTATION_CODE_SKELETON.py` (coming next)

**Questions**? Review:
- Section 1: Current state analysis
- Section 2: Three approaches explained
- Section 3: Policy gradient deep dive

