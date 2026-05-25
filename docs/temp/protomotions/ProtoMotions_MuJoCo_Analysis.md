# ProtoMotions MuJoCo Inference Analysis

## Executive Summary

**Yes, ProtoMotions has a fully working MuJoCo inference path** with:
- Complete inference script (`inference_agent.py`)
- Full MuJoCo backend simulator (`protomotions/simulator/mujoco/simulator.py`)
- Two PD control modes (implicit and explicit)
- ONNX policy loading and inference via Lightning Fabric agents
- Comprehensive configuration and initialization system

The MuJoCo backend differs significantly from your `run_smpl_rl_tracker.py` in architecture, PD handling, control flow, and initialization.

---

## Key Files

| File | Purpose | Lines |
|------|---------|-------|
| `/protomotions/inference_agent.py` | Main inference entry point | 391 |
| `/protomotions/simulator/mujoco/simulator.py` | MuJoCo physics backend | 1298 |
| `/protomotions/simulator/mujoco/config.py` | MuJoCo configuration | 54 |
| `/protomotions/simulator/base_simulator/simulator.py` | Base simulator interface | ~1500 |
| `/protomotions/agents/base_agent/agent.py` | Policy inference wrapper | ~1000+ |

---

## 1. INITIALIZATION PATH

### 1.1 Entry Point: `inference_agent.py` (lines 183-386)

```python
def main():
    # Load frozen configs from checkpoint
    resolved_configs_path = checkpoint.parent / "resolved_configs_inference.pt"
    resolved_configs = torch.load(resolved_configs_path, map_location="cpu")
    
    robot_config = resolved_configs["robot"]
    simulator_config = resolved_configs["simulator"]
    env_config = resolved_configs["env"]
    agent_config = resolved_configs["agent"]
    
    # MuJoCo-specific: force CPU
    accelerator = "cpu" if args.simulator == "mujoco" else "gpu"
    fabric_config = FabricConfig(accelerator=accelerator, devices=1, ...)
    fabric: Fabric = Fabric(**asdict(fabric_config))
    fabric.launch()
    
    # Build components
    components = build_all_components(
        terrain_config=terrain_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        ...
    )
    
    # Create env (auto-initializes simulator)
    env: BaseEnv = EnvClass(config=env_config, ...)
    
    # Load agent with checkpoint
    agent: BaseAgent = AgentClass(config=agent_config, env=env, fabric=fabric)
    agent.setup()
    agent.load(args.checkpoint, load_env=False)
```

**Key differences from `run_smpl_rl_tracker.py`:**
- Uses `torch.load()` to restore frozen configs from `resolved_configs_inference.pt`
- Employs `Lightning Fabric` for unified device/distributed handling
- Forces CPU for MuJoCo (`accelerator = "cpu"`)
- Creates full `BaseEnv` wrapper (not direct simulator access)
- Loads ONNX model via agent, not manual ONNX loading

---

### 1.2 MuJoCo Simulator Creation: `simulator.py._create_simulation()` (lines 294-383)

```python
def _create_simulation(self) -> None:
    asset_root = self.robot_config.asset.asset_root
    asset_file = self.robot_config.asset.asset_file_name
    asset_path = os.path.join(asset_root, asset_file)
    
    # Resolve projectile config BEFORE loading MJCF
    self._resolve_proj_config()
    
    # STRIP MJCF: remove sensors, add visual settings & ground
    self.model = self._load_mjcf_stripped(asset_path, self._proj_config)
    self.data = mujoco.MjData(self.model)
    
    # Set physics timestep
    self.model.opt.timestep = 1.0 / self.config.sim.fps  # Default: 1000 Hz
    
    # Zero passive forces from MJCF (we handle PD control via actuators)
    self._zero_passive_forces()
    
    # Override armature and frictionloss from robot config
    self._override_joint_properties()
    
    # Disable robot self-collisions if configured
    if not self.robot_config.asset.self_collisions:
        self._disable_self_collisions()
    
    # Build actuator-to-DOF mapping
    self._build_actuator_mapping()
    
    # Cache PD control parameters
    self._setup_control_parameters()
    
    # Configure actuators based on PD mode
    use_implicit_pd = getattr(self.config, "use_implicit_pd", True)
    if use_implicit_pd:
        self._configure_actuators_for_pd()  # Position actuators
    else:
        self._configure_explicit_pd()  # Motor actuators + manual PD
    
    # Extract COM offsets for velocity semantics correction
    self._extract_body_com_offsets()
    
    # Initialize viewer if not headless
    if not self.headless:
        self._init_viewer()
```

---

## 2. ACTUATOR_GEAR HANDLING

### ProtoMotions Approach: **LEAVES XML DEFAULT**

ProtoMotions does **NOT** modify `<actuator>` `gear` attributes. Instead:

1. **MJCF Stripping** (lines 124-172):
   - Removes `<sensor>` elements that reference missing sites
   - Adds visual settings, ground plane, lighting
   - **Does NOT modify `<actuator>` sections**

2. **Joint Property Overrides** (lines 398-435):
   - Overrides ONLY `dof_armature` (inertia) via robot_config
   - Overrides ONLY `dof_frictionloss` (zeroes it)
   - **Does NOT touch `actuator_gear`**

```python
def _override_joint_properties(self) -> None:
    """Override armature and frictionloss from robot config.
    
    The MJCF has default values (e.g. armature=0.03 for all joints) that
    may differ from the robot config's per-joint values. Newton and IsaacGym
    override these; we must do the same.
    """
    control_info = self.robot_config.control.control_info
    dof_start = 6 if self._has_free_joint else 0
    
    for i in range(self.model.njnt):
        jnt_type = self.model.jnt_type[i]
        if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
            continue
        
        jnt_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
        dof_addr = self.model.jnt_dofadr[i]
        dof_idx = dof_addr - dof_start
        
        if jnt_name in control_info:
            info = control_info[jnt_name]
            # Override armature only
            if info.armature is not None:
                self.model.dof_armature[dof_addr] = info.armature
            # Frame comment mentions frictionloss zeroed
            # self.model.dof_frictionloss[dof_addr] = 0.0
```

**Comparison to `run_smpl_rl_tracker.py`:**
- If your script manually sets `gear`, ProtoMotions does not
- ProtoMotions relies on MJCF defaults for `gear`
- If you need non-default `gear`, add it to MJCF or modify `_override_joint_properties()`

---

## 3. PD GAINS HANDLING

### Two Modes: Implicit vs. Explicit

#### Mode 1: **Implicit PD** (Default, `use_implicit_pd=True`)

**Configuration** (lines 494-566):
```python
def _configure_actuators_for_pd(self) -> None:
    """Convert motor actuators to position (PD) actuators.
    
    MuJoCo position actuators compute PD torques implicitly at every substep:
        force = kp * (ctrl - q) - kd * qd
    
    This is achieved by setting:
        gainprm[0] = kp
        biasprm = [0, -kp, -kd]
        biastype = mjBIAS_AFFINE (1)
    """
    for act_idx in range(self.model.nu):
        jnt_id = self.model.actuator_trnid[act_idx, 0]
        dof_addr = self.model.jnt_dofadr[jnt_id]
        dof_idx = dof_addr - dof_start
        
        # Get kp, kd from robot_config.control.control_info
        jnt_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
        if jnt_name in self.robot_config.control.control_info:
            info = self.robot_config.control.control_info[jnt_name]
            kp = info.stiffness
            kd = info.damping
            effort = info.effort_limit if info.effort_limit else 1000.0
        
        # Configure as position actuator:
        # force = gainprm[0] * ctrl + biasprm[0] + biasprm[1] * q + biasprm[2] * qd
        #       = kp * ctrl + 0 + (-kp) * q + (-kd) * qd
        #       = kp * (ctrl - q) - kd * qd
        self.model.actuator_gainprm[act_idx, 0] = kp
        self.model.actuator_biastype[act_idx] = 1  # mjBIAS_AFFINE
        self.model.actuator_biasprm[act_idx, 0] = 0.0
        self.model.actuator_biasprm[act_idx, 1] = -kp
        self.model.actuator_biasprm[act_idx, 2] = -kd
        
        # Set force limits on actuator
        self.model.actuator_forcerange[act_idx, 0] = -effort
        self.model.actuator_forcerange[act_idx, 1] = effort
        self.model.actuator_forcelimited[act_idx] = 1
```

**Application** (lines 1102-1132):
```python
def _apply_simulator_pd_targets(self, pd_targets: torch.Tensor) -> None:
    """Apply PD position targets.
    
    In implicit mode, write position targets to data.ctrl.
    MuJoCo position actuators compute PD internally each substep.
    """
    targets = pd_targets[0].detach().cpu().numpy()
    
    # Apply EMA filter (optional)
    alpha = self._action_filter_alpha  # Default: 1.0 (no filtering)
    if alpha < 1.0:
        if self._prev_pd_targets is None:
            self._prev_pd_targets = targets.copy()
        targets = alpha * targets + (1.0 - alpha) * self._prev_pd_targets
        self._prev_pd_targets = targets.copy()
    
    # Implicit: write position targets to ctrl (MuJoCo handles PD)
    self.data.ctrl[self._dof_to_actuator] = targets
```

**How it works:**
- `data.ctrl[i]` = target position for actuator i
- MuJoCo computes: `force = kp * (ctrl - q) - kd * qd` at each physics substep
- **No ONNX output reinterpretation needed** — policy outputs positions directly

---

#### Mode 2: **Explicit PD** (Optional, `use_implicit_pd=False`)

**Setup** (lines 568-591):
```python
def _configure_explicit_pd(self) -> None:
    """Set up explicit PD mode: keep motor actuators, cache sim-order gains."""
    control_info = self.robot_config.control.control_info
    
    for act_idx in range(self.model.nu):
        jnt_id = self.model.actuator_trnid[act_idx, 0]
        jnt_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
        if jnt_name in control_info:
            effort = control_info[jnt_name].effort_limit or 1000.0
            self.model.actuator_forcerange[act_idx, 0] = -effort
            self.model.actuator_forcerange[act_idx, 1] = effort
            self.model.actuator_forcelimited[act_idx] = 1

def _cache_sim_order_pd_gains(self) -> None:
    """Cache PD gains reordered to sim DOF order for explicit PD mode."""
    dof_to_sim = self.data_conversion.dof_convert_to_sim.numpy()
    self._kp_sim = self._kp[dof_to_sim]
    self._kd_sim = self._kd[dof_to_sim]
    self._effort_limits_sim = self._effort_limits[dof_to_sim]
```

**Per-substep recomputation** (lines 593-610):
```python
def _recompute_explicit_pd(self) -> None:
    """Compute PD torques from current state and write to data.ctrl.
    
    Called at each physics substep in explicit PD mode.
    """
    if self._pd_targets_sim is None:
        return
    
    if self._has_free_joint:
        q = self.data.qpos[7:]
        qd = self.data.qvel[6:]
    else:
        q = self.data.qpos[:]
        qd = self.data.qvel[:]
    
    torques = self._kp_sim * (self._pd_targets_sim - q) - self._kd_sim * qd
    torques = np.clip(torques, -self._effort_limits_sim, self._effort_limits_sim)
    self._apply_torques_to_ctrl(torques.astype(np.float32))
```

**How it works:**
- Policy outputs positions: `target_q[i]`
- At each 1kHz physics substep, compute: `torque[i] = kp * (target_q[i] - q[i]) - kd * qd[i]`
- Matches RoboJuDo and real hardware PD loops (not per-decimation, but per-physics-step)

---

## 4. CONTROL APPLICATION FLOW

### Physics Stepping with Decimation (lines 748-790)

```python
def _physics_step(self) -> None:
    """Execute physics step with decimation.
    
    Decimation = 20 (default): 50 Hz control loop
    Physics = 1000 Hz, so 20 steps per control call
    """
    from protomotions.robot_configs.base import ControlType
    
    # Apply control (calls _apply_simulator_pd_targets or _apply_simulator_torques)
    self._apply_control()  # <-- Happens ONCE per 20ms step
    
    use_implicit_pd = getattr(self.config, "use_implicit_pd", True)
    use_explicit_substep_pd = (
        not use_implicit_pd
        and self.control_type == ControlType.BUILT_IN_PD
    )
    
    if use_explicit_substep_pd:
        # Explicit PD: recompute torques from current state at EACH substep
        for _ in range(self.decimation):  # 20 iterations of 1ms each
            self._recompute_explicit_pd()  # <-- Recompute PD torques
            mujoco.mj_step(self.model, self.data)
    else:
        # Implicit PD (position actuators) or TORQUE/PROPORTIONAL mode
        # Control held constant over all 20 physics steps
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)
    
    self._step_count += 1
    
    # Sync viewer if active
    if self.viewer is not None and self._viewer_initialized:
        self.viewer.sync()
```

### Base Simulator Control Pipeline (lines 1158-1226)

```python
def _apply_control(self) -> None:
    """Apply control based on control type.
    
    Actions are expected to be pre-processed by ActionProcessor in the network:
    - For BUILT_IN_PD/PROPORTIONAL: actions are PD targets
    - For TORQUE: actions are torques
    """
    if self.control_type == ControlType.BUILT_IN_PD:
        targets = self._common_actions  # [num_envs, num_dofs]
        
        # Apply action noise if configured
        if "action_noise" in self._domain_randomization:
            targets = targets.clone()
            targets[..., dof_indices] += action_noise
        
        # Convert from common DOF order to simulator DOF order
        sim_targets = targets[:, self.data_conversion.dof_convert_to_sim]
        
        # Call MuJoCo-specific implementation
        self._apply_simulator_pd_targets(sim_targets)
    
    elif self.control_type == ControlType.PROPORTIONAL:
        # Manual PD at control loop rate (not per-physics-step)
        targets = self._common_actions
        common_dof_state = self._get_simulator_dof_state().convert_to_common(...)
        
        # Compute torques: τ = kp * (target - q) - kd * qd
        torques = (
            self._common_p_gains * (targets - common_dof_state.dof_pos)
            - self._common_d_gains * common_dof_state.dof_vel
        )
        torques = torch.clip(torques, -self._torque_limits_common, ...)
        sim_torques = torques[:, self.data_conversion.dof_convert_to_sim]
        self._apply_simulator_torques(sim_torques)
    
    elif self.control_type == ControlType.TORQUE:
        # Direct torque control
        torques = self._common_actions
        torques = torch.clip(torques, -self._torque_limits_common, ...)
        sim_torques = torques[:, self.data_conversion.dof_convert_to_sim]
        self._apply_simulator_torques(sim_torques)
```

**Key timing:**
- Policy inference: once per 50ms (control loop)
- Action application: once per 50ms via `_apply_control()`
- Physics stepping: 20 × 1ms per 50ms control loop
- For implicit PD: targets held constant over all 20 physics steps
- For explicit PD: torques recomputed at 1kHz (per-physics-step)

---

## 5. ONNX POLICY LOADING & INFERENCE

ProtoMotions doesn't directly expose ONNX loading in `inference_agent.py`. Instead:

### Via Agent System (lines 363-364)

```python
agent: BaseAgent = AgentClass(config=agent_config, env=env, fabric=fabric)
agent.setup()
agent.load(args.checkpoint, load_env=False)
```

The `agent.load()` method (in `base_agent/agent.py`) handles:
- Loading PyTorch checkpoint (containing ONNX or native model)
- Restoring model weights
- Setting up evaluator

### Evaluation Loop (lines 367-382)

```python
if args.full_eval:
    agent.evaluator.eval_count = 0
    evaluation_log, evaluated_score = agent.evaluator.evaluate()
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for key, value in sorted(evaluation_log.items()):
        print(f"  {key}: {value:.6f}")
    print("=" * 60)
else:
    agent.evaluator.simple_test_policy(collect_metrics=True)
```

The evaluator calls the agent's policy in a loop, which internally uses the loaded model to compute actions.

---

## 6. INITIALIZATION SPECIAL HANDLING

### 6.1 Free Joint Handling (lines 712-742)

```python
def _set_simulator_env_state(
    self,
    new_states: ResetState,
    new_object_states: Optional[ObjectState] = None,
    env_ids: Optional[torch.Tensor] = None,
) -> None:
    """Set simulator state (qpos/qvel) and recompute FK."""
    root_pos = new_states.root_pos[0].cpu().numpy().copy()
    root_rot = new_states.root_rot[0].cpu().numpy()  # wxyz
    root_vel = new_states.root_vel[0].cpu().numpy()
    root_ang_vel = new_states.root_ang_vel[0].cpu().numpy()
    dof_pos = new_states.dof_pos[0].cpu().numpy()
    dof_vel = new_states.dof_vel[0].cpu().numpy()
    
    if self._has_free_joint:
        self.data.qpos[0:3] = root_pos
        self.data.qpos[3:7] = root_rot  # wxyz
        self.data.qpos[7 : 7 + self._num_actuated_dofs] = dof_pos
        
        self.data.qvel[0:3] = root_vel
        # Convert root angular velocity from WORLD frame to LOCAL frame
        # MuJoCo qvel[3:6] for a free joint expects local-frame angular velocity
        root_ang_vel_local = self._quat_rotate_inverse_np(root_rot, root_ang_vel)
        self.data.qvel[3:6] = root_ang_vel_local
        self.data.qvel[6 : 6 + self._num_actuated_dofs] = dof_vel
    else:
        self.data.qpos[: self._num_actuated_dofs] = dof_pos
        self.data.qvel[: self._num_actuated_dofs] = dof_vel
    
    # Recompute forward kinematics
    mujoco.mj_forward(self.model, self.data)
```

**Key:** Angular velocity converted from world to local frame (MuJoCo convention).

### 6.2 COM Velocity Correction (lines 860-979)

MuJoCo `data.cvel` returns velocity at body frame origin, but IsaacGym returns velocity at COM.

```python
def _extract_body_com_offsets(self) -> None:
    """Extract COM offsets for each body from MuJoCo model.
    
    **Velocity Semantics Note:**
    - MuJoCo data.cvel returns velocity at the body frame origin
    - IsaacGym returns velocity at the body center-of-mass (COM)
    - To match IsaacGym semantics: v_COM = v_frame_origin + ω × r_offset
    """
    nb = self._num_robot_bodies
    body_com_offsets = np.zeros((nb, 3), dtype=np.float32)
    
    for body_idx in range(1, 1 + nb):
        geom_start = self.model.body_geomadr[body_idx]
        geom_count = self.model.body_geomnum[body_idx]
        
        if geom_count == 0:
            continue
        
        # Average geom positions to estimate COM offset
        com_pos = np.zeros(3, dtype=np.float32)
        for geom_offset in range(geom_count):
            geom_idx = geom_start + geom_offset
            com_pos += self.model.geom_pos[geom_idx]
        
        body_com_offsets[body_idx - 1] = com_pos / geom_count
    
    self._body_com_offsets = body_com_offsets
```

Applied in `_get_simulator_bodies_state()`:
```python
def _apply_com_velocity_correction(...) -> np.ndarray:
    """Apply COM offset correction to body velocities."""
    nb = body_vel_frame.shape[0]
    body_vel_com = body_vel_frame.copy()
    
    for i in range(nb):
        com_offset_local = self._body_com_offsets[i]
        if np.linalg.norm(com_offset_local) < 1e-8:
            continue
        
        # Rotate COM offset to world frame using body quaternion
        quat_wxyz = body_rot[i]
        # ... quaternion rotation math ...
        com_offset_world = ...
        
        # Apply correction: v_COM = v_frame + ω × r_offset
        cross_product = np.cross(body_ang_vel[i], com_offset_world)
        body_vel_com[i] += cross_product
    
    return body_vel_com
```

---

## 7. KEY DIFFERENCES FROM `run_smpl_rl_tracker.py`

| Aspect | ProtoMotions | Typical `run_smpl_rl_tracker.py` |
|--------|--------------|----------------------------------|
| **Entry point** | `inference_agent.py` with argparse | Direct script (often hard-coded) |
| **Config loading** | Frozen `resolved_configs_inference.pt` | Manual YAML parsing |
| **Device handling** | Lightning Fabric (unified) | Manual torch.device |
| **Simulator** | Abstract `Simulator` base class | Direct MuJoCo calls |
| **Environment** | `BaseEnv` wrapper | Direct simulator interaction |
| **Policy loading** | Via `BaseAgent.load()` | Manual ONNX session/torch loading |
| **Control type** | Abstract (BUILT_IN_PD, PROPORTIONAL, TORQUE) | Often hard-coded |
| **PD application** | Two modes (implicit/explicit) | Usually single mode |
| **Decimation** | Explicit config (`sim.decimation = 20`) | Often magic number |
| **Viewer** | MuJoCo passive viewer + callbacks | Often custom or omitted |
| **Action noise** | Configurable domain randomization | Often disabled in inference |

---

## 8. SUMMARY TABLE: PD HANDLING

| Parameter | Method | Where Set | When Applied |
|-----------|--------|-----------|--------------|
| **actuator_gear** | Not modified | MJCF default | At model load |
| **dof_armature** | Override from robot_config | `_override_joint_properties()` | Before simulation starts |
| **kp, kd** | From robot_config.control.control_info | `_setup_control_parameters()` | Before simulation starts |
| **PD targets/torques** | From policy output | `_apply_simulator_pd_targets()` / `_apply_simulator_torques()` | Once per 50ms (control loop) |
| **Implicit PD computation** | MuJoCo position actuators | Physics loop (1kHz) | 20× per control loop |
| **Explicit PD computation** | `_recompute_explicit_pd()` | Physics loop (1kHz) | 20× per control loop (if enabled) |

---

## 9. REPRODUCTION: How to Run Inference

```bash
cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/ProtoMotions

# Simple inference (policy testing)
python protomotions/inference_agent.py \
    --checkpoint results/model/last.ckpt \
    --simulator mujoco \
    --num-envs 1

# Full evaluation with metrics
python protomotions/inference_agent.py \
    --checkpoint results/model/last.ckpt \
    --simulator mujoco \
    --num-envs 1 \
    --full-eval

# Headless (no viewer)
python protomotions/inference_agent.py \
    --checkpoint results/model/last.ckpt \
    --simulator mujoco \
    --headless
```

---

## 10. RECOMMENDATIONS FOR YOUR USE CASE

If you want to match ProtoMotions' approach:

1. **Use Implicit PD** (default):
   - Set `use_implicit_pd=True` in MuJoCo config
   - Let MuJoCo handle PD internally at 1kHz
   - Policy outputs position targets directly

2. **Leave actuator_gear at XML default** unless you have specific requirements

3. **Override dof_armature** only if different from MJCF:
   - Check robot_config.control.control_info for per-joint values
   - ProtoMotions prints these during initialization

4. **Use action clipping + decimation**:
   - Policy output clipping + PD target acceleration clamping
   - 20 physics steps per control step (1kHz physics, 50Hz control)

5. **Handle COM velocity correction** if matching IsaacGym semantics

