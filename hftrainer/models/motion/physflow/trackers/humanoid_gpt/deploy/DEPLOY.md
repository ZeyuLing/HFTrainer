# Deployment Guide for Unitree G1

This directory contains the deployment pipeline of **Humanoid-GPT** for Unitree G1.
The same tracking inference stack is used in simulation and on hardware.

Main entry point:

```bash
python -m deploy.play_track
```

## Overview

The deployment stack supports:

- **Simulation mode**: walk control, online retargeting, and offline trajectory tracking in MuJoCo.
- **Real-robot mode**: low-level DDS control on Unitree G1 with shared observation/action computation.

Core files:


| File              | Description                                                              |
|-------------------|--------------------------------------------------------------------------|
| `play_track.py`   | Unified runtime entry for simulation and real robot                      |
| `walk_policy.py`  | ONNX walk policy wrapper                                                 |
| `retarget.py`     | Online mocap retarget subprocess (PNLink / Xsens)                        |
| `real_robot.py`   | Low-level robot interface (IMU/joints readout and PD command publishing) |
| `hand_control.py` | Dex3-1 hand controller                                                   |
| `keyboard_cmd.py` | Keyboard UI for mode/velocity control                                    |
| `constants.py`    | Deploy constants (PD gains, motor IDs, DDS topics)                       |


## Installation

All commands below are executed from repository root.

### 1. Base environment for Humanoid-GPT

```bash
conda create -n h-gpt python=3.12 -y
conda activate h-gpt
pip install -e .
```

### 2. Download third-party libraries

```bash
pip install gdown
gdown https://drive.google.com/uc?id=1ArtgwKxVHXTO4KXsKXPLdhy1yAtKKnz9 -O thirdparty.zip
unzip thirdparty.zip
rm thirdparty.zip
```

Alternatively, download `[thirdparty.zip](https://drive.google.com/file/d/1bfgFhrv6tfuDOkt11AOJAO2IHTRXlYey/view?usp=sharing)` manually and extract it to the repository root so that a `thirdparty/` folder appears at the top level.

After extraction, the directory should look like:

```
thirdparty/
├── GMR-galbot/          # Online retargeting (Section 3)
├── noitom/              # PNLink mocap backend (Section 3)
├── cyclonedds/          # DDS middleware for real-robot communication (Section 4)
└── unitree_sdk2_python/ # Unitree G1 SDK Python bindings (Section 4)
```

### 3. Online retargeting dependencies

```bash
pip install -e thirdparty/GMR-galbot
pip install -e thirdparty/noitom
```

`noitom` is required for the default `pnlink` mocap backend.

### 4. Real-robot dependencies

Build CycloneDDS:

```bash
cd thirdparty/cyclonedds
mkdir -p build install
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install
cd ../../..
```

Install Unitree SDK Python:

```bash
export CYCLONEDDS_HOME="$PWD/thirdparty/cyclonedds/install"
pip install -e thirdparty/unitree_sdk2_python
```

### 5. TensorRT acceleration (real mode)

Real mode enforces TensorRT backend (`strict_trt=True`).

```bash
pip uninstall onnxruntime -y
pip install onnxruntime-gpu tensorrt-cu12
```

You may need to add this into bashrc:

```bash
# Expose TensorRT / NVIDIA runtime libs from the h-gpt env to the dynamic linker
for _d in "$HOME/miniconda3/envs/h-gpt/lib"/python*/site-packages/{tensorrt_libs,nvidia/*/lib}; do
  [ -d "$_d" ] && export LD_LIBRARY_PATH="$_d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done
unset _d
```

```bash
python - <<'PY'
import onnxruntime as ort
print(ort.get_available_providers())
PY
```

`TensorrtExecutionProvider` must appear in the provider list.

### 6. Noitom / Xsens online streaming (online retarget mode)

Online retarget mode consumes a live mocap stream produced by the capture suit's
PC software. Run that software on a **separate Windows machine** and stream the
skeleton data over the network to this Linux host; GMR retargets it to the G1 in
real time. Pick the backend with `--mocap_type` (`pnlink` for Noitom,
`xsens` for Xsens).

**Network setup (both backends).** Put the Windows PC and the Linux host on the
same LAN — a direct Ethernet cable works best for latency, and Wi-Fi can be used
in parallel for internet. Give each machine a static IP in the same subnet and
verify reachability:

```bash
# Windows (cmd):  find the Ethernet adapter's IPv4 address
ipconfig
# Linux:          find this host's IP and ping the Windows PC
ip addr
ping <windows_pc_ip>
```

In the Windows software, set the **destination / target address** to this Linux
host's **LAN IP** (not `127.0.0.1`), and make sure the **protocol** and **port**
match the values you pass on the Linux side.

#### Noitom Axis Studio (`--mocap_type pnlink`)

1. Install **Axis Studio** on the Windows PC, connect the Perception Neuron suit,
  and complete calibration.
2. Open `Settings → BVH Broadcasting` and enable broadcasting
  (`BVH - Capture` for live capture, or `BVH - Edit` to replay a recording).
3. Recommended broadcast settings:
  - Skeleton: **Axis Studio**, Rotation: **YXZ**, Displacement: **checked**
  - Frame Format: **Binary**, *Use old header format*: **unchecked**
  - Protocol: **UDP**
  - Local Address: **the Windows PC LAN IP**
  - Destination Address: `**<linux_host_ip>:<port>`**
4. On the Linux host:

```bash
python -m deploy.play_track --real --net <nic_name> --mocap_type pnlink
```

#### Xsens MVN Analyze / Animate (`--mocap_type xsens`)

1. Install **MVN Analyze / Animate** on the Windows PC, connect the Xsens suit,
  and complete calibration.
2. Go to `Options → Preferences → Miscellaneous → Network Streamer` (or
  `Options → Network Streamer`) and **Add** a target destination:
  - Host: `**<linux_host_ip>`** (the Linux machine running deploy)
  - Port: `**9763**` (MVN default)
  - Protocol: **TCP** or **UDP** (must match `--xsens_protocol`)
  - Format: **Position + Orientation (Quaternion)**
  - Enable the stream by ticking the checkbox next to it.
3. On the Linux host (match host/port/protocol to MVN):

```bash
python -m deploy.play_track --real --net <nic_name> \
    --mocap_type xsens --xsens_host 0.0.0.0 --xsens_port 9763 --xsens_protocol tcp
```

> `--xsens_host` is the **local bind address** of the receiver (`0.0.0.0`
> listens on all interfaces); `--xsens_port` / `--xsens_protocol` must equal the
> MVN Network Streamer settings above.

## Robot Bring-Up (Real Mode)

For initial tests, suspend the robot for safety.

1. Power on the battery (short press, then long press for ~2 s).
2. After head indicator stabilization, enter debug mode via `L2 + R2`.
3. Optionally verify mode switching with `L2 + A` (position) and `L2 + B` (damping).

Network setup:

1. Connect host and robot via Ethernet.
2. Configure host IP in the same subnet as the robot.
3. Verify connectivity: `ping <robot_ip>`.
4. Find network interface name:

```bash
ifconfig
# or
ip addr
```

Pass the interface name to `--net`.

## Running

### Simulation

```bash
python -m deploy.play_track
python -m deploy.play_track --no-mocap
python -m deploy.play_track --track-dir storage/test
python -m deploy.play_track --track-dir storage/test/human_walking_50Hz_29dof.npz
```

### Real robot

```bash
python -m deploy.play_track --real --net <nic_name>
python -m deploy.play_track --real --net <nic_name> --enable-hand
python -m deploy.play_track --real --net <nic_name> --visualize-retarget False
```

## Control Interface

### Keyboard control (GUI)


| Key     | Function                                           |
|---------|----------------------------------------------------|
| `0`     | Walk mode                                          |
| `1`     | Online retarget mode                               |
| `2`-`9` | Offline trajectory modes (sorted from `track_dir`) |
| `W/S`   | Linear velocity x (+/-)                            |
| `A/D`   | Linear velocity y (+/-)                            |
| `Q/E`   | Yaw rate (+/-)                                     |
| `R`     | Reset simulation (simulation mode only)            |
| ```     | Exit simulation loop (simulation mode only)        |


Mode keys are single-character digits; in practice, keep offline trajectories within modes `2..9`.

### Remote controller sequence (real robot)

1. `start`: damping to default posture.
2. `A`: enter locomotion/tracking loop.
3. `select`: emergency stop and return to damping.

## Main CLI Arguments


| Argument               | Default                            | Meaning                                        |
|------------------------|------------------------------------|------------------------------------------------|
| `--real`               | `False`                            | Enable real-robot mode                         |
| `--net`                | `enx00e04c161320`                  | DDS network interface                          |
| `--freq`               | `50`                               | Control frequency (Hz)                         |
| `--onnx-walk`          | `storage/ckpts/G1-Walk/...onnx`    | Walk policy path                               |
| `--onnx-track`         | `storage/ckpts/G1-TrackV5/...onnx` | Tracking policy path                           |
| `--policy-type`        | `mlp`                              | Policy architecture (`mlp`)                    |
| `--track-dir`          | `storage/test`                     | Offline trajectory folder or single `.npz`     |
| `--no-mocap`           | `False`                            | Disable online mocap in simulation             |
| `--mocap-type`         | `pnlink`                           | `pnlink` or `xsens`                            |
| `--human-height`       | `1.6`                              | Retargeting height prior                       |
| `--visualize-retarget` | `True`                             | Enable retarget visualization process          |
| `--enable-hand`        | `False`                            | Enable Dex3-1 hand control                     |
| `--debug`              | `False`                            | Real mode without low-level command publishing |
