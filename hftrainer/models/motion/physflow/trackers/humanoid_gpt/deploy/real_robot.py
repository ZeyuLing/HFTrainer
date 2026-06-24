"""Real-robot low-level control for Unitree G1.

Provides:
- LowLevelControlG1: read sensor state and send joint-level PD targets.
- RealRobotState: adapter that writes sensor readings into a MuJoCo
  mj_data / State so that G1TrackInferFn.update_state() works unchanged.
"""

from __future__ import annotations

import ctypes
import time
import struct
import mujoco
import numpy as np

from deploy.constants import (
    NUM_JOINT, DEFAULT_QPOS, TOPIC_LOWCMD, TOPIC_LOWSTATE, MotorID,
)

from tracking.infer_utils import State
from tracking.constants import KPs, KDs


# ---------------------------------------------------------------------------
# Fast HG LowCmd byte-mirror + CRC
# ---------------------------------------------------------------------------
#
# The unitree SDK's CRC.Crc(cmd) re-serializes the entire LowCmd (35 motors x
# 7 fields + struct.pack with 245 args + a 250-iter Python __Trans loop) every
# call -- ~0.85 ms per step on the Jetson.  We replace that with a numpy-backed
# byte mirror that matches the IDL packing format exactly: a single bulk numpy
# write to update q / kp / kd, followed by a direct ctypes call to the same
# C ``crc32_core`` from libcrc_aarch64.so.  CRC value is bit-identical to the
# reference path (verified against unitree_sdk2py.utils.crc.CRC).
#
# Format string (from unitree_sdk2py.utils.crc):
#     '<2B2x' + 'B3x5fI' * 35 + '5I'   -> total 1004 bytes
# Layout:
#     offset 0:  mode_pr (u8), mode_machine (u8), 2x pad
#     offset 4 + 28*i (i=0..34) per motor:
#         mode (u8) + 3x pad,
#         q (f32), dq (f32), tau (f32), kp (f32), kd (f32),
#         reserve (u32)
#     offset 984:  reserve[0..3] (4x u32)
#     offset 1000: crc (u32, excluded from the CRC calculation)
# CRC range: first 250 uint32 words.

_HG_LOW_CMD_BYTES = 1004
_HG_LOW_CMD_N_WORDS = 250  # (1004 // 4) - 1 (skip the trailing crc field)
_HG_LOW_CMD_N_MOTORS = 35
_HG_LOW_CMD_MOTOR_BASE = 4
_HG_LOW_CMD_MOTOR_BLOCK = 28
_HG_LOW_CMD_RESERVE_BASE = 984

_HG_MOTOR_DTYPE = np.dtype(
    [
        ("mode", "<u1"),
        ("_pad", "<u1", (3,)),
        ("q", "<f4"),
        ("dq", "<f4"),
        ("tau", "<f4"),
        ("kp", "<f4"),
        ("kd", "<f4"),
        ("reserve", "<u4"),
    ],
    align=False,
)
assert _HG_MOTOR_DTYPE.itemsize == _HG_LOW_CMD_MOTOR_BLOCK


class FastHGLowCmdPacker:
    """Byte-mirror of an HG ``LowCmd_`` for fast CRC computation.

    Writes q / kp / kd into a fixed 1004-byte buffer via numpy bulk indexing
    (avoiding per-field Python attribute access + struct.pack), then invokes
    the unitree SDK's existing C ``crc32_core`` directly through ctypes.

    Intended usage::

        packer = FastHGLowCmdPacker(low_cmd, crc_lib)
        # every control step:
        packer.write_motors(active_ids, q_target, kps, kds)
        crc_value = packer.crc()
        low_cmd.crc = crc_value
    """

    __slots__ = (
        "_buf",
        "_motors_view",
        "_u4",
        "_ptr",
        "_crc_fn",
    )

    def __init__(self, low_cmd, crc_lib) -> None:
        self._buf = np.zeros(_HG_LOW_CMD_BYTES, dtype=np.uint8)
        # Fill static fields from the existing LowCmd
        self._buf[0] = int(low_cmd.mode_pr) & 0xFF
        self._buf[1] = int(low_cmd.mode_machine) & 0xFF
        # Per-motor: copy mode + reserve bytes (rest stay zero)
        for i in range(_HG_LOW_CMD_N_MOTORS):
            base = _HG_LOW_CMD_MOTOR_BASE + _HG_LOW_CMD_MOTOR_BLOCK * i
            self._buf[base] = int(low_cmd.motor_cmd[i].mode) & 0xFF
            struct.pack_into(
                "<I", self._buf, base + 24,
                int(getattr(low_cmd.motor_cmd[i], "reserve", 0)) & 0xFFFFFFFF,
            )
        # Whole-cmd reserve[0..3] block (4x uint32)
        try:
            res = list(low_cmd.reserve)
        except TypeError:
            res = [0, 0, 0, 0]
        for i in range(4):
            struct.pack_into(
                "<I", self._buf, _HG_LOW_CMD_RESERVE_BASE + 4 * i,
                int(res[i]) & 0xFFFFFFFF,
            )

        # Numpy structured view over the 35-motor array (zero-copy alias)
        motor_bytes_end = (
            _HG_LOW_CMD_MOTOR_BASE
            + _HG_LOW_CMD_MOTOR_BLOCK * _HG_LOW_CMD_N_MOTORS
        )
        self._motors_view = self._buf[
            _HG_LOW_CMD_MOTOR_BASE:motor_bytes_end
        ].view(_HG_MOTOR_DTYPE)
        # uint32 view + cached ctypes pointer for the C CRC call
        self._u4 = self._buf.view(np.uint32)
        self._ptr = self._u4.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        # Make sure argtypes/restype are set on the shared lib (cheap, idempotent).
        crc_lib.crc32_core.argtypes = (
            ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        )
        crc_lib.crc32_core.restype = ctypes.c_uint32
        self._crc_fn = crc_lib.crc32_core

    def write_motors(
        self,
        active_ids: np.ndarray,
        q: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
    ) -> None:
        """Bulk-update q / kp / kd for the given motor ids."""
        v = self._motors_view
        v["q"][active_ids] = q[active_ids]
        v["kp"][active_ids] = kp[active_ids]
        v["kd"][active_ids] = kd[active_ids]

    def write_damping(
        self, active_ids: np.ndarray, kd_value: float = 8.0
    ) -> None:
        """Set q=qd=tau=kp=0, kd=kd_value for the given motor ids."""
        v = self._motors_view
        v["q"][active_ids] = 0.0
        v["dq"][active_ids] = 0.0
        v["tau"][active_ids] = 0.0
        v["kp"][active_ids] = 0.0
        v["kd"][active_ids] = kd_value

    def crc(self) -> int:
        """Compute CRC32 over the first 250 uint32 words of the byte mirror.

        Bit-identical to ``unitree_sdk2py.utils.crc.CRC.Crc(low_cmd)`` for
        an HG LowCmd whose contents match the byte mirror.
        """
        return int(self._crc_fn(self._ptr, _HG_LOW_CMD_N_WORDS))


# ---------------------------------------------------------------------------
# Unitree SDK helpers (deferred imports keep sim-only usage SDK-free)
# ---------------------------------------------------------------------------

class KeyMap:
    R1 = 0; L1 = 1; start = 2; select = 3
    R2 = 4; L2 = 5; F1 = 6; F2 = 7
    A = 8; B = 9; X = 10; Y = 11
    up = 12; right = 13; down = 14; left = 15


class RemoteController:
    def __init__(self):
        self.lx = self.ly = self.rx = self.ry = 0
        self.button = [0] * 16

    def set(self, data):
        keys = struct.unpack("H", data[2:4])[0]
        for i in range(16):
            self.button[i] = (keys & (1 << i)) >> i
        self.lx = struct.unpack("f", data[4:8])[0]
        self.rx = struct.unpack("f", data[8:12])[0]
        self.ry = struct.unpack("f", data[12:16])[0]
        self.ly = struct.unpack("f", data[20:24])[0]


class _UnitreeMotor:
    class MotorMode:
        PR = 0
        AB = 1

    @staticmethod
    def create_damping_cmd(cmd):
        for i in range(len(cmd.motor_cmd)):
            cmd.motor_cmd[i].q = 0
            cmd.motor_cmd[i].qd = 0
            cmd.motor_cmd[i].kp = 0
            cmd.motor_cmd[i].kd = 8
            cmd.motor_cmd[i].tau = 0

    @staticmethod
    def create_zero_cmd(cmd):
        for i in range(len(cmd.motor_cmd)):
            cmd.motor_cmd[i].q = 0
            cmd.motor_cmd[i].qd = 0
            cmd.motor_cmd[i].kp = 0
            cmd.motor_cmd[i].kd = 0
            cmd.motor_cmd[i].tau = 0

    @staticmethod
    def init_cmd_hg(cmd, mode_machine, mode_pr):
        cmd.mode_machine = mode_machine
        cmd.mode_pr = mode_pr
        for i in range(len(cmd.motor_cmd)):
            cmd.motor_cmd[i].mode = 1
            cmd.motor_cmd[i].q = 0
            cmd.motor_cmd[i].qd = 0
            cmd.motor_cmd[i].kp = 0
            cmd.motor_cmd[i].kd = 0
            cmd.motor_cmd[i].tau = 0


# ---------------------------------------------------------------------------
# Low-level G1 controller
# ---------------------------------------------------------------------------

class LowLevelControlG1:
    """Low-level Unitree G1 motor controller via DDS."""

    def __init__(
        self,
        active_j2m_ids=MotorID.FULL,
        ctrl_dt: float = 0.02,
        debug: bool = False,
    ):
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG, LowState_ as LowStateHG
        from unitree_sdk2py.utils.crc import CRC

        self.debug = debug
        self.ctrl_dt = ctrl_dt
        self.default_qpos = DEFAULT_QPOS.copy()
        self._kps = KPs.copy()
        self._kds = KDs.copy()
        self.active_j2m_ids = list(active_j2m_ids)
        self._crc = CRC()

        self.joint_qpos = np.zeros(NUM_JOINT, dtype=np.float32)
        self.joint_qvel = np.zeros(NUM_JOINT, dtype=np.float32)
        self.joint_torque = np.zeros(NUM_JOINT, dtype=np.float32)
        self.root_quat = np.zeros(4, dtype=np.float32)
        self.root_gyro = np.zeros(3, dtype=np.float32)

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = _UnitreeMotor.MotorMode.PR
        self.mode_machine_ = 0
        self.remote = RemoteController()

        self._LowCmdHG = LowCmdHG
        self._pub = ChannelPublisher(TOPIC_LOWCMD, LowCmdHG)
        self._pub.Init()
        self._sub = ChannelSubscriber(TOPIC_LOWSTATE, LowStateHG)
        self._sub.Init(self._handle_low_state, 10)

        self._wait_low_state()
        _UnitreeMotor.init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)

        # Fast byte-mirror CRC + scratch numpy arrays for active motors.  The
        # mirror is cached after init_cmd_hg() so its mode_pr / mode_machine /
        # motor_cmd[i].mode bytes match the IDL struct.
        self._fast_packer = FastHGLowCmdPacker(self.low_cmd, self._crc.crc_lib)
        self._active_ids_np = np.asarray(self.active_j2m_ids, dtype=np.intp)
        # Pre-extract numpy float32 views of constants so ``write_motors`` can
        # bulk-copy without per-call dtype conversion.
        self._kps_f32 = np.ascontiguousarray(self._kps, dtype=np.float32)
        self._kds_f32 = np.ascontiguousarray(self._kds, dtype=np.float32)

        print("[LowLevelControlG1] Connected.")

    def _wait_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.ctrl_dt)

    def _handle_low_state(self, msg):
        self.low_state = msg
        self.mode_machine_ = msg.mode_machine
        self.remote.set(msg.wireless_remote)

    def _send(self):
        # Slow path: re-pack and CRC via the SDK helper (~0.85 ms on Jetson).
        # Kept for backward-compat with callers that mutate ``self.low_cmd``
        # directly (e.g. legacy code paths).  Production code should use
        # :py:meth:`fast_step` which writes through the byte-mirror.
        self.low_cmd.crc = self._crc.Crc(self.low_cmd)
        if not self.debug:
            self._pub.Write(self.low_cmd)

    def get_sensor_state(self):
        """Read joint + IMU state from the robot."""
        full_ids = MotorID.FULL
        for i in range(NUM_JOINT):
            mid = full_ids[i]
            self.joint_qpos[i] = self.low_state.motor_state[mid].q
            self.joint_qvel[i] = self.low_state.motor_state[mid].dq
            self.joint_torque[i] = self.low_state.motor_state[mid].tau_est
        # ``imu_state.quaternion`` / ``gyroscope`` are SDK arrays; index them
        # directly to avoid the ~10 us np.array(...) constructor.
        q = self.low_state.imu_state.quaternion
        g = self.low_state.imu_state.gyroscope
        self.root_quat[0] = q[0]; self.root_quat[1] = q[1]
        self.root_quat[2] = q[2]; self.root_quat[3] = q[3]
        self.root_gyro[0] = g[0]; self.root_gyro[1] = g[1]; self.root_gyro[2] = g[2]
        # Return views (caller consumes within the same control step before
        # the next get_sensor_state overwrites them; saves 4 numpy copies).
        return (
            self.root_quat,
            self.root_gyro,
            self.joint_qpos,
            self.joint_qvel,
        )

    # ------------------------------------------------------------------
    # Fast pack + CRC path
    # ------------------------------------------------------------------

    def fast_pack_motor_cmd(
        self,
        tar_qpos: np.ndarray,
        kps: np.ndarray | None = None,
        kds: np.ndarray | None = None,
    ) -> None:
        """Write q / kp / kd into BOTH the byte-mirror and the IDL struct.

        Numpy bulk-writes the byte mirror (used by :py:meth:`fast_compute_crc`)
        and per-motor attribute-writes the IDL struct (used by DDS publish).
        Both paths see the same values, so the CRC is correct for the message
        we publish.
        """
        if kps is None:
            kps = self._kps_f32
        elif kps is not self._kps_f32:
            kps = np.ascontiguousarray(kps, dtype=np.float32)
        if kds is None:
            kds = self._kds_f32
        elif kds is not self._kds_f32:
            kds = np.ascontiguousarray(kds, dtype=np.float32)
        tar_qpos = np.ascontiguousarray(tar_qpos, dtype=np.float32)

        active = self._active_ids_np
        # Byte-mirror bulk write
        self._fast_packer.write_motors(active, tar_qpos, kps, kds)
        # IDL-struct write (only fields the DDS message actually needs).
        # qd / tau remain at 0 from init_cmd_hg(); mode stays at 1.
        cmd = self.low_cmd
        for mid in self.active_j2m_ids:
            mc = cmd.motor_cmd[mid]
            mc.q = float(tar_qpos[mid])
            mc.kp = float(kps[mid])
            mc.kd = float(kds[mid])

    def fast_pack_damping(self, kd_value: float = 8.0) -> None:
        """Write damping (q=kp=0, kd=kd_value) to all 35 motors.

        Mirrors :py:meth:`_UnitreeMotor.create_damping_cmd` (every motor
        slot, not just the active ones) so that disconnected / passive
        joints also receive the safety damping torque.
        """
        all_ids = np.arange(_HG_LOW_CMD_N_MOTORS, dtype=np.intp)
        self._fast_packer.write_damping(all_ids, kd_value=kd_value)
        cmd = self.low_cmd
        for i in range(_HG_LOW_CMD_N_MOTORS):
            mc = cmd.motor_cmd[i]
            mc.q = 0.0
            mc.qd = 0.0
            mc.tau = 0.0
            mc.kp = 0.0
            mc.kd = float(kd_value)

    def fast_compute_crc(self) -> int:
        """CRC over the byte-mirror; also stamps it into ``low_cmd.crc``."""
        crc = self._fast_packer.crc()
        self.low_cmd.crc = crc
        return crc

    def fast_publish(self) -> None:
        if not self.debug:
            self._pub.Write(self.low_cmd)

    def fast_step(
        self,
        tar_qpos: np.ndarray,
        kps: np.ndarray | None = None,
        kds: np.ndarray | None = None,
    ) -> None:
        """Like :py:meth:`step` but uses the byte-mirror CRC path."""
        self.fast_pack_motor_cmd(tar_qpos, kps, kds)
        self.fast_compute_crc()
        self.fast_publish()

    # Make the standard step() route through the fast path so legacy callers
    # automatically benefit (no API change).  Behaviour is identical except
    # that qd / tau remain at their init defaults of 0 (the previous
    # implementation re-wrote them to 0 every step anyway).
    def step(self, tar_qpos: np.ndarray, kps=None, kds=None):
        """Send PD targets to the robot (fast path)."""
        self.fast_step(tar_qpos, kps, kds)

    def set_motor_damping(self):
        """Send damping cmd (fast path)."""
        self.fast_pack_damping(kd_value=8.0)
        self.fast_compute_crc()
        self.fast_publish()

    def move_to_default_pos(self, duration: float = 2.0, dt: float = 0.02):
        curr_qpos = np.zeros(NUM_JOINT, dtype=np.float32)
        for mid in self.active_j2m_ids:
            curr_qpos[mid] = self.low_state.motor_state[mid].q
        num_steps = int(duration / dt)
        init_kps = np.zeros_like(self._kps)
        for i in range(1, num_steps + 1):
            if self.remote.button[KeyMap.select] == 1:
                self.set_motor_damping()
                break
            alpha = i / num_steps
            _kps = init_kps * (1 - alpha) + self._kps * alpha
            _qpos = curr_qpos * (1 - alpha) + self.default_qpos * alpha
            self.step(_qpos, _kps, self._kds)
            time.sleep(dt)


# ---------------------------------------------------------------------------
# Adapter: real sensor data -> MuJoCo State for G1TrackInferFn
# ---------------------------------------------------------------------------

class RealRobotState:
    """Creates a MuJoCo State from real-robot sensor readings.

    G1TrackInferFn.update_state() reads qpos, qvel, sensor data, and
    body xpos/xmat from State.mj_data.  We populate a phantom mj_data
    with real sensor readings and call mj_forward() so that FK quantities
    (xpos, xmat, cvel, site data) are consistent.
    """

    def __init__(self, mj_model: mujoco.MjModel):
        self.mj_model = mj_model
        self.mj_data = mujoco.MjData(mj_model)

    def build_state(
        self,
        root_quat: np.ndarray,
        root_gyro: np.ndarray,
        joint_qpos: np.ndarray,
        joint_qvel: np.ndarray,
    ) -> State:
        """Populate mj_data from sensor readings and compute FK."""
        d = self.mj_data
        # Root: use nominal standing height (IMU has no position)
        d.qpos[:3] = np.array([0.0, 0.0, 0.78], dtype=np.float32)
        d.qpos[3:7] = root_quat
        d.qpos[7:] = joint_qpos
        # Root angular velocity (body frame) -> cvel for correct keypoint velocities
        d.qvel[3:6] = root_gyro
        d.qvel[6:] = joint_qvel

        # IMU gyro -> write into sensordata for gyro_pelvis sensor
        sensor_id = self.mj_model.sensor("gyro_pelvis").id
        adr = self.mj_model.sensor_adr[sensor_id]
        dim = self.mj_model.sensor_dim[sensor_id]
        d.sensordata[adr: adr + dim] = root_gyro

        mujoco.mj_forward(self.mj_model, d)
        return State(mj_data=d)
