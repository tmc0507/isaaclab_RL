#!/usr/bin/env python3
"""
PID Velocity Tuner + Hybrid TCN Plant + Adaptive AI PID Tuner - V4.2

Mục tiêu V4.2:
- Giữ toàn bộ pipeline nhận dạng motor V3: Hybrid Physics + TCN residual + deep ensemble,
  history theo thời gian, multi-step rollout và free-run validation.
- Thêm một mạng AI thứ hai (PIDGainNet) để đề xuất trực tiếp Kp, Ki, Kd từ "plant fingerprint"
  (chữ ký đáp ứng động học) thay vì chỉ quét toàn bộ không gian PID.
- PIDGainNet được meta-train end-to-end qua learned plant: randomize động học quanh plant hiện tại,
  chạy diagnostic response, AI sinh PID, rồi backprop qua closed-loop simulation để tối ưu gain.
  Không cần tạo nhãn PID thủ công cho từng plant.
- Sau AI inference, V4 dùng Local Differential Evolution quanh nghiệm AI để tinh chỉnh nhẹ và
  kiểm chứng bằng learned plant. Global DE của V3 vẫn được giữ làm fallback/baseline.
- Không tự động ghi PID xuống ESP32. Chỉ Apply sau khi candidate đã verify và người dùng xác nhận.
- V4.1 tích hợp trainer V3.1: CUDA/CPU/Auto, AMP, live batch progress, ETA và validation thưa hơn.
- V4.1 đánh giá PID ở CẢ HAI CHIỀU quay (+/- target) để tận dụng dữ liệu motor thuận/nghịch và tăng độ bền Sim-to-Real.
- V4.2 sửa lỗi GUI AI Tuner: nút TRAIN AI bị chồng hàng Kd max, thiếu lbl_ai_live làm callback lỗi, và đồng bộ trạng thái enable/disable của các nút.

Kiến trúc:
  Real data -> HybridMotorTCN (plant model)
                  |
                  +-> diagnostic response -> PIDGainNet -> [Kp,Ki,Kd]
                                                   |
                                                   +-> local DE refinement
                                                   +-> simulator verification
                                                   +-> manual Apply to ESP32

Plant model vẫn không dùng target/error làm feature; PIDGainNet mới chỉ dùng target/rpm/pwm/error
ở diagnostic sequence để nhận biết trạng thái closed-loop hiện tại của hệ.
"""

from __future__ import annotations

import sys
import os
import json
import time
import math
import random
import re
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLineEdit, QPushButton, QLabel, QComboBox,
    QTextEdit, QProgressBar, QTabWidget, QGroupBox, QSpinBox,
    QDoubleSpinBox, QFileDialog, QCheckBox, QMessageBox,
)
from PyQt6.QtCore import QTimer, QThread, pyqtSignal

import serial
import serial.tools.list_ports


# ============================================================
# CONFIG
# ============================================================

BAUD_RATE = 115200
SERIAL_TIMEOUT = 0.05
NOMINAL_DT = 0.020             # ESP32 PID 50 Hz
PWM_LIMIT = 250.0              # giữ đúng giới hạn controller trong bản cũ
INTEGRAL_LIMIT = 50.0
PLOT_WINDOW_S = 5.0

# V3 model defaults
DEFAULT_HISTORY_LEN = 25          # 25 samples @ 50 Hz = 0.5 s history
DEFAULT_ROLLOUT_HORIZON = 12      # 240 ms differentiable rollout during training
DEFAULT_TCN_CHANNELS = 64
DEFAULT_TCN_LEVELS = 4            # dilation 1,2,4,8
DEFAULT_ENSEMBLE_SIZE = 3
DEFAULT_DROPOUT = 0.05
MAX_SEQUENCE_GAP_S = 0.150        # split independent recordings if gap is too large
PHYSICS_RIDGE = 1e-4
ROLLOUT_LOSS_WEIGHT = 1.5
INPUT_NOISE_STD = 0.005           # small normalized augmentation

# V3.1/V4.1 training-speed + progress defaults
VALIDATE_EVERY_EPOCHS = 5
LIVE_UPDATES_PER_EPOCH = 10
CPU_LEAVE_FREE_CORES = 1
V4_AI_VALIDATE_EVERY_EPOCHS = 5


# V4 Adaptive AI PID Tuner defaults
V4_DIAG_SEGMENT_STEPS = 10      # 6 segments -> 1.20 s @ 50 Hz
V4_EVAL_SEGMENT_STEPS = 14      # 6 segments -> 1.68 s @ 50 Hz
V4_TUNER_EPOCHS = 120
V4_TUNER_BATCH = 6
V4_TUNER_LR = 3e-4
V4_BPTT_STEPS = 20              # truncated backprop để giảm GPU/RAM nhưng vẫn giữ dynamics ngắn hạn
V4_RANDOM_SEVERITY = 0.25
V4_LOCAL_REFINE_FRAC = 0.15
V4_LOCAL_REFINE_EVALS = 80
V4_VERIFY_MAX_OVERSHOOT_PCT = 5.0
V4_VERIFY_MAX_SATURATION = 0.50
V4_VERIFY_MAX_OOD = 0.25

# ============================================================
# PID OPTIMIZATION PRIORITY
# ============================================================
# Ưu tiên: overshoot thấp -> settling time ngắn -> rise time nhanh.
# Settling time được tính trong dải ±2% thay vì ±5%.
SETTLING_BAND_FRAC = 0.02

# Overshoot <= 1% được xem là vùng chấp nhận được.
# Vượt quá mức này sẽ bị phạt rất mạnh để optimizer không đổi
# overshoot lấy thời gian đáp ứng nhanh.
OVERSHOOT_LIMIT_PCT = 1.0

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================
# UTILS
# ============================================================

def safe_dt(t_now: float, t_prev: float, nominal: float = NOMINAL_DT) -> float:
    """
    Ước lượng dt của plant từ timestamp host nhưng lượng tử theo chu kỳ PID.

    Serial/OS có jitter nên chênh lệch timestamp nhận gói không phải dt vật lý chính xác.
    Firmware hiện chạy nominal 20 ms; vì vậy 18/21/23 ms đều quy về 20 ms. Nếu
    mất một sample và gap gần 40 ms thì giữ 40 ms.
    """
    dt = t_now - t_prev
    if not np.isfinite(dt) or dt < 0.005 or dt > 0.100:
        return nominal
    n = int(np.clip(round(dt / nominal), 1, 5))
    return float(n * nominal)


def robust_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if len(y_true) < 2:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


# ============================================================
# V3 HYBRID PHYSICS + TCN ENSEMBLE PLANT MODEL
# ============================================================

class CausalConv1d(nn.Module):
    """1D causal convolution: output tại t chỉ nhìn t và quá khứ."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size=kernel_size,
            dilation=dilation, padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.left_pad > 0:
            x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = F.silu(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = F.silu(y)
        y = self.dropout(y)
        return x + y


class TemporalMotorMember(nn.Module):
    """Một TCN member dự đoán residual delta-RPM đã được standardize."""

    def __init__(self, input_size: int, channels: int, levels: int,
                 kernel_size: int = 3, dropout: float = DEFAULT_DROPOUT):
        super().__init__()
        self.input_proj = nn.Conv1d(input_size, channels, kernel_size=1)
        self.blocks = nn.ModuleList([
            TemporalResidualBlock(
                channels=channels,
                kernel_size=kernel_size,
                dilation=2 ** i,
                dropout=dropout,
            )
            for i in range(levels)
        ])
        self.head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, max(16, channels // 2)),
            nn.SiLU(),
            nn.Linear(max(16, channels // 2), 1),
        )

    def forward(self, x_btf: torch.Tensor) -> torch.Tensor:
        # [B,T,F] -> [B,F,T]
        x = x_btf.transpose(1, 2)
        x = F.silu(self.input_proj(x))
        for block in self.blocks:
            x = block(x)
        # causal feature cuối cùng đại diện toàn bộ history
        return self.head(x[:, :, -1])


class HybridMotorTCN(nn.Module):
    """
    Learned motor plant cho V3.

    Neural input mỗi sample:
        [rpm, pwm, drpm, dpwm, dt]

    TCN nhìn HISTORY_LEN samples. Output là residual delta-RPM so với một physics
    prior tuyến tính-nonlinear rất nhẹ. Physics prior giúp extrapolation bớt vô lý;
    TCN học phần ma sát/dead-zone/trễ/phi tuyến còn thiếu.
    """

    FEATURE_NAMES = ["rpm", "pwm", "drpm", "dpwm", "dt"]

    def __init__(self, history_len: int = DEFAULT_HISTORY_LEN,
                 channels: int = DEFAULT_TCN_CHANNELS,
                 levels: int = DEFAULT_TCN_LEVELS,
                 ensemble_size: int = DEFAULT_ENSEMBLE_SIZE,
                 kernel_size: int = 3,
                 dropout: float = DEFAULT_DROPOUT):
        super().__init__()
        self.input_size = len(self.FEATURE_NAMES)
        self.history_len = int(history_len)
        self.channels = int(channels)
        self.levels = int(levels)
        self.ensemble_size = int(ensemble_size)
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)

        self.members = nn.ModuleList([
            TemporalMotorMember(
                input_size=self.input_size,
                channels=self.channels,
                levels=self.levels,
                kernel_size=self.kernel_size,
                dropout=self.dropout,
            )
            for _ in range(self.ensemble_size)
        ])

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_trained = False

        # acceleration prior:
        # a = c0 + c1*rpm + c2*pwm + c3*tanh(rpm/5) + c4*tanh(pwm/20)
        self.physics_coef = np.zeros(5, dtype=np.float64)

        self.feature_min: Optional[np.ndarray] = None
        self.feature_max: Optional[np.ndarray] = None
        self.train_rpm_min: Optional[float] = None
        self.train_rpm_max: Optional[float] = None
        self.train_pwm_min: Optional[float] = None
        self.train_pwm_max: Optional[float] = None
        self.val_metrics: Dict[str, float] = {}

        # scalar runtime state cho GUI/PID optimizer. IsaacLab sau này sẽ dùng
        # vectorized history riêng cho từng environment.
        self._runtime_history: List[np.ndarray] = []
        self._last_runtime_rpm = 0.0
        self._last_runtime_pwm = 0.0

    @property
    def model_config(self) -> dict:
        return {
            "history_len": self.history_len,
            "channels": self.channels,
            "levels": self.levels,
            "ensemble_size": self.ensemble_size,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
        }

    def forward(self, x_btf: torch.Tensor) -> torch.Tensor:
        """Return [ensemble, batch, 1] residual predictions in scaled space."""
        return torch.stack([m(x_btf) for m in self.members], dim=0)

    @staticmethod
    def physics_design_numpy(rpm: np.ndarray, pwm: np.ndarray) -> np.ndarray:
        rpm = np.asarray(rpm, dtype=np.float64)
        pwm = np.asarray(pwm, dtype=np.float64)
        return np.column_stack([
            np.ones_like(rpm),
            rpm,
            pwm,
            np.tanh(rpm / 5.0),
            np.tanh(pwm / 20.0),
        ])

    def fit_physics_prior(self, rpm: np.ndarray, pwm: np.ndarray,
                          dt: np.ndarray, delta_rpm: np.ndarray) -> None:
        rpm = np.asarray(rpm, dtype=np.float64).reshape(-1)
        pwm = np.asarray(pwm, dtype=np.float64).reshape(-1)
        dt = np.asarray(dt, dtype=np.float64).reshape(-1)
        delta_rpm = np.asarray(delta_rpm, dtype=np.float64).reshape(-1)
        dt_safe = np.clip(dt, 0.005, 0.100)
        accel = delta_rpm / dt_safe
        A = self.physics_design_numpy(rpm, pwm)

        # ridge rất nhẹ, không regularize bias quá mạnh
        reg = PHYSICS_RIDGE * np.eye(A.shape[1], dtype=np.float64)
        reg[0, 0] *= 0.01
        try:
            self.physics_coef = np.linalg.solve(A.T @ A + reg, A.T @ accel)
        except np.linalg.LinAlgError:
            self.physics_coef = np.linalg.lstsq(A, accel, rcond=None)[0]

    def physics_delta_numpy(self, rpm, pwm, dt):
        rpm_a = np.asarray(rpm, dtype=np.float64)
        pwm_a = np.asarray(pwm, dtype=np.float64)
        dt_a = np.asarray(dt, dtype=np.float64)
        c = self.physics_coef
        accel = (
            c[0] + c[1] * rpm_a + c[2] * pwm_a
            + c[3] * np.tanh(rpm_a / 5.0)
            + c[4] * np.tanh(pwm_a / 20.0)
        )
        return accel * dt_a

    def physics_delta_torch(self, rpm: torch.Tensor, pwm: torch.Tensor,
                            dt: torch.Tensor) -> torch.Tensor:
        c = torch.as_tensor(self.physics_coef, dtype=rpm.dtype, device=rpm.device)
        accel = (
            c[0] + c[1] * rpm + c[2] * pwm
            + c[3] * torch.tanh(rpm / 5.0)
            + c[4] * torch.tanh(pwm / 20.0)
        )
        return accel * dt

    @staticmethod
    def make_feature(rpm: float, pwm: float, drpm: float,
                     dpwm: float, dt: float) -> np.ndarray:
        dt = float(np.clip(dt, 0.005, 0.100))
        return np.asarray([rpm, pwm, drpm, dpwm, dt], dtype=np.float64)

    def _pad_history(self, history: List[np.ndarray]) -> np.ndarray:
        if not history:
            history = [self.make_feature(0.0, 0.0, 0.0, 0.0, NOMINAL_DT)]
        if len(history) < self.history_len:
            pad = [history[0].copy() for _ in range(self.history_len - len(history))]
            history = pad + history
        return np.asarray(history[-self.history_len:], dtype=np.float64)

    def predict_delta_from_history(self, history_raw: np.ndarray,
                                   return_uncertainty: bool = False):
        if not self.is_trained:
            return (0.0, 0.0) if return_uncertainty else 0.0

        hist = np.asarray(history_raw, dtype=np.float64)
        if hist.ndim != 2 or hist.shape[1] != self.input_size:
            raise ValueError(f"history must have shape [T,{self.input_size}]")
        if len(hist) < self.history_len:
            hist = self._pad_history([x.copy() for x in hist])
        else:
            hist = hist[-self.history_len:]

        x_s = self.scaler_X.transform(hist).astype(np.float32)
        x_t = torch.from_numpy(x_s).unsqueeze(0)
        self.eval()
        with torch.inference_mode():
            pred_s = self.forward(x_t).squeeze(-1).squeeze(-1).cpu().numpy()

        # mỗi member output residual standardized
        residual_members = pred_s * float(self.scaler_y.scale_[0]) + float(self.scaler_y.mean_[0])
        residual_mean = float(np.mean(residual_members))
        residual_std = float(np.std(residual_members))

        rpm, pwm, _, _, dt = hist[-1]
        delta = float(self.physics_delta_numpy(rpm, pwm, dt) + residual_mean)
        if return_uncertainty:
            return delta, residual_std
        return delta

    def reset_runtime(self, rpm: float = 0.0, pwm: float = 0.0,
                      dt: float = NOMINAL_DT) -> None:
        base = self.make_feature(rpm, pwm, 0.0, 0.0, dt)
        self._runtime_history = [base.copy() for _ in range(max(0, self.history_len - 1))]
        self._last_runtime_rpm = float(rpm)
        self._last_runtime_pwm = float(pwm)

    def predict_next_rpm(self, rpm: float, pwm: float, dt: float,
                         return_uncertainty: bool = False):
        if not self._runtime_history:
            self.reset_runtime(rpm=rpm, pwm=pwm, dt=dt)

        drpm = float(rpm - self._last_runtime_rpm)
        dpwm = float(pwm - self._last_runtime_pwm)
        feat = self.make_feature(rpm, pwm, drpm, dpwm, dt)
        self._runtime_history.append(feat)
        self._runtime_history = self._runtime_history[-self.history_len:]

        delta, unc = self.predict_delta_from_history(
            np.asarray(self._runtime_history), return_uncertainty=True
        )
        self._last_runtime_rpm = float(rpm)
        self._last_runtime_pwm = float(pwm)
        next_rpm = float(rpm + delta)
        if return_uncertainty:
            return next_rpm, float(unc)
        return next_rpm

    def ood_amount(self, rpm: float, pwm: float, dt: float) -> float:
        if self.feature_min is None or self.feature_max is None:
            return 0.0
        if not self._runtime_history:
            drpm = 0.0
            dpwm = 0.0
        else:
            drpm = rpm - self._last_runtime_rpm
            dpwm = pwm - self._last_runtime_pwm
        x = self.make_feature(rpm, pwm, drpm, dpwm, dt)
        lo = self.feature_min
        hi = self.feature_max
        span = np.maximum(hi - lo, 1e-6)
        below = np.maximum(0.0, lo - x) / span
        above = np.maximum(0.0, x - hi) / span
        return float(np.mean(below + above))


# ============================================================
# PID CONTROLLER (MÔ PHỎNG GIỐNG FIRMWARE HIỆN TẠI)
# ============================================================

class PIDController:
    """PID velocity controller mô phỏng logic conditional anti-windup của firmware."""

    def __init__(self, kp=1.0, ki=0.0, kd=0.0,
                 output_limit=PWM_LIMIT, integral_limit=INTEGRAL_LIMIT):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.output_limit = float(output_limit)
        self.integral_limit = float(integral_limit)
        self.integral = 0.0
        self.last_error = 0.0
        self.output = 0.0

    def reset(self, initial_error: float = 0.0):
        self.integral = 0.0
        self.last_error = float(initial_error)
        self.output = 0.0

    def compute(self, error: float, dt: float) -> float:
        if dt <= 0.0:
            return self.output
        dt = float(np.clip(dt, 1e-4, 0.2))
        derivative = (error - self.last_error) / dt

        # Giống firmware: xem output với integral hiện tại có đang saturation không.
        before_integral = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )
        saturated_high = before_integral >= self.output_limit
        saturated_low = before_integral <= -self.output_limit

        # Chỉ tích phân nếu không saturation, hoặc error đang kéo output ra khỏi saturation.
        if ((not saturated_high and not saturated_low)
                or (saturated_high and error < 0.0)
                or (saturated_low and error > 0.0)):
            self.integral += error * dt
            self.integral = float(np.clip(
                self.integral, -self.integral_limit, self.integral_limit
            ))

        self.last_error = float(error)
        raw = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.output = float(np.clip(raw, -self.output_limit, self.output_limit))
        return self.output

    def set_gains(self, kp, ki, kd):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)


# ============================================================
# SERIAL DATA COLLECTION
# ============================================================

class DataCollectorThread(QThread):
    data_received = pyqtSignal(dict)
    connection_changed = pyqtSignal(bool, str)
    error_occurred = pyqtSignal(str)

    LINE_RE = re.compile(
        r"rpm\s*=\s*([-+]?\d*\.?\d+)\s+"
        r"set\s*=\s*([-+]?\d*\.?\d+)\s+"
        r"pwm\s*=\s*([-+]?\d*\.?\d+)\s+"
        r"err\s*=\s*([-+]?\d*\.?\d+)",
        re.IGNORECASE,
    )

    def __init__(self):
        super().__init__()
        self.serial_conn = None
        self.running = False
        self.write_lock = threading.Lock()
        self.debug_serial = False

    def connect_to_port(self, port):
        try:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()

            self.serial_conn = serial.Serial(
                port, BAUD_RATE, timeout=SERIAL_TIMEOUT
            )
            time.sleep(2.0)  # ESP32 có thể reset khi mở serial
            self.serial_conn.reset_input_buffer()

            self.running = True
            if not self.isRunning():
                self.start()
            self.connection_changed.emit(True, port)
            return True

        except Exception as e:
            self.error_occurred.emit(str(e))
            return False

    def disconnect(self):
        self.running = False
        try:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
        except Exception:
            pass
        self.serial_conn = None
        self.connection_changed.emit(False, "")

    def send_command(self, cmd):
        if self.serial_conn and self.serial_conn.is_open:
            try:
                with self.write_lock:
                    self.serial_conn.write((cmd.strip() + "\n").encode())
                    self.serial_conn.flush()
            except Exception as e:
                self.error_occurred.emit(f"Write error: {e}")

    def parse_line(self, line: str) -> Optional[dict]:
        m = self.LINE_RE.search(line)
        if not m:
            return None
        rpm, target, pwm, error = map(float, m.groups())
        return {
            "rpm": rpm,
            "target": target,
            "pwm": pwm,
            "error": error,
            "time": time.perf_counter(),
        }

    def run(self):
        while self.running:
            try:
                if not self.serial_conn or not self.serial_conn.is_open:
                    break

                raw = self.serial_conn.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                if self.debug_serial:
                    print(f"[SERIAL] {line}")

                data = self.parse_line(line)
                if data is not None:
                    self.data_received.emit(data)

            except Exception as e:
                if self.running:
                    self.error_occurred.emit(f"Read error: {e}")
                break

        self.running = False


# ============================================================
# V3 TRAINING THREAD: HISTORY + MULTI-STEP ROLLOUT
# ============================================================

class TrainingThread(QThread):
    """V3.1 trainer.

    Model architecture and data semantics are the same as V3.
    V3.1 mainly improves execution:
    - live batch progress without flooding QTextEdit
    - ETA / epoch timing
    - Auto / CPU / CUDA selection
    - CUDA mixed precision (AMP) when available
    - pinned host memory + non-blocking copies on CUDA
    - expensive full rollout validation only every N epochs
    """

    progress = pyqtSignal(int, str)       # persistent log messages
    live_status = pyqtSignal(int, str)    # replace/update one live status label
    finished_training = pyqtSignal(object, dict)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        data,
        model: HybridMotorTCN,
        epochs=400,
        learning_rate=1e-3,
        batch_size=128,
        rollout_horizon=DEFAULT_ROLLOUT_HORIZON,
        device_preference="auto",
        validate_every=VALIDATE_EVERY_EPOCHS,
    ):
        super().__init__()
        self.data = list(data)
        self.model = model
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.rollout_horizon = int(rollout_horizon)
        self.device_preference = str(device_preference).strip().lower()
        self.validate_every = max(1, int(validate_every))
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    @staticmethod
    def _sanitize_samples(data: List[dict]) -> List[dict]:
        clean = []
        for i, d in enumerate(data):
            try:
                rpm = float(d["rpm"])
                pwm = float(d["pwm"])
                t = float(d.get("time", i * NOMINAL_DT))
            except (KeyError, TypeError, ValueError):
                continue
            if not (np.isfinite(rpm) and np.isfinite(pwm) and np.isfinite(t)):
                continue
            clean.append({"rpm": rpm, "pwm": pwm, "time": t})
        return clean

    def _build_segments(self) -> List[dict]:
        clean = self._sanitize_samples(self.data)
        if len(clean) < 4:
            return []

        groups: List[List[dict]] = []
        cur: List[dict] = []
        prev_t = None
        for d in clean:
            t = d["time"]
            new_segment = (
                prev_t is not None
                and (t <= prev_t or (t - prev_t) > MAX_SEQUENCE_GAP_S)
            )
            if new_segment and len(cur) >= 4:
                groups.append(cur)
                cur = []
            cur.append(d)
            prev_t = t
        if len(cur) >= 4:
            groups.append(cur)

        segments = []
        for g in groups:
            t = np.asarray([x["time"] for x in g], dtype=np.float64)
            rpm = np.asarray([x["rpm"] for x in g], dtype=np.float64)
            pwm = np.asarray([x["pwm"] for x in g], dtype=np.float64)
            n = len(rpm)
            if n < 4:
                continue

            dt = np.asarray([
                safe_dt(t[i + 1], t[i]) for i in range(n - 1)
            ], dtype=np.float64)
            rpm_state = rpm[:-1]
            pwm_state = pwm[:-1]
            drpm = np.zeros_like(rpm_state)
            dpwm = np.zeros_like(pwm_state)
            if len(rpm_state) > 1:
                drpm[1:] = np.diff(rpm_state)
                dpwm[1:] = np.diff(pwm_state)
            delta = np.diff(rpm)

            valid = (
                np.isfinite(rpm_state) & np.isfinite(pwm_state)
                & np.isfinite(drpm) & np.isfinite(dpwm)
                & np.isfinite(dt) & np.isfinite(delta)
                & (np.abs(drpm) < 5000.0)
                & (np.abs(delta) < 5000.0)
            )
            # Không xoá điểm ở giữa vì sẽ phá sequence; segment tiếp tại invalid.
            # Dữ liệu parser bình thường sẽ valid toàn bộ.
            if not np.all(valid):
                good_idx = np.where(valid)[0]
                if len(good_idx) < 4:
                    continue
                # giữ đoạn contiguous dài nhất
                runs = np.split(good_idx, np.where(np.diff(good_idx) != 1)[0] + 1)
                run = max(runs, key=len)
                a, b = int(run[0]), int(run[-1] + 1)
                rpm = rpm[a:b + 1]
                pwm = pwm[a:b + 1]
                t = t[a:b + 1]
                dt = dt[a:b]
                rpm_state = rpm[:-1]
                pwm_state = pwm[:-1]
                drpm = np.zeros_like(rpm_state)
                dpwm = np.zeros_like(pwm_state)
                if len(rpm_state) > 1:
                    drpm[1:] = np.diff(rpm_state)
                    dpwm[1:] = np.diff(pwm_state)
                delta = np.diff(rpm)

            features = np.column_stack([rpm_state, pwm_state, drpm, dpwm, dt])
            segments.append({
                "rpm": rpm,
                "pwm": pwm,
                "time": t,
                "dt": dt,
                "rpm_state": rpm_state,
                "pwm_state": pwm_state,
                "features": features,
                "delta": delta,
            })
        return segments

    def _prepare_dataset(self):
        segments = self._build_segments()
        H = self.model.history_len
        K = self.rollout_horizon
        usable = [s for s in segments if len(s["delta"]) >= 2 * H + 2 * K + 10]
        if not usable:
            raise ValueError(
                f"Không có recording đủ dài cho split không-leak với history={H}, rollout={K}. "
                "Hãy thu sequence dài hơn hoặc giảm history/rollout."
            )

        # --------------------------------------------------------
        # Purged chronological split.
        # Train và validation không được dùng các window chồng lên cùng sample.
        # Đây là điểm quan trọng với time-series: random/global split sẽ làm metric
        # đẹp giả vì hai window kề nhau chia sẻ gần như toàn bộ history.
        # --------------------------------------------------------
        cuts = []
        trans_rpm, trans_pwm, trans_dt, trans_delta = [], [], [], []
        for seg in usable:
            T = len(seg["delta"])
            cut = int(0.80 * T)
            # đảm bảo cả train và val còn đủ history + rollout sau purge
            cut = max(H + K + 1, cut)
            cut = min(cut, T - H - K - 1)
            cuts.append(cut)
            trans_rpm.append(seg["rpm_state"][:cut])
            trans_pwm.append(seg["pwm_state"][:cut])
            trans_dt.append(seg["dt"][:cut])
            trans_delta.append(seg["delta"][:cut])

        pr = np.concatenate(trans_rpm)
        pp = np.concatenate(trans_pwm)
        pd = np.concatenate(trans_dt)
        py = np.concatenate(trans_delta)
        self.model.fit_physics_prior(pr, pp, pd, py)

        train_lists = ([], [], [], [], [])
        val_lists = ([], [], [], [], [])

        def append_window(dst, seg, residual, i):
            X_l, y_l, fp_l, fd_l, fr_l = dst
            X_l.append(seg["features"][i - H + 1:i + 1])
            y_l.append([residual[i]])
            fp_l.append(seg["pwm_state"][i:i + K])
            fd_l.append(seg["dt"][i:i + K])
            fr_l.append(seg["rpm"][i + 1:i + K + 1])

        for seg, cut in zip(usable, cuts):
            T = len(seg["delta"])
            residual = seg["delta"] - self.model.physics_delta_numpy(
                seg["rpm_state"], seg["pwm_state"], seg["dt"]
            )

            # Train: cả prediction horizon phải nằm trước cut.
            train_last_i = cut - K
            for i in range(H - 1, train_last_i + 1):
                append_window(train_lists, seg, residual, i)

            # Validation: toàn bộ history phải bắt đầu sau cut.
            # Khoảng giữa bị purge, cố ý không dùng để loại overlap leakage.
            val_first_i = cut + H - 1
            for i in range(val_first_i, T - K + 1):
                append_window(val_lists, seg, residual, i)

        def pack(lists):
            X_l, y_l, fp_l, fd_l, fr_l = lists
            return (
                np.asarray(X_l, dtype=np.float64),
                np.asarray(y_l, dtype=np.float64),
                np.asarray(fp_l, dtype=np.float64),
                np.asarray(fd_l, dtype=np.float64),
                np.asarray(fr_l, dtype=np.float64),
            )

        X_train, y_train, fpwm_train, fdt_train, frpm_train = pack(train_lists)
        X_val, y_val, fpwm_val, fdt_val, frpm_val = pack(val_lists)

        if len(X_train) < 100 or len(X_val) < 30:
            raise ValueError(
                f"Sau purged split chỉ còn train={len(X_train)}, val={len(X_val)} windows. "
                "Cần thu thêm dữ liệu để validation time-series đáng tin cậy."
            )

        self.model.scaler_X.fit(X_train.reshape(-1, self.model.input_size))
        self.model.scaler_y.fit(y_train)
        flat_train = X_train.reshape(-1, self.model.input_size)
        self.model.feature_min = np.min(flat_train, axis=0)
        self.model.feature_max = np.max(flat_train, axis=0)
        self.model.train_rpm_min = float(np.min(flat_train[:, 0]))
        self.model.train_rpm_max = float(np.max(flat_train[:, 0]))
        self.model.train_pwm_min = float(np.min(flat_train[:, 1]))
        self.model.train_pwm_max = float(np.max(flat_train[:, 1]))

        return (
            usable,
            (X_train, y_train, fpwm_train, fdt_train, frpm_train),
            (X_val, y_val, fpwm_val, fdt_val, frpm_val),
        )

    def _rollout_member(self, member: nn.Module, hist_raw: torch.Tensor,
                        future_pwm: torch.Tensor, future_dt: torch.Tensor,
                        x_mean: torch.Tensor, x_scale: torch.Tensor,
                        y_mean: torch.Tensor, y_scale: torch.Tensor) -> torch.Tensor:
        """Differentiable recursive rollout for one member. Returns [B,K]."""
        hist = hist_raw
        preds = []
        K = future_pwm.shape[1]
        for h in range(K):
            hist_s = (hist - x_mean) / x_scale
            res_s = member(hist_s).squeeze(-1)
            residual = res_s * y_scale + y_mean

            rpm = hist[:, -1, 0]
            pwm = hist[:, -1, 1]
            dt = hist[:, -1, 4]
            delta_phys = self.model.physics_delta_torch(rpm, pwm, dt)
            rpm_next = rpm + delta_phys + residual
            preds.append(rpm_next)

            if h + 1 < K:
                pwm_next = future_pwm[:, h + 1]
                dt_next = future_dt[:, h + 1]
                drpm_next = rpm_next - rpm
                dpwm_next = pwm_next - pwm
                feat_next = torch.stack([
                    rpm_next, pwm_next, drpm_next, dpwm_next, dt_next
                ], dim=1)
                hist = torch.cat([hist[:, 1:, :], feat_next.unsqueeze(1)], dim=1)
        return torch.stack(preds, dim=1)

    def _rollout_ensemble(self, hist_raw: torch.Tensor,
                          future_pwm: torch.Tensor, future_dt: torch.Tensor,
                          x_mean: torch.Tensor, x_scale: torch.Tensor,
                          y_mean: torch.Tensor, y_scale: torch.Tensor) -> torch.Tensor:
        """Recursive rollout using ensemble mean, same rule as runtime simulation."""
        hist = hist_raw
        preds = []
        K = future_pwm.shape[1]
        for h in range(K):
            hist_s = (hist - x_mean) / x_scale
            member_s = torch.stack([
                m(hist_s).squeeze(-1) for m in self.model.members
            ], dim=0)
            res_s = torch.mean(member_s, dim=0)
            residual = res_s * y_scale + y_mean
            rpm = hist[:, -1, 0]
            pwm = hist[:, -1, 1]
            dt = hist[:, -1, 4]
            rpm_next = rpm + self.model.physics_delta_torch(rpm, pwm, dt) + residual
            preds.append(rpm_next)
            if h + 1 < K:
                pwm_next = future_pwm[:, h + 1]
                dt_next = future_dt[:, h + 1]
                feat_next = torch.stack([
                    rpm_next,
                    pwm_next,
                    rpm_next - rpm,
                    pwm_next - pwm,
                    dt_next,
                ], dim=1)
                hist = torch.cat([hist[:, 1:, :], feat_next.unsqueeze(1)], dim=1)
        return torch.stack(preds, dim=1)

    def _free_run_validation(self, segments: List[dict], max_steps: int = 4000) -> dict:
        H = self.model.history_len
        true_all, pred_all = [], []
        steps = 0
        self.model.eval()
        for seg in segments:
            T = len(seg["delta"])
            if T < H + 5:
                continue
            # evaluate on final 20% of each recording; seed history with measured data
            start = max(H - 1, int(0.80 * T))
            hist = seg["features"][start - H + 1:start + 1].copy()
            current_rpm = float(seg["rpm_state"][start])

            for j in range(start, T):
                if steps >= max_steps:
                    break
                if j == start:
                    # hist already contains transition j input
                    pass
                else:
                    pwm_j = float(seg["pwm_state"][j])
                    dt_j = float(seg["dt"][j])
                    prev_feat = hist[-1]
                    feat = self.model.make_feature(
                        current_rpm,
                        pwm_j,
                        current_rpm - float(prev_feat[0]),
                        pwm_j - float(prev_feat[1]),
                        dt_j,
                    )
                    hist = np.vstack([hist[1:], feat])

                delta = self.model.predict_delta_from_history(hist)
                next_pred = float(current_rpm + delta)
                true_next = float(seg["rpm"][j + 1])
                pred_all.append(next_pred)
                true_all.append(true_next)
                current_rpm = next_pred
                steps += 1
            if steps >= max_steps:
                break

        if not true_all:
            return {"free_run_mae_rpm": float("nan"), "free_run_rmse_rpm": float("nan"), "free_run_steps": 0}
        yt = np.asarray(true_all)
        yp = np.asarray(pred_all)
        return {
            "free_run_mae_rpm": float(np.mean(np.abs(yp - yt))),
            "free_run_rmse_rpm": float(np.sqrt(np.mean((yp - yt) ** 2))),
            "free_run_steps": int(len(yt)),
        }


    @staticmethod
    def _fmt_seconds(seconds: float) -> str:
        if not np.isfinite(seconds) or seconds < 0:
            return "--"
        seconds = int(round(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h:d}h {m:02d}m {s:02d}s"
        if m:
            return f"{m:d}m {s:02d}s"
        return f"{s:d}s"

    def _select_device(self) -> torch.device:
        pref = self.device_preference
        if pref == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Bạn chọn CUDA nhưng PyTorch hiện không thấy CUDA. "
                    "Hãy chọn Auto/CPU hoặc cài PyTorch CUDA đúng môi trường."
                )
            return torch.device("cuda")
        if pref == "cpu":
            return torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _validate(
        self,
        val_loader,
        device,
        x_mean,
        x_scale,
        y_mean,
        y_scale,
        rpm_scale,
        huber,
        rollout_huber,
        non_blocking: bool,
    ) -> float:
        self.model.eval()
        val_losses = []
        with torch.inference_mode():
            for xb, yb, fpb, fdb, frb in val_loader:
                xb = xb.to(device, non_blocking=non_blocking)
                yb = yb.to(device, non_blocking=non_blocking)
                fpb = fpb.to(device, non_blocking=non_blocking)
                fdb = fdb.to(device, non_blocking=non_blocking)
                frb = frb.to(device, non_blocking=non_blocking)

                hist_s = (xb - x_mean) / x_scale
                pred_members = torch.stack(
                    [m(hist_s) for m in self.model.members], dim=0
                )
                pred_mean = torch.mean(pred_members, dim=0)

                y_scaled = (yb - y_mean) / y_scale
                one = huber(pred_mean, y_scaled)

                roll = self._rollout_ensemble(
                    xb, fpb, fdb, x_mean, x_scale, y_mean, y_scale
                )
                rloss = rollout_huber(
                    roll / rpm_scale,
                    frb / rpm_scale,
                )
                val_losses.append(
                    float((one + ROLLOUT_LOSS_WEIGHT * rloss).item())
                )

        return float(np.mean(val_losses)) if val_losses else float("inf")

    def run(self):
        try:
            if len(self.data) < 300:
                self.error_occurred.emit(
                    "Cần ít nhất ~300 samples; khuyến nghị vài nghìn đến vài chục nghìn samples."
                )
                return

            train_wall_start = time.perf_counter()

            self.progress.emit(3, "Building contiguous time-series segments...")
            segments, train_pack, val_pack = self._prepare_dataset()
            X_train, y_train, fpwm_train, fdt_train, frpm_train = train_pack
            X_val, y_val, fpwm_val, fdt_val, frpm_val = val_pack

            self.progress.emit(
                6,
                f"Train windows={len(X_train)}, val={len(X_val)}, "
                f"history={self.model.history_len}, rollout={self.rollout_horizon}, "
                f"ensemble={self.model.ensemble_size}"
            )
            self.progress.emit(
                7,
                "Physics prior coef=" + np.array2string(
                    self.model.physics_coef, precision=5
                )
            )

            device = self._select_device()

            if device.type == "cpu":
                # PyTorch already parallelizes Conv/Linear. Leave one CPU core
                # for GUI/OS where possible so the app remains responsive.
                try:
                    cpu_total = os.cpu_count() or 1
                    cpu_threads = max(1, cpu_total - CPU_LEAVE_FREE_CORES)
                    torch.set_num_threads(cpu_threads)
                except Exception:
                    cpu_threads = torch.get_num_threads()
                self.progress.emit(
                    8,
                    f"Training device: cpu | torch threads={torch.get_num_threads()}"
                )
            else:
                try:
                    name = torch.cuda.get_device_name(device)
                except Exception:
                    name = "CUDA GPU"
                self.progress.emit(
                    8,
                    f"Training device: cuda | {name} | AMP=ON"
                )
                try:
                    torch.set_float32_matmul_precision("high")
                except Exception:
                    pass

            self.model.to(device)

            pin_memory = device.type == "cuda"
            non_blocking = pin_memory

            # Keep raw tensors because recursive rollout constructs its next
            # history from predicted RPM. Scaling is performed in graph.
            train_ds = TensorDataset(
                torch.from_numpy(X_train.astype(np.float32)),
                torch.from_numpy(y_train.astype(np.float32)),
                torch.from_numpy(fpwm_train.astype(np.float32)),
                torch.from_numpy(fdt_train.astype(np.float32)),
                torch.from_numpy(frpm_train.astype(np.float32)),
            )
            val_ds = TensorDataset(
                torch.from_numpy(X_val.astype(np.float32)),
                torch.from_numpy(y_val.astype(np.float32)),
                torch.from_numpy(fpwm_val.astype(np.float32)),
                torch.from_numpy(fdt_val.astype(np.float32)),
                torch.from_numpy(frpm_val.astype(np.float32)),
            )

            train_loader = DataLoader(
                train_ds,
                batch_size=min(self.batch_size, len(train_ds)),
                shuffle=True,
                drop_last=False,
                pin_memory=pin_memory,
                num_workers=0,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=min(max(self.batch_size, 512), len(val_ds)),
                shuffle=False,
                drop_last=False,
                pin_memory=pin_memory,
                num_workers=0,
            )

            total_train_batches = max(1, len(train_loader))
            live_every = max(
                1,
                total_train_batches // max(1, LIVE_UPDATES_PER_EPOCH)
            )

            x_mean = torch.as_tensor(
                self.model.scaler_X.mean_,
                dtype=torch.float32,
                device=device,
            ).view(1, 1, -1)
            x_scale = torch.as_tensor(
                self.model.scaler_X.scale_,
                dtype=torch.float32,
                device=device,
            ).view(1, 1, -1)
            y_mean = torch.tensor(
                float(self.model.scaler_y.mean_[0]),
                dtype=torch.float32,
                device=device,
            )
            y_scale = torch.tensor(
                float(self.model.scaler_y.scale_[0]),
                dtype=torch.float32,
                device=device,
            )
            rpm_scale = max(float(np.std(frpm_train)), 20.0)

            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=2e-4,
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=4,     # counted in validation events in V3.1
                min_lr=1e-5,
            )
            huber = nn.SmoothL1Loss(beta=0.5)
            rollout_huber = nn.SmoothL1Loss(beta=0.10)

            # CUDA AMP. Keep CPU path pure FP32.
            use_amp = device.type == "cuda"
            grad_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

            best_val = float("inf")
            best_state = None
            no_improve = 0
            patience_validations = 10
            epoch_done = 0
            last_train = float("inf")
            last_val = float("inf")
            epoch_time_ema = None

            self.progress.emit(
                9,
                f"Training started | batches/epoch={total_train_batches} | "
                f"validate every {self.validate_every} epoch(s)"
            )

            for epoch in range(self.epochs):
                if self._stop_requested:
                    self.progress.emit(0, "Training cancelled.")
                    return

                epoch_start = time.perf_counter()
                self.model.train()
                batch_losses = []

                for batch_idx, (xb, yb, fpb, fdb, frb) in enumerate(
                    train_loader, start=1
                ):
                    if self._stop_requested:
                        self.progress.emit(0, "Training cancelled.")
                        return

                    xb = xb.to(device, non_blocking=non_blocking)
                    yb = yb.to(device, non_blocking=non_blocking)
                    fpb = fpb.to(device, non_blocking=non_blocking)
                    fdb = fdb.to(device, non_blocking=non_blocking)
                    frb = frb.to(device, non_blocking=non_blocking)

                    y_scaled = (yb - y_mean) / y_scale
                    optimizer.zero_grad(set_to_none=True)

                    # Compute normalized measured history once per batch.
                    hist_s_base = (xb - x_mean) / x_scale

                    with torch.cuda.amp.autocast(enabled=use_amp):
                        loss_total = torch.zeros(
                            (), dtype=torch.float32, device=device
                        )

                        for member in self.model.members:
                            if INPUT_NOISE_STD > 0:
                                hist_s = (
                                    hist_s_base
                                    + INPUT_NOISE_STD
                                    * torch.randn_like(hist_s_base)
                                )
                            else:
                                hist_s = hist_s_base

                            pred_s = member(hist_s)
                            one_loss = huber(pred_s, y_scaled)

                            pred_roll = self._rollout_member(
                                member,
                                xb,
                                fpb,
                                fdb,
                                x_mean,
                                x_scale,
                                y_mean,
                                y_scale,
                            )
                            roll_loss = rollout_huber(
                                pred_roll / rpm_scale,
                                frb / rpm_scale,
                            )
                            loss_total = (
                                loss_total
                                + one_loss
                                + ROLLOUT_LOSS_WEIGHT * roll_loss
                            )

                        loss_total = (
                            loss_total / self.model.ensemble_size
                        )

                    if use_amp:
                        grad_scaler.scale(loss_total).backward()
                        grad_scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=5.0
                        )
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                    else:
                        loss_total.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=5.0
                        )
                        optimizer.step()

                    batch_losses.append(float(loss_total.detach().item()))

                    # Live batch status: update one QLabel, not the QTextEdit.
                    if (
                        batch_idx == 1
                        or batch_idx == total_train_batches
                        or batch_idx % live_every == 0
                    ):
                        elapsed_epoch = time.perf_counter() - epoch_start
                        sec_per_batch = elapsed_epoch / max(batch_idx, 1)
                        eta_epoch = (
                            total_train_batches - batch_idx
                        ) * sec_per_batch

                        frac = (
                            epoch + batch_idx / total_train_batches
                        ) / max(self.epochs, 1)
                        pct = 10 + int(77 * frac)

                        live_loss = float(
                            np.mean(batch_losses[-min(20, len(batch_losses)):])
                        )

                        self.live_status.emit(
                            pct,
                            f"Epoch {epoch+1}/{self.epochs} | "
                            f"Batch {batch_idx}/{total_train_batches} | "
                            f"loss≈{live_loss:.5f} | "
                            f"ETA epoch {self._fmt_seconds(eta_epoch)}"
                        )

                last_train = (
                    float(np.mean(batch_losses))
                    if batch_losses else float("inf")
                )

                should_validate = (
                    epoch == 0
                    or (epoch + 1) % self.validate_every == 0
                    or (epoch + 1) == self.epochs
                )

                if should_validate:
                    self.live_status.emit(
                        10 + int(77 * (epoch + 1) / self.epochs),
                        f"Epoch {epoch+1}/{self.epochs} | validating..."
                    )

                    last_val = self._validate(
                        val_loader,
                        device,
                        x_mean,
                        x_scale,
                        y_mean,
                        y_scale,
                        rpm_scale,
                        huber,
                        rollout_huber,
                        non_blocking,
                    )
                    scheduler.step(last_val)

                    if last_val < best_val - 1e-7:
                        best_val = last_val
                        best_state = {
                            k: v.detach().cpu().clone()
                            for k, v in self.model.state_dict().items()
                        }
                        no_improve = 0
                    else:
                        no_improve += 1

                epoch_done = epoch + 1
                epoch_elapsed = time.perf_counter() - epoch_start
                if epoch_time_ema is None:
                    epoch_time_ema = epoch_elapsed
                else:
                    epoch_time_ema = (
                        0.85 * epoch_time_ema + 0.15 * epoch_elapsed
                    )

                remaining_epochs = self.epochs - epoch_done
                eta_total = remaining_epochs * epoch_time_ema
                lr = optimizer.param_groups[0]["lr"]
                pct = 10 + int(77 * epoch_done / self.epochs)

                if should_validate:
                    self.progress.emit(
                        pct,
                        f"Epoch {epoch_done}/{self.epochs} | "
                        f"train={last_train:.6f} | val={last_val:.6f} | "
                        f"lr={lr:.2e} | epoch={self._fmt_seconds(epoch_elapsed)} | "
                        f"ETA≈{self._fmt_seconds(eta_total)}"
                    )
                elif epoch_done % 5 == 0:
                    self.progress.emit(
                        pct,
                        f"Epoch {epoch_done}/{self.epochs} | "
                        f"train={last_train:.6f} | val=skipped | "
                        f"lr={lr:.2e} | epoch={self._fmt_seconds(epoch_elapsed)} | "
                        f"ETA≈{self._fmt_seconds(eta_total)}"
                    )

                if (
                    should_validate
                    and no_improve >= patience_validations
                ):
                    self.progress.emit(
                        pct,
                        f"Early stopping at epoch {epoch_done} "
                        f"after {no_improve} validation checks without improvement."
                    )
                    break

            # If training ended before any validation for an unexpected reason,
            # run one now so there is always a best checkpoint.
            if best_state is None:
                last_val = self._validate(
                    val_loader,
                    device,
                    x_mean,
                    x_scale,
                    y_mean,
                    y_scale,
                    rpm_scale,
                    huber,
                    rollout_huber,
                    non_blocking,
                )
                best_val = last_val
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }

            self.model.load_state_dict(best_state)
            self.model.to(device)
            self.model.eval()

            self.live_status.emit(89, "Final validation: one-step + rollout metrics...")

            # Final one-step and K-step validation metrics.
            one_true, one_pred = [], []
            roll_true, roll_pred = [], []
            member_std = []

            with torch.inference_mode():
                for xb, yb, fpb, fdb, frb in val_loader:
                    xb = xb.to(device, non_blocking=non_blocking)
                    fpb = fpb.to(device, non_blocking=non_blocking)
                    fdb = fdb.to(device, non_blocking=non_blocking)

                    hist_s = (xb - x_mean) / x_scale
                    pm = torch.stack(
                        [
                            m(hist_s).squeeze(-1)
                            for m in self.model.members
                        ],
                        dim=0,
                    )
                    pm_raw = pm * y_scale + y_mean
                    resid_mean = torch.mean(pm_raw, dim=0)
                    resid_std = torch.std(
                        pm_raw, dim=0, unbiased=False
                    )

                    rpm_now = xb[:, -1, 0]
                    pwm_now = xb[:, -1, 1]
                    dt_now = xb[:, -1, 4]
                    delta_pred = (
                        self.model.physics_delta_torch(
                            rpm_now, pwm_now, dt_now
                        )
                        + resid_mean
                    )
                    rpm_next_pred = rpm_now + delta_pred
                    rpm_next_true = frb[:, 0].to(
                        device, non_blocking=non_blocking
                    )

                    one_pred.extend(
                        rpm_next_pred.cpu().numpy().tolist()
                    )
                    one_true.extend(
                        rpm_next_true.cpu().numpy().tolist()
                    )
                    member_std.extend(
                        resid_std.cpu().numpy().tolist()
                    )

                    rpred = self._rollout_ensemble(
                        xb,
                        fpb,
                        fdb,
                        x_mean,
                        x_scale,
                        y_mean,
                        y_scale,
                    )
                    roll_pred.extend(
                        rpred.cpu().numpy().reshape(-1).tolist()
                    )
                    roll_true.extend(
                        frb.numpy().reshape(-1).tolist()
                    )

            one_true_a = np.asarray(one_true, dtype=np.float64)
            one_pred_a = np.asarray(one_pred, dtype=np.float64)
            roll_true_a = np.asarray(roll_true, dtype=np.float64)
            roll_pred_a = np.asarray(roll_pred, dtype=np.float64)

            self.model.to("cpu")
            self.model.eval()
            self.model.is_trained = True

            one_mae = float(
                np.mean(np.abs(one_pred_a - one_true_a))
            )
            one_rmse = float(
                np.sqrt(np.mean((one_pred_a - one_true_a) ** 2))
            )
            one_r2 = float(
                robust_r2(one_true_a, one_pred_a)
            )
            roll_mae = float(
                np.mean(np.abs(roll_pred_a - roll_true_a))
            )
            roll_rmse = float(
                np.sqrt(np.mean((roll_pred_a - roll_true_a) ** 2))
            )
            unc_mean = (
                float(np.mean(member_std))
                if member_std else 0.0
            )

            self.progress.emit(
                92, "Running recursive free-run validation..."
            )
            self.live_status.emit(
                92, "Running recursive free-run validation..."
            )
            free_metrics = self._free_run_validation(segments)

            total_wall = time.perf_counter() - train_wall_start

            self.model.val_metrics = {
                "one_step_mae_rpm": one_mae,
                "one_step_rmse_rpm": one_rmse,
                "one_step_r2_rpm": one_r2,
                "rollout_horizon_steps": self.rollout_horizon,
                "rollout_mae_rpm": roll_mae,
                "rollout_rmse_rpm": roll_rmse,
                "ensemble_residual_std_mean_rpm": unc_mean,
                **free_metrics,
            }

            stats = {
                "trainer_version": "3.1",
                "epochs": epoch_done,
                "train_loss_composite": last_train,
                "best_val_loss_composite": best_val,
                "train_windows": int(len(X_train)),
                "val_windows": int(len(X_val)),
                "history_len": int(self.model.history_len),
                "rollout_horizon": int(self.rollout_horizon),
                "ensemble_size": int(self.model.ensemble_size),
                "training_device": str(device),
                "validate_every_epochs": int(self.validate_every),
                "total_training_time_s": float(total_wall),
                "train_rpm_range": [
                    self.model.train_rpm_min,
                    self.model.train_rpm_max,
                ],
                "train_pwm_range": [
                    self.model.train_pwm_min,
                    self.model.train_pwm_max,
                ],
                "physics_coef": self.model.physics_coef.tolist(),
                **self.model.val_metrics,
            }

            self.progress.emit(
                100,
                f"V3.1 Hybrid TCN training complete | "
                f"total={self._fmt_seconds(total_wall)}"
            )
            self.live_status.emit(
                100, "DONE - V3.1 training complete."
            )
            self.finished_training.emit(self.model, stats)

        except Exception as e:
            try:
                self.model.to("cpu")
            except Exception:
                pass
            self.error_occurred.emit(f"Training error: {e}")


# ============================================================
# PID OPTIMIZATION
# ============================================================

@dataclass
class ResponseMetrics:
    iae: float
    itae: float
    overshoot_pct: float
    rise_time_s: float
    settling_time_s: float
    steady_state_error_pct: float
    control_effort: float
    control_variation: float
    saturation_fraction: float
    ood_mean: float
    score: float


class PIDOptimizer:
    """Tối ưu PID trên learned motor model bằng Differential Evolution."""

    def __init__(self, model: HybridMotorTCN, target_rpm=100.0,
                 dt=NOMINAL_DT, duration=4.0,
                 kp_bounds=(0.0, 5.0), ki_bounds=(0.0, 10.0), kd_bounds=(0.0, 2.0),
                 both_directions=True):
        self.model = model
        self.target_rpm = float(target_rpm)
        self.dt = float(dt)
        self.duration = float(duration)
        self.bounds = np.asarray([kp_bounds, ki_bounds, kd_bounds], dtype=np.float64)
        self.rng = np.random.default_rng(RANDOM_SEED)
        self.both_directions = bool(both_directions)

    def simulate_pid(self, kp, ki, kd, target=None, initial_rpm=0.0):
        target = self.target_rpm if target is None else float(target)
        pid = PIDController(kp, ki, kd)
        rpm = float(initial_rpm)
        initial_error = target - rpm
        pid.reset(initial_error=initial_error)
        self.model.reset_runtime(rpm=initial_rpm, pwm=0.0, dt=self.dt)

        n = max(2, int(self.duration / self.dt))
        rows = []

        for k in range(n):
            error = target - rpm
            pwm = pid.compute(error, self.dt)
            ood = self.model.ood_amount(rpm, pwm, self.dt)
            rpm_next, uncertainty = self.model.predict_next_rpm(
                rpm, pwm, self.dt, return_uncertainty=True
            )

            # Ensemble disagreement được cộng nhẹ vào OOD để optimizer tránh vùng
            # model không chắc chắn dù feature vẫn nằm gần min/max training.
            ood += min(1.0, uncertainty / max(abs(target), 10.0))

            if not np.isfinite(rpm_next) or abs(rpm_next) > 1e6:
                return [], True

            rows.append({
                "t": k * self.dt,
                "rpm": rpm,
                "target": target,
                "pwm": pwm,
                "error": error,
                "ood": ood,
                "uncertainty": uncertainty,
            })
            rpm = float(rpm_next)

        return rows, False

    def calculate_metrics(self, data, target) -> ResponseMetrics:
        if not data or abs(target) < 1e-9:
            return ResponseMetrics(*(float("inf"),) * 11)

        rpm = np.asarray([d["rpm"] for d in data], dtype=np.float64)
        pwm = np.asarray([d["pwm"] for d in data], dtype=np.float64)
        err = target - rpm
        t = np.arange(len(rpm), dtype=np.float64) * self.dt
        scale = max(abs(target), 1.0)

        iae = float(np.sum(np.abs(err)) * self.dt / scale)
        itae = float(np.sum(t * np.abs(err)) * self.dt / (scale * max(self.duration, self.dt)))

        if target > 0:
            overshoot = max(0.0, float(np.max(rpm) - target)) / scale * 100.0
        else:
            overshoot = max(0.0, float(target - np.min(rpm))) / scale * 100.0

        # rise time 10 -> 90%, theo chiều target
        sgn = 1.0 if target >= 0 else -1.0
        prog = sgn * rpm
        tgt = abs(target)
        idx10 = np.where(prog >= 0.10 * tgt)[0]
        idx90 = np.where(prog >= 0.90 * tgt)[0]
        if len(idx10) and len(idx90) and idx90[0] >= idx10[0]:
            rise_time = float((idx90[0] - idx10[0]) * self.dt)
        else:
            rise_time = self.duration * 1.5

        # Settling time đúng nghĩa: từ thời điểm i trở đi TẤT CẢ mẫu
        # đều nằm trong dải xác lập. Dùng ±2% để tiêu chí chặt hơn.
        band = SETTLING_BAND_FRAC * scale
        inside = np.abs(err) <= band
        suffix_all_inside = np.logical_and.accumulate(inside[::-1])[::-1]
        idx_settle = np.where(suffix_all_inside)[0]
        settling_time = float(idx_settle[0] * self.dt) if len(idx_settle) else self.duration * 1.5

        tail_n = max(10, int(0.20 * len(rpm)))
        ss = float(np.mean(rpm[-tail_n:]))
        ss_error_pct = abs(target - ss) / scale * 100.0

        control_effort = float(np.sum(np.abs(pwm)) * self.dt / (PWM_LIMIT * self.duration))
        if len(pwm) > 1:
            control_variation = float(np.mean(np.abs(np.diff(pwm))) / PWM_LIMIT)
        else:
            control_variation = 0.0
        saturation_fraction = float(np.mean(np.abs(pwm) >= 0.98 * PWM_LIMIT))
        ood_mean = float(np.mean([d["ood"] for d in data]))

        # ========================================================
        # SCORE PRIORITY:
        #   1) Overshoot rất thấp
        #   2) Settling time ngắn (dải ±2%)
        #   3) Rise time nhanh
        #   4) Vẫn giữ tracking, PWM và độ ổn định hợp lý
        # ========================================================

        overshoot_excess = max(0.0, overshoot - OVERSHOOT_LIMIT_PCT)

        score = (
            1.5 * iae +
            1.0 * itae +
            0.90 * overshoot +
            0.8 * rise_time +
            8.0 * settling_time +
            0.20 * ss_error_pct +
            0.5 * control_effort +
            1.0 * control_variation +
            2.0 * saturation_fraction +
            30.0 * ood_mean
        )

        # Overshoot > 1% bị phạt mạnh.
        score += (
            25.0 * overshoot_excess +
            10.0 * overshoot_excess * overshoot_excess
        )

        # Mức phạt mạnh nếu phản hồi chạy quá xa setpoint.
        if np.max(np.abs(rpm)) > 3.0 * scale + 50.0:
            score += 1000.0

        return ResponseMetrics(
            iae=iae,
            itae=itae,
            overshoot_pct=overshoot,
            rise_time_s=rise_time,
            settling_time_s=settling_time,
            steady_state_error_pct=ss_error_pct,
            control_effort=control_effort,
            control_variation=control_variation,
            saturation_fraction=saturation_fraction,
            ood_mean=ood_mean,
            score=float(score),
        )

    def evaluate_candidate(self, gains: np.ndarray):
        """Evaluate one PID on several magnitudes and, by default, both directions.

        V4 only evaluated the sign of target_rpm in the DE path. Real motor/driver
        dynamics are not perfectly symmetric, and the collected dataset contains
        both positive and negative operation. V4.1 therefore scores the same PID
        on +/- {0.6, 1.0, 1.2} * |target|.
        """
        kp, ki, kd = map(float, gains)

        mag = max(5.0, abs(self.target_rpm))
        mags = [
            max(5.0, 0.60 * mag),
            max(5.0, 1.00 * mag),
            max(5.0, 1.20 * mag),
        ]

        main_sign = -1.0 if self.target_rpm < 0 else 1.0
        targets = [main_sign * x for x in mags]

        if self.both_directions:
            targets += [-main_sign * x for x in mags]

        metrics = []
        metrics_by_target = {}

        for target in targets:
            data, unstable = self.simulate_pid(
                kp, ki, kd, target=target
            )
            if unstable:
                return 1e9, None

            m = self.calculate_metrics(data, target)
            if not np.isfinite(m.score):
                return 1e9, None

            metrics.append(m)
            metrics_by_target[float(target)] = m

        scores = np.asarray(
            [m.score for m in metrics], dtype=np.float64
        )

        # Mean performance plus a worst-case term.
        robust_score = float(
            np.mean(scores) + 0.35 * np.max(scores)
        )

        # Report metric at the user's requested main target.
        report_target = main_sign * mags[1]
        report_metric = metrics_by_target.get(
            float(report_target), metrics[1]
        )

        return robust_score, report_metric

    def optimize(self, max_evaluations=240, callback=None, stop_fn=None, extra_seeds=None):
        """
        Differential Evolution/rand/1/bin.
        max_evaluations là budget số candidate evaluations gần đúng.
        """
        dim = 3
        pop_size = min(30, max(12, int(math.sqrt(max_evaluations) * 1.5)))
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]

        # Latin-hypercube-like initialization để phủ search space tốt hơn random thuần.
        population = np.empty((pop_size, dim), dtype=np.float64)
        for d in range(dim):
            bins = (np.arange(pop_size) + self.rng.random(pop_size)) / pop_size
            self.rng.shuffle(bins)
            population[:, d] = lo[d] + bins * (hi[d] - lo[d])

        scores = np.empty(pop_size, dtype=np.float64)
        metrics_cache = [None] * pop_size
        evaluations = 0

        # Luôn chèn vài seed PID có ý nghĩa.
        seeds = []
        if extra_seeds is not None:
            for seed in extra_seeds:
                seed = np.asarray(seed, dtype=np.float64).reshape(3)
                seeds.append(seed)
        seeds.extend([
            np.array([1.0, 0.0, 0.0]),
            np.array([0.6, 0.0, 0.0]),
            np.array([0.5, min(1.0, hi[1]), 0.0]),
        ])
        for i, seed in enumerate(seeds[:pop_size]):
            population[i] = np.clip(seed, lo, hi)

        for i in range(pop_size):
            if stop_fn and stop_fn():
                raise RuntimeError("Optimization cancelled")
            scores[i], metrics_cache[i] = self.evaluate_candidate(population[i])
            evaluations += 1

        best_idx = int(np.argmin(scores))
        best = population[best_idx].copy()
        best_score = float(scores[best_idx])
        best_metrics = metrics_cache[best_idx]

        if callback:
            callback(evaluations, max_evaluations, best, best_score, best_metrics)

        F = 0.75
        CR = 0.85

        while evaluations < max_evaluations:
            for i in range(pop_size):
                if evaluations >= max_evaluations:
                    break
                if stop_fn and stop_fn():
                    raise RuntimeError("Optimization cancelled")

                pool = [j for j in range(pop_size) if j != i]
                a, b, c = self.rng.choice(pool, size=3, replace=False)
                mutant = population[a] + F * (population[b] - population[c])
                mutant = np.clip(mutant, lo, hi)

                cross = self.rng.random(dim) < CR
                cross[self.rng.integers(0, dim)] = True
                trial = np.where(cross, mutant, population[i])

                trial_score, trial_metrics = self.evaluate_candidate(trial)
                evaluations += 1

                if trial_score < scores[i]:
                    population[i] = trial
                    scores[i] = trial_score
                    metrics_cache[i] = trial_metrics

                    if trial_score < best_score:
                        best = trial.copy()
                        best_score = float(trial_score)
                        best_metrics = trial_metrics

                if callback and (evaluations % 5 == 0 or evaluations >= max_evaluations):
                    callback(evaluations, max_evaluations, best, best_score, best_metrics)

        return float(best[0]), float(best[1]), float(best[2]), best_score, best_metrics


# ============================================================
# V4 ADAPTIVE AI PID TUNER
# ============================================================

class PIDGainNet(nn.Module):
    """
    Network thứ hai: diagnostic sequence -> Kp, Ki, Kd.

    Input mỗi sample của diagnostic sequence (đã normalize):
        [target, rpm, pwm, error, drpm]

    Output được ép vào PID bounds bằng sigmoid, vì vậy AI không thể sinh gain
    ra ngoài giới hạn do người dùng đặt.
    """

    def __init__(self, kp_bounds=(0.0, 5.0), ki_bounds=(0.0, 10.0),
                 kd_bounds=(0.0, 2.0), channels=64, dropout=0.05):
        super().__init__()
        self.bounds = np.asarray([kp_bounds, ki_bounds, kd_bounds], dtype=np.float64)
        self.channels = int(channels)
        self.dropout = float(dropout)
        self.encoder = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(32, self.channels, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv1d(self.channels, self.channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Dropout(self.dropout),
        )
        # mean-pool + max-pool + 2 target context features
        self.head = nn.Sequential(
            nn.Linear(2 * self.channels + 2, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )
        self.is_trained = False
        self.training_meta: Dict[str, float] = {}

    @property
    def model_config(self):
        return {
            "kp_bounds": tuple(self.bounds[0].tolist()),
            "ki_bounds": tuple(self.bounds[1].tolist()),
            "kd_bounds": tuple(self.bounds[2].tolist()),
            "channels": self.channels,
            "dropout": self.dropout,
        }

    def _bounds_torch(self, device, dtype):
        b = torch.as_tensor(self.bounds, dtype=dtype, device=device)
        return b[:, 0], b[:, 1]

    def initialize_from_gains(self, gains):
        """Khởi tạo output gần PID đang chạy để meta-training bắt đầu ở vùng an toàn hơn."""
        g = np.asarray(gains, dtype=np.float64).reshape(3)
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        p = np.clip((g - lo) / np.maximum(hi - lo, 1e-9), 1e-4, 1.0 - 1e-4)
        logits = np.log(p / (1.0 - p))
        last = self.head[-1]
        with torch.no_grad():
            last.weight.zero_()
            last.bias.copy_(torch.as_tensor(logits, dtype=last.bias.dtype))

    def forward(self, seq_btf: torch.Tensor, target_rpm: torch.Tensor) -> torch.Tensor:
        # seq [B,T,5]
        x = seq_btf.transpose(1, 2)
        z = self.encoder(x)
        z_mean = torch.mean(z, dim=-1)
        z_max = torch.amax(z, dim=-1)

        tr = target_rpm.reshape(-1)
        target_mag = torch.log1p(torch.abs(tr)) / math.log1p(500.0)
        target_sign = torch.sign(tr)
        ctx = torch.stack([target_mag, target_sign], dim=-1)
        h = torch.cat([z_mean, z_max, ctx], dim=-1)
        logits = self.head(h)
        lo, hi = self._bounds_torch(logits.device, logits.dtype)
        return lo + torch.sigmoid(logits) * (hi - lo)

    def predict_gains(self, seq_btf: torch.Tensor, target_rpm: float):
        self.eval()
        device = next(self.parameters()).device
        seq = seq_btf.to(device=device, dtype=torch.float32)
        target = torch.full((seq.shape[0],), float(target_rpm), device=device)
        with torch.inference_mode():
            gains = self(seq, target).cpu().numpy()
        return gains


def _sample_plant_params(batch: int, device: torch.device, severity: float,
                         rng: Optional[np.random.Generator] = None):
    """
    Domain randomization quanh learned plant hiện tại.
    Không khẳng định đây là tham số vật lý trực tiếp; chúng mô phỏng thay đổi tổng hợp
    do nguồn/driver, ma sát, tải, dead-zone và delay sau thời gian sử dụng.
    """
    rng = np.random.default_rng() if rng is None else rng
    s = float(np.clip(severity, 0.0, 0.60))
    def unif(a, b):
        return torch.as_tensor(rng.uniform(a, b, size=batch), dtype=torch.float32, device=device)

    return {
        "actuation_scale": unif(1.0 - s, 1.0 + s),
        "response_scale": unif(1.0 - 0.8 * s, 1.0 + 0.8 * s),
        "drag_rate": unif(0.0, 1.20 * s),       # additional 1/s damping
        "deadzone": unif(0.0, 40.0 * s),       # PWM counts
        "delay_steps": torch.as_tensor(
            rng.integers(0, max(1, int(round(1 + 8 * s))) + 1, size=batch),
            dtype=torch.long, device=device,
        ),
    }


def _nominal_plant_params(batch: int, device: torch.device):
    return {
        "actuation_scale": torch.ones(batch, device=device),
        "response_scale": torch.ones(batch, device=device),
        "drag_rate": torch.zeros(batch, device=device),
        "deadzone": torch.zeros(batch, device=device),
        "delay_steps": torch.zeros(batch, dtype=torch.long, device=device),
    }


class DifferentiablePlantBatch:
    """
    Vectorized differentiable wrapper quanh HybridMotorTCN.

    Nó KHÔNG sửa weights của plant model. Mục tiêu là cho gradient từ closed-loop
    performance đi xuyên qua plant tới PIDGainNet trong meta-training.
    """

    def __init__(self, model: HybridMotorTCN, params: dict, batch: int,
                 device: torch.device, dt: float = NOMINAL_DT):
        self.model = model
        self.params = params
        self.batch = int(batch)
        self.device = device
        self.dt = float(dt)
        self.history_len = model.history_len

        self.x_mean = torch.as_tensor(
            model.scaler_X.mean_, dtype=torch.float32, device=device
        ).view(1, 1, -1)
        self.x_scale = torch.as_tensor(
            model.scaler_X.scale_, dtype=torch.float32, device=device
        ).view(1, 1, -1)
        self.y_mean = torch.tensor(float(model.scaler_y.mean_[0]), device=device)
        self.y_scale = torch.tensor(float(model.scaler_y.scale_[0]), device=device)

        max_delay = int(torch.max(params["delay_steps"]).item()) if self.batch else 0
        self.max_delay = max(0, max_delay)
        self.reset()

    def reset(self, rpm0: Optional[torch.Tensor] = None):
        if rpm0 is None:
            rpm0 = torch.zeros(self.batch, dtype=torch.float32, device=self.device)
        rpm0 = rpm0.reshape(self.batch)
        z = torch.zeros_like(rpm0)
        dt = torch.full_like(rpm0, self.dt)
        feat = torch.stack([rpm0, z, z, z, dt], dim=-1)
        self.history = feat[:, None, :].repeat(1, self.history_len, 1)
        self.last_rpm = rpm0
        self.last_pwm = z
        self.delay_queue = torch.zeros(
            self.batch, self.max_delay + 1, dtype=torch.float32, device=self.device
        )

    def step(self, rpm: torch.Tensor, pwm_cmd: torch.Tensor):
        rpm = rpm.reshape(self.batch)
        pwm_cmd = pwm_cmd.reshape(self.batch)

        # command delay: delay=0 dùng sample mới nhất.
        self.delay_queue = torch.cat(
            [self.delay_queue[:, 1:], pwm_cmd[:, None]], dim=1
        ) if self.delay_queue.shape[1] > 1 else pwm_cmd[:, None]
        gather_idx = self.max_delay - torch.clamp(
            self.params["delay_steps"], 0, self.max_delay
        )
        delayed = self.delay_queue.gather(1, gather_idx[:, None]).squeeze(1)

        # H-bridge/motor dead-zone + effective actuation strength.
        mag = torch.relu(torch.abs(delayed) - self.params["deadzone"])
        pwm_eff = torch.sign(delayed) * mag * self.params["actuation_scale"]
        pwm_eff = torch.clamp(pwm_eff, -PWM_LIMIT, PWM_LIMIT)

        drpm = rpm - self.last_rpm
        dpwm = pwm_eff - self.last_pwm
        dtv = torch.full_like(rpm, self.dt)
        feat = torch.stack([rpm, pwm_eff, drpm, dpwm, dtv], dim=-1)
        self.history = torch.cat([self.history[:, 1:, :], feat[:, None, :]], dim=1)

        hist_s = (self.history - self.x_mean) / self.x_scale
        residual_scaled = torch.stack(
            [member(hist_s).squeeze(-1) for member in self.model.members], dim=0
        )
        residual = torch.mean(residual_scaled, dim=0) * self.y_scale + self.y_mean
        uncertainty = torch.std(
            residual_scaled * self.y_scale + self.y_mean,
            dim=0, unbiased=False,
        )
        physics = self.model.physics_delta_torch(rpm, pwm_eff, dtv)
        delta = self.params["response_scale"] * (physics + residual)
        delta = delta - self.params["drag_rate"] * rpm * self.dt

        next_rpm = rpm + delta
        # guard against numerical explosion during meta-training.
        next_rpm = torch.clamp(next_rpm, -5000.0, 5000.0)
        self.last_rpm = rpm
        self.last_pwm = pwm_eff
        return next_rpm, uncertainty


def _piecewise_schedule(target_rpm: float, fractions, segment_steps: int,
                        batch: int, device: torch.device):
    vals = []
    mag = abs(float(target_rpm))
    sign = 1.0 if target_rpm >= 0 else -1.0
    for f in fractions:
        vals.extend([sign * mag * float(f)] * int(segment_steps))
    seq = torch.as_tensor(vals, dtype=torch.float32, device=device)
    return seq[:, None].repeat(1, batch)  # [T,B]


def _run_pid_sequence(plant: DifferentiablePlantBatch, gains: torch.Tensor,
                      target_seq_tb: torch.Tensor, keep_graph: bool = True):
    """Differentiable PID + plant rollout. Return normalized diagnostic raw arrays."""
    B = plant.batch
    device = plant.device
    rpm = torch.zeros(B, device=device)
    integral = torch.zeros(B, device=device)
    first_target = target_seq_tb[0]
    last_error = first_target - rpm       # derivative kick protection
    prev_pwm = torch.zeros(B, device=device)
    prev_rpm_out = rpm

    rows = []
    for target in target_seq_tb:
        error = target - rpm
        derivative = (error - last_error) / plant.dt

        raw_before = (
            gains[:, 0] * error
            + gains[:, 1] * integral
            + gains[:, 2] * derivative
        )
        sat_hi = raw_before >= PWM_LIMIT
        sat_lo = raw_before <= -PWM_LIMIT
        can_integrate = (~sat_hi & ~sat_lo) | (sat_hi & (error < 0)) | (sat_lo & (error > 0))
        int_candidate = torch.clamp(
            integral + error * plant.dt, -INTEGRAL_LIMIT, INTEGRAL_LIMIT
        )
        integral = torch.where(can_integrate, int_candidate, integral)

        raw = (
            gains[:, 0] * error
            + gains[:, 1] * integral
            + gains[:, 2] * derivative
        )
        pwm = torch.clamp(raw, -PWM_LIMIT, PWM_LIMIT)
        rpm_next, uncertainty = plant.step(rpm, pwm)
        drpm = rpm - prev_rpm_out
        rows.append((target, rpm, pwm, error, drpm, uncertainty))

        prev_rpm_out = rpm
        rpm = rpm_next
        prev_pwm = pwm
        last_error = error

        # Truncated BPTT: giữ gradient qua một cửa sổ động học đủ dài, sau đó
        # detach state để meta-training không giữ toàn bộ graph hàng trăm bước.
        if keep_graph and V4_BPTT_STEPS > 0 and (len(rows) % V4_BPTT_STEPS == 0):
            rpm = rpm.detach()
            integral = integral.detach()
            last_error = last_error.detach()
            prev_rpm_out = prev_rpm_out.detach()
            prev_pwm = prev_pwm.detach()
            plant.history = plant.history.detach()
            plant.last_rpm = plant.last_rpm.detach()
            plant.last_pwm = plant.last_pwm.detach()
            plant.delay_queue = plant.delay_queue.detach()

    # stack each variable to [B,T]
    stacked = [torch.stack([r[i] for r in rows], dim=1) for i in range(6)]
    return stacked


def _diagnostic_features(model: HybridMotorTCN, params: dict, baseline_gains,
                         target_rpm: float, batch: int, device: torch.device):
    plant = DifferentiablePlantBatch(model, params, batch, device)
    baseline = torch.as_tensor(
        np.asarray(baseline_gains, dtype=np.float32), device=device
    ).view(1, 3).repeat(batch, 1)
    schedule = _piecewise_schedule(
        target_rpm,
        fractions=[0.45, 1.0, 0.25, -0.45, -1.0, 0.0],
        segment_steps=V4_DIAG_SEGMENT_STEPS,
        batch=batch,
        device=device,
    )
    target, rpm, pwm, error, drpm, _ = _run_pid_sequence(
        plant, baseline, schedule, keep_graph=False
    )
    scale = max(abs(float(target_rpm)), 10.0)
    seq = torch.stack([
        target / scale,
        rpm / scale,
        pwm / PWM_LIMIT,
        error / scale,
        drpm / scale,
    ], dim=-1)  # [B,T,5]
    return torch.clamp(seq, -10.0, 10.0)


def _closed_loop_objective(model: HybridMotorTCN, params: dict, gains: torch.Tensor,
                           target_rpm: float, batch: int, device: torch.device):
    plant = DifferentiablePlantBatch(model, params, batch, device)
    schedule = _piecewise_schedule(
        target_rpm,
        fractions=[0.60, 1.00, 0.0, -0.60, -1.00, 0.0],
        segment_steps=V4_EVAL_SEGMENT_STEPS,
        batch=batch,
        device=device,
    )
    target, rpm, pwm, error, drpm, uncertainty = _run_pid_sequence(
        plant, gains, schedule
    )
    scale = max(abs(float(target_rpm)), 10.0)
    abs_err = torch.abs(error) / scale

    # Time weight resets in every segment, approximating ITAE after each command change.
    T = error.shape[1]
    phase = torch.arange(T, device=device) % V4_EVAL_SEGMENT_STEPS
    phase = phase.float() / max(V4_EVAL_SEGMENT_STEPS - 1, 1)
    itae = torch.mean(abs_err * phase[None, :], dim=1)
    iae = torch.mean(abs_err, dim=1)

    nonzero = (torch.abs(target) > 1e-6).float()
    direction = torch.sign(target)
    progress = direction * rpm
    overshoot = torch.relu(progress - torch.abs(target)) / scale * nonzero
    overshoot_pen = torch.mean(overshoot ** 2, dim=1)

    # Tail tracking: last 25% of every segment should sit close to target.
    idx = torch.arange(T, device=device)
    pos_in_seg = idx % V4_EVAL_SEGMENT_STEPS
    tail_mask = (pos_in_seg >= int(0.75 * V4_EVAL_SEGMENT_STEPS)).float()[None, :]
    tail_err = torch.sum(abs_err * tail_mask, dim=1) / torch.sum(tail_mask)

    effort = torch.mean(torch.abs(pwm) / PWM_LIMIT, dim=1)
    if T > 1:
        variation = torch.mean(torch.abs(pwm[:, 1:] - pwm[:, :-1]) / PWM_LIMIT, dim=1)
    else:
        variation = torch.zeros_like(effort)
    sat = torch.mean(torch.relu(torch.abs(pwm) / PWM_LIMIT - 0.95) ** 2, dim=1)
    unstable = torch.mean(
        torch.relu(torch.abs(rpm) / (3.0 * scale + 50.0) - 1.0) ** 2,
        dim=1,
    )
    unc = torch.mean(uncertainty / scale, dim=1)

    # Composite differentiable proxy matching V3 priorities.
    per_case = (
        2.0 * iae
        + 1.0 * itae
        + 10.0 * overshoot_pen
        + 5.0 * tail_err
        + 0.15 * effort
        + 0.50 * variation
        + 4.0 * sat
        + 15.0 * unstable
        + 0.10 * unc
    )
    return torch.mean(per_case), {
        "iae": torch.mean(iae).detach(),
        "itae": torch.mean(itae).detach(),
        "tail": torch.mean(tail_err).detach(),
        "overshoot_proxy": torch.mean(overshoot_pen).detach(),
        "effort": torch.mean(effort).detach(),
        "variation": torch.mean(variation).detach(),
        "saturation_proxy": torch.mean(sat).detach(),
    }


class MetaTunerTrainingThread(QThread):
    progress = pyqtSignal(int, str)
    live_status = pyqtSignal(int, str)
    finished_training = pyqtSignal(object, dict)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        plant_model: HybridMotorTCN,
        tuner: PIDGainNet,
        baseline_gains,
        target_rpm=100.0,
        epochs=V4_TUNER_EPOCHS,
        batch_size=V4_TUNER_BATCH,
        learning_rate=V4_TUNER_LR,
        random_severity=V4_RANDOM_SEVERITY,
        device_preference="auto",
        validate_every=V4_AI_VALIDATE_EVERY_EPOCHS,
    ):
        super().__init__()
        self.plant_model = plant_model
        self.tuner = tuner
        self.baseline_gains = tuple(map(float, baseline_gains))
        self.target_rpm = float(target_rpm)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.random_severity = float(random_severity)
        self.device_preference = str(device_preference).strip().lower()
        self.validate_every = max(1, int(validate_every))
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    @staticmethod
    def _fmt_seconds(seconds: float) -> str:
        if not np.isfinite(seconds) or seconds < 0:
            return "--"
        seconds = int(round(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h:d}h {m:02d}m {s:02d}s"
        if m:
            return f"{m:d}m {s:02d}s"
        return f"{s:d}s"

    def _select_device(self) -> torch.device:
        pref = self.device_preference
        if pref == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Bạn chọn CUDA nhưng PyTorch hiện không thấy CUDA. "
                    "Hãy cài/activate đúng PyTorch CUDA hoặc chọn Auto/CPU."
                )
            return torch.device("cuda")
        if pref == "cpu":
            return torch.device("cpu")
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def run(self):
        device = torch.device("cpu")
        old_req = None

        try:
            device = self._select_device()
            if not self.plant_model.is_trained:
                raise RuntimeError("Plant model chưa train/load.")
            if abs(self.target_rpm) < 1e-6:
                raise RuntimeError("Target RPM phải khác 0.")

            wall_start = time.perf_counter()

            if device.type == "cuda":
                gpu_name = torch.cuda.get_device_name(device)
                self.progress.emit(
                    1,
                    f"V4.2 AI tuner meta-training on CUDA | {gpu_name} | FP32"
                )
            else:
                self.progress.emit(
                    1,
                    f"V4.2 AI tuner meta-training on CPU | "
                    f"threads={torch.get_num_threads()}"
                )

            self.progress.emit(
                2,
                "Objective schedule contains both +RPM and -RPM commands; "
                "local/global DE also verify both directions."
            )

            self.plant_model.to(device).eval()

            # Freeze plant weights while preserving gradients through plant
            # operations with respect to PIDGainNet output/action trajectory.
            old_req = [
                p.requires_grad
                for p in self.plant_model.parameters()
            ]
            for p in self.plant_model.parameters():
                p.requires_grad_(False)

            self.tuner.to(device)
            self.tuner.initialize_from_gains(
                self.baseline_gains
            )

            opt = optim.AdamW(
                self.tuner.parameters(),
                lr=self.learning_rate,
                weight_decay=1e-4,
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode="min",
                factor=0.5,
                patience=4,
                min_lr=1e-5,
            )

            # Fixed validation plants make validation comparable over time.
            val_rng = np.random.default_rng(
                RANDOM_SEED + 1001
            )
            val_batch = max(8, self.batch_size)
            val_params = _sample_plant_params(
                val_batch,
                device,
                self.random_severity,
                val_rng,
            )

            best_val = float("inf")
            best_state = None
            no_improve = 0
            patience_validations = 10
            train_loss_value = float("inf")
            val_loss_value = float("inf")
            epoch_done = 0
            epoch_time_ema = None

            train_rng = np.random.default_rng(
                RANDOM_SEED + 2026
            )

            for epoch in range(self.epochs):
                if self._stop_requested:
                    raise RuntimeError(
                        "AI tuner training cancelled"
                    )

                epoch_start = time.perf_counter()

                params = _sample_plant_params(
                    self.batch_size,
                    device,
                    self.random_severity,
                    train_rng,
                )

                # Diagnostic is treated as observed plant fingerprint.
                with torch.no_grad():
                    diag = _diagnostic_features(
                        self.plant_model,
                        params,
                        self.baseline_gains,
                        self.target_rpm,
                        self.batch_size,
                        device,
                    )

                self.tuner.train()

                target_ctx = torch.full(
                    (self.batch_size,),
                    self.target_rpm,
                    dtype=torch.float32,
                    device=device,
                )

                gains = self.tuner(
                    diag, target_ctx
                )

                loss, pieces = _closed_loop_objective(
                    self.plant_model,
                    params,
                    gains,
                    self.target_rpm,
                    self.batch_size,
                    device,
                )

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.tuner.parameters(), 5.0
                )
                opt.step()

                train_loss_value = float(
                    loss.detach().item()
                )

                should_validate = (
                    epoch == 0
                    or (epoch + 1) % self.validate_every == 0
                    or (epoch + 1) == self.epochs
                )

                if should_validate:
                    self.tuner.eval()
                    self.live_status.emit(
                        5 + int(88 * (epoch + 1) / max(self.epochs, 1)),
                        f"AI Epoch {epoch+1}/{self.epochs} | validating..."
                    )

                    with torch.no_grad():
                        vdiag = _diagnostic_features(
                            self.plant_model,
                            val_params,
                            self.baseline_gains,
                            self.target_rpm,
                            val_batch,
                            device,
                        )
                        vt = torch.full(
                            (val_batch,),
                            self.target_rpm,
                            device=device,
                        )
                        vg = self.tuner(vdiag, vt)
                        vloss, _ = _closed_loop_objective(
                            self.plant_model,
                            val_params,
                            vg,
                            self.target_rpm,
                            val_batch,
                            device,
                        )
                        val_loss_value = float(
                            vloss.item()
                        )

                    scheduler.step(val_loss_value)

                    if val_loss_value < best_val - 1e-6:
                        best_val = val_loss_value
                        best_state = {
                            k: v.detach().cpu().clone()
                            for k, v in self.tuner.state_dict().items()
                        }
                        no_improve = 0
                    else:
                        no_improve += 1

                epoch_done = epoch + 1
                epoch_elapsed = (
                    time.perf_counter() - epoch_start
                )

                if epoch_time_ema is None:
                    epoch_time_ema = epoch_elapsed
                else:
                    epoch_time_ema = (
                        0.85 * epoch_time_ema
                        + 0.15 * epoch_elapsed
                    )

                eta_total = (
                    self.epochs - epoch_done
                ) * epoch_time_ema

                gmean = (
                    gains.detach()
                    .mean(dim=0)
                    .cpu()
                    .numpy()
                )

                pct = 5 + int(
                    88 * epoch_done / max(self.epochs, 1)
                )

                self.live_status.emit(
                    pct,
                    f"AI Epoch {epoch_done}/{self.epochs} | "
                    f"train={train_loss_value:.5f} | "
                    f"PID≈({gmean[0]:.4f},"
                    f"{gmean[1]:.4f},"
                    f"{gmean[2]:.5f}) | "
                    f"ETA≈{self._fmt_seconds(eta_total)}"
                )

                if should_validate:
                    self.progress.emit(
                        pct,
                        f"AI Epoch {epoch_done}/{self.epochs} | "
                        f"train={train_loss_value:.5f} | "
                        f"val={val_loss_value:.5f} | "
                        f"mean PID=({gmean[0]:.4f},"
                        f"{gmean[1]:.4f},"
                        f"{gmean[2]:.5f}) | "
                        f"epoch={self._fmt_seconds(epoch_elapsed)} | "
                        f"ETA≈{self._fmt_seconds(eta_total)}"
                    )
                elif epoch_done % 5 == 0:
                    self.progress.emit(
                        pct,
                        f"AI Epoch {epoch_done}/{self.epochs} | "
                        f"train={train_loss_value:.5f} | "
                        f"val=skipped | "
                        f"ETA≈{self._fmt_seconds(eta_total)}"
                    )

                if (
                    should_validate
                    and no_improve >= patience_validations
                ):
                    self.progress.emit(
                        94,
                        f"AI tuner early stopping at epoch "
                        f"{epoch_done} after {no_improve} "
                        f"validation checks without improvement."
                    )
                    break

            if best_state is None:
                raise RuntimeError(
                    "Không tạo được AI tuner state."
                )

            self.tuner.load_state_dict(best_state)
            self.tuner.eval()
            self.tuner.is_trained = True

            # Nominal proposal after meta-training.
            nominal_params = _nominal_plant_params(
                1, device
            )
            with torch.no_grad():
                ndiag = _diagnostic_features(
                    self.plant_model,
                    nominal_params,
                    self.baseline_gains,
                    self.target_rpm,
                    1,
                    device,
                )
                nt = torch.tensor(
                    [self.target_rpm],
                    device=device,
                )
                ng = (
                    self.tuner(ndiag, nt)[0]
                    .cpu()
                    .numpy()
                )

            total_wall = (
                time.perf_counter() - wall_start
            )

            stats = {
                "trainer_version": "4.1",
                "epochs": epoch_done,
                "best_val_objective": best_val,
                "last_train_objective": train_loss_value,
                "target_rpm": self.target_rpm,
                "both_directions": True,
                "random_severity": self.random_severity,
                "baseline_gains": list(
                    self.baseline_gains
                ),
                "nominal_ai_pid": ng.tolist(),
                "device": str(device),
                "validate_every_epochs": self.validate_every,
                "training_time_s": total_wall,
            }

            self.tuner.training_meta = stats

            self.tuner.to("cpu")
            self.plant_model.to("cpu")

            if old_req is not None:
                for p, req in zip(
                    self.plant_model.parameters(),
                    old_req,
                ):
                    p.requires_grad_(req)

            self.progress.emit(
                100,
                f"V4.2 AI tuner training complete | "
                f"total={self._fmt_seconds(total_wall)}"
            )
            self.live_status.emit(
                100,
                "DONE - V4.2 AI PID tuner trained."
            )
            self.finished_training.emit(
                self.tuner, stats
            )

        except Exception as e:
            try:
                self.tuner.to("cpu")
                self.plant_model.to("cpu")
                if old_req is not None:
                    for p, req in zip(
                        self.plant_model.parameters(),
                        old_req,
                    ):
                        p.requires_grad_(req)
            except Exception:
                pass

            self.error_occurred.emit(str(e))


class AIHybridTuneThread(QThread):
    progress = pyqtSignal(int, str)
    finished_tuning = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, plant_model: HybridMotorTCN, tuner: PIDGainNet,
                 baseline_gains, target_rpm: float, duration: float,
                 global_bounds: np.ndarray, refine_fraction=V4_LOCAL_REFINE_FRAC,
                 refine_evals=V4_LOCAL_REFINE_EVALS):
        super().__init__()
        self.plant_model = plant_model
        self.tuner = tuner
        self.baseline_gains = tuple(map(float, baseline_gains))
        self.target_rpm = float(target_rpm)
        self.duration = float(duration)
        self.global_bounds = np.asarray(global_bounds, dtype=np.float64).reshape(3, 2)
        self.refine_fraction = float(refine_fraction)
        self.refine_evals = int(refine_evals)
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    @staticmethod
    def _verify(metrics: Optional[ResponseMetrics], duration: float):
        reasons = []
        if metrics is None:
            return False, ["Không có response metrics"]
        if not np.isfinite(metrics.score):
            reasons.append("score không finite")
        if metrics.overshoot_pct > V4_VERIFY_MAX_OVERSHOOT_PCT:
            reasons.append(f"overshoot {metrics.overshoot_pct:.2f}% > {V4_VERIFY_MAX_OVERSHOOT_PCT:.1f}%")
        if metrics.saturation_fraction > V4_VERIFY_MAX_SATURATION:
            reasons.append(f"PWM saturation {100*metrics.saturation_fraction:.1f}% quá cao")
        if metrics.ood_mean > V4_VERIFY_MAX_OOD:
            reasons.append(f"OOD {metrics.ood_mean:.3f} quá cao")
        if metrics.settling_time_s >= 1.49 * duration:
            reasons.append("không xác lập trong simulation window")
        return len(reasons) == 0, reasons

    def run(self):
        try:
            if not self.plant_model.is_trained or not self.tuner.is_trained:
                raise RuntimeError("Cần plant model và AI PID tuner đã train/load.")

            self.progress.emit(5, "Generating nominal diagnostic fingerprint...")
            device = torch.device("cpu")
            self.plant_model.to(device).eval()
            self.tuner.to(device).eval()
            params = _nominal_plant_params(1, device)
            with torch.no_grad():
                diag = _diagnostic_features(
                    self.plant_model, params, self.baseline_gains,
                    self.target_rpm, 1, device,
                )
                candidate = self.tuner.predict_gains(diag, self.target_rpm)[0]

            evaluator = PIDOptimizer(
                self.plant_model,
                target_rpm=self.target_rpm,
                duration=self.duration,
                kp_bounds=tuple(self.global_bounds[0]),
                ki_bounds=tuple(self.global_bounds[1]),
                kd_bounds=tuple(self.global_bounds[2]),
            )
            ai_score, ai_metrics = evaluator.evaluate_candidate(candidate)
            self.progress.emit(
                20,
                f"AI proposal Kp={candidate[0]:.5f} Ki={candidate[1]:.5f} "
                f"Kd={candidate[2]:.6f} | score={ai_score:.4f}",
            )

            final = candidate.copy()
            final_score = ai_score
            final_metrics = ai_metrics

            if self.refine_evals > 0:
                lo_g = self.global_bounds[:, 0]
                hi_g = self.global_bounds[:, 1]
                full_span = np.maximum(hi_g - lo_g, 1e-9)
                half = np.maximum(np.abs(candidate) * self.refine_fraction,
                                  0.02 * full_span)
                lo = np.maximum(lo_g, candidate - half)
                hi = np.minimum(hi_g, candidate + half)
                # avoid zero-width ranges
                hi = np.maximum(hi, lo + 1e-7)

                local = PIDOptimizer(
                    self.plant_model,
                    target_rpm=self.target_rpm,
                    duration=self.duration,
                    kp_bounds=(float(lo[0]), float(hi[0])),
                    ki_bounds=(float(lo[1]), float(hi[1])),
                    kd_bounds=(float(lo[2]), float(hi[2])),
                )

                def cb(done, total, best, score, metrics):
                    if self._stop_requested:
                        return
                    pct = 20 + int(70 * done / max(total, 1))
                    self.progress.emit(
                        pct,
                        f"Local refine {done}/{total} | score={score:.4f} | "
                        f"PID=({best[0]:.4f},{best[1]:.4f},{best[2]:.5f})",
                    )

                r = local.optimize(
                    max_evaluations=max(12, self.refine_evals),
                    callback=cb,
                    stop_fn=lambda: self._stop_requested,
                    extra_seeds=[candidate],
                )
                refined = np.asarray(r[:3], dtype=np.float64)
                refined_score = float(r[3])
                refined_metrics = r[4]
                if refined_score <= final_score:
                    final = refined
                    final_score = refined_score
                    final_metrics = refined_metrics

            verified, reasons = self._verify(final_metrics, self.duration)
            self.progress.emit(95, "Verifying final PID candidate...")
            self.finished_tuning.emit({
                "ai_candidate": candidate.tolist(),
                "ai_score": float(ai_score),
                "ai_metrics": ai_metrics,
                "final_pid": final.tolist(),
                "final_score": float(final_score),
                "final_metrics": final_metrics,
                "verified": bool(verified),
                "verify_reasons": reasons,
            })
        except Exception as e:
            self.error_occurred.emit(str(e))


class OptimizationThread(QThread):
    progress = pyqtSignal(int, str)
    finished_optimization = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, optimizer: PIDOptimizer, max_evaluations: int):
        super().__init__()
        self.optimizer = optimizer
        self.max_evaluations = int(max_evaluations)
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        try:
            def cb(done, total, best, score, metrics):
                pct = int(100 * done / max(total, 1))
                kp, ki, kd = best
                extra = ""
                if metrics is not None:
                    extra = (
                        f" | OS={metrics.overshoot_pct:.1f}% "
                        f"Ts={metrics.settling_time_s:.2f}s "
                        f"SSE={metrics.steady_state_error_pct:.2f}%"
                    )
                self.progress.emit(
                    pct,
                    f"Eval {done}/{total} | best Kp={kp:.4f} Ki={ki:.4f} "
                    f"Kd={kd:.4f} | score={score:.4f}{extra}",
                )

            result = self.optimizer.optimize(
                max_evaluations=self.max_evaluations,
                callback=cb,
                stop_fn=lambda: self._stop_requested,
            )
            self.finished_optimization.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))


# ============================================================
# MAIN GUI
# ============================================================

class PIDTrainerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PID Velocity Tuner - Adaptive AI V4.2")
        self.resize(1500, 850)

        self.collected_data: List[dict] = []
        self.realtime_data = {"time": [], "rpm": [], "target": [], "pwm": [], "error": []}
        self.recording = False
        self.recording_start_index = 0
        self.start_time = time.perf_counter()

        self.model = HybridMotorTCN()
        self.is_trained = False
        self.optimized_pid: Optional[Tuple[float, float, float]] = None
        self.optimized_metrics: Optional[ResponseMetrics] = None

        self.data_thread = DataCollectorThread()
        self.data_thread.data_received.connect(self.on_data_received)
        self.data_thread.connection_changed.connect(self.on_connection_changed)
        self.data_thread.error_occurred.connect(self.on_error)

        self.train_thread: Optional[TrainingThread] = None
        self.opt_thread: Optional[OptimizationThread] = None
        self.pid_tuner: Optional[PIDGainNet] = None
        self.pid_tuner_trained = False
        self.meta_tuner_thread: Optional[MetaTunerTrainingThread] = None
        self.ai_tune_thread: Optional[AIHybridTuneThread] = None
        self.ai_last_verified = False

        self._collection_timers: List[QTimer] = []

        self.init_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(50)

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        left_panel = QWidget()
        left_panel.setFixedWidth(390)
        left_layout = QVBoxLayout(left_panel)

        tabs = QTabWidget()
        tabs.addTab(self.create_serial_tab(), "Serial")
        tabs.addTab(self.create_data_tab(), "Data")
        tabs.addTab(self.create_train_tab(), "Train Plant")
        tabs.addTab(self.create_optimize_tab(), "Global DE")
        tabs.addTab(self.create_ai_tuner_tab(), "AI Tuner V4.2")
        left_layout.addWidget(tabs)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel("left", "RPM")
        self.plot_widget.setLabel("bottom", "Time", units="s")

        # Luôn hiển thị thời gian theo giây (s).
        # Tắt SI prefix để PyQtGraph không tự đổi s -> ks.
        self.plot_widget.getAxis("bottom").enableAutoSIPrefix(False)

        self.plot_widget.setTitle("Motor Response - Realtime")
        self.plot_widget.addLegend()

        self.curve_rpm = self.plot_widget.plot(
            pen=pg.mkPen("b", width=2), name="Actual RPM"
        )
        self.curve_target = self.plot_widget.plot(
            pen=pg.mkPen("r", width=2, style=pg.QtCore.Qt.PenStyle.DashLine),
            name="Target",
        )
        self.curve_pwm = self.plot_widget.plot(
            pen=pg.mkPen("g", width=1), name="PWM/10"
        )

        self.lbl_status = QLabel("Disconnected")
        self.lbl_status.setStyleSheet("font-weight:bold;color:#e74c3c;font-size:14px;")
        self.lbl_data_count = QLabel("Training data: 0 samples")
        self.lbl_model_status = QLabel("Model: Not trained")
        self.lbl_model_status.setStyleSheet("font-weight:bold;")

        status_bar = QHBoxLayout()
        status_bar.addWidget(self.lbl_status)
        status_bar.addWidget(self.lbl_data_count)
        status_bar.addWidget(self.lbl_model_status)
        status_bar.addStretch()

        right_layout.addWidget(self.plot_widget)
        right_layout.addLayout(status_bar)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

    def create_serial_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        group = QGroupBox("Serial Connection")
        gl = QGridLayout()
        self.combo_port = QComboBox()
        self.refresh_ports()
        gl.addWidget(QLabel("Port:"), 0, 0)
        gl.addWidget(self.combo_port, 0, 1)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh_ports)
        gl.addWidget(self.btn_refresh, 0, 2)

        self.btn_connect = QPushButton("CONNECT")
        self.btn_connect.clicked.connect(self.toggle_connection)
        self.btn_connect.setStyleSheet("font-weight:bold;height:35px;")
        gl.addWidget(self.btn_connect, 1, 0, 1, 3)
        group.setLayout(gl)
        layout.addWidget(group)

        group2 = QGroupBox("Manual Command")
        g2 = QHBoxLayout()
        self.input_cmd = QLineEdit()
        self.input_cmd.setPlaceholderText("pid=1.0,0.01,0.1")
        self.btn_send = QPushButton("Send")
        self.btn_send.clicked.connect(self.send_manual_cmd)
        g2.addWidget(self.input_cmd)
        g2.addWidget(self.btn_send)
        group2.setLayout(g2)
        layout.addWidget(group2)

        group3 = QGroupBox("Quick PID")
        g3 = QGridLayout()
        self.input_kp = QLineEdit("0.6")
        self.input_ki = QLineEdit("0.0")
        self.input_kd = QLineEdit("0.0")
        self.input_target = QLineEdit("100")

        for row, (name, w) in enumerate([
            ("Kp:", self.input_kp),
            ("Ki:", self.input_ki),
            ("Kd:", self.input_kd),
            ("Target RPM:", self.input_target),
        ]):
            g3.addWidget(QLabel(name), row, 0)
            g3.addWidget(w, row, 1)

        self.btn_set_pid = QPushButton("SET PID")
        self.btn_set_pid.clicked.connect(self.set_pid_manual)
        self.btn_set_target = QPushButton("SET TARGET")
        self.btn_set_target.clicked.connect(self.set_target_manual)
        self.btn_stop = QPushButton("STOP")
        self.btn_stop.clicked.connect(self.stop_motor)
        self.btn_stop.setStyleSheet("background:#e74c3c;color:white;font-weight:bold;")

        g3.addWidget(self.btn_set_pid, 4, 0, 1, 2)
        g3.addWidget(self.btn_set_target, 5, 0, 1, 2)
        g3.addWidget(self.btn_stop, 6, 0, 1, 2)
        group3.setLayout(g3)
        layout.addWidget(group3)
        layout.addStretch()
        return widget

    def create_data_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        group = QGroupBox("Fixed Target Collection")
        g = QGridLayout()
        self.input_auto_target = QSpinBox()
        self.input_auto_target.setRange(-10000, 10000)
        self.input_auto_target.setValue(100)
        self.input_auto_duration = QSpinBox()
        self.input_auto_duration.setRange(3, 120)
        self.input_auto_duration.setValue(10)

        g.addWidget(QLabel("Target RPM:"), 0, 0)
        g.addWidget(self.input_auto_target, 0, 1)
        g.addWidget(QLabel("Duration (s):"), 1, 0)
        g.addWidget(self.input_auto_duration, 1, 1)

        self.btn_auto_collect = QPushButton("START FIXED COLLECT")
        self.btn_auto_collect.clicked.connect(self.start_auto_collect)
        self.btn_stop_collect = QPushButton("STOP COLLECT")
        self.btn_stop_collect.clicked.connect(self.stop_collection)
        self.btn_stop_collect.setEnabled(False)
        g.addWidget(self.btn_auto_collect, 2, 0, 1, 2)
        g.addWidget(self.btn_stop_collect, 3, 0, 1, 2)
        group.setLayout(g)
        layout.addWidget(group)

        group2 = QGroupBox("Rich Excitation / Multi Target")
        g2 = QGridLayout()
        self.input_range_from = QSpinBox()
        self.input_range_from.setRange(-10000, 9999)
        self.input_range_from.setValue(20)
        self.input_range_to = QSpinBox()
        self.input_range_to.setRange(-9999, 10000)
        self.input_range_to.setValue(150)
        self.input_range_step = QSpinBox()
        self.input_range_step.setRange(5, 1000)
        self.input_range_step.setValue(30)
        self.input_each_duration = QDoubleSpinBox()
        self.input_each_duration.setRange(0.5, 20.0)
        self.input_each_duration.setValue(2.0)
        self.input_each_duration.setSingleStep(0.5)

        for row, (name, w) in enumerate([
            ("From RPM:", self.input_range_from),
            ("To RPM:", self.input_range_to),
            ("Step:", self.input_range_step),
            ("Each (s):", self.input_each_duration),
        ]):
            g2.addWidget(QLabel(name), row, 0)
            g2.addWidget(w, row, 1)

        self.chk_random_order = QCheckBox("Randomize target order")
        self.chk_random_order.setChecked(True)
        g2.addWidget(self.chk_random_order, 4, 0, 1, 2)

        self.btn_multi_collect = QPushButton("RICH EXCITATION COLLECT")
        self.btn_multi_collect.clicked.connect(self.start_multi_collect)
        self.btn_multi_collect.setStyleSheet(
            "font-weight:bold;height:35px;background:#8e44ad;color:white;"
        )
        g2.addWidget(self.btn_multi_collect, 5, 0, 1, 2)
        group2.setLayout(g2)
        layout.addWidget(group2)

        group3 = QGroupBox("Data Management")
        g3 = QHBoxLayout()
        self.btn_save_data = QPushButton("Save Data")
        self.btn_load_data = QPushButton("Load Data")
        self.btn_clear_data = QPushButton("Clear Data")
        self.btn_save_data.clicked.connect(self.save_data)
        self.btn_load_data.clicked.connect(self.load_data)
        self.btn_clear_data.clicked.connect(self.clear_data)
        g3.addWidget(self.btn_save_data)
        g3.addWidget(self.btn_load_data)
        g3.addWidget(self.btn_clear_data)
        group3.setLayout(g3)
        layout.addWidget(group3)

        self.progress_bar = QProgressBar()
        self.lbl_collect_status = QLabel("Ready")
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.lbl_collect_status)
        layout.addStretch()
        return widget

    def create_train_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        group = QGroupBox("V3.1 Hybrid TCN Plant Training")
        g = QGridLayout()

        self.input_history = QSpinBox()
        self.input_history.setRange(8, 100)
        self.input_history.setValue(DEFAULT_HISTORY_LEN)

        self.input_channels = QSpinBox()
        self.input_channels.setRange(16, 256)
        self.input_channels.setValue(DEFAULT_TCN_CHANNELS)

        self.input_ensemble = QSpinBox()
        self.input_ensemble.setRange(1, 5)
        self.input_ensemble.setValue(DEFAULT_ENSEMBLE_SIZE)

        self.input_rollout = QSpinBox()
        self.input_rollout.setRange(2, 50)
        self.input_rollout.setValue(DEFAULT_ROLLOUT_HORIZON)

        self.input_lr = QDoubleSpinBox()
        self.input_lr.setRange(0.00001, 0.01)
        self.input_lr.setDecimals(5)
        self.input_lr.setValue(0.001)

        self.input_epochs = QSpinBox()
        self.input_epochs.setRange(50, 5000)
        self.input_epochs.setValue(400)

        self.input_batch = QSpinBox()
        self.input_batch.setRange(16, 2048)
        self.input_batch.setValue(128)

        self.input_train_device = QComboBox()
        self.input_train_device.addItems(["Auto", "CUDA", "CPU"])
        self.input_train_device.setCurrentText("Auto")

        for row, (name, w) in enumerate([
            ("History samples:", self.input_history),
            ("TCN channels:", self.input_channels),
            ("Ensemble size:", self.input_ensemble),
            ("Rollout steps:", self.input_rollout),
            ("Learning rate:", self.input_lr),
            ("Epochs:", self.input_epochs),
            ("Batch size:", self.input_batch),
            ("Device:", self.input_train_device),
        ]):
            g.addWidget(QLabel(name), row, 0)
            g.addWidget(w, row, 1)

        self.btn_train = QPushButton("TRAIN V3.1 PLANT MODEL")
        self.btn_train.clicked.connect(self.start_training)
        self.btn_train.setStyleSheet(
            "font-weight:bold;height:40px;background:#27ae60;color:white;"
        )
        g.addWidget(self.btn_train, 8, 0, 1, 2)
        group.setLayout(g)
        layout.addWidget(group)

        group2 = QGroupBox("Training Log")
        g2 = QVBoxLayout()

        self.lbl_train_live = QLabel("Idle")
        self.lbl_train_live.setWordWrap(True)
        self.lbl_train_live.setStyleSheet(
            "font-weight:bold;color:#2980b9;padding:4px;"
        )
        g2.addWidget(self.lbl_train_live)

        self.txt_train_log = QTextEdit()
        self.txt_train_log.setReadOnly(True)
        g2.addWidget(self.txt_train_log)
        group2.setLayout(g2)
        layout.addWidget(group2)

        group3 = QGroupBox("Model")
        g3 = QHBoxLayout()
        self.btn_save_model = QPushButton("Save V3/V3.1 Plant")
        self.btn_load_model = QPushButton("Load V3/V3.1 Plant")
        self.btn_save_model.clicked.connect(self.save_model)
        self.btn_load_model.clicked.connect(self.load_model)
        g3.addWidget(self.btn_save_model)
        g3.addWidget(self.btn_load_model)
        group3.setLayout(g3)
        layout.addWidget(group3)
        return widget

    def create_optimize_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        group = QGroupBox("PID Optimization - Differential Evolution")
        g = QGridLayout()

        self.input_opt_target = QSpinBox()
        self.input_opt_target.setRange(-10000, 10000)
        self.input_opt_target.setValue(100)
        self.input_opt_eval = QSpinBox()
        self.input_opt_eval.setRange(60, 2000)
        self.input_opt_eval.setValue(300)
        self.input_opt_duration = QDoubleSpinBox()
        self.input_opt_duration.setRange(1.0, 15.0)
        self.input_opt_duration.setValue(4.0)
        self.input_opt_duration.setSingleStep(0.5)

        self.input_kp_max = QDoubleSpinBox()
        self.input_kp_max.setRange(0.01, 100.0)
        self.input_kp_max.setValue(5.0)
        self.input_ki_max = QDoubleSpinBox()
        self.input_ki_max.setRange(0.01, 100.0)
        self.input_ki_max.setValue(10.0)
        self.input_kd_max = QDoubleSpinBox()
        self.input_kd_max.setRange(0.001, 20.0)
        self.input_kd_max.setDecimals(3)
        self.input_kd_max.setValue(2.0)

        self.input_both_directions = QCheckBox(
            "Evaluate both +RPM and -RPM"
        )
        self.input_both_directions.setChecked(True)

        rows = [
            ("Target RPM:", self.input_opt_target),
            ("Eval budget:", self.input_opt_eval),
            ("Simulation (s):", self.input_opt_duration),
            ("Kp max:", self.input_kp_max),
            ("Ki max:", self.input_ki_max),
            ("Kd max:", self.input_kd_max),
        ]
        for row, (name, w) in enumerate(rows):
            g.addWidget(QLabel(name), row, 0)
            g.addWidget(w, row, 1)

        g.addWidget(
            self.input_both_directions,
            len(rows), 0, 1, 2
        )

        self.btn_optimize = QPushButton("OPTIMIZE PID")
        self.btn_optimize.clicked.connect(self.start_optimization)
        self.btn_optimize.setStyleSheet(
            "font-weight:bold;height:40px;background:#f39c12;color:white;"
        )
        g.addWidget(self.btn_optimize, len(rows) + 1, 0, 1, 2)

        self.btn_apply_pid = QPushButton("APPLY TO ESP32")
        self.btn_apply_pid.clicked.connect(self.apply_optimized_pid)
        self.btn_apply_pid.setEnabled(False)
        g.addWidget(self.btn_apply_pid, len(rows) + 2, 0, 1, 2)
        group.setLayout(g)
        layout.addWidget(group)

        group2 = QGroupBox("Best Result")
        g2 = QGridLayout()
        self.lbl_opt_kp = QLabel("---")
        self.lbl_opt_ki = QLabel("---")
        self.lbl_opt_kd = QLabel("---")
        self.lbl_opt_score = QLabel("---")
        self.lbl_opt_metrics = QLabel("---")
        self.lbl_opt_metrics.setWordWrap(True)

        for row, (name, w) in enumerate([
            ("Best Kp:", self.lbl_opt_kp),
            ("Best Ki:", self.lbl_opt_ki),
            ("Best Kd:", self.lbl_opt_kd),
            ("Score:", self.lbl_opt_score),
        ]):
            g2.addWidget(QLabel(name), row, 0)
            g2.addWidget(w, row, 1)
        g2.addWidget(QLabel("Metrics:"), 4, 0)
        g2.addWidget(self.lbl_opt_metrics, 4, 1)
        group2.setLayout(g2)
        layout.addWidget(group2)

        group3 = QGroupBox("Optimization Log")
        g3 = QVBoxLayout()
        self.txt_opt_log = QTextEdit()
        self.txt_opt_log.setReadOnly(True)
        g3.addWidget(self.txt_opt_log)
        group3.setLayout(g3)
        layout.addWidget(group3)
        return widget


    def create_ai_tuner_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        g0 = QGroupBox("Baseline PID used for diagnostic")
        grid0 = QGridLayout()
        self.ai_base_kp = QDoubleSpinBox(); self.ai_base_kp.setRange(0.0, 100.0); self.ai_base_kp.setDecimals(6); self.ai_base_kp.setValue(0.26925)
        self.ai_base_ki = QDoubleSpinBox(); self.ai_base_ki.setRange(0.0, 100.0); self.ai_base_ki.setDecimals(6); self.ai_base_ki.setValue(1.45549)
        self.ai_base_kd = QDoubleSpinBox(); self.ai_base_kd.setRange(0.0, 20.0); self.ai_base_kd.setDecimals(6); self.ai_base_kd.setValue(0.00287)
        self.ai_target = QSpinBox(); self.ai_target.setRange(-10000, 10000); self.ai_target.setValue(100)
        for r, (name, w) in enumerate([
            ("Base Kp:", self.ai_base_kp), ("Base Ki:", self.ai_base_ki),
            ("Base Kd:", self.ai_base_kd), ("Target RPM:", self.ai_target),
        ]):
            grid0.addWidget(QLabel(name), r, 0); grid0.addWidget(w, r, 1)
        g0.setLayout(grid0); layout.addWidget(g0)

        g1 = QGroupBox("Meta-train PIDGainNet")
        grid1 = QGridLayout()
        self.ai_epochs = QSpinBox(); self.ai_epochs.setRange(20, 3000); self.ai_epochs.setValue(V4_TUNER_EPOCHS)
        self.ai_batch = QSpinBox(); self.ai_batch.setRange(2, 128); self.ai_batch.setValue(V4_TUNER_BATCH)
        self.ai_lr = QDoubleSpinBox(); self.ai_lr.setRange(0.00001, 0.01); self.ai_lr.setDecimals(6); self.ai_lr.setValue(V4_TUNER_LR)
        self.ai_random = QDoubleSpinBox(); self.ai_random.setRange(0.0, 0.60); self.ai_random.setDecimals(2); self.ai_random.setSingleStep(0.05); self.ai_random.setValue(V4_RANDOM_SEVERITY)

        self.ai_device = QComboBox()
        self.ai_device.addItems(["Auto", "CUDA", "CPU"])
        self.ai_device.setCurrentText("Auto")

        self.ai_kp_max = QDoubleSpinBox(); self.ai_kp_max.setRange(0.01, 100.0); self.ai_kp_max.setValue(5.0)
        self.ai_ki_max = QDoubleSpinBox(); self.ai_ki_max.setRange(0.01, 100.0); self.ai_ki_max.setValue(10.0)
        self.ai_kd_max = QDoubleSpinBox(); self.ai_kd_max.setRange(0.0001, 20.0); self.ai_kd_max.setDecimals(4); self.ai_kd_max.setValue(2.0)
        for r, (name, w) in enumerate([
            ("Epochs:", self.ai_epochs), ("Batch plants:", self.ai_batch),
            ("Learning rate:", self.ai_lr), ("Random severity:", self.ai_random),
            ("Device:", self.ai_device),
            ("Kp max:", self.ai_kp_max), ("Ki max:", self.ai_ki_max), ("Kd max:", self.ai_kd_max),
        ]):
            grid1.addWidget(QLabel(name), r, 0); grid1.addWidget(w, r, 1)
        self.btn_train_ai = QPushButton("TRAIN AI PID TUNER")
        self.btn_train_ai.clicked.connect(self.start_ai_tuner_training)
        self.btn_train_ai.setStyleSheet("font-weight:bold;height:36px;background:#16a085;color:white;")
        grid1.addWidget(self.btn_train_ai, 8, 0, 1, 2)
        g1.setLayout(grid1); layout.addWidget(g1)

        g2 = QGroupBox("AI inference + local DE refinement")
        grid2 = QGridLayout()
        self.ai_refine_pct = QDoubleSpinBox(); self.ai_refine_pct.setRange(0.0, 1.0); self.ai_refine_pct.setDecimals(2); self.ai_refine_pct.setSingleStep(0.05); self.ai_refine_pct.setValue(V4_LOCAL_REFINE_FRAC)
        self.ai_refine_evals = QSpinBox(); self.ai_refine_evals.setRange(0, 2000); self.ai_refine_evals.setValue(V4_LOCAL_REFINE_EVALS)
        self.ai_duration = QDoubleSpinBox(); self.ai_duration.setRange(1.0, 15.0); self.ai_duration.setValue(4.0); self.ai_duration.setSingleStep(0.5)
        for r, (name, w) in enumerate([
            ("Local range ±frac:", self.ai_refine_pct),
            ("Local evals:", self.ai_refine_evals),
            ("Verify duration (s):", self.ai_duration),
        ]):
            grid2.addWidget(QLabel(name), r, 0); grid2.addWidget(w, r, 1)
        self.btn_ai_tune = QPushButton("AI TUNE + LOCAL REFINE")
        self.btn_ai_tune.clicked.connect(self.start_ai_hybrid_tuning)
        self.btn_ai_tune.setEnabled(False)
        self.btn_ai_tune.setStyleSheet("font-weight:bold;height:38px;background:#2980b9;color:white;")
        grid2.addWidget(self.btn_ai_tune, 3, 0, 1, 2)
        self.btn_apply_ai = QPushButton("APPLY VERIFIED PID TO ESP32")
        self.btn_apply_ai.clicked.connect(self.apply_ai_pid)
        self.btn_apply_ai.setEnabled(False)
        grid2.addWidget(self.btn_apply_ai, 4, 0, 1, 2)
        g2.setLayout(grid2); layout.addWidget(g2)

        g3 = QGroupBox("AI Tuner Model")
        row = QHBoxLayout()
        self.btn_save_ai = QPushButton("Save AI Tuner")
        self.btn_load_ai = QPushButton("Load AI Tuner")
        self.btn_save_ai.clicked.connect(self.save_ai_tuner)
        self.btn_load_ai.clicked.connect(self.load_ai_tuner)
        row.addWidget(self.btn_save_ai); row.addWidget(self.btn_load_ai)
        g3.setLayout(row); layout.addWidget(g3)

        self.lbl_ai_status = QLabel("AI tuner: not trained")
        self.lbl_ai_status.setWordWrap(True)

        # V4.2 bug fix: this label is used by the training callbacks.
        # V4.1 referenced self.lbl_ai_live but never created it, so clicking
        # TRAIN AI PID TUNER raised AttributeError after disabling the button.
        self.lbl_ai_live = QLabel("AI training: idle")
        self.lbl_ai_live.setWordWrap(True)
        self.lbl_ai_live.setStyleSheet(
            "font-weight:bold;color:#16a085;padding:4px;"
        )

        self.txt_ai_log = QTextEdit()
        self.txt_ai_log.setReadOnly(True)

        self.btn_train_ai.setToolTip(
            "Cần train/load Plant model trước. Sau đó nút này train PIDGainNet."
        )
        self.btn_ai_tune.setToolTip(
            "Chỉ bật sau khi AI PID tuner đã train hoặc load thành công."
        )

        layout.addWidget(self.lbl_ai_status)
        layout.addWidget(self.lbl_ai_live)
        layout.addWidget(self.txt_ai_log)

        # Plant chưa sẵn sàng khi GUI vừa khởi động.
        self._refresh_ai_controls()
        return widget

    # --------------------------------------------------------
    # SERIAL
    # --------------------------------------------------------

    def refresh_ports(self):
        current = self.combo_port.currentText() if hasattr(self, "combo_port") else ""
        if hasattr(self, "combo_port"):
            self.combo_port.clear()
            ports = serial.tools.list_ports.comports()
            for p in sorted(ports, key=lambda x: x.device):
                self.combo_port.addItem(p.device)
            if current:
                idx = self.combo_port.findText(current)
                if idx >= 0:
                    self.combo_port.setCurrentIndex(idx)

    def toggle_connection(self):
        if self.data_thread.running:
            self.data_thread.disconnect()
        else:
            port = self.combo_port.currentText()
            if port:
                self.data_thread.connect_to_port(port)

    def on_connection_changed(self, connected, port):
        if connected:
            self.btn_connect.setText("DISCONNECT")
            self.lbl_status.setText(f"Connected: {port}")
            self.lbl_status.setStyleSheet("font-weight:bold;color:#27ae60;font-size:14px;")
        else:
            self.btn_connect.setText("CONNECT")
            self.lbl_status.setText("Disconnected")
            self.lbl_status.setStyleSheet("font-weight:bold;color:#e74c3c;font-size:14px;")

    def send_manual_cmd(self):
        cmd = self.input_cmd.text().strip()
        if cmd:
            self.data_thread.send_command(cmd)
            self.txt_train_log.append(f">>> {cmd}")

    def set_pid_manual(self):
        try:
            kp = float(self.input_kp.text())
            ki = float(self.input_ki.text())
            kd = float(self.input_kd.text())
            self.data_thread.send_command(f"pid={kp},{ki},{kd}")
        except ValueError:
            self.on_error("PID không hợp lệ")

    def set_target_manual(self):
        try:
            target = float(self.input_target.text())

            # Không reset thời gian/đồ thị khi đổi target.
            # Trục thời gian tiếp tục chạy liên tục.
            self.data_thread.send_command(f"v={target}")
        except ValueError:
            self.on_error("Target RPM không hợp lệ")

    def stop_motor(self):
        self.data_thread.send_command("stop")

    # --------------------------------------------------------
    # DATA COLLECTION
    # --------------------------------------------------------

    def on_data_received(self, data):
        current_s = float(data["time"] - self.start_time)
        self.realtime_data["time"].append(current_s)
        self.realtime_data["rpm"].append(data["rpm"])
        self.realtime_data["target"].append(data["target"])
        self.realtime_data["pwm"].append(data["pwm"])
        self.realtime_data["error"].append(data["error"])

        while (self.realtime_data["time"] and
               self.realtime_data["time"][-1] - self.realtime_data["time"][0] > PLOT_WINDOW_S):
            for key in self.realtime_data:
                self.realtime_data[key].pop(0)

        if self.recording:
            self.collected_data.append(data)
            self.lbl_data_count.setText(f"Training data: {len(self.collected_data)} samples")

    def on_error(self, msg):
        self.lbl_status.setText(f"Error: {msg}")
        self.lbl_status.setStyleSheet("font-weight:bold;color:#e74c3c;font-size:14px;")

    def reset_realtime_plot(self):
        """Reset đồ thị real-time và lấy mốc thời gian mới tại 0 s."""
        self.start_time = time.perf_counter()
        self.realtime_data = {
            "time": [],
            "rpm": [],
            "target": [],
            "pwm": [],
            "error": [],
        }

        # Xóa dữ liệu đang hiển thị ngay lập tức, không chờ sample Serial mới.
        self.curve_rpm.setData([], [])
        self.curve_target.setData([], [])
        self.curve_pwm.setData([], [])

    def _begin_recording(self):
        self.recording = True
        self.recording_start_index = len(self.collected_data)

        # Mỗi lần bắt đầu thu dữ liệu là một phép thử mới -> bắt đầu từ 0 s.
        self.reset_realtime_plot()

        self.btn_auto_collect.setEnabled(False)
        self.btn_multi_collect.setEnabled(False)
        self.btn_stop_collect.setEnabled(True)

    def _end_recording(self, stop_motor=True):
        if stop_motor:
            self.data_thread.send_command("stop")
        self.recording = False
        self.btn_auto_collect.setEnabled(True)
        self.btn_multi_collect.setEnabled(True)
        self.btn_stop_collect.setEnabled(False)
        new_samples = len(self.collected_data) - self.recording_start_index
        self.lbl_collect_status.setText(
            f"Collection DONE: +{new_samples} samples | total={len(self.collected_data)}"
        )
        self.progress_bar.setValue(100)

    def start_auto_collect(self):
        if not self.data_thread.running:
            self.lbl_collect_status.setText("ERROR: Connect ESP32 first.")
            return

        target = self.input_auto_target.value()
        duration = self.input_auto_duration.value()
        self._begin_recording()
        self.data_thread.send_command(f"v={target}")
        self.lbl_collect_status.setText(f"Collecting {target} RPM for {duration}s...")
        QTimer.singleShot(duration * 1000, self._end_recording)

    def stop_collection(self):
        self._end_recording(stop_motor=True)

    def start_multi_collect(self):
        if not self.data_thread.running:
            self.lbl_collect_status.setText("ERROR: Connect ESP32 first.")
            return

        start = self.input_range_from.value()
        end = self.input_range_to.value()
        step = self.input_range_step.value()
        each_ms = int(self.input_each_duration.value() * 1000)

        if start == end:
            self.lbl_collect_status.setText("ERROR: From và To phải khác nhau.")
            return

        direction = 1 if end > start else -1
        values = list(range(start, end + direction, direction * step))
        if values[-1] != end:
            values.append(end)

        # Cho ANN thấy cả tăng/giảm setpoint; random order giúp giảm tương quan dữ liệu.
        if self.chk_random_order.isChecked():
            random.shuffle(values)
        else:
            values = values + list(reversed(values[:-1]))

        self._begin_recording()
        total = len(values)

        def collect_next(idx):
            if not self.recording:
                return
            if idx >= total:
                self._end_recording(stop_motor=True)
                return

            target = values[idx]
            self.data_thread.send_command(f"v={target}")
            self.progress_bar.setValue(int(100 * idx / max(total, 1)))
            self.lbl_collect_status.setText(
                f"Rich excitation: {target} RPM ({idx+1}/{total})"
            )
            QTimer.singleShot(each_ms, lambda: collect_next(idx + 1))

        collect_next(0)

    def save_data(self):
        if not self.collected_data:
            self.lbl_collect_status.setText("No training data to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Data", "training_data_v3.json", "JSON Files (*.json)"
        )
        if path:
            payload = {
                "format_version": 3,
                "nominal_dt": NOMINAL_DT,
                "samples": self.collected_data,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            self.lbl_collect_status.setText(f"Saved {len(self.collected_data)} samples to {path}")

    def load_data(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Data", "", "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict) and "samples" in obj:
                data = obj["samples"]
            elif isinstance(obj, list):
                data = obj  # tương thích file JSON bản cũ
            else:
                raise ValueError("Unsupported data format")
            self.collected_data = list(data)
            self.lbl_data_count.setText(f"Training data: {len(self.collected_data)} samples")
            self.lbl_collect_status.setText(f"Loaded {len(self.collected_data)} samples")
        except Exception as e:
            self.lbl_collect_status.setText(f"Load error: {e}")

    def clear_data(self):
        self.collected_data.clear()
        self.lbl_data_count.setText("Training data: 0 samples")
        self.lbl_collect_status.setText("Training data cleared.")

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    def start_training(self):
        if len(self.collected_data) < 300:
            self.txt_train_log.setText(
                "Need at least ~300 samples. Khuyến nghị >= 5000 samples, nhiều RPM/PWM và cả tăng/giảm tốc."
            )
            return
        if self.train_thread and self.train_thread.isRunning():
            return

        history = self.input_history.value()
        channels = self.input_channels.value()
        ensemble = self.input_ensemble.value()
        rollout = self.input_rollout.value()
        epochs = self.input_epochs.value()
        lr = self.input_lr.value()
        batch = self.input_batch.value()
        device_pref = self.input_train_device.currentText().strip().lower()

        self.model = HybridMotorTCN(
            history_len=history,
            channels=channels,
            levels=DEFAULT_TCN_LEVELS,
            ensemble_size=ensemble,
            kernel_size=3,
            dropout=DEFAULT_DROPOUT,
        )
        self.train_thread = TrainingThread(
            self.collected_data,
            self.model,
            epochs=epochs,
            learning_rate=lr,
            batch_size=batch,
            rollout_horizon=rollout,
            device_preference=device_pref,
            validate_every=VALIDATE_EVERY_EPOCHS,
        )
        self.train_thread.progress.connect(self.on_train_progress)
        self.train_thread.live_status.connect(self.on_train_live_status)
        self.train_thread.finished_training.connect(self.on_train_finished)
        self.train_thread.error_occurred.connect(self.on_train_error)
        self.btn_train.setEnabled(False)
        self.progress_bar.setValue(0)
        self.lbl_train_live.setText("Preparing plant training...")
        self.txt_train_log.clear()
        self.txt_train_log.append(
            f"V4.1 plant: history={history}, channels={channels}, "
            f"ensemble={ensemble}, rollout={rollout}, device={device_pref}"
        )
        self.train_thread.start()

    def on_train_progress(self, pct, msg):
        self.progress_bar.setValue(pct)
        self.txt_train_log.append(msg)

    def on_train_live_status(self, pct, msg):
        self.progress_bar.setValue(pct)
        self.lbl_train_live.setText(msg)

    def on_train_finished(self, model, stats):
        self.model = model
        self.is_trained = True
        self.btn_train.setEnabled(True)
        one = stats.get("one_step_rmse_rpm", float("nan"))
        roll = stats.get("rollout_rmse_rpm", float("nan"))
        free = stats.get("free_run_rmse_rpm", float("nan"))
        self.lbl_model_status.setText(
            f"V4.1 Plant | RMSE 1-step={one:.2f}, rollout={roll:.2f}, free-run={free:.2f} RPM"
        )
        self.lbl_model_status.setStyleSheet("font-weight:bold;color:#27ae60;")
        self.txt_train_log.append("\n=== V4.1 PLANT TRAINING METRICS ===")
        self.txt_train_log.append(json.dumps(stats, indent=2))
        self._refresh_ai_controls()

    def on_train_error(self, msg):
        self.btn_train.setEnabled(True)
        self.lbl_train_live.setText("ERROR")
        self.txt_train_log.append(f"ERROR: {msg}")

    def _scaler_to_dict(self, scaler: StandardScaler):
        return {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "var": scaler.var_.tolist(),
            "n_features_in": int(scaler.n_features_in_),
            "n_samples_seen": int(np.asarray(scaler.n_samples_seen_).reshape(-1)[0]),
        }

    def _restore_scaler(self, scaler: StandardScaler, d: dict):
        scaler.mean_ = np.asarray(d["mean"], dtype=np.float64)
        scaler.scale_ = np.asarray(d["scale"], dtype=np.float64)
        scaler.var_ = np.asarray(d["var"], dtype=np.float64)
        scaler.n_features_in_ = int(d["n_features_in"])
        scaler.n_samples_seen_ = int(d.get("n_samples_seen", 1))

    def save_model(self):
        if not self.model.is_trained:
            self.txt_train_log.append("Model chưa train.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save V3/V3.1 Plant Model", "pid_motor_hybrid_tcn_v3_1.pth", "PyTorch Model (*.pth)"
        )
        if not path:
            return

        # Save CPU state for portable loading.
        self.model.to("cpu")
        ckpt = {
            "format_version": 3,
            "model_type": "HybridMotorTCN",
            "model_config": self.model.model_config,
            "model_state": self.model.state_dict(),
            "scaler_X": self._scaler_to_dict(self.model.scaler_X),
            "scaler_y": self._scaler_to_dict(self.model.scaler_y),
            "physics_coef": self.model.physics_coef.tolist(),
            "feature_min": None if self.model.feature_min is None else self.model.feature_min.tolist(),
            "feature_max": None if self.model.feature_max is None else self.model.feature_max.tolist(),
            "train_rpm_min": self.model.train_rpm_min,
            "train_rpm_max": self.model.train_rpm_max,
            "train_pwm_min": self.model.train_pwm_min,
            "train_pwm_max": self.model.train_pwm_max,
            "val_metrics": self.model.val_metrics,
            "nominal_dt": NOMINAL_DT,
            "feature_names": self.model.FEATURE_NAMES,
            "is_trained": True,
        }
        torch.save(ckpt, path)
        self.txt_train_log.append(f"V3 model saved: {path}")

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load V3/V3.1 Plant Model", "", "PyTorch Model (*.pth)"
        )
        if not path:
            return
        try:
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                ckpt = torch.load(path, map_location="cpu")

            if int(ckpt.get("format_version", 0)) != 3:
                raise ValueError(
                    "Checkpoint không phải V3. V2 dùng MLP one-step nên không load trực tiếp; "
                    "hãy load data JSON cũ và train lại V3."
                )
            if ckpt.get("model_type") != "HybridMotorTCN":
                raise ValueError("Unsupported V3 model_type")

            cfg = ckpt["model_config"]
            self.model = HybridMotorTCN(**cfg)
            self.model.load_state_dict(ckpt["model_state"])
            self._restore_scaler(self.model.scaler_X, ckpt["scaler_X"])
            self._restore_scaler(self.model.scaler_y, ckpt["scaler_y"])
            self.model.physics_coef = np.asarray(ckpt["physics_coef"], dtype=np.float64)
            self.model.feature_min = (
                None if ckpt.get("feature_min") is None
                else np.asarray(ckpt["feature_min"], dtype=np.float64)
            )
            self.model.feature_max = (
                None if ckpt.get("feature_max") is None
                else np.asarray(ckpt["feature_max"], dtype=np.float64)
            )
            self.model.train_rpm_min = ckpt.get("train_rpm_min")
            self.model.train_rpm_max = ckpt.get("train_rpm_max")
            self.model.train_pwm_min = ckpt.get("train_pwm_min")
            self.model.train_pwm_max = ckpt.get("train_pwm_max")
            self.model.val_metrics = ckpt.get("val_metrics", {})
            self.model.is_trained = True
            self.model.eval()
            self.is_trained = True

            # Sync UI to loaded architecture.
            if hasattr(self, "input_history"):
                self.input_history.setValue(self.model.history_len)
                self.input_channels.setValue(self.model.channels)
                self.input_ensemble.setValue(self.model.ensemble_size)

            m = self.model.val_metrics
            self.lbl_model_status.setText(
                f"V3 Loaded | RMSE 1-step={m.get('one_step_rmse_rpm', float('nan')):.2f}, "
                f"free-run={m.get('free_run_rmse_rpm', float('nan')):.2f} RPM"
            )
            self.lbl_model_status.setStyleSheet("font-weight:bold;color:#27ae60;")
            self.txt_train_log.append(f"V3 model loaded: {path}")
            self._refresh_ai_controls()
        except Exception as e:
            self.txt_train_log.append(f"LOAD ERROR: {e}")

    # --------------------------------------------------------
    # OPTIMIZATION
    # --------------------------------------------------------

    def start_optimization(self):
        if not self.is_trained or not self.model.is_trained:
            self.txt_opt_log.setText("Train/load Hybrid TCN plant model first.")
            return
        if self.opt_thread and self.opt_thread.isRunning():
            return

        target = float(self.input_opt_target.value())
        if abs(target) < 1e-6:
            self.txt_opt_log.setText("Target RPM must be non-zero for tuning.")
            return

        optimizer = PIDOptimizer(
            self.model,
            target_rpm=target,
            duration=self.input_opt_duration.value(),
            kp_bounds=(0.0, self.input_kp_max.value()),
            ki_bounds=(0.0, self.input_ki_max.value()),
            kd_bounds=(0.0, self.input_kd_max.value()),
            both_directions=self.input_both_directions.isChecked(),
        )
        budget = self.input_opt_eval.value()

        self.txt_opt_log.clear()
        dirs = "both +/- directions" if self.input_both_directions.isChecked() else "selected direction only"
        self.txt_opt_log.append(
            f"Optimizing robust PID around |target|={abs(target):.1f} RPM | "
            f"{dirs} | budget={budget} evaluations..."
        )
        self.btn_optimize.setEnabled(False)
        self.btn_apply_pid.setEnabled(False)
        self.progress_bar.setValue(0)

        self.opt_thread = OptimizationThread(optimizer, budget)
        self.opt_thread.progress.connect(self.on_opt_progress)
        self.opt_thread.finished_optimization.connect(self.on_opt_finished)
        self.opt_thread.error_occurred.connect(self.on_opt_error)
        self.opt_thread.start()

    def on_opt_progress(self, pct, msg):
        self.progress_bar.setValue(pct)
        # log vừa đủ, tránh QTextEdit quá nặng
        if pct % 5 == 0:
            self.txt_opt_log.append(msg)

    def on_opt_finished(self, result):
        kp, ki, kd, score, metrics = result
        self.optimized_pid = (kp, ki, kd)
        self.optimized_metrics = metrics

        self.lbl_opt_kp.setText(f"{kp:.6f}")
        self.lbl_opt_ki.setText(f"{ki:.6f}")
        self.lbl_opt_kd.setText(f"{kd:.6f}")
        self.lbl_opt_score.setText(f"{score:.4f}")

        if metrics is not None:
            self.lbl_opt_metrics.setText(
                f"OS={metrics.overshoot_pct:.2f}% | "
                f"Tr={metrics.rise_time_s:.3f}s | "
                f"Ts={metrics.settling_time_s:.3f}s | "
                f"SSE={metrics.steady_state_error_pct:.2f}% | "
                f"sat={100*metrics.saturation_fraction:.1f}% | "
                f"OOD={metrics.ood_mean:.4f}"
            )

        self.txt_opt_log.append("\n=== OPTIMAL PID ===")
        self.txt_opt_log.append(f"Kp={kp:.6f}  Ki={ki:.6f}  Kd={kd:.6f}")
        self.txt_opt_log.append(f"Robust score={score:.4f}")
        self.txt_opt_log.append(
            "NOTE: Learned-model result is a candidate. Test on hardware with low target first before full-speed use."
        )

        self.progress_bar.setValue(100)
        self.btn_optimize.setEnabled(True)
        self.btn_apply_pid.setEnabled(True)

    def on_opt_error(self, msg):
        self.btn_optimize.setEnabled(True)
        self.txt_opt_log.append(f"OPTIMIZATION ERROR: {msg}")

    def apply_optimized_pid(self):
        if self.optimized_pid is None:
            return
        if not self.data_thread.running:
            self.txt_opt_log.append("ESP32 chưa connected.")
            return

        kp, ki, kd = self.optimized_pid
        answer = QMessageBox.question(
            self,
            "Apply optimized PID",
            f"Apply Kp={kp:.6f}, Ki={ki:.6f}, Kd={kd:.6f} to ESP32?\n\n"
            "Khuyến nghị test target thấp trước và sẵn sàng STOP motor.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        self.data_thread.send_command(f"pid={kp},{ki},{kd}")
        self.input_kp.setText(f"{kp:.6f}")
        self.input_ki.setText(f"{ki:.6f}")
        self.input_kd.setText(f"{kd:.6f}")
        self.txt_opt_log.append("PID applied to ESP32.")


    # --------------------------------------------------------
    # V4 AI PID TUNER
    # --------------------------------------------------------

    def _refresh_ai_controls(self):
        """Đồng bộ trạng thái các nút AI theo plant/tuner hiện có."""
        plant_ready = bool(
            getattr(self, "is_trained", False)
            and getattr(self, "model", None) is not None
            and getattr(self.model, "is_trained", False)
        )
        tuner_ready = bool(
            getattr(self, "pid_tuner_trained", False)
            and getattr(self, "pid_tuner", None) is not None
            and getattr(self.pid_tuner, "is_trained", False)
        )
        meta_running = bool(
            getattr(self, "meta_tuner_thread", None)
            and self.meta_tuner_thread.isRunning()
        )
        tune_running = bool(
            getattr(self, "ai_tune_thread", None)
            and self.ai_tune_thread.isRunning()
        )

        if hasattr(self, "btn_train_ai"):
            self.btn_train_ai.setEnabled(plant_ready and not meta_running and not tune_running)
        if hasattr(self, "btn_ai_tune"):
            self.btn_ai_tune.setEnabled(
                plant_ready and tuner_ready and not meta_running and not tune_running
            )
        if hasattr(self, "btn_apply_ai"):
            self.btn_apply_ai.setEnabled(
                bool(getattr(self, "ai_last_verified", False))
                and not meta_running and not tune_running
            )

        if hasattr(self, "lbl_ai_status"):
            if not plant_ready:
                self.lbl_ai_status.setText(
                    "AI tuner: cần Train/Load Plant model trước"
                )
            elif not tuner_ready:
                self.lbl_ai_status.setText(
                    "AI tuner: Plant ready → có thể TRAIN AI PID TUNER"
                )

    def _ai_baseline(self):
        return (self.ai_base_kp.value(), self.ai_base_ki.value(), self.ai_base_kd.value())

    def _ai_bounds(self):
        return np.asarray([
            [0.0, self.ai_kp_max.value()],
            [0.0, self.ai_ki_max.value()],
            [0.0, self.ai_kd_max.value()],
        ], dtype=np.float64)

    def start_ai_tuner_training(self):
        try:
            if not self.is_trained or not self.model.is_trained:
                self.txt_ai_log.setText(
                    "Train/load Hybrid TCN plant model trước."
                )
                self._refresh_ai_controls()
                return
            if self.meta_tuner_thread and self.meta_tuner_thread.isRunning():
                return

            target = float(self.ai_target.value())
            if abs(target) < 1e-6:
                self.txt_ai_log.setText("Target RPM phải khác 0.")
                self._refresh_ai_controls()
                return

            b = self._ai_bounds()
            self.pid_tuner = PIDGainNet(
                kp_bounds=tuple(b[0]),
                ki_bounds=tuple(b[1]),
                kd_bounds=tuple(b[2]),
            )
            self.pid_tuner_trained = False
            self.ai_last_verified = False
            self.txt_ai_log.clear()
            self.lbl_ai_live.setText(
                "Preparing V4.2 AI meta-training..."
            )
            self.txt_ai_log.append(
                "V4.2 meta-training: diagnostic response -> PIDGainNet -> "
                "differentiable closed-loop loss; objective includes +RPM and -RPM."
            )

            self.meta_tuner_thread = MetaTunerTrainingThread(
                self.model,
                self.pid_tuner,
                baseline_gains=self._ai_baseline(),
                target_rpm=target,
                epochs=self.ai_epochs.value(),
                batch_size=self.ai_batch.value(),
                learning_rate=self.ai_lr.value(),
                random_severity=self.ai_random.value(),
                device_preference=self.ai_device.currentText().strip().lower(),
                validate_every=V4_AI_VALIDATE_EVERY_EPOCHS,
            )
            self.meta_tuner_thread.progress.connect(
                self.on_ai_train_progress
            )
            self.meta_tuner_thread.live_status.connect(
                self.on_ai_live_status
            )
            self.meta_tuner_thread.finished_training.connect(
                self.on_ai_train_finished
            )
            self.meta_tuner_thread.error_occurred.connect(
                self.on_ai_train_error
            )

            # Disable/enable buttons from one central place.
            self._refresh_ai_controls()
            self.meta_tuner_thread.start()
            # isRunning() may not turn True until immediately after start,
            # so explicitly disable the two tuning buttons here.
            self.btn_train_ai.setEnabled(False)
            self.btn_ai_tune.setEnabled(False)
            self.btn_apply_ai.setEnabled(False)

        except Exception as e:
            self.lbl_ai_live.setText("ERROR")
            self.txt_ai_log.append(
                f"AI START ERROR: {type(e).__name__}: {e}"
            )
            self._refresh_ai_controls()

    def on_ai_train_progress(self, pct, msg):
        self.progress_bar.setValue(pct)
        self.txt_ai_log.append(msg)

    def on_ai_live_status(self, pct, msg):
        self.progress_bar.setValue(pct)
        self.lbl_ai_live.setText(msg)

    def on_ai_train_finished(self, tuner, stats):
        self.pid_tuner = tuner
        self.pid_tuner_trained = True
        self.lbl_ai_status.setText(
            f"AI tuner trained | best val objective={stats.get('best_val_objective', float('nan')):.5f}"
        )
        self.lbl_ai_status.setStyleSheet("font-weight:bold;color:#16a085;")
        self.txt_ai_log.append("\n=== AI TUNER V4.2 ===")
        self.txt_ai_log.append(json.dumps(stats, indent=2))
        self.lbl_ai_live.setText("DONE - AI PID tuner trained")
        self._refresh_ai_controls()

    def on_ai_train_error(self, msg):
        self.pid_tuner_trained = False
        self.lbl_ai_live.setText("ERROR")
        self.txt_ai_log.append(f"AI TRAIN ERROR: {msg}")
        self._refresh_ai_controls()

    def start_ai_hybrid_tuning(self):
        if not self.pid_tuner_trained or self.pid_tuner is None or not self.pid_tuner.is_trained:
            self.txt_ai_log.append("Train/load AI PID tuner trước.")
            return
        if not self.model.is_trained:
            self.txt_ai_log.append("Plant model chưa sẵn sàng.")
            return
        if self.ai_tune_thread and self.ai_tune_thread.isRunning():
            return

        self.ai_last_verified = False
        self.btn_apply_ai.setEnabled(False)
        self.btn_ai_tune.setEnabled(False)
        self.txt_ai_log.append("\nRunning AI proposal + local DE refinement...")
        self.ai_tune_thread = AIHybridTuneThread(
            self.model,
            self.pid_tuner,
            baseline_gains=self._ai_baseline(),
            target_rpm=float(self.ai_target.value()),
            duration=self.ai_duration.value(),
            global_bounds=self._ai_bounds(),
            refine_fraction=self.ai_refine_pct.value(),
            refine_evals=self.ai_refine_evals.value(),
        )
        self.ai_tune_thread.progress.connect(self.on_ai_tune_progress)
        self.ai_tune_thread.finished_tuning.connect(self.on_ai_tune_finished)
        self.ai_tune_thread.error_occurred.connect(self.on_ai_tune_error)
        self.ai_tune_thread.start()

    def on_ai_tune_progress(self, pct, msg):
        self.progress_bar.setValue(pct)
        if pct % 5 == 0 or pct < 25 or pct >= 90:
            self.txt_ai_log.append(msg)

    def on_ai_tune_finished(self, result):
        self.btn_ai_tune.setEnabled(True)
        ai = result["ai_candidate"]
        final = result["final_pid"]
        self.optimized_pid = tuple(map(float, final))
        self.optimized_metrics = result["final_metrics"]
        self.ai_last_verified = bool(result["verified"])

        self.txt_ai_log.append("\n=== V4 RESULT ===")
        self.txt_ai_log.append(
            f"AI direct: Kp={ai[0]:.6f} Ki={ai[1]:.6f} Kd={ai[2]:.6f} "
            f"score={result['ai_score']:.4f}"
        )
        self.txt_ai_log.append(
            f"Final:     Kp={final[0]:.6f} Ki={final[1]:.6f} Kd={final[2]:.6f} "
            f"score={result['final_score']:.4f}"
        )
        m = result["final_metrics"]
        if m is not None:
            self.txt_ai_log.append(
                f"OS={m.overshoot_pct:.2f}% | Tr={m.rise_time_s:.3f}s | "
                f"Ts={m.settling_time_s:.3f}s | SSE={m.steady_state_error_pct:.2f}% | "
                f"sat={100*m.saturation_fraction:.1f}% | OOD={m.ood_mean:.4f}"
            )

        if self.ai_last_verified:
            self.lbl_ai_status.setText("AI candidate: VERIFIED in learned simulator")
            self.lbl_ai_status.setStyleSheet("font-weight:bold;color:#27ae60;")
            self.btn_apply_ai.setEnabled(True)
            self.txt_ai_log.append("VERIFY PASS. Vẫn nên test target thấp trên hardware trước.")
        else:
            reasons = result.get("verify_reasons", [])
            self.lbl_ai_status.setText("AI candidate: VERIFY FAILED")
            self.lbl_ai_status.setStyleSheet("font-weight:bold;color:#c0392b;")
            self.btn_apply_ai.setEnabled(False)
            self.txt_ai_log.append("VERIFY FAIL: " + "; ".join(reasons))

        self.progress_bar.setValue(100)

    def on_ai_tune_error(self, msg):
        self._refresh_ai_controls()
        self.txt_ai_log.append(f"AI TUNE ERROR: {msg}")

    def save_ai_tuner(self):
        if self.pid_tuner is None or not self.pid_tuner.is_trained:
            self.txt_ai_log.append("AI tuner chưa train.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save V4 AI PID Tuner", "pid_gain_net_v4.pth", "PyTorch Model (*.pth)"
        )
        if not path:
            return
        self.pid_tuner.to("cpu")
        ckpt = {
            "format_version": 4,
            "model_type": "PIDGainNet",
            "model_config": self.pid_tuner.model_config,
            "model_state": self.pid_tuner.state_dict(),
            "training_meta": self.pid_tuner.training_meta,
            "is_trained": True,
        }
        torch.save(ckpt, path)
        self.txt_ai_log.append(f"AI tuner saved: {path}")

    def load_ai_tuner(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load V4 AI PID Tuner", "", "PyTorch Model (*.pth)"
        )
        if not path:
            return
        try:
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                ckpt = torch.load(path, map_location="cpu")
            if int(ckpt.get("format_version", 0)) != 4 or ckpt.get("model_type") != "PIDGainNet":
                raise ValueError("Không phải V4 PIDGainNet checkpoint.")
            self.pid_tuner = PIDGainNet(**ckpt["model_config"])
            self.pid_tuner.load_state_dict(ckpt["model_state"])
            self.pid_tuner.training_meta = ckpt.get("training_meta", {})
            self.pid_tuner.is_trained = True
            self.pid_tuner.eval()
            self.pid_tuner_trained = True
            self._refresh_ai_controls()
            self.lbl_ai_status.setText("AI tuner loaded")
            self.lbl_ai_status.setStyleSheet("font-weight:bold;color:#16a085;")
            self.txt_ai_log.append(f"AI tuner loaded: {path}")
        except Exception as e:
            self.txt_ai_log.append(f"AI LOAD ERROR: {e}")

    def apply_ai_pid(self):
        if not self.ai_last_verified or self.optimized_pid is None:
            self.txt_ai_log.append("PID chưa VERIFY PASS; không apply.")
            return
        if not self.data_thread.running:
            self.txt_ai_log.append("ESP32 chưa connected.")
            return
        kp, ki, kd = self.optimized_pid
        answer = QMessageBox.question(
            self,
            "Apply V4 verified PID",
            f"Apply Kp={kp:.6f}, Ki={ki:.6f}, Kd={kd:.6f} to ESP32?\n\n"
            "Candidate đã pass learned-simulator verify, nhưng vẫn nên test RPM thấp trước.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.data_thread.send_command(f"pid={kp},{ki},{kd}")
        self.input_kp.setText(f"{kp:.6f}")
        self.input_ki.setText(f"{ki:.6f}")
        self.input_kd.setText(f"{kd:.6f}")
        self.txt_ai_log.append("Verified V4 PID applied to ESP32.")

    # --------------------------------------------------------
    # GUI UPDATE / CLOSE
    # --------------------------------------------------------

    def update_gui(self):
        if self.realtime_data["time"]:
            t = self.realtime_data["time"]
            self.curve_rpm.setData(t, self.realtime_data["rpm"])
            self.curve_target.setData(t, self.realtime_data["target"])
            self.curve_pwm.setData(t, [p / 10.0 for p in self.realtime_data["pwm"]])

    def closeEvent(self, event):
        self.recording = False
        if self.train_thread and self.train_thread.isRunning():
            self.train_thread.request_stop()
            self.train_thread.wait(1500)
        if self.opt_thread and self.opt_thread.isRunning():
            self.opt_thread.request_stop()
            self.opt_thread.wait(1500)
        if self.meta_tuner_thread and self.meta_tuner_thread.isRunning():
            self.meta_tuner_thread.request_stop()
            self.meta_tuner_thread.wait(1500)
        if self.ai_tune_thread and self.ai_tune_thread.isRunning():
            self.ai_tune_thread.request_stop()
            self.ai_tune_thread.wait(1500)
        self.data_thread.disconnect()
        if self.data_thread.isRunning():
            self.data_thread.wait(1000)
        event.accept()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PIDTrainerGUI()
    window.show()
    sys.exit(app.exec())