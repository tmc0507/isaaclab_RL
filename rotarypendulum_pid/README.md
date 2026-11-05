# Rotary pendulum PID

Task `Isaac-Rotary-Pendulum-PID`, tích hợp từ
`/home/cuong/RotaryPendulum-IsaacLab-master` vào package `isaaclab_assets`.
Task dùng QUBE-Servo 2: PPO xuất 6 hệ số cho hai bộ PID nối tầng,
12 quan sát; mô phỏng motor và vật lý 500 Hz, policy 100 Hz.

```text
rotarypendulum_pid/
├── __init__.py                 # Đăng ký Gym task
├── rotary_cfg.py               # Robot, motor và tần số điều khiển
├── rotary_env_cfg.py           # Scene, observation, action, reward, reset
├── agents/rsl_rl_ppo_cfg.py     # PPO
├── mdp/                        # PID, motor, command, observation, reward, termination
└── urdf_file/qube_servo2/       # URDF và STL; USD được sinh ở lần chạy đầu
```

## Train

Chạy từ workspace IsaacLab, với môi trường đã cài trên máy này:

```bash
cd /home/cuong/isaaclab/IsaacLab
conda activate env_isaaclab
source _isaac_sim/setup_conda_env.sh
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Rotary-Pendulum-PID --num_envs 16 --headless
```

Mặc định train 1500 iterations. Thêm `--max_iterations 2` để chạy thử nhanh.
Lệnh ghi checkpoint và TensorBoard vào `logs/rsl_rl/rotary_pendulum_pid/`.
Giữ `--num_envs 16` khi bắt đầu trên GPU 4 GB của máy này; chỉ tăng khi còn đủ VRAM.
Cấu hình nguồn vẫn có mặc định 4096 môi trường nếu bỏ tùy chọn này.

## Play

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Rotary-Pendulum-PID --num_envs 4 \
  --checkpoint logs/rsl_rl/rotary_pendulum_pid/<run>/model_<iteration>.pt
```

## Kiểm tra

Sau khi kích hoạt cùng môi trường ở trên:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./isaaclab.sh -p -m pytest \
  source/isaaclab_assets/test/test_rotary_pid.py -v -s
```

Tắt tự nạp plugin pytest để tránh plugin ROS của hệ thống dùng Python 3.12
trong môi trường Isaac Sim dùng Python 3.11. Các kiểm tra gồm mô hình motor,
khối lượng/quán tính/mesh URDF, đăng ký task, rollout hữu hạn và reset PID.

Bản tích hợp giữ thông số vật lý, reward và PID của nguồn; chuyển API
tensor và `set_joint_effort_target` sang phiên bản IsaacLab đang dùng.
Firmware, CAD STEP, script phân tích và checkpoint cũ vẫn nằm trong thư mục nguồn.
Kiểm tra khởi chạy train không đánh giá độ hội tụ hay hiệu quả trên phần cứng thật.
