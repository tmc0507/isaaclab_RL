# 🚀 Isaac Lab Custom Assets

<p align="center">
  <b>Custom Assets for NVIDIA Isaac Sim / Isaac Lab</b><br>
  Mô phỏng robot · Reinforcement Learning · PPO Training · Control System · Robotics AI
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Isaac%20Sim-Robotics%20Simulation-76B900?style=for-the-badge&logo=nvidia&logoColor=white">
  <img src="https://img.shields.io/badge/Isaac%20Lab-Robot%20Learning-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Reinforcement%20Learning-PPO-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/Language-Python-yellow?style=for-the-badge&logo=python&logoColor=white">
</p>

---

## 📌 Mục lục

- [1. Giới thiệu chung](#1-giới-thiệu-chung)
- [2. Link tải và tài liệu cài đặt](#2-link-tải-và-tài-liệu-cài-đặt)
- [3. Isaac Sim là gì?](#3-isaac-sim-là-gì)
- [4. Isaac Lab là gì?](#4-isaac-lab-là-gì)
- [5. Cấu trúc asset hiện tại](#5-cấu-trúc-asset-hiện-tại)
- [6. Vị trí đặt asset trong Isaac Lab](#6-vị-trí-đặt-asset-trong-isaac-lab)
- [7. Thêm asset vào `__init__.py`](#7-thêm-asset-vào-__init__py)
- [8. Test import asset](#8-test-import-asset)
- [9. Kiểm tra environment](#9-kiểm-tra-environment)
- [10. Lệnh train tổng quát](#10-lệnh-train-tổng-quát)
- [11. Lệnh play policy](#11-lệnh-play-policy)
- [12. Quy trình phát triển đề xuất](#12-quy-trình-phát-triển-đề-xuất)
- [13. Lỗi thường gặp](#13-lỗi-thường-gặp)
- [14. Phần phát triển sau](#14-phần-phát-triển-sau)

---

## 1. Giới thiệu chung

Repository này chứa các **custom asset** được xây dựng để sử dụng trong **NVIDIA Isaac Sim** và **Isaac Lab**.

Mục tiêu của project là chuẩn bị các asset có thể tích hợp trực tiếp vào Isaac Lab để phục vụ cho:

- Mô phỏng hệ thống robot trong môi trường vật lý.
- Xây dựng môi trường Reinforcement Learning.
- Train policy điều khiển bằng PPO.
- Kiểm tra hành vi robot trong simulation.
- Phát triển thuật toán điều khiển trước khi triển khai thực nghiệm.

Ở giai đoạn đầu, README này chỉ tập trung vào phần **cài đặt, thêm asset và kiểm tra import**.

Phần mô tả chi tiết cho từng hệ thống như:

- Xe cân bằng
- Con lắc ngược quay

sẽ được trình bày ở các phần sau.

---

## 2. Link tải và tài liệu cài đặt

Trước khi sử dụng asset trong repository này, cần cài đặt Isaac Sim và Isaac Lab đúng phiên bản tương thích.

| Nội dung | Link chính thức |
|---|---|
| NVIDIA Isaac Sim | [https://developer.nvidia.com/isaac/sim](https://developer.nvidia.com/isaac/sim) |
| Isaac Sim GitHub | [https://github.com/isaac-sim/IsaacSim](https://github.com/isaac-sim/IsaacSim) |
| NVIDIA Isaac Lab | [https://developer.nvidia.com/isaac/lab](https://developer.nvidia.com/isaac/lab) |
| Isaac Lab Documentation | [https://isaac-sim.github.io/IsaacLab/](https://isaac-sim.github.io/IsaacLab/) |
| Isaac Lab GitHub | [https://github.com/isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) |
| Isaac Lab Installation Guide | [https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) |

> **Lưu ý:** Isaac Sim và Isaac Lab cần đúng phiên bản tương thích.  
> Nên kiểm tra hướng dẫn cài đặt chính thức trước khi copy asset vào project.

---

## 3. Isaac Sim là gì?

**Isaac Sim** là nền tảng mô phỏng robot của NVIDIA, được xây dựng trên Omniverse.

Isaac Sim được sử dụng để mô phỏng, kiểm tra và đánh giá robot trong môi trường 3D có vật lý thực tế.

Các chức năng chính:

- Mô phỏng rigid body.
- Mô phỏng joint và articulation.
- Import robot model theo định dạng USD.
- Kiểm tra va chạm và chuyển động.
- Mô phỏng cảm biến.
- Hiển thị môi trường 3D trực quan.
- Hỗ trợ tăng tốc mô phỏng bằng GPU.

Trong project này, Isaac Sim đóng vai trò là **physics simulation backend**.

---

## 4. Isaac Lab là gì?

**Isaac Lab** là framework robot learning được xây dựng trên Isaac Sim.

Isaac Lab hỗ trợ quá trình phát triển môi trường học tăng cường cho robot, bao gồm:

- Khai báo robot asset.
- Cấu hình actuator.
- Thiết kế observation space.
- Thiết kế action space.
- Thiết kế reward function.
- Đăng ký environment.
- Train policy bằng các thuật toán Reinforcement Learning.

Trong project này, Isaac Lab đóng vai trò là framework chính để kết nối:

```bash
Robot Asset → Simulation Environment → Reinforcement Learning Policy
```

---

## 5. Cấu trúc asset hiện tại

Repository hiện tại gồm 2 folder asset:

```bash
.
├── balancerac_v1
└── rotarypendulum_v1
```

Trong đó:

| Folder | Vai trò |
|---|---|
| `balancerac_v1` | Custom asset cho hệ thống xe cân bằng |
| `rotarypendulum_v1` | Custom asset cho hệ thống con lắc ngược quay |

Ở README này, 2 folder trên chỉ được giới thiệu ở mức tổng quát.  
Phần giải thích chi tiết cấu trúc, joint, actuator, observation và reward sẽ được viết riêng ở phần sau.

---

## 6. Vị trí đặt asset trong Isaac Lab

Không cần tạo thêm folder `robots`.

Chỉ cần copy trực tiếp 2 folder asset vào trong package `isaaclab_assets`.

Cấu trúc mong muốn:

```bash
IsaacLab
└── source
    └── isaaclab_assets
        └── isaaclab_assets
            ├── balancerac_v1
            ├── rotarypendulum_v1
            └── __init__.py
```

Vị trí cần đặt asset:

```bash
IsaacLab/source/isaaclab_assets/isaaclab_assets/
```

---

## 6.1. Copy asset vào Isaac Lab

### Linux

Từ thư mục đang chứa 2 folder asset, chạy:

```bash
cp -r balancerac_v1 IsaacLab/source/isaaclab_assets/isaaclab_assets/
cp -r rotarypendulum_v1 IsaacLab/source/isaaclab_assets/isaaclab_assets/
```

### Windows

Copy thủ công 2 folder:

```bash
balancerac_v1
rotarypendulum_v1
```

Sau đó paste vào:

```bash
IsaacLab/source/isaaclab_assets/isaaclab_assets/
```

Sau khi copy xong, cấu trúc sẽ là:

```bash
IsaacLab/source/isaaclab_assets/isaaclab_assets/
├── balancerac_v1
├── rotarypendulum_v1
└── __init__.py
```

---

## 7. Thêm asset vào `__init__.py`

Mở file:

```bash
IsaacLab/source/isaaclab_assets/isaaclab_assets/__init__.py
```

Thêm các dòng sau:

```python
from .balancerac_v1 import *
from .rotarypendulum_v1 import *
```

Mục đích của bước này là giúp Python nhận diện các custom asset như một phần của package `isaaclab_assets`.

Sau khi thêm, có thể import asset theo dạng:

```python
from isaaclab_assets.balancerac_v1 import *
from isaaclab_assets.rotarypendulum_v1 import *
```

---

## 8. Test import asset

Di chuyển vào thư mục gốc của Isaac Lab:

```bash
cd IsaacLab
```

### Linux

Test `balancerac_v1`:

```bash
./isaaclab.sh -p -c "from isaaclab_assets.balancerac_v1 import *; print('balancerac_v1 imported successfully')"
```

Test `rotarypendulum_v1`:

```bash
./isaaclab.sh -p -c "from isaaclab_assets.rotarypendulum_v1 import *; print('rotarypendulum_v1 imported successfully')"
```

### Windows

Test `balancerac_v1`:

```bash
isaaclab.bat -p -c "from isaaclab_assets.balancerac_v1 import *; print('balancerac_v1 imported successfully')"
```

Test `rotarypendulum_v1`:

```bash
isaaclab.bat -p -c "from isaaclab_assets.rotarypendulum_v1 import *; print('rotarypendulum_v1 imported successfully')"
```

Kết quả mong muốn:

```bash
balancerac_v1 imported successfully
rotarypendulum_v1 imported successfully
```

Nếu xuất hiện kết quả trên, asset đã được Isaac Lab nhận diện thành công.

---

## 9. Kiểm tra environment

Sau khi asset đã được thêm vào Isaac Lab, kiểm tra danh sách environment bằng lệnh:

### Linux

```bash
./isaaclab.sh -p scripts/environments/list_envs.py
```

### Windows

```bash
isaaclab.bat -p scripts/environments/list_envs.py
```

Nếu environment đã được đăng ký đúng, task sẽ xuất hiện trong danh sách.

Ví dụ:

```bash
Isaac-CustomTask-v0
```

> Tên task thực tế phụ thuộc vào file cấu hình environment của project.

---

## 10. Lệnh train tổng quát

Sau khi environment đã được đăng ký, có thể train policy bằng script Reinforcement Learning của Isaac Lab.

### Linux

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task <TASK_NAME> \
    --num_envs <NUMBER_OF_ENVIRONMENTS>
```

### Windows

```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py ^
    --task <TASK_NAME> ^
    --num_envs <NUMBER_OF_ENVIRONMENTS>
```

Ví dụ:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-CustomTask-v0 \
    --num_envs 64
```

Giải thích tham số:

| Tham số | Ý nghĩa |
|---|---|
| `--task` | Tên environment đã đăng ký trong Isaac Lab |
| `--num_envs` | Số lượng môi trường mô phỏng chạy song song |

---

## 11. Lệnh play policy

Sau khi train xong, có thể chạy thử policy bằng lệnh:

### Linux

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task <TASK_NAME> \
    --num_envs 1
```

### Windows

```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py ^
    --task <TASK_NAME> ^
    --num_envs 1
```

Lệnh này dùng để kiểm tra hành vi robot sau khi policy đã được huấn luyện.

---

## 12. Quy trình phát triển đề xuất

Quy trình làm việc khuyến nghị:

```bash
1. Cài đặt Isaac Sim
2. Cài đặt Isaac Lab
3. Chuẩn bị custom asset folder
4. Copy asset vào isaaclab_assets
5. Thêm import trong __init__.py
6. Test import asset
7. Đăng ký environment
8. Kiểm tra environment list
9. Chạy simulation test
10. Train policy bằng PPO
11. Chạy thử policy đã train
12. Tuning observation, reward và actuator
```

---

## 13. Lỗi thường gặp

### 13.1. Không import được asset

Kiểm tra các điểm sau:

- Folder asset đã được đặt đúng trong `isaaclab_assets`.
- Tên folder trong lệnh import trùng với tên folder thực tế.
- Folder asset có file `__init__.py`.
- File `isaaclab_assets/__init__.py` đã thêm đúng import.

Cấu hình đúng:

```python
from .balancerac_v1 import *
from .rotarypendulum_v1 import *
```

---

### 13.2. Không tìm thấy task

Nếu Isaac Lab báo lỗi:

```bash
Task not found
```

Hãy kiểm tra lại danh sách environment:

```bash
./isaaclab.sh -p scripts/environments/list_envs.py
```

Sau đó dùng đúng tên task xuất hiện trong danh sách.

---

### 13.3. Simulation chạy nhưng robot không chuyển động

Kiểm tra các mục sau:

- Đường dẫn USD file.
- Articulation root path.
- Joint name.
- Actuator configuration.
- Action scale.
- Initial joint position.
- Observation space.
- Reward function.
- Termination condition.

---

### 13.4. Import đúng nhưng train lỗi

Một số nguyên nhân thường gặp:

- Environment chưa được register.
- Tên task truyền vào `--task` chưa đúng.
- Thiếu dependency của Isaac Lab.
- Sai đường dẫn asset.
- Sai tên joint trong actuator config.
- File USD chưa đúng articulation root.

---

## 14. Phần phát triển sau

Các phần chi tiết sẽ được bổ sung sau để README không bị quá dài ở phần cài đặt ban đầu.

Dự kiến bổ sung:

```bash
14.1. Balancer Robot Asset
14.2. Rotary Pendulum Asset
14.3. Cấu trúc USD
14.4. Joint và actuator
14.5. Observation space
14.6. Action space
14.7. Reward function
14.8. PPO training configuration
14.9. Training result
14.10. Policy evaluation
```

---

## 15. Ghi chú kỹ thuật

Repository này phục vụ cho nghiên cứu mô phỏng robot và học tăng cường.

Trước khi triển khai policy từ simulation sang phần cứng thật, cần kiểm tra kỹ:

- Giới hạn góc joint.
- Giới hạn torque.
- Giới hạn vận tốc.
- Tần số điều khiển.
- Độ ổn định của sensor feedback.
- Điều kiện an toàn khi robot mất cân bằng.
- Emergency stop.
- Sai số giữa mô phỏng và thực tế.

---

## 16. Tác giả

Project được phát triển cho mục đích nghiên cứu:

- Robotics Simulation
- Reinforcement Learning
- Control System
- PPO Training
- Isaac Sim / Isaac Lab

<p align="center">
  <b>Developed for Robotics AI Research</b>
</p>
