# Isaac Lab Custom Robot Assets

<p align="center">
  <b>Custom Robot Assets for Isaac Sim and Isaac Lab</b><br>
  Robotics Simulation · Reinforcement Learning · Control System Development
</p>

---

## 1. General Overview

This repository contains custom robot asset packages designed for simulation and reinforcement learning development using **NVIDIA Isaac Sim** and **Isaac Lab**.

The main purpose of this project is to prepare reusable robot assets that can be integrated into Isaac Lab for:

- Physics-based robot simulation
- Reinforcement learning environment development
- Control policy training
- Robot behavior testing
- System modeling and validation

At the current stage, this README focuses on the **general setup workflow** for adding custom robot assets into Isaac Lab.  
Detailed documentation for each specific robot system will be added in later sections.

---

## 2. About Isaac Sim

**Isaac Sim** is a robotics simulation platform used to build, simulate, and test robotic systems in a high-fidelity physics environment.

It supports:

- Robot model simulation
- Rigid body and joint physics
- Sensor simulation
- USD-based robot assets
- Realistic visualization
- GPU-accelerated simulation

Isaac Sim is commonly used in robotics research and development to validate robot behavior before deploying control algorithms to real hardware.

---

## 3. About Isaac Lab

**Isaac Lab** is a robot learning framework built on top of Isaac Sim.

It provides tools for creating reinforcement learning environments, defining robot assets, configuring actuators, designing observations and rewards, and training policies using learning algorithms such as PPO.

In this project, Isaac Lab is used as the main framework for connecting custom robot assets with simulation tasks and reinforcement learning workflows.

---

## 4. Repository Structure

This repository currently contains the following asset folders:

```bash
.
├── balancerac_v1
└── rotarypendulum_v1
```

These folders are custom robot asset packages that should be added into the Isaac Lab asset directory.

At this stage:

- `balancerac_v1` is kept as a custom robot asset package.
- `rotarypendulum_v1` is kept as a custom robot asset package.
- Detailed explanation for each robot system will be documented later.

---

## 5. Target Isaac Lab Asset Structure

After adding the custom folders into Isaac Lab, the recommended structure should look like this:

```bash
IsaacLab
└── source
    └── isaaclab_assets
        └── isaaclab_assets
            └── robots
                ├── balancerac_v1
                ├── rotarypendulum_v1
                └── __init__.py
```

The two folders must be placed inside:

```bash
IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/
```

---

## 6. Add Custom Assets into Isaac Lab

### Linux

From the folder that contains `balancerac_v1` and `rotarypendulum_v1`, run:

```bash
cp -r balancerac_v1 IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/
cp -r rotarypendulum_v1 IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/
```

### Windows

Copy these two folders manually:

```bash
balancerac_v1
rotarypendulum_v1
```

Paste them into:

```bash
IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/
```

The final result should be:

```bash
IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/
├── balancerac_v1
├── rotarypendulum_v1
└── __init__.py
```

---

## 7. Register Custom Assets

Open the following file:

```bash
IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/__init__.py
```

Add the following lines:

```python
from .balancerac_v1 import *
from .rotarypendulum_v1 import *
```

This step allows Isaac Lab to recognize and import the custom robot asset packages.

---

## 8. Check Folder Requirements

Each custom asset folder should contain an `__init__.py` file.

Example:

```bash
balancerac_v1
├── __init__.py
└── ...

rotarypendulum_v1
├── __init__.py
└── ...
```

If `__init__.py` is missing, Python may not recognize the folder as an importable package.

---

## 9. Test Asset Import

Go to the Isaac Lab root directory:

```bash
cd IsaacLab
```

### Linux

Test `balancerac_v1`:

```bash
./isaaclab.sh -p -c "from isaaclab_assets.robots.balancerac_v1 import *; print('balancerac_v1 imported successfully')"
```

Test `rotarypendulum_v1`:

```bash
./isaaclab.sh -p -c "from isaaclab_assets.robots.rotarypendulum_v1 import *; print('rotarypendulum_v1 imported successfully')"
```

### Windows

Test `balancerac_v1`:

```bash
isaaclab.bat -p -c "from isaaclab_assets.robots.balancerac_v1 import *; print('balancerac_v1 imported successfully')"
```

Test `rotarypendulum_v1`:

```bash
isaaclab.bat -p -c "from isaaclab_assets.robots.rotarypendulum_v1 import *; print('rotarypendulum_v1 imported successfully')"
```

Expected output:

```bash
balancerac_v1 imported successfully
rotarypendulum_v1 imported successfully
```

---

## 10. Verify Isaac Lab Environment List

After registering the assets, check whether the related environments are available:

### Linux

```bash
./isaaclab.sh -p scripts/environments/list_envs.py
```

### Windows

```bash
isaaclab.bat -p scripts/environments/list_envs.py
```

If custom tasks have already been registered, they should appear in the environment list.

---

## 11. General Training Command Format

After the environment is registered, training can be started using the Isaac Lab reinforcement learning scripts.

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

Example:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-CustomTask-v0 \
    --num_envs 64
```

Replace `<TASK_NAME>` with the exact task name registered in Isaac Lab.

---

## 12. General Policy Test Command Format

After training, the learned policy can be tested using:

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

This command loads the trained policy and runs the robot behavior in simulation.

---

## 13. Recommended Development Workflow

A recommended workflow for developing custom Isaac Lab robot assets is:

```bash
1. Prepare robot model and asset folder
2. Add asset folder into Isaac Lab
3. Register asset import in __init__.py
4. Test Python import
5. Register simulation environment
6. Check environment list
7. Run simulation test
8. Train reinforcement learning policy
9. Play trained policy
10. Tune observations, rewards, and actuator settings
```

---

## 14. Common Issues

### Asset import error

If the asset cannot be imported, check:

- The folder is placed in the correct Isaac Lab robot asset directory.
- The folder contains an `__init__.py` file.
- The import line is added correctly in `robots/__init__.py`.
- The folder name in the import statement matches the actual folder name.

Correct format:

```python
from .balancerac_v1 import *
from .rotarypendulum_v1 import *
```

---

### Task not found

If Isaac Lab shows:

```bash
Task not found
```

Check whether the task has been registered correctly.

Run:

```bash
./isaaclab.sh -p scripts/environments/list_envs.py
```

Then use the exact task name shown in the environment list.

---

### Simulation opens but robot does not move

Check the following configuration items:

- USD file path
- Articulation root path
- Joint names
- Actuator configuration
- Action scale
- Initial joint position
- Observation configuration
- Reward function
- Termination conditions

---

## 15. Future Documentation

Detailed technical documentation will be added later for each robot system.

Planned sections:

```bash
16. Balancer Robot Asset
17. Rotary Inverted Pendulum Asset
18. Reinforcement Learning Task Configuration
19. Observation and Action Space
20. Reward Function Design
21. Training Results
22. Policy Evaluation
```

These sections will describe the robot structure, control objective, reinforcement learning setup, and simulation results in more detail.

---

## 16. Notes

This repository is intended for simulation-based robotics research and development.

Before applying any trained policy to real hardware, it is necessary to validate:

- Joint limits
- Torque limits
- Motor safety constraints
- Sensor feedback reliability
- Control frequency
- Emergency stop behavior
- Real-world testing conditions

---

## 17. Author

Developed for robotics simulation, reinforcement learning, and control system research using Isaac Sim and Isaac Lab.

