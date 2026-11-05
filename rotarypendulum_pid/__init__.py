"""QUBE-Servo 2 rotary pendulum with PPO-tuned cascade PID gains."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Rotary-Pendulum-PID",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rotary_env_cfg:PidEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PidPPORunnerCfg",
    },
)
