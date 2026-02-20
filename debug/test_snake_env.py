# script to make stable baseline 3 learn snake
# (made to test the env)
import gymnasium as gym

import os, time, sys, torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from env.snake_env import SnakeEnv




class SmallCNN(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        
        input_channels = observation_space.shape[0]
        
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, features_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.conv1(obs)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        x = self.fc2(x)

        return x






# ================ CONFIGS ================
config = {
    "gamma": 0.99,  # discount factor
    "gae_lambda": 0.90,  # GAE lambda (note: in SB3 is 'gae_lambda', not 'lambda')
    "clip_range": 0.1,  # clip parameter (note: in SB3 is 'clip_range', not 'epsilon')
    "learning_rate": 1e-4,  # learning rate
    "n_epochs": 10,  # training epochs per update
    "batch_size": 256,
    "vf_coef": 0.5,  # value loss coefficient
    "ent_coef": 0.1,  # entropy coefficient
    "n_steps": 2048,  # number of steps per update
    "verbose": 1,
}
STEPS = 10e6  # Training steps (actions taken)


if __name__ == "__main__":
    gym.register(
        id="SnakeEnv",
        entry_point=SnakeEnv,  # pyright: ignore
        max_episode_steps=15 * 15 * 10,
    )
    env = make_vec_env("SnakeEnv", n_envs=14, monitor_dir="debug/sb3_logs/monitor/")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PPO(
        "CnnPolicy",    
        env,
        policy_kwargs=dict(
            features_extractor_class=SmallCNN,
            normalize_images=False,
        ),
        device=device,
        **config,
    )

    # Training with progress bar
    print("\n=== Starting Training ===")
    print(f"Total timesteps: {STEPS}")

    model.learn(
        total_timesteps=STEPS,
        progress_bar=True,
        log_interval=1,
    )

    # Save the final model
    model.save("debug/sb3_logs/final_model")

    env.close()

    # =============== TESTING TRAINED MODEL ===============

    # Visualize a game
    env = SnakeEnv(False)
    print("\n=== Visualizing Gameplay ===")
    obs, _ = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done1, done2, info = env.step(action)
        done = done1 or done2
        total_reward += reward
        step_count += 1

        # Render the game
        env.render()

        # Add small delay for visualization
        time.sleep(0.4)

        # Print game info every 10 steps
        if step_count % 10 == 0:
            print(
                f"Step: {step_count}, Reward: {reward:.2f}, Total: {total_reward:.2f}"
            )
            print(f"Score: {info.get('score', 0)}, Game Over: {done}")

    print(f"\n=== Game Finished ===")
    print(f"Final Score: {info.get('score', 0)}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Steps: {step_count}")


    # Close the env
    env.close()
