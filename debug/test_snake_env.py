# script to make stable baseline 3 learn snake
# (made to test the env)
import gymnasium as gym

import os, time, sys, torch
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from env.snake_env import SnakeEnv

# ================ CONFIGS ================
config = {
    "gamma": 0.95,  # discount factor
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
STEPS = 1000000  # Training steps (actions taken)
EVAL_FREQ = 500  # Evaluation frequency
SAVE_FREQ = 100000  # Model saving frequency
VISUALIZE_FREQUENCY = 50  # Interval between games (played by the model) shown
BUFFER_SIZE = 2048  # Number of stepls the model plays before learning


if __name__ == "__main__":
    gym.register(
        id="SnakeEnv",
        entry_point=SnakeEnv,  # pyright: ignore
        max_episode_steps=15 * 15 * 10,
    )
    env = make_vec_env("SnakeEnv", n_envs=14, monitor_dir="debug/sb3_logs/monitor/")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PPO(
        "MlpPolicy",  # Using MLP policy for 1D observation
        env,
        policy_kwargs=dict(
            net_arch=[128 * 3, 128, 128, 32]
        ),  # 2 hidden layers of 128 neurons
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
    print("\n=== Testing Trained Model ===")

    # Load the best model (or use final model)
    # best_model_path = "debug/sb3_logs/best_model/best_model.zip"
    # if os.path.exists(best_model_path):
    #     model = PPO.load(best_model_path)
    # else:
    #     model = PPO.load("debug/sb3_logs/final_model.zip")
    

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
        time.sleep(0.8)

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
