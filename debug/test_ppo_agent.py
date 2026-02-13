# script to make the ppo agent learn cartople 
# (made to test ppo agent learning ability)

import gymnasium as gym
import os, time, sys
from pathlib import Path
from agent.ppo_agent import PPOAgent



TIMESTEPS = 100_000  # Training timestamps


if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    agent = PPOAgent(env)

    # Hyperparameters settings
    agent.timestamps_per_batch = 2048
    
    # Train the agent
    agent.learn(TIMESTEPS)

    env.close()

    # Testing trained model
    test_env = gym.make("CartPole-v1", render_mode="human")   
    obs, _ = test_env.reset()
    stop = False
    total_reward = 0
    step_count = 0

    while not stop:
        action, log_prob = agent.get_action(obs)
        obs, reward, termin, trunc, _ = test_env.step(action)
        total_reward += reward
        stop = termin
        step_count += 1
        
        time.sleep(0.033)
    print(f"Test Reward: {total_reward} in {step_count} steps")

    test_env.close()