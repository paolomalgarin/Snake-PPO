# script to make the ppo agent learn cartople 
# (made to test ppo agent learning ability)

import gymnasium as gym
import os, time, sys
from pathlib import Path

# Adding the 'agent' dir to path
env_path = Path(__file__).parent.parent / 'agent'
sys.path.insert(0, str(env_path))

from ppo_agent import PPOAgent


# ================ CONFIGS ================
config = {
    'gamma': 0.99,      # discount factor
    'lambda': 0.95,     # GAE lambda
    'epsilon': 0.2,     # clip parameter
    'lr': 3e-4,         # learning rate
    'epochs': 10,       # training epochs per update
    'batch_size': 64,
    'value_coef': 0.5,  # value loss coefficient
    'entropy_coef': 0.01
}
EPOCHS = 50  # Training epochs
VISUALIZE_FREQUENCY = 50  # Interval between games (played by the model) shown
BUFFER_SIZE = 2048  # Number of stepls the model plays before learning



if __name__ == "__main__":    
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    agent = PPOAgent(obs_dim=4, action_dim=2, config=config)
    

    # =============== TRAINING LOOP ===============
    for episode in range(1, EPOCHS + 1):
        print(f"\n=== Episode {episode}/{EPOCHS} ===")

        # Get expiriances from env
        buffer = agent.collect_rollout(env, num_steps=BUFFER_SIZE)

        # Calculate episode stats
        total_reward = sum(buffer.rewards)
        avg_value = sum(buffer.values) / len(buffer.values) if buffer.values else 0
        
        print(f"Total reward: {total_reward:.2f}")
        print(f"Average value: {avg_value:.2f}")
        print(f"Steps collected: {len(buffer.rewards)}")

        # Make agent learn
        agent.ppo_update()
    
    # Close the env
    env.close()

    # Testing trained model    
    test_env = gym.make("CartPole-v1", render_mode="human")
    obs, _ = test_env.reset()
    stop = False
    total_reward = 0
    step_count = 0

    while not stop:
        action, log_prob, value = agent.select_actions(obs)
        obs, reward, termin, trunc, _ = test_env.step(action)
        total_reward += reward
        stop = termin
        step_count += 1
        
        time.sleep(0.033)
    print(f"Test Reward: {total_reward} in {step_count} steps")

    test_env.close()