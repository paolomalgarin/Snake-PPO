# script to make the ppo agent learn cartople 
# (made to test ppo agent learning ability)

import gymnasium as gym
import time, argparse
from agent.ppo_agent import PPOAgent


TIMESTEPS = 100_000  # Training timestamps


if __name__ == "__main__":

    # 
    parser = argparse.ArgumentParser(description='test_ppo_agent.py arguments')
    
    parser.add_argument('--ts', type=int, default=None, help='Total training timestamps')
    args = parser.parse_args()
    
    if args.ts is not None:
        TIMESTEPS = args.ts


    print("Creating environment...")
    
    # Agent and environment initialization
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    agent = PPOAgent(env)
    
    print("environment created!")
    
    # Agent hyperparameters settings
    agent.timestamps_per_batch = 2048
    

    # Agent training
    print("\n=== TRAINING ===")
    
    agent.learn(TIMESTEPS)

    env.close()


    # Trained agent testing
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