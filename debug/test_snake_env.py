# script to make stable baseline 3 learn snake 
# (made to test the env)

import os, time, sys
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# Adding the 'agent' dir to path
env_path = Path(__file__).parent.parent / 'env'
sys.path.insert(0, str(env_path))

from snake_env import SnakeEnv

# ================ CONFIGS ================
config = {
    'gamma': 0.99,      # discount factor
    'gae_lambda': 0.95, # GAE lambda (note: in SB3 is 'gae_lambda', not 'lambda')
    'clip_range': 0.2,  # clip parameter (note: in SB3 is 'clip_range', not 'epsilon')
    'learning_rate': 3e-4, # learning rate
    'n_epochs': 10,     # training epochs per update
    'batch_size': 64,
    'vf_coef': 0.5,     # value loss coefficient
    'ent_coef': 0.01,   # entropy coefficient
    'n_steps': 2048,    # number of steps per update
    'verbose': 1

}
EPOCHS = 1000  # Training epochs
EVAL_FREQ = 50  # Evaluation frequency
SAVE_FREQ = 1000  # Model saving frequency
VISUALIZE_FREQUENCY = 50  # Interval between games (played by the model) shown
BUFFER_SIZE = 2048  # Number of stepls the model plays before learning



if __name__ == "__main__":    
    eval_env = SnakeEnv()
    train_env = Monitor(SnakeEnv())
    model = PPO(
        "MlpPolicy",  # Using MLP policy for 1D observation
        train_env,
        policy_kwargs=dict(net_arch=[128, 128]),  # 2 hidden layers of 128 neurons
        **config
    )
    

    # =============== TRAINING LOOP ===============
    # Create callbacks for evaluation and saving
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='debug/sb3_logs/best_model/',
        log_path='debug/sb3_logs/eval/',
        eval_freq=EVAL_FREQ,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path='debug/sb3_logs/checkpoints/',
        name_prefix='snake_ppo'
    )
    
    # Training with progress bar
    print("\n=== Starting Training ===")
    print(f"Total timesteps (epochs): {EPOCHS}")
    
    model.learn(
        total_timesteps=EPOCHS,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save the final model
    model.save("debug/sb3_logs/final_model")
    
    # Close training env
    train_env.close()
    
    # =============== TESTING TRAINED MODEL ===============
    print("\n=== Testing Trained Model ===")
    
    # Load the best model (or use final model)
    best_model_path = 'debug/sb3_logs/best_model/best_model.zip'
    if os.path.exists(best_model_path):
        model = PPO.load(best_model_path)
    else:
        model = PPO.load("debug/sb3_logs/final_model.zip")
    
    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(
        model, 
        eval_env, 
        n_eval_episodes=10, 
        deterministic=True
    )
    print(f"\nEvaluation Results:")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Visualize a game
    print("\n=== Visualizing Gameplay ===")
    obs, _ = eval_env.reset()
    done = False
    total_reward = 0
    step_count = 0
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1
        
        # Render the game
        eval_env.render()
        
        # Add small delay for visualization
        time.sleep(0.8)
        
        # Print game info every 10 steps
        if step_count % 10 == 0:
            print(f"Step: {step_count}, Reward: {reward:.2f}, Total: {total_reward:.2f}")
            print(f"Score: {info.get('score', 0)}, Game Over: {done}")
    
    print(f"\n=== Game Finished ===")
    print(f"Final Score: {info.get('score', 0)}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Steps: {step_count}")
    
    # Play a few more games to see performance
    print("\n=== Playing 5 More Games for Statistics ===")
    scores = []
    steps_per_game = []
    
    for game in range(5):
        obs, _ = eval_env.reset()
        done = False
        game_reward = 0
        game_steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            game_reward += reward
            game_steps += 1
        
        scores.append(info.get('score', 0))
        steps_per_game.append(game_steps)
        
        print(f"Game {game+1}: Score = {scores[-1]}, Steps = {game_steps}, Reward = {game_reward:.2f}")
    
    print(f"\n=== Summary Statistics ===")
    print(f"Average Score: {sum(scores)/len(scores):.2f}")
    print(f"Average Steps: {sum(steps_per_game)/len(steps_per_game):.2f}")
    print(f"Max Score: {max(scores)}")
    print(f"Min Score: {min(scores)}")
    
    # Close the env
    eval_env.close()