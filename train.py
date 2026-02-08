# Training loop
from agent.ppo_agent import PPOAgent
from env.snake_env import SnakeEnv
import os, time

# ================ CONFIGS ================
config = {
    'gamma': 0.95,      # discount factor
    'lambda': 0.90,     # GAE lambda
    'epsilon': 0.15,     # clip parameter
    'lr': 2e-4,         # learning rate
    'epochs': 4,       # training epochs per update
    'batch_size': 32,
    'value_coef': 0.5,  # value loss coefficient
    'entropy_coef': 0.1
}
EPOCHS = 50000  # Training epochs
SAVING_FREQUENCY = 1000  # Interval between model saves
VISUALIZE_FREQUENCY = 500  # Interval between games (played by the model) shown
BUFFER_SIZE = 1024  # Number of stepls the model plays before learning

log_data = {
    'episodes': [],
    'rewards': [],
    'avg_rewards': [],
    'avg_scores': [],
    'steps': [],
    'avg_values': []
}



if __name__ == "__main__":
    # Making the dirs
    os.makedirs(os.path.join('resoults', 'config'), exist_ok=True)
    os.makedirs(os.path.join('resoults', 'checkpoints'), exist_ok=True)
    
    env = SnakeEnv()
    agent = PPOAgent(obs_dim=env.OBS_LENGTH, action_dim=env.ACTION_LENGTH, config=config)

    # Saving config
    import json
    with open(os.path.join('resoults', 'config', 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # =============== TRAINING LOOP ===============
    for episode in range(1, EPOCHS + 1):
        print(f"\n=== Episode {episode}/{EPOCHS} ===")

        # Get expiriances from env
        buffer = agent.collect_rollout(env, num_steps=BUFFER_SIZE)

        # Calculate episode stats
        total_reward = sum(buffer.rewards)
        avg_reward = sum(buffer.rewards) / len(buffer.rewards) if buffer.rewards else 0
        avg_score = sum(buffer.scores) / len(buffer.scores) if buffer.values else 0
        avg_value = sum(buffer.values) / len(buffer.values) if buffer.values else 0
        
        print(f"\033[90mTotal reward: {total_reward:.2f}\033[0m")
        print(f"Average reward: \033[92m{avg_reward:.2f}\033[0m")
        print(f"Average score: \033[93m{avg_score:.2f}\033[0m")
        print(f"\033[90mSteps collected: {len(buffer.rewards)}\033[0m")

        # Save episode logs
        log_data['episodes'].append(episode)
        log_data['rewards'].append(total_reward)
        log_data['avg_rewards'].append(avg_reward)
        log_data['avg_scores'].append(avg_score)
        log_data['steps'].append(len(buffer.rewards))
        log_data['avg_values'].append(avg_value)

        # Save checkpoint
        if episode % SAVING_FREQUENCY == 0 and episode != EPOCHS:
            agent.save(path='resoults/checkpoints', model_name=f'checkpoint_{episode}')
        
        # Visualize play
        if episode % VISUALIZE_FREQUENCY == 0:
            env.reset()
            stop = False
            while not stop: 
                action, log_prob, value = agent.select_actions(env._get_obs())
                _, _, termin, trunc, _ = env.step(action)
                env.render()

                stop = termin or trunc
                time.sleep(0.2)
                
        # Make agent learn
        agent.ppo_update()
            

    # Saving final model
    agent.save(model_name='final_model')

    # Saving training logs
    os.mkdir(os.path.join('resoults', 'logs'))
    with open(os.path.join('resoults', 'logs', 'training_log.json'), 'w') as f:
        json.dump(log_data, f, indent=4)