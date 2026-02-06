# Training loop
from agent.ppo_agent import PPOAgent
from env.snake_env import SnakeEnv
import os, time

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
EPOCHS = 2000  # Training epochs
SAVING_FREQUENCY = 500  # Interval between model saves
VISUALIZE_FREQUENCY = 50  # Interval between games (played by the model) shown
BUFFER_SIZE = 2048  # Number of stepls the model plays before learning

log_data = {
    'episodes': [],
    'rewards': [],
    'steps': [],
    'avg_values': []
}



if __name__ == "__main__":
    # Making the dirs
    os.makedirs(os.path.join('agent', 'resoults', 'config'), exist_ok=True)
    os.makedirs(os.path.join('agent', 'resoults', 'checkpoints'), exist_ok=True)
    
    env = SnakeEnv()
    agent = PPOAgent(obs_dim=229, action_dim=3, config=config)

    # Saving config
    import json
    with open(os.path.join('agent', 'resoults', 'config', 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=4)

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

        # Save episode logs
        log_data['episodes'].append(episode)
        log_data['rewards'].append(total_reward)
        log_data['steps'].append(len(buffer.rewards))
        log_data['avg_values'].append(avg_value)

        # Save checkpoint
        if episode % SAVING_FREQUENCY == 0 and episode != EPOCHS:
            agent.save(path='agent/resoults/checkpoints', model_name=f'checkpoint_{episode}.pth')
        
        # Visualize play
        if episode % VISUALIZE_FREQUENCY == 0:
            env.reset()
            stop = False
            while not stop: 
                action, log_prob, value = agent.select_actions(env._get_obs())
                _, _, termin, trunc, _ = env.step(action)
                env.render()

                stop = termin or trunc
                time.sleep(1.3)
                
        # Make agent learn
        agent.ppo_update()
            

    # Saving final model
    agent.save(model_name='final_model')

    # Saving training logs
    with open(os.path.join('agent', 'resoults', 'logs', 'training_log.json'), 'w') as f:
        json.dump(log_data, f, indent=4)