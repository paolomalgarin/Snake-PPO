# Visualizzazione modello allenato in partita

from agent.ppo_agent import PPOAgent
from env.snake_env import SnakeEnv
import os, time


PATH = os.path.join('resoults', 'final_model.pth')

if __name__ == "__main__":
    # Making the dirs
    env = SnakeEnv()
    agent = PPOAgent(obs_dim=env.OBS_LENGTH, action_dim=env.ACTION_LENGTH)

    agent.load(PATH)
    env.reset()
    stop = False
    while not stop: 
        action, log_prob, value = agent.select_actions(env._get_obs())
        _, _, termin, trunc, _ = env.step(action)
        env.render()

        stop = termin or trunc
        time.sleep(0.2)


    env.close()