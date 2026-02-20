# Script to visualize trained model playing

from agent.ppo_agent import PPOAgent
from env.snake_env import SnakeEnv
import os, time, argparse


PATH = os.path.join('resoults', 'model', 'final_model.pth')


if __name__ == "__main__":
    # Handle params
    parser = argparse.ArgumentParser(description='trainig arguments')
    
    parser.add_argument('--path', type=int, default=None, help='Path to the model file (file name included). It can be both absolute or relative to project\'s root folder')
    parser.add_argument('--disable-gui', type=bool, default=False, help='no value needed, deactivates gui on env (it will use the cli)')
    args = parser.parse_args()
    
    if args.path is not None:
        PATH = args.path
    use_gui = not args.disable_gui


    # Initialize env and agent
    env = SnakeEnv(useGui=use_gui)
    agent = PPOAgent(env)

    # Load weights
    print('Loading weights...')
    agent.load(PATH)

    # Play game
    obs, _ = env.reset()
    stop = False
    while not stop: 
        action, log_prob = agent.get_action(obs)
        obs, _, termin, trunc, _ = env.step(action)
        env.render()

        stop = termin or trunc
        time.sleep(0.2)


    # Close env
    env.close()