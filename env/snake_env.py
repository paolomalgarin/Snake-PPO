# Wrapper Gymnasium

from gymnasium import Env
from snake_game import SnakeGame


class SnakeEnv(Env):
    
    def __init__(self):
        self.game = SnakeGame()
    
    def step(self):
        pass