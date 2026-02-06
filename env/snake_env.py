# Wrapper Gymnasium

from gymnasium import Env, spaces
import numpy as np
from .snake_game import SnakeGame, Point, Direction


class SnakeEnv(Env):
    ACTION_LENGTH = 3
    OBS_LENGTH = 679

    def __init__(self):
        self.game = SnakeGame()
        self.action_space = spaces.Discrete(3) # 3 actions: move forward, turn right, turn left
        # OBS_LENGTH: array 1D with the entire grid 3 times (food, body, head) + the direction (4)
        self.OBS_LENGTH = self.game.gridHeight * self.game.gridWidth * 3 + 4
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.OBS_LENGTH, ),
            dtype=np.float32
        )
        self.max_steps = self.game.gridHeight * self.game.gridWidth * 10
        self.prev_score = self.game.score
        self.prev_food_distance = self.game.getFoodDistance()
        self.steps = 0
    
    def reset(self, seed=None):
        self.game.reset()
        self.steps = 0
        self.prev_score = self.game.score
        self.prev_food_distance = self.game.getFoodDistance()
        self.max_steps = self.game.gridHeight * self.game.gridWidth * 10

        obs = self._get_obs()
        info = {}

        return obs, info

    def step(self, action):
        self.steps += 1

        # 3 actions: 0 = move forward, 1 = turn right, 2 = turn left
        match action:
            case 0:
                newDir = self.game.direction
            case 1:
                newDir = self.game.direction.getRightTurn()
            case 2:
                newDir = self.game.direction.getRightTurn().getOpposite()

        self.game.changeDir(newDir)
        self.game.move()

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self.game.isGameOver
        truncated = self.steps >= self.max_steps

        info = {
            "score": self.game.score,
            "steps": self.steps
        }

        return obs, reward, terminated, truncated, info


    def render(self):
        #renders the current game state
        self.game.displayCMD() # (temporary)

    def close(self):
        # closes pygame and widows
        pass

    def _get_obs(self):
        grid = np.zeros((self.OBS_LENGTH), dtype=np.float32)
        
        gridLength = self.game.gridWidth * self.game.gridHeight
        bodyGridStart = 0
        headGridStart = gridLength
        foodGridStart = gridLength * 2
        directionStart = gridLength * 3

        for row in range(self.game.gridHeight):
            for col in range(self.game.gridWidth):
                currentPoint = Point(col, row)

                if(currentPoint in self.game.body):
                    grid[bodyGridStart + row * self.game.gridWidth + col] = 1
                elif(currentPoint == self.game.head):
                    grid[headGridStart + row * self.game.gridWidth + col] = 1
                elif(currentPoint == self.game.food):
                    grid[foodGridStart + row * self.game.gridWidth + col] = 1
            
            
        direction_mapping = {
            Direction.UP: 0,
            Direction.DOWN: 1,
            Direction.LEFT: 2,
            Direction.RIGHT: 3
        }
        grid[directionStart + direction_mapping[self.game.direction]] = 1
        
        return grid
    
    def _compute_reward(self):
        reward = 0

        # -0.01 for each step to promote faster game
        reward -= 0.01
        
        # +10 if eats food (and add more steps)
        if self.game.score > self.prev_score:
            self.prev_score = self.game.score
            reward += 10

        # +0.1 if gets closer to food, -0.1 if gets furder from food
        new_food_distance = self.game.getFoodDistance()
        if(new_food_distance < self.prev_food_distance):
            reward += 0.1
        else: 
            reward -= 0.1
        self.prev_food_distance = new_food_distance

        # -10 if dies
        if self.game.isGameOver:
            reward -= 10
        
        return float(reward)