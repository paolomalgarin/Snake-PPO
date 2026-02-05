# Wrapper Gymnasium

from gymnasium import Env, spaces
import numpy as np
from snake_game import SnakeGame, Point, Direction


class SnakeEnv(Env):
    
    def __init__(self):
        self.game = SnakeGame()
        self.action_space = spaces.Discrete(3) # 3 actions: move forward, turn right, turn left
        self.observation_space = spaces.Box({
            "grid": spaces.Box(
                # grid with 0 = empty squares, 1 = snake body, 2 = snake head, 3 = food
                low=0,
                high=3,
                shape=(self.game.gridHeight, self.game.gridWidth),
                dtype=np.int32
            ),
            "direction": spaces.Discrete(4)  # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
        })
        self.max_steps = self.game.gridHeight * self.game.gridWidth
        self.prev_score = self.game.score
        self.prev_food_distance = self.game.getFoodDistance()
    
    def reset(self):
        self.game.reset()
        self.steps = 0
        self.prev_score = self.game.score
        self.prev_food_distance = self.game.getFoodDistance()
        self.max_steps = self.game.gridHeight * self.game.gridWidth

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
        grid = np.zeros((self.game.gridHeight, self.game.gridWidth), dtype=np.int32)

        for row in range(self.game.gridHeight):
            for col in range(self.game.gridWidth):
                currentPoint = Point(col, row)
                
                if(currentPoint in self.game.body):
                    grid[row][col] = 1
                elif(currentPoint == self.game.head):
                    grid[row][col] = 2
                elif(currentPoint == self.game.food):
                    grid[row][col] = 3
                else:
                    grid[row][col] = 0
            
            
        direction_mapping = {
            Direction.UP: 0,
            Direction.DOWN: 1,
            Direction.LEFT: 2,
            Direction.RIGHT: 3
        }
        direction_value = direction_mapping[self.game.direction]
        
        return {
            "grid": grid,
            "direction": direction_value
        }
    
    def _compute_reward(self):
        reward = 0

        # -0.01 for each step to promote faster game
        reward -= 0.01
        
        # +10 if eats food (and add more steps)
        if self.game.score > self.prev_score:
            self.max_steps = self.steps +  self.game.gridHeight * self.game.gridWidth
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
        
        return reward