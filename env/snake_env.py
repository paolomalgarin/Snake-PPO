# Wrapper Gymnasium

from gymnasium import Env, spaces
import numpy as np
from env.snake_game import SnakeGame, Point, Direction


class SnakeEnv(Env):

    def __init__(self, useGui = False):
        self.game = SnakeGame(useGui=useGui)
        self.game.reset()
        
        self.useGui = useGui

        self.action_space = spaces.Discrete(
            4
        )  # 3 actions: up, down, right, left
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3, self.game.gridHeight, self.game.gridWidth), dtype=np.float32
        )

        self.max_steps = self.game.gridHeight * self.game.gridWidth * 10
        self.steps = 0

        self.prev_score = self.game.score
        self.prev_food_distance = self.game.getFoodDistance()

    def reset(self, seed=None, options=None):
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

        # 4 actions: 0 = up, 1 = down, 2 = right, 3 = left
        match action:
            case 0:
                newDir = Direction.UP
            case 1:
                newDir = Direction.DOWN
            case 2:
                newDir = Direction.RIGHT
            case 3:
                newDir = Direction.LEFT

        self.game.changeDir(newDir)
        self.game.move()

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self.game.isGameOver
        truncated = self.steps >= self.max_steps

        info = {"score": self.game.score, "steps": self.steps}

        return obs, reward, terminated, truncated, info

    def render(self):
        # renders the current game state
        if(not self.useGui):
            self.game.displayCMD()  # (temporary)
        else:
            self.game.drawWindow()

    def close(self):
        # closes pygame and widows
        self.game.close()

    def _get_obs(self):
        grid = np.zeros((self.observation_space.shape), dtype=np.float32)


        for row in range(self.game.gridHeight):
            for col in range(self.game.gridWidth):
                currentPoint = Point(col, row)

                if currentPoint == self.game.head:
                    grid[0][row][col] = 1.0
                elif currentPoint in self.game.body:
                    idx = self.game.body.index(currentPoint)
                    grid[1][row][col] = 1.0 + idx
                elif currentPoint == self.game.food:
                    grid[2][row][col] = 1.0

        return grid

    def _compute_reward(self):
        reward = 0
        
        if self.game.score > self.prev_score:
            self.prev_score = self.game.score
            reward += 1

        return float(reward)

    def _print_obs(self, obs):
        print("OBSERVATIONS:")
        for i in range(3):
            for j in range(15):
                for k in range(15):
                    toprint = (
                        "1"
                        if obs[(i * 15 * 15) + (j * 15 + k)] == 1
                        else (
                            "."
                            if obs[(i * 15 * 15) + (j * 15 + k)] == 0
                            else obs[(i * 15 * 15) + (j * 15 + k)]
                        )
                    )
                    print(toprint, end=" ")
                print()
            print()
        print(
            obs[15 * 15 * 3],
            obs[15 * 15 * 3 + 1],
            obs[15 * 15 * 3 + 2],
            obs[15 * 15 * 3 + 3],
        )
