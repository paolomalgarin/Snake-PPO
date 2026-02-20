# Wrapper Gymnasium

from gymnasium import Env, spaces
import numpy as np
from env.snake_game import SnakeGame, Point, Direction


class SnakeEnv(Env):

    def __init__(self, useGui = False):
        self.game = SnakeGame(useGui=useGui)
        self.useGui = useGui
        self.game.reset()
        self.action_space = spaces.Discrete(
            4
        )  # 3 actions: up, down, right, left
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3, self.game.gridHeight, self.game.gridWidth), dtype=np.float32
        )
        self.max_steps = self.game.gridHeight * self.game.gridWidth * 10
        self.prev_score = self.game.score
        self.prev_food_distance = self.game.getFoodDistance()
        self.loopBuffer = []  # buffer containing every position the head of the snake has been (get resetted every time the snake eats) to check if the snake is circling 
        self.steps = 0

    def reset(self, seed=None, options=None):
        self.game.reset()
        self.steps = 0
        self.prev_score = self.game.score
        self.prev_food_distance = self.game.getFoodDistance()
        self.max_steps = self.game.gridHeight * self.game.gridWidth * 10
        self.loopBuffer.clear()  # buffer containing every position the head of the snake has been (get resetted every time the snake eats) to check if the snake is circling 

        obs = self._get_obs()
        info = {}

        return obs, info

    def step(self, action):
        self.steps += 1

        # Add the OLD position in the anti-circling buffer
        self.loopBuffer.append(self.game.head)

        # 4 actions: 0 = up, 1 = down, 2 = right, 3 = left
        match action:
            # case 0:
            #     newDir = self.game.direction
            # case 1:
            #     newDir = self.game.direction.getRightTurn()
            # case 2:
            #     newDir = self.game.direction.getRightTurn().getOpposite()
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
                    grid[1][row][col] = 1.0
                elif currentPoint == self.game.food:
                    grid[2][row][col] = 1.0

        # direction_mapping = {
        #     Direction.UP: 0,
        #     Direction.DOWN: 1,
        #     Direction.LEFT: 2,
        #     Direction.RIGHT: 3,
        # }
        # grid[directionStart + direction_mapping[self.game.direction]] = 1.0

        return grid

    def _compute_reward(self):
        reward = 0


        # # +0.1 if gets closer to food, -0.1 if gets furder from food
        # new_food_distance = self.game.getFoodDistance()

        # if self.game.score == self.prev_score:  # avoids giving penalities immediatly after the snake eat
        #     if new_food_distance < self.prev_food_distance:
        #         # reward += 0.1
        #         pass
        #     else:
        #         # reward -= 0.18
        #         reward -= 0.3

        # self.prev_food_distance = new_food_distance
        
        # +10 if eats food (and add more steps)
        if self.game.score > self.prev_score:
            self.prev_score = self.game.score
            reward += 1
            
            # # Flush the anti-circling buffer
            # self.loopBuffer.clear()

        
        # # penality for circling
        # if(self.game.head in self.loopBuffer):
        #     reward -= 0.2

        # -10 if dies
        # if self.game.isGameOver:
        #     reward = -1

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
