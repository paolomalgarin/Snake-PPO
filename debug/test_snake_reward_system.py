# script to play the game using the SnakeGame class
# (made to test the game logic)

from pathlib import Path
import sys
import argparse
from env.snake_env import SnakeEnv


if __name__ == "__main__":
    # ------------------------ game setup ------------------------
    game = SnakeEnv()


    # ------------------------ game logic (debug part) ------------------------
    game.reset()
    game.render()

    isGameOver = False
    while not isGameOver:
        # Get input
        user_input = input("Move (a=left, d=right, w=forward, q=quit): ").lower()
        
        print("\n\n")
        
        move = -1
        match user_input:
            case 'w':
                move = 0
            case 'd':
                move = 1
            case 'a':
                move = 2
            case 'q':
                break
            case _:
                continue  # Invalid input, skip move
        
        obs, reward, terminated, truncated, info = game.step(move)
        isGameOver = terminated or truncated

        game.render()
        print(f"Step Reward: {reward}")

    print("=== GAME OVER! ===")
    print("[Score " + str(game.game.score) + "]\n")

    game.close()