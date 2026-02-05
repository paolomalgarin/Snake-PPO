# script to play the game using the SnakeGame class
# (made to test the game logic)

from pathlib import Path
import sys
import argparse

# Aggiungo la directory 'env' al path
env_path = Path(__file__).parent.parent / 'env'
sys.path.insert(0, str(env_path))

from snake_game import SnakeGame, Direction


if __name__ == "__main__":
    # ------------------------ game setup ------------------------
    parser = argparse.ArgumentParser(description='Snake game')
    gameWidth, gameHeight = 10, 10

    parser.add_argument('--w', type=int, default=10, help='Width of the grid')
    parser.add_argument('--h', type=int, default=10, help='Height of the grid')
    args = parser.parse_args()

    if args.w is not None:
        gameWidth = args.w
    if args.h is not None:
        gameHeight = args.h

    game = SnakeGame(gameWidth, gameHeight)


    # ------------------------ game logic (debug part) ------------------------
    game.spawnFood()
    game.displayCMD()

    while not game.isGameOver:
        # Get input
        user_input = input("Move (w=up, s=down, a=left, d=right, q=quit): ").lower()
        
        print("\n\n")
        
        match user_input:
            case 'w':
                game.changeDir(Direction.UP)
            case 's':
                game.changeDir(Direction.DOWN)
            case 'a':
                game.changeDir(Direction.LEFT)
            case 'd':
                game.changeDir(Direction.RIGHT)
            case 'q':
                break
            case _:
                continue  # Invalid input, skip move
        
        game.move()
        game.displayCMD()

    print("=== GAME OVER! ===")
    print("[Score " + str(game.score) + "]\n")