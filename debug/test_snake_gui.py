
from env.snake_game import SnakeGame, Direction
import pygame
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Snake game')
    
    parser.add_argument('--guiH', type=int, default=680, help='Height of the gui window')
    args = parser.parse_args()

    if args.guiH is not None:
        guiHeight = args.guiH

    game = SnakeGame(15, 10, True, windowHeight=guiHeight)
    game.reset()

    print("Move (w=up, s=down, a=left, d=right, q=quit)")

    game.drawWindow()

    running = True
    while running:
        event = pygame.event.wait()
        
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                running = False
            elif event.key == pygame.K_w:
                game.changeDir(Direction.UP)
            elif event.key == pygame.K_s:
                game.changeDir(Direction.DOWN)
            elif event.key == pygame.K_d:
                game.changeDir(Direction.RIGHT)
            elif event.key == pygame.K_a:
                game.changeDir(Direction.LEFT)
            game.move()
            game.drawWindow()

    game.close()