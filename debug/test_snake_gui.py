
from env.snake_game import SnakeGame, Direction
import pygame



if __name__ == "__main__":
    game = SnakeGame(15, 10, True)
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