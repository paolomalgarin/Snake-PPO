# Logica snake
from enum import Enum
from typing import NamedTuple
import random, math


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def getOpposite(self):
        match self:
            case Direction.UP:
                return Direction.DOWN
            case Direction.DOWN:
                return Direction.UP
            case Direction.LEFT:
                return Direction.RIGHT
            case Direction.RIGHT:
                return Direction.LEFT

class Point(NamedTuple):
    x: int
    y: int


class SnakeGame:
    
    def __init__(self, gridW: int = 15, gridH: int = 15):
        self.gridWidth = gridW
        self.gridHeight = gridH

        self.head = Point(0, 0)
        self.body = []
        self.food = None

        self.direction = Direction.RIGHT

        self.score = 0
        self.isGameOver = False

    def spawnFood(self):
        randX = -1
        randY = -1

        attempt = 0
        maxAttempts = 100
        found = False

        while not found: # simula un do-while perchè non esiste in python 
            randX = random.randint(0, self.gridWidth - 1)
            randY = random.randint(0, self.gridHeight - 1)
            candidate = Point(randX, randY)

            attempt += 1
            found = ((candidate not in self.body and candidate != self.head) or attempt >= maxAttempts) 

        if(attempt >= maxAttempts):
            print("no free cell found to spawn food, spawning in occupied cell")

        self.food = Point(randX, randY)

    def hittedWall(self):
        horizontalWallHit = self.head.x < 0 or self.head.x >= self.gridWidth
        verticalWallHit = self.head.y < 0 or self.head.y >= self.gridHeight
        return horizontalWallHit or verticalWallHit
    
    def hittedBody(self):
        return self.head in self.body
    
    def grow(self):
        # inserisce un nuovo elemento alla fine del body
        # in modo che venga rimosso al posto dell'ultimo pezzo della coda quando chiamo move,
        # allungando il body
        self.body.append(Point(-1, -1)) # (aggiunta in coda)

    def moveHead(self):
        # calcola la nuova posizione della testa
        x, y = self.head.x, self.head.y
        
        match self.direction:
            case Direction.UP:
                y -= 1
            case Direction.DOWN:
                y += 1
            case Direction.LEFT:
                x -= 1
            case Direction.RIGHT:
                x += 1
        
        # mette la testa in quella posizione
        self.head = Point(x, y)

    def changeDir(self, dir: Direction):
        if(dir.getOpposite() == self.direction):
            return False
        else:
            self.direction = dir
            return True

    def move(self):
        # aggiungo la testa corrente nel body
        self.body.insert(0, self.head) # (aggiunta in testa)

        self.moveHead()

        if(self.head == self.food):
            self.spawnFood()
            self.grow()
            self.score += 1

        if(self.hittedWall() or self.hittedBody()):
            self.isGameOver = True

        self.body.pop() # (rimozione in coda)


    def displayCMD(self):
        gameString = " "

        for row in range(self.gridHeight):
            for col in range(self.gridWidth):
                currentPoint = Point(col, row)
                
                if(currentPoint in self.body):
                    gameString += "▢ "
                elif(currentPoint == self.head):
                    gameString += "▣ "
                elif(currentPoint == self.food):
                    gameString += "◎ "
                else:
                    gameString += ". "
            gameString += "\n "
        
        print(gameString)

    def getFoodDistance(self):
        return math.sqrt(math.pow(self.head.x - self.food.x, 2) + math.pow(self.head.y - self.food.y, 2))
    
    def reset(self):
        self.head = Point(0, 0)
        self.direction = Direction.RIGHT
        self.score = 0
        self.isGameOver = False

        while self.body.__len__() > 0:
            self.body.pop()

        self.spawnFood()