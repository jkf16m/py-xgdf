"""Small pygame Snake example used as a gq workspace test."""

from __future__ import annotations

import random
import sys

import pygame


CELL = 24
COLUMNS, ROWS = 25, 20
WIDTH, HEIGHT = COLUMNS * CELL, ROWS * CELL
TICK = 10


def new_food(snake: list[tuple[int, int]]) -> tuple[int, int]:
    """Return a random cell not occupied by the snake."""
    available = [(x, y) for y in range(ROWS) for x in range(COLUMNS) if (x, y) not in snake]
    return random.choice(available)


def draw(screen: pygame.Surface, snake: list[tuple[int, int]], food: tuple[int, int]) -> None:
    """Draw the board, snake, and food."""
    screen.fill((18, 18, 24))
    for x, y in snake:
        pygame.draw.rect(screen, (72, 210, 110), (x * CELL, y * CELL, CELL - 2, CELL - 2))
    x, y = food
    pygame.draw.rect(screen, (235, 75, 75), (x * CELL, y * CELL, CELL - 2, CELL - 2))
    pygame.display.flip()


def main() -> int:
    """Run one game until the window closes or the snake collides."""
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Snake")
    clock = pygame.time.Clock()
    snake = [(COLUMNS // 2, ROWS // 2)]
    direction = (1, 0)
    queued = direction
    food = new_food(snake)
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                choices = {
                    pygame.K_UP: (0, -1), pygame.K_w: (0, -1),
                    pygame.K_DOWN: (0, 1), pygame.K_s: (0, 1),
                    pygame.K_LEFT: (-1, 0), pygame.K_a: (-1, 0),
                    pygame.K_RIGHT: (1, 0), pygame.K_d: (1, 0),
                }
                candidate = choices.get(event.key)
                if candidate and candidate != (-direction[0], -direction[1]):
                    queued = candidate

        direction = queued
        head = (snake[0][0] + direction[0], snake[0][1] + direction[1])
        if not (0 <= head[0] < COLUMNS and 0 <= head[1] < ROWS) or head in snake:
            running = False
            continue
        snake.insert(0, head)
        if head == food:
            food = new_food(snake)
        else:
            snake.pop()
        draw(screen, snake, food)
        clock.tick(TICK)

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
