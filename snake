import random
from collections import deque
import pygame
from blocks import *
import numpy as np


NUM_CHANNELS = 4
NUM_ACTIONS = 4


class SnakeStateTransition:
    DX, DY = [-1, 0, 1, 0], [0, 1, 0, -1]

    def __init__(self, field_size, field, num_feed, obstacles_pos):
        self.field_height, self.field_width = field_size
        self.field = field.copy()
        self.points = 1
        self.feed_pos = []
        print("self.points : ", self.points)
        for pos in obstacles_pos:
            self.field[pos[0]][pos[1]] = ObstacleBlock.get_code()
        for _ in range(num_feed):
            self._generate_feed()
        self.head_pos = self.generate_random_point()
        self.field[self.head_pos[0]][self.head_pos[1]] = SnakeHeadBlock.get_code(0)

    def _generate_feed(self):
        empty_blocks = []
        for i in range(self.field_height):
            for j in range(self.field_width):
                if self.field[i][j] == EmptyBlock.get_code():
                    empty_blocks.append((i, j))

        if len(empty_blocks) > 0:
            x, y = random.sample(empty_blocks, 1)[0]
            self.field[x, y] = FeedBlock.get_code()
            self._generate_obstacles(x, y)
            self.feed_pos.append([x, y])

    def _generate_obstacles(self, x, y):
        random_direction = random.randint(0, 3)
        i = 0
        while i < 3:
            pos_x, pos_y = x + self.DX[random_direction], y + self.DY[random_direction]
            if 0 <= pos_x < self.field_width and 0 <= pos_y < self.field_height and self.field[pos_x][pos_y] == EmptyBlock.get_code():
                self.field[pos_x][pos_y] = ObstacleBlock.get_code()
                return
            random_direction = (random_direction + 1) % 4
            i += 1

    def generate_random_point(self):
        while True:
            x = random.randint(0, self.field_width-1)
            y = random.randint(0, self.field_height-1)
            if self.field[x][y] == 0:
                return x, y

    def get_state(self):
        return np.eye(NUM_CHANNELS)[self.field]

    def get_length(self):
        # return len(self.snake) + 1
        return self.points

    def move_forward(self, action):
        old_head_pos = self.head_pos
        hx = self.head_pos[0] + SnakeStateTransition.DX[action]
        hy = self.head_pos[1] + SnakeStateTransition.DY[action]
        if hx < 0 or hx >= self.field_height or hy < 0 or hy >= self.field_width \
                or ObstacleBlock.contains(self.field[hx][hy]):
            return -2, True

        is_feed = FeedBlock.contains(self.field[hx][hy])

        self.field[self.head_pos[0], self.head_pos[1]] = EmptyBlock.get_code()
        self.field[hx, hy] = SnakeHeadBlock.get_code(0)
        self.head_pos = hx, hy
        if is_feed:
            self.feed_pos.remove([hx, hy])
            self._generate_feed()
            self.points += 1
            return 2, False
        distance1 = self.closest_point(old_head_pos, self.feed_pos)
        distance2 = self.closest_point(self.head_pos, self.feed_pos)
        if distance1 > distance2:
            return 1, False
        else:
            return -1, False

    def closest_point(self, point, points):
        '''
        计算某个点和某个list中所有点的曼哈顿距离，并返回距离最小的点和距离
        '''
        closest_distance = float('inf')
        for p in points:
            d = self.manhattan_distance(point, p)
            if d < closest_distance:
                closest_distance = d
        return closest_distance

    def manhattan_distance(self, point1, point2):
        '''
        计算两个点之间的曼哈顿距离
        '''
        x1, y1 = point1
        x2, y2 = point2
        return abs(x1 - x2) + abs(y1 - y2)

    # def turn_left(self):
    #     self.direction = (self.direction + 3) % 4
    #     return self.move_forward()
    #
    # def turn_right(self):
    #     self.direction = (self.direction + 1) % 4
    #     return self.move_forward()


class SnakeAction:
    MOVE_FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2


class Snake:
    ACTIONS = {
        SnakeAction.MOVE_FORWARD: 'move_forward',
        SnakeAction.TURN_LEFT: 'turn_left',
        SnakeAction.TURN_RIGHT: 'turn_right'
    }

    def __init__(self, level_loader, is_play=True, block_pixels=30, map_num=1):
        self.level_loader = level_loader
        self.block_pixels = block_pixels

        self.field_height, self.field_width = self.level_loader.get_field_size()
        self.obstacles_pos = []
        for _ in range(map_num):
            self.obstacles_pos.append(self.update_obstacles())
        self.map_index = 0
        pygame.init()
        if is_play:
            self.screen = pygame.display.set_mode((
                self.field_width * block_pixels,
                self.field_height * block_pixels
            ))
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.state_transition = SnakeStateTransition(
            self.level_loader.get_field_size(),
            self.level_loader.get_field(),
            self.level_loader.get_num_feed(),
            self.obstacles_pos[self.map_index]
        )
        self.tot_reward = 0
        return self.state_transition.get_state()

    def update_obstacles(self):
        obstacles_pos = []
        num_obstacles_pos = self.level_loader.get_obstacles()
        empty_blocks = []
        field = self.level_loader.get_field()
        _field = field.copy()
        for i in range(self.field_height):
            for j in range(self.field_width):
                if _field[i][j] == EmptyBlock.get_code():
                    empty_blocks.append((i, j))
        for _ in range(num_obstacles_pos):
            x, y = random.sample(empty_blocks, 1)[0]
            _field[x, y] = ObstacleBlock.get_code()
            obstacles_pos.append((x, y))
        return obstacles_pos

    def update_map_index(self):
        self.map_index = 0

    def step(self, action):
        reward, done = self.state_transition.move_forward(action)
        self.tot_reward += reward
        return self.state_transition.get_state(), reward, done

    def get_length(self):
        return self.state_transition.get_length()

    def quit(self):
        pygame.quit()

    def render(self, fps):
        # '''
        pygame.display.set_caption('length: {}'.format(self.state_transition.get_length()))
        pygame.event.pump()
        self.screen.fill((255, 255, 255))

        for i in range(self.field_height):
            for j in range(self.field_width):
                cp = get_color_points(self.state_transition.field[i][j])
                if cp is None:
                    continue
                pygame.draw.polygon(
                    self.screen,
                    cp[0],
                    (cp[1] + [j, i])*self.block_pixels
                )

        pygame.display.flip()
        # '''
        self.clock.tick(fps)

    def save_image(self, save_path):
        pygame.image.save(self.screen, save_path)
