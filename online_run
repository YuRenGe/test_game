import random

import keras
import numpy as np
from collections import deque

from blocks import EmptyBlock, ObstacleBlock, FeedBlock, SnakeHeadBlock, SnakeBodyBlock, SnakeTailBlock
from snake import NUM_CHANNELS


class Soldier:
    DX, DY = [-1, 0, 1, 0], [0, 1, 0, -1]

    def __init__(self, role_id,  init_pos, field):
        self.id = role_id
        self.current_pos = init_pos
        self.path = []      # 自己经过的轨迹都是0
        self.path.append(init_pos)
        self.field = field
        self.start_x = -1
        self.start_y = -1
        self.target_pos = None
        self.field_size = 21

    def get_start_map_point(self):
        self.field[self.current_pos[0]][self.current_pos[1]] = SnakeHeadBlock.get_code(0)
        self.construct_11x11_map()

    def get_current_state(self):
        field = self.field[self.start_x:self.start_x+11, self.start_y:self.start_y+11]
        # 可以在此处控制显示几个目标点
        return np.eye(NUM_CHANNELS)[field]

    def step(self, action):
        next_pos = self.current_pos[0] + self.DX[action], self.current_pos[1] + self.DY[action]
        if next_pos[0] < 0 or next_pos[0] >= self.field_size or next_pos[1] < 0 or \
                next_pos[1] >= self.field_size or self.field[next_pos[0]][next_pos[1]] == ObstacleBlock.get_code():
            next_pos = self.current_pos
        self.field[self.current_pos[0]][self.current_pos[1]] = EmptyBlock.get_code()
        self.field[next_pos[0]][next_pos[1]] = ObstacleBlock.get_code()
        self.current_pos = next_pos
        self.path.append(next_pos)
        return self.field

    def update_env_info(self, field, target_pos):         # 此时的field里面已经有所有信息了
        self.field = field
        self.target_pos = target_pos
        # 将自己填充到地图上
        self.field[self.current_pos[0]][self.current_pos[1]] = SnakeHeadBlock.get_code(0)
        # 将目标点填充到地图上
        for pos in self.target_pos:
            if pos not in self.path and self.field[pos[0]][pos[1]] == EmptyBlock.get_code():         # 走过的路就不再放置目标点
                self.field[pos[0]][pos[1]] = FeedBlock.get_code()
        if self.start_x < 0:
            self.get_start_map_point()

    def construct_11x11_map(self):
        sub_map_index = []
        sub_map_list = []
        # 找出所有的子地图
        for i in range(11):
            for j in range(11):
                sub_map = self.field[i:i+11, j:j+11]
                sub_map_index.append((i, j))
                sub_map_list.append(sub_map)
        # 从所有子地图中找出所有包含该士兵位置的子地图
        sub_map_with_role_list = []
        sub_map_with_role_start_point = []
        for index, sub_map in enumerate(sub_map_list):
            if np.sum(sub_map == SnakeHeadBlock.get_code(0)) != 0:
                sub_map_with_role_start_point.append(sub_map_index[index])
                sub_map_with_role_list.append(sub_map)

        enemy_counts = [np.sum(sub_map == FeedBlock.get_code()) for sub_map in sub_map_with_role_list]
        max_index = np.argmax(enemy_counts)
        max_sub_map = sub_map_with_role_list[max_index]
        self.start_x, self.start_y = sub_map_with_role_start_point[max_index]
        return max_sub_map, enemy_counts[max_index]


class Env:
    DX, DY = [-1, 0, 1, 0], [0, 1, 0, -1]

    def __init__(self):
        self.model = keras.models.load_model("CoreGeek/checkpoints/model_best.h5")
        self.field_size = 21
        self.field = None
        self.field_target_point = None
        self.soldiers_info = {}
        self.enemy_pos = []
        self.target_pos = []

    def update_env(self, json_data):
        self.enemy_pos.clear()
        self.soldiers_info.clear()
        if self.field is None:
            self.field = np.full((self.field_size, self.field_size), EmptyBlock.get_code())
            zones = json_data["mapInfo"]["zones"]
            for zone in zones:
                if zone["roleType"] == "mountain":
                    self.field[zone["pos"]["x"]][zone["pos"]["y"]] = ObstacleBlock.get_code()
        enemy_team_roles = json_data["players"]["teamEnemy"]["posList"]
        for role in enemy_team_roles:
            self.enemy_pos.append((role["x"], role["y"]))
        our_team_roles = json_data["players"]["teamOur"]["roles"]
        for role in our_team_roles:
            pos = role["pos"]["x"], role["pos"]["y"]
            self.soldiers_info[role['id']] = Soldier(role['id'], pos, self.field.copy())
        print("ok")

    def construct_target_pos(self):
        field1 = self.field.copy()
        for _, soldier in self.soldiers_info.items():
            field1[soldier.current_pos[0]][soldier.current_pos[1]] = ObstacleBlock.get_code()       # 目标点不要出现在己方位置上面
        for pos in self.enemy_pos:
            field1[pos[0]][pos[1]] = ObstacleBlock.get_code()       # 目标点不要出现在敌方位置上面
        self.target_pos.clear()
        for pos in self.enemy_pos:
            target_pos = self._generate_target_pos(pos, field1)
            if target_pos is not None:
                self.target_pos.append(target_pos)

    def _generate_target_pos(self, pos, field):
        DX, DY = [0, 1, 1, 1, 0, -1, -1, -1], [1, 1, 0, -1, -1, -1, 0, 1]
        random_direction = random.randint(0, 7)
        i = 0
        while i < 8:
            pos_x, pos_y = pos[0] + DX[random_direction], pos[1] + DY[random_direction]
            if 0 <= pos_x < self.field_size and 0 <= pos_y < self.field_size and field[pos_x][
                pos_y] == EmptyBlock.get_code():
                return pos_x, pos_y
            random_direction = (random_direction + 1) % 8
            i += 1
        return None

    def run(self):
        field1 = self.field.copy()
        # 更新敌方角色
        for enemy_pos in self.enemy_pos:
            field1[enemy_pos[0]][enemy_pos[1]] = ObstacleBlock.get_code()
        for _ in range(10):
            field2 = field1.copy()           # 包含有敌方位置信息的地图
            for _, soldier in self.soldiers_info.items():    # 先将己方所有士兵以障碍物的状态填充到地图上
                field2[soldier.path[-1][0]][soldier.path[-1][1]] = ObstacleBlock.get_code()
            for _, soldier in self.soldiers_info.items():
                soldier.update_env_info(field2, self.target_pos)
                current_state = soldier.get_current_state()
                action = np.argmax(self.model.predict(np.array([current_state])))
                field2 = soldier.step(action)
        soldiers = []
        for soldier_id, solider in self.soldiers_info.items():
            soldier = {'roleId': soldier_id}
            pos_list = []
            for pos in solider.path[-10:]:
                pos_list.append({'x': pos[0], 'y': pos[1]})
            soldier['posList'] = pos_list
            soldiers.append(soldier)
        res = {"soldiers": soldiers}
        print(res)
        return res
