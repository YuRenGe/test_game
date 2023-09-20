import yaml
from blocks import *


class LevelLoader:
    def __init__(self, level_filepath):
        with open(level_filepath) as f:
            self.level = yaml.safe_load(f)

        s = self.level['map_size']
        h, w = s, s
        self.field_size = h, w

        self.field = np.full(self.field_size, EmptyBlock.get_code())

    def get_field_size(self):
        return self.field_size

    def get_field(self):
        return self.field

    def get_num_feed(self):
        return self.level['num_feed']

    def get_obstacles(self):
        return self.level['num_obstacles']

    # def get_initial_head_position(self):
    #     return self.initial_head_position
    #
    # def get_initial_tail_position(self):
    #     return self.initial_tail_position
    #
    # def get_initial_snake(self):
    #     return self.initial_snake
