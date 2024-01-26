"""Real-world Task Level 1"""

import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils

class RealWorld1(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.final_goal = "Pack only the dice that come up with even numbers in the box"
        self.lang_template = "pack the {color} dice in the box"
        self.task_completed_desc = "pack the dice in the box"
        self.color_list = ['red', 'green', 'yellow']
        self.assets_root = True
        
    def reset(self, env):
        super().reset(env)
        self.lang_goals.append(self.lang_template.format(color='red'))
 