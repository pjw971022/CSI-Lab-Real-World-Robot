"""Real-world Task Level 1"""

import numpy as np
from ravens.tasks.task import Task
from ravens.utils import utils

class RealWorldPackingObjects(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.final_goal_template = "pack the {shape} shape in the green box"
        self.lang_template = "move the {obj} in the {receptacle}"
        self.task_completed_desc = "done packing shapes"
        self.categories = ["circle", "rectangle",  "ring", "square"]
        self.receptacles = ["green box", "gray box" ]
        self.objects = ["ball", "cup", "fan", "dice", "tape", "vr controller", "bean", "candy case"]
        self.assets_root = True
        self.lang_initial_state = ''
        self.admissible_actions = [f'move the {obj} in the {receptacle}' 
                                   for obj in self.objects 
                                   for receptacle in self.receptacles ]

    def reset(self, ):
        super().reset()
        self.lang_goals.append(self.lang_template.format(obj='trash', receptacle='trash can'))

    def get_final_lang_goal(self):
        if len(self.lang_goals) == 0:
            return self.task_completed_desc
        else:
            # random_shape = np.random.choice(self.categories)
            selected_shape = 'circle'
            return self.final_goal_template.format(shape=selected_shape) 
        