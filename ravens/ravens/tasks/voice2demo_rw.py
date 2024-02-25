"""Real-world Task Level 1"""

import numpy as np
from ravens.tasks.task import Task
from ravens.utils import utils

class RealWorldVoice2Demo(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 5
        self.final_goal = ""
        self.lang_template = ""
        self.task_completed_desc = "done making a word"
        self.categories = ''
        self.objects = ['red dice', 'green dice', 'yellow dice', 'bottle', 'lotion', 'cup', 'sponge', 'pencil holder', 'yellow pencil', 'green basket', 'stain', '']
        self.receptacles = []
        self.assets_root = True
        self.lang_initial_state = ''
        self.admissible_actions = []
        self.extract_state_prompt = \
                                    f'Extract only the alphabet from the image. ' \
                                    f'Format of alphabet is <object 1, object 2, object 3>. you must adhere strictly to the format.'
                
    def reset(self, ):
        super().reset()
        self.lang_goals.append(self.lang_template.format(obj='trash', receptacle='trash can'))

    def get_final_lang_goal(self):
        return self.final_goal
        