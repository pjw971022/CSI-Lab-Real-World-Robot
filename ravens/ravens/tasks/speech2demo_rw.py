"""Real-world Task Level 1"""

import numpy as np
from ravens.tasks.task import Task
from ravens.utils import utils

class RealWorldspeech2Demo(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 15
        self.final_goal = ""
        self.lang_template = ""
        self.task_completed_desc = "done making a word"
        self.categories = ''
        self.objects = ['red block', 'green block', 'yellow block', 'bottle', 'lotion', 'cup', 'sponge', 'pencil holder', 'yellow pencil',
                'green basket', 'stain', 'toy car' ,'baseball', 'tennis ball' ] 
        self.receptacles = []
        self.assets_root = True
        self.lang_initial_state = ''
        self.admissible_actions = []
        self.extract_state_prompt = "Tell me every objects on the black table. Example) There are letter A, letter B, letter C and letter D  on the desk."            
    def reset(self, ):
        super().reset()
        self.lang_goals.append(self.lang_template.format(obj='trash', receptacle='trash can'))

    def get_final_lang_goal(self):
        return self.final_goal
        