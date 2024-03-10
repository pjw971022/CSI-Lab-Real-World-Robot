"""Real-world Task Level 1"""

import numpy as np
from ravens.tasks.task import Task
from ravens.utils import utils

class RealWorldMakingWord(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 5
        self.final_goal = "make a word using the given alphabet puzzles"
        self.lang_template = "move the {obj} in the {receptacle}"
        self.task_completed_desc = "done making a word"
        self.categories = ''
        self.objects = ['M','O','R','E']
        self.receptacles = ["first paper", "second paper", "third paper", "fourth paper"]
        self.assets_root = True
        self.lang_initial_state = ''
        self.admissible_actions = [f'move the {obj} in the {receptacle}' 
                                   for obj in self.categories 
                                   for receptacle in self.receptacles ]
        self.extract_state_prompt = \
                                    f'Extract only the alphabet from the image. ' \
                                    f'Format of alphabet is <object 1, object 2, object 3>. you must adhere strictly to the format.'
                
    def reset(self, ):
        super().reset()
        self.lang_goals.append(self.lang_template.format(obj='trash', receptacle='trash can'))

    def get_final_lang_goal(self):
        return self.final_goal
        