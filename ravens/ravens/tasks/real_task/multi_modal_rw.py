"""Real-world Task Level 1"""

import numpy as np
from ravens.tasks.task import Task
from ravens.utils import utils

class RealWorldMultimodal(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 15
        self.extract_state_prompt = "Tell me every objects on the desk. Example) There are letter A, tennis ball, toy car on the desk." #          

        self.task_completed_desc = "done"
        self.final_goal = ""
        self.lang_template = ""
        self.categories = ''
        self.objects =[]
        self.receptacles = []
        self.assets_root = True
        self.lang_initial_state = ''
        self.admissible_actions = []

    def reset(self, ):
        super().reset()

    def get_final_lang_goal(self):
        return self.final_goal
        