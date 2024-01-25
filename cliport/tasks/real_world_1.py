"""Real-world Task Level 1"""

import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils

class RealWorld1(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 3
        self.lang_template = "align the brown box with the green corner"
        self.task_completed_desc = "done with alignment"