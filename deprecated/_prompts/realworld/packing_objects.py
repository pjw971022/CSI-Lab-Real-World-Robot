class PromptRealWorldPackingObjects:
    def __int__(self, n_shot=1):
        self.n_shot = n_shot

    def prompt(self): # @ Success / Fail
        prompt = \
        '[Goal] pack the square shape in the gray box. ' \
        '[State of Step 1] dice, candy case, pencil, spoon, candy. ' \
        '[Plan 1] move the dice in the gray box. [Failure] ' \
        '[State of Step 1] dice, candy case, pencil, spoon, candy. ' \
        '[Plan 1] move the dice in the gray box. [Success] ' \
        '[State of Step 2] candy case, pencil, spoon, candy. ' \
        '[Plan 2] move the candy case in the gray box. [Failure] ' \
        '[State of Step 3] candy case, pencil, spoon, candy. ' \
        '[Plan 3] move the candy case in the gray box. [Success] ' \
        '[State of Step 4] pencil, spoon, candy. ' \
        '[Plan 4] done cleaning up the table '        
        return prompt
