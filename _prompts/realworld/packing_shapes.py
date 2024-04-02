class PromptRealWorldPackingShapes:
    def __int__(self, n_shot=1):
        self.n_shot = n_shot

    def prompt(self): # @ Success / Fail
        prompt = \
        '[Goal] move the square in the third paper' \
        '[State of Step 1] circle, retangle, square, triangle' \
        '[Plan 1] move the square in the third paper' \
        '[State of Step 2] circle, retangle, triangle' \
        '[Plan 2] done moving the square '     
        return prompt
