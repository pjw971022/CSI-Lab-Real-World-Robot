class PromptRealWorldMultiModal:
    def __int__(self, n_shot=1):
        self.n_shot = n_shot
        
    def prompt(self, command_modality): # @ Success / Fail
        if command_modality <2:
            prompt = \
            '[Goal] pick up the red dice. ' \
            '[Context] There are red dice, tennis ball, toy car on the desk. \nPossible Actions: move, pick, place, rotate, push, pull, sweep.' \
            '[Plan 1] pick up the <red dice>.' \
            '[Plan 2] done picking up the red dice. '
        else:
            prompt = \
            '[Goal] imitate my behavior. [bebavior description] The person pulled the toy car and placed them in the red box.' \
            '[Context] There are red dice, tennis ball, toy car on the desk. \nPossible Actions: move, pick, place, rotate, push, pull, sweep.' \
            '[Plan 1] pull the <toy car>.' \
            '[Plan 1] place in the <red box>.' \
            '[Plan 2] done imitating your behavior. '
        
        # prompt += \
        # f'{prompt}' \
        # '[Goal] pick up the toy car. ' \
        # '[Context] There are red dice, tennis ball, toy car on the desk. \nPossible Actions: move, pick, place, rotate, push, pull, sweep.' \
        # '[Plan 1] pick up the <red dice>.' \
        # '[Plan 2] done picking up the red objects. '
        return prompt