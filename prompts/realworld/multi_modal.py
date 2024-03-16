class PromptRealWorldMultiModal:
    def __int__(self, n_shot=1):
        self.n_shot = n_shot
        
    def prompt(self, command_modality): # @ Success / Fail
        if command_modality ==0:
            prompt = \
            '[Goal] Solve towers of hanoi. ' \
            '[Context] green ring on top of blue ring. ' \
            'blue ring on top of brown ring. brown ring in first side' \
            'The rings can be moved in first side, second side, third side. ' \
            '[Plan 1] move the <green ring> to the <third side>. ' \
            '[Plan 2] move the <blue ring> to the <second side>. ' \
            '[Plan 3] move the <green ring> to the <second side>. ' \
            '[Plan 4] move the <brown ring> to the <third side>. ' \
            '[Plan 5] move the <green ring> to the <first side>. ' \
            '[Plan 6] move the <blue ring> to the <third side>. ' \
            '[Plan 7] move the <green ring> to the <third side>. ' \
            '[Plan 8] done solving rings in rods. '
        elif command_modality ==1:
            prompt = \
            '[Goal] pick up the red dice. ' \
            '[Context] There are red dice, tennis ball, toy car on the desk. \nPossible Actions: move, pick, place, rotate, push, pull, sweep.' \
            '[Plan 1] pick up the <red dice>.' \
            '[Plan 2] done picking up the red dice. '
        else:
            prompt = \
            '[Goal] imitate my behavior in this video. you have to only consider the [Possible objects] ' \
            '[Context] This video shows a hand playing with a red toy car on a black surface. The hand moves the car around in different directions.' \
            '[Possible Actions] move, pick, place, rotate, push, pull, sweep.' \
            '[Possible objects] toy car, red box.' \
            '[Plan 1] pull the <toy car>.' \
            '[Plan 2] place in the <red box>.' \
            '[Plan 3] done imitating your behavior. '
        
        # prompt += \
        # f'{prompt}' \
        # '[Goal] pick up the toy car. ' \
        # '[Context] There are red dice, tennis ball, toy car on the desk. \nPossible Actions: move, pick, place, rotate, push, pull, sweep.' \
        # '[Plan 1] pick up the <red dice>.' \
        # '[Plan 2] done picking up the red objects. '
        return prompt