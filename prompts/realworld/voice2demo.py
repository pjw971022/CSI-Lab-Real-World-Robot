class PromptRealWorldVoice2Demo:
    def __int__(self, n_shot=1):
        self.n_shot = n_shot

    def prompt(self): # @ Success / Fail
        prompt = \
        '[Goal] stack 2 dice on the table. ' \
        '[Context] There are blue and purple dice on the desk. \nAll possible objects: purple dice, blue dice, yellow dice, bottle, lotion, cup, pencil holder, green basket, stain. \nPossible Actions: Move, Rotate, Push, Pull, Sweep.' \
        '[Plan 1] move the blue dice on the purple dice ' \
        '[Plan 2] done stacking up the table '     
        prompt = \
        f'{prompt}' \
        '[Goal] clean up blocks. ' \
        '[Context] There are red and blue block on the desk. \nAll possible objects: red block, blue block, bottle, lotion, cup, yellow pencil, green basket, stain. \nPossible Actions: Move, Rotate, Push, Pull, Sweep.' \
        '[Plan 1] move the red block in the green basket. ' \
        '[Plan 2] move the blue block in the green basket. ' \
        '[Plan 3] done cleaning up the blocks. '
        prompt = \
        f'{prompt}' \
        '[Goal] open the cola. ' \
        '[Context] There are coke on the desk. \nAll possible objects: coke, sponge, pencil holder, yellow pencil. \nPossible Actions: Move, Rotate, Push, Pull, Sweep.' \
        '[Plan 1] pick up the cola bottle cap . ' \
        '[Plan 2] rotate the cola bottle cap. ' \
        '[Plan 3] place the anywhere on the table. ' \
        '[Plan 4] done opening the coke. '
        # prompt = \
        # f'{prompt}' \
        # '[Goal] wipe off a stain. ' \
        # '[Context] There are a sponge and stain on the desk. \nAll possible objects: coke, sponge, pencil holder, yellow pencil, green basket, stain. \nPossible Actions: Move, Rotate, Push, Pull, Sweep.' \

        # @@@
        return prompt
