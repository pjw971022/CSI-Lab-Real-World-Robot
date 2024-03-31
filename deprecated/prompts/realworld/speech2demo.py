class PromptRealWorldspeech2Demo:
    def __int__(self, n_shot=1):
        self.n_shot = n_shot
        
    def prompt(self): # @ Success / Fail
        prompt = \
        '[Goal] put only the red objects in the box. ' \
        '[Context] There are 2 red object, 2 yellow object and 3 green object  on the desk. \nAll possible objects: green dice, gray basket, green block, red dice, red block, yellow dice, yellow block. \nPossible Actions: move, pick, place, rotate, push, pull, sweep.' \
        '[Plan 1] move the first <red object> in the <gray basket>. ' \
        '[Plan 2] move the second <red block> in the <gray basket>. ' \
        '[Plan 3] done putting up the red objects. '
        prompt += \
        f'{prompt}' \
        '[Goal] clean up blocks. ' \
        '[Context] There are red and blue block on the desk. \nAll possible objects: red block, blue block, bottle, lotion, cup, yellow pencil, green basket, stain. \nPossible Actions: move, pick, place, rotate, push, pull, sweep.' \
        '[Plan 1] move the <red block> in the <green basket>. ' \
        '[Plan 2] move the <blue block> in the <green basket>. ' \
        '[Plan 3] done cleaning up the blocks. '

        prompt += \
        f'{prompt}' \
        '[Goal] take out all the items from the basket. ' \
        '[Context] There is a red cube and a tennis ball inside the basket. \nAll possible objects: tennis ball, sponge, red cube,pencil holder, yellow pencil. \nPossible Actions: move, pick, place, rotate, push, pull, sweep.' \
        '[Plan 1] move the <red cube> to <anywhere>. ' \
        '[Plan 2] move the <tennis ball> to <anywhere>. ' \
        '[Plan 3] done taking out all the items. '
        # '[Goal] open the cola. ' \
        # '[Context] There are coke on the desk. \nAll possible objects: coke, sponge, pencil holder, yellow pencil. \nPossible Actions: move, pick, place, rotate, push, pull, sweep.' \
        # '[Plan 1] rotate the <cola bottle cap>. ' \
        # '[Plan 2] place the <anywhere> on the table. ' \
        # '[Plan 3] done opening the coke. '
        prompt += \
        f'{prompt}' \
        '[Goal] clean up the spilled water ' \
        '[Context] Water is spilling on the table. \nAll possible objects: bottle, coke, sponge, pencil holder, yellow pencil. \nPossible Actions: move, pick, place, rotate, push, pull, sweep.' \
        '[Plan 1] pick up the <sponge> . ' \
        '[Plan 2] go to the <ready pose>. ' \
        '[Plan 3] sweep the <stain>. ' \
        '[Plan 4] place <anywhere> on the table. '\
        '[Plan 5] done wiping the spilled water. '
        prompt += \
        f'{prompt}' \
        '[Goal] drag the toy truck go to the goal' \
        '[Context] There toy truck , ball, puzzle on the desk. \nAll possible objects: toy truck, block, dice. \nPossible Actions: move, pick, place, rotate, push, pull, sweep.' \
        '[Plan 1] pull the <toy truck>. ' \
        '[Plan 2] go to the <ready pose>. ' \
        '[Plan 3] done playing fun with the toy truck. '
        # '[Goal] dispense lotion onto my hands.' \
        # '[Context] There lotion the desk. \nAll possible objects: coke, sponge, pencil holder, lotion, bottle. \nPossible Actions: move, pick, place, rotate, push, pull, sweep.' \
        # '[Plan 1] push the <lotion>. ' \
        # '[Plan 2] go to the <ready pose>. ' \
        # '[Plan 3] done pumping the lotion. '
        prompt += \
        f'{prompt}' \
        '[Goal] make the meal. ' \
        '[Context] There are apple, toast and  plate. \nAll possible objects: toast, apple, knife, banana, plate, . \nPossible Actions: move, pick, place, rotate, push, pull, sweep.' \
        '[Plan 1] move the <toast> in the <napkin>. ' \
        '[Plan 2] move the <apple> in the <napkin>. ' \
        '[Plan 3] done setting.'
        prompt = \
        '[Goal] Place each alphabet puzzle in alphabetical order. ' \
        '[Context] There are letter I, letter Q, letter J and letter F  on the desk. \nPossible Actions: move, pick, place, rotate, push, pull, sweep.' \
        '[Plan 1] move the <letter F> in the <first paper>. ' \
        '[Plan 2] move the <letter I> in the <second paper>. ' \
        '[Plan 3] move the <letter J> in the <third paper>. ' \
        '[Plan 4] move the <letter Q> in the <fourth paper>. ' \
        '[Plan 5] done placing up the alphabet puzzle. '
        prompt = \
        f'{prompt}' \
        '[Goal] Place each alphabet puzzle in alphabetical order. ' \
        '[Context] There are letter L, letter A, letter M and letter Z  on the desk. \nPossible Actions: move, pick, place, rotate, push, pull, sweep.' \
        '[Plan 1] move the <letter A> in the <first paper>. ' \
        '[Plan 2] move the <letter L> in the <second paper>. ' \
        '[Plan 3] move the <letter M> in the <third paper>. ' \
        '[Plan 4] move the <letter Z> in the <fourth paper>. ' \
        '[Plan 5] done placing up the alphabet puzzle. '

        return prompt