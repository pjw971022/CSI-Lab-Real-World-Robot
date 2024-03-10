class PromptRealworldHanoiSolve:
    def __init__(self, n_shot=1):
        self.n_shot = n_shot

    def prompt(self):
        if self.n_shot > 0:
            prompt = \
                '[Goal] Solve towers of hanoi. ' \
                '[Initial State] green ring on top of blue ring. ' \
                'blue ring on top of brown ring. brown ring in lighter brown side' \
                'The rings can be moved in lighter brown side, middle of the stand, darker brown side. ' \
                '[Step 1] move the green ring to the darker brown side. ' \
                '[Step 2] move the blue ring to the middle of the stand. ' \
                '[Step 3] move the green ring to the middle of the stand. ' \
                '[Step 4] move the brown ring to the darker brown side. ' \
                '[Step 5] move the green ring to the lighter brown side. ' \
                '[Step 6] move the blue ring to the darker brown side. ' \
                '[Step 7] move the green ring to the darker brown side. ' \
                '[Step 8] done putting rings in rods. '
        return prompt