class PromptRealworldHanoiSolve:
    def __init__(self, n_shot=1):
        self.n_shot = n_shot

    def prompt(self):
        if self.n_shot > 0:
            prompt = \
                '[Goal] Solve towers of hanoi. ' \
                '[Initial State] green ring on top of blue ring. ' \
                'blue ring on top of brown ring. brown ring in first side' \
                'The rings can be moved in first side, second side, third side. ' \
                '[Step 1] move the <green ring> to the <third side>. ' \
                '[Step 2] move the <blue ring> to the <second side>. ' \
                '[Step 3] move the <green ring> to the <second side>. ' \
                '[Step 4] move the <brown ring> to the <third side>. ' \
                '[Step 5] move the <green ring> to the <first side>. ' \
                '[Step 6] move the <blue ring> to the <third side>. ' \
                '[Step 7] move the <green ring> to the <third side>. ' \
                '[Step 8] done putting rings in rods. '
        return prompt