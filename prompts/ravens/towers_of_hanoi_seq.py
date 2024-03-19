class PromptRavensTowersOfHanoiSeq:
    def __init__(self, n_shot=1):
        self.n_shot = n_shot

    def prompt(self):
        # prompt = \
        #     'Task: move the brown disk in rod 3. ' \
        #     'Initial State: red disk on top of brown disk. ' \
        #     'brown disk on top of blue disk. blue disk rod 1. ' \
        #     'The locations for moving are lighter brown side, middle of the stand, ' \
        #     'darker brown side. ' \
        #     f'Step 1:put red disk in middle of the stand. ' \
        #     f'Step 2:put brown disk in darker brown side. ' \
        #     f'Step 3:done putting disks in stands. '
        # prompt = '[Game] Tower of Hanoi. '

        prompt = \
            '[Goal] move the gray disk in rod 2. ' \
            '[Initial State] red disk on top of gray disk. ' \
            'gray disk in rod 1. ' \
            'The disks can be moved in rod 1, rod 2, rod 3. ' \
            '[Step 1] put red disk in rod 3. ' \
            '[Step 2] put gray disk in rod 2. ' \
            '[Step 3] done putting disks in rods. '

        prompt = \
            f'{prompt}' \
            '[Goal] move the yellow disk in rod 3. ' \
            '[Initial State] red disk on top of green disk. ' \
            'green disk on top of yellow disk. yellow disk in rod 1. ' \
            'The disks can be moved in rod 1, rod 2, rod 3.  ' \
            '[Step 1] put red disk in rod 3. ' \
            '[Step 2] put green disk in rod 2. ' \
            '[Step 3] put red disk in rod 2. ' \
            '[Step 4] put yellow disk in rod 3. ' \
            '[Step 5] done putting disks in rods. '

        prompt = \
            f'{prompt}' \
            '[Goal] move the blue disk in rod 2. ' \
            '[Initial State] blue disk on top of cyan disk. cyan disk in rod 1. ' \
            'The disks can be moved in rod 1, rod 2, rod 3. ' \
            '[Step 1] put blue disk in rod 2. ' \
            '[Step 2] done putting disks in rods. '

        # prompt = \
        #     f'{prompt}' \
        #     '[Goal] move the blue disk to the middle of the stand. ' \
        #     '[Initial State] cyan disk on top of brown disk. ' \
        #     'brown disk on top of blue disk. blue disk in rod 1. ' \
        #     'TThe disks can be moved in rod 1, rod 2, rod 3. ' \
        #     '[Step 1] put cyan disk in middle of the stand. ' \
        #     '[Step 2] put brown disk in darker brown side. ' \
        #     '[Step 3] put cyan disk in darker brown side. ' \
        #     '[Step 4] put blue disk in middle of the stand. ' \
        #     '[Step 5] done putting disks in stands. '

        # prompt = \
        #     f'{prompt}' \
        #     'Initial State: brown disk on top of green disk. ' \
        #     'green disk on top of gray disk. gray disk in rod 1. ' \
        #     'The locations for moving are lighter brown side, middle of the stand, ' \
        #     'darker brown side. ' \
        #     'Task: move the gray disk to the middle of the stand. ' \
        #     f'Step 1:put brown disk in middle of the stand. ' \
        #     f'Step 2:put green disk in darker brown side. ' \
        #     f'Step 3:put brown disk in darker brown side. ' \
        #     f'Step 4:put gray disk in middle of the stand. ' \
        #     f'Step 5:done putting disks in stands. '

        return prompt


class PromptRavensTowersOfHanoiSeqSeen:
    def __init__(self, n_shot=1):
        self.n_shot = n_shot

    def prompt(self):
        # prompt = \
        #     'Task: move the brown disk in rod 3. ' \
        #     'Initial State: red disk on top of brown disk. ' \
        #     'brown disk on top of blue disk. blue disk rod 1. ' \
        #     'The locations for moving are lighter brown side, middle of the stand, ' \
        #     'darker brown side. ' \
        #     f'Step 1:put red disk in middle of the stand. ' \
        #     f'Step 2:put brown disk in darker brown side. ' \
        #     f'Step 3:done putting disks in stands. '
        # prompt = '[Game] Tower of Hanoi. '
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
                
        prompt = \
            '[Goal] move the gray ring in middle of the stand. ' \
            '[Initial State] red ring on top of gray ring. ' \
            'gray ring in lighter brown side. ' \
            'The rings can be moved in lighter brown side, middle of the stand, darker brown side. ' \
            '[Step 1] move red ring in the darker brown side. ' \
            '[Step 2] move gray ring in the middle of the stand. ' \
            '[Step 3] done putting rings in rods. '

        prompt = \
            f'{prompt}' \
            '[Goal] move the yellow ring in darker brown side. ' \
            '[Initial State] red ring on top of green ring. ' \
            'green ring on top of yellow ring. yellow ring in lighter brown side. ' \
            'The rings can be moved in lighter brown side, middle of the stand, darker brown side.  ' \
            '[Step 1] move red ring in the darker brown side. ' \
            '[Step 2] move green ring in the middle of the stand. ' \
            '[Step 3] move red ring in the middle of the stand. ' \
            '[Step 4] move yellow ring in the darker brown side. ' \
            '[Step 5] done putting rings in rods. '

        prompt = \
            f'{prompt}' \
            '[Goal] move the blue ring in middle of the stand. ' \
            '[Initial State] blue ring on top of cyan ring. cyan ring in lighter brown side. ' \
            'The rings can be moved in lighter brown side, middle of the stand, darker brown side. ' \
            '[Step 1] move blue ring in the middle of the stand. ' \
            '[Step 2] done putting rings in rods. '
        
        prompt = \
            f'{prompt}' \
            '[Goal] flip the towers of hanoi. ' \
            '[Initial State] red ring on top of green ring. ' \
            'green ring on top of yellow ring. yellow ring in lighter brown side. ' \
            'The rings can be moved in lighter brown side, middle of the stand, darker brown side.  ' \
            '[Step 1] move red ring in the middle of the stand. ' \
            '[Step 2] move green ring in the middle of the stand. ' \
            '[Step 3] move yellow ring in the middle of the stand. ' \
            '[Step 4] done putting rings in rods. '
        return prompt

class PromptRavensTowersOfHanoiSeqSolve: # cliport version
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
                
        # if self.n_shot > 1:
        #     prompt = \
        #         f'{prompt}' \
        #         '[Goal] Solve towers of hanoi. ' \
        #         '[Initial State] brown ring on top of red ring. ' \
        #         'red ring on top of yellow ring. yellow ring in lighter brown side' \
        #         'The rings can be moved in lighter brown side, middle of the stand, darker brown side. ' \
        #         '[Step 1] move the brown ring to the darker brown side. ' \
        #         '[Step 2] move the red ring to the middle of the stand. ' \
        #         '[Step 3] move the brown ring to the middle of the stand. ' \
        #         '[Step 4] move the yellow ring to the darker brown side. ' \
        #         '[Step 5] move the brown ring to the lighter brown side. ' \
        #         '[Step 6] move the red ring to the darker brown side. ' \
        #         '[Step 7] move the brown ring to the darker brown side. ' \
        #         '[Step 8] done putting rings in rods. '
                
        # if self.n_shot > 2:
        #     prompt = \
        #         f'{prompt}' \
        #         '[Goal] Solve towers of hanoi. ' \
        #         '[Initial State] gray ring on top of yellow ring. ' \
        #         'yellow ring on top of cyan ring. cyan ring in lighter brown side' \
        #         'The rings can be moved in lighter brown side, middle of the stand, darker brown side. ' \
        #         '[Step 1] move the gray ring to the darker brown side. ' \
        #         '[Step 2] move the yellow ring to the middle of the stand. ' \
        #         '[Step 3] move the gray ring to the middle of the stand. ' \
        #         '[Step 4] move the cyan ring to the darker brown side. ' \
        #         '[Step 5] move the gray ring to the lighter brown side. ' \
        #         '[Step 6] move the yellow ring to the darker brown side. ' \
        #         '[Step 7] move the gray ring to the darker brown side. ' \
        #         '[Step 8] done putting rings in rods. '

        return prompt
