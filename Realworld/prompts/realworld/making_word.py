class PromptRealWorldMakingWord:
    def __int__(self, n_shot=1):
        self.n_shot = n_shot
    def video2prompt(self, ):
        prompt = \
        '''
        In this video, a human is rearranging letters to form a word. The human first picks up the letter "M" and places it on the first card. Then, the human picks up the letter "O" and places it on the second card. The human then picks up the letter "R" and places it on the third card. Finally, the human picks up the letter "E" and places it on the fourth card. The human has now formed the word "MORE".
        This is video description. I want to know how to do this task sequencially.
        Our place have only 2 objects. That is paper and alphabets. Other objects can't use in planning sentence. And planning sentence don't need to use upper letter in first word.
        Plan this task like this sentence. You should response only plan sentence. And input enter between plan.
        [Plan 1] move the D in the first paper.
        [Plan 2] move the O in the second paper. 
        [Plan 3] move the G in the third paper.
        [Plan 4] done making a DOG.
        '''
        return prompt
        
    def prompt(self): # @ Success / Fail
        prompt = \
        '[Goal] make a word using the given alphabet puzzles. ' \
        '[Description] The nth of the word must go into the nth paper.' \
        '[State of Step 1] A, D, O, R, G. ' \
        '[Plan 1] move the D in the first paper. ' \
        '[State of Step 2] A, O, R, G. ' \
        '[Plan 2] move the O in the second paper. ' \
        '[State of Step 3] A, O, R, G. ' \
        '[Plan 3] move the G in the third paper. ' \
        '[State of Step 4] A, R. ' \
        '[Plan 4] done making a DOG. '
        prompt = \
        f'{prompt}' \
        '[Goal] make a word using the given alphabet puzzles. ' \
        '[Description] The nth of the word must go into the nth paper.' \
        '[State of Step 1] A, F, O, K, E. ' \
        '[Plan 1] move the F in the first paper. ' \
        '[State of Step 2] A, O, K, E. ' \
        '[Plan 2] move the A in the second paper. ' \
        '[State of Step 3] O, K, E. ' \
        '[Plan 3] move the K in the third paper. ' \
        '[State of Step 4] O, E. ' \
        '[Plan 4] move the E in the fourth paper. ' \
        '[State of Step 5] O. ' \
        '[Plan 5] done making a FAKE. '

        # prompt = \
        # '[Goal] make a word using the given alphabet puzzles. ' \
        # '[State of Step 1] A, D, O, R, G. ' \
        # '[Plan 1] move the D in the first paper. [Failure] ' \
        # '[State of Step 2] A, D, O, R, G. ' \
        # '[Plan 2] move the D in the first paper. [Success] ' \
        # '[State of Step 3] A, O, R, G. ' \
        # '[Plan 3] move the O in the second paper. [Success] ' \
        # '[State of Step 4] A, O, R, G. ' \
        # '[Plan 4] move the G in the third paper. [Success] ' \
        # '[State of Step 5] A, R. ' \
        # '[Plan 5] done making a word '
        return prompt
