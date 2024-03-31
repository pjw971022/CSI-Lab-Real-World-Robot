class PromptRealWorldCleanup:
    def __int__(self, n_shot=1):
        self.n_shot = n_shot

    def prompt(self): # @
        prompt = \
            '[Goal] clean up the trash and tidy the table. ' \
            '[State of Step 1] trash, office supplies, fan, snack. ' \
            '[Plan 1] move the trash in the trash can. ' \
            '[State of Step 2] trash, office supplies, fan, snack. ' \
            '[Plan 2] move the trash in the trash can. ' \
            '[State of Step 3] office supplies, fan, snack. ' \
            '[Plan 3] move the fan in the box. ' \
            '[State of Step 4] office supplies,snack. ' \
            '[Plan 4] move the office supplies in the box. ' \
            '[State of Step 5] snack. ' \
            '[Plan 5] move the snack in the box. ' \
            '[State of Step 6] ' \
            '[Plan 6] done cleaning up the table '
        
        return prompt
