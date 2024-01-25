class PromptRealWorld1:
    def __int__(self, n_shot=1):
        self.n_shot = n_shot

    def prompt(self):

        prompt = \
            '[Goal] "~~~" ' \
            '[Initial State] ~~~ ' \
            f'[Step 1] ~~~. ' \
            f'[Step 2] ~~~. ' \
            f'[Step 3] ~~~. ' \
            f'[Step 4] done ~~~. '

        return prompt
