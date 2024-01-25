"""Prompts for all envs."""
from prompts.ravens.put_block_in_bowl import PromptRavensPutBlockInBowl
from prompts.ravens.towers_of_hanoi_seq import PromptRavensTowersOfHanoiSeq, PromptRavensTowersOfHanoiSeqSeen, PromptRavensTowersOfHanoiSeqSolve
from prompts.babyai.babyai_pickup import PromptBabyAIPickup,PromptBabyAIPickupConstraint
from prompts.virtualhome.virtualhome import PromptVirtualHome
from prompts.ravens.real_world_1 import PromptRealWorld1
names = {
    'real-world-1':PromptRealWorld1,
    'put-block-in-bowl': PromptRavensPutBlockInBowl,
    'towers-of-hanoi-seq': PromptRavensTowersOfHanoiSeq,
    'towers-of-hanoi-seq-seen-colors': PromptRavensTowersOfHanoiSeqSeen,
    'babyai-pickup': PromptBabyAIPickup,
    'babyai-pickup-constraint': PromptBabyAIPickupConstraint,
    'towers-of-hanoi-seq-seen-colors-full': PromptRavensTowersOfHanoiSeqSolve,
    'virtualhome': PromptVirtualHome
}
