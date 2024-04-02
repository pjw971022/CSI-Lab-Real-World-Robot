"""Prompts for all envs."""
<<<<<<< HEAD:_prompts/__init__.py
from _prompts.ravens.put_block_in_bowl import PromptRavensPutBlockInBowl
from _prompts.ravens.towers_of_hanoi_seq import PromptRavensTowersOfHanoiSeq, PromptRavensTowersOfHanoiSeqSeen, PromptRavensTowersOfHanoiSeqSolve
from _prompts.babyai.babyai_pickup import PromptBabyAIPickup,PromptBabyAIPickupConstraint
from _prompts.virtualhome.virtualhome import PromptVirtualHome
from _prompts.realworld.cleanup import PromptRealWorldCleanup
from _prompts.realworld.packing_shapes import PromptRealWorldPackingShapes
from _prompts.realworld.packing_objects import PromptRealWorldPackingObjects
from _prompts.realworld.making_word import PromptRealWorldMakingWord
from _prompts.realworld.speech2demo import PromptRealWorldspeech2Demo
=======
from prompts.ravens.put_block_in_bowl import PromptRavensPutBlockInBowl
from prompts.ravens.towers_of_hanoi_seq import PromptRavensTowersOfHanoiSeq, PromptRavensTowersOfHanoiSeqSeen, PromptRavensTowersOfHanoiSeqSolve
from prompts.babyai.babyai_pickup import PromptBabyAIPickup,PromptBabyAIPickupConstraint
from prompts.virtualhome.virtualhome import PromptVirtualHome
from prompts.realworld.cleanup import PromptRealWorldCleanup
from prompts.realworld.packing_shapes import PromptRealWorldPackingShapes
from prompts.realworld.packing_objects import PromptRealWorldPackingObjects
from prompts.realworld.making_word import PromptRealWorldMakingWord
from prompts.realworld.speech2demo import PromptRealWorldspeech2Demo
from prompts.realworld.multi_modal import PromptRealWorldMultiModal
>>>>>>> 20e4f73066fb102265a9e612c7b8494efa30ed73:deprecated/prompts/__init__.py
names = {
    'real-world-multi-modal':PromptRealWorldMultiModal,
    'real-world-speech2demo':PromptRealWorldspeech2Demo,
    'real-world-cleanup':PromptRealWorldCleanup,
    'real-world-packing-objects': PromptRealWorldPackingObjects,
    'real-world-packing-shapes': PromptRealWorldPackingShapes,
    'real-world-making-word': PromptRealWorldMakingWord,    
    'put-block-in-bowl': PromptRavensPutBlockInBowl,
    'towers-of-hanoi-seq': PromptRavensTowersOfHanoiSeq,
    'towers-of-hanoi-seq-seen-colors': PromptRavensTowersOfHanoiSeqSeen,
    'babyai-pickup': PromptBabyAIPickup,
    'babyai-pickup-constraint': PromptBabyAIPickupConstraint,
    'towers-of-hanoi-seq-seen-colors-full': PromptRavensTowersOfHanoiSeqSolve,
    'virtualhome': PromptVirtualHome
}
