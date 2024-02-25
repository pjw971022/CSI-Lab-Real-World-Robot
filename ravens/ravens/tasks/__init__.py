"""Ravens tasks."""

from ravens.tasks.align_box_corner import AlignBoxCorner
from ravens.tasks.assembling_kits import AssemblingKits
from ravens.tasks.assembling_kits import AssemblingKitsEasy
from ravens.tasks.assembling_kits_seq import AssemblingKitsSeqSeenColors
from ravens.tasks.assembling_kits_seq import AssemblingKitsSeqUnseenColors
from ravens.tasks.assembling_kits_seq import AssemblingKitsSeqFull
from ravens.tasks.block_insertion import BlockInsertion
from ravens.tasks.block_insertion import BlockInsertionEasy
from ravens.tasks.block_insertion import BlockInsertionNoFixture
from ravens.tasks.block_insertion import BlockInsertionSixDof
from ravens.tasks.block_insertion import BlockInsertionTranslation
from ravens.tasks.manipulating_rope import ManipulatingRope
from ravens.tasks.align_rope import AlignRope
from ravens.tasks.packing_boxes import PackingBoxes
from ravens.tasks.packing_shapes import PackingShapes
from ravens.tasks.packing_boxes_pairs import PackingBoxesPairsSeenColors
from ravens.tasks.packing_boxes_pairs import PackingBoxesPairsUnseenColors
from ravens.tasks.packing_boxes_pairs import PackingBoxesPairsFull
from ravens.tasks.packing_google_objects import PackingSeenGoogleObjectsSeq
from ravens.tasks.packing_google_objects import PackingUnseenGoogleObjectsSeq
from ravens.tasks.packing_google_objects import PackingSeenGoogleObjectsGroup
from ravens.tasks.packing_google_objects import PackingUnseenGoogleObjectsGroup
from ravens.tasks.palletizing_boxes import PalletizingBoxes
from ravens.tasks.place_red_in_green import PlaceRedInGreen
from ravens.tasks.put_block_in_bowl import PutBlockInBowl
from ravens.tasks.stack_block_pyramid import StackBlockPyramid
from ravens.tasks.stack_block_pyramid_seq import StackBlockPyramidSeqSeenColors
from ravens.tasks.stack_block_pyramid_seq import StackBlockPyramidSeqUnseenColors
from ravens.tasks.stack_block_pyramid_seq import StackBlockPyramidSeqFull
from ravens.tasks.sweeping_piles import SweepingPiles
from ravens.tasks.separating_piles import SeparatingPilesSeenColors
from ravens.tasks.separating_piles import SeparatingPilesUnseenColors
from ravens.tasks.separating_piles import SeparatingPilesFull
from ravens.tasks.task import Task
from ravens.tasks.towers_of_hanoi import TowersOfHanoi
from ravens.tasks.towers_of_hanoi_seq import TowersOfHanoiSeq
from ravens.tasks.towers_of_hanoi_seq_cliport import TowersOfHanoiSeqCliport
from ravens.tasks.cleanup_rw import RealWorldCleanup
from ravens.tasks.packing_shapes_rw import RealWorldPackingShapes
from ravens.tasks.packing_objects_rw import RealWorldPackingObjects
from ravens.tasks.making_word_rw import RealWorldMakingWord
from ravens.tasks.voice2demo_rw import RealWorldVoice2Demo

names = {
    # demo conditioned
    'align-box-corner': AlignBoxCorner,
    'assembling-kits': AssemblingKits,
    'assembling-kits-easy': AssemblingKitsEasy,
    'block-insertion': BlockInsertion,
    'block-insertion-easy': BlockInsertionEasy,
    'block-insertion-nofixture': BlockInsertionNoFixture,
    'block-insertion-sixdof': BlockInsertionSixDof,
    'block-insertion-translation': BlockInsertionTranslation,
    'manipulating-rope': ManipulatingRope,
    'packing-boxes': PackingBoxes,
    'palletizing-boxes': PalletizingBoxes,
    'place-red-in-green': PlaceRedInGreen,
    'stack-block-pyramid': StackBlockPyramid,
    'sweeping-piles': SweepingPiles,
    'towers-of-hanoi': TowersOfHanoi,

    # goal conditioned
    'align-rope': AlignRope,
    'assembling-kits-seq-seen-colors': AssemblingKitsSeqSeenColors,
    'assembling-kits-seq-unseen-colors': AssemblingKitsSeqUnseenColors,
    'assembling-kits-seq-full': AssemblingKitsSeqFull,
    'packing-shapes': PackingShapes,
    'packing-boxes-pairs-seen-colors': PackingBoxesPairsSeenColors,
    'packing-boxes-pairs-unseen-colors': PackingBoxesPairsUnseenColors,
    'packing-boxes-pairs-full': PackingBoxesPairsFull,
    'packing-seen-google-objects-seq': PackingSeenGoogleObjectsSeq,
    'packing-unseen-google-objects-seq': PackingUnseenGoogleObjectsSeq,
    'packing-seen-google-objects-group': PackingSeenGoogleObjectsGroup,
    'packing-unseen-google-objects-group': PackingUnseenGoogleObjectsGroup,
    'put-block-in-bowl': PutBlockInBowl,
    'stack-block-pyramid-seq-seen-colors': StackBlockPyramidSeqSeenColors,
    'stack-block-pyramid-seq-unseen-colors': StackBlockPyramidSeqUnseenColors,
    'stack-block-pyramid-seq-full': StackBlockPyramidSeqFull,
    'separating-piles-seen-colors': SeparatingPilesSeenColors,
    'separating-piles-unseen-colors': SeparatingPilesUnseenColors,
    'separating-piles-full': SeparatingPilesFull,
    'towers-of-hanoi-seq': TowersOfHanoiSeq,
    'towers-of-hanoi-seq-seen-colors': TowersOfHanoiSeqCliport,
    'real-world-cleanup': RealWorldCleanup,
    'real-world-packing-shapes': RealWorldPackingShapes,
    'real-world-packing-objects': RealWorldPackingObjects,
    'real-world-making-word':RealWorldMakingWord,
    'real-world-voice2demo':RealWorldVoice2Demo,
}
