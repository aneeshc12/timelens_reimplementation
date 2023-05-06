import torch as th
import torch.nn.functional as F
from timelens import refine_warp_network, warp_network
from timelens.superslomo import unet

def _pack_input_for_attention_computation(example):
    fusion = example["middle"]["fusion"]
    number_of_examples, _, height, width = fusion.size()
    return th.cat(
        [
            example["after"]["flow"],
            example["middle"]["after_refined_warped"],
            example["before"]["flow"],
            example["middle"]["before_refined_warped"],
            example["middle"]["fusion"],
            th.Tensor(example["middle"]["weight"])
            .view(-1, 1, 1, 1)
            .expand(number_of_examples, 1, height, width)
            .type(fusion.type()),
        ],
        dim=1,
    )
"""
cats all 6 things considered when calculating attention
1. tau to before optical flow
2. tau to after optical flow
3. tau to before refined warp (stored in middle)
4. tau to after refined warp (stored in middle)
5. middle fusion image
6. tau value (expanded into a 4d tensor to be compatible for concatenation)
"""


def _compute_weighted_average(attention, before_refined, after_refined, fusion):
    return (
        attention[:, 0, ...].unsqueeze(1) * before_refined
        + attention[:, 1, ...].unsqueeze(1) * after_refined
        + attention[:, 2, ...].unsqueeze(1) * fusion
    )


class AttentionAverage(refine_warp_network.RefineWarp):
    """
    inherits from refinewarp network
        which inherits from warp network AND fusion network (uses both to refine)

    all of those functions and attributes can be called here
    """

    def __init__(self):
        warp_network.Warp.__init__(self)
        self.fusion_network = unet.UNet(2 * 3 + 2 * 5, 3, False)
        self.flow_refinement_network = unet.UNet(9, 4, False)
        """
        all three heads defined here with appropriate channels
        """

        self.attention_network = unet.UNet(14, 3, False)
        """
        computes attention
        """

    def run_fast(self, example):
        example['middle']['before_refined_warped'], \
        example['middle']['after_refined_warped'] = refine_warp_network.RefineWarp.run_fast(self, example)
        """
        add refined and warped images to ["middle"] from refine warp
        (picks up synthesis and initial optical flow from example itself)

        calls parent class function
        """

        attention_scores = self.attention_network(
            _pack_input_for_attention_computation(example)
        )
        """
        cats all 6 things to consider together, run it through the attention UNet
        """

        attention = F.softmax(attention_scores, dim=1)
        average = _compute_weighted_average(
            attention,
            example['middle']['before_refined_warped'],
            example['middle']['after_refined_warped'],
            example['middle']['fusion']
        )
        """
        softmax and compute the weighted average
        """

        return average, attention
    """
    fast method,
    uses refine warp fast (notes in refine warp part)
    """

    def run_attention_averaging(self, example):
        refine_warp_network.RefineWarp.run_and_pack_to_example(self, example)       #"""only thing that changes"""
        """
        calls parent class function
        """

        attention_scores = self.attention_network(
            _pack_input_for_attention_computation(example)
        )
        attention = F.softmax(attention_scores, dim=1)
        average = _compute_weighted_average(
            attention,
            example["middle"]["before_refined_warped"],
            example["middle"]["after_refined_warped"],
            example["middle"]["fusion"],
        )
        return average, attention
    """
    slow method ig
    refine warp net packs by itself instead of just outputting
    """

    def run_and_pack_to_example(self, example):
        (
            example["middle"]["attention_average"],
            example["middle"]["attention"],
        ) = self.run_attention_averaging(example)
    """
    afaik not used
    """

    def forward(self, example):
        return self.run_attention_averaging(example)
    """
    slow method
    """
