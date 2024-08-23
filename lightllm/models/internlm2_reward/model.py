import os
import json
import torch

from lightllm.models.internlm2_reward.layer_infer.post_layer_infer import Internlm2RewardPostLayerInfer
from lightllm.models.internlm2_reward.layer_weights.pre_and_post_layer_weight import (
    Internlm2RewardPreAndPostLayerWeight,
)
from lightllm.models.internlm2.model import Internlm2TpPartModel


class Internlm2RewardTpPartModel(Internlm2TpPartModel):
    # weight class
    pre_and_post_weight_class = Internlm2RewardPreAndPostLayerWeight

    post_layer_infer_class = Internlm2RewardPostLayerInfer

    def __init__(self, kvargs):
        super().__init__(kvargs)
