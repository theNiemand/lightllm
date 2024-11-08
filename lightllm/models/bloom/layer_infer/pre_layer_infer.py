import torch
from vllm.distributed import tensor_model_parallel_all_reduce
from lightllm.common.basemodel import PreLayerInferTpl
from lightllm.common.basemodel import InferStateInfo
from lightllm.models.bloom.layer_weights.pre_and_post_layer_weight import BloomPreAndPostLayerWeight
from lightllm.utils.infer_utils import mark_cost_time
from lightllm.models.bloom.triton_kernel.layernorm import layernorm_forward
from lightllm.models.llama.triton_kernel.embedding import embedding


class BloomPreLayerInfer(PreLayerInferTpl):
    """ """

    def __init__(self, tp_rank, world_size, network_config, mode):
        super().__init__(tp_rank, world_size, network_config, mode)
        self.eps_ = network_config["layer_norm_epsilon"]
        tp_vocab_size_ = network_config["vocab_size"] // self.world_size_
        self.vob_start_id_ = tp_vocab_size_ * self.tp_rank_
        self.vob_end_id_ = tp_vocab_size_ * (self.tp_rank_ + 1)
        return

    def _norm(self, input, infer_state, layer_weight: BloomPreAndPostLayerWeight) -> torch.Tensor:
        return layernorm_forward(input, layer_weight.pre_norm_weight_, layer_weight.pre_norm_bias_, eps=self.eps_)

    def context_forward(self, input_ids, infer_state: InferStateInfo, layer_weight: BloomPreAndPostLayerWeight):
        total_token_num = infer_state.total_token_num
        input_ids = input_ids[0:total_token_num]

        input_embdings = self.alloc_tensor(
            (input_ids.shape[0], layer_weight.wte_weight_.shape[1]), dtype=layer_weight.data_type_
        )
        embedding(input_ids, layer_weight.wte_weight_, self.vob_start_id_, self.vob_end_id_, input_embdings)
        if self.world_size_ > 1:
            input_embdings = tensor_model_parallel_all_reduce(input_embdings)
        input_embdings = self._norm(input_embdings, infer_state, layer_weight)
        return input_embdings

    def token_forward(self, input_ids, infer_state: InferStateInfo, layer_weight: BloomPreAndPostLayerWeight):
        input_embdings = self.alloc_tensor(
            (input_ids.shape[0], layer_weight.wte_weight_.shape[1]), dtype=layer_weight.data_type_
        )
        embedding(input_ids, layer_weight.wte_weight_, self.vob_start_id_, self.vob_end_id_, input_embdings)
        if self.world_size_ > 1:
            input_embdings = tensor_model_parallel_all_reduce(input_embdings)
        input_embdings = self._norm(input_embdings, infer_state, layer_weight)
        return input_embdings
