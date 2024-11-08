import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np

from lightllm.models.bloom.layer_weights.pre_and_post_layer_weight import BloomPreAndPostLayerWeight
from einops import rearrange
from lightllm.common.basemodel import InferStateInfo, PostLayerInferTpl
from lightllm.models.bloom.triton_kernel.layernorm import layernorm_forward


class BloomPostLayerInfer(PostLayerInferTpl):
    """ """

    def __init__(self, tp_rank, world_size, network_config, mode):
        super().__init__(tp_rank, world_size, network_config, mode)
        assert network_config["vocab_size"] % self.world_size_ == 0
        self.eps_ = network_config["layer_norm_epsilon"]
        self.vocab_size_ = network_config["vocab_size"]
        self.embed_dim_ = network_config["n_embed"]
        return

    def _norm(self, input, infer_state, layer_weight: BloomPreAndPostLayerWeight) -> torch.Tensor:
        return layernorm_forward(input, layer_weight.final_norm_weight_, layer_weight.final_norm_bias_, eps=self.eps_)

    def token_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight: BloomPreAndPostLayerWeight):
        batch_size = infer_state.batch_size
        last_input = self.alloc_tensor(
            (batch_size, self.embed_dim_), device=input_embdings.device, dtype=input_embdings.dtype
        )
        if infer_state.is_prefill:
            last_index = (
                torch.cumsum(infer_state.b_seq_len - infer_state.b_ready_cache_len, dim=0, dtype=torch.long) - 1
            )
            last_input[:, :] = input_embdings[last_index, :]
        else:
            last_input[:, :] = input_embdings[-batch_size:, :]

        input_embdings_dtype = input_embdings.dtype
        input_embdings = None
        last_input = self._norm(last_input, infer_state, layer_weight)
        last_input = rearrange(last_input, "batch embed_dim -> embed_dim batch").contiguous().reshape(-1, batch_size)
        # (split_vob_size, hidden_size) (hidden_size, batch_size)
        logic_batch = torch.mm(layer_weight.lm_head_weight_, last_input)
        last_input = None
        if self.world_size_ == 1:
            gather_data = logic_batch
        else:
            gather_data = self.alloc_tensor(
                (self.vocab_size_, batch_size), device=logic_batch.device, dtype=input_embdings_dtype
            )
            split_size = self.vocab_size_ // self.world_size_
            dist.all_gather(
                [gather_data[i * split_size : (i + 1) * split_size, :] for i in range(self.world_size_)],
                logic_batch,
                group=None,
                async_op=False,
            )
        logic_batch = None

        ans_logics = gather_data.permute(1, 0).float()
        gather_data = None
        return ans_logics
