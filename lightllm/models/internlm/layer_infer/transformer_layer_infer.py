import torch
import torch.functional as F
import numpy as np

from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.internlm.layer_weights.transformer_layer_weight import InternlmTransformerLayerWeight
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.infer_struct import LlamaInferStateInfo


class InternlmTransformerLayerInfer(LlamaTransformerLayerInfer):
    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        return

    def _get_qkv(
        self, input, cache_kv, infer_state: LlamaInferStateInfo, layer_weight: InternlmTransformerLayerWeight
    ) -> torch.Tensor:
        input = input.view(-1, self.embed_dim_)
        q = self.alloc_tensor((input.size(0), layer_weight.q_weight_.size(1)), dtype=input.dtype)
        torch.addmm(layer_weight.q_bias_, input, layer_weight.q_weight_, beta=1.0, alpha=1.0, out=q)
        torch.addmm(
            layer_weight.kv_bias_,
            input,
            layer_weight.kv_weight_,
            beta=1.0,
            alpha=1.0,
            out=cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_),
        )
        rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, 0 : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
        )
        return q, cache_kv

    def _get_o(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: InternlmTransformerLayerWeight
    ) -> torch.Tensor:
        input = input.view(-1, self.tp_o_head_num_ * self.head_dim_)
        o_tensor = self.alloc_tensor((input.size(0), layer_weight.o_weight_.size(1)), input.dtype)
        torch.addmm(layer_weight.o_bias_, input, layer_weight.o_weight_, beta=1.0 / self.world_size_, out=o_tensor)
        return o_tensor
