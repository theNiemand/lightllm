import torch
import torch.functional as F
import numpy as np
from typing import Tuple
import triton
from lightllm.models.llama.infer_struct import LlamaInferStateInfo

from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer, rotary_emb_fwd

from lightllm.models.qwen2.layer_weights.transformer_layer_weight import Qwen2TransformerLayerWeight


class Qwen2TransformerLayerInfer(LlamaTransformerLayerInfer):
    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        return

    def _get_qkv(
        self,
        input: torch.Tensor,
        cache_kv: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: Qwen2TransformerLayerWeight,
    ) -> torch.Tensor:
        input = input.view(-1, self.embed_dim_)
        dtype = input.dtype
        q = self.alloc_tensor((input.shape[0], layer_weight.q_weight_.shape[1]), dtype=dtype)
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
