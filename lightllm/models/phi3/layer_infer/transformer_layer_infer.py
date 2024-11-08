import torch
import torch.functional as F
import numpy as np
from functools import partial

from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.phi3.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.phi3.triton_kernel.context_flashattention_nopad import (
    context_attention_fwd,
    context_attention_fwd_no_prompt_cache,
)
from lightllm.models.phi3.triton_kernel.destindex_copy_kv import destindex_copy_kv
from lightllm.models.phi3.layer_weights.transformer_layer_weight import Phi3TransformerLayerWeight
from lightllm.models.llama.infer_struct import LlamaInferStateInfo


class Phi3TransformerLayerInfer(LlamaTransformerLayerInfer):
    """ """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        return

    def _bind_attention(self):
        self._context_attention_kernel = partial(Phi3TransformerLayerInfer._context_attention_kernel, self)
        self._copy_kv_to_mem_cache = partial(Phi3TransformerLayerInfer._copy_kv_to_mem_cache_normal, self)
        self._token_attention_kernel = partial(Phi3TransformerLayerInfer._token_decode_attention_flashdecoding, self)
        return

    def _get_qkv(self, input_emb, cache_kv, infer_state: LlamaInferStateInfo, layer_weight: Phi3TransformerLayerWeight):
        q = torch.mm(input_emb.view(-1, self.embed_dim_), layer_weight.q_weight_)
        torch.mm(
            input_emb.view(-1, self.embed_dim_),
            layer_weight.kv_weight_,
            out=cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_),
        )
        rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, 0 : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
        )
        return q, cache_kv

    def _copy_kv_to_mem_cache(self, buffer, mem_index, mem_manager):
        destindex_copy_kv(buffer, mem_index, mem_manager.kv_buffer[self.layer_num_])
        return

    def _context_attention_kernel(
        self, q, kv, infer_state: LlamaInferStateInfo, layer_weight, out=None
    ) -> torch.Tensor:
        o_tensor = torch.empty_like(q) if out is None else out
        if infer_state.use_dynamic_prompt_cache:
            kv = infer_state.mem_manager.kv_buffer[self.layer_num_]
            context_attention_fwd(
                q.view(-1, self.tp_q_head_num_, self.head_dim_),
                kv[:, 0 : self.tp_k_head_num_, :],
                kv[:, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :],
                o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_),
                infer_state.b_req_idx,
                infer_state.b_start_loc,
                infer_state.b_seq_len,
                infer_state.b_ready_cache_len,
                infer_state.max_len_in_batch,
                infer_state.req_manager.req_to_token_indexs,
            )
        else:
            context_attention_fwd_no_prompt_cache(
                q.view(-1, self.tp_q_head_num_, self.head_dim_),
                kv[:, 0 : self.tp_k_head_num_, :],
                kv[:, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :],
                o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_),
                infer_state.b_start_loc,
                infer_state.b_seq_len,
                infer_state.max_len_in_batch,
            )

        return o_tensor

    def _token_decode_attention_flashdecoding(self, q, infer_state: LlamaInferStateInfo, layer_weight, out=None):
        from lightllm.models.phi3.triton_kernel.flash_decoding import token_decode_attention_flash_decoding

        cache_k = infer_state.mem_manager.kv_buffer[self.layer_num_][:, 0 : self.tp_k_head_num_, :]
        cache_v = infer_state.mem_manager.kv_buffer[self.layer_num_][
            :, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :
        ]
        return token_decode_attention_flash_decoding(
            q,
            infer_state,
            self.tp_q_head_num_,
            self.head_dim_,
            cache_k,
            cache_v,
            out=out,
            alloc_tensor_func=self.alloc_tensor,
        )
