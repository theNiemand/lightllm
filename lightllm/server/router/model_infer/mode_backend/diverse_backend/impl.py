import torch
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.utils.infer_utils import set_random_seed
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.server.router.model_infer.infer_batch import (
    g_infer_context,
    InferReq,
    InferReqGroup,
    InferSamplingParams,
)
from lightllm.server.core.objs import ReqRunStatus, FinishStatus
from lightllm.utils.log_utils import init_logger
from lightllm.server.tokenizer import get_tokenizer
from lightllm.server.router.model_infer.mode_backend.beamsearch.pre_process import (
    prepare_prefill_inputs,
    prepare_decode_inputs,
)
from .post_process import sample


class DiversehBackend(ModeBackend):
    def __init__(self) -> None:
        super().__init__()

    def init_custom(self):
        pass

    @calculate_time(show=False, min_cost_ms=300)
    def prefill_batch(self, batch_id):
        return self.forward(batch_id, is_prefill=True)

    @calculate_time(show=True, min_cost_ms=200)
    def decode_batch(self, batch_id):
        return self.forward(batch_id, is_prefill=False)

    def build_group(self, batch):
        for r_id in batch.request_ids:
            req = g_infer_context.requests_mapping[r_id]
            group_req_id = req.group_req_id
            best_of = req.sampling_param.best_of
            if group_req_id not in g_infer_context.group_mapping:
                g_infer_context.group_mapping[group_req_id] = InferReqGroup(group_req_id=group_req_id, best_of=best_of)
            g_infer_context.group_mapping[group_req_id].add_req(r_id)

    def forward(self, batch_id, is_prefill):
        # special code for return all prompt_logprobs
        output_dict = {}
        batch = self.cache.pop(batch_id)
        if is_prefill:
            self.build_group(batch)
            kwargs, run_reqs = prepare_prefill_inputs(batch, self.radix_cache, self.is_multimodal)
        else:
            kwargs, run_reqs = prepare_decode_inputs(batch, self.radix_cache)

        logits = self.model.forward(**kwargs)
        next_token_id_groups, next_token_logprob_groups = sample(
            logits, run_reqs, is_prefill, self.model.vocab_size, self.model.req_manager, self.eos_id
        )
        start = 0
        for req_group_obj in run_reqs:
            # prefill and decode is same
            alive_req_id = []
            if req_group_obj.best_of != req_group_obj.refs:
                req_group_obj.update_filter()
            for i in range(req_group_obj.best_of):
                req_obj = req_group_obj.get_req(i)
                req_obj.cur_kv_len = len(req_obj.input_token_ids)
                req_obj.input_token_ids.append(next_token_id_groups[start + i])
                req_obj.out_token_id_count[next_token_id_groups[start + i]] += 1
                req_obj.cum_logprob += next_token_logprob_groups[start + i]
                req_obj.update_finish_status(self.eos_id)
                metadata = {
                    "id": int(next_token_id_groups[start + i]),
                    "logprob": float(next_token_logprob_groups[start + i]),
                    "cumlogprob": req_obj.cum_logprob,
                }
                output_dict[req_obj.r_id] = (
                    req_obj.req_status,
                    req_obj.cur_kv_len,
                    req_obj.get_output_len(),
                    [(int(next_token_id_groups[start + i]), metadata)],
                    req_obj.finish_status.value,  # 转化为整数，避免传送大对象,
                    None,
                )  # 请求状态， 当前占用的kv的长度， 当前输出token的数量， 输出的token的id和元信息列表， 是否推理结束的状态， 额外保留参数
                if not req_obj.finish_status.is_finished():
                    alive_req_id.append(req_obj.r_id)
            start += req_group_obj.best_of
            req_group_obj.best_of = len(alive_req_id)
            req_group_obj.req_group = alive_req_id

        self.cache[batch.batch_id] = batch
        return output_dict
