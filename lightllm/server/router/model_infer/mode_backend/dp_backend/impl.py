import torch
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.utils.infer_utils import set_random_seed
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq, InferSamplingParams
from lightllm.server.core.objs import ReqRunStatus, FinishStatus
from lightllm.utils.log_utils import init_logger
from lightllm.server.router.model_infer.mode_backend.continues_batch.post_process import sample
from .pre_process import prepare_prefill_inputs, prepare_decode_inputs


class DPBackend(ModeBackend):
    def __init__(self) -> None:
        super().__init__()

    @calculate_time(show=False, min_cost_ms=300)
    def prefill_batch(self, batch_id):
        return self.forward(batch_id, is_prefill=True)

    @calculate_time(show=True, min_cost_ms=200)
    def decode_batch(self, batch_id):
        return self.forward(batch_id, is_prefill=False)

    def forward(self, batch_id, is_prefill):
        output_dict = {}
        batch = self.cache.pop(batch_id)
        if is_prefill:
            kwargs, run_reqs, padding_token_num = prepare_prefill_inputs(batch, self.radix_cache, self.is_multimodal)
        else:
            kwargs, run_reqs, padding_token_num = prepare_decode_inputs(batch, self.radix_cache)

        logits = self.model.forward(**kwargs)
        if len(run_reqs) == 0:
            self.cache[batch.batch_id] = batch
            return output_dict
        else:
            if padding_token_num != 0:
                logits = logits[:-padding_token_num]
        next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

        for req_obj, next_token_id, next_token_logprob in zip(run_reqs, next_token_ids, next_token_logprobs):
            # prefill and decode is same
            req_obj: InferReq = req_obj
            req_obj.cur_kv_len = len(req_obj.input_token_ids)
            req_obj.input_token_ids.append(next_token_id)
            req_obj.out_token_id_count[next_token_id] += 1
            req_obj.update_finish_status(self.eos_id)

            metadata = {
                "id": int(next_token_id),
                "logprob": float(next_token_logprob),
            }
            output_dict[req_obj.r_id] = (
                req_obj.req_status,
                req_obj.cur_kv_len,
                req_obj.get_output_len(),
                [(int(next_token_id), metadata)],
                req_obj.finish_status.value,  # 转化为整数，避免传送大对象,
                None,
            )  # 请求状态， 当前占用的kv的长度， 当前输出token的数量， 输出的token的id和元信息列表， 是否推理结束的状态， 额外保留参数
        self.cache[batch.batch_id] = batch
        return output_dict
