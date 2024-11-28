import time
import asyncio
import enum
from dataclasses import dataclass
from .sampling_params import SamplingParams
from .multimodal_params import MultimodalParams
from typing import Dict, List, Optional, Tuple, Union
from lightllm.server.req_id_generator import convert_sub_id_to_group_id
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class ReqRunStatus(enum.Enum):
    WAIT_IN_QUEUE = 0  # 在队列中等待
    RUNNING = 1  # 运行
    PAUSED_AND_OFFLOAD = 2  # 暂停卸载KV
    RERUNNING_FROM_OFFLOAD = 3  # 从卸载KV中恢复


class FinishStatus(enum.Enum):
    NO_FINISH = 0  # 没有结束
    FINISHED_STOP = 1  # 因为遇到了STOP token 而结束
    FINISHED_LENGTH = 2  # 因为长度达到了最大长度而结束
    FINISHED_ABORT = 3  # 因为请求被中止而结束

    def is_finished(self):
        return 1 <= self.value <= 3

    def is_aborted(self):
        return self == FinishStatus.FINISHED_ABORT

    def get_finish_reason(self):
        if self == FinishStatus.FINISHED_STOP:
            finish_reason = "stop"
        elif self == FinishStatus.FINISHED_LENGTH:
            finish_reason = "length"
        elif self == FinishStatus.FINISHED_ABORT:
            finish_reason = "abort"
        else:
            finish_reason = None
        return finish_reason


class Req:
    def __init__(self, request_id, prompt_ids, sample_params: SamplingParams, multimodal_params: MultimodalParams):
        self.request_id = request_id
        self.group_req_id = convert_sub_id_to_group_id(request_id)
        self.prompt_ids = prompt_ids
        self.input_len = len(prompt_ids)
        self.max_output_len = sample_params.max_new_tokens
        self.sample_params = sample_params
        self.multimodal_params = multimodal_params
        # output_ids 和 output_metadata_list 的维护暂时没有用了，先注释掉，不清除，后续可能会有用
        # self.output_ids = []
        # self.output_metadata_list = []

        self.req_status = ReqRunStatus.WAIT_IN_QUEUE
        self.finish_status = FinishStatus.NO_FINISH
        self.cur_kv_len = 0  # 当前已经占用掉 token 的 kv len 长度
        self.cur_output_len = 0  # 当前已经生成token 长度
        self.start_time = 0  # 请求开始时间
        self.prompt_cache_len = 0  # cache 命中长度
        return

    def to_rpc_obj(self):
        return {
            "request_id": self.request_id,
            "group_req_id": self.group_req_id,
            "input_id": self.prompt_ids,
            "sampling_param": self.sample_params.to_dict(),
            "multimodal_params": self.multimodal_params.to_dict(),
            "req_status": self.req_status,
            "dp_index": self.sample_params.suggested_dp_index,
        }

    def __repr__(self):
        return f"request_id(n={self.request_id}, " f"prompt_ids={self.prompt_ids}, "

    def get_used_tokens(self):
        return max(0, self.cur_kv_len)

    def get_tuple_tokens(self, is_busy, router_max_new_token_len):
        raise Exception("need to impl")

    def get_decode_need_tokens(self):
        raise Exception("need to impl")

    def get_first_router_need_tokens(self):
        raise Exception("need to impl")


class NormalReq(Req):
    def __init__(self, request_id, prompt_ids, sample_params: SamplingParams, multimodal_params: MultimodalParams):
        super().__init__(request_id, prompt_ids, sample_params, multimodal_params)
        return

    def get_tuple_tokens(self, is_busy, router_max_new_token_len):
        """
        普通continues batch调度模式, 先prefill 后 decode 的估计方式 的实现
        """
        has_out_len = self.cur_output_len
        if self.sample_params.ignore_eos:
            cur_max_new_token_len = self.max_output_len
        elif is_busy:
            cur_max_new_token_len = self.max_output_len
        else:
            # 用当前输出长度的 1.1 倍作为预估输出长度的另一个参考量，用于更新估计的最大输出长度量
            # 后续会更新为更合理的统计条件概率估计方式 to do
            cur_max_new_token_len = min(self.max_output_len, max(int(1.1 * has_out_len), router_max_new_token_len))

        if self.req_status == ReqRunStatus.RUNNING:
            return (self.input_len + has_out_len, max(0, cur_max_new_token_len - has_out_len - 1))
        elif self.req_status == ReqRunStatus.WAIT_IN_QUEUE:
            return (self.input_len + 1, max(0, cur_max_new_token_len - 1 - 1))
        elif self.req_status == ReqRunStatus.PAUSED_AND_OFFLOAD:
            return (self.input_len + has_out_len + 1, max(0, cur_max_new_token_len - has_out_len - 1 - 1))
        else:
            assert False, "error state"
        return

    def get_decode_need_tokens(self):
        if self.req_status == ReqRunStatus.RUNNING:
            return 1
        else:
            assert False, "error state"

    def get_first_router_need_tokens(self):
        if self.req_status == ReqRunStatus.WAIT_IN_QUEUE:
            return self.input_len
        elif self.req_status == ReqRunStatus.PAUSED_AND_OFFLOAD:
            return self.input_len + self.cur_output_len
        else:
            assert False, "error state"


class TokenHealingReq(NormalReq):
    def __init__(self, request_id, prompt_ids, sample_params: SamplingParams, multimodal_params: MultimodalParams):
        super().__init__(request_id, prompt_ids, sample_params, multimodal_params)

        for prefix_token_num in range(2, -1, -1):
            if len(self.prompt_ids) > prefix_token_num:
                self.prefix_token_ids = self.prompt_ids[-prefix_token_num:]
                del self.prompt_ids[-prefix_token_num:]
                self.prefix_token_num = prefix_token_num
                break

        # 因为原始的输出token数量，会被中间的前缀补全占用decode次数，
        # 所以默认多添加一些decode步数
        self.added_max_new_tokens = 6
        self.sample_params.max_new_tokens += self.added_max_new_tokens

    def get_tuple_tokens(self, is_busy, router_max_new_token_len):
        ans = super().get_tuple_tokens(is_busy, router_max_new_token_len)
        # token healing mode 下，由于估计的生成token数据对应的生存周期可能会不准确
        # 所以为了缓解调度带来的显存估计问题，对于生成token的长度 + 3来缓解可能的估计
        # 错误问题。
        return (ans[0], ans[1] + self.prefix_token_num + self.added_max_new_tokens)

    def to_rpc_obj(self):
        ans = super().to_rpc_obj()
        ans["prefix_token_ids"] = self.prefix_token_ids
        return ans


class SplitFuseReq(Req):
    def __init__(
        self,
        request_id,
        prompt_ids,
        sample_params: SamplingParams,
        multimodal_params: MultimodalParams,
        splitfuse_block_size=None,
    ):
        super().__init__(request_id, prompt_ids, sample_params, multimodal_params)
        self.splitfuse_block_size = splitfuse_block_size
        return

    def get_tuple_tokens(self, is_busy, router_max_new_token_len):
        """
        splitfuse 调度模式的实现
        """
        has_out_len = self.cur_output_len
        if self.sample_params.ignore_eos:
            cur_max_new_token_len = self.max_output_len
        elif is_busy:
            cur_max_new_token_len = self.max_output_len
        else:
            cur_max_new_token_len = min(self.max_output_len, max(int(1.1 * has_out_len), router_max_new_token_len))

        if self.req_status == ReqRunStatus.RUNNING:
            return (
                self.input_len + has_out_len,
                max(
                    0,
                    (self.input_len + has_out_len - self.cur_kv_len + self.splitfuse_block_size - 1)
                    // self.splitfuse_block_size
                    + cur_max_new_token_len
                    - has_out_len
                    - 1,
                ),
            )
        elif self.req_status == ReqRunStatus.WAIT_IN_QUEUE:
            return (
                self.input_len,
                max(
                    0,
                    (self.input_len + self.splitfuse_block_size - 1) // self.splitfuse_block_size
                    + cur_max_new_token_len
                    - 1,
                ),
            )
        elif self.req_status == ReqRunStatus.PAUSED_AND_OFFLOAD:
            return (
                self.input_len + has_out_len,
                max(
                    0,
                    (self.input_len + has_out_len + self.splitfuse_block_size - 1) // self.splitfuse_block_size
                    + cur_max_new_token_len
                    - has_out_len
                    - 1,
                ),
            )
        else:
            assert False, "error state"
        return

    def get_decode_need_tokens(self):
        """
        splitfuse 调度模式的实现
        """
        if self.req_status == ReqRunStatus.RUNNING:
            return min(self.input_len + self.cur_output_len - self.cur_kv_len, self.splitfuse_block_size)
        else:
            assert False, "error state"

    def get_first_router_need_tokens(self):
        if self.req_status == ReqRunStatus.WAIT_IN_QUEUE:
            return min(self.input_len, self.splitfuse_block_size)
        elif self.req_status == ReqRunStatus.PAUSED_AND_OFFLOAD:
            return min(self.input_len + self.cur_output_len, self.splitfuse_block_size)
        else:
            assert False, "error state"


class ReqDetokenizationState:
    def __init__(
        self,
        group_req_id: int,
        prompt_ids: List[int],
        max_output_len: int,
        ignore_eos: bool,
        skip_special_tokens: bool,
        add_spaces_between_special_tokens: bool,
        print_eos_token: bool,
        best_of: int,
    ) -> None:
        self.request_id = None
        self.group_req_id = group_req_id
        self.prompt_ids = prompt_ids
        self.output_ids = []
        self.output_tokens = []
        self.output_str = ""
        self.sub_texts = []
        self.current_sub_text = []
        self.max_output_len = max_output_len
        self.ignore_eos = ignore_eos
        self.gen_metadata = {}
        self.skip_special_tokens = skip_special_tokens
        self.add_spaces_between_special_tokens = add_spaces_between_special_tokens
        self.print_eos_token = print_eos_token
        self.best_of = best_of


class Batch:
    def __init__(self, batch_id, reqs: List[Req], dp_size: int):
        self.batch_id = batch_id
        self.reqs = reqs
        self.id_to_reqs = {req.request_id: req for req in reqs}
        self.dp_size = dp_size

        # 该参数只会在batch init， prefill， decode 后进行更新，并在剔除请求时减少
        # 在 batch rpc init 之后才会被填充正确的值，初始化为 None
        self.batch_decode_need_tokens = [None for _ in range(dp_size)]
        return

    def input_tokens(self):
        batch_input_tokens = 0
        for req in self.reqs:
            batch_input_tokens += req.input_len
        return batch_input_tokens

    def mark_and_get_finished_req_and_preupdate_status(self):
        unfinished_req_ids, finished_req_ids = [], []
        for req in self.reqs:
            if req.finish_status.is_finished():
                finished_req_ids.append(req.request_id)
                req_dp_index = req.sample_params.suggested_dp_index
                # 标记的时候，也同时更新一些这些请求被移除掉的更新量，有点dirty
                self.batch_decode_need_tokens[req_dp_index] -= req.get_decode_need_tokens()
            else:
                unfinished_req_ids.append(req.request_id)

        return unfinished_req_ids, finished_req_ids

    def filter_out_finished_req(self, unfinished_req_ids, finished_req_ids):
        # update batch
        if len(finished_req_ids) != 0:
            self.reqs = [self.id_to_reqs[req_id] for req_id in unfinished_req_ids]
            self.id_to_reqs = {req.request_id: req for req in self.reqs}
        return

    def pop_req(self, req_id):
        self.reqs = [req for req in self.reqs if req.request_id != req_id]
        req = self.id_to_reqs[req_id]
        self.id_to_reqs.pop(req_id)
        req_dp_index = req.sample_params.suggested_dp_index
        self.batch_decode_need_tokens[req_dp_index] -= req.get_decode_need_tokens()
        return

    def is_clear(self):
        return len(self.reqs) == 0

    def merge(self, mini_batch: "Batch"):
        for _req in mini_batch.reqs:
            self.reqs.append(_req)
        self.id_to_reqs = {req.request_id: req for req in self.reqs}
        for dp_index in range(self.dp_size):
            self.batch_decode_need_tokens[dp_index] += mini_batch.batch_decode_need_tokens[dp_index]
        return

    def dp_merge(self, mini_batch: "Batch"):
        if mini_batch is None:
            return

        for _req in mini_batch.reqs:
            self.reqs.append(_req)
        self.id_to_reqs = {req.request_id: req for req in self.reqs}
        return

    def __repr__(self):
        return f"batch_id={self.batch_id}, " f"reqs={self.reqs}, "

    def simple_log(self):
        return f"batch_id={self.batch_id}, time:{time.time()}s req_ids:{[req.request_id for req in self.reqs]}"


class BatchTokenIdOut:
    def __init__(self):
        self.reqs_infs: List[Tuple[int, int, Dict, int]] = []  # [req_id, new_token_id, gen_metadata, finish_status]


class BatchStrOut:
    def __init__(self):
        self.reqs_infs: List[Tuple[int, str, Dict, int]] = []  # [req_id, token_str, gen_metadata, finish_status]


class AbortReq:
    def __init__(self, group_req_id):
        self.group_req_id = group_req_id
