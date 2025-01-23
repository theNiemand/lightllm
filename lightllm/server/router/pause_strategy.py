import uuid
import numpy as np
from typing import List, Tuple
from .batch import Batch, Req
from lightllm.server.router.req_queue.base_queue import BaseQueue
from lightllm.server.core.objs import ReqRunStatus


class Strategy:
    def ordering_reqs(self, batch: Batch):
        raise not NotImplemented()


class Fcfs(Strategy):
    def __init__(self) -> None:
        super().__init__()

    def ordering_reqs(self, batch: Batch):
        reqs = [req for req in batch.reqs]
        return sorted(reqs, key=lambda req: req.request_id, reverse=True)


class Sfj(Strategy):
    def __init__(self) -> None:
        super().__init__()

    def ordering_reqs(self, batch: Batch):
        reqs = [req for req in batch.reqs]
        return sorted(reqs, key=lambda req: req.max_output_len - len(req.output_ids), reverse=True)


class Hrnn(Strategy):
    def __init__(self) -> None:
        super().__init__()

    def ordering_reqs(self, batch: Batch):
        reqs = [req for req in batch.reqs]
        return sorted(
            reqs,
            key=lambda req: (req.input_len + req.max_output_len - len(req.output_ids)) / req.input_len,
            reverse=True,
        )


def select_paused_reqs(batch: Batch, strategy: Strategy, req_queue: BaseQueue, max_total_token_num):
    reqs: List[Req] = strategy.ordering_reqs(batch)
    if len(reqs) == 0:
        return []

    group_req_id = reqs[0].group_req_id
    pause_reqs = []
    for req in reqs:
        if req.group_req_id == group_req_id:
            pause_reqs.append(req)
            batch.pop_req(req.request_id)
        else:
            break

    # 更新请求状态
    for req in pause_reqs:
        req.req_status = ReqRunStatus.PAUSED_AND_OFFLOAD
        # 重构后改由推理进程进行长度设置。
        # req.shm_cur_kv_len = 0

    req_queue.back_to_wait_list(pause_reqs)

    return pause_reqs
