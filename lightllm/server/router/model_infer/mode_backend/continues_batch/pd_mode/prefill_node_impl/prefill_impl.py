import os
import time
import threading
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from typing import List
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.utils.infer_utils import set_random_seed
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.server.router.model_infer.infer_batch import InferBatch, InferReq, InferSamplingParams, requests_mapping
from lightllm.server.core.objs import ReqRunStatus, FinishStatus
from lightllm.server.pd_io_struct import KVMoveTask, DecodeNodeInfo
from lightllm.utils.log_utils import init_logger
from ...pre_process import prepare_prefill_inputs, prepare_decode_inputs
from ...post_process import sample
from lightllm.common.basemodel.infer_lock import g_router_lock, g_infer_state_lock
from rpyc.utils.server import ThreadedServer
from .prefill_task_cache import g_kv_move_task_cache
from lightllm.utils.device_utils import kv_trans_use_p2p

logger = init_logger(__name__)


class ContinuesBatchBackendForPrefillNode(ModeBackend):
    def __init__(self, info_queue: mp.Queue, mem_queue: mp.Queue) -> None:
        super().__init__()
        self.info_queue: mp.Queue = info_queue
        self.mem_queue: mp.Queue = mem_queue

    def init_custom(self):
        self.lock_nccl_group = dist.new_group(backend="gloo")
        from .prefill_infer_rpyc import PDPrefillInferRpcServer

        socket_path = f"/tmp/prefill_node_infer_rpyc_{self.pd_rpyc_ports[self.tp_rank]}"
        if os.path.exists(socket_path):
            os.remove(socket_path)

        t = ThreadedServer(
            PDPrefillInferRpcServer(self), socket_path=socket_path, protocol_config={"allow_pickle": True}
        )
        threading.Thread(target=lambda: t.start(), daemon=True).start()

        if kv_trans_use_p2p():
            from ..p2p_fix import reduce_tensor

            mp.reductions.reduce_tensor.__code__ = reduce_tensor.__code__

        return

    @calculate_time(show=False, min_cost_ms=300)
    def prefill_batch(self, batch_id):
        ans = self.forward(batch_id, is_prefill=True)
        return ans

    @calculate_time(show=True, min_cost_ms=200)
    def decode_batch(self, batch_id):
        return self.forward(batch_id, is_prefill=False)

    def forward(self, batch_id, is_prefill):
        # special code for return all prompt_logprobs
        batch: InferBatch = self.cache.pop(batch_id)
        if is_prefill:
            kwargs, run_reqs = prepare_prefill_inputs(batch, self.radix_cache, self.is_multimodal)
        else:
            kwargs, run_reqs = prepare_decode_inputs(batch, self.radix_cache)

        logits = self.model.forward(**kwargs)
        next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

        for req_obj, next_token_id, next_token_logprob in zip(run_reqs, next_token_ids, next_token_logprobs):
            
            if self.tp_rank < self.dp_size:
                # prefill and decode is same
                req_obj: InferReq = req_obj
                req_obj.shm_req.cur_kv_len = len(req_obj.get_input_token_ids())
                req_obj.set_next_gen_token_id(next_token_id, next_token_logprob)
                req_obj.out_token_id_count[next_token_id] += 1
                req_obj.update_finish_status(self.eos_id)
                req_obj.shm_req.candetoken_out_len = req_obj.shm_req.cur_output_len
                
        if is_prefill:
            self.prefill_req_handle_and_frozen_tokens(run_reqs)

        self.cache[batch.batch_id] = batch
        return

    def prefill_req_handle_and_frozen_tokens(self, run_reqs: List[InferReq]):
        # 提前在radix cache中回收相关的信息，并添加引用信息
        if self.tp_rank < self.dp_size:
            logger.info("prefill_req_handle_and_frozen_tokens")
        g_infer_state_lock.acquire()
        try:
            for req in run_reqs:
                req:InferReq = req
                key = req.get_input_token_ids()[0:req.shm_req.cur_kv_len]
                key = torch.tensor(key, dtype=torch.int64, device="cpu")
                value = self.model.req_manager.req_to_token_indexs[req.req_idx][: req.shm_req.cur_kv_len].detach().cpu()
                prefix_len = self.radix_cache.insert(key, value)
                self.model.mem_manager.free(self.model.req_manager.req_to_token_indexs[req.req_idx][:prefix_len])
                if req.shared_kv_node is not None:
                    self.radix_cache.dec_node_ref_counter(req.shared_kv_node)
                    req.shared_kv_node = None
                
                # 等待所有的请求都将radix cache 操作完成，不然下面的 req.shm_req.cur_kv_len = 0 可能会让
                # 其他进程得到错误的 kv 长度信息，然后执行错误的操作。
                dist.barrier()
                if self.tp_rank < self.dp_size:
                    req.shm_req.cur_kv_len = 0
                
                if req.shm_req.sample_params.move_kv_to_decode_node.exists:
                    # 注意兼容纯tp 和 tp dp 混合模式的逻辑
                    if self.tp_rank < self.dp_size:
                        g_router_lock.acquire()
                        self.shared_token_load.add_frozened_token_count(len(key), self.tp_rank)
                        g_router_lock.release()

                    share_node, kv_len, value = self.radix_cache.match_prefix(key, update_refs=True)
                    assert len(key) == len(value)
                    # 将下面的请求放入到任务队列中, 注意要使用raidx cache 返回的value
                    decode_node_info = DecodeNodeInfo(**req.shm_req.sample_params.move_kv_to_decode_node.to_dict())
                    task = KVMoveTask(
                        group_request_id=req.shm_req.group_req_id,
                        input_tokens=key.tolist(),
                        prefill_token_indexes=value.tolist(),
                        decode_token_indexes=None,
                        prefill_node_id=self.args.pd_node_id,
                        decode_node=decode_node_info,
                        move_kv_len=None,
                        prefill_dp_index=0 if self.dp_size == 1 else self.tp_rank,
                        decode_dp_index=None,
                        mark_start_time=time.time(),
                    )
                    g_kv_move_task_cache[task.group_request_id] = (task, share_node)

                    # 注意兼容纯 tp 和 tp dp 混合模式的逻辑
                    if self.tp_rank < self.dp_size:
                        self.info_queue.put(task)
        except BaseException as e:
            logger.exception(str(e))
        g_infer_state_lock.release()
        if self.tp_rank < self.dp_size:
            logger.info("prefill_req_handle_and_frozen_tokens end")
        return
