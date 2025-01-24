import os
import asyncio
import torch.multiprocessing as mp
import multiprocessing
import threading
import inspect
from datetime import timedelta
from typing import Dict, List, Tuple
from lightllm.server.router.model_infer.mode_backend import (
    ContinuesBatchBackend,
    ReturnPromptLogProbBackend,
    SplitFuseBackend,
    DiversehBackend,
    RewardModelBackend,
    TokenHealingBackend,
    SimpleConstraintBackend,
    FirstTokenConstraintBackend,
    ContinuesBatchBackendForPrefillNode,
    ContinuesBatchBackendForDecodeNode,
    DPBackend,
)
from lightllm.server.core.objs import RpcShmParams, RpcShmResults, ShmSyncStatusArray
from lightllm.utils.log_utils import init_logger
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.process_check import start_parent_check_thread

logger = init_logger(__name__)


class ModelRpcServer:
    def __init__(
        self,
        args,
        tp_rank: int,
        rpc_event: multiprocessing.Event,
        rpc_finished_event: multiprocessing.Event,
        info_queue: mp.Queue,
        mem_queue: mp.Queue,
    ):
        super().__init__()
        self.args = args
        self.world_size = self.args.tp
        self.info_queue = info_queue
        self.mem_queue = mem_queue
        self.rpc_event = rpc_event
        self.rpc_finished_event = rpc_finished_event

        self.rpc_shm_params = RpcShmParams()
        self.rpc_shm_params.create_or_link_shm()
        self.rpc_shm_results = RpcShmResults()
        self.rpc_shm_results.create_or_link_shm()
        self.rpc_shm_sync_status = ShmSyncStatusArray(self.world_size)
        self.rpc_shm_sync_status.create_or_link_shm()

        self.tp_rank = tp_rank

        # 多卡才是跨进程的
        if self.args.tp != 1:
            self.loop_thread = threading.Thread(target=self.rpc_loop)
            self.loop_thread.start()
        return

    def rpc_loop(self):
        error_count = 0
        while True:
            try:
                self.rpc_event.wait()
                func_name, args = self.rpc_shm_params.read_func_params()

                ans = getattr(self, func_name)(*args)
                if ans is not None and self.tp_rank == 0:
                    self.rpc_shm_results.write_func_result(func_name=func_name, ret=ans)

                # 下面得执行顺序不可随意交换, 否则容易出现同步或者死锁问题。
                self.rpc_shm_sync_status.add_mark(self.tp_rank)
                while not self.rpc_shm_sync_status.run_finished():
                    pass

                self.rpc_event.clear()

                self.rpc_shm_sync_status.add_mark1(self.tp_rank)
                while not self.rpc_shm_sync_status.run_finished1():
                    pass

                if self.tp_rank == 0:
                    self.rpc_finished_event.set()

            except BaseException as e:
                logger.exception(str(e))
                error_count += 1

            if error_count >= 3:
                logger.error("infer process error to exit")
                os._exit(-1)

        return

    def init_model(self, kvargs):
        # 填充真正的 rank_id 参数
        kvargs["rank_id"] = self.tp_rank
        self.world_size = kvargs["world_size"]
        is_splitfuse_mode = kvargs.get("is_splitfuse_mode", False)
        return_all_prompt_logprobs = kvargs.get("return_all_prompt_logprobs", False)
        use_reward_model = kvargs.get("use_reward_model", False)
        diverse_mode = kvargs.get("diverse_mode", False)
        is_token_healing = kvargs.get("is_token_healing", False)
        is_first_token_constraint_mode = kvargs.get("is_first_token_constraint_mode", False)
        if kvargs.get("args", None) is not None:
            is_simple_constraint_mode = kvargs.get("args", None).simple_constraint_mode
            is_prefill_node = kvargs.get("args", None).run_mode == "prefill"
            is_decode_node = kvargs.get("args", None).run_mode == "decode"
        else:
            is_simple_constraint_mode = False
            is_prefill_node = False
            is_decode_node = False
        # use_dynamic_prompt_cache = kvargs.get("use_dynamic_prompt_cache", False)
        if is_prefill_node:
            self.backend = ContinuesBatchBackendForPrefillNode(self.info_queue, self.mem_queue)
        elif is_decode_node:
            self.backend = ContinuesBatchBackendForDecodeNode(self.info_queue, self.mem_queue)
        elif use_reward_model:
            self.backend = RewardModelBackend()
        elif is_splitfuse_mode:
            self.backend = SplitFuseBackend()
        elif return_all_prompt_logprobs:
            self.backend = ReturnPromptLogProbBackend()
        elif diverse_mode:
            self.backend = DiversehBackend()
        elif is_token_healing:
            self.backend = TokenHealingBackend()
        elif is_simple_constraint_mode:
            self.backend = SimpleConstraintBackend()
        elif is_first_token_constraint_mode:
            self.backend = FirstTokenConstraintBackend()
        elif kvargs.get("dp_size", 1) > 1:
            self.backend = DPBackend()
        else:
            self.backend = ContinuesBatchBackend()

        logger.info(f"use {self.backend.__class__.__name__}")
        self.backend.init_model(kvargs)

        return

    def prefill(self, reqs):
        try:
            return self.backend.prefill(reqs)
        except Exception as e:
            err_msg = str(e)
            logger.exception(f"Batch prefill encountered an unexpected ERROR: {err_msg}")
            raise e

    def decode(self):
        try:
            return self.backend.decode()
        except Exception as e:
            err_msg = str(e)
            logger.exception(f"Batch decode encountered an unexpected ERROR: {err_msg}")
            raise e

    def pause_reqs(self, req_ids):
        return self.backend.pause_reqs(req_ids)

    def get_max_total_token_num(self):
        return self.backend.get_max_total_token_num()


class ModelRpcClient:
    def __init__(self, model_infer_servers: List[ModelRpcServer], world_size, rpc_event, rpc_finished_event):
        # model_infer_servers 是传入的推理服务对象，但是在重构后，
        # 单卡不使用rpc 通信的时候，里面才有真实对象，当多卡使用rpc
        # 以后，model_infer_servers 传入的是 None 数组
        if world_size == 1:
            self.model_infer_server: ModelRpcServer = model_infer_servers[0]
        else:
            self.model_infer_server: ModelRpcServer = None

        self.world_size = world_size
        self.use_rpc = self.world_size != 1
        self.rpc_shm_params = RpcShmParams()
        self.rpc_shm_params.create_or_link_shm()
        self.rpc_shm_results = RpcShmResults()
        self.rpc_shm_results.create_or_link_shm()

        self.rpc_event = rpc_event
        self.rpc_finished_event = rpc_finished_event
        return

    async def init_model(self, kvargs):
        if self.use_rpc:
            self.rpc_shm_params.write_func_params("init_model", (kvargs,))
            self.rpc_event.set()

            self.rpc_finished_event.wait()
            self.rpc_finished_event.clear()
            return
        else:
            self.model_infer_server.init_model(kvargs)
            return

    async def prefill(self, reqs):
        if self.use_rpc:
            self.rpc_shm_params.write_func_params("prefill", (reqs,))
            self.rpc_event.set()

            await asyncio.to_thread(self.rpc_finished_event.wait)
            self.rpc_finished_event.clear()
            return
        else:
            self.model_infer_server.prefill(reqs)
            return

    async def decode(self):
        if self.use_rpc:
            self.rpc_shm_params.write_func_params("decode", ())
            self.rpc_event.set()

            await asyncio.to_thread(self.rpc_finished_event.wait)
            self.rpc_finished_event.clear()
            return
        else:
            self.model_infer_server.decode()
            return

    async def pause_reqs(self, req_ids):
        if self.use_rpc:
            self.rpc_shm_params.write_func_params("pause_reqs", (req_ids,))
            self.rpc_event.set()

            self.rpc_finished_event.wait()
            self.rpc_finished_event.clear()
            return
        else:
            self.model_infer_server.pause_reqs(req_ids)
            return

    async def get_max_total_token_num(self):
        if self.use_rpc:
            self.rpc_shm_params.write_func_params("get_max_total_token_num", ())
            self.rpc_event.set()

            self.rpc_finished_event.wait()
            self.rpc_finished_event.clear()
            func_name, ret = self.rpc_shm_results.read_func_result()
            assert func_name == "get_max_total_token_num"
            return ret
        else:
            return self.model_infer_server.get_max_total_token_num()


def _init_env(
    args,
    tp_rank,
    info_queue,
    mem_queue,
    router_lock,
    rpc_event: mp.Event,
    rpc_finished_event: mp.Event,
    success_event: mp.Event,
):
    import lightllm.utils.rpyc_fix_utils as _

    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)
    start_parent_check_thread()

    # 将调度锁注册到全局的共享变量中
    from lightllm.common.basemodel.infer_lock import g_router_lock

    g_router_lock.obj = router_lock

    model_rpc_server = ModelRpcServer(args, tp_rank, rpc_event, rpc_finished_event, info_queue, mem_queue)
    success_event.set()

    model_rpc_server.loop_thread.join()
    return


async def start_model_process(
    args,
    tp_rank,
    rpc_event,
    rpc_finished_event,
    world_size,
    info_queue: mp.Queue,
    mem_queue: mp.Queue,
    router_lock: mp.Queue,
):
    import lightllm.utils.rpyc_fix_utils as _

    # 单卡时不使用 rpc
    if world_size == 1:
        return ModelRpcServer(args, tp_rank, rpc_event, rpc_finished_event, info_queue, mem_queue)

    success_event = mp.Event()
    proc = mp.Process(
        target=_init_env,
        args=(args, tp_rank, info_queue, mem_queue, router_lock, rpc_event, rpc_finished_event, success_event),
    )
    proc.start()
    success_event.wait(timeout=40)
    assert proc.is_alive()

    return None
