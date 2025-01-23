import os

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import json
import torch
from typing import final

from lightllm.common.basemodel.layer_weights.hf_load_utils import load_hf_weights
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.basemodel.splitfuse_infer_struct import SplitFuseInferStateInfo
from lightllm.common.mem_manager import MemoryManager
from lightllm.common.req_manager import ReqManager
from lightllm.common.infer_utils import init_req_to_token_indexes
from lightllm.common.build_utils import repair_config
from lightllm.common.basemodel.triton_kernel.copy_kv_index_to_req import copy_kv_index_to_req
from lightllm.common.basemodel.triton_kernel.splitfuse_copy_kv_index_to_req import splitfuse_copy_kv_index_to_req
from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager
from lightllm.common.basemodel.cuda_graph import CudaGraph
from lightllm.common.quantization import Quantcfg
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

torch.backends.cudnn.enabled = True


class TpPartBaseModel:
    # weight class
    pre_and_post_weight_class = None
    transformer_weight_class = None

    # infer class
    pre_layer_infer_class = None
    post_layer_infer_class = None
    transformer_layer_infer_class = None

    # infer state class
    infer_state_class = InferStateInfo
    splitfuse_infer_state_class = SplitFuseInferStateInfo

    def __init__(self, kvargs):
        self.run_mode = kvargs["run_mode"]
        self.tp_rank_ = kvargs["tp_rank"]
        self.world_size_ = kvargs["world_size"]
        self.weight_dir_ = kvargs["weight_dir"]
        self.max_total_token_num = kvargs["max_total_token_num"]
        self.batch_max_tokens = kvargs.get("batch_max_tokens", None)
        self.load_way = kvargs.get("load_way", "HF")
        self.mode = kvargs.get("mode", [])
        self.weight_dict = kvargs.get("weight_dict", None)
        self.finetune_config = kvargs.get("finetune_config", None)
        self.max_req_num = kvargs.get("max_req_num", 1000)
        self.max_seq_length = kvargs.get("max_seq_length", 1024 * 5)
        # is_token_healing 和 return_all_prompt_logics 是有排斥关系的两个模式，只能单独有一个生效
        # 主要是在prefill阶段返回多少个token的用于后续处理相关。
        self.is_token_healing = kvargs.get("is_token_healing", False)
        self.return_all_prompt_logics = kvargs.get("return_all_prompt_logics", False)
        assert not (self.is_token_healing and self.return_all_prompt_logics), "can not be true in same time"
        self.use_dynamic_prompt_cache = kvargs.get("use_dynamic_prompt_cache", False)
        self.data_type = kvargs.get("data_type", "float16")
        self.graph_max_batch_size = kvargs.get("graph_max_batch_size", 16)
        self.graph_max_len_in_batch = kvargs.get("graph_max_len_in_batch", 8192)
        self.disable_cudagraph = kvargs.get("disable_cudagraph", False)
        self.quant_type = kvargs.get("quant_type", None)
        self.quant_cfg_path = kvargs.get("quant_cfg", None)
        self.mem_fraction = kvargs.get("mem_fraction", 0.9)

        self._init_datatype()
        self._init_config()
        self._verify_must()
        self._verify_params()
        self._init_quant()

        # 更连续的显存分配可以有更好的性能
        if self.max_total_token_num is None:
            self._init_weights()
            self._init_mem_manager()
        else:
            self._init_mem_manager()
            self._init_weights()

        self._init_kv_move_buffer()
        self._check_mem_size()
        self._init_req_manager()
        self._init_infer_layer()
        self._init_some_value()
        self._init_custom()
        self._init_cudagraph()
        self._check_max_len_infer()
        torch.cuda.empty_cache()
        return

    def _init_config(self):
        with open(os.path.join(self.weight_dir_, "config.json"), "r") as json_file:
            self.config = json.load(json_file)
        # rename keys
        repair_config(self.config, same_names=["num_attention_heads", "n_head"])
        repair_config(self.config, same_names=["hidden_size", "n_embd", "n_embed"])
        repair_config(self.config, same_names=["num_hidden_layers", "n_layer"])
        if self.finetune_config:
            self.config["vocab_size"] = self.finetune_config.vocab_size

        return

    @final
    def _verify_must(self):
        assert self.config["num_attention_heads"] % self.world_size_ == 0
        return

    def _verify_params(self):
        assert self.load_way == "HF", "only support HF format weights"
        assert self.config["num_key_value_heads"] % self.world_size_ == 0
        return

    def _init_quant(self):
        self.quant_cfg = Quantcfg(self.config, self.quant_type, self.quant_cfg_path)
        logger.info(f"Initial quantization. " f"The default quantization method is {self.quant_cfg.quant_type}")

    def _init_weights(self):
        self.pre_post_weight = self.pre_and_post_weight_class(
            self.tp_rank_, self.world_size_, self.data_type, network_config=self.config, mode=self.mode
        )
        self.trans_layers_weight = [
            self.transformer_weight_class(
                i,
                self.tp_rank_,
                self.world_size_,
                self.data_type,
                network_config=self.config,
                mode=self.mode,
                quant_cfg=self.quant_cfg,
            )
            for i in range(self.config["n_layer"])
        ]
        load_hf_weights(
            self.data_type,
            weight_dir=self.weight_dir_,
            pre_post_layer=self.pre_post_weight,
            transformer_layer_list=self.trans_layers_weight,
            weight_dict=self.weight_dict,
        )
        self.pre_post_weight.verify_load()
        [weight.verify_load() for weight in self.trans_layers_weight]
        return

    def _init_mem_manager(self):
        assert self.config["num_attention_heads"] % self.world_size_ == 0
        self.mem_manager = MemoryManager(
            self.max_total_token_num,
            dtype=self.data_type,
            head_num=self.config["num_attention_heads"] // self.world_size_,
            head_dim=self.config["n_embed"] // self.config["num_attention_heads"],
            layer_num=self.config["n_layer"],
            mem_fraction=self.mem_fraction,
        )
        return

    def _init_kv_move_buffer(self):
        # p d 分离的推理模式下才需要做这一步初始化
        if self.run_mode in ["prefill", "decode"]:
            self.mem_manager.alloc_kv_move_buffer(self.mem_manager.size)

    def _check_mem_size(self):
        self.max_total_token_num = self.mem_manager.size
        assert self.max_seq_length < self.max_total_token_num
        return

    def _init_req_manager(self):
        create_max_seq_len = 0

        if self.batch_max_tokens is not None:
            create_max_seq_len = max(create_max_seq_len, self.batch_max_tokens)
        if self.max_seq_length is not None:
            create_max_seq_len = max(create_max_seq_len, self.max_seq_length)

        self.req_manager = ReqManager(self.max_req_num, create_max_seq_len, self.mem_manager)
        return

    def _init_infer_layer(self):
        self.pre_infer = self.pre_layer_infer_class(
            tp_rank=self.tp_rank_, world_size=self.world_size_, network_config=self.config, mode=self.mode
        )
        self.post_infer = self.post_layer_infer_class(
            tp_rank=self.tp_rank_, world_size=self.world_size_, network_config=self.config, mode=self.mode
        )
        self.layers_infer = [
            self.transformer_layer_infer_class(
                i, tp_rank=self.tp_rank_, world_size=self.world_size_, network_config=self.config, mode=self.mode
            )
            for i in range(self.config["n_layer"])
        ]
        return

    def _init_some_value(self):
        # Dealing with head_dim_!=n_embed // num_attention_heads scenarios, such as mistral 13B
        head_dim_ = self.config["n_embed"] // self.config["num_attention_heads"]
        self.head_dim_ = self.config.get("head_dim", head_dim_)
        self.tp_k_head_num_ = self.config["num_key_value_heads"] // self.world_size_
        self.tp_v_head_num_ = self.tp_k_head_num_
        self.layers_num = self.config["n_layer"]
        self.vocab_size = self.config["vocab_size"]
        return

    def _init_datatype(self):
        if self.data_type in ["fp16", "float16"]:
            self.data_type = torch.float16
        elif self.data_type in ["bf16", "bfloat16"]:
            self.data_type = torch.bfloat16
        elif self.data_type in ["fp32", "float32"]:
            self.data_type = torch.float32
        else:
            raise ValueError(f"Unsupport datatype {self.data_type}!")

    def _init_cudagraph(self):
        self.graph = (
            None if self.disable_cudagraph else CudaGraph(self.graph_max_batch_size, self.graph_max_len_in_batch)
        )
        if self.graph is not None:
            self.graph.warmup(self)

    def _init_custom(self):
        pass

    @torch.no_grad()
    def forward(
        self,
        batch_size,
        total_token_num,
        max_len_in_batch,
        input_ids: torch.Tensor,
        mem_indexes: torch.Tensor,
        b_req_idx: torch.Tensor,
        b_start_loc: torch.Tensor,
        b_seq_len: torch.Tensor,
        b_ready_cache_len: torch.Tensor = None,
        multimodal_params=None,
        is_prefill=True,
    ):
        assert mem_indexes.is_cuda

        if is_prefill:
            return self._prefill(
                batch_size,
                total_token_num,
                max_len_in_batch,
                input_ids,
                mem_indexes,
                b_req_idx,
                b_start_loc,
                b_seq_len,
                b_ready_cache_len,
                multimodal_params,
            )
        else:
            return self._decode(
                batch_size,
                total_token_num,
                max_len_in_batch,
                input_ids,
                mem_indexes,
                b_req_idx,
                b_start_loc,
                b_seq_len,
                multimodal_params,
            )

    def _prefill(
        self,
        batch_size,
        total_token_num,
        max_len_in_batch,
        input_ids,
        mem_indexes,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        b_ready_cache_len,
        multimodal_params,
    ):
        infer_state = self.infer_state_class()
        infer_state.is_prefill = True
        infer_state.is_token_healing = self.is_token_healing
        infer_state.return_all_prompt_logics = self.return_all_prompt_logics
        infer_state.use_dynamic_prompt_cache = self.use_dynamic_prompt_cache
        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_len_in_batch = max_len_in_batch
        assert b_req_idx.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0]
        infer_state.b_req_idx = b_req_idx
        infer_state.b_start_loc = b_start_loc
        infer_state.b_seq_len = b_seq_len
        if b_ready_cache_len is not None:
            infer_state.b_ready_cache_len = b_ready_cache_len
        else:
            infer_state.b_ready_cache_len = torch.zeros_like(b_seq_len, dtype=b_seq_len.dtype, device=b_seq_len.device)
        infer_state.multimodal_params = multimodal_params

        infer_state.mem_manager = self.mem_manager
        infer_state.req_manager = self.req_manager

        infer_state.mem_is_contiguous = False
        infer_state.mem_index = mem_indexes
        infer_state.kv_buffer = torch.empty(
            (input_ids.shape[0], self.tp_k_head_num_ + self.tp_v_head_num_, self.head_dim_),
            dtype=self.data_type,
            device="cuda",
        )

        init_req_to_token_indexes(
            self.req_manager.req_to_token_indexs,
            b_req_idx,
            b_seq_len,
            infer_state.b_ready_cache_len,
            max_len_in_batch,
            infer_state.mem_index,
        )

        infer_state.init_some_extra_state(self, input_ids)
        predict_logics = self._context_forward(input_ids, infer_state)
        return predict_logics

    def _decode(
        self,
        batch_size,
        total_token_num,
        max_len_in_batch,
        input_ids,
        mem_indexes,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        multimodal_params,
    ):
        infer_state = self.infer_state_class()
        infer_state.is_prefill = False
        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_len_in_batch = max_len_in_batch
        infer_state.use_dynamic_prompt_cache = self.use_dynamic_prompt_cache
        assert b_req_idx.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0]
        infer_state.b_req_idx = b_req_idx
        infer_state.b_start_loc = b_start_loc
        infer_state.b_seq_len = b_seq_len
        infer_state.multimodal_params = multimodal_params

        infer_state.mem_manager = self.mem_manager
        infer_state.req_manager = self.req_manager

        # 在使用 cuda graph 特性的时候，必须保证每次推理的流程一致
        # 所以不再使用分配连续的mem带来的优化，保证推理流程的一致
        infer_state.mem_is_contiguous = False
        infer_state.mem_index = mem_indexes
        infer_state.kv_buffer = torch.empty(
            (batch_size, self.tp_k_head_num_ + self.tp_v_head_num_, self.head_dim_),
            dtype=self.data_type,
            device="cuda",
        )
        copy_kv_index_to_req(self.req_manager.req_to_token_indexs, b_req_idx, b_seq_len, infer_state.mem_index)

        infer_state.init_some_extra_state(self, input_ids)
        if self.graph is not None and self.graph.can_run(batch_size, max_len_in_batch):
            if self.graph.need_capture(batch_size):
                infer_state.is_cuda_graph = True
                predict_logics = self.graph.capture_decode(self._token_forward, input_ids, infer_state)
            else:
                predict_logics = self.graph.replay(input_ids, infer_state)
        else:
            predict_logics = self._token_forward(input_ids, infer_state)
        return predict_logics

    @torch.no_grad()
    def splitfuse_forward(
        self,
        input_ids,
        mem_indexes,
        decode_req_num,
        decode_total_token_num,
        decode_b_req_idx: torch.Tensor,
        decode_b_start_loc: torch.Tensor,
        decode_b_seq_len: torch.Tensor,
        decode_max_len_in_batch,
        prefill_req_num,
        prefill_b_req_idx: torch.Tensor,
        prefill_b_split_start_loc: torch.Tensor,
        prefill_b_split_ready_cache_len: torch.Tensor,
        prefill_max_split_seq_len_in_batch,
        prefill_b_seq_len: torch.Tensor,
    ):

        infer_state = self.splitfuse_infer_state_class()
        infer_state.use_dynamic_prompt_cache = self.use_dynamic_prompt_cache
        infer_state.batch_size = decode_req_num + prefill_req_num

        infer_state.decode_req_num = decode_req_num
        infer_state.decode_total_token_num = decode_total_token_num
        infer_state.decode_b_req_idx = decode_b_req_idx
        infer_state.decode_b_start_loc = decode_b_start_loc
        infer_state.decode_b_seq_len = decode_b_seq_len
        infer_state.decode_max_len_in_batch = decode_max_len_in_batch

        infer_state.prefill_req_num = prefill_req_num
        infer_state.prefill_b_req_idx = prefill_b_req_idx
        infer_state.prefill_b_split_start_loc = prefill_b_split_start_loc
        infer_state.prefill_b_split_ready_cache_len = prefill_b_split_ready_cache_len
        infer_state.prefill_max_split_seq_len_in_batch = prefill_max_split_seq_len_in_batch
        infer_state.prefill_b_seq_len = prefill_b_seq_len
        # infer_state.event = [torch.cuda.Event() for _ in range(self.layers_num)]

        infer_state.mem_manager = self.mem_manager
        infer_state.req_manager = self.req_manager

        alloc_size = len(input_ids)
        infer_state.mem_is_contiguous = False
        infer_state.mem_index = mem_indexes
        infer_state.kv_buffer = torch.empty(
            (alloc_size, self.tp_k_head_num_ + self.tp_v_head_num_, self.head_dim_),
            dtype=self.data_type,
            device="cuda",
        )

        # decode 部分
        if decode_req_num != 0:
            copy_kv_index_to_req(
                self.req_manager.req_to_token_indexs,
                decode_b_req_idx,
                decode_b_seq_len,
                infer_state.mem_index[0:decode_req_num],
            )

        # split prefill 部分
        if prefill_req_num != 0:
            splitfuse_copy_kv_index_to_req(
                self.req_manager.req_to_token_indexs,
                prefill_b_req_idx,
                prefill_b_split_ready_cache_len,
                prefill_b_seq_len,
                infer_state.mem_index[decode_req_num:],
            )

        infer_state.init_some_extra_state(self, input_ids)
        infer_state.create_inner_decode_infer_status()
        infer_state.create_inner_prefill_infer_status()
        predict_logics = self._splitfuse_forward(input_ids, infer_state)
        return predict_logics

    @final
    def _context_forward(self, input_ids, infer_state: InferStateInfo):
        g_cache_manager.cache_env_in()
        cuda_input_ids = input_ids
        input_embs = self.pre_infer.context_forward(cuda_input_ids, infer_state, self.pre_post_weight)
        for i in range(0, self.layers_num):
            input_embs = self.layers_infer[i].context_forward(input_embs, infer_state, self.trans_layers_weight[i])
        predict_logics = self.post_infer.token_forward(input_embs, infer_state, self.pre_post_weight)
        g_cache_manager.cache_env_out()
        return predict_logics

    @final
    def _token_forward(self, input_ids, infer_state: InferStateInfo):
        g_cache_manager.cache_env_in(
            is_cuda_graph=infer_state.is_cuda_graph,
            cur_batch_size=infer_state.batch_size,
            cuda_graph_max_batch_size=self.graph_max_batch_size,
        )
        cuda_input_ids = input_ids
        input_embs = self.pre_infer.token_forward(cuda_input_ids, infer_state, self.pre_post_weight)
        for i in range(0, self.layers_num):
            input_embs = self.layers_infer[i].token_forward(input_embs, infer_state, self.trans_layers_weight[i])
        predict_logics = self.post_infer.token_forward(input_embs, infer_state, self.pre_post_weight)
        g_cache_manager.cache_env_out()
        return predict_logics

    @final
    def _splitfuse_forward(self, input_ids, infer_state: SplitFuseInferStateInfo):
        g_cache_manager.cache_env_in()
        cuda_input_ids = input_ids
        input_embs = self.pre_infer.splitfuse_forward(cuda_input_ids, infer_state, self.pre_post_weight)
        for i in range(0, self.layers_num):
            input_embs = self.layers_infer[i].splitfuse_forward(input_embs, infer_state, self.trans_layers_weight[i])
        predict_logics = self.post_infer.splitfuse_forward(input_embs, infer_state, self.pre_post_weight)
        g_cache_manager.cache_env_out()
        return predict_logics

    @final
    @torch.no_grad()
    def _check_max_len_infer(self):
        disable_check_max_len_infer = os.getenv("DISABLE_CHECK_MAX_LEN_INFER", None) is not None
        if disable_check_max_len_infer:
            logger.info("disable_check_max_len_infer is true")
            return

        # 模拟最大长度进行 prefill，观察是否出现 OOM
        try:
            logger.info("begin check max_len infer")
            dummy_input_ids = torch.ones(self.batch_max_tokens, dtype=torch.int32, device="cuda")
            b_req_idx = torch.tensor([self.req_manager.alloc()], dtype=torch.int32, device="cuda")
            mem_indexes = self.mem_manager.alloc(len(dummy_input_ids)).cuda()
            b_seq_len = torch.ones(1, dtype=torch.int32, device="cuda")
            b_seq_len[:] = self.batch_max_tokens
            b_ready_cache_len = torch.zeros(1, dtype=torch.int32, device="cuda")
            b_start_loc = torch.arange(0, 1, dtype=torch.int32, device="cuda")
            total_token_num = self.batch_max_tokens
            logics = self.forward(
                1,
                total_token_num,
                self.batch_max_tokens,
                dummy_input_ids,
                mem_indexes,
                b_req_idx,
                b_start_loc,
                b_seq_len,
                b_ready_cache_len=b_ready_cache_len,
                is_prefill=True,
                multimodal_params=[],
            )
            prob_out = torch.softmax(logics, dim=-1)
            logics = None
            torch.argmax(prob_out, dim=1, keepdim=True)
            prob_out = None
            self.req_manager.free_all()
            self.mem_manager.free_all()
            logger.info(f"check max_len {self.batch_max_tokens} infer ok")
        except (RuntimeError, torch.OutOfMemoryError) as e:
            logger.exception(str(e))
            exception_str = (
                "check max len infer fail, you can try:"
                "1.Set the --mem_fraction or --max_total_token_num startup parameter to a smaller value."
                "2.Set the --max_req_total_len to a smaller value."
                "3.Set the --batch_max_tokens startup parameter to a smaller value."
            )
            logger.error(exception_str)
            raise Exception(exception_str)
        return
