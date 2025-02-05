import os
import sys
import time
import uuid
import subprocess
import signal
from lightllm.utils.net_utils import alloc_can_use_network_port, PortLocker
from lightllm.utils.start_utils import process_manager
from .metrics.manager import start_metric_manager
from .embed_cache.manager import start_cache_manager
from .visualserver.manager import start_visual_process
from lightllm.utils.log_utils import init_logger
from lightllm.utils.envs_utils import set_env_start_args, set_unique_server_name
from .detokenization.manager import start_detokenization_process
from .router.manager import start_router_process
from lightllm.utils.process_check import is_process_active

logger = init_logger(__name__)


def setup_signal_handlers(http_server_process, process_manager):
    def signal_handler(sig, frame):
        if sig == signal.SIGINT:
            logger.info("Received SIGINT (Ctrl+C), forcing immediate exit...")
            if http_server_process and http_server_process.poll() is None:
                http_server_process.kill()

            process_manager.terminate_all_processes()
            logger.info("All processes have been forcefully terminated.")
            sys.exit(0)
        elif sig == signal.SIGTERM:
            logger.info("Received SIGTERM, shutting down gracefully...")
            if http_server_process and http_server_process.poll() is None:
                http_server_process.send_signal(signal.SIGTERM)

                start_time = time.time()
                while (time.time() - start_time) < 60:
                    if not is_process_active(http_server_process.pid):
                        logger.info("httpserver exit")
                        break
                    time.sleep(1)

                if time.time() - start_time < 60:
                    logger.info("HTTP server has exited gracefully")
                else:
                    logger.warning("HTTP server did not exit in time, killing it...")
                    http_server_process.kill()

            process_manager.terminate_all_processes()
            logger.info("All processes have been terminated gracefully.")
            sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    logger.info(f"start process pid {os.getpid()}")
    logger.info(f"http server pid {http_server_process.pid}")
    return


def set_env(args):
    import os

    if args.static_quant:
        os.environ["STATIC_QUANT"] = "1"
    set_unique_server_name(args)
    set_env_start_args(args)
    return


def normal_or_p_d_start(args):

    if args.run_mode not in ["normal", "prefill", "decode"]:
        return

    assert args.zmq_mode in ["tcp://", "ipc:///tmp/"]

    # 确保单机上多实列不冲突
    if args.zmq_mode == "ipc:///tmp/":
        zmq_mode = f"{args.zmq_mode}_{str(args.nccl_port)}_"
        args.zmq_mode = None  # args 的参数不能直接设置，只能先设置None，再设置才能成功
        args.zmq_mode = zmq_mode
        logger.info(f"zmq mode head: {args.zmq_mode}")

    logger.info(f"use tgi api: {args.use_tgi_api}")

    assert (
        args.mem_fraction > 0 and args.mem_fraction < 1
    ), f"Invalid mem_fraction {args.mem_fraction}, The expected value is between 0 and 1."

    if args.static_quant:
        assert args.quant_type == "vllm-w8a8", "Only static parameter loading for vllm-w8a8 is supported."

    # splitfuse_mode 和 cuda_graph 不能同时开启
    if args.splitfuse_mode:
        assert args.disable_cudagraph

    # 这些模式不能同时设置。
    assert [
        args.splitfuse_mode,
        args.diverse_mode,
        args.token_healing_mode,
        args.use_reward_model,
        args.return_all_prompt_logprobs,
        args.first_token_constraint_mode,
    ].count(True) <= 1
    # 部分模式目前还无法与dynamic_prompt_cache一起跑，to do。
    if args.use_dynamic_prompt_cache:
        assert args.token_healing_mode is False

    # 部分模式还不能支持与高级动态调度算法协同，to do.
    if args.diverse_mode:
        assert args.router_token_ratio == 0.0

    # 检查GPU数量是否足够
    total_required_gpus = args.visual_dp * args.visual_tp
    if len(args.visual_gpu_ids) < total_required_gpus:
        raise ValueError(
            f"Not enough GPUs specified. You need at least {total_required_gpus}, but got {len(args.visual_gpu_ids)}."
        )
    else:
        args.visual_gpu_ids = args.visual_gpu_ids[:total_required_gpus]

    # 检查visual_nccl_port数量是否足够
    if len(args.visual_nccl_ports) < args.visual_dp:
        raise ValueError(
            f"Not enough visual_nccl_ports specified. You need at least {args.visual_dp}, "
            f"but got ({len(args.visual_nccl_ports)})."
        )
    else:
        args.visual_nccl_ports = args.visual_nccl_ports[: args.visual_dp]

    if not args.splitfuse_mode:
        # 普通模式下
        if args.batch_max_tokens is None:
            args.batch_max_tokens = args.max_req_total_len
        else:
            assert args.batch_max_tokens >= args.max_req_total_len, "batch_max_tokens must >= max_req_total_len"
    else:
        # splitfuse 模式下
        # assert args.batch_max_tokens is not None, "need to set by yourself"
        if args.batch_max_tokens is None:
            args.batch_max_tokens = min(args.max_req_total_len, 16 * args.splitfuse_block_size)

        assert (
            args.batch_max_tokens > args.splitfuse_block_size
        ), "splitfuse_mode, batch_max_tokens must >= splitfuse_block_size"

    # help to manage data stored on Ceph
    if "s3://" in args.model_dir:
        from lightllm.utils.petrel_helper import s3_model_prepare

        s3_model_prepare(args.model_dir)

    # 如果args.eos_id 是 None, 从 config.json 中读取 eos_token_id 相关的信息，赋值给 args
    if args.eos_id is None:
        from lightllm.utils.config_utils import get_eos_token_ids

        args.eos_id = get_eos_token_ids(args.model_dir)

    if args.data_type is None:
        from lightllm.utils.config_utils import get_dtype

        args.data_type = get_dtype(args.model_dir)
        assert args.data_type in ["fp16", "float16", "bf16", "bfloat16", "fp32", "float32"]

    already_uesd_ports = args.visual_nccl_ports + [args.nccl_port, args.port]
    if args.run_mode == "decode":
        already_uesd_ports = args.visual_nccl_ports + [args.nccl_port, args.port, args.pd_decode_rpyc_port]

    # 提前锁定端口，防止在单个机器上启动多个实列的时候，要到模型启动的时候才能
    # 捕获到端口设置冲突的问题
    ports_locker = PortLocker(already_uesd_ports)
    ports_locker.lock_port()

    can_use_ports = alloc_can_use_network_port(
        num=6 + args.tp + args.tp + args.visual_dp * args.visual_tp, used_nccl_ports=already_uesd_ports
    )
    logger.info(f"alloced ports: {can_use_ports}")
    router_port, detokenization_port, detokenization_pub_port, visual_port, cache_port, metric_port = can_use_ports[0:6]
    model_rpc_ports = can_use_ports[6 : 6 + args.tp]
    can_use_ports = can_use_ports[6 + args.tp :]

    visual_model_tp_ports = []
    for _ in range(args.visual_dp):
        tp_ports_for_dp = can_use_ports[0 : args.visual_tp]
        can_use_ports = can_use_ports[args.visual_tp :]
        visual_model_tp_ports.append(tp_ports_for_dp)

    # 将申请好的端口放入args参数中
    args.router_port = router_port
    args.detokenization_port = detokenization_port
    args.detokenization_pub_port = detokenization_pub_port
    args.visual_port = visual_port
    args.cache_port = cache_port
    args.metric_port = metric_port

    # 申请在 p d 分离模式下，会用的端口
    args.pd_tp_infer_rpyc_ports = can_use_ports[0 : args.tp]
    # p d 分离模式下用于标识节点的id
    args.pd_node_id = uuid.uuid4().int
    # p 节点用来建立torch kv 传输分布组的可用端口范围
    args.pd_p_allowed_port_min = 20000
    args.pd_p_allowed_port_max = 30000

    # p d 分离模式下，decode节点的调度间隙是0
    if args.run_mode == "decode":
        args.router_max_wait_tokens = 0

    set_env(args)
    logger.info(f"all start args:{args}")

    ports_locker.release_port()

    if args.enable_multimodal:
        process_manager.start_submodule_processes(
            start_funcs=[
                start_cache_manager,
            ],
            start_args=[(cache_port, args)],
        )
        process_manager.start_submodule_processes(
            start_funcs=[
                start_visual_process,
            ],
            start_args=[
                (args, router_port, visual_port, cache_port, visual_model_tp_ports),
            ],
        )

    process_manager.start_submodule_processes(
        start_funcs=[
            start_metric_manager,
        ],
        start_args=[(metric_port, args)],
    )

    process_manager.start_submodule_processes(
        start_funcs=[start_router_process, start_detokenization_process],
        start_args=[
            (args, router_port, detokenization_port, model_rpc_ports, metric_port),
            (args, detokenization_port, detokenization_pub_port),
        ],
    )

    # 启动 gunicorn
    command = [
        "gunicorn",
        "--workers",
        f"{args.httpserver_workers}",
        "--worker-class",
        "uvicorn.workers.UvicornWorker",
        "--bind",
        f"{args.host}:{args.port}",
        "--log-level",
        "info",
        "--access-logfile",
        "-",
        "--error-logfile",
        "-",
        "lightllm.server.api_http:app",
    ]

    # 启动子进程
    http_server_process = subprocess.Popen(command)

    if "s3://" in args.model_dir:
        from lightllm.utils.petrel_helper import s3_model_clear

        s3_model_clear(args.model_dir)

    if args.health_monitor:
        from lightllm.server.health_monitor.manager import start_health_check_process

        process_manager.start_submodule_processes(start_funcs=[start_health_check_process], start_args=[(args,)])
    setup_signal_handlers(http_server_process, process_manager)
    http_server_process.wait()
    return


def pd_master_start(args):
    if args.run_mode != "pd_master":
        return

    logger.info(f"use tgi api: {args.use_tgi_api}")
    logger.info(f"all start args:{args}")

    can_use_ports = alloc_can_use_network_port(num=1, used_nccl_ports=[args.nccl_port, args.port])
    metric_port = can_use_ports[0]

    args.metric_port = metric_port

    set_env(args)

    process_manager.start_submodule_processes(
        start_funcs=[
            start_metric_manager,
        ],
        start_args=[(metric_port, args)],
    )

    command = [
        "gunicorn",
        "--workers",
        "1",
        "--worker-class",
        "uvicorn.workers.UvicornWorker",
        "--bind",
        f"{args.host}:{args.port}",
        "--log-level",
        "info",
        "--access-logfile",
        "-",
        "--error-logfile",
        "-",
        "--preload",
        "lightllm.server.api_http:app",
    ]

    http_server_process = subprocess.Popen(command)

    if args.health_monitor:
        from lightllm.server.health_monitor.manager import start_health_check_process

        process_manager.start_submodule_processes(start_funcs=[start_health_check_process], start_args=[(args,)])

    setup_signal_handlers(http_server_process, process_manager)
    http_server_process.wait()
