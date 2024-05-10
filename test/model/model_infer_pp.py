import numpy as np
from multiprocessing import Queue
import multiprocessing


def test_model_inference(world_size, nnodes, pp_rank, model_dir, model_class, batch_size, input_len, output_len, mode):
    ans_queue = Queue()
    workers = []
    for rank_id in range(world_size):
        model_kvargs = {
            "tp_rank": rank_id,
            "global_rank": rank_id + pp_rank * world_size,
            "nnodes": nnodes,
            "pp_rank": pp_rank,
            "world_size": world_size,
            
            "weight_dir": model_dir,
            "max_total_token_num": batch_size * (input_len + output_len),
            "load_way": "HF",
            "mode": mode,
            "max_req_num": batch_size,
            "max_seq_length": (input_len + output_len),
        }

        proc = multiprocessing.Process(
            target=tppart_model_infer, args=(model_class, model_kvargs, batch_size, input_len, output_len, ans_queue)
        )
        proc.start()
        workers.append(proc)

    for proc in workers:
        proc.join()

    assert not ans_queue.empty()
    while not ans_queue.empty():
        assert ans_queue.get()
    return


def tppart_model_infer(model_class, model_kvargs, batch_size, input_len, output_len, ans_queue):
    import torch
    import torch.distributed as dist

    tp_rank = model_kvargs["tp_rank"]
    global_rank = model_kvargs["global_rank"]
    world_size = model_kvargs["world_size"]
    
    nnodes = model_kvargs["nnodes"]

    print('Start init', global_rank, world_size * nnodes)
    dist.init_process_group("nccl", init_method="tcp://10.142.5.49:23456", rank=global_rank, world_size=world_size * nnodes)
    torch.cuda.set_device(tp_rank)

    tp_lists = [list(range(i * world_size, (i + 1) * world_size)) for i in range(nnodes)]

    if tp_rank == 0:
        print("tp_lists:", tp_lists)
    tp_groups = tuple([dist.new_group(tp_list) for tp_list in tp_lists])

    pp_lists = [[tp_list[0] - world_size] + tp_list for tp_list in tp_lists[1:]] + [[(nnodes - 1) * world_size] + tp_lists[0]]
    if tp_rank == 0:
        print("pp_lists:", pp_lists)
    pp_groups = tuple([dist.new_group(pp_list) for pp_list in pp_lists])

    model_kvargs["tp_groups"] = tp_groups
    model_kvargs["pp_groups"] = pp_groups

    print("Setdevice", tp_rank)

    dist.barrier()
    print("Init finish", "global_rank:", global_rank)
    torch.cuda.empty_cache()

    model_part = model_class(model_kvargs)
    # warm up
    test_data = np.vstack([np.arange(5, input_len + 5) for _ in range(batch_size)])
    test_data = test_data.reshape(-1)
    test_data = torch.from_numpy(test_data).cuda()

    b_req_idx = model_part.req_manager.alloc(batch_size).int()
    b_start_loc = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    b_ready_cache_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    for i in range(batch_size):
        b_start_loc[i] = i * input_len
        b_seq_len[i] = input_len

    total_token_num = input_len * batch_size

    logics = model_part.forward(
        batch_size,
        total_token_num,
        input_len,
        test_data,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        b_ready_cache_len=b_ready_cache_len,
        is_prefill=True,
    )

    torch.save(logics, 'logics_pp.pt')

    prob_out = torch.softmax(logics, dim=-1)
    predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
    predict_ids = predict_ids.detach().cpu().numpy()

    for i in range(output_len):
        b_start_loc = b_start_loc + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
        total_token_num += batch_size
        b_seq_len += 1
        logics = model_part.forward(
            batch_size,
            total_token_num,
            input_len + i + 1,
            torch.from_numpy(predict_ids).cuda().reshape(-1),
            b_req_idx,
            b_start_loc,
            b_seq_len,
            is_prefill=False,
        )
        prob_out = torch.softmax(logics, dim=-1)
        predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
        predict_ids = predict_ids.detach().cpu().numpy()

    model_part.mem_manager.free_all()
    model_part.req_manager.free_all()

    if tp_rank == 0:
        print("can use mem size:", model_part.mem_manager.can_use_mem_size)
        print("can use req size:", model_part.req_manager.can_use_req_size)

    b_req_idx = None
    b_start_loc = None
    b_seq_len = None

    dist.barrier()
    import time

    torch.cuda.synchronize()
    start_time = time.time()

    prefill_start_time = time.time()

    b_req_idx = model_part.req_manager.alloc(batch_size).int()
    b_start_loc = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    for i in range(batch_size):
        b_start_loc[i] = i * input_len
        b_seq_len[i] = input_len

    total_token_num = batch_size * input_len
    logics = model_part.forward(
        batch_size,
        total_token_num,
        input_len,
        test_data,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        b_ready_cache_len=b_ready_cache_len,
        is_prefill=True,
    )
    prob_out = torch.softmax(logics, dim=-1)
    predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
    predict_ids = predict_ids.detach().cpu().numpy()

    torch.cuda.synchronize()
    if tp_rank == 0:
        print("prefill time cost:", (time.time() - prefill_start_time) * 1000)

    for i in range(output_len):
        torch.cuda.synchronize()
        step_start = time.time()
        b_start_loc = b_start_loc + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
        total_token_num += batch_size
        b_seq_len += 1

        logics = model_part.forward(
            batch_size,
            total_token_num,
            input_len + i + 1,
            torch.from_numpy(predict_ids).cuda().reshape(-1),
            b_req_idx,
            b_start_loc,
            b_seq_len,
            is_prefill=False,
        )
        prob_out = torch.softmax(logics, dim=-1)
        predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
        predict_ids = predict_ids.detach().cpu().numpy()
        torch.cuda.synchronize()
        if i % 100 == 0 or i == output_len - 1:
            if tp_rank == 0:
                print(i, "step cost time:", (time.time() - step_start) * 1000)

    torch.cuda.synchronize()
    end_time = time.time()

    if tp_rank == 0:
        print("time total cost(ms):", (end_time - start_time) * 1000)
    ans_queue.put(True)

    return
