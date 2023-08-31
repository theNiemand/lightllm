import torch
import numpy as np
from multiprocessing import Queue
import multiprocessing

def compute_cosine_similarity(a, b):
    a = torch.tensor(a).flatten()
    b = torch.tensor(b).flatten()
    return torch.cosine_similarity(a, b, dim=0).item()

def test_model_alignment(world_size, offset, model_dir, model_class, offload_class, batch_size, input_len, output_len):
    test_model_inference(world_size, offset, model_dir, model_class, batch_size, input_len, output_len, cache_num=0)
    test_model_inference(world_size, offset, model_dir, offload_class, batch_size, input_len, output_len, cache_num=4)
    original_model_output = torch.load('test_no_cache.pth')
    offload_model_output = torch.load('test_cache.pth')
    assert len(original_model_output) == len(offload_model_output)
    cos = compute_cosine_similarity(original_model_output, offload_model_output)
    print(cos)
    assert cos > 0.99, f'cosine similarity is {cos}'

def test_model_inference(world_size, offset, model_dir, model_class, batch_size, input_len, output_len, cache_num):
    if world_size > 1:
        ans_queue = Queue()
        workers = []
        for rank_id in range(world_size):
            proc = multiprocessing.Process(target=tppart_model_infer, args=(rank_id, offset, world_size, ans_queue, model_dir, model_class, batch_size, input_len, output_len, cache_num))
            proc.start()
            workers.append(proc)

        for proc in workers:
            proc.join()
            print(f'process {proc.pid} joined')
    else:
        tppart_model_infer(0, offset, 1, None, model_dir, model_class, batch_size, input_len, output_len, cache_num)

def save_buffer(buffer_type, mem_manager, cache):
    if cache:
        for idx, tensor in enumerate(getattr(mem_manager, buffer_type)):
            torch.save(tensor, f'cache_{buffer_type}_{idx}.pth')
    else:
        for idx, tensor in enumerate(getattr(mem_manager, buffer_type)):
            torch.save(tensor, f'no_cache_{buffer_type}_{idx}.pth')


def tppart_model_infer(rank_id, offset, world_size, ans_queue, model_dir, model_class, batch_size, input_len, output_len, cache_num=0):
    import torch
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group('nccl', init_method='tcp://127.0.0.1:28765', rank=rank_id, world_size=world_size)
    torch.cuda.set_device(rank_id + offset)

    dist.barrier()
    torch.cuda.empty_cache()

    kwargs = {}
    if cache_num > 0:
        kwargs["cache_num"] = cache_num
    model_part = model_class(dist.get_rank(), 
                             dist.get_world_size(), 
                             max_total_token_num= batch_size * (input_len + output_len), 
                             weight_dir=model_dir, 
                             load_way="HF",
                             **kwargs)
    # warm up
    test_data = np.vstack([np.arange(5, input_len + 5) for _ in range(batch_size)])
    test_data = test_data.reshape(-1)
    test_data = torch.from_numpy(test_data).cuda()

    b_loc = torch.zeros(batch_size, input_len + output_len, dtype=torch.long, device="cuda")
    b_start_loc = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    for i in range(batch_size):
        b_loc[i, 0:input_len] = i * input_len + torch.arange(0, input_len, dtype=torch.int32, device="cuda")
        b_start_loc[i] = i * input_len
        b_seq_len[i] = input_len


    output_result = []
    total_token_num = input_len * batch_size
    logics = model_part.forward(batch_size, 
                                total_token_num, 
                                input_len, 
                                test_data,
                                b_loc,
                                b_start_loc,
                                b_seq_len,
                                is_prefill=True)
    output_result.extend(logics.detach().cpu().tolist())
    prob_out = torch.softmax(logics, dim=-1)
    predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
    predict_ids = predict_ids.detach().cpu().numpy()

    torch.cuda.synchronize()
    # if cache_num > 0:
    #     for idx, (key, value) in enumerate(zip(model_part.mem_manager.debug_key_buffer_cpu, model_part.mem_manager.debug_value_buffer_cpu)):
    #         assert key[:10].max() > 0, f'idx {idx} key norm is {key[:10].norm(2)}'
    #         assert value[:10].max() > 0, f'idx {idx} value norm is {value[:10].norm(2)}'
    for i in range(output_len):
        b_loc[:, input_len + i] = total_token_num + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
        b_start_loc = b_start_loc + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
        total_token_num += batch_size
        b_seq_len += 1
        logics = model_part.forward(batch_size, total_token_num, input_len + i + 1, torch.from_numpy(
            predict_ids).cuda().reshape(-1), b_loc, b_start_loc, b_seq_len, is_prefill=False)
        output_result.extend(logics.detach().cpu().tolist())
        prob_out = torch.softmax(logics, dim=-1)
        predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
        predict_ids = predict_ids.detach().cpu().numpy()
        # if cache_num > 0:
        #     torch.cuda.synchronize()
        #     for idx, (key, value) in enumerate(zip(model_part.mem_manager.debug_key_buffer_cpu, model_part.mem_manager.debug_value_buffer_cpu)):
        #         assert key[:10].max() > 0, f'idx {idx} key norm is {key[:10].norm(2)}'
        #         assert value[:10].max() > 0, f'idx {idx} value norm is {value[:10].norm(2)}'
        
    max_len_in_batch = input_len + output_len
    for i in range(batch_size):
        model_part.mem_manager.free(b_loc[i, max_len_in_batch - b_seq_len[i]:max_len_in_batch])
    
    torch.cuda.synchronize()
    print(f"rank {rank_id} finished")

    if cache_num <= 0 and rank_id == 0:
        torch.save(output_result, f'test_no_cache.pth')
        save_buffer('key_buffer', model_part.mem_manager, cache=False)
        save_buffer('value_buffer', model_part.mem_manager, cache=False)
    if cache_num > 0 and rank_id == 0:
        torch.save(output_result, f'test_cache.pth')
        save_buffer('debug_key_buffer_cpu', model_part.mem_manager, cache=True)
        save_buffer('debug_value_buffer_cpu', model_part.mem_manager, cache=True)
    return