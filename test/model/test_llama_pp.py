import os
import sys
import unittest
from model_infer_pp import test_model_inference

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# class TestLlamaInfer(unittest.TestCase):
#     def test_llama_infer(self):

if __name__ == "__main__":
    from lightllm.models.llama_pp.model import LlamaTpPpPartModel
    import torch

    torch.multiprocessing.set_start_method("spawn")

    test_model_inference(
        world_size=4,
        nnodes=2,
        pp_rank=int(os.getenv("PP_RANK", 0)),
        model_dir="/data/test_models/llama-7b",
        model_class=LlamaTpPpPartModel,
        batch_size=20,
        input_len=1024,
        output_len=1024,
        mode=[],
    )
