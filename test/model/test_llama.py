import os
import sys
import unittest
from model_infer import test_model_inference

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestLlamaInfer(unittest.TestCase):
    def test_llama_infer(self):

        test_model_inference(
            world_size=2,
            model_dir="/data/test_models/llama-7b",
            model_class=LlamaTpPartModel,
            batch_size=20,
            input_len=1024,
            output_len=1024,
            mode=[],
        )
        return


if __name__ == "__main__":
    import torch

    torch.multiprocessing.set_start_method("spawn")
    from lightllm.models.llama.model import LlamaTpPartModel
    test_model_inference(
        world_size=1,
        model_dir="/data/test_models/llama-7b",
        model_class=LlamaTpPartModel,
        batch_size=20,
        input_len=1024,
        output_len=1024,
        mode=[],
    )
