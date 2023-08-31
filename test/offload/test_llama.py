import os
import sys
import unittest
from model_infer import test_model_alignment
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class TestLlamaInfer(unittest.TestCase):

    def test_llama_infer(self):
        from lightllm.models.llama.model import LlamaTpPartModel, LlamaTpPartOffloadModel
        test_model_alignment(world_size=1, 
                             offset=0,
                             model_dir="/hosthome/data/models/llama-7b-hf", 
                             model_class=LlamaTpPartModel, 
                             offload_class=LlamaTpPartOffloadModel,
                             batch_size=5, 
                             input_len=10, 
                             output_len=20)
        return

if __name__ == '__main__':
    unittest.main()