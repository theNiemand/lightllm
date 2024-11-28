import torch
from .quantize_method import QuantizationMethod
from .registry import QUANTMETHODS
from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager
import torch.nn.functional as F

try:
    HAS_VLLM = True
    import vllm
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.w8a8_utils import apply_int8_linear
except:
    HAS_VLLM = False


class vLLMBaseQuantizationMethod(QuantizationMethod):
    def __init__(self):
        super().__init__()
        assert HAS_VLLM, "vllm is not installed, you can't use quant api of it"

    def quantize(self, weight: torch.Tensor):
        """ """
        pass

    def apply(self, input_tensor, weights, bias=None, out=None, workspace=None):
        """ """
        pass


@QUANTMETHODS.register(["vllm-w8a8"])
class vLLMw8a8QuantizationMethod(vLLMBaseQuantizationMethod):
    def __init__(self):
        super().__init__()

    def quantize(self, weight: torch.Tensor):
        if hasattr(weight, "scale"):
            return weight.data.transpose(0, 1).cuda(), weight.scale.cuda()
        weight = weight.float()
        scale = weight.abs().max(dim=-1)[0] / 127
        weight = weight.transpose(0, 1) / scale.reshape(1, -1)
        weight = torch.round(weight.clamp(min=-128, max=127)).to(dtype=torch.int8)
        return weight.cuda(), scale.cuda()

    def apply(self, input_tensor, weights, bias=None, out=None, workspace=None):
        x_q, x_scale, x_zp = ops.scaled_int8_quant(input_tensor, scale=None, azp=None, symmetric=True)
        m = input_tensor.shape[0]
        n = weights[0].shape[1]
        if out is None:
            out = g_cache_manager.alloc_tensor(
                (m, n), input_tensor.dtype, device=input_tensor.device, is_graph_out=False
            )
        torch.ops._C.cutlass_scaled_mm(out, x_q, weights[0], x_scale, weights[1], bias)
        return out


@QUANTMETHODS.register(["vllm-fp8w8a8"])
class vLLMFP8w8a8QuantizationMethod(vLLMBaseQuantizationMethod):
    def __init__(self):
        super().__init__()
        self.is_moe = False

    def quantize(self, weight: torch.Tensor):
        if self.is_moe:
            return self.quantize_moe(weight)
        qweight, weight_scale = ops.scaled_fp8_quant(weight.cuda(), scale=None, use_per_token_if_dynamic=True)
        return qweight.transpose(0, 1), weight_scale

    def quantize_moe(self, weight):
        num_experts = weight.shape[0]
        qweights = []
        weight_scales = []
        qweights = torch.empty_like(weight, dtype=torch.float8_e4m3fn).cuda()
        for i in range(num_experts):
            qweight, weight_scale = ops.scaled_fp8_quant(weight[i].cuda(), scale=None, use_per_token_if_dynamic=False)
            qweights[i] = qweight
            weight_scales.append(weight_scale)
        weight_scale = torch.cat(weight_scales, dim=0).reshape(-1)
        return qweights, weight_scale

    def apply(self, input_tensor, weights, bias=None, out=None, workspace=None):
        x_q, x_scale = ops.scaled_fp8_quant(input_tensor, scale=None, scale_ub=None, use_per_token_if_dynamic=True)
        m = input_tensor.shape[0]
        n = weights[0].shape[1]
        if out is None:
            out = g_cache_manager.alloc_tensor(
                (m, n), input_tensor.dtype, device=input_tensor.device, is_graph_out=False
            )
        torch.ops._C.cutlass_scaled_mm(out, x_q, weights[0], x_scale, weights[1], bias)
        return out
