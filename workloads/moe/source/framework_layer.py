"""Framework layer and fixed initialization used for the validated artifacts.

Source reference only. Execute the fixed workload with ../run.py.
The original generator constructed and ran this layer under torch.inference_mode().
Its compiler/backend settings are recorded in ../docs/performance.md.
"""
import numpy as np
import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig


def mix_bits(x):
    x = np.asarray(x, dtype=np.uint32).copy()
    x ^= x >> 16
    x *= np.uint32(0x7feb352d)
    x ^= x >> 15
    x *= np.uint32(0x846ca68b)
    return x ^ (x >> 16)


def initialize(tensor, seed, scale, offset=0):
    flat = tensor.view(-1)
    for start in range(0, flat.numel(), 1 << 22):
        end = min(start + (1 << 22), flat.numel())
        indices = np.arange(offset + start, offset + end, dtype=np.uint64)
        keys = indices.astype(np.uint32) ^ mix_bits((indices >> 32).astype(np.uint32))
        values = ((mix_bits(keys ^ np.uint32(seed)) % 17).astype(np.int32) - 8)
        flat[start:end].copy_(torch.from_numpy(values.astype(np.float32) * scale))


def make_layer(case):
    h, width, experts = case["hidden"], case["intermediate"], case["experts"]
    config = TransformerConfig(
        num_layers=1, hidden_size=h, num_attention_heads=max(1, h // 128),
        ffn_hidden_size=width, moe_ffn_hidden_size=width,
        num_moe_experts=experts, moe_router_topk=case["top_k"],
        moe_router_pre_softmax=False, moe_router_dtype="fp32",
        moe_router_load_balancing_type="none", moe_aux_loss_coeff=0,
        moe_token_dispatcher_type="alltoall", moe_grouped_gemm=True,
        moe_permute_fusion=True, moe_expert_capacity_factor=None,
        gated_linear_unit=True, activation_func=F.silu, add_bias_linear=False,
        bias_activation_fusion=True, gradient_accumulation_fusion=False,
        bf16=True, params_dtype=torch.bfloat16, perform_initialization=False,
        hidden_dropout=0, attention_dropout=0,
    )
    layer = build_module(get_moe_module_spec(True, experts, True), config=config).cuda().eval()
    initialize_layer(layer, case)
    return layer


def initialize_layer(layer, case):
    h, width, experts = case["hidden"], case["intermediate"], case["experts"]
    seed = case["seed"]
    initialize(layer.router.weight, seed + 1, 1 / 128)
    for e in range(experts):
        initialize(getattr(layer.experts.linear_fc1, "weight" + str(e)), seed + 2,
                   1 / 256, e * 2 * width * h)
        initialize(getattr(layer.experts.linear_fc2, "weight" + str(e)), seed + 3,
                   1 / 256, e * h * width)


def initialize_parallel():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.distributed.init_process_group(
        "nccl", init_method="tcp://127.0.0.1:29500", rank=0, world_size=1)
    parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(2026)
