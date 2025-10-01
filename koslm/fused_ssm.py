# koslm/fused_ssm.py
"""
Python wrapper for the fused CUDA KOSLM SSM step.

This module calls the compiled extension `koslm_cuda` which must expose:
 - fused_ssm_forward(...)
 - fused_ssm_backward(...)

Forward baseline (matches your Python implementation):
 - gates_2H: [B, 2H] -> (z_raw, m_raw)
 - z_raw_b = z_raw + z_bias
 - m_raw_b = m_raw + m_bias
 - z = tanh(z_raw_b)
 - M = tanh(m_raw_b) * m_scale  (clamped)
 - innov = clamp(z - M * (A_prev * c_prev), -20, 20)
 - raw1 = innov @ W1.T + b1
 - gelu = GELU(raw1)
 - rawK = gelu @ W2.T + b2
 - K_t = tanh(rawK) * (k_scale*0.5)
 - A_t = clamp((1 - K_t*M) * A_base, 0, 0.95)
 - B_t = clamp(K_t, -0.95, 0.95)
 - c_t = A_t * c_prev + B_t * z
 - scale_down = clamp(max_norm / c_norm, max=1.0)
 - c_t *= scale_down
 - h_t = M * c_t

This wrapper packs inputs/weights and calls the CUDA extension. The backward pulls gradients
back from the extension and returns gradients in the same layout expected by PyTorch.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

# The compiled extension must be named koslm_cuda in setup.py
import sys, os
sys.path.append(os.path.dirname(__file__))  # 加入当前目录到 sys.path
import koslm_cuda  # 直接导入 .pyd


class FusedKOSLMSSMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                gates_2H: torch.Tensor,
                c_prev: torch.Tensor,
                A_prev: torch.Tensor,
                A_base: torch.Tensor,
                W1: torch.Tensor,
                b1: torch.Tensor,
                W2: torch.Tensor,
                b2: torch.Tensor,
                z_bias: torch.Tensor,
                m_bias: torch.Tensor,
                k_scale: float,
                m_scale: float,
                max_norm: float):
        """
        Calls C++/CUDA fused forward.

        Returns:
          out_h, out_c, out_A
        """
        # Ensure contiguous (C++ expects contiguous pointers)
        gates_2H = gates_2H.contiguous()
        c_prev = c_prev.contiguous()
        A_prev = A_prev.contiguous()
        A_base = A_base.contiguous()
        W1 = W1.contiguous()
        b1 = b1.contiguous()
        W2 = W2.contiguous()
        b2 = b2.contiguous()
        z_bias = z_bias.contiguous()
        m_bias = m_bias.contiguous()

        outputs = koslm_cuda.fused_ssm_forward(
            gates_2H, c_prev, A_prev, A_base,
            W1, b1, W2, b2,
            z_bias, m_bias,
            float(k_scale), float(m_scale), float(max_norm)
        )
        # outputs expected:
        # [0] out_h
        # [1] out_c
        # [2] out_A
        # [3] batch_sumsq
        # [4] innov
        # [5] raw1
        # [6] gelu
        # [7] rawK
        # [8] M_out
        # [9] c_no_scale
        # [10] z_raw_b
        # [11] m_raw_b

        out_h = outputs[0]
        out_c = outputs[1]
        out_A = outputs[2]

        # Save everything needed for backward.
        # We'll pass to fused_ssm_backward the exact items the C++ expects.
        ctx.save_for_backward(
            gates_2H, c_prev, A_prev, A_base,
            W1, b1, W2, b2,
            out_A, out_c, outputs[3],  # batch_sumsq
            outputs[4],  # innov
            outputs[5],  # raw1
            outputs[6],  # gelu
            outputs[7],  # rawK
            outputs[8],  # M_out
            outputs[9],  # c_no_scale
            outputs[10],  # z_raw_b
            outputs[11],  # m_raw_b
        )
        ctx.k_scale = float(k_scale)
        ctx.m_scale = float(m_scale)
        ctx.max_norm = float(max_norm)

        return out_h, out_c, out_A

    @staticmethod
    def backward(ctx,
                 grad_out_h: torch.Tensor,
                 grad_out_c: torch.Tensor,
                 grad_out_A: Optional[torch.Tensor] = None):
        # Unpack saved tensors. Order matches save_for_backward above.
        (gates_2H, c_prev, A_prev, A_base,
         W1, b1, W2, b2,
         out_A, out_c, batch_sumsq,
         innov, raw1, gelu, rawK,
         M_out, c_no_scale, z_raw_b, m_raw_b) = ctx.saved_tensors

        # Ensure contiguous
        grad_out_h = grad_out_h.contiguous()
        grad_out_c = grad_out_c.contiguous()

        # Call C++ backward. The C++ expects this exact signature & returns gradients:
        # returns a vector of tensors:
        # grad_z_raw, grad_m_raw, grad_c_prev, grad_A_prev, grad_A_base,
        # grad_W1, grad_b1, grad_W2, grad_b2
        grads = koslm_cuda.fused_ssm_backward(
            grad_out_h, grad_out_c,
            gates_2H,  # C++ uses this for shapes if needed (not strictly necessary but present)
            c_prev, A_prev, A_base,
            W1, b1, W2, b2,
            out_A, out_c, batch_sumsq,
            innov, raw1, gelu, rawK,
            M_out, c_no_scale,
            z_raw_b, m_raw_b,
            ctx.k_scale, ctx.m_scale, ctx.max_norm
        )

        # Unpack grads returned by the C++ function
        # grads order: grad_z_raw, grad_m_raw, grad_c_prev, grad_A_prev, grad_A_base, grad_W1, grad_b1, grad_W2, grad_b2
        grad_z_raw = grads[0]
        grad_m_raw = grads[1]
        grad_c_prev = grads[2]
        grad_A_prev = grads[3]
        grad_A_base = grads[4]
        grad_W1 = grads[5]
        grad_b1 = grads[6]
        grad_W2 = grads[7]
        grad_b2 = grads[8]

        # Convert grad_z_raw / grad_m_raw into grads for biases (z_bias, m_bias)
        # Since z_raw_b = z_raw + z_bias, gradient w.r.t z_bias is sum over batch of grad_z_raw
        grad_z_bias = grad_z_raw.sum(dim=0)
        grad_m_bias = grad_m_raw.sum(dim=0)

        # grad for gates_2H is concat([grad_z_raw, grad_m_raw])
        grad_gates = torch.cat([grad_z_raw, grad_m_raw], dim=-1)

        # Return gradients for all forward inputs in the same order as forward(...)
        # forward signature:
        # (gates_2H, c_prev, A_prev, A_base, W1,b1,W2,b2, z_bias, m_bias, k_scale, m_scale, max_norm)
        # For non-tensor scalar args (k_scale,m_scale,max_norm) return None.
        return (grad_gates,  # gates_2H
                grad_c_prev,  # c_prev
                grad_A_prev,  # A_prev
                grad_A_base,  # A_base
                grad_W1,  # W1
                grad_b1,  # b1
                grad_W2,  # W2
                grad_b2,  # b2
                grad_z_bias,  # z_bias
                grad_m_bias,  # m_bias
                None, None, None)  # k_scale, m_scale, max_norm (non-tensors)


class FusedKOSLMSSMModule(nn.Module):
    """
    Convenient nn.Module wrapper that creates the k_proj parameters if not provided.
    k_proj: Linear(H, 3H) -> GELU -> Linear(3H, H)
    Keeps z_bias and m_bias as parameters.
    """

    def __init__(self, H: int, *, init_k_proj: Optional[nn.Module] = None,
                 k_scale: float = 0.9, m_scale: float = 1.5, max_norm: float = 100.0):
        super().__init__()
        self.H = H
        self.k_scale = float(k_scale)
        self.m_scale = float(m_scale)
        self.max_norm = float(max_norm)

        if init_k_proj is not None:
            # user-provided sequential or module with matching params
            # Expect init_k_proj to be nn.Sequential(Linear(H,3H), GELU(), Linear(3H,H))
            assert hasattr(init_k_proj, 'to'), "init_k_proj should be an nn.Module"
            # We'll extract weights if it's the expected shape
            self.k_proj = init_k_proj
            # Ensure there are two linear layers
            # The code below will attempt to find W1,b1 and W2,b2 on forward
        else:
            # create default k_proj layers (same as original python)
            self.k_proj = nn.Sequential(
                nn.Linear(H, H * 3),
                nn.GELU(),
                nn.Linear(H * 3, H),
            )

        # biases used in z and m projection (like your python code)
        self.z_bias = nn.Parameter(torch.zeros(H))
        self.m_bias = nn.Parameter(torch.zeros(H))

    def forward(self, gates_2H: torch.Tensor, c_prev: torch.Tensor, A_prev: torch.Tensor, A_base: torch.Tensor):
        """
        gates_2H: [B, 2H]
        c_prev: [B, H]
        A_prev: [B, H]
        A_base: [H]
        """
        # extract W1,b1,W2,b2 from self.k_proj
        # support two cases:
        # - self.k_proj is nn.Sequential(Linear, GELU, Linear)
        # - or user passed a custom module with same param names
        # We'll attempt to extract in the common sequential layout:
        seq = self.k_proj
        if isinstance(seq, nn.Sequential) and len(seq) >= 3 and isinstance(seq[0], nn.Linear) and isinstance(seq[2],
                                                                                                             nn.Linear):
            W1 = seq[0].weight
            b1 = seq[0].bias if seq[0].bias is not None else torch.zeros(seq[0].out_features, device=W1.device,
                                                                         dtype=W1.dtype)
            W2 = seq[2].weight
            b2 = seq[2].bias if seq[2].bias is not None else torch.zeros(seq[2].out_features, device=W2.device,
                                                                         dtype=W2.dtype)
        else:
            # fallback: try to find linear modules by attribute names
            # iterate parameters to find matching shapes (best-effort)
            all_linears = [m for m in seq.modules() if isinstance(m, nn.Linear)]
            if len(all_linears) >= 2:
                W1 = all_linears[0].weight
                b1 = all_linears[0].bias if all_linears[0].bias is not None else torch.zeros(
                    all_linears[0].out_features, device=W1.device, dtype=W1.dtype)
                W2 = all_linears[1].weight
                b2 = all_linears[1].bias if all_linears[1].bias is not None else torch.zeros(
                    all_linears[1].out_features, device=W2.device, dtype=W2.dtype)
            else:
                raise RuntimeError("k_proj must contain two nn.Linear layers (Linear(H,3H) and Linear(3H,H))")

        # call fused autograd function
        out_h, out_c, out_A = FusedKOSLMSSMFunction.apply(
            gates_2H, c_prev, A_prev, A_base,
            W1, b1, W2, b2,
            self.z_bias, self.m_bias,
            self.k_scale, self.m_scale, self.max_norm
        )
        return out_h, out_c, out_A
