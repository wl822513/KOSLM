# fused_lstm/fused_lstm.py
import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch import Tensor
import fused_ssm_innov_cuda  # 第一个 .cu 编译后的模块
import fused_ssm_step_cuda  # 第二个 .cu 编译后的模块

@dataclass
class KOSLMConfig:
    d_model: int  # D
    n_layers: int
    d_inner: int  # N
    d_state: int = 2  # N in paper/comments
    d_conv: int = 2

    bias: bool = False
    conv_bias: bool = True


# KOSLM模型
class KOSLM(nn.Module):
    def __init__(self, config: KOSLMConfig):
        super().__init__()

        self.config = config
        self.layers = nn.ModuleList([KOSLMBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        # projects block input from D to 2*ED (two branches)

    def forward(self, x):  # 将输入通过所有层进行处理。
        # x : (B, L, D)
        # y : (B, L, D)

        for layer in self.layers:
            x = layer(self.norm(x))

        return x


class KOSLMBlock(nn.Module):
    def __init__(self, config, segment_len=16, use_ckpt=False):
        super().__init__()
        self.cell = KOSLMCell(config)
        self.segment_len = segment_len
        self.use_ckpt = use_ckpt
        self.H = config.d_inner

        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

    def _segment_scan(self, x_seg, h0, c0, At):
        # x_seg: [B, S, D], 逐步 scan
        B, S, _ = x_seg.shape
        h, c = h0, c0
        hs = []
        for t in range(S):
            xt = x_seg[:, t, :]
            if self.use_ckpt and self.training:
                h, c, At = checkpoint(lambda _x, _h, _c, _At: self.cell(_x, (_h, _c), _At)[:3],
                                     xt, h, c, At, use_reentrant=False)
            else:
                h, c, At = self.cell(xt, (h, c), At)
            hs.append(h.unsqueeze(1))
        return torch.cat(hs, dim=1), (h, c), At

    def forward(self, x):
        """
        x: [B, L, D]
        """
        B, L, _ = x.shape
        device = x.device
        h = torch.zeros(B, self.H, device=device)
        c = torch.zeros(B, self.H, device=device)
        At = torch.zeros(B, self.H, device=device)
        outputs = []
        for s in range(0, L, self.segment_len):
            seg = x[:, s:s+self.segment_len, :]
            y_seg, (h, c), At = self._segment_scan(seg, h, c, At)
            outputs.append(y_seg)
        output = torch.cat(outputs, dim=1)  # [B,L,H]
        output = self.out_proj(output)
        return output  # [B,L,H]


class KOSLMCell(nn.Module):
    def __init__(self, config, bias=True):
        super().__init__()
        self.config = config
        H = config.d_inner
        D_in = config.d_model

        # 上游“门”产生器：把 (x_t, h_{t-1}) 变成 2H 的门值 (z_raw, m_raw)
        self.weight_ih = nn.Linear(D_in,  2*H, bias=bias)
        self.weight_hh = nn.Linear(H,     2*H, bias=bias)
        # 更好的初始化：让“保留”更强，避免早期梯度消失
        nn.init.xavier_uniform_(self.weight_ih.weight)
        nn.init.orthogonal_(self.weight_hh.weight)
        if bias:
            nn.init.zeros_(self.weight_ih.bias)
            nn.init.zeros_(self.weight_hh.bias)

        self.ssm = KOSLMSSM(H)

    def forward(self, x, hx, At):
        h_prev, c_prev = hx  # [B,H], [B,H]
        gates = self.weight_ih(x) + self.weight_hh(h_prev)  # [B,2H]
        h_t, c_t, A_t = self.ssm(gates, c_prev, At)
        return h_t, c_t, A_t


class KOSLMSSM(nn.Module):
    """
    与理论一致且稳定的一步更新（全 PyTorch）
    - c_t: [B, H]
    - A:   基础稳定对角（或块对角），这里示例对角
    - M_t, K_t: 对角结构（向量）
    """
    def __init__(self, H: int, base_decay_min=0.01, base_decay_max=0.2):
        super().__init__()
        self.H = H

        # 连续时间稳定对角 -lambda_i < 0
        lambdas = torch.linspace(base_decay_min, base_decay_max, H)  # 正数
        self.log_lambda = nn.Parameter(lambdas.log())  # 训练时仍保持正值，通过 exp 保持正，再加负号成稳定

        # 门控与增益的投影：输入 [B,H] -> 输出 [B,H]
        self.k_proj = nn.Sequential(
            nn.Linear(H, H*3),
            nn.GELU(),
            nn.Linear(H*3, H),
        )

    def forward(ctx, z_raw, m_raw, c_prev, A_prev, A_base, k_scale, m_scale, max_norm):
        """
        z_raw: [B,H]
        m_raw: [B,H]
        c_prev: [B,H]
        A_prev: [B,H]
        A_base: [H]
        k_scale: float
        m_scale: float
        max_norm: float
        """
        # 1. compute innov on GPU
        innov = fused_ssm_innov_cuda.fused_ssm_innov_cuda(z_raw, m_raw, c_prev, A_prev, m_scale)

        # 2. compute K in Python (k_proj 可以是 Linear 层或者其他投影)
        # 注意 K_t 已经在 Python 里计算好
        K_t = torch.tanh(self.k_proj(innov)) * k_scale * 0.5  # [B,H]

        # 3. prepare workspace for step
        B, H = z_raw.shape
        out_c = torch.empty_like(c_prev)
        out_h = torch.empty_like(c_prev)
        out_A = torch.empty_like(c_prev)
        batch_sumsq = torch.zeros(B, device=z_raw.device, dtype=z_raw.dtype)

        # 4. forward SSM step on GPU
        fused_ssm_step_cuda.ssm_step_forward_cuda(
            innov, K_t, m_raw, c_prev, A_prev, A_base,
            out_c, out_h, out_A, batch_sumsq,
            B, H, m_scale, max_norm
        )

        # 5. apply scaling
        fused_ssm_step_cuda.ssm_apply_scale_cuda(
            out_h, out_c, batch_sumsq,
            B, H, max_norm
        )

        ctx.save_for_backward(innov, K_t, m_raw, c_prev, A_prev, A_base, out_c, out_A, batch_sumsq)
        ctx.m_scale = m_scale
        ctx.max_norm = max_norm

        return out_h, out_c, out_A

    @staticmethod
    def backward(ctx, grad_h, grad_c, grad_A=None):
        innov, K_t, m_raw, c_prev, A_prev, A_base, out_c, out_A, batch_sumsq = ctx.saved_tensors
        m_scale = ctx.m_scale
        max_norm = ctx.max_norm

        grad_innov = torch.zeros_like(innov)
        grad_c_prev = torch.zeros_like(c_prev)
        grad_A_prev = torch.zeros_like(A_prev)

        B, H = grad_h.shape

        # backward SSM step
        fused_ssm_step_cuda.ssm_step_backward_cuda(
            grad_h, grad_c, innov, K_t, m_raw, c_prev, A_prev, A_base,
            out_c, out_A, batch_sumsq,
            grad_innov, grad_c_prev, grad_A_prev,
            B, H, m_scale, max_norm
        )

        # 这里 grad_innov 会返回给 Python 计算 rawK 的梯度，如果 k_proj 是 nn.Linear，则可以 backward
        return grad_innov, torch.zeros_like(m_raw), grad_c_prev, grad_A_prev, torch.zeros_like(A_base), None, None, None


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output





