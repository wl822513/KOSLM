# fused_lstm_koslm_opt_no_out.py
import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.checkpoint import checkpoint

@dataclass
class KOSLMConfig:
    d_model: int  # D
    n_layers: int
    d_inner: int  # N
    d_state: int = 2  # N in paper/comments
    d_conv: int = 2

    bias: bool = False
    conv_bias: bool = True


# ---------------- KOSLM ----------------
class KOSLM(nn.Module):
    def __init__(self, config: KOSLMConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([KOSLMBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)

    def forward(self, x):
        # x: [B, L, D]
        for layer in self.layers:
            x = layer(self.norm(x))
        return x


# ---------------- KOSLMBlock（segment-level input-proj 向量化） ----------------
class KOSLMBlock(nn.Module):
    def __init__(self, config: KOSLMConfig):
        super().__init__()

        self.config = config

        self.conv1d = nn.Conv1d(in_channels=config.d_model, out_channels=config.d_model,
                                kernel_size=config.d_conv, bias=config.conv_bias,
                                groups=config.d_model,
                                padding=config.d_conv - 1)

        self.in_proj = nn.Linear(config.d_model, config.d_inner, bias=config.bias)

        # projects r to input-dependent M
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)
        self.KOSLMCell = KOSLMCell(config)

    def forward(self, x):
        # x : (B, L, D)
        # y : (B, L, D)
        _, L, _ = x.shape
        r = self.in_proj(x)  # (B, L, H)

        # x branch
        x = x.transpose(1, 2)  # (B, D, L)
        x = self.conv1d(x)[:, :, :L]  # depthwise convolution over time, with a short filter
        x = x.transpose(1, 2)  # (B, L, D)

        x = F.silu(x)
        y = self.ssm(x)
        output = self.out_proj(y + r)

        return output

    def ssm(self, x):
        # x : (B, L, H)
        # y : (B, L, H)
        B, L, _ = x.shape

        device = x.device
        # 初始状态
        h_t = torch.randn(B, self.config.d_inner, device=device)  # (B, H)
        c_t = torch.randn(B, self.config.d_inner, 1, device=device)  # (B, H, 1)
        A_t = torch.randn(B, self.config.d_inner, self.config.d_inner, device=device)  # (B, H)

        outputs = []

        # 遍历序列长度 L:如96,132,336,720
        for t in range(L):
            x_t = x[:, t, :]  # (B, H)
            h_t, c_t, A_t = self.KOSLMCell(x_t, (h_t, c_t), A_t)
            outputs.append(h_t.unsqueeze(1))  # 收集输出 (B, 1, H)

        # 拼接所有时间步的输出
        y = torch.cat(outputs, dim=1)  # (B, L, H)
        return y


# ---------------- KOSLMCell ----------------
class KOSLMCell(nn.Module):
    def __init__(self, config: KOSLMConfig, bias=True):
        super().__init__()
        self.config = config
        self.KOSLMSSM = KOSLMSSM(config.d_inner)

        # S4D real initialization
        A = torch.linspace(1, config.d_state + 1, config.d_inner, dtype=torch.float32).repeat(config.d_inner, 1)  # 均匀递增到 H
        # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.A_log = nn.Parameter(torch.log(A))

        # --- RNN-style params ---
        self.weight_ih = nn.Parameter(torch.Tensor(2*config.d_inner, config.d_model))
        self.weight_hh = nn.Parameter(torch.Tensor(2*config.d_inner, config.d_inner))
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(2*config.d_inner))
            self.bias_hh = nn.Parameter(torch.Tensor(2*config.d_inner))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        # --- Kalman gain projection W, b ---
        self.W_k = nn.Parameter(torch.Tensor(config.d_inner, config.d_inner))  # [H, H]
        if bias:
            self.b_k = nn.Parameter(torch.Tensor(config.d_inner))              # [H]
        else:
            self.register_parameter('b_k', None)

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / (self.config.d_inner ** 0.5)
        # only reset trainable weights/bias, not A_log
        nn.init.uniform_(self.weight_ih, -std, std)
        nn.init.uniform_(self.weight_hh, -std, std)
        # biases
        if self.bias_ih is not None:
            nn.init.zeros_(self.bias_ih)
        if self.bias_hh is not None:
            nn.init.zeros_(self.bias_hh)

        # Kalman gain W, b
        nn.init.xavier_uniform_(self.W_k)
        if self.b_k is not None:
            nn.init.zeros_(self.b_k)

    def forward(self, x: Tensor, hx: tuple, At_prev: Tensor):
        """
        x: [B, input_size]
        hx: (h, c) each [B, hidden_size]
        returns: (h_new, c_new)
        """
        h, c = hx
        # compute gates separately using efficient matmuls (PyTorch/ATen uses optimized GEMM)
        # Note: we follow convention gates = W_ih x + b_ih + W_hh h + b_hh but arranged via matmul
        # W_ih: [2H, in], x: [B, in] => x @ W_ih.T -> [B, 2H]
        igates = F.linear(x, self.weight_ih, self.bias_ih)    # [B, 2H]
        hgates = F.linear(h, self.weight_hh, self.bias_hh)    # [B, 2H]
        gates = igates + hgates

        A = -torch.exp(self.A_log.float())  # (H, H)

        # call fused kernel (or cpu fallback)
        # h_new, c_new, A_t = koslm_cuda.forward(gates, c, A, At_prev, self.W_k, self.b_k)
        h_new, c_new, A_t = self.KOSLMSSM(gates, c, A, At_prev, self.W_k, self.b_k)
        return h_new, c_new, A_t


# ---------------- KOSLMSSM (autograd-safe, fused-friendly) ----------------
class KOSLMSSM(nn.Module):
    """
    Kalman-Optimal Selective Long-term Memory (KOSLM) 单步 Cell (PyTorch 实现)
    与 koslm_cuda.cu 的 forward 完全一致，但用纯 Python / PyTorch 实现。
    """

    def __init__(self, H: int):
        """
        Args:
            H (int): 隐状态维度
        """
        super().__init__()
        self.H = H
        self.k_proj = nn.Sequential(
            nn.Linear(H, H*3),  # 第一层，全连接
            nn.ReLU(),  # 激活函数
            nn.Linear(H*3, H),  # 第二层，全连接
        )

    def forward(self, gates, c_prev, A, At_prev, W_k, b_k):
        """
        Args:
            gates:   [B, 2H] -> 包含 z, m
            c_prev:  [B, H, 1]
            At_prev: [B, H, H]

        Returns:
            h_new: [B, H]
            c_new: [B, H, H]
            A_t:   [B, H, H]
        """
        B, twoH = gates.shape
        assert twoH == 2 * self.H, f"gates.shape[1]={twoH} must equal 2H"

        # -------- 分解 gates --------
        gates_reshaped = gates.view(B, 2, self.H).unsqueeze(-1)
        z = torch.tanh(gates_reshaped[:, 0, :])   # [B, H, 1]
        M = torch.sigmoid(gates_reshaped[:, 1, :])  # [B, H, 1]

        # -------- innovation --------
        innov = z - (At_prev * c_prev * M)  # innov = z - m * Ac_prev  [B, H, H]
        # -------- K_t --------
        K_t = self.k_proj(innov)
        # -------- A_t 更新 --------
        I = torch.eye(self.H, device=gates.device).unsqueeze(0).expand(B, -1, -1)  # [B,H,H]
        A_t = torch.sigmoid(I - K_t * M) * A  # [B, H, H]

        # -------- B_t --------
        B_t = K_t  # [B, H, H]

        # -------- 状态更新 --------
        c_new = A_t * c_prev + B_t * z  # [B, H, H]
        h_new = (c_new * M).mean(dim=-1)     # [B, H]，用均值或 sum 来降维
        return h_new, c_new, A_t


# ---------------- RMSNorm ----------------
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output
