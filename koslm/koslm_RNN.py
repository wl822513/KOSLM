# fused_lstm/fused_lstm.py
import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch import Tensor
# from .koslmssm import KOSLMSSM
# from . import koslm_cuda


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

    def _segment_scan(self, x_seg, h0, c0):
        # x_seg: [B, S, D], 逐步 scan
        B, S, _ = x_seg.shape
        h, c = h0, c0
        hs = []
        for t in range(S):
            xt = x_seg[:, t, :]
            if self.use_ckpt and self.training:
                h, c = checkpoint(lambda _x, _h, _c: self.cell(_x, (_h, _c))[:3],
                                     xt, h, c, use_reentrant=False)
            else:
                h, c = self.cell(xt, (h, c))
            hs.append(h.unsqueeze(1))
        return torch.cat(hs, dim=1), (h, c)

    def forward(self, x):
        """
        x: [B, L, D]
        """
        B, L, _ = x.shape
        device = x.device
        h = torch.zeros(B, self.H, device=device)
        c = torch.zeros(B, self.H, device=device)
        outputs = []
        for s in range(0, L, self.segment_len):
            seg = x[:, s:s+self.segment_len, :]
            y_seg, (h, c) = self._segment_scan(seg, h, c)
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

    def forward(self, x, hx):
        h_prev, c_prev = hx  # [B,H], [B,H]
        gates = self.weight_ih(x) + self.weight_hh(h_prev)  # [B,2H]
        h_t, c_t = self.ssm(gates, c_prev)
        return h_t, c_t


def proj_tanh_scale(x, scale=0.9):
    # 有界投影，避免过大增益
    return scale * torch.tanh(x)


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

        # 可选：对 M（观测矩阵的对角）也学习一个偏置，提高可观测性且避免饱和
        self.m_bias = nn.Parameter(torch.zeros(H))
        self.z_bias = nn.Parameter(torch.zeros(H))

        # 约束幅值的可调上限
        self.k_scale = 0.9
        self.m_scale = 1.5

    def forward(self, gates_2H: torch.Tensor, c_prev: torch.Tensor):
        """
        gates_2H: [B, 2H] -> (z_raw, m_raw)
        c_prev:   [B, H]
        A_prev:  [B, H]
        返回:
          h_t:  [B, H]   (读出)
          c_t:  [B, H]   (状态)
          A_t:  [B, H]   (对角，便于监控)
        """
        B, twoH = gates_2H.shape
        H = self.H
        assert twoH == 2*H

        z_raw, m_raw = gates_2H.split(H, dim=-1)

        # z_t, M_t（对角向量）
        # z 用 tanh，附加小偏置，避免过早饱和
        z_t = torch.tanh(z_raw + self.z_bias)                              # [B,H]
        M_t = proj_tanh_scale(m_raw + self.m_bias, scale=self.m_scale)     # [B,H] （对角向量，可取 [-m_scale,m_scale]）

        # 基础稳定 A（对角离散），并做选择性反馈：(I - K_t M_t) A
        A_base = -torch.exp(self.log_lambda)  # [H]

        # innovation：z - M * (A * c_prev)
        # 对角结构等价： M_t ∘ (A_base ∘ c_prev) ; 其中 ∘ 为逐元素，对角乘法的等价实现
        Ac_prev = A_base * c_prev                                          # [B,H]  (对角A)
        Mc_prev = M_t * Ac_prev                                            # [B,H]  (对角M)
        innov = z_t - Mc_prev                                              # [B,H]

        # 1) 更保守的 k_proj 初始化（如果第一次运行需要的话，见后面 init 建议）
        # 2) 对 innov 做小的缩放（避免初期过大），并增加残差 clip
        innov = torch.clamp(innov, min=-20.0, max=20.0)  # 防止极值传播

        # 3) K_t 先通过 k_proj，再做 tanh 限幅，最后缩放到小的范围内
        #    这里把 scale 进一步降低以确保 B_t 不会导致大增益
        raw_K = self.k_proj(innov)  # [B,H]
        # optional small-weighted residual to keep initial K small: K = eps * raw_K + (1-eps)*0
        K_t = torch.tanh(raw_K) * (self.k_scale * 0.5)  # 降低 scale，默认 self.k_scale=0.9 -> 0.45

        # 4) 确保 M_t 受限（已使用 proj_tanh_scale），但再 clamp 一步
        M_t = torch.clamp(M_t, min=-self.m_scale, max=self.m_scale)

        # 5) 计算 A_t，随后立刻 clamp 到稳定区间 [A_min, A_max]
        A_t = (1.0 - K_t * M_t) * A_base  # [B,H]
        # # 强制 A_t 的上界，例如 0.95，防止 1.0 附近的几何放大
        A_t = torch.clamp(A_t, min=0.0, max=0.95)  # 0<=A_t<=0.95

        # 6) B_t 同样 clamp（避免 1.0 级别）
        B_t = torch.clamp(K_t, min=-0.95, max=0.95)

        # 7) 前向状态更新：在更新后对 c_t 做可选的 LayerNorm/RMSNorm 缓和
        c_t = A_t * c_prev + B_t * z_t  # [B,H]

        # 8) 额外归一化：小幅度归一化 c_t，防止逐段累积出界（保留比例）
        #    这个是安全网：把 c_t 缩放到合理范围（可学习 scale）
        #    你可以注释掉或通过超参开关来启用/禁用
        c_norm = torch.sqrt((c_t ** 2).mean(dim=-1, keepdim=True) + 1e-6)  # [B,1]
        max_norm = 100.0  # 允许的最大范数（可调）
        # 如果范数过大，则按比例缩回
        scale_down = torch.clamp(max_norm / c_norm, max=1.0)
        c_t = c_t * scale_down

        # 9) 读出
        h_t = M_t * c_t
        # ------------------- 安全 / 数值稳定策略结束 -------------------
        return h_t, c_t


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output




