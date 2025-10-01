// fused_ssm_cuda_kernel.cu
// Fused elementwise CUDA kernels for KOSLMSSM step
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

extern "C" {

// forward elementwise: compute c_no_scale = A_t*c_prev + B_t*z, out_A=A_t, store M, accumulate batch_sumsq
__global__ void fused_elemwise_forward(
    const float* __restrict__ z_raw,    // [B,H] (already z_raw + z_bias applied in C++)
    const float* __restrict__ m_raw,    // [B,H] (already m_raw + m_bias applied in C++)
    const float* __restrict__ c_prev,   // [B,H]
    const float* __restrict__ A_prev,   // [B,H]  (not used in fused stage except for innov calc done in C++)
    const float* __restrict__ A_base,   // [H]
    const float* __restrict__ rawK,     // [B,H] (from gelu@W2 + b2)
    float* __restrict__ c_no_scale,     // [B,H] out
    float* __restrict__ out_A,          // [B,H] out
    float* __restrict__ M_out,          // [B,H] store M for backward
    float* __restrict__ batch_sumsq,    // [B] accumulate sumsq
    int B, int H,
    float k_scale,
    float m_scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H;
    if (idx >= total) return;
    int b = idx / H;
    int h = idx % H;

    // z_raw and m_raw are passed after adding biases in C++ side
    float z = tanhf(z_raw[idx]);
    float M = tanhf(m_raw[idx]) * m_scale;
    if (M > m_scale) M = m_scale;
    if (M < -m_scale) M = -m_scale;

    float cprev = c_prev[idx];
    float Abase = A_base[h];
    float rawk = rawK[idx];

    float K = tanhf(rawk) * (k_scale * 0.5f);
    if (K > 0.95f) K = 0.95f;
    if (K < -0.95f) K = -0.95f;

    float At = (1.0f - K * M) * Abase;
    if (At > 0.95f) At = 0.95f;
    if (At < 0.0f) At = 0.0f;

    float Bt = K;

    float cns = At * cprev + Bt * z;
    c_no_scale[idx] = cns;
    atomicAdd(&batch_sumsq[b], cns * cns);

    out_A[idx] = At;
    M_out[idx] = M;
}

// apply scale_down and compute out_c/out_h
__global__ void fused_apply_scale(
    float* __restrict__ out_c,  // [B,H] input=c_no_scale -> output scaled
    float* __restrict__ out_h,  // [B,H] output computed as M * out_c
    const float* __restrict__ M_in, // [B,H]
    const float* __restrict__ batch_sumsq, // [B]
    int B, int H, float max_norm
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H;
    if (idx >= total) return;
    int b = idx / H;
    float mean_sq = batch_sumsq[b] / float(H);
    float c_norm = sqrtf(mean_sq + 1e-6f);
    float scale_down = 1.0f;
    if (c_norm > max_norm) scale_down = max_norm / c_norm;

    out_c[idx] = out_c[idx] * scale_down;
    out_h[idx] = M_in[idx] * out_c[idx];
}


// backward elementwise: compute contributions that don't require matmul (we will compute weight-related grads in C++)
// This kernel computes per-element:
// - grad_c_no_scale primary contributions from g_out (combination of grad_c and grad_h via M)
// - dL/dAt partial (to be used to compute grad_A_base)
// - store values needed for matmul-based grads (grad_rawK placeholder filled by C++)
__global__ void fused_elemwise_backward(
    const float* __restrict__ grad_h,  // [B,H]
    const float* __restrict__ grad_c,  // [B,H]
    const float* __restrict__ z_raw,   // [B,H] (z_raw + z_bias)
    const float* __restrict__ m_raw,   // [B,H] (m_raw + m_bias)
    const float* __restrict__ c_prev,  // [B,H]
    const float* __restrict__ A_prev,  // [B,H]
    const float* __restrict__ A_base,  // [H]
    const float* __restrict__ rawK,    // [B,H]
    const float* __restrict__ c_no_scale, // [B,H] (before scaling)
    const float* __restrict__ out_A,   // [B,H]
    const float* __restrict__ M_in,    // [B,H]
    const float* __restrict__ batch_sumsq, // [B]
    float* __restrict__ grad_c_no_scale, // [B,H] output primary (will include scale effect later)
    float* __restrict__ grad_At,          // [B,H] = dL/dAt (primary)
    float* __restrict__ grad_rawK_partial,// [B,H] (partial dL/dK from c path; final dL/drawK computed in C++)
    int B, int H,
    float k_scale,
    float m_scale,
    float max_norm
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H;
    if (idx >= total) return;
    int b = idx / H;
    int h = idx % H;

    float M = tanhf(m_raw[idx]) * m_scale;
    if (M > m_scale) M = m_scale;
    if (M < -m_scale) M = -m_scale;

    float g_h = grad_h[idx];
    float g_c = grad_c[idx];

    // combined upstream gradient on post-scale out_c:
    float g_out = g_c + g_h * M; // because out_h = M * out_c

    // compute scale_down later in C++ (we need batch_sumsq)
    // primary grad on c_no_scale pre-scale = g_out * scale_down  (scale_down applied later)
    // we store g_out here to be multiplied by scale_down in C++ (vectorized)
    grad_c_no_scale[idx] = g_out;

    // dL/dAt (pre-scale) = g_c_no_scale * c_prev ; but g_c_no_scale = g_out * scale_down; store g_out*c_prev here
    grad_At[idx] = g_out * c_prev[idx];

    // contribute dL/dK from c path: dL/dK_from_c = g_c_no_scale * z_t -> store g_out * z (z = tanh(z_raw))
    float z = tanhf(z_raw[idx]);
    grad_rawK_partial[idx] = g_out * z;
}
} // extern "C"
