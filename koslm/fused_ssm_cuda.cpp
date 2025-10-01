// fused_ssm_cuda.cpp
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <iostream>

// CUDA kernel declarations
extern "C" {
    void fused_elemwise_forward(const float* z_raw, const float* m_raw, const float* c_prev, const float* A_prev,
                                const float* A_base, const float* rawK,
                                float* c_no_scale, float* out_A, float* M_out, float* batch_sumsq,
                                int B, int H, float k_scale, float m_scale);

    void fused_apply_scale(float* out_c, float* out_h, const float* M_in, const float* batch_sumsq,
                           int B, int H, float max_norm);

    void fused_elemwise_backward(const float* grad_h, const float* grad_c, const float* z_raw, const float* m_raw,
                                 const float* c_prev, const float* A_prev, const float* A_base, const float* rawK,
                                 const float* c_no_scale, const float* out_A, const float* M_in, const float* batch_sumsq,
                                 float* grad_c_no_scale, float* grad_At, float* grad_rawK_partial,
                                 int B, int H, float k_scale, float m_scale, float max_norm);
}

// forward wrapper
std::vector<at::Tensor> fused_ssm_forward(
    at::Tensor gates_2H, // [B, 2H] split to z_raw, m_raw
    at::Tensor c_prev,   // [B,H]
    at::Tensor A_prev,   // [B,H]
    at::Tensor A_base,   // [H]
    at::Tensor W1, at::Tensor b1, at::Tensor W2, at::Tensor b2, // k_proj params
    at::Tensor z_bias, at::Tensor m_bias, // [H] biases
    float k_scale, float m_scale, float max_norm
) {
    TORCH_CHECK(gates_2H.is_cuda(), "gates_2H must be CUDA");
    int B = gates_2H.size(0);
    int twoH = gates_2H.size(1);
    int H = twoH / 2;
    auto opts = gates_2H.options();

    // split gates
    auto z_raw = gates_2H.slice(1, 0, H).contiguous();
    auto m_raw = gates_2H.slice(1, H, 2*H).contiguous();

    // apply biases
    auto z_raw_b = z_raw + z_bias.unsqueeze(0); // [B,H]
    auto m_raw_b = m_raw + m_bias.unsqueeze(0); // [B,H]

    // z_t, M_t and innov
    auto z = at::tanh(z_raw_b);
    auto M = at::tanh(m_raw_b) * m_scale;
    M = at::clamp(M, -m_scale, m_scale);

    auto Ac_prev = A_prev * c_prev;
    auto innov = z - M * Ac_prev;
    innov = at::clamp(innov, -20.0f, 20.0f);

    // k_proj: raw1 = innov @ W1.T + b1
    auto raw1 = at::matmul(innov, W1.t());
    raw1 = raw1 + b1.unsqueeze(0);
    auto gelu = at::gelu(raw1);

    // rawK = gelu @ W2.T + b2
    auto rawK = at::matmul(gelu, W2.t());
    rawK = rawK + b2.unsqueeze(0);

    // allocate outputs and intermediates
    auto c_no_scale = at::empty_like(c_prev);
    auto out_A = at::empty_like(c_prev);
    auto M_out = at::empty_like(c_prev);
    auto batch_sumsq = at::zeros({B}, opts);

    // call fused elementwise forward kernel
    fused_elemwise_forward(
        z_raw_b.data_ptr<float>(),
        m_raw_b.data_ptr<float>(),
        c_prev.data_ptr<float>(),
        A_prev.data_ptr<float>(),
        A_base.data_ptr<float>(),
        rawK.data_ptr<float>(),
        c_no_scale.data_ptr<float>(),
        out_A.data_ptr<float>(),
        M_out.data_ptr<float>(),
        batch_sumsq.data_ptr<float>(),
        B, H, k_scale, m_scale
    );

    // now produce out_c and out_h by applying scale
    auto out_c = c_no_scale.clone(); // kernel wrote c_no_scale; we will scale in kernel
    auto out_h = at::empty_like(out_c);
    fused_apply_scale(out_c.data_ptr<float>(), out_h.data_ptr<float>(), M_out.data_ptr<float>(),
                      batch_sumsq.data_ptr<float>(), B, H, max_norm);

    // save intermediates for backward
    return {out_h, out_c, out_A, batch_sumsq, innov, raw1, gelu, rawK, M_out, c_no_scale, z_raw_b, m_raw_b};
}

// backward wrapper
std::vector<at::Tensor> fused_ssm_backward(
    at::Tensor grad_out_h, at::Tensor grad_out_c,
    at::Tensor gates_2H,
    at::Tensor c_prev, at::Tensor A_prev, at::Tensor A_base,
    at::Tensor W1, at::Tensor b1, at::Tensor W2, at::Tensor b2,
    at::Tensor out_A, at::Tensor out_c, at::Tensor batch_sumsq,
    at::Tensor innov, at::Tensor raw1, at::Tensor gelu, at::Tensor rawK,
    at::Tensor M_out, at::Tensor c_no_scale,
    at::Tensor z_raw_b, at::Tensor m_raw_b,
    float k_scale, float m_scale, float max_norm
) {
    TORCH_CHECK(grad_out_h.is_cuda(), "grad must be cuda");
    int B = grad_out_h.size(0);
    int H = grad_out_h.size(1);
    auto opts = grad_out_h.options();

    // ... backward implementation 保持不变 ...

    // 注意: 这里保持你的原始 backward 代码不变
    // 返回 grad tensors
    return {
        /* grad_z_raw, grad_m_raw, grad_c_prev, grad_A_prev,
           grad_A_base, grad_W1, grad_b1, grad_W2, grad_b2 */
    };
}

// pybind11 module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_ssm_forward", &fused_ssm_forward, "Fused SSM forward (matmul + fused elem kernels)");
    m.def("fused_ssm_backward", &fused_ssm_backward, "Fused SSM backward (matmul + fused elem kernels)");
}
