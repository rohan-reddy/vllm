# GEMM + Static FP8 Output Quantization: Implementation Plan

**Related:** vllm-project/vllm#25179 (Other Hardware — Hopper, AMD, etc.)

## Problem Statement

In FP8 inference chains (e.g., consecutive FP8 linear layers in a transformer), the current
execution flow is two separate kernel launches:

```
1. cutlass_scaled_mm(out_f16, a_fp8, b_fp8, a_scales, b_scales)  → fp16/bf16 output
2. static_scaled_fp8_quant(out_fp8, out_f16, out_scale)           → FP8 output
```

This requires materializing and re-reading a full fp16/bf16 intermediate tensor from HBM.
A fused kernel can do both steps in the GEMM epilogue, saving significant memory bandwidth.

## Answer to the Key Question

> "Can we do this with our existing cutlass kernel epilogues?"

**Yes.** The existing CUTLASS 3.x EVT (Epilogue Visitor Tree) infrastructure is already
templated over both `ElementD` (output element type) and the epilogue class. The dispatch
template `cutlass_3x_gemm_sm90_fp8<ElementAB_, ElementD_, Epilogue_>` already accepts
arbitrary epilogue classes and output element types.

What we need to add:

1. A new epilogue class `ScaledEpilogueStaticFP8Quant` in
   `csrc/cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp`
2. New kernel dispatch configs with `ElementD = cutlass::float_e4m3_t`
3. New CUDA entry points
4. Python bindings

## Technical Design

### New Epilogue: `ScaledEpilogueStaticFP8Quant`

Extends `ScaledEpilogueBase` with an additional output scale node.

**EVT computation chain:**

```
D [fp8_e4m3] = fp8_saturate(ScaleOut × (ScaleA × (ScaleB × Accum)))
```

Where `ScaleOut = 1/out_scale` (reciprocal is precomputed on CPU to avoid division in kernel).

**Implementation sketch:**

```cpp
template <typename ElementAcc, typename ElementD_fp8, typename TileShape>
struct ScaledEpilogueStaticFP8Quant
    : private ScaledEpilogueBase<ElementAcc, ElementD_fp8, TileShape> {
private:
  using SUPER = ScaledEpilogueBase<ElementAcc, ElementD_fp8, TileShape>;
  using Accum  = typename SUPER::Accum;
  using ScaleA = typename SUPER::template ColOrScalarLoad<float>;   // per-col or scalar
  using ScaleB = typename SUPER::template RowOrScalarLoad<float>;   // per-row or scalar
  using RcpScaleOut = cutlass::epilogue::fusion::Sm90ScalarBroadcast<float>; // scalar

  // Step 1: ScaleB × Accum → float
  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTCompute0 = cutlass::epilogue::fusion::Sm90EVT<Compute0, ScaleB, Accum>;

  // Step 2: ScaleA × result0 → float
  using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTCompute1 = cutlass::epilogue::fusion::Sm90EVT<Compute1, ScaleA, EVTCompute0>;

  // Step 3: (1/out_scale) × result1 → fp8_e4m3  (cast with saturation)
  using Compute2 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, ElementD_fp8, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

public:
  using EVTCompute = cutlass::epilogue::fusion::Sm90EVT<Compute2, RcpScaleOut, EVTCompute1>;
  using ArgumentType = typename EVTCompute::Arguments;

  static ArgumentType prepare_args(
      torch::Tensor const& a_scales,
      torch::Tensor const& b_scales,
      torch::Tensor const& out_scale) {  // scalar tensor [1]
    // Precompute reciprocal on CPU
    float rcp_out_scale = 1.0f / out_scale.item<float>();
    // ... build EVT args
  }
};
```

### Kernel Dispatch

In `csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm90_fp8_dispatch.cuh`, add
dispatch configs instantiated with `ElementD = cutlass::float_e4m3_t` and
`Epilogue = ScaledEpilogueStaticFP8Quant`.

The tile shapes and cluster shapes can be reused from the existing SM90 fp8 configs.

### New Entry Points

New file `csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm90_fp8_with_quant.cu`:

```cpp
void cutlass_scaled_mm_sm90_fp8_with_static_fp8_quant(
    torch::Tensor& out,          // allocated FP8 output (float_e4m3_t)
    torch::Tensor const& a,      // FP8 input A
    torch::Tensor const& b,      // FP8 input B
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales,
    torch::Tensor const& out_scale);  // scalar float tensor
```

### Python Bindings (`vllm/_custom_ops.py`)

```python
def cutlass_scaled_mm_with_fp8_quant(
    out: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    out_scale: torch.Tensor,
) -> None:
    torch.ops._C.cutlass_scaled_mm_with_fp8_quant(out, a, b, a_scales, b_scales, out_scale)
```

### Integration in `Fp8LinearMethod`

In `vllm/model_executor/layers/quantization/fp8.py`, update `Fp8LinearMethod.apply()`
to detect when the caller wants an FP8 output (for the next FP8 layer's input) and use
the fused kernel path.

## Implementation Scope

| Variant | Difficulty | Priority |
|---|---|---|
| Per-tensor (scalar) static output scale | Low | **P0** |
| Per-token (row-wise) static output scale | Medium | P1 |
| Dynamic per-tensor output quant (needs online max reduction) | High | P2 |
| Dynamic per-token output quant | Very High | Future |
| Group/block-wise output scale (aligned with block GEMM tiles) | Medium | P1 |

> Note from #25179: *"Dynamic quant might be harder, again could be easier with group over
> per-token"* — per-group dynamic quant can potentially reuse the tile-level accumulators
> from block GEMM to compute per-group max values without a separate pass.

## Hardware Targets

- **SM90 (Hopper H100/H200)** — primary target; full CUTLASS 3.x EVT support
- **SM89 (Ada Lovelace RTX 4090)** — requires parallel `scaled_mm_epilogues_c2x.hpp` epilogue
- **SM100/SM120 (Blackwell)** — may already be in progress per #25179 Blackwell section
- **AMD (MI300X)** — CUTLASS approach is NVIDIA-only; a portable Triton kernel is preferred

## Key Files to Modify

| File | Change |
|---|---|
| `csrc/cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp` | Add `ScaledEpilogueStaticFP8Quant` |
| `csrc/cutlass_extensions/epilogue/scaled_mm_epilogues_c2x.hpp` | SM89 equivalent |
| `csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm90_fp8_dispatch.cuh` | New dispatch configs |
| `csrc/quantization/w8a8/cutlass/c3x/scaled_mm_kernels.hpp` | New kernel declarations |
| `csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm90_fp8_with_quant.cu` | New file: entry point |
| `csrc/quantization/w8a8/cutlass/scaled_mm_entry.cu` | Register new kernel |
| `vllm/_custom_ops.py` | Python binding |
| `vllm/model_executor/layers/quantization/fp8.py` | Integrate fused path |

## Open Questions

1. Does CUTLASS 3.x `CollectiveBuilder` support `float_e4m3_t` as `ElementD` output
   for SM90? Needs empirical verification — the accumulator is `float`, and the final
   cast to `float_e4m3_t` must be supported by the epilogue store path.

2. Should `out_scale` accept per-token (row-wise) scales as well as scalar for the
   static case? This would require `ColOrScalarLoad<float>` instead of scalar broadcast.

3. For SM89 (C2x): The `Sm80EVT` pattern is used instead of `Sm90EVT`. The epilogue
   can be structured similarly but the specific CUTLASS compute nodes may differ.

4. For portability to AMD: Should we implement this as a Triton kernel that works
   across hardware, rather than a CUTLASS-only solution? #25179 prefers portable Triton.
