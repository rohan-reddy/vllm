# Context: GEMM + Static FP8 Output Quant on Hopper

## What We Are Building

A fused CUTLASS SM90 (Hopper/H100) GEMM kernel that outputs FP8 directly using a
static pre-calibrated output scale, instead of the current two-step:
  GEMM → BF16 → separate quantize → FP8

Parent issue: https://github.com/vllm-project/vllm/issues/25179
Working branch: claude/fix-gemm-fp8-hopper-DiIeh

---

## Codebase Architecture (Key Files)

### CUDA / CUTLASS Layer

| File | Role |
|------|------|
| `csrc/cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp` | All SM90 epilogue EVT definitions. Add `ScaledEpilogueFP8Out` here. |
| `csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm90_fp8_dispatch.cuh` | Dispatch logic for SM90 FP8 GEMM (tile shape selection by M/N/K). Mirror this for FP8-out variant. |
| `csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm90_fp8.cu` | Thin wrapper that calls dispatch. Mirror for FP8-out variant. |
| `csrc/quantization/w8a8/cutlass/scaled_mm_entry.cu` | Top-level C++ entry that routes to SM90/SM89/etc. Add `cutlass_scaled_mm_fp8_out()` here. |
| `csrc/quantization/w8a8/cutlass/c3x/scaled_mm.cuh` | Generic SM90 kernel template `cutlass_3x_gemm<Epilogue>`. Reused as-is. |

### Python / Integration Layer

| File | Role |
|------|------|
| `vllm/_custom_ops.py` | Python bindings for custom CUDA ops. Add `cutlass_scaled_mm_fp8_out()`. |
| `vllm/model_executor/layers/quantization/fp8.py` | `Fp8LinearMethod` — `apply()` chooses GEMM path. Hook fused path here when `activation_scheme="static"` on SM90. |
| `vllm/model_executor/layers/quantization/kernels/scaled_mm/cutlass.py` | `CutlassFP8ScaledMMLinearKernel.apply_scaled_mm()` — add `apply_scaled_mm_fp8_out()`. |
| `vllm/model_executor/layers/quantization/kernels/scaled_mm/__init__.py` | Kernel selection logic. |
| `vllm/model_executor/layers/quantization/input_quant_fp8.py` | `QuantFP8` op. `static=True` uses pre-computed scale. |
| `vllm/model_executor/layers/quantization/utils/w8a8_utils.py` | Utilities: `requantize_with_max_scale`, `cutlass_fp8_supported`. |

### Tests

| File | Role |
|------|------|
| `tests/kernels/quantization/test_cutlass_scaled_mm.py` | CUTLASS kernel unit tests — add tests for FP8-out here. |
| `tests/kernels/quantization/test_fp8_quant.py` | FP8 quantization op tests. |

---

## Current Epilogue Structure (to understand before changing)

All epilogues live in `scaled_mm_epilogues_c3x.hpp`.
They use CUTLASS Epilogue Visitor Trees (EVT).

### Existing ScaledEpilogue (BF16 output):
```
EVTCompute0 = Sm90EVT<multiply, float, float>(ScaleB, Accum)
EVTCompute  = Sm90EVT<multiply, ElementD, float>(ScaleA, EVTCompute0)
ElementD = bfloat16_t
```

### New ScaledEpilogueFP8Out (FP8 output):
```
EVTCompute0  = Sm90EVT<multiply, float, float>(ScaleB, Accum)
EVTCompute1  = Sm90EVT<multiply, float, float>(ScaleA, EVTCompute0)
EVTCompute   = Sm90EVT<multiply, ElementD, float>(ScaleOut, EVTCompute1)
ElementD = cutlass::float_e4m3_t  ← KEY CHANGE
```

ScaleOut is a new per-tensor static scalar (shape [1,1] or scalar tensor).
CUTLASS automatically clamps on cast to float_e4m3_t, so no explicit clamp needed.

---

## Static FP8 Quantization Flow (existing, for context)

1. Weights loaded as FP8 with `weight_scale` (per-tensor or per-channel)
2. If `activation_scheme="static"`: `layer.input_scale` is registered (per-tensor scalar)
   - In `process_weights_after_loading()`: `input_scale = input_scale.max()` (line ~429)
3. `apply()` calls `fp8_linear.apply_weights()` → `QuantFP8(static=True)` quantizes input
4. `cutlass_scaled_mm(A_fp8, B_fp8, scale_a, scale_b, out_dtype=bfloat16)` called
5. Output is BF16 (then separately quantized for next layer)

The NEW path would:
- Skip step 5 producing BF16
- Instead call `cutlass_scaled_mm_fp8_out(A_fp8, B_fp8, scale_a, scale_b, scale_out)`
- Output is FP8 directly

`scale_out` = the input_scale of the NEXT layer. This requires passing it through.

---

## Implementation Steps (ordered)

### Step 1 — New epilogue (no GPU needed to write, compilation only)
Add `ScaledEpilogueFP8Out` and `ScaledEpilogueBiasFP8Out` to:
`csrc/cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp`

Pattern: copy `ScaledEpilogue` (lines ~137-169), change:
- `ElementD` to `cutlass::float_e4m3_t`
- Add `ScaleOut` as a third EVT level
- Add `scale_out` to the `Arguments` struct and `args_from_tensor()`

### Step 2 — New dispatch file (no GPU needed)
Create `csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm90_fp8_fp8out_dispatch.cuh`
Copy `scaled_mm_sm90_fp8_dispatch.cuh`, replace epilogue template args with FP8-out version.
All tile shapes (M<=16, M<=64, M<=128, large M) remain identical.

### Step 3 — New kernel .cu file (no GPU needed)
Create `csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm90_fp8_fp8out.cu`
Copy `scaled_mm_sm90_fp8.cu`, add `scale_out` parameter, call new dispatch.

### Step 4 — Entry point (no GPU needed)
In `scaled_mm_entry.cu`, add:
```cpp
void cutlass_scaled_mm_fp8_out(
    torch::Tensor& out,           // FP8 output pre-allocated
    torch::Tensor const& a,       // FP8 input
    torch::Tensor const& b,       // FP8 weight
    torch::Tensor const& scale_a, // [1,1] or [M,1]
    torch::Tensor const& scale_b, // [1,1] or [1,N]
    torch::Tensor const& scale_out, // [1,1] static scalar
    c10::optional<torch::Tensor> const& bias
)
```
Route to SM90 FP8-out kernel for SM90, return error for other arches.

### Step 5 — Op registration + Python binding (no GPU needed)
- `csrc/ops.h`: declare `cutlass_scaled_mm_fp8_out`
- `vllm/_custom_ops.py`: add Python wrapper
- `CMakeLists.txt`: add new .cu file to build

### Step 6 — Python integration in fp8.py (no GPU needed)
In `Fp8LinearMethod.apply()`, add logic:
```python
if (self.act_q_static and
    output_scale is not None and
    cutlass_fp8_supported() and
    is_sm90()):
    return ops.cutlass_scaled_mm_fp8_out(
        x_q, w, scale_a=x_s, scale_b=w_s, scale_out=output_scale, bias=bias)
```

### Step 7 — Tests (can write without GPU, run needs H100)
In `tests/kernels/quantization/test_cutlass_scaled_mm.py`:
```python
@pytest.mark.skipif(not is_sm90(), reason="SM90 only")
def test_cutlass_fp8_gemm_fp8_out(M, N, K, ...):
    # Reference: BF16 GEMM + separate ops.scaled_fp8_quant
    # Actual: cutlass_scaled_mm_fp8_out
    # Assert close within FP8 rounding tolerance
```

---

## Minimizing H100 GPU Time

See section below in main notes.

---

## Important CUTLASS Notes

- SM90 uses `KernelTmaWarpSpecializedPingpongFP8FastAccum` scheduler
- FP8 accumulation is in float32 internally
- `float_e4m3_t` max value is 448.0; CUTLASS clamps on output cast
- EVT args passed via `Args` struct to `CollectiveEpilogue::Arguments`
- `args_from_tensor()` helper in `ScaledEpilogueBase` builds the arg tree from tensors

---

## Key Definitions Locations

- `Sm90EVT`, `Sm90Compute`: in CUTLASS headers (installed via submodule)
- `KernelTmaWarpSpecializedPingpongFP8FastAccum`: `cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp`
- `CollectiveBuilder` for SM90: `cutlass/gemm/collective/builders/sm90_gmma_builder.inl`
- `float_e4m3_t`: `cutlass/numeric_types.h`
