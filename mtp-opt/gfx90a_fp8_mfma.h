// gfx90a FP8-to-BF16 decode for paged_attention_v1
// Replaces the CDNA3-only FP8 MFMA with: decode FP8->BF16 + BF16 MFMA
// Same principle as the W8A16 GEMM kernel (proven exact on all 254 codes).

#include <hip/hip_fp8.h>
#include <hip/bfloat16.h>

// Software FP8 e4m3fn -> float (works on all archs; the __hip_fp8_e4m3
// conversion operator is implemented in software on pre-CDNA3)
__device__ __forceinline__ float sw_fp8e4m3_to_f32(uint8_t v) {
    __hip_fp8_e4m3 fn;
    __builtin_memcpy(&fn, &v, 1);
    return static_cast<float>(fn);
}

// Decode 8 FP8 bytes (packed in a long) -> 2 x _B16x4 BF16
__device__ __forceinline__ void sw_fp8x8_to_bf16x2(
    const long& inp,
    _B16x4* out /* out[2] */
) {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&inp);
    for (int half = 0; half < 2; half++) {
        for (int j = 0; j < 4; j++) {
            float f = sw_fp8e4m3_to_f32(bytes[half * 4 + j]);
            out[half].bf16[j] = __float2bfloat16(f);
        }
    }
}

// gfx90a replacement for gcn_mfma16x16x32_instr<__hip_fp8_e4m3>:
// decode both operands to BF16, use two BF16 MFMA calls (K=16 each = K=32)
template <typename T, int absz, int cbid, int blgp>
__device__ __forceinline__ floatx4 gcn_mfma16x16x32_instr_gfx90a(
    const long& inpA,
    const long& inpB,
    const floatx4& inpC
) {
    _B16x4 A_bf16[2], B_bf16[2];
    sw_fp8x8_to_bf16x2(inpA, A_bf16);
    sw_fp8x8_to_bf16x2(inpB, B_bf16);

    // Two BF16 MFMA calls, accumulating into the same C fragment
    floatx4 result = gcn_mfma16x16x16_instr<__hip_bfloat16, absz, cbid, blgp>(
        A_bf16[0], B_bf16[0], inpC);
    result = gcn_mfma16x16x16_instr<__hip_bfloat16, absz, cbid, blgp>(
        A_bf16[1], B_bf16[1], result);
    return result;
}
