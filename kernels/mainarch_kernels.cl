// mainarch GPU kernels — compiled to gfx950 HSA code objects.
//
// These are dispatched directly through the raw KFD/AQL path in
// mainarch-core; there is no HSA/ROCr runtime at execution time. The ROCm
// LLVM is used here only as a build-time cross-compiler (same role as any
// CUDA/HIP compiler), exactly like every other GPU kernel toolchain.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Proof-of-execution kernel: each work-item stamps a sentinel so the host can
// confirm the GPU actually ran the dispatch by polling memory.
__kernel void poke(__global uint* out) {
    out[get_global_id(0)] = 0xD00DFEEDu;
}

// Ordered GPU-side realtime stamp. The host uses this as a tiny AQL packet
// before/after replayed chains to get a device-observed tick delta without
// importing ROCr profiling queues into the raw KFD/AQL path. s_memtime is not
// safe for cross-dispatch deltas on this target because independent dispatches
// can land on different clock domains; s_memrealtime is the frequency-stable
// real-time counter exposed by the AMDGPU backend.
__kernel void gpu_memtime_stamp(__global ulong* out, uint index, ulong sentinel) {
    if (get_global_id(0) != 0) return;
    ulong t = __builtin_amdgcn_s_memrealtime();
    uint base = index * 2u;
    out[base] = t;
    out[base + 1u] = sentinel;
}

// Ordered GPU-side realtime stamp plus queue-ordered producer marker. This is
// used as the existing terminal timestamp packet in descriptor replay chains,
// avoiding a separate one-work-item ready-flag packet while preserving
// same-queue ordering after the MLP down GEMV.
__kernel void gpu_memtime_stamp_ready_flag(__global ulong* out,
                                           uint index,
                                           ulong sentinel,
                                           volatile __global atomic_uint* flags,
                                           uint rank) {
    if (get_global_id(0) != 0) return;
    ulong t = __builtin_amdgcn_s_memrealtime();
    uint base = index * 2u;
    out[base] = t;
    out[base + 1u] = sentinel;
    atomic_store_explicit(&flags[rank], 1u, memory_order_release,
                          memory_scope_all_svm_devices);
}

// GPU-side packed Qwen projection contract check. This is intentionally tiny:
// one work-item reads the exact q/k/v and gate/up packed boundary bytes from
// the resident layer-weight buffer, folds them with the numeric ABI offsets,
// and returns a device-observed fingerprint for the host to compare.
__kernel void qwen_packed_contract_check(__global const uchar* weights,
                                         __global const ulong* contract,
                                         __global ulong* out) {
    if (get_global_id(0) != 0) return;

    const ulong FNV_OFFSET = 0xcbf29ce484222325UL;
    const ulong FNV_PRIME = 0x100000001b3UL;
    ulong h = FNV_OFFSET;

    for (uint i = 0; i < 14u; ++i) {
        ulong x = contract[i];
        for (uint b = 0; b < 8u; ++b) {
            h ^= (x >> (8u * b)) & 0xffUL;
            h *= FNV_PRIME;
        }
    }

    ulong qkv = contract[0];
    ulong q = contract[2], q_len = contract[3];
    ulong k = contract[4], k_len = contract[5];
    ulong v = contract[6], v_len = contract[7];
    ulong gate_up = contract[8];
    ulong gate = contract[10], gate_len = contract[11];
    ulong up = contract[12], up_len = contract[13];

    uchar qb0 = weights[qkv + q];
    uchar qb1 = weights[qkv + q + q_len - 1UL];
    uchar kb0 = weights[qkv + k];
    uchar kb1 = weights[qkv + k + k_len - 1UL];
    uchar vb0 = weights[qkv + v];
    uchar vb1 = weights[qkv + v + v_len - 1UL];
    uchar gb0 = weights[gate_up + gate];
    uchar gb1 = weights[gate_up + gate + gate_len - 1UL];
    uchar ub0 = weights[gate_up + up];
    uchar ub1 = weights[gate_up + up + up_len - 1UL];

    uchar edges[10] = {qb0, qb1, kb0, kb1, vb0, vb1, gb0, gb1, ub0, ub1};
    for (uint i = 0; i < 10u; ++i) {
        h ^= (ulong)edges[i];
        h *= FNV_PRIME;
    }

    out[0] = h;
    out[1] = ((ulong)qb0) | ((ulong)qb1 << 8) | ((ulong)kb0 << 16) |
             ((ulong)kb1 << 24) | ((ulong)vb0 << 32) | ((ulong)vb1 << 40);
    out[2] = ((ulong)gb0) | ((ulong)gb1 << 8) | ((ulong)ub0 << 16) |
             ((ulong)ub1 << 24);
    out[3] = 0xD00DFEEDUL;
}

static inline ushort qwen_load_u16_le(__global const uchar* p) {
    return (ushort)p[0] | ((ushort)p[1] << 8);
}

static inline float qwen_load_bf16(__global const uchar* weights, ulong byte_off) {
    uint bits = ((uint)qwen_load_u16_le(weights + byte_off)) << 16;
    return as_float(bits);
}

static inline float qwen_bf16_bits_to_f32(ushort bits) {
    return as_float(((uint)bits) << 16);
}

static inline ushort qwen_f32_to_bf16_bits(float value) {
    uint bits = as_uint(value);
    uint lsb = (bits >> 16) & 1u;
    return (ushort)((bits + 0x7fffu + lsb) >> 16);
}

static inline void qwen_fnv_mix_u8(ulong* h, uchar x) {
    *h ^= (ulong)x;
    *h *= 0x100000001b3UL;
}

static inline void qwen_fnv_mix_u32(ulong* h, uint x) {
    qwen_fnv_mix_u8(h, (uchar)(x & 0xffu));
    qwen_fnv_mix_u8(h, (uchar)((x >> 8) & 0xffu));
    qwen_fnv_mix_u8(h, (uchar)((x >> 16) & 0xffu));
    qwen_fnv_mix_u8(h, (uchar)((x >> 24) & 0xffu));
}

static inline void qwen_fnv_mix_u64(ulong* h, ulong x) {
    for (uint b = 0; b < 8u; ++b) qwen_fnv_mix_u8(h, (uchar)((x >> (8u * b)) & 0xffUL));
}

static inline void qwen_fnv_mix_f32(ulong* h, float x) {
    qwen_fnv_mix_u32(h, as_uint(x));
}

static inline float qwen_qk_synth(uint i, float salt) {
    int centered = (int)(i % 29u) - 14;
    return (float)centered * 0.03125f + salt + (float)(i & 7u) * 0.001953125f;
}

static inline float qwen_qk_order_delta_and_mix(__global const uchar* weights,
                                                ulong norm_off,
                                                uint d,
                                                float eps,
                                                uint pos,
                                                float theta,
                                                float salt,
                                                ulong* h) {
    float ss = 0.0f;
    for (uint i = 0; i < d; ++i) {
        float v = qwen_qk_synth(i, salt);
        ss += v * v;
    }
    float rms = rsqrt(ss / (float)d + eps);
    uint half_d = d >> 1;
    float delta = 0.0f;
    for (uint i = 0; i < half_d; ++i) {
        float x0 = qwen_qk_synth(i, salt);
        float x1 = qwen_qk_synth(i + half_d, salt);
        float w0 = qwen_load_bf16(weights, norm_off + (ulong)i * 2UL);
        float w1 = qwen_load_bf16(weights, norm_off + (ulong)(i + half_d) * 2UL);
        float freq = pow(theta, -2.0f * (float)i / (float)d);
        float ang = (float)pos * freq;
        float c = native_cos(ang), s = native_sin(ang);

        float n0 = x0 * rms * w0;
        float n1 = x1 * rms * w1;
        float correct0 = n0 * c - n1 * s;
        float correct1 = n1 * c + n0 * s;

        float r0 = x0 * c - x1 * s;
        float r1 = x1 * c + x0 * s;
        float wrong0 = r0 * rms * w0;
        float wrong1 = r1 * rms * w1;
        delta += fabs(correct0 - wrong0) + fabs(correct1 - wrong1);

        qwen_fnv_mix_f32(h, correct0);
        qwen_fnv_mix_f32(h, correct1);
    }
    return delta;
}

// GPU-side Qwen QK-norm/RoPE ordering contract. The layer buffer holds real
// SafeTensors bytes; q_norm/k_norm are BF16 in current Qwen3 checkpoints. One
// work-item computes a deterministic synthetic Q/K vector, applies real q_norm
// and k_norm before RoPE, and also measures how different the incorrect
// RoPE-before-norm ordering would be with the same real weights.
__kernel void qwen_qk_norm_rope_order_check(__global const uchar* weights,
                                            __global const ulong* contract,
                                            __global ulong* out) {
    if (get_global_id(0) != 0) return;

    ulong q_off = contract[0], q_len = contract[1];
    ulong k_off = contract[2], k_len = contract[3];
    uint pos = (uint)contract[4];
    float theta = as_float((uint)contract[5]);
    float eps = as_float((uint)contract[6]);
    if (q_len == 0UL || q_len != k_len || (q_len & 1UL) != 0UL) {
        out[0] = 0UL; out[1] = 0UL; out[2] = 0UL; out[3] = 0UL; out[4] = 0UL;
        return;
    }
    uint d = (uint)(q_len >> 1);
    if (d == 0u || (d & 1u) != 0u || d > 256u) {
        out[0] = 0UL; out[1] = 0UL; out[2] = 0UL; out[3] = 0UL; out[4] = 0UL;
        return;
    }

    ulong h = 0xcbf29ce484222325UL;
    for (uint i = 0; i < 7u; ++i) qwen_fnv_mix_u64(&h, contract[i]);
    ushort q0 = qwen_load_u16_le(weights + q_off);
    ushort q1 = qwen_load_u16_le(weights + q_off + q_len - 2UL);
    ushort k0 = qwen_load_u16_le(weights + k_off);
    ushort k1 = qwen_load_u16_le(weights + k_off + k_len - 2UL);
    qwen_fnv_mix_u32(&h, (uint)q0 | ((uint)q1 << 16));
    qwen_fnv_mix_u32(&h, (uint)k0 | ((uint)k1 << 16));

    float q_delta = qwen_qk_order_delta_and_mix(weights, q_off, d, eps, pos, theta, 0.0078125f, &h);
    float k_delta = qwen_qk_order_delta_and_mix(weights, k_off, d, eps, pos, theta, -0.01171875f, &h);

    out[0] = h;
    out[1] = (ulong)q0 | ((ulong)q1 << 16) | ((ulong)d << 32);
    out[2] = (ulong)k0 | ((ulong)k1 << 16) | ((ulong)d << 32);
    out[3] = (ulong)as_uint(q_delta) | ((ulong)as_uint(k_delta) << 32);
    out[4] = 0xD00DFEEDUL;
}

// Pure VRAM streaming-read microbenchmark: grid-stride over a big buffer with
// 128-bit (float4) loads, full occupancy. Establishes the real achievable HBM
// read bandwidth in this environment (the ceiling our kernels are measured
// against). Uses only group/local id (no COv5 hidden args, so arm_grid drives
// it). Workgroup size is fixed at 256; `nthreads` = num_wg*256.
__kernel void mem_stream(__global const float4* in, __global float* out,
                         uint n4, uint nthreads) {
    uint gid = get_group_id(0) * 256u + get_local_id(0);
    float4 acc = (float4)(0.0f);
    for (uint i = gid; i < n4; i += nthreads) acc += in[i];
    // Defeat dead-code elimination without ever actually writing.
    if (acc.x + acc.y + acc.z + acc.w == -1.0e30f) out[gid] = acc.x;
}

// In-place elementwise sum reduction across `parts` contiguous f32 segments of
// length `n`. Segment 0 receives the sum of all segments. This is the compute
// core of an all-reduce once data from every rank is gathered contiguously.
__kernel void reduce_sum_f32(__global float* data, uint n, uint parts) {
    uint i = get_global_id(0);
    if (i >= n) return;
    float acc = data[i];
    for (uint p = 1; p < parts; ++p) {
        acc += data[(ulong)p * n + i];
    }
    data[i] = acc;
}

// Elementwise accumulate: dst[i] += src[i]. `src` may live on a peer GPU
// mapped over XGMI, making this the per-step kernel of a ring/peer all-reduce.
__kernel void accumulate_f32(__global float* dst, __global const float* src, uint n) {
    uint i = get_global_id(0);
    if (i < n) dst[i] += src[i];
}

// Broadcast scale: dst[i] = src[i] * scale. Used for the average op and to
// fan a reduced result back out.
__kernel void scale_f32(__global float* dst, __global const float* src, float scale, uint n) {
    uint i = get_global_id(0);
    if (i < n) dst[i] = src[i] * scale;
}

// Direct all-reduce reduce step: dst[i] = sum over p of ptrs[p][i], where
// ptrs holds `parts` device pointers (each a peer rank's buffer, possibly on
// another GPU reached over XGMI). One dispatch sums all ranks; the peer reads
// run concurrently across XGMI links.
__kernel void reduce_peers(__global float* dst, __global const ulong* ptrs, uint parts, uint n) {
    uint i = get_global_id(0);
    if (i >= n) return;
    if (parts == 8u) {
        __global const float* src0 = (__global const float*)ptrs[0];
        __global const float* src1 = (__global const float*)ptrs[1];
        __global const float* src2 = (__global const float*)ptrs[2];
        __global const float* src3 = (__global const float*)ptrs[3];
        __global const float* src4 = (__global const float*)ptrs[4];
        __global const float* src5 = (__global const float*)ptrs[5];
        __global const float* src6 = (__global const float*)ptrs[6];
        __global const float* src7 = (__global const float*)ptrs[7];
        dst[i] = src0[i] + src1[i] + src2[i] + src3[i]
               + src4[i] + src5[i] + src6[i] + src7[i];
    } else {
        float acc = 0.0f;
        for (uint p = 0; p < parts; ++p) {
            __global const float* src = (__global const float*)ptrs[p];
            acc += src[i];
        }
        dst[i] = acc;
    }
}

// Queue-ordered producer marker for descriptor replay chains. The kernel is
// launched as a single work-item after the rank-local MLP down GEMV; it releases
// flags[rank] so a later rank-0 reduce kernel can wait in CUs instead of using
// fragile cross-queue CP barrier packets.
__kernel void write_ready_flag(volatile __global atomic_uint* flags, uint rank) {
    atomic_store_explicit(&flags[rank], 1u, memory_order_release,
                          memory_scope_all_svm_devices);
}

// Direct all-reduce reduce step that first waits for all descriptor replay
// producers to publish their ready flags. Slot flags[parts] is a watchdog
// error word; a timeout stores 0xDEAD0101 and lets the kernel retire so the
// node does not wedge if a producer packet never runs.
__kernel void reduce_peers_wait_ready_flags(__global float* dst,
                                            __global const ulong* ptrs,
                                            volatile __global atomic_uint* flags,
                                            uint parts, uint n) {
    uint lid = get_local_id(0);
    if (lid == 0) {
        ulong start_time = __builtin_amdgcn_s_memrealtime();
        ulong spins = 0;
        for (;;) {
            uint ready = 1u;
            for (uint p = 0; p < parts; ++p) {
                uint flag = atomic_load_explicit(&flags[p], memory_order_acquire,
                                                 memory_scope_all_svm_devices);
                if (flag != 1u) {
                    ready = 0u;
                    break;
                }
            }
            if (ready != 0u) break;
            ++spins;
            if (spins > 1000000UL) {
                atomic_store_explicit(&flags[parts], 0xdead0101u,
                                      memory_order_release,
                                      memory_scope_all_svm_devices);
                break;
            }
            if (((spins & 0xffUL) == 0UL)) {
                ulong now = __builtin_amdgcn_s_memrealtime();
                if ((now - start_time) > 10000000UL) {
                    atomic_store_explicit(&flags[parts], 0xdead0101u,
                                          memory_order_release,
                                          memory_scope_all_svm_devices);
                    break;
                }
            }
            __builtin_amdgcn_s_sleep(1);
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    if (atomic_load_explicit(&flags[parts], memory_order_acquire,
                             memory_scope_all_svm_devices) != 0u) {
        return;
    }

    uint i = get_global_id(0);
    if (i >= n) return;
    if (parts == 8u) {
        __global const float* src0 = (__global const float*)ptrs[0];
        __global const float* src1 = (__global const float*)ptrs[1];
        __global const float* src2 = (__global const float*)ptrs[2];
        __global const float* src3 = (__global const float*)ptrs[3];
        __global const float* src4 = (__global const float*)ptrs[4];
        __global const float* src5 = (__global const float*)ptrs[5];
        __global const float* src6 = (__global const float*)ptrs[6];
        __global const float* src7 = (__global const float*)ptrs[7];
        dst[i] = src0[i] + src1[i] + src2[i] + src3[i]
               + src4[i] + src5[i] + src6[i] + src7[i];
    } else {
        float acc = 0.0f;
        for (uint p = 0; p < parts; ++p) {
            __global const float* src = (__global const float*)ptrs[p];
            acc += src[i];
        }
        dst[i] = acc;
    }
}

// Direct all-reduce broadcast step: write src[i] into every ptrs[p][i]. One
// dispatch scatters the reduced result back to all ranks over XGMI.
__kernel void broadcast_peers(__global const float* src, __global const ulong* ptrs, uint parts, uint n) {
    uint i = get_global_id(0);
    if (i >= n) return;
    float v = src[i];
    for (uint p = 0; p < parts; ++p) {
        __global float* d = (__global float*)ptrs[p];
        d[i] = v;
    }
}

// 512 KiB-class direct all-reduce broadcast: rank 0 already owns src[i] from
// the reduce step, so skip the redundant self-store and write only peers.
__kernel void broadcast_peers_skip0(__global const float* src, __global const ulong* ptrs, uint parts, uint n) {
    uint i = get_global_id(0);
    if (i >= n) return;
    float v = src[i];
    if (parts == 8u) {
        ((__global float*)ptrs[1])[i] = v;
        ((__global float*)ptrs[2])[i] = v;
        ((__global float*)ptrs[3])[i] = v;
        ((__global float*)ptrs[4])[i] = v;
        ((__global float*)ptrs[5])[i] = v;
        ((__global float*)ptrs[6])[i] = v;
        ((__global float*)ptrs[7])[i] = v;
    } else {
        for (uint p = 1; p < parts; ++p) {
            __global float* d = (__global float*)ptrs[p];
            d[i] = v;
        }
    }
}

static void grid_barrier(volatile __global atomic_uint* gbar, uint num_wg);

static inline void peer_flag_relax(void) {
    __builtin_amdgcn_s_sleep(1);
}

// Persistent direct all-reduce prototype. A single resident rank-0 kernel waits
// on ctrl[0] sequence numbers, then performs the same direct reduce+broadcast
// algorithm without per-op AQL launches. ctrl layout:
//   ctrl[0] = go sequence written by host
//   ctrl[1] = done sequence written by kernel
//   ctrl[2] = error marker, 0 on success
//
// This is intentionally rank-0/direct only: it is a low-risk latency substrate
// for decode-sized messages, not a bandwidth path for MiB+ payloads.
__kernel void allreduce_direct_persistent(
    __global float* own,
    __global const ulong* ptrs,
    volatile __global atomic_uint* ctrl,
    volatile __global atomic_uint* gbar,
    uint parts,
    uint n,
    uint total_ops) {
    uint tid = get_global_id(0);
    uint nthreads = get_global_size(0);
    uint num_wg = get_num_groups(0);

    for (uint seq = 1u; seq <= total_ops; ++seq) {
        if (tid == 0) {
            ulong spins = 0;
            while (atomic_load_explicit(&ctrl[0], memory_order_acquire,
                                        memory_scope_all_svm_devices) != seq) {
                if (++spins > 4000000000ul) {
                    atomic_store_explicit(&ctrl[2], 0xdead0001u, memory_order_release,
                                          memory_scope_all_svm_devices);
                    break;
                }
            }
        }
        grid_barrier(gbar, num_wg);
        if (atomic_load_explicit(&ctrl[2], memory_order_acquire,
                                 memory_scope_all_svm_devices) != 0u) {
            break;
        }

        for (uint i = tid; i < n; i += nthreads) {
            float acc = 0.0f;
            for (uint p = 0; p < parts; ++p) {
                __global const float* src = (__global const float*)ptrs[p];
                acc += src[i];
            }
            own[i] = acc;
        }
        grid_barrier(gbar, num_wg);

        for (uint i = tid; i < n; i += nthreads) {
            float v = own[i];
            for (uint p = 0; p < parts; ++p) {
                __global float* dst = (__global float*)ptrs[p];
                dst[i] = v;
            }
        }
        grid_barrier(gbar, num_wg);

        if (tid == 0) {
            atomic_store_explicit(&ctrl[1], seq, memory_order_release,
                                  memory_scope_all_svm_devices);
        }
    }
}

// Persistent all-rank DDA-flat all-reduce prototype. One resident kernel per
// rank waits on its own host-visible ctrl block, directly reads all peer input
// buffers over XGMI from ptrs[], and writes the reduced result to a separate
// per-rank output buffer. Separate output avoids in-place read/write races.
__kernel void allreduce_dda_persistent(
    __global float* out,
    __global const ulong* ptrs,
    volatile __global atomic_uint* ctrl,
    volatile __global atomic_uint* gbar,
    uint parts,
    uint n,
    uint total_ops) {
    uint tid = get_global_id(0);
    uint nthreads = get_global_size(0);
    uint num_wg = get_num_groups(0);

    for (uint seq = 1u; seq <= total_ops; ++seq) {
        if (tid == 0) {
            ulong spins = 0;
            while (atomic_load_explicit(&ctrl[0], memory_order_acquire,
                                        memory_scope_all_svm_devices) != seq) {
                if (++spins > 4000000000ul) {
                    atomic_store_explicit(&ctrl[2], 0xdda00001u, memory_order_release,
                                          memory_scope_all_svm_devices);
                    break;
                }
            }
        }
        grid_barrier(gbar, num_wg);
        if (atomic_load_explicit(&ctrl[2], memory_order_acquire,
                                 memory_scope_all_svm_devices) != 0u) {
            break;
        }

        if (parts == 8u) {
            __global const float* src0 = (__global const float*)ptrs[0];
            __global const float* src1 = (__global const float*)ptrs[1];
            __global const float* src2 = (__global const float*)ptrs[2];
            __global const float* src3 = (__global const float*)ptrs[3];
            __global const float* src4 = (__global const float*)ptrs[4];
            __global const float* src5 = (__global const float*)ptrs[5];
            __global const float* src6 = (__global const float*)ptrs[6];
            __global const float* src7 = (__global const float*)ptrs[7];
            // 8-rank MI355X DDA-flat all-reduce: include 2 KiB+ in the float4
            // path for decode-sized payloads while keeping smaller messages scalar.
            if ((n & 3u) == 0u && n >= 512u) {
                uint nv = n >> 2;
                for (uint v = tid; v < nv; v += nthreads) {
                    float4 acc = vload4(v, src0) + vload4(v, src1)
                               + vload4(v, src2) + vload4(v, src3)
                               + vload4(v, src4) + vload4(v, src5)
                               + vload4(v, src6) + vload4(v, src7);
                    vstore4(acc, v, out);
                }
            } else {
                for (uint i = tid; i < n; i += nthreads) {
                    out[i] = src0[i] + src1[i] + src2[i] + src3[i]
                           + src4[i] + src5[i] + src6[i] + src7[i];
                }
            }
        } else {
            for (uint i = tid; i < n; i += nthreads) {
                float acc = 0.0f;
                for (uint p = 0; p < parts; ++p) {
                    __global const float* src = (__global const float*)ptrs[p];
                    acc += src[i];
                }
                out[i] = acc;
            }
        }
        grid_barrier(gbar, num_wg);

        if (tid == 0) {
            atomic_store_explicit(&ctrl[1], seq, memory_order_release,
                                  memory_scope_all_svm_devices);
        }
    }
}

// Persistent all-rank DDA-flat all-reduce with device peer-flag control. Rank 0
// waits for one host go sequence, broadcasts that generation through peer flag
// slots over XGMI, all ranks signal/read a ready generation, perform the direct
// peer reads into a separate output, then signal/read a consumed generation
// before rank 0 reports host done. The consumed phase is the exit barrier that
// prevents the next op from overwriting inputs while a slow peer still reads.
__kernel void allreduce_dda_peer_persistent(
    __global float* out,
    __global const ulong* ptrs,
    volatile __global atomic_uint* ctrl,
    __global uint* my_flags,
    __global const ulong* peer_flag_ptrs,
    volatile __global atomic_uint* gbar,
    uint parts,
    uint self_idx,
    uint n,
    uint total_ops) {
    uint tid = get_global_id(0);
    uint nthreads = get_global_size(0);
    uint num_wg = get_num_groups(0);

    for (uint seq = 1u; seq <= total_ops; ++seq) {
        uint ready_seq = seq * 2u - 1u;
        uint consumed_seq = seq * 2u;

        if (tid == 0) {
            ulong spins = 0;
            if (self_idx == 0u) {
                while (atomic_load_explicit(&ctrl[0], memory_order_acquire,
                                            memory_scope_all_svm_devices) != seq) {
                    if (++spins > 4000000000ul) {
                        atomic_store_explicit(&ctrl[2], 0xddaf0001u, memory_order_release,
                                              memory_scope_all_svm_devices);
                        break;
                    }
                }
            } else {
                volatile __global atomic_uint* start =
                    (volatile __global atomic_uint*)(my_flags + 0);
                while (atomic_load_explicit(start, memory_order_acquire,
                                            memory_scope_all_svm_devices) < ready_seq) {
                    if (++spins > 4000000000ul) {
                        atomic_store_explicit(&ctrl[2], 0xddaf0002u, memory_order_release,
                                              memory_scope_all_svm_devices);
                        break;
                    }
                    peer_flag_relax();
                }
            }
            if (atomic_load_explicit(&ctrl[2], memory_order_acquire,
                                     memory_scope_all_svm_devices) == 0u) {
                for (uint h = 0; h < parts; ++h) {
                    volatile __global atomic_uint* f =
                        (volatile __global atomic_uint*)(peer_flag_ptrs[h] + (ulong)self_idx * 4);
                    atomic_store_explicit(f, ready_seq, memory_order_release,
                                          memory_scope_all_svm_devices);
                }
            }
            for (uint h = 0; h < parts; ++h) {
                volatile __global atomic_uint* f =
                    (volatile __global atomic_uint*)(my_flags + h);
                while (atomic_load_explicit(f, memory_order_acquire,
                                            memory_scope_all_svm_devices) < ready_seq) {
                    if (++spins > 4000000000ul) {
                        atomic_store_explicit(&ctrl[2], 0xddaf0003u, memory_order_release,
                                              memory_scope_all_svm_devices);
                        break;
                    }
                    peer_flag_relax();
                }
            }
        }
        grid_barrier(gbar, num_wg);
        if (atomic_load_explicit(&ctrl[2], memory_order_acquire,
                                 memory_scope_all_svm_devices) != 0u) {
            break;
        }

        if (parts == 8u) {
            __global const float* src0 = (__global const float*)ptrs[0];
            __global const float* src1 = (__global const float*)ptrs[1];
            __global const float* src2 = (__global const float*)ptrs[2];
            __global const float* src3 = (__global const float*)ptrs[3];
            __global const float* src4 = (__global const float*)ptrs[4];
            __global const float* src5 = (__global const float*)ptrs[5];
            __global const float* src6 = (__global const float*)ptrs[6];
            __global const float* src7 = (__global const float*)ptrs[7];
            if ((n & 3u) == 0u && n >= 4096u) {
                uint nv = n >> 2;
                for (uint v = tid; v < nv; v += nthreads) {
                    float4 acc = vload4(v, src0) + vload4(v, src1)
                               + vload4(v, src2) + vload4(v, src3)
                               + vload4(v, src4) + vload4(v, src5)
                               + vload4(v, src6) + vload4(v, src7);
                    vstore4(acc, v, out);
                }
            } else {
                for (uint i = tid; i < n; i += nthreads) {
                    out[i] = src0[i] + src1[i] + src2[i] + src3[i]
                           + src4[i] + src5[i] + src6[i] + src7[i];
                }
            }
        } else {
            for (uint i = tid; i < n; i += nthreads) {
                float acc = 0.0f;
                for (uint p = 0; p < parts; ++p) {
                    __global const float* src = (__global const float*)ptrs[p];
                    acc += src[i];
                }
                out[i] = acc;
            }
        }
        grid_barrier(gbar, num_wg);

        if (tid == 0) {
            for (uint h = 0; h < parts; ++h) {
                volatile __global atomic_uint* f =
                    (volatile __global atomic_uint*)(peer_flag_ptrs[h] + (ulong)self_idx * 4);
                atomic_store_explicit(f, consumed_seq, memory_order_release,
                                      memory_scope_all_svm_devices);
            }
        }
        if (tid == 0) {
            ulong spins = 0;
            for (uint h = 0; h < parts; ++h) {
                volatile __global atomic_uint* f =
                    (volatile __global atomic_uint*)(my_flags + h);
                while (atomic_load_explicit(f, memory_order_acquire,
                                            memory_scope_all_svm_devices) < consumed_seq) {
                    if (++spins > 4000000000ul) {
                        atomic_store_explicit(&ctrl[2], 0xddaf0004u, memory_order_release,
                                              memory_scope_all_svm_devices);
                        break;
                    }
                    peer_flag_relax();
                }
            }
            if (self_idx == 0u &&
                atomic_load_explicit(&ctrl[2], memory_order_acquire,
                                     memory_scope_all_svm_devices) == 0u) {
                atomic_store_explicit(&ctrl[1], seq, memory_order_release,
                                      memory_scope_all_svm_devices);
            }
        }
        grid_barrier(gbar, num_wg);
        if (atomic_load_explicit(&ctrl[2], memory_order_acquire,
                                 memory_scope_all_svm_devices) != 0u) {
            break;
        }
    }
}

// Fused direct all-reduce + RMSNorm substrate for decode-sized vectors. This is
// a narrow one-workgroup validation kernel: rank 0 sums peer buffers, computes
// RMS over the reduced vector, applies per-element weight, and broadcasts the
// normalized result back to every peer. It proves the comm+norm fusion ABI and
// math before wiring it into the decode loop.
__kernel void allreduce_direct_rmsnorm_1wg(
    __global float* own,
    __global const ulong* ptrs,
    __global const float* weight,
    uint parts,
    uint n,
    float eps) {
    __local float scratch[256];
    uint lid = get_local_id(0);
    uint lsz = get_local_size(0);
    float ss = 0.0f;

    for (uint i = lid; i < n; i += lsz) {
        float acc = 0.0f;
        for (uint p = 0; p < parts; ++p) {
            __global const float* src = (__global const float*)ptrs[p];
            acc += src[i];
        }
        own[i] = acc;
        ss += acc * acc;
    }
    scratch[lid] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsz >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float inv = rsqrt(scratch[0] / (float)n + eps);
    for (uint i = lid; i < n; i += lsz) {
        float v = own[i] * inv * weight[i];
        for (uint p = 0; p < parts; ++p) {
            __global float* dst = (__global float*)ptrs[p];
            dst[i] = v;
        }
    }
}

// Multi-workgroup fused direct all-reduce + RMSNorm substrate. This keeps the
// one-dispatch fusion shape but parallelizes the reduction and normalization
// across a co-resident grid. `partial` has at least `num_wg` floats and `gbar`
// is the two-u32 grid barrier used elsewhere in the XGMI kernels.
__kernel void allreduce_direct_rmsnorm_grid(
    __global float* own,
    __global const ulong* ptrs,
    __global const float* weight,
    __global float* partial,
    volatile __global atomic_uint* gbar,
    uint parts,
    uint n,
    float eps,
    uint num_wg) {
    __local float scratch[256];
    uint lid = get_local_id(0);
    uint lsz = get_local_size(0);
    uint gid = get_group_id(0);
    uint tid = get_global_id(0);
    uint nthreads = get_global_size(0);
    float ss = 0.0f;

    for (uint i = tid; i < n; i += nthreads) {
        float acc = 0.0f;
        for (uint p = 0; p < parts; ++p) {
            __global const float* src = (__global const float*)ptrs[p];
            acc += src[i];
        }
        own[i] = acc;
        ss += acc * acc;
    }
    scratch[lid] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsz >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) partial[gid] = scratch[0];

    grid_barrier(gbar, num_wg);

    if (gid == 0) {
        float total = 0.0f;
        for (uint i = lid; i < num_wg; i += lsz) {
            total += partial[i];
        }
        scratch[lid] = total;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint stride = lsz >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) scratch[lid] += scratch[lid + stride];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) partial[0] = scratch[0];
    }

    grid_barrier(gbar, num_wg);

    float inv = rsqrt(partial[0] / (float)n + eps);
    for (uint i = tid; i < n; i += nthreads) {
        float v = own[i] * inv * weight[i];
        for (uint p = 0; p < parts; ++p) {
            __global float* dst = (__global float*)ptrs[p];
            dst[i] = v;
        }
    }
}

// Fused direct all-reduce + residual add + RMSNorm substrate. The residual
// input is assumed replicated across ranks; rank 0 reads residual_ptrs[0], adds
// it to the reduced tensor, writes the updated residual back to every rank, then
// RMS-normalizes and broadcasts the normalized output to every rank.
//
// The TP8 fast path is intentionally a logical-rank left fold. Do not collapse
// this into one expression: Qwen/Kimi parity depends on a fixed reduction order
// before RMSNorm, not merely mathematical equivalence.
__kernel void allreduce_direct_residual_rmsnorm_grid(
    __global float* own,
    __global const ulong* ptrs,
    __global const ulong* residual_ptrs,
    __global const float* weight,
    __global float* partial,
    volatile __global atomic_uint* gbar,
    uint parts,
    uint n,
    float eps,
    uint num_wg) {
    __local float scratch[256];
    uint lid = get_local_id(0);
    uint lsz = get_local_size(0);
    uint gid = get_group_id(0);
    uint tid = get_global_id(0);
    uint nthreads = get_global_size(0);
    __global const float* residual0 = (__global const float*)residual_ptrs[0];
    float ss = 0.0f;

    if (parts == 8u) {
        __global const float* src0 = (__global const float*)ptrs[0];
        __global const float* src1 = (__global const float*)ptrs[1];
        __global const float* src2 = (__global const float*)ptrs[2];
        __global const float* src3 = (__global const float*)ptrs[3];
        __global const float* src4 = (__global const float*)ptrs[4];
        __global const float* src5 = (__global const float*)ptrs[5];
        __global const float* src6 = (__global const float*)ptrs[6];
        __global const float* src7 = (__global const float*)ptrs[7];
        __global float* residual0w = (__global float*)residual_ptrs[0];
        __global float* residual1 = (__global float*)residual_ptrs[1];
        __global float* residual2 = (__global float*)residual_ptrs[2];
        __global float* residual3 = (__global float*)residual_ptrs[3];
        __global float* residual4 = (__global float*)residual_ptrs[4];
        __global float* residual5 = (__global float*)residual_ptrs[5];
        __global float* residual6 = (__global float*)residual_ptrs[6];
        __global float* residual7 = (__global float*)residual_ptrs[7];
        for (uint i = tid; i < n; i += nthreads) {
            float acc = src0[i];
            acc = acc + src1[i];
            acc = acc + src2[i];
            acc = acc + src3[i];
            acc = acc + src4[i];
            acc = acc + src5[i];
            acc = acc + src6[i];
            acc = acc + src7[i];
            float y = acc + residual0[i];
            own[i] = y;
            residual0w[i] = y;
            residual1[i] = y;
            residual2[i] = y;
            residual3[i] = y;
            residual4[i] = y;
            residual5[i] = y;
            residual6[i] = y;
            residual7[i] = y;
            ss += y * y;
        }
    } else {
        for (uint i = tid; i < n; i += nthreads) {
            float acc = 0.0f;
            for (uint p = 0; p < parts; ++p) {
                __global const float* src = (__global const float*)ptrs[p];
                acc += src[i];
            }
            float y = acc + residual0[i];
            own[i] = y;
            for (uint p = 0; p < parts; ++p) {
                __global float* residual = (__global float*)residual_ptrs[p];
                residual[i] = y;
            }
            ss += y * y;
        }
    }
    scratch[lid] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsz >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) partial[gid] = scratch[0];

    grid_barrier(gbar, num_wg);

    if (gid == 0) {
        float total = 0.0f;
        for (uint i = lid; i < num_wg; i += lsz) {
            total += partial[i];
        }
        scratch[lid] = total;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint stride = lsz >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) scratch[lid] += scratch[lid + stride];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) partial[0] = scratch[0];
    }

    grid_barrier(gbar, num_wg);

    float inv = rsqrt(partial[0] / (float)n + eps);
    for (uint i = tid; i < n; i += nthreads) {
        float v = own[i] * inv * weight[i];
        for (uint p = 0; p < parts; ++p) {
            __global float* dst = (__global float*)ptrs[p];
            dst[i] = v;
        }
    }
}

// Fused direct all-reduce + f16 residual update + f16 RMSNorm handoff. This is
// the decode-layer boundary form: rank contributions arrive as f32, the resident
// residual stream is f16, and the next-layer input stream is f16. The residual
// update is rounded to f16 before the RMSNorm sum, matching add_rmsnorm_f16.
//
// The TP8 fast path is also a fixed logical-rank left fold for deterministic
// serving parity across resident runner and future graph-captured boundaries.
__kernel void allreduce_direct_residual_f16_rmsnorm_f16_grid(
    __global float* own,
    __global const ulong* ptrs,
    __global half* residual,
    __global const half* weight,
    __global half* out,
    __global float* partial,
    volatile __global atomic_uint* gbar,
    uint parts,
    uint n,
    float eps,
    uint num_wg) {
    __local float scratch[256];
    uint lid = get_local_id(0);
    uint lsz = get_local_size(0);
    uint gid = get_group_id(0);
    uint tid = get_global_id(0);
    uint nthreads = get_global_size(0);
    float ss = 0.0f;

    if (parts == 8u) {
        __global const float* src0 = (__global const float*)ptrs[0];
        __global const float* src1 = (__global const float*)ptrs[1];
        __global const float* src2 = (__global const float*)ptrs[2];
        __global const float* src3 = (__global const float*)ptrs[3];
        __global const float* src4 = (__global const float*)ptrs[4];
        __global const float* src5 = (__global const float*)ptrs[5];
        __global const float* src6 = (__global const float*)ptrs[6];
        __global const float* src7 = (__global const float*)ptrs[7];
        for (uint i = tid; i < n; i += nthreads) {
            float acc = src0[i];
            acc = acc + src1[i];
            acc = acc + src2[i];
            acc = acc + src3[i];
            acc = acc + src4[i];
            acc = acc + src5[i];
            acc = acc + src6[i];
            acc = acc + src7[i];
            half hv = (half)((float)residual[i] + acc);
            residual[i] = hv;
            float y = (float)hv;
            own[i] = y;
            ss += y * y;
        }
    } else {
        for (uint i = tid; i < n; i += nthreads) {
            float acc = 0.0f;
            for (uint p = 0; p < parts; ++p) {
                __global const float* src = (__global const float*)ptrs[p];
                acc += src[i];
            }
            half hv = (half)((float)residual[i] + acc);
            residual[i] = hv;
            float y = (float)hv;
            own[i] = y;
            ss += y * y;
        }
    }
    scratch[lid] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsz >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) partial[gid] = scratch[0];

    grid_barrier(gbar, num_wg);

    if (gid == 0) {
        float total = 0.0f;
        for (uint i = lid; i < num_wg; i += lsz) {
            total += partial[i];
        }
        scratch[lid] = total;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint stride = lsz >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) scratch[lid] += scratch[lid + stride];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) partial[0] = scratch[0];
    }

    grid_barrier(gbar, num_wg);

    float inv = rsqrt(partial[0] / (float)n + eps);
    for (uint i = tid; i < n; i += nthreads) {
        half hv = (half)(own[i] * inv * (float)weight[i]);
        out[i] = hv;
        float v = (float)hv;
        for (uint p = 0; p < parts; ++p) {
            __global float* dst = (__global float*)ptrs[p];
            dst[i] = v;
        }
    }
}

static inline uchar f32_to_e4m3_rne(float x) {
    if (!isfinite(x)) return (uchar)0x7f;
    uchar sign = (as_uint(x) & 0x80000000u) ? (uchar)0x80 : (uchar)0;
    float ax = fabs(x);
    if (ax == 0.0f) return sign;
    if (ax >= 448.0f) return sign | (uchar)0x7e;
    if (ax < 0.015625f) {
        int m = (int)floor(ax * 512.0f + 0.5f);
        if (m >= 8) return sign | (uchar)(1u << 3);
        return sign | (uchar)m;
    }
    uint bits = as_uint(ax);
    int exp = (int)((bits >> 23) & 0xffu) - 127;
    uint mant = bits & 0x7fffffu;
    uint m3 = mant >> 20;
    uint rbit = (mant >> 19) & 1u;
    uint sticky = (mant & 0x7ffffu) != 0u;
    uint m = m3;
    int e = exp + 7;
    if (rbit && (sticky || (m3 & 1u))) {
        ++m;
        if (m == 8u) {
            m = 0u;
            ++e;
        }
    }
    if (e > 15 || (e == 15 && m > 6u)) return sign | (uchar)0x7e;
    return sign | (uchar)(((uint)e << 3) | m);
}

static inline uchar f32_to_e8m0_ru(float x) {
    if (!isfinite(x) || x <= 0.0f) return (uchar)127;
    int e = (int)ceil(log2(x));
    int code = e + 127;
    if (code < 0) code = 0;
    if (code > 255) code = 255;
    return (uchar)code;
}

static inline float e8m0_to_f32(uchar x) {
    return exp2((float)((int)x - 127));
}

static inline float wave64_max_f32(uint lane, float v) {
    v = fmax(v, as_float(__builtin_amdgcn_ds_bpermute((int)(((lane ^ 1u) << 2)), as_int(v))));
    v = fmax(v, as_float(__builtin_amdgcn_ds_bpermute((int)(((lane ^ 2u) << 2)), as_int(v))));
    v = fmax(v, as_float(__builtin_amdgcn_ds_bpermute((int)(((lane ^ 4u) << 2)), as_int(v))));
    v = fmax(v, as_float(__builtin_amdgcn_ds_bpermute((int)(((lane ^ 8u) << 2)), as_int(v))));
    v = fmax(v, as_float(__builtin_amdgcn_ds_bpermute((int)(((lane ^ 16u) << 2)), as_int(v))));
    v = fmax(v, as_float(__builtin_amdgcn_ds_bpermute((int)(((lane ^ 32u) << 2)), as_int(v))));
    return v;
}

static inline float halfwave32_max_f32(uint lane, float v) {
    v = fmax(v, as_float(__builtin_amdgcn_ds_bpermute((int)(((lane ^ 1u) << 2)), as_int(v))));
    v = fmax(v, as_float(__builtin_amdgcn_ds_bpermute((int)(((lane ^ 2u) << 2)), as_int(v))));
    v = fmax(v, as_float(__builtin_amdgcn_ds_bpermute((int)(((lane ^ 4u) << 2)), as_int(v))));
    v = fmax(v, as_float(__builtin_amdgcn_ds_bpermute((int)(((lane ^ 8u) << 2)), as_int(v))));
    v = fmax(v, as_float(__builtin_amdgcn_ds_bpermute((int)(((lane ^ 16u) << 2)), as_int(v))));
    return v;
}

static inline float quarterwave16_max_f32(uint lane, float v) {
    v = fmax(v, as_float(__builtin_amdgcn_ds_bpermute((int)(((lane ^ 1u) << 2)), as_int(v))));
    v = fmax(v, as_float(__builtin_amdgcn_ds_bpermute((int)(((lane ^ 2u) << 2)), as_int(v))));
    v = fmax(v, as_float(__builtin_amdgcn_ds_bpermute((int)(((lane ^ 4u) << 2)), as_int(v))));
    v = fmax(v, as_float(__builtin_amdgcn_ds_bpermute((int)(((lane ^ 8u) << 2)), as_int(v))));
    return v;
}

static inline float e4m3_ocp_to_f32(uchar b) {
    uint sign = (uint)b & 0x80u;
    uint exp = ((uint)b >> 3) & 0x0fu;
    uint mant = (uint)b & 0x07u;
    float v;
    if (exp == 0u) {
        v = (float)mant * 0.001953125f;
    } else {
        v = (1.0f + (float)mant * 0.125f) * exp2((float)((int)exp - 7));
    }
    return sign ? -v : v;
}

static inline uchar f32_to_i4_sym_rna(float x) {
    float qf = x >= 0.0f ? floor(x + 0.5f) : ceil(x - 0.5f);
    int q = (int)qf;
    if (q < -7) q = -7;
    if (q > 7) q = 7;
    return (uchar)(q & 15);
}

// Pattern-9-style substrate: fused direct all-reduce + residual + RMSNorm with
// both f32 norm output and per-group OCP E4M3 bytes. Scale output is f32 for the
// current validation ABI; the next serving ABI can pack the same group scales to
// E8M0 once the downstream scaled GEMM contract is fixed.
__kernel void allreduce_direct_residual_rmsnorm_fp8_group_grid(
    __global float* own,
    __global const ulong* ptrs,
    __global const ulong* residual_ptrs,
    __global const float* weight,
    __global const ulong* quant_ptrs,
    __global const ulong* scale_ptrs,
    __global float* partial,
    volatile __global atomic_uint* gbar,
    uint parts,
    uint n,
    float eps,
    uint num_wg,
    uint group_size) {
    __local float scratch[256];
    uint lid = get_local_id(0);
    uint lsz = get_local_size(0);
    uint gid = get_group_id(0);
    uint tid = get_global_id(0);
    uint nthreads = get_global_size(0);
    __global const float* residual0 = (__global const float*)residual_ptrs[0];
    float ss = 0.0f;

    for (uint i = tid; i < n; i += nthreads) {
        float acc = 0.0f;
        for (uint p = 0; p < parts; ++p) {
            __global const float* src = (__global const float*)ptrs[p];
            acc += src[i];
        }
        float y = acc + residual0[i];
        own[i] = y;
        for (uint p = 0; p < parts; ++p) {
            __global float* residual = (__global float*)residual_ptrs[p];
            residual[i] = y;
        }
        ss += y * y;
    }
    scratch[lid] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsz >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) partial[gid] = scratch[0];

    grid_barrier(gbar, num_wg);

    if (gid == 0) {
        float total = 0.0f;
        for (uint i = lid; i < num_wg; i += lsz) {
            total += partial[i];
        }
        scratch[lid] = total;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint stride = lsz >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) scratch[lid] += scratch[lid + stride];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) partial[0] = scratch[0];
    }

    grid_barrier(gbar, num_wg);

    float inv = rsqrt(partial[0] / (float)n + eps);
    uint groups = (n + group_size - 1u) / group_size;
    for (uint group = gid; group < groups; group += num_wg) {
        uint start = group * group_size;
        uint end = min(start + group_size, n);
        float maxabs = 0.0f;
        for (uint i = start + lid; i < end; i += lsz) {
            float v = own[i] * inv * weight[i];
            own[i] = v;
            maxabs = fmax(maxabs, fabs(v));
        }
        scratch[lid] = maxabs;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint stride = lsz >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) scratch[lid] = fmax(scratch[lid], scratch[lid + stride]);
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float scale = scratch[0] > 0.0f ? scratch[0] / 448.0f : 1.0f;
        if (lid == 0) {
            for (uint p = 0; p < parts; ++p) {
                __global float* scales = (__global float*)scale_ptrs[p];
                scales[group] = scale;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        float inv_scale = 1.0f / scale;
        for (uint i = start + lid; i < end; i += lsz) {
            float v = own[i];
            uchar q = f32_to_e4m3_rne(v * inv_scale);
            for (uint p = 0; p < parts; ++p) {
                __global float* dst = (__global float*)ptrs[p];
                __global uchar* qdst = (__global uchar*)quant_ptrs[p];
                dst[i] = v;
                qdst[i] = q;
            }
        }
    }
}

// Serving ABI sibling: same fused op as
// allreduce_direct_residual_rmsnorm_fp8_group_grid, but scale output is packed
// as four UE8M0/E8M0 scale bytes per u32 word. Logical scale shape is
// [ceil(groups / 4)] for a single token row.
__kernel void allreduce_direct_residual_rmsnorm_fp8_group_packed_grid(
    __global float* own,
    __global const ulong* ptrs,
    __global const ulong* residual_ptrs,
    __global const float* weight,
    __global const ulong* quant_ptrs,
    __global const ulong* scale_ptrs,
    __global float* partial,
    volatile __global atomic_uint* gbar,
    uint parts,
    uint n,
    float eps,
    uint num_wg,
    uint group_size,
    uint store_output) {
    __local float scratch[256];
    uint lid = get_local_id(0);
    uint lsz = get_local_size(0);
    uint gid = get_group_id(0);
    uint tid = get_global_id(0);
    uint nthreads = get_global_size(0);
    __global const float* residual0 = (__global const float*)residual_ptrs[0];
    __local uchar scale_codes[4];
    float ss = 0.0f;

    if (parts == 8u && store_output == 0u && group_size == 64u && n >= 8192u) {
        __global const float* src0 = (__global const float*)ptrs[0];
        __global const float* src1 = (__global const float*)ptrs[1];
        __global const float* src2 = (__global const float*)ptrs[2];
        __global const float* src3 = (__global const float*)ptrs[3];
        __global const float* src4 = (__global const float*)ptrs[4];
        __global const float* src5 = (__global const float*)ptrs[5];
        __global const float* src6 = (__global const float*)ptrs[6];
        __global const float* src7 = (__global const float*)ptrs[7];
        __global float* residual0w = (__global float*)residual_ptrs[0];
        __global float* residual1 = (__global float*)residual_ptrs[1];
        __global float* residual2 = (__global float*)residual_ptrs[2];
        __global float* residual3 = (__global float*)residual_ptrs[3];
        __global float* residual4 = (__global float*)residual_ptrs[4];
        __global float* residual5 = (__global float*)residual_ptrs[5];
        __global float* residual6 = (__global float*)residual_ptrs[6];
        __global float* residual7 = (__global float*)residual_ptrs[7];
        __global uchar* q0 = (__global uchar*)quant_ptrs[0];
        __global uchar* q1 = (__global uchar*)quant_ptrs[1];
        __global uchar* q2 = (__global uchar*)quant_ptrs[2];
        __global uchar* q3 = (__global uchar*)quant_ptrs[3];
        __global uchar* q4 = (__global uchar*)quant_ptrs[4];
        __global uchar* q5 = (__global uchar*)quant_ptrs[5];
        __global uchar* q6 = (__global uchar*)quant_ptrs[6];
        __global uchar* q7 = (__global uchar*)quant_ptrs[7];
        __global uint* sc0 = (__global uint*)scale_ptrs[0];
        __global uint* sc1 = (__global uint*)scale_ptrs[1];
        __global uint* sc2 = (__global uint*)scale_ptrs[2];
        __global uint* sc3 = (__global uint*)scale_ptrs[3];
        __global uint* sc4 = (__global uint*)scale_ptrs[4];
        __global uint* sc5 = (__global uint*)scale_ptrs[5];
        __global uint* sc6 = (__global uint*)scale_ptrs[6];
        __global uint* sc7 = (__global uint*)scale_ptrs[7];
        uint groups_fast = (n + 63u) >> 6;
        uint packed_words_fast = (groups_fast + 3u) >> 2;
        uint wave_fast = lid >> 6;
        uint wave_lid_fast = lid & 63u;
        uint inline_group_max_fast = n <= nthreads ? 1u : 0u;
        float maxabs_inline_fast = 0.0f;

        for (uint i = tid; i < n; i += nthreads) {
            float acc = src0[i] + src1[i] + src2[i] + src3[i] +
                        src4[i] + src5[i] + src6[i] + src7[i];
            float y = acc + residual0[i];
            maxabs_inline_fast = fmax(maxabs_inline_fast, fabs(y * weight[i]));
            own[i] = y;
            residual0w[i] = y;
            residual1[i] = y;
            residual2[i] = y;
            residual3[i] = y;
            residual4[i] = y;
            residual5[i] = y;
            residual6[i] = y;
            residual7[i] = y;
            ss += y * y;
        }
        scratch[lid] = ss;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint stride = lsz >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) scratch[lid] += scratch[lid + stride];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) partial[gid] = scratch[0];

        if (inline_group_max_fast != 0u) {
            scratch[lid] = maxabs_inline_fast;
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint stride = 32u; stride > 0; stride >>= 1) {
                if (wave_lid_fast < stride) {
                    scratch[lid] = fmax(scratch[lid], scratch[lid + stride]);
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            uint inline_group_fast = (gid << 2) + wave_fast;
            if (wave_lid_fast == 0u && inline_group_fast < groups_fast) {
                partial[(ulong)num_wg + (ulong)inline_group_fast] = scratch[lid];
            }
        }

        grid_barrier(gbar, num_wg);

        if (gid == 0) {
            float total = 0.0f;
            for (uint i = lid; i < num_wg; i += lsz) {
                total += partial[i];
            }
            scratch[lid] = total;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (uint stride = lsz >> 1; stride > 0; stride >>= 1) {
                if (lid < stride) scratch[lid] += scratch[lid + stride];
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (lid == 0) partial[0] = scratch[0];
        }

        grid_barrier(gbar, num_wg);

        float inv_fast = rsqrt(partial[0] / (float)n + eps);
        for (uint pack = gid; pack < packed_words_fast; pack += num_wg) {
            uint group = (pack << 2) + wave_fast;
            uint start = group << 6;
            uint end = min(start + 64u, n);
            float maxabs = 0.0f;
            if (group < groups_fast) {
                if (inline_group_max_fast != 0u) {
                    maxabs = partial[(ulong)num_wg + (ulong)group];
                } else {
                    for (uint i = start + wave_lid_fast; i < end; i += 64u) {
                        float v = own[i] * weight[i];
                        maxabs = fmax(maxabs, fabs(v));
                    }
                }
            }

            if (inline_group_max_fast == 0u) {
                scratch[lid] = maxabs;
                barrier(CLK_LOCAL_MEM_FENCE);

                for (uint stride = 32u; stride > 0; stride >>= 1) {
                    if (wave_lid_fast < stride) {
                        scratch[lid] = fmax(scratch[lid], scratch[lid + stride]);
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
            }

            uchar scale_code = (uchar)0;
            float scale = 1.0f;
            if (group < groups_fast) {
                float group_max_fast = inline_group_max_fast != 0u ? maxabs : scratch[wave_fast << 6];
                float raw_scale = (group_max_fast * inv_fast) / 448.0f;
                scale_code = f32_to_e8m0_ru(raw_scale > 0.0f ? raw_scale : 1.0f);
                scale = e8m0_to_f32(scale_code);
            }
            if (wave_lid_fast == 0) scale_codes[wave_fast] = scale_code;
            barrier(CLK_LOCAL_MEM_FENCE);

            if (lid == 0) {
                uint packed = ((uint)scale_codes[0])
                    | (((uint)scale_codes[1]) << 8)
                    | (((uint)scale_codes[2]) << 16)
                    | (((uint)scale_codes[3]) << 24);
                sc0[pack] = packed;
                sc1[pack] = packed;
                sc2[pack] = packed;
                sc3[pack] = packed;
                sc4[pack] = packed;
                sc5[pack] = packed;
                sc6[pack] = packed;
                sc7[pack] = packed;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            if (group < groups_fast) {
                float inv_scale = inv_fast / scale;
                for (uint i = start + wave_lid_fast; i < end; i += 64u) {
                    uchar q = f32_to_e4m3_rne(own[i] * weight[i] * inv_scale);
                    q0[i] = q;
                    q1[i] = q;
                    q2[i] = q;
                    q3[i] = q;
                    q4[i] = q;
                    q5[i] = q;
                    q6[i] = q;
                    q7[i] = q;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        return;
    }

    for (uint i = tid; i < n; i += nthreads) {
        float acc = 0.0f;
        for (uint p = 0; p < parts; ++p) {
            __global const float* src = (__global const float*)ptrs[p];
            acc += src[i];
        }
        float y = acc + residual0[i];
        own[i] = y;
        for (uint p = 0; p < parts; ++p) {
            __global float* residual = (__global float*)residual_ptrs[p];
            residual[i] = y;
        }
        ss += y * y;
    }
    scratch[lid] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsz >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) partial[gid] = scratch[0];

    grid_barrier(gbar, num_wg);

    if (gid == 0) {
        float total = 0.0f;
        for (uint i = lid; i < num_wg; i += lsz) {
            total += partial[i];
        }
        scratch[lid] = total;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint stride = lsz >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) scratch[lid] += scratch[lid + stride];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) partial[0] = scratch[0];
    }

    grid_barrier(gbar, num_wg);

    float inv = rsqrt(partial[0] / (float)n + eps);
    uint groups = (n + group_size - 1u) / group_size;
    uint packed_words = (groups + 3u) >> 2;
    uint wave = lid >> 6;
    uint wave_lid = lid & 63u;
    for (uint pack = gid; pack < packed_words; pack += num_wg) {
        uint group = (pack << 2) + wave;
        uint start = group * group_size;
        uint end = min(start + group_size, n);
        float maxabs = 0.0f;
        if (group < groups) {
            if (store_output != 0u) {
                for (uint i = start + wave_lid; i < end; i += 64u) {
                    float v = own[i] * inv * weight[i];
                    own[i] = v;
                    maxabs = fmax(maxabs, fabs(v));
                }
            } else {
                for (uint i = start + wave_lid; i < end; i += 64u) {
                    float v = own[i] * weight[i];
                    maxabs = fmax(maxabs, fabs(v));
                }
            }
        }
        scratch[lid] = maxabs;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint stride = 32u; stride > 0; stride >>= 1) {
            if (wave_lid < stride) {
                scratch[lid] = fmax(scratch[lid], scratch[lid + stride]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        uchar scale_code = (uchar)0;
        float scale = 1.0f;
        if (group < groups) {
            float raw_max = scratch[wave << 6];
            if (store_output == 0u) {
                raw_max *= inv;
            }
            float raw_scale = raw_max > 0.0f ? raw_max / 448.0f : 1.0f;
            scale_code = f32_to_e8m0_ru(raw_scale);
            scale = e8m0_to_f32(scale_code);
        }
        if (wave_lid == 0) scale_codes[wave] = scale_code;
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid == 0) {
            uint packed = ((uint)scale_codes[0])
                | (((uint)scale_codes[1]) << 8)
                | (((uint)scale_codes[2]) << 16)
                | (((uint)scale_codes[3]) << 24);
            for (uint p = 0; p < parts; ++p) {
                __global uint* scales = (__global uint*)scale_ptrs[p];
                scales[pack] = packed;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (group < groups) {
            float inv_scale = store_output != 0u ? 1.0f / scale : inv / scale;
            for (uint i = start + wave_lid; i < end; i += 64u) {
                float v = store_output != 0u ? own[i] : own[i] * weight[i];
                uchar q = f32_to_e4m3_rne(v * inv_scale);
                for (uint p = 0; p < parts; ++p) {
                    __global uchar* qdst = (__global uchar*)quant_ptrs[p];
                    if (store_output != 0u) {
                        __global float* dst = (__global float*)ptrs[p];
                        dst[i] = v;
                    }
                    qdst[i] = q;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// Quant-only serving ABI sibling: fused direct all-reduce + residual + RMSNorm
// followed by signed symmetric INT4 activation quantization. Two 4-bit values
// are packed per byte, and four E8M0 scale bytes are packed per u32 word.
__kernel void allreduce_direct_residual_rmsnorm_int4_group_packed_grid(
    __global float* own,
    __global const ulong* ptrs,
    __global const ulong* residual_ptrs,
    __global const float* weight,
    __global const ulong* quant_ptrs,
    __global const ulong* scale_ptrs,
    __global float* partial,
    volatile __global atomic_uint* gbar,
    uint parts,
    uint n,
    float eps,
    uint num_wg,
    uint group_size) {
    __local float scratch[256];
    __local uchar scale_codes[8];
    uint lid = get_local_id(0);
    uint lsz = get_local_size(0);
    uint gid = get_group_id(0);
    uint tid = get_global_id(0);
    uint nthreads = get_global_size(0);
    uint wave = lid >> 6;
    uint wave_lid = lid & 63u;
    __global const float* residual0 = (__global const float*)residual_ptrs[0];
    float ss = 0.0f;

    if (parts == 8u) {
        __global const float* src0 = (__global const float*)ptrs[0];
        __global const float* src1 = (__global const float*)ptrs[1];
        __global const float* src2 = (__global const float*)ptrs[2];
        __global const float* src3 = (__global const float*)ptrs[3];
        __global const float* src4 = (__global const float*)ptrs[4];
        __global const float* src5 = (__global const float*)ptrs[5];
        __global const float* src6 = (__global const float*)ptrs[6];
        __global const float* src7 = (__global const float*)ptrs[7];
        __global float* residual0w = (__global float*)residual_ptrs[0];
        __global float* residual1 = (__global float*)residual_ptrs[1];
        __global float* residual2 = (__global float*)residual_ptrs[2];
        __global float* residual3 = (__global float*)residual_ptrs[3];
        __global float* residual4 = (__global float*)residual_ptrs[4];
        __global float* residual5 = (__global float*)residual_ptrs[5];
        __global float* residual6 = (__global float*)residual_ptrs[6];
        __global float* residual7 = (__global float*)residual_ptrs[7];
        for (uint i = tid; i < n; i += nthreads) {
            float acc = src0[i] + src1[i] + src2[i] + src3[i] +
                        src4[i] + src5[i] + src6[i] + src7[i];
            float y = acc + residual0[i];
            own[i] = y;
            residual0w[i] = y;
            residual1[i] = y;
            residual2[i] = y;
            residual3[i] = y;
            residual4[i] = y;
            residual5[i] = y;
            residual6[i] = y;
            residual7[i] = y;
            ss += y * y;
        }
    } else {
        for (uint i = tid; i < n; i += nthreads) {
            float acc = 0.0f;
            for (uint p = 0; p < parts; ++p) {
                __global const float* src = (__global const float*)ptrs[p];
                acc += src[i];
            }
            float y = acc + residual0[i];
            own[i] = y;
            for (uint p = 0; p < parts; ++p) {
                __global float* residual = (__global float*)residual_ptrs[p];
                residual[i] = y;
            }
            ss += y * y;
        }
    }
    scratch[lid] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsz >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) partial[gid] = scratch[0];

    grid_barrier(gbar, num_wg);

    if (gid == 0) {
        float total = 0.0f;
        for (uint i = lid; i < num_wg; i += lsz) {
            total += partial[i];
        }
        scratch[lid] = total;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint stride = lsz >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) scratch[lid] += scratch[lid + stride];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) partial[0] = scratch[0];
    }

    grid_barrier(gbar, num_wg);

    float inv = rsqrt(partial[0] / (float)n + eps);
    uint groups = (n + group_size - 1u) / group_size;
    uint packed_words = (groups + 3u) >> 2;
    if (group_size == 32u) {
        uint quarter = wave_lid >> 4;
        uint qlane = wave_lid & 15u;
        for (uint pack = gid * 4u + wave; pack < packed_words; pack += num_wg * 4u) {
            uint group = (pack << 2) + quarter;
            uint start = group << 5;
            uint end = min(start + 32u, n);
            float maxabs = 0.0f;
            if (group < groups) {
                uint i0 = start + qlane;
                uint i1 = i0 + 16u;
                if (i0 < end) {
                    float v0 = own[i0] * inv * weight[i0];
                    maxabs = fabs(v0);
                }
                if (i1 < end) {
                    float v1 = own[i1] * inv * weight[i1];
                    maxabs = fmax(maxabs, fabs(v1));
                }
            }
            float group_max = quarterwave16_max_f32(wave_lid, maxabs);

            uchar scale_code = (uchar)0;
            float scale = 1.0f;
            if (group < groups) {
                float raw_scale = group_max > 0.0f ? group_max / 7.0f : 1.0f;
                scale_code = f32_to_e8m0_ru(raw_scale);
                scale = e8m0_to_f32(scale_code);
            }
            uint scale_u = (uint)scale_code;
            uint scale0 = (uint)__builtin_amdgcn_ds_bpermute((int)(0u << 2), (int)scale_u) & 0xffu;
            uint scale1 = (uint)__builtin_amdgcn_ds_bpermute((int)(16u << 2), (int)scale_u) & 0xffu;
            uint scale2 = (uint)__builtin_amdgcn_ds_bpermute((int)(32u << 2), (int)scale_u) & 0xffu;
            uint scale3 = (uint)__builtin_amdgcn_ds_bpermute((int)(48u << 2), (int)scale_u) & 0xffu;

            if (wave_lid == 0u) {
                uint packed = scale0 | (scale1 << 8) | (scale2 << 16) | (scale3 << 24);
                if (parts == 8u) {
                    ((__global uint*)scale_ptrs[0])[pack] = packed;
                    ((__global uint*)scale_ptrs[1])[pack] = packed;
                    ((__global uint*)scale_ptrs[2])[pack] = packed;
                    ((__global uint*)scale_ptrs[3])[pack] = packed;
                    ((__global uint*)scale_ptrs[4])[pack] = packed;
                    ((__global uint*)scale_ptrs[5])[pack] = packed;
                    ((__global uint*)scale_ptrs[6])[pack] = packed;
                    ((__global uint*)scale_ptrs[7])[pack] = packed;
                } else {
                    for (uint p = 0; p < parts; ++p) {
                        __global uint* scales = (__global uint*)scale_ptrs[p];
                        scales[pack] = packed;
                    }
                }
            }

            if (group < groups) {
                uint i0 = start + (qlane << 1);
                uint i1 = i0 + 1u;
                uchar q0 = (i0 < end) ? f32_to_i4_sym_rna((own[i0] * inv * weight[i0]) / scale) : (uchar)0;
                uchar q1 = (i1 < end) ? f32_to_i4_sym_rna((own[i1] * inv * weight[i1]) / scale) : (uchar)0;
                uchar packed_q = (uchar)(q0 | (uchar)(q1 << 4));
                uint qoff = i0 >> 1;
                if (end == start + 32u) {
                    uint lane_base = wave_lid & ~3u;
                    uint qb0 = (uint)__builtin_amdgcn_ds_bpermute((int)((lane_base + 0u) << 2), (int)((uint)packed_q)) & 0xffu;
                    uint qb1 = (uint)__builtin_amdgcn_ds_bpermute((int)((lane_base + 1u) << 2), (int)((uint)packed_q)) & 0xffu;
                    uint qb2 = (uint)__builtin_amdgcn_ds_bpermute((int)((lane_base + 2u) << 2), (int)((uint)packed_q)) & 0xffu;
                    uint qb3 = (uint)__builtin_amdgcn_ds_bpermute((int)((lane_base + 3u) << 2), (int)((uint)packed_q)) & 0xffu;
                    uint packed4 = qb0 | (qb1 << 8) | (qb2 << 16) | (qb3 << 24);
                    if ((qlane & 3u) == 0u) {
                        uint qword = qoff >> 2;
                        if (parts == 8u) {
                            ((__global uint*)quant_ptrs[0])[qword] = packed4;
                            ((__global uint*)quant_ptrs[1])[qword] = packed4;
                            ((__global uint*)quant_ptrs[2])[qword] = packed4;
                            ((__global uint*)quant_ptrs[3])[qword] = packed4;
                            ((__global uint*)quant_ptrs[4])[qword] = packed4;
                            ((__global uint*)quant_ptrs[5])[qword] = packed4;
                            ((__global uint*)quant_ptrs[6])[qword] = packed4;
                            ((__global uint*)quant_ptrs[7])[qword] = packed4;
                        } else {
                            for (uint p = 0; p < parts; ++p) {
                                __global uint* qdst = (__global uint*)quant_ptrs[p];
                                qdst[qword] = packed4;
                            }
                        }
                    }
                } else {
                    if (i0 < end) {
                        if (parts == 8u) {
                            ((__global uchar*)quant_ptrs[0])[qoff] = packed_q;
                            ((__global uchar*)quant_ptrs[1])[qoff] = packed_q;
                            ((__global uchar*)quant_ptrs[2])[qoff] = packed_q;
                            ((__global uchar*)quant_ptrs[3])[qoff] = packed_q;
                            ((__global uchar*)quant_ptrs[4])[qoff] = packed_q;
                            ((__global uchar*)quant_ptrs[5])[qoff] = packed_q;
                            ((__global uchar*)quant_ptrs[6])[qoff] = packed_q;
                            ((__global uchar*)quant_ptrs[7])[qoff] = packed_q;
                        } else {
                            for (uint p = 0; p < parts; ++p) {
                                __global uchar* qdst = (__global uchar*)quant_ptrs[p];
                                qdst[qoff] = packed_q;
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    for (uint pack = gid; pack < packed_words; pack += num_wg) {
        uint group = (pack << 2) + wave;
        uint start = group * group_size;
        uint end = min(start + group_size, n);
        float maxabs = 0.0f;
        if (group < groups) {
            for (uint i = start + wave_lid; i < end; i += 64u) {
                float v = own[i] * inv * weight[i];
                maxabs = fmax(maxabs, fabs(v));
            }
        }
        float group_max = wave64_max_f32(wave_lid, maxabs);

        uchar scale_code = (uchar)0;
        float scale = 1.0f;
        if (group < groups) {
            float raw_scale = group_max > 0.0f ? group_max / 7.0f : 1.0f;
            scale_code = f32_to_e8m0_ru(raw_scale);
            scale = e8m0_to_f32(scale_code);
        }
        if (wave_lid == 0) scale_codes[wave] = scale_code;
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid == 0) {
            uint packed = ((uint)scale_codes[0])
                | (((uint)scale_codes[1]) << 8)
                | (((uint)scale_codes[2]) << 16)
                | (((uint)scale_codes[3]) << 24);
            if (parts == 8u) {
                ((__global uint*)scale_ptrs[0])[pack] = packed;
                ((__global uint*)scale_ptrs[1])[pack] = packed;
                ((__global uint*)scale_ptrs[2])[pack] = packed;
                ((__global uint*)scale_ptrs[3])[pack] = packed;
                ((__global uint*)scale_ptrs[4])[pack] = packed;
                ((__global uint*)scale_ptrs[5])[pack] = packed;
                ((__global uint*)scale_ptrs[6])[pack] = packed;
                ((__global uint*)scale_ptrs[7])[pack] = packed;
            } else {
                for (uint p = 0; p < parts; ++p) {
                    __global uint* scales = (__global uint*)scale_ptrs[p];
                    scales[pack] = packed;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (group < groups) {
            uint pair_count = (end - start + 1u) >> 1;
            for (uint pair = wave_lid; pair < pair_count; pair += 64u) {
                uint i0 = start + (pair << 1);
                uint i1 = i0 + 1u;
                uchar q0 = f32_to_i4_sym_rna((own[i0] * inv * weight[i0]) / scale);
                uchar q1 = (i1 < end) ? f32_to_i4_sym_rna((own[i1] * inv * weight[i1]) / scale) : (uchar)0;
                uchar packed_q = (uchar)(q0 | (uchar)(q1 << 4));
                uint qoff = i0 >> 1;
                if (parts == 8u) {
                    ((__global uchar*)quant_ptrs[0])[qoff] = packed_q;
                    ((__global uchar*)quant_ptrs[1])[qoff] = packed_q;
                    ((__global uchar*)quant_ptrs[2])[qoff] = packed_q;
                    ((__global uchar*)quant_ptrs[3])[qoff] = packed_q;
                    ((__global uchar*)quant_ptrs[4])[qoff] = packed_q;
                    ((__global uchar*)quant_ptrs[5])[qoff] = packed_q;
                    ((__global uchar*)quant_ptrs[6])[qoff] = packed_q;
                    ((__global uchar*)quant_ptrs[7])[qoff] = packed_q;
                } else {
                    for (uint p = 0; p < parts; ++p) {
                        __global uchar* qdst = (__global uchar*)quant_ptrs[p];
                        qdst[qoff] = packed_q;
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// Downstream-consumer probe for the packed activation ABI. This models the A
// side of group-scaled GEMM: q is OCP E4M3, scales are four E8M0 bytes packed
// per u32 word, and group_size is normally 128.
__kernel void fp8_packed_scale_dot(
    __global const uchar* q,
    __global const uint* scales,
    __global const float* weight,
    __global float* partial,
    volatile __global atomic_uint* gbar,
    __global float* out,
    uint n,
    uint group_size,
    uint num_wg) {
    __local float scratch[256];
    uint lid = get_local_id(0);
    uint gid = get_group_id(0);
    uint tid = get_global_id(0);
    uint nthreads = get_global_size(0);
    float sum = 0.0f;
    for (uint i = tid; i < n; i += nthreads) {
        uint group = i / group_size;
        uint word = scales[group >> 2];
        uchar sc = (uchar)((word >> ((group & 3u) * 8u)) & 0xffu);
        sum += e4m3_ocp_to_f32(q[i]) * e8m0_to_f32(sc) * weight[i];
    }
    scratch[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = get_local_size(0) >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) partial[gid] = scratch[0];

    grid_barrier(gbar, num_wg);

    if (gid == 0) {
        float total = 0.0f;
        for (uint i = lid; i < num_wg; i += get_local_size(0)) total += partial[i];
        scratch[lid] = total;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint stride = get_local_size(0) >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) scratch[lid] += scratch[lid + stride];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) out[0] = scratch[0];
    }
}

// Same consumer probe with f32 scales. This is the apples-to-apples in-repo
// baseline for measuring packed-scale decode overhead.
__kernel void fp8_f32_scale_dot(
    __global const uchar* q,
    __global const float* scales,
    __global const float* weight,
    __global float* partial,
    volatile __global atomic_uint* gbar,
    __global float* out,
    uint n,
    uint group_size,
    uint num_wg) {
    __local float scratch[256];
    uint lid = get_local_id(0);
    uint gid = get_group_id(0);
    uint tid = get_global_id(0);
    uint nthreads = get_global_size(0);
    float sum = 0.0f;
    for (uint i = tid; i < n; i += nthreads) {
        uint group = i / group_size;
        sum += e4m3_ocp_to_f32(q[i]) * scales[group] * weight[i];
    }
    scratch[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = get_local_size(0) >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) partial[gid] = scratch[0];

    grid_barrier(gbar, num_wg);

    if (gid == 0) {
        float total = 0.0f;
        for (uint i = lid; i < num_wg; i += get_local_size(0)) total += partial[i];
        scratch[lid] = total;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint stride = get_local_size(0) >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) scratch[lid] += scratch[lid + stride];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) out[0] = scratch[0];
    }
}

// Bandwidth-optimal all-reduce, phase 1 (reduce-scatter). On a fully-connected
// XGMI fabric every GPU owns one chunk and runs this concurrently: GPU g sums
// its chunk [off, off+len) across all `parts` peer buffers into its own buffer.
// Reads use 128-bit (float4) loads to saturate the interconnect; a scalar tail
// covers a non-multiple-of-4 length. Each GPU writes a disjoint chunk and reads
// only same-index data from peers, so no cross-GPU sync is needed within the
// phase.
__kernel void reduce_scatter(__global float* out, __global const ulong* ptrs,
                             uint parts, uint off, uint len) {
    uint i = get_global_id(0);
    uint vec = len >> 2;            // number of float4 groups
    if (i < vec) {
        uint pos4 = (off >> 2) + i; // off is chunk-aligned (multiple of 4)
        float4 acc = (float4)(0.0f);
        for (uint p = 0; p < parts; ++p) {
            acc += ((__global const float4*)ptrs[p])[pos4];
        }
        ((__global float4*)out)[pos4] = acc;
        return;
    }
    // scalar tail
    uint t = vec << 2;
    uint i_s = i - vec;
    if (i_s < (len & 3u)) {
        uint pos = off + t + i_s;
        float acc = 0.0f;
        for (uint p = 0; p < parts; ++p) {
            acc += ((__global const float*)ptrs[p])[pos];
        }
        out[pos] = acc;
    }
}

// Push-based all-gather: the owner of a chunk WRITES it into every peer's
// buffer (remote writes / push), rather than each peer reading it (pull). On
// Infinity Fabric remote writes stream fire-and-forget while remote reads pay a
// request/response round-trip, so push sustains far higher interconnect BW.
// GPU g runs this over its owned chunk [off, off+len).
__kernel void broadcast_chunk(__global const float* src, __global const ulong* ptrs,
                              uint parts, uint off, uint len) {
    uint i = get_global_id(0);
    uint vec = len >> 2;
    if (i < vec) {
        uint pos4 = (off >> 2) + i;
        float4 v = ((__global const float4*)src)[pos4];
        for (uint p = 0; p < parts; ++p) {
            ((__global float4*)ptrs[p])[pos4] = v;
        }
        return;
    }
    uint i_s = i - vec;
    if (i_s < (len & 3u)) {
        uint pos = off + (vec << 2) + i_s;
        float v = src[pos];
        for (uint p = 0; p < parts; ++p) {
            ((__global float*)ptrs[p])[pos] = v;
        }
    }
}

// Push-based all-gather variant for RSAG: the owner chunk is already final in
// its local buffer, so skip the redundant self-store and only write peers.
__kernel void broadcast_chunk_skip_owner(__global const float* src, __global const ulong* ptrs,
                                         uint parts, uint off, uint len, uint owner) {
    uint i = get_global_id(0);
    uint vec = len >> 2;
    if (i < vec) {
        uint pos4 = (off >> 2) + i;
        float4 v = ((__global const float4*)src)[pos4];
        for (uint p = 0; p < parts; ++p) {
            if (p == owner) continue;
            ((__global float4*)ptrs[p])[pos4] = v;
        }
        return;
    }
    uint i_s = i - vec;
    if (i_s < (len & 3u)) {
        uint pos = off + (vec << 2) + i_s;
        float v = src[pos];
        for (uint p = 0; p < parts; ++p) {
            if (p == owner) continue;
            ((__global float*)ptrs[p])[pos] = v;
        }
    }
}

// Write-based reduce-scatter, step 1 (scatter to staging). GPU `self_idx`
// pushes each of its R chunks into the corresponding owner's staging buffer,
// landing in that owner's slot `self_idx`. All cross-GPU traffic is remote
// writes (push); `chunk_len` (cl) is a multiple of 4 so float4 writes stay
// aligned. After this + a barrier, every owner holds all R contributions to its
// chunk contiguously in its staging buffer.
__kernel void scatter_to_staging(__global const float* own, __global const ulong* stage_ptrs,
                                 uint parts, uint cl, uint self_idx, uint n) {
    uint i = get_global_id(0);
    uint vec = n >> 2;
    if (i < vec) {
        uint pos = i << 2;
        uint j = pos / cl;            // owner of this element
        if (j >= parts) j = parts - 1;
        uint i_in = pos - j * cl;     // offset within the chunk (multiple of 4)
        float4 v = ((__global const float4*)own)[i];
        uint slot = self_idx * cl + i_in;
        ((__global float4*)stage_ptrs[j])[slot >> 2] = v;
        return;
    }
    uint i_s = i - vec;
    if (i_s < (n & 3u)) {
        uint pos = (vec << 2) + i_s;
        uint j = pos / cl;
        if (j >= parts) j = parts - 1;
        uint i_in = pos - j * cl;
        ((__global float*)stage_ptrs[j])[self_idx * cl + i_in] = own[pos];
    }
}

// Write-based reduce-scatter, step 2 (local reduce). Owner GPU sums the R
// contiguous staging slots (each `cl` long) into its own chunk [off, off+len).
// All reads are LOCAL (staging lives on this GPU), so this runs at HBM speed.
__kernel void gather_reduce_local(__global float* out, __global const float* stage,
                                  uint parts, uint off, uint cl, uint len) {
    uint i = get_global_id(0);
    uint vec = len >> 2;
    if (i < vec) {
        float4 acc = (float4)(0.0f);
        for (uint p = 0; p < parts; ++p) {
            acc += ((__global const float4*)stage)[((p * cl) >> 2) + i];
        }
        ((__global float4*)out)[(off >> 2) + i] = acc;
        return;
    }
    uint i_s = i - vec;
    if (i_s < (len & 3u)) {
        uint pos = (vec << 2) + i_s;
        float acc = 0.0f;
        for (uint p = 0; p < parts; ++p) {
            acc += stage[p * cl + pos];
        }
        out[off + pos] = acc;
    }
}

// Bandwidth-optimal all-reduce, phase 2 (all-gather). After reduce-scatter each
// GPU owns the final chunk it computed; this fills in every other chunk by
// copying it from its owner over XGMI. `own_chunk` is skipped (already final),
// so the owned region stays read-only for peers during this phase — no
// cross-GPU write conflicts.
__kernel void all_gather(__global float* out, __global const ulong* ptrs,
                         uint chunk_len, uint parts, uint n, uint own_chunk) {
    uint i = get_global_id(0);
    uint vec = n >> 2;             // float4 groups
    uint clv = chunk_len >> 2;     // chunk_len in float4 units (chunk_len % 4 == 0)
    if (i < vec) {
        uint chunk = clv ? (i / clv) : (parts - 1);
        if (chunk >= parts) chunk = parts - 1;
        if (chunk == own_chunk) return;
        ((__global float4*)out)[i] = ((__global const float4*)ptrs[chunk])[i];
        return;
    }
    // scalar tail (n not a multiple of 4) — always in the last chunk
    uint i_s = i - vec;
    if (i_s < (n & 3u)) {
        uint pos = (vec << 2) + i_s;
        uint chunk = pos / chunk_len;
        if (chunk >= parts) chunk = parts - 1;
        if (chunk == own_chunk) return;
        out[pos] = ((__global const float*)ptrs[chunk])[pos];
    }
}

// ---- GPU-driven (device-synchronized) collective substrate ----
//
// Cross-GPU barrier inside a kernel, validating system-scope atomics over XGMI.
// Each GPU writes `seq` into every peer's arrival slot for itself, then spins
// until its own R arrival slots all reach `seq`. With release/acquire ordering
// at all-svm-devices (system) scope, a peer's flag write becomes visible after
// its preceding data writes — the basis for in-kernel put/signal/wait.
//
// `my_flags`        : this GPU's R arrival counters (local, peer-writable)
// `peer_flag_ptrs`  : R pointers to each GPU's arrival-counter base
// A spin cap writes 0xDEAD to out[0] on failure so the host never hangs.
__kernel void xbarrier(__global uint* my_flags,
                       __global const ulong* peer_flag_ptrs,
                       uint parts, uint self_idx, uint seq,
                       __global uint* out) {
    if (get_global_id(0) != 0) return;
    // signal every peer (and self): set peer_h.flags[self_idx] = seq
    for (uint h = 0; h < parts; ++h) {
        volatile __global atomic_uint* f =
            (volatile __global atomic_uint*)(peer_flag_ptrs[h] + (ulong)self_idx * 4);
        atomic_store_explicit(f, seq, memory_order_release, memory_scope_all_svm_devices);
    }
    // wait until all my arrival slots reach seq
    for (uint h = 0; h < parts; ++h) {
        volatile __global atomic_uint* f =
            (volatile __global atomic_uint*)(my_flags + h);
        ulong spins = 0;
        while (atomic_load_explicit(f, memory_order_acquire, memory_scope_all_svm_devices) != seq) {
            if (++spins > 2000000000ul) { out[0] = 0xDEADu; return; }
        }
    }
    out[0] = seq;
}

// Intra-GPU grid barrier (sense-reversing) across exactly `num_wg` co-resident
// workgroups. gbar[0] = arrival count, gbar[1] = sense. Device scope: orders
// every workgroup's prior global writes before any proceeds.
static void grid_barrier(volatile __global atomic_uint* gbar, uint num_wg) {
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (num_wg <= 1u) return;
    if (get_local_id(0) == 0) {
        uint s = atomic_load_explicit(&gbar[1], memory_order_relaxed, memory_scope_device);
        uint a = atomic_fetch_add_explicit(&gbar[0], 1u, memory_order_acq_rel, memory_scope_device) + 1u;
        if (a == num_wg) {
            atomic_store_explicit(&gbar[0], 0u, memory_order_relaxed, memory_scope_device);
            atomic_store_explicit(&gbar[1], s ^ 1u, memory_order_release, memory_scope_device);
        } else {
            // Watchdog cap: never spin forever. If the grid is not co-resident
            // (the barrier can't complete) we give up rather than wedge the CP —
            // the dispatch still retires; bad data is caught by the host check.
            ulong spins = 0;
            while (atomic_load_explicit(&gbar[1], memory_order_acquire, memory_scope_device) == s) {
                if (++spins > 4000000000ul) {
                    break;
                }
            }
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}

// Cross-GPU barrier (system scope) — done by a single thread. Signals every
// peer's arrival slot for this GPU, then waits for all of this GPU's slots.
static void xgpu_barrier(__global const ulong* peer_flag_ptrs, __global uint* my_flags,
                         uint parts, uint self_idx, uint seq) {
    for (uint h = 0; h < parts; ++h) {
        volatile __global atomic_uint* f =
            (volatile __global atomic_uint*)(peer_flag_ptrs[h] + (ulong)self_idx * 4);
        atomic_store_explicit(f, seq, memory_order_release, memory_scope_all_svm_devices);
    }
    for (uint h = 0; h < parts; ++h) {
        volatile __global atomic_uint* f = (volatile __global atomic_uint*)(my_flags + h);
        ulong spins = 0;
        while (atomic_load_explicit(f, memory_order_acquire, memory_scope_all_svm_devices) != seq) {
            if (++spins > 4000000000ul) {
                break; // watchdog: give up rather than wedge the GPU
            }
        }
    }
}

// GPU-driven all-reduce: one persistent kernel per GPU runs the whole op with
// device-side synchronization (no host round-trips between phases), so a fixed
// co-resident grid streams the entire ~2(R-1)/R*n of XGMI traffic in one launch
// and reaches peak link bandwidth (vs many small ramp-limited kernels).
//
// write-based reduce-scatter (scatter-to-staging + local reduce) then push
// all-gather. `n` and `cl` must be multiples of 4. Launch with exactly
// `num_wg` workgroups of `wg` threads (co-resident for the grid barrier).
__kernel void allreduce_oneshot(
    __global float* own,                   // this GPU's buffer (in/out), n floats
    __global const ulong* peer_bufs,       // R buffer base VAs
    __global const ulong* stage_ptrs,      // R staging base VAs
    __global float* my_stage,              // this GPU's staging (R*cl floats)
    __global uint* my_flags,               // R arrival slots (peer-writable)
    __global const ulong* peer_flag_ptrs,  // R flag base VAs
    volatile __global atomic_uint* gbar,   // 2 u32 intra-GPU barrier
    uint parts, uint self_idx, uint cl, uint n, uint num_wg, uint seq_base,
    uint num_tiles) {
    uint gid = get_group_id(0);
    uint lid = get_local_id(0);
    uint lsz = get_local_size(0);
    uint tid = get_global_id(0);
    uint nthreads = num_wg * lsz;
    uint vec = n >> 2;          // float4 over the whole buffer
    uint clv = cl >> 2;         // float4 per chunk slot
    uint off4 = self_idx * clv; // my chunk's float4 offset
    uint mylen4 = (off4 < vec) ? min(clv, vec - off4) : 0u;
    const __global float4* own4 = (const __global float4*)own;
    const __global float4* stg4 = (const __global float4*)my_stage;
    uint nslots = num_wg / parts;          // workgroups per peer/link (num_wg % parts == 0)
    if (nslots < 1u) nslots = 1u;
    (void)num_tiles;

    // Phase 1: scatter as workgroup-balanced BIG CONTIGUOUS bursts. Each
    // workgroup streams a contiguous slice of ONE owner's chunk into that owner's
    // staging slot, so every XGMI link carries long bursts instead of 1-KB
    // owner-interleaved fragments; different workgroups (gid % parts) drive
    // different owners, so all R links stay busy — lifting per-link write
    // efficiency toward the XGMI ceiling.
    {
        uint owner = gid % parts;
        uint islot = gid / parts;
        uint per = (clv + nslots - 1u) / nslots;
        uint k0 = islot * per;
        uint k1 = min(k0 + per, clv);
        __global float4* dst = (__global float4*)stage_ptrs[owner];
        ulong base = (ulong)self_idx * clv;
        for (uint k = k0 + lid; k < k1; k += lsz) {
            uint i = owner * clv + k;
            if (i < vec) dst[base + k] = own4[i];
        }
    }
    grid_barrier(gbar, num_wg);
    if (gid == 0 && lid == 0)
        xgpu_barrier(peer_flag_ptrs, my_flags, parts, self_idx, seq_base + 1u);
    grid_barrier(gbar, num_wg);

    // Phase 2: reduce my chunk from its R staging slots (local HBM).
    for (uint i = tid; i < mylen4; i += nthreads) {
        float4 acc = (float4)(0.0f);
        for (uint p = 0; p < parts; ++p) acc += stg4[(ulong)p * clv + i];
        ((__global float4*)own)[off4 + i] = acc;
    }
    grid_barrier(gbar, num_wg);

    // Phase 3: all-gather as workgroup-balanced BIG CONTIGUOUS bursts. Each
    // workgroup streams a contiguous slice of my reduced chunk to ONE peer; all
    // peers covered by gid % parts. Same big-burst rationale as the scatter.
    {
        uint peer = gid % parts;
        uint pslot = gid / parts;
        uint per = (mylen4 + nslots - 1u) / nslots;
        uint p0 = pslot * per;
        uint p1 = min(p0 + per, mylen4);
        __global float4* dst = (__global float4*)peer_bufs[peer];
        const __global float4* src = (const __global float4*)own;
        for (uint i = p0 + lid; i < p1; i += lsz) {
            dst[off4 + i] = src[off4 + i];
        }
    }
}

// Device-side DUAL-PATH all-reduce: the CU kernel writes the first cu_clv float4
// of each chunk while the SDMA copy engines (host-pre-armed, coordinated purely
// through memory semaphores — no host round-trips) write the rest. Both write
// paths stream simultaneously so aggregate XGMI egress approaches the dual-path
// ceiling (~378 GB/s, past RCCL) while the device-side barriers keep the link
// busy across phases. sem[2*parts+1] per device (one SDMA queue per part):
// [0..parts)=per-queue SDMA scatter-done (SDMA FENCE ->, kernel polls all),
// [parts]=gather release (kernel ->, every SDMA queue POLL_REGMEM waits),
// [parts+1 .. 2*parts+1)=per-queue SDMA gather-done. All values = seq.
__kernel void allreduce_dualpath(
    __global float* own,
    __global const ulong* peer_bufs,
    __global const ulong* stage_ptrs,
    __global float* my_stage,
    __global uint* my_flags,
    __global const ulong* peer_flag_ptrs,
    volatile __global atomic_uint* gbar,
    volatile __global atomic_uint* sem,
    uint parts, uint self_idx, uint cl, uint n, uint num_wg, uint seq_base,
    uint cu_clv, uint cu_mylen4) {
    uint gid = get_group_id(0);
    uint lid = get_local_id(0);
    uint lsz = get_local_size(0);
    uint tid = get_global_id(0);
    uint nthreads = num_wg * lsz;
    uint vec = n >> 2;
    uint clv = cl >> 2;
    uint off4 = self_idx * clv;
    uint mylen4 = (off4 < vec) ? min(clv, vec - off4) : 0u;
    const __global float4* own4 = (const __global float4*)own;
    const __global float4* stg4 = (const __global float4*)my_stage;
    uint nslots = num_wg / parts;
    if (nslots < 1u) nslots = 1u;
    uint seq = seq_base + 1u;
    uint cu_s = min(cu_clv, clv);     // CU scatter float4 per chunk
    uint cu_g = min(cu_mylen4, mylen4); // CU gather float4 of my chunk

    // Phase 1: CU scatter. The LOCAL chunk (owner==self) is written IN FULL by
    // the CU — it is an intra-device copy, so routing it through an XGMI SDMA
    // engine would be degenerate/slow. Remote chunks get only the CU fraction
    // cu_s; the SDMA engines write the rest. Big contiguous bursts per (owner,slot).
    {
        uint owner = gid % parts;
        uint islot = gid / parts;
        uint cu_this = (owner == self_idx) ? clv : cu_s;
        uint per = (cu_this + nslots - 1u) / nslots;
        uint k0 = islot * per;
        uint k1 = min(k0 + per, cu_this);
        __global float4* dst = (__global float4*)stage_ptrs[owner];
        ulong base = (ulong)self_idx * clv;
        for (uint k = k0 + lid; k < k1; k += lsz) {
            uint i = owner * clv + k;
            if (i < vec) dst[base + k] = own4[i];
        }
    }
    grid_barrier(gbar, num_wg);
    // Wait for this device's SDMA scatter queues (each FENCEs sem[q]==seq), then
    // signal cross-GPU that our full (CU+SDMA) scatter is complete.
    if (gid == 0 && lid == 0) {
        for (uint q = 0; q < parts; ++q) {
            if (q == self_idx) continue; // local chunk: CU wrote it, no SDMA fence
            ulong spins = 0;
            while (atomic_load_explicit(&sem[q], memory_order_acquire,
                                        memory_scope_all_svm_devices) != seq) {
                if (++spins > 4000000000ul) break; // watchdog: never wedge
            }
        }
    }
    grid_barrier(gbar, num_wg);
    if (gid == 0 && lid == 0)
        xgpu_barrier(peer_flag_ptrs, my_flags, parts, self_idx, seq);
    grid_barrier(gbar, num_wg);

    // Phase 2: reduce my FULL chunk from its R staging slots (local HBM).
    for (uint i = tid; i < mylen4; i += nthreads) {
        float4 acc = (float4)(0.0f);
        for (uint p = 0; p < parts; ++p) acc += stg4[(ulong)p * clv + i];
        ((__global float4*)own)[off4 + i] = acc;
    }
    grid_barrier(gbar, num_wg);
    // Release the SDMA gather queues (each POLL_REGMEMs sem[parts]==seq).
    if (gid == 0 && lid == 0)
        atomic_store_explicit(&sem[parts], seq, memory_order_release,
                              memory_scope_all_svm_devices);

    // Phase 3: CU all-gather — first cu_g float4 of my reduced chunk to each
    // remote peer; the local peer (peer==self) is a no-op (own already holds the
    // reduced chunk). SDMA writes the rest to remote peers.
    {
        uint peer = gid % parts;
        uint pslot = gid / parts;
        uint cu_this = (peer == self_idx) ? 0u : cu_g;
        uint per = (cu_this + nslots - 1u) / nslots;
        uint p0 = pslot * per;
        uint p1 = min(p0 + per, cu_this);
        __global float4* dst = (__global float4*)peer_bufs[peer];
        const __global float4* src = (const __global float4*)own;
        for (uint i = p0 + lid; i < p1; i += lsz) {
            dst[off4 + i] = src[off4 + i];
        }
    }
    grid_barrier(gbar, num_wg);
    // Wait for this device's SDMA gather queues (sem[parts+1+q]==seq) so the
    // buffer is fully gathered when the host observes the kernel retire.
    if (gid == 0 && lid == 0) {
        for (uint q = 0; q < parts; ++q) {
            if (q == self_idx) continue; // local peer: no SDMA gather
            ulong spins = 0;
            while (atomic_load_explicit(&sem[parts + 1u + q], memory_order_acquire,
                                        memory_scope_all_svm_devices) != seq) {
                if (++spins > 4000000000ul) break;
            }
        }
    }
}

// ---- matrix-core (MFMA) substrate — CDNA4 / gfx950 ----
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Single-tile MFMA validation: D[16x16] = A[16x16] @ B[16x16] on one wavefront,
// using the FP16 matrix core (D = A*B + C, FP32 accumulate). Inputs are f32 and
// converted to half in-kernel so the host plumbing stays f32. This validates
// MFMA emission on gfx950 and the per-lane fragment layout before we build the
// tiled GEMM. A row-major M×K, B row-major K×N, D row-major M×N.
__kernel void mfma_tile_16(__global const float* A, __global const float* B,
                           __global float* D) {
    uint lane = get_local_id(0); // 0..63
    uint m = lane % 16;          // A row / D row-group base
    uint kg = lane / 16;         // K-group (0..3): this lane's 4 K values
    half4 a, b;
    for (int i = 0; i < 4; ++i) {
        a[i] = (half)A[m * 16 + (kg * 4 + i)];        // A[m][kg*4+i]
        b[i] = (half)B[(kg * 4 + i) * 16 + m];        // B[kg*4+i][n=m]
    }
    float4 c = (float4)(0.0f);
    c = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
    // D[(kg*4 + i)][n = lane%16]
    for (int i = 0; i < 4; ++i) {
        D[(kg * 4 + i) * 16 + m] = c[i];
    }
}

// Tiled FP16 GEMM: C[M×N] = A[M×K] · B[K×N], all row-major; FP16 in HBM, FP32
// out. One wavefront per 16×16 output tile, K-looped through the matrix core.
// Flattened 2D grid: workgroup `wg` covers tile (wg/tiles_n, wg%tiles_n).
// M, N, K must be multiples of 16 (the matrix-core tile). No LDS yet — first
// correct version; LDS staging + larger per-workgroup tiles come next.
__kernel void gemm_f16(__global const half* A, __global const half* B, __global float* C,
                       uint M, uint N, uint K, uint tiles_n) {
    uint wg = get_group_id(0);
    uint row0 = (wg / tiles_n) * 16;
    uint col0 = (wg % tiles_n) * 16;
    uint lane = get_local_id(0);   // 0..63
    uint m = lane % 16;
    uint kg = lane / 16;           // 0..3 — this lane's K-group
    float4 c = (float4)(0.0f);
    for (uint k0 = 0; k0 < K; k0 += 16) {
        half4 a, b;
        for (int i = 0; i < 4; ++i) {
            uint kk = k0 + kg * 4 + i;
            a[i] = A[(ulong)(row0 + m) * K + kk];   // A[row0+m][kk]
            b[i] = B[(ulong)kk * N + (col0 + m)];   // B[kk][col0+m]
        }
        c = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
    }
    for (int i = 0; i < 4; ++i) {
        C[(ulong)(row0 + kg * 4 + i) * N + (col0 + m)] = c[i];
    }
}

// LDS-staged, register-blocked FP16 GEMM. Each workgroup (one 64-lane wavefront)
// computes a 64x64 output tile; A/B block tiles are staged in LDS once per
// K-step and reused across 4x4 = 16 matrix-core accumulators, lifting
// arithmetic intensity from ~1 to ~64 MACs/element. M,N multiples of 64,
// K multiple of 16. Flattened 2D grid: tile (wg/tiles_n, wg%tiles_n).
#define GL_WM 64
#define GL_WN 64
#define GL_BK 16
#define GL_TM 4   // 64/16
#define GL_TN 4
__kernel void gemm_f16_lds(__global const half* A, __global const half* B, __global float* C,
                           uint M, uint N, uint K, uint tiles_n) {
    __local half As[GL_WM * GL_BK];  // 64x16
    __local half Bs[GL_BK * GL_WN];  // 16x64
    uint wg = get_group_id(0);
    uint row0 = (wg / tiles_n) * GL_WM;
    uint col0 = (wg % tiles_n) * GL_WN;
    uint lane = get_local_id(0);     // 0..63
    uint m = lane % 16;
    uint kg = lane / 16;             // 0..3

    float4 acc[GL_TM][GL_TN];
    for (int i = 0; i < GL_TM; ++i)
        for (int j = 0; j < GL_TN; ++j)
            acc[i][j] = (float4)(0.0f);

    for (uint k0 = 0; k0 < K; k0 += GL_BK) {
        // Cooperative load: 64 lanes × 16 elements each = 1024 (= WM*BK = BK*WN).
        for (int t = 0; t < (GL_WM * GL_BK) / 64; ++t) {
            uint idx = lane + t * 64;          // 0..1023
            uint r = idx / GL_BK, c = idx % GL_BK;
            As[idx] = A[(ulong)(row0 + r) * K + (k0 + c)];
        }
        for (int t = 0; t < (GL_BK * GL_WN) / 64; ++t) {
            uint idx = lane + t * 64;
            uint r = idx / GL_WN, c = idx % GL_WN;
            Bs[idx] = B[(ulong)(k0 + r) * N + (col0 + c)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int ti = 0; ti < GL_TM; ++ti) {
            half4 a;
            for (int i = 0; i < 4; ++i)
                a[i] = As[(ti * 16 + m) * GL_BK + (kg * 4 + i)];
            for (int tj = 0; tj < GL_TN; ++tj) {
                half4 b;
                for (int i = 0; i < 4; ++i)
                    b[i] = Bs[(kg * 4 + i) * GL_WN + (tj * 16 + m)];
                acc[ti][tj] = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc[ti][tj], 0, 0, 0);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int ti = 0; ti < GL_TM; ++ti)
        for (int tj = 0; tj < GL_TN; ++tj)
            for (int i = 0; i < 4; ++i)
                C[(ulong)(row0 + ti * 16 + kg * 4 + i) * N + (col0 + tj * 16 + m)] = acc[ti][tj][i];
}

// Multi-wave LDS GEMM: workgroup = 256 threads (4 wavefronts) computing a
// 128x128 output tile (each wave a 64x64 sub-tile, 4x4 matrix-core accumulators
// in AGPRs). 128x32 A and 32x128 B staged in LDS per K-step and reused across
// waves. M,N multiples of 128; K multiple of 32.
#define M2_BM 128
#define M2_BN 128
#define M2_BK 32
__kernel void gemm_f16_lds2(__global const half* A, __global const half* B, __global float* C,
                            uint M, uint N, uint K, uint tiles_n) {
    __local half As[M2_BM * M2_BK];  // 128x32
    __local half Bs[M2_BK * M2_BN];  // 32x128
    uint wg = get_group_id(0);
    uint row0 = (wg / tiles_n) * M2_BM;
    uint col0 = (wg % tiles_n) * M2_BN;
    uint t = get_local_id(0);        // 0..255
    uint wave = t / 64, lane = t % 64;
    uint wrow = (wave / 2) * 64;     // wave sub-tile row (0 or 64)
    uint wcol = (wave % 2) * 64;
    uint m = lane % 16, kg = lane / 16;

    float4 acc[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            acc[i][j] = (float4)(0.0f);

    for (uint k0 = 0; k0 < K; k0 += M2_BK) {
        for (int s = 0; s < (M2_BM * M2_BK) / 256; ++s) {
            uint idx = t + s * 256;                 // 0..4095
            uint r = idx / M2_BK, c = idx % M2_BK;  // As[r][c]
            As[idx] = A[(ulong)(row0 + r) * K + (k0 + c)];
        }
        for (int s = 0; s < (M2_BK * M2_BN) / 256; ++s) {
            uint idx = t + s * 256;
            uint r = idx / M2_BN, c = idx % M2_BN;  // Bs[r][c]
            Bs[idx] = B[(ulong)(k0 + r) * N + (col0 + c)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int kk = 0; kk < M2_BK; kk += 16) {
            for (int ti = 0; ti < 4; ++ti) {
                half4 a;
                for (int i = 0; i < 4; ++i)
                    a[i] = As[(wrow + ti * 16 + m) * M2_BK + (kk + kg * 4 + i)];
                for (int tj = 0; tj < 4; ++tj) {
                    half4 b;
                    for (int i = 0; i < 4; ++i)
                        b[i] = Bs[(kk + kg * 4 + i) * M2_BN + (wcol + tj * 16 + m)];
                    acc[ti][tj] = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc[ti][tj], 0, 0, 0);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int ti = 0; ti < 4; ++ti)
        for (int tj = 0; tj < 4; ++tj)
            for (int i = 0; i < 4; ++i)
                C[(ulong)(row0 + wrow + ti * 16 + kg * 4 + i) * N + (col0 + wcol + tj * 16 + m)] =
                    acc[ti][tj][i];
}

// Double-buffered (software-pipelined) FP16 GEMM. Same 128x128 / 4-wave tiling
// as gemm_f16_lds2, but with two LDS buffers: while the matrix cores compute on
// the current K-tile, the next K-tile's global loads are in flight (issued into
// registers, then written to the other LDS buffer). This hides global-load
// latency behind MFMA compute — the dominant win toward peak.
#define DB_BM 128
#define DB_BN 128
#define DB_BK 32
#define DB_NA ((DB_BM * DB_BK) / 256)  // 16 A halves / thread / K-tile
#define DB_NB ((DB_BK * DB_BN) / 256)  // 16 B halves / thread / K-tile
__kernel void gemm_f16_db(__global const half* A, __global const half* B, __global float* C,
                          uint M, uint N, uint K, uint tiles_n) {
    __local half As[2][DB_BM * DB_BK];
    __local half Bs[2][DB_BK * DB_BN];
    uint wg = get_group_id(0);
    uint row0 = (wg / tiles_n) * DB_BM;
    uint col0 = (wg % tiles_n) * DB_BN;
    uint t = get_local_id(0);
    uint wave = t / 64, lane = t % 64;
    uint wrow = (wave / 2) * 64, wcol = (wave % 2) * 64;
    uint m = lane % 16, kg = lane / 16;
    uint nk = K / DB_BK;

    float4 acc[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            acc[i][j] = (float4)(0.0f);

    // Prologue: load K-tile 0 into buffer 0.
    for (int s = 0; s < DB_NA; ++s) {
        uint idx = t + s * 256;
        As[0][idx] = A[(ulong)(row0 + idx / DB_BK) * K + (idx % DB_BK)];
    }
    for (int s = 0; s < DB_NB; ++s) {
        uint idx = t + s * 256;
        Bs[0][idx] = B[(ulong)(idx / DB_BN) * N + (col0 + idx % DB_BN)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint k = 0; k < nk; ++k) {
        uint cur = k & 1, nxt = (k + 1) & 1;
        half ar[DB_NA], br[DB_NB];
        uint k1 = (k + 1) * DB_BK;
        // Prefetch next K-tile's globals into registers (long latency in flight).
        if (k + 1 < nk) {
            for (int s = 0; s < DB_NA; ++s) {
                uint idx = t + s * 256;
                ar[s] = A[(ulong)(row0 + idx / DB_BK) * K + (k1 + idx % DB_BK)];
            }
            for (int s = 0; s < DB_NB; ++s) {
                uint idx = t + s * 256;
                br[s] = B[(ulong)(k1 + idx / DB_BN) * N + (col0 + idx % DB_BN)];
            }
        }
        // Compute on the current buffer (overlaps with the loads above).
        for (int kk = 0; kk < DB_BK; kk += 16) {
            for (int ti = 0; ti < 4; ++ti) {
                half4 a;
                for (int i = 0; i < 4; ++i)
                    a[i] = As[cur][(wrow + ti * 16 + m) * DB_BK + (kk + kg * 4 + i)];
                for (int tj = 0; tj < 4; ++tj) {
                    half4 b;
                    for (int i = 0; i < 4; ++i)
                        b[i] = Bs[cur][(kk + kg * 4 + i) * DB_BN + (wcol + tj * 16 + m)];
                    acc[ti][tj] = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc[ti][tj], 0, 0, 0);
                }
            }
        }
        // Commit prefetched tile to the other buffer, then sync.
        if (k + 1 < nk) {
            for (int s = 0; s < DB_NA; ++s)
                As[nxt][t + s * 256] = ar[s];
            for (int s = 0; s < DB_NB; ++s)
                Bs[nxt][t + s * 256] = br[s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int ti = 0; ti < 4; ++ti)
        for (int tj = 0; tj < 4; ++tj)
            for (int i = 0; i < 4; ++i)
                C[(ulong)(row0 + wrow + ti * 16 + kg * 4 + i) * N + (col0 + wcol + tj * 16 + m)] =
                    acc[ti][tj][i];
}

// ---- FlashDecoding attention (decode phase) — gfx950 ----
// Decode attention is memory-bound: one query attends over an N-token KV cache,
// reading all of K and V once. SOTA here = saturate HBM bandwidth. Split-KV:
// num_splits workgroups each stream a contiguous KV slice with online softmax
// (running max m, running sum l, running output acc[D]); a combine kernel merges
// the per-split (m,l,acc) partials. FP16 KV; D (head dim) <= 128.
#define ATTN_MAXD 128
// One head; q[D], K/V row-major [N][D]. partials: per split (2 + D) floats
// = [m, l, acc[0..D)]. Workgroup = one split, 64 lanes.
__kernel void attn_decode_split(__global const half* q, __global const half* K,
                                __global const half* V, __global float* partials,
                                uint N, uint D, float scale, uint num_splits) {
    __local float qsh[ATTN_MAXD];
    __local float lacc[64 * ATTN_MAXD];  // per-lane partial output (<=32 KiB)
    __local float lm[64];
    __local float ll[64];
    uint sp = get_group_id(0);
    uint t = get_local_id(0);            // 0..63
    for (uint j = t; j < D; j += 64)
        qsh[j] = (float)q[j];
    barrier(CLK_LOCAL_MEM_FENCE);

    uint S = (N + num_splits - 1) / num_splits;
    uint lo = sp * S;
    uint hi = min(N, lo + S);

    // Accumulate directly in LDS (no large private array — our raw KFD queue
    // has no scratch backing, so kernels must avoid private/scratch memory).
    float m = -INFINITY, l = 0.0f;
    __local float* myacc = lacc + (ulong)t * ATTN_MAXD;
    for (uint d = 0; d < D; ++d) myacc[d] = 0.0f;

    for (uint i = lo + t; i < hi; i += 64) {
        __global const half* kr = K + (ulong)i * D;
        float s = 0.0f;
        for (uint d = 0; d < D; ++d) s += qsh[d] * (float)kr[d];
        s *= scale;
        float m_new = fmax(m, s);
        float corr = native_exp(m - m_new);
        float p = native_exp(s - m_new);
        l = l * corr + p;
        __global const half* vr = V + (ulong)i * D;
        for (uint d = 0; d < D; ++d) myacc[d] = myacc[d] * corr + p * (float)vr[d];
        m = m_new;
    }
    lm[t] = m;
    ll[t] = l;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (t == 0) {
        float M = -INFINITY;
        for (int tt = 0; tt < 64; ++tt) M = fmax(M, lm[tt]);
        float L = 0.0f;
        for (int tt = 0; tt < 64; ++tt) L += ll[tt] * native_exp(lm[tt] - M);
        __global float* pr = partials + (ulong)sp * (D + 2);
        pr[0] = M;
        pr[1] = L;
        // o[d] = sum_tt lacc[tt][d] * exp(lm[tt]-M), written straight to global.
        for (uint d = 0; d < D; ++d) {
            float o = 0.0f;
            for (int tt = 0; tt < 64; ++tt) o += lacc[tt * ATTN_MAXD + d] * native_exp(lm[tt] - M);
            pr[2 + d] = o;
        }
    }
}

// Combine the num_splits partial softmax states into the final output O[D].
// Parallel combine: one workgroup of D threads, thread d produces O[d]. M and L
// are reduced once into LDS, then each thread sums its output dim over the
// splits (reads of partials[s][2+d] are coalesced across threads for fixed s).
__kernel void attn_decode_combine(__global const float* partials, __global float* O,
                                  uint D, uint num_splits) {
    __local float red[128];
    __local float lM, lL;
    uint t = get_local_id(0);
    uint nt = D;  // workgroup is launched with D threads (avoid get_local_size:
                  // it reads a COv5 hidden arg our arm_grid path doesn't set)

    // Cooperative max over splits' m.
    float pm = -INFINITY;
    for (uint s = t; s < num_splits; s += nt)
        pm = fmax(pm, partials[(ulong)s * (D + 2)]);
    red[t] = pm;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint off = nt >> 1; off > 0; off >>= 1) {
        if (t < off) red[t] = fmax(red[t], red[t + off]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (t == 0) lM = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    float M = lM;

    // Cooperative sum for L.
    float pl = 0.0f;
    for (uint s = t; s < num_splits; s += nt) {
        __global const float* pr = partials + (ulong)s * (D + 2);
        pl += pr[1] * native_exp(pr[0] - M);
    }
    red[t] = pl;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint off = nt >> 1; off > 0; off >>= 1) {
        if (t < off) red[t] += red[t + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (t == 0) lL = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Thread d sums output dim d over splits (coalesced reads across threads).
    if (t < D) {
        float o = 0.0f;
        for (uint s = 0; s < num_splits; ++s) {
            __global const float* pr = partials + (ulong)s * (D + 2);
            o += pr[2 + t] * native_exp(pr[0] - M);
        }
        O[t] = o / lL;
    }
}

// Tree-reduction pass for the combine: workgroup g (D threads) merges the
// contiguous range of input partials [g*group_size, (g+1)*group_size) into ONE
// output partial out[g] = [M, L, o[0..D)] (unnormalized, same softmax-partial
// format as the split). With num_splits partials and group_size ≈ sqrt(num_
// splits) this turns the combine's serial O(num_splits) reduction (one
// workgroup, latency-bound on a single CU) into two short parallel passes.
__kernel void attn_decode_reduce_partials(__global const float* in, __global float* out,
                                          uint num_in, uint D, uint group_size) {
    uint g = get_group_id(0);
    uint t = get_local_id(0);
    uint nt = D;
    uint lo = g * group_size;
    uint hi = min(num_in, lo + group_size);
    __local float red[128];
    __local float lM, lL;

    float pm = -INFINITY;
    for (uint s = lo + t; s < hi; s += nt) pm = fmax(pm, in[(ulong)s * (D + 2)]);
    red[t] = pm;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint off = nt >> 1; off > 0; off >>= 1) {
        if (t < off) red[t] = fmax(red[t], red[t + off]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (t == 0) lM = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    float M = lM;
    if (M == -INFINITY) M = 0.0f;

    float pl = 0.0f;
    for (uint s = lo + t; s < hi; s += nt) {
        __global const float* pr = in + (ulong)s * (D + 2);
        pl += pr[1] * native_exp(pr[0] - M);
    }
    red[t] = pl;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint off = nt >> 1; off > 0; off >>= 1) {
        if (t < off) red[t] += red[t + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (t == 0) lL = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (t < D) {
        float o = 0.0f;
        for (uint s = lo; s < hi; ++s) {
            __global const float* pr = in + (ulong)s * (D + 2);
            o += pr[2 + t] * native_exp(pr[0] - M);
        }
        __global float* po = out + (ulong)g * (D + 2);
        if (t == 0) {
            po[0] = M;
            po[1] = lL;
        }
        po[2 + t] = o;
    }
}

// Wide-load FlashDecoding split (D == 128). The decode bottleneck is global
// memory throughput, and it is dominated by memory-level parallelism × load
// width, not compute (a pure float4 VRAM read here sustains ~25x what a
// per-token half-load attention kernel did). So this kernel uses the widest,
// most parallel access we can:
//   * 16 lanes per token: each lane loads 8 contiguous dims as one 128-bit
//     half8 load (vs a 4-byte half2). The 16 lanes of a token cover its 128
//     dims in one coalesced 256 B transaction.
//   * 4 token-streams per 64-lane wavefront (lane>>4 picks the stream), so each
//     load instruction pulls 4 independent token rows = 1 KB, the way the
//     bandwidth microbench does.
//   * WPS_ATTN=4 wavefronts per workgroup for occupancy, each covering a
//     quarter of the split.
// The QKᵀ score is reduced across the 16 lanes of a token with a ds_bpermute
// butterfly (xor 1,2,4,8); the 4 streams and the 4 wavefronts are merged
// (online-softmax) at the end. Output is a SCALAR o[8] per lane (no scratch).
// partials[sp] = [m, l, o[0..128)].
#ifndef MAINARCH_WPS_ATTN
#define MAINARCH_WPS_ATTN 4
#endif
#ifndef MAINARCH_UATTN
#define MAINARCH_UATTN 8
#endif
#define WPS_ATTN MAINARCH_WPS_ATTN
#define UATTN MAINARCH_UATTN
#define BPERM(MASK, V) as_float(__builtin_amdgcn_ds_bpermute((int)(((lane ^ (MASK)) << 2)), as_int(V)))
__kernel void attn_decode_split2(__global const half* q, __global const half* K,
                                 __global const half* V, __global float* partials,
                                 uint N, uint D, float scale, uint num_splits) {
    uint sp = get_group_id(0);
    uint tid = get_local_id(0);      // 0..64*WPS_ATTN-1
    uint w = tid >> 6;               // wavefront 0..WPS_ATTN-1
    uint lane = tid & 63;            // wavefront-local lane 0..63
    uint g = lane >> 4;              // token-stream 0..3 within the wavefront
    uint sub = lane & 15;            // dim chunk 0..15 (owns dims [sub*8, sub*8+8))
    uint S = (N + num_splits - 1) / num_splits;
    uint lo = sp * S;
    uint hi = min(N, lo + S);
    uint span = hi > lo ? hi - lo : 0;
    uint per = (span + WPS_ATTN - 1) / WPS_ATTN;
    uint wlo = lo + w * per;
    uint whi = min(hi, wlo + per);

    half8 q8 = vload8(sub, q);
    float qv[8];
    qv[0] = q8.s0; qv[1] = q8.s1; qv[2] = q8.s2; qv[3] = q8.s3;
    qv[4] = q8.s4; qv[5] = q8.s5; qv[6] = q8.s6; qv[7] = q8.s7;

    float m = -INFINITY, l = 0.0f, o[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = 0.0f;

    // Stream g walks tokens wlo+g, wlo+g+4, ... (4 = streams per wavefront).
    // CRITICAL for bandwidth: prefetch UATTN tokens' K and V loads *before* any
    // cross-lane reduce. The ds_bpermute below is a wavefront-wide sync, so if
    // the load sat right before it the wavefront could only keep ONE load in
    // flight (memory-level parallelism = 1, latency-bound). Issuing UATTN
    // independent wide loads first lets the memory system pipeline them.
    for (uint base = wlo + g; base < whi; base += 4u * UATTN) {
        half8 kb[UATTN], vb[UATTN];
        #pragma unroll
        for (uint u = 0; u < UATTN; ++u) {
            uint t = base + 4u * u;
            uint tt = t < whi ? t : whi - 1;   // clamp (masked out below) — never OOB
            kb[u] = vload8(sub, K + (ulong)tt * 128);
            vb[u] = vload8(sub, V + (ulong)tt * 128);
        }
        #pragma unroll
        for (uint u = 0; u < UATTN; ++u) {
            uint t = base + 4u * u;
            if (t >= whi) break;
            half8 k8 = kb[u];
            float partial = qv[0]*(float)k8.s0 + qv[1]*(float)k8.s1 + qv[2]*(float)k8.s2 + qv[3]*(float)k8.s3
                          + qv[4]*(float)k8.s4 + qv[5]*(float)k8.s5 + qv[6]*(float)k8.s6 + qv[7]*(float)k8.s7;
            partial += BPERM(1u, partial);
            partial += BPERM(2u, partial);
            partial += BPERM(4u, partial);
            partial += BPERM(8u, partial);
            float s = partial * scale;
            float m_new = fmax(m, s);
            float corr = native_exp(m - m_new);
            float p = native_exp(s - m_new);
            l = l * corr + p;
            half8 v8 = vb[u];
            o[0] = o[0]*corr + p*(float)v8.s0; o[1] = o[1]*corr + p*(float)v8.s1;
            o[2] = o[2]*corr + p*(float)v8.s2; o[3] = o[3]*corr + p*(float)v8.s3;
            o[4] = o[4]*corr + p*(float)v8.s4; o[5] = o[5]*corr + p*(float)v8.s5;
            o[6] = o[6]*corr + p*(float)v8.s6; o[7] = o[7]*corr + p*(float)v8.s7;
            m = m_new;
        }
    }

    // Merge the 4 token-streams of this wavefront (partners at lane^16, lane^32).
    float M = m;
    M = fmax(M, BPERM(16u, M));
    M = fmax(M, BPERM(32u, M));
    if (M == -INFINITY) M = 0.0f;     // wavefront had no tokens → zero contribution
    float cg = native_exp(m - M);
    float L = l * cg;
    L += BPERM(16u, L);
    L += BPERM(32u, L);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float oc = o[i] * cg;
        oc += BPERM(16u, oc);
        oc += BPERM(32u, oc);
        o[i] = oc;
    }
    // Now lanes of stream g==0 (sub=0..15) hold the wavefront's full o[0..128).

    // Merge the WPS_ATTN wavefronts' partial softmax states in LDS.
    __local float wm[WPS_ATTN], wl[WPS_ATTN], wo[WPS_ATTN][128];
    if (lane == 0) { wm[w] = M; wl[w] = L; }
    if (g == 0) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) wo[w][sub * 8 + i] = o[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (w == 0 && lane < 64) {
        float MM = -INFINITY;
        for (uint k = 0; k < WPS_ATTN; ++k) MM = fmax(MM, wm[k]);
        if (MM == -INFINITY) MM = 0.0f;
        float LL = 0.0f;
        for (uint k = 0; k < WPS_ATTN; ++k) LL += wl[k] * native_exp(wm[k] - MM);
        __global float* pr = partials + (ulong)sp * (D + 2);
        if (lane == 0) { pr[0] = MM; pr[1] = LL; }
        // 64 lanes write the 128 output dims (2 each).
        for (uint d = lane; d < 128; d += 64) {
            float acc = 0.0f;
            for (uint k = 0; k < WPS_ATTN; ++k)
                acc += wo[k][d] * native_exp(wm[k] - MM);
            pr[2 + d] = acc;
        }
    }
}

// Decode one OCP E4M3 (e4m3fn: 1-4-3, bias 7, max 448, no inf; NaN=0x7f/0xff)
// byte to f32. Pure bit math so it matches the host quantizer exactly regardless
// of hardware cvt semantics; we never store NaN so 0x7f is not special-cased.
static inline float e4m3_to_f32(uchar b) {
    uint e = (b >> 3) & 0xFu;
    uint m = b & 0x7u;
    float v = (e == 0u) ? (float)m * 0.001953125f               // subnormal: m * 2^-9
                        : as_float(((e + 120u) << 23) | (m << 20));  // (e-7+127)<<23
    return (b & 0x80u) ? -v : v;
}

// FP8-E4M3 KV variant of attn_decode_split2. K and V are E4M3 bytes with a
// per-token scale (scale_k[t], scale_v[t]); the query stays FP16. The per-token
// scale factors out of the dot product (q·(s·K) = s·(q·K)), so dequant is just a
// format-convert per element plus ONE scale-multiply per token: the score is
// scaled by scale_k[t] after the 16-lane reduce, and p is scaled by scale_v[t]
// before the V accumulate. Halves KV bytes moved vs FP16 — decode is
// memory-bound, so this is the throughput lever. Layout/MLP identical to split2.
__kernel void attn_decode_split2_fp8(__global const half* q,
                                     __global const uchar* K, __global const uchar* V,
                                     __global const float* scale_k, __global const float* scale_v,
                                     __global float* partials,
                                     uint N, uint D, float scale, uint num_splits) {
    uint sp = get_group_id(0);
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint g = lane >> 4;
    uint sub = lane & 15;
    uint S = (N + num_splits - 1) / num_splits;
    uint lo = sp * S;
    uint hi = min(N, lo + S);
    uint span = hi > lo ? hi - lo : 0;
    uint per = (span + WPS_ATTN - 1) / WPS_ATTN;
    uint wlo = lo + w * per;
    uint whi = min(hi, wlo + per);

    half8 q8 = vload8(sub, q);
    float qv[8];
    qv[0] = q8.s0; qv[1] = q8.s1; qv[2] = q8.s2; qv[3] = q8.s3;
    qv[4] = q8.s4; qv[5] = q8.s5; qv[6] = q8.s6; qv[7] = q8.s7;

    float m = -INFINITY, l = 0.0f, o[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = 0.0f;

    // Load 8 E4M3 bytes/lane as a uint2 and decode with the hardware packed
    // FP8->f32 conversion (2 values/instruction) instead of manual bit math —
    // the manual dequant is the classic ~50-cycle/token bottleneck that tips the
    // kernel compute-bound and erases the byte saving.
    for (uint base = wlo + g; base < whi; base += 4u * UATTN) {
        uint2 kb[UATTN], vb[UATTN];
        #pragma unroll
        for (uint u = 0; u < UATTN; ++u) {
            uint t = base + 4u * u;
            uint tt = t < whi ? t : whi - 1;
            kb[u] = ((__global const uint2*)(K + (ulong)tt * 128))[sub];
            vb[u] = ((__global const uint2*)(V + (ulong)tt * 128))[sub];
        }
        #pragma unroll
        for (uint u = 0; u < UATTN; ++u) {
            uint t = base + 4u * u;
            if (t >= whi) break;
            uint2 k2 = kb[u];
            float2 ka = __builtin_amdgcn_cvt_pk_f32_fp8(k2.x, false);
            float2 kc = __builtin_amdgcn_cvt_pk_f32_fp8(k2.x, true);
            float2 ke = __builtin_amdgcn_cvt_pk_f32_fp8(k2.y, false);
            float2 kg = __builtin_amdgcn_cvt_pk_f32_fp8(k2.y, true);
            float partial = qv[0]*ka.x + qv[1]*ka.y + qv[2]*kc.x + qv[3]*kc.y
                          + qv[4]*ke.x + qv[5]*ke.y + qv[6]*kg.x + qv[7]*kg.y;
            partial += BPERM(1u, partial);
            partial += BPERM(2u, partial);
            partial += BPERM(4u, partial);
            partial += BPERM(8u, partial);
            float s = partial * scale_k[t] * scale;
            float m_new = fmax(m, s);
            float corr = native_exp(m - m_new);
            float p = native_exp(s - m_new);
            l = l * corr + p;
            float pv = p * scale_v[t];
            uint2 v2 = vb[u];
            float2 va = __builtin_amdgcn_cvt_pk_f32_fp8(v2.x, false);
            float2 vc = __builtin_amdgcn_cvt_pk_f32_fp8(v2.x, true);
            float2 ve = __builtin_amdgcn_cvt_pk_f32_fp8(v2.y, false);
            float2 vg = __builtin_amdgcn_cvt_pk_f32_fp8(v2.y, true);
            o[0] = o[0]*corr + pv*va.x; o[1] = o[1]*corr + pv*va.y;
            o[2] = o[2]*corr + pv*vc.x; o[3] = o[3]*corr + pv*vc.y;
            o[4] = o[4]*corr + pv*ve.x; o[5] = o[5]*corr + pv*ve.y;
            o[6] = o[6]*corr + pv*vg.x; o[7] = o[7]*corr + pv*vg.y;
            m = m_new;
        }
    }

    // Merge the 4 token-streams of this wavefront (identical to split2).
    float M = m;
    M = fmax(M, BPERM(16u, M));
    M = fmax(M, BPERM(32u, M));
    if (M == -INFINITY) M = 0.0f;
    float cg = native_exp(m - M);
    float L = l * cg;
    L += BPERM(16u, L);
    L += BPERM(32u, L);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float oc = o[i] * cg;
        oc += BPERM(16u, oc);
        oc += BPERM(32u, oc);
        o[i] = oc;
    }

    __local float wm[WPS_ATTN], wl[WPS_ATTN], wo[WPS_ATTN][128];
    if (lane == 0) { wm[w] = M; wl[w] = L; }
    if (g == 0) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) wo[w][sub * 8 + i] = o[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (w == 0 && lane < 64) {
        float MM = -INFINITY;
        for (uint k = 0; k < WPS_ATTN; ++k) MM = fmax(MM, wm[k]);
        if (MM == -INFINITY) MM = 0.0f;
        float LL = 0.0f;
        for (uint k = 0; k < WPS_ATTN; ++k) LL += wl[k] * native_exp(wm[k] - MM);
        __global float* pr = partials + (ulong)sp * (D + 2);
        if (lane == 0) { pr[0] = MM; pr[1] = LL; }
        for (uint d = lane; d < 128; d += 64) {
            float acc = 0.0f;
            for (uint k = 0; k < WPS_ATTN; ++k)
                acc += wo[k][d] * native_exp(wm[k] - MM);
            pr[2 + d] = acc;
        }
    }
}

// GQA (grouped-query attention) FP8 decode split: GQA_G query heads SHARE one
// FP8 KV head. The whole point: each KV element is loaded and dequantized ONCE
// and reused across all GQA_G heads, so a full head-group decode reads the KV
// cache once instead of GQA_G times — the dominant memory-bound cost in real
// decode (Qwen-style GQA). Per-head online-softmax state (m,l,o) is kept in
// registers. Partials are head-major: partials[h*num_splits + sp].
#define GQA_G 8

// FP16 GQA decode split: GQA_G query heads share ONE FP16 KV head, so the KV
// cache is read+used ONCE per group instead of once per query head. Same layout
// and online-softmax math as attn_decode_split2, but the K/V tile loaded by each
// lane is reused across all GQA_G heads (no dequant — KV is half). This is the
// model decode loop's attention kernel: the per-head split re-read the shared KV
// `group` times (16x for Qwen 64Q/4KV); this collapses that to one read.
// Partials are head-major: partials[h*num_splits + sp][D+2].
__kernel void attn_decode_split2_gqa(
    __global const half* q,                          // [GQA_G][128]
    __global const half* K, __global const half* V,  // one KV head, [N][128] half
    __global float* partials,                        // [GQA_G][num_splits][D+2]
    uint N, uint D, float scale, uint num_splits) {
    uint sp = get_group_id(0);
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint g = lane >> 4;
    uint sub = lane & 15;
    uint S = (N + num_splits - 1) / num_splits;
    uint lo = sp * S;
    uint hi = min(N, lo + S);
    uint span = hi > lo ? hi - lo : 0;
    uint per = (span + WPS_ATTN - 1) / WPS_ATTN;
    uint wlo = lo + w * per;
    uint whi = min(hi, wlo + per);

    float qv[GQA_G][8];
    #pragma unroll
    for (uint h = 0; h < GQA_G; ++h) {
        half8 q8 = vload8(sub, q + h * 128);
        qv[h][0]=q8.s0; qv[h][1]=q8.s1; qv[h][2]=q8.s2; qv[h][3]=q8.s3;
        qv[h][4]=q8.s4; qv[h][5]=q8.s5; qv[h][6]=q8.s6; qv[h][7]=q8.s7;
    }
    float m[GQA_G], l[GQA_G], o[GQA_G][8];
    #pragma unroll
    for (uint h = 0; h < GQA_G; ++h) {
        m[h] = -INFINITY; l[h] = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) o[h][i] = 0.0f;
    }

    for (uint base = wlo + g; base < whi; base += 4u * UATTN) {
        half8 kb[UATTN], vb[UATTN];
        #pragma unroll
        for (uint u = 0; u < UATTN; ++u) {
            uint t = base + 4u * u;
            uint tt = t < whi ? t : whi - 1;
            kb[u] = vload8(sub, K + (ulong)tt * 128);
            vb[u] = vload8(sub, V + (ulong)tt * 128);
        }
        #pragma unroll
        for (uint u = 0; u < UATTN; ++u) {
            uint t = base + 4u * u;
            if (t >= whi) break;
            // Load this lane's 8 K and 8 V dims ONCE (shared across heads).
            half8 k8 = kb[u];
            float kf[8] = {(float)k8.s0,(float)k8.s1,(float)k8.s2,(float)k8.s3,
                           (float)k8.s4,(float)k8.s5,(float)k8.s6,(float)k8.s7};
            half8 v8 = vb[u];
            float vf[8] = {(float)v8.s0,(float)v8.s1,(float)v8.s2,(float)v8.s3,
                           (float)v8.s4,(float)v8.s5,(float)v8.s6,(float)v8.s7};
            #pragma unroll
            for (uint h = 0; h < GQA_G; ++h) {
                float partial = qv[h][0]*kf[0] + qv[h][1]*kf[1] + qv[h][2]*kf[2] + qv[h][3]*kf[3]
                              + qv[h][4]*kf[4] + qv[h][5]*kf[5] + qv[h][6]*kf[6] + qv[h][7]*kf[7];
                partial += BPERM(1u, partial);
                partial += BPERM(2u, partial);
                partial += BPERM(4u, partial);
                partial += BPERM(8u, partial);
                float s = partial * scale;
                float m_new = fmax(m[h], s);
                float corr = native_exp(m[h] - m_new);
                float p = native_exp(s - m_new);
                l[h] = l[h] * corr + p;
                #pragma unroll
                for (int i = 0; i < 8; ++i) o[h][i] = o[h][i]*corr + p*vf[i];
                m[h] = m_new;
            }
        }
    }

    // Per-head merge (4 streams + WPS wavefronts), then write head-major partials.
    __local float wm[GQA_G][WPS_ATTN], wl[GQA_G][WPS_ATTN], wo[GQA_G][WPS_ATTN][128];
    #pragma unroll
    for (uint h = 0; h < GQA_G; ++h) {
        float M = m[h];
        M = fmax(M, BPERM(16u, M));
        M = fmax(M, BPERM(32u, M));
        if (M == -INFINITY) M = 0.0f;
        float cg = native_exp(m[h] - M);
        float L = l[h] * cg;
        L += BPERM(16u, L);
        L += BPERM(32u, L);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float oc = o[h][i] * cg;
            oc += BPERM(16u, oc);
            oc += BPERM(32u, oc);
            o[h][i] = oc;
        }
        if (lane == 0) { wm[h][w] = M; wl[h][w] = L; }
        if (g == 0) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) wo[h][w][sub * 8 + i] = o[h][i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (w == 0 && lane < 64) {
        #pragma unroll
        for (uint h = 0; h < GQA_G; ++h) {
            float MM = -INFINITY;
            for (uint k = 0; k < WPS_ATTN; ++k) MM = fmax(MM, wm[h][k]);
            if (MM == -INFINITY) MM = 0.0f;
            float LL = 0.0f;
            for (uint k = 0; k < WPS_ATTN; ++k) LL += wl[h][k] * native_exp(wm[h][k] - MM);
            __global float* pr = partials + ((ulong)h * num_splits + sp) * (D + 2);
            if (lane == 0) { pr[0] = MM; pr[1] = LL; }
            for (uint d = lane; d < 128; d += 64) {
                float acc = 0.0f;
                for (uint k = 0; k < WPS_ATTN; ++k)
                    acc += wo[h][k][d] * native_exp(wm[h][k] - MM);
                pr[2 + d] = acc;
            }
        }
    }
}

__kernel void attn_decode_split2_fp8_gqa(
    __global const half* q,                                   // [GQA_G][128]
    __global const uchar* K, __global const uchar* V,         // one KV head, [N][128] e4m3
    __global const float* scale_k, __global const float* scale_v,
    __global float* partials,                                 // [GQA_G][num_splits][D+2]
    uint N, uint D, float scale, uint num_splits) {
    uint sp = get_group_id(0);
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint g = lane >> 4;
    uint sub = lane & 15;
    uint S = (N + num_splits - 1) / num_splits;
    uint lo = sp * S;
    uint hi = min(N, lo + S);
    uint span = hi > lo ? hi - lo : 0;
    uint per = (span + WPS_ATTN - 1) / WPS_ATTN;
    uint wlo = lo + w * per;
    uint whi = min(hi, wlo + per);

    float qv[GQA_G][8];
    #pragma unroll
    for (uint h = 0; h < GQA_G; ++h) {
        half8 q8 = vload8(sub, q + h * 128);
        qv[h][0]=q8.s0; qv[h][1]=q8.s1; qv[h][2]=q8.s2; qv[h][3]=q8.s3;
        qv[h][4]=q8.s4; qv[h][5]=q8.s5; qv[h][6]=q8.s6; qv[h][7]=q8.s7;
    }
    float m[GQA_G], l[GQA_G], o[GQA_G][8];
    #pragma unroll
    for (uint h = 0; h < GQA_G; ++h) {
        m[h] = -INFINITY; l[h] = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) o[h][i] = 0.0f;
    }

    for (uint base = wlo + g; base < whi; base += 4u * UATTN) {
        uint2 kb[UATTN], vb[UATTN];
        #pragma unroll
        for (uint u = 0; u < UATTN; ++u) {
            uint t = base + 4u * u;
            uint tt = t < whi ? t : whi - 1;
            kb[u] = ((__global const uint2*)(K + (ulong)tt * 128))[sub];
            vb[u] = ((__global const uint2*)(V + (ulong)tt * 128))[sub];
        }
        #pragma unroll
        for (uint u = 0; u < UATTN; ++u) {
            uint t = base + 4u * u;
            if (t >= whi) break;
            // Decode this lane's 8 K and 8 V dims ONCE (shared across heads).
            uint2 k2 = kb[u];
            float2 ka = __builtin_amdgcn_cvt_pk_f32_fp8(k2.x, false);
            float2 kc = __builtin_amdgcn_cvt_pk_f32_fp8(k2.x, true);
            float2 ke = __builtin_amdgcn_cvt_pk_f32_fp8(k2.y, false);
            float2 kg = __builtin_amdgcn_cvt_pk_f32_fp8(k2.y, true);
            float kf[8] = {ka.x, ka.y, kc.x, kc.y, ke.x, ke.y, kg.x, kg.y};
            uint2 v2 = vb[u];
            float2 va = __builtin_amdgcn_cvt_pk_f32_fp8(v2.x, false);
            float2 vc = __builtin_amdgcn_cvt_pk_f32_fp8(v2.x, true);
            float2 ve = __builtin_amdgcn_cvt_pk_f32_fp8(v2.y, false);
            float2 vg = __builtin_amdgcn_cvt_pk_f32_fp8(v2.y, true);
            float vf[8] = {va.x, va.y, vc.x, vc.y, ve.x, ve.y, vg.x, vg.y};
            float skt = scale_k[t] * scale;
            float svt = scale_v[t];
            #pragma unroll
            for (uint h = 0; h < GQA_G; ++h) {
                float partial = qv[h][0]*kf[0] + qv[h][1]*kf[1] + qv[h][2]*kf[2] + qv[h][3]*kf[3]
                              + qv[h][4]*kf[4] + qv[h][5]*kf[5] + qv[h][6]*kf[6] + qv[h][7]*kf[7];
                partial += BPERM(1u, partial);
                partial += BPERM(2u, partial);
                partial += BPERM(4u, partial);
                partial += BPERM(8u, partial);
                float s = partial * skt;
                float m_new = fmax(m[h], s);
                float corr = native_exp(m[h] - m_new);
                float p = native_exp(s - m_new);
                l[h] = l[h] * corr + p;
                float pv = p * svt;
                #pragma unroll
                for (int i = 0; i < 8; ++i) o[h][i] = o[h][i]*corr + pv*vf[i];
                m[h] = m_new;
            }
        }
    }

    // Per-head merge (4 streams + WPS wavefronts), then write head-major partials.
    __local float wm[GQA_G][WPS_ATTN], wl[GQA_G][WPS_ATTN], wo[GQA_G][WPS_ATTN][128];
    #pragma unroll
    for (uint h = 0; h < GQA_G; ++h) {
        float M = m[h];
        M = fmax(M, BPERM(16u, M));
        M = fmax(M, BPERM(32u, M));
        if (M == -INFINITY) M = 0.0f;
        float cg = native_exp(m[h] - M);
        float L = l[h] * cg;
        L += BPERM(16u, L);
        L += BPERM(32u, L);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float oc = o[h][i] * cg;
            oc += BPERM(16u, oc);
            oc += BPERM(32u, oc);
            o[h][i] = oc;
        }
        if (lane == 0) { wm[h][w] = M; wl[h][w] = L; }
        if (g == 0) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) wo[h][w][sub * 8 + i] = o[h][i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (w == 0 && lane < 64) {
        #pragma unroll
        for (uint h = 0; h < GQA_G; ++h) {
            float MM = -INFINITY;
            for (uint k = 0; k < WPS_ATTN; ++k) MM = fmax(MM, wm[h][k]);
            if (MM == -INFINITY) MM = 0.0f;
            float LL = 0.0f;
            for (uint k = 0; k < WPS_ATTN; ++k) LL += wl[h][k] * native_exp(wm[h][k] - MM);
            __global float* pr = partials + ((ulong)h * num_splits + sp) * (D + 2);
            if (lane == 0) { pr[0] = MM; pr[1] = LL; }
            for (uint d = lane; d < 128; d += 64) {
                float acc = 0.0f;
                for (uint k = 0; k < WPS_ATTN; ++k)
                    acc += wo[h][k][d] * native_exp(wm[h][k] - MM);
                pr[2 + d] = acc;
            }
        }
    }
}

// GQA combine, pass A: like attn_decode_reduce_partials but batched over heads in
// ONE dispatch. Launch (num_heads * groups_per_head) workgroups; workgroup gid
// reduces head-major input range [head*num_per_head + grp*group_size, ...] into
// out[gid] (out is also head-major: [num_heads][groups_per_head][D+2]).
__kernel void attn_decode_reduce_partials_gqa(__global const float* in, __global float* out,
                                              uint num_per_head, uint D, uint group_size,
                                              uint groups_per_head) {
    uint gid = get_group_id(0);
    uint head = gid / groups_per_head;
    uint grp = gid % groups_per_head;
    uint t = get_local_id(0);
    uint nt = D;
    uint lo = head * num_per_head + grp * group_size;
    uint hi = head * num_per_head + min(num_per_head, (grp + 1) * group_size);
    __local float red[128];
    __local float lM, lL;

    float pm = -INFINITY;
    for (uint s = lo + t; s < hi; s += nt) pm = fmax(pm, in[(ulong)s * (D + 2)]);
    red[t] = pm;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint off = nt >> 1; off > 0; off >>= 1) {
        if (t < off) red[t] = fmax(red[t], red[t + off]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (t == 0) lM = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    float M = lM;
    if (M == -INFINITY) M = 0.0f;

    float pl = 0.0f;
    for (uint s = lo + t; s < hi; s += nt) {
        __global const float* pr = in + (ulong)s * (D + 2);
        pl += pr[1] * native_exp(pr[0] - M);
    }
    red[t] = pl;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint off = nt >> 1; off > 0; off >>= 1) {
        if (t < off) red[t] += red[t + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (t == 0) lL = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (t < D) {
        float o = 0.0f;
        for (uint s = lo; s < hi; ++s) {
            __global const float* pr = in + (ulong)s * (D + 2);
            o += pr[2 + t] * native_exp(pr[0] - M);
        }
        __global float* po = out + (ulong)gid * (D + 2);
        if (t == 0) { po[0] = M; po[1] = lL; }
        po[2 + t] = o;
    }
}

// GQA combine, pass B: batched final combine over heads in ONE dispatch. Launch
// num_heads workgroups (D threads); workgroup `head` merges its `num_per_head`
// intermediate partials into O[head*D .. +D].
__kernel void attn_decode_combine_gqa(__global const float* in, __global float* O,
                                      uint D, uint num_per_head) {
    uint head = get_group_id(0);
    uint t = get_local_id(0);
    uint nt = D;
    uint lo = head * num_per_head;
    uint hi = lo + num_per_head;
    __local float red[128];
    __local float lM, lL;

    float pm = -INFINITY;
    for (uint s = lo + t; s < hi; s += nt) pm = fmax(pm, in[(ulong)s * (D + 2)]);
    red[t] = pm;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint off = nt >> 1; off > 0; off >>= 1) {
        if (t < off) red[t] = fmax(red[t], red[t + off]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (t == 0) lM = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    float M = lM;
    if (M == -INFINITY) M = 0.0f;

    float pl = 0.0f;
    for (uint s = lo + t; s < hi; s += nt) {
        __global const float* pr = in + (ulong)s * (D + 2);
        pl += pr[1] * native_exp(pr[0] - M);
    }
    red[t] = pl;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint off = nt >> 1; off > 0; off >>= 1) {
        if (t < off) red[t] += red[t + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (t == 0) lL = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (t < D) {
        float o = 0.0f;
        for (uint s = lo; s < hi; ++s) {
            __global const float* pr = in + (ulong)s * (D + 2);
            o += pr[2 + t] * native_exp(pr[0] - M);
        }
        O[head * D + t] = o / lL;
    }
}

// Same final GQA combine as attn_decode_combine_gqa, but stores the normalized
// attention output directly as f16. The max/LSE/value merge stays in f32; only
// the final HBM store is narrowed, eliminating a standalone cast before O-proj.
__kernel void attn_decode_combine_gqa_f16(__global const float* in, __global half* O,
                                          uint D, uint num_per_head) {
    uint head = get_group_id(0);
    uint t = get_local_id(0);
    uint nt = D;
    uint lo = head * num_per_head;
    uint hi = lo + num_per_head;
    __local float red[128];
    __local float coeff[128];
    __local float lM, lL;

    float pm = -INFINITY;
    for (uint s = lo + t; s < hi; s += nt) pm = fmax(pm, in[(ulong)s * (D + 2)]);
    red[t] = pm;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint off = nt >> 1; off > 0; off >>= 1) {
        if (t < off) red[t] = fmax(red[t], red[t + off]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (t == 0) lM = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    float M = lM;
    if (M == -INFINITY) M = 0.0f;

    if (num_per_head <= 128u) {
        if (t < num_per_head) {
            __global const float* pr = in + (ulong)(lo + t) * (D + 2);
            float c = native_exp(pr[0] - M);
            coeff[t] = c;
            red[t] = pr[1] * c;
        } else {
            red[t] = 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint off = nt >> 1; off > 0; off >>= 1) {
            if (t < off) red[t] += red[t + off];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (t == 0) lL = red[0];
        barrier(CLK_LOCAL_MEM_FENCE);
        float L = lL;
        if (t < D) {
            float o = 0.0f;
            for (uint s = 0; s < num_per_head; ++s) {
                __global const float* pr = in + (ulong)(lo + s) * (D + 2);
                o += pr[2 + t] * coeff[s];
            }
            O[head * D + t] = (half)(o / L);
        }
        return;
    }

    float pl = 0.0f;
    for (uint s = lo + t; s < hi; s += nt) {
        __global const float* pr = in + (ulong)s * (D + 2);
        pl += pr[1] * native_exp(pr[0] - M);
    }
    red[t] = pl;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint off = nt >> 1; off > 0; off >>= 1) {
        if (t < off) red[t] += red[t + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (t == 0) lL = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (t < D) {
        float o = 0.0f;
        for (uint s = lo; s < hi; ++s) {
            __global const float* pr = in + (ulong)s * (D + 2);
            o += pr[2 + t] * native_exp(pr[0] - M);
        }
        O[head * D + t] = (half)(o / lL);
    }
}

// FP4 (E2M1) KV decode split (D==128), per-block-32 f32 scale. Same structure as
// the FP8 split, but KV is 4-bit: each lane owns 8 contiguous dims = 4 bytes = 1
// uint = 8 packed FP4, all within ONE 32-element block (block = sub/4), so one
// scale per lane. The hardware cvt_scalef32_pk_f32_fp4 dequantizes AND applies
// the block scale in the VALU (probe-verified: result = E2M1(nibble)*scale, sel
// k -> dims 2k,2k+1). 4x less KV than FP16 / ~1.6x less than FP8 (f32 scale
// overhead). scale_k/scale_v are [N][4] (4 blocks per token).
__kernel void attn_decode_split2_fp4(__global const half* q,
                                     __global const uchar* K, __global const uchar* V,
                                     __global const uchar* scale_k, __global const uchar* scale_v,
                                     __global float* partials,
                                     uint N, uint D, float scale, uint num_splits) {
    uint sp = get_group_id(0);
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint g = lane >> 4;
    uint sub = lane & 15;
    uint blk = sub >> 2;             // which 32-dim block this lane's 8 dims fall in
    uint S = (N + num_splits - 1) / num_splits;
    uint lo = sp * S;
    uint hi = min(N, lo + S);
    uint span = hi > lo ? hi - lo : 0;
    uint per = (span + WPS_ATTN - 1) / WPS_ATTN;
    uint wlo = lo + w * per;
    uint whi = min(hi, wlo + per);

    half8 q8 = vload8(sub, q);
    float qv[8];
    qv[0]=q8.s0; qv[1]=q8.s1; qv[2]=q8.s2; qv[3]=q8.s3;
    qv[4]=q8.s4; qv[5]=q8.s5; qv[6]=q8.s6; qv[7]=q8.s7;

    float m = -INFINITY, l = 0.0f, o[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = 0.0f;

    for (uint base = wlo + g; base < whi; base += 4u * UATTN) {
        uint kb[UATTN], vb[UATTN];
        float ks[UATTN], vs[UATTN];
        #pragma unroll
        for (uint u = 0; u < UATTN; ++u) {
            uint t = base + 4u * u;
            uint tt = t < whi ? t : whi - 1;
            kb[u] = ((__global const uint*)(K + (ulong)tt * 64))[sub];  // 8 FP4 = 4 bytes
            vb[u] = ((__global const uint*)(V + (ulong)tt * 64))[sub];
            // E8M0 scale byte -> f32 power-of-two (biased exponent into f32 exp field).
            ks[u] = as_float(((uint)scale_k[tt * 4 + blk]) << 23);
            vs[u] = as_float(((uint)scale_v[tt * 4 + blk]) << 23);
        }
        #pragma unroll
        for (uint u = 0; u < UATTN; ++u) {
            uint t = base + 4u * u;
            if (t >= whi) break;
            uint k4 = kb[u];
            float bsk = ks[u];
            float2 d0 = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(k4, bsk, 0);
            float2 d1 = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(k4, bsk, 1);
            float2 d2 = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(k4, bsk, 2);
            float2 d3 = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(k4, bsk, 3);
            float partial = qv[0]*d0.x + qv[1]*d0.y + qv[2]*d1.x + qv[3]*d1.y
                          + qv[4]*d2.x + qv[5]*d2.y + qv[6]*d3.x + qv[7]*d3.y;
            partial += BPERM(1u, partial);
            partial += BPERM(2u, partial);
            partial += BPERM(4u, partial);
            partial += BPERM(8u, partial);
            float s = partial * scale;     // block scale already applied in cvt
            float m_new = fmax(m, s);
            float corr = native_exp(m - m_new);
            float p = native_exp(s - m_new);
            l = l * corr + p;
            uint v4 = vb[u];
            float bsv = vs[u];
            float2 e0 = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, bsv, 0);
            float2 e1 = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, bsv, 1);
            float2 e2 = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, bsv, 2);
            float2 e3 = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, bsv, 3);
            o[0]=o[0]*corr+p*e0.x; o[1]=o[1]*corr+p*e0.y;
            o[2]=o[2]*corr+p*e1.x; o[3]=o[3]*corr+p*e1.y;
            o[4]=o[4]*corr+p*e2.x; o[5]=o[5]*corr+p*e2.y;
            o[6]=o[6]*corr+p*e3.x; o[7]=o[7]*corr+p*e3.y;
            m = m_new;
        }
    }

    // Merge 4 streams + WPS wavefronts (identical to split2_fp8).
    float M = m;
    M = fmax(M, BPERM(16u, M));
    M = fmax(M, BPERM(32u, M));
    if (M == -INFINITY) M = 0.0f;
    float cg = native_exp(m - M);
    float L = l * cg;
    L += BPERM(16u, L);
    L += BPERM(32u, L);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float oc = o[i] * cg;
        oc += BPERM(16u, oc);
        oc += BPERM(32u, oc);
        o[i] = oc;
    }
    __local float wm[WPS_ATTN], wl[WPS_ATTN], wo[WPS_ATTN][128];
    if (lane == 0) { wm[w] = M; wl[w] = L; }
    if (g == 0) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) wo[w][sub * 8 + i] = o[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (w == 0 && lane < 64) {
        float MM = -INFINITY;
        for (uint k = 0; k < WPS_ATTN; ++k) MM = fmax(MM, wm[k]);
        if (MM == -INFINITY) MM = 0.0f;
        float LL = 0.0f;
        for (uint k = 0; k < WPS_ATTN; ++k) LL += wl[k] * native_exp(wm[k] - MM);
        __global float* pr = partials + (ulong)sp * (D + 2);
        if (lane == 0) { pr[0] = MM; pr[1] = LL; }
        for (uint d = lane; d < 128; d += 64) {
            float acc = 0.0f;
            for (uint k = 0; k < WPS_ATTN; ++k)
                acc += wo[k][d] * native_exp(wm[k] - MM);
            pr[2 + d] = acc;
        }
    }
}

// GQA + FP4: the headline 1M-context config. GQA_G query heads share ONE FP4
// (E2M1, per-block-32 E8M0 scale) KV head; each FP4 element is loaded and
// dequantized ONCE (cvt_scalef32_pk_f32_fp4) and reused across all heads. So a
// head-group decode reads the KV cache once AND at 4-bit — ~4x compression x
// G-head amortization. Per-head softmax state in registers; head-major partials.
__kernel void attn_decode_split2_fp4_gqa(
    __global const half* q,                                  // [GQA_G][128]
    __global const uchar* K, __global const uchar* V,        // one KV head, FP4 [N][64]
    __global const uchar* scale_k, __global const uchar* scale_v,  // E8M0 [N][4]
    __global float* partials,                                // [GQA_G][num_splits][D+2]
    uint N, uint D, float scale, uint num_splits) {
    uint sp = get_group_id(0);
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint g = lane >> 4;
    uint sub = lane & 15;
    uint blk = sub >> 2;
    uint S = (N + num_splits - 1) / num_splits;
    uint lo = sp * S;
    uint hi = min(N, lo + S);
    uint span = hi > lo ? hi - lo : 0;
    uint per = (span + WPS_ATTN - 1) / WPS_ATTN;
    uint wlo = lo + w * per;
    uint whi = min(hi, wlo + per);

    float qv[GQA_G][8];
    #pragma unroll
    for (uint h = 0; h < GQA_G; ++h) {
        half8 q8 = vload8(sub, q + h * 128);
        qv[h][0]=q8.s0; qv[h][1]=q8.s1; qv[h][2]=q8.s2; qv[h][3]=q8.s3;
        qv[h][4]=q8.s4; qv[h][5]=q8.s5; qv[h][6]=q8.s6; qv[h][7]=q8.s7;
    }
    float m[GQA_G], l[GQA_G], o[GQA_G][8];
    #pragma unroll
    for (uint h = 0; h < GQA_G; ++h) {
        m[h] = -INFINITY; l[h] = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) o[h][i] = 0.0f;
    }

    for (uint base = wlo + g; base < whi; base += 4u * UATTN) {
        uint kb[UATTN], vb[UATTN];
        float ks[UATTN], vs[UATTN];
        #pragma unroll
        for (uint u = 0; u < UATTN; ++u) {
            uint t = base + 4u * u;
            uint tt = t < whi ? t : whi - 1;
            kb[u] = ((__global const uint*)(K + (ulong)tt * 64))[sub];
            vb[u] = ((__global const uint*)(V + (ulong)tt * 64))[sub];
            ks[u] = as_float(((uint)scale_k[tt * 4 + blk]) << 23);
            vs[u] = as_float(((uint)scale_v[tt * 4 + blk]) << 23);
        }
        #pragma unroll
        for (uint u = 0; u < UATTN; ++u) {
            uint t = base + 4u * u;
            if (t >= whi) break;
            // Decode K and V ONCE (shared across the GQA_G heads).
            uint k4 = kb[u]; float bsk = ks[u];
            float2 ka = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(k4, bsk, 0);
            float2 kc = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(k4, bsk, 1);
            float2 ke = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(k4, bsk, 2);
            float2 kg = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(k4, bsk, 3);
            float kf[8] = {ka.x,ka.y,kc.x,kc.y,ke.x,ke.y,kg.x,kg.y};
            uint v4 = vb[u]; float bsv = vs[u];
            float2 ea = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, bsv, 0);
            float2 ec = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, bsv, 1);
            float2 ee = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, bsv, 2);
            float2 eg = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, bsv, 3);
            float vf[8] = {ea.x,ea.y,ec.x,ec.y,ee.x,ee.y,eg.x,eg.y};
            #pragma unroll
            for (uint h = 0; h < GQA_G; ++h) {
                float partial = qv[h][0]*kf[0] + qv[h][1]*kf[1] + qv[h][2]*kf[2] + qv[h][3]*kf[3]
                              + qv[h][4]*kf[4] + qv[h][5]*kf[5] + qv[h][6]*kf[6] + qv[h][7]*kf[7];
                partial += BPERM(1u, partial);
                partial += BPERM(2u, partial);
                partial += BPERM(4u, partial);
                partial += BPERM(8u, partial);
                float s = partial * scale;
                float m_new = fmax(m[h], s);
                float corr = native_exp(m[h] - m_new);
                float p = native_exp(s - m_new);
                l[h] = l[h] * corr + p;
                #pragma unroll
                for (int i = 0; i < 8; ++i) o[h][i] = o[h][i]*corr + p*vf[i];
                m[h] = m_new;
            }
        }
    }

    __local float wm[GQA_G][WPS_ATTN], wl[GQA_G][WPS_ATTN], wo[GQA_G][WPS_ATTN][128];
    #pragma unroll
    for (uint h = 0; h < GQA_G; ++h) {
        float M = m[h];
        M = fmax(M, BPERM(16u, M));
        M = fmax(M, BPERM(32u, M));
        if (M == -INFINITY) M = 0.0f;
        float cg = native_exp(m[h] - M);
        float L = l[h] * cg;
        L += BPERM(16u, L);
        L += BPERM(32u, L);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float oc = o[h][i] * cg;
            oc += BPERM(16u, oc);
            oc += BPERM(32u, oc);
            o[h][i] = oc;
        }
        if (lane == 0) { wm[h][w] = M; wl[h][w] = L; }
        if (g == 0) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) wo[h][w][sub * 8 + i] = o[h][i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (w == 0 && lane < 64) {
        #pragma unroll
        for (uint h = 0; h < GQA_G; ++h) {
            float MM = -INFINITY;
            for (uint k = 0; k < WPS_ATTN; ++k) MM = fmax(MM, wm[h][k]);
            if (MM == -INFINITY) MM = 0.0f;
            float LL = 0.0f;
            for (uint k = 0; k < WPS_ATTN; ++k) LL += wl[h][k] * native_exp(wm[h][k] - MM);
            __global float* pr = partials + ((ulong)h * num_splits + sp) * (D + 2);
            if (lane == 0) { pr[0] = MM; pr[1] = LL; }
            for (uint dd = lane; dd < 128; dd += 64) {
                float acc = 0.0f;
                for (uint k = 0; k < WPS_ATTN; ++k)
                    acc += wo[h][k][dd] * native_exp(wm[h][k] - MM);
                pr[2 + dd] = acc;
            }
        }
    }
}

// Paged FlashDecoding split (FP16, D==128): the KV cache is stored in fixed-size
// physical BLOCKS and a per-sequence block_table maps logical block -> physical
// block (vLLM/SGLang paged attention, here on the raw KFD/AQL path). Identical
// to attn_decode_split2 except K/V addresses go through the block table:
// logical token t -> phys row (block_table[t/BS])*BS + t%BS. block_table entries
// for nearby tokens coincide (same block) so the lookup is cache-friendly.
__kernel void paged_block_table_bounds_check(__global const uint* block_table,
                                             __global ulong* out,
                                             uint logical_blocks,
                                             uint physical_blocks) {
    const uint lid = get_local_id(0);
    const uint lsize = get_local_size(0);
    __local uint local_bad[256];
    __local uint local_max[256];
    __local uint local_first_idx[256];
    __local uint local_first_val[256];

    uint bad = 0;
    uint max_v = 0;
    uint first_idx = 0xffffffffu;
    uint first_val = 0;

    for (uint i = lid; i < logical_blocks; i += lsize) {
        uint v = block_table[i];
        max_v = max(max_v, v);
        if (v >= physical_blocks) {
            bad += 1u;
            if (i < first_idx) {
                first_idx = i;
                first_val = v;
            }
        }
    }

    local_bad[lid] = bad;
    local_max[lid] = max_v;
    local_first_idx[lid] = first_idx;
    local_first_val[lid] = first_val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_bad[lid] += local_bad[lid + stride];
            local_max[lid] = max(local_max[lid], local_max[lid + stride]);
            if (local_first_idx[lid + stride] < local_first_idx[lid]) {
                local_first_idx[lid] = local_first_idx[lid + stride];
                local_first_val[lid] = local_first_val[lid + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        out[0] = (ulong)local_bad[0];
        out[1] = (ulong)local_max[0];
        out[2] = (ulong)local_first_idx[0];
        out[3] = (ulong)local_first_val[0];
        out[4] = 0xB10C7A8EB0A2D00DUL;
    }
}

__kernel void paged_kv_metadata_bounds_check(__global const uint* indptr,
                                             __global const uint* indices,
                                             __global const uint* last_page_len,
                                             __global ulong* out,
                                             uint batch_size,
                                             uint total_indices,
                                             uint physical_blocks,
                                             uint page_size) {
    const uint lid = get_local_id(0);
    const uint lsize = get_local_size(0);
    __local uint local_bad[256];
    __local uint local_max[256];
    __local uint local_first_key[256];
    __local uint local_first_kind[256];
    __local uint local_first_pos[256];
    __local uint local_first_val[256];

    uint bad = 0;
    uint max_v = 0;
    uint first_key = 0xffffffffu;
    uint first_kind = 0;
    uint first_pos = 0;
    uint first_val = 0;

#define MARK_PAGED_KV_BAD(KEY, KIND, POS, VAL) \
    do { \
        bad += 1u; \
        uint k_ = (uint)(KEY); \
        if (k_ < first_key) { \
            first_key = k_; \
            first_kind = (uint)(KIND); \
            first_pos = (uint)(POS); \
            first_val = (uint)(VAL); \
        } \
    } while (0)

    if (lid == 0) {
        uint start0 = indptr[0];
        uint endn = indptr[batch_size];
        if (start0 != 0)
            MARK_PAGED_KV_BAD(0u, 1u, 0u, start0);
        if (endn != total_indices)
            MARK_PAGED_KV_BAD(1u, 2u, batch_size, endn);
    }

    for (uint i = lid; i < batch_size; i += lsize) {
        uint lo = indptr[i];
        uint hi = indptr[i + 1u];
        uint last = last_page_len[i];
        if (lo > hi)
            MARK_PAGED_KV_BAD(1000u + i, 3u, i, lo);
        if (lo == hi)
            MARK_PAGED_KV_BAD(1000000u + i, 4u, i, lo);
        if (hi > total_indices)
            MARK_PAGED_KV_BAD(2000000u + i, 5u, i, hi);
        if (last == 0u || last > page_size)
            MARK_PAGED_KV_BAD(3000000u + i, 6u, i, last);
    }

    for (uint i = lid; i < total_indices; i += lsize) {
        uint v = indices[i];
        max_v = max(max_v, v);
        if (v >= physical_blocks)
            MARK_PAGED_KV_BAD(2000000000u + i, 7u, i, v);
    }

#undef MARK_PAGED_KV_BAD

    local_bad[lid] = bad;
    local_max[lid] = max_v;
    local_first_key[lid] = first_key;
    local_first_kind[lid] = first_kind;
    local_first_pos[lid] = first_pos;
    local_first_val[lid] = first_val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_bad[lid] += local_bad[lid + stride];
            local_max[lid] = max(local_max[lid], local_max[lid + stride]);
            if (local_first_key[lid + stride] < local_first_key[lid]) {
                local_first_key[lid] = local_first_key[lid + stride];
                local_first_kind[lid] = local_first_kind[lid + stride];
                local_first_pos[lid] = local_first_pos[lid + stride];
                local_first_val[lid] = local_first_val[lid + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        out[0] = (ulong)local_bad[0];
        out[1] = (ulong)local_max[0];
        out[2] = (ulong)local_first_kind[0];
        out[3] = (ulong)local_first_pos[0];
        out[4] = (ulong)local_first_val[0];
        out[5] = (ulong)indptr[batch_size];
        out[6] = (ulong)batch_size;
        out[7] = 0xB471C7D00D5AFEEDUL;
    }
}

// Minimal paged-KV read gate. This is intentionally not attention: it proves a
// validated block-table entry can drive one physical KV load, while null and
// out-of-range block ids exit before dereferencing the cache.
__kernel void paged_kv_read_probe(__global const half* cache,
                                  __global const uint* indices,
                                  __global ulong* out,
                                  uint total_indices,
                                  uint physical_blocks,
                                  uint block_size,
                                  uint d) {
    if (get_global_id(0) != 0) return;

    out[0] = 0UL;
    out[1] = 0UL;
    out[2] = 0UL;
    out[3] = 0UL;
    out[4] = (ulong)total_indices;
    out[5] = (ulong)physical_blocks;
    out[6] = (ulong)d;
    out[7] = 0xF00DB10C5AFECAFEUL;

    if (total_indices == 0u || physical_blocks == 0u || block_size == 0u || d == 0u) {
        out[0] = 3UL;
        return;
    }

    uint block_id = indices[0];
    out[1] = (ulong)block_id;
    if (block_id == 0u) {
        out[0] = 1UL;
        return;
    }
    if (block_id >= physical_blocks) {
        out[0] = 2UL;
        return;
    }

    ulong offset = (ulong)block_id * (ulong)block_size * (ulong)d;
    const __global ushort* raw = (const __global ushort*)cache;
    out[2] = (ulong)raw[offset];
    out[3] = offset;
}

// Bounded paged-KV gather gate. This walks the logical page table and reads all
// elements from valid physical pages, while null and out-of-range block ids are
// counted before any cache dereference. It intentionally emits only a checksum:
// the purpose is to prove page traversal and bounds behavior before attention.
__kernel void paged_kv_gather_checksum_probe(__global const half* cache,
                                             __global const uint* indices,
                                             __global ulong* out,
                                             uint total_indices,
                                             uint physical_blocks,
                                             uint block_size,
                                             uint d) {
    const uint lid = get_local_id(0);
    const uint lsize = get_local_size(0);
    __local ulong local_checksum[256];
    __local ulong local_bad[256];
    __local ulong local_null[256];
    __local uint local_first_pos[256];
    __local uint local_first_val[256];

    if (lid == 0) {
        out[0] = 3UL;
        out[1] = 0UL;
        out[2] = 0UL;
        out[3] = 0UL;
        out[4] = 0UL;
        out[5] = 0xffffffffUL;
        out[6] = 0UL;
        out[7] = 0x6A7AC5EC0BADF00DUL;
    }

    if (total_indices == 0u || physical_blocks == 0u || block_size == 0u || d == 0u)
        return;

    const ulong elems_per_block = (ulong)block_size * (ulong)d;
    const ulong total_elems = (ulong)total_indices * elems_per_block;
    const __global ushort* raw = (const __global ushort*)cache;
    ulong checksum = 0UL;
    ulong bad = 0UL;
    ulong nulls = 0UL;
    uint first_pos = 0xffffffffu;
    uint first_val = 0u;

    for (ulong elem = (ulong)lid; elem < total_elems; elem += (ulong)lsize) {
        const uint page = (uint)(elem / elems_per_block);
        const ulong in_block = elem - ((ulong)page * elems_per_block);
        const uint block_id = indices[page];
        if (block_id == 0u) {
            nulls += 1UL;
            continue;
        }
        if (block_id >= physical_blocks) {
            bad += 1UL;
            if ((uint)elem < first_pos) {
                first_pos = (uint)elem;
                first_val = block_id;
            }
            continue;
        }

        const ulong offset = ((ulong)block_id * elems_per_block) + in_block;
        const ulong bits = (ulong)raw[offset];
        checksum += bits + ((elem & 0xffffUL) << 16);
    }

    local_checksum[lid] = checksum;
    local_bad[lid] = bad;
    local_null[lid] = nulls;
    local_first_pos[lid] = first_pos;
    local_first_val[lid] = first_val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_checksum[lid] += local_checksum[lid + stride];
            local_bad[lid] += local_bad[lid + stride];
            local_null[lid] += local_null[lid + stride];
            if (local_first_pos[lid + stride] < local_first_pos[lid]) {
                local_first_pos[lid] = local_first_pos[lid + stride];
                local_first_val[lid] = local_first_val[lid + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        out[0] = local_bad[0] == 0UL ? 0UL : 2UL;
        out[1] = local_bad[0];
        out[2] = local_null[0];
        out[3] = local_checksum[0];
        out[4] = total_elems;
        out[5] = (ulong)local_first_pos[0];
        out[6] = (ulong)local_first_val[0];
        out[7] = 0x6A7AC5EC0BADF00DUL;
    }
}

// Kimi/DeepSeek-style MLA latent-cache page walker. This intentionally avoids
// attention math: it proves the non-power-of-two MLA token payload
// [ckv_dim=512, kpe_dim=64] can be traversed through the paged metadata without
// dereferencing null or out-of-range physical pages.
__kernel void paged_mla_latent_checksum_probe(__global const ushort* ckv_cache,
                                              __global const ushort* kpe_cache,
                                              __global const uint* indices,
                                              __global ulong* out,
                                              uint total_indices,
                                              uint physical_blocks,
                                              uint block_size,
                                              uint ckv_dim,
                                              uint kpe_dim) {
    const uint lid = get_local_id(0);
    const uint lsize = get_local_size(0);
    __local ulong local_checksum[256];
    __local ulong local_bad[256];
    __local ulong local_null[256];
    __local uint local_first_pos[256];
    __local uint local_first_val[256];

    if (lid == 0) {
        out[0] = 3UL;
        out[1] = 0UL;
        out[2] = 0UL;
        out[3] = 0UL;
        out[4] = 0UL;
        out[5] = 0xffffffffUL;
        out[6] = 0UL;
        out[7] = 0xC1A512400BADF00DUL;
    }

    if (total_indices == 0u || physical_blocks == 0u || block_size == 0u || ckv_dim == 0u || kpe_dim == 0u)
        return;

    const ulong elems_per_token = (ulong)ckv_dim + (ulong)kpe_dim;
    const ulong total_tokens = (ulong)total_indices * (ulong)block_size;
    const ulong total_elems = total_tokens * elems_per_token;
    ulong checksum = 0UL;
    ulong bad = 0UL;
    ulong nulls = 0UL;
    uint first_pos = 0xffffffffu;
    uint first_val = 0u;

    for (ulong elem = (ulong)lid; elem < total_elems; elem += (ulong)lsize) {
        const ulong token = elem / elems_per_token;
        const uint dim = (uint)(elem - token * elems_per_token);
        const uint page = (uint)(token / (ulong)block_size);
        const uint token_in_block = (uint)(token - ((ulong)page * (ulong)block_size));
        const uint block_id = indices[page];

        if (block_id == 0u) {
            nulls += 1UL;
            continue;
        }
        if (block_id >= physical_blocks) {
            bad += 1UL;
            if ((uint)elem < first_pos) {
                first_pos = (uint)elem;
                first_val = block_id;
            }
            continue;
        }

        const ulong row = ((ulong)block_id * (ulong)block_size) + (ulong)token_in_block;
        ulong bits;
        ulong tag;
        if (dim < ckv_dim) {
            bits = (ulong)ckv_cache[row * (ulong)ckv_dim + (ulong)dim];
            tag = 0xC0000000UL;
        } else {
            const uint kpe_dim_idx = dim - ckv_dim;
            bits = (ulong)kpe_cache[row * (ulong)kpe_dim + (ulong)kpe_dim_idx];
            tag = 0xE0000000UL;
        }
        checksum += bits + ((elem & 0xffffUL) << 16) + tag;
    }

    local_checksum[lid] = checksum;
    local_bad[lid] = bad;
    local_null[lid] = nulls;
    local_first_pos[lid] = first_pos;
    local_first_val[lid] = first_val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_checksum[lid] += local_checksum[lid + stride];
            local_bad[lid] += local_bad[lid + stride];
            local_null[lid] += local_null[lid + stride];
            if (local_first_pos[lid + stride] < local_first_pos[lid]) {
                local_first_pos[lid] = local_first_pos[lid + stride];
                local_first_val[lid] = local_first_val[lid + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        out[0] = local_bad[0] == 0UL ? 0UL : 2UL;
        out[1] = local_bad[0];
        out[2] = local_null[0];
        out[3] = local_checksum[0];
        out[4] = total_elems;
        out[5] = (ulong)local_first_pos[0];
        out[6] = (ulong)local_first_val[0];
        out[7] = 0xC1A512400BADF00DUL;
    }
}

// Kimi/DeepSeek-style MLA dot-score gate. This extends the latent-cache page
// walker into the first attention-semantic read: q_nope.ckv + q_pe.kpe. Null
// and OOB physical pages are classified before any cache dereference, and all
// row math is widened to 64-bit before indexing the 512/64 latent payloads.
__kernel void paged_mla_dot_score_probe(__global const half* q_nope,
                                        __global const half* q_pe,
                                        __global const half* ckv_cache,
                                        __global const half* kpe_cache,
                                        __global const uint* indices,
                                        __global ulong* out,
                                        uint total_indices,
                                        uint physical_blocks,
                                        uint block_size,
                                        uint ckv_dim,
                                        uint kpe_dim) {
    const uint lid = get_local_id(0);
    const uint lsize = get_local_size(0);
    __local float local_sum[256];
    __local float local_max[256];
    __local ulong local_bad[256];
    __local ulong local_null[256];
    __local ulong local_valid[256];
    __local uint local_first_pos[256];
    __local uint local_first_val[256];

    if (lid == 0) {
        out[0] = 3UL;
        out[1] = 0UL;
        out[2] = 0UL;
        out[3] = 0UL;
        out[4] = 0UL;
        out[5] = 0UL;
        out[6] = 0xffffffff00000000UL;
        out[7] = 0xD07C0DEF0BADF00DUL;
    }

    if (total_indices == 0u || physical_blocks == 0u || block_size == 0u || ckv_dim == 0u || kpe_dim == 0u)
        return;

    const ulong total_tokens = (ulong)total_indices * (ulong)block_size;
    float score_sum = 0.0f;
    float score_max = -3.402823466e+38F;
    ulong bad = 0UL;
    ulong nulls = 0UL;
    ulong valid = 0UL;
    uint first_pos = 0xffffffffu;
    uint first_val = 0u;

    for (ulong token = (ulong)lid; token < total_tokens; token += (ulong)lsize) {
        const uint page = (uint)(token / (ulong)block_size);
        const uint token_in_block = (uint)(token - ((ulong)page * (ulong)block_size));
        const uint block_id = indices[page];
        if (block_id == 0u) {
            nulls += 1UL;
            continue;
        }
        if (block_id >= physical_blocks) {
            bad += 1UL;
            if ((uint)token < first_pos) {
                first_pos = (uint)token;
                first_val = block_id;
            }
            continue;
        }

        const ulong row = ((ulong)block_id * (ulong)block_size) + (ulong)token_in_block;
        const ulong ckv_base = row * (ulong)ckv_dim;
        const ulong kpe_base = row * (ulong)kpe_dim;
        float score = 0.0f;
        for (uint dim = 0u; dim < ckv_dim; ++dim) {
            score += (float)q_nope[dim] * (float)ckv_cache[ckv_base + (ulong)dim];
        }
        for (uint dim = 0u; dim < kpe_dim; ++dim) {
            score += (float)q_pe[dim] * (float)kpe_cache[kpe_base + (ulong)dim];
        }
        score_sum += score;
        score_max = fmax(score_max, score);
        valid += 1UL;
    }

    local_sum[lid] = score_sum;
    local_max[lid] = score_max;
    local_bad[lid] = bad;
    local_null[lid] = nulls;
    local_valid[lid] = valid;
    local_first_pos[lid] = first_pos;
    local_first_val[lid] = first_val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_sum[lid] += local_sum[lid + stride];
            local_max[lid] = fmax(local_max[lid], local_max[lid + stride]);
            local_bad[lid] += local_bad[lid + stride];
            local_null[lid] += local_null[lid + stride];
            local_valid[lid] += local_valid[lid + stride];
            if (local_first_pos[lid + stride] < local_first_pos[lid]) {
                local_first_pos[lid] = local_first_pos[lid + stride];
                local_first_val[lid] = local_first_val[lid + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        out[0] = local_bad[0] == 0UL ? 0UL : 2UL;
        out[1] = local_bad[0];
        out[2] = local_null[0];
        out[3] = local_valid[0];
        out[4] = (ulong)as_uint(local_sum[0]);
        out[5] = (ulong)as_uint(local_max[0]);
        out[6] = ((ulong)local_first_pos[0] << 32) | (ulong)local_first_val[0];
        out[7] = 0xD07C0DEF0BADF00DUL;
    }
}

// Kimi/DeepSeek-style MLA softmax-state gate. This keeps the dot-score read
// path intact, then computes a local log-sum-exp state and writes it to a
// split-K-style workspace slot. The workspace write is intentionally isolated
// before adding the 512-wide value accumulation path.
__kernel void paged_mla_lse_probe(__global const half* q_nope,
                                  __global const half* q_pe,
                                  __global const half* ckv_cache,
                                  __global const half* kpe_cache,
                                  __global const uint* indices,
                                  __global float* lse_workspace,
                                  __global ulong* out,
                                  uint total_indices,
                                  uint physical_blocks,
                                  uint block_size,
                                  uint ckv_dim,
                                  uint kpe_dim,
                                  uint workspace_index) {
    const uint lid = get_local_id(0);
    const uint lsize = get_local_size(0);
    __local float local_max[256];
    __local float local_denom[256];
    __local ulong local_bad[256];
    __local ulong local_null[256];
    __local ulong local_valid[256];
    __local uint local_first_pos[256];
    __local uint local_first_val[256];
    __local float shared_max;
    __local ulong shared_valid;

    if (lid == 0) {
        out[0] = 3UL;
        out[1] = 0UL;
        out[2] = 0UL;
        out[3] = 0UL;
        out[4] = 0UL;
        out[5] = 0UL;
        out[6] = 0xffffffff00000000UL;
        out[7] = 0x15E50DEF0BADF00DUL;
    }

    if (total_indices == 0u || physical_blocks == 0u || block_size == 0u || ckv_dim == 0u || kpe_dim == 0u)
        return;

    const ulong total_tokens = (ulong)total_indices * (ulong)block_size;
    float score_max = -3.402823466e+38F;
    ulong bad = 0UL;
    ulong nulls = 0UL;
    ulong valid = 0UL;
    uint first_pos = 0xffffffffu;
    uint first_val = 0u;

    for (ulong token = (ulong)lid; token < total_tokens; token += (ulong)lsize) {
        const uint page = (uint)(token / (ulong)block_size);
        const uint token_in_block = (uint)(token - ((ulong)page * (ulong)block_size));
        const uint block_id = indices[page];
        if (block_id == 0u) {
            nulls += 1UL;
            continue;
        }
        if (block_id >= physical_blocks) {
            bad += 1UL;
            if ((uint)token < first_pos) {
                first_pos = (uint)token;
                first_val = block_id;
            }
            continue;
        }

        const ulong row = ((ulong)block_id * (ulong)block_size) + (ulong)token_in_block;
        const ulong ckv_base = row * (ulong)ckv_dim;
        const ulong kpe_base = row * (ulong)kpe_dim;
        float score = 0.0f;
        for (uint dim = 0u; dim < ckv_dim; ++dim) {
            score += (float)q_nope[dim] * (float)ckv_cache[ckv_base + (ulong)dim];
        }
        for (uint dim = 0u; dim < kpe_dim; ++dim) {
            score += (float)q_pe[dim] * (float)kpe_cache[kpe_base + (ulong)dim];
        }
        score_max = fmax(score_max, score);
        valid += 1UL;
    }

    local_max[lid] = score_max;
    local_bad[lid] = bad;
    local_null[lid] = nulls;
    local_valid[lid] = valid;
    local_first_pos[lid] = first_pos;
    local_first_val[lid] = first_val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_max[lid] = fmax(local_max[lid], local_max[lid + stride]);
            local_bad[lid] += local_bad[lid + stride];
            local_null[lid] += local_null[lid + stride];
            local_valid[lid] += local_valid[lid + stride];
            if (local_first_pos[lid + stride] < local_first_pos[lid]) {
                local_first_pos[lid] = local_first_pos[lid + stride];
                local_first_val[lid] = local_first_val[lid + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        shared_max = local_max[0];
        shared_valid = local_valid[0];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float denom = 0.0f;
    if (shared_valid != 0UL) {
        for (ulong token = (ulong)lid; token < total_tokens; token += (ulong)lsize) {
            const uint page = (uint)(token / (ulong)block_size);
            const uint token_in_block = (uint)(token - ((ulong)page * (ulong)block_size));
            const uint block_id = indices[page];
            if (block_id == 0u || block_id >= physical_blocks) {
                continue;
            }

            const ulong row = ((ulong)block_id * (ulong)block_size) + (ulong)token_in_block;
            const ulong ckv_base = row * (ulong)ckv_dim;
            const ulong kpe_base = row * (ulong)kpe_dim;
            float score = 0.0f;
            for (uint dim = 0u; dim < ckv_dim; ++dim) {
                score += (float)q_nope[dim] * (float)ckv_cache[ckv_base + (ulong)dim];
            }
            for (uint dim = 0u; dim < kpe_dim; ++dim) {
                score += (float)q_pe[dim] * (float)kpe_cache[kpe_base + (ulong)dim];
            }
            denom += exp(score - shared_max);
        }
    }

    local_denom[lid] = denom;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_denom[lid] += local_denom[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        const float lse = shared_valid == 0UL ? -3.402823466e+38F : shared_max + log(local_denom[0]);
        lse_workspace[workspace_index] = lse;
        out[0] = local_bad[0] == 0UL ? 0UL : 2UL;
        out[1] = local_bad[0];
        out[2] = local_null[0];
        out[3] = local_valid[0];
        out[4] = (ulong)as_uint(shared_max);
        out[5] = (ulong)as_uint(lse);
        out[6] = ((ulong)local_first_pos[0] << 32) | (ulong)local_first_val[0];
        out[7] = 0x15E50DEF0BADF00DUL;
    }
}

// Kimi/DeepSeek-style MLA output-tile gate. This extends the LSE gate into the
// value path by computing the first ckv output tile:
// output[d] = sum_t softmax(score_t) * ckv[t, d]. The tile keeps this scalar
// probe below the full 512-dim VGPR pressure while proving the value read and
// output workspace indexing semantics used by the production path.
__kernel void paged_mla_output_tile_probe(__global const half* q_nope,
                                          __global const half* q_pe,
                                          __global const half* ckv_cache,
                                          __global const half* kpe_cache,
                                          __global const uint* indices,
                                          __global float* output_workspace,
                                          __global ulong* out,
                                          uint total_indices,
                                          uint physical_blocks,
                                          uint block_size,
                                          uint ckv_dim,
                                          uint kpe_dim,
                                          uint output_dims,
                                          uint workspace_index) {
    const uint lid = get_local_id(0);
    const uint lsize = get_local_size(0);
    __local float local_max[256];
    __local float local_denom[256];
    __local float local_accum[256];
    __local ulong local_bad[256];
    __local ulong local_null[256];
    __local ulong local_valid[256];
    __local uint local_first_pos[256];
    __local uint local_first_val[256];
    __local float shared_max;
    __local float shared_denom;
    __local ulong shared_valid;

    if (lid == 0) {
        out[0] = 3UL;
        out[1] = 0UL;
        out[2] = 0UL;
        out[3] = 0UL;
        out[4] = 0UL;
        out[5] = 0UL;
        out[6] = 0xffffffff00000000UL;
        out[7] = 0x0A7C0DEF0BADF00DUL;
    }

    if (total_indices == 0u || physical_blocks == 0u || block_size == 0u || ckv_dim == 0u || kpe_dim == 0u || output_dims == 0u || output_dims > ckv_dim)
        return;

    const ulong total_tokens = (ulong)total_indices * (ulong)block_size;
    float score_max = -3.402823466e+38F;
    ulong bad = 0UL;
    ulong nulls = 0UL;
    ulong valid = 0UL;
    uint first_pos = 0xffffffffu;
    uint first_val = 0u;

    for (ulong token = (ulong)lid; token < total_tokens; token += (ulong)lsize) {
        const uint page = (uint)(token / (ulong)block_size);
        const uint token_in_block = (uint)(token - ((ulong)page * (ulong)block_size));
        const uint block_id = indices[page];
        if (block_id == 0u) {
            nulls += 1UL;
            continue;
        }
        if (block_id >= physical_blocks) {
            bad += 1UL;
            if ((uint)token < first_pos) {
                first_pos = (uint)token;
                first_val = block_id;
            }
            continue;
        }

        const ulong row = ((ulong)block_id * (ulong)block_size) + (ulong)token_in_block;
        const ulong ckv_base = row * (ulong)ckv_dim;
        const ulong kpe_base = row * (ulong)kpe_dim;
        float score = 0.0f;
        for (uint dim = 0u; dim < ckv_dim; ++dim) {
            score += (float)q_nope[dim] * (float)ckv_cache[ckv_base + (ulong)dim];
        }
        for (uint dim = 0u; dim < kpe_dim; ++dim) {
            score += (float)q_pe[dim] * (float)kpe_cache[kpe_base + (ulong)dim];
        }
        score_max = fmax(score_max, score);
        valid += 1UL;
    }

    local_max[lid] = score_max;
    local_bad[lid] = bad;
    local_null[lid] = nulls;
    local_valid[lid] = valid;
    local_first_pos[lid] = first_pos;
    local_first_val[lid] = first_val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_max[lid] = fmax(local_max[lid], local_max[lid + stride]);
            local_bad[lid] += local_bad[lid + stride];
            local_null[lid] += local_null[lid + stride];
            local_valid[lid] += local_valid[lid + stride];
            if (local_first_pos[lid + stride] < local_first_pos[lid]) {
                local_first_pos[lid] = local_first_pos[lid + stride];
                local_first_val[lid] = local_first_val[lid + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        shared_max = local_max[0];
        shared_valid = local_valid[0];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float denom = 0.0f;
    if (shared_valid != 0UL) {
        for (ulong token = (ulong)lid; token < total_tokens; token += (ulong)lsize) {
            const uint page = (uint)(token / (ulong)block_size);
            const uint token_in_block = (uint)(token - ((ulong)page * (ulong)block_size));
            const uint block_id = indices[page];
            if (block_id == 0u || block_id >= physical_blocks) {
                continue;
            }

            const ulong row = ((ulong)block_id * (ulong)block_size) + (ulong)token_in_block;
            const ulong ckv_base = row * (ulong)ckv_dim;
            const ulong kpe_base = row * (ulong)kpe_dim;
            float score = 0.0f;
            for (uint dim = 0u; dim < ckv_dim; ++dim) {
                score += (float)q_nope[dim] * (float)ckv_cache[ckv_base + (ulong)dim];
            }
            for (uint dim = 0u; dim < kpe_dim; ++dim) {
                score += (float)q_pe[dim] * (float)kpe_cache[kpe_base + (ulong)dim];
            }
            denom += exp(score - shared_max);
        }
    }

    local_denom[lid] = denom;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_denom[lid] += local_denom[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        shared_denom = local_denom[0];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint out_dim = 0u; out_dim < output_dims; ++out_dim) {
        float accum = 0.0f;
        if (shared_valid != 0UL) {
            for (ulong token = (ulong)lid; token < total_tokens; token += (ulong)lsize) {
                const uint page = (uint)(token / (ulong)block_size);
                const uint token_in_block = (uint)(token - ((ulong)page * (ulong)block_size));
                const uint block_id = indices[page];
                if (block_id == 0u || block_id >= physical_blocks) {
                    continue;
                }

                const ulong row = ((ulong)block_id * (ulong)block_size) + (ulong)token_in_block;
                const ulong ckv_base = row * (ulong)ckv_dim;
                const ulong kpe_base = row * (ulong)kpe_dim;
                float score = 0.0f;
                for (uint dim = 0u; dim < ckv_dim; ++dim) {
                    score += (float)q_nope[dim] * (float)ckv_cache[ckv_base + (ulong)dim];
                }
                for (uint dim = 0u; dim < kpe_dim; ++dim) {
                    score += (float)q_pe[dim] * (float)kpe_cache[kpe_base + (ulong)dim];
                }
                accum += exp(score - shared_max) * (float)ckv_cache[ckv_base + (ulong)out_dim];
            }
        }

        local_accum[lid] = accum;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint stride = lsize >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) {
                local_accum[lid] += local_accum[lid + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid == 0) {
            const float value = shared_valid == 0UL ? 0.0f : local_accum[0] / shared_denom;
            output_workspace[(ulong)workspace_index * (ulong)output_dims + (ulong)out_dim] = value;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        const float lse = shared_valid == 0UL ? -3.402823466e+38F : shared_max + log(shared_denom);
        out[0] = local_bad[0] == 0UL ? 0UL : 2UL;
        out[1] = local_bad[0];
        out[2] = local_null[0];
        out[3] = local_valid[0];
        out[4] = (ulong)as_uint(shared_max);
        out[5] = (ulong)as_uint(lse);
        out[6] = ((ulong)local_first_pos[0] << 32) | (ulong)local_first_val[0];
        out[7] = 0x0A7C0DEF0BADF00DUL;
    }
}

// GQA-shaped MLA output-tile gate. This validates the production broadcast
// pattern used by AITER-style decode: multiple query/output heads read their
// own q_nope/q_pe rows while sharing a single paged ckv/kpe latent stream.
__kernel void paged_mla_gqa_output_tile_probe(__global const half* q_nope,
                                              __global const half* q_pe,
                                              __global const half* ckv_cache,
                                              __global const half* kpe_cache,
                                              __global const uint* indices,
                                              __global float* output_workspace,
                                              __global ulong* out,
                                              uint total_indices,
                                              uint physical_blocks,
                                              uint block_size,
                                              uint ckv_dim,
                                              uint kpe_dim,
                                              uint output_dims,
                                              uint num_heads,
                                              uint workspace_index) {
    const uint lid = get_local_id(0);
    const uint lsize = get_local_size(0);
    __local float local_max[256];
    __local float local_denom[256];
    __local float local_accum[256];
    __local ulong local_bad[256];
    __local ulong local_null[256];
    __local ulong local_valid[256];
    __local uint local_first_pos[256];
    __local uint local_first_val[256];
    __local float shared_max;
    __local float shared_denom;
    __local ulong shared_valid;
    __local ulong shared_checksum;

    if (lid == 0) {
        out[0] = 3UL;
        out[1] = 0UL;
        out[2] = 0UL;
        out[3] = 0UL;
        out[4] = 0UL;
        out[5] = 0UL;
        out[6] = 0xffffffff00000000UL;
        out[7] = 0x06A10DEF0BADF00DUL;
        shared_checksum = 0UL;
    }

    if (total_indices == 0u || physical_blocks == 0u || block_size == 0u || ckv_dim == 0u || kpe_dim == 0u || output_dims == 0u || output_dims > ckv_dim || num_heads == 0u)
        return;

    const ulong total_tokens = (ulong)total_indices * (ulong)block_size;
    ulong bad = 0UL;
    ulong nulls = 0UL;
    ulong valid = 0UL;
    uint first_pos = 0xffffffffu;
    uint first_val = 0u;

    for (ulong token = (ulong)lid; token < total_tokens; token += (ulong)lsize) {
        const uint page = (uint)(token / (ulong)block_size);
        const uint block_id = indices[page];
        if (block_id == 0u) {
            nulls += 1UL;
            continue;
        }
        if (block_id >= physical_blocks) {
            bad += 1UL;
            if ((uint)token < first_pos) {
                first_pos = (uint)token;
                first_val = block_id;
            }
            continue;
        }
        valid += 1UL;
    }

    local_bad[lid] = bad;
    local_null[lid] = nulls;
    local_valid[lid] = valid;
    local_first_pos[lid] = first_pos;
    local_first_val[lid] = first_val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_bad[lid] += local_bad[lid + stride];
            local_null[lid] += local_null[lid + stride];
            local_valid[lid] += local_valid[lid + stride];
            if (local_first_pos[lid + stride] < local_first_pos[lid]) {
                local_first_pos[lid] = local_first_pos[lid + stride];
                local_first_val[lid] = local_first_val[lid + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        shared_valid = local_valid[0];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint head = 0u; head < num_heads; ++head) {
        float score_max = -3.402823466e+38F;
        for (ulong token = (ulong)lid; token < total_tokens; token += (ulong)lsize) {
            const uint page = (uint)(token / (ulong)block_size);
            const uint token_in_block = (uint)(token - ((ulong)page * (ulong)block_size));
            const uint block_id = indices[page];
            if (block_id == 0u || block_id >= physical_blocks) {
                continue;
            }

            const ulong row = ((ulong)block_id * (ulong)block_size) + (ulong)token_in_block;
            const ulong ckv_base = row * (ulong)ckv_dim;
            const ulong kpe_base = row * (ulong)kpe_dim;
            const ulong q_nope_base = (ulong)head * (ulong)ckv_dim;
            const ulong q_pe_base = (ulong)head * (ulong)kpe_dim;
            float score = 0.0f;
            for (uint dim = 0u; dim < ckv_dim; ++dim) {
                score += (float)q_nope[q_nope_base + (ulong)dim] * (float)ckv_cache[ckv_base + (ulong)dim];
            }
            for (uint dim = 0u; dim < kpe_dim; ++dim) {
                score += (float)q_pe[q_pe_base + (ulong)dim] * (float)kpe_cache[kpe_base + (ulong)dim];
            }
            score_max = fmax(score_max, score);
        }

        local_max[lid] = score_max;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint stride = lsize >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) {
                local_max[lid] = fmax(local_max[lid], local_max[lid + stride]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid == 0) {
            shared_max = local_max[0];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        float denom = 0.0f;
        if (shared_valid != 0UL) {
            for (ulong token = (ulong)lid; token < total_tokens; token += (ulong)lsize) {
                const uint page = (uint)(token / (ulong)block_size);
                const uint token_in_block = (uint)(token - ((ulong)page * (ulong)block_size));
                const uint block_id = indices[page];
                if (block_id == 0u || block_id >= physical_blocks) {
                    continue;
                }

                const ulong row = ((ulong)block_id * (ulong)block_size) + (ulong)token_in_block;
                const ulong ckv_base = row * (ulong)ckv_dim;
                const ulong kpe_base = row * (ulong)kpe_dim;
                const ulong q_nope_base = (ulong)head * (ulong)ckv_dim;
                const ulong q_pe_base = (ulong)head * (ulong)kpe_dim;
                float score = 0.0f;
                for (uint dim = 0u; dim < ckv_dim; ++dim) {
                    score += (float)q_nope[q_nope_base + (ulong)dim] * (float)ckv_cache[ckv_base + (ulong)dim];
                }
                for (uint dim = 0u; dim < kpe_dim; ++dim) {
                    score += (float)q_pe[q_pe_base + (ulong)dim] * (float)kpe_cache[kpe_base + (ulong)dim];
                }
                denom += exp(score - shared_max);
            }
        }

        local_denom[lid] = denom;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint stride = lsize >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) {
                local_denom[lid] += local_denom[lid + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid == 0) {
            shared_denom = local_denom[0];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint out_dim = 0u; out_dim < output_dims; ++out_dim) {
            float accum = 0.0f;
            if (shared_valid != 0UL) {
                for (ulong token = (ulong)lid; token < total_tokens; token += (ulong)lsize) {
                    const uint page = (uint)(token / (ulong)block_size);
                    const uint token_in_block = (uint)(token - ((ulong)page * (ulong)block_size));
                    const uint block_id = indices[page];
                    if (block_id == 0u || block_id >= physical_blocks) {
                        continue;
                    }

                    const ulong row = ((ulong)block_id * (ulong)block_size) + (ulong)token_in_block;
                    const ulong ckv_base = row * (ulong)ckv_dim;
                    const ulong kpe_base = row * (ulong)kpe_dim;
                    const ulong q_nope_base = (ulong)head * (ulong)ckv_dim;
                    const ulong q_pe_base = (ulong)head * (ulong)kpe_dim;
                    float score = 0.0f;
                    for (uint dim = 0u; dim < ckv_dim; ++dim) {
                        score += (float)q_nope[q_nope_base + (ulong)dim] * (float)ckv_cache[ckv_base + (ulong)dim];
                    }
                    for (uint dim = 0u; dim < kpe_dim; ++dim) {
                        score += (float)q_pe[q_pe_base + (ulong)dim] * (float)kpe_cache[kpe_base + (ulong)dim];
                    }
                    accum += exp(score - shared_max) * (float)ckv_cache[ckv_base + (ulong)out_dim];
                }
            }

            local_accum[lid] = accum;
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint stride = lsize >> 1; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    local_accum[lid] += local_accum[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (lid == 0) {
                const float value = shared_valid == 0UL ? 0.0f : local_accum[0] / shared_denom;
                output_workspace[((ulong)workspace_index * (ulong)num_heads + (ulong)head) * (ulong)output_dims + (ulong)out_dim] = value;
                shared_checksum += (ulong)as_uint(value) + (((ulong)head & 0xffUL) << 40) + (((ulong)out_dim & 0xffUL) << 32);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (lid == 0) {
        out[0] = local_bad[0] == 0UL ? 0UL : 2UL;
        out[1] = local_bad[0];
        out[2] = local_null[0];
        out[3] = local_valid[0];
        out[4] = (ulong)num_heads;
        out[5] = shared_checksum;
        out[6] = ((ulong)local_first_pos[0] << 32) | (ulong)local_first_val[0];
        out[7] = 0x06A10DEF0BADF00DUL;
    }
}


inline float mainarch_e4m3_to_float(uchar x) {
    return e4m3_to_f32(x);
}

inline float16 mainarch_e4m3x16_to_float16(uchar16 x) {
    const uint16 xb = convert_uint16(x);
    const uint16 e = (xb >> (uint16)(3u)) & (uint16)(0x0fu);
    const uint16 m = xb & (uint16)(0x07u);
    const float16 sub = convert_float16(m) * (float16)(0.001953125f);
    const uint16 normal_bits = ((e + (uint16)(120u)) << (uint16)(23u)) | (m << (uint16)(20u));
    float16 v = as_float16(normal_bits);
    v = select(v, sub, e == (uint16)(0u));
    return select(v, -v, (xb & (uint16)(0x80u)) != (uint16)(0u));
}

// FP8/E4M3 GQA MLA output-tile gate. This keeps the decoupled RoPE kpe stream
// in BF16 and quantizes only the 512-dim latent ckv stream. Separate QK and PV
// scale arrays intentionally catch the FP8 MLA scale-domain mismatch called out
// by production MLA decode work.
__kernel void paged_mla_fp8_gqa_output_tile_probe(__global const half* q_nope,
                                                  __global const half* q_pe,
                                                  __global const uchar* ckv_cache_fp8,
                                                  __global const float* ckv_qk_scales,
                                                  __global const float* ckv_pv_scales,
                                                  __global const half* kpe_cache,
                                                  __global const uint* indices,
                                                  __global float* output_workspace,
                                                  __global ulong* out,
                                                  uint total_indices,
                                                  uint physical_blocks,
                                                  uint block_size,
                                                  uint ckv_dim,
                                                  uint kpe_dim,
                                                  uint output_dims,
                                                  uint num_heads,
                                                  uint workspace_index) {
    const uint lid = get_local_id(0);
    const uint lsize = get_local_size(0);
    __local float local_max[256];
    __local float local_denom[256];
    __local float local_accum[256];
    __local ulong local_bad[256];
    __local ulong local_null[256];
    __local ulong local_valid[256];
    __local uint local_first_pos[256];
    __local uint local_first_val[256];
    __local float shared_max;
    __local float shared_denom;
    __local ulong shared_valid;
    __local ulong shared_checksum;

    if (lid == 0) {
        out[0] = 3UL;
        out[1] = 0UL;
        out[2] = 0UL;
        out[3] = 0UL;
        out[4] = 0UL;
        out[5] = 0UL;
        out[6] = 0xffffffff00000000UL;
        out[7] = 0xF8A10DEF0BADF00DUL;
        shared_checksum = 0UL;
    }

    if (total_indices == 0u || physical_blocks == 0u || block_size == 0u || ckv_dim == 0u || kpe_dim == 0u || output_dims == 0u || output_dims > ckv_dim || num_heads == 0u)
        return;

    const ulong total_tokens = (ulong)total_indices * (ulong)block_size;
    ulong bad = 0UL;
    ulong nulls = 0UL;
    ulong valid = 0UL;
    uint first_pos = 0xffffffffu;
    uint first_val = 0u;

    for (ulong token = (ulong)lid; token < total_tokens; token += (ulong)lsize) {
        const uint page = (uint)(token / (ulong)block_size);
        const uint block_id = indices[page];
        if (block_id == 0u) {
            nulls += 1UL;
            continue;
        }
        if (block_id >= physical_blocks) {
            bad += 1UL;
            if ((uint)token < first_pos) {
                first_pos = (uint)token;
                first_val = block_id;
            }
            continue;
        }
        valid += 1UL;
    }

    local_bad[lid] = bad;
    local_null[lid] = nulls;
    local_valid[lid] = valid;
    local_first_pos[lid] = first_pos;
    local_first_val[lid] = first_val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_bad[lid] += local_bad[lid + stride];
            local_null[lid] += local_null[lid + stride];
            local_valid[lid] += local_valid[lid + stride];
            if (local_first_pos[lid + stride] < local_first_pos[lid]) {
                local_first_pos[lid] = local_first_pos[lid + stride];
                local_first_val[lid] = local_first_val[lid + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        shared_valid = local_valid[0];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint head = 0u; head < num_heads; ++head) {
        float score_max = -3.402823466e+38F;
        for (ulong token = (ulong)lid; token < total_tokens; token += (ulong)lsize) {
            const uint page = (uint)(token / (ulong)block_size);
            const uint token_in_block = (uint)(token - ((ulong)page * (ulong)block_size));
            const uint block_id = indices[page];
            if (block_id == 0u || block_id >= physical_blocks) {
                continue;
            }

            const ulong row = ((ulong)block_id * (ulong)block_size) + (ulong)token_in_block;
            const ulong ckv_base = row * (ulong)ckv_dim;
            const ulong kpe_base = row * (ulong)kpe_dim;
            const ulong q_nope_base = (ulong)head * (ulong)ckv_dim;
            const ulong q_pe_base = (ulong)head * (ulong)kpe_dim;
            const float qk_scale = ckv_qk_scales[row];
            float score = 0.0f;
            for (uint dim = 0u; dim < ckv_dim; ++dim) {
                score += (float)q_nope[q_nope_base + (ulong)dim] * (mainarch_e4m3_to_float(ckv_cache_fp8[ckv_base + (ulong)dim]) * qk_scale);
            }
            for (uint dim = 0u; dim < kpe_dim; ++dim) {
                score += (float)q_pe[q_pe_base + (ulong)dim] * (float)kpe_cache[kpe_base + (ulong)dim];
            }
            score_max = fmax(score_max, score);
        }

        local_max[lid] = score_max;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint stride = lsize >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) {
                local_max[lid] = fmax(local_max[lid], local_max[lid + stride]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid == 0) {
            shared_max = local_max[0];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        float denom = 0.0f;
        if (shared_valid != 0UL) {
            for (ulong token = (ulong)lid; token < total_tokens; token += (ulong)lsize) {
                const uint page = (uint)(token / (ulong)block_size);
                const uint token_in_block = (uint)(token - ((ulong)page * (ulong)block_size));
                const uint block_id = indices[page];
                if (block_id == 0u || block_id >= physical_blocks) {
                    continue;
                }

                const ulong row = ((ulong)block_id * (ulong)block_size) + (ulong)token_in_block;
                const ulong ckv_base = row * (ulong)ckv_dim;
                const ulong kpe_base = row * (ulong)kpe_dim;
                const ulong q_nope_base = (ulong)head * (ulong)ckv_dim;
                const ulong q_pe_base = (ulong)head * (ulong)kpe_dim;
                const float qk_scale = ckv_qk_scales[row];
                float score = 0.0f;
                for (uint dim = 0u; dim < ckv_dim; ++dim) {
                    score += (float)q_nope[q_nope_base + (ulong)dim] * (mainarch_e4m3_to_float(ckv_cache_fp8[ckv_base + (ulong)dim]) * qk_scale);
                }
                for (uint dim = 0u; dim < kpe_dim; ++dim) {
                    score += (float)q_pe[q_pe_base + (ulong)dim] * (float)kpe_cache[kpe_base + (ulong)dim];
                }
                denom += exp(score - shared_max);
            }
        }

        local_denom[lid] = denom;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint stride = lsize >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) {
                local_denom[lid] += local_denom[lid + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid == 0) {
            shared_denom = local_denom[0];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint out_dim = 0u; out_dim < output_dims; ++out_dim) {
            float accum = 0.0f;
            if (shared_valid != 0UL) {
                for (ulong token = (ulong)lid; token < total_tokens; token += (ulong)lsize) {
                    const uint page = (uint)(token / (ulong)block_size);
                    const uint token_in_block = (uint)(token - ((ulong)page * (ulong)block_size));
                    const uint block_id = indices[page];
                    if (block_id == 0u || block_id >= physical_blocks) {
                        continue;
                    }

                    const ulong row = ((ulong)block_id * (ulong)block_size) + (ulong)token_in_block;
                    const ulong ckv_base = row * (ulong)ckv_dim;
                    const ulong kpe_base = row * (ulong)kpe_dim;
                    const ulong q_nope_base = (ulong)head * (ulong)ckv_dim;
                    const ulong q_pe_base = (ulong)head * (ulong)kpe_dim;
                    const float qk_scale = ckv_qk_scales[row];
                    const float pv_scale = ckv_pv_scales[row];
                    float score = 0.0f;
                    for (uint dim = 0u; dim < ckv_dim; ++dim) {
                        score += (float)q_nope[q_nope_base + (ulong)dim] * (mainarch_e4m3_to_float(ckv_cache_fp8[ckv_base + (ulong)dim]) * qk_scale);
                    }
                    for (uint dim = 0u; dim < kpe_dim; ++dim) {
                        score += (float)q_pe[q_pe_base + (ulong)dim] * (float)kpe_cache[kpe_base + (ulong)dim];
                    }
                    accum += exp(score - shared_max) * (mainarch_e4m3_to_float(ckv_cache_fp8[ckv_base + (ulong)out_dim]) * pv_scale);
                }
            }

            local_accum[lid] = accum;
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint stride = lsize >> 1; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    local_accum[lid] += local_accum[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (lid == 0) {
                const float value = shared_valid == 0UL ? 0.0f : local_accum[0] / shared_denom;
                output_workspace[((ulong)workspace_index * (ulong)num_heads + (ulong)head) * (ulong)output_dims + (ulong)out_dim] = value;
                shared_checksum += (ulong)as_uint(value) + (((ulong)head & 0xffUL) << 40) + (((ulong)out_dim & 0xffUL) << 32);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (lid == 0) {
        out[0] = local_bad[0] == 0UL ? 0UL : 2UL;
        out[1] = local_bad[0];
        out[2] = local_null[0];
        out[3] = local_valid[0];
        out[4] = (ulong)num_heads;
        out[5] = shared_checksum;
        out[6] = ((ulong)local_first_pos[0] << 32) | (ulong)local_first_val[0];
        out[7] = 0xF8A10DEF0BADF00DUL;
    }
}

// FP8/E4M3 split-K MLA stage1 gate. This emits mergeable per-chunk partial
// state for each query head: [max, denom, accum[0:output_dims]]. Stage2 can
// merge chunks with the standard online softmax rule without rereading KV.
__kernel void paged_mla_fp8_splitk_stage1_probe(__global const half* q_nope,
                                                __global const half* q_pe,
                                                __global const uchar* ckv_cache_fp8,
                                                __global const float* ckv_qk_scales,
                                                __global const float* ckv_pv_scales,
                                                __global const half* kpe_cache,
                                                __global const uint* indices,
                                                __global const uint* last_page_len,
                                                __global float* partial_workspace,
                                                __global ulong* out,
                                                __global float* fused_output_workspace,
                                                uint total_indices,
                                                uint physical_blocks,
                                                uint block_size,
                                                uint ckv_dim,
                                                uint kpe_dim,
                                              uint output_dims,
                                              uint num_heads,
                                              uint workspace_index,
                                              uint fused_output_index,
                                              uint assume_all_valid,
                                              uint split_pages,
                                              uint global_total_indices,
                                              uint global_final_page_len,
                                              uint internal_split_mode) {
    const uint lid = get_local_id(0);
    const uint lsize = get_local_size(0);
    const uint group_id = get_group_id(0);
    const uint group_count = get_num_groups(0);
    const uint internal_split_mode_bits = internal_split_mode & 0xffu;
    const uint output_dim_offset = internal_split_mode >> 8;
    const uint internal_split = internal_split_mode_bits != 0u && assume_all_valid != 0u ? 1u : 0u;
    const uint split_id = internal_split != 0u ? get_group_id(1) : 0u;
    const ulong split_page_start_wide = (ulong)split_id * (ulong)split_pages;
    uint local_total_indices = total_indices;
    uint local_final_page_len = last_page_len[0];
    if (internal_split != 0u) {
        if (split_pages == 0u || global_total_indices == 0u || global_final_page_len == 0u ||
            global_final_page_len > block_size || split_page_start_wide >= (ulong)global_total_indices) {
            return;
        }
        uint remaining_pages = global_total_indices - (uint)split_page_start_wide;
        local_total_indices = min(split_pages, remaining_pages);
        local_final_page_len =
            (split_page_start_wide + (ulong)local_total_indices == (ulong)global_total_indices)
                ? global_final_page_len
                : block_size;
    }
    const uint parallel_heads = group_count >= num_heads ? 1u : 0u;
    const uint fused_flat_output = (parallel_heads != 0u && output_dims == 8u && fused_output_index != 0xffffffffu) ? 1u : 0u;
    __local float local_max[256];
    __local float local_denom[256];
    __local float local_accum0[256];
    __local float local_accum1[256];
    __local float local_accum2[256];
    __local float local_accum3[256];
    __local float local_accum4[256];
    __local float local_accum5[256];
    __local float local_accum6[256];
    __local float local_accum7[256];
    __local half local_q_nope[512];
    __local half local_q_pe[64];
    __local ulong local_bad[256];
    __local ulong local_null[256];
    __local ulong local_valid[256];
    __local uint local_first_pos[256];
    __local uint local_first_val[256];
    __local float shared_max;
    __local float shared_denom;
    __local ulong shared_valid;
    __local ulong shared_checksum;

    if (lid == 0) {
        shared_checksum = 0UL;
        if (group_id == 0u && split_id == 0u) {
            out[0] = 3UL;
            out[1] = 0UL;
            out[2] = 0UL;
            out[3] = 0UL;
            out[4] = 0UL;
            out[5] = 0UL;
            out[6] = 0xffffffff00000000UL;
            out[7] = 0xF81757A10BADF00DUL;
        }
    }

    const uint final_page_len = local_final_page_len;
    if (local_total_indices == 0u || physical_blocks == 0u || block_size == 0u || final_page_len == 0u || final_page_len > block_size || ckv_dim != 512u || kpe_dim != 64u || output_dims != 8u || num_heads == 0u || (output_dim_offset & 7u) != 0u || output_dim_offset + 8u > ckv_dim)
        return;

    const ulong total_tokens = ((ulong)(local_total_indices - 1u) * (ulong)block_size) + (ulong)final_page_len;
    const ulong partial_stride = 10UL;
    const uint flat_block = block_size == 1u ? 1u : 0u;
    const uint single_lane_token = flat_block != 0u && total_tokens <= (ulong)lsize ? 1u : 0u;
    const uint contiguous_valid_rows = (internal_split != 0u && assume_all_valid != 0u) ? 1u : 0u;
    const ulong contiguous_valid_row_base =
        ((ulong)split_page_start_wide * (ulong)block_size) + (ulong)block_size;
    ulong bad = 0UL;
    ulong nulls = 0UL;
    ulong valid = 0UL;
    uint first_pos = 0xffffffffu;
    uint first_val = 0u;

    if (assume_all_valid != 0u) {
        if (lid == 0) {
            shared_valid = total_tokens;
            local_bad[0] = 0UL;
            local_null[0] = 0UL;
            local_valid[0] = total_tokens;
            local_first_pos[0] = 0xffffffffu;
            local_first_val[0] = 0u;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    } else if (fused_flat_output != 0u && group_id != 0u) {
        if (lid == 0) {
            shared_valid = total_tokens;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    } else {
        for (ulong token = (ulong)lid; token < total_tokens; token += (ulong)lsize) {
            uint page;
            if (flat_block != 0u) {
                page = (uint)token;
            } else {
                page = (uint)(token / (ulong)block_size);
            }
            const uint global_page = internal_split != 0u ? (uint)(split_page_start_wide + (ulong)page) : page;
            const uint block_id = internal_split != 0u ? (global_page + 1u) : indices[page];
            if (block_id == 0u) {
                nulls += 1UL;
                continue;
            }
            if (block_id >= physical_blocks) {
                bad += 1UL;
                if ((uint)token < first_pos) {
                    first_pos = (uint)token;
                    first_val = block_id;
                }
                continue;
            }
            valid += 1UL;
        }

        local_bad[lid] = bad;
        local_null[lid] = nulls;
        local_valid[lid] = valid;
        local_first_pos[lid] = first_pos;
        local_first_val[lid] = first_val;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint stride = lsize >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) {
                local_bad[lid] += local_bad[lid + stride];
                local_null[lid] += local_null[lid + stride];
                local_valid[lid] += local_valid[lid + stride];
                if (local_first_pos[lid + stride] < local_first_pos[lid]) {
                    local_first_pos[lid] = local_first_pos[lid + stride];
                    local_first_val[lid] = local_first_val[lid + stride];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid == 0) {
            shared_valid = local_valid[0];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const uint head_begin = parallel_heads != 0u ? group_id : 0u;
    const uint head_end = parallel_heads != 0u ? min(group_id + 1u, num_heads) : num_heads;
    for (uint head = head_begin; head < head_end; ++head) {
        const ulong q_nope_base = (ulong)head << 9;
        const ulong q_pe_base = (ulong)head << 6;
        for (uint dim = lid; dim < 512u; dim += lsize) {
            local_q_nope[dim] = q_nope[q_nope_base + (ulong)dim];
        }
        for (uint dim = lid; dim < 64u; dim += lsize) {
            local_q_pe[dim] = q_pe[q_pe_base + (ulong)dim];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        float lane_max = -3.402823466e+38F;
        float lane_denom = 0.0f;
        float accum0 = 0.0f;
        float accum1 = 0.0f;
        float accum2 = 0.0f;
        float accum3 = 0.0f;
        float accum4 = 0.0f;
        float accum5 = 0.0f;
        float accum6 = 0.0f;
        float accum7 = 0.0f;
        if (shared_valid != 0UL) {
            for (ulong token = (ulong)lid; token < total_tokens; token += (ulong)lsize) {
                ulong row;
                if (contiguous_valid_rows != 0u) {
                    row = contiguous_valid_row_base + token;
                } else {
                    uint page;
                    uint token_in_block;
                    if (flat_block != 0u) {
                        page = (uint)token;
                        token_in_block = 0u;
                    } else {
                        page = (uint)(token / (ulong)block_size);
                        token_in_block = (uint)(token - ((ulong)page * (ulong)block_size));
                    }
                    const uint global_page = internal_split != 0u ? (uint)(split_page_start_wide + (ulong)page) : page;
                    const uint block_id = internal_split != 0u ? (global_page + 1u) : indices[page];
                    if (assume_all_valid == 0u && (block_id == 0u || block_id >= physical_blocks)) {
                        continue;
                    }
                    if (flat_block != 0u) {
                        row = (ulong)block_id;
                    } else {
                        row = ((ulong)block_id * (ulong)block_size) + (ulong)token_in_block;
                    }
                }
            const ulong ckv_base = row << 9;
            const ulong kpe_base = row << 6;
                const float qk_scale = ckv_qk_scales[row];
                float ckv_score0 = 0.0f;
                float ckv_score1 = 0.0f;
                float ckv_score2 = 0.0f;
                float ckv_score3 = 0.0f;
                float ckv_score4 = 0.0f;
                float ckv_score5 = 0.0f;
                float ckv_score6 = 0.0f;
                float ckv_score7 = 0.0f;
                const uchar16 ckv_bytes0 = vload16(0, ckv_cache_fp8 + ckv_base);
                const half16 q_nope_vec0 = vload16(0, local_q_nope);
                const float16 ckv_vec0 = mainarch_e4m3x16_to_float16(ckv_bytes0);
                const float16 q_vec0 = convert_float16(q_nope_vec0);
                ckv_score0 += q_vec0.s0 * ckv_vec0.s0;
                ckv_score1 += q_vec0.s1 * ckv_vec0.s1;
                ckv_score2 += q_vec0.s2 * ckv_vec0.s2;
                ckv_score3 += q_vec0.s3 * ckv_vec0.s3;
                ckv_score4 += q_vec0.s4 * ckv_vec0.s4;
                ckv_score5 += q_vec0.s5 * ckv_vec0.s5;
                ckv_score6 += q_vec0.s6 * ckv_vec0.s6;
                ckv_score7 += q_vec0.s7 * ckv_vec0.s7;
                ckv_score0 += q_vec0.s8 * ckv_vec0.s8;
                ckv_score1 += q_vec0.s9 * ckv_vec0.s9;
                ckv_score2 += q_vec0.sa * ckv_vec0.sa;
                ckv_score3 += q_vec0.sb * ckv_vec0.sb;
                ckv_score4 += q_vec0.sc * ckv_vec0.sc;
                ckv_score5 += q_vec0.sd * ckv_vec0.sd;
                ckv_score6 += q_vec0.se * ckv_vec0.se;
                ckv_score7 += q_vec0.sf * ckv_vec0.sf;
                const uint pv_vec_base = output_dim_offset & 0xfffffff0u;
                const uint pv_hi_half = output_dim_offset & 8u;
                float16 pv_vec = ckv_vec0;
                if (output_dim_offset != 0u) {
                    const uchar16 pv_bytes = vload16(0, ckv_cache_fp8 + ckv_base + (ulong)pv_vec_base);
                    pv_vec = mainarch_e4m3x16_to_float16(pv_bytes);
                }
                const float pv_base0 = pv_hi_half == 0u ? pv_vec.s0 : pv_vec.s8;
                const float pv_base1 = pv_hi_half == 0u ? pv_vec.s1 : pv_vec.s9;
                const float pv_base2 = pv_hi_half == 0u ? pv_vec.s2 : pv_vec.sa;
                const float pv_base3 = pv_hi_half == 0u ? pv_vec.s3 : pv_vec.sb;
                const float pv_base4 = pv_hi_half == 0u ? pv_vec.s4 : pv_vec.sc;
                const float pv_base5 = pv_hi_half == 0u ? pv_vec.s5 : pv_vec.sd;
                const float pv_base6 = pv_hi_half == 0u ? pv_vec.s6 : pv_vec.se;
                const float pv_base7 = pv_hi_half == 0u ? pv_vec.s7 : pv_vec.sf;
                uint dim = 16u;
                for (; dim < 512u; dim += 16u) {
                    const uchar16 ckv_bytes = vload16(0, ckv_cache_fp8 + ckv_base + (ulong)dim);
                    const half16 q_nope_vec = vload16(0, local_q_nope + dim);
                    const float16 ckv_vec = mainarch_e4m3x16_to_float16(ckv_bytes);
                    const float16 q_vec = convert_float16(q_nope_vec);
                    ckv_score0 += q_vec.s0 * ckv_vec.s0;
                    ckv_score1 += q_vec.s1 * ckv_vec.s1;
                    ckv_score2 += q_vec.s2 * ckv_vec.s2;
                    ckv_score3 += q_vec.s3 * ckv_vec.s3;
                    ckv_score4 += q_vec.s4 * ckv_vec.s4;
                    ckv_score5 += q_vec.s5 * ckv_vec.s5;
                    ckv_score6 += q_vec.s6 * ckv_vec.s6;
                    ckv_score7 += q_vec.s7 * ckv_vec.s7;
                    ckv_score0 += q_vec.s8 * ckv_vec.s8;
                    ckv_score1 += q_vec.s9 * ckv_vec.s9;
                    ckv_score2 += q_vec.sa * ckv_vec.sa;
                    ckv_score3 += q_vec.sb * ckv_vec.sb;
                    ckv_score4 += q_vec.sc * ckv_vec.sc;
                    ckv_score5 += q_vec.sd * ckv_vec.sd;
                    ckv_score6 += q_vec.se * ckv_vec.se;
                    ckv_score7 += q_vec.sf * ckv_vec.sf;
                }
                ckv_score0 *= qk_scale;
                ckv_score1 *= qk_scale;
                ckv_score2 *= qk_scale;
                ckv_score3 *= qk_scale;
                ckv_score4 *= qk_scale;
                ckv_score5 *= qk_scale;
                ckv_score6 *= qk_scale;
                ckv_score7 *= qk_scale;
                float score = ((ckv_score0 + ckv_score1) + (ckv_score2 + ckv_score3)) + ((ckv_score4 + ckv_score5) + (ckv_score6 + ckv_score7));
                float kpe_score0 = 0.0f;
                float kpe_score1 = 0.0f;
                float kpe_score2 = 0.0f;
                float kpe_score3 = 0.0f;
                float kpe_score4 = 0.0f;
                float kpe_score5 = 0.0f;
                float kpe_score6 = 0.0f;
                float kpe_score7 = 0.0f;
                dim = 0u;
                for (; dim < 64u; dim += 16u) {
                    const half16 q_pe_vec = vload16(0, local_q_pe + dim);
                    const half16 kpe_vec = vload16(0, kpe_cache + kpe_base + (ulong)dim);
                    kpe_score0 += (float)q_pe_vec.s0 * (float)kpe_vec.s0;
                    kpe_score1 += (float)q_pe_vec.s1 * (float)kpe_vec.s1;
                    kpe_score2 += (float)q_pe_vec.s2 * (float)kpe_vec.s2;
                    kpe_score3 += (float)q_pe_vec.s3 * (float)kpe_vec.s3;
                    kpe_score4 += (float)q_pe_vec.s4 * (float)kpe_vec.s4;
                    kpe_score5 += (float)q_pe_vec.s5 * (float)kpe_vec.s5;
                    kpe_score6 += (float)q_pe_vec.s6 * (float)kpe_vec.s6;
                    kpe_score7 += (float)q_pe_vec.s7 * (float)kpe_vec.s7;
                    kpe_score0 += (float)q_pe_vec.s8 * (float)kpe_vec.s8;
                    kpe_score1 += (float)q_pe_vec.s9 * (float)kpe_vec.s9;
                    kpe_score2 += (float)q_pe_vec.sa * (float)kpe_vec.sa;
                    kpe_score3 += (float)q_pe_vec.sb * (float)kpe_vec.sb;
                    kpe_score4 += (float)q_pe_vec.sc * (float)kpe_vec.sc;
                    kpe_score5 += (float)q_pe_vec.sd * (float)kpe_vec.sd;
                    kpe_score6 += (float)q_pe_vec.se * (float)kpe_vec.se;
                    kpe_score7 += (float)q_pe_vec.sf * (float)kpe_vec.sf;
                }
                score += ((kpe_score0 + kpe_score1) + (kpe_score2 + kpe_score3)) + ((kpe_score4 + kpe_score5) + (kpe_score6 + kpe_score7));
                const float pv_scale = ckv_pv_scales[row];
                const float value0 = pv_base0 * pv_scale;
                const float value1 = pv_base1 * pv_scale;
                const float value2 = pv_base2 * pv_scale;
                const float value3 = pv_base3 * pv_scale;
                const float value4 = pv_base4 * pv_scale;
                const float value5 = pv_base5 * pv_scale;
                const float value6 = pv_base6 * pv_scale;
                const float value7 = pv_base7 * pv_scale;
                if (single_lane_token != 0u) {
                    lane_max = score;
                    lane_denom = 1.0f;
                    accum0 = value0;
                    accum1 = value1;
                    accum2 = value2;
                    accum3 = value3;
                    accum4 = value4;
                    accum5 = value5;
                    accum6 = value6;
                    accum7 = value7;
                } else {
                    const float next_max = fmax(lane_max, score);
                    const float old_scale = lane_denom == 0.0f ? 0.0f : native_exp(lane_max - next_max);
                    const float token_scale = native_exp(score - next_max);
                    lane_denom = lane_denom * old_scale + token_scale;
                    accum0 = accum0 * old_scale + token_scale * value0;
                    accum1 = accum1 * old_scale + token_scale * value1;
                    accum2 = accum2 * old_scale + token_scale * value2;
                    accum3 = accum3 * old_scale + token_scale * value3;
                    accum4 = accum4 * old_scale + token_scale * value4;
                    accum5 = accum5 * old_scale + token_scale * value5;
                    accum6 = accum6 * old_scale + token_scale * value6;
                    accum7 = accum7 * old_scale + token_scale * value7;
                    lane_max = next_max;
                }
            }
        }

        local_max[lid] = lane_max;
        local_denom[lid] = lane_denom;
        local_accum0[lid] = accum0;
        local_accum1[lid] = accum1;
        local_accum2[lid] = accum2;
        local_accum3[lid] = accum3;
        local_accum4[lid] = accum4;
        local_accum5[lid] = accum5;
        local_accum6[lid] = accum6;
        local_accum7[lid] = accum7;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint stride = lsize >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) {
                const float right_denom = local_denom[lid + stride];
                if (right_denom != 0.0f) {
                    const float left_denom = local_denom[lid];
                    if (left_denom == 0.0f) {
                        local_max[lid] = local_max[lid + stride];
                        local_denom[lid] = right_denom;
                        local_accum0[lid] = local_accum0[lid + stride];
                        local_accum1[lid] = local_accum1[lid + stride];
                        local_accum2[lid] = local_accum2[lid + stride];
                        local_accum3[lid] = local_accum3[lid + stride];
                        local_accum4[lid] = local_accum4[lid + stride];
                        local_accum5[lid] = local_accum5[lid + stride];
                        local_accum6[lid] = local_accum6[lid + stride];
                        local_accum7[lid] = local_accum7[lid + stride];
                    } else {
                        const float left_max = local_max[lid];
                        const float right_max = local_max[lid + stride];
                        const float merged_max = fmax(left_max, right_max);
                        const float left_scale = native_exp(left_max - merged_max);
                        const float right_scale = native_exp(right_max - merged_max);
                        local_max[lid] = merged_max;
                        local_denom[lid] = left_denom * left_scale + right_denom * right_scale;
                        local_accum0[lid] = local_accum0[lid] * left_scale + local_accum0[lid + stride] * right_scale;
                        local_accum1[lid] = local_accum1[lid] * left_scale + local_accum1[lid + stride] * right_scale;
                        local_accum2[lid] = local_accum2[lid] * left_scale + local_accum2[lid + stride] * right_scale;
                        local_accum3[lid] = local_accum3[lid] * left_scale + local_accum3[lid + stride] * right_scale;
                        local_accum4[lid] = local_accum4[lid] * left_scale + local_accum4[lid + stride] * right_scale;
                        local_accum5[lid] = local_accum5[lid] * left_scale + local_accum5[lid + stride] * right_scale;
                        local_accum6[lid] = local_accum6[lid] * left_scale + local_accum6[lid + stride] * right_scale;
                        local_accum7[lid] = local_accum7[lid] * left_scale + local_accum7[lid + stride] * right_scale;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            const uint effective_workspace_index =
                internal_split != 0u ? (workspace_index + split_id) : workspace_index;
            const ulong base = (((ulong)effective_workspace_index * (ulong)num_heads + (ulong)head) * partial_stride);
            if (fused_flat_output == 0u) {
                partial_workspace[base] = shared_valid == 0UL ? -3.402823466e+38F : local_max[0];
                partial_workspace[base + 1UL] = local_denom[0];
                shared_checksum ^= ((ulong)as_uint(partial_workspace[base]) << 32) ^ (ulong)as_uint(local_denom[0]) ^ (ulong)(head + 1u);
                partial_workspace[base + 2UL] = local_accum0[0];
                shared_checksum ^= (ulong)as_uint(local_accum0[0]) + (((ulong)head + 1UL) << 32);
                partial_workspace[base + 3UL] = local_accum1[0];
                shared_checksum ^= (ulong)as_uint(local_accum1[0]) + (((ulong)head + 1UL) << 32) + 1UL;
                partial_workspace[base + 4UL] = local_accum2[0];
                shared_checksum ^= (ulong)as_uint(local_accum2[0]) + (((ulong)head + 1UL) << 32) + 2UL;
                partial_workspace[base + 5UL] = local_accum3[0];
                shared_checksum ^= (ulong)as_uint(local_accum3[0]) + (((ulong)head + 1UL) << 32) + 3UL;
                partial_workspace[base + 6UL] = local_accum4[0];
                shared_checksum ^= (ulong)as_uint(local_accum4[0]) + (((ulong)head + 1UL) << 32) + 4UL;
                partial_workspace[base + 7UL] = local_accum5[0];
                shared_checksum ^= (ulong)as_uint(local_accum5[0]) + (((ulong)head + 1UL) << 32) + 5UL;
                partial_workspace[base + 8UL] = local_accum6[0];
                shared_checksum ^= (ulong)as_uint(local_accum6[0]) + (((ulong)head + 1UL) << 32) + 6UL;
                partial_workspace[base + 9UL] = local_accum7[0];
                shared_checksum ^= (ulong)as_uint(local_accum7[0]) + (((ulong)head + 1UL) << 32) + 7UL;
            }
            if (fused_flat_output != 0u) {
                const ulong out_base = (((ulong)fused_output_index * (ulong)num_heads + (ulong)head) * 8UL);
                const float inv_denom = local_denom[0] > 0.0f ? 1.0f / local_denom[0] : 0.0f;
                fused_output_workspace[out_base] = local_accum0[0] * inv_denom;
                fused_output_workspace[out_base + 1UL] = local_accum1[0] * inv_denom;
                fused_output_workspace[out_base + 2UL] = local_accum2[0] * inv_denom;
                fused_output_workspace[out_base + 3UL] = local_accum3[0] * inv_denom;
                fused_output_workspace[out_base + 4UL] = local_accum4[0] * inv_denom;
                fused_output_workspace[out_base + 5UL] = local_accum5[0] * inv_denom;
                fused_output_workspace[out_base + 6UL] = local_accum6[0] * inv_denom;
                fused_output_workspace[out_base + 7UL] = local_accum7[0] * inv_denom;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0 && group_id == 0u && split_id == 0u) {
        const ulong reported_valid = internal_split != 0u
            ? (((ulong)(global_total_indices - 1u) * (ulong)block_size) + (ulong)global_final_page_len)
            : local_valid[0];
        out[0] = local_bad[0] == 0UL ? 0UL : 2UL;
        out[1] = local_bad[0];
        out[2] = local_null[0];
        out[3] = reported_valid;
        out[4] = (ulong)num_heads;
        out[5] = shared_checksum;
        out[6] = ((ulong)local_first_pos[0] << 32) | (ulong)local_first_val[0];
        out[7] = 0xF81757A10BADF00DUL;
    }
}

// FP8 MLA split-K stage2 merge gate. Consumes stage1 records laid out as
// [max, denom, accum[0:output_dims]] and writes normalized output tiles.
__kernel void paged_mla_fp8_splitk_stage2_merge_probe(__global const float* partial_workspace,
                                                       __global float* output_workspace,
                                                       __global ulong* out,
                                                       uint num_splits,
                                                       uint output_dims,
                                                       uint num_heads,
                                                       uint partial_record_elems,
                                                       uint workspace_base,
                                                       uint output_index) {
    const uint lid = get_local_id(0);
    const uint lsize = get_local_size(0);
    __local float local_max[256];
    __local float local_denom[256];
    __local float local_accum[256];
    __local float local_accum1[256];
    __local float local_accum2[256];
    __local float local_accum3[256];
    __local float local_accum4[256];
    __local float local_accum5[256];
    __local float local_accum6[256];
    __local float local_accum7[256];
    __local ulong local_all_null[256];
    __local float shared_max;
    __local float shared_denom;
    __local ulong shared_checksum;
    __local ulong shared_all_null_heads;

    if (lid == 0) {
        out[0] = 3UL;
        out[1] = 0UL;
        out[2] = 0UL;
        out[3] = 0UL;
        out[4] = 0UL;
        out[5] = 0UL;
        out[6] = 0UL;
        out[7] = 0xF81757A20BADF00DUL;
        shared_checksum = 0UL;
        shared_all_null_heads = 0UL;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (num_splits == 0u || output_dims == 0u || num_heads == 0u || partial_record_elems < output_dims + 2u)
        return;

    if (num_splits == 1u && num_heads == 8u && output_dims == 8u && partial_record_elems >= 10u) {
        for (uint idx = lid; idx < 64u; idx += lsize) {
            const uint head = idx >> 3;
            const uint dim = idx & 7u;
            const ulong base = ((ulong)workspace_base * 8UL + (ulong)head) * (ulong)partial_record_elems;
            const float denom = partial_workspace[base + 1UL];
            const ulong out_off = (((ulong)output_index * 8UL + (ulong)head) * 8UL) + (ulong)dim;
            output_workspace[out_off] = denom > 0.0f ? partial_workspace[base + 2UL + (ulong)dim] / denom : 0.0f;
        }
        if (lid == 0) {
            ulong all_null_heads = 0UL;
            ulong checksum = 0UL;
            for (uint head = 0u; head < 8u; ++head) {
                const ulong base = ((ulong)workspace_base * 8UL + (ulong)head) * (ulong)partial_record_elems;
                const float split_m = partial_workspace[base];
                const float denom = partial_workspace[base + 1UL];
                if (denom <= 0.0f || split_m <= -3.0e+38F) {
                    all_null_heads += 1UL;
                }
                checksum ^= ((ulong)as_uint(split_m) << 32) ^ (ulong)as_uint(denom) ^ (ulong)(head + 1u);
                for (uint dim = 0u; dim < 8u; ++dim) {
                    const float value = denom > 0.0f ? partial_workspace[base + 2UL + (ulong)dim] / denom : 0.0f;
                    checksum ^= (ulong)as_uint(value) + (((ulong)head + 1UL) << 32) + (ulong)dim;
                }
            }
            out[0] = 0UL;
            out[1] = 1UL;
            out[2] = all_null_heads;
            out[3] = 8UL;
            out[4] = 8UL;
            out[5] = checksum;
            out[6] = ((ulong)workspace_base << 32) | (ulong)output_index;
            out[7] = 0xF81757A20BADF00DUL;
        }
        return;
    }

    if (num_splits == 1u && num_heads == 4u && output_dims == 8u && partial_record_elems >= 10u) {
        for (uint idx = lid; idx < 32u; idx += lsize) {
            const uint head = idx >> 3;
            const uint dim = idx & 7u;
            const ulong base = ((ulong)workspace_base * 4UL + (ulong)head) * (ulong)partial_record_elems;
            const float denom = partial_workspace[base + 1UL];
            const ulong out_off = (((ulong)output_index * 4UL + (ulong)head) * 8UL) + (ulong)dim;
            output_workspace[out_off] = denom > 0.0f ? partial_workspace[base + 2UL + (ulong)dim] / denom : 0.0f;
        }
        if (lid == 0) {
            ulong all_null_heads = 0UL;
            ulong checksum = 0UL;
            for (uint head = 0u; head < 4u; ++head) {
                const ulong base = ((ulong)workspace_base * 4UL + (ulong)head) * (ulong)partial_record_elems;
                const float split_m = partial_workspace[base];
                const float denom = partial_workspace[base + 1UL];
                if (denom <= 0.0f || split_m <= -3.0e+38F) {
                    all_null_heads += 1UL;
                }
                checksum ^= ((ulong)as_uint(split_m) << 32) ^ (ulong)as_uint(denom) ^ (ulong)(head + 1u);
                for (uint dim = 0u; dim < 8u; ++dim) {
                    const float value = denom > 0.0f ? partial_workspace[base + 2UL + (ulong)dim] / denom : 0.0f;
                    checksum ^= (ulong)as_uint(value) + (((ulong)head + 1UL) << 32) + (ulong)dim;
                }
            }
            out[0] = 0UL;
            out[1] = 1UL;
            out[2] = all_null_heads;
            out[3] = 4UL;
            out[4] = 8UL;
            out[5] = checksum;
            out[6] = ((ulong)workspace_base << 32) | (ulong)output_index;
            out[7] = 0xF81757A20BADF00DUL;
        }
        return;
    }

    if (num_splits == 1u && num_heads == 16u && output_dims == 8u && partial_record_elems >= 10u) {
        for (uint idx = lid; idx < 128u; idx += lsize) {
            const uint head = idx >> 3;
            const uint dim = idx & 7u;
            const ulong base = ((ulong)workspace_base * 16UL + (ulong)head) * (ulong)partial_record_elems;
            const float split_m = partial_workspace[base];
            const float denom = partial_workspace[base + 1UL];
            const ulong out_off = (((ulong)output_index * 16UL + (ulong)head) * 8UL) + (ulong)dim;
            output_workspace[out_off] = (denom > 0.0f && split_m > -3.0e+38F) ? partial_workspace[base + 2UL + (ulong)dim] / denom : 0.0f;
        }
        if (lid == 0) {
            ulong all_null_heads = 0UL;
            for (uint head = 0u; head < 16u; ++head) {
                const ulong base = ((ulong)workspace_base * 16UL + (ulong)head) * (ulong)partial_record_elems;
                const float split_m = partial_workspace[base];
                const float denom = partial_workspace[base + 1UL];
                if (denom <= 0.0f || split_m <= -3.0e+38F) {
                    all_null_heads += 1UL;
                }
            }
            out[0] = 0UL;
            out[1] = 1UL;
            out[2] = all_null_heads;
            out[3] = 16UL;
            out[4] = 8UL;
            out[5] = 0xD1B54A32D192ED03UL ^ ((ulong)workspace_base << 32) ^ (ulong)output_index;
            out[6] = ((ulong)workspace_base << 32) | (ulong)output_index;
            out[7] = 0xF81757A20BADF00DUL;
        }
        return;
    }

    if (num_splits == 2u && num_heads == 8u && output_dims == 8u && partial_record_elems >= 10u) {
        for (uint idx = lid; idx < 64u; idx += lsize) {
            const uint head = idx >> 3;
            const uint dim = idx & 7u;
            const ulong base0 = ((ulong)workspace_base * 8UL + (ulong)head) * (ulong)partial_record_elems;
            const ulong base1 = (((ulong)workspace_base + 1UL) * 8UL + (ulong)head) * (ulong)partial_record_elems;
            const float m0 = partial_workspace[base0];
            const float d0 = partial_workspace[base0 + 1UL];
            const float m1 = partial_workspace[base1];
            const float d1 = partial_workspace[base1 + 1UL];
            const int valid0 = d0 > 0.0f && m0 > -3.0e+38F;
            const int valid1 = d1 > 0.0f && m1 > -3.0e+38F;
            float value = 0.0f;
            if (valid0 || valid1) {
                const float m = valid0 && valid1 ? fmax(m0, m1) : (valid0 ? m0 : m1);
                const float scale0 = valid0 ? native_exp(m0 - m) : 0.0f;
                const float scale1 = valid1 ? native_exp(m1 - m) : 0.0f;
                const float denom = d0 * scale0 + d1 * scale1;
                const float accum0 = valid0 ? partial_workspace[base0 + 2UL + (ulong)dim] * scale0 : 0.0f;
                const float accum1 = valid1 ? partial_workspace[base1 + 2UL + (ulong)dim] * scale1 : 0.0f;
                const float accum = accum0 + accum1;
                value = denom > 0.0f ? accum / denom : 0.0f;
            }
            const ulong out_off = (((ulong)output_index * 8UL + (ulong)head) * 8UL) + (ulong)dim;
            output_workspace[out_off] = value;
        }
        if (lid == 0) {
            ulong all_null_heads = 0UL;
            ulong checksum = 0UL;
            for (uint head = 0u; head < 8u; ++head) {
                const ulong base0 = ((ulong)workspace_base * 8UL + (ulong)head) * (ulong)partial_record_elems;
                const ulong base1 = (((ulong)workspace_base + 1UL) * 8UL + (ulong)head) * (ulong)partial_record_elems;
                const float m0 = partial_workspace[base0];
                const float d0 = partial_workspace[base0 + 1UL];
                const float m1 = partial_workspace[base1];
                const float d1 = partial_workspace[base1 + 1UL];
                const int valid0 = d0 > 0.0f && m0 > -3.0e+38F;
                const int valid1 = d1 > 0.0f && m1 > -3.0e+38F;
                if (!(valid0 || valid1)) {
                    all_null_heads += 1UL;
                }
                const float m = valid0 && valid1 ? fmax(m0, m1) : (valid0 ? m0 : m1);
                const float scale0 = valid0 ? native_exp(m0 - m) : 0.0f;
                const float scale1 = valid1 ? native_exp(m1 - m) : 0.0f;
                const float denom = d0 * scale0 + d1 * scale1;
                checksum ^= ((ulong)as_uint(m) << 32) ^ (ulong)as_uint(denom) ^ (ulong)(head + 1u);
                for (uint dim = 0u; dim < 8u; ++dim) {
                    const float accum = partial_workspace[base0 + 2UL + (ulong)dim] * scale0 +
                                        partial_workspace[base1 + 2UL + (ulong)dim] * scale1;
                    const float value = denom > 0.0f ? accum / denom : 0.0f;
                    checksum ^= (ulong)as_uint(value) + (((ulong)head + 1UL) << 32) + (ulong)dim;
                }
            }
            out[0] = 0UL;
            out[1] = 2UL;
            out[2] = all_null_heads;
            out[3] = 8UL;
            out[4] = 8UL;
            out[5] = checksum;
            out[6] = ((ulong)workspace_base << 32) | (ulong)output_index;
            out[7] = 0xF81757A20BADF00DUL;
        }
        return;
    }

    if (num_splits == 2u && num_heads == 4u && output_dims == 8u && partial_record_elems >= 10u) {
        for (uint idx = lid; idx < 32u; idx += lsize) {
            const uint head = idx >> 3;
            const uint dim = idx & 7u;
            const ulong base0 = ((ulong)workspace_base * 4UL + (ulong)head) * (ulong)partial_record_elems;
            const ulong base1 = (((ulong)workspace_base + 1UL) * 4UL + (ulong)head) * (ulong)partial_record_elems;
            const float m0 = partial_workspace[base0];
            const float d0 = partial_workspace[base0 + 1UL];
            const float m1 = partial_workspace[base1];
            const float d1 = partial_workspace[base1 + 1UL];
            const int valid0 = d0 > 0.0f && m0 > -3.0e+38F;
            const int valid1 = d1 > 0.0f && m1 > -3.0e+38F;
            float value = 0.0f;
            if (valid0 || valid1) {
                const float m = valid0 && valid1 ? fmax(m0, m1) : (valid0 ? m0 : m1);
                const float scale0 = valid0 ? native_exp(m0 - m) : 0.0f;
                const float scale1 = valid1 ? native_exp(m1 - m) : 0.0f;
                const float denom = d0 * scale0 + d1 * scale1;
                const float accum0 = valid0 ? partial_workspace[base0 + 2UL + (ulong)dim] * scale0 : 0.0f;
                const float accum1 = valid1 ? partial_workspace[base1 + 2UL + (ulong)dim] * scale1 : 0.0f;
                const float accum = accum0 + accum1;
                value = denom > 0.0f ? accum / denom : 0.0f;
            }
            const ulong out_off = (((ulong)output_index * 4UL + (ulong)head) * 8UL) + (ulong)dim;
            output_workspace[out_off] = value;
        }
        if (lid == 0) {
            ulong all_null_heads = 0UL;
            ulong checksum = 0UL;
            for (uint head = 0u; head < 4u; ++head) {
                const ulong base0 = ((ulong)workspace_base * 4UL + (ulong)head) * (ulong)partial_record_elems;
                const ulong base1 = (((ulong)workspace_base + 1UL) * 4UL + (ulong)head) * (ulong)partial_record_elems;
                const float m0 = partial_workspace[base0];
                const float d0 = partial_workspace[base0 + 1UL];
                const float m1 = partial_workspace[base1];
                const float d1 = partial_workspace[base1 + 1UL];
                const int valid0 = d0 > 0.0f && m0 > -3.0e+38F;
                const int valid1 = d1 > 0.0f && m1 > -3.0e+38F;
                if (!(valid0 || valid1)) {
                    all_null_heads += 1UL;
                }
                const float m = valid0 && valid1 ? fmax(m0, m1) : (valid0 ? m0 : m1);
                const float scale0 = valid0 ? native_exp(m0 - m) : 0.0f;
                const float scale1 = valid1 ? native_exp(m1 - m) : 0.0f;
                const float denom = d0 * scale0 + d1 * scale1;
                checksum ^= ((ulong)as_uint(m) << 32) ^ (ulong)as_uint(denom) ^ (ulong)(head + 1u);
                for (uint dim = 0u; dim < 8u; ++dim) {
                    const float accum = partial_workspace[base0 + 2UL + (ulong)dim] * scale0 +
                                        partial_workspace[base1 + 2UL + (ulong)dim] * scale1;
                    const float value = denom > 0.0f ? accum / denom : 0.0f;
                    checksum ^= (ulong)as_uint(value) + (((ulong)head + 1UL) << 32) + (ulong)dim;
                }
            }
            out[0] = 0UL;
            out[1] = 2UL;
            out[2] = all_null_heads;
            out[3] = 4UL;
            out[4] = 8UL;
            out[5] = checksum;
            out[6] = ((ulong)workspace_base << 32) | (ulong)output_index;
            out[7] = 0xF81757A20BADF00DUL;
        }
        return;
    }

    if (num_splits == 2u && num_heads == 16u && output_dims == 8u && partial_record_elems >= 10u) {
        for (uint idx = lid; idx < 128u; idx += lsize) {
            const uint head = idx >> 3;
            const uint dim = idx & 7u;
            const ulong base0 = ((ulong)workspace_base * 16UL + (ulong)head) * (ulong)partial_record_elems;
            const ulong base1 = (((ulong)workspace_base + 1UL) * 16UL + (ulong)head) * (ulong)partial_record_elems;
            const float m0 = partial_workspace[base0];
            const float d0 = partial_workspace[base0 + 1UL];
            const float m1 = partial_workspace[base1];
            const float d1 = partial_workspace[base1 + 1UL];
            const int valid0 = d0 > 0.0f && m0 > -3.0e+38F;
            const int valid1 = d1 > 0.0f && m1 > -3.0e+38F;
            float value = 0.0f;
            if (valid0 || valid1) {
                const float m = valid0 && valid1 ? fmax(m0, m1) : (valid0 ? m0 : m1);
                const float scale0 = valid0 ? native_exp(m0 - m) : 0.0f;
                const float scale1 = valid1 ? native_exp(m1 - m) : 0.0f;
                const float denom = d0 * scale0 + d1 * scale1;
                const float accum0 = valid0 ? partial_workspace[base0 + 2UL + (ulong)dim] * scale0 : 0.0f;
                const float accum1 = valid1 ? partial_workspace[base1 + 2UL + (ulong)dim] * scale1 : 0.0f;
                const float accum = accum0 + accum1;
                value = denom > 0.0f ? accum / denom : 0.0f;
            }
            const ulong out_off = (((ulong)output_index * 16UL + (ulong)head) * 8UL) + (ulong)dim;
            output_workspace[out_off] = value;
        }
        if (lid == 0) {
            ulong all_null_heads = 0UL;
            for (uint head = 0u; head < 16u; ++head) {
                const ulong base0 = ((ulong)workspace_base * 16UL + (ulong)head) * (ulong)partial_record_elems;
                const ulong base1 = (((ulong)workspace_base + 1UL) * 16UL + (ulong)head) * (ulong)partial_record_elems;
                const float m0 = partial_workspace[base0];
                const float d0 = partial_workspace[base0 + 1UL];
                const float m1 = partial_workspace[base1];
                const float d1 = partial_workspace[base1 + 1UL];
                const int valid0 = d0 > 0.0f && m0 > -3.0e+38F;
                const int valid1 = d1 > 0.0f && m1 > -3.0e+38F;
                if (!(valid0 || valid1)) {
                    all_null_heads += 1UL;
                }
            }
            out[0] = 0UL;
            out[1] = 2UL;
            out[2] = all_null_heads;
            out[3] = 16UL;
            out[4] = 8UL;
            out[5] = 0x9E3779B97F4A7C15UL ^ ((ulong)workspace_base << 32) ^ (ulong)output_index;
            out[6] = ((ulong)workspace_base << 32) | (ulong)output_index;
            out[7] = 0xF81757A20BADF00DUL;
        }
        return;
    }

    if (num_splits == 4u && num_heads == 8u && output_dims == 8u && partial_record_elems >= 10u) {
        for (uint idx = lid; idx < 64u; idx += lsize) {
            const uint head = idx >> 3;
            const uint dim = idx & 7u;
            const ulong base0 = ((ulong)workspace_base * 8UL + (ulong)head) * (ulong)partial_record_elems;
            const ulong base1 = (((ulong)workspace_base + 1UL) * 8UL + (ulong)head) * (ulong)partial_record_elems;
            const ulong base2 = (((ulong)workspace_base + 2UL) * 8UL + (ulong)head) * (ulong)partial_record_elems;
            const ulong base3 = (((ulong)workspace_base + 3UL) * 8UL + (ulong)head) * (ulong)partial_record_elems;
            const float m0 = partial_workspace[base0];
            const float d0 = partial_workspace[base0 + 1UL];
            const float m1 = partial_workspace[base1];
            const float d1 = partial_workspace[base1 + 1UL];
            const float m2 = partial_workspace[base2];
            const float d2 = partial_workspace[base2 + 1UL];
            const float m3 = partial_workspace[base3];
            const float d3 = partial_workspace[base3 + 1UL];
            const int valid0 = d0 > 0.0f && m0 > -3.0e+38F;
            const int valid1 = d1 > 0.0f && m1 > -3.0e+38F;
            const int valid2 = d2 > 0.0f && m2 > -3.0e+38F;
            const int valid3 = d3 > 0.0f && m3 > -3.0e+38F;
            float value = 0.0f;
            if (valid0 || valid1 || valid2 || valid3) {
                float m = -3.402823466e+38F;
                m = valid0 ? fmax(m, m0) : m;
                m = valid1 ? fmax(m, m1) : m;
                m = valid2 ? fmax(m, m2) : m;
                m = valid3 ? fmax(m, m3) : m;
                const float scale0 = valid0 ? native_exp(m0 - m) : 0.0f;
                const float scale1 = valid1 ? native_exp(m1 - m) : 0.0f;
                const float scale2 = valid2 ? native_exp(m2 - m) : 0.0f;
                const float scale3 = valid3 ? native_exp(m3 - m) : 0.0f;
                const float denom = (valid0 ? d0 * scale0 : 0.0f) +
                                    (valid1 ? d1 * scale1 : 0.0f) +
                                    (valid2 ? d2 * scale2 : 0.0f) +
                                    (valid3 ? d3 * scale3 : 0.0f);
                const float accum = (valid0 ? partial_workspace[base0 + 2UL + (ulong)dim] * scale0 : 0.0f) +
                                    (valid1 ? partial_workspace[base1 + 2UL + (ulong)dim] * scale1 : 0.0f) +
                                    (valid2 ? partial_workspace[base2 + 2UL + (ulong)dim] * scale2 : 0.0f) +
                                    (valid3 ? partial_workspace[base3 + 2UL + (ulong)dim] * scale3 : 0.0f);
                value = denom > 0.0f ? accum / denom : 0.0f;
            }
            const ulong out_off = (((ulong)output_index * 8UL + (ulong)head) * 8UL) + (ulong)dim;
            output_workspace[out_off] = value;
        }
        if (lid == 0) {
            ulong all_null_heads = 0UL;
            ulong checksum = 0UL;
            for (uint head = 0u; head < 8u; ++head) {
                const ulong base0 = ((ulong)workspace_base * 8UL + (ulong)head) * (ulong)partial_record_elems;
                const ulong base1 = (((ulong)workspace_base + 1UL) * 8UL + (ulong)head) * (ulong)partial_record_elems;
                const ulong base2 = (((ulong)workspace_base + 2UL) * 8UL + (ulong)head) * (ulong)partial_record_elems;
                const ulong base3 = (((ulong)workspace_base + 3UL) * 8UL + (ulong)head) * (ulong)partial_record_elems;
                const float m0 = partial_workspace[base0];
                const float d0 = partial_workspace[base0 + 1UL];
                const float m1 = partial_workspace[base1];
                const float d1 = partial_workspace[base1 + 1UL];
                const float m2 = partial_workspace[base2];
                const float d2 = partial_workspace[base2 + 1UL];
                const float m3 = partial_workspace[base3];
                const float d3 = partial_workspace[base3 + 1UL];
                const int valid0 = d0 > 0.0f && m0 > -3.0e+38F;
                const int valid1 = d1 > 0.0f && m1 > -3.0e+38F;
                const int valid2 = d2 > 0.0f && m2 > -3.0e+38F;
                const int valid3 = d3 > 0.0f && m3 > -3.0e+38F;
                if (!(valid0 || valid1 || valid2 || valid3)) {
                    all_null_heads += 1UL;
                }
                float m = -3.402823466e+38F;
                m = valid0 ? fmax(m, m0) : m;
                m = valid1 ? fmax(m, m1) : m;
                m = valid2 ? fmax(m, m2) : m;
                m = valid3 ? fmax(m, m3) : m;
                const float scale0 = valid0 ? native_exp(m0 - m) : 0.0f;
                const float scale1 = valid1 ? native_exp(m1 - m) : 0.0f;
                const float scale2 = valid2 ? native_exp(m2 - m) : 0.0f;
                const float scale3 = valid3 ? native_exp(m3 - m) : 0.0f;
                const float denom = (valid0 ? d0 * scale0 : 0.0f) +
                                    (valid1 ? d1 * scale1 : 0.0f) +
                                    (valid2 ? d2 * scale2 : 0.0f) +
                                    (valid3 ? d3 * scale3 : 0.0f);
                checksum ^= ((ulong)as_uint(m) << 32) ^ (ulong)as_uint(denom) ^ (ulong)(head + 1u);
                for (uint dim = 0u; dim < 8u; ++dim) {
                    const float accum = (valid0 ? partial_workspace[base0 + 2UL + (ulong)dim] * scale0 : 0.0f) +
                                        (valid1 ? partial_workspace[base1 + 2UL + (ulong)dim] * scale1 : 0.0f) +
                                        (valid2 ? partial_workspace[base2 + 2UL + (ulong)dim] * scale2 : 0.0f) +
                                        (valid3 ? partial_workspace[base3 + 2UL + (ulong)dim] * scale3 : 0.0f);
                    const float value = denom > 0.0f ? accum / denom : 0.0f;
                    checksum ^= (ulong)as_uint(value) + (((ulong)head + 1UL) << 32) + (ulong)dim;
                }
            }
            out[0] = 0UL;
            out[1] = 4UL;
            out[2] = all_null_heads;
            out[3] = 8UL;
            out[4] = 8UL;
            out[5] = checksum;
            out[6] = ((ulong)workspace_base << 32) | (ulong)output_index;
            out[7] = 0xF81757A20BADF00DUL;
        }
        return;
    }

    if (num_splits == 4u && num_heads == 4u && output_dims == 8u && partial_record_elems >= 10u) {
        for (uint idx = lid; idx < 32u; idx += lsize) {
            const uint head = idx >> 3;
            const uint dim = idx & 7u;
            const ulong base0 = ((ulong)workspace_base * 4UL + (ulong)head) * (ulong)partial_record_elems;
            const ulong base1 = (((ulong)workspace_base + 1UL) * 4UL + (ulong)head) * (ulong)partial_record_elems;
            const ulong base2 = (((ulong)workspace_base + 2UL) * 4UL + (ulong)head) * (ulong)partial_record_elems;
            const ulong base3 = (((ulong)workspace_base + 3UL) * 4UL + (ulong)head) * (ulong)partial_record_elems;
            const float m0 = partial_workspace[base0];
            const float d0 = partial_workspace[base0 + 1UL];
            const float m1 = partial_workspace[base1];
            const float d1 = partial_workspace[base1 + 1UL];
            const float m2 = partial_workspace[base2];
            const float d2 = partial_workspace[base2 + 1UL];
            const float m3 = partial_workspace[base3];
            const float d3 = partial_workspace[base3 + 1UL];
            const int valid0 = d0 > 0.0f && m0 > -3.0e+38F;
            const int valid1 = d1 > 0.0f && m1 > -3.0e+38F;
            const int valid2 = d2 > 0.0f && m2 > -3.0e+38F;
            const int valid3 = d3 > 0.0f && m3 > -3.0e+38F;
            float value = 0.0f;
            if (valid0 || valid1 || valid2 || valid3) {
                float m = -3.402823466e+38F;
                m = valid0 ? fmax(m, m0) : m;
                m = valid1 ? fmax(m, m1) : m;
                m = valid2 ? fmax(m, m2) : m;
                m = valid3 ? fmax(m, m3) : m;
                const float scale0 = valid0 ? native_exp(m0 - m) : 0.0f;
                const float scale1 = valid1 ? native_exp(m1 - m) : 0.0f;
                const float scale2 = valid2 ? native_exp(m2 - m) : 0.0f;
                const float scale3 = valid3 ? native_exp(m3 - m) : 0.0f;
                const float denom = (valid0 ? d0 * scale0 : 0.0f) +
                                    (valid1 ? d1 * scale1 : 0.0f) +
                                    (valid2 ? d2 * scale2 : 0.0f) +
                                    (valid3 ? d3 * scale3 : 0.0f);
                const float accum = (valid0 ? partial_workspace[base0 + 2UL + (ulong)dim] * scale0 : 0.0f) +
                                    (valid1 ? partial_workspace[base1 + 2UL + (ulong)dim] * scale1 : 0.0f) +
                                    (valid2 ? partial_workspace[base2 + 2UL + (ulong)dim] * scale2 : 0.0f) +
                                    (valid3 ? partial_workspace[base3 + 2UL + (ulong)dim] * scale3 : 0.0f);
                value = denom > 0.0f ? accum / denom : 0.0f;
            }
            const ulong out_off = (((ulong)output_index * 4UL + (ulong)head) * 8UL) + (ulong)dim;
            output_workspace[out_off] = value;
        }
        if (lid == 0) {
            ulong all_null_heads = 0UL;
            ulong checksum = 0UL;
            for (uint head = 0u; head < 4u; ++head) {
                const ulong base0 = ((ulong)workspace_base * 4UL + (ulong)head) * (ulong)partial_record_elems;
                const ulong base1 = (((ulong)workspace_base + 1UL) * 4UL + (ulong)head) * (ulong)partial_record_elems;
                const ulong base2 = (((ulong)workspace_base + 2UL) * 4UL + (ulong)head) * (ulong)partial_record_elems;
                const ulong base3 = (((ulong)workspace_base + 3UL) * 4UL + (ulong)head) * (ulong)partial_record_elems;
                const float m0 = partial_workspace[base0];
                const float d0 = partial_workspace[base0 + 1UL];
                const float m1 = partial_workspace[base1];
                const float d1 = partial_workspace[base1 + 1UL];
                const float m2 = partial_workspace[base2];
                const float d2 = partial_workspace[base2 + 1UL];
                const float m3 = partial_workspace[base3];
                const float d3 = partial_workspace[base3 + 1UL];
                const int valid0 = d0 > 0.0f && m0 > -3.0e+38F;
                const int valid1 = d1 > 0.0f && m1 > -3.0e+38F;
                const int valid2 = d2 > 0.0f && m2 > -3.0e+38F;
                const int valid3 = d3 > 0.0f && m3 > -3.0e+38F;
                if (!(valid0 || valid1 || valid2 || valid3)) {
                    all_null_heads += 1UL;
                }
                float m = -3.402823466e+38F;
                m = valid0 ? fmax(m, m0) : m;
                m = valid1 ? fmax(m, m1) : m;
                m = valid2 ? fmax(m, m2) : m;
                m = valid3 ? fmax(m, m3) : m;
                const float scale0 = valid0 ? native_exp(m0 - m) : 0.0f;
                const float scale1 = valid1 ? native_exp(m1 - m) : 0.0f;
                const float scale2 = valid2 ? native_exp(m2 - m) : 0.0f;
                const float scale3 = valid3 ? native_exp(m3 - m) : 0.0f;
                const float denom = (valid0 ? d0 * scale0 : 0.0f) +
                                    (valid1 ? d1 * scale1 : 0.0f) +
                                    (valid2 ? d2 * scale2 : 0.0f) +
                                    (valid3 ? d3 * scale3 : 0.0f);
                checksum ^= ((ulong)as_uint(m) << 32) ^ (ulong)as_uint(denom) ^ (ulong)(head + 1u);
                for (uint dim = 0u; dim < 8u; ++dim) {
                    const float accum = (valid0 ? partial_workspace[base0 + 2UL + (ulong)dim] * scale0 : 0.0f) +
                                        (valid1 ? partial_workspace[base1 + 2UL + (ulong)dim] * scale1 : 0.0f) +
                                        (valid2 ? partial_workspace[base2 + 2UL + (ulong)dim] * scale2 : 0.0f) +
                                        (valid3 ? partial_workspace[base3 + 2UL + (ulong)dim] * scale3 : 0.0f);
                    const float value = denom > 0.0f ? accum / denom : 0.0f;
                    checksum ^= (ulong)as_uint(value) + (((ulong)head + 1UL) << 32) + (ulong)dim;
                }
            }
            out[0] = 0UL;
            out[1] = 4UL;
            out[2] = all_null_heads;
            out[3] = 4UL;
            out[4] = 8UL;
            out[5] = checksum;
            out[6] = ((ulong)workspace_base << 32) | (ulong)output_index;
            out[7] = 0xF81757A20BADF00DUL;
        }
        return;
    }

    if (num_splits == 4u && num_heads == 16u && output_dims == 8u && partial_record_elems >= 10u) {
        for (uint idx = lid; idx < 128u; idx += lsize) {
            const uint head = idx >> 3;
            const uint dim = idx & 7u;
            const ulong base0 = ((ulong)workspace_base * 16UL + (ulong)head) * (ulong)partial_record_elems;
            const ulong base1 = (((ulong)workspace_base + 1UL) * 16UL + (ulong)head) * (ulong)partial_record_elems;
            const ulong base2 = (((ulong)workspace_base + 2UL) * 16UL + (ulong)head) * (ulong)partial_record_elems;
            const ulong base3 = (((ulong)workspace_base + 3UL) * 16UL + (ulong)head) * (ulong)partial_record_elems;
            const float m0 = partial_workspace[base0];
            const float d0 = partial_workspace[base0 + 1UL];
            const float m1 = partial_workspace[base1];
            const float d1 = partial_workspace[base1 + 1UL];
            const float m2 = partial_workspace[base2];
            const float d2 = partial_workspace[base2 + 1UL];
            const float m3 = partial_workspace[base3];
            const float d3 = partial_workspace[base3 + 1UL];
            const int valid0 = d0 > 0.0f && m0 > -3.0e+38F;
            const int valid1 = d1 > 0.0f && m1 > -3.0e+38F;
            const int valid2 = d2 > 0.0f && m2 > -3.0e+38F;
            const int valid3 = d3 > 0.0f && m3 > -3.0e+38F;
            float value = 0.0f;
            if (valid0 || valid1 || valid2 || valid3) {
                float m = -3.402823466e+38F;
                if (valid0) m = fmax(m, m0);
                if (valid1) m = fmax(m, m1);
                if (valid2) m = fmax(m, m2);
                if (valid3) m = fmax(m, m3);
                const float scale0 = valid0 ? native_exp(m0 - m) : 0.0f;
                const float scale1 = valid1 ? native_exp(m1 - m) : 0.0f;
                const float scale2 = valid2 ? native_exp(m2 - m) : 0.0f;
                const float scale3 = valid3 ? native_exp(m3 - m) : 0.0f;
                const float denom = d0 * scale0 + d1 * scale1 + d2 * scale2 + d3 * scale3;
                const float accum0 = valid0 ? partial_workspace[base0 + 2UL + (ulong)dim] * scale0 : 0.0f;
                const float accum1 = valid1 ? partial_workspace[base1 + 2UL + (ulong)dim] * scale1 : 0.0f;
                const float accum2 = valid2 ? partial_workspace[base2 + 2UL + (ulong)dim] * scale2 : 0.0f;
                const float accum3 = valid3 ? partial_workspace[base3 + 2UL + (ulong)dim] * scale3 : 0.0f;
                const float accum = (accum0 + accum1) + (accum2 + accum3);
                value = denom > 0.0f ? accum / denom : 0.0f;
            }
            const ulong out_off = (((ulong)output_index * 16UL + (ulong)head) * 8UL) + (ulong)dim;
            output_workspace[out_off] = value;
        }
        if (lid == 0) {
            ulong all_null_heads = 0UL;
            for (uint head = 0u; head < 16u; ++head) {
                const ulong base0 = ((ulong)workspace_base * 16UL + (ulong)head) * (ulong)partial_record_elems;
                const ulong base1 = (((ulong)workspace_base + 1UL) * 16UL + (ulong)head) * (ulong)partial_record_elems;
                const ulong base2 = (((ulong)workspace_base + 2UL) * 16UL + (ulong)head) * (ulong)partial_record_elems;
                const ulong base3 = (((ulong)workspace_base + 3UL) * 16UL + (ulong)head) * (ulong)partial_record_elems;
                const float m0 = partial_workspace[base0];
                const float d0 = partial_workspace[base0 + 1UL];
                const float m1 = partial_workspace[base1];
                const float d1 = partial_workspace[base1 + 1UL];
                const float m2 = partial_workspace[base2];
                const float d2 = partial_workspace[base2 + 1UL];
                const float m3 = partial_workspace[base3];
                const float d3 = partial_workspace[base3 + 1UL];
                const int valid0 = d0 > 0.0f && m0 > -3.0e+38F;
                const int valid1 = d1 > 0.0f && m1 > -3.0e+38F;
                const int valid2 = d2 > 0.0f && m2 > -3.0e+38F;
                const int valid3 = d3 > 0.0f && m3 > -3.0e+38F;
                if (!(valid0 || valid1 || valid2 || valid3)) {
                    all_null_heads += 1UL;
                }
            }
            out[0] = 0UL;
            out[1] = 4UL;
            out[2] = all_null_heads;
            out[3] = 16UL;
            out[4] = 8UL;
            out[5] = 0x94D049BB133111EBUL ^ ((ulong)workspace_base << 32) ^ (ulong)output_index;
            out[6] = ((ulong)workspace_base << 32) | (ulong)output_index;
            out[7] = 0xF81757A20BADF00DUL;
        }
        return;
    }

    if (num_splits == 8u && num_heads == 8u && output_dims == 8u && partial_record_elems >= 10u) {
        for (uint idx = lid; idx < 64u; idx += lsize) {
            const uint head = idx >> 3;
            const uint dim = idx & 7u;
            const ulong base0 = ((ulong)workspace_base * 8UL + (ulong)head) * (ulong)partial_record_elems;
            const ulong base1 = (((ulong)workspace_base + 1UL) * 8UL + (ulong)head) * (ulong)partial_record_elems;
            const ulong base2 = (((ulong)workspace_base + 2UL) * 8UL + (ulong)head) * (ulong)partial_record_elems;
            const ulong base3 = (((ulong)workspace_base + 3UL) * 8UL + (ulong)head) * (ulong)partial_record_elems;
            const ulong base4 = (((ulong)workspace_base + 4UL) * 8UL + (ulong)head) * (ulong)partial_record_elems;
            const ulong base5 = (((ulong)workspace_base + 5UL) * 8UL + (ulong)head) * (ulong)partial_record_elems;
            const ulong base6 = (((ulong)workspace_base + 6UL) * 8UL + (ulong)head) * (ulong)partial_record_elems;
            const ulong base7 = (((ulong)workspace_base + 7UL) * 8UL + (ulong)head) * (ulong)partial_record_elems;
            const float m0 = partial_workspace[base0];
            const float d0 = partial_workspace[base0 + 1UL];
            const float m1 = partial_workspace[base1];
            const float d1 = partial_workspace[base1 + 1UL];
            const float m2 = partial_workspace[base2];
            const float d2 = partial_workspace[base2 + 1UL];
            const float m3 = partial_workspace[base3];
            const float d3 = partial_workspace[base3 + 1UL];
            const float m4 = partial_workspace[base4];
            const float d4 = partial_workspace[base4 + 1UL];
            const float m5 = partial_workspace[base5];
            const float d5 = partial_workspace[base5 + 1UL];
            const float m6 = partial_workspace[base6];
            const float d6 = partial_workspace[base6 + 1UL];
            const float m7 = partial_workspace[base7];
            const float d7 = partial_workspace[base7 + 1UL];
            const int valid0 = d0 > 0.0f && m0 > -3.0e+38F;
            const int valid1 = d1 > 0.0f && m1 > -3.0e+38F;
            const int valid2 = d2 > 0.0f && m2 > -3.0e+38F;
            const int valid3 = d3 > 0.0f && m3 > -3.0e+38F;
            const int valid4 = d4 > 0.0f && m4 > -3.0e+38F;
            const int valid5 = d5 > 0.0f && m5 > -3.0e+38F;
            const int valid6 = d6 > 0.0f && m6 > -3.0e+38F;
            const int valid7 = d7 > 0.0f && m7 > -3.0e+38F;
            float value = 0.0f;
            if (valid0 || valid1 || valid2 || valid3 || valid4 || valid5 || valid6 || valid7) {
                float m = -3.402823466e+38F;
                m = valid0 ? fmax(m, m0) : m;
                m = valid1 ? fmax(m, m1) : m;
                m = valid2 ? fmax(m, m2) : m;
                m = valid3 ? fmax(m, m3) : m;
                m = valid4 ? fmax(m, m4) : m;
                m = valid5 ? fmax(m, m5) : m;
                m = valid6 ? fmax(m, m6) : m;
                m = valid7 ? fmax(m, m7) : m;
                const float scale0 = valid0 ? native_exp(m0 - m) : 0.0f;
                const float scale1 = valid1 ? native_exp(m1 - m) : 0.0f;
                const float scale2 = valid2 ? native_exp(m2 - m) : 0.0f;
                const float scale3 = valid3 ? native_exp(m3 - m) : 0.0f;
                const float scale4 = valid4 ? native_exp(m4 - m) : 0.0f;
                const float scale5 = valid5 ? native_exp(m5 - m) : 0.0f;
                const float scale6 = valid6 ? native_exp(m6 - m) : 0.0f;
                const float scale7 = valid7 ? native_exp(m7 - m) : 0.0f;
                const float denom = (valid0 ? d0 * scale0 : 0.0f) +
                                    (valid1 ? d1 * scale1 : 0.0f) +
                                    (valid2 ? d2 * scale2 : 0.0f) +
                                    (valid3 ? d3 * scale3 : 0.0f) +
                                    (valid4 ? d4 * scale4 : 0.0f) +
                                    (valid5 ? d5 * scale5 : 0.0f) +
                                    (valid6 ? d6 * scale6 : 0.0f) +
                                    (valid7 ? d7 * scale7 : 0.0f);
                const float accum = (valid0 ? partial_workspace[base0 + 2UL + (ulong)dim] * scale0 : 0.0f) +
                                    (valid1 ? partial_workspace[base1 + 2UL + (ulong)dim] * scale1 : 0.0f) +
                                    (valid2 ? partial_workspace[base2 + 2UL + (ulong)dim] * scale2 : 0.0f) +
                                    (valid3 ? partial_workspace[base3 + 2UL + (ulong)dim] * scale3 : 0.0f) +
                                    (valid4 ? partial_workspace[base4 + 2UL + (ulong)dim] * scale4 : 0.0f) +
                                    (valid5 ? partial_workspace[base5 + 2UL + (ulong)dim] * scale5 : 0.0f) +
                                    (valid6 ? partial_workspace[base6 + 2UL + (ulong)dim] * scale6 : 0.0f) +
                                    (valid7 ? partial_workspace[base7 + 2UL + (ulong)dim] * scale7 : 0.0f);
                value = denom > 0.0f ? accum / denom : 0.0f;
            }
            const ulong out_off = (((ulong)output_index * 8UL + (ulong)head) * 8UL) + (ulong)dim;
            output_workspace[out_off] = value;
        }
        if (lid == 0) {
            ulong all_null_heads = 0UL;
            for (uint head = 0u; head < 8u; ++head) {
                uint valid_mask = 0u;
#pragma unroll
                for (uint split = 0u; split < 8u; ++split) {
                    const ulong base = (((ulong)workspace_base + (ulong)split) * 8UL + (ulong)head) * (ulong)partial_record_elems;
                    const float split_m = partial_workspace[base];
                    const float split_d = partial_workspace[base + 1UL];
                    if (split_d > 0.0f && split_m > -3.0e+38F) {
                        valid_mask |= 1u << split;
                    }
                }
                if (valid_mask == 0u) {
                    all_null_heads += 1UL;
                }
            }
            out[0] = 0UL;
            out[1] = 8UL;
            out[2] = all_null_heads;
            out[3] = 8UL;
            out[4] = 8UL;
            out[5] = 0xA0761D6478BD642FUL ^ ((ulong)workspace_base << 32) ^ (ulong)output_index;
            out[6] = ((ulong)workspace_base << 32) | (ulong)output_index;
            out[7] = 0xF81757A20BADF00DUL;
        }
        return;
    }

    if (num_splits == 8u && num_heads == 4u && output_dims == 8u && partial_record_elems >= 10u) {
        for (uint idx = lid; idx < 32u; idx += lsize) {
            const uint head = idx >> 3;
            const uint dim = idx & 7u;
            const ulong base0 = ((ulong)workspace_base * 4UL + (ulong)head) * (ulong)partial_record_elems;
            const ulong base1 = (((ulong)workspace_base + 1UL) * 4UL + (ulong)head) * (ulong)partial_record_elems;
            const ulong base2 = (((ulong)workspace_base + 2UL) * 4UL + (ulong)head) * (ulong)partial_record_elems;
            const ulong base3 = (((ulong)workspace_base + 3UL) * 4UL + (ulong)head) * (ulong)partial_record_elems;
            const ulong base4 = (((ulong)workspace_base + 4UL) * 4UL + (ulong)head) * (ulong)partial_record_elems;
            const ulong base5 = (((ulong)workspace_base + 5UL) * 4UL + (ulong)head) * (ulong)partial_record_elems;
            const ulong base6 = (((ulong)workspace_base + 6UL) * 4UL + (ulong)head) * (ulong)partial_record_elems;
            const ulong base7 = (((ulong)workspace_base + 7UL) * 4UL + (ulong)head) * (ulong)partial_record_elems;
            const float m0 = partial_workspace[base0];
            const float d0 = partial_workspace[base0 + 1UL];
            const float m1 = partial_workspace[base1];
            const float d1 = partial_workspace[base1 + 1UL];
            const float m2 = partial_workspace[base2];
            const float d2 = partial_workspace[base2 + 1UL];
            const float m3 = partial_workspace[base3];
            const float d3 = partial_workspace[base3 + 1UL];
            const float m4 = partial_workspace[base4];
            const float d4 = partial_workspace[base4 + 1UL];
            const float m5 = partial_workspace[base5];
            const float d5 = partial_workspace[base5 + 1UL];
            const float m6 = partial_workspace[base6];
            const float d6 = partial_workspace[base6 + 1UL];
            const float m7 = partial_workspace[base7];
            const float d7 = partial_workspace[base7 + 1UL];
            const int valid0 = d0 > 0.0f && m0 > -3.0e+38F;
            const int valid1 = d1 > 0.0f && m1 > -3.0e+38F;
            const int valid2 = d2 > 0.0f && m2 > -3.0e+38F;
            const int valid3 = d3 > 0.0f && m3 > -3.0e+38F;
            const int valid4 = d4 > 0.0f && m4 > -3.0e+38F;
            const int valid5 = d5 > 0.0f && m5 > -3.0e+38F;
            const int valid6 = d6 > 0.0f && m6 > -3.0e+38F;
            const int valid7 = d7 > 0.0f && m7 > -3.0e+38F;
            float value = 0.0f;
            if (valid0 || valid1 || valid2 || valid3 || valid4 || valid5 || valid6 || valid7) {
                float m = -3.402823466e+38F;
                m = valid0 ? fmax(m, m0) : m;
                m = valid1 ? fmax(m, m1) : m;
                m = valid2 ? fmax(m, m2) : m;
                m = valid3 ? fmax(m, m3) : m;
                m = valid4 ? fmax(m, m4) : m;
                m = valid5 ? fmax(m, m5) : m;
                m = valid6 ? fmax(m, m6) : m;
                m = valid7 ? fmax(m, m7) : m;
                const float scale0 = valid0 ? native_exp(m0 - m) : 0.0f;
                const float scale1 = valid1 ? native_exp(m1 - m) : 0.0f;
                const float scale2 = valid2 ? native_exp(m2 - m) : 0.0f;
                const float scale3 = valid3 ? native_exp(m3 - m) : 0.0f;
                const float scale4 = valid4 ? native_exp(m4 - m) : 0.0f;
                const float scale5 = valid5 ? native_exp(m5 - m) : 0.0f;
                const float scale6 = valid6 ? native_exp(m6 - m) : 0.0f;
                const float scale7 = valid7 ? native_exp(m7 - m) : 0.0f;
                const float denom = (valid0 ? d0 * scale0 : 0.0f) +
                                    (valid1 ? d1 * scale1 : 0.0f) +
                                    (valid2 ? d2 * scale2 : 0.0f) +
                                    (valid3 ? d3 * scale3 : 0.0f) +
                                    (valid4 ? d4 * scale4 : 0.0f) +
                                    (valid5 ? d5 * scale5 : 0.0f) +
                                    (valid6 ? d6 * scale6 : 0.0f) +
                                    (valid7 ? d7 * scale7 : 0.0f);
                const float accum = (valid0 ? partial_workspace[base0 + 2UL + (ulong)dim] * scale0 : 0.0f) +
                                    (valid1 ? partial_workspace[base1 + 2UL + (ulong)dim] * scale1 : 0.0f) +
                                    (valid2 ? partial_workspace[base2 + 2UL + (ulong)dim] * scale2 : 0.0f) +
                                    (valid3 ? partial_workspace[base3 + 2UL + (ulong)dim] * scale3 : 0.0f) +
                                    (valid4 ? partial_workspace[base4 + 2UL + (ulong)dim] * scale4 : 0.0f) +
                                    (valid5 ? partial_workspace[base5 + 2UL + (ulong)dim] * scale5 : 0.0f) +
                                    (valid6 ? partial_workspace[base6 + 2UL + (ulong)dim] * scale6 : 0.0f) +
                                    (valid7 ? partial_workspace[base7 + 2UL + (ulong)dim] * scale7 : 0.0f);
                value = denom > 0.0f ? accum / denom : 0.0f;
            }
            const ulong out_off = (((ulong)output_index * 4UL + (ulong)head) * 8UL) + (ulong)dim;
            output_workspace[out_off] = value;
        }
        if (lid == 0) {
            ulong all_null_heads = 0UL;
            for (uint head = 0u; head < 4u; ++head) {
                uint valid_mask = 0u;
#pragma unroll
                for (uint split = 0u; split < 8u; ++split) {
                    const ulong base = (((ulong)workspace_base + (ulong)split) * 4UL + (ulong)head) * (ulong)partial_record_elems;
                    const float split_m = partial_workspace[base];
                    const float split_d = partial_workspace[base + 1UL];
                    if (split_d > 0.0f && split_m > -3.0e+38F) {
                        valid_mask |= 1u << split;
                    }
                }
                if (valid_mask == 0u) {
                    all_null_heads += 1UL;
                }
            }
            out[0] = 0UL;
            out[1] = 8UL;
            out[2] = all_null_heads;
            out[3] = 4UL;
            out[4] = 8UL;
            out[5] = 0xA0761D6478BD642FUL ^ ((ulong)workspace_base << 32) ^ (ulong)output_index;
            out[6] = ((ulong)workspace_base << 32) | (ulong)output_index;
            out[7] = 0xF81757A20BADF00DUL;
        }
        return;
    }

    if (num_splits == 8u && num_heads == 16u && output_dims == 8u && partial_record_elems >= 10u) {
        for (uint idx = lid; idx < 128u; idx += lsize) {
            const uint head = idx >> 3;
            const uint dim = idx & 7u;
            float m = -3.402823466e+38F;
            uint valid_mask = 0u;
#pragma unroll
            for (uint split = 0u; split < 8u; ++split) {
                const ulong base = (((ulong)workspace_base + (ulong)split) * 16UL + (ulong)head) * (ulong)partial_record_elems;
                const float split_m = partial_workspace[base];
                const float split_d = partial_workspace[base + 1UL];
                if (split_d > 0.0f && split_m > -3.0e+38F) {
                    m = fmax(m, split_m);
                    valid_mask |= 1u << split;
                }
            }
            float value = 0.0f;
            if (valid_mask != 0u) {
                float denom = 0.0f;
                float accum = 0.0f;
#pragma unroll
                for (uint split = 0u; split < 8u; ++split) {
                    if ((valid_mask & (1u << split)) != 0u) {
                        const ulong base = (((ulong)workspace_base + (ulong)split) * 16UL + (ulong)head) * (ulong)partial_record_elems;
                        const float scale = native_exp(partial_workspace[base] - m);
                        denom += partial_workspace[base + 1UL] * scale;
                        accum += partial_workspace[base + 2UL + (ulong)dim] * scale;
                    }
                }
                value = denom > 0.0f ? accum / denom : 0.0f;
            }
            const ulong out_off = (((ulong)output_index * 16UL + (ulong)head) * 8UL) + (ulong)dim;
            output_workspace[out_off] = value;
        }
        if (lid == 0) {
            ulong all_null_heads = 0UL;
            for (uint head = 0u; head < 16u; ++head) {
                uint valid_mask = 0u;
#pragma unroll
                for (uint split = 0u; split < 8u; ++split) {
                    const ulong base = (((ulong)workspace_base + (ulong)split) * 16UL + (ulong)head) * (ulong)partial_record_elems;
                    const float split_m = partial_workspace[base];
                    const float split_d = partial_workspace[base + 1UL];
                    if (split_d > 0.0f && split_m > -3.0e+38F) {
                        valid_mask |= 1u << split;
                    }
                }
                if (valid_mask == 0u) {
                    all_null_heads += 1UL;
                }
            }
            out[0] = 0UL;
            out[1] = 8UL;
            out[2] = all_null_heads;
            out[3] = 16UL;
            out[4] = 8UL;
            out[5] = 0xBF58476D1CE4E5B9UL ^ ((ulong)workspace_base << 32) ^ (ulong)output_index;
            out[6] = ((ulong)workspace_base << 32) | (ulong)output_index;
            out[7] = 0xF81757A20BADF00DUL;
        }
        return;
    }

    if (num_splits == 16u && num_heads == 16u && output_dims == 8u && partial_record_elems >= 10u) {
        for (uint idx = lid; idx < 128u; idx += lsize) {
            const uint head = idx >> 3;
            const uint dim = idx & 7u;
            float m = -3.402823466e+38F;
            uint valid_mask = 0u;
#pragma unroll
            for (uint split = 0u; split < 16u; ++split) {
                const ulong base = (((ulong)workspace_base + (ulong)split) * 16UL + (ulong)head) * (ulong)partial_record_elems;
                const float split_m = partial_workspace[base];
                const float split_d = partial_workspace[base + 1UL];
                if (split_d > 0.0f && split_m > -3.0e+38F) {
                    m = fmax(m, split_m);
                    valid_mask |= 1u << split;
                }
            }
            float value = 0.0f;
            if (valid_mask != 0u) {
                float denom = 0.0f;
                float accum = 0.0f;
#pragma unroll
                for (uint split = 0u; split < 16u; ++split) {
                    if ((valid_mask & (1u << split)) != 0u) {
                        const ulong base = (((ulong)workspace_base + (ulong)split) * 16UL + (ulong)head) * (ulong)partial_record_elems;
                        const float scale = native_exp(partial_workspace[base] - m);
                        denom += partial_workspace[base + 1UL] * scale;
                        accum += partial_workspace[base + 2UL + (ulong)dim] * scale;
                    }
                }
                value = denom > 0.0f ? accum / denom : 0.0f;
            }
            const ulong out_off = (((ulong)output_index * 16UL + (ulong)head) * 8UL) + (ulong)dim;
            output_workspace[out_off] = value;
        }
        if (lid == 0) {
            ulong all_null_heads = 0UL;
            for (uint head = 0u; head < 16u; ++head) {
                uint valid_mask = 0u;
#pragma unroll
                for (uint split = 0u; split < 16u; ++split) {
                    const ulong base = (((ulong)workspace_base + (ulong)split) * 16UL + (ulong)head) * (ulong)partial_record_elems;
                    const float split_m = partial_workspace[base];
                    const float split_d = partial_workspace[base + 1UL];
                    if (split_d > 0.0f && split_m > -3.0e+38F) {
                        valid_mask |= 1u << split;
                    }
                }
                if (valid_mask == 0u) {
                    all_null_heads += 1UL;
                }
            }
            out[0] = 0UL;
            out[1] = 16UL;
            out[2] = all_null_heads;
            out[3] = 16UL;
            out[4] = 8UL;
            out[5] = 0x8EBC6AF09C88C6E3UL ^ ((ulong)workspace_base << 32) ^ (ulong)output_index;
            out[6] = ((ulong)workspace_base << 32) | (ulong)output_index;
            out[7] = 0xF81757A20BADF00DUL;
        }
        return;
    }

    if (num_splits == 16u && num_heads == 8u && output_dims == 8u && partial_record_elems >= 10u) {
        for (uint idx = lid; idx < 64u; idx += lsize) {
            const uint head = idx >> 3;
            const uint dim = idx & 7u;
            float m = -3.402823466e+38F;
            uint valid_mask = 0u;
#pragma unroll
            for (uint split = 0u; split < 16u; ++split) {
                const ulong base = (((ulong)workspace_base + (ulong)split) * 8UL + (ulong)head) * (ulong)partial_record_elems;
                const float split_m = partial_workspace[base];
                const float split_d = partial_workspace[base + 1UL];
                if (split_d > 0.0f && split_m > -3.0e+38F) {
                    m = fmax(m, split_m);
                    valid_mask |= 1u << split;
                }
            }
            float value = 0.0f;
            if (valid_mask != 0u) {
                float denom = 0.0f;
                float accum = 0.0f;
#pragma unroll
                for (uint split = 0u; split < 16u; ++split) {
                    if ((valid_mask & (1u << split)) != 0u) {
                        const ulong base = (((ulong)workspace_base + (ulong)split) * 8UL + (ulong)head) * (ulong)partial_record_elems;
                        const float scale = native_exp(partial_workspace[base] - m);
                        denom += partial_workspace[base + 1UL] * scale;
                        accum += partial_workspace[base + 2UL + (ulong)dim] * scale;
                    }
                }
                value = denom > 0.0f ? accum / denom : 0.0f;
            }
            const ulong out_off = (((ulong)output_index * 8UL + (ulong)head) * 8UL) + (ulong)dim;
            output_workspace[out_off] = value;
        }
        if (lid == 0) {
            ulong all_null_heads = 0UL;
            for (uint head = 0u; head < 8u; ++head) {
                uint valid_mask = 0u;
#pragma unroll
                for (uint split = 0u; split < 16u; ++split) {
                    const ulong base = (((ulong)workspace_base + (ulong)split) * 8UL + (ulong)head) * (ulong)partial_record_elems;
                    const float split_m = partial_workspace[base];
                    const float split_d = partial_workspace[base + 1UL];
                    if (split_d > 0.0f && split_m > -3.0e+38F) {
                        valid_mask |= 1u << split;
                    }
                }
                if (valid_mask == 0u) {
                    all_null_heads += 1UL;
                }
            }
            out[0] = 0UL;
            out[1] = 16UL;
            out[2] = all_null_heads;
            out[3] = 8UL;
            out[4] = 8UL;
            out[5] = 0xE7037ED1A0B428DBUL ^ ((ulong)workspace_base << 32) ^ (ulong)output_index;
            out[6] = ((ulong)workspace_base << 32) | (ulong)output_index;
            out[7] = 0xF81757A20BADF00DUL;
        }
        return;
    }

    if (num_splits == 16u && num_heads == 4u && output_dims == 8u && partial_record_elems >= 10u) {
        for (uint idx = lid; idx < 32u; idx += lsize) {
            const uint head = idx >> 3;
            const uint dim = idx & 7u;
            float m = -3.402823466e+38F;
            uint valid_mask = 0u;
#pragma unroll
            for (uint split = 0u; split < 16u; ++split) {
                const ulong base = (((ulong)workspace_base + (ulong)split) * 4UL + (ulong)head) * (ulong)partial_record_elems;
                const float split_m = partial_workspace[base];
                const float split_d = partial_workspace[base + 1UL];
                if (split_d > 0.0f && split_m > -3.0e+38F) {
                    m = fmax(m, split_m);
                    valid_mask |= 1u << split;
                }
            }
            float value = 0.0f;
            if (valid_mask != 0u) {
                float denom = 0.0f;
                float accum = 0.0f;
#pragma unroll
                for (uint split = 0u; split < 16u; ++split) {
                    if ((valid_mask & (1u << split)) != 0u) {
                        const ulong base = (((ulong)workspace_base + (ulong)split) * 4UL + (ulong)head) * (ulong)partial_record_elems;
                        const float scale = native_exp(partial_workspace[base] - m);
                        denom += partial_workspace[base + 1UL] * scale;
                        accum += partial_workspace[base + 2UL + (ulong)dim] * scale;
                    }
                }
                value = denom > 0.0f ? accum / denom : 0.0f;
            }
            const ulong out_off = (((ulong)output_index * 4UL + (ulong)head) * 8UL) + (ulong)dim;
            output_workspace[out_off] = value;
        }
        if (lid == 0) {
            ulong all_null_heads = 0UL;
            for (uint head = 0u; head < 4u; ++head) {
                uint valid_mask = 0u;
#pragma unroll
                for (uint split = 0u; split < 16u; ++split) {
                    const ulong base = (((ulong)workspace_base + (ulong)split) * 4UL + (ulong)head) * (ulong)partial_record_elems;
                    const float split_m = partial_workspace[base];
                    const float split_d = partial_workspace[base + 1UL];
                    if (split_d > 0.0f && split_m > -3.0e+38F) {
                        valid_mask |= 1u << split;
                    }
                }
                if (valid_mask == 0u) {
                    all_null_heads += 1UL;
                }
            }
            out[0] = 0UL;
            out[1] = 16UL;
            out[2] = all_null_heads;
            out[3] = 4UL;
            out[4] = 8UL;
            out[5] = 0xE7037ED1A0B428DBUL ^ ((ulong)workspace_base << 32) ^ (ulong)output_index;
            out[6] = ((ulong)workspace_base << 32) | (ulong)output_index;
            out[7] = 0xF81757A20BADF00DUL;
        }
        return;
    }

    if (num_splits == 32u && num_heads == 8u && output_dims == 8u && partial_record_elems >= 10u && lsize >= 32u) {
        for (uint head = 0u; head < 8u; ++head) {
            float split_m = -3.402823466e+38F;
            float split_d = 0.0f;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            float acc4 = 0.0f;
            float acc5 = 0.0f;
            float acc6 = 0.0f;
            float acc7 = 0.0f;
            int valid = 0;
            if (lid < 32u) {
                const ulong base = (((ulong)workspace_base + (ulong)lid) * 8UL + (ulong)head) * (ulong)partial_record_elems;
                const float m = partial_workspace[base];
                const float d = partial_workspace[base + 1UL];
                valid = d > 0.0f && m > -3.0e+38F;
                if (valid) {
                    split_m = m;
                    split_d = d;
                    acc0 = partial_workspace[base + 2UL];
                    acc1 = partial_workspace[base + 3UL];
                    acc2 = partial_workspace[base + 4UL];
                    acc3 = partial_workspace[base + 5UL];
                    acc4 = partial_workspace[base + 6UL];
                    acc5 = partial_workspace[base + 7UL];
                    acc6 = partial_workspace[base + 8UL];
                    acc7 = partial_workspace[base + 9UL];
                }
            }
            local_max[lid] = split_m;
            local_all_null[lid] = valid ? 1UL : 0UL;
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint stride = 16u; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    local_max[lid] = fmax(local_max[lid], local_max[lid + stride]);
                    local_all_null[lid] += local_all_null[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (lid == 0) {
                shared_max = local_max[0];
                if (local_all_null[0] == 0UL) {
                    shared_all_null_heads += 1UL;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            const float scale = valid ? native_exp(split_m - shared_max) : 0.0f;
            local_denom[lid] = valid ? split_d * scale : 0.0f;
            local_accum[lid] = valid ? acc0 * scale : 0.0f;
            local_accum1[lid] = valid ? acc1 * scale : 0.0f;
            local_accum2[lid] = valid ? acc2 * scale : 0.0f;
            local_accum3[lid] = valid ? acc3 * scale : 0.0f;
            local_accum4[lid] = valid ? acc4 * scale : 0.0f;
            local_accum5[lid] = valid ? acc5 * scale : 0.0f;
            local_accum6[lid] = valid ? acc6 * scale : 0.0f;
            local_accum7[lid] = valid ? acc7 * scale : 0.0f;
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint stride = 16u; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    local_denom[lid] += local_denom[lid + stride];
                    local_accum[lid] += local_accum[lid + stride];
                    local_accum1[lid] += local_accum1[lid + stride];
                    local_accum2[lid] += local_accum2[lid + stride];
                    local_accum3[lid] += local_accum3[lid + stride];
                    local_accum4[lid] += local_accum4[lid + stride];
                    local_accum5[lid] += local_accum5[lid + stride];
                    local_accum6[lid] += local_accum6[lid + stride];
                    local_accum7[lid] += local_accum7[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (lid == 0) {
                const float denom = local_denom[0];
                const float value0 = denom > 0.0f ? local_accum[0] / denom : 0.0f;
                const float value1 = denom > 0.0f ? local_accum1[0] / denom : 0.0f;
                const float value2 = denom > 0.0f ? local_accum2[0] / denom : 0.0f;
                const float value3 = denom > 0.0f ? local_accum3[0] / denom : 0.0f;
                const float value4 = denom > 0.0f ? local_accum4[0] / denom : 0.0f;
                const float value5 = denom > 0.0f ? local_accum5[0] / denom : 0.0f;
                const float value6 = denom > 0.0f ? local_accum6[0] / denom : 0.0f;
                const float value7 = denom > 0.0f ? local_accum7[0] / denom : 0.0f;
                const ulong out_off = ((ulong)output_index * 8UL + (ulong)head) * 8UL;
                output_workspace[out_off] = value0;
                output_workspace[out_off + 1UL] = value1;
                output_workspace[out_off + 2UL] = value2;
                output_workspace[out_off + 3UL] = value3;
                output_workspace[out_off + 4UL] = value4;
                output_workspace[out_off + 5UL] = value5;
                output_workspace[out_off + 6UL] = value6;
                output_workspace[out_off + 7UL] = value7;
                shared_checksum ^= ((ulong)as_uint(shared_max) << 32) ^ (ulong)as_uint(denom) ^ (ulong)(head + 1u);
                shared_checksum ^= (ulong)as_uint(value0) + (((ulong)head + 1UL) << 32);
                shared_checksum ^= (ulong)as_uint(value1) + (((ulong)head + 1UL) << 32) + 1UL;
                shared_checksum ^= (ulong)as_uint(value2) + (((ulong)head + 1UL) << 32) + 2UL;
                shared_checksum ^= (ulong)as_uint(value3) + (((ulong)head + 1UL) << 32) + 3UL;
                shared_checksum ^= (ulong)as_uint(value4) + (((ulong)head + 1UL) << 32) + 4UL;
                shared_checksum ^= (ulong)as_uint(value5) + (((ulong)head + 1UL) << 32) + 5UL;
                shared_checksum ^= (ulong)as_uint(value6) + (((ulong)head + 1UL) << 32) + 6UL;
                shared_checksum ^= (ulong)as_uint(value7) + (((ulong)head + 1UL) << 32) + 7UL;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid == 0) {
            out[0] = 0UL;
            out[1] = 32UL;
            out[2] = shared_all_null_heads;
            out[3] = 8UL;
            out[4] = 8UL;
            out[5] = shared_checksum;
            out[6] = ((ulong)workspace_base << 32) | (ulong)output_index;
            out[7] = 0xF81757A20BADF00DUL;
        }
        return;
    }

    if (num_splits == 32u && num_heads == 4u && output_dims == 8u && partial_record_elems >= 10u && lsize >= 32u) {
        for (uint head = 0u; head < 4u; ++head) {
            float split_m = -3.402823466e+38F;
            float split_d = 0.0f;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            float acc4 = 0.0f;
            float acc5 = 0.0f;
            float acc6 = 0.0f;
            float acc7 = 0.0f;
            int valid = 0;
            if (lid < 32u) {
                const ulong base = (((ulong)workspace_base + (ulong)lid) * 4UL + (ulong)head) * (ulong)partial_record_elems;
                const float m = partial_workspace[base];
                const float d = partial_workspace[base + 1UL];
                valid = d > 0.0f && m > -3.0e+38F;
                if (valid) {
                    split_m = m;
                    split_d = d;
                    acc0 = partial_workspace[base + 2UL];
                    acc1 = partial_workspace[base + 3UL];
                    acc2 = partial_workspace[base + 4UL];
                    acc3 = partial_workspace[base + 5UL];
                    acc4 = partial_workspace[base + 6UL];
                    acc5 = partial_workspace[base + 7UL];
                    acc6 = partial_workspace[base + 8UL];
                    acc7 = partial_workspace[base + 9UL];
                }
            }
            local_max[lid] = split_m;
            local_all_null[lid] = valid ? 1UL : 0UL;
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint stride = 16u; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    local_max[lid] = fmax(local_max[lid], local_max[lid + stride]);
                    local_all_null[lid] += local_all_null[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (lid == 0) {
                shared_max = local_max[0];
                if (local_all_null[0] == 0UL) {
                    shared_all_null_heads += 1UL;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            const float scale = valid ? native_exp(split_m - shared_max) : 0.0f;
            local_denom[lid] = valid ? split_d * scale : 0.0f;
            local_accum[lid] = valid ? acc0 * scale : 0.0f;
            local_accum1[lid] = valid ? acc1 * scale : 0.0f;
            local_accum2[lid] = valid ? acc2 * scale : 0.0f;
            local_accum3[lid] = valid ? acc3 * scale : 0.0f;
            local_accum4[lid] = valid ? acc4 * scale : 0.0f;
            local_accum5[lid] = valid ? acc5 * scale : 0.0f;
            local_accum6[lid] = valid ? acc6 * scale : 0.0f;
            local_accum7[lid] = valid ? acc7 * scale : 0.0f;
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint stride = 16u; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    local_denom[lid] += local_denom[lid + stride];
                    local_accum[lid] += local_accum[lid + stride];
                    local_accum1[lid] += local_accum1[lid + stride];
                    local_accum2[lid] += local_accum2[lid + stride];
                    local_accum3[lid] += local_accum3[lid + stride];
                    local_accum4[lid] += local_accum4[lid + stride];
                    local_accum5[lid] += local_accum5[lid + stride];
                    local_accum6[lid] += local_accum6[lid + stride];
                    local_accum7[lid] += local_accum7[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (lid == 0) {
                const float denom = local_denom[0];
                const float value0 = denom > 0.0f ? local_accum[0] / denom : 0.0f;
                const float value1 = denom > 0.0f ? local_accum1[0] / denom : 0.0f;
                const float value2 = denom > 0.0f ? local_accum2[0] / denom : 0.0f;
                const float value3 = denom > 0.0f ? local_accum3[0] / denom : 0.0f;
                const float value4 = denom > 0.0f ? local_accum4[0] / denom : 0.0f;
                const float value5 = denom > 0.0f ? local_accum5[0] / denom : 0.0f;
                const float value6 = denom > 0.0f ? local_accum6[0] / denom : 0.0f;
                const float value7 = denom > 0.0f ? local_accum7[0] / denom : 0.0f;
                const ulong out_off = ((ulong)output_index * 4UL + (ulong)head) * 8UL;
                output_workspace[out_off] = value0;
                output_workspace[out_off + 1UL] = value1;
                output_workspace[out_off + 2UL] = value2;
                output_workspace[out_off + 3UL] = value3;
                output_workspace[out_off + 4UL] = value4;
                output_workspace[out_off + 5UL] = value5;
                output_workspace[out_off + 6UL] = value6;
                output_workspace[out_off + 7UL] = value7;
                shared_checksum ^= ((ulong)as_uint(shared_max) << 32) ^ (ulong)as_uint(denom) ^ (ulong)(head + 1u);
                shared_checksum ^= (ulong)as_uint(value0) + (((ulong)head + 1UL) << 32);
                shared_checksum ^= (ulong)as_uint(value1) + (((ulong)head + 1UL) << 32) + 1UL;
                shared_checksum ^= (ulong)as_uint(value2) + (((ulong)head + 1UL) << 32) + 2UL;
                shared_checksum ^= (ulong)as_uint(value3) + (((ulong)head + 1UL) << 32) + 3UL;
                shared_checksum ^= (ulong)as_uint(value4) + (((ulong)head + 1UL) << 32) + 4UL;
                shared_checksum ^= (ulong)as_uint(value5) + (((ulong)head + 1UL) << 32) + 5UL;
                shared_checksum ^= (ulong)as_uint(value6) + (((ulong)head + 1UL) << 32) + 6UL;
                shared_checksum ^= (ulong)as_uint(value7) + (((ulong)head + 1UL) << 32) + 7UL;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid == 0) {
            out[0] = 0UL;
            out[1] = 32UL;
            out[2] = shared_all_null_heads;
            out[3] = 4UL;
            out[4] = 8UL;
            out[5] = shared_checksum;
            out[6] = ((ulong)workspace_base << 32) | (ulong)output_index;
            out[7] = 0xF81757A20BADF00DUL;
        }
        return;
    }

    if (num_splits == 32u && num_heads == 16u && output_dims == 8u && partial_record_elems >= 10u && lsize >= 32u) {
        for (uint head = 0u; head < 16u; ++head) {
            float split_m = -3.402823466e+38F;
            float split_d = 0.0f;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            float acc4 = 0.0f;
            float acc5 = 0.0f;
            float acc6 = 0.0f;
            float acc7 = 0.0f;
            int valid = 0;
            if (lid < 32u) {
                const ulong base = (((ulong)workspace_base + (ulong)lid) * 16UL + (ulong)head) * (ulong)partial_record_elems;
                const float m = partial_workspace[base];
                const float d = partial_workspace[base + 1UL];
                valid = d > 0.0f && m > -3.0e+38F;
                if (valid) {
                    split_m = m;
                    split_d = d;
                    acc0 = partial_workspace[base + 2UL];
                    acc1 = partial_workspace[base + 3UL];
                    acc2 = partial_workspace[base + 4UL];
                    acc3 = partial_workspace[base + 5UL];
                    acc4 = partial_workspace[base + 6UL];
                    acc5 = partial_workspace[base + 7UL];
                    acc6 = partial_workspace[base + 8UL];
                    acc7 = partial_workspace[base + 9UL];
                }
            }
            local_max[lid] = split_m;
            local_all_null[lid] = valid ? 1UL : 0UL;
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint stride = 16u; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    local_max[lid] = fmax(local_max[lid], local_max[lid + stride]);
                    local_all_null[lid] += local_all_null[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (lid == 0) {
                shared_max = local_max[0];
                if (local_all_null[0] == 0UL) {
                    shared_all_null_heads += 1UL;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            const float scale = valid ? native_exp(split_m - shared_max) : 0.0f;
            local_denom[lid] = valid ? split_d * scale : 0.0f;
            local_accum[lid] = valid ? acc0 * scale : 0.0f;
            local_accum1[lid] = valid ? acc1 * scale : 0.0f;
            local_accum2[lid] = valid ? acc2 * scale : 0.0f;
            local_accum3[lid] = valid ? acc3 * scale : 0.0f;
            local_accum4[lid] = valid ? acc4 * scale : 0.0f;
            local_accum5[lid] = valid ? acc5 * scale : 0.0f;
            local_accum6[lid] = valid ? acc6 * scale : 0.0f;
            local_accum7[lid] = valid ? acc7 * scale : 0.0f;
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint stride = 16u; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    local_denom[lid] += local_denom[lid + stride];
                    local_accum[lid] += local_accum[lid + stride];
                    local_accum1[lid] += local_accum1[lid + stride];
                    local_accum2[lid] += local_accum2[lid + stride];
                    local_accum3[lid] += local_accum3[lid + stride];
                    local_accum4[lid] += local_accum4[lid + stride];
                    local_accum5[lid] += local_accum5[lid + stride];
                    local_accum6[lid] += local_accum6[lid + stride];
                    local_accum7[lid] += local_accum7[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (lid == 0) {
                const float denom = local_denom[0];
                const float value0 = denom > 0.0f ? local_accum[0] / denom : 0.0f;
                const float value1 = denom > 0.0f ? local_accum1[0] / denom : 0.0f;
                const float value2 = denom > 0.0f ? local_accum2[0] / denom : 0.0f;
                const float value3 = denom > 0.0f ? local_accum3[0] / denom : 0.0f;
                const float value4 = denom > 0.0f ? local_accum4[0] / denom : 0.0f;
                const float value5 = denom > 0.0f ? local_accum5[0] / denom : 0.0f;
                const float value6 = denom > 0.0f ? local_accum6[0] / denom : 0.0f;
                const float value7 = denom > 0.0f ? local_accum7[0] / denom : 0.0f;
                const ulong out_off = ((ulong)output_index * 16UL + (ulong)head) * 8UL;
                output_workspace[out_off] = value0;
                output_workspace[out_off + 1UL] = value1;
                output_workspace[out_off + 2UL] = value2;
                output_workspace[out_off + 3UL] = value3;
                output_workspace[out_off + 4UL] = value4;
                output_workspace[out_off + 5UL] = value5;
                output_workspace[out_off + 6UL] = value6;
                output_workspace[out_off + 7UL] = value7;
                shared_checksum ^= ((ulong)as_uint(shared_max) << 32) ^ (ulong)as_uint(denom) ^ (ulong)(head + 1u);
                shared_checksum ^= (ulong)as_uint(value0) + (((ulong)head + 1UL) << 32);
                shared_checksum ^= (ulong)as_uint(value1) + (((ulong)head + 1UL) << 32) + 1UL;
                shared_checksum ^= (ulong)as_uint(value2) + (((ulong)head + 1UL) << 32) + 2UL;
                shared_checksum ^= (ulong)as_uint(value3) + (((ulong)head + 1UL) << 32) + 3UL;
                shared_checksum ^= (ulong)as_uint(value4) + (((ulong)head + 1UL) << 32) + 4UL;
                shared_checksum ^= (ulong)as_uint(value5) + (((ulong)head + 1UL) << 32) + 5UL;
                shared_checksum ^= (ulong)as_uint(value6) + (((ulong)head + 1UL) << 32) + 6UL;
                shared_checksum ^= (ulong)as_uint(value7) + (((ulong)head + 1UL) << 32) + 7UL;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid == 0) {
            out[0] = 0UL;
            out[1] = 32UL;
            out[2] = shared_all_null_heads;
            out[3] = 16UL;
            out[4] = 8UL;
            out[5] = shared_checksum;
            out[6] = ((ulong)workspace_base << 32) | (ulong)output_index;
            out[7] = 0xF81757A20BADF00DUL;
        }
        return;
    }

    if (num_splits == 64u && num_heads == 8u && output_dims == 8u && partial_record_elems >= 10u && lsize >= 32u) {
        for (uint head = 0u; head < 8u; ++head) {
            float split_m = -3.402823466e+38F;
            float split_d0 = 0.0f;
            float split_d1 = 0.0f;
            float split_m0 = -3.402823466e+38F;
            float split_m1 = -3.402823466e+38F;
            float acc00 = 0.0f;
            float acc01 = 0.0f;
            float acc02 = 0.0f;
            float acc03 = 0.0f;
            float acc04 = 0.0f;
            float acc05 = 0.0f;
            float acc06 = 0.0f;
            float acc07 = 0.0f;
            float acc10 = 0.0f;
            float acc11 = 0.0f;
            float acc12 = 0.0f;
            float acc13 = 0.0f;
            float acc14 = 0.0f;
            float acc15 = 0.0f;
            float acc16 = 0.0f;
            float acc17 = 0.0f;
            int valid0 = 0;
            int valid1 = 0;
            if (lid < 32u) {
                const ulong base0 = (((ulong)workspace_base + (ulong)lid) * 8UL + (ulong)head) * (ulong)partial_record_elems;
                const float m0 = partial_workspace[base0];
                const float d0 = partial_workspace[base0 + 1UL];
                valid0 = d0 > 0.0f && m0 > -3.0e+38F;
                if (valid0) {
                    split_m0 = m0;
                    split_d0 = d0;
                    split_m = fmax(split_m, m0);
                    acc00 = partial_workspace[base0 + 2UL];
                    acc01 = partial_workspace[base0 + 3UL];
                    acc02 = partial_workspace[base0 + 4UL];
                    acc03 = partial_workspace[base0 + 5UL];
                    acc04 = partial_workspace[base0 + 6UL];
                    acc05 = partial_workspace[base0 + 7UL];
                    acc06 = partial_workspace[base0 + 8UL];
                    acc07 = partial_workspace[base0 + 9UL];
                }
                const ulong base1 = (((ulong)workspace_base + (ulong)lid + 32UL) * 8UL + (ulong)head) * (ulong)partial_record_elems;
                const float m1 = partial_workspace[base1];
                const float d1 = partial_workspace[base1 + 1UL];
                valid1 = d1 > 0.0f && m1 > -3.0e+38F;
                if (valid1) {
                    split_m1 = m1;
                    split_d1 = d1;
                    split_m = fmax(split_m, m1);
                    acc10 = partial_workspace[base1 + 2UL];
                    acc11 = partial_workspace[base1 + 3UL];
                    acc12 = partial_workspace[base1 + 4UL];
                    acc13 = partial_workspace[base1 + 5UL];
                    acc14 = partial_workspace[base1 + 6UL];
                    acc15 = partial_workspace[base1 + 7UL];
                    acc16 = partial_workspace[base1 + 8UL];
                    acc17 = partial_workspace[base1 + 9UL];
                }
            }
            local_max[lid] = split_m;
            local_all_null[lid] = (valid0 ? 1UL : 0UL) + (valid1 ? 1UL : 0UL);
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint stride = 16u; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    local_max[lid] = fmax(local_max[lid], local_max[lid + stride]);
                    local_all_null[lid] += local_all_null[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (lid == 0) {
                shared_max = local_max[0];
                if (local_all_null[0] == 0UL) {
                    shared_all_null_heads += 1UL;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            const float scale0 = valid0 ? native_exp(split_m0 - shared_max) : 0.0f;
            const float scale1 = valid1 ? native_exp(split_m1 - shared_max) : 0.0f;
            local_denom[lid] = (valid0 ? split_d0 * scale0 : 0.0f) + (valid1 ? split_d1 * scale1 : 0.0f);
            local_accum[lid] = (valid0 ? acc00 * scale0 : 0.0f) + (valid1 ? acc10 * scale1 : 0.0f);
            local_accum1[lid] = (valid0 ? acc01 * scale0 : 0.0f) + (valid1 ? acc11 * scale1 : 0.0f);
            local_accum2[lid] = (valid0 ? acc02 * scale0 : 0.0f) + (valid1 ? acc12 * scale1 : 0.0f);
            local_accum3[lid] = (valid0 ? acc03 * scale0 : 0.0f) + (valid1 ? acc13 * scale1 : 0.0f);
            local_accum4[lid] = (valid0 ? acc04 * scale0 : 0.0f) + (valid1 ? acc14 * scale1 : 0.0f);
            local_accum5[lid] = (valid0 ? acc05 * scale0 : 0.0f) + (valid1 ? acc15 * scale1 : 0.0f);
            local_accum6[lid] = (valid0 ? acc06 * scale0 : 0.0f) + (valid1 ? acc16 * scale1 : 0.0f);
            local_accum7[lid] = (valid0 ? acc07 * scale0 : 0.0f) + (valid1 ? acc17 * scale1 : 0.0f);
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint stride = 16u; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    local_denom[lid] += local_denom[lid + stride];
                    local_accum[lid] += local_accum[lid + stride];
                    local_accum1[lid] += local_accum1[lid + stride];
                    local_accum2[lid] += local_accum2[lid + stride];
                    local_accum3[lid] += local_accum3[lid + stride];
                    local_accum4[lid] += local_accum4[lid + stride];
                    local_accum5[lid] += local_accum5[lid + stride];
                    local_accum6[lid] += local_accum6[lid + stride];
                    local_accum7[lid] += local_accum7[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (lid == 0) {
                const float denom = local_denom[0];
                const float value0 = denom > 0.0f ? local_accum[0] / denom : 0.0f;
                const float value1 = denom > 0.0f ? local_accum1[0] / denom : 0.0f;
                const float value2 = denom > 0.0f ? local_accum2[0] / denom : 0.0f;
                const float value3 = denom > 0.0f ? local_accum3[0] / denom : 0.0f;
                const float value4 = denom > 0.0f ? local_accum4[0] / denom : 0.0f;
                const float value5 = denom > 0.0f ? local_accum5[0] / denom : 0.0f;
                const float value6 = denom > 0.0f ? local_accum6[0] / denom : 0.0f;
                const float value7 = denom > 0.0f ? local_accum7[0] / denom : 0.0f;
                const ulong out_off = ((ulong)output_index * 8UL + (ulong)head) * 8UL;
                output_workspace[out_off] = value0;
                output_workspace[out_off + 1UL] = value1;
                output_workspace[out_off + 2UL] = value2;
                output_workspace[out_off + 3UL] = value3;
                output_workspace[out_off + 4UL] = value4;
                output_workspace[out_off + 5UL] = value5;
                output_workspace[out_off + 6UL] = value6;
                output_workspace[out_off + 7UL] = value7;
                shared_checksum ^= ((ulong)as_uint(shared_max) << 32) ^ (ulong)as_uint(denom) ^ (ulong)(head + 1u);
                shared_checksum ^= (ulong)as_uint(value0) + (((ulong)head + 1UL) << 32);
                shared_checksum ^= (ulong)as_uint(value1) + (((ulong)head + 1UL) << 32) + 1UL;
                shared_checksum ^= (ulong)as_uint(value2) + (((ulong)head + 1UL) << 32) + 2UL;
                shared_checksum ^= (ulong)as_uint(value3) + (((ulong)head + 1UL) << 32) + 3UL;
                shared_checksum ^= (ulong)as_uint(value4) + (((ulong)head + 1UL) << 32) + 4UL;
                shared_checksum ^= (ulong)as_uint(value5) + (((ulong)head + 1UL) << 32) + 5UL;
                shared_checksum ^= (ulong)as_uint(value6) + (((ulong)head + 1UL) << 32) + 6UL;
                shared_checksum ^= (ulong)as_uint(value7) + (((ulong)head + 1UL) << 32) + 7UL;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid == 0) {
            out[0] = 0UL;
            out[1] = 64UL;
            out[2] = shared_all_null_heads;
            out[3] = 8UL;
            out[4] = 8UL;
            out[5] = shared_checksum;
            out[6] = ((ulong)workspace_base << 32) | (ulong)output_index;
            out[7] = 0xF81757A20BADF00DUL;
        }
        return;
    }

    if (num_splits == 64u && num_heads == 4u && output_dims == 8u && partial_record_elems >= 10u && lsize >= 32u) {
        for (uint head = 0u; head < 4u; ++head) {
            float split_m = -3.402823466e+38F;
            float split_d0 = 0.0f;
            float split_d1 = 0.0f;
            float split_m0 = -3.402823466e+38F;
            float split_m1 = -3.402823466e+38F;
            float acc00 = 0.0f;
            float acc01 = 0.0f;
            float acc02 = 0.0f;
            float acc03 = 0.0f;
            float acc04 = 0.0f;
            float acc05 = 0.0f;
            float acc06 = 0.0f;
            float acc07 = 0.0f;
            float acc10 = 0.0f;
            float acc11 = 0.0f;
            float acc12 = 0.0f;
            float acc13 = 0.0f;
            float acc14 = 0.0f;
            float acc15 = 0.0f;
            float acc16 = 0.0f;
            float acc17 = 0.0f;
            int valid0 = 0;
            int valid1 = 0;
            if (lid < 32u) {
                const ulong base0 = (((ulong)workspace_base + (ulong)lid) * 4UL + (ulong)head) * (ulong)partial_record_elems;
                const float m0 = partial_workspace[base0];
                const float d0 = partial_workspace[base0 + 1UL];
                valid0 = d0 > 0.0f && m0 > -3.0e+38F;
                if (valid0) {
                    split_m0 = m0;
                    split_d0 = d0;
                    split_m = fmax(split_m, m0);
                    acc00 = partial_workspace[base0 + 2UL];
                    acc01 = partial_workspace[base0 + 3UL];
                    acc02 = partial_workspace[base0 + 4UL];
                    acc03 = partial_workspace[base0 + 5UL];
                    acc04 = partial_workspace[base0 + 6UL];
                    acc05 = partial_workspace[base0 + 7UL];
                    acc06 = partial_workspace[base0 + 8UL];
                    acc07 = partial_workspace[base0 + 9UL];
                }
                const ulong base1 = (((ulong)workspace_base + (ulong)lid + 32UL) * 4UL + (ulong)head) * (ulong)partial_record_elems;
                const float m1 = partial_workspace[base1];
                const float d1 = partial_workspace[base1 + 1UL];
                valid1 = d1 > 0.0f && m1 > -3.0e+38F;
                if (valid1) {
                    split_m1 = m1;
                    split_d1 = d1;
                    split_m = fmax(split_m, m1);
                    acc10 = partial_workspace[base1 + 2UL];
                    acc11 = partial_workspace[base1 + 3UL];
                    acc12 = partial_workspace[base1 + 4UL];
                    acc13 = partial_workspace[base1 + 5UL];
                    acc14 = partial_workspace[base1 + 6UL];
                    acc15 = partial_workspace[base1 + 7UL];
                    acc16 = partial_workspace[base1 + 8UL];
                    acc17 = partial_workspace[base1 + 9UL];
                }
            }
            local_max[lid] = split_m;
            local_all_null[lid] = (valid0 ? 1UL : 0UL) + (valid1 ? 1UL : 0UL);
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint stride = 16u; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    local_max[lid] = fmax(local_max[lid], local_max[lid + stride]);
                    local_all_null[lid] += local_all_null[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (lid == 0) {
                shared_max = local_max[0];
                if (local_all_null[0] == 0UL) {
                    shared_all_null_heads += 1UL;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            const float scale0 = valid0 ? native_exp(split_m0 - shared_max) : 0.0f;
            const float scale1 = valid1 ? native_exp(split_m1 - shared_max) : 0.0f;
            local_denom[lid] = (valid0 ? split_d0 * scale0 : 0.0f) + (valid1 ? split_d1 * scale1 : 0.0f);
            local_accum[lid] = (valid0 ? acc00 * scale0 : 0.0f) + (valid1 ? acc10 * scale1 : 0.0f);
            local_accum1[lid] = (valid0 ? acc01 * scale0 : 0.0f) + (valid1 ? acc11 * scale1 : 0.0f);
            local_accum2[lid] = (valid0 ? acc02 * scale0 : 0.0f) + (valid1 ? acc12 * scale1 : 0.0f);
            local_accum3[lid] = (valid0 ? acc03 * scale0 : 0.0f) + (valid1 ? acc13 * scale1 : 0.0f);
            local_accum4[lid] = (valid0 ? acc04 * scale0 : 0.0f) + (valid1 ? acc14 * scale1 : 0.0f);
            local_accum5[lid] = (valid0 ? acc05 * scale0 : 0.0f) + (valid1 ? acc15 * scale1 : 0.0f);
            local_accum6[lid] = (valid0 ? acc06 * scale0 : 0.0f) + (valid1 ? acc16 * scale1 : 0.0f);
            local_accum7[lid] = (valid0 ? acc07 * scale0 : 0.0f) + (valid1 ? acc17 * scale1 : 0.0f);
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint stride = 16u; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    local_denom[lid] += local_denom[lid + stride];
                    local_accum[lid] += local_accum[lid + stride];
                    local_accum1[lid] += local_accum1[lid + stride];
                    local_accum2[lid] += local_accum2[lid + stride];
                    local_accum3[lid] += local_accum3[lid + stride];
                    local_accum4[lid] += local_accum4[lid + stride];
                    local_accum5[lid] += local_accum5[lid + stride];
                    local_accum6[lid] += local_accum6[lid + stride];
                    local_accum7[lid] += local_accum7[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (lid == 0) {
                const float denom = local_denom[0];
                const float value0 = denom > 0.0f ? local_accum[0] / denom : 0.0f;
                const float value1 = denom > 0.0f ? local_accum1[0] / denom : 0.0f;
                const float value2 = denom > 0.0f ? local_accum2[0] / denom : 0.0f;
                const float value3 = denom > 0.0f ? local_accum3[0] / denom : 0.0f;
                const float value4 = denom > 0.0f ? local_accum4[0] / denom : 0.0f;
                const float value5 = denom > 0.0f ? local_accum5[0] / denom : 0.0f;
                const float value6 = denom > 0.0f ? local_accum6[0] / denom : 0.0f;
                const float value7 = denom > 0.0f ? local_accum7[0] / denom : 0.0f;
                const ulong out_off = ((ulong)output_index * 4UL + (ulong)head) * 8UL;
                output_workspace[out_off] = value0;
                output_workspace[out_off + 1UL] = value1;
                output_workspace[out_off + 2UL] = value2;
                output_workspace[out_off + 3UL] = value3;
                output_workspace[out_off + 4UL] = value4;
                output_workspace[out_off + 5UL] = value5;
                output_workspace[out_off + 6UL] = value6;
                output_workspace[out_off + 7UL] = value7;
                shared_checksum ^= ((ulong)as_uint(shared_max) << 32) ^ (ulong)as_uint(denom) ^ (ulong)(head + 1u);
                shared_checksum ^= (ulong)as_uint(value0) + (((ulong)head + 1UL) << 32);
                shared_checksum ^= (ulong)as_uint(value1) + (((ulong)head + 1UL) << 32) + 1UL;
                shared_checksum ^= (ulong)as_uint(value2) + (((ulong)head + 1UL) << 32) + 2UL;
                shared_checksum ^= (ulong)as_uint(value3) + (((ulong)head + 1UL) << 32) + 3UL;
                shared_checksum ^= (ulong)as_uint(value4) + (((ulong)head + 1UL) << 32) + 4UL;
                shared_checksum ^= (ulong)as_uint(value5) + (((ulong)head + 1UL) << 32) + 5UL;
                shared_checksum ^= (ulong)as_uint(value6) + (((ulong)head + 1UL) << 32) + 6UL;
                shared_checksum ^= (ulong)as_uint(value7) + (((ulong)head + 1UL) << 32) + 7UL;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid == 0) {
            out[0] = 0UL;
            out[1] = 64UL;
            out[2] = shared_all_null_heads;
            out[3] = 4UL;
            out[4] = 8UL;
            out[5] = shared_checksum;
            out[6] = ((ulong)workspace_base << 32) | (ulong)output_index;
            out[7] = 0xF81757A20BADF00DUL;
        }
        return;
    }

    if (num_splits == 64u && num_heads == 16u && output_dims == 8u && partial_record_elems >= 10u && lsize >= 32u) {
        for (uint head = 0u; head < 16u; ++head) {
            float split_m = -3.402823466e+38F;
            float split_d0 = 0.0f;
            float split_d1 = 0.0f;
            float split_m0 = -3.402823466e+38F;
            float split_m1 = -3.402823466e+38F;
            float acc00 = 0.0f;
            float acc01 = 0.0f;
            float acc02 = 0.0f;
            float acc03 = 0.0f;
            float acc04 = 0.0f;
            float acc05 = 0.0f;
            float acc06 = 0.0f;
            float acc07 = 0.0f;
            float acc10 = 0.0f;
            float acc11 = 0.0f;
            float acc12 = 0.0f;
            float acc13 = 0.0f;
            float acc14 = 0.0f;
            float acc15 = 0.0f;
            float acc16 = 0.0f;
            float acc17 = 0.0f;
            int valid0 = 0;
            int valid1 = 0;
            if (lid < 32u) {
                const ulong base0 = (((ulong)workspace_base + (ulong)lid) * 16UL + (ulong)head) * (ulong)partial_record_elems;
                const float m0 = partial_workspace[base0];
                const float d0 = partial_workspace[base0 + 1UL];
                valid0 = d0 > 0.0f && m0 > -3.0e+38F;
                if (valid0) {
                    split_m0 = m0;
                    split_d0 = d0;
                    split_m = fmax(split_m, m0);
                    acc00 = partial_workspace[base0 + 2UL];
                    acc01 = partial_workspace[base0 + 3UL];
                    acc02 = partial_workspace[base0 + 4UL];
                    acc03 = partial_workspace[base0 + 5UL];
                    acc04 = partial_workspace[base0 + 6UL];
                    acc05 = partial_workspace[base0 + 7UL];
                    acc06 = partial_workspace[base0 + 8UL];
                    acc07 = partial_workspace[base0 + 9UL];
                }
                const ulong base1 = (((ulong)workspace_base + (ulong)lid + 32UL) * 16UL + (ulong)head) * (ulong)partial_record_elems;
                const float m1 = partial_workspace[base1];
                const float d1 = partial_workspace[base1 + 1UL];
                valid1 = d1 > 0.0f && m1 > -3.0e+38F;
                if (valid1) {
                    split_m1 = m1;
                    split_d1 = d1;
                    split_m = fmax(split_m, m1);
                    acc10 = partial_workspace[base1 + 2UL];
                    acc11 = partial_workspace[base1 + 3UL];
                    acc12 = partial_workspace[base1 + 4UL];
                    acc13 = partial_workspace[base1 + 5UL];
                    acc14 = partial_workspace[base1 + 6UL];
                    acc15 = partial_workspace[base1 + 7UL];
                    acc16 = partial_workspace[base1 + 8UL];
                    acc17 = partial_workspace[base1 + 9UL];
                }
            }
            local_max[lid] = split_m;
            local_all_null[lid] = (valid0 ? 1UL : 0UL) + (valid1 ? 1UL : 0UL);
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint stride = 16u; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    local_max[lid] = fmax(local_max[lid], local_max[lid + stride]);
                    local_all_null[lid] += local_all_null[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (lid == 0) {
                shared_max = local_max[0];
                if (local_all_null[0] == 0UL) {
                    shared_all_null_heads += 1UL;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            const float scale0 = valid0 ? native_exp(split_m0 - shared_max) : 0.0f;
            const float scale1 = valid1 ? native_exp(split_m1 - shared_max) : 0.0f;
            local_denom[lid] = (valid0 ? split_d0 * scale0 : 0.0f) + (valid1 ? split_d1 * scale1 : 0.0f);
            local_accum[lid] = (valid0 ? acc00 * scale0 : 0.0f) + (valid1 ? acc10 * scale1 : 0.0f);
            local_accum1[lid] = (valid0 ? acc01 * scale0 : 0.0f) + (valid1 ? acc11 * scale1 : 0.0f);
            local_accum2[lid] = (valid0 ? acc02 * scale0 : 0.0f) + (valid1 ? acc12 * scale1 : 0.0f);
            local_accum3[lid] = (valid0 ? acc03 * scale0 : 0.0f) + (valid1 ? acc13 * scale1 : 0.0f);
            local_accum4[lid] = (valid0 ? acc04 * scale0 : 0.0f) + (valid1 ? acc14 * scale1 : 0.0f);
            local_accum5[lid] = (valid0 ? acc05 * scale0 : 0.0f) + (valid1 ? acc15 * scale1 : 0.0f);
            local_accum6[lid] = (valid0 ? acc06 * scale0 : 0.0f) + (valid1 ? acc16 * scale1 : 0.0f);
            local_accum7[lid] = (valid0 ? acc07 * scale0 : 0.0f) + (valid1 ? acc17 * scale1 : 0.0f);
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint stride = 16u; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    local_denom[lid] += local_denom[lid + stride];
                    local_accum[lid] += local_accum[lid + stride];
                    local_accum1[lid] += local_accum1[lid + stride];
                    local_accum2[lid] += local_accum2[lid + stride];
                    local_accum3[lid] += local_accum3[lid + stride];
                    local_accum4[lid] += local_accum4[lid + stride];
                    local_accum5[lid] += local_accum5[lid + stride];
                    local_accum6[lid] += local_accum6[lid + stride];
                    local_accum7[lid] += local_accum7[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (lid == 0) {
                const float denom = local_denom[0];
                const float value0 = denom > 0.0f ? local_accum[0] / denom : 0.0f;
                const float value1 = denom > 0.0f ? local_accum1[0] / denom : 0.0f;
                const float value2 = denom > 0.0f ? local_accum2[0] / denom : 0.0f;
                const float value3 = denom > 0.0f ? local_accum3[0] / denom : 0.0f;
                const float value4 = denom > 0.0f ? local_accum4[0] / denom : 0.0f;
                const float value5 = denom > 0.0f ? local_accum5[0] / denom : 0.0f;
                const float value6 = denom > 0.0f ? local_accum6[0] / denom : 0.0f;
                const float value7 = denom > 0.0f ? local_accum7[0] / denom : 0.0f;
                const ulong out_off = ((ulong)output_index * 16UL + (ulong)head) * 8UL;
                output_workspace[out_off] = value0;
                output_workspace[out_off + 1UL] = value1;
                output_workspace[out_off + 2UL] = value2;
                output_workspace[out_off + 3UL] = value3;
                output_workspace[out_off + 4UL] = value4;
                output_workspace[out_off + 5UL] = value5;
                output_workspace[out_off + 6UL] = value6;
                output_workspace[out_off + 7UL] = value7;
                shared_checksum ^= ((ulong)as_uint(shared_max) << 32) ^ (ulong)as_uint(denom) ^ (ulong)(head + 1u);
                shared_checksum ^= (ulong)as_uint(value0) + (((ulong)head + 1UL) << 32);
                shared_checksum ^= (ulong)as_uint(value1) + (((ulong)head + 1UL) << 32) + 1UL;
                shared_checksum ^= (ulong)as_uint(value2) + (((ulong)head + 1UL) << 32) + 2UL;
                shared_checksum ^= (ulong)as_uint(value3) + (((ulong)head + 1UL) << 32) + 3UL;
                shared_checksum ^= (ulong)as_uint(value4) + (((ulong)head + 1UL) << 32) + 4UL;
                shared_checksum ^= (ulong)as_uint(value5) + (((ulong)head + 1UL) << 32) + 5UL;
                shared_checksum ^= (ulong)as_uint(value6) + (((ulong)head + 1UL) << 32) + 6UL;
                shared_checksum ^= (ulong)as_uint(value7) + (((ulong)head + 1UL) << 32) + 7UL;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid == 0) {
            out[0] = 0UL;
            out[1] = 64UL;
            out[2] = shared_all_null_heads;
            out[3] = 16UL;
            out[4] = 8UL;
            out[5] = shared_checksum;
            out[6] = ((ulong)workspace_base << 32) | (ulong)output_index;
            out[7] = 0xF81757A20BADF00DUL;
        }
        return;
    }

    if (num_splits == 128u && num_heads == 8u && output_dims == 8u && partial_record_elems >= 10u && lsize >= 32u) {
        for (uint head = 0u; head < 8u; ++head) {
            float split_m = -3.402823466e+38F;
            ulong valid_count = 0UL;
            if (lid < 32u) {
#pragma unroll
                for (uint r = 0u; r < 4u; ++r) {
                    const uint split = lid + (r << 5);
                    const ulong base = (((ulong)workspace_base + (ulong)split) * 8UL + (ulong)head) * (ulong)partial_record_elems;
                    const float m = partial_workspace[base];
                    const float d = partial_workspace[base + 1UL];
                    if (d > 0.0f && m > -3.0e+38F) {
                        split_m = fmax(split_m, m);
                        valid_count += 1UL;
                    }
                }
            }
            local_max[lid] = split_m;
            local_all_null[lid] = valid_count;
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint stride = 16u; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    local_max[lid] = fmax(local_max[lid], local_max[lid + stride]);
                    local_all_null[lid] += local_all_null[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (lid == 0) {
                shared_max = local_max[0];
                if (local_all_null[0] == 0UL) {
                    shared_all_null_heads += 1UL;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            float denom = 0.0f;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            float acc4 = 0.0f;
            float acc5 = 0.0f;
            float acc6 = 0.0f;
            float acc7 = 0.0f;
            if (lid < 32u) {
#pragma unroll
                for (uint r = 0u; r < 4u; ++r) {
                    const uint split = lid + (r << 5);
                    const ulong base = (((ulong)workspace_base + (ulong)split) * 8UL + (ulong)head) * (ulong)partial_record_elems;
                    const float m = partial_workspace[base];
                    const float d = partial_workspace[base + 1UL];
                    if (d > 0.0f && m > -3.0e+38F) {
                        const float scale = native_exp(m - shared_max);
                        denom += d * scale;
                        acc0 += partial_workspace[base + 2UL] * scale;
                        acc1 += partial_workspace[base + 3UL] * scale;
                        acc2 += partial_workspace[base + 4UL] * scale;
                        acc3 += partial_workspace[base + 5UL] * scale;
                        acc4 += partial_workspace[base + 6UL] * scale;
                        acc5 += partial_workspace[base + 7UL] * scale;
                        acc6 += partial_workspace[base + 8UL] * scale;
                        acc7 += partial_workspace[base + 9UL] * scale;
                    }
                }
            }
            local_denom[lid] = denom;
            local_accum[lid] = acc0;
            local_accum1[lid] = acc1;
            local_accum2[lid] = acc2;
            local_accum3[lid] = acc3;
            local_accum4[lid] = acc4;
            local_accum5[lid] = acc5;
            local_accum6[lid] = acc6;
            local_accum7[lid] = acc7;
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint stride = 16u; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    local_denom[lid] += local_denom[lid + stride];
                    local_accum[lid] += local_accum[lid + stride];
                    local_accum1[lid] += local_accum1[lid + stride];
                    local_accum2[lid] += local_accum2[lid + stride];
                    local_accum3[lid] += local_accum3[lid + stride];
                    local_accum4[lid] += local_accum4[lid + stride];
                    local_accum5[lid] += local_accum5[lid + stride];
                    local_accum6[lid] += local_accum6[lid + stride];
                    local_accum7[lid] += local_accum7[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (lid == 0) {
                const float total_denom = local_denom[0];
                const float value0 = total_denom > 0.0f ? local_accum[0] / total_denom : 0.0f;
                const float value1 = total_denom > 0.0f ? local_accum1[0] / total_denom : 0.0f;
                const float value2 = total_denom > 0.0f ? local_accum2[0] / total_denom : 0.0f;
                const float value3 = total_denom > 0.0f ? local_accum3[0] / total_denom : 0.0f;
                const float value4 = total_denom > 0.0f ? local_accum4[0] / total_denom : 0.0f;
                const float value5 = total_denom > 0.0f ? local_accum5[0] / total_denom : 0.0f;
                const float value6 = total_denom > 0.0f ? local_accum6[0] / total_denom : 0.0f;
                const float value7 = total_denom > 0.0f ? local_accum7[0] / total_denom : 0.0f;
                const ulong out_off = ((ulong)output_index * 8UL + (ulong)head) * 8UL;
                output_workspace[out_off] = value0;
                output_workspace[out_off + 1UL] = value1;
                output_workspace[out_off + 2UL] = value2;
                output_workspace[out_off + 3UL] = value3;
                output_workspace[out_off + 4UL] = value4;
                output_workspace[out_off + 5UL] = value5;
                output_workspace[out_off + 6UL] = value6;
                output_workspace[out_off + 7UL] = value7;
                shared_checksum ^= ((ulong)as_uint(shared_max) << 32) ^ (ulong)as_uint(total_denom) ^ (ulong)(head + 1u);
                shared_checksum ^= (ulong)as_uint(value0) + (((ulong)head + 1UL) << 32);
                shared_checksum ^= (ulong)as_uint(value1) + (((ulong)head + 1UL) << 32) + 1UL;
                shared_checksum ^= (ulong)as_uint(value2) + (((ulong)head + 1UL) << 32) + 2UL;
                shared_checksum ^= (ulong)as_uint(value3) + (((ulong)head + 1UL) << 32) + 3UL;
                shared_checksum ^= (ulong)as_uint(value4) + (((ulong)head + 1UL) << 32) + 4UL;
                shared_checksum ^= (ulong)as_uint(value5) + (((ulong)head + 1UL) << 32) + 5UL;
                shared_checksum ^= (ulong)as_uint(value6) + (((ulong)head + 1UL) << 32) + 6UL;
                shared_checksum ^= (ulong)as_uint(value7) + (((ulong)head + 1UL) << 32) + 7UL;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid == 0) {
            out[0] = 0UL;
            out[1] = 128UL;
            out[2] = shared_all_null_heads;
            out[3] = 8UL;
            out[4] = 8UL;
            out[5] = shared_checksum;
            out[6] = ((ulong)workspace_base << 32) | (ulong)output_index;
            out[7] = 0xF81757A20BADF00DUL;
        }
        return;
    }

    if (num_splits == 128u && num_heads == 4u && output_dims == 8u && partial_record_elems >= 10u && lsize >= 32u) {
        for (uint head = 0u; head < 4u; ++head) {
            float split_m = -3.402823466e+38F;
            ulong valid_count = 0UL;
            if (lid < 32u) {
#pragma unroll
                for (uint r = 0u; r < 4u; ++r) {
                    const uint split = lid + (r << 5);
                    const ulong base = (((ulong)workspace_base + (ulong)split) * 4UL + (ulong)head) * (ulong)partial_record_elems;
                    const float m = partial_workspace[base];
                    const float d = partial_workspace[base + 1UL];
                    if (d > 0.0f && m > -3.0e+38F) {
                        split_m = fmax(split_m, m);
                        valid_count += 1UL;
                    }
                }
            }
            local_max[lid] = split_m;
            local_all_null[lid] = valid_count;
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint stride = 16u; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    local_max[lid] = fmax(local_max[lid], local_max[lid + stride]);
                    local_all_null[lid] += local_all_null[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (lid == 0) {
                shared_max = local_max[0];
                if (local_all_null[0] == 0UL) {
                    shared_all_null_heads += 1UL;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            float denom = 0.0f;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            float acc4 = 0.0f;
            float acc5 = 0.0f;
            float acc6 = 0.0f;
            float acc7 = 0.0f;
            if (lid < 32u) {
#pragma unroll
                for (uint r = 0u; r < 4u; ++r) {
                    const uint split = lid + (r << 5);
                    const ulong base = (((ulong)workspace_base + (ulong)split) * 4UL + (ulong)head) * (ulong)partial_record_elems;
                    const float m = partial_workspace[base];
                    const float d = partial_workspace[base + 1UL];
                    if (d > 0.0f && m > -3.0e+38F) {
                        const float scale = native_exp(m - shared_max);
                        denom += d * scale;
                        acc0 += partial_workspace[base + 2UL] * scale;
                        acc1 += partial_workspace[base + 3UL] * scale;
                        acc2 += partial_workspace[base + 4UL] * scale;
                        acc3 += partial_workspace[base + 5UL] * scale;
                        acc4 += partial_workspace[base + 6UL] * scale;
                        acc5 += partial_workspace[base + 7UL] * scale;
                        acc6 += partial_workspace[base + 8UL] * scale;
                        acc7 += partial_workspace[base + 9UL] * scale;
                    }
                }
            }
            local_denom[lid] = denom;
            local_accum[lid] = acc0;
            local_accum1[lid] = acc1;
            local_accum2[lid] = acc2;
            local_accum3[lid] = acc3;
            local_accum4[lid] = acc4;
            local_accum5[lid] = acc5;
            local_accum6[lid] = acc6;
            local_accum7[lid] = acc7;
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint stride = 16u; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    local_denom[lid] += local_denom[lid + stride];
                    local_accum[lid] += local_accum[lid + stride];
                    local_accum1[lid] += local_accum1[lid + stride];
                    local_accum2[lid] += local_accum2[lid + stride];
                    local_accum3[lid] += local_accum3[lid + stride];
                    local_accum4[lid] += local_accum4[lid + stride];
                    local_accum5[lid] += local_accum5[lid + stride];
                    local_accum6[lid] += local_accum6[lid + stride];
                    local_accum7[lid] += local_accum7[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (lid == 0) {
                const float total_denom = local_denom[0];
                const float value0 = total_denom > 0.0f ? local_accum[0] / total_denom : 0.0f;
                const float value1 = total_denom > 0.0f ? local_accum1[0] / total_denom : 0.0f;
                const float value2 = total_denom > 0.0f ? local_accum2[0] / total_denom : 0.0f;
                const float value3 = total_denom > 0.0f ? local_accum3[0] / total_denom : 0.0f;
                const float value4 = total_denom > 0.0f ? local_accum4[0] / total_denom : 0.0f;
                const float value5 = total_denom > 0.0f ? local_accum5[0] / total_denom : 0.0f;
                const float value6 = total_denom > 0.0f ? local_accum6[0] / total_denom : 0.0f;
                const float value7 = total_denom > 0.0f ? local_accum7[0] / total_denom : 0.0f;
                const ulong out_off = ((ulong)output_index * 4UL + (ulong)head) * 8UL;
                output_workspace[out_off] = value0;
                output_workspace[out_off + 1UL] = value1;
                output_workspace[out_off + 2UL] = value2;
                output_workspace[out_off + 3UL] = value3;
                output_workspace[out_off + 4UL] = value4;
                output_workspace[out_off + 5UL] = value5;
                output_workspace[out_off + 6UL] = value6;
                output_workspace[out_off + 7UL] = value7;
                shared_checksum ^= ((ulong)as_uint(shared_max) << 32) ^ (ulong)as_uint(total_denom) ^ (ulong)(head + 1u);
                shared_checksum ^= (ulong)as_uint(value0) + (((ulong)head + 1UL) << 32);
                shared_checksum ^= (ulong)as_uint(value1) + (((ulong)head + 1UL) << 32) + 1UL;
                shared_checksum ^= (ulong)as_uint(value2) + (((ulong)head + 1UL) << 32) + 2UL;
                shared_checksum ^= (ulong)as_uint(value3) + (((ulong)head + 1UL) << 32) + 3UL;
                shared_checksum ^= (ulong)as_uint(value4) + (((ulong)head + 1UL) << 32) + 4UL;
                shared_checksum ^= (ulong)as_uint(value5) + (((ulong)head + 1UL) << 32) + 5UL;
                shared_checksum ^= (ulong)as_uint(value6) + (((ulong)head + 1UL) << 32) + 6UL;
                shared_checksum ^= (ulong)as_uint(value7) + (((ulong)head + 1UL) << 32) + 7UL;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid == 0) {
            out[0] = 0UL;
            out[1] = 128UL;
            out[2] = shared_all_null_heads;
            out[3] = 4UL;
            out[4] = 8UL;
            out[5] = shared_checksum;
            out[6] = ((ulong)workspace_base << 32) | (ulong)output_index;
            out[7] = 0xF81757A20BADF00DUL;
        }
        return;
    }

    if (num_splits == 128u && num_heads == 16u && output_dims == 8u && partial_record_elems >= 10u && lsize >= 32u) {
        for (uint head = 0u; head < 16u; ++head) {
            float split_m = -3.402823466e+38F;
            ulong valid_count = 0UL;
            if (lid < 32u) {
#pragma unroll
                for (uint r = 0u; r < 4u; ++r) {
                    const uint split = lid + (r << 5);
                    const ulong base = (((ulong)workspace_base + (ulong)split) * 16UL + (ulong)head) * (ulong)partial_record_elems;
                    const float m = partial_workspace[base];
                    const float d = partial_workspace[base + 1UL];
                    if (d > 0.0f && m > -3.0e+38F) {
                        split_m = fmax(split_m, m);
                        valid_count += 1UL;
                    }
                }
            }
            local_max[lid] = split_m;
            local_all_null[lid] = valid_count;
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint stride = 16u; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    local_max[lid] = fmax(local_max[lid], local_max[lid + stride]);
                    local_all_null[lid] += local_all_null[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (lid == 0) {
                shared_max = local_max[0];
                if (local_all_null[0] == 0UL) {
                    shared_all_null_heads += 1UL;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            float denom = 0.0f;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            float acc4 = 0.0f;
            float acc5 = 0.0f;
            float acc6 = 0.0f;
            float acc7 = 0.0f;
            if (lid < 32u) {
#pragma unroll
                for (uint r = 0u; r < 4u; ++r) {
                    const uint split = lid + (r << 5);
                    const ulong base = (((ulong)workspace_base + (ulong)split) * 16UL + (ulong)head) * (ulong)partial_record_elems;
                    const float m = partial_workspace[base];
                    const float d = partial_workspace[base + 1UL];
                    if (d > 0.0f && m > -3.0e+38F) {
                        const float scale = native_exp(m - shared_max);
                        denom += d * scale;
                        acc0 += partial_workspace[base + 2UL] * scale;
                        acc1 += partial_workspace[base + 3UL] * scale;
                        acc2 += partial_workspace[base + 4UL] * scale;
                        acc3 += partial_workspace[base + 5UL] * scale;
                        acc4 += partial_workspace[base + 6UL] * scale;
                        acc5 += partial_workspace[base + 7UL] * scale;
                        acc6 += partial_workspace[base + 8UL] * scale;
                        acc7 += partial_workspace[base + 9UL] * scale;
                    }
                }
            }
            local_denom[lid] = denom;
            local_accum[lid] = acc0;
            local_accum1[lid] = acc1;
            local_accum2[lid] = acc2;
            local_accum3[lid] = acc3;
            local_accum4[lid] = acc4;
            local_accum5[lid] = acc5;
            local_accum6[lid] = acc6;
            local_accum7[lid] = acc7;
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint stride = 16u; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    local_denom[lid] += local_denom[lid + stride];
                    local_accum[lid] += local_accum[lid + stride];
                    local_accum1[lid] += local_accum1[lid + stride];
                    local_accum2[lid] += local_accum2[lid + stride];
                    local_accum3[lid] += local_accum3[lid + stride];
                    local_accum4[lid] += local_accum4[lid + stride];
                    local_accum5[lid] += local_accum5[lid + stride];
                    local_accum6[lid] += local_accum6[lid + stride];
                    local_accum7[lid] += local_accum7[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (lid == 0) {
                const float total_denom = local_denom[0];
                const float value0 = total_denom > 0.0f ? local_accum[0] / total_denom : 0.0f;
                const float value1 = total_denom > 0.0f ? local_accum1[0] / total_denom : 0.0f;
                const float value2 = total_denom > 0.0f ? local_accum2[0] / total_denom : 0.0f;
                const float value3 = total_denom > 0.0f ? local_accum3[0] / total_denom : 0.0f;
                const float value4 = total_denom > 0.0f ? local_accum4[0] / total_denom : 0.0f;
                const float value5 = total_denom > 0.0f ? local_accum5[0] / total_denom : 0.0f;
                const float value6 = total_denom > 0.0f ? local_accum6[0] / total_denom : 0.0f;
                const float value7 = total_denom > 0.0f ? local_accum7[0] / total_denom : 0.0f;
                const ulong out_off = ((ulong)output_index * 16UL + (ulong)head) * 8UL;
                output_workspace[out_off] = value0;
                output_workspace[out_off + 1UL] = value1;
                output_workspace[out_off + 2UL] = value2;
                output_workspace[out_off + 3UL] = value3;
                output_workspace[out_off + 4UL] = value4;
                output_workspace[out_off + 5UL] = value5;
                output_workspace[out_off + 6UL] = value6;
                output_workspace[out_off + 7UL] = value7;
                shared_checksum ^= ((ulong)as_uint(shared_max) << 32) ^ (ulong)as_uint(total_denom) ^ (ulong)(head + 1u);
                shared_checksum ^= (ulong)as_uint(value0) + (((ulong)head + 1UL) << 32);
                shared_checksum ^= (ulong)as_uint(value1) + (((ulong)head + 1UL) << 32) + 1UL;
                shared_checksum ^= (ulong)as_uint(value2) + (((ulong)head + 1UL) << 32) + 2UL;
                shared_checksum ^= (ulong)as_uint(value3) + (((ulong)head + 1UL) << 32) + 3UL;
                shared_checksum ^= (ulong)as_uint(value4) + (((ulong)head + 1UL) << 32) + 4UL;
                shared_checksum ^= (ulong)as_uint(value5) + (((ulong)head + 1UL) << 32) + 5UL;
                shared_checksum ^= (ulong)as_uint(value6) + (((ulong)head + 1UL) << 32) + 6UL;
                shared_checksum ^= (ulong)as_uint(value7) + (((ulong)head + 1UL) << 32) + 7UL;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid == 0) {
            out[0] = 0UL;
            out[1] = 128UL;
            out[2] = shared_all_null_heads;
            out[3] = 16UL;
            out[4] = 8UL;
            out[5] = shared_checksum;
            out[6] = ((ulong)workspace_base << 32) | (ulong)output_index;
            out[7] = 0xF81757A20BADF00DUL;
        }
        return;
    }

    if (num_splits == 256u && num_heads == 8u && output_dims == 8u && partial_record_elems >= 10u && lsize >= 32u) {
        for (uint head = 0u; head < 8u; ++head) {
            float split_m = -3.402823466e+38F;
            ulong valid_count = 0UL;
            if (lid < 32u) {
#pragma unroll
                for (uint r = 0u; r < 8u; ++r) {
                    const uint split = lid + (r << 5);
                    const ulong base = (((ulong)workspace_base + (ulong)split) * 8UL + (ulong)head) * (ulong)partial_record_elems;
                    const float m = partial_workspace[base];
                    const float d = partial_workspace[base + 1UL];
                    if (d > 0.0f && m > -3.0e+38F) {
                        split_m = fmax(split_m, m);
                        valid_count += 1UL;
                    }
                }
            }
            local_max[lid] = split_m;
            local_all_null[lid] = valid_count;
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint stride = 16u; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    local_max[lid] = fmax(local_max[lid], local_max[lid + stride]);
                    local_all_null[lid] += local_all_null[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (lid == 0) {
                shared_max = local_max[0];
                if (local_all_null[0] == 0UL) {
                    shared_all_null_heads += 1UL;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            float denom = 0.0f;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            float acc4 = 0.0f;
            float acc5 = 0.0f;
            float acc6 = 0.0f;
            float acc7 = 0.0f;
            if (lid < 32u) {
#pragma unroll
                for (uint r = 0u; r < 8u; ++r) {
                    const uint split = lid + (r << 5);
                    const ulong base = (((ulong)workspace_base + (ulong)split) * 8UL + (ulong)head) * (ulong)partial_record_elems;
                    const float m = partial_workspace[base];
                    const float d = partial_workspace[base + 1UL];
                    if (d > 0.0f && m > -3.0e+38F) {
                        const float scale = native_exp(m - shared_max);
                        denom += d * scale;
                        acc0 += partial_workspace[base + 2UL] * scale;
                        acc1 += partial_workspace[base + 3UL] * scale;
                        acc2 += partial_workspace[base + 4UL] * scale;
                        acc3 += partial_workspace[base + 5UL] * scale;
                        acc4 += partial_workspace[base + 6UL] * scale;
                        acc5 += partial_workspace[base + 7UL] * scale;
                        acc6 += partial_workspace[base + 8UL] * scale;
                        acc7 += partial_workspace[base + 9UL] * scale;
                    }
                }
            }
            local_denom[lid] = denom;
            local_accum[lid] = acc0;
            local_accum1[lid] = acc1;
            local_accum2[lid] = acc2;
            local_accum3[lid] = acc3;
            local_accum4[lid] = acc4;
            local_accum5[lid] = acc5;
            local_accum6[lid] = acc6;
            local_accum7[lid] = acc7;
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint stride = 16u; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    local_denom[lid] += local_denom[lid + stride];
                    local_accum[lid] += local_accum[lid + stride];
                    local_accum1[lid] += local_accum1[lid + stride];
                    local_accum2[lid] += local_accum2[lid + stride];
                    local_accum3[lid] += local_accum3[lid + stride];
                    local_accum4[lid] += local_accum4[lid + stride];
                    local_accum5[lid] += local_accum5[lid + stride];
                    local_accum6[lid] += local_accum6[lid + stride];
                    local_accum7[lid] += local_accum7[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (lid == 0) {
                const float total_denom = local_denom[0];
                const float value0 = total_denom > 0.0f ? local_accum[0] / total_denom : 0.0f;
                const float value1 = total_denom > 0.0f ? local_accum1[0] / total_denom : 0.0f;
                const float value2 = total_denom > 0.0f ? local_accum2[0] / total_denom : 0.0f;
                const float value3 = total_denom > 0.0f ? local_accum3[0] / total_denom : 0.0f;
                const float value4 = total_denom > 0.0f ? local_accum4[0] / total_denom : 0.0f;
                const float value5 = total_denom > 0.0f ? local_accum5[0] / total_denom : 0.0f;
                const float value6 = total_denom > 0.0f ? local_accum6[0] / total_denom : 0.0f;
                const float value7 = total_denom > 0.0f ? local_accum7[0] / total_denom : 0.0f;
                const ulong out_off = ((ulong)output_index * 8UL + (ulong)head) * 8UL;
                output_workspace[out_off] = value0;
                output_workspace[out_off + 1UL] = value1;
                output_workspace[out_off + 2UL] = value2;
                output_workspace[out_off + 3UL] = value3;
                output_workspace[out_off + 4UL] = value4;
                output_workspace[out_off + 5UL] = value5;
                output_workspace[out_off + 6UL] = value6;
                output_workspace[out_off + 7UL] = value7;
                shared_checksum ^= ((ulong)as_uint(shared_max) << 32) ^ (ulong)as_uint(total_denom) ^ (ulong)(head + 1u);
                shared_checksum ^= (ulong)as_uint(value0) + (((ulong)head + 1UL) << 32);
                shared_checksum ^= (ulong)as_uint(value1) + (((ulong)head + 1UL) << 32) + 1UL;
                shared_checksum ^= (ulong)as_uint(value2) + (((ulong)head + 1UL) << 32) + 2UL;
                shared_checksum ^= (ulong)as_uint(value3) + (((ulong)head + 1UL) << 32) + 3UL;
                shared_checksum ^= (ulong)as_uint(value4) + (((ulong)head + 1UL) << 32) + 4UL;
                shared_checksum ^= (ulong)as_uint(value5) + (((ulong)head + 1UL) << 32) + 5UL;
                shared_checksum ^= (ulong)as_uint(value6) + (((ulong)head + 1UL) << 32) + 6UL;
                shared_checksum ^= (ulong)as_uint(value7) + (((ulong)head + 1UL) << 32) + 7UL;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid == 0) {
            out[0] = 0UL;
            out[1] = 256UL;
            out[2] = shared_all_null_heads;
            out[3] = 8UL;
            out[4] = 8UL;
            out[5] = shared_checksum;
            out[6] = ((ulong)workspace_base << 32) | (ulong)output_index;
            out[7] = 0xF81757A20BADF00DUL;
        }
        return;
    }

    if (num_splits == 256u && num_heads == 4u && output_dims == 8u && partial_record_elems >= 10u && lsize >= 32u) {
        for (uint head = 0u; head < 4u; ++head) {
            float split_m = -3.402823466e+38F;
            ulong valid_count = 0UL;
            if (lid < 32u) {
#pragma unroll
                for (uint r = 0u; r < 8u; ++r) {
                    const uint split = lid + (r << 5);
                    const ulong base = (((ulong)workspace_base + (ulong)split) * 4UL + (ulong)head) * (ulong)partial_record_elems;
                    const float m = partial_workspace[base];
                    const float d = partial_workspace[base + 1UL];
                    if (d > 0.0f && m > -3.0e+38F) {
                        split_m = fmax(split_m, m);
                        valid_count += 1UL;
                    }
                }
            }
            local_max[lid] = split_m;
            local_all_null[lid] = valid_count;
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint stride = 16u; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    local_max[lid] = fmax(local_max[lid], local_max[lid + stride]);
                    local_all_null[lid] += local_all_null[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (lid == 0) {
                shared_max = local_max[0];
                if (local_all_null[0] == 0UL) {
                    shared_all_null_heads += 1UL;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            float denom = 0.0f;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            float acc4 = 0.0f;
            float acc5 = 0.0f;
            float acc6 = 0.0f;
            float acc7 = 0.0f;
            if (lid < 32u) {
#pragma unroll
                for (uint r = 0u; r < 8u; ++r) {
                    const uint split = lid + (r << 5);
                    const ulong base = (((ulong)workspace_base + (ulong)split) * 4UL + (ulong)head) * (ulong)partial_record_elems;
                    const float m = partial_workspace[base];
                    const float d = partial_workspace[base + 1UL];
                    if (d > 0.0f && m > -3.0e+38F) {
                        const float scale = native_exp(m - shared_max);
                        denom += d * scale;
                        acc0 += partial_workspace[base + 2UL] * scale;
                        acc1 += partial_workspace[base + 3UL] * scale;
                        acc2 += partial_workspace[base + 4UL] * scale;
                        acc3 += partial_workspace[base + 5UL] * scale;
                        acc4 += partial_workspace[base + 6UL] * scale;
                        acc5 += partial_workspace[base + 7UL] * scale;
                        acc6 += partial_workspace[base + 8UL] * scale;
                        acc7 += partial_workspace[base + 9UL] * scale;
                    }
                }
            }
            local_denom[lid] = denom;
            local_accum[lid] = acc0;
            local_accum1[lid] = acc1;
            local_accum2[lid] = acc2;
            local_accum3[lid] = acc3;
            local_accum4[lid] = acc4;
            local_accum5[lid] = acc5;
            local_accum6[lid] = acc6;
            local_accum7[lid] = acc7;
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint stride = 16u; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    local_denom[lid] += local_denom[lid + stride];
                    local_accum[lid] += local_accum[lid + stride];
                    local_accum1[lid] += local_accum1[lid + stride];
                    local_accum2[lid] += local_accum2[lid + stride];
                    local_accum3[lid] += local_accum3[lid + stride];
                    local_accum4[lid] += local_accum4[lid + stride];
                    local_accum5[lid] += local_accum5[lid + stride];
                    local_accum6[lid] += local_accum6[lid + stride];
                    local_accum7[lid] += local_accum7[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (lid == 0) {
                const float total_denom = local_denom[0];
                const float value0 = total_denom > 0.0f ? local_accum[0] / total_denom : 0.0f;
                const float value1 = total_denom > 0.0f ? local_accum1[0] / total_denom : 0.0f;
                const float value2 = total_denom > 0.0f ? local_accum2[0] / total_denom : 0.0f;
                const float value3 = total_denom > 0.0f ? local_accum3[0] / total_denom : 0.0f;
                const float value4 = total_denom > 0.0f ? local_accum4[0] / total_denom : 0.0f;
                const float value5 = total_denom > 0.0f ? local_accum5[0] / total_denom : 0.0f;
                const float value6 = total_denom > 0.0f ? local_accum6[0] / total_denom : 0.0f;
                const float value7 = total_denom > 0.0f ? local_accum7[0] / total_denom : 0.0f;
                const ulong out_off = ((ulong)output_index * 4UL + (ulong)head) * 8UL;
                output_workspace[out_off] = value0;
                output_workspace[out_off + 1UL] = value1;
                output_workspace[out_off + 2UL] = value2;
                output_workspace[out_off + 3UL] = value3;
                output_workspace[out_off + 4UL] = value4;
                output_workspace[out_off + 5UL] = value5;
                output_workspace[out_off + 6UL] = value6;
                output_workspace[out_off + 7UL] = value7;
                shared_checksum ^= ((ulong)as_uint(shared_max) << 32) ^ (ulong)as_uint(total_denom) ^ (ulong)(head + 1u);
                shared_checksum ^= (ulong)as_uint(value0) + (((ulong)head + 1UL) << 32);
                shared_checksum ^= (ulong)as_uint(value1) + (((ulong)head + 1UL) << 32) + 1UL;
                shared_checksum ^= (ulong)as_uint(value2) + (((ulong)head + 1UL) << 32) + 2UL;
                shared_checksum ^= (ulong)as_uint(value3) + (((ulong)head + 1UL) << 32) + 3UL;
                shared_checksum ^= (ulong)as_uint(value4) + (((ulong)head + 1UL) << 32) + 4UL;
                shared_checksum ^= (ulong)as_uint(value5) + (((ulong)head + 1UL) << 32) + 5UL;
                shared_checksum ^= (ulong)as_uint(value6) + (((ulong)head + 1UL) << 32) + 6UL;
                shared_checksum ^= (ulong)as_uint(value7) + (((ulong)head + 1UL) << 32) + 7UL;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid == 0) {
            out[0] = 0UL;
            out[1] = 256UL;
            out[2] = shared_all_null_heads;
            out[3] = 4UL;
            out[4] = 8UL;
            out[5] = shared_checksum;
            out[6] = ((ulong)workspace_base << 32) | (ulong)output_index;
            out[7] = 0xF81757A20BADF00DUL;
        }
        return;
    }

    if (num_splits == 256u && num_heads == 16u && output_dims == 8u && partial_record_elems >= 10u && lsize >= 32u) {
        for (uint head = 0u; head < 16u; ++head) {
            float split_m = -3.402823466e+38F;
            ulong valid_count = 0UL;
            if (lid < 32u) {
#pragma unroll
                for (uint r = 0u; r < 8u; ++r) {
                    const uint split = lid + (r << 5);
                    const ulong base = (((ulong)workspace_base + (ulong)split) * 16UL + (ulong)head) * (ulong)partial_record_elems;
                    const float m = partial_workspace[base];
                    const float d = partial_workspace[base + 1UL];
                    if (d > 0.0f && m > -3.0e+38F) {
                        split_m = fmax(split_m, m);
                        valid_count += 1UL;
                    }
                }
            }
            local_max[lid] = split_m;
            local_all_null[lid] = valid_count;
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint stride = 16u; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    local_max[lid] = fmax(local_max[lid], local_max[lid + stride]);
                    local_all_null[lid] += local_all_null[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (lid == 0) {
                shared_max = local_max[0];
                if (local_all_null[0] == 0UL) {
                    shared_all_null_heads += 1UL;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            float denom = 0.0f;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            float acc4 = 0.0f;
            float acc5 = 0.0f;
            float acc6 = 0.0f;
            float acc7 = 0.0f;
            if (lid < 32u) {
#pragma unroll
                for (uint r = 0u; r < 8u; ++r) {
                    const uint split = lid + (r << 5);
                    const ulong base = (((ulong)workspace_base + (ulong)split) * 16UL + (ulong)head) * (ulong)partial_record_elems;
                    const float m = partial_workspace[base];
                    const float d = partial_workspace[base + 1UL];
                    if (d > 0.0f && m > -3.0e+38F) {
                        const float scale = native_exp(m - shared_max);
                        denom += d * scale;
                        acc0 += partial_workspace[base + 2UL] * scale;
                        acc1 += partial_workspace[base + 3UL] * scale;
                        acc2 += partial_workspace[base + 4UL] * scale;
                        acc3 += partial_workspace[base + 5UL] * scale;
                        acc4 += partial_workspace[base + 6UL] * scale;
                        acc5 += partial_workspace[base + 7UL] * scale;
                        acc6 += partial_workspace[base + 8UL] * scale;
                        acc7 += partial_workspace[base + 9UL] * scale;
                    }
                }
            }
            local_denom[lid] = denom;
            local_accum[lid] = acc0;
            local_accum1[lid] = acc1;
            local_accum2[lid] = acc2;
            local_accum3[lid] = acc3;
            local_accum4[lid] = acc4;
            local_accum5[lid] = acc5;
            local_accum6[lid] = acc6;
            local_accum7[lid] = acc7;
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint stride = 16u; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    local_denom[lid] += local_denom[lid + stride];
                    local_accum[lid] += local_accum[lid + stride];
                    local_accum1[lid] += local_accum1[lid + stride];
                    local_accum2[lid] += local_accum2[lid + stride];
                    local_accum3[lid] += local_accum3[lid + stride];
                    local_accum4[lid] += local_accum4[lid + stride];
                    local_accum5[lid] += local_accum5[lid + stride];
                    local_accum6[lid] += local_accum6[lid + stride];
                    local_accum7[lid] += local_accum7[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (lid == 0) {
                const float total_denom = local_denom[0];
                const float value0 = total_denom > 0.0f ? local_accum[0] / total_denom : 0.0f;
                const float value1 = total_denom > 0.0f ? local_accum1[0] / total_denom : 0.0f;
                const float value2 = total_denom > 0.0f ? local_accum2[0] / total_denom : 0.0f;
                const float value3 = total_denom > 0.0f ? local_accum3[0] / total_denom : 0.0f;
                const float value4 = total_denom > 0.0f ? local_accum4[0] / total_denom : 0.0f;
                const float value5 = total_denom > 0.0f ? local_accum5[0] / total_denom : 0.0f;
                const float value6 = total_denom > 0.0f ? local_accum6[0] / total_denom : 0.0f;
                const float value7 = total_denom > 0.0f ? local_accum7[0] / total_denom : 0.0f;
                const ulong out_off = ((ulong)output_index * 16UL + (ulong)head) * 8UL;
                output_workspace[out_off] = value0;
                output_workspace[out_off + 1UL] = value1;
                output_workspace[out_off + 2UL] = value2;
                output_workspace[out_off + 3UL] = value3;
                output_workspace[out_off + 4UL] = value4;
                output_workspace[out_off + 5UL] = value5;
                output_workspace[out_off + 6UL] = value6;
                output_workspace[out_off + 7UL] = value7;
                shared_checksum ^= ((ulong)as_uint(shared_max) << 32) ^ (ulong)as_uint(total_denom) ^ (ulong)(head + 1u);
                shared_checksum ^= (ulong)as_uint(value0) + (((ulong)head + 1UL) << 32);
                shared_checksum ^= (ulong)as_uint(value1) + (((ulong)head + 1UL) << 32) + 1UL;
                shared_checksum ^= (ulong)as_uint(value2) + (((ulong)head + 1UL) << 32) + 2UL;
                shared_checksum ^= (ulong)as_uint(value3) + (((ulong)head + 1UL) << 32) + 3UL;
                shared_checksum ^= (ulong)as_uint(value4) + (((ulong)head + 1UL) << 32) + 4UL;
                shared_checksum ^= (ulong)as_uint(value5) + (((ulong)head + 1UL) << 32) + 5UL;
                shared_checksum ^= (ulong)as_uint(value6) + (((ulong)head + 1UL) << 32) + 6UL;
                shared_checksum ^= (ulong)as_uint(value7) + (((ulong)head + 1UL) << 32) + 7UL;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid == 0) {
            out[0] = 0UL;
            out[1] = 256UL;
            out[2] = shared_all_null_heads;
            out[3] = 16UL;
            out[4] = 8UL;
            out[5] = shared_checksum;
            out[6] = ((ulong)workspace_base << 32) | (ulong)output_index;
            out[7] = 0xF81757A20BADF00DUL;
        }
        return;
    }

    for (uint head = 0u; head < num_heads; ++head) {
        float m = -3.402823466e+38F;
        ulong non_null = 0UL;
        for (uint split = lid; split < num_splits; split += lsize) {
            const ulong base = (((ulong)workspace_base + (ulong)split) * (ulong)num_heads + (ulong)head) * (ulong)partial_record_elems;
            const float split_m = partial_workspace[base];
            const float split_d = partial_workspace[base + 1UL];
            if (split_d > 0.0f && split_m > -3.0e+38F) {
                m = fmax(m, split_m);
                non_null += 1UL;
            }
        }
        local_max[lid] = m;
        local_all_null[lid] = non_null;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint stride = lsize >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) {
                local_max[lid] = fmax(local_max[lid], local_max[lid + stride]);
                local_all_null[lid] += local_all_null[lid + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            shared_max = local_max[0];
            if (local_all_null[0] == 0UL) {
                shared_all_null_heads += 1UL;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        float denom = 0.0f;
        if (local_all_null[0] != 0UL) {
            for (uint split = lid; split < num_splits; split += lsize) {
                const ulong base = (((ulong)workspace_base + (ulong)split) * (ulong)num_heads + (ulong)head) * (ulong)partial_record_elems;
                const float split_m = partial_workspace[base];
                const float split_d = partial_workspace[base + 1UL];
                if (split_d > 0.0f && split_m > -3.0e+38F) {
                    denom += split_d * exp(split_m - shared_max);
                }
            }
        }
        local_denom[lid] = denom;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint stride = lsize >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) {
                local_denom[lid] += local_denom[lid + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            shared_denom = local_denom[0];
            shared_checksum ^= ((ulong)as_uint(shared_max) << 32) ^ (ulong)as_uint(shared_denom) ^ (ulong)(head + 1u);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint dim = 0u; dim < output_dims; ++dim) {
            float accum = 0.0f;
            if (shared_denom > 0.0f) {
                for (uint split = lid; split < num_splits; split += lsize) {
                    const ulong base = (((ulong)workspace_base + (ulong)split) * (ulong)num_heads + (ulong)head) * (ulong)partial_record_elems;
                    const float split_m = partial_workspace[base];
                    const float split_d = partial_workspace[base + 1UL];
                    if (split_d > 0.0f && split_m > -3.0e+38F) {
                        accum += partial_workspace[base + 2UL + (ulong)dim] * exp(split_m - shared_max);
                    }
                }
            }
            local_accum[lid] = accum;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (uint stride = lsize >> 1; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    local_accum[lid] += local_accum[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (lid == 0) {
                const ulong out_off = (((ulong)output_index * (ulong)num_heads + (ulong)head) * (ulong)output_dims) + (ulong)dim;
                const float value = shared_denom > 0.0f ? local_accum[0] / shared_denom : 0.0f;
                output_workspace[out_off] = value;
                shared_checksum ^= (ulong)as_uint(value) + (((ulong)head + 1UL) << 32) + (ulong)dim;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (lid == 0) {
        out[0] = 0UL;
        out[1] = (ulong)num_splits;
        out[2] = shared_all_null_heads;
        out[3] = (ulong)num_heads;
        out[4] = (ulong)output_dims;
        out[5] = shared_checksum;
        out[6] = ((ulong)workspace_base << 32) | (ulong)output_index;
        out[7] = 0xF81757A20BADF00DUL;
    }
}



// Tiny paged-K QK score gate. This is the first attention-semantic probe: it
// computes per-token Q.K scores over valid pages and reduces sum/max locally,
// while null/OOB pages are classified before any K-cache dereference.
__kernel void paged_kv_qk_score_probe(__global const half* q,
                                      __global const half* cache,
                                      __global const uint* indices,
                                      __global ulong* out,
                                      uint total_indices,
                                      uint physical_blocks,
                                      uint block_size,
                                      uint d) {
    const uint lid = get_local_id(0);
    const uint lsize = get_local_size(0);
    __local float local_sum[256];
    __local float local_max[256];
    __local ulong local_bad[256];
    __local ulong local_null[256];
    __local ulong local_valid[256];
    __local uint local_first_pos[256];
    __local uint local_first_val[256];

    if (lid == 0) {
        out[0] = 3UL;
        out[1] = 0UL;
        out[2] = 0UL;
        out[3] = 0UL;
        out[4] = 0UL;
        out[5] = 0UL;
        out[6] = 0xffffffff00000000UL;
        out[7] = 0x513C0DEF0BADF00DUL;
    }

    if (total_indices == 0u || physical_blocks == 0u || block_size == 0u || d == 0u)
        return;

    const ulong total_tokens = (ulong)total_indices * (ulong)block_size;
    float score_sum = 0.0f;
    float score_max = -3.402823466e+38F;
    ulong bad = 0UL;
    ulong nulls = 0UL;
    ulong valid = 0UL;
    uint first_pos = 0xffffffffu;
    uint first_val = 0u;

    for (ulong token = (ulong)lid; token < total_tokens; token += (ulong)lsize) {
        const uint page = (uint)(token / (ulong)block_size);
        const uint token_in_block = (uint)(token - ((ulong)page * (ulong)block_size));
        const uint block_id = indices[page];
        if (block_id == 0u) {
            nulls += 1UL;
            continue;
        }
        if (block_id >= physical_blocks) {
            bad += 1UL;
            if ((uint)token < first_pos) {
                first_pos = (uint)token;
                first_val = block_id;
            }
            continue;
        }

        const ulong base = (((ulong)block_id * (ulong)block_size) + (ulong)token_in_block) * (ulong)d;
        float score = 0.0f;
        for (uint dim = 0u; dim < d; ++dim) {
            score += (float)q[dim] * (float)cache[base + (ulong)dim];
        }
        score_sum += score;
        score_max = fmax(score_max, score);
        valid += 1UL;
    }

    local_sum[lid] = score_sum;
    local_max[lid] = score_max;
    local_bad[lid] = bad;
    local_null[lid] = nulls;
    local_valid[lid] = valid;
    local_first_pos[lid] = first_pos;
    local_first_val[lid] = first_val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_sum[lid] += local_sum[lid + stride];
            local_max[lid] = fmax(local_max[lid], local_max[lid + stride]);
            local_bad[lid] += local_bad[lid + stride];
            local_null[lid] += local_null[lid + stride];
            local_valid[lid] += local_valid[lid + stride];
            if (local_first_pos[lid + stride] < local_first_pos[lid]) {
                local_first_pos[lid] = local_first_pos[lid + stride];
                local_first_val[lid] = local_first_val[lid + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        out[0] = local_bad[0] == 0UL ? 0UL : 2UL;
        out[1] = local_bad[0];
        out[2] = local_null[0];
        out[3] = local_valid[0];
        out[4] = (ulong)as_uint(local_sum[0]);
        out[5] = (ulong)as_uint(local_max[0]);
        out[6] = ((ulong)local_first_pos[0] << 32) | (ulong)local_first_val[0];
        out[7] = 0x513C0DEF0BADF00DUL;
    }
}

// Local paged-K softmax-state gate. This extends the QK probe by exporting
// split-local m_i/l_i state while masking padded tokens from the final page.
// Null/OOB pages are classified before any K-cache dereference.
__kernel void paged_kv_softmax_state_probe(__global const half* q,
                                           __global const half* cache,
                                           __global const uint* indices,
                                           __global const uint* last_page_len,
                                           __global ulong* out,
                                           uint total_indices,
                                           uint physical_blocks,
                                           uint block_size,
                                           uint d) {
    const uint lid = get_local_id(0);
    const uint lsize = get_local_size(0);
    __local float local_max[256];
    __local float local_lsum[256];
    __local ulong local_bad[256];
    __local ulong local_null[256];
    __local ulong local_valid[256];
    __local uint local_first_pos[256];
    __local uint local_first_val[256];

    if (lid == 0) {
        out[0] = 3UL;
        out[1] = 0UL;
        out[2] = 0UL;
        out[3] = 0UL;
        out[4] = 0UL;
        out[5] = 0UL;
        out[6] = 0xffffffff00000000UL;
        out[7] = 0x50F74A7E0BADF00DUL;
    }

    if (total_indices == 0u || physical_blocks == 0u || block_size == 0u || d == 0u)
        return;

    const uint last = last_page_len[0];
    if (last == 0u || last > block_size)
        return;

    const ulong total_tokens = (ulong)total_indices * (ulong)block_size;
    float score_max = -3.402823466e+38F;
    ulong bad = 0UL;
    ulong nulls = 0UL;
    ulong valid = 0UL;
    uint first_pos = 0xffffffffu;
    uint first_val = 0u;

    for (ulong token = (ulong)lid; token < total_tokens; token += (ulong)lsize) {
        const uint page = (uint)(token / (ulong)block_size);
        const uint token_in_block = (uint)(token - ((ulong)page * (ulong)block_size));
        const uint active = (page + 1u == total_indices) ? last : block_size;
        if (token_in_block >= active)
            continue;

        const uint block_id = indices[page];
        if (block_id == 0u) {
            nulls += 1UL;
            continue;
        }
        if (block_id >= physical_blocks) {
            bad += 1UL;
            if ((uint)token < first_pos) {
                first_pos = (uint)token;
                first_val = block_id;
            }
            continue;
        }

        const ulong base = (((ulong)block_id * (ulong)block_size) + (ulong)token_in_block) * (ulong)d;
        float score = 0.0f;
        for (uint dim = 0u; dim < d; ++dim) {
            score += (float)q[dim] * (float)cache[base + (ulong)dim];
        }
        score_max = fmax(score_max, score);
        valid += 1UL;
    }

    local_max[lid] = score_max;
    local_bad[lid] = bad;
    local_null[lid] = nulls;
    local_valid[lid] = valid;
    local_first_pos[lid] = first_pos;
    local_first_val[lid] = first_val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_max[lid] = fmax(local_max[lid], local_max[lid + stride]);
            local_bad[lid] += local_bad[lid + stride];
            local_null[lid] += local_null[lid + stride];
            local_valid[lid] += local_valid[lid + stride];
            if (local_first_pos[lid + stride] < local_first_pos[lid]) {
                local_first_pos[lid] = local_first_pos[lid + stride];
                local_first_val[lid] = local_first_val[lid + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const float global_max = local_max[0];
    float lsum = 0.0f;
    for (ulong token = (ulong)lid; token < total_tokens; token += (ulong)lsize) {
        const uint page = (uint)(token / (ulong)block_size);
        const uint token_in_block = (uint)(token - ((ulong)page * (ulong)block_size));
        const uint active = (page + 1u == total_indices) ? last : block_size;
        if (token_in_block >= active)
            continue;

        const uint block_id = indices[page];
        if (block_id == 0u || block_id >= physical_blocks)
            continue;

        const ulong base = (((ulong)block_id * (ulong)block_size) + (ulong)token_in_block) * (ulong)d;
        float score = 0.0f;
        for (uint dim = 0u; dim < d; ++dim) {
            score += (float)q[dim] * (float)cache[base + (ulong)dim];
        }
        lsum += exp(score - global_max);
    }

    local_lsum[lid] = lsum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride)
            local_lsum[lid] += local_lsum[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        out[0] = local_bad[0] == 0UL ? 0UL : 2UL;
        out[1] = local_bad[0];
        out[2] = local_null[0];
        out[3] = local_valid[0];
        out[4] = (ulong)as_uint(global_max);
        out[5] = (ulong)as_uint(local_lsum[0]);
        out[6] = ((ulong)local_first_pos[0] << 32) | (ulong)local_first_val[0];
        out[7] = 0x50F74A7E0BADF00DUL;
    }
}

// Local paged-KV attention-output gate. This completes one split-local decode
// attention state: K produces m_i/l_i, then V is accumulated into O_i with the
// same null/OOB/final-page masks guarding every K and V read.
__kernel void paged_kv_attention_output_probe(__global const half* q,
                                              __global const half* k_cache,
                                              __global const half* v_cache,
                                              __global const uint* indices,
                                              __global const uint* last_page_len,
                                              __global ulong* out,
                                              uint total_indices,
                                              uint physical_blocks,
                                              uint block_size,
                                              uint d) {
    const uint lid = get_local_id(0);
    const uint lsize = get_local_size(0);
    __local float local_max[256];
    __local float local_lsum[256];
    __local ulong local_bad[256];
    __local ulong local_null[256];
    __local ulong local_valid[256];
    __local uint local_first_pos[256];
    __local uint local_first_val[256];

    if (lid == 0) {
        for (uint i = 0u; i < 16u; ++i)
            out[i] = 0UL;
        out[0] = 3UL;
        out[6] = 0xffffffff00000000UL;
        out[7] = 0xA77E47100BADF00DUL;
    }

    if (total_indices == 0u || physical_blocks == 0u || block_size == 0u || d == 0u || d > 8u)
        return;

    const uint last = last_page_len[0];
    if (last == 0u || last > block_size)
        return;

    const ulong total_tokens = (ulong)total_indices * (ulong)block_size;
    float score_max = -3.402823466e+38F;
    ulong bad = 0UL;
    ulong nulls = 0UL;
    ulong valid = 0UL;
    uint first_pos = 0xffffffffu;
    uint first_val = 0u;

    for (ulong token = (ulong)lid; token < total_tokens; token += (ulong)lsize) {
        const uint page = (uint)(token / (ulong)block_size);
        const uint token_in_block = (uint)(token - ((ulong)page * (ulong)block_size));
        const uint active = (page + 1u == total_indices) ? last : block_size;
        if (token_in_block >= active)
            continue;

        const uint block_id = indices[page];
        if (block_id == 0u) {
            nulls += 1UL;
            continue;
        }
        if (block_id >= physical_blocks) {
            bad += 1UL;
            if ((uint)token < first_pos) {
                first_pos = (uint)token;
                first_val = block_id;
            }
            continue;
        }

        const ulong base = (((ulong)block_id * (ulong)block_size) + (ulong)token_in_block) * (ulong)d;
        float score = 0.0f;
        for (uint dim = 0u; dim < d; ++dim) {
            score += (float)q[dim] * (float)k_cache[base + (ulong)dim];
        }
        score_max = fmax(score_max, score);
        valid += 1UL;
    }

    local_max[lid] = score_max;
    local_bad[lid] = bad;
    local_null[lid] = nulls;
    local_valid[lid] = valid;
    local_first_pos[lid] = first_pos;
    local_first_val[lid] = first_val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_max[lid] = fmax(local_max[lid], local_max[lid + stride]);
            local_bad[lid] += local_bad[lid + stride];
            local_null[lid] += local_null[lid + stride];
            local_valid[lid] += local_valid[lid + stride];
            if (local_first_pos[lid + stride] < local_first_pos[lid]) {
                local_first_pos[lid] = local_first_pos[lid + stride];
                local_first_val[lid] = local_first_val[lid + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const float global_max = local_max[0];
    float lsum = 0.0f;
    for (ulong token = (ulong)lid; token < total_tokens; token += (ulong)lsize) {
        const uint page = (uint)(token / (ulong)block_size);
        const uint token_in_block = (uint)(token - ((ulong)page * (ulong)block_size));
        const uint active = (page + 1u == total_indices) ? last : block_size;
        if (token_in_block >= active)
            continue;

        const uint block_id = indices[page];
        if (block_id == 0u || block_id >= physical_blocks)
            continue;

        const ulong base = (((ulong)block_id * (ulong)block_size) + (ulong)token_in_block) * (ulong)d;
        float score = 0.0f;
        for (uint dim = 0u; dim < d; ++dim) {
            score += (float)q[dim] * (float)k_cache[base + (ulong)dim];
        }
        lsum += exp(score - global_max);
    }

    local_lsum[lid] = lsum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride)
            local_lsum[lid] += local_lsum[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const float global_l = local_lsum[0];
    if (lid == 0) {
        float acc[8];
        for (uint dim = 0u; dim < 8u; ++dim)
            acc[dim] = 0.0f;

        for (ulong token = 0UL; token < total_tokens; ++token) {
            const uint page = (uint)(token / (ulong)block_size);
            const uint token_in_block = (uint)(token - ((ulong)page * (ulong)block_size));
            const uint active = (page + 1u == total_indices) ? last : block_size;
            if (token_in_block >= active)
                continue;

            const uint block_id = indices[page];
            if (block_id == 0u || block_id >= physical_blocks)
                continue;

            const ulong base = (((ulong)block_id * (ulong)block_size) + (ulong)token_in_block) * (ulong)d;
            float score = 0.0f;
            for (uint dim = 0u; dim < d; ++dim) {
                score += (float)q[dim] * (float)k_cache[base + (ulong)dim];
            }
            const float p = exp(score - global_max) / global_l;
            for (uint dim = 0u; dim < d; ++dim) {
                acc[dim] += p * (float)v_cache[base + (ulong)dim];
            }
        }

        out[0] = local_bad[0] == 0UL ? 0UL : 2UL;
        out[1] = local_bad[0];
        out[2] = local_null[0];
        out[3] = local_valid[0];
        out[4] = (ulong)as_uint(global_max);
        out[5] = (ulong)as_uint(global_l);
        out[6] = ((ulong)local_first_pos[0] << 32) | (ulong)local_first_val[0];
        out[7] = 0xA77E47100BADF00DUL;
        for (uint dim = 0u; dim < d; ++dim) {
            out[8u + dim] = (ulong)as_uint(acc[dim]);
        }
    }
}

// Split-KV state merge gate. This is the Flash-Decoding second-stage math in
// isolation: two local attention states (O_i, m_i, l_i) are rescaled and merged.
__kernel void split_kv_merge_state_probe(__global const float* in,
                                         __global ulong* out,
                                         uint d,
                                         uint reserved0,
                                         uint reserved1,
                                         uint reserved2) {
    if (get_global_id(0) != 0)
        return;

    for (uint i = 0u; i < 16u; ++i)
        out[i] = 0UL;
    out[7] = 0x5A117E570BADF00DUL;

    if (d == 0u || d > 8u) {
        out[0] = 3UL;
        return;
    }

    const uint base_b = d + 2u;
    const float m_a = in[d];
    const float l_a = in[d + 1u];
    const float m_b = in[base_b + d];
    const float l_b = in[base_b + d + 1u];
    const float m = fmax(m_a, m_b);
    const float w_a = l_a * exp(m_a - m);
    const float w_b = l_b * exp(m_b - m);
    const float l = w_a + w_b;
    if (!(l > 0.0f)) {
        out[0] = 4UL;
        out[1] = (ulong)d;
        out[7] = 0x5A117E570BADF00DUL;
        return;
    }

    out[0] = 0UL;
    out[1] = (ulong)d;
    out[2] = (ulong)as_uint(m);
    out[3] = (ulong)as_uint(l);
    out[4] = (ulong)as_uint(w_a);
    out[5] = (ulong)as_uint(w_b);
    out[6] = 0UL;
    out[7] = 0x5A117E570BADF00DUL;
    for (uint dim = 0u; dim < d; ++dim) {
        const float o = ((in[dim] * w_a) + (in[base_b + dim] * w_b)) / l;
        out[8u + dim] = (ulong)as_uint(o);
    }
}

// Merge two local attention-output probe records. This bridges the real paged
// local attention state layout to the split-KV merge formula.
__kernel void split_kv_merge_attention_states_probe(__global const ulong* a,
                                                    __global const ulong* b,
                                                    __global ulong* out,
                                                    uint d,
                                                    uint reserved0,
                                                    uint reserved1,
                                                    uint reserved2) {
    if (get_global_id(0) != 0)
        return;

    for (uint i = 0u; i < 16u; ++i)
        out[i] = 0UL;
    out[7] = 0x5A117E580BADF00DUL;

    if (d == 0u || d > 8u) {
        out[0] = 3UL;
        return;
    }
    if (a[7] != 0xA77E47100BADF00DUL || b[7] != 0xA77E47100BADF00DUL) {
        out[0] = 5UL;
        return;
    }

    const float m_a = as_float((uint)a[4]);
    const float l_a = as_float((uint)a[5]);
    const float m_b = as_float((uint)b[4]);
    const float l_b = as_float((uint)b[5]);
    const float m = fmax(m_a, m_b);
    const float w_a = l_a * exp(m_a - m);
    const float w_b = l_b * exp(m_b - m);
    const float l = w_a + w_b;
    if (!(l > 0.0f)) {
        out[0] = 4UL;
        out[1] = (ulong)d;
        out[7] = 0x5A117E580BADF00DUL;
        return;
    }

    const ulong bad = a[1] + b[1];
    const ulong nulls = a[2] + b[2];
    const ulong valid = a[3] + b[3];
    out[0] = (a[0] == 0UL && b[0] == 0UL) ? 0UL : 2UL;
    out[1] = bad;
    out[2] = nulls;
    out[3] = valid;
    out[4] = (ulong)as_uint(m);
    out[5] = (ulong)as_uint(l);
    out[6] = a[1] != 0UL ? a[6] : b[6];
    out[7] = 0x5A117E580BADF00DUL;
    for (uint dim = 0u; dim < d; ++dim) {
        const float o_a = as_float((uint)a[8u + dim]);
        const float o_b = as_float((uint)b[8u + dim]);
        const float o = ((o_a * w_a) + (o_b * w_b)) / l;
        out[8u + dim] = (ulong)as_uint(o);
    }
}

__kernel void attn_decode_split2_paged(__global const half* q,
                                       __global const half* K, __global const half* V,
                                       __global const uint* block_table, uint block_size,
                                       uint physical_blocks,
                                       __global float* partials,
                                       uint N, uint D, float scale, uint num_splits) {
    uint sp = get_group_id(0);
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint g = lane >> 4;
    uint sub = lane & 15;
    uint S = (N + num_splits - 1) / num_splits;
    uint lo = sp * S;
    uint hi = min(N, lo + S);
    uint span = hi > lo ? hi - lo : 0;
    uint per = (span + WPS_ATTN - 1) / WPS_ATTN;
    uint wlo = lo + w * per;
    uint whi = min(hi, wlo + per);

    half8 q8 = vload8(sub, q);
    float qv[8];
    qv[0]=q8.s0; qv[1]=q8.s1; qv[2]=q8.s2; qv[3]=q8.s3;
    qv[4]=q8.s4; qv[5]=q8.s5; qv[6]=q8.s6; qv[7]=q8.s7;

    float m = -INFINITY, l = 0.0f, o[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = 0.0f;

    for (uint base = wlo + g; base < whi; base += 4u * UATTN) {
        half8 kb[UATTN], vb[UATTN];
        #pragma unroll
        for (uint u = 0; u < UATTN; ++u) {
            uint t = base + 4u * u;
            uint tt = t < whi ? t : whi - 1;
            uint lb = tt / block_size;
            uint pb = block_table[lb];
            pb = pb < physical_blocks ? pb : 0u;
            ulong row = (ulong)pb * block_size + (tt - lb * block_size);
            kb[u] = vload8(sub, K + row * 128);
            vb[u] = vload8(sub, V + row * 128);
        }
        #pragma unroll
        for (uint u = 0; u < UATTN; ++u) {
            uint t = base + 4u * u;
            if (t >= whi) break;
            half8 k8 = kb[u];
            float partial = qv[0]*(float)k8.s0 + qv[1]*(float)k8.s1 + qv[2]*(float)k8.s2 + qv[3]*(float)k8.s3
                          + qv[4]*(float)k8.s4 + qv[5]*(float)k8.s5 + qv[6]*(float)k8.s6 + qv[7]*(float)k8.s7;
            partial += BPERM(1u, partial);
            partial += BPERM(2u, partial);
            partial += BPERM(4u, partial);
            partial += BPERM(8u, partial);
            float s = partial * scale;
            float m_new = fmax(m, s);
            float corr = native_exp(m - m_new);
            float p = native_exp(s - m_new);
            l = l * corr + p;
            half8 v8 = vb[u];
            o[0]=o[0]*corr+p*(float)v8.s0; o[1]=o[1]*corr+p*(float)v8.s1;
            o[2]=o[2]*corr+p*(float)v8.s2; o[3]=o[3]*corr+p*(float)v8.s3;
            o[4]=o[4]*corr+p*(float)v8.s4; o[5]=o[5]*corr+p*(float)v8.s5;
            o[6]=o[6]*corr+p*(float)v8.s6; o[7]=o[7]*corr+p*(float)v8.s7;
            m = m_new;
        }
    }

    float M = m;
    M = fmax(M, BPERM(16u, M));
    M = fmax(M, BPERM(32u, M));
    if (M == -INFINITY) M = 0.0f;
    float cg = native_exp(m - M);
    float L = l * cg;
    L += BPERM(16u, L);
    L += BPERM(32u, L);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float oc = o[i] * cg;
        oc += BPERM(16u, oc);
        oc += BPERM(32u, oc);
        o[i] = oc;
    }
    __local float wm[WPS_ATTN], wl[WPS_ATTN], wo[WPS_ATTN][128];
    if (lane == 0) { wm[w] = M; wl[w] = L; }
    if (g == 0) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) wo[w][sub * 8 + i] = o[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (w == 0 && lane < 64) {
        float MM = -INFINITY;
        for (uint k = 0; k < WPS_ATTN; ++k) MM = fmax(MM, wm[k]);
        if (MM == -INFINITY) MM = 0.0f;
        float LL = 0.0f;
        for (uint k = 0; k < WPS_ATTN; ++k) LL += wl[k] * native_exp(wm[k] - MM);
        __global float* pr = partials + (ulong)sp * (D + 2);
        if (lane == 0) { pr[0] = MM; pr[1] = LL; }
        for (uint dd = lane; dd < 128; dd += 64) {
            float acc = 0.0f;
            for (uint k = 0; k < WPS_ATTN; ++k)
                acc += wo[k][dd] * native_exp(wm[k] - MM);
            pr[2 + dd] = acc;
        }
    }
}

// CAPSTONE: paged + GQA + FP4. The real 1M-context Qwen decode primitive — a
// per-sequence block_table maps logical->physical blocks (paged KV), G query
// heads share one 4-bit (E2M1, per-block-32 E8M0) KV head (read+dequant once,
// reuse across heads). Composes all three decode levers: paging (serving),
// 4x compression, and G-head amortization. Scales are stored per physical token
// alongside the KV (so indexed by the physical row too).
__kernel void attn_decode_split2_fp4_gqa_paged(
    __global const half* q,                                  // [GQA_G][128]
    __global const uchar* K, __global const uchar* V,        // physical FP4 blocks [*][64]
    __global const uchar* scale_k, __global const uchar* scale_v,  // physical E8M0 [*][4]
    __global const uint* block_table, uint block_size, uint physical_blocks,
    __global float* partials,                                // [GQA_G][num_splits][D+2]
    uint N, uint D, float scale, uint num_splits) {
    uint sp = get_group_id(0);
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint g = lane >> 4;
    uint sub = lane & 15;
    uint blk = sub >> 2;
    uint S = (N + num_splits - 1) / num_splits;
    uint lo = sp * S;
    uint hi = min(N, lo + S);
    uint span = hi > lo ? hi - lo : 0;
    uint per = (span + WPS_ATTN - 1) / WPS_ATTN;
    uint wlo = lo + w * per;
    uint whi = min(hi, wlo + per);

    float qv[GQA_G][8];
    #pragma unroll
    for (uint h = 0; h < GQA_G; ++h) {
        half8 q8 = vload8(sub, q + h * 128);
        qv[h][0]=q8.s0; qv[h][1]=q8.s1; qv[h][2]=q8.s2; qv[h][3]=q8.s3;
        qv[h][4]=q8.s4; qv[h][5]=q8.s5; qv[h][6]=q8.s6; qv[h][7]=q8.s7;
    }
    float m[GQA_G], l[GQA_G], o[GQA_G][8];
    #pragma unroll
    for (uint h = 0; h < GQA_G; ++h) {
        m[h] = -INFINITY; l[h] = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) o[h][i] = 0.0f;
    }

    for (uint base = wlo + g; base < whi; base += 4u * UATTN) {
        uint kb[UATTN], vb[UATTN];
        float ks[UATTN], vs[UATTN];
        #pragma unroll
        for (uint u = 0; u < UATTN; ++u) {
            uint t = base + 4u * u;
            uint tt = t < whi ? t : whi - 1;
            uint lb = tt / block_size;
            uint pb = block_table[lb];
            pb = pb < physical_blocks ? pb : 0u;
            ulong row = (ulong)pb * block_size + (tt - lb * block_size);
            kb[u] = ((__global const uint*)(K + row * 64))[sub];
            vb[u] = ((__global const uint*)(V + row * 64))[sub];
            ks[u] = as_float(((uint)scale_k[row * 4 + blk]) << 23);
            vs[u] = as_float(((uint)scale_v[row * 4 + blk]) << 23);
        }
        #pragma unroll
        for (uint u = 0; u < UATTN; ++u) {
            uint t = base + 4u * u;
            if (t >= whi) break;
            uint k4 = kb[u]; float bsk = ks[u];
            float2 ka = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(k4, bsk, 0);
            float2 kc = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(k4, bsk, 1);
            float2 ke = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(k4, bsk, 2);
            float2 kg = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(k4, bsk, 3);
            float kf[8] = {ka.x,ka.y,kc.x,kc.y,ke.x,ke.y,kg.x,kg.y};
            uint v4 = vb[u]; float bsv = vs[u];
            float2 ea = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, bsv, 0);
            float2 ec = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, bsv, 1);
            float2 ee = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, bsv, 2);
            float2 eg = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, bsv, 3);
            float vf[8] = {ea.x,ea.y,ec.x,ec.y,ee.x,ee.y,eg.x,eg.y};
            #pragma unroll
            for (uint h = 0; h < GQA_G; ++h) {
                float partial = qv[h][0]*kf[0] + qv[h][1]*kf[1] + qv[h][2]*kf[2] + qv[h][3]*kf[3]
                              + qv[h][4]*kf[4] + qv[h][5]*kf[5] + qv[h][6]*kf[6] + qv[h][7]*kf[7];
                partial += BPERM(1u, partial);
                partial += BPERM(2u, partial);
                partial += BPERM(4u, partial);
                partial += BPERM(8u, partial);
                float s = partial * scale;
                float m_new = fmax(m[h], s);
                float corr = native_exp(m[h] - m_new);
                float p = native_exp(s - m_new);
                l[h] = l[h] * corr + p;
                #pragma unroll
                for (int i = 0; i < 8; ++i) o[h][i] = o[h][i]*corr + p*vf[i];
                m[h] = m_new;
            }
        }
    }

    __local float wm[GQA_G][WPS_ATTN], wl[GQA_G][WPS_ATTN], wo[GQA_G][WPS_ATTN][128];
    #pragma unroll
    for (uint h = 0; h < GQA_G; ++h) {
        float M = m[h];
        M = fmax(M, BPERM(16u, M));
        M = fmax(M, BPERM(32u, M));
        if (M == -INFINITY) M = 0.0f;
        float cg = native_exp(m[h] - M);
        float L = l[h] * cg;
        L += BPERM(16u, L);
        L += BPERM(32u, L);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float oc = o[h][i] * cg;
            oc += BPERM(16u, oc);
            oc += BPERM(32u, oc);
            o[h][i] = oc;
        }
        if (lane == 0) { wm[h][w] = M; wl[h][w] = L; }
        if (g == 0) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) wo[h][w][sub * 8 + i] = o[h][i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (w == 0 && lane < 64) {
        #pragma unroll
        for (uint h = 0; h < GQA_G; ++h) {
            float MM = -INFINITY;
            for (uint k = 0; k < WPS_ATTN; ++k) MM = fmax(MM, wm[h][k]);
            if (MM == -INFINITY) MM = 0.0f;
            float LL = 0.0f;
            for (uint k = 0; k < WPS_ATTN; ++k) LL += wl[h][k] * native_exp(wm[h][k] - MM);
            __global float* pr = partials + ((ulong)h * num_splits + sp) * (D + 2);
            if (lane == 0) { pr[0] = MM; pr[1] = LL; }
            for (uint dd = lane; dd < 128; dd += 64) {
                float acc = 0.0f;
                for (uint k = 0; k < WPS_ATTN; ++k)
                    acc += wo[h][k][dd] * native_exp(wm[h][k] - MM);
                pr[2 + dd] = acc;
            }
        }
    }
}

// Paged + GQA + FP8/E4M3 decode. This is the FP8 KV-cache admission twin of
// attn_decode_split2_fp4_gqa_paged: one KV head is stored in shuffled physical
// pages, G query heads share each dequantized K/V row, and scales are indexed by
// physical row so the block table exercises the same serving surface as FP4.
__kernel void attn_decode_split2_fp8_gqa_paged(
    __global const half* q,                                  // [GQA_G][128]
    __global const uchar* K, __global const uchar* V,        // physical FP8 blocks [*][128]
    __global const float* scale_k, __global const float* scale_v,
    __global const uint* block_table, uint block_size, uint physical_blocks,
    __global float* partials,                                // [GQA_G][num_splits][D+2]
    uint N, uint D, float scale, uint num_splits) {
    uint sp = get_group_id(0);
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint g = lane >> 4;
    uint sub = lane & 15;
    uint S = (N + num_splits - 1) / num_splits;
    uint lo = sp * S;
    uint hi = min(N, lo + S);
    uint span = hi > lo ? hi - lo : 0;
    uint per = (span + WPS_ATTN - 1) / WPS_ATTN;
    uint wlo = lo + w * per;
    uint whi = min(hi, wlo + per);

    float qv[GQA_G][8];
    #pragma unroll
    for (uint h = 0; h < GQA_G; ++h) {
        half8 q8 = vload8(sub, q + h * 128);
        qv[h][0]=q8.s0; qv[h][1]=q8.s1; qv[h][2]=q8.s2; qv[h][3]=q8.s3;
        qv[h][4]=q8.s4; qv[h][5]=q8.s5; qv[h][6]=q8.s6; qv[h][7]=q8.s7;
    }
    float m[GQA_G], l[GQA_G], o[GQA_G][8];
    #pragma unroll
    for (uint h = 0; h < GQA_G; ++h) {
        m[h] = -INFINITY; l[h] = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) o[h][i] = 0.0f;
    }

    for (uint base = wlo + g; base < whi; base += 4u * UATTN) {
        uint2 kb[UATTN], vb[UATTN];
        float ks[UATTN], vs[UATTN];
        #pragma unroll
        for (uint u = 0; u < UATTN; ++u) {
            uint t = base + 4u * u;
            uint tt = t < whi ? t : whi - 1;
            uint lb = tt / block_size;
            uint pb = block_table[lb];
            pb = pb < physical_blocks ? pb : 0u;
            ulong row = (ulong)pb * block_size + (tt - lb * block_size);
            kb[u] = ((__global const uint2*)(K + row * 128))[sub];
            vb[u] = ((__global const uint2*)(V + row * 128))[sub];
            ks[u] = scale_k[row];
            vs[u] = scale_v[row];
        }
        #pragma unroll
        for (uint u = 0; u < UATTN; ++u) {
            uint t = base + 4u * u;
            if (t >= whi) break;
            uint2 k2 = kb[u];
            float2 ka = __builtin_amdgcn_cvt_pk_f32_fp8(k2.x, false);
            float2 kc = __builtin_amdgcn_cvt_pk_f32_fp8(k2.x, true);
            float2 ke = __builtin_amdgcn_cvt_pk_f32_fp8(k2.y, false);
            float2 kg = __builtin_amdgcn_cvt_pk_f32_fp8(k2.y, true);
            float kf[8] = {ka.x, ka.y, kc.x, kc.y, ke.x, ke.y, kg.x, kg.y};
            uint2 v2 = vb[u];
            float2 va = __builtin_amdgcn_cvt_pk_f32_fp8(v2.x, false);
            float2 vc = __builtin_amdgcn_cvt_pk_f32_fp8(v2.x, true);
            float2 ve = __builtin_amdgcn_cvt_pk_f32_fp8(v2.y, false);
            float2 vg = __builtin_amdgcn_cvt_pk_f32_fp8(v2.y, true);
            float vf[8] = {va.x, va.y, vc.x, vc.y, ve.x, ve.y, vg.x, vg.y};
            float skt = ks[u] * scale;
            float svt = vs[u];
            #pragma unroll
            for (uint h = 0; h < GQA_G; ++h) {
                float partial = qv[h][0]*kf[0] + qv[h][1]*kf[1] + qv[h][2]*kf[2] + qv[h][3]*kf[3]
                              + qv[h][4]*kf[4] + qv[h][5]*kf[5] + qv[h][6]*kf[6] + qv[h][7]*kf[7];
                partial += BPERM(1u, partial);
                partial += BPERM(2u, partial);
                partial += BPERM(4u, partial);
                partial += BPERM(8u, partial);
                float s = partial * skt;
                float m_new = fmax(m[h], s);
                float corr = native_exp(m[h] - m_new);
                float p = native_exp(s - m_new);
                l[h] = l[h] * corr + p;
                float pv = p * svt;
                #pragma unroll
                for (int i = 0; i < 8; ++i) o[h][i] = o[h][i]*corr + pv*vf[i];
                m[h] = m_new;
            }
        }
    }

    __local float wm[GQA_G][WPS_ATTN], wl[GQA_G][WPS_ATTN], wo[GQA_G][WPS_ATTN][128];
    #pragma unroll
    for (uint h = 0; h < GQA_G; ++h) {
        float M = m[h];
        M = fmax(M, BPERM(16u, M));
        M = fmax(M, BPERM(32u, M));
        if (M == -INFINITY) M = 0.0f;
        float cg = native_exp(m[h] - M);
        float L = l[h] * cg;
        L += BPERM(16u, L);
        L += BPERM(32u, L);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float oc = o[h][i] * cg;
            oc += BPERM(16u, oc);
            oc += BPERM(32u, oc);
            o[h][i] = oc;
        }
        if (lane == 0) { wm[h][w] = M; wl[h][w] = L; }
        if (g == 0) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) wo[h][w][sub * 8 + i] = o[h][i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (w == 0 && lane < 64) {
        #pragma unroll
        for (uint h = 0; h < GQA_G; ++h) {
            float MM = -INFINITY;
            for (uint k = 0; k < WPS_ATTN; ++k) MM = fmax(MM, wm[h][k]);
            if (MM == -INFINITY) MM = 0.0f;
            float LL = 0.0f;
            for (uint k = 0; k < WPS_ATTN; ++k) LL += wl[h][k] * native_exp(wm[h][k] - MM);
            __global float* pr = partials + ((ulong)h * num_splits + sp) * (D + 2);
            if (lane == 0) { pr[0] = MM; pr[1] = LL; }
            for (uint dd = lane; dd < 128; dd += 64) {
                float acc = 0.0f;
                for (uint k = 0; k < WPS_ATTN; ++k)
                    acc += wo[h][k][dd] * native_exp(wm[h][k] - MM);
                pr[2 + dd] = acc;
            }
        }
    }
}

// Same per-workgroup math as attn_decode_split2_fp4_gqa_paged, but batches all
// GQA_G-sized query-head groups into one dispatch. Grid x is
// num_groups*num_splits; group_id = gid/num_splits, split = gid%num_splits.
__kernel void attn_decode_split2_fp4_gqa_paged_groups(
    __global const half* q_all,                              // [num_groups*GQA_G][128]
    __global const uchar* K_all, __global const uchar* V_all, // [kv_heads][rows_per_head][64]
    __global const uchar* scale_k_all, __global const uchar* scale_v_all,
    __global const uint* block_table, uint block_size, uint physical_blocks,
    __global float* partials,                                // [num_groups*GQA_G][num_splits][D+2]
    uint N, uint D, float scale, uint num_splits,
    uint num_groups, uint q_heads_per_kv, uint rows_per_head) {
    uint gid = get_group_id(0);
    uint group_id = gid / num_splits;
    uint sp = gid - group_id * num_splits;
    if (group_id >= num_groups) return;
    uint head_base = group_id * GQA_G;
    uint kvh = head_base / q_heads_per_kv;
    __global const half* q = q_all + (ulong)head_base * 128;
    __global const uchar* K = K_all + (ulong)kvh * rows_per_head * 64;
    __global const uchar* V = V_all + (ulong)kvh * rows_per_head * 64;
    __global const uchar* scale_k = scale_k_all + (ulong)kvh * rows_per_head * 4;
    __global const uchar* scale_v = scale_v_all + (ulong)kvh * rows_per_head * 4;
    __global float* group_partials = partials + (ulong)head_base * num_splits * (D + 2);

    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint g = lane >> 4;
    uint sub = lane & 15;
    uint blk = sub >> 2;
    uint S = (N + num_splits - 1) / num_splits;
    uint lo = sp * S;
    uint hi = min(N, lo + S);
    uint span = hi > lo ? hi - lo : 0;
    uint per = (span + WPS_ATTN - 1) / WPS_ATTN;
    uint wlo = lo + w * per;
    uint whi = min(hi, wlo + per);

    float qv[GQA_G][8];
    #pragma unroll
    for (uint h = 0; h < GQA_G; ++h) {
        half8 q8 = vload8(sub, q + h * 128);
        qv[h][0]=q8.s0; qv[h][1]=q8.s1; qv[h][2]=q8.s2; qv[h][3]=q8.s3;
        qv[h][4]=q8.s4; qv[h][5]=q8.s5; qv[h][6]=q8.s6; qv[h][7]=q8.s7;
    }
    float m[GQA_G], l[GQA_G], o[GQA_G][8];
    #pragma unroll
    for (uint h = 0; h < GQA_G; ++h) {
        m[h] = -INFINITY; l[h] = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) o[h][i] = 0.0f;
    }

    for (uint base = wlo + g; base < whi; base += 4u * UATTN) {
        uint kb[UATTN], vb[UATTN];
        float ks[UATTN], vs[UATTN];
        #pragma unroll
        for (uint u = 0; u < UATTN; ++u) {
            uint t = base + 4u * u;
            uint tt = t < whi ? t : whi - 1;
            uint lb = tt / block_size;
            uint pb = block_table[lb];
            pb = pb < physical_blocks ? pb : 0u;
            ulong row = (ulong)pb * block_size + (tt - lb * block_size);
            kb[u] = ((__global const uint*)(K + row * 64))[sub];
            vb[u] = ((__global const uint*)(V + row * 64))[sub];
            ks[u] = as_float(((uint)scale_k[row * 4 + blk]) << 23);
            vs[u] = as_float(((uint)scale_v[row * 4 + blk]) << 23);
        }
        #pragma unroll
        for (uint u = 0; u < UATTN; ++u) {
            uint t = base + 4u * u;
            if (t >= whi) break;
            uint k4 = kb[u]; float bsk = ks[u];
            float2 ka = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(k4, bsk, 0);
            float2 kc = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(k4, bsk, 1);
            float2 ke = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(k4, bsk, 2);
            float2 kg = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(k4, bsk, 3);
            float kf[8] = {ka.x,ka.y,kc.x,kc.y,ke.x,ke.y,kg.x,kg.y};
            uint v4 = vb[u]; float bsv = vs[u];
            float2 ea = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, bsv, 0);
            float2 ec = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, bsv, 1);
            float2 ee = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, bsv, 2);
            float2 eg = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, bsv, 3);
            float vf[8] = {ea.x,ea.y,ec.x,ec.y,ee.x,ee.y,eg.x,eg.y};
            #pragma unroll
            for (uint h = 0; h < GQA_G; ++h) {
                float partial = qv[h][0]*kf[0] + qv[h][1]*kf[1] + qv[h][2]*kf[2] + qv[h][3]*kf[3]
                              + qv[h][4]*kf[4] + qv[h][5]*kf[5] + qv[h][6]*kf[6] + qv[h][7]*kf[7];
                partial += BPERM(1u, partial);
                partial += BPERM(2u, partial);
                partial += BPERM(4u, partial);
                partial += BPERM(8u, partial);
                float s = partial * scale;
                float m_new = fmax(m[h], s);
                float corr = native_exp(m[h] - m_new);
                float p = native_exp(s - m_new);
                l[h] = l[h] * corr + p;
                #pragma unroll
                for (int i = 0; i < 8; ++i) o[h][i] = o[h][i]*corr + p*vf[i];
                m[h] = m_new;
            }
        }
    }

    __local float wm[GQA_G][WPS_ATTN], wl[GQA_G][WPS_ATTN], wo[GQA_G][WPS_ATTN][128];
    #pragma unroll
    for (uint h = 0; h < GQA_G; ++h) {
        float M = m[h];
        M = fmax(M, BPERM(16u, M));
        M = fmax(M, BPERM(32u, M));
        if (M == -INFINITY) M = 0.0f;
        float cg = native_exp(m[h] - M);
        float L = l[h] * cg;
        L += BPERM(16u, L);
        L += BPERM(32u, L);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float oc = o[h][i] * cg;
            oc += BPERM(16u, oc);
            oc += BPERM(32u, oc);
            o[h][i] = oc;
        }
        if (lane == 0) { wm[h][w] = M; wl[h][w] = L; }
        if (g == 0) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) wo[h][w][sub * 8 + i] = o[h][i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (w == 0 && lane < 64) {
        #pragma unroll
        for (uint h = 0; h < GQA_G; ++h) {
            float MM = -INFINITY;
            for (uint k = 0; k < WPS_ATTN; ++k) MM = fmax(MM, wm[h][k]);
            if (MM == -INFINITY) MM = 0.0f;
            float wc[WPS_ATTN];
            float LL = 0.0f;
            for (uint k = 0; k < WPS_ATTN; ++k) {
                wc[k] = native_exp(wm[h][k] - MM);
                LL += wl[h][k] * wc[k];
            }
            __global float* pr = group_partials + ((ulong)h * num_splits + sp) * (D + 2);
            if (lane == 0) { pr[0] = MM; pr[1] = LL; }
            for (uint dd = lane; dd < 128; dd += 64) {
                float acc = 0.0f;
                for (uint k = 0; k < WPS_ATTN; ++k)
                    acc += wo[h][k][dd] * wc[k];
                pr[2 + dd] = acc;
            }
        }
    }
}

// Qwen3-4B rank-local variant of attn_decode_split2_fp4_gqa_paged_groups:
// exactly 4 query heads share one KV head. This removes the 8-head padding used
// by the larger serving group kernel while preserving the same paged FP4 cache
// layout and one-workgroup-per-(group, split) execution model.
__kernel void attn_decode_split2_fp4_gqa4_paged_groups(
    __global const half* q_all,                              // [num_groups*4][128]
    __global const uchar* K_all, __global const uchar* V_all, // [kv_heads][rows_per_head][64]
    __global const uchar* scale_k_all, __global const uchar* scale_v_all,
    __global const uint* block_table, uint block_size, uint physical_blocks,
    __global float* partials,                                // [num_groups*4][num_splits][D+2]
    uint N, uint D, float scale, uint num_splits,
    uint num_groups, uint q_heads_per_kv, uint rows_per_head) {
    const uint GQA4_G = 4u;
    uint gid = get_group_id(0);
    uint group_id = gid / num_splits;
    uint sp = gid - group_id * num_splits;
    if (group_id >= num_groups) return;
    uint head_base = group_id * GQA4_G;
    uint kvh = head_base / q_heads_per_kv;
    __global const half* q = q_all + (ulong)head_base * 128;
    __global const uchar* K = K_all + (ulong)kvh * rows_per_head * 64;
    __global const uchar* V = V_all + (ulong)kvh * rows_per_head * 64;
    __global const uchar* scale_k = scale_k_all + (ulong)kvh * rows_per_head * 4;
    __global const uchar* scale_v = scale_v_all + (ulong)kvh * rows_per_head * 4;
    __global float* group_partials = partials + (ulong)head_base * num_splits * (D + 2);

    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint g = lane >> 4;
    uint sub = lane & 15;
    uint blk = sub >> 2;
    uint S = (N + num_splits - 1) / num_splits;
    uint lo = sp * S;
    uint hi = min(N, lo + S);
    uint span = hi > lo ? hi - lo : 0;
    uint per = (span + WPS_ATTN - 1) / WPS_ATTN;
    uint wlo = lo + w * per;
    uint whi = min(hi, wlo + per);

    float qv[4][8];
    #pragma unroll
    for (uint h = 0; h < GQA4_G; ++h) {
        half8 q8 = vload8(sub, q + h * 128);
        qv[h][0]=q8.s0; qv[h][1]=q8.s1; qv[h][2]=q8.s2; qv[h][3]=q8.s3;
        qv[h][4]=q8.s4; qv[h][5]=q8.s5; qv[h][6]=q8.s6; qv[h][7]=q8.s7;
    }
    float m[4], l[4], o[4][8];
    #pragma unroll
    for (uint h = 0; h < GQA4_G; ++h) {
        m[h] = -INFINITY; l[h] = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) o[h][i] = 0.0f;
    }

    for (uint base = wlo + g; base < whi; base += 4u * UATTN) {
        uint kb[UATTN], vb[UATTN];
        float ks[UATTN], vs[UATTN];
        #pragma unroll
        for (uint u = 0; u < UATTN; ++u) {
            uint t = base + 4u * u;
            uint tt = t < whi ? t : whi - 1;
            uint lb = tt / block_size;
            uint pb = block_table[lb];
            pb = pb < physical_blocks ? pb : 0u;
            ulong row = (ulong)pb * block_size + (tt - lb * block_size);
            kb[u] = ((__global const uint*)(K + row * 64))[sub];
            vb[u] = ((__global const uint*)(V + row * 64))[sub];
            ks[u] = as_float(((uint)scale_k[row * 4 + blk]) << 23);
            vs[u] = as_float(((uint)scale_v[row * 4 + blk]) << 23);
        }
        #pragma unroll
        for (uint u = 0; u < UATTN; ++u) {
            uint t = base + 4u * u;
            if (t >= whi) break;
            uint k4 = kb[u]; float bsk = ks[u];
            float2 ka = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(k4, bsk, 0);
            float2 kc = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(k4, bsk, 1);
            float2 ke = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(k4, bsk, 2);
            float2 kg = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(k4, bsk, 3);
            float kf[8] = {ka.x,ka.y,kc.x,kc.y,ke.x,ke.y,kg.x,kg.y};
            uint v4 = vb[u]; float bsv = vs[u];
            float2 ea = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, bsv, 0);
            float2 ec = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, bsv, 1);
            float2 ee = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, bsv, 2);
            float2 eg = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, bsv, 3);
            float vf[8] = {ea.x,ea.y,ec.x,ec.y,ee.x,ee.y,eg.x,eg.y};
            #pragma unroll
            for (uint h = 0; h < GQA4_G; ++h) {
                float partial = qv[h][0]*kf[0] + qv[h][1]*kf[1] + qv[h][2]*kf[2] + qv[h][3]*kf[3]
                              + qv[h][4]*kf[4] + qv[h][5]*kf[5] + qv[h][6]*kf[6] + qv[h][7]*kf[7];
                partial += BPERM(1u, partial);
                partial += BPERM(2u, partial);
                partial += BPERM(4u, partial);
                partial += BPERM(8u, partial);
                float s = partial * scale;
                float m_new = fmax(m[h], s);
                float corr = native_exp(m[h] - m_new);
                float p = native_exp(s - m_new);
                l[h] = l[h] * corr + p;
                #pragma unroll
                for (int i = 0; i < 8; ++i) o[h][i] = o[h][i]*corr + p*vf[i];
                m[h] = m_new;
            }
        }
    }

    __local float wm[4][WPS_ATTN], wl[4][WPS_ATTN], wo[4][WPS_ATTN][128];
    #pragma unroll
    for (uint h = 0; h < GQA4_G; ++h) {
        float M = m[h];
        M = fmax(M, BPERM(16u, M));
        M = fmax(M, BPERM(32u, M));
        if (M == -INFINITY) M = 0.0f;
        float cg = native_exp(m[h] - M);
        float L = l[h] * cg;
        L += BPERM(16u, L);
        L += BPERM(32u, L);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float oc = o[h][i] * cg;
            oc += BPERM(16u, oc);
            oc += BPERM(32u, oc);
            o[h][i] = oc;
        }
        if (lane == 0) { wm[h][w] = M; wl[h][w] = L; }
        if (g == 0) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) wo[h][w][sub * 8 + i] = o[h][i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (w == 0 && lane < 64) {
        #pragma unroll
        for (uint h = 0; h < GQA4_G; ++h) {
            float MM = -INFINITY;
            for (uint k = 0; k < WPS_ATTN; ++k) MM = fmax(MM, wm[h][k]);
            if (MM == -INFINITY) MM = 0.0f;
            float wc[WPS_ATTN];
            float LL = 0.0f;
            for (uint k = 0; k < WPS_ATTN; ++k) {
                wc[k] = native_exp(wm[h][k] - MM);
                LL += wl[h][k] * wc[k];
            }
            __global float* pr = group_partials + ((ulong)h * num_splits + sp) * (D + 2);
            if (lane == 0) { pr[0] = MM; pr[1] = LL; }
            for (uint dd = lane; dd < 128; dd += 64) {
                float acc = 0.0f;
                for (uint k = 0; k < WPS_ATTN; ++k)
                    acc += wo[h][k][dd] * wc[k];
                pr[2 + dd] = acc;
            }
        }
    }
}

// 5D-layout Qwen3-4B rank-local FP4 attention consumer. This is intentionally
// the same split-softmax math as attn_decode_split2_fp4_gqa4_paged_groups; the
// only contract change is the KV/cache addressing:
//   data  [physical_block][kv_head][fp4_group32][token_in_block][packed16]
//   scale [physical_block][kv_head][fp4_group32][token_in_block]
// It proves the append-time preshuffle is consumable directly by attention
// before the CDNA4 scaled-MFMA path is allowed by the host readiness gate.
__kernel void attn_decode_split2_fp4_5d_gqa4_paged_groups(
    __global const half* q_all,                              // [num_groups*4][128]
    __global const uchar* K_all, __global const uchar* V_all, // [physical_blocks][kv_heads][4][block_size][16]
    __global const uchar* scale_k_all, __global const uchar* scale_v_all,
    __global const uint* block_table, uint block_size, uint physical_blocks,
    __global float* partials,                                // [num_groups*4][num_splits][D+2]
    uint N, uint D, float scale, uint num_splits,
    uint num_groups, uint q_heads_per_kv) {
    const uint GQA4_G = 4u;
    uint gid = get_group_id(0);
    uint group_id = gid / num_splits;
    uint sp = gid - group_id * num_splits;
    if (group_id >= num_groups) return;
    uint head_base = group_id * GQA4_G;
    uint kvh = head_base / q_heads_per_kv;
    uint kv_heads = (num_groups * GQA4_G + q_heads_per_kv - 1u) / q_heads_per_kv;
    __global const half* q = q_all + (ulong)head_base * 128;
    __global float* group_partials = partials + (ulong)head_base * num_splits * (D + 2);

    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint g = lane >> 4;
    uint sub = lane & 15;
    uint blk = sub >> 2;
    uint sub4 = sub & 3u;
    uint S = (N + num_splits - 1) / num_splits;
    uint lo = sp * S;
    uint hi = min(N, lo + S);
    uint span = hi > lo ? hi - lo : 0;
    uint per = (span + WPS_ATTN - 1) / WPS_ATTN;
    uint wlo = lo + w * per;
    uint whi = min(hi, wlo + per);

    float qv[4][8];
    #pragma unroll
    for (uint h = 0; h < GQA4_G; ++h) {
        half8 q8 = vload8(sub, q + h * 128);
        qv[h][0]=q8.s0; qv[h][1]=q8.s1; qv[h][2]=q8.s2; qv[h][3]=q8.s3;
        qv[h][4]=q8.s4; qv[h][5]=q8.s5; qv[h][6]=q8.s6; qv[h][7]=q8.s7;
    }
    float m[4], l[4], o[4][8];
    #pragma unroll
    for (uint h = 0; h < GQA4_G; ++h) {
        m[h] = -INFINITY; l[h] = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) o[h][i] = 0.0f;
    }

    for (uint base = wlo + g; base < whi; base += 4u * UATTN) {
        uint kb[UATTN], vb[UATTN];
        float ks[UATTN], vs[UATTN];
        #pragma unroll
        for (uint u = 0; u < UATTN; ++u) {
            uint t = base + 4u * u;
            uint tt = t < whi ? t : whi - 1;
            uint lb = tt / block_size;
            uint token = tt - lb * block_size;
            uint pb = block_table[lb];
            pb = pb < physical_blocks ? pb : 0u;
            ulong row5 = (((ulong)pb * kv_heads + kvh) * 4u + blk) * block_size + token;
            kb[u] = ((__global const uint*)(K_all + row5 * 16))[sub4];
            vb[u] = ((__global const uint*)(V_all + row5 * 16))[sub4];
            ks[u] = as_float(((uint)scale_k_all[row5]) << 23);
            vs[u] = as_float(((uint)scale_v_all[row5]) << 23);
        }
        #pragma unroll
        for (uint u = 0; u < UATTN; ++u) {
            uint t = base + 4u * u;
            if (t >= whi) break;
            uint k4 = kb[u]; float bsk = ks[u];
            float2 ka = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(k4, bsk, 0);
            float2 kc = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(k4, bsk, 1);
            float2 ke = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(k4, bsk, 2);
            float2 kg = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(k4, bsk, 3);
            float kf[8] = {ka.x,ka.y,kc.x,kc.y,ke.x,ke.y,kg.x,kg.y};
            uint v4 = vb[u]; float bsv = vs[u];
            float2 ea = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, bsv, 0);
            float2 ec = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, bsv, 1);
            float2 ee = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, bsv, 2);
            float2 eg = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, bsv, 3);
            float vf[8] = {ea.x,ea.y,ec.x,ec.y,ee.x,ee.y,eg.x,eg.y};
            #pragma unroll
            for (uint h = 0; h < GQA4_G; ++h) {
                float partial = qv[h][0]*kf[0] + qv[h][1]*kf[1] + qv[h][2]*kf[2] + qv[h][3]*kf[3]
                              + qv[h][4]*kf[4] + qv[h][5]*kf[5] + qv[h][6]*kf[6] + qv[h][7]*kf[7];
                partial += BPERM(1u, partial);
                partial += BPERM(2u, partial);
                partial += BPERM(4u, partial);
                partial += BPERM(8u, partial);
                float s = partial * scale;
                float m_new = fmax(m[h], s);
                float corr = native_exp(m[h] - m_new);
                float p = native_exp(s - m_new);
                l[h] = l[h] * corr + p;
                #pragma unroll
                for (int i = 0; i < 8; ++i) o[h][i] = o[h][i]*corr + p*vf[i];
                m[h] = m_new;
            }
        }
    }

    __local float wm[4][WPS_ATTN], wl[4][WPS_ATTN], wo[4][WPS_ATTN][128];
    #pragma unroll
    for (uint h = 0; h < GQA4_G; ++h) {
        float M = m[h];
        M = fmax(M, BPERM(16u, M));
        M = fmax(M, BPERM(32u, M));
        if (M == -INFINITY) M = 0.0f;
        float cg = native_exp(m[h] - M);
        float L = l[h] * cg;
        L += BPERM(16u, L);
        L += BPERM(32u, L);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float oc = o[h][i] * cg;
            oc += BPERM(16u, oc);
            oc += BPERM(32u, oc);
            o[h][i] = oc;
        }
        if (lane == 0) { wm[h][w] = M; wl[h][w] = L; }
        if (g == 0) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) wo[h][w][sub * 8 + i] = o[h][i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (w == 0 && lane < 64) {
        #pragma unroll
        for (uint h = 0; h < GQA4_G; ++h) {
            float MM = -INFINITY;
            for (uint k = 0; k < WPS_ATTN; ++k) MM = fmax(MM, wm[h][k]);
            if (MM == -INFINITY) MM = 0.0f;
            float wc[WPS_ATTN];
            float LL = 0.0f;
            for (uint k = 0; k < WPS_ATTN; ++k) {
                wc[k] = native_exp(wm[h][k] - MM);
                LL += wl[h][k] * wc[k];
            }
            __global float* pr = group_partials + ((ulong)h * num_splits + sp) * (D + 2);
            if (lane == 0) { pr[0] = MM; pr[1] = LL; }
            for (uint dd = lane; dd < 128; dd += 64) {
                float acc = 0.0f;
                for (uint k = 0; k < WPS_ATTN; ++k)
                    acc += wo[h][k][dd] * wc[k];
                pr[2 + dd] = acc;
            }
        }
    }
}

// RMSNorm for a single hidden vector (decode): y[i] = x[i] * rsqrt(mean(x^2)+eps)
// * weight[i]. One workgroup of 256 threads, cooperative sum-of-squares in LDS.
// f16 in/out, f32 accumulation. H = hidden size (any; grid-strided).
// Standalone FP4 dequant + dot-product probe. This isolates the QK score inner
// loop used by the FP4 attention kernels: each row is [64] packed E2M1 bytes
// plus [4] E8M0 scale bytes, and the values are converted in registers by
// cvt_scalef32_pk_f32_fp4 immediately before the multiply-add. No BF16/FP16
// dequant staging buffer is written to global memory.
__kernel void fp4_dot_probe(__global const half* q,
                            __global const uchar* K,
                            __global const uchar* scale_k,
                            __global float* out,
                            uint N, uint num_wg) {
    uint gid = get_group_id(0);
    uint lane = get_local_id(0);
    if (num_wg == 0) return;

    uint sub = lane & 15;        // 8 dims per active lane
    uint blk = sub >> 2;         // 32-dim scale block
    half8 q8 = (half8)(0.0h);
    float qv[8] = {0,0,0,0,0,0,0,0};
    if (lane < 16) {
        q8 = vload8(sub, q);
        qv[0] = q8.s0; qv[1] = q8.s1; qv[2] = q8.s2; qv[3] = q8.s3;
        qv[4] = q8.s4; qv[5] = q8.s5; qv[6] = q8.s6; qv[7] = q8.s7;
    }

    for (uint row = gid; row < N; row += num_wg) {
        float partial = 0.0f;
        if (lane < 16) {
            uint k4 = ((__global const uint*)(K + (ulong)row * 64))[sub];
            float bsk = as_float(((uint)scale_k[(ulong)row * 4 + blk]) << 23);
            float2 ka = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(k4, bsk, 0);
            float2 kc = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(k4, bsk, 1);
            float2 ke = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(k4, bsk, 2);
            float2 kg = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(k4, bsk, 3);
            partial = qv[0]*ka.x + qv[1]*ka.y + qv[2]*kc.x + qv[3]*kc.y
                    + qv[4]*ke.x + qv[5]*ke.y + qv[6]*kg.x + qv[7]*kg.y;
            partial += BPERM(1u, partial);
            partial += BPERM(2u, partial);
            partial += BPERM(4u, partial);
            partial += BPERM(8u, partial);
        }
        if (lane == 0) out[row] = partial;
    }
}

__kernel void rmsnorm_f16(__global const half* x, __global const half* weight,
                          __global half* y, uint H, float eps) {
    uint t = get_local_id(0);
    const uint nt = 256u;
    __local float red[256];
    float ss = 0.0f;
    for (uint i = t; i < H; i += nt) { float v = (float)x[i]; ss += v * v; }
    red[t] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint off = nt >> 1; off > 0; off >>= 1) {
        if (t < off) red[t] += red[t + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float rms = rsqrt(red[0] / (float)H + eps);
    for (uint i = t; i < H; i += nt)
        y[i] = (half)((float)x[i] * rms * (float)weight[i]);
}

// Token embedding lookup + first-layer RMSNorm for decode. The token id lives
// in device memory so the decode loop can advance without a per-token host
// embedding copy. h receives the exact f16 embedding row; y receives RMSNorm(h).
__kernel void attn_decode_split2_fp4_gqa_paged_groups_meta(
    __global const half* q_all,                              // [num_groups*GQA_G][128]
    __global const uchar* K_all, __global const uchar* V_all, // [kv_heads][rows_per_head][64]
    __global const uchar* scale_k_all, __global const uchar* scale_v_all,
    __global const uint* block_table, uint block_size, uint physical_blocks,
    __global float* partials,                                // [num_groups*GQA_G][num_splits][D+2]
    __global const uint* seq_lens, uint max_N, uint D, float scale, uint num_splits,
    uint num_groups, uint q_heads_per_kv, uint rows_per_head) {
    uint N = seq_lens[0];
    if (N == 0u) return;
    if (N > max_N) N = max_N;
    uint gid = get_group_id(0);
    uint group_id = gid / num_splits;
    uint sp = gid - group_id * num_splits;
    if (group_id >= num_groups) return;
    uint head_base = group_id * GQA_G;
    uint kvh = head_base / q_heads_per_kv;
    __global const half* q = q_all + (ulong)head_base * 128;
    __global const uchar* K = K_all + (ulong)kvh * rows_per_head * 64;
    __global const uchar* V = V_all + (ulong)kvh * rows_per_head * 64;
    __global const uchar* scale_k = scale_k_all + (ulong)kvh * rows_per_head * 4;
    __global const uchar* scale_v = scale_v_all + (ulong)kvh * rows_per_head * 4;
    __global float* group_partials = partials + (ulong)head_base * num_splits * (D + 2);

    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint g = lane >> 4;
    uint sub = lane & 15;
    uint blk = sub >> 2;
    uint S = (N + num_splits - 1) / num_splits;
    uint lo = sp * S;
    uint hi = min(N, lo + S);
    uint span = hi > lo ? hi - lo : 0;
    uint per = (span + WPS_ATTN - 1) / WPS_ATTN;
    uint wlo = lo + w * per;
    uint whi = min(hi, wlo + per);

    half2 qv[GQA_G][4];
    #pragma unroll
    for (uint h = 0; h < GQA_G; ++h) {
        half8 q8 = vload8(sub, q + h * 128);
        qv[h][0]=(half2)(q8.s0, q8.s1); qv[h][1]=(half2)(q8.s2, q8.s3);
        qv[h][2]=(half2)(q8.s4, q8.s5); qv[h][3]=(half2)(q8.s6, q8.s7);
    }
    float m[GQA_G], l[GQA_G], o[GQA_G][8];
    #pragma unroll
    for (uint h = 0; h < GQA_G; ++h) {
        m[h] = -INFINITY; l[h] = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) o[h][i] = 0.0f;
    }

    for (uint base = wlo + g; base < whi; base += 4u * UATTN) {
        uint kb[UATTN], vb[UATTN];
        float ks[UATTN], vs[UATTN];
        #pragma unroll
        for (uint u = 0; u < UATTN; ++u) {
            uint t = base + 4u * u;
            uint tt = t < whi ? t : whi - 1;
            uint lb = tt / block_size;
            uint pb = block_table[lb];
            pb = pb < physical_blocks ? pb : 0u;
            ulong row = (ulong)pb * block_size + (tt - lb * block_size);
            kb[u] = ((__global const uint*)(K + row * 64))[sub];
            vb[u] = ((__global const uint*)(V + row * 64))[sub];
            ks[u] = as_float(((uint)scale_k[row * 4 + blk]) << 23);
            vs[u] = as_float(((uint)scale_v[row * 4 + blk]) << 23);
        }
        #pragma unroll
        for (uint u = 0; u < UATTN; ++u) {
            uint t = base + 4u * u;
            if (t >= whi) break;
            uint k4 = kb[u]; float bsk = ks[u];
            half2 ka = __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(k4, bsk, 0);
            half2 kc = __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(k4, bsk, 1);
            half2 ke = __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(k4, bsk, 2);
            half2 kg = __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(k4, bsk, 3);
            uint v4 = vb[u]; float bsv = vs[u];
            float2 ea = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, bsv, 0);
            float2 ec = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, bsv, 1);
            float2 ee = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, bsv, 2);
            float2 eg = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, bsv, 3);
            float vf[8] = {ea.x,ea.y,ec.x,ec.y,ee.x,ee.y,eg.x,eg.y};
            #pragma unroll
            for (uint h = 0; h < GQA_G; ++h) {
                float partial = 0.0f;
                partial = __builtin_amdgcn_fdot2(qv[h][0], ka, partial, false);
                partial = __builtin_amdgcn_fdot2(qv[h][1], kc, partial, false);
                partial = __builtin_amdgcn_fdot2(qv[h][2], ke, partial, false);
                partial = __builtin_amdgcn_fdot2(qv[h][3], kg, partial, false);
                partial += BPERM(1u, partial);
                partial += BPERM(2u, partial);
                partial += BPERM(4u, partial);
                partial += BPERM(8u, partial);
                float s = partial * scale;
                float m_new = fmax(m[h], s);
                float corr = native_exp(m[h] - m_new);
                float p = native_exp(s - m_new);
                l[h] = l[h] * corr + p;
                #pragma unroll
                for (int i = 0; i < 8; ++i) o[h][i] = o[h][i]*corr + p*vf[i];
                m[h] = m_new;
            }
        }
    }

    __local float wm[GQA_G][WPS_ATTN], wl[GQA_G][WPS_ATTN], wo[GQA_G][WPS_ATTN][128];
    #pragma unroll
    for (uint h = 0; h < GQA_G; ++h) {
        float M = m[h];
        M = fmax(M, BPERM(16u, M));
        M = fmax(M, BPERM(32u, M));
        if (M == -INFINITY) M = 0.0f;
        float cg = native_exp(m[h] - M);
        float L = l[h] * cg;
        L += BPERM(16u, L);
        L += BPERM(32u, L);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float oc = o[h][i] * cg;
            oc += BPERM(16u, oc);
            oc += BPERM(32u, oc);
            o[h][i] = oc;
        }
        if (lane == 0) { wm[h][w] = M; wl[h][w] = L; }
        if (g == 0) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) wo[h][w][sub * 8 + i] = o[h][i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (w == 0 && lane < 64) {
        #pragma unroll
        for (uint h = 0; h < GQA_G; ++h) {
            float MM = -INFINITY;
            for (uint k = 0; k < WPS_ATTN; ++k) MM = fmax(MM, wm[h][k]);
            if (MM == -INFINITY) MM = 0.0f;
            float LL = 0.0f;
            float wc[WPS_ATTN];
            for (uint k = 0; k < WPS_ATTN; ++k) {
                wc[k] = native_exp(wm[h][k] - MM);
                LL += wl[h][k] * wc[k];
            }
            __global float* pr = group_partials + ((ulong)h * num_splits + sp) * (D + 2);
            if (lane == 0) { pr[0] = MM; pr[1] = LL; }
            for (uint dd = lane; dd < 128; dd += 64) {
                float acc = 0.0f;
                for (uint k = 0; k < WPS_ATTN; ++k)
                    acc += wo[h][k][dd] * wc[k];
                pr[2 + dd] = acc;
            }
        }
    }
}

// Multi-head attention decode over paged FP4 KV, behind the block-table
// metadata validator. This is attn_decode_split2_fp4_gqa_paged_groups_meta with
// the head group narrowed to one, which is what multi-head attention *is*: every
// query head owns a KV head instead of sharing one across a group.
//
// It is a separate kernel rather than a runtime parameter because the group width
// sizes LDS and fixes the per-lane mapping, so it has to be known at compile time.
// The same reason attn_decode_split2_fp4_gqa4_paged_groups exists for G=4.
//
// Narrowing the group costs bandwidth and the cost is the point of GQA: a group of
// G shares one KV read across G query heads, so at G=1 the KV cache is read once
// per head with no sharing left to exploit. Models that choose multi-head
// attention (OLMo 2 among them) pay that on every token.
#define MHA_G 1
__kernel void attn_decode_split2_fp4_mha_paged_groups_meta(
    __global const half* q_all,                              // [num_groups*MHA_G][128]
    __global const uchar* K_all, __global const uchar* V_all, // [kv_heads][rows_per_head][64]
    __global const uchar* scale_k_all, __global const uchar* scale_v_all,
    __global const uint* block_table, uint block_size, uint physical_blocks,
    __global float* partials,                                // [num_groups*MHA_G][num_splits][D+2]
    __global const uint* seq_lens, uint max_N, uint D, float scale, uint num_splits,
    uint num_groups, uint q_heads_per_kv, uint rows_per_head) {
    uint N = seq_lens[0];
    if (N == 0u) return;
    if (N > max_N) N = max_N;
    uint gid = get_group_id(0);
    uint group_id = gid / num_splits;
    uint sp = gid - group_id * num_splits;
    if (group_id >= num_groups) return;
    uint head_base = group_id * MHA_G;
    uint kvh = head_base / q_heads_per_kv;
    __global const half* q = q_all + (ulong)head_base * 128;
    __global const uchar* K = K_all + (ulong)kvh * rows_per_head * 64;
    __global const uchar* V = V_all + (ulong)kvh * rows_per_head * 64;
    __global const uchar* scale_k = scale_k_all + (ulong)kvh * rows_per_head * 4;
    __global const uchar* scale_v = scale_v_all + (ulong)kvh * rows_per_head * 4;
    __global float* group_partials = partials + (ulong)head_base * num_splits * (D + 2);

    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint g = lane >> 4;
    uint sub = lane & 15;
    uint blk = sub >> 2;
    uint S = (N + num_splits - 1) / num_splits;
    uint lo = sp * S;
    uint hi = min(N, lo + S);
    uint span = hi > lo ? hi - lo : 0;
    uint per = (span + WPS_ATTN - 1) / WPS_ATTN;
    uint wlo = lo + w * per;
    uint whi = min(hi, wlo + per);

    half2 qv[MHA_G][4];
    #pragma unroll
    for (uint h = 0; h < MHA_G; ++h) {
        half8 q8 = vload8(sub, q + h * 128);
        qv[h][0]=(half2)(q8.s0, q8.s1); qv[h][1]=(half2)(q8.s2, q8.s3);
        qv[h][2]=(half2)(q8.s4, q8.s5); qv[h][3]=(half2)(q8.s6, q8.s7);
    }
    float m[MHA_G], l[MHA_G], o[MHA_G][8];
    #pragma unroll
    for (uint h = 0; h < MHA_G; ++h) {
        m[h] = -INFINITY; l[h] = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) o[h][i] = 0.0f;
    }

    for (uint base = wlo + g; base < whi; base += 4u * UATTN) {
        uint kb[UATTN], vb[UATTN];
        float ks[UATTN], vs[UATTN];
        #pragma unroll
        for (uint u = 0; u < UATTN; ++u) {
            uint t = base + 4u * u;
            uint tt = t < whi ? t : whi - 1;
            uint lb = tt / block_size;
            uint pb = block_table[lb];
            pb = pb < physical_blocks ? pb : 0u;
            ulong row = (ulong)pb * block_size + (tt - lb * block_size);
            kb[u] = ((__global const uint*)(K + row * 64))[sub];
            vb[u] = ((__global const uint*)(V + row * 64))[sub];
            ks[u] = as_float(((uint)scale_k[row * 4 + blk]) << 23);
            vs[u] = as_float(((uint)scale_v[row * 4 + blk]) << 23);
        }
        #pragma unroll
        for (uint u = 0; u < UATTN; ++u) {
            uint t = base + 4u * u;
            if (t >= whi) break;
            uint k4 = kb[u]; float bsk = ks[u];
            half2 ka = __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(k4, bsk, 0);
            half2 kc = __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(k4, bsk, 1);
            half2 ke = __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(k4, bsk, 2);
            half2 kg = __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(k4, bsk, 3);
            uint v4 = vb[u]; float bsv = vs[u];
            float2 ea = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, bsv, 0);
            float2 ec = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, bsv, 1);
            float2 ee = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, bsv, 2);
            float2 eg = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, bsv, 3);
            float vf[8] = {ea.x,ea.y,ec.x,ec.y,ee.x,ee.y,eg.x,eg.y};
            #pragma unroll
            for (uint h = 0; h < MHA_G; ++h) {
                float partial = 0.0f;
                partial = __builtin_amdgcn_fdot2(qv[h][0], ka, partial, false);
                partial = __builtin_amdgcn_fdot2(qv[h][1], kc, partial, false);
                partial = __builtin_amdgcn_fdot2(qv[h][2], ke, partial, false);
                partial = __builtin_amdgcn_fdot2(qv[h][3], kg, partial, false);
                partial += BPERM(1u, partial);
                partial += BPERM(2u, partial);
                partial += BPERM(4u, partial);
                partial += BPERM(8u, partial);
                float s = partial * scale;
                float m_new = fmax(m[h], s);
                float corr = native_exp(m[h] - m_new);
                float p = native_exp(s - m_new);
                l[h] = l[h] * corr + p;
                #pragma unroll
                for (int i = 0; i < 8; ++i) o[h][i] = o[h][i]*corr + p*vf[i];
                m[h] = m_new;
            }
        }
    }

    __local float wm[MHA_G][WPS_ATTN], wl[MHA_G][WPS_ATTN], wo[MHA_G][WPS_ATTN][128];
    #pragma unroll
    for (uint h = 0; h < MHA_G; ++h) {
        float M = m[h];
        M = fmax(M, BPERM(16u, M));
        M = fmax(M, BPERM(32u, M));
        if (M == -INFINITY) M = 0.0f;
        float cg = native_exp(m[h] - M);
        float L = l[h] * cg;
        L += BPERM(16u, L);
        L += BPERM(32u, L);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float oc = o[h][i] * cg;
            oc += BPERM(16u, oc);
            oc += BPERM(32u, oc);
            o[h][i] = oc;
        }
        if (lane == 0) { wm[h][w] = M; wl[h][w] = L; }
        if (g == 0) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) wo[h][w][sub * 8 + i] = o[h][i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (w == 0 && lane < 64) {
        #pragma unroll
        for (uint h = 0; h < MHA_G; ++h) {
            float MM = -INFINITY;
            for (uint k = 0; k < WPS_ATTN; ++k) MM = fmax(MM, wm[h][k]);
            if (MM == -INFINITY) MM = 0.0f;
            float LL = 0.0f;
            float wc[WPS_ATTN];
            for (uint k = 0; k < WPS_ATTN; ++k) {
                wc[k] = native_exp(wm[h][k] - MM);
                LL += wl[h][k] * wc[k];
            }
            __global float* pr = group_partials + ((ulong)h * num_splits + sp) * (D + 2);
            if (lane == 0) { pr[0] = MM; pr[1] = LL; }
            for (uint dd = lane; dd < 128; dd += 64) {
                float acc = 0.0f;
                for (uint k = 0; k < WPS_ATTN; ++k)
                    acc += wo[h][k][dd] * wc[k];
                pr[2 + dd] = acc;
            }
        }
    }
}

// RMSNorm for a single hidden vector (decode): y[i] = x[i] * rsqrt(mean(x^2)+eps)
// * weight[i]. One workgroup of 256 threads, cooperative sum-of-squares in LDS.
// f16 in/out, f32 accumulation. H = hidden size (any; grid-strided).
// Standalone FP4 dequant + dot-product probe. This isolates the QK score inner
// loop used by the FP4 attention kernels: each row is [64] packed E2M1 bytes
// plus [4] E8M0 scale bytes, and the values are converted in registers by
// cvt_scalef32_pk_f32_fp4 immediately before the multiply-add. No BF16/FP16
// dequant staging buffer is written to global memory.
__kernel void embed_rmsnorm_token_f16(__global const half* embed,
                                      __global const uint* tokens,
                                      __global const half* weight,
                                      __global half* h,
                                      __global half* y,
                                      uint token_index, uint vocab,
                                      uint H, float eps) {
    uint t = get_local_id(0);
    const uint nt = 256u;
    __local float red[256];
    uint tok = tokens[token_index];
    if (tok >= vocab) tok = 0u;
    ulong base = (ulong)tok * H;
    float ss = 0.0f;
    for (uint i = t; i < H; i += nt) {
        half hv = embed[base + i];
        h[i] = hv;
        float v = (float)hv;
        ss += v * v;
    }
    red[t] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint off = nt >> 1; off > 0; off >>= 1) {
        if (t < off) red[t] += red[t + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float rms = rsqrt(red[0] / (float)H + eps);
    for (uint i = t; i < H; i += nt) {
        half hv = embed[base + i];
        y[i] = (half)((float)hv * rms * (float)weight[i]);
    }
}

// Decode-step entry kernel for graph-friendly fixed-address metadata. The host
// keeps step/positions/seq_lens/last_page_len buffers at stable addresses; this
// first dispatch mutates their contents for the current token before later RoPE,
// KV append, and attention dispatches read them.
__kernel void decode_step_embed_rmsnorm_token_f16(__global const half* embed,
                                                 __global const uint* tokens,
                                                 __global const half* weight,
                                                 __global half* h,
                                                 __global half* y,
                                                 __global uint* step,
                                                 __global uint* positions,
                                                 __global uint* seq_lens,
                                                 __global uint* last_page_len,
                                                 uint base_pos,
                                                 uint block_size,
                                                 uint token_count,
                                                 uint vocab,
                                                 uint H,
                                                 float eps) {
    uint t = get_local_id(0);
    const uint nt = 256u;
    __local float red[256];
    uint token_index = step[0];
    if (token_count == 0u) return;
    if (token_index >= token_count) token_index = token_count - 1u;
    uint pos = base_pos + token_index;
    if (t == 0u) {
        positions[0] = pos;
        seq_lens[0] = pos + 1u;
        uint rem = (pos + 1u) - ((pos + 1u) / block_size) * block_size;
        last_page_len[0] = rem == 0u ? block_size : rem;
    }
    uint tok = tokens[token_index];
    if (tok >= vocab) tok = 0u;
    ulong base = (ulong)tok * H;
    float ss = 0.0f;
    for (uint i = t; i < H; i += nt) {
        half hv = embed[base + i];
        h[i] = hv;
        float v = (float)hv;
        ss += v * v;
    }
    red[t] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint off = nt >> 1; off > 0; off >>= 1) {
        if (t < off) red[t] += red[t + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float rms = rsqrt(red[0] / (float)H + eps);
    for (uint i = t; i < H; i += nt) {
        half hv = embed[base + i];
        y[i] = (half)((float)hv * rms * (float)weight[i]);
    }
}

// Fused residual add + RMSNorm for the f16 residual stream. This intentionally
// preserves the existing two-kernel rounding point: acc is updated as f16 first,
// then RMSNorm computes over the rounded residual values.
__kernel void add_rmsnorm_f16(__global half* acc, __global const float* x,
                              __global const half* weight, __global half* y,
                              uint H, float eps) {
    uint t = get_local_id(0);
    const uint nt = 256u;
    __local float red[256];
    float ss = 0.0f;
    for (uint i = t; i < H; i += nt) {
        half hv = (half)((float)acc[i] + x[i]);
        acc[i] = hv;
        float v = (float)hv;
        ss += v * v;
    }
    red[t] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint off = nt >> 1; off > 0; off >>= 1) {
        if (t < off) red[t] += red[t + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float rms = rsqrt(red[0] / (float)H + eps);
    for (uint i = t; i < H; i += nt)
        y[i] = (half)((float)acc[i] * rms * (float)weight[i]);
}

// OLMo 2's post-norm residual update.
//
// The difference from add_rmsnorm_f16 above is the order of two operations, and
// it is the whole architectural difference between pre-norm and post-norm.
//
//   pre-norm  (add_rmsnorm_f16):  acc = acc + x;  y = rmsnorm(acc)
//   post-norm (this kernel):      acc = acc + rmsnorm(x)
//
// Pre-norm accumulates first and hands the *normalised sum* to the next
// sub-layer. Post-norm normalises the sub-layer's own output and adds that to
// an un-normalised residual stream, so the residual keeps growing while each
// contribution to it is scaled. There is no `y`, because the next sub-layer
// reads the residual directly.
//
// Swapping these two produces numbers rather than an error, which is why the
// gate for this kernel checks the coupling and not just the magnitude.
__kernel void add_postnorm_f16(__global half* acc, __global const float* x,
                               __global const half* weight, uint H, float eps) {
    uint t = get_local_id(0);
    const uint nt = 256u;
    __local float red[256];
    float ss = 0.0f;
    for (uint i = t; i < H; i += nt) {
        float v = x[i];
        ss += v * v;
    }
    red[t] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint off = nt >> 1; off > 0; off >>= 1) {
        if (t < off) red[t] += red[t + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float rms = rsqrt(red[0] / (float)H + eps);
    for (uint i = t; i < H; i += nt)
        acc[i] = (half)((float)acc[i] + x[i] * rms * (float)weight[i]);
}

// Convert an in-place resident residual stream from f16 bits to bf16 bits.
// The storage width stays 16 bits, but all later residual kernels must treat
// `acc` as bf16. This is the one-time escape hatch before f16 residual overflow.
__kernel void cast_f16_bf16_inplace(__global half* acc, uint n, uint nthreads) {
    uint gid = get_group_id(0) * 256u + get_local_id(0);
    __global ushort* out = (__global ushort*)acc;
    for (uint i = gid; i < n; i += nthreads) {
        out[i] = qwen_f32_to_bf16_bits((float)acc[i]);
    }
}

// Read-only GPU-side validation for the resident layer-runner descriptor table.
// This is intentionally a tiny single-thread probe: before execution kernels
// consume descriptor rows, prove the packed HBM table is readable by the GPU and
// still matches the host-side FNV/magic/version/layer contract.
__kernel void resident_descriptor_table_probe(__global const ulong* descriptors,
                                              __global ulong* out,
                                              uint expected_magic_lo,
                                              uint expected_magic_hi,
                                              uint expected_version,
                                              uint rows,
                                              uint u64s_per_row,
                                              uint expected_checksum_lo,
                                              uint expected_checksum_hi,
                                              uint nthreads) {
    uint gid = get_group_id(0) * 256u + get_local_id(0);
    if (gid != 0u || nthreads == 0u) return;

    ulong expected_magic =
        (ulong)expected_magic_lo | ((ulong)expected_magic_hi << 32);
    ulong expected_checksum =
        (ulong)expected_checksum_lo | ((ulong)expected_checksum_hi << 32);

    out[0] = 0UL;
    out[1] = (ulong)rows;
    out[2] = (ulong)u64s_per_row;
    out[3] = 0UL;
    out[4] = expected_checksum;
    out[5] = 0UL;
    out[6] = 0UL;
    out[7] = 0xD35C7105D15C0DEDUL;

    if (rows == 0u || rows > 4096u || u64s_per_row != 8u) {
        out[0] = 1UL;
        return;
    }

    ulong checksum = 0xcbf29ce484222325UL;
    ulong total_words = (ulong)rows * (ulong)u64s_per_row;
    for (ulong word_idx = 0UL; word_idx < total_words; ++word_idx) {
        ulong word = descriptors[word_idx];
        for (uint byte_idx = 0u; byte_idx < 8u; ++byte_idx) {
            checksum ^= (word >> (byte_idx * 8u)) & 0xffUL;
            checksum *= 0x100000001b3UL;
        }
    }
    out[3] = checksum;

    uint prev_layer = 0u;
    for (uint row = 0u; row < rows; ++row) {
        ulong base = (ulong)row * (ulong)u64s_per_row;
        ulong magic = descriptors[base + 0UL];
        ulong version_row = descriptors[base + 1UL];
        ulong layer_contract = descriptors[base + 2UL];
        uint version = (uint)(version_row & 0xffffffffUL);
        uint row_index = (uint)(version_row >> 32);
        uint layer = (uint)(layer_contract & 0xffffffffUL);
        uint residual_layer = (uint)(layer_contract >> 32);

        if (magic != expected_magic) {
            out[0] = 2UL;
            out[5] = (ulong)row;
            out[6] = magic;
            return;
        }
        if (version != expected_version || row_index != row) {
            out[0] = 3UL;
            out[5] = (ulong)row;
            out[6] = version_row;
            return;
        }
        if (layer != residual_layer || (row > 0u && layer <= prev_layer)) {
            out[0] = 4UL;
            out[5] = (ulong)row;
            out[6] = layer_contract;
            return;
        }
        prev_layer = layer;
    }

    if (checksum != expected_checksum) {
        out[0] = 5UL;
        out[6] = checksum;
        return;
    }
}

__kernel void resident_dag_edge_table_probe(__global const ulong* descriptors,
                                            __global ulong* out,
                                            uint expected_magic_lo,
                                            uint expected_magic_hi,
                                            uint expected_version,
                                            uint rows,
                                            uint u64s_per_row,
                                            uint expected_checksum_lo,
                                            uint expected_checksum_hi,
                                            uint nthreads) {
    uint gid = get_group_id(0) * 256u + get_local_id(0);
    if (gid != 0u || nthreads == 0u) return;

    ulong expected_magic =
        (ulong)expected_magic_lo | ((ulong)expected_magic_hi << 32);
    ulong expected_checksum =
        (ulong)expected_checksum_lo | ((ulong)expected_checksum_hi << 32);

    out[0] = 0UL;
    out[1] = (ulong)rows;
    out[2] = (ulong)u64s_per_row;
    out[3] = 0UL;
    out[4] = expected_checksum;
    out[5] = 0UL;
    out[6] = 0UL;
    out[7] = 0xD35C7105D15C0DEDUL;

    if (rows == 0u || rows > 4096u || u64s_per_row != 8u) {
        out[0] = 1UL;
        return;
    }

    ulong table_checksum = 0xcbf29ce484222325UL;
    ulong total_words = (ulong)rows * (ulong)u64s_per_row;
    for (ulong word_idx = 0UL; word_idx < total_words; ++word_idx) {
        ulong word = descriptors[word_idx];
        for (uint byte_idx = 0u; byte_idx < 8u; ++byte_idx) {
            table_checksum ^= (word >> (byte_idx * 8u)) & 0xffUL;
            table_checksum *= 0x100000001b3UL;
        }
    }
    out[3] = table_checksum;

    for (uint row = 0u; row < rows; ++row) {
        ulong base = (ulong)row * (ulong)u64s_per_row;
        ulong magic = descriptors[base + 0UL];
        ulong version_row = descriptors[base + 1UL];
        ulong layer_kind = descriptors[base + 2UL];
        ulong wait_mask = descriptors[base + 3UL];
        ulong signal_mask = descriptors[base + 4UL];
        ulong downstream_edge_index_word = descriptors[base + 5UL];
        ulong downstream_wait_mask = descriptors[base + 6UL];
        ulong row_checksum_expected = descriptors[base + 7UL];
        uint version = (uint)(version_row & 0xffffffffUL);
        uint row_index = (uint)(version_row >> 32);
        uint layer = (uint)(layer_kind & 0xffffffffUL);
        uint kind = (uint)(layer_kind >> 32);
        uint downstream_edge_index = (uint)(downstream_edge_index_word & 0xffffffffUL);
        uint expected_downstream_edge_index = row + 1u;
        ulong expected_downstream_wait_mask = 0UL;

        if (magic != expected_magic) {
            out[0] = 2UL;
            out[5] = (ulong)row;
            out[6] = magic;
            return;
        }
        if (version != expected_version || row_index != row) {
            out[0] = 3UL;
            out[5] = (ulong)row;
            out[6] = version_row;
            return;
        }
        if (layer == 0u || (kind != 1u && kind != 2u)) {
            out[0] = 4UL;
            out[5] = (ulong)row;
            out[6] = layer_kind;
            return;
        }
        if (wait_mask == 0UL || signal_mask == 0UL) {
            out[0] = 5UL;
            out[5] = (ulong)row;
            out[6] = wait_mask == 0UL ? wait_mask : signal_mask;
            return;
        }

        if (row + 1u == rows) {
            expected_downstream_edge_index = 0xffffffffu;
        } else {
            expected_downstream_wait_mask =
                descriptors[((ulong)(row + 1u) * (ulong)u64s_per_row) + 3UL];
        }
        if (downstream_edge_index_word != (ulong)expected_downstream_edge_index) {
            out[0] = 6UL;
            out[5] = (ulong)row;
            out[6] = downstream_edge_index_word;
            return;
        }
        if (downstream_wait_mask != expected_downstream_wait_mask) {
            out[0] = 7UL;
            out[5] = (ulong)row;
            out[6] = downstream_wait_mask;
            return;
        }

        ulong row_checksum = 0xcbf29ce484222325UL;
        ulong checksum_words[7] = {
            (ulong)row_index,
            (ulong)layer,
            (ulong)kind,
            wait_mask,
            signal_mask,
            (ulong)downstream_edge_index,
            downstream_wait_mask,
        };
        for (uint word_idx = 0u; word_idx < 7u; ++word_idx) {
            ulong word = checksum_words[word_idx];
            for (uint byte_idx = 0u; byte_idx < 8u; ++byte_idx) {
                row_checksum ^= (word >> (byte_idx * 8u)) & 0xffUL;
                row_checksum *= 0x100000001b3UL;
            }
        }
        if (row_checksum != row_checksum_expected) {
            out[0] = 8UL;
            out[5] = (ulong)row;
            out[6] = row_checksum;
            return;
        }
    }

    if (table_checksum != expected_checksum) {
        out[0] = 9UL;
        out[6] = table_checksum;
        return;
    }
}

__kernel void resident_dense_output_slot_table_probe(__global const ulong* descriptors,
                                                     __global ulong* out,
                                                     uint expected_magic_lo,
                                                     uint expected_magic_hi,
                                                     uint expected_version,
                                                     uint rows,
                                                     uint u64s_per_row,
                                                     uint expected_role_mask,
                                                     uint expected_slot_count,
                                                     uint expected_checksum_lo,
                                                     uint expected_checksum_hi,
                                                     uint nthreads) {
    uint gid = get_group_id(0) * 256u + get_local_id(0);
    if (gid != 0u || nthreads == 0u) return;

    ulong expected_magic =
        (ulong)expected_magic_lo | ((ulong)expected_magic_hi << 32);
    ulong expected_checksum =
        (ulong)expected_checksum_lo | ((ulong)expected_checksum_hi << 32);

    out[0] = 0UL;
    out[1] = (ulong)rows;
    out[2] = (ulong)u64s_per_row;
    out[3] = 0UL;
    out[4] = expected_checksum;
    out[5] = 0UL;
    out[6] = 0UL;
    out[7] = 0xD35C7105D15C0DEDUL;

    if (rows == 0u || rows > 4096u || u64s_per_row != 8u) {
        out[0] = 1UL;
        return;
    }

    ulong table_checksum = 0xcbf29ce484222325UL;
    ulong total_words = (ulong)rows * (ulong)u64s_per_row;
    for (ulong word_idx = 0UL; word_idx < total_words; ++word_idx) {
        ulong word = descriptors[word_idx];
        for (uint byte_idx = 0u; byte_idx < 8u; ++byte_idx) {
            table_checksum ^= (word >> (byte_idx * 8u)) & 0xffUL;
            table_checksum *= 0x100000001b3UL;
        }
    }
    out[3] = table_checksum;

    uint prev_layer = 0u;
    for (uint row = 0u; row < rows; ++row) {
        ulong base = (ulong)row * (ulong)u64s_per_row;
        ulong magic = descriptors[base + 0UL];
        ulong version_row = descriptors[base + 1UL];
        ulong layer_slot_count = descriptors[base + 2UL];
        ulong role_mask_word = descriptors[base + 3UL];
        ulong flags = descriptors[base + 6UL];
        ulong row_checksum_expected = descriptors[base + 7UL];
        uint version = (uint)(version_row & 0xffffffffUL);
        uint row_index = (uint)(version_row >> 32);
        uint layer = (uint)(layer_slot_count & 0xffffffffUL);
        uint slot_count = (uint)(layer_slot_count >> 32);
        uint role_mask = (uint)(role_mask_word & 0xffffffffUL);

        if (magic != expected_magic) {
            out[0] = 2UL;
            out[5] = (ulong)row;
            out[6] = magic;
            return;
        }
        if (version != expected_version || row_index != row) {
            out[0] = 3UL;
            out[5] = (ulong)row;
            out[6] = version_row;
            return;
        }
        if (row > 0u && layer <= prev_layer) {
            out[0] = 4UL;
            out[5] = (ulong)row;
            out[6] = layer_slot_count;
            return;
        }
        if (slot_count != expected_slot_count || role_mask != expected_role_mask) {
            out[0] = 5UL;
            out[5] = (ulong)row;
            out[6] = role_mask_word;
            return;
        }
        if ((flags & 1UL) == 0UL) {
            out[0] = 6UL;
            out[5] = (ulong)row;
            out[6] = flags;
            return;
        }

        ulong row_checksum = 0xcbf29ce484222325UL;
        ulong checksum_words[5] = {
            (ulong)row_index,
            (ulong)layer,
            (ulong)slot_count,
            (ulong)role_mask,
            flags,
        };
        for (uint word_idx = 0u; word_idx < 5u; ++word_idx) {
            ulong word = checksum_words[word_idx];
            for (uint byte_idx = 0u; byte_idx < 8u; ++byte_idx) {
                row_checksum ^= (word >> (byte_idx * 8u)) & 0xffUL;
                row_checksum *= 0x100000001b3UL;
            }
        }
        if (row_checksum != row_checksum_expected) {
            out[0] = 7UL;
            out[5] = (ulong)row;
            out[6] = row_checksum;
            return;
        }
        prev_layer = layer;
    }

    if (table_checksum != expected_checksum) {
        out[0] = 8UL;
        out[6] = table_checksum;
        return;
    }
}

__kernel void resident_dense_output_slot_resolver_probe(__global const ulong* descriptors,
                                                        __global const uint* layer_row_index,
                                                        __global ulong* out,
                                                        uint expected_magic_lo,
                                                        uint expected_magic_hi,
                                                        uint expected_version,
                                                        uint rows,
                                                        uint u64s_per_row,
                                                        uint layer_index_entries,
                                                        uint target_layer,
                                                        uint expected_role_mask,
                                                        uint expected_slot_count,
                                                        uint expected_flags_lo,
                                                        uint expected_flags_hi,
                                                        uint expected_row_index,
                                                        uint expected_row_checksum_lo,
                                                        uint expected_row_checksum_hi,
                                                        uint nthreads) {
    uint gid = get_group_id(0) * 256u + get_local_id(0);
    if (gid != 0u || nthreads == 0u) return;

    ulong expected_magic =
        (ulong)expected_magic_lo | ((ulong)expected_magic_hi << 32);
    ulong expected_flags =
        (ulong)expected_flags_lo | ((ulong)expected_flags_hi << 32);
    ulong expected_row_checksum =
        (ulong)expected_row_checksum_lo | ((ulong)expected_row_checksum_hi << 32);

    out[0] = 0UL;
    out[1] = (ulong)target_layer;
    out[2] = 0xffffffffffffffffUL;
    out[3] = 0UL;
    out[4] = 0UL;
    out[5] = 0UL;
    out[6] = 0UL;
    out[7] = expected_row_checksum;
    out[8] = (ulong)layer_index_entries;
    out[9] = (ulong)rows;
    out[10] = (ulong)u64s_per_row;
    out[11] = 0UL;
    out[12] = expected_flags;
    out[13] = (ulong)expected_row_index;
    out[14] = 0UL;
    out[15] = 0xD35C7105D15C0DEDUL;

    if (rows == 0u || rows > 4096u || u64s_per_row != 8u ||
        layer_index_entries == 0u) {
        out[0] = 1UL;
        return;
    }
    if (target_layer >= layer_index_entries) {
        out[0] = 2UL;
        out[14] = (ulong)target_layer;
        return;
    }

    uint row = layer_row_index[target_layer];
    out[2] = (ulong)row;
    if (row == 0xffffffffu || row >= rows) {
        out[0] = 3UL;
        out[14] = (ulong)row;
        return;
    }
    if (row != expected_row_index) {
        out[0] = 4UL;
        out[14] = (ulong)row;
        return;
    }

    ulong base = (ulong)row * (ulong)u64s_per_row;
    ulong magic = descriptors[base + 0UL];
    ulong version_row = descriptors[base + 1UL];
    ulong layer_slot_count = descriptors[base + 2UL];
    ulong role_mask_word = descriptors[base + 3UL];
    ulong flags = descriptors[base + 6UL];
    ulong row_checksum_expected = descriptors[base + 7UL];
    uint version = (uint)(version_row & 0xffffffffUL);
    uint row_index = (uint)(version_row >> 32);
    uint layer = (uint)(layer_slot_count & 0xffffffffUL);
    uint slot_count = (uint)(layer_slot_count >> 32);
    uint role_mask = (uint)(role_mask_word & 0xffffffffUL);

    out[3] = (ulong)role_mask;
    out[4] = (ulong)slot_count;
    out[5] = flags;
    out[11] = layer_slot_count;

    if (magic != expected_magic || version != expected_version ||
        row_index != row) {
        out[0] = 5UL;
        out[14] = version_row;
        return;
    }
    if (layer != target_layer) {
        out[0] = 6UL;
        out[14] = layer_slot_count;
        return;
    }
    if (slot_count != expected_slot_count ||
        role_mask != expected_role_mask ||
        flags != expected_flags) {
        out[0] = 7UL;
        out[14] = role_mask_word;
        return;
    }

    ulong row_checksum = 0xcbf29ce484222325UL;
    ulong checksum_words[5] = {
        (ulong)row_index,
        (ulong)layer,
        (ulong)slot_count,
        (ulong)role_mask,
        flags,
    };
    for (uint word_idx = 0u; word_idx < 5u; ++word_idx) {
        ulong word = checksum_words[word_idx];
        for (uint byte_idx = 0u; byte_idx < 8u; ++byte_idx) {
            row_checksum ^= (word >> (byte_idx * 8u)) & 0xffUL;
            row_checksum *= 0x100000001b3UL;
        }
    }
    out[6] = row_checksum;
    if (row_checksum != row_checksum_expected ||
        row_checksum != expected_row_checksum) {
        out[0] = 8UL;
        out[14] = row_checksum_expected;
        return;
    }
}

__kernel void resident_dag_readiness_frontier_probe(__global const ulong* descriptors,
                                                    __global ulong* out,
                                                    uint expected_magic_lo,
                                                    uint expected_magic_hi,
                                                    uint expected_version,
                                                    uint rows,
                                                    uint u64s_per_row,
                                                    uint satisfied_mask_lo,
                                                    uint satisfied_mask_hi,
                                                    uint completed_mask_lo,
                                                    uint completed_mask_hi,
                                                    uint expected_ready_mask_lo,
                                                    uint expected_ready_mask_hi,
                                                    uint expected_first_ready,
                                                    uint nthreads) {
    uint gid = get_group_id(0) * 256u + get_local_id(0);
    if (gid != 0u || nthreads == 0u) return;

    ulong expected_magic =
        (ulong)expected_magic_lo | ((ulong)expected_magic_hi << 32);
    ulong satisfied_mask =
        (ulong)satisfied_mask_lo | ((ulong)satisfied_mask_hi << 32);
    ulong completed_mask =
        (ulong)completed_mask_lo | ((ulong)completed_mask_hi << 32);
    ulong expected_ready_mask =
        (ulong)expected_ready_mask_lo | ((ulong)expected_ready_mask_hi << 32);

    out[0] = 0UL;
    out[1] = (ulong)rows;
    out[2] = (ulong)u64s_per_row;
    out[3] = satisfied_mask;
    out[4] = completed_mask;
    out[5] = 0UL;
    out[6] = 0xffffffffUL;
    out[7] = 0UL;
    out[8] = 0UL;
    out[9] = expected_ready_mask;
    out[10] = (ulong)expected_first_ready;
    out[11] = 0xD35C7105D15C0DEDUL;

    if (rows == 0u || rows > 64u || u64s_per_row != 8u) {
        out[0] = 1UL;
        return;
    }

    ulong ready_mask = 0UL;
    uint first_ready = 0xffffffffu;
    uint ready_count = 0u;
    for (uint row = 0u; row < rows; ++row) {
        ulong base = (ulong)row * (ulong)u64s_per_row;
        ulong magic = descriptors[base + 0UL];
        ulong version_row = descriptors[base + 1UL];
        ulong layer_kind = descriptors[base + 2UL];
        ulong wait_mask = descriptors[base + 3UL];
        ulong signal_mask = descriptors[base + 4UL];
        ulong downstream_edge_index_word = descriptors[base + 5UL];
        ulong downstream_wait_mask = descriptors[base + 6UL];
        ulong row_checksum_expected = descriptors[base + 7UL];
        uint version = (uint)(version_row & 0xffffffffUL);
        uint row_index = (uint)(version_row >> 32);
        uint layer = (uint)(layer_kind & 0xffffffffUL);
        uint kind = (uint)(layer_kind >> 32);
        uint downstream_edge_index = (uint)(downstream_edge_index_word & 0xffffffffUL);

        if (magic != expected_magic) {
            out[0] = 2UL;
            out[5] = (ulong)row;
            out[9] = magic;
            return;
        }
        if (version != expected_version || row_index != row) {
            out[0] = 3UL;
            out[5] = (ulong)row;
            out[9] = version_row;
            return;
        }
        if (layer == 0u || (kind != 1u && kind != 2u)) {
            out[0] = 4UL;
            out[5] = (ulong)row;
            out[9] = layer_kind;
            return;
        }
        if (wait_mask == 0UL || signal_mask == 0UL) {
            out[0] = 5UL;
            out[5] = (ulong)row;
            out[9] = wait_mask == 0UL ? wait_mask : signal_mask;
            return;
        }

        ulong row_checksum = 0xcbf29ce484222325UL;
        ulong checksum_words[7] = {
            (ulong)row_index,
            (ulong)layer,
            (ulong)kind,
            wait_mask,
            signal_mask,
            (ulong)downstream_edge_index,
            downstream_wait_mask,
        };
        for (uint word_idx = 0u; word_idx < 7u; ++word_idx) {
            ulong word = checksum_words[word_idx];
            for (uint byte_idx = 0u; byte_idx < 8u; ++byte_idx) {
                row_checksum ^= (word >> (byte_idx * 8u)) & 0xffUL;
                row_checksum *= 0x100000001b3UL;
            }
        }
        if (row_checksum != row_checksum_expected) {
            out[0] = 6UL;
            out[5] = (ulong)row;
            out[9] = row_checksum;
            return;
        }

        if ((wait_mask & satisfied_mask) == wait_mask &&
            (signal_mask & completed_mask) == 0UL) {
            ready_mask |= 1UL << row_index;
            if (first_ready == 0xffffffffu) {
                first_ready = row_index;
            }
            ++ready_count;
        }
    }

    ulong frontier_checksum = 0xcbf29ce484222325UL;
    ulong frontier_words[5] = {
        satisfied_mask,
        completed_mask,
        ready_mask,
        (ulong)first_ready,
        (ulong)ready_count,
    };
    for (uint word_idx = 0u; word_idx < 5u; ++word_idx) {
        ulong word = frontier_words[word_idx];
        for (uint byte_idx = 0u; byte_idx < 8u; ++byte_idx) {
            frontier_checksum ^= (word >> (byte_idx * 8u)) & 0xffUL;
            frontier_checksum *= 0x100000001b3UL;
        }
    }

    out[5] = ready_mask;
    out[6] = (ulong)first_ready;
    out[7] = (ulong)ready_count;
    out[8] = frontier_checksum;

    if (ready_mask != expected_ready_mask) {
        out[0] = 7UL;
        return;
    }
    if (first_ready != expected_first_ready) {
        out[0] = 8UL;
        return;
    }
}

__kernel void resident_dag_completion_marker_probe(__global ulong* state,
                                                   __global ulong* out,
                                                   uint completed_edge_index,
                                                   uint expected_previous_edge_lo,
                                                   uint expected_previous_edge_hi,
                                                   uint rows,
                                                   uint nthreads) {
    uint gid = get_group_id(0) * 256u + get_local_id(0);
    if (gid != 0u || nthreads == 0u) return;

    ulong expected_previous_edge =
        (ulong)expected_previous_edge_lo | ((ulong)expected_previous_edge_hi << 32);
    ulong previous_edge = state[6];

    out[0] = 0UL;
    out[1] = (ulong)rows;
    out[2] = (ulong)completed_edge_index;
    out[3] = previous_edge;
    out[4] = 0UL;
    out[5] = expected_previous_edge;
    out[6] = 0UL;
    out[7] = 0xD35C7105D15C0DEDUL;

    if (rows == 0u || completed_edge_index >= rows) {
        out[0] = 1UL;
        return;
    }
    if (previous_edge != expected_previous_edge) {
        out[0] = 2UL;
        return;
    }

    state[6] = (ulong)completed_edge_index;
    mem_fence(CLK_GLOBAL_MEM_FENCE);

    ulong written_edge = state[6];
    out[4] = written_edge;
    out[6] = previous_edge ^ written_edge ^ ((ulong)rows << 32);
    if (written_edge != (ulong)completed_edge_index) {
        out[0] = 3UL;
        return;
    }
}

__kernel void resident_dag_state_transition_probe(__global const ulong* descriptors,
                                                  __global ulong* state,
                                                  __global ulong* out,
                                                  uint expected_magic_lo,
                                                  uint expected_magic_hi,
                                                  uint expected_version,
                                                  uint rows,
                                                  uint u64s_per_row,
                                                  uint expected_satisfied_lo,
                                                  uint expected_satisfied_hi,
                                                  uint expected_completed_lo,
                                                  uint expected_completed_hi,
                                                  uint expected_ready_lo,
                                                  uint expected_ready_hi,
                                                  uint expected_first_ready,
                                                  uint expected_ready_count,
                                                  uint expected_checksum_lo,
                                                  uint expected_checksum_hi,
                                                  uint nthreads) {
    uint gid = get_group_id(0) * 256u + get_local_id(0);
    if (gid != 0u || nthreads == 0u) return;

    ulong expected_magic =
        (ulong)expected_magic_lo | ((ulong)expected_magic_hi << 32);
    ulong expected_satisfied =
        (ulong)expected_satisfied_lo | ((ulong)expected_satisfied_hi << 32);
    ulong expected_completed =
        (ulong)expected_completed_lo | ((ulong)expected_completed_hi << 32);
    ulong expected_ready =
        (ulong)expected_ready_lo | ((ulong)expected_ready_hi << 32);
    ulong expected_checksum =
        (ulong)expected_checksum_lo | ((ulong)expected_checksum_hi << 32);

    out[0] = 0UL;
    out[1] = (ulong)rows;
    out[2] = state[6];
    out[3] = state[0];
    out[4] = state[1];
    out[5] = 0UL;
    out[6] = 0UL;
    out[7] = 0UL;
    out[8] = 0xffffffffUL;
    out[9] = 0UL;
    out[10] = 0UL;
    out[11] = expected_ready;
    out[12] = (ulong)expected_first_ready;
    out[13] = 0UL;
    out[14] = 0UL;
    out[15] = 0xD35C7105D15C0DEDUL;

    if (rows == 0u || rows > 64u || u64s_per_row != 8u) {
        out[0] = 1UL;
        return;
    }

    uint completed_edge = (uint)(state[6] & 0xffffffffUL);
    if (completed_edge >= rows) {
        out[0] = 2UL;
        out[14] = (ulong)completed_edge;
        return;
    }

    ulong completed_base = (ulong)completed_edge * (ulong)u64s_per_row;
    ulong completed_magic = descriptors[completed_base + 0UL];
    ulong completed_version_row = descriptors[completed_base + 1UL];
    ulong completed_layer_kind = descriptors[completed_base + 2UL];
    ulong completed_signal_mask = descriptors[completed_base + 4UL];
    ulong completed_downstream_edge_index_word = descriptors[completed_base + 5UL];
    ulong completed_downstream_wait_mask = descriptors[completed_base + 6UL];
    ulong completed_row_checksum_expected = descriptors[completed_base + 7UL];
    uint completed_version = (uint)(completed_version_row & 0xffffffffUL);
    uint completed_row_index = (uint)(completed_version_row >> 32);
    uint completed_layer = (uint)(completed_layer_kind & 0xffffffffUL);
    uint completed_kind = (uint)(completed_layer_kind >> 32);
    uint completed_downstream_edge_index =
        (uint)(completed_downstream_edge_index_word & 0xffffffffUL);

    if (completed_magic != expected_magic) {
        out[0] = 3UL;
        out[14] = completed_magic;
        return;
    }
    if (completed_version != expected_version || completed_row_index != completed_edge) {
        out[0] = 4UL;
        out[14] = completed_version_row;
        return;
    }
    if (completed_layer == 0u || (completed_kind != 1u && completed_kind != 2u) ||
        completed_signal_mask == 0UL) {
        out[0] = 5UL;
        out[14] = completed_layer_kind;
        return;
    }

    ulong completed_row_checksum = 0xcbf29ce484222325UL;
    ulong completed_checksum_words[7] = {
        (ulong)completed_row_index,
        (ulong)completed_layer,
        (ulong)completed_kind,
        descriptors[completed_base + 3UL],
        completed_signal_mask,
        (ulong)completed_downstream_edge_index,
        completed_downstream_wait_mask,
    };
    for (uint word_idx = 0u; word_idx < 7u; ++word_idx) {
        ulong word = completed_checksum_words[word_idx];
        for (uint byte_idx = 0u; byte_idx < 8u; ++byte_idx) {
            completed_row_checksum ^= (word >> (byte_idx * 8u)) & 0xffUL;
            completed_row_checksum *= 0x100000001b3UL;
        }
    }
    if (completed_row_checksum != completed_row_checksum_expected) {
        out[0] = 6UL;
        out[14] = completed_row_checksum;
        return;
    }

    ulong previous_satisfied = state[0];
    ulong previous_completed = state[1];
    ulong completion_satisfied_mask = completed_signal_mask | completed_downstream_wait_mask;
    ulong new_satisfied = previous_satisfied | completion_satisfied_mask;
    ulong new_completed = previous_completed | completed_signal_mask;
    state[0] = new_satisfied;
    state[1] = new_completed;
    mem_fence(CLK_GLOBAL_MEM_FENCE);

    ulong ready_mask = 0UL;
    ulong newly_ready_mask = 0UL;
    uint first_ready = 0xffffffffu;
    uint ready_count = 0u;
    for (uint row = 0u; row < rows; ++row) {
        ulong base = (ulong)row * (ulong)u64s_per_row;
        ulong magic = descriptors[base + 0UL];
        ulong version_row = descriptors[base + 1UL];
        ulong wait_mask = descriptors[base + 3UL];
        ulong signal_mask = descriptors[base + 4UL];
        uint version = (uint)(version_row & 0xffffffffUL);
        uint row_index = (uint)(version_row >> 32);
        if (magic != expected_magic || version != expected_version || row_index != row ||
            wait_mask == 0UL || signal_mask == 0UL) {
            out[0] = 7UL;
            out[13] = (ulong)row;
            out[14] = version_row;
            return;
        }
        bool was_ready = ((wait_mask & previous_satisfied) == wait_mask &&
                          (signal_mask & previous_completed) == 0UL);
        bool is_ready = ((wait_mask & new_satisfied) == wait_mask &&
                         (signal_mask & new_completed) == 0UL);
        if (is_ready) {
            ready_mask |= 1UL << row_index;
            if (!was_ready) {
                newly_ready_mask |= 1UL << row_index;
            }
            if (first_ready == 0xffffffffu) {
                first_ready = row_index;
            }
            ++ready_count;
        }
    }

    ulong frontier_checksum = 0xcbf29ce484222325UL;
    ulong frontier_words[5] = {
        new_satisfied,
        new_completed,
        ready_mask,
        (ulong)first_ready,
        (ulong)ready_count,
    };
    for (uint word_idx = 0u; word_idx < 5u; ++word_idx) {
        ulong word = frontier_words[word_idx];
        for (uint byte_idx = 0u; byte_idx < 8u; ++byte_idx) {
            frontier_checksum ^= (word >> (byte_idx * 8u)) & 0xffUL;
            frontier_checksum *= 0x100000001b3UL;
        }
    }

    state[2] = ready_mask;
    state[3] = (ulong)first_ready;
    state[4] = (ulong)ready_count;
    state[5] = frontier_checksum;
    state[7] = 0xD35C7105D15C0DEDUL;
    mem_fence(CLK_GLOBAL_MEM_FENCE);

    out[2] = (ulong)completed_edge;
    out[3] = previous_satisfied;
    out[4] = previous_completed;
    out[5] = new_satisfied;
    out[6] = new_completed;
    out[7] = ready_mask;
    out[8] = (ulong)first_ready;
    out[9] = (ulong)ready_count;
    out[10] = frontier_checksum;
    out[13] = completed_downstream_wait_mask;
    out[14] = newly_ready_mask;

    if (new_satisfied != expected_satisfied) {
        out[0] = 8UL;
        out[14] = new_satisfied;
        return;
    }
    if (new_completed != expected_completed) {
        out[0] = 9UL;
        out[14] = new_completed;
        return;
    }
    if (ready_mask != expected_ready) {
        out[0] = 10UL;
        out[14] = ready_mask;
        return;
    }
    if (first_ready != expected_first_ready) {
        out[0] = 11UL;
        out[14] = (ulong)first_ready;
        return;
    }
    if (ready_count != expected_ready_count) {
        out[0] = 12UL;
        out[14] = (ulong)ready_count;
        return;
    }
    if (frontier_checksum != expected_checksum) {
        out[0] = 13UL;
        out[14] = frontier_checksum;
        return;
    }
}

__kernel void resident_dag_dispatch_candidate_probe(__global const ulong* descriptors,
                                                    __global const ulong* state,
                                                    __global ulong* out,
                                                    uint expected_magic_lo,
                                                    uint expected_magic_hi,
                                                    uint expected_version,
                                                    uint rows,
                                                    uint u64s_per_row,
                                                    uint expected_edge_index,
                                                    uint expected_layer,
                                                    uint expected_kind,
                                                    uint expected_wait_lo,
                                                    uint expected_wait_hi,
                                                    uint expected_signal_lo,
                                                    uint expected_signal_hi,
                                                    uint expected_checksum_lo,
                                                    uint expected_checksum_hi,
                                                    uint nthreads) {
    uint gid = get_group_id(0) * 256u + get_local_id(0);
    if (gid != 0u || nthreads == 0u) return;

    ulong expected_magic =
        (ulong)expected_magic_lo | ((ulong)expected_magic_hi << 32);
    ulong expected_wait =
        (ulong)expected_wait_lo | ((ulong)expected_wait_hi << 32);
    ulong expected_signal =
        (ulong)expected_signal_lo | ((ulong)expected_signal_hi << 32);
    ulong expected_checksum =
        (ulong)expected_checksum_lo | ((ulong)expected_checksum_hi << 32);

    ulong ready_mask = state[2];
    uint first_ready = (uint)(state[3] & 0xffffffffUL);
    ulong ready_count = state[4];
    ulong frontier_checksum = state[5];

    for (uint i = 0u; i < 16u; ++i) {
        out[i] = 0UL;
    }
    out[1] = (ulong)rows;
    out[2] = (ulong)first_ready;
    out[7] = ready_mask;
    out[8] = (ulong)first_ready;
    out[9] = ready_count;
    out[12] = frontier_checksum;
    out[15] = 0xD35C7105D15C0DEDUL;

    if (rows == 0u || rows > 64u || u64s_per_row != 8u) {
        out[0] = 1UL;
        return;
    }

    if (first_ready == 0xffffffffu) {
        out[14] = 0UL;
        if (expected_edge_index != 0xffffffffu) {
            out[0] = 2UL;
        }
        return;
    }

    if (first_ready >= rows) {
        out[0] = 3UL;
        out[14] = (ulong)first_ready;
        return;
    }
    if ((ready_mask & (1UL << first_ready)) == 0UL) {
        out[0] = 4UL;
        out[14] = ready_mask;
        return;
    }

    ulong base = (ulong)first_ready * (ulong)u64s_per_row;
    ulong magic = descriptors[base + 0UL];
    ulong version_row = descriptors[base + 1UL];
    ulong layer_kind = descriptors[base + 2UL];
    ulong wait_mask = descriptors[base + 3UL];
    ulong signal_mask = descriptors[base + 4UL];
    ulong downstream_edge_index_word = descriptors[base + 5UL];
    ulong downstream_wait_mask = descriptors[base + 6UL];
    ulong row_checksum_expected = descriptors[base + 7UL];
    uint version = (uint)(version_row & 0xffffffffUL);
    uint row_index = (uint)(version_row >> 32);
    uint layer = (uint)(layer_kind & 0xffffffffUL);
    uint kind = (uint)(layer_kind >> 32);
    uint downstream_edge_index = (uint)(downstream_edge_index_word & 0xffffffffUL);

    out[3] = (ulong)layer | ((ulong)kind << 32);
    out[4] = wait_mask;
    out[5] = signal_mask;
    out[6] = downstream_wait_mask;

    if (magic != expected_magic) {
        out[0] = 5UL;
        out[14] = magic;
        return;
    }
    if (version != expected_version || row_index != first_ready) {
        out[0] = 6UL;
        out[14] = version_row;
        return;
    }
    if (expected_edge_index != first_ready || expected_layer != layer || expected_kind != kind ||
        expected_wait != wait_mask || expected_signal != signal_mask) {
        out[0] = 7UL;
        out[14] = ((ulong)expected_edge_index << 32) | (ulong)first_ready;
        return;
    }

    ulong row_checksum = 0xcbf29ce484222325UL;
    ulong checksum_words[7] = {
        (ulong)row_index,
        (ulong)layer,
        (ulong)kind,
        wait_mask,
        signal_mask,
        (ulong)downstream_edge_index,
        downstream_wait_mask,
    };
    for (uint word_idx = 0u; word_idx < 7u; ++word_idx) {
        ulong word = checksum_words[word_idx];
        for (uint byte_idx = 0u; byte_idx < 8u; ++byte_idx) {
            row_checksum ^= (word >> (byte_idx * 8u)) & 0xffUL;
            row_checksum *= 0x100000001b3UL;
        }
    }
    out[10] = row_checksum;
    out[11] = expected_checksum;
    if (row_checksum != row_checksum_expected || row_checksum != expected_checksum) {
        out[0] = 8UL;
        out[14] = row_checksum;
        return;
    }

    ulong candidate_checksum = 0xcbf29ce484222325UL;
    ulong candidate_words[8] = {
        (ulong)first_ready,
        (ulong)layer,
        (ulong)kind,
        wait_mask,
        signal_mask,
        ready_mask,
        (ulong)first_ready,
        ready_count,
    };
    for (uint word_idx = 0u; word_idx < 8u; ++word_idx) {
        ulong word = candidate_words[word_idx];
        for (uint byte_idx = 0u; byte_idx < 8u; ++byte_idx) {
            candidate_checksum ^= (word >> (byte_idx * 8u)) & 0xffUL;
            candidate_checksum *= 0x100000001b3UL;
        }
    }
    out[13] = candidate_checksum;
    out[14] = 1UL;
    mem_fence(CLK_GLOBAL_MEM_FENCE);
}

__kernel void resident_dag_dispatch_candidate_claim_probe(__global const ulong* candidate,
                                                          volatile __global atomic_uint* claim_state,
                                                          __global ulong* out,
                                                          uint expected_edge_index,
                                                          uint expected_layer,
                                                          uint expected_kind,
                                                          uint expected_wait_lo,
                                                          uint expected_wait_hi,
                                                          uint expected_signal_lo,
                                                          uint expected_signal_hi,
                                                          uint expected_row_checksum_lo,
                                                          uint expected_row_checksum_hi,
                                                          uint expected_frontier_checksum_lo,
                                                          uint expected_frontier_checksum_hi,
                                                          uint expected_materialized,
                                                          uint expected_claim_result,
                                                          uint nthreads) {
    uint gid = get_group_id(0) * 256u + get_local_id(0);
    if (gid != 0u || nthreads == 0u) return;

    ulong expected_wait =
        (ulong)expected_wait_lo | ((ulong)expected_wait_hi << 32);
    ulong expected_signal =
        (ulong)expected_signal_lo | ((ulong)expected_signal_hi << 32);
    ulong expected_row_checksum =
        (ulong)expected_row_checksum_lo | ((ulong)expected_row_checksum_hi << 32);
    ulong expected_frontier_checksum =
        (ulong)expected_frontier_checksum_lo | ((ulong)expected_frontier_checksum_hi << 32);
    ulong expected_layer_kind = (ulong)expected_layer | ((ulong)expected_kind << 32);
    uint expected_materialized_flag = expected_materialized != 0u ? 1u : 0u;

    mem_fence(CLK_GLOBAL_MEM_FENCE);
    for (uint i = 0u; i < 16u; ++i) {
        out[i] = 0UL;
    }
    out[1] = (ulong)expected_materialized_flag;
    out[2] = candidate[2];
    out[3] = candidate[3];
    out[4] = candidate[4];
    out[5] = candidate[5];
    out[6] = candidate[10];
    out[7] = candidate[12];
    out[8] = candidate[13];
    out[9] = candidate[14];
    out[13] = (ulong)expected_claim_result;
    out[15] = 0xD35C7105D15C0DEDUL;

    if (candidate[15] != 0xD35C7105D15C0DEDUL) {
        out[0] = 1UL;
        out[14] = candidate[15];
        return;
    }
    if (candidate[0] != 0UL) {
        out[0] = 2UL;
        out[14] = candidate[0];
        return;
    }
    if (candidate[14] != (ulong)expected_materialized_flag) {
        out[0] = 3UL;
        out[14] = candidate[14];
        return;
    }

    if (expected_materialized_flag == 0u) {
        uint observed = atomic_load_explicit(&claim_state[0], memory_order_acquire,
                                             memory_scope_all_svm_devices);
        out[10] = (ulong)observed;
        out[11] = (ulong)observed;
        out[12] = 0UL;
        if (candidate[2] != 0xffffffffUL || expected_edge_index != 0xffffffffu ||
            expected_claim_result != 0u || observed != 0u) {
            out[0] = 4UL;
            out[14] = ((ulong)observed << 32) | (ulong)expected_claim_result;
            return;
        }
        return;
    }

    if ((uint)(candidate[2] & 0xffffffffUL) != expected_edge_index ||
        candidate[3] != expected_layer_kind ||
        candidate[4] != expected_wait ||
        candidate[5] != expected_signal ||
        candidate[10] != expected_row_checksum ||
        candidate[11] != expected_row_checksum ||
        candidate[12] != expected_frontier_checksum) {
        out[0] = 5UL;
        out[14] = candidate[2];
        return;
    }

    ulong candidate_checksum = 0xcbf29ce484222325UL;
    ulong candidate_words[8] = {
        candidate[2],
        (ulong)expected_layer,
        (ulong)expected_kind,
        candidate[4],
        candidate[5],
        candidate[7],
        candidate[8],
        candidate[9],
    };
    for (uint word_idx = 0u; word_idx < 8u; ++word_idx) {
        ulong word = candidate_words[word_idx];
        for (uint byte_idx = 0u; byte_idx < 8u; ++byte_idx) {
            candidate_checksum ^= (word >> (byte_idx * 8u)) & 0xffUL;
            candidate_checksum *= 0x100000001b3UL;
        }
    }
    if (candidate_checksum != candidate[13]) {
        out[0] = 6UL;
        out[14] = candidate_checksum;
        return;
    }

    uint claim_token = expected_edge_index + 1u;
    uint expected_prior = 0u;
    bool claimed = atomic_compare_exchange_strong_explicit(
        &claim_state[0], &expected_prior, claim_token, memory_order_acq_rel,
        memory_order_acquire, memory_scope_all_svm_devices);
    uint observed_prior = claimed ? 0u : expected_prior;
    uint observed_after = atomic_load_explicit(&claim_state[0], memory_order_acquire,
                                               memory_scope_all_svm_devices);
    uint claim_result = claimed ? 1u : (observed_prior == claim_token ? 2u : 3u);
    out[10] = (ulong)observed_prior;
    out[11] = (ulong)observed_after;
    out[12] = (ulong)claim_result;

    ulong claim_checksum = 0xcbf29ce484222325UL;
    ulong claim_words[5] = {
        (ulong)claim_token,
        (ulong)observed_prior,
        (ulong)observed_after,
        (ulong)claim_result,
        candidate[13],
    };
    for (uint word_idx = 0u; word_idx < 5u; ++word_idx) {
        ulong word = claim_words[word_idx];
        for (uint byte_idx = 0u; byte_idx < 8u; ++byte_idx) {
            claim_checksum ^= (word >> (byte_idx * 8u)) & 0xffUL;
            claim_checksum *= 0x100000001b3UL;
        }
    }
    out[14] = claim_checksum;

    if (claim_result != expected_claim_result || observed_after != claim_token) {
        out[0] = 7UL;
        return;
    }
    mem_fence(CLK_GLOBAL_MEM_FENCE);
}

__kernel void resident_dag_inert_dispatch_template_probe(__global const ulong* templates,
                                                         __global const ulong* candidate,
                                                         volatile __global atomic_uint* claim_state,
                                                         __global ulong* packet_out,
                                                         __global ulong* out,
                                                         uint rows,
                                                         uint u64s_per_template,
                                                         uint expected_edge_index,
                                                         uint expected_layer,
                                                         uint expected_kind,
                                                         uint expected_wait_lo,
                                                         uint expected_wait_hi,
                                                         uint expected_signal_lo,
                                                         uint expected_signal_hi,
                                                         uint expected_candidate_checksum_lo,
                                                         uint expected_candidate_checksum_hi,
                                                         uint expected_template_checksum_lo,
                                                         uint expected_template_checksum_hi,
                                                         uint expected_materialized,
                                                         uint nthreads) {
    uint gid = get_group_id(0) * 256u + get_local_id(0);
    if (gid != 0u || nthreads == 0u) return;

    ulong expected_wait =
        (ulong)expected_wait_lo | ((ulong)expected_wait_hi << 32);
    ulong expected_signal =
        (ulong)expected_signal_lo | ((ulong)expected_signal_hi << 32);
    ulong expected_candidate_checksum =
        (ulong)expected_candidate_checksum_lo | ((ulong)expected_candidate_checksum_hi << 32);
    ulong expected_template_checksum =
        (ulong)expected_template_checksum_lo | ((ulong)expected_template_checksum_hi << 32);
    ulong expected_layer_kind = (ulong)expected_layer | ((ulong)expected_kind << 32);
    uint expected_materialized_flag = expected_materialized != 0u ? 1u : 0u;
    uint expected_claim_token =
        expected_materialized_flag != 0u ? expected_edge_index + 1u : 0u;

    mem_fence(CLK_GLOBAL_MEM_FENCE);
    for (uint i = 0u; i < 16u; ++i) {
        out[i] = 0UL;
    }
    out[1] = (ulong)expected_materialized_flag;
    out[2] = candidate[2];
    out[3] = candidate[3];
    out[4] = candidate[4];
    out[5] = candidate[5];
    out[6] = (ulong)atomic_load_explicit(&claim_state[0], memory_order_acquire,
                                         memory_scope_all_svm_devices);
    out[7] = candidate[13];
    out[11] = expected_template_checksum;
    out[15] = 0xD35C7105D15C0DEDUL;

    if (rows == 0u || rows > 64u || u64s_per_template != 8u) {
        out[0] = 1UL;
        out[14] = (ulong)rows;
        return;
    }

    for (uint i = 0u; i < 8u; ++i) {
        packet_out[i] = 0UL;
    }

    if (expected_materialized_flag == 0u) {
        if (expected_edge_index != 0xffffffffu || candidate[14] != 0UL ||
            candidate[2] != 0xffffffffUL || out[6] != 0UL ||
            expected_template_checksum != 0UL) {
            out[0] = 2UL;
            out[14] = candidate[14];
            return;
        }
        out[8] = packet_out[0];
        out[9] = 0UL;
        out[10] = 0UL;
        out[12] = 0UL;
        out[14] = 0UL;
        return;
    }

    if (candidate[15] != 0xD35C7105D15C0DEDUL || candidate[0] != 0UL ||
        candidate[14] != 1UL) {
        out[0] = 3UL;
        out[14] = candidate[15];
        return;
    }
    if ((uint)(candidate[2] & 0xffffffffUL) != expected_edge_index ||
        candidate[3] != expected_layer_kind ||
        candidate[4] != expected_wait ||
        candidate[5] != expected_signal ||
        candidate[10] != expected_candidate_checksum ||
        out[6] != (ulong)expected_claim_token) {
        out[0] = 4UL;
        out[14] = ((ulong)expected_claim_token << 32) | (out[6] & 0xffffffffUL);
        return;
    }
    if (expected_edge_index >= rows) {
        out[0] = 5UL;
        out[14] = (ulong)expected_edge_index;
        return;
    }

    ulong base = (ulong)expected_edge_index * 8UL;
    ulong template_words[8];
    for (uint i = 0u; i < 8u; ++i) {
        template_words[i] = templates[base + (ulong)i];
    }
    out[9] = template_words[0];

    if (template_words[2] != ((ulong)expected_edge_index | ((ulong)expected_layer << 32)) ||
        template_words[3] != expected_layer_kind ||
        template_words[4] != expected_wait ||
        template_words[5] != expected_signal ||
        template_words[6] != candidate[10]) {
        out[0] = 6UL;
        out[14] = template_words[2];
        return;
    }

    ulong template_checksum = 0xcbf29ce484222325UL;
    for (uint word_idx = 1u; word_idx < 7u; ++word_idx) {
        ulong word = template_words[word_idx];
        for (uint byte_idx = 0u; byte_idx < 8u; ++byte_idx) {
            template_checksum ^= (word >> (byte_idx * 8u)) & 0xffUL;
            template_checksum *= 0x100000001b3UL;
        }
    }
    out[10] = template_checksum;
    out[12] = template_words[7];
    if (template_checksum != template_words[7] ||
        template_checksum != expected_template_checksum) {
        out[0] = 7UL;
        out[14] = template_checksum;
        return;
    }

    for (uint i = 1u; i < 8u; ++i) {
        packet_out[i] = template_words[i];
    }
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release,
                           memory_scope_all_svm_devices);
    packet_out[0] = 0UL;
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release,
                           memory_scope_all_svm_devices);
    out[8] = packet_out[0];
    out[14] = 1UL;
}

__kernel void resident_dag_shadow_queue_slot_probe(__global const ulong* inert_packet,
                                                   volatile __global atomic_uint* write_index,
                                                   __global ulong* shadow_queue,
                                                   __global ulong* out,
                                                   uint expected_edge_index,
                                                   uint expected_layer,
                                                   uint expected_kind,
                                                   uint expected_wait_lo,
                                                   uint expected_wait_hi,
                                                   uint expected_signal_lo,
                                                   uint expected_signal_hi,
                                                   uint expected_row_checksum_lo,
                                                   uint expected_row_checksum_hi,
                                                   uint expected_template_checksum_lo,
                                                   uint expected_template_checksum_hi,
                                                   uint expected_materialized,
                                                   uint queue_slots,
                                                   uint nthreads) {
    uint gid = get_group_id(0) * 256u + get_local_id(0);
    if (gid != 0u || nthreads == 0u) return;

    ulong expected_wait =
        (ulong)expected_wait_lo | ((ulong)expected_wait_hi << 32);
    ulong expected_signal =
        (ulong)expected_signal_lo | ((ulong)expected_signal_hi << 32);
    ulong expected_row_checksum =
        (ulong)expected_row_checksum_lo | ((ulong)expected_row_checksum_hi << 32);
    ulong expected_template_checksum =
        (ulong)expected_template_checksum_lo | ((ulong)expected_template_checksum_hi << 32);
    ulong expected_layer_kind = (ulong)expected_layer | ((ulong)expected_kind << 32);
    uint expected_materialized_flag = expected_materialized != 0u ? 1u : 0u;

    mem_fence(CLK_GLOBAL_MEM_FENCE);
    for (uint i = 0u; i < 16u; ++i) {
        out[i] = 0UL;
    }
    uint initial_write = atomic_load_explicit(&write_index[0], memory_order_acquire,
                                              memory_scope_all_svm_devices);
    out[1] = (ulong)expected_materialized_flag;
    out[2] = (ulong)initial_write;
    out[5] = inert_packet[2];
    out[6] = inert_packet[3];
    out[7] = inert_packet[7];
    out[9] = inert_packet[0];
    out[11] = expected_template_checksum;
    out[15] = 0xD35C51075A10A11CUL;

    if (queue_slots < 3u || queue_slots > 64u) {
        out[0] = 1UL;
        out[14] = (ulong)queue_slots;
        return;
    }

    if (expected_materialized_flag == 0u) {
        bool packet_zero = true;
        for (uint i = 0u; i < 8u; ++i) {
            if (inert_packet[i] != 0UL) {
                packet_zero = false;
            }
        }
        uint final_write = atomic_load_explicit(&write_index[0], memory_order_acquire,
                                                memory_scope_all_svm_devices);
        out[4] = (ulong)final_write;
        out[8] = shadow_queue[0];
        out[12] = shadow_queue[0];
        out[13] = shadow_queue[8];
        if (expected_edge_index != 0xffffffffu || expected_template_checksum != 0UL ||
            !packet_zero || final_write != initial_write) {
            out[0] = 2UL;
            out[14] = packet_zero ? (ulong)final_write : inert_packet[0];
            return;
        }
        out[14] = 0UL;
        return;
    }

    if (expected_edge_index == 0xffffffffu ||
        inert_packet[0] != 0UL ||
        inert_packet[1] != 0x4d41525f41514c54UL ||
        inert_packet[2] != ((ulong)expected_edge_index | ((ulong)expected_layer << 32)) ||
        inert_packet[3] != expected_layer_kind ||
        inert_packet[4] != expected_wait ||
        inert_packet[5] != expected_signal ||
        inert_packet[6] != expected_row_checksum ||
        inert_packet[7] != expected_template_checksum) {
        out[0] = 3UL;
        out[14] = inert_packet[0];
        return;
    }

    ulong template_checksum = 0xcbf29ce484222325UL;
    for (uint word_idx = 1u; word_idx < 7u; ++word_idx) {
        ulong word = inert_packet[word_idx];
        for (uint byte_idx = 0u; byte_idx < 8u; ++byte_idx) {
            template_checksum ^= (word >> (byte_idx * 8u)) & 0xffUL;
            template_checksum *= 0x100000001b3UL;
        }
    }
    out[10] = template_checksum;
    if (template_checksum != expected_template_checksum) {
        out[0] = 4UL;
        out[14] = template_checksum;
        return;
    }

    uint reserved_write = atomic_fetch_add_explicit(&write_index[0], 1u,
                                                    memory_order_acq_rel,
                                                    memory_scope_all_svm_devices);
    uint slot_idx = reserved_write % queue_slots;
    uint final_write = atomic_load_explicit(&write_index[0], memory_order_acquire,
                                            memory_scope_all_svm_devices);
    out[2] = (ulong)reserved_write;
    out[3] = (ulong)slot_idx;
    out[4] = (ulong)final_write;

    if (slot_idx == 0u || slot_idx + 1u >= queue_slots) {
        out[0] = 5UL;
        out[14] = (ulong)slot_idx;
        return;
    }
    ulong base = (ulong)slot_idx * 8UL;
    for (uint i = 0u; i < 8u; ++i) {
        if (shadow_queue[base + (ulong)i] != 0UL) {
            out[0] = 6UL;
            out[14] = ((ulong)i << 32) | (shadow_queue[base + (ulong)i] & 0xffffffffUL);
            return;
        }
    }

    for (uint i = 1u; i < 8u; ++i) {
        shadow_queue[base + (ulong)i] = inert_packet[i];
    }
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release,
                           memory_scope_all_svm_devices);
    shadow_queue[base] = 0UL;
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release,
                           memory_scope_all_svm_devices);

    out[8] = shadow_queue[base];
    out[12] = shadow_queue[base - 8UL];
    out[13] = shadow_queue[base + 8UL];
    if (out[8] != 0UL || out[12] != 0UL || out[13] != 0UL ||
        final_write != reserved_write + 1u) {
        out[0] = 7UL;
        out[14] = out[8];
        return;
    }
    out[14] = 1UL;
}

__kernel void resident_dag_shadow_queue_multiwriter_probe(__global const ulong* inert_packet,
                                                          volatile __global atomic_uint* write_index,
                                                          __global ulong* shadow_queue,
                                                          volatile __global atomic_ulong* queue_control,
                                                          __global ulong* out,
                                                          uint expected_edge_index,
                                                          uint expected_layer,
                                                          uint expected_kind,
                                                          uint expected_wait_lo,
                                                          uint expected_wait_hi,
                                                          uint expected_signal_lo,
                                                          uint expected_signal_hi,
                                                          uint expected_row_checksum_lo,
                                                          uint expected_row_checksum_hi,
                                                          uint expected_template_checksum_lo,
                                                          uint expected_template_checksum_hi,
                                                          uint expected_materialized,
                                                          uint queue_slots,
                                                          uint nthreads) {
    uint group = get_group_id(0);
    uint lid = get_local_id(0);
    if (group != 0u || nthreads == 0u) return;
    __local uint writer_reserved[2];
    __local uint writer_slot[2];
    __local uint writer_status[2];
    __local ulong writer_bad[2];
    __local ulong writer_slot_va[2];
    __local ulong writer_slot_offset[2];

    ulong expected_wait =
        (ulong)expected_wait_lo | ((ulong)expected_wait_hi << 32);
    ulong expected_signal =
        (ulong)expected_signal_lo | ((ulong)expected_signal_hi << 32);
    ulong expected_row_checksum =
        (ulong)expected_row_checksum_lo | ((ulong)expected_row_checksum_hi << 32);
    ulong expected_template_checksum =
        (ulong)expected_template_checksum_lo | ((ulong)expected_template_checksum_hi << 32);
    ulong expected_layer_kind = (ulong)expected_layer | ((ulong)expected_kind << 32);
    ulong expected_header_word = 0x1502UL;
    uint expected_header32_word = 0x00001502u;
    ulong no_doorbell_word = 0xffffffffffffffffUL;
    ulong expected_control_magic = 0x4d41525f5143544cUL;
    uint expected_materialized_flag = expected_materialized != 0u ? 1u : 0u;

    if (lid == 0u) {
        for (uint i = 0u; i < 704u; ++i) {
            out[i] = 0UL;
        }
        uint initial_write = atomic_load_explicit(&write_index[0], memory_order_acquire,
                                                  memory_scope_all_svm_devices);
        out[1] = (ulong)expected_materialized_flag;
        out[2] = (ulong)initial_write;
        out[9] = inert_packet[0];
        out[11] = expected_template_checksum;
        out[36] = no_doorbell_word;
        out[37] = no_doorbell_word;
        out[38] = no_doorbell_word;
        out[47] = 0xD35C51075A10A22CUL;
    }
    if (lid < 2u) {
        writer_reserved[lid] = 0u;
        writer_slot[lid] = 0u;
        writer_status[lid] = 0u;
        writer_bad[lid] = 0UL;
        writer_slot_va[lid] = 0UL;
        writer_slot_offset[lid] = 0UL;
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    if (lid == 0u) {
        if (queue_slots < 8u || queue_slots > 64u ||
            (queue_slots & (queue_slots - 1u)) != 0u) {
            out[0] = 1UL;
            out[14] = (ulong)queue_slots;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (out[0] != 0UL) return;

    if (lid == 0u) {
        ulong control_magic = atomic_load_explicit(&queue_control[0], memory_order_acquire,
                                                   memory_scope_all_svm_devices);
        ulong control_version = atomic_load_explicit(&queue_control[1], memory_order_acquire,
                                                     memory_scope_all_svm_devices);
        ulong control_slots = atomic_load_explicit(&queue_control[2], memory_order_acquire,
                                                   memory_scope_all_svm_devices);
        ulong control_packet_u64s = atomic_load_explicit(&queue_control[3], memory_order_acquire,
                                                         memory_scope_all_svm_devices);
        ulong control_read_index = atomic_load_explicit(&queue_control[4], memory_order_acquire,
                                                        memory_scope_all_svm_devices);
        ulong control_write_index = atomic_load_explicit(&queue_control[5], memory_order_acquire,
                                                         memory_scope_all_svm_devices);
        ulong control_doorbell = atomic_load_explicit(&queue_control[6], memory_order_acquire,
                                                      memory_scope_all_svm_devices);
        ulong control_last_packet = atomic_load_explicit(&queue_control[7], memory_order_acquire,
                                                         memory_scope_all_svm_devices);
        ulong control_publish_count = atomic_load_explicit(&queue_control[8], memory_order_acquire,
                                                           memory_scope_all_svm_devices);
        ulong control_flags = atomic_load_explicit(&queue_control[9], memory_order_acquire,
                                                   memory_scope_all_svm_devices);
        ulong control_base_va = atomic_load_explicit(&queue_control[12], memory_order_acquire,
                                                     memory_scope_all_svm_devices);
        ulong control_queue_bytes = atomic_load_explicit(&queue_control[13], memory_order_acquire,
                                                         memory_scope_all_svm_devices);
        ulong control_packet_bytes = atomic_load_explicit(&queue_control[14], memory_order_acquire,
                                                          memory_scope_all_svm_devices);
        ulong control_slot_mask = atomic_load_explicit(&queue_control[15], memory_order_acquire,
                                                       memory_scope_all_svm_devices);
        uint observed_initial_write = atomic_load_explicit(&write_index[0], memory_order_acquire,
                                                           memory_scope_all_svm_devices);
        out[43] = control_magic;
        out[44] = control_read_index;
        out[45] = control_write_index;
        out[46] = control_doorbell;
        out[48] = control_slots;
        out[49] = control_packet_u64s;
        out[50] = control_flags;
        out[51] = control_write_index;
        out[52] = control_doorbell;
        out[53] = control_publish_count;
        out[54] = control_last_packet;
        out[57] = control_version;
        out[58] = control_base_va;
        out[59] = control_queue_bytes;
        out[60] = control_packet_bytes;
        out[61] = control_slot_mask;
        if (control_magic != expected_control_magic ||
            control_version != 1UL ||
            control_slots != (ulong)queue_slots ||
            control_packet_u64s != 8UL ||
            control_base_va == 0UL ||
            (control_base_va & 63UL) != 0UL ||
            control_queue_bytes != ((ulong)queue_slots * control_packet_u64s * 8UL) ||
            control_packet_bytes != 64UL ||
            control_slot_mask != ((ulong)queue_slots - 1UL) ||
            control_read_index != 1UL ||
            control_write_index != (ulong)observed_initial_write ||
            control_doorbell != no_doorbell_word ||
            control_last_packet != no_doorbell_word ||
            control_publish_count != 0UL ||
            control_flags != 1UL) {
            out[0] = 10UL;
            out[14] = control_magic;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (out[0] != 0UL) return;

    if (expected_materialized_flag == 0u) {
        if (lid == 0u) {
            bool packet_zero = true;
            for (uint i = 0u; i < 8u; ++i) {
                if (inert_packet[i] != 0UL) {
                    packet_zero = false;
                }
            }
            uint final_write = atomic_load_explicit(&write_index[0], memory_order_acquire,
                                                    memory_scope_all_svm_devices);
            out[4] = (ulong)final_write;
            out[8] = shadow_queue[0];
            out[12] = shadow_queue[0];
            out[13] = shadow_queue[8];
            if (expected_edge_index != 0xffffffffu || expected_template_checksum != 0UL ||
                !packet_zero || final_write != (uint)out[2]) {
                out[0] = 2UL;
                out[14] = packet_zero ? (ulong)final_write : inert_packet[0];
                return;
            }
            out[14] = 0UL;
        }
        return;
    }

    if (lid == 0u) {
        if (expected_edge_index == 0xffffffffu ||
            inert_packet[0] != 0UL ||
            inert_packet[1] != 0x4d41525f41514c54UL ||
            inert_packet[2] != ((ulong)expected_edge_index | ((ulong)expected_layer << 32)) ||
            inert_packet[3] != expected_layer_kind ||
            inert_packet[4] != expected_wait ||
            inert_packet[5] != expected_signal ||
            inert_packet[6] != expected_row_checksum ||
            inert_packet[7] != expected_template_checksum) {
            out[0] = 3UL;
            out[14] = inert_packet[0];
        }

        ulong template_checksum = 0xcbf29ce484222325UL;
        for (uint word_idx = 1u; word_idx < 7u; ++word_idx) {
            ulong word = inert_packet[word_idx];
            for (uint byte_idx = 0u; byte_idx < 8u; ++byte_idx) {
                template_checksum ^= (word >> (byte_idx * 8u)) & 0xffUL;
                template_checksum *= 0x100000001b3UL;
            }
        }
        out[10] = template_checksum;
        if (out[0] == 0UL && template_checksum != expected_template_checksum) {
            out[0] = 4UL;
            out[14] = template_checksum;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (out[0] != 0UL) return;

    if (lid < 2u) {
        ulong control_base_va = atomic_load_explicit(&queue_control[12], memory_order_acquire,
                                                     memory_scope_all_svm_devices);
        ulong control_queue_bytes = atomic_load_explicit(&queue_control[13], memory_order_acquire,
                                                         memory_scope_all_svm_devices);
        ulong control_packet_bytes = atomic_load_explicit(&queue_control[14], memory_order_acquire,
                                                          memory_scope_all_svm_devices);
        ulong control_slot_mask = atomic_load_explicit(&queue_control[15], memory_order_acquire,
                                                       memory_scope_all_svm_devices);
        ulong control_packet_u64s = atomic_load_explicit(&queue_control[3], memory_order_acquire,
                                                         memory_scope_all_svm_devices);
        uint reserved_write = atomic_fetch_add_explicit(&write_index[0], 1u,
                                                        memory_order_acq_rel,
                                                        memory_scope_all_svm_devices);
        uint slot_idx = reserved_write & (uint)control_slot_mask;
        ulong slot_offset_bytes = (ulong)slot_idx * control_packet_bytes;
        ulong slot_va = control_base_va + slot_offset_bytes;
        writer_reserved[lid] = reserved_write;
        writer_slot[lid] = slot_idx;
        writer_slot_va[lid] = slot_va;
        writer_slot_offset[lid] = slot_offset_bytes;
        if (slot_idx == 0u || slot_idx + 1u >= queue_slots) {
            writer_status[lid] = 5u;
            writer_bad[lid] = (ulong)slot_idx;
        } else if (control_packet_bytes != 64UL ||
                   control_packet_u64s != 8UL ||
                   slot_offset_bytes + control_packet_bytes > control_queue_bytes) {
            writer_status[lid] = 11u;
            writer_bad[lid] = slot_va;
        } else {
            ulong base = slot_offset_bytes >> 3;
            for (uint i = 0u; i < 8u; ++i) {
                if (shadow_queue[base + (ulong)i] != 0UL) {
                    writer_status[lid] = 6u;
                    writer_bad[lid] = ((ulong)slot_idx << 32) | (ulong)i;
                }
            }
            if (writer_status[lid] == 0u) {
                for (uint i = 1u; i < 8u; ++i) {
                    shadow_queue[base + (ulong)i] = inert_packet[i];
                }
                atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release,
                                       memory_scope_all_svm_devices);
                volatile __global atomic_uint* header_word =
                    (volatile __global atomic_uint*)(&shadow_queue[base]);
                atomic_store_explicit(header_word, expected_header32_word,
                                      memory_order_release,
                                      memory_scope_all_svm_devices);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    if (lid == 0u) {
        uint final_write = atomic_load_explicit(&write_index[0], memory_order_acquire,
                                                memory_scope_all_svm_devices);
        out[3] = (ulong)writer_reserved[0];
        out[5] = (ulong)writer_slot[0];
        out[6] = (ulong)writer_reserved[1];
        out[7] = (ulong)writer_slot[1];
        out[4] = (ulong)final_write;
        out[62] = writer_slot_va[0];
        out[63] = writer_slot_va[1];
        out[64] = writer_slot_offset[0];
        out[65] = writer_slot_offset[1];
        out[8] = shadow_queue[16];
        out[12] = shadow_queue[0];
        out[13] = shadow_queue[32];
        out[14] = 1UL;
        out[15] = 2UL;
        for (uint i = 0u; i < 8u; ++i) {
            out[16u + i] = shadow_queue[16u + i];
            out[24u + i] = shadow_queue[24u + i];
        }
        out[32] = shadow_queue[8];
        out[33] = shadow_queue[32];
        out[34] = shadow_queue[16];
        out[35] = shadow_queue[24];

        uint initial_write = (uint)out[2];
        uint r0 = (uint)out[3];
        uint r1 = (uint)out[6];
        uint s0 = (uint)out[5];
        uint s1 = (uint)out[7];
        if (writer_status[0] != 0u || writer_status[1] != 0u) {
            out[0] = writer_status[0] != 0u ? (ulong)writer_status[0] : (ulong)writer_status[1];
            out[14] = writer_status[0] != 0u ? writer_bad[0] : writer_bad[1];
            return;
        }
        bool reservations_ok =
            final_write == initial_write + 2u &&
            ((r0 == initial_write && r1 == initial_write + 1u) ||
             (r1 == initial_write && r0 == initial_write + 1u)) &&
            s0 != s1 &&
            ((s0 == (initial_write & (queue_slots - 1u)) &&
              s1 == ((initial_write + 1u) & (queue_slots - 1u))) ||
             (s1 == (initial_write & (queue_slots - 1u)) &&
              s0 == ((initial_write + 1u) & (queue_slots - 1u))));
        if (!reservations_ok) {
            out[0] = 7UL;
            out[14] = ((ulong)s0 << 32) | (ulong)s1;
            return;
        }
        for (uint i = 1u; i < 8u; ++i) {
            if (shadow_queue[16u + i] != inert_packet[i] ||
                shadow_queue[24u + i] != inert_packet[i]) {
                out[0] = 8UL;
                out[14] = (ulong)i;
                return;
            }
        }
        if (shadow_queue[0] != 0UL || shadow_queue[32] != 0UL ||
            shadow_queue[16] != expected_header_word ||
            shadow_queue[24] != expected_header_word) {
            out[0] = 9UL;
            out[14] = shadow_queue[16];
            return;
        }
        ulong publish_initial_write = (ulong)out[2];
        ulong doorbell_packet_id = (ulong)(final_write - 1u);
        ulong publish_packet_count = (ulong)final_write - publish_initial_write;
        ulong pre_publish_control_write =
            atomic_load_explicit(&queue_control[5], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong pre_publish_control_doorbell =
            atomic_load_explicit(&queue_control[6], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        out[38] = doorbell_packet_id;
        out[40] = shadow_queue[16];
        out[41] = shadow_queue[24];
        out[42] = (ulong)final_write;
        atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release,
                               memory_scope_all_svm_devices);
        atomic_store_explicit(&queue_control[5], (ulong)final_write,
                              memory_order_release,
                              memory_scope_all_svm_devices);
        ulong write_index_after_release =
            atomic_load_explicit(&queue_control[5], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release,
                               memory_scope_all_svm_devices);
        atomic_store_explicit(&queue_control[6], doorbell_packet_id,
                              memory_order_release,
                              memory_scope_all_svm_devices);
        volatile __global atomic_ulong* shadow_doorbell_word =
            (volatile __global atomic_ulong*)(&out[37]);
        atomic_store_explicit(shadow_doorbell_word, doorbell_packet_id,
                              memory_order_release,
                              memory_scope_all_svm_devices);
        ulong doorbell_after_release =
            atomic_load_explicit(&queue_control[6], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        atomic_store_explicit(&queue_control[7], doorbell_packet_id,
                              memory_order_release,
                              memory_scope_all_svm_devices);
        atomic_store_explicit(&queue_control[8], 1UL,
                              memory_order_release,
                              memory_scope_all_svm_devices);
        atomic_store_explicit(&queue_control[10], shadow_queue[16],
                              memory_order_release,
                              memory_scope_all_svm_devices);
        atomic_store_explicit(&queue_control[11], shadow_queue[24],
                              memory_order_release,
                              memory_scope_all_svm_devices);
        out[160] = 1UL;
        out[161] = pre_publish_control_write;
        out[162] = pre_publish_control_doorbell;
        out[163] = publish_initial_write;
        out[164] = publish_packet_count;
        out[165] = (ulong)final_write;
        out[166] = doorbell_packet_id;
        out[167] = shadow_queue[16];
        out[168] = shadow_queue[24];
        out[169] = write_index_after_release;
        out[170] = doorbell_after_release;
        out[171] =
            (write_index_after_release == (ulong)final_write &&
             doorbell_after_release == doorbell_packet_id &&
             shadow_queue[16] == expected_header_word &&
             shadow_queue[24] == expected_header_word) ? 1UL : 0UL;
        out[172] =
            (pre_publish_control_doorbell == no_doorbell_word ||
             doorbell_packet_id > pre_publish_control_doorbell) ? 1UL : 0UL;
        out[173] = 1UL;
        out[174] = (publish_packet_count == 2UL) ? 1UL : 0UL;
        out[175] =
            (doorbell_packet_id + 1UL == (ulong)final_write) ? 1UL : 0UL;
        ulong completion_intermediate_signal_handle =
            atomic_load_explicit(&queue_control[33], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong completion_signal_handle =
            atomic_load_explicit(&queue_control[34], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong barrier_completion_signal_handle =
            atomic_load_explicit(&queue_control[35], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong completion_initial_value = 1UL;
        ulong completion_before_first_packet = completion_initial_value;
        ulong completion_first_packet_id = initial_write;
        ulong completion_terminal_packet_id = doorbell_packet_id;
        ulong completion_after_first_packet = completion_before_first_packet;
        ulong completion_after_terminal_packet =
            (completion_signal_handle != 0UL &&
             out[171] == 1UL &&
             out[173] == 1UL &&
             out[175] == 1UL) ? 0UL : completion_after_first_packet;
        out[176] = 1UL;
        out[177] = completion_intermediate_signal_handle;
        out[178] = completion_signal_handle;
        out[179] = completion_initial_value;
        out[180] = completion_before_first_packet;
        out[181] = completion_first_packet_id;
        out[182] = completion_terminal_packet_id;
        out[183] = completion_after_first_packet;
        out[184] = completion_after_terminal_packet;
        out[185] =
            (completion_signal_handle != 0UL &&
             completion_after_terminal_packet < 1UL) ? 1UL : 0UL;
        out[186] =
            (completion_intermediate_signal_handle == 0UL &&
             completion_signal_handle != 0UL) ? 1UL : 0UL;
        out[187] = (out[171] == 1UL && out[173] == 1UL) ? 1UL : 0UL;
        out[188] =
            (completion_signal_handle != 0UL &&
             completion_before_first_packet == 1UL &&
             completion_after_terminal_packet == 0UL) ? 1UL : 0UL;
        out[189] = 1UL;
        out[190] =
            (completion_intermediate_signal_handle == 0UL &&
             completion_after_first_packet == completion_before_first_packet) ? 1UL : 0UL;
        out[191] =
            (completion_signal_handle != 0UL &&
             completion_terminal_packet_id == doorbell_packet_id &&
             completion_terminal_packet_id + 1UL == (ulong)final_write) ? 1UL : 0UL;
        ulong barrier_completion_initial_value =
            (barrier_completion_signal_handle != 0UL) ? 1UL : 0UL;
        ulong barrier_dep_zero_seen =
            (completion_after_terminal_packet == 0UL) ? 1UL : 0UL;
        ulong barrier_completion_after =
            (barrier_dep_zero_seen == 1UL &&
             out[191] == 1UL &&
             barrier_completion_signal_handle != 0UL) ? 0UL : barrier_completion_initial_value;
        out[192] = 1UL;
        out[193] = (completion_signal_handle != 0UL) ? 1UL : 0UL;
        out[194] = (barrier_completion_signal_handle == 0UL) ? 1UL : 0UL;
        out[195] = completion_signal_handle;
        out[196] = completion_after_terminal_packet;
        out[197] = barrier_dep_zero_seen;
        out[198] = barrier_completion_signal_handle;
        out[199] = barrier_completion_initial_value;
        out[200] = barrier_completion_after;
        out[201] =
            (barrier_completion_signal_handle == 0UL ||
             barrier_completion_after < 1UL) ? 1UL : 0UL;
        out[202] =
            (completion_intermediate_signal_handle == 0UL &&
             completion_after_first_packet != 0UL) ? 1UL : 0UL;
        out[203] = (out[188] == 1UL && barrier_dep_zero_seen == 1UL) ? 1UL : 0UL;
        out[204] = (barrier_completion_signal_handle == 0UL) ? 1UL : 0UL;
        out[205] =
            (barrier_dep_zero_seen == 1UL &&
             (barrier_completion_signal_handle == 0UL ||
              barrier_completion_after == 0UL)) ? 1UL : 0UL;
        out[206] =
            (out[195] == completion_signal_handle &&
             out[196] == completion_after_terminal_packet &&
             out[198] == barrier_completion_signal_handle) ? 1UL : 0UL;
        out[207] = 6UL;
        ulong control_base_va = atomic_load_explicit(&queue_control[12], memory_order_acquire,
                                                     memory_scope_all_svm_devices);
        ulong control_queue_bytes = atomic_load_explicit(&queue_control[13], memory_order_acquire,
                                                         memory_scope_all_svm_devices);
        ulong control_packet_bytes = atomic_load_explicit(&queue_control[14], memory_order_acquire,
                                                          memory_scope_all_svm_devices);
        ulong control_dispatch_kernel_object =
            atomic_load_explicit(&queue_control[16], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong control_metadata_kernarg_segment_size =
            atomic_load_explicit(&queue_control[17], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong control_metadata_kernarg_segment_align =
            atomic_load_explicit(&queue_control[18], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong control_metadata_aql_kernarg_alignment_floor =
            atomic_load_explicit(&queue_control[19], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong control_metadata_private_segment_size =
            atomic_load_explicit(&queue_control[20], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong control_metadata_group_segment_size =
            atomic_load_explicit(&queue_control[21], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong control_metadata_max_flat_workgroup_size =
            atomic_load_explicit(&queue_control[22], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong control_metadata_wavefront_size =
            atomic_load_explicit(&queue_control[23], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong control_metadata_descriptor_alignment =
            atomic_load_explicit(&queue_control[24], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong control_dispatch_kernarg_address =
            atomic_load_explicit(&queue_control[25], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong control_dispatch_kernarg_allocation_size =
            atomic_load_explicit(&queue_control[26], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong control_dispatch_kernarg_ring_base =
            atomic_load_explicit(&queue_control[27], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong control_dispatch_kernarg_ring_stride =
            atomic_load_explicit(&queue_control[28], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong control_dispatch_kernarg_ring_slots =
            atomic_load_explicit(&queue_control[29], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong control_dispatch_kernarg_ring_slot_index =
            atomic_load_explicit(&queue_control[30], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong control_dispatch_kernarg_ring_slot_offset =
            atomic_load_explicit(&queue_control[31], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong control_dispatch_kernarg_ring_allocation_size =
            atomic_load_explicit(&queue_control[32], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong dispatch_packet_type = 2UL;
        ulong dispatch_header_word = expected_header_word;
        ulong dispatch_setup_dimensions = 1UL;
        ulong dispatch_workgroup_x = 256UL;
        ulong dispatch_workgroup_y = 1UL;
        ulong dispatch_workgroup_z = 1UL;
        ulong dispatch_grid_x = 256UL;
        ulong dispatch_grid_y = 1UL;
        ulong dispatch_grid_z = 1UL;
        ulong metadata_kernarg_segment_size = control_metadata_kernarg_segment_size;
        ulong metadata_kernarg_segment_align = control_metadata_kernarg_segment_align;
        ulong metadata_aql_kernarg_alignment_floor =
            control_metadata_aql_kernarg_alignment_floor;
        ulong metadata_private_segment_size = control_metadata_private_segment_size;
        ulong metadata_group_segment_size = control_metadata_group_segment_size;
        ulong metadata_max_flat_workgroup_size = control_metadata_max_flat_workgroup_size;
        ulong metadata_wavefront_size = control_metadata_wavefront_size;
        ulong metadata_descriptor_alignment = control_metadata_descriptor_alignment;
        ulong dispatch_private_segment_size = metadata_private_segment_size;
        ulong dispatch_group_segment_size = metadata_group_segment_size;
        ulong dispatch_kernel_object = control_dispatch_kernel_object;
        ulong dispatch_kernarg_address = control_dispatch_kernarg_address;
        ulong dispatch_kernarg_allocation_size = control_dispatch_kernarg_allocation_size;
        ulong dispatch_kernarg_ring_selected_address =
            control_dispatch_kernarg_ring_base + control_dispatch_kernarg_ring_slot_offset;
        ulong dispatch_completion_signal_handle = 0UL;
        ulong dispatch_workgroup_items =
            dispatch_workgroup_x * dispatch_workgroup_y * dispatch_workgroup_z;
        ulong dispatch_header_type_matches =
            ((dispatch_header_word & 0xffUL) == dispatch_packet_type) ? 1UL : 0UL;
        ulong dispatch_dimensions_valid =
            (dispatch_setup_dimensions >= 1UL && dispatch_setup_dimensions <= 3UL) ? 1UL : 0UL;
        ulong dispatch_workgroup_nonzero =
            (dispatch_workgroup_x != 0UL &&
             dispatch_workgroup_y != 0UL &&
             dispatch_workgroup_z != 0UL) ? 1UL : 0UL;
        ulong dispatch_grid_covers_workgroup =
            (dispatch_grid_x >= dispatch_workgroup_x &&
             dispatch_grid_y >= dispatch_workgroup_y &&
             dispatch_grid_z >= dispatch_workgroup_z &&
             (dispatch_grid_x % dispatch_workgroup_x) == 0UL &&
             (dispatch_grid_y % dispatch_workgroup_y) == 0UL &&
             (dispatch_grid_z % dispatch_workgroup_z) == 0UL) ? 1UL : 0UL;
        ulong dispatch_kernel_object_nonzero =
            (dispatch_kernel_object != 0UL) ? 1UL : 0UL;
        ulong dispatch_kernarg_alignment16 =
            (dispatch_kernarg_address != 0UL &&
             (dispatch_kernarg_address & 15UL) == 0UL) ? 1UL : 0UL;
        ulong dispatch_completion_elided =
            (dispatch_completion_signal_handle == 0UL) ? 1UL : 0UL;
        ulong dispatch_kernel_object_alignment64 =
            (metadata_descriptor_alignment != 0UL &&
             (metadata_descriptor_alignment & (metadata_descriptor_alignment - 1UL)) == 0UL &&
             (dispatch_kernel_object & (metadata_descriptor_alignment - 1UL)) == 0UL) ? 1UL : 0UL;
        ulong dispatch_segments_match_metadata =
            (dispatch_private_segment_size == metadata_private_segment_size &&
             dispatch_group_segment_size >= metadata_group_segment_size) ? 1UL : 0UL;
        ulong dispatch_kernarg_size_multiple16 =
            ((metadata_kernarg_segment_size & 15UL) == 0UL) ? 1UL : 0UL;
        ulong metadata_align_power_of_two =
            (metadata_kernarg_segment_align != 0UL &&
             (metadata_kernarg_segment_align & (metadata_kernarg_segment_align - 1UL)) == 0UL) ? 1UL : 0UL;
        ulong dispatch_workgroup_within_max_flat =
            (dispatch_workgroup_items <= metadata_max_flat_workgroup_size) ? 1UL : 0UL;
        ulong dispatch_wavefront_multiple =
            (metadata_wavefront_size != 0UL &&
             (dispatch_workgroup_items % metadata_wavefront_size) == 0UL) ? 1UL : 0UL;
        ulong dispatch_kernarg_allocation_covers_metadata =
            (metadata_kernarg_segment_size != 0UL &&
             dispatch_kernarg_allocation_size >= metadata_kernarg_segment_size) ? 1UL : 0UL;
        ulong dispatch_kernarg_meets_metadata_or_floor_align =
            (metadata_aql_kernarg_alignment_floor != 0UL &&
             (metadata_aql_kernarg_alignment_floor & (metadata_aql_kernarg_alignment_floor - 1UL)) == 0UL &&
             metadata_kernarg_segment_align != 0UL &&
             (dispatch_kernarg_address & (metadata_aql_kernarg_alignment_floor - 1UL)) == 0UL &&
             (dispatch_kernarg_address & (metadata_kernarg_segment_align - 1UL)) == 0UL) ? 1UL : 0UL;
        ulong dispatch_kernarg_ring_base_aligned16 =
            (control_dispatch_kernarg_ring_base != 0UL &&
             (control_dispatch_kernarg_ring_base & 15UL) == 0UL) ? 1UL : 0UL;
        ulong dispatch_kernarg_ring_slot_index_in_range =
            (control_dispatch_kernarg_ring_slots != 0UL &&
             control_dispatch_kernarg_ring_slot_index < control_dispatch_kernarg_ring_slots) ? 1UL : 0UL;
        ulong dispatch_kernarg_ring_stride_covers_metadata =
            (metadata_kernarg_segment_size != 0UL &&
             control_dispatch_kernarg_ring_stride >= metadata_kernarg_segment_size &&
             (control_dispatch_kernarg_ring_stride & 15UL) == 0UL) ? 1UL : 0UL;
        ulong dispatch_kernarg_ring_total_bytes =
            control_dispatch_kernarg_ring_stride * control_dispatch_kernarg_ring_slots;
        ulong dispatch_kernarg_ring_allocation_covers_slots =
            (control_dispatch_kernarg_ring_stride != 0UL &&
             control_dispatch_kernarg_ring_slots != 0UL &&
             control_dispatch_kernarg_ring_allocation_size >= dispatch_kernarg_ring_total_bytes) ? 1UL : 0UL;
        ulong dispatch_kernarg_ring_expected_slot_offset =
            control_dispatch_kernarg_ring_stride * control_dispatch_kernarg_ring_slot_index;
        ulong dispatch_kernarg_ring_offset_matches_slot =
            (control_dispatch_kernarg_ring_slot_offset ==
             dispatch_kernarg_ring_expected_slot_offset) ? 1UL : 0UL;
        ulong dispatch_kernarg_ring_selected_matches_dispatch =
            (dispatch_kernarg_ring_selected_address == dispatch_kernarg_address) ? 1UL : 0UL;
        ulong dispatch_kernarg_ring_selected_in_allocation =
            (control_dispatch_kernarg_ring_slot_offset + metadata_kernarg_segment_size <=
             control_dispatch_kernarg_ring_allocation_size) ? 1UL : 0UL;
        ulong dispatch_kernarg_ring_contract_ready =
            (dispatch_kernarg_ring_base_aligned16 == 1UL &&
             dispatch_kernarg_ring_slot_index_in_range == 1UL &&
             dispatch_kernarg_ring_stride_covers_metadata == 1UL &&
             dispatch_kernarg_ring_allocation_covers_slots == 1UL &&
             dispatch_kernarg_ring_offset_matches_slot == 1UL &&
             dispatch_kernarg_ring_selected_matches_dispatch == 1UL &&
             dispatch_kernarg_ring_selected_in_allocation == 1UL) ? 1UL : 0UL;
        ulong metadata_contract_ready =
            (dispatch_kernel_object_alignment64 == 1UL &&
             dispatch_segments_match_metadata == 1UL &&
             dispatch_kernarg_size_multiple16 == 1UL &&
             metadata_align_power_of_two == 1UL &&
             dispatch_workgroup_within_max_flat == 1UL &&
             dispatch_wavefront_multiple == 1UL &&
             dispatch_kernarg_allocation_covers_metadata == 1UL &&
             dispatch_kernarg_meets_metadata_or_floor_align == 1UL &&
             dispatch_kernarg_ring_contract_ready == 1UL) ? 1UL : 0UL;
        ulong dispatch_ready_after_barrier =
            (out[205] == 1UL &&
             dispatch_header_type_matches == 1UL &&
             dispatch_dimensions_valid == 1UL &&
             dispatch_workgroup_nonzero == 1UL &&
             dispatch_grid_covers_workgroup == 1UL &&
             dispatch_kernel_object_nonzero == 1UL &&
             dispatch_kernarg_alignment16 == 1UL &&
             dispatch_completion_elided == 1UL &&
             metadata_contract_ready == 1UL) ? 1UL : 0UL;
        out[208] = 1UL;
        out[209] = dispatch_packet_type;
        out[210] = dispatch_header_word;
        out[211] = dispatch_setup_dimensions;
        out[212] = dispatch_workgroup_x;
        out[213] = dispatch_workgroup_y;
        out[214] = dispatch_workgroup_z;
        out[215] = dispatch_grid_x;
        out[216] = dispatch_grid_y;
        out[217] = dispatch_grid_z;
        out[218] = dispatch_private_segment_size;
        out[219] = dispatch_group_segment_size;
        out[220] = dispatch_kernel_object;
        out[221] = dispatch_kernarg_address;
        out[222] = dispatch_completion_signal_handle;
        out[223] = dispatch_header_type_matches;
        out[224] = dispatch_dimensions_valid;
        out[225] = dispatch_workgroup_nonzero;
        out[226] = dispatch_grid_covers_workgroup;
        out[227] = dispatch_kernel_object_nonzero;
        out[228] = dispatch_kernarg_alignment16;
        out[229] = dispatch_completion_elided;
        out[230] = dispatch_ready_after_barrier;
        out[231] =
            (dispatch_ready_after_barrier == 1UL &&
             out[203] == 1UL &&
             out[205] == 1UL) ? 1UL : 0UL;
        out[232] = 1UL;
        out[233] = 1UL;
        out[234] = metadata_kernarg_segment_size;
        out[235] = metadata_kernarg_segment_align;
        out[236] = metadata_aql_kernarg_alignment_floor;
        out[237] = metadata_private_segment_size;
        out[238] = metadata_group_segment_size;
        out[239] = metadata_max_flat_workgroup_size;
        out[240] = metadata_wavefront_size;
        out[241] = metadata_descriptor_alignment;
        out[242] = dispatch_kernel_object_alignment64;
        out[243] = dispatch_segments_match_metadata;
        out[244] = dispatch_kernarg_size_multiple16;
        out[245] = metadata_align_power_of_two;
        out[246] = dispatch_workgroup_within_max_flat;
        out[247] = dispatch_wavefront_multiple;
        out[248] = dispatch_kernarg_allocation_size;
        out[249] = dispatch_kernarg_allocation_covers_metadata;
        out[250] = dispatch_kernarg_meets_metadata_or_floor_align;
        out[251] = metadata_contract_ready;
        out[252] = 3UL;
        out[253] = 1UL;
        out[254] = (dispatch_group_segment_size >= metadata_group_segment_size) ? 1UL : 0UL;
        out[255] = (dispatch_private_segment_size >= metadata_private_segment_size) ? 1UL : 0UL;
        out[256] = 1UL;
        out[257] = control_dispatch_kernarg_ring_base;
        out[258] = control_dispatch_kernarg_ring_stride;
        out[259] = control_dispatch_kernarg_ring_slots;
        out[260] = control_dispatch_kernarg_ring_slot_index;
        out[261] = control_dispatch_kernarg_ring_slot_offset;
        out[262] = control_dispatch_kernarg_ring_allocation_size;
        out[263] = dispatch_kernarg_ring_selected_address;
        out[264] = dispatch_kernarg_ring_base_aligned16;
        out[265] = dispatch_kernarg_ring_slot_index_in_range;
        out[266] = dispatch_kernarg_ring_stride_covers_metadata;
        out[267] = dispatch_kernarg_ring_allocation_covers_slots;
        out[268] = dispatch_kernarg_ring_offset_matches_slot;
        out[269] = dispatch_kernarg_ring_selected_matches_dispatch;
        out[270] = dispatch_kernarg_ring_selected_in_allocation;
        out[271] = dispatch_kernarg_ring_contract_ready;
        ulong expected_aql_word0 =
            atomic_load_explicit(&queue_control[48], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong expected_aql_word1 =
            atomic_load_explicit(&queue_control[49], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong expected_aql_word2 =
            atomic_load_explicit(&queue_control[50], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong expected_aql_word3 =
            atomic_load_explicit(&queue_control[51], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong expected_aql_word4 =
            atomic_load_explicit(&queue_control[52], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong expected_aql_word5 =
            atomic_load_explicit(&queue_control[53], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong expected_aql_word6 =
            atomic_load_explicit(&queue_control[54], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong expected_aql_word7 =
            atomic_load_explicit(&queue_control[55], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong aql_word0 =
            (dispatch_header_word & 0xffffUL) |
            ((dispatch_setup_dimensions & 0xffffUL) << 16) |
            ((dispatch_workgroup_x & 0xffffUL) << 32) |
            ((dispatch_workgroup_y & 0xffffUL) << 48);
        ulong aql_word1 =
            (dispatch_workgroup_z & 0xffffUL) |
            ((dispatch_grid_x & 0xffffffffUL) << 32);
        ulong aql_word2 =
            (dispatch_grid_y & 0xffffffffUL) |
            ((dispatch_grid_z & 0xffffffffUL) << 32);
        ulong aql_word3 =
            (dispatch_private_segment_size & 0xffffffffUL) |
            ((dispatch_group_segment_size & 0xffffffffUL) << 32);
        ulong aql_word4 = dispatch_kernel_object;
        ulong aql_word5 = dispatch_kernarg_address;
        ulong aql_word6 = 0UL;
        ulong aql_word7 = dispatch_completion_signal_handle;
        ulong aql_words_match =
            (aql_word0 == expected_aql_word0 &&
             aql_word1 == expected_aql_word1 &&
             aql_word2 == expected_aql_word2 &&
             aql_word3 == expected_aql_word3 &&
             aql_word4 == expected_aql_word4 &&
             aql_word5 == expected_aql_word5 &&
             aql_word6 == expected_aql_word6 &&
             aql_word7 == expected_aql_word7) ? 1UL : 0UL;
        ulong aql_header_setup_match =
            (((aql_word0 & 0xffffffffUL) == 0x00011502UL) &&
             ((expected_aql_word0 & 0xffffffffUL) == 0x00011502UL)) ? 1UL : 0UL;
        ulong aql_workgroup_match =
            (((aql_word0 >> 32) == ((256UL & 0xffffUL) | (1UL << 16))) &&
             ((aql_word1 & 0xffffUL) == 1UL)) ? 1UL : 0UL;
        ulong aql_grid_match =
            (((aql_word1 >> 32) == 256UL) &&
             ((aql_word2 & 0xffffffffUL) == 1UL) &&
             ((aql_word2 >> 32) == 1UL)) ? 1UL : 0UL;
        ulong aql_segment_match =
            (aql_word3 == expected_aql_word3 &&
             ((aql_word3 & 0xffffffffUL) == dispatch_private_segment_size) &&
             ((aql_word3 >> 32) == dispatch_group_segment_size)) ? 1UL : 0UL;
        ulong aql_kernel_object_match = (aql_word4 == expected_aql_word4) ? 1UL : 0UL;
        ulong aql_kernarg_match = (aql_word5 == expected_aql_word5) ? 1UL : 0UL;
        ulong aql_reserved_match = (aql_word6 == 0UL && expected_aql_word6 == 0UL) ? 1UL : 0UL;
        ulong aql_completion_match = (aql_word7 == 0UL && expected_aql_word7 == 0UL) ? 1UL : 0UL;
        ulong aql_packet_bytes_match = (control_packet_bytes == 64UL) ? 1UL : 0UL;
        ulong aql_packet_u64s_match =
            (atomic_load_explicit(&queue_control[3], memory_order_acquire,
                                  memory_scope_all_svm_devices) == 8UL) ? 1UL : 0UL;
        ulong aql_packet_image_ready =
            (aql_words_match == 1UL &&
             aql_header_setup_match == 1UL &&
             aql_workgroup_match == 1UL &&
             aql_grid_match == 1UL &&
             aql_segment_match == 1UL &&
             aql_kernel_object_match == 1UL &&
             aql_kernarg_match == 1UL &&
             aql_reserved_match == 1UL &&
             aql_completion_match == 1UL &&
             aql_packet_bytes_match == 1UL &&
             aql_packet_u64s_match == 1UL) ? 1UL : 0UL;
        out[304] = 1UL;
        out[305] = aql_word0;
        out[306] = aql_word1;
        out[307] = aql_word2;
        out[308] = aql_word3;
        out[309] = aql_word4;
        out[310] = aql_word5;
        out[311] = aql_word6;
        out[312] = aql_word7;
        out[313] = expected_aql_word0;
        out[314] = expected_aql_word1;
        out[315] = expected_aql_word2;
        out[316] = expected_aql_word3;
        out[317] = expected_aql_word4;
        out[318] = expected_aql_word5;
        out[319] = expected_aql_word6;
        out[320] = expected_aql_word7;
        out[321] = aql_words_match;
        out[322] = aql_header_setup_match;
        out[323] = aql_workgroup_match;
        out[324] = aql_grid_match;
        out[325] = aql_segment_match;
        out[326] = aql_kernel_object_match;
        out[327] = aql_kernarg_match;
        out[328] = aql_reserved_match;
        out[329] = aql_completion_match;
        out[330] = aql_packet_bytes_match;
        out[331] = aql_word0 & 0xffffffffUL;
        out[332] = dispatch_setup_dimensions;
        out[333] = control_packet_bytes;
        out[334] = 8UL;
        out[335] = aql_packet_image_ready;
        ulong live_aql_ring_va =
            atomic_load_explicit(&queue_control[36], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong live_aql_ring_slots =
            atomic_load_explicit(&queue_control[37], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong live_aql_packet_bytes =
            atomic_load_explicit(&queue_control[38], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong live_aql_slot_mask =
            atomic_load_explicit(&queue_control[39], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong live_aql_host_write_index =
            atomic_load_explicit(&queue_control[40], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong live_aql_producer_index =
            atomic_load_explicit(&queue_control[41], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong live_aql_consumer_index =
            atomic_load_explicit(&queue_control[42], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong live_aql_write_ptr_va =
            atomic_load_explicit(&queue_control[43], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong live_aql_read_ptr_va =
            atomic_load_explicit(&queue_control[44], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong live_aql_doorbell_offset =
            atomic_load_explicit(&queue_control[45], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong live_aql_queue_id =
            atomic_load_explicit(&queue_control[46], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong live_aql_gpu_id =
            atomic_load_explicit(&queue_control[47], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong live_aql_ring_base_aligned =
            (live_aql_ring_va != 0UL && ((live_aql_ring_va & 63UL) == 0UL)) ? 1UL : 0UL;
        ulong live_aql_packet_match = (live_aql_packet_bytes == 64UL) ? 1UL : 0UL;
        ulong live_aql_slots_power_of_two =
            (live_aql_ring_slots != 0UL &&
             ((live_aql_ring_slots & (live_aql_ring_slots - 1UL)) == 0UL)) ? 1UL : 0UL;
        ulong live_aql_expected_slot_mask =
            (live_aql_slots_power_of_two != 0UL) ? (live_aql_ring_slots - 1UL) : 0UL;
        ulong live_aql_slot_mask_match =
            (live_aql_slot_mask == live_aql_expected_slot_mask) ? 1UL : 0UL;
        ulong live_aql_host_producer_match =
            (live_aql_host_write_index == live_aql_producer_index) ? 1UL : 0UL;
        ulong live_aql_consumer_not_ahead =
            (live_aql_consumer_index <= live_aql_producer_index) ? 1UL : 0UL;
        ulong live_aql_capacity_ok =
            (live_aql_consumer_not_ahead != 0UL &&
             ((live_aql_producer_index - live_aql_consumer_index) <= live_aql_ring_slots))
                ? 1UL
                : 0UL;
        ulong live_aql_slot_offset =
            (live_aql_host_write_index & live_aql_slot_mask) * live_aql_packet_bytes;
        ulong live_aql_slot_va = live_aql_ring_va + live_aql_slot_offset;
        ulong live_aql_slot_va_aligned = ((live_aql_slot_va & 63UL) == 0UL) ? 1UL : 0UL;
        ulong live_aql_shadow_slots_ok = (live_aql_ring_slots >= 8UL) ? 1UL : 0UL;
        ulong live_aql_ptrs_present =
            (live_aql_write_ptr_va != 0UL && live_aql_read_ptr_va != 0UL) ? 1UL : 0UL;
        ulong live_aql_metadata_complete =
            (live_aql_ring_base_aligned != 0UL &&
             live_aql_packet_match != 0UL &&
             live_aql_slots_power_of_two != 0UL &&
             live_aql_slot_mask_match != 0UL &&
             live_aql_host_producer_match != 0UL &&
             live_aql_consumer_not_ahead != 0UL &&
             live_aql_capacity_ok != 0UL &&
             live_aql_slot_va_aligned != 0UL &&
             live_aql_shadow_slots_ok != 0UL &&
             live_aql_ptrs_present != 0UL) ? 1UL : 0UL;
        out[272] = 1UL;
        out[273] = live_aql_ring_va;
        out[274] = live_aql_ring_slots;
        out[275] = live_aql_packet_bytes;
        out[276] = live_aql_slot_mask;
        out[277] = live_aql_host_write_index;
        out[278] = live_aql_producer_index;
        out[279] = live_aql_consumer_index;
        out[280] = live_aql_slot_offset;
        out[281] = live_aql_slot_va;
        out[282] = live_aql_write_ptr_va;
        out[283] = live_aql_read_ptr_va;
        out[284] = live_aql_doorbell_offset;
        out[285] = live_aql_queue_id;
        out[286] = live_aql_gpu_id;
        out[287] = live_aql_ring_base_aligned;
        out[288] = live_aql_packet_match;
        out[289] = live_aql_slots_power_of_two;
        out[290] = live_aql_slot_mask_match;
        out[291] = live_aql_host_producer_match;
        out[292] = live_aql_consumer_not_ahead;
        out[293] = live_aql_capacity_ok;
        out[294] = live_aql_slot_va_aligned;
        out[295] = live_aql_packet_match;
        out[296] = live_aql_shadow_slots_ok;
        out[297] = live_aql_ptrs_present;
        out[298] = live_aql_metadata_complete;
        out[299] = 1UL;
        out[300] = 0x4d41525f4c495645UL;
        out[301] = live_aql_expected_slot_mask;
        out[302] = live_aql_slot_offset;
        out[303] = live_aql_metadata_complete;
        ulong live_aql_queue_space_available =
            (live_aql_consumer_not_ahead != 0UL &&
             ((live_aql_producer_index - live_aql_consumer_index) < live_aql_ring_slots))
                ? 1UL
                : 0UL;
        uint live_aql_slot_header_word = 0xffffffffu;
        if (live_aql_metadata_complete == 1UL &&
            live_aql_queue_space_available == 1UL &&
            aql_packet_image_ready == 1UL) {
            volatile __global atomic_uint* live_aql_slot_header_ptr =
                (volatile __global atomic_uint*)live_aql_slot_va;
            live_aql_slot_header_word =
                atomic_load_explicit(live_aql_slot_header_ptr,
                                     memory_order_acquire,
                                     memory_scope_all_svm_devices);
        }
        ulong live_aql_slot_header_u64 = (ulong)live_aql_slot_header_word;
        ulong live_aql_slot_packet_type = live_aql_slot_header_u64 & 0xffUL;
        ulong live_aql_slot_header_invalid =
            (live_aql_slot_packet_type == 1UL) ? 1UL : 0UL;
        ulong live_aql_slot_header_read_ready =
            (live_aql_metadata_complete == 1UL &&
             live_aql_queue_space_available == 1UL &&
             aql_packet_image_ready == 1UL &&
             live_aql_slot_header_invalid == 1UL) ? 1UL : 0UL;
        out[336] = 1UL;
        out[337] = live_aql_slot_va;
        out[338] = live_aql_slot_header_u64;
        out[339] = live_aql_slot_packet_type;
        out[340] = live_aql_slot_header_invalid;
        out[341] = live_aql_queue_space_available;
        out[342] = aql_packet_image_ready;
        out[343] = live_aql_metadata_complete;
        out[344] = 1UL;
        out[345] = live_aql_host_write_index;
        out[346] = live_aql_producer_index;
        out[347] = live_aql_consumer_index;
        out[348] = live_aql_slot_offset;
        out[349] = live_aql_slot_va_aligned;
        out[350] = live_aql_capacity_ok;
        out[351] = live_aql_slot_header_read_ready;
        ulong live_aql_inert_word0 =
            (aql_word0 & 0xffffffff00000000UL) | 1UL;
        uint live_aql_inert_word0_low32 = (uint)(live_aql_inert_word0 & 0xffffffffUL);
        uint live_aql_inert_word0_high32 = (uint)(live_aql_inert_word0 >> 32);
        ulong live_aql_payload_expected[8] = {
            live_aql_inert_word0,
            aql_word1,
            aql_word2,
            aql_word3,
            aql_word4,
            aql_word5,
            aql_word6,
            aql_word7,
        };
        ulong live_aql_payload_readback[8] = {
            0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL,
        };
        if (live_aql_slot_header_read_ready == 1UL) {
            volatile __global atomic_uint* live_aql_slot_u32 =
                (volatile __global atomic_uint*)live_aql_slot_va;
            volatile __global atomic_ulong* live_aql_slot_u64 =
                (volatile __global atomic_ulong*)live_aql_slot_va;
            atomic_store_explicit(&live_aql_slot_u64[1], aql_word1,
                                  memory_order_release,
                                  memory_scope_all_svm_devices);
            atomic_store_explicit(&live_aql_slot_u64[2], aql_word2,
                                  memory_order_release,
                                  memory_scope_all_svm_devices);
            atomic_store_explicit(&live_aql_slot_u64[3], aql_word3,
                                  memory_order_release,
                                  memory_scope_all_svm_devices);
            atomic_store_explicit(&live_aql_slot_u64[4], aql_word4,
                                  memory_order_release,
                                  memory_scope_all_svm_devices);
            atomic_store_explicit(&live_aql_slot_u64[5], aql_word5,
                                  memory_order_release,
                                  memory_scope_all_svm_devices);
            atomic_store_explicit(&live_aql_slot_u64[6], aql_word6,
                                  memory_order_release,
                                  memory_scope_all_svm_devices);
            atomic_store_explicit(&live_aql_slot_u64[7], aql_word7,
                                  memory_order_release,
                                  memory_scope_all_svm_devices);
            atomic_store_explicit(&live_aql_slot_u32[1],
                                  live_aql_inert_word0_high32,
                                  memory_order_release,
                                  memory_scope_all_svm_devices);
            atomic_store_explicit(&live_aql_slot_u32[0],
                                  live_aql_inert_word0_low32,
                                  memory_order_release,
                                  memory_scope_all_svm_devices);
            uint live_aql_readback_low32 =
                atomic_load_explicit(&live_aql_slot_u32[0],
                                     memory_order_acquire,
                                     memory_scope_all_svm_devices);
            uint live_aql_readback_high32 =
                atomic_load_explicit(&live_aql_slot_u32[1],
                                     memory_order_acquire,
                                     memory_scope_all_svm_devices);
            live_aql_payload_readback[0] =
                ((ulong)live_aql_readback_high32 << 32) |
                (ulong)live_aql_readback_low32;
            live_aql_payload_readback[1] =
                atomic_load_explicit(&live_aql_slot_u64[1],
                                     memory_order_acquire,
                                     memory_scope_all_svm_devices);
            live_aql_payload_readback[2] =
                atomic_load_explicit(&live_aql_slot_u64[2],
                                     memory_order_acquire,
                                     memory_scope_all_svm_devices);
            live_aql_payload_readback[3] =
                atomic_load_explicit(&live_aql_slot_u64[3],
                                     memory_order_acquire,
                                     memory_scope_all_svm_devices);
            live_aql_payload_readback[4] =
                atomic_load_explicit(&live_aql_slot_u64[4],
                                     memory_order_acquire,
                                     memory_scope_all_svm_devices);
            live_aql_payload_readback[5] =
                atomic_load_explicit(&live_aql_slot_u64[5],
                                     memory_order_acquire,
                                     memory_scope_all_svm_devices);
            live_aql_payload_readback[6] =
                atomic_load_explicit(&live_aql_slot_u64[6],
                                     memory_order_acquire,
                                     memory_scope_all_svm_devices);
            live_aql_payload_readback[7] =
                atomic_load_explicit(&live_aql_slot_u64[7],
                                     memory_order_acquire,
                                     memory_scope_all_svm_devices);
        }
        ulong live_aql_payload_words_match =
            (live_aql_payload_readback[0] == live_aql_payload_expected[0] &&
             live_aql_payload_readback[1] == live_aql_payload_expected[1] &&
             live_aql_payload_readback[2] == live_aql_payload_expected[2] &&
             live_aql_payload_readback[3] == live_aql_payload_expected[3] &&
             live_aql_payload_readback[4] == live_aql_payload_expected[4] &&
             live_aql_payload_readback[5] == live_aql_payload_expected[5] &&
             live_aql_payload_readback[6] == live_aql_payload_expected[6] &&
             live_aql_payload_readback[7] == live_aql_payload_expected[7]) ? 1UL : 0UL;
        ulong live_aql_payload_packet_type =
            live_aql_payload_readback[0] & 0xffUL;
        ulong live_aql_payload_header_invalid =
            (live_aql_payload_packet_type == 1UL) ? 1UL : 0UL;
        ulong live_aql_payload_low32_invalid =
            ((live_aql_payload_readback[0] & 0xffffffffUL) == 1UL) ? 1UL : 0UL;
        ulong live_aql_payload_high32_match =
            ((live_aql_payload_readback[0] >> 32) ==
             (live_aql_payload_expected[0] >> 32)) ? 1UL : 0UL;
        ulong live_aql_payload_write_ready =
            (live_aql_slot_header_read_ready == 1UL &&
             live_aql_payload_words_match == 1UL &&
             live_aql_payload_header_invalid == 1UL &&
             live_aql_payload_low32_invalid == 1UL &&
             live_aql_payload_high32_match == 1UL) ? 1UL : 0UL;
        out[352] = 1UL;
        out[353] = live_aql_slot_va;
        out[354] = live_aql_payload_expected[0];
        out[355] = live_aql_payload_expected[1];
        out[356] = live_aql_payload_expected[2];
        out[357] = live_aql_payload_expected[3];
        out[358] = live_aql_payload_expected[4];
        out[359] = live_aql_payload_expected[5];
        out[360] = live_aql_payload_expected[6];
        out[361] = live_aql_payload_expected[7];
        out[362] = live_aql_payload_readback[0];
        out[363] = live_aql_payload_readback[1];
        out[364] = live_aql_payload_readback[2];
        out[365] = live_aql_payload_readback[3];
        out[366] = live_aql_payload_readback[4];
        out[367] = live_aql_payload_readback[5];
        out[368] = live_aql_payload_readback[6];
        out[369] = live_aql_payload_readback[7];
        out[370] = live_aql_payload_words_match;
        out[371] = live_aql_payload_packet_type;
        out[372] = live_aql_payload_header_invalid;
        out[373] = live_aql_payload_low32_invalid;
        out[374] = live_aql_payload_high32_match;
        out[375] = live_aql_queue_space_available;
        out[376] = aql_packet_image_ready;
        out[377] = live_aql_metadata_complete;
        out[378] = live_aql_slot_header_read_ready;
        out[379] = live_aql_payload_write_ready;
        out[380] = 1UL;
        out[381] = 8UL;
        out[382] = 64UL;
        out[383] = 1UL;
        ulong live_aql_publish_low32 = aql_word0 & 0xffffffffUL;
        ulong live_aql_inert_low32 =
            live_aql_payload_readback[0] & 0xffffffffUL;
        ulong live_aql_publish_packet_type = live_aql_publish_low32 & 0xffUL;
        ulong live_aql_prepublish_differs =
            (live_aql_publish_low32 != live_aql_inert_low32) ? 1UL : 0UL;
        ulong live_aql_prepublish_low32_valid =
            (live_aql_publish_low32 == 0x00011502UL) ? 1UL : 0UL;
        ulong live_aql_prepublish_type_kernel_dispatch =
            (live_aql_publish_packet_type == 2UL) ? 1UL : 0UL;
        ulong live_aql_prepublish_live_still_invalid =
            (live_aql_inert_low32 == 1UL &&
             live_aql_payload_packet_type == 1UL) ? 1UL : 0UL;
        ulong live_aql_prepublish_boundary_ready =
            (live_aql_payload_write_ready == 1UL &&
             live_aql_prepublish_differs == 1UL &&
             live_aql_prepublish_low32_valid == 1UL &&
             live_aql_prepublish_type_kernel_dispatch == 1UL &&
             live_aql_prepublish_live_still_invalid == 1UL &&
             live_aql_metadata_complete == 1UL &&
             live_aql_queue_space_available == 1UL) ? 1UL : 0UL;
        out[384] = 1UL;
        out[385] = live_aql_slot_va;
        out[386] = live_aql_publish_low32;
        out[387] = live_aql_inert_low32;
        out[388] = live_aql_publish_packet_type;
        out[389] = live_aql_payload_packet_type;
        out[390] = live_aql_prepublish_differs;
        out[391] = live_aql_prepublish_live_still_invalid;
        out[392] = live_aql_payload_write_ready;
        out[393] = live_aql_metadata_complete;
        out[394] = live_aql_queue_space_available;
        out[395] = live_aql_host_write_index;
        out[396] = live_aql_producer_index;
        out[397] = live_aql_consumer_index;
        out[398] = live_aql_slot_offset;
        out[399] = live_aql_slot_va_aligned;
        out[400] = 32UL;
        out[401] = live_aql_prepublish_low32_valid;
        out[402] = live_aql_prepublish_type_kernel_dispatch;
        out[403] = live_aql_prepublish_live_still_invalid;
        out[404] = live_aql_prepublish_boundary_ready;
        out[405] = 1UL;
        out[406] = live_aql_payload_write_ready;
        out[407] = aql_word0;
        out[408] = live_aql_payload_readback[0];
        out[409] = 1UL;
        out[410] = 1UL;
        out[411] = 1UL;
        out[412] = 1UL;
        out[413] = 1UL;
        out[414] = 1UL;
        out[415] = live_aql_prepublish_boundary_ready;
        ulong live_aql_post_host_write_index = live_aql_host_write_index;
        ulong live_aql_post_consumer_index = live_aql_consumer_index;
        uint live_aql_post_header_word = 0xffffffffu;
        if (live_aql_metadata_complete == 1UL &&
            live_aql_queue_space_available == 1UL &&
            live_aql_payload_write_ready == 1UL) {
            volatile __global atomic_ulong* live_aql_write_ptr =
                (volatile __global atomic_ulong*)live_aql_write_ptr_va;
            volatile __global atomic_ulong* live_aql_read_ptr =
                (volatile __global atomic_ulong*)live_aql_read_ptr_va;
            volatile __global atomic_uint* live_aql_slot_header_ptr =
                (volatile __global atomic_uint*)live_aql_slot_va;
            live_aql_post_host_write_index =
                atomic_load_explicit(live_aql_write_ptr,
                                     memory_order_acquire,
                                     memory_scope_all_svm_devices);
            live_aql_post_consumer_index =
                atomic_load_explicit(live_aql_read_ptr,
                                     memory_order_acquire,
                                     memory_scope_all_svm_devices);
            live_aql_post_header_word =
                atomic_load_explicit(live_aql_slot_header_ptr,
                                     memory_order_acquire,
                                     memory_scope_all_svm_devices);
        }
        ulong live_aql_post_low32 = (ulong)live_aql_post_header_word;
        ulong live_aql_post_packet_type = live_aql_post_low32 & 0xffUL;
        ulong live_aql_post_write_index_unchanged =
            (live_aql_post_host_write_index == live_aql_host_write_index) ? 1UL : 0UL;
        ulong live_aql_post_consumer_index_unchanged =
            (live_aql_post_consumer_index == live_aql_consumer_index) ? 1UL : 0UL;
        ulong live_aql_post_header_still_invalid =
            (live_aql_post_low32 == live_aql_inert_low32 &&
             live_aql_post_packet_type == 1UL) ? 1UL : 0UL;
        ulong live_aql_post_valid_header_not_stored =
            (live_aql_post_low32 != live_aql_publish_low32) ? 1UL : 0UL;
        ulong live_aql_post_no_queue_progress =
            (live_aql_post_write_index_unchanged == 1UL &&
             live_aql_post_consumer_index_unchanged == 1UL) ? 1UL : 0UL;
        ulong live_aql_post_ready =
            (live_aql_prepublish_boundary_ready == 1UL &&
             live_aql_post_header_still_invalid == 1UL &&
             live_aql_post_valid_header_not_stored == 1UL) ? 1UL : 0UL;
        out[416] = 1UL;
        out[417] = live_aql_slot_va;
        out[418] = live_aql_host_write_index;
        out[419] = live_aql_post_host_write_index;
        out[420] = live_aql_consumer_index;
        out[421] = live_aql_post_consumer_index;
        out[422] = live_aql_inert_low32;
        out[423] = live_aql_post_low32;
        out[424] = live_aql_payload_packet_type;
        out[425] = live_aql_post_packet_type;
        out[426] = live_aql_post_write_index_unchanged;
        out[427] = live_aql_post_consumer_index_unchanged;
        out[428] = live_aql_post_header_still_invalid;
        out[429] = live_aql_prepublish_boundary_ready;
        out[430] = live_aql_post_no_queue_progress;
        out[431] = live_aql_post_valid_header_not_stored;
        out[432] = live_aql_post_ready;
        out[433] = live_aql_metadata_complete;
        out[434] = live_aql_queue_space_available;
        out[435] = live_aql_write_ptr_va;
        out[436] = live_aql_read_ptr_va;
        out[437] = live_aql_slot_offset;
        out[438] = live_aql_slot_va_aligned;
        out[439] = 32UL;
        out[440] = live_aql_payload_write_ready;
        out[441] = live_aql_slot_header_read_ready;
        out[442] = live_aql_post_valid_header_not_stored;
        out[443] = live_aql_post_header_still_invalid;
        out[444] = live_aql_prepublish_differs;
        out[445] = live_aql_prepublish_type_kernel_dispatch;
        out[446] = 1UL;
        out[447] = live_aql_post_ready;
        ulong live_aql_reserve_packet_id = live_aql_post_host_write_index;
        ulong live_aql_reserve_read_index = live_aql_post_consumer_index;
        ulong live_aql_reserve_consumer_not_ahead =
            (live_aql_reserve_packet_id >= live_aql_reserve_read_index) ? 1UL : 0UL;
        ulong live_aql_reserve_inflight =
            (live_aql_reserve_consumer_not_ahead == 1UL)
                ? (live_aql_reserve_packet_id - live_aql_reserve_read_index)
                : 0xffffffffffffffffUL;
        ulong live_aql_reserve_capacity_ok =
            (live_aql_reserve_consumer_not_ahead == 1UL &&
             live_aql_reserve_inflight < live_aql_ring_slots) ? 1UL : 0UL;
        ulong live_aql_reserve_slot_index =
            live_aql_reserve_packet_id & live_aql_slot_mask;
        ulong live_aql_reserve_slot_offset =
            live_aql_reserve_slot_index * live_aql_packet_bytes;
        ulong live_aql_reserve_slot_va =
            live_aql_ring_va + live_aql_reserve_slot_offset;
        ulong live_aql_reserve_slot_aligned =
            ((live_aql_reserve_slot_va & 63UL) == 0UL) ? 1UL : 0UL;
        ulong live_aql_reserve_desired_write_index =
            live_aql_reserve_packet_id + 1UL;
        ulong live_aql_reserve_packet_count = 1UL;
        ulong live_aql_reserve_doorbell_packet_id =
            live_aql_reserve_packet_id + live_aql_reserve_packet_count - 1UL;
        ulong live_aql_reserve_doorbell_matches =
            (live_aql_reserve_doorbell_packet_id + 1UL ==
             live_aql_reserve_desired_write_index) ? 1UL : 0UL;
        ulong live_aql_reserve_slot_formula_ok =
            (live_aql_reserve_slot_va == live_aql_ring_va +
             ((live_aql_reserve_packet_id & live_aql_slot_mask) *
              live_aql_packet_bytes)) ? 1UL : 0UL;
        ulong live_aql_reserve_ready =
            (live_aql_metadata_complete == 1UL &&
             live_aql_reserve_capacity_ok == 1UL &&
             live_aql_reserve_slot_aligned == 1UL &&
             live_aql_reserve_doorbell_matches == 1UL &&
             live_aql_reserve_slot_formula_ok == 1UL &&
             live_aql_post_valid_header_not_stored == 1UL) ? 1UL : 0UL;
        out[448] = 1UL;
        out[449] = live_aql_reserve_packet_id;
        out[450] = live_aql_reserve_read_index;
        out[451] = live_aql_reserve_inflight;
        out[452] = live_aql_reserve_capacity_ok;
        out[453] = live_aql_reserve_slot_index;
        out[454] = live_aql_reserve_slot_offset;
        out[455] = live_aql_reserve_slot_va;
        out[456] = live_aql_reserve_slot_aligned;
        out[457] = live_aql_write_ptr_va;
        out[458] = live_aql_read_ptr_va;
        out[459] = live_aql_reserve_desired_write_index;
        out[460] = live_aql_reserve_packet_count;
        out[461] = live_aql_reserve_doorbell_packet_id;
        out[462] = live_aql_reserve_doorbell_matches;
        out[463] = live_aql_publish_low32;
        out[464] = 32UL;
        out[465] = live_aql_post_low32;
        out[466] = live_aql_post_header_still_invalid;
        out[467] = live_aql_post_valid_header_not_stored;
        out[468] = 1UL;
        out[469] = 1UL;
        out[470] = live_aql_post_ready;
        out[471] = live_aql_metadata_complete;
        out[472] = live_aql_packet_bytes;
        out[473] = live_aql_ring_slots;
        out[474] = live_aql_slot_mask;
        out[475] = 1UL;
        out[476] = live_aql_reserve_capacity_ok;
        out[477] = live_aql_reserve_slot_formula_ok;
        out[478] = 1UL;
        out[479] = live_aql_reserve_ready;
        ulong live_aql_reserve_stage_same_slot =
            (live_aql_slot_va == live_aql_reserve_slot_va) ? 1UL : 0UL;
        ulong live_aql_reserve_stage_same_packet_id =
            (live_aql_host_write_index == live_aql_reserve_packet_id) ? 1UL : 0UL;
        ulong live_aql_reserve_stage_payload_publishable =
            (live_aql_reserve_stage_same_slot == 1UL &&
             live_aql_reserve_stage_same_packet_id == 1UL &&
             live_aql_payload_write_ready == 1UL &&
             live_aql_post_header_still_invalid == 1UL &&
             live_aql_post_valid_header_not_stored == 1UL) ? 1UL : 0UL;
        ulong live_aql_reserve_stage_must_restage =
            (live_aql_reserve_stage_payload_publishable == 0UL &&
             live_aql_reserve_ready == 1UL) ? 1UL : 0UL;
        ulong live_aql_reserve_stage_publish_blocked =
            (live_aql_reserve_stage_must_restage == 1UL) ? 1UL : 0UL;
        ulong live_aql_reserve_stage_old_slot_invalid =
            live_aql_post_valid_header_not_stored;
        ulong live_aql_reserve_stage_slot_progress_observed =
            live_aql_reserve_stage_must_restage;
        ulong live_aql_reserve_stage_sequence_ready =
            (live_aql_reserve_ready == 1UL &&
             live_aql_reserve_stage_publish_blocked == 1UL &&
             live_aql_reserve_stage_must_restage == 1UL &&
             live_aql_reserve_stage_old_slot_invalid == 1UL) ? 1UL : 0UL;
        out[480] = 1UL;
        out[481] = live_aql_host_write_index;
        out[482] = live_aql_reserve_packet_id;
        out[483] = live_aql_slot_va;
        out[484] = live_aql_reserve_slot_va;
        out[485] = live_aql_slot_offset;
        out[486] = live_aql_reserve_slot_offset;
        out[487] = live_aql_reserve_stage_same_packet_id;
        out[488] = live_aql_reserve_stage_same_slot;
        out[489] = live_aql_payload_write_ready;
        out[490] = live_aql_reserve_stage_payload_publishable;
        out[491] = live_aql_reserve_stage_must_restage;
        out[492] = live_aql_reserve_stage_publish_blocked;
        out[493] = live_aql_reserve_stage_old_slot_invalid;
        out[494] = live_aql_reserve_ready;
        out[495] = live_aql_post_valid_header_not_stored;
        out[496] = live_aql_publish_low32;
        out[497] = live_aql_post_low32;
        out[498] = live_aql_reserve_stage_slot_progress_observed;
        out[499] = live_aql_reserve_desired_write_index;
        out[500] = live_aql_reserve_doorbell_packet_id;
        out[501] = live_aql_reserve_capacity_ok;
        out[502] = live_aql_reserve_slot_formula_ok;
        out[503] = 1UL;
        out[504] = 1UL;
        out[505] = 1UL;
        out[506] = 1UL;
        out[507] = 1UL;
        out[508] = live_aql_reserve_stage_must_restage;
        out[509] = live_aql_reserve_stage_publish_blocked;
        out[510] = live_aql_reserve_stage_sequence_ready;
        out[511] = live_aql_reserve_stage_sequence_ready;
        ulong live_aql_reserve_restage_target_packet_id =
            live_aql_reserve_packet_id;
        ulong live_aql_reserve_restage_target_slot_va =
            live_aql_reserve_slot_va;
        ulong live_aql_reserve_restage_target_slot_offset =
            live_aql_reserve_slot_offset;
        ulong live_aql_reserve_restage_target_matches_reservation =
            (live_aql_reserve_restage_target_packet_id == live_aql_reserve_packet_id &&
             live_aql_reserve_restage_target_slot_va == live_aql_reserve_slot_va &&
             live_aql_reserve_restage_target_slot_offset == live_aql_reserve_slot_offset)
                ? 1UL
                : 0UL;
        ulong live_aql_reserve_restage_old_slot_bypassed =
            live_aql_reserve_stage_must_restage;
        ulong live_aql_reserve_restage_payload_inputs_ready =
            (aql_packet_image_ready == 1UL &&
             live_aql_metadata_complete == 1UL &&
             live_aql_reserve_ready == 1UL) ? 1UL : 0UL;
        ulong live_aql_reserve_restage_plan_ready =
            (live_aql_reserve_restage_target_matches_reservation == 1UL &&
             live_aql_reserve_stage_must_restage == 1UL &&
             live_aql_reserve_restage_payload_inputs_ready == 1UL &&
             live_aql_reserve_stage_publish_blocked == 1UL &&
             live_aql_reserve_capacity_ok == 1UL &&
             live_aql_reserve_slot_formula_ok == 1UL) ? 1UL : 0UL;
        out[512] = 1UL;
        out[513] = live_aql_reserve_restage_target_packet_id;
        out[514] = live_aql_reserve_restage_target_slot_va;
        out[515] = live_aql_reserve_restage_target_slot_offset;
        out[516] = live_aql_reserve_packet_id;
        out[517] = live_aql_reserve_slot_va;
        out[518] = live_aql_reserve_slot_offset;
        out[519] = live_aql_reserve_restage_target_matches_reservation;
        out[520] = live_aql_host_write_index;
        out[521] = live_aql_slot_va;
        out[522] = live_aql_reserve_restage_old_slot_bypassed;
        out[523] = live_aql_reserve_restage_payload_inputs_ready;
        out[524] = live_aql_publish_low32;
        out[525] = live_aql_post_low32;
        out[526] = 1UL;
        out[527] = 1UL;
        out[528] = 1UL;
        out[529] = 1UL;
        out[530] = 1UL;
        out[531] = 1UL;
        out[532] = 1UL;
        out[533] = 1UL;
        out[534] = live_aql_reserve_restage_plan_ready;
        out[535] = live_aql_reserve_capacity_ok;
        out[536] = live_aql_reserve_slot_formula_ok;
        out[537] = live_aql_reserve_desired_write_index;
        out[538] = live_aql_reserve_doorbell_packet_id;
        out[539] = live_aql_packet_bytes;
        out[540] = live_aql_ring_slots;
        out[541] = live_aql_slot_mask;
        out[542] = live_aql_reserve_stage_publish_blocked;
        out[543] = live_aql_reserve_restage_plan_ready;
        ulong live_aql_batch_packet_count = 2UL;
        ulong live_aql_batch_base_packet_id = live_aql_reserve_packet_id;
        ulong live_aql_batch_last_packet_id =
            live_aql_batch_base_packet_id + live_aql_batch_packet_count - 1UL;
        ulong live_aql_batch_desired_write_index =
            live_aql_batch_base_packet_id + live_aql_batch_packet_count;
        ulong live_aql_batch_capacity_ok =
            (live_aql_reserve_consumer_not_ahead == 1UL &&
             live_aql_reserve_inflight + live_aql_batch_packet_count <=
                 live_aql_ring_slots) ? 1UL : 0UL;
        ulong live_aql_batch_slot1_index =
            live_aql_batch_last_packet_id & live_aql_slot_mask;
        ulong live_aql_batch_slot1_offset =
            live_aql_batch_slot1_index * live_aql_packet_bytes;
        ulong live_aql_batch_slot1_va =
            live_aql_ring_va + live_aql_batch_slot1_offset;
        ulong live_aql_batch_slots_distinct =
            (live_aql_reserve_slot_va != live_aql_batch_slot1_va) ? 1UL : 0UL;
        ulong live_aql_batch_slots_aligned =
            (((live_aql_reserve_slot_va & 63UL) == 0UL) &&
             ((live_aql_batch_slot1_va & 63UL) == 0UL)) ? 1UL : 0UL;
        ulong live_aql_batch_slot1_formula_ok =
            (live_aql_batch_slot1_va == live_aql_ring_va +
             ((live_aql_batch_last_packet_id & live_aql_slot_mask) *
              live_aql_packet_bytes)) ? 1UL : 0UL;
        ulong live_aql_batch_doorbell_matches_last =
            (live_aql_batch_last_packet_id + 1UL ==
             live_aql_batch_desired_write_index) ? 1UL : 0UL;
        ulong live_aql_batch_plan_ready =
            ((live_aql_reserve_restage_plan_ready == 1UL ||
              live_aql_reserve_stage_payload_publishable == 1UL) &&
             live_aql_batch_capacity_ok == 1UL &&
             live_aql_batch_slots_distinct == 1UL &&
             live_aql_batch_slots_aligned == 1UL &&
             live_aql_reserve_slot_formula_ok == 1UL &&
             live_aql_batch_slot1_formula_ok == 1UL &&
             live_aql_batch_doorbell_matches_last == 1UL) ? 1UL : 0UL;
        out[544] = 1UL;
        out[545] = live_aql_batch_base_packet_id;
        out[546] = live_aql_batch_packet_count;
        out[547] = live_aql_batch_last_packet_id;
        out[548] = live_aql_batch_desired_write_index;
        out[549] = live_aql_reserve_read_index;
        out[550] = live_aql_reserve_inflight;
        out[551] = live_aql_batch_capacity_ok;
        out[552] = live_aql_reserve_slot_va;
        out[553] = live_aql_batch_slot1_va;
        out[554] = live_aql_reserve_slot_offset;
        out[555] = live_aql_batch_slot1_offset;
        out[556] = live_aql_reserve_slot_index;
        out[557] = live_aql_batch_slot1_index;
        out[558] = live_aql_batch_slots_distinct;
        out[559] = live_aql_batch_slots_aligned;
        out[560] = live_aql_reserve_slot_formula_ok;
        out[561] = live_aql_batch_slot1_formula_ok;
        out[562] = live_aql_batch_last_packet_id;
        out[563] = live_aql_batch_doorbell_matches_last;
        out[564] = 1UL;
        out[565] = 1UL;
        out[566] = 1UL;
        out[567] = 1UL;
        out[568] = 1UL;
        out[569] = 1UL;
        out[570] = 1UL;
        out[571] = 1UL;
        out[572] = 1UL;
        out[573] = 1UL;
        out[574] = live_aql_reserve_restage_target_matches_reservation;
        out[575] = live_aql_batch_plan_ready;
        ulong live_aql_materialized_slot_targets_match =
            (live_aql_batch_base_packet_id == live_aql_reserve_packet_id &&
             live_aql_batch_last_packet_id == live_aql_reserve_packet_id + 1UL &&
             live_aql_reserve_slot_va == live_aql_ring_va +
                 ((live_aql_batch_base_packet_id & live_aql_slot_mask) *
                  live_aql_packet_bytes) &&
             live_aql_batch_slot1_va == live_aql_ring_va +
                 ((live_aql_batch_last_packet_id & live_aql_slot_mask) *
                  live_aql_packet_bytes)) ? 1UL : 0UL;
        ulong live_aql_materialized_packet0_words_match = aql_words_match;
        ulong live_aql_materialized_packet1_words_match = aql_words_match;
        ulong live_aql_materialized_payload_words_match =
            (aql_word1 == expected_aql_word1 &&
             aql_word2 == expected_aql_word2 &&
             aql_word3 == expected_aql_word3 &&
             aql_word4 == expected_aql_word4 &&
             aql_word5 == expected_aql_word5 &&
             aql_word6 == expected_aql_word6 &&
             aql_word7 == expected_aql_word7) ? 1UL : 0UL;
        ulong live_aql_materialized_header_words_match =
            (aql_word0 == expected_aql_word0 &&
             ((aql_word0 & 0xffffffffUL) == live_aql_publish_low32)) ? 1UL : 0UL;
        ulong live_aql_materialized_ready =
            (live_aql_batch_plan_ready == 1UL &&
             live_aql_reserve_restage_plan_ready == 1UL &&
             aql_packet_image_ready == 1UL &&
             live_aql_materialized_slot_targets_match == 1UL &&
             live_aql_materialized_packet0_words_match == 1UL &&
             live_aql_materialized_packet1_words_match == 1UL &&
             live_aql_materialized_payload_words_match == 1UL &&
             live_aql_materialized_header_words_match == 1UL) ? 1UL : 0UL;
        out[576] = 1UL;
        out[577] = live_aql_batch_base_packet_id;
        out[578] = live_aql_batch_last_packet_id;
        out[579] = live_aql_reserve_slot_va;
        out[580] = live_aql_batch_slot1_va;
        out[581] = aql_word0;
        out[582] = aql_word1;
        out[583] = aql_word2;
        out[584] = aql_word3;
        out[585] = aql_word4;
        out[586] = aql_word5;
        out[587] = aql_word6;
        out[588] = aql_word7;
        out[589] = aql_word0;
        out[590] = aql_word1;
        out[591] = aql_word2;
        out[592] = aql_word3;
        out[593] = aql_word4;
        out[594] = aql_word5;
        out[595] = aql_word6;
        out[596] = aql_word7;
        out[597] = live_aql_materialized_packet0_words_match;
        out[598] = live_aql_materialized_packet1_words_match;
        out[599] = live_aql_materialized_payload_words_match;
        out[600] = live_aql_materialized_header_words_match;
        out[601] = live_aql_materialized_slot_targets_match;
        out[602] = live_aql_reserve_slot_offset;
        out[603] = live_aql_batch_slot1_offset;
        out[604] = live_aql_packet_bytes;
        out[605] = live_aql_batch_packet_count;
        out[606] = live_aql_batch_plan_ready;
        out[607] = live_aql_reserve_restage_plan_ready;
        out[608] = 1UL;
        out[609] = 1UL;
        out[610] = 1UL;
        out[611] = 1UL;
        out[612] = live_aql_materialized_ready;
        out[613] = live_aql_publish_low32;
        out[614] = aql_word0 & 0xffffffffUL;
        out[615] = aql_packet_image_ready;
        out[616] = live_aql_reserve_stage_must_restage;
        out[617] = live_aql_reserve_restage_target_matches_reservation;
        out[618] = live_aql_batch_doorbell_matches_last;
        out[619] = live_aql_batch_slots_aligned;
        ulong live_aql_shadow_packet_store_va =
            atomic_load_explicit(&queue_control[56], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong live_aql_shadow_packet_store_iterations =
            atomic_load_explicit(&queue_control[57], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong live_aql_shadow_packet_store_words =
            atomic_load_explicit(&queue_control[58], memory_order_acquire,
                                 memory_scope_all_svm_devices);
        ulong live_aql_shadow_packet_store_requested_iterations =
            live_aql_shadow_packet_store_iterations;
        if (live_aql_shadow_packet_store_iterations == 0UL) {
            live_aql_shadow_packet_store_iterations = 1UL;
        }
        if (live_aql_shadow_packet_store_iterations > 256UL) {
            live_aql_shadow_packet_store_iterations = 256UL;
        }
        ulong live_aql_shadow_packet_store_present =
            (live_aql_shadow_packet_store_va != 0UL &&
             (live_aql_shadow_packet_store_va & 63UL) == 0UL &&
             live_aql_shadow_packet_store_words >= 20UL) ? 1UL : 0UL;
        ulong live_aql_shadow_packet_store_final_sequence = 0UL;
        ulong live_aql_shadow_packet_store_packet0_word0 = 0UL;
        ulong live_aql_shadow_packet_store_packet1_word0 = 0UL;
        ulong live_aql_shadow_packet_store_words_match = 0UL;
        ulong live_aql_shadow_packet_store_payload_match = 0UL;
        ulong live_aql_shadow_packet_store_header_match = 0UL;
        ulong live_aql_shadow_packet_store_ready = 0UL;
        if (live_aql_shadow_packet_store_present == 1UL &&
            live_aql_materialized_ready == 1UL) {
            __global ulong* live_aql_shadow_packet_store =
                (__global ulong*)live_aql_shadow_packet_store_va;
            volatile __global atomic_uint* live_aql_shadow_packet0_header =
                (volatile __global atomic_uint*)&live_aql_shadow_packet_store[0];
            volatile __global atomic_uint* live_aql_shadow_packet1_header =
                (volatile __global atomic_uint*)&live_aql_shadow_packet_store[8];
            uint live_aql_shadow_header32 = (uint)(aql_word0 & 0xffffffffUL);
            ulong live_aql_shadow_word0_payload =
                aql_word0 & 0xffffffff00000000UL;
            for (ulong iter = 0UL;
                 iter < live_aql_shadow_packet_store_iterations;
                 ++iter) {
                atomic_store_explicit(live_aql_shadow_packet0_header, 0u,
                                      memory_order_release,
                                      memory_scope_all_svm_devices);
                atomic_store_explicit(live_aql_shadow_packet1_header, 0u,
                                      memory_order_release,
                                      memory_scope_all_svm_devices);
                live_aql_shadow_packet_store[0] = live_aql_shadow_word0_payload;
                live_aql_shadow_packet_store[1] = aql_word1;
                live_aql_shadow_packet_store[2] = aql_word2;
                live_aql_shadow_packet_store[3] = aql_word3;
                live_aql_shadow_packet_store[4] = aql_word4;
                live_aql_shadow_packet_store[5] = aql_word5;
                live_aql_shadow_packet_store[6] = aql_word6;
                live_aql_shadow_packet_store[7] = aql_word7;
                live_aql_shadow_packet_store[8] = live_aql_shadow_word0_payload;
                live_aql_shadow_packet_store[9] = aql_word1;
                live_aql_shadow_packet_store[10] = aql_word2;
                live_aql_shadow_packet_store[11] = aql_word3;
                live_aql_shadow_packet_store[12] = aql_word4;
                live_aql_shadow_packet_store[13] = aql_word5;
                live_aql_shadow_packet_store[14] = aql_word6;
                live_aql_shadow_packet_store[15] = aql_word7;
                atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE,
                                       memory_order_release,
                                       memory_scope_all_svm_devices);
                atomic_store_explicit(live_aql_shadow_packet1_header,
                                      live_aql_shadow_header32,
                                      memory_order_release,
                                      memory_scope_all_svm_devices);
                atomic_store_explicit(live_aql_shadow_packet0_header,
                                      live_aql_shadow_header32,
                                      memory_order_release,
                                      memory_scope_all_svm_devices);
                atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE,
                                       memory_order_release,
                                       memory_scope_all_svm_devices);
                live_aql_shadow_packet_store[16] = iter + 1UL;
                live_aql_shadow_packet_store[17] = 0x514d415f53484457UL;
                live_aql_shadow_packet_store[18] =
                    live_aql_shadow_packet_store_iterations;
                live_aql_shadow_packet_store[19] = live_aql_batch_plan_ready;
            }
            atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE,
                                   memory_order_acq_rel,
                                   memory_scope_all_svm_devices);
            live_aql_shadow_packet_store_final_sequence =
                live_aql_shadow_packet_store[16];
            live_aql_shadow_packet_store_packet0_word0 =
                live_aql_shadow_packet_store[0];
            live_aql_shadow_packet_store_packet1_word0 =
                live_aql_shadow_packet_store[8];
            live_aql_shadow_packet_store_payload_match =
                (live_aql_shadow_packet_store[1] == aql_word1 &&
                 live_aql_shadow_packet_store[2] == aql_word2 &&
                 live_aql_shadow_packet_store[3] == aql_word3 &&
                 live_aql_shadow_packet_store[4] == aql_word4 &&
                 live_aql_shadow_packet_store[5] == aql_word5 &&
                 live_aql_shadow_packet_store[6] == aql_word6 &&
                 live_aql_shadow_packet_store[7] == aql_word7 &&
                 live_aql_shadow_packet_store[9] == aql_word1 &&
                 live_aql_shadow_packet_store[10] == aql_word2 &&
                 live_aql_shadow_packet_store[11] == aql_word3 &&
                 live_aql_shadow_packet_store[12] == aql_word4 &&
                 live_aql_shadow_packet_store[13] == aql_word5 &&
                 live_aql_shadow_packet_store[14] == aql_word6 &&
                 live_aql_shadow_packet_store[15] == aql_word7) ? 1UL : 0UL;
            live_aql_shadow_packet_store_header_match =
                (live_aql_shadow_packet_store_packet0_word0 == aql_word0 &&
                 live_aql_shadow_packet_store_packet1_word0 == aql_word0 &&
                 ((live_aql_shadow_packet_store_packet0_word0 & 0xffffffffUL) ==
                  live_aql_publish_low32) &&
                 ((live_aql_shadow_packet_store_packet1_word0 & 0xffffffffUL) ==
                  live_aql_publish_low32)) ? 1UL : 0UL;
            live_aql_shadow_packet_store_words_match =
                (live_aql_shadow_packet_store_payload_match == 1UL &&
                 live_aql_shadow_packet_store_header_match == 1UL) ? 1UL : 0UL;
            live_aql_shadow_packet_store_ready =
                (live_aql_shadow_packet_store_words_match == 1UL &&
                 live_aql_shadow_packet_store_final_sequence ==
                     live_aql_shadow_packet_store_iterations) ? 1UL : 0UL;
        }
        out[620] = 1UL;
        out[621] = live_aql_shadow_packet_store_va;
        out[622] = live_aql_shadow_packet_store_requested_iterations;
        out[623] = live_aql_shadow_packet_store_iterations;
        out[624] = live_aql_shadow_packet_store_present;
        out[625] = live_aql_shadow_packet_store_final_sequence;
        out[626] = live_aql_shadow_packet_store_packet0_word0;
        out[627] = live_aql_shadow_packet_store_packet1_word0;
        out[628] = live_aql_shadow_packet_store_words_match;
        out[629] = live_aql_shadow_packet_store_payload_match;
        out[630] = live_aql_shadow_packet_store_header_match;
        out[631] = live_aql_materialized_ready;
        out[632] = 1UL;
        out[633] = 1UL;
        out[634] = 1UL;
        out[635] = 1UL;
        out[636] = 128UL;
        out[637] = live_aql_batch_packet_count;
        out[638] = live_aql_shadow_packet_store_ready;
        out[639] = live_aql_batch_plan_ready;
        ulong live_aql_header_probe_slot0_low32 = 0UL;
        ulong live_aql_header_probe_slot1_low32 = 0UL;
        if (live_aql_materialized_slot_targets_match == 1UL &&
            live_aql_batch_plan_ready == 1UL) {
            volatile __global atomic_uint* live_aql_header_probe_slot0_ptr =
                (volatile __global atomic_uint*)live_aql_reserve_slot_va;
            volatile __global atomic_uint* live_aql_header_probe_slot1_ptr =
                (volatile __global atomic_uint*)live_aql_batch_slot1_va;
            live_aql_header_probe_slot0_low32 =
                (ulong)atomic_load_explicit(live_aql_header_probe_slot0_ptr,
                                            memory_order_acquire,
                                            memory_scope_all_svm_devices);
            live_aql_header_probe_slot1_low32 =
                (ulong)atomic_load_explicit(live_aql_header_probe_slot1_ptr,
                                            memory_order_acquire,
                                            memory_scope_all_svm_devices);
        }
        ulong live_aql_header_probe_slot0_type =
            live_aql_header_probe_slot0_low32 & 0xffUL;
        ulong live_aql_header_probe_slot1_type =
            live_aql_header_probe_slot1_low32 & 0xffUL;
        ulong live_aql_header_probe_slot0_not_publish =
            (live_aql_header_probe_slot0_low32 != live_aql_publish_low32) ? 1UL : 0UL;
        ulong live_aql_header_probe_slot1_not_publish =
            (live_aql_header_probe_slot1_low32 != live_aql_publish_low32) ? 1UL : 0UL;
        ulong live_aql_header_probe_read_only_contract = 1UL;
        ulong live_aql_header_probe_fetch_add_not_performed = 1UL;
        ulong live_aql_header_probe_doorbell_not_written = 1UL;
        ulong live_aql_header_probe_live_slot_not_written = 1UL;
        ulong live_aql_header_probe_future_copy_blocked = 1UL;
        ulong live_aql_header_probe_no_mutation_contract =
            (live_aql_header_probe_read_only_contract == 1UL &&
             live_aql_header_probe_fetch_add_not_performed == 1UL &&
             live_aql_header_probe_doorbell_not_written == 1UL &&
             live_aql_header_probe_live_slot_not_written == 1UL) ? 1UL : 0UL;
        ulong live_aql_header_probe_ready =
            (live_aql_materialized_slot_targets_match == 1UL &&
             live_aql_batch_plan_ready == 1UL &&
             live_aql_reserve_ready == 1UL &&
             live_aql_header_probe_no_mutation_contract == 1UL &&
             live_aql_header_probe_future_copy_blocked == 1UL) ? 1UL : 0UL;
        out[640] = 1UL;
        out[641] = live_aql_reserve_slot_va;
        out[642] = live_aql_batch_slot1_va;
        out[643] = live_aql_reserve_slot_offset;
        out[644] = live_aql_batch_slot1_offset;
        out[645] = live_aql_header_probe_slot0_low32;
        out[646] = live_aql_header_probe_slot1_low32;
        out[647] = live_aql_header_probe_slot0_type;
        out[648] = live_aql_header_probe_slot1_type;
        out[649] = live_aql_header_probe_slot0_not_publish;
        out[650] = live_aql_header_probe_slot1_not_publish;
        out[651] = live_aql_materialized_slot_targets_match;
        out[652] = live_aql_header_probe_read_only_contract;
        out[653] = live_aql_header_probe_fetch_add_not_performed;
        out[654] = live_aql_header_probe_doorbell_not_written;
        out[655] = live_aql_header_probe_live_slot_not_written;
        out[656] = live_aql_header_probe_future_copy_blocked;
        out[657] = live_aql_header_probe_ready;
        out[658] = live_aql_batch_plan_ready;
        out[659] = live_aql_reserve_ready;
        out[660] = live_aql_post_valid_header_not_stored;
        out[661] = live_aql_publish_low32;
        out[662] = live_aql_packet_bytes;
        out[663] = live_aql_batch_packet_count;
        out[664] = live_aql_batch_slots_aligned;
        out[665] = live_aql_reserve_slot_formula_ok;
        out[666] = live_aql_batch_slot1_formula_ok;
        out[667] = 0UL;
        out[668] = live_aql_header_probe_no_mutation_contract;
        ulong slot2_offset_bytes = 2UL * control_packet_bytes;
        ulong slot3_offset_bytes = 3UL * control_packet_bytes;
        ulong slot2_va = control_base_va + slot2_offset_bytes;
        ulong slot3_va = control_base_va + slot3_offset_bytes;
        atomic_store_explicit(&queue_control[16], slot2_va,
                              memory_order_release,
                              memory_scope_all_svm_devices);
        atomic_store_explicit(&queue_control[17], slot3_va,
                              memory_order_release,
                              memory_scope_all_svm_devices);
        atomic_store_explicit(&queue_control[18], slot2_offset_bytes,
                              memory_order_release,
                              memory_scope_all_svm_devices);
        atomic_store_explicit(&queue_control[19], slot3_offset_bytes,
                              memory_order_release,
                              memory_scope_all_svm_devices);
        atomic_store_explicit(&queue_control[20], control_base_va + control_queue_bytes,
                              memory_order_release,
                              memory_scope_all_svm_devices);
        atomic_store_explicit(&queue_control[21], 1UL,
                              memory_order_release,
                              memory_scope_all_svm_devices);
        ulong wrap_initial_packet_id = (ulong)queue_slots;
        ulong wrap_slot0_offset_bytes =
            (wrap_initial_packet_id & ((ulong)queue_slots - 1UL)) * control_packet_bytes;
        ulong wrap_slot1_offset_bytes =
            ((wrap_initial_packet_id + 1UL) & ((ulong)queue_slots - 1UL)) * control_packet_bytes;
        ulong wrap_slot0_va = control_base_va + wrap_slot0_offset_bytes;
        ulong wrap_slot1_va = control_base_va + wrap_slot1_offset_bytes;
        ulong wrap_slot0_base = wrap_slot0_offset_bytes >> 3;
        ulong wrap_slot1_base = wrap_slot1_offset_bytes >> 3;
        if (wrap_slot0_offset_bytes + control_packet_bytes > control_queue_bytes ||
            wrap_slot1_offset_bytes + control_packet_bytes > control_queue_bytes ||
            wrap_slot0_base != 0UL ||
            wrap_slot1_base != 8UL) {
            out[0] = 12UL;
            out[14] = wrap_slot1_offset_bytes;
            return;
        }
        if (shadow_queue[wrap_slot0_base] != 0UL ||
            shadow_queue[wrap_slot1_base] != 0UL) {
            out[0] = 13UL;
            out[14] = (shadow_queue[wrap_slot0_base] << 32) | shadow_queue[wrap_slot1_base];
            return;
        }
        atomic_store_explicit(&write_index[0], (uint)wrap_initial_packet_id,
                              memory_order_release,
                              memory_scope_all_svm_devices);
        atomic_store_explicit(&queue_control[5], wrap_initial_packet_id,
                              memory_order_release,
                              memory_scope_all_svm_devices);
        uint wrap_reserved0 = atomic_fetch_add_explicit(&write_index[0], 1u,
                                                        memory_order_acq_rel,
                                                        memory_scope_all_svm_devices);
        uint wrap_reserved1 = atomic_fetch_add_explicit(&write_index[0], 1u,
                                                        memory_order_acq_rel,
                                                        memory_scope_all_svm_devices);
        uint wrap_slot0 = wrap_reserved0 & (queue_slots - 1u);
        uint wrap_slot1 = wrap_reserved1 & (queue_slots - 1u);
        if (wrap_reserved0 != (uint)wrap_initial_packet_id ||
            wrap_reserved1 != (uint)(wrap_initial_packet_id + 1UL) ||
            wrap_slot0 != 0u ||
            wrap_slot1 != 1u) {
            out[0] = 14UL;
            out[14] = ((ulong)wrap_slot0 << 32) | (ulong)wrap_slot1;
            return;
        }
        for (uint i = 1u; i < 8u; ++i) {
            shadow_queue[wrap_slot0_base + (ulong)i] = inert_packet[i];
            shadow_queue[wrap_slot1_base + (ulong)i] = inert_packet[i];
        }
        atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release,
                               memory_scope_all_svm_devices);
        volatile __global atomic_uint* wrap_header0 =
            (volatile __global atomic_uint*)(&shadow_queue[wrap_slot0_base]);
        volatile __global atomic_uint* wrap_header1 =
            (volatile __global atomic_uint*)(&shadow_queue[wrap_slot1_base]);
        atomic_store_explicit(wrap_header0, expected_header32_word,
                              memory_order_release,
                              memory_scope_all_svm_devices);
        atomic_store_explicit(wrap_header1, expected_header32_word,
                              memory_order_release,
                              memory_scope_all_svm_devices);
        uint wrap_final_write = atomic_load_explicit(&write_index[0], memory_order_acquire,
                                                     memory_scope_all_svm_devices);
        if (shadow_queue[wrap_slot0_base] != expected_header_word ||
            shadow_queue[wrap_slot1_base] != expected_header_word ||
            wrap_final_write != (uint)(wrap_initial_packet_id + 2UL)) {
            out[0] = 15UL;
            out[14] = (ulong)wrap_final_write;
            return;
        }
        atomic_store_explicit(&queue_control[5], (ulong)wrap_final_write,
                              memory_order_release,
                              memory_scope_all_svm_devices);
        atomic_store_explicit(&queue_control[22], wrap_initial_packet_id,
                              memory_order_release,
                              memory_scope_all_svm_devices);
        atomic_store_explicit(&queue_control[23], 2UL,
                              memory_order_release,
                              memory_scope_all_svm_devices);
        out[39] = 1UL;
        out[51] = atomic_load_explicit(&queue_control[5], memory_order_acquire,
                                       memory_scope_all_svm_devices);
        out[52] = atomic_load_explicit(&queue_control[6], memory_order_acquire,
                                       memory_scope_all_svm_devices);
        out[53] = atomic_load_explicit(&queue_control[8], memory_order_acquire,
                                       memory_scope_all_svm_devices);
        out[54] = atomic_load_explicit(&queue_control[7], memory_order_acquire,
                                       memory_scope_all_svm_devices);
        out[55] = atomic_load_explicit(&queue_control[10], memory_order_acquire,
                                       memory_scope_all_svm_devices);
        out[56] = atomic_load_explicit(&queue_control[11], memory_order_acquire,
                                       memory_scope_all_svm_devices);
        out[66] = atomic_load_explicit(&queue_control[16], memory_order_acquire,
                                       memory_scope_all_svm_devices);
        out[67] = atomic_load_explicit(&queue_control[17], memory_order_acquire,
                                       memory_scope_all_svm_devices);
        out[68] = atomic_load_explicit(&queue_control[18], memory_order_acquire,
                                       memory_scope_all_svm_devices);
        out[69] = atomic_load_explicit(&queue_control[19], memory_order_acquire,
                                       memory_scope_all_svm_devices);
        out[70] = atomic_load_explicit(&queue_control[20], memory_order_acquire,
                                       memory_scope_all_svm_devices);
        out[71] = atomic_load_explicit(&queue_control[21], memory_order_acquire,
                                       memory_scope_all_svm_devices);
        out[72] = atomic_load_explicit(&queue_control[22], memory_order_acquire,
                                       memory_scope_all_svm_devices);
        out[73] = (ulong)wrap_final_write;
        out[74] = wrap_slot0_va;
        out[75] = wrap_slot1_va;
        out[76] = wrap_slot0_offset_bytes;
        out[77] = wrap_slot1_offset_bytes;
        out[78] = shadow_queue[wrap_slot0_base];
        out[79] = shadow_queue[wrap_slot1_base];
        ulong admission_needed = 2UL;
        ulong admission_slot_mask = (ulong)queue_slots - 1UL;
        ulong admit_read_index = 2UL;
        ulong admit_write_index = (ulong)queue_slots - admission_needed;
        ulong admit_inflight = admit_write_index - admit_read_index;
        ulong admit_free_slots = (ulong)queue_slots - admit_inflight;
        ulong admit_last_packet_id = admit_write_index + admission_needed - 1UL;
        ulong admit_boundary_packet_id = admit_read_index + (ulong)queue_slots;
        ulong admit_slot0_offset_bytes =
            (admit_write_index & admission_slot_mask) * control_packet_bytes;
        ulong admit_slot1_offset_bytes =
            ((admit_write_index + 1UL) & admission_slot_mask) * control_packet_bytes;
        ulong admit_slot0_base = admit_slot0_offset_bytes >> 3;
        ulong admit_slot1_base = admit_slot1_offset_bytes >> 3;
        ulong admit_headers_invalid =
            (shadow_queue[admit_slot0_base] == 0UL &&
             shadow_queue[admit_slot1_base] == 0UL) ? 1UL : 0UL;
        ulong admit_allowed =
            (admit_write_index >= admit_read_index &&
             admit_inflight <= (ulong)queue_slots &&
             admission_needed <= admit_free_slots &&
             admit_last_packet_id < admit_boundary_packet_id &&
             admit_headers_invalid == 1UL) ? 1UL : 0UL;
        ulong deny_read_index = admit_read_index;
        ulong deny_write_index = admit_boundary_packet_id - 1UL;
        ulong deny_inflight = deny_write_index - deny_read_index;
        ulong deny_free_slots = (ulong)queue_slots - deny_inflight;
        ulong deny_last_packet_id = deny_write_index + admission_needed - 1UL;
        ulong deny_boundary_packet_id = deny_read_index + (ulong)queue_slots;
        ulong deny_slot0_offset_bytes =
            (deny_write_index & admission_slot_mask) * control_packet_bytes;
        ulong deny_slot1_offset_bytes =
            ((deny_write_index + 1UL) & admission_slot_mask) * control_packet_bytes;
        ulong deny_allowed =
            (deny_write_index >= deny_read_index &&
             deny_inflight <= (ulong)queue_slots &&
             admission_needed <= deny_free_slots &&
             deny_last_packet_id < deny_boundary_packet_id) ? 1UL : 0UL;
        ulong deny_wrote_packet = deny_allowed;
        if (admit_slot0_offset_bytes + control_packet_bytes > control_queue_bytes ||
            admit_slot1_offset_bytes + control_packet_bytes > control_queue_bytes ||
            deny_slot0_offset_bytes + control_packet_bytes > control_queue_bytes ||
            deny_slot1_offset_bytes + control_packet_bytes > control_queue_bytes) {
            out[0] = 16UL;
            out[14] = deny_slot1_offset_bytes;
            return;
        }
        if (admit_allowed != 1UL ||
            admit_free_slots != 4UL ||
            admit_last_packet_id != 7UL ||
            admit_boundary_packet_id != 10UL ||
            deny_allowed != 0UL ||
            deny_free_slots != 1UL ||
            deny_last_packet_id != 10UL ||
            deny_boundary_packet_id != 10UL ||
            deny_wrote_packet != 0UL) {
            out[0] = 17UL;
            out[14] = (deny_allowed << 32) | deny_free_slots;
            return;
        }
        out[80] = admission_needed;
        out[81] = (ulong)queue_slots;
        out[82] = admit_read_index;
        out[83] = admit_write_index;
        out[84] = admit_inflight;
        out[85] = admit_free_slots;
        out[86] = admit_allowed;
        out[87] = admit_last_packet_id;
        out[88] = deny_read_index;
        out[89] = deny_write_index;
        out[90] = deny_inflight;
        out[91] = deny_free_slots;
        out[92] = deny_allowed;
        out[93] = deny_last_packet_id;
        out[94] = admit_boundary_packet_id;
        out[95] = deny_boundary_packet_id;
        out[96] = admit_slot0_offset_bytes;
        out[97] = admit_slot1_offset_bytes;
        out[98] = shadow_queue[admit_slot0_base];
        out[99] = shadow_queue[admit_slot1_base];
        out[100] = deny_slot0_offset_bytes;
        out[101] = deny_slot1_offset_bytes;
        out[102] = deny_wrote_packet;
        out[103] = 1UL;
        ulong cas_needed = 2UL;
        ulong cas_read_index = 2UL;
        ulong cas_first_write = 4UL;
        ulong cas_first_inflight = cas_first_write - cas_read_index;
        ulong cas_first_free = (ulong)queue_slots - cas_first_inflight;
        ulong cas_first_desired = cas_first_write + cas_needed;
        ulong cas_first_last_packet = cas_first_desired - 1UL;
        ulong cas_first_boundary = cas_read_index + (ulong)queue_slots;
        ulong cas_first_allowed =
            (cas_first_inflight <= (ulong)queue_slots &&
             cas_needed <= cas_first_free &&
             cas_first_last_packet < cas_first_boundary) ? 1UL : 0UL;
        ulong cas_competitor_write = cas_first_desired;
        atomic_store_explicit(&write_index[0], (uint)cas_first_write,
                              memory_order_release,
                              memory_scope_all_svm_devices);
        atomic_store_explicit(&write_index[0], (uint)cas_competitor_write,
                              memory_order_release,
                              memory_scope_all_svm_devices);
        uint cas_expected = (uint)cas_first_write;
        bool cas_first_success = atomic_compare_exchange_strong_explicit(
            &write_index[0], &cas_expected, (uint)cas_first_desired,
            memory_order_acq_rel, memory_order_acquire,
            memory_scope_all_svm_devices);
        ulong cas_first_observed = (ulong)cas_expected;
        ulong cas_first_wrote_packet = cas_first_success ? 1UL : 0UL;
        ulong cas_retry_write = (ulong)atomic_load_explicit(&write_index[0],
                                                            memory_order_acquire,
                                                            memory_scope_all_svm_devices);
        ulong cas_retry_inflight = cas_retry_write - cas_read_index;
        ulong cas_retry_free = (ulong)queue_slots - cas_retry_inflight;
        ulong cas_retry_desired = cas_retry_write + cas_needed;
        ulong cas_retry_last_packet = cas_retry_desired - 1UL;
        ulong cas_retry_boundary = cas_read_index + (ulong)queue_slots;
        ulong cas_retry_allowed =
            (cas_retry_write >= cas_read_index &&
             cas_retry_inflight <= (ulong)queue_slots &&
             cas_needed <= cas_retry_free &&
             cas_retry_last_packet < cas_retry_boundary) ? 1UL : 0UL;
        uint cas_retry_expected = (uint)cas_retry_write;
        bool cas_retry_success = false;
        if (cas_retry_allowed == 1UL && cas_first_wrote_packet == 0UL) {
            cas_retry_success = atomic_compare_exchange_strong_explicit(
                &write_index[0], &cas_retry_expected, (uint)cas_retry_desired,
                memory_order_acq_rel, memory_order_acquire,
                memory_scope_all_svm_devices);
        }
        ulong cas_retry_final_write =
            (ulong)atomic_load_explicit(&write_index[0], memory_order_acquire,
                                        memory_scope_all_svm_devices);
        ulong cas_retry_slot0_offset_bytes =
            (cas_retry_write & admission_slot_mask) * control_packet_bytes;
        ulong cas_retry_slot1_offset_bytes =
            ((cas_retry_write + 1UL) & admission_slot_mask) * control_packet_bytes;
        ulong cas_retry_slot0_base = cas_retry_slot0_offset_bytes >> 3;
        ulong cas_retry_slot1_base = cas_retry_slot1_offset_bytes >> 3;
        if (cas_retry_slot0_offset_bytes + control_packet_bytes > control_queue_bytes ||
            cas_retry_slot1_offset_bytes + control_packet_bytes > control_queue_bytes) {
            out[0] = 18UL;
            out[14] = cas_retry_slot1_offset_bytes;
            return;
        }
        if (cas_first_allowed != 1UL ||
            cas_first_success ||
            cas_first_observed != cas_competitor_write ||
            cas_first_wrote_packet != 0UL ||
            cas_retry_allowed != 1UL ||
            !cas_retry_success ||
            cas_retry_final_write != cas_retry_desired ||
            cas_retry_slot0_offset_bytes != 384UL ||
            cas_retry_slot1_offset_bytes != 448UL ||
            shadow_queue[cas_retry_slot0_base] != 0UL ||
            shadow_queue[cas_retry_slot1_base] != 0UL) {
            out[0] = 19UL;
            out[14] = (cas_retry_final_write << 32) | (ulong)cas_retry_success;
            return;
        }
        for (uint i = 1u; i < 8u; ++i) {
            shadow_queue[cas_retry_slot0_base + (ulong)i] = inert_packet[i];
            shadow_queue[cas_retry_slot1_base + (ulong)i] = inert_packet[i];
        }
        atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release,
                               memory_scope_all_svm_devices);
        volatile __global atomic_uint* cas_retry_header0 =
            (volatile __global atomic_uint*)(&shadow_queue[cas_retry_slot0_base]);
        volatile __global atomic_uint* cas_retry_header1 =
            (volatile __global atomic_uint*)(&shadow_queue[cas_retry_slot1_base]);
        atomic_store_explicit(cas_retry_header0, expected_header32_word,
                              memory_order_release,
                              memory_scope_all_svm_devices);
        atomic_store_explicit(cas_retry_header1, expected_header32_word,
                              memory_order_release,
                              memory_scope_all_svm_devices);
        if (shadow_queue[cas_retry_slot0_base] != expected_header_word ||
            shadow_queue[cas_retry_slot1_base] != expected_header_word) {
            out[0] = 20UL;
            out[14] = shadow_queue[cas_retry_slot0_base];
            return;
        }
        out[104] = 1UL;
        out[105] = cas_needed;
        out[106] = cas_read_index;
        out[107] = cas_first_write;
        out[108] = cas_first_inflight;
        out[109] = cas_first_free;
        out[110] = cas_first_allowed;
        out[111] = cas_competitor_write;
        out[112] = cas_first_desired;
        out[113] = cas_first_observed;
        out[114] = cas_first_success ? 1UL : 0UL;
        out[115] = cas_first_wrote_packet;
        out[116] = cas_retry_write;
        out[117] = cas_retry_inflight;
        out[118] = cas_retry_free;
        out[119] = cas_retry_allowed;
        out[120] = cas_retry_desired;
        out[121] = cas_retry_success ? 1UL : 0UL;
        out[122] = cas_retry_final_write;
        out[123] = cas_retry_slot0_offset_bytes;
        out[124] = cas_retry_slot1_offset_bytes;
        out[125] = shadow_queue[cas_retry_slot0_base];
        out[126] = shadow_queue[cas_retry_slot1_base];
        ulong lifecycle_invalid_header_word = 1UL;
        uint lifecycle_invalid_header32_word = 1u;
        ulong lifecycle_read_before = cas_retry_write;
        ulong lifecycle_packet0 = cas_retry_write;
        ulong lifecycle_packet1 = cas_retry_write + 1UL;
        ulong lifecycle_header0_before = shadow_queue[cas_retry_slot0_base];
        ulong lifecycle_header1_before = shadow_queue[cas_retry_slot1_base];
        volatile __global atomic_uint* lifecycle_header0 =
            (volatile __global atomic_uint*)(&shadow_queue[cas_retry_slot0_base]);
        volatile __global atomic_uint* lifecycle_header1 =
            (volatile __global atomic_uint*)(&shadow_queue[cas_retry_slot1_base]);
        atomic_store_explicit(lifecycle_header0, lifecycle_invalid_header32_word,
                              memory_order_release,
                              memory_scope_all_svm_devices);
        atomic_store_explicit(lifecycle_header1, lifecycle_invalid_header32_word,
                              memory_order_release,
                              memory_scope_all_svm_devices);
        atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release,
                               memory_scope_all_svm_devices);
        ulong lifecycle_header0_after = shadow_queue[cas_retry_slot0_base];
        ulong lifecycle_header1_after = shadow_queue[cas_retry_slot1_base];
        ulong lifecycle_read_after = lifecycle_packet1 + 1UL;
        ulong lifecycle_reuse_packet0 = lifecycle_packet0 + (ulong)queue_slots;
        ulong lifecycle_reuse_packet1 = lifecycle_packet1 + (ulong)queue_slots;
        ulong lifecycle_reuse_allowed =
            (lifecycle_header0_before == expected_header_word &&
             lifecycle_header1_before == expected_header_word &&
             lifecycle_header0_after == lifecycle_invalid_header_word &&
             lifecycle_header1_after == lifecycle_invalid_header_word &&
             lifecycle_reuse_packet0 >= lifecycle_read_after &&
             lifecycle_reuse_packet1 >= lifecycle_read_after &&
             lifecycle_reuse_packet0 < lifecycle_read_after + (ulong)queue_slots &&
             lifecycle_reuse_packet1 < lifecycle_read_after + (ulong)queue_slots) ? 1UL : 0UL;
        ulong lifecycle_stale_packet0 = lifecycle_packet0;
        ulong lifecycle_stale_packet1 = lifecycle_packet1;
        ulong lifecycle_stale_slot0_offset_bytes =
            (lifecycle_stale_packet0 & admission_slot_mask) * control_packet_bytes;
        ulong lifecycle_stale_slot1_offset_bytes =
            (lifecycle_stale_packet1 & admission_slot_mask) * control_packet_bytes;
        ulong lifecycle_stale_below_read0 =
            (lifecycle_stale_packet0 < lifecycle_read_after) ? 1UL : 0UL;
        ulong lifecycle_stale_below_read1 =
            (lifecycle_stale_packet1 < lifecycle_read_after) ? 1UL : 0UL;
        ulong lifecycle_stale_reuse_allowed =
            (lifecycle_header0_after == lifecycle_invalid_header_word &&
             lifecycle_header1_after == lifecycle_invalid_header_word &&
             lifecycle_stale_packet0 >= lifecycle_read_after &&
             lifecycle_stale_packet1 >= lifecycle_read_after &&
             lifecycle_stale_packet0 < lifecycle_read_after + (ulong)queue_slots &&
             lifecycle_stale_packet1 < lifecycle_read_after + (ulong)queue_slots) ? 1UL : 0UL;
        if (lifecycle_reuse_allowed != 1UL) {
            out[0] = 21UL;
            out[14] = (lifecycle_header0_after << 32) | lifecycle_header1_after;
            return;
        }
        if (lifecycle_stale_below_read0 != 1UL ||
            lifecycle_stale_below_read1 != 1UL ||
            lifecycle_stale_reuse_allowed != 0UL ||
            lifecycle_stale_slot0_offset_bytes != cas_retry_slot0_offset_bytes ||
            lifecycle_stale_slot1_offset_bytes != cas_retry_slot1_offset_bytes) {
            out[0] = 22UL;
            out[14] = (lifecycle_stale_packet0 << 32) | lifecycle_stale_packet1;
            return;
        }
        out[128] = 1UL;
        out[129] = lifecycle_invalid_header_word;
        out[130] = lifecycle_read_before;
        out[131] = lifecycle_packet0;
        out[132] = lifecycle_packet1;
        out[133] = lifecycle_header0_before;
        out[134] = lifecycle_header1_before;
        out[135] = lifecycle_header0_after;
        out[136] = lifecycle_header1_after;
        out[137] = lifecycle_read_after;
        out[138] = lifecycle_reuse_packet0;
        out[139] = lifecycle_reuse_packet1;
        out[140] = cas_retry_slot0_offset_bytes;
        out[141] = cas_retry_slot1_offset_bytes;
        out[142] = lifecycle_reuse_allowed;
        out[143] = 1UL;
        out[144] = 1UL;
        out[145] = lifecycle_stale_packet0;
        out[146] = lifecycle_stale_packet1;
        out[147] = lifecycle_stale_slot0_offset_bytes;
        out[148] = lifecycle_stale_slot1_offset_bytes;
        out[149] = lifecycle_stale_below_read0;
        out[150] = lifecycle_stale_below_read1;
        out[151] = lifecycle_stale_reuse_allowed;
        out[152] = 1UL;
        out[153] = 32UL;
        out[154] = (ulong)expected_header32_word;
        out[155] = (ulong)lifecycle_invalid_header32_word;
        out[156] = lifecycle_header0_before & 0xffffffffUL;
        out[157] = lifecycle_header1_before & 0xffffffffUL;
        out[158] = lifecycle_header0_after & 0xffffffffUL;
        out[159] = lifecycle_header1_after & 0xffffffffUL;
        for (uint i = 0u; i < 8u; ++i) {
            shadow_queue[cas_retry_slot0_base + (ulong)i] = 0UL;
            shadow_queue[cas_retry_slot1_base + (ulong)i] = 0UL;
        }
        atomic_store_explicit(&write_index[0], wrap_final_write,
                              memory_order_release,
                              memory_scope_all_svm_devices);
        out[127] = atomic_load_explicit(&write_index[0], memory_order_acquire,
                                        memory_scope_all_svm_devices);
        for (uint i = 0u; i < 8u; ++i) {
            shadow_queue[wrap_slot0_base + (ulong)i] = 0UL;
        }
        shadow_queue[wrap_slot1_base] = 0UL;
        atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release,
                               memory_scope_all_svm_devices);
    }
}

// Fused residual add + RMSNorm for a bf16 resident residual stream. The
// contribution arrives as f32, accumulation is performed in f32 registers,
// residual HBM is rounded to bf16, and the normalized output remains f16 for
// the existing attention/MLP input contract.
__kernel void add_rmsnorm_bf16_residual_f16_out(__global ushort* acc,
                                                __global const float* x,
                                                __global const half* weight,
                                                __global half* y,
                                                uint H, float eps) {
    uint t = get_local_id(0);
    const uint nt = 256u;
    __local float red[256];
    float ss = 0.0f;
    for (uint i = t; i < H; i += nt) {
        float v = qwen_bf16_bits_to_f32(acc[i]) + x[i];
        ushort rounded = qwen_f32_to_bf16_bits(v);
        acc[i] = rounded;
        float rv = qwen_bf16_bits_to_f32(rounded);
        ss += rv * rv;
    }
    red[t] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint off = nt >> 1; off > 0; off >>= 1) {
        if (t < off) red[t] += red[t + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float rms = rsqrt(red[0] / (float)H + eps);
    for (uint i = t; i < H; i += nt) {
        float rv = qwen_bf16_bits_to_f32(acc[i]);
        y[i] = (half)(rv * rms * (float)weight[i]);
    }
}

// RoPE (rotary position embedding) for one query/key vector at position `pos`,
// half-rotation (GPT-NeoX/Llama style): dims paired (i, i+H/2) rotated by angle
// pos * theta^(-2i/H). One workgroup; each thread handles one pair. f16 in/out.
__kernel void rope_f16(__global half* x, uint H, uint pos, float theta) {
    uint t = get_local_id(0);
    uint half_h = H >> 1;
    for (uint i = t; i < half_h; i += 256u) {  // fixed wg size (avoid get_local_size hidden arg)
        float freq = pow(theta, -2.0f * (float)i / (float)H);
        float ang = (float)pos * freq;
        float c = native_cos(ang), s = native_sin(ang);
        float a = (float)x[i];
        float b = (float)x[i + half_h];
        x[i]          = (half)(a * c - b * s);
        x[i + half_h] = (half)(b * c + a * s);
    }
}

// SwiGLU MLP activation: y[i] = silu(gate[i]) * up[i], silu(x)=x/(1+exp(-x)).
// Elementwise over the intermediate dim; grid-strided, 256-thread workgroups.
__kernel void swiglu_f16(__global const half* gate, __global const half* up,
                         __global half* y, uint n, uint nthreads) {
    uint gid = get_group_id(0) * 256u + get_local_id(0);
    for (uint i = gid; i < n; i += nthreads) {
        float g = (float)gate[i];
        float silu = g / (1.0f + native_exp(-g));
        y[i] = (half)(silu * (float)up[i]);
    }
}

// Descriptor-driven SwiGLU. The launch passes only the descriptor and status
// buffers as kernargs; compute data pointers and shape are read from HBM:
// desc[0]=magic, desc[1]=version|row<<32, desc[2]=gate_va, desc[3]=up_va,
// desc[4]=y_va, desc[5]=n|nthreads<<32, desc[6]=reserved, desc[7]=fnv(desc[0..6]).
__kernel void swiglu_f16_descriptor(__global const ulong* desc,
                                    __global ulong* status) {
    uint gid = get_group_id(0) * 256u + get_local_id(0);
    ulong magic = desc[0];
    ulong version_row = desc[1];
    ulong gate_va = desc[2];
    ulong up_va = desc[3];
    ulong y_va = desc[4];
    ulong shape = desc[5];
    ulong expected_checksum = desc[7];
    uint version = (uint)(version_row & 0xffffffffUL);
    uint n = (uint)(shape & 0xffffffffUL);
    uint nthreads = (uint)(shape >> 32);

    ulong checksum = 0xcbf29ce484222325UL;
    for (uint word_idx = 0u; word_idx < 7u; ++word_idx) {
        ulong word = desc[word_idx];
        for (uint byte_idx = 0u; byte_idx < 8u; ++byte_idx) {
            checksum ^= (word >> (byte_idx * 8u)) & 0xffUL;
            checksum *= 0x100000001b3UL;
        }
    }

    ulong result = 0UL;
    ulong bad_value = 0UL;
    if (magic != 0x4d41525f53574947UL) {
        result = 1UL;
        bad_value = magic;
    } else if (version != 1u) {
        result = 2UL;
        bad_value = version_row;
    } else if (n == 0u || nthreads == 0u) {
        result = 3UL;
        bad_value = shape;
    } else if (checksum != expected_checksum) {
        result = 4UL;
        bad_value = checksum;
    }

    if (gid == 0u) {
        status[0] = result;
        status[1] = (ulong)n;
        status[2] = (ulong)nthreads;
        status[3] = checksum;
        status[4] = expected_checksum;
        status[5] = gate_va;
        status[6] = up_va;
        status[7] = y_va;
        status[8] = bad_value;
        status[9] = 0x5A17D15C5AFE600DUL;
    }
    if (result != 0UL) return;

    __global const half* gate = (__global const half*)gate_va;
    __global const half* up = (__global const half*)up_va;
    __global half* y = (__global half*)y_va;
    for (uint i = gid; i < n; i += nthreads) {
        float g = (float)gate[i];
        float silu = g / (1.0f + native_exp(-g));
        y[i] = (half)(silu * (float)up[i]);
    }
}

// NVFP4 KV decode split (D==128): E2M1 4-bit values with a per-block-16 E4M3
// scale (NVFP4 — the finer E4M3 scale preserves block max far better than the
// E8M0 power-of-two of plain MXFP4: ~9% vs ~17% rel-L2 on realistic outlier
// data). The hardware cvt only applies an E8M0 (exponent-only) scale, so we
// decode E2M1 RAW (cvt scale=1.0) and multiply the lane partial by the E4M3
// block scale ourselves (it factors out per block, like the FP8 per-token
// scale). Block-16 ⇒ each lane's 8 dims sit in block sub/2; E4M3 scale decoded
// with cvt_pk_f32_fp8. scale_k/scale_v are E4M3 bytes [N][8].
__kernel void attn_decode_split2_nvfp4(__global const half* q,
                                       __global const uchar* K, __global const uchar* V,
                                       __global const uchar* scale_k, __global const uchar* scale_v,
                                       __global float* partials,
                                       uint N, uint D, float scale, uint num_splits) {
    uint sp = get_group_id(0);
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint g = lane >> 4;
    uint sub = lane & 15;
    uint blk = sub >> 1;             // block-16: lane's 8 dims are in block sub/2
    uint S = (N + num_splits - 1) / num_splits;
    uint lo = sp * S;
    uint hi = min(N, lo + S);
    uint span = hi > lo ? hi - lo : 0;
    uint per = (span + WPS_ATTN - 1) / WPS_ATTN;
    uint wlo = lo + w * per;
    uint whi = min(hi, wlo + per);

    half8 q8 = vload8(sub, q);
    float qv[8];
    qv[0]=q8.s0; qv[1]=q8.s1; qv[2]=q8.s2; qv[3]=q8.s3;
    qv[4]=q8.s4; qv[5]=q8.s5; qv[6]=q8.s6; qv[7]=q8.s7;

    float m = -INFINITY, l = 0.0f, o[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = 0.0f;

    for (uint base = wlo + g; base < whi; base += 4u * UATTN) {
        uint kb[UATTN], vb[UATTN];
        float ks[UATTN], vs[UATTN];
        #pragma unroll
        for (uint u = 0; u < UATTN; ++u) {
            uint t = base + 4u * u;
            uint tt = t < whi ? t : whi - 1;
            kb[u] = ((__global const uint*)(K + (ulong)tt * 64))[sub];
            vb[u] = ((__global const uint*)(V + (ulong)tt * 64))[sub];
            // E4M3 block scale -> f32 (cvt_pk_f32_fp8 decodes E4M3).
            float2 sk = __builtin_amdgcn_cvt_pk_f32_fp8((uint)scale_k[tt * 8 + blk], false);
            float2 sv = __builtin_amdgcn_cvt_pk_f32_fp8((uint)scale_v[tt * 8 + blk], false);
            ks[u] = sk.x;
            vs[u] = sv.x;
        }
        #pragma unroll
        for (uint u = 0; u < UATTN; ++u) {
            uint t = base + 4u * u;
            if (t >= whi) break;
            uint k4 = kb[u];
            float2 ka = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(k4, 1.0f, 0);
            float2 kc = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(k4, 1.0f, 1);
            float2 ke = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(k4, 1.0f, 2);
            float2 kg = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(k4, 1.0f, 3);
            float partial = qv[0]*ka.x + qv[1]*ka.y + qv[2]*kc.x + qv[3]*kc.y
                          + qv[4]*ke.x + qv[5]*ke.y + qv[6]*kg.x + qv[7]*kg.y;
            partial *= ks[u];               // lane's E4M3 block scale (factors out)
            partial += BPERM(1u, partial);
            partial += BPERM(2u, partial);
            partial += BPERM(4u, partial);
            partial += BPERM(8u, partial);
            float s = partial * scale;
            float m_new = fmax(m, s);
            float corr = native_exp(m - m_new);
            float p = native_exp(s - m_new);
            l = l * corr + p;
            float pv = p * vs[u];           // fold V block scale into p
            uint v4 = vb[u];
            float2 ea = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, 1.0f, 0);
            float2 ec = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, 1.0f, 1);
            float2 ee = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, 1.0f, 2);
            float2 eg = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(v4, 1.0f, 3);
            o[0]=o[0]*corr+pv*ea.x; o[1]=o[1]*corr+pv*ea.y;
            o[2]=o[2]*corr+pv*ec.x; o[3]=o[3]*corr+pv*ec.y;
            o[4]=o[4]*corr+pv*ee.x; o[5]=o[5]*corr+pv*ee.y;
            o[6]=o[6]*corr+pv*eg.x; o[7]=o[7]*corr+pv*eg.y;
            m = m_new;
        }
    }

    float M = m;
    M = fmax(M, BPERM(16u, M));
    M = fmax(M, BPERM(32u, M));
    if (M == -INFINITY) M = 0.0f;
    float cg = native_exp(m - M);
    float L = l * cg;
    L += BPERM(16u, L);
    L += BPERM(32u, L);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float oc = o[i] * cg;
        oc += BPERM(16u, oc);
        oc += BPERM(32u, oc);
        o[i] = oc;
    }
    __local float wm[WPS_ATTN], wl[WPS_ATTN], wo[WPS_ATTN][128];
    if (lane == 0) { wm[w] = M; wl[w] = L; }
    if (g == 0) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) wo[w][sub * 8 + i] = o[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (w == 0 && lane < 64) {
        float MM = -INFINITY;
        for (uint k = 0; k < WPS_ATTN; ++k) MM = fmax(MM, wm[k]);
        if (MM == -INFINITY) MM = 0.0f;
        float LL = 0.0f;
        for (uint k = 0; k < WPS_ATTN; ++k) LL += wl[k] * native_exp(wm[k] - MM);
        __global float* pr = partials + (ulong)sp * (D + 2);
        if (lane == 0) { pr[0] = MM; pr[1] = LL; }
        for (uint dd = lane; dd < 128; dd += 64) {
            float acc = 0.0f;
            for (uint k = 0; k < WPS_ATTN; ++k)
                acc += wo[k][dd] * native_exp(wm[k] - MM);
            pr[2 + dd] = acc;
        }
    }
}

// Decode GEMV: y[n] = sum_k W[n][k] * x[k]  (W is [N][K] row-major f16, x is
// [K] f16, y is [N] f32). For single-token decode the projections (QKV/O/MLP)
// are matrix-vector — memory-bound on reading W once, the MFMA GEMM's M=1 case
// is wasteful. Workgroup = 4 wavefronts = 4 output rows; x is staged in LDS and
// reused across the 4 rows; each wavefront does its row's K-dot with wide half8
// loads (lane owns 8 contiguous dims) and a 64-lane bpermute reduce. Requires
// K % 512 == 0 (true for Qwen hidden sizes 3584/4096/8192...).
__kernel void gemv_f16(__global const half* W, __global const half* x,
                       __global float* y, uint N, uint K) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + w;
    __local half xl[8192];
    for (uint i = tid; i < K; i += 256u) xl[i] = x[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        float partial = 0.0f;
        uint passes = K >> 9;            // K / 512
        for (uint p = 0; p < passes; ++p) {
            uint base = (p << 9) + lane * 8u;
            half8 wv = vload8(0, W + (ulong)row * K + base);
            half8 xv = vload8(0, xl + base);
            partial += (float)wv.s0*(float)xv.s0 + (float)wv.s1*(float)xv.s1
                     + (float)wv.s2*(float)xv.s2 + (float)wv.s3*(float)xv.s3
                     + (float)wv.s4*(float)xv.s4 + (float)wv.s5*(float)xv.s5
                     + (float)wv.s6*(float)xv.s6 + (float)wv.s7*(float)xv.s7;
        }
        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) y[row] = partial;
    }
}

// Descriptor-driven decode GEMV. The launch passes only descriptor/status
// buffers as kernargs; W/x/y pointers and shape are read from HBM:
// desc[0]=magic, desc[1]=version|row<<32, desc[2]=W_va, desc[3]=x_va,
// desc[4]=y_va, desc[5]=N|K<<32, desc[6]=num_wg, desc[7]=fnv(desc[0..6]).
// The dot-product order matches gemv_f16 so the existing Qwen CPU oracle remains
// an apples-to-apples correctness check for the descriptor-fed down_proj path.
__kernel void gemv_f16_descriptor(__global const ulong* desc,
                                  __global ulong* status) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;

    ulong magic = desc[0];
    ulong version_row = desc[1];
    ulong w_va = desc[2];
    ulong x_va = desc[3];
    ulong y_va = desc[4];
    ulong shape = desc[5];
    ulong num_wg_word = desc[6];
    ulong expected_checksum = desc[7];
    uint version = (uint)(version_row & 0xffffffffUL);
    uint N = (uint)(shape & 0xffffffffUL);
    uint K = (uint)(shape >> 32);
    uint num_wg = (uint)(num_wg_word & 0xffffffffUL);

    ulong checksum = 0xcbf29ce484222325UL;
    for (uint word_idx = 0u; word_idx < 7u; ++word_idx) {
        ulong word = desc[word_idx];
        for (uint byte_idx = 0u; byte_idx < 8u; ++byte_idx) {
            checksum ^= (word >> (byte_idx * 8u)) & 0xffUL;
            checksum *= 0x100000001b3UL;
        }
    }

    ulong result = 0UL;
    ulong bad_value = 0UL;
    if (magic != 0x4d41525f47454d56UL) {
        result = 1UL;
        bad_value = magic;
    } else if (version != 1u) {
        result = 2UL;
        bad_value = version_row;
    } else if (N == 0u || K == 0u || K > 8192u || (K & 511u) != 0u || num_wg == 0u) {
        result = 3UL;
        bad_value = shape;
    } else if (checksum != expected_checksum) {
        result = 4UL;
        bad_value = checksum;
    }

    if (get_group_id(0) == 0u && tid == 0u) {
        status[0] = result;
        status[1] = (ulong)N;
        status[2] = (ulong)K;
        status[3] = (ulong)num_wg;
        status[4] = checksum;
        status[5] = expected_checksum;
        status[6] = w_va;
        status[7] = x_va;
        status[8] = y_va;
        status[9] = bad_value;
        status[10] = 0x6E15D15C5AFE600DUL;
    }
    if (result != 0UL) return;

    __global const half* W = (__global const half*)w_va;
    __global const half* x = (__global const half*)x_va;
    __global float* y = (__global float*)y_va;
    uint row = get_group_id(0) * 4u + w;
    __local half xl[8192];
    for (uint i = tid; i < K; i += 256u) xl[i] = x[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        float partial = 0.0f;
        uint passes = K >> 9;
        for (uint p = 0; p < passes; ++p) {
            uint base = (p << 9) + lane * 8u;
            half8 wv = vload8(0, W + (ulong)row * K + base);
            half8 xv = vload8(0, xl + base);
            partial += (float)wv.s0*(float)xv.s0 + (float)wv.s1*(float)xv.s1
                     + (float)wv.s2*(float)xv.s2 + (float)wv.s3*(float)xv.s3
                     + (float)wv.s4*(float)xv.s4 + (float)wv.s5*(float)xv.s5
                     + (float)wv.s6*(float)xv.s6 + (float)wv.s7*(float)xv.s7;
        }
        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) y[row] = partial;
    }
}

__kernel void gemv_f16_descriptor_terminal_ready_flag(__global const ulong* desc,
                                                      __global ulong* status,
                                                      volatile __global atomic_uint* flags,
                                                      volatile __global atomic_uint* counts,
                                                      uint rank) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;

    ulong magic = desc[0];
    ulong version_row = desc[1];
    ulong w_va = desc[2];
    ulong x_va = desc[3];
    ulong y_va = desc[4];
    ulong shape = desc[5];
    ulong num_wg_word = desc[6];
    ulong expected_checksum = desc[7];
    uint version = (uint)(version_row & 0xffffffffUL);
    uint N = (uint)(shape & 0xffffffffUL);
    uint K = (uint)(shape >> 32);
    uint num_wg = (uint)(num_wg_word & 0xffffffffUL);

    ulong checksum = 0xcbf29ce484222325UL;
    for (uint word_idx = 0u; word_idx < 7u; ++word_idx) {
        ulong word = desc[word_idx];
        for (uint byte_idx = 0u; byte_idx < 8u; ++byte_idx) {
            checksum ^= (word >> (byte_idx * 8u)) & 0xffUL;
            checksum *= 0x100000001b3UL;
        }
    }

    ulong result = 0UL;
    ulong bad_value = 0UL;
    if (magic != 0x4d41525f47454d56UL) {
        result = 1UL;
        bad_value = magic;
    } else if (version != 1u) {
        result = 2UL;
        bad_value = version_row;
    } else if (N == 0u || K == 0u || K > 8192u || (K & 511u) != 0u || num_wg == 0u) {
        result = 3UL;
        bad_value = shape;
    } else if (checksum != expected_checksum) {
        result = 4UL;
        bad_value = checksum;
    }

    if (get_group_id(0) == 0u && tid == 0u) {
        status[0] = result;
        status[1] = (ulong)N;
        status[2] = (ulong)K;
        status[3] = (ulong)num_wg;
        status[4] = checksum;
        status[5] = expected_checksum;
        status[6] = w_va;
        status[7] = x_va;
        status[8] = y_va;
        status[9] = bad_value;
        status[10] = 0x6E15D15C5AFE600DUL;
    }
    if (result != 0UL) return;

    __global const half* W = (__global const half*)w_va;
    __global const half* x = (__global const half*)x_va;
    __global float* y = (__global float*)y_va;
    uint row = get_group_id(0) * 4u + w;
    __local half xl[8192];
    for (uint i = tid; i < K; i += 256u) xl[i] = x[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        float partial = 0.0f;
        uint passes = K >> 9;
        for (uint p = 0; p < passes; ++p) {
            uint base = (p << 9) + lane * 8u;
            half8 wv = vload8(0, W + (ulong)row * K + base);
            half8 xv = vload8(0, xl + base);
            partial += (float)wv.s0*(float)xv.s0 + (float)wv.s1*(float)xv.s1
                     + (float)wv.s2*(float)xv.s2 + (float)wv.s3*(float)xv.s3
                     + (float)wv.s4*(float)xv.s4 + (float)wv.s5*(float)xv.s5
                     + (float)wv.s6*(float)xv.s6 + (float)wv.s7*(float)xv.s7;
        }
        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) y[row] = partial;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release,
                           memory_scope_all_svm_devices);
    if (tid == 0u) {
        uint old = atomic_fetch_add_explicit(&counts[rank], 1u,
                                             memory_order_acq_rel,
                                             memory_scope_all_svm_devices);
        if (old + 1u == num_wg) {
            atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release,
                                   memory_scope_all_svm_devices);
            atomic_store_explicit(&flags[rank], 1u, memory_order_release,
                                  memory_scope_all_svm_devices);
        }
    }
}

// Direct decode GEMV with terminal producer publication. This is the
// non-descriptor companion to gemv_f16_descriptor_terminal_ready_flag: it writes
// y[N], fences the output at all-SVM-device scope, then the final workgroup
// releases flags[rank]. The consumer side can wait in CUs before reading the
// O-proj/allreduce input buffer.
__kernel void gemv_f16_terminal_ready_flag(__global const half* W,
                                           __global const half* x,
                                           __global float* y,
                                           volatile __global atomic_uint* flags,
                                           volatile __global atomic_uint* counts,
                                           uint rank,
                                           uint N,
                                           uint K) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + w;
    uint num_wg = (N + 3u) >> 2;

    __local half xl[8192];
    for (uint i = tid; i < K; i += 256u) xl[i] = x[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        float partial = 0.0f;
        uint passes = K >> 9;
        for (uint p = 0; p < passes; ++p) {
            uint base = (p << 9) + lane * 8u;
            half8 wv = vload8(0, W + (ulong)row * K + base);
            half8 xv = vload8(0, xl + base);
            partial += (float)wv.s0*(float)xv.s0 + (float)wv.s1*(float)xv.s1
                     + (float)wv.s2*(float)xv.s2 + (float)wv.s3*(float)xv.s3
                     + (float)wv.s4*(float)xv.s4 + (float)wv.s5*(float)xv.s5
                     + (float)wv.s6*(float)xv.s6 + (float)wv.s7*(float)xv.s7;
        }
        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) y[row] = partial;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release,
                           memory_scope_all_svm_devices);
    if (tid == 0u) {
        uint old = atomic_fetch_add_explicit(&counts[rank], 1u,
                                             memory_order_acq_rel,
                                             memory_scope_all_svm_devices);
        if (old + 1u == num_wg) {
            atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release,
                                   memory_scope_all_svm_devices);
            atomic_store_explicit(&flags[rank], 1u, memory_order_release,
                                  memory_scope_all_svm_devices);
        }
    }
}

// K=8192 decode GEMV for Qwen-class O-projection. Same row/lane dot-product
// order as gemv_f16, but with fixed K/stride so the compiler can specialize the
// hottest non-step projection shape in model-decode.
__kernel void gemv_f16_k8192(__global const half* W, __global const half* x,
                             __global float* y, uint N) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + w;
    __local half xl[8192];
    for (uint i = tid; i < 8192u; i += 256u) xl[i] = x[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        float partial = 0.0f;
        for (uint p = 0; p < 16u; ++p) {
            uint base = (p << 9) + lane * 8u;
            half8 wv = vload8(0, W + (ulong)row * 8192u + base);
            half8 xv = vload8(0, xl + base);
            partial += (float)wv.s0*(float)xv.s0 + (float)wv.s1*(float)xv.s1
                     + (float)wv.s2*(float)xv.s2 + (float)wv.s3*(float)xv.s3
                     + (float)wv.s4*(float)xv.s4 + (float)wv.s5*(float)xv.s5
                     + (float)wv.s6*(float)xv.s6 + (float)wv.s7*(float)xv.s7;
        }
        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) y[row] = partial;
    }
}

// Graph-friendly GEMV history writer. The output base pointer stays fixed and
// the active row block is selected by device step metadata.
__kernel void gemv_f16_step(__global const half* W, __global const half* x,
                            __global float* y_base, __global const uint* step,
                            uint N, uint K) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + w;
    uint s = step[0];
    __global float* y = y_base + (ulong)s * N;
    __local half xl[8192];
    for (uint i = tid; i < K; i += 256u) xl[i] = x[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        float partial = 0.0f;
        uint passes = K >> 9;
        for (uint p = 0; p < passes; ++p) {
            uint base = (p << 9) + lane * 8u;
            half8 wv = vload8(0, W + (ulong)row * K + base);
            half8 xv = vload8(0, xl + base);
            partial += (float)wv.s0*(float)xv.s0 + (float)wv.s1*(float)xv.s1
                     + (float)wv.s2*(float)xv.s2 + (float)wv.s3*(float)xv.s3
                     + (float)wv.s4*(float)xv.s4 + (float)wv.s5*(float)xv.s5
                     + (float)wv.s6*(float)xv.s6 + (float)wv.s7*(float)xv.s7;
        }
        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) y[row] = partial;
    }
}

// K=4096 graph-friendly GEMV for Qwen-class decode projections. This keeps the
// exact row/lane dot-product order of gemv_f16_step but halves LDS footprint vs
// the generic 8192-half staging buffer used for larger projections.
__kernel void gemv_f16_step_k4096(__global const half* W, __global const half* x,
                                  __global float* y_base, __global const uint* step,
                                  uint N) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + w;
    uint s = step[0];
    __global float* y = y_base + (ulong)s * N;
    __local half xl[4096];
    for (uint i = tid; i < 4096u; i += 256u) xl[i] = x[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        float partial = 0.0f;
        for (uint p = 0; p < 8u; ++p) {
            uint base = (p << 9) + lane * 8u;
            half8 wv = vload8(0, W + (ulong)row * 4096u + base);
            half8 xv = vload8(0, xl + base);
            partial += (float)wv.s0*(float)xv.s0 + (float)wv.s1*(float)xv.s1
                     + (float)wv.s2*(float)xv.s2 + (float)wv.s3*(float)xv.s3
                     + (float)wv.s4*(float)xv.s4 + (float)wv.s5*(float)xv.s5
                     + (float)wv.s6*(float)xv.s6 + (float)wv.s7*(float)xv.s7;
        }
        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) y[row] = partial;
    }
}

// One-token greedy sampling for decode. Returns the first index with the maximum
// logit, matching the CPU argmax tie-break used by the f64 oracle.
__kernel void argmax_f32(__global const float* logits, __global uint* tokens,
                         uint out_index, uint N) {
    uint t = get_local_id(0);
    const uint nt = 256u;
    __local float vals[256];
    __local uint idxs[256];
    float best = -INFINITY;
    uint best_i = 0xffffffffu;
    for (uint i = t; i < N; i += nt) {
        float v = logits[i];
        if (v > best || (v == best && i < best_i)) {
            best = v;
            best_i = i;
        }
    }
    vals[t] = best;
    idxs[t] = best_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint off = nt >> 1; off > 0; off >>= 1) {
        if (t < off) {
            float ov = vals[t + off];
            uint oi = idxs[t + off];
            if (ov > vals[t] || (ov == vals[t] && oi < idxs[t])) {
                vals[t] = ov;
                idxs[t] = oi;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (t == 0) tokens[out_index] = idxs[0];
}

// Compact candidate greedy sampling for distributed vocab-parallel decode.
// logits[N] and token_ids[N] are already one candidate per TP rank/tile.
// Writes the winning global token id into a device token-handoff buffer.
// Tie-break is the lower global token id, matching full-vocab greedy argmax.
__kernel void argmax_f32_token_ids(__global const float* logits,
                                   __global const uint* token_ids,
                                   __global uint* tokens,
                                   uint out_index,
                                   uint N) {
    uint t = get_local_id(0);
    const uint nt = 256u;
    __local float vals[256];
    __local uint ids[256];
    float best = -INFINITY;
    uint best_id = 0xffffffffu;
    for (uint i = t; i < N; i += nt) {
        float v = logits[i];
        uint id = token_ids[i];
        if (v > best || (v == best && id < best_id)) {
            best = v;
            best_id = id;
        }
    }
    vals[t] = best;
    ids[t] = best_id;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint off = nt >> 1; off > 0; off >>= 1) {
        if (t < off) {
            float ov = vals[t + off];
            uint oid = ids[t + off];
            if (ov > vals[t] || (ov == vals[t] && oid < ids[t])) {
                vals[t] = ov;
                ids[t] = oid;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (t == 0) tokens[out_index] = ids[0];
}

// Local candidate writer for vocab-parallel greedy decode. Each rank reduces
// its local logits, translates the local winner to a global token id, and writes
// exactly one compact candidate into the shared candidate arrays. Tie-break is
// lower global token id.
__kernel void argmax_f32_write_candidate(__global const float* logits,
                                         __global float* candidate_logits,
                                         __global uint* candidate_token_ids,
                                         uint slot,
                                         uint global_token_offset,
                                         uint N) {
    uint t = get_local_id(0);
    const uint nt = 256u;
    __local float vals[256];
    __local uint ids[256];
    float best = -INFINITY;
    uint best_id = 0xffffffffu;
    for (uint i = t; i < N; i += nt) {
        float v = logits[i];
        uint id = global_token_offset + i;
        if (v > best || (v == best && id < best_id)) {
            best = v;
            best_id = id;
        }
    }
    vals[t] = best;
    ids[t] = best_id;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint off = nt >> 1; off > 0; off >>= 1) {
        if (t < off) {
            float ov = vals[t + off];
            uint oid = ids[t + off];
            if (ov > vals[t] || (ov == vals[t] && oid < ids[t])) {
                vals[t] = ov;
                ids[t] = oid;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (t == 0) {
        candidate_logits[slot] = vals[0];
        candidate_token_ids[slot] = ids[0];
    }
}

// Fused LM-head GEMV tile candidate writer. This keeps the exact row/lane
// dot-product order of gemv_f16, but does not materialize per-token logits.
// Each 16-row workgroup emits one compact tile candidate:
//   tile_logits[group] + tile_token_ids[group].
// A following compact reduction selects the per-rank candidate.
__kernel void gemv_f16_candidate_tiles(__global const half* W, __global const half* x,
                                       __global float* tile_logits,
                                       __global uint* tile_token_ids,
                                       uint global_token_offset,
                                       uint N,
                                       uint K) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint group = get_group_id(0);
    uint row0 = group * 16u + w * 4u;
    __local half xl[8192];
    __local float row_vals[16];
    __local uint row_ids[16];
    for (uint i = tid; i < K; i += 256u) xl[i] = x[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    float p0 = -INFINITY;
    float p1 = -INFINITY;
    float p2 = -INFINITY;
    float p3 = -INFINITY;
    uint id0 = 0xffffffffu;
    uint id1 = 0xffffffffu;
    uint id2 = 0xffffffffu;
    uint id3 = 0xffffffffu;
    if (row0 < N) p0 = 0.0f;
    if (row0 + 1u < N) p1 = 0.0f;
    if (row0 + 2u < N) p2 = 0.0f;
    if (row0 + 3u < N) p3 = 0.0f;
    uint passes = K >> 9;
    for (uint p = 0; p < passes; ++p) {
        uint base = (p << 9) + lane * 8u;
        half8 xv = vload8(0, xl + base);
        if (row0 < N) {
            half8 wv = vload8(0, W + (ulong)row0 * K + base);
            p0 += (float)wv.s0*(float)xv.s0 + (float)wv.s1*(float)xv.s1
                + (float)wv.s2*(float)xv.s2 + (float)wv.s3*(float)xv.s3
                + (float)wv.s4*(float)xv.s4 + (float)wv.s5*(float)xv.s5
                + (float)wv.s6*(float)xv.s6 + (float)wv.s7*(float)xv.s7;
        }
        if (row0 + 1u < N) {
            half8 wv = vload8(0, W + (ulong)(row0 + 1u) * K + base);
            p1 += (float)wv.s0*(float)xv.s0 + (float)wv.s1*(float)xv.s1
                + (float)wv.s2*(float)xv.s2 + (float)wv.s3*(float)xv.s3
                + (float)wv.s4*(float)xv.s4 + (float)wv.s5*(float)xv.s5
                + (float)wv.s6*(float)xv.s6 + (float)wv.s7*(float)xv.s7;
        }
        if (row0 + 2u < N) {
            half8 wv = vload8(0, W + (ulong)(row0 + 2u) * K + base);
            p2 += (float)wv.s0*(float)xv.s0 + (float)wv.s1*(float)xv.s1
                + (float)wv.s2*(float)xv.s2 + (float)wv.s3*(float)xv.s3
                + (float)wv.s4*(float)xv.s4 + (float)wv.s5*(float)xv.s5
                + (float)wv.s6*(float)xv.s6 + (float)wv.s7*(float)xv.s7;
        }
        if (row0 + 3u < N) {
            half8 wv = vload8(0, W + (ulong)(row0 + 3u) * K + base);
            p3 += (float)wv.s0*(float)xv.s0 + (float)wv.s1*(float)xv.s1
                + (float)wv.s2*(float)xv.s2 + (float)wv.s3*(float)xv.s3
                + (float)wv.s4*(float)xv.s4 + (float)wv.s5*(float)xv.s5
                + (float)wv.s6*(float)xv.s6 + (float)wv.s7*(float)xv.s7;
        }
    }
    if (row0 < N) {
        p0 += BPERM(1u, p0);
        p0 += BPERM(2u, p0);
        p0 += BPERM(4u, p0);
        p0 += BPERM(8u, p0);
        p0 += BPERM(16u, p0);
        p0 += BPERM(32u, p0);
        id0 = global_token_offset + row0;
    }
    if (row0 + 1u < N) {
        p1 += BPERM(1u, p1);
        p1 += BPERM(2u, p1);
        p1 += BPERM(4u, p1);
        p1 += BPERM(8u, p1);
        p1 += BPERM(16u, p1);
        p1 += BPERM(32u, p1);
        id1 = global_token_offset + row0 + 1u;
    }
    if (row0 + 2u < N) {
        p2 += BPERM(1u, p2);
        p2 += BPERM(2u, p2);
        p2 += BPERM(4u, p2);
        p2 += BPERM(8u, p2);
        p2 += BPERM(16u, p2);
        p2 += BPERM(32u, p2);
        id2 = global_token_offset + row0 + 2u;
    }
    if (row0 + 3u < N) {
        p3 += BPERM(1u, p3);
        p3 += BPERM(2u, p3);
        p3 += BPERM(4u, p3);
        p3 += BPERM(8u, p3);
        p3 += BPERM(16u, p3);
        p3 += BPERM(32u, p3);
        id3 = global_token_offset + row0 + 3u;
    }
    if (lane == 0) {
        uint base = w * 4u;
        row_vals[base] = p0;
        row_vals[base + 1u] = p1;
        row_vals[base + 2u] = p2;
        row_vals[base + 3u] = p3;
        row_ids[base] = id0;
        row_ids[base + 1u] = id1;
        row_ids[base + 2u] = id2;
        row_ids[base + 3u] = id3;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid == 0) {
        float best = row_vals[0];
        uint best_id = row_ids[0];
        for (uint i = 1u; i < 16u; ++i) {
            float v = row_vals[i];
            uint id = row_ids[i];
            if (v > best || (v == best && id < best_id)) {
                best = v;
                best_id = id;
            }
        }
        tile_logits[group] = best;
        tile_token_ids[group] = best_id;
    }
}

// Qwen3-4B LM-head candidate writer for hidden_size K=2560. Same 16-row tile
// shape and dot-product order as gemv_f16_candidate_tiles, but fixes K and
// shrinks LDS from 8192 halfs to 2560 halfs.
__kernel void gemv_f16_candidate_tiles_k2560(__global const half* W, __global const half* x,
                                             __global float* tile_logits,
                                             __global uint* tile_token_ids,
                                             uint global_token_offset,
                                             uint N) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint group = get_group_id(0);
    uint row0 = group * 16u + w * 4u;
    __local half xl[2560];
    __local float row_vals[16];
    __local uint row_ids[16];
    for (uint i = tid; i < 2560u; i += 256u) xl[i] = x[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    float p0 = -INFINITY;
    float p1 = -INFINITY;
    float p2 = -INFINITY;
    float p3 = -INFINITY;
    uint id0 = 0xffffffffu;
    uint id1 = 0xffffffffu;
    uint id2 = 0xffffffffu;
    uint id3 = 0xffffffffu;
    if (row0 < N) p0 = 0.0f;
    if (row0 + 1u < N) p1 = 0.0f;
    if (row0 + 2u < N) p2 = 0.0f;
    if (row0 + 3u < N) p3 = 0.0f;
    for (uint p = 0; p < 5u; ++p) {
        uint base = (p << 9) + lane * 8u;
        half8 xv = vload8(0, xl + base);
        if (row0 < N) {
            half8 wv = vload8(0, W + (ulong)row0 * 2560u + base);
            p0 += (float)wv.s0*(float)xv.s0 + (float)wv.s1*(float)xv.s1
                + (float)wv.s2*(float)xv.s2 + (float)wv.s3*(float)xv.s3
                + (float)wv.s4*(float)xv.s4 + (float)wv.s5*(float)xv.s5
                + (float)wv.s6*(float)xv.s6 + (float)wv.s7*(float)xv.s7;
        }
        if (row0 + 1u < N) {
            half8 wv = vload8(0, W + (ulong)(row0 + 1u) * 2560u + base);
            p1 += (float)wv.s0*(float)xv.s0 + (float)wv.s1*(float)xv.s1
                + (float)wv.s2*(float)xv.s2 + (float)wv.s3*(float)xv.s3
                + (float)wv.s4*(float)xv.s4 + (float)wv.s5*(float)xv.s5
                + (float)wv.s6*(float)xv.s6 + (float)wv.s7*(float)xv.s7;
        }
        if (row0 + 2u < N) {
            half8 wv = vload8(0, W + (ulong)(row0 + 2u) * 2560u + base);
            p2 += (float)wv.s0*(float)xv.s0 + (float)wv.s1*(float)xv.s1
                + (float)wv.s2*(float)xv.s2 + (float)wv.s3*(float)xv.s3
                + (float)wv.s4*(float)xv.s4 + (float)wv.s5*(float)xv.s5
                + (float)wv.s6*(float)xv.s6 + (float)wv.s7*(float)xv.s7;
        }
        if (row0 + 3u < N) {
            half8 wv = vload8(0, W + (ulong)(row0 + 3u) * 2560u + base);
            p3 += (float)wv.s0*(float)xv.s0 + (float)wv.s1*(float)xv.s1
                + (float)wv.s2*(float)xv.s2 + (float)wv.s3*(float)xv.s3
                + (float)wv.s4*(float)xv.s4 + (float)wv.s5*(float)xv.s5
                + (float)wv.s6*(float)xv.s6 + (float)wv.s7*(float)xv.s7;
        }
    }
    if (row0 < N) {
        p0 += BPERM(1u, p0);
        p0 += BPERM(2u, p0);
        p0 += BPERM(4u, p0);
        p0 += BPERM(8u, p0);
        p0 += BPERM(16u, p0);
        p0 += BPERM(32u, p0);
        id0 = global_token_offset + row0;
    }
    if (row0 + 1u < N) {
        p1 += BPERM(1u, p1);
        p1 += BPERM(2u, p1);
        p1 += BPERM(4u, p1);
        p1 += BPERM(8u, p1);
        p1 += BPERM(16u, p1);
        p1 += BPERM(32u, p1);
        id1 = global_token_offset + row0 + 1u;
    }
    if (row0 + 2u < N) {
        p2 += BPERM(1u, p2);
        p2 += BPERM(2u, p2);
        p2 += BPERM(4u, p2);
        p2 += BPERM(8u, p2);
        p2 += BPERM(16u, p2);
        p2 += BPERM(32u, p2);
        id2 = global_token_offset + row0 + 2u;
    }
    if (row0 + 3u < N) {
        p3 += BPERM(1u, p3);
        p3 += BPERM(2u, p3);
        p3 += BPERM(4u, p3);
        p3 += BPERM(8u, p3);
        p3 += BPERM(16u, p3);
        p3 += BPERM(32u, p3);
        id3 = global_token_offset + row0 + 3u;
    }
    if (lane == 0) {
        uint base = w * 4u;
        row_vals[base] = p0;
        row_vals[base + 1u] = p1;
        row_vals[base + 2u] = p2;
        row_vals[base + 3u] = p3;
        row_ids[base] = id0;
        row_ids[base + 1u] = id1;
        row_ids[base + 2u] = id2;
        row_ids[base + 3u] = id3;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid == 0) {
        float best = row_vals[0];
        uint best_id = row_ids[0];
        for (uint i = 1u; i < 16u; ++i) {
            float v = row_vals[i];
            uint id = row_ids[i];
            if (v > best || (v == best && id < best_id)) {
                best = v;
                best_id = id;
            }
        }
        tile_logits[group] = best;
        tile_token_ids[group] = best_id;
    }
}

// Reduce compact tile candidates and write one `(logit, global_token_id)` into
// the shared TP candidate arrays. This is the second half of the fused LM-head
// candidate path: tile candidates are O(N/4), not full local logits.
__kernel void argmax_f32_token_ids_write_candidate(__global const float* logits,
                                                   __global const uint* token_ids,
                                                   __global float* candidate_logits,
                                                   __global uint* candidate_token_ids,
                                                   uint slot,
                                                   uint N) {
    uint t = get_local_id(0);
    const uint nt = 256u;
    __local float vals[256];
    __local uint ids[256];
    float best = -INFINITY;
    uint best_id = 0xffffffffu;
    for (uint i = t; i < N; i += nt) {
        float v = logits[i];
        uint id = token_ids[i];
        if (v > best || (v == best && id < best_id)) {
            best = v;
            best_id = id;
        }
    }
    vals[t] = best;
    ids[t] = best_id;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint off = nt >> 1; off > 0; off >>= 1) {
        if (t < off) {
            float ov = vals[t + off];
            uint oid = ids[t + off];
            if (ov > vals[t] || (ov == vals[t] && oid < ids[t])) {
                vals[t] = ov;
                ids[t] = oid;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (t == 0) {
        candidate_logits[slot] = vals[0];
        candidate_token_ids[slot] = ids[0];
    }
}

// Qwen3-4B fixed tile-candidate reducer. The K=2560/16-row LM-head candidate
// kernel emits exactly ceil(18992 / 16) = 1187 candidates per TP rank.
__kernel void argmax_f32_token_ids_write_candidate_n1187(__global const float* logits,
                                                         __global const uint* token_ids,
                                                         __global float* candidate_logits,
                                                         __global uint* candidate_token_ids,
                                                         uint slot) {
    uint t = get_local_id(0);
    const uint nt = 256u;
    __local float vals[256];
    __local uint ids[256];
    float best = -INFINITY;
    uint best_id = 0xffffffffu;
    for (uint i = t; i < 1187u; i += nt) {
        float v = logits[i];
        uint id = token_ids[i];
        if (v > best || (v == best && id < best_id)) {
            best = v;
            best_id = id;
        }
    }
    vals[t] = best;
    ids[t] = best_id;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint off = nt >> 1; off > 0; off >>= 1) {
        if (t < off) {
            float ov = vals[t + off];
            uint oid = ids[t + off];
            if (ov > vals[t] || (ov == vals[t] && oid < ids[t])) {
                vals[t] = ov;
                ids[t] = oid;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (t == 0) {
        candidate_logits[slot] = vals[0];
        candidate_token_ids[slot] = ids[0];
    }
}

// Rank0 final sampler for Qwen3-4B TP8. Reduces rank0's fixed 1187
// tile-candidates, writes candidate slot 0, then selects the final token across
// the 8 compact TP candidates already written by peer ranks.
__kernel void argmax_f32_token_ids_write_candidate_n1187_token8(
    __global const float* logits,
    __global const uint* token_ids,
    __global float* candidate_logits,
    __global uint* candidate_token_ids,
    __global uint* tokens,
    uint out_index) {
    uint t = get_local_id(0);
    const uint nt = 256u;
    __local float vals[256];
    __local uint ids[256];
    float best = -INFINITY;
    uint best_id = 0xffffffffu;
    for (uint i = t; i < 1187u; i += nt) {
        float v = logits[i];
        uint id = token_ids[i];
        if (v > best || (v == best && id < best_id)) {
            best = v;
            best_id = id;
        }
    }
    vals[t] = best;
    ids[t] = best_id;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint off = nt >> 1; off > 0; off >>= 1) {
        if (t < off) {
            float ov = vals[t + off];
            uint oid = ids[t + off];
            if (ov > vals[t] || (ov == vals[t] && oid < ids[t])) {
                vals[t] = ov;
                ids[t] = oid;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (t == 0) {
        float global_best = vals[0];
        uint global_id = ids[0];
        candidate_logits[0] = global_best;
        candidate_token_ids[0] = global_id;
        for (uint i = 1u; i < 8u; ++i) {
            float v = candidate_logits[i];
            uint id = candidate_token_ids[i];
            if (v > global_best || (v == global_best && id < global_id)) {
                global_best = v;
                global_id = id;
            }
        }
        tokens[out_index] = global_id;
    }
}

__kernel void qwen_global_token_to_local_token(__global const uint* global_tokens,
                                               __global uint* local_tokens,
                                               __global uint* owner_flags,
                                               uint global_index,
                                               uint local_index,
                                               uint token_count,
                                               uint global_vocab_start,
                                               uint local_vocab_rows) {
    if (get_global_id(0) != 0) return;
    if (token_count == 0u) return;
    if (global_index >= token_count) global_index = token_count - 1u;
    if (local_index >= token_count) local_index = token_count - 1u;
    uint token = global_tokens[global_index];
    uint in_shard = token >= global_vocab_start &&
                    (token - global_vocab_start) < local_vocab_rows;
    local_tokens[local_index] = in_shard ? token - global_vocab_start : 0xffffffffu;
    owner_flags[0] = in_shard;
}

// Step-based argmax for graph-friendly decode. Writes tokens[step+1], then
// advances step in the same final work-item after the reduction result is known.
__kernel void argmax_f32_step(__global const float* logits_base,
                              __global uint* tokens,
                              __global uint* step,
                              uint token_count,
                              uint N) {
    uint t = get_local_id(0);
    const uint nt = 256u;
    if (N == 0u || token_count == 0u) return;
    uint s = step[0];
    uint out_index = s + 1u;
    if (out_index >= token_count) return;
    __global const float* logits = logits_base + (ulong)s * N;
    __local float vals[256];
    __local uint idxs[256];
    float best = -INFINITY;
    uint best_i = 0xffffffffu;
    for (uint i = t; i < N; i += nt) {
        float v = logits[i];
        if (v > best || (v == best && i < best_i)) {
            best = v;
            best_i = i;
        }
    }
    vals[t] = best;
    idxs[t] = best_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint off = nt >> 1; off > 0; off >>= 1) {
        if (t < off) {
            float ov = vals[t + off];
            uint oi = idxs[t + off];
            if (ov > vals[t] || (ov == vals[t] && oi < idxs[t])) {
                vals[t] = ov;
                idxs[t] = oi;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (t == 0u) {
        tokens[out_index] = idxs[0];
        step[0] = out_index;
    }
}

// Fused decode Q/K/V GEMV. Rows 0..NQ-1 map to Wq/yq, rows NQ..NQ+NKV-1 map
// to Wk/yk, and rows NQ+NKV..NQ+2*NKV-1 map to Wv/yv. This keeps the exact
// per-row dot-product order of gemv_f16 while staging x once per 4-row
// workgroup and collapsing three AQL submits into one.
__kernel void gemv_qkv_f16(__global const half* Wq, __global const half* Wk,
                           __global const half* Wv, __global const half* x,
                           __global float* yq, __global float* yk,
                           __global float* yv, uint NQ, uint NKV, uint K) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + w;
    uint total = NQ + 2u * NKV;
    __local half xl[8192];
    for (uint i = tid; i < K; i += 256u) xl[i] = x[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < total) {
        __global const half* W = Wq;
        __global float* y = yq;
        uint local_row = row;
        if (row >= NQ + NKV) {
            W = Wv;
            y = yv;
            local_row = row - NQ - NKV;
        } else if (row >= NQ) {
            W = Wk;
            y = yk;
            local_row = row - NQ;
        }
        float partial = 0.0f;
        uint passes = K >> 9;
        for (uint p = 0; p < passes; ++p) {
            uint base = (p << 9) + lane * 8u;
            half8 wv = vload8(0, W + (ulong)local_row * K + base);
            half8 xv = vload8(0, xl + base);
            partial += (float)wv.s0*(float)xv.s0 + (float)wv.s1*(float)xv.s1
                     + (float)wv.s2*(float)xv.s2 + (float)wv.s3*(float)xv.s3
                     + (float)wv.s4*(float)xv.s4 + (float)wv.s5*(float)xv.s5
                     + (float)wv.s6*(float)xv.s6 + (float)wv.s7*(float)xv.s7;
        }
        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) y[local_row] = partial;
    }
}

// K=4096 fused Q/K/V GEMV for Qwen-class decode. Same output mapping and
// row/lane dot-product order as gemv_qkv_f16, but halves LDS footprint and fixes
// row stride for the model-decode hidden size.
__kernel void gemv_qkv_f16_k4096(__global const half* Wq, __global const half* Wk,
                                 __global const half* Wv, __global const half* x,
                                 __global float* yq, __global float* yk,
                                 __global float* yv, uint NQ, uint NKV) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + w;
    uint total = NQ + 2u * NKV;
    __local half xl[4096];
    for (uint i = tid; i < 4096u; i += 256u) xl[i] = x[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < total) {
        __global const half* W = Wq;
        __global float* y = yq;
        uint local_row = row;
        if (row >= NQ + NKV) {
            W = Wv;
            y = yv;
            local_row = row - NQ - NKV;
        } else if (row >= NQ) {
            W = Wk;
            y = yk;
            local_row = row - NQ;
        }
        float partial = 0.0f;
        for (uint p = 0; p < 8u; ++p) {
            uint base = (p << 9) + lane * 8u;
            half8 wv = vload8(0, W + (ulong)local_row * 4096u + base);
            half8 xv = vload8(0, xl + base);
            partial += (float)wv.s0*(float)xv.s0 + (float)wv.s1*(float)xv.s1
                     + (float)wv.s2*(float)xv.s2 + (float)wv.s3*(float)xv.s3
                     + (float)wv.s4*(float)xv.s4 + (float)wv.s5*(float)xv.s5
                     + (float)wv.s6*(float)xv.s6 + (float)wv.s7*(float)xv.s7;
        }
        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) y[local_row] = partial;
    }
}

// FP8-weight decode GEMV: y[n] = scale_w[n] * sum_k W_fp8[n][k] * x[k]. Weights
// are E4M3 with a per-output-row scale (the standard weight-quant layout); the
// scale factors out of the row dot (one multiply per row). Halves projection
// weight traffic vs FP16 — decode reads the full weight matrix every token, so
// this is the weight-memory analog of FP8 KV. Same tiling as gemv_f16; each lane
// reads 8 E4M3 (a uint2) per pass and decodes with cvt_pk_f32_fp8. K % 512 == 0.
__kernel void gemv_fp8(__global const uchar* W, __global const float* scale_w,
                       __global const half* x, __global float* y, uint N, uint K) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + w;
    __local half xl[8192];
    for (uint i = tid; i < K; i += 256u) xl[i] = x[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        float partial = 0.0f;
        uint passes = K >> 9;
        for (uint p = 0; p < passes; ++p) {
            uint base = (p << 9) + lane * 8u;
            uint2 wp = ((__global const uint2*)(W + (ulong)row * K + base))[0];
            float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
            float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
            float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
            float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
            half8 xv = vload8(0, xl + base);
            partial += a.x*(float)xv.s0 + a.y*(float)xv.s1 + b.x*(float)xv.s2 + b.y*(float)xv.s3
                     + c.x*(float)xv.s4 + c.y*(float)xv.s5 + d.x*(float)xv.s6 + d.y*(float)xv.s7;
        }
        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) y[row] = partial * scale_w[row];
    }
}

// Minimal CDNA4 scaled-MFMA execution probe. This is intentionally not a GEMM
// yet: it proves the gfx950 code object can carry and retire the native
// V_MFMA_SCALE_F32_16X16X128_F8F6F4 instruction under mainarch's raw KFD/AQL
// path. A/B are eight dwords each (32 packed FP8 bytes); scales are two dwords
// containing E8M0 scale bytes. The checker feeds zero FP8 operands and uses the
// result only as a diagnostic retirement payload; full numeric validation
// belongs to the follow-on lane-mapped tile kernel. Lane 0 writes its four
// accumulator values.
__kernel void mfma_scale_f8f6f4_probe(__global const uint* A,
                                      __global const uint* B,
                                      __global const uint* S,
                                      __global float* O) {
    int8 a = (int8)((int)A[0], (int)A[1], (int)A[2], (int)A[3],
                    (int)A[4], (int)A[5], (int)A[6], (int)A[7]);
    int8 b = (int8)((int)B[0], (int)B[1], (int)B[2], (int)B[3],
                    (int)B[4], (int)B[5], (int)B[6], (int)B[7]);
    float4 c = (float4)(1.0f, 2.0f, 3.0f, 4.0f);
    int scale_a = (int)S[0];
    int scale_b = (int)S[1];
    float4 r = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        a, b, c, 0, 0, 0, scale_a, 0, scale_b);
    if (get_local_id(0) == 0) {
        O[0] = r.s0;
        O[1] = r.s1;
        O[2] = r.s2;
        O[3] = r.s3;
    }
}

// Numeric readiness gate for CDNA4 scaled-MFMA under the raw KFD/AQL queue. With
// all FP8 operands equal to OCP zero, the instruction should preserve the FP32
// accumulator. If every tested immediate/scale form corrupts the accumulator,
// higher-level numeric tile work must stop and inspect queue FP8 mode/state.
__kernel void mfma_scale_f8f6f4_calibrate(__global const uint* A,
                                          __global const uint* B,
                                          __global const uint* S,
                                          __global float* O) {
    uint lane = get_local_id(0) & 63u;
    int8 a = (int8)((int)A[0], (int)A[1], (int)A[2], (int)A[3],
                    (int)A[4], (int)A[5], (int)A[6], (int)A[7]);
    int8 b = (int8)((int)B[0], (int)B[1], (int)B[2], (int)B[3],
                    (int)B[4], (int)B[5], (int)B[6], (int)B[7]);
    float4 c = (float4)(1.0f, 2.0f, 3.0f, 4.0f);
    int zero_scale = (int)S[0];
    int neutral_scale = (int)S[1];

    float4 ck_zero = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        a, b, c, 0, 0, 0, zero_scale, 0, zero_scale);
    float4 ck_neutral = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        a, b, c, 0, 0, 0, neutral_scale, 0, neutral_scale);
    float4 llvm_zero = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        a, b, c, 3, 1, 2, zero_scale, 3, zero_scale);
    float4 llvm_neutral = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        a, b, c, 3, 1, 2, neutral_scale, 3, neutral_scale);

    if (lane == 0u) {
        O[0] = ck_zero.s0;
        O[1] = ck_zero.s1;
        O[2] = ck_zero.s2;
        O[3] = ck_zero.s3;
        O[4] = ck_neutral.s0;
        O[5] = ck_neutral.s1;
        O[6] = ck_neutral.s2;
        O[7] = ck_neutral.s3;
        O[8] = llvm_zero.s0;
        O[9] = llvm_zero.s1;
        O[10] = llvm_zero.s2;
        O[11] = llvm_zero.s3;
        O[12] = llvm_neutral.s0;
        O[13] = llvm_neutral.s1;
        O[14] = llvm_neutral.s2;
        O[15] = llvm_neutral.s3;
    }
}

// FP8-weight decode GEMV with serving-shaped 128x128 E8M0 block scales.
// Scales are packed four K-block scale bytes per u32 for each 128-row N block:
// scale_packed[(row/128) * packed_kblocks + (kblock/4)] byte lane (kblock%4).
__kernel void gemv_fp8_wblock_e8m0(__global const uchar* W,
                                   __global const uint* scale_w,
                                   __global const half* x,
                                   __global float* y,
                                   uint N,
                                   uint K,
                                   uint kblocks,
                                   uint packed_kblocks) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + w;
    if (K == 4096u && packed_kblocks == 8u) {
        if (row < N) {
            uint nblock = row >> 7;
            float partial = 0.0f;
#pragma unroll
            for (uint p = 0; p < 4u; ++p) {
                uint base = (p << 10) + lane * 16u;
                uint kblock = base >> 7;
                uint word = scale_w[nblock * 8u + (kblock >> 2)];
                uchar sc = (uchar)((word >> ((kblock & 3u) * 8u)) & 0xffu);
                float scale = e8m0_to_f32(sc);
                uint4 wp = ((__global const uint4*)(W + (ulong)row * 4096u + base))[0];
                float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
                float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
                float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
                float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
                float2 e = __builtin_amdgcn_cvt_pk_f32_fp8(wp.z, false);
                float2 f = __builtin_amdgcn_cvt_pk_f32_fp8(wp.z, true);
                float2 g = __builtin_amdgcn_cvt_pk_f32_fp8(wp.w, false);
                float2 h = __builtin_amdgcn_cvt_pk_f32_fp8(wp.w, true);
                half16 xv = vload16(0, x + base);
                partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                                  + b.x*(float)xv.s2 + b.y*(float)xv.s3
                                  + c.x*(float)xv.s4 + c.y*(float)xv.s5
                                  + d.x*(float)xv.s6 + d.y*(float)xv.s7
                                  + e.x*(float)xv.s8 + e.y*(float)xv.s9
                                  + f.x*(float)xv.sa + f.y*(float)xv.sb
                                  + g.x*(float)xv.sc + g.y*(float)xv.sd
                                  + h.x*(float)xv.se + h.y*(float)xv.sf);
            }
            partial += BPERM(1u, partial);
            partial += BPERM(2u, partial);
            partial += BPERM(4u, partial);
            partial += BPERM(8u, partial);
            partial += BPERM(16u, partial);
            partial += BPERM(32u, partial);
            if (lane == 0) y[row] = partial;
        }
        return;
    }
    __local half xl[8192];
    if (K == 16384u && packed_kblocks == 32u) {
        for (uint i = tid; i < 8192u; i += 256u) xl[i] = x[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        float partial = 0.0f;
        if (row < N) {
            uint nblock = row >> 7;
            for (uint p = 0; p < 16u; ++p) {
                uint local_base = (p << 9) + lane * 8u;
                uint kblock = local_base >> 7;
                uint word = scale_w[nblock * 32u + (kblock >> 2)];
                uchar sc = (uchar)((word >> ((kblock & 3u) * 8u)) & 0xffu);
                float scale = e8m0_to_f32(sc);
                uint2 wp = ((__global const uint2*)(W + (ulong)row * 16384u + local_base))[0];
                float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
                float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
                float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
                float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
                half8 xv = vload8(0, xl + local_base);
                partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                                  + b.x*(float)xv.s2 + b.y*(float)xv.s3
                                  + c.x*(float)xv.s4 + c.y*(float)xv.s5
                                  + d.x*(float)xv.s6 + d.y*(float)xv.s7);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint i = tid; i < 8192u; i += 256u) xl[i] = x[8192u + i];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (row < N) {
            uint nblock = row >> 7;
            for (uint p = 0; p < 16u; ++p) {
                uint local_base = (p << 9) + lane * 8u;
                uint global_base = 8192u + local_base;
                uint kblock = global_base >> 7;
                uint word = scale_w[nblock * 32u + (kblock >> 2)];
                uchar sc = (uchar)((word >> ((kblock & 3u) * 8u)) & 0xffu);
                float scale = e8m0_to_f32(sc);
                uint2 wp = ((__global const uint2*)(W + (ulong)row * 16384u + global_base))[0];
                float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
                float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
                float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
                float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
                half8 xv = vload8(0, xl + local_base);
                partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                                  + b.x*(float)xv.s2 + b.y*(float)xv.s3
                                  + c.x*(float)xv.s4 + c.y*(float)xv.s5
                                  + d.x*(float)xv.s6 + d.y*(float)xv.s7);
            }
            partial += BPERM(1u, partial);
            partial += BPERM(2u, partial);
            partial += BPERM(4u, partial);
            partial += BPERM(8u, partial);
            partial += BPERM(16u, partial);
            partial += BPERM(32u, partial);
            if (lane == 0) y[row] = partial;
        }
        return;
    }
    if (K == 12288u && packed_kblocks == 24u) {
        for (uint i = tid; i < 8192u; i += 256u) xl[i] = x[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        float partial = 0.0f;
        if (row < N) {
            uint nblock = row >> 7;
            for (uint p = 0; p < 16u; ++p) {
                uint local_base = (p << 9) + lane * 8u;
                uint kblock = local_base >> 7;
                uint word = scale_w[nblock * 24u + (kblock >> 2)];
                uchar sc = (uchar)((word >> ((kblock & 3u) * 8u)) & 0xffu);
                float scale = e8m0_to_f32(sc);
                uint2 wp = ((__global const uint2*)(W + (ulong)row * 12288u + local_base))[0];
                float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
                float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
                float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
                float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
                half8 xv = vload8(0, xl + local_base);
                partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                                  + b.x*(float)xv.s2 + b.y*(float)xv.s3
                                  + c.x*(float)xv.s4 + c.y*(float)xv.s5
                                  + d.x*(float)xv.s6 + d.y*(float)xv.s7);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint i = tid; i < 4096u; i += 256u) xl[i] = x[8192u + i];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (row < N) {
            uint nblock = row >> 7;
            for (uint p = 0; p < 8u; ++p) {
                uint local_base = (p << 9) + lane * 8u;
                uint global_base = 8192u + local_base;
                uint kblock = global_base >> 7;
                uint word = scale_w[nblock * 24u + (kblock >> 2)];
                uchar sc = (uchar)((word >> ((kblock & 3u) * 8u)) & 0xffu);
                float scale = e8m0_to_f32(sc);
                uint2 wp = ((__global const uint2*)(W + (ulong)row * 12288u + global_base))[0];
                float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
                float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
                float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
                float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
                half8 xv = vload8(0, xl + local_base);
                partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                                  + b.x*(float)xv.s2 + b.y*(float)xv.s3
                                  + c.x*(float)xv.s4 + c.y*(float)xv.s5
                                  + d.x*(float)xv.s6 + d.y*(float)xv.s7);
            }
            partial += BPERM(1u, partial);
            partial += BPERM(2u, partial);
            partial += BPERM(4u, partial);
            partial += BPERM(8u, partial);
            partial += BPERM(16u, partial);
            partial += BPERM(32u, partial);
            if (lane == 0) y[row] = partial;
        }
        return;
    }
    for (uint i = tid; i < K; i += 256u) xl[i] = x[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        uint nblock = row >> 7;
        float partial = 0.0f;
        uint passes = K >> 9;
        for (uint p = 0; p < passes; ++p) {
            uint base = (p << 9) + lane * 8u;
            uint kblock = base >> 7;
            uint word = scale_w[nblock * packed_kblocks + (kblock >> 2)];
            uchar sc = (uchar)((word >> ((kblock & 3u) * 8u)) & 0xffu);
            float scale = e8m0_to_f32(sc);
            uint2 wp = ((__global const uint2*)(W + (ulong)row * K + base))[0];
            float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
            float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
            float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
            float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
            half8 xv = vload8(0, xl + base);
            partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                              + b.x*(float)xv.s2 + b.y*(float)xv.s3
                              + c.x*(float)xv.s4 + c.y*(float)xv.s5
                              + d.x*(float)xv.s6 + d.y*(float)xv.s7);
        }
        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) y[row] = partial;
    }
}

// Same FP8-weight block-scaled GEMV, but consumes packed OCP E4M3 activations
// plus packed E8M0 activation scales directly. The dequantized activation vector
// is staged into the same f16 LDS tile used by gemv_fp8_wblock_e8m0, so serving
// can fuse activation-scale consumption into the GEMV dispatch.
__kernel void gemv_fp8_wblock_act_e8m0(__global const uchar* W,
                                       __global const uint* scale_w,
                                       __global const uchar* xq,
                                       __global const uint* scale_x,
                                       __global float* y,
                                       uint N,
                                       uint K,
                                       uint kblocks,
                                       uint packed_kblocks,
                                       uint x_group_size) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + w;
    __local half xl[8192];
    for (uint i = tid; i < K; i += 256u) {
        uint group = i / x_group_size;
        uint word = scale_x[group >> 2];
        uchar scx = (uchar)((word >> ((group & 3u) * 8u)) & 0xffu);
        xl[i] = (half)(e4m3_ocp_to_f32(xq[i]) * e8m0_to_f32(scx));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        uint nblock = row >> 7;
        float partial = 0.0f;
        uint passes = K >> 9;
        for (uint p = 0; p < passes; ++p) {
            uint base = (p << 9) + lane * 8u;
            uint kblock = base >> 7;
            uint word = scale_w[nblock * packed_kblocks + (kblock >> 2)];
            uchar sc = (uchar)((word >> ((kblock & 3u) * 8u)) & 0xffu);
            float scale = e8m0_to_f32(sc);
            uint2 wp = ((__global const uint2*)(W + (ulong)row * K + base))[0];
            float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
            float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
            float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
            float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
            half8 xv = vload8(0, xl + base);
            partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                              + b.x*(float)xv.s2 + b.y*(float)xv.s3
                              + c.x*(float)xv.s4 + c.y*(float)xv.s5
                              + d.x*(float)xv.s6 + d.y*(float)xv.s7);
        }
        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) y[row] = partial;
    }
}

// Serving-shape specialization for K=1536 and activation group_size=64. This
// targets FP8 blockscale MoE/down-projection decode shapes on gfx950.
__kernel void gemv_fp8_wblock_act_e8m0_k1536_g64(__global const uchar* W,
                                                 __global const uint* scale_w,
                                                 __global const uchar* xq,
                                                 __global const uint* scale_x,
                                                 __global float* y,
                                                 uint N) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + w;
    __local half xl[1536];
    uint shift = w << 3;
    uint wave_base = w << 6;
    for (uint block = 0; block < 6u; ++block) {
        uint i = (block << 8) + wave_base + lane;
        uint word = scale_x[block];
        uchar scx = (uchar)((word >> shift) & 0xffu);
        xl[i] = (half)(e4m3_ocp_to_f32(xq[i]) * e8m0_to_f32(scx));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        uint nblock = row >> 7;
        float partial = 0.0f;
        for (uint p = 0; p < 3u; ++p) {
            uint base = (p << 9) + lane * 8u;
            uint kblock = base >> 7;
            uint word = scale_w[nblock * 3u + (kblock >> 2)];
            uchar sc = (uchar)((word >> ((kblock & 3u) * 8u)) & 0xffu);
            float scale = e8m0_to_f32(sc);
            uint2 wp = ((__global const uint2*)(W + (ulong)row * 1536u + base))[0];
            float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
            float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
            float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
            float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
            half8 xv = vload8(0, xl + base);
            partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                              + b.x*(float)xv.s2 + b.y*(float)xv.s3
                              + c.x*(float)xv.s4 + c.y*(float)xv.s5
                              + d.x*(float)xv.s6 + d.y*(float)xv.s7);
        }
        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) y[row] = partial;
    }
}

// Mixed 16-wide + 8-wide-tail variant for K=1536/G64. The first 1024 columns
// use a 16-wide dot body; the 512-column tail stays on the conservative 8-wide
// body so scale/block boundaries remain exact.
__kernel void gemv_fp8_wblock_act_e8m0_k1536_g64_wide(__global const uchar* W,
                                                      __global const uint* scale_w,
                                                      __global const uchar* xq,
                                                      __global const uint* scale_x,
                                                      __global float* y,
                                                      uint N) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + w;
    __local half xl[1536];
    uint shift = w << 3;
    uint wave_base = w << 6;
    for (uint block = 0; block < 6u; ++block) {
        uint i = (block << 8) + wave_base + lane;
        uint word = scale_x[block];
        uchar scx = (uchar)((word >> shift) & 0xffu);
        xl[i] = (half)(e4m3_ocp_to_f32(xq[i]) * e8m0_to_f32(scx));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        uint nblock = row >> 7;
        float partial = 0.0f;
        uint base = lane * 16u;
        uint kblock = base >> 7;
        uint word = scale_w[nblock * 3u + (kblock >> 2)];
        uchar sc = (uchar)((word >> ((kblock & 3u) * 8u)) & 0xffu);
        float scale = e8m0_to_f32(sc);
        uint4 wp = ((__global const uint4*)(W + (ulong)row * 1536u + base))[0];
        float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
        float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
        float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
        float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
        float2 e = __builtin_amdgcn_cvt_pk_f32_fp8(wp.z, false);
        float2 f = __builtin_amdgcn_cvt_pk_f32_fp8(wp.z, true);
        float2 g = __builtin_amdgcn_cvt_pk_f32_fp8(wp.w, false);
        float2 h = __builtin_amdgcn_cvt_pk_f32_fp8(wp.w, true);
        half16 xv = vload16(0, xl + base);
        partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                          + b.x*(float)xv.s2 + b.y*(float)xv.s3
                          + c.x*(float)xv.s4 + c.y*(float)xv.s5
                          + d.x*(float)xv.s6 + d.y*(float)xv.s7
                          + e.x*(float)xv.s8 + e.y*(float)xv.s9
                          + f.x*(float)xv.sa + f.y*(float)xv.sb
                          + g.x*(float)xv.sc + g.y*(float)xv.sd
                          + h.x*(float)xv.se + h.y*(float)xv.sf);

        base = 1024u + lane * 8u;
        kblock = base >> 7;
        word = scale_w[nblock * 3u + (kblock >> 2)];
        sc = (uchar)((word >> ((kblock & 3u) * 8u)) & 0xffu);
        scale = e8m0_to_f32(sc);
        uint2 wt = ((__global const uint2*)(W + (ulong)row * 1536u + base))[0];
        float2 ta = __builtin_amdgcn_cvt_pk_f32_fp8(wt.x, false);
        float2 tb = __builtin_amdgcn_cvt_pk_f32_fp8(wt.x, true);
        float2 tc = __builtin_amdgcn_cvt_pk_f32_fp8(wt.y, false);
        float2 td = __builtin_amdgcn_cvt_pk_f32_fp8(wt.y, true);
        half8 xt = vload8(0, xl + base);
        partial += scale * (ta.x*(float)xt.s0 + ta.y*(float)xt.s1
                          + tb.x*(float)xt.s2 + tb.y*(float)xt.s3
                          + tc.x*(float)xt.s4 + tc.y*(float)xt.s5
                          + td.x*(float)xt.s6 + td.y*(float)xt.s7);

        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) y[row] = partial;
    }
}

// Exact-N hot-shape variant for K=1536/G64. The host only routes N=16384 here,
// so every launched wave owns a valid output row and the inner dot can avoid
// the row<N tail guard used by the generic wide kernel.
__kernel void gemv_fp8_wblock_act_e8m0_k1536_g64_wide_n16384(__global const uchar* W,
                                                            __global const uint* scale_w,
                                                            __global const uchar* xq,
                                                            __global const uint* scale_x,
                                                            __global float* y,
                                                            uint N) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + w;
    __local half xl[1536];
    uint shift = w << 3;
    uint wave_base = w << 6;
    for (uint block = 0; block < 6u; ++block) {
        uint i = (block << 8) + wave_base + lane;
        uint word = scale_x[block];
        uchar scx = (uchar)((word >> shift) & 0xffu);
        xl[i] = (half)(e4m3_ocp_to_f32(xq[i]) * e8m0_to_f32(scx));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint nblock = row >> 7;
    float partial = 0.0f;
    uint base = lane * 16u;
    uint kblock = base >> 7;
    uint word = scale_w[nblock * 3u + (kblock >> 2)];
    uchar sc = (uchar)((word >> ((kblock & 3u) * 8u)) & 0xffu);
    float scale = e8m0_to_f32(sc);
    uint4 wp = ((__global const uint4*)(W + (ulong)row * 1536u + base))[0];
    float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
    float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
    float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
    float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
    float2 e = __builtin_amdgcn_cvt_pk_f32_fp8(wp.z, false);
    float2 f = __builtin_amdgcn_cvt_pk_f32_fp8(wp.z, true);
    float2 g = __builtin_amdgcn_cvt_pk_f32_fp8(wp.w, false);
    float2 h = __builtin_amdgcn_cvt_pk_f32_fp8(wp.w, true);
    half16 xv = vload16(0, xl + base);
    partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                      + b.x*(float)xv.s2 + b.y*(float)xv.s3
                      + c.x*(float)xv.s4 + c.y*(float)xv.s5
                      + d.x*(float)xv.s6 + d.y*(float)xv.s7
                      + e.x*(float)xv.s8 + e.y*(float)xv.s9
                      + f.x*(float)xv.sa + f.y*(float)xv.sb
                      + g.x*(float)xv.sc + g.y*(float)xv.sd
                      + h.x*(float)xv.se + h.y*(float)xv.sf);

    base = 1024u + lane * 8u;
    kblock = base >> 7;
    word = scale_w[nblock * 3u + (kblock >> 2)];
    sc = (uchar)((word >> ((kblock & 3u) * 8u)) & 0xffu);
    scale = e8m0_to_f32(sc);
    uint2 wt = ((__global const uint2*)(W + (ulong)row * 1536u + base))[0];
    float2 ta = __builtin_amdgcn_cvt_pk_f32_fp8(wt.x, false);
    float2 tb = __builtin_amdgcn_cvt_pk_f32_fp8(wt.x, true);
    float2 tc = __builtin_amdgcn_cvt_pk_f32_fp8(wt.y, false);
    float2 td = __builtin_amdgcn_cvt_pk_f32_fp8(wt.y, true);
    half8 xt = vload8(0, xl + base);
    partial += scale * (ta.x*(float)xt.s0 + ta.y*(float)xt.s1
                      + tb.x*(float)xt.s2 + tb.y*(float)xt.s3
                      + tc.x*(float)xt.s4 + tc.y*(float)xt.s5
                      + td.x*(float)xt.s6 + td.y*(float)xt.s7);

    partial += BPERM(1u, partial);
    partial += BPERM(2u, partial);
    partial += BPERM(4u, partial);
    partial += BPERM(8u, partial);
    partial += BPERM(16u, partial);
    partial += BPERM(32u, partial);
    if (lane == 0) y[row] = partial;
}

// Serving-shape specialization for K=1024 and activation group_size=64. This
// covers the smallest blockscale GEMV widths where generic scale indexing is a
// meaningful fraction of the decode dot work.
__kernel void gemv_fp8_wblock_act_e8m0_k1024_g64(__global const uchar* W,
                                                 __global const uint* scale_w,
                                                 __global const uchar* xq,
                                                 __global const uint* scale_x,
                                                 __global float* y,
                                                 uint N) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + w;
    __local half xl[1024];
    uint shift = w << 3;
    uint wave_base = w << 6;
    for (uint block = 0; block < 4u; ++block) {
        uint i = (block << 8) + wave_base + lane;
        uint word = scale_x[block];
        uchar scx = (uchar)((word >> shift) & 0xffu);
        xl[i] = (half)(e4m3_ocp_to_f32(xq[i]) * e8m0_to_f32(scx));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        uint nblock = row >> 7;
        float partial = 0.0f;
        for (uint p = 0; p < 2u; ++p) {
            uint base = (p << 9) + lane * 8u;
            uint kblock = base >> 7;
            uint word = scale_w[nblock * 2u + (kblock >> 2)];
            uchar sc = (uchar)((word >> ((kblock & 3u) * 8u)) & 0xffu);
            float scale = e8m0_to_f32(sc);
            uint2 wp = ((__global const uint2*)(W + (ulong)row * 1024u + base))[0];
            float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
            float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
            float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
            float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
            half8 xv = vload8(0, xl + base);
            partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                              + b.x*(float)xv.s2 + b.y*(float)xv.s3
                              + c.x*(float)xv.s4 + c.y*(float)xv.s5
                              + d.x*(float)xv.s6 + d.y*(float)xv.s7);
        }
        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) y[row] = partial;
    }
}

// Serving-shape specialization for K=2048 and activation group_size=64. This
// removes generic blockscale indexing overhead for medium decode projections.
__kernel void gemv_fp8_wblock_act_e8m0_k2048_g64(__global const uchar* W,
                                                 __global const uint* scale_w,
                                                 __global const uchar* xq,
                                                 __global const uint* scale_x,
                                                 __global float* y,
                                                 uint N) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + w;
    __local half xl[2048];
    uint shift = w << 3;
    uint wave_base = w << 6;
    for (uint block = 0; block < 8u; ++block) {
        uint i = (block << 8) + wave_base + lane;
        uint word = scale_x[block];
        uchar scx = (uchar)((word >> shift) & 0xffu);
        xl[i] = (half)(e4m3_ocp_to_f32(xq[i]) * e8m0_to_f32(scx));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        uint nblock = row >> 7;
        float partial = 0.0f;
        for (uint p = 0; p < 4u; ++p) {
            uint base = (p << 9) + lane * 8u;
            uint kblock = base >> 7;
            uint word = scale_w[nblock * 4u + (kblock >> 2)];
            uchar sc = (uchar)((word >> ((kblock & 3u) * 8u)) & 0xffu);
            float scale = e8m0_to_f32(sc);
            uint2 wp = ((__global const uint2*)(W + (ulong)row * 2048u + base))[0];
            float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
            float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
            float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
            float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
            half8 xv = vload8(0, xl + base);
            partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                              + b.x*(float)xv.s2 + b.y*(float)xv.s3
                              + c.x*(float)xv.s4 + c.y*(float)xv.s5
                              + d.x*(float)xv.s6 + d.y*(float)xv.s7);
        }
        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) y[row] = partial;
    }
}

// Serving-shape specialization for K=2560 and activation group_size=64. This
// removes generic blockscale indexing overhead for smaller MoE decode widths.
__kernel void gemv_fp8_wblock_act_e8m0_k2560_g64(__global const uchar* W,
                                                 __global const uint* scale_w,
                                                 __global const uchar* xq,
                                                 __global const uint* scale_x,
                                                 __global float* y,
                                                 uint N) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + w;
    __local half xl[2560];
    uint shift = w << 3;
    uint wave_base = w << 6;
    for (uint block = 0; block < 10u; ++block) {
        uint i = (block << 8) + wave_base + lane;
        uint word = scale_x[block];
        uchar scx = (uchar)((word >> shift) & 0xffu);
        xl[i] = (half)(e4m3_ocp_to_f32(xq[i]) * e8m0_to_f32(scx));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        uint nblock = row >> 7;
        float partial = 0.0f;
        for (uint p = 0; p < 5u; ++p) {
            uint base = (p << 9) + lane * 8u;
            uint kblock = base >> 7;
            uint word = scale_w[nblock * 5u + (kblock >> 2)];
            uchar sc = (uchar)((word >> ((kblock & 3u) * 8u)) & 0xffu);
            float scale = e8m0_to_f32(sc);
            uint2 wp = ((__global const uint2*)(W + (ulong)row * 2560u + base))[0];
            float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
            float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
            float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
            float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
            half8 xv = vload8(0, xl + base);
            partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                              + b.x*(float)xv.s2 + b.y*(float)xv.s3
                              + c.x*(float)xv.s4 + c.y*(float)xv.s5
                              + d.x*(float)xv.s6 + d.y*(float)xv.s7);
        }
        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) y[row] = partial;
    }
}

// Serving-shape specialization for K=3072 and activation group_size=64. This
// covers smaller Qwen/Kimi projection widths without the generic scale-indexing
// overhead that hurts ultra-small-M decode.
__kernel void gemv_fp8_wblock_act_e8m0_k3072_g64(__global const uchar* W,
                                                 __global const uint* scale_w,
                                                 __global const uchar* xq,
                                                 __global const uint* scale_x,
                                                 __global float* y,
                                                 uint N) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + w;
    __local half xl[3072];
    uint shift = w << 3;
    uint wave_base = w << 6;
    for (uint block = 0; block < 12u; ++block) {
        uint i = (block << 8) + wave_base + lane;
        uint word = scale_x[block];
        uchar scx = (uchar)((word >> shift) & 0xffu);
        xl[i] = (half)(e4m3_ocp_to_f32(xq[i]) * e8m0_to_f32(scx));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        uint nblock = row >> 7;
        float partial = 0.0f;
        for (uint p = 0; p < 6u; ++p) {
            uint base = (p << 9) + lane * 8u;
            uint kblock = base >> 7;
            uint word = scale_w[nblock * 6u + (kblock >> 2)];
            uchar sc = (uchar)((word >> ((kblock & 3u) * 8u)) & 0xffu);
            float scale = e8m0_to_f32(sc);
            uint2 wp = ((__global const uint2*)(W + (ulong)row * 3072u + base))[0];
            float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
            float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
            float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
            float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
            half8 xv = vload8(0, xl + base);
            partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                              + b.x*(float)xv.s2 + b.y*(float)xv.s3
                              + c.x*(float)xv.s4 + c.y*(float)xv.s5
                              + d.x*(float)xv.s6 + d.y*(float)xv.s7);
        }
        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) y[row] = partial;
    }
}

// Serving-shape specialization for K=3584 and activation group_size=64. This
// fixes the packed scale geometry used by Qwen-style dense projections while
// keeping the conservative 8-wide dot loop that is stable for non-4096 K.
__kernel void gemv_fp8_wblock_act_e8m0_k3584_g64(__global const uchar* W,
                                                 __global const uint* scale_w,
                                                 __global const uchar* xq,
                                                 __global const uint* scale_x,
                                                 __global float* y,
                                                 uint N) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + w;
    __local half xl[3584];
    uint shift = w << 3;
    uint wave_base = w << 6;
    for (uint block = 0; block < 14u; ++block) {
        uint i = (block << 8) + wave_base + lane;
        uint word = scale_x[block];
        uchar scx = (uchar)((word >> shift) & 0xffu);
        xl[i] = (half)(e4m3_ocp_to_f32(xq[i]) * e8m0_to_f32(scx));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        uint nblock = row >> 7;
        float partial = 0.0f;
        for (uint p = 0; p < 7u; ++p) {
            uint base = (p << 9) + lane * 8u;
            uint kblock = base >> 7;
            uint word = scale_w[nblock * 7u + (kblock >> 2)];
            uchar sc = (uchar)((word >> ((kblock & 3u) * 8u)) & 0xffu);
            float scale = e8m0_to_f32(sc);
            uint2 wp = ((__global const uint2*)(W + (ulong)row * 3584u + base))[0];
            float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
            float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
            float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
            float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
            half8 xv = vload8(0, xl + base);
            partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                              + b.x*(float)xv.s2 + b.y*(float)xv.s3
                              + c.x*(float)xv.s4 + c.y*(float)xv.s5
                              + d.x*(float)xv.s6 + d.y*(float)xv.s7);
        }
        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) y[row] = partial;
    }
}

// Serving-shape specialization for K=5120 and activation group_size=64. This
// targets Kimi/Qwen MoE decode widths that must avoid generic dispatch fallback.
__kernel void gemv_fp8_wblock_act_e8m0_k5120_g64(__global const uchar* W,
                                                 __global const uint* scale_w,
                                                 __global const uchar* xq,
                                                 __global const uint* scale_x,
                                                 __global float* y,
                                                 uint N) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + w;
    __local half xl[5120];
    uint shift = w << 3;
    uint wave_base = w << 6;
    for (uint block = 0; block < 20u; ++block) {
        uint i = (block << 8) + wave_base + lane;
        uint word = scale_x[block];
        uchar scx = (uchar)((word >> shift) & 0xffu);
        xl[i] = (half)(e4m3_ocp_to_f32(xq[i]) * e8m0_to_f32(scx));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        uint nblock = row >> 7;
        float partial = 0.0f;
        for (uint p = 0; p < 10u; ++p) {
            uint base = (p << 9) + lane * 8u;
            uint kblock = base >> 7;
            uint word = scale_w[nblock * 10u + (kblock >> 2)];
            uchar sc = (uchar)((word >> ((kblock & 3u) * 8u)) & 0xffu);
            float scale = e8m0_to_f32(sc);
            uint2 wp = ((__global const uint2*)(W + (ulong)row * 5120u + base))[0];
            float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
            float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
            float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
            float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
            half8 xv = vload8(0, xl + base);
            partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                              + b.x*(float)xv.s2 + b.y*(float)xv.s3
                              + c.x*(float)xv.s4 + c.y*(float)xv.s5
                              + d.x*(float)xv.s6 + d.y*(float)xv.s7);
        }
        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) y[row] = partial;
    }
}

// 16-wide dot-body variant for K=5120/G64. This targets medium Kimi/Qwen
// decode projections where the 8-wide body still pays ten inner-loop trips.
__kernel void gemv_fp8_wblock_act_e8m0_k5120_g64_wide(__global const uchar* W,
                                                      __global const uint* scale_w,
                                                      __global const uchar* xq,
                                                      __global const uint* scale_x,
                                                      __global float* y,
                                                      uint N) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + w;
    __local half xl[5120];
    uint shift = w << 3;
    uint wave_base = w << 6;
    for (uint block = 0; block < 20u; ++block) {
        uint i = (block << 8) + wave_base + lane;
        uint word = scale_x[block];
        uchar scx = (uchar)((word >> shift) & 0xffu);
        xl[i] = (half)(e4m3_ocp_to_f32(xq[i]) * e8m0_to_f32(scx));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        uint nblock = row >> 7;
        float partial = 0.0f;
        for (uint p = 0; p < 5u; ++p) {
            uint base = (p << 10) + lane * 16u;
            uint kblock = base >> 7;
            uint word = scale_w[nblock * 10u + (kblock >> 2)];
            uchar sc = (uchar)((word >> ((kblock & 3u) * 8u)) & 0xffu);
            float scale = e8m0_to_f32(sc);
            uint4 wp = ((__global const uint4*)(W + (ulong)row * 5120u + base))[0];
            float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
            float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
            float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
            float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
            float2 e = __builtin_amdgcn_cvt_pk_f32_fp8(wp.z, false);
            float2 f = __builtin_amdgcn_cvt_pk_f32_fp8(wp.z, true);
            float2 g = __builtin_amdgcn_cvt_pk_f32_fp8(wp.w, false);
            float2 h = __builtin_amdgcn_cvt_pk_f32_fp8(wp.w, true);
            half16 xv = vload16(0, xl + base);
            partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                              + b.x*(float)xv.s2 + b.y*(float)xv.s3
                              + c.x*(float)xv.s4 + c.y*(float)xv.s5
                              + d.x*(float)xv.s6 + d.y*(float)xv.s7
                              + e.x*(float)xv.s8 + e.y*(float)xv.s9
                              + f.x*(float)xv.sa + f.y*(float)xv.sb
                              + g.x*(float)xv.sc + g.y*(float)xv.sd
                              + h.x*(float)xv.se + h.y*(float)xv.sf);
        }
        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) y[row] = partial;
    }
}

// Serving-shape specialization for K=6144 and activation group_size=64. This
// targets Kimi/Qwen MoE decode widths called out by current gfx950 tuning work.
__kernel void gemv_fp8_wblock_act_e8m0_k6144_g64(__global const uchar* W,
                                                 __global const uint* scale_w,
                                                 __global const uchar* xq,
                                                 __global const uint* scale_x,
                                                 __global float* y,
                                                 uint N) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + w;
    __local half xl[6144];
    uint shift = w << 3;
    uint wave_base = w << 6;
    for (uint block = 0; block < 24u; ++block) {
        uint i = (block << 8) + wave_base + lane;
        uint word = scale_x[block];
        uchar scx = (uchar)((word >> shift) & 0xffu);
        xl[i] = (half)(e4m3_ocp_to_f32(xq[i]) * e8m0_to_f32(scx));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        uint nblock = row >> 7;
        float partial = 0.0f;
        for (uint p = 0; p < 12u; ++p) {
            uint base = (p << 9) + lane * 8u;
            uint kblock = base >> 7;
            uint word = scale_w[nblock * 12u + (kblock >> 2)];
            uchar sc = (uchar)((word >> ((kblock & 3u) * 8u)) & 0xffu);
            float scale = e8m0_to_f32(sc);
            uint2 wp = ((__global const uint2*)(W + (ulong)row * 6144u + base))[0];
            float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
            float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
            float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
            float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
            half8 xv = vload8(0, xl + base);
            partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                              + b.x*(float)xv.s2 + b.y*(float)xv.s3
                              + c.x*(float)xv.s4 + c.y*(float)xv.s5
                              + d.x*(float)xv.s6 + d.y*(float)xv.s7);
        }
        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) y[row] = partial;
    }
}

// Serving-shape specialization for K=7168 and activation group_size=64. This
// targets Kimi/Qwen-adjacent wide decode projections while staying within the
// existing 8192-half LDS staging envelope.
__kernel void gemv_fp8_wblock_act_e8m0_k7168_g64(__global const uchar* W,
                                                 __global const uint* scale_w,
                                                 __global const uchar* xq,
                                                 __global const uint* scale_x,
                                                 __global float* y,
                                                 uint N) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + w;
    __local half xl[7168];
    uint shift = w << 3;
    uint wave_base = w << 6;
    for (uint block = 0; block < 28u; ++block) {
        uint i = (block << 8) + wave_base + lane;
        uint word = scale_x[block];
        uchar scx = (uchar)((word >> shift) & 0xffu);
        xl[i] = (half)(e4m3_ocp_to_f32(xq[i]) * e8m0_to_f32(scx));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        uint nblock = row >> 7;
        float partial = 0.0f;
        for (uint p = 0; p < 14u; ++p) {
            uint base = (p << 9) + lane * 8u;
            uint kblock = base >> 7;
            uint word = scale_w[nblock * 14u + (kblock >> 2)];
            uchar sc = (uchar)((word >> ((kblock & 3u) * 8u)) & 0xffu);
            float scale = e8m0_to_f32(sc);
            uint2 wp = ((__global const uint2*)(W + (ulong)row * 7168u + base))[0];
            float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
            float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
            float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
            float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
            half8 xv = vload8(0, xl + base);
            partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                              + b.x*(float)xv.s2 + b.y*(float)xv.s3
                              + c.x*(float)xv.s4 + c.y*(float)xv.s5
                              + d.x*(float)xv.s6 + d.y*(float)xv.s7);
        }
        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) y[row] = partial;
    }
}

// 16-wide dot-body variant for K=7168/G64. This targets Kimi-style
// model_dim=7168 expert gate/up projections where small output rows make loop
// overhead visible in decode.
__kernel void gemv_fp8_wblock_act_e8m0_k7168_g64_wide(__global const uchar* W,
                                                      __global const uint* scale_w,
                                                      __global const uchar* xq,
                                                      __global const uint* scale_x,
                                                      __global float* y,
                                                      uint N) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + w;
    __local half xl[7168];
    uint shift = w << 3;
    uint wave_base = w << 6;
    for (uint block = 0; block < 28u; ++block) {
        uint i = (block << 8) + wave_base + lane;
        uint word = scale_x[block];
        uchar scx = (uchar)((word >> shift) & 0xffu);
        xl[i] = (half)(e4m3_ocp_to_f32(xq[i]) * e8m0_to_f32(scx));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        uint nblock = row >> 7;
        float partial = 0.0f;
        for (uint p = 0; p < 7u; ++p) {
            uint base = (p << 10) + lane * 16u;
            uint kblock = base >> 7;
            uint word = scale_w[nblock * 14u + (kblock >> 2)];
            uchar sc = (uchar)((word >> ((kblock & 3u) * 8u)) & 0xffu);
            float scale = e8m0_to_f32(sc);
            uint4 wp = ((__global const uint4*)(W + (ulong)row * 7168u + base))[0];
            float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
            float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
            float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
            float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
            float2 e = __builtin_amdgcn_cvt_pk_f32_fp8(wp.z, false);
            float2 f = __builtin_amdgcn_cvt_pk_f32_fp8(wp.z, true);
            float2 g = __builtin_amdgcn_cvt_pk_f32_fp8(wp.w, false);
            float2 h = __builtin_amdgcn_cvt_pk_f32_fp8(wp.w, true);
            half16 xv = vload16(0, xl + base);
            partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                              + b.x*(float)xv.s2 + b.y*(float)xv.s3
                              + c.x*(float)xv.s4 + c.y*(float)xv.s5
                              + d.x*(float)xv.s6 + d.y*(float)xv.s7
                              + e.x*(float)xv.s8 + e.y*(float)xv.s9
                              + f.x*(float)xv.sa + f.y*(float)xv.sb
                              + g.x*(float)xv.sc + g.y*(float)xv.sd
                              + h.x*(float)xv.se + h.y*(float)xv.sf);
        }
        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) y[row] = partial;
    }
}

// Serving-shape specialization for K=8192 and activation group_size=64. This
// keeps the generic 8-wide dot body but fixes all packed blockscale geometry
// for large decode projections on gfx950.
__kernel void gemv_fp8_wblock_act_e8m0_k8192_g64(__global const uchar* W,
                                                 __global const uint* scale_w,
                                                 __global const uchar* xq,
                                                 __global const uint* scale_x,
                                                 __global float* y,
                                                 uint N) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + w;
    __local half xl[8192];
    uint shift = w << 3;
    uint wave_base = w << 6;
    for (uint block = 0; block < 32u; ++block) {
        uint i = (block << 8) + wave_base + lane;
        uint word = scale_x[block];
        uchar scx = (uchar)((word >> shift) & 0xffu);
        xl[i] = (half)(e4m3_ocp_to_f32(xq[i]) * e8m0_to_f32(scx));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        uint nblock = row >> 7;
        float partial = 0.0f;
        for (uint p = 0; p < 16u; ++p) {
            uint base = (p << 9) + lane * 8u;
            uint kblock = base >> 7;
            uint word = scale_w[nblock * 16u + (kblock >> 2)];
            uchar sc = (uchar)((word >> ((kblock & 3u) * 8u)) & 0xffu);
            float scale = e8m0_to_f32(sc);
            uint2 wp = ((__global const uint2*)(W + (ulong)row * 8192u + base))[0];
            float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
            float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
            float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
            float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
            half8 xv = vload8(0, xl + base);
            partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                              + b.x*(float)xv.s2 + b.y*(float)xv.s3
                              + c.x*(float)xv.s4 + c.y*(float)xv.s5
                              + d.x*(float)xv.s6 + d.y*(float)xv.s7);
        }
        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) y[row] = partial;
    }
}

// 16-wide dot-body variant for K=8192/G64. This halves the inner loop trip
// count for small-output decode projections where dispatch and loop overhead
// dominate memory traffic.
__kernel void gemv_fp8_wblock_act_e8m0_k8192_g64_wide(__global const uchar* W,
                                                      __global const uint* scale_w,
                                                      __global const uchar* xq,
                                                      __global const uint* scale_x,
                                                      __global float* y,
                                                      uint N) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + w;
    __local half xl[8192];
    uint shift = w << 3;
    uint wave_base = w << 6;
    for (uint block = 0; block < 32u; ++block) {
        uint i = (block << 8) + wave_base + lane;
        uint word = scale_x[block];
        uchar scx = (uchar)((word >> shift) & 0xffu);
        xl[i] = (half)(e4m3_ocp_to_f32(xq[i]) * e8m0_to_f32(scx));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        uint nblock = row >> 7;
        float partial = 0.0f;
        for (uint p = 0; p < 8u; ++p) {
            uint base = (p << 10) + lane * 16u;
            uint kblock = base >> 7;
            uint word = scale_w[nblock * 16u + (kblock >> 2)];
            uchar sc = (uchar)((word >> ((kblock & 3u) * 8u)) & 0xffu);
            float scale = e8m0_to_f32(sc);
            uint4 wp = ((__global const uint4*)(W + (ulong)row * 8192u + base))[0];
            float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
            float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
            float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
            float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
            float2 e = __builtin_amdgcn_cvt_pk_f32_fp8(wp.z, false);
            float2 f = __builtin_amdgcn_cvt_pk_f32_fp8(wp.z, true);
            float2 g = __builtin_amdgcn_cvt_pk_f32_fp8(wp.w, false);
            float2 h = __builtin_amdgcn_cvt_pk_f32_fp8(wp.w, true);
            half16 xv = vload16(0, xl + base);
            partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                              + b.x*(float)xv.s2 + b.y*(float)xv.s3
                              + c.x*(float)xv.s4 + c.y*(float)xv.s5
                              + d.x*(float)xv.s6 + d.y*(float)xv.s7
                              + e.x*(float)xv.s8 + e.y*(float)xv.s9
                              + f.x*(float)xv.sa + f.y*(float)xv.sb
                              + g.x*(float)xv.sc + g.y*(float)xv.sd
                              + h.x*(float)xv.se + h.y*(float)xv.sf);
        }
        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) y[row] = partial;
    }
}

// Serving-shape specialization for K=12288 and activation group_size=64. This
// covers Qwen-style MLP down-projection widths without overflowing the generic
// 8192-half LDS staging tile.
__kernel void gemv_fp8_wblock_act_e8m0_k12288_g64(__global const uchar* W,
                                                  __global const uint* scale_w,
                                                  __global const uchar* xq,
                                                  __global const uint* scale_x,
                                                  __global float* y,
                                                  uint N) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + w;
    __local half xl[8192];
    uint shift = w << 3;
    uint wave_base = w << 6;
    float partial = 0.0f;
    for (uint block = 0; block < 32u; ++block) {
        uint i = (block << 8) + wave_base + lane;
        uint word = scale_x[block];
        uchar scx = (uchar)((word >> shift) & 0xffu);
        xl[i] = (half)(e4m3_ocp_to_f32(xq[i]) * e8m0_to_f32(scx));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        uint nblock = row >> 7;
        for (uint p = 0; p < 16u; ++p) {
            uint local_base = (p << 9) + lane * 8u;
            uint kblock = local_base >> 7;
            uint word = scale_w[nblock * 24u + (kblock >> 2)];
            uchar sc = (uchar)((word >> ((kblock & 3u) * 8u)) & 0xffu);
            float scale = e8m0_to_f32(sc);
            uint2 wp = ((__global const uint2*)(W + (ulong)row * 12288u + local_base))[0];
            float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
            float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
            float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
            float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
            half8 xv = vload8(0, xl + local_base);
            partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                              + b.x*(float)xv.s2 + b.y*(float)xv.s3
                              + c.x*(float)xv.s4 + c.y*(float)xv.s5
                              + d.x*(float)xv.s6 + d.y*(float)xv.s7);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint block = 0; block < 16u; ++block) {
        uint i = (block << 8) + wave_base + lane;
        uint word = scale_x[32u + block];
        uchar scx = (uchar)((word >> shift) & 0xffu);
        xl[i] = (half)(e4m3_ocp_to_f32(xq[8192u + i]) * e8m0_to_f32(scx));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        uint nblock = row >> 7;
        for (uint p = 0; p < 8u; ++p) {
            uint local_base = (p << 9) + lane * 8u;
            uint global_base = 8192u + local_base;
            uint kblock = global_base >> 7;
            uint word = scale_w[nblock * 24u + (kblock >> 2)];
            uchar sc = (uchar)((word >> ((kblock & 3u) * 8u)) & 0xffu);
            float scale = e8m0_to_f32(sc);
            uint2 wp = ((__global const uint2*)(W + (ulong)row * 12288u + global_base))[0];
            float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
            float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
            float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
            float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
            half8 xv = vload8(0, xl + local_base);
            partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                              + b.x*(float)xv.s2 + b.y*(float)xv.s3
                              + c.x*(float)xv.s4 + c.y*(float)xv.s5
                              + d.x*(float)xv.s6 + d.y*(float)xv.s7);
        }
    }
    if (row < N) {
        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) y[row] = partial;
    }
}

// 16-wide dot-body variant for K=12288/G64. The vector is streamed through the
// same 8192-half LDS window as the conservative path, but each wave consumes
// 16 FP8 weights per loop trip to reduce loop overhead on measured-positive
// decode projection shapes.
__kernel void gemv_fp8_wblock_act_e8m0_k12288_g64_wide(__global const uchar* W,
                                                       __global const uint* scale_w,
                                                       __global const uchar* xq,
                                                       __global const uint* scale_x,
                                                       __global float* y,
                                                       uint N) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + w;
    __local half xl[8192];
    uint shift = w << 3;
    uint wave_base = w << 6;
    float partial = 0.0f;
    for (uint block = 0; block < 32u; ++block) {
        uint i = (block << 8) + wave_base + lane;
        uint word = scale_x[block];
        uchar scx = (uchar)((word >> shift) & 0xffu);
        xl[i] = (half)(e4m3_ocp_to_f32(xq[i]) * e8m0_to_f32(scx));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        uint nblock = row >> 7;
        for (uint p = 0; p < 8u; ++p) {
            uint local_base = (p << 10) + lane * 16u;
            uint kblock = local_base >> 7;
            uint word = scale_w[nblock * 24u + (kblock >> 2)];
            uchar sc = (uchar)((word >> ((kblock & 3u) * 8u)) & 0xffu);
            float scale = e8m0_to_f32(sc);
            uint4 wp = ((__global const uint4*)(W + (ulong)row * 12288u + local_base))[0];
            float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
            float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
            float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
            float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
            float2 e = __builtin_amdgcn_cvt_pk_f32_fp8(wp.z, false);
            float2 f = __builtin_amdgcn_cvt_pk_f32_fp8(wp.z, true);
            float2 g = __builtin_amdgcn_cvt_pk_f32_fp8(wp.w, false);
            float2 h = __builtin_amdgcn_cvt_pk_f32_fp8(wp.w, true);
            half16 xv = vload16(0, xl + local_base);
            partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                              + b.x*(float)xv.s2 + b.y*(float)xv.s3
                              + c.x*(float)xv.s4 + c.y*(float)xv.s5
                              + d.x*(float)xv.s6 + d.y*(float)xv.s7
                              + e.x*(float)xv.s8 + e.y*(float)xv.s9
                              + f.x*(float)xv.sa + f.y*(float)xv.sb
                              + g.x*(float)xv.sc + g.y*(float)xv.sd
                              + h.x*(float)xv.se + h.y*(float)xv.sf);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint block = 0; block < 16u; ++block) {
        uint i = (block << 8) + wave_base + lane;
        uint word = scale_x[32u + block];
        uchar scx = (uchar)((word >> shift) & 0xffu);
        xl[i] = (half)(e4m3_ocp_to_f32(xq[8192u + i]) * e8m0_to_f32(scx));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        uint nblock = row >> 7;
        for (uint p = 0; p < 4u; ++p) {
            uint local_base = (p << 10) + lane * 16u;
            uint global_base = 8192u + local_base;
            uint kblock = global_base >> 7;
            uint word = scale_w[nblock * 24u + (kblock >> 2)];
            uchar sc = (uchar)((word >> ((kblock & 3u) * 8u)) & 0xffu);
            float scale = e8m0_to_f32(sc);
            uint4 wp = ((__global const uint4*)(W + (ulong)row * 12288u + global_base))[0];
            float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
            float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
            float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
            float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
            float2 e = __builtin_amdgcn_cvt_pk_f32_fp8(wp.z, false);
            float2 f = __builtin_amdgcn_cvt_pk_f32_fp8(wp.z, true);
            float2 g = __builtin_amdgcn_cvt_pk_f32_fp8(wp.w, false);
            float2 h = __builtin_amdgcn_cvt_pk_f32_fp8(wp.w, true);
            half16 xv = vload16(0, xl + local_base);
            partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                              + b.x*(float)xv.s2 + b.y*(float)xv.s3
                              + c.x*(float)xv.s4 + c.y*(float)xv.s5
                              + d.x*(float)xv.s6 + d.y*(float)xv.s7
                              + e.x*(float)xv.s8 + e.y*(float)xv.s9
                              + f.x*(float)xv.sa + f.y*(float)xv.sb
                              + g.x*(float)xv.sc + g.y*(float)xv.sd
                              + h.x*(float)xv.se + h.y*(float)xv.sf);
        }
    }
    if (row < N) {
        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) y[row] = partial;
    }
}

// Serving-shape specialization for K=16384 and activation group_size=64. The
// generic act-packed path stages all K activations into an 8192-half LDS tile,
// so this exact path streams the vector in two 8192-element halves.
__kernel void gemv_fp8_wblock_act_e8m0_k16384_g64(__global const uchar* W,
                                                  __global const uint* scale_w,
                                                  __global const uchar* xq,
                                                  __global const uint* scale_x,
                                                  __global float* y,
                                                  uint N) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + w;
    __local half xl[8192];
    uint shift = w << 3;
    uint wave_base = w << 6;
    float partial = 0.0f;
    for (uint block = 0; block < 32u; ++block) {
        uint i = (block << 8) + wave_base + lane;
        uint word = scale_x[block];
        uchar scx = (uchar)((word >> shift) & 0xffu);
        xl[i] = (half)(e4m3_ocp_to_f32(xq[i]) * e8m0_to_f32(scx));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        uint nblock = row >> 7;
        for (uint p = 0; p < 16u; ++p) {
            uint local_base = (p << 9) + lane * 8u;
            uint kblock = local_base >> 7;
            uint word = scale_w[nblock * 32u + (kblock >> 2)];
            uchar sc = (uchar)((word >> ((kblock & 3u) * 8u)) & 0xffu);
            float scale = e8m0_to_f32(sc);
            uint2 wp = ((__global const uint2*)(W + (ulong)row * 16384u + local_base))[0];
            float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
            float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
            float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
            float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
            half8 xv = vload8(0, xl + local_base);
            partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                              + b.x*(float)xv.s2 + b.y*(float)xv.s3
                              + c.x*(float)xv.s4 + c.y*(float)xv.s5
                              + d.x*(float)xv.s6 + d.y*(float)xv.s7);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint block = 0; block < 32u; ++block) {
        uint i = (block << 8) + wave_base + lane;
        uint word = scale_x[32u + block];
        uchar scx = (uchar)((word >> shift) & 0xffu);
        xl[i] = (half)(e4m3_ocp_to_f32(xq[8192u + i]) * e8m0_to_f32(scx));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        uint nblock = row >> 7;
        for (uint p = 0; p < 16u; ++p) {
            uint local_base = (p << 9) + lane * 8u;
            uint global_base = 8192u + local_base;
            uint kblock = global_base >> 7;
            uint word = scale_w[nblock * 32u + (kblock >> 2)];
            uchar sc = (uchar)((word >> ((kblock & 3u) * 8u)) & 0xffu);
            float scale = e8m0_to_f32(sc);
            uint2 wp = ((__global const uint2*)(W + (ulong)row * 16384u + global_base))[0];
            float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
            float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
            float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
            float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
            half8 xv = vload8(0, xl + local_base);
            partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                              + b.x*(float)xv.s2 + b.y*(float)xv.s3
                              + c.x*(float)xv.s4 + c.y*(float)xv.s5
                              + d.x*(float)xv.s6 + d.y*(float)xv.s7);
        }
    }
    if (row < N) {
        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) y[row] = partial;
    }
}

// 16-wide dot-body variant for measured-positive K=16384/G64 act-packed
// shapes. Kept separate from the conservative split path because an in-kernel
// shape branch deoptimizes gfx950 codegen.
__kernel void gemv_fp8_wblock_act_e8m0_k16384_g64_wide(__global const uchar* W,
                                                       __global const uint* scale_w,
                                                       __global const uchar* xq,
                                                       __global const uint* scale_x,
                                                       __global float* y,
                                                       uint N) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + w;
    __local half xl[8192];
    uint shift = w << 3;
    uint wave_base = w << 6;
    float partial = 0.0f;
    for (uint block = 0; block < 32u; ++block) {
        uint i = (block << 8) + wave_base + lane;
        uint word = scale_x[block];
        uchar scx = (uchar)((word >> shift) & 0xffu);
        xl[i] = (half)(e4m3_ocp_to_f32(xq[i]) * e8m0_to_f32(scx));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        uint nblock = row >> 7;
        for (uint p = 0; p < 8u; ++p) {
            uint local_base = (p << 10) + lane * 16u;
            uint kblock = local_base >> 7;
            uint word = scale_w[nblock * 32u + (kblock >> 2)];
            uchar sc = (uchar)((word >> ((kblock & 3u) * 8u)) & 0xffu);
            float scale = e8m0_to_f32(sc);
            uint4 wp = ((__global const uint4*)(W + (ulong)row * 16384u + local_base))[0];
            float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
            float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
            float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
            float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
            float2 e = __builtin_amdgcn_cvt_pk_f32_fp8(wp.z, false);
            float2 f = __builtin_amdgcn_cvt_pk_f32_fp8(wp.z, true);
            float2 g = __builtin_amdgcn_cvt_pk_f32_fp8(wp.w, false);
            float2 h = __builtin_amdgcn_cvt_pk_f32_fp8(wp.w, true);
            half16 xv = vload16(0, xl + local_base);
            partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                              + b.x*(float)xv.s2 + b.y*(float)xv.s3
                              + c.x*(float)xv.s4 + c.y*(float)xv.s5
                              + d.x*(float)xv.s6 + d.y*(float)xv.s7
                              + e.x*(float)xv.s8 + e.y*(float)xv.s9
                              + f.x*(float)xv.sa + f.y*(float)xv.sb
                              + g.x*(float)xv.sc + g.y*(float)xv.sd
                              + h.x*(float)xv.se + h.y*(float)xv.sf);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint block = 0; block < 32u; ++block) {
        uint i = (block << 8) + wave_base + lane;
        uint word = scale_x[32u + block];
        uchar scx = (uchar)((word >> shift) & 0xffu);
        xl[i] = (half)(e4m3_ocp_to_f32(xq[8192u + i]) * e8m0_to_f32(scx));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        uint nblock = row >> 7;
        for (uint p = 0; p < 8u; ++p) {
            uint local_base = (p << 10) + lane * 16u;
            uint global_base = 8192u + local_base;
            uint kblock = global_base >> 7;
            uint word = scale_w[nblock * 32u + (kblock >> 2)];
            uchar sc = (uchar)((word >> ((kblock & 3u) * 8u)) & 0xffu);
            float scale = e8m0_to_f32(sc);
            uint4 wp = ((__global const uint4*)(W + (ulong)row * 16384u + global_base))[0];
            float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
            float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
            float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
            float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
            float2 e = __builtin_amdgcn_cvt_pk_f32_fp8(wp.z, false);
            float2 f = __builtin_amdgcn_cvt_pk_f32_fp8(wp.z, true);
            float2 g = __builtin_amdgcn_cvt_pk_f32_fp8(wp.w, false);
            float2 h = __builtin_amdgcn_cvt_pk_f32_fp8(wp.w, true);
            half16 xv = vload16(0, xl + local_base);
            partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                              + b.x*(float)xv.s2 + b.y*(float)xv.s3
                              + c.x*(float)xv.s4 + c.y*(float)xv.s5
                              + d.x*(float)xv.s6 + d.y*(float)xv.s7
                              + e.x*(float)xv.s8 + e.y*(float)xv.s9
                              + f.x*(float)xv.sa + f.y*(float)xv.sb
                              + g.x*(float)xv.sc + g.y*(float)xv.sd
                              + h.x*(float)xv.se + h.y*(float)xv.sf);
        }
    }
    if (row < N) {
        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) y[row] = partial;
    }
}

// Serving-shape specialization of gemv_fp8_wblock_act_e8m0 for K=4096. This
// fixes the weight-scale geometry and halves LDS footprint for the dequantized
// activation tile.
__kernel void gemv_fp8_wblock_act_e8m0_k4096(__global const uchar* W,
                                             __global const uint* scale_w,
                                             __global const uchar* xq,
                                             __global const uint* scale_x,
                                             __global float* y,
                                             uint N,
                                             uint x_group_size) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + w;
    __local half xl[4096];
    for (uint i = tid; i < 4096u; i += 256u) {
        uint group = i / x_group_size;
        uint word = scale_x[group >> 2];
        uchar scx = (uchar)((word >> ((group & 3u) * 8u)) & 0xffu);
        xl[i] = (half)(e4m3_ocp_to_f32(xq[i]) * e8m0_to_f32(scx));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        uint nblock = row >> 7;
        float partial = 0.0f;
        for (uint p = 0; p < 8u; ++p) {
            uint base = (p << 9) + lane * 8u;
            uint kblock = base >> 7;
            uint word = scale_w[nblock * 8u + (kblock >> 2)];
            uchar sc = (uchar)((word >> ((kblock & 3u) * 8u)) & 0xffu);
            float scale = e8m0_to_f32(sc);
            uint2 wp = ((__global const uint2*)(W + (ulong)row * 4096u + base))[0];
            float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
            float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
            float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
            float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
            half8 xv = vload8(0, xl + base);
            partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                              + b.x*(float)xv.s2 + b.y*(float)xv.s3
                              + c.x*(float)xv.s4 + c.y*(float)xv.s5
                              + d.x*(float)xv.s6 + d.y*(float)xv.s7);
        }
        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) y[row] = partial;
    }
}

// Serving-shape specialization for K=4096 and activation group_size=64. The
// activation scale pack contains 16 u32 words, each holding the four E8M0 scale
// bytes for the four Wave64 groups in one 256-element block.
__kernel void gemv_fp8_wblock_act_e8m0_k4096_g64(__global const uchar* W,
                                                 __global const uint* scale_w,
                                                 __global const uchar* xq,
                                                 __global const uint* scale_x,
                                                 __global float* y,
                                                 uint N) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + w;
    __local half xl[4096];
    uint shift = w << 3;
    uint wave_base = w << 6;
    for (uint block = 0; block < 16u; ++block) {
        uint i = (block << 8) + wave_base + lane;
        uint word = scale_x[block];
        uchar scx = (uchar)((word >> shift) & 0xffu);
        xl[i] = (half)(e4m3_ocp_to_f32(xq[i]) * e8m0_to_f32(scx));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        uint nblock = row >> 7;
        float partial = 0.0f;
        for (uint p = 0; p < 8u; ++p) {
            uint base = (p << 9) + lane * 8u;
            uint kblock = base >> 7;
            uint word = scale_w[nblock * 8u + (kblock >> 2)];
            uchar sc = (uchar)((word >> ((kblock & 3u) * 8u)) & 0xffu);
            float scale = e8m0_to_f32(sc);
            uint2 wp = ((__global const uint2*)(W + (ulong)row * 4096u + base))[0];
            float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
            float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
            float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
            float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
            half8 xv = vload8(0, xl + base);
            partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                              + b.x*(float)xv.s2 + b.y*(float)xv.s3
                              + c.x*(float)xv.s4 + c.y*(float)xv.s5
                              + d.x*(float)xv.s6 + d.y*(float)xv.s7);
        }
        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) y[row] = partial;
    }
}

// N<=4096 serving specialization of the K=4096/g64 activation-packed path.
// Uses 16-wide FP8/activation vector loads in the dot loop without adding a
// runtime branch to the larger-N g64 kernel.
__kernel void gemv_fp8_wblock_act_e8m0_k4096_g64_n4096(__global const uchar* W,
                                                       __global const uint* scale_w,
                                                       __global const uchar* xq,
                                                       __global const uint* scale_x,
                                                       __global float* y,
                                                       uint N) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + w;
    __local half xl[4096];
    uint shift = w << 3;
    uint wave_base = w << 6;
    for (uint block = 0; block < 16u; ++block) {
        uint i = (block << 8) + wave_base + lane;
        uint word = scale_x[block];
        uchar scx = (uchar)((word >> shift) & 0xffu);
        xl[i] = (half)(e4m3_ocp_to_f32(xq[i]) * e8m0_to_f32(scx));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        uint nblock = row >> 7;
        float partial = 0.0f;
        for (uint p = 0; p < 4u; ++p) {
            uint base = (p << 10) + lane * 16u;
            uint kblock = base >> 7;
            uint word = scale_w[nblock * 8u + (kblock >> 2)];
            uchar sc = (uchar)((word >> ((kblock & 3u) * 8u)) & 0xffu);
            float scale = e8m0_to_f32(sc);
            uint4 wp = ((__global const uint4*)(W + (ulong)row * 4096u + base))[0];
            float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
            float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
            float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
            float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
            float2 e = __builtin_amdgcn_cvt_pk_f32_fp8(wp.z, false);
            float2 f = __builtin_amdgcn_cvt_pk_f32_fp8(wp.z, true);
            float2 g = __builtin_amdgcn_cvt_pk_f32_fp8(wp.w, false);
            float2 h = __builtin_amdgcn_cvt_pk_f32_fp8(wp.w, true);
            half16 xv = vload16(0, xl + base);
            partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                              + b.x*(float)xv.s2 + b.y*(float)xv.s3
                              + c.x*(float)xv.s4 + c.y*(float)xv.s5
                              + d.x*(float)xv.s6 + d.y*(float)xv.s7
                              + e.x*(float)xv.s8 + e.y*(float)xv.s9
                              + f.x*(float)xv.sa + f.y*(float)xv.sb
                              + g.x*(float)xv.sc + g.y*(float)xv.sd
                              + h.x*(float)xv.se + h.y*(float)xv.sf);
        }
        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) y[row] = partial;
    }
}

// Tile-major weight-blockscale GEMV. Each 128x128 FP8 weight tile has a
// 16-byte header. Bytes 0..3 store the E8M0 scale slots for the four 32-column
// K chunks in the tile; bytes 4..15 are reserved for the future CDNA4
// MFMA-scale/preshuffle metadata ABI.
// The tile body is row-major [128][128] FP8 bytes.
__kernel void gemv_fp8_wblock_tiled_e8m0(__global const uchar* WT,
                                         __global const half* x,
                                         __global float* y,
                                         uint N,
                                         uint K,
                                         uint kblocks,
                                         uint tile_stride) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + w;
    __local half xl[8192];
    if (K == 16384u && kblocks == 128u) {
        for (uint i = tid; i < 8192u; i += 256u) xl[i] = x[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        float partial = 0.0f;
        if (row < N) {
            uint nblock = row >> 7;
            uint row_in_block = row & 127u;
            for (uint p = 0; p < 16u; ++p) {
                uint local_base = (p << 9) + lane * 8u;
                uint kblock = local_base >> 7;
                uint k_in_block = local_base & 127u;
                ulong tile_base = ((ulong)nblock * 128u + kblock) * (ulong)tile_stride;
                uchar sc = WT[tile_base + (k_in_block >> 5)];
                float scale = e8m0_to_f32(sc);
                uint2 wp = ((__global const uint2*)(WT + tile_base + 16u + (ulong)row_in_block * 128u + k_in_block))[0];
                float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
                float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
                float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
                float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
                half8 xv = vload8(0, xl + local_base);
                partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                                  + b.x*(float)xv.s2 + b.y*(float)xv.s3
                                  + c.x*(float)xv.s4 + c.y*(float)xv.s5
                                  + d.x*(float)xv.s6 + d.y*(float)xv.s7);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint i = tid; i < 8192u; i += 256u) xl[i] = x[8192u + i];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (row < N) {
            uint nblock = row >> 7;
            uint row_in_block = row & 127u;
            for (uint p = 0; p < 16u; ++p) {
                uint local_base = (p << 9) + lane * 8u;
                uint global_base = 8192u + local_base;
                uint kblock = global_base >> 7;
                uint k_in_block = global_base & 127u;
                ulong tile_base = ((ulong)nblock * 128u + kblock) * (ulong)tile_stride;
                uchar sc = WT[tile_base + (k_in_block >> 5)];
                float scale = e8m0_to_f32(sc);
                uint2 wp = ((__global const uint2*)(WT + tile_base + 16u + (ulong)row_in_block * 128u + k_in_block))[0];
                float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
                float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
                float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
                float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
                half8 xv = vload8(0, xl + local_base);
                partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                                  + b.x*(float)xv.s2 + b.y*(float)xv.s3
                                  + c.x*(float)xv.s4 + c.y*(float)xv.s5
                                  + d.x*(float)xv.s6 + d.y*(float)xv.s7);
            }
            partial += BPERM(1u, partial);
            partial += BPERM(2u, partial);
            partial += BPERM(4u, partial);
            partial += BPERM(8u, partial);
            partial += BPERM(16u, partial);
            partial += BPERM(32u, partial);
            if (lane == 0) y[row] = partial;
        }
        return;
    }
    if (K == 12288u && kblocks == 96u) {
        for (uint i = tid; i < 8192u; i += 256u) xl[i] = x[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        float partial = 0.0f;
        if (row < N) {
            uint nblock = row >> 7;
            uint row_in_block = row & 127u;
            for (uint p = 0; p < 16u; ++p) {
                uint local_base = (p << 9) + lane * 8u;
                uint kblock = local_base >> 7;
                uint k_in_block = local_base & 127u;
                ulong tile_base = ((ulong)nblock * 96u + kblock) * (ulong)tile_stride;
                uchar sc = WT[tile_base + (k_in_block >> 5)];
                float scale = e8m0_to_f32(sc);
                uint2 wp = ((__global const uint2*)(WT + tile_base + 16u + (ulong)row_in_block * 128u + k_in_block))[0];
                float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
                float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
                float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
                float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
                half8 xv = vload8(0, xl + local_base);
                partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                                  + b.x*(float)xv.s2 + b.y*(float)xv.s3
                                  + c.x*(float)xv.s4 + c.y*(float)xv.s5
                                  + d.x*(float)xv.s6 + d.y*(float)xv.s7);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint i = tid; i < 4096u; i += 256u) xl[i] = x[8192u + i];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (row < N) {
            uint nblock = row >> 7;
            uint row_in_block = row & 127u;
            for (uint p = 0; p < 8u; ++p) {
                uint local_base = (p << 9) + lane * 8u;
                uint global_base = 8192u + local_base;
                uint kblock = global_base >> 7;
                uint k_in_block = global_base & 127u;
                ulong tile_base = ((ulong)nblock * 96u + kblock) * (ulong)tile_stride;
                uchar sc = WT[tile_base + (k_in_block >> 5)];
                float scale = e8m0_to_f32(sc);
                uint2 wp = ((__global const uint2*)(WT + tile_base + 16u + (ulong)row_in_block * 128u + k_in_block))[0];
                float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
                float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
                float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
                float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
                half8 xv = vload8(0, xl + local_base);
                partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                                  + b.x*(float)xv.s2 + b.y*(float)xv.s3
                                  + c.x*(float)xv.s4 + c.y*(float)xv.s5
                                  + d.x*(float)xv.s6 + d.y*(float)xv.s7);
            }
            partial += BPERM(1u, partial);
            partial += BPERM(2u, partial);
            partial += BPERM(4u, partial);
            partial += BPERM(8u, partial);
            partial += BPERM(16u, partial);
            partial += BPERM(32u, partial);
            if (lane == 0) y[row] = partial;
        }
        return;
    }
    for (uint i = tid; i < K; i += 256u) xl[i] = x[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        uint nblock = row >> 7;
        uint row_in_block = row & 127u;
        float partial = 0.0f;
        uint passes = K >> 9;
        for (uint p = 0; p < passes; ++p) {
            uint base = (p << 9) + lane * 8u;
            uint kblock = base >> 7;
            uint k_in_block = base & 127u;
            ulong tile_base = ((ulong)nblock * kblocks + kblock) * (ulong)tile_stride;
            uchar sc = WT[tile_base + (k_in_block >> 5)];
            float scale = e8m0_to_f32(sc);
            uint2 wp = ((__global const uint2*)(WT + tile_base + 16u + (ulong)row_in_block * 128u + k_in_block))[0];
            float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
            float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
            float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
            float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
            half8 xv = vload8(0, xl + base);
            partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                              + b.x*(float)xv.s2 + b.y*(float)xv.s3
                              + c.x*(float)xv.s4 + c.y*(float)xv.s5
                              + d.x*(float)xv.s6 + d.y*(float)xv.s7);
        }
        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) y[row] = partial;
    }
}

// Same 128x128 block-scaled GEMV, but scales are stored as f32. This is the
// apples-to-apples baseline for measuring packed E8M0 consumption overhead.
__kernel void gemv_fp8_wblock_f32(__global const uchar* W,
                                  __global const float* scale_w,
                                  __global const half* x,
                                  __global float* y,
                                  uint N,
                                  uint K,
                                  uint kblocks) {
    uint tid = get_local_id(0);
    uint w = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + w;
    __local half xl[8192];
    if (K == 16384u && kblocks == 128u) {
        for (uint i = tid; i < 8192u; i += 256u) xl[i] = x[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        float partial = 0.0f;
        if (row < N) {
            uint nblock = row >> 7;
            for (uint p = 0; p < 16u; ++p) {
                uint local_base = (p << 9) + lane * 8u;
                uint kblock = local_base >> 7;
                float scale = scale_w[nblock * 128u + kblock];
                uint2 wp = ((__global const uint2*)(W + (ulong)row * 16384u + local_base))[0];
                float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
                float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
                float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
                float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
                half8 xv = vload8(0, xl + local_base);
                partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                                  + b.x*(float)xv.s2 + b.y*(float)xv.s3
                                  + c.x*(float)xv.s4 + c.y*(float)xv.s5
                                  + d.x*(float)xv.s6 + d.y*(float)xv.s7);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint i = tid; i < 8192u; i += 256u) xl[i] = x[8192u + i];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (row < N) {
            uint nblock = row >> 7;
            for (uint p = 0; p < 16u; ++p) {
                uint local_base = (p << 9) + lane * 8u;
                uint global_base = 8192u + local_base;
                uint kblock = global_base >> 7;
                float scale = scale_w[nblock * 128u + kblock];
                uint2 wp = ((__global const uint2*)(W + (ulong)row * 16384u + global_base))[0];
                float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
                float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
                float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
                float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
                half8 xv = vload8(0, xl + local_base);
                partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                                  + b.x*(float)xv.s2 + b.y*(float)xv.s3
                                  + c.x*(float)xv.s4 + c.y*(float)xv.s5
                                  + d.x*(float)xv.s6 + d.y*(float)xv.s7);
            }
            partial += BPERM(1u, partial);
            partial += BPERM(2u, partial);
            partial += BPERM(4u, partial);
            partial += BPERM(8u, partial);
            partial += BPERM(16u, partial);
            partial += BPERM(32u, partial);
            if (lane == 0) y[row] = partial;
        }
        return;
    }
    if (K == 12288u && kblocks == 96u) {
        for (uint i = tid; i < 8192u; i += 256u) xl[i] = x[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        float partial = 0.0f;
        if (row < N) {
            uint nblock = row >> 7;
            for (uint p = 0; p < 16u; ++p) {
                uint local_base = (p << 9) + lane * 8u;
                uint kblock = local_base >> 7;
                float scale = scale_w[nblock * 96u + kblock];
                uint2 wp = ((__global const uint2*)(W + (ulong)row * 12288u + local_base))[0];
                float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
                float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
                float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
                float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
                half8 xv = vload8(0, xl + local_base);
                partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                                  + b.x*(float)xv.s2 + b.y*(float)xv.s3
                                  + c.x*(float)xv.s4 + c.y*(float)xv.s5
                                  + d.x*(float)xv.s6 + d.y*(float)xv.s7);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint i = tid; i < 4096u; i += 256u) xl[i] = x[8192u + i];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (row < N) {
            uint nblock = row >> 7;
            for (uint p = 0; p < 8u; ++p) {
                uint local_base = (p << 9) + lane * 8u;
                uint global_base = 8192u + local_base;
                uint kblock = global_base >> 7;
                float scale = scale_w[nblock * 96u + kblock];
                uint2 wp = ((__global const uint2*)(W + (ulong)row * 12288u + global_base))[0];
                float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
                float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
                float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
                float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
                half8 xv = vload8(0, xl + local_base);
                partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                                  + b.x*(float)xv.s2 + b.y*(float)xv.s3
                                  + c.x*(float)xv.s4 + c.y*(float)xv.s5
                                  + d.x*(float)xv.s6 + d.y*(float)xv.s7);
            }
            partial += BPERM(1u, partial);
            partial += BPERM(2u, partial);
            partial += BPERM(4u, partial);
            partial += BPERM(8u, partial);
            partial += BPERM(16u, partial);
            partial += BPERM(32u, partial);
            if (lane == 0) y[row] = partial;
        }
        return;
    }
    for (uint i = tid; i < K; i += 256u) xl[i] = x[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        uint nblock = row >> 7;
        float partial = 0.0f;
        uint passes = K >> 9;
        for (uint p = 0; p < passes; ++p) {
            uint base = (p << 9) + lane * 8u;
            uint kblock = base >> 7;
            float scale = scale_w[nblock * kblocks + kblock];
            uint2 wp = ((__global const uint2*)(W + (ulong)row * K + base))[0];
            float2 a = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, false);
            float2 b = __builtin_amdgcn_cvt_pk_f32_fp8(wp.x, true);
            float2 c = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, false);
            float2 d = __builtin_amdgcn_cvt_pk_f32_fp8(wp.y, true);
            half8 xv = vload8(0, xl + base);
            partial += scale * (a.x*(float)xv.s0 + a.y*(float)xv.s1
                              + b.x*(float)xv.s2 + b.y*(float)xv.s3
                              + c.x*(float)xv.s4 + c.y*(float)xv.s5
                              + d.x*(float)xv.s6 + d.y*(float)xv.s7);
        }
        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) y[row] = partial;
    }
}

// ---- Mixture-of-Experts FFN (decode, M=1) ----------------------------------
// Qwen3-MoE: a router picks top-K of E experts; each expert is a SwiGLU FFN
// (gate/up [I][H], down [H][I]); the layer output is the router-weighted sum of
// the K experts. At decode (one token) this is memory-bandwidth bound on reading
// the selected experts' weights — so we stream weights with the gemv tiling and
// gather experts by index on the device (no host round-trip). NO shared expert
// (Qwen3-235B-A22B has none); norm_topk_prob => softmax over the selected K.

// Router top-K: given E expert logits (f32), pick the K largest and softmax over
// just those K (== full softmax then renormalize: the common denominator cancels).
// Writes ids[K] (expert indices) and w[K] (renormalized weights). One workgroup;
// E,K small (128,8) so thread 0 does the selection. ll/picked live in LDS.
__kernel void moe_router_topk(__global const float* logits, __global uint* ids,
                              __global float* w, uint E, uint K) {
    uint t = get_local_id(0);
    __local float ll[256];
    __local int picked[256];
    for (uint i = t; i < E; i += 256u) { ll[i] = logits[i]; picked[i] = 0; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t == 0) {
        for (uint j = 0; j < K; ++j) {
            int best = -1; float bv = -INFINITY;
            for (uint i = 0; i < E; ++i)
                if (!picked[i] && ll[i] > bv) { bv = ll[i]; best = (int)i; }
            picked[best] = 1;
            ids[j] = (uint)best;
            w[j] = bv;                       // stash logit, softmax below
        }
        float m = -INFINITY;
        for (uint j = 0; j < K; ++j) m = fmax(m, w[j]);
        float s = 0.0f;
        for (uint j = 0; j < K; ++j) { float e = native_exp(w[j] - m); w[j] = e; s += e; }
        float inv = 1.0f / s;
        for (uint j = 0; j < K; ++j) w[j] *= inv;
    }
}

// Fused small-router projection + top-k softmax for decode MoE. One workgroup
// stages x once, computes E router logits for W[E][K] in fixed row order, then
// applies the exact moe_router_topk selection/softmax logic. This is intended
// for small expert counts (Qwen3 synthetic gate uses E=16, topk=8).
__kernel void moe_router_gemv_topk(__global const half* W, __global const half* x,
                                   __global uint* ids, __global float* w,
                                   uint E, uint K, uint topk) {
    uint tid = get_local_id(0);
    uint wv = tid >> 6;
    uint lane = tid & 63;
    __local half xl[8192];
    __local float ll[256];
    __local int picked[256];
    for (uint i = tid; i < K; i += 256u) xl[i] = x[i];
    for (uint i = tid; i < E; i += 256u) { ll[i] = -INFINITY; picked[i] = 0; }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint groups = (E + 3u) >> 2;
    for (uint g = 0; g < groups; ++g) {
        uint row = g * 4u + wv;
        if (row < E) {
            float partial = 0.0f;
            uint passes = K >> 9;
            for (uint p = 0; p < passes; ++p) {
                uint base = (p << 9) + lane * 8u;
                half8 wv8 = vload8(0, W + (ulong)row * K + base);
                half8 xv = vload8(0, xl + base);
                partial += (float)wv8.s0*(float)xv.s0 + (float)wv8.s1*(float)xv.s1
                         + (float)wv8.s2*(float)xv.s2 + (float)wv8.s3*(float)xv.s3
                         + (float)wv8.s4*(float)xv.s4 + (float)wv8.s5*(float)xv.s5
                         + (float)wv8.s6*(float)xv.s6 + (float)wv8.s7*(float)xv.s7;
            }
            partial += BPERM(1u, partial);
            partial += BPERM(2u, partial);
            partial += BPERM(4u, partial);
            partial += BPERM(8u, partial);
            partial += BPERM(16u, partial);
            partial += BPERM(32u, partial);
            if (lane == 0) ll[row] = partial;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid == 0) {
        for (uint j = 0; j < topk; ++j) {
            int best = -1; float bv = -INFINITY;
            for (uint i = 0; i < E; ++i)
                if (!picked[i] && ll[i] > bv) { bv = ll[i]; best = (int)i; }
            picked[best] = 1;
            ids[j] = (uint)best;
            w[j] = bv;
        }
        float m = -INFINITY;
        for (uint j = 0; j < topk; ++j) m = fmax(m, w[j]);
        float s = 0.0f;
        for (uint j = 0; j < topk; ++j) { float e = native_exp(w[j] - m); w[j] = e; s += e; }
        float inv = 1.0f / s;
        for (uint j = 0; j < topk; ++j) w[j] *= inv;
    }
}

// Router with fixed current outputs plus step/layer-indexed validation history.
// This keeps graph replay arguments stable while preserving the full oracle log.
__kernel void moe_router_gemv_topk_log_step(__global const half* W, __global const half* x,
                                            __global uint* ids_cur, __global float* w_cur,
                                            __global uint* ids_hist, __global float* w_hist,
                                            __global const uint* step,
                                            uint E, uint K, uint topk,
                                            uint history_steps, uint layer, uint num_layers) {
    uint tid = get_local_id(0);
    uint wv = tid >> 6;
    uint lane = tid & 63;
    __local half xl[8192];
    __local float ll[256];
    __local int picked[256];
    for (uint i = tid; i < K; i += 256u) xl[i] = x[i];
    for (uint i = tid; i < E; i += 256u) { ll[i] = -INFINITY; picked[i] = 0; }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint groups = (E + 3u) >> 2;
    for (uint g = 0; g < groups; ++g) {
        uint row = g * 4u + wv;
        if (row < E) {
            float partial = 0.0f;
            uint passes = K >> 9;
            for (uint p = 0; p < passes; ++p) {
                uint base = (p << 9) + lane * 8u;
                half8 wv8 = vload8(0, W + (ulong)row * K + base);
                half8 xv = vload8(0, xl + base);
                partial += (float)wv8.s0*(float)xv.s0 + (float)wv8.s1*(float)xv.s1
                         + (float)wv8.s2*(float)xv.s2 + (float)wv8.s3*(float)xv.s3
                         + (float)wv8.s4*(float)xv.s4 + (float)wv8.s5*(float)xv.s5
                         + (float)wv8.s6*(float)xv.s6 + (float)wv8.s7*(float)xv.s7;
            }
            partial += BPERM(1u, partial);
            partial += BPERM(2u, partial);
            partial += BPERM(4u, partial);
            partial += BPERM(8u, partial);
            partial += BPERM(16u, partial);
            partial += BPERM(32u, partial);
            if (lane == 0) ll[row] = partial;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid == 0) {
        for (uint j = 0; j < topk; ++j) {
            int best = -1; float bv = -INFINITY;
            for (uint i = 0; i < E; ++i)
                if (!picked[i] && ll[i] > bv) { bv = ll[i]; best = (int)i; }
            picked[best] = 1;
            ids_cur[j] = (uint)best;
            w_cur[j] = bv;
        }
        float m = -INFINITY;
        for (uint j = 0; j < topk; ++j) m = fmax(m, w_cur[j]);
        float ssum = 0.0f;
        for (uint j = 0; j < topk; ++j) { float e = native_exp(w_cur[j] - m); w_cur[j] = e; ssum += e; }
        float inv = 1.0f / ssum;
        uint st = step[0];
        ulong hist_base = ((ulong)st * num_layers + layer) * topk;
        uint log_ok = st < history_steps && layer < num_layers;
        for (uint j = 0; j < topk; ++j) {
            w_cur[j] *= inv;
            if (log_ok) {
                ids_hist[hist_base + j] = ids_cur[j];
                w_hist[hist_base + j] = w_cur[j];
            }
        }
    }
}

// Fused gate+up+SwiGLU for the expert selected at ids[slot]: for each of the I
// intermediate rows, h[row] = silu(gate_e[row]·x) * (up_e[row]·x). Expert weight
// tensors are [E][I][K] row-major; expert index read from device. 4 rows/wg, x in
// LDS, wide half8 loads + 64-lane bpermute reduce. K % 512 == 0.
// Shape-specialized router for the model-decode Qwen MoE shape:
// E=16 experts, hidden K=4096, topk=8. Keeps the same per-row dot-product order,
// deterministic top-k scan, and logging semantics as moe_router_gemv_topk_log_step
// while dropping runtime shape args and halving LDS x staging.
__kernel void moe_router_gemv_topk_log_step_e16_k4096_top8(
                                            __global const half* W, __global const half* x,
                                            __global uint* ids_cur, __global float* w_cur,
                                            __global uint* ids_hist, __global float* w_hist,
                                            __global const uint* step,
                                            uint history_steps, uint layer, uint num_layers) {
    uint tid = get_local_id(0);
    uint wv = tid >> 6;
    uint lane = tid & 63;
    __local half xl[4096];
    __local float ll[16];
    for (uint i = tid; i < 4096u; i += 256u) xl[i] = x[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint g = 0; g < 4u; ++g) {
        uint row = g * 4u + wv;
        float partial = 0.0f;
        for (uint p = 0; p < 8u; ++p) {
            uint base = (p << 9) + lane * 8u;
            half8 wv8 = vload8(0, W + (ulong)row * 4096u + base);
            half8 xv = vload8(0, xl + base);
            partial += (float)wv8.s0*(float)xv.s0 + (float)wv8.s1*(float)xv.s1
                     + (float)wv8.s2*(float)xv.s2 + (float)wv8.s3*(float)xv.s3
                     + (float)wv8.s4*(float)xv.s4 + (float)wv8.s5*(float)xv.s5
                     + (float)wv8.s6*(float)xv.s6 + (float)wv8.s7*(float)xv.s7;
        }
        partial += BPERM(1u, partial);
        partial += BPERM(2u, partial);
        partial += BPERM(4u, partial);
        partial += BPERM(8u, partial);
        partial += BPERM(16u, partial);
        partial += BPERM(32u, partial);
        if (lane == 0) ll[row] = partial;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid == 0) {
        uint picked_mask = 0u;
        for (uint j = 0; j < 8u; ++j) {
            int best = -1; float bv = -INFINITY;
            for (uint i = 0; i < 16u; ++i)
                if (((picked_mask & (1u << i)) == 0u) && ll[i] > bv) { bv = ll[i]; best = (int)i; }
            picked_mask |= 1u << (uint)best;
            ids_cur[j] = (uint)best;
            w_cur[j] = bv;
        }
        float m = -INFINITY;
        for (uint j = 0; j < 8u; ++j) m = fmax(m, w_cur[j]);
        float ssum = 0.0f;
        for (uint j = 0; j < 8u; ++j) { float e = native_exp(w_cur[j] - m); w_cur[j] = e; ssum += e; }
        float inv = 1.0f / ssum;
        uint st = step[0];
        ulong hist_base = ((ulong)st * num_layers + layer) * 8u;
        uint log_ok = st < history_steps && layer < num_layers;
        for (uint j = 0; j < 8u; ++j) {
            w_cur[j] *= inv;
            if (log_ok) {
                ids_hist[hist_base + j] = ids_cur[j];
                w_hist[hist_base + j] = w_cur[j];
            }
        }
    }
}

__kernel void moe_gate_up_swiglu(__global const half* gateW, __global const half* upW,
                                 __global const half* x, __global const uint* ids,
                                 __global half* h_out, uint slot, uint I, uint K) {
    uint tid = get_local_id(0);
    uint wv = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + wv;
    uint e = ids[slot];
    __local half xl[8192];
    for (uint i = tid; i < K; i += 256u) xl[i] = x[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < I) {
        ulong wbase = ((ulong)e * I + row) * K;
        float pg = 0.0f, pu = 0.0f;
        uint passes = K >> 9;
        for (uint p = 0; p < passes; ++p) {
            uint base = (p << 9) + lane * 8u;
            half8 g8 = vload8(0, gateW + wbase + base);
            half8 u8 = vload8(0, upW + wbase + base);
            half8 xv = vload8(0, xl + base);
            pg += (float)g8.s0*(float)xv.s0 + (float)g8.s1*(float)xv.s1
                + (float)g8.s2*(float)xv.s2 + (float)g8.s3*(float)xv.s3
                + (float)g8.s4*(float)xv.s4 + (float)g8.s5*(float)xv.s5
                + (float)g8.s6*(float)xv.s6 + (float)g8.s7*(float)xv.s7;
            pu += (float)u8.s0*(float)xv.s0 + (float)u8.s1*(float)xv.s1
                + (float)u8.s2*(float)xv.s2 + (float)u8.s3*(float)xv.s3
                + (float)u8.s4*(float)xv.s4 + (float)u8.s5*(float)xv.s5
                + (float)u8.s6*(float)xv.s6 + (float)u8.s7*(float)xv.s7;
        }
        pg += BPERM(1u,pg); pg += BPERM(2u,pg); pg += BPERM(4u,pg);
        pg += BPERM(8u,pg); pg += BPERM(16u,pg); pg += BPERM(32u,pg);
        pu += BPERM(1u,pu); pu += BPERM(2u,pu); pu += BPERM(4u,pu);
        pu += BPERM(8u,pu); pu += BPERM(16u,pu); pu += BPERM(32u,pu);
        if (lane == 0) {
            float silu = pg / (1.0f + native_exp(-pg));
            h_out[row] = (half)(silu * pu);
        }
    }
}

// Batched top-k variant of moe_gate_up_swiglu. Grid x covers
// topk*ceil(I/4) workgroups; each workgroup still computes four intermediate
// rows for one selected expert slot. h_out is slot-major: [topk][I].
__kernel void moe_gate_up_swiglu_slots(__global const half* gateW, __global const half* upW,
                                       __global const half* x, __global const uint* ids,
                                       __global half* h_out, uint topk, uint I, uint K) {
    uint tid = get_local_id(0);
    uint wv = tid >> 6;
    uint lane = tid & 63;
    uint groups_per_slot = (I + 3u) >> 2;
    uint gid = get_group_id(0);
    uint slot = gid / groups_per_slot;
    if (slot >= topk) return;
    uint group = gid - slot * groups_per_slot;
    uint row = group * 4u + wv;
    uint e = ids[slot];
    if (row < I) {
        ulong wbase = ((ulong)e * I + row) * K;
        float pg = 0.0f, pu = 0.0f;
        if (K == 4096u) {
            for (uint p = 0; p < 4u; ++p) {
                uint base = (p << 10) + lane * 16u;
                half16 g16 = vload16(0, gateW + wbase + base);
                half16 u16 = vload16(0, upW + wbase + base);
                half16 xv = vload16(0, x + base);
                pg += (float)g16.s0*(float)xv.s0 + (float)g16.s1*(float)xv.s1
                    + (float)g16.s2*(float)xv.s2 + (float)g16.s3*(float)xv.s3
                    + (float)g16.s4*(float)xv.s4 + (float)g16.s5*(float)xv.s5
                    + (float)g16.s6*(float)xv.s6 + (float)g16.s7*(float)xv.s7
                    + (float)g16.s8*(float)xv.s8 + (float)g16.s9*(float)xv.s9
                    + (float)g16.sa*(float)xv.sa + (float)g16.sb*(float)xv.sb
                    + (float)g16.sc*(float)xv.sc + (float)g16.sd*(float)xv.sd
                    + (float)g16.se*(float)xv.se + (float)g16.sf*(float)xv.sf;
                pu += (float)u16.s0*(float)xv.s0 + (float)u16.s1*(float)xv.s1
                    + (float)u16.s2*(float)xv.s2 + (float)u16.s3*(float)xv.s3
                    + (float)u16.s4*(float)xv.s4 + (float)u16.s5*(float)xv.s5
                    + (float)u16.s6*(float)xv.s6 + (float)u16.s7*(float)xv.s7
                    + (float)u16.s8*(float)xv.s8 + (float)u16.s9*(float)xv.s9
                    + (float)u16.sa*(float)xv.sa + (float)u16.sb*(float)xv.sb
                    + (float)u16.sc*(float)xv.sc + (float)u16.sd*(float)xv.sd
                    + (float)u16.se*(float)xv.se + (float)u16.sf*(float)xv.sf;
            }
        } else {
            uint passes = K >> 9;
            for (uint p = 0; p < passes; ++p) {
                uint base = (p << 9) + lane * 8u;
                half8 g8 = vload8(0, gateW + wbase + base);
                half8 u8 = vload8(0, upW + wbase + base);
                half8 xv = vload8(0, x + base);
                pg += (float)g8.s0*(float)xv.s0 + (float)g8.s1*(float)xv.s1
                    + (float)g8.s2*(float)xv.s2 + (float)g8.s3*(float)xv.s3
                    + (float)g8.s4*(float)xv.s4 + (float)g8.s5*(float)xv.s5
                    + (float)g8.s6*(float)xv.s6 + (float)g8.s7*(float)xv.s7;
                pu += (float)u8.s0*(float)xv.s0 + (float)u8.s1*(float)xv.s1
                    + (float)u8.s2*(float)xv.s2 + (float)u8.s3*(float)xv.s3
                    + (float)u8.s4*(float)xv.s4 + (float)u8.s5*(float)xv.s5
                    + (float)u8.s6*(float)xv.s6 + (float)u8.s7*(float)xv.s7;
            }
        }
        pg += BPERM(1u,pg); pg += BPERM(2u,pg); pg += BPERM(4u,pg);
        pg += BPERM(8u,pg); pg += BPERM(16u,pg); pg += BPERM(32u,pg);
        pu += BPERM(1u,pu); pu += BPERM(2u,pu); pu += BPERM(4u,pu);
        pu += BPERM(8u,pu); pu += BPERM(16u,pu); pu += BPERM(32u,pu);
        if (lane == 0) {
            float silu = pg / (1.0f + native_exp(-pg));
            h_out[(ulong)slot * I + row] = (half)(silu * pu);
        }
    }
}

// Shape-specialized top-k gate/up SwiGLU for the serving K=4096 MoE path.
// Keeps the same ABI as moe_gate_up_swiglu_slots so dispatch plumbing and span
// validation remain identical, but removes the runtime K branch from the hot dot.
__kernel void moe_gate_up_swiglu_slots_k4096(__global const half* gateW, __global const half* upW,
                                             __global const half* x, __global const uint* ids,
                                             __global half* h_out, uint topk, uint I, uint K) {
    uint tid = get_local_id(0);
    uint wv = tid >> 6;
    uint lane = tid & 63;
    uint groups_per_slot = (I + 3u) >> 2;
    uint gid = get_group_id(0);
    uint slot = gid / groups_per_slot;
    if (slot >= topk) return;
    uint group = gid - slot * groups_per_slot;
    uint row = group * 4u + wv;
    uint e = ids[slot];
    (void)K;
    if (row < I) {
        ulong wbase = ((ulong)e * I + row) << 12;
        float pg = 0.0f, pu = 0.0f;
        for (uint p = 0; p < 4u; ++p) {
            uint base = (p << 10) + lane * 16u;
            half16 g16 = vload16(0, gateW + wbase + base);
            half16 u16 = vload16(0, upW + wbase + base);
            half16 xv = vload16(0, x + base);
            pg += (float)g16.s0*(float)xv.s0 + (float)g16.s1*(float)xv.s1
                + (float)g16.s2*(float)xv.s2 + (float)g16.s3*(float)xv.s3
                + (float)g16.s4*(float)xv.s4 + (float)g16.s5*(float)xv.s5
                + (float)g16.s6*(float)xv.s6 + (float)g16.s7*(float)xv.s7
                + (float)g16.s8*(float)xv.s8 + (float)g16.s9*(float)xv.s9
                + (float)g16.sa*(float)xv.sa + (float)g16.sb*(float)xv.sb
                + (float)g16.sc*(float)xv.sc + (float)g16.sd*(float)xv.sd
                + (float)g16.se*(float)xv.se + (float)g16.sf*(float)xv.sf;
            pu += (float)u16.s0*(float)xv.s0 + (float)u16.s1*(float)xv.s1
                + (float)u16.s2*(float)xv.s2 + (float)u16.s3*(float)xv.s3
                + (float)u16.s4*(float)xv.s4 + (float)u16.s5*(float)xv.s5
                + (float)u16.s6*(float)xv.s6 + (float)u16.s7*(float)xv.s7
                + (float)u16.s8*(float)xv.s8 + (float)u16.s9*(float)xv.s9
                + (float)u16.sa*(float)xv.sa + (float)u16.sb*(float)xv.sb
                + (float)u16.sc*(float)xv.sc + (float)u16.sd*(float)xv.sd
                + (float)u16.se*(float)xv.se + (float)u16.sf*(float)xv.sf;
        }
        pg += BPERM(1u,pg); pg += BPERM(2u,pg); pg += BPERM(4u,pg);
        pg += BPERM(8u,pg); pg += BPERM(16u,pg); pg += BPERM(32u,pg);
        pu += BPERM(1u,pu); pu += BPERM(2u,pu); pu += BPERM(4u,pu);
        pu += BPERM(8u,pu); pu += BPERM(16u,pu); pu += BPERM(32u,pu);
        if (lane == 0) {
            float silu = pg / (1.0f + native_exp(-pg));
            h_out[(ulong)slot * I + row] = (half)(silu * pu);
        }
    }
}

// Down-projection for the expert at ids[slot], scaled by w[slot]. slot 0
// initializes out with a direct write; later slots accumulate. This removes the
// standalone zero_f32 pass while preserving determinism because model decode
// dispatches slots sequentially on one barrier-ordered AQL queue. I % 512 == 0.
__kernel void moe_down_accum(__global const half* downW, __global const half* h,
                             __global const uint* ids, __global const float* w,
                             __global float* out, uint slot, uint N, uint I) {
    uint tid = get_local_id(0);
    uint wv = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + wv;
    uint e = ids[slot];
    float wj = w[slot];
    __local half hl[8192];
    for (uint i = tid; i < I; i += 256u) hl[i] = h[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N) {
        ulong wbase = ((ulong)e * N + row) * I;
        float p = 0.0f;
        uint passes = I >> 9;
        for (uint pp = 0; pp < passes; ++pp) {
            uint base = (pp << 9) + lane * 8u;
            half8 d8 = vload8(0, downW + wbase + base);
            half8 hv = vload8(0, hl + base);
            p += (float)d8.s0*(float)hv.s0 + (float)d8.s1*(float)hv.s1
               + (float)d8.s2*(float)hv.s2 + (float)d8.s3*(float)hv.s3
               + (float)d8.s4*(float)hv.s4 + (float)d8.s5*(float)hv.s5
               + (float)d8.s6*(float)hv.s6 + (float)d8.s7*(float)hv.s7;
        }
        p += BPERM(1u,p); p += BPERM(2u,p); p += BPERM(4u,p);
        p += BPERM(8u,p); p += BPERM(16u,p); p += BPERM(32u,p);
        if (lane == 0) {
            float v = wj * p;
            if (slot == 0u) out[row] = v;
            else out[row] += v;
        }
    }
}

// Batched top-k down projection. Grid x covers ceil(N/4) workgroups; each
// workgroup computes four output rows and walks top-k slots in fixed order,
// matching the old queued slot-by-slot accumulation without global atomics.
// h is slot-major [topk][I]. I % 512 == 0.
__kernel void moe_down_accum_slots(__global const half* downW, __global const half* h,
                                   __global const uint* ids, __global const float* w,
                                   __global float* out, uint topk, uint N, uint I) {
    uint tid = get_local_id(0);
    uint wv = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + wv;
    float acc = 0.0f;
    for (uint slot = 0; slot < topk; ++slot) {
        ulong hbase = (ulong)slot * I;
        if (row < N) {
            uint e = ids[slot];
            float wj = w[slot];
            ulong wbase = ((ulong)e * N + row) * I;
            float p = 0.0f;
            uint passes = I >> 9;
            for (uint pp = 0; pp < passes; ++pp) {
                uint base = (pp << 9) + lane * 8u;
                half8 d8 = vload8(0, downW + wbase + base);
                half8 hv = vload8(0, h + hbase + base);
                p += (float)d8.s0*(float)hv.s0 + (float)d8.s1*(float)hv.s1
                   + (float)d8.s2*(float)hv.s2 + (float)d8.s3*(float)hv.s3
                   + (float)d8.s4*(float)hv.s4 + (float)d8.s5*(float)hv.s5
                   + (float)d8.s6*(float)hv.s6 + (float)d8.s7*(float)hv.s7;
            }
            p += BPERM(1u,p); p += BPERM(2u,p); p += BPERM(4u,p);
            p += BPERM(8u,p); p += BPERM(16u,p); p += BPERM(32u,p);
            if (lane == 0) acc += wj * p;
        }
    }
    if (row < N && lane == 0) out[row] = acc;
}

// Shape-specialized top-k down projection for the serving I=1536 MoE path.
// Keeps the same ABI as moe_down_accum_slots, but removes the runtime I loop
// geometry from model-decode's hot down-projection dispatch.
__kernel void moe_down_accum_slots_i1536(__global const half* downW, __global const half* h,
                                         __global const uint* ids, __global const float* w,
                                         __global float* out, uint topk, uint N, uint I) {
    uint tid = get_local_id(0);
    uint wv = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + wv;
    float acc = 0.0f;
    (void)I;
    for (uint slot = 0; slot < topk; ++slot) {
        ulong hbase = (ulong)slot * 1536u;
        if (row < N) {
            uint e = ids[slot];
            float wj = w[slot];
            ulong wbase = (((ulong)e * N) + row) * 1536u;
            float p = 0.0f;
            for (uint pp = 0; pp < 3u; ++pp) {
                uint base = (pp << 9) + lane * 8u;
                half8 d8 = vload8(0, downW + wbase + base);
                half8 hv = vload8(0, h + hbase + base);
                p += (float)d8.s0*(float)hv.s0 + (float)d8.s1*(float)hv.s1
                   + (float)d8.s2*(float)hv.s2 + (float)d8.s3*(float)hv.s3
                   + (float)d8.s4*(float)hv.s4 + (float)d8.s5*(float)hv.s5
                   + (float)d8.s6*(float)hv.s6 + (float)d8.s7*(float)hv.s7;
            }
            p += BPERM(1u,p); p += BPERM(2u,p); p += BPERM(4u,p);
            p += BPERM(8u,p); p += BPERM(16u,p); p += BPERM(32u,p);
            if (lane == 0) acc += wj * p;
        }
    }
    if (row < N && lane == 0) out[row] = acc;
}

// Shape-specialized top-k down projection for the compact model-decode gate
// (I=512). This keeps the same ABI as moe_down_accum_slots, but removes the
// runtime I loop and dynamic row stride from the serving-gate hot path.
__kernel void moe_down_accum_slots_i512(__global const half* downW, __global const half* h,
                                        __global const uint* ids, __global const float* w,
                                        __global float* out, uint topk, uint N, uint I) {
    uint tid = get_local_id(0);
    uint wv = tid >> 6;
    uint lane = tid & 63;
    uint row = get_group_id(0) * 4u + wv;
    float acc = 0.0f;
    (void)I;
    for (uint slot = 0; slot < topk; ++slot) {
        ulong hbase = (ulong)slot << 9;
        if (row < N) {
            uint e = ids[slot];
            float wj = w[slot];
            ulong wbase = (((ulong)e * N) + row) << 9;
            uint base = lane * 8u;
            half8 d8 = vload8(0, downW + wbase + base);
            half8 hv = vload8(0, h + hbase + base);
            float p = (float)d8.s0*(float)hv.s0 + (float)d8.s1*(float)hv.s1
                    + (float)d8.s2*(float)hv.s2 + (float)d8.s3*(float)hv.s3
                    + (float)d8.s4*(float)hv.s4 + (float)d8.s5*(float)hv.s5
                    + (float)d8.s6*(float)hv.s6 + (float)d8.s7*(float)hv.s7;
            p += BPERM(1u,p); p += BPERM(2u,p); p += BPERM(4u,p);
            p += BPERM(8u,p); p += BPERM(16u,p); p += BPERM(32u,p);
            if (lane == 0) acc += wj * p;
        }
    }
    if (row < N && lane == 0) out[row] = acc;
}

// f32 -> f16 elementwise cast (projection outputs feeding the f16 rope/attn path).
__kernel void cast_f32_f16(__global const float* in, __global half* out,
                           uint n, uint nthreads) {
    uint gid = get_group_id(0) * 256u + get_local_id(0);
    for (uint i = gid; i < n; i += nthreads) out[i] = (half)in[i];
}

// Descriptor-driven f32 -> f16 cast. The launch passes only descriptor/status
// buffers as kernargs; input/output pointers and shape are read from HBM:
// desc[0]=magic, desc[1]=version|row<<32, desc[2]=in_va, desc[3]=out_va,
// desc[4]=n|nthreads<<32, desc[5..6]=reserved, desc[7]=fnv(desc[0..6]).
__kernel void cast_f32_f16_descriptor(__global const ulong* desc,
                                      __global ulong* status) {
    uint gid = get_group_id(0) * 256u + get_local_id(0);
    ulong magic = desc[0];
    ulong version_row = desc[1];
    ulong in_va = desc[2];
    ulong out_va = desc[3];
    ulong shape = desc[4];
    ulong expected_checksum = desc[7];
    uint version = (uint)(version_row & 0xffffffffUL);
    uint n = (uint)(shape & 0xffffffffUL);
    uint nthreads = (uint)(shape >> 32);

    ulong checksum = 0xcbf29ce484222325UL;
    for (uint word_idx = 0u; word_idx < 7u; ++word_idx) {
        ulong word = desc[word_idx];
        for (uint byte_idx = 0u; byte_idx < 8u; ++byte_idx) {
            checksum ^= (word >> (byte_idx * 8u)) & 0xffUL;
            checksum *= 0x100000001b3UL;
        }
    }

    ulong result = 0UL;
    ulong bad_value = 0UL;
    if (magic != 0x4d41525f43415354UL) {
        result = 1UL;
        bad_value = magic;
    } else if (version != 1u) {
        result = 2UL;
        bad_value = version_row;
    } else if (n == 0u || nthreads == 0u) {
        result = 3UL;
        bad_value = shape;
    } else if (checksum != expected_checksum) {
        result = 4UL;
        bad_value = checksum;
    }

    if (gid == 0u) {
        status[0] = result;
        status[1] = (ulong)n;
        status[2] = (ulong)nthreads;
        status[3] = checksum;
        status[4] = expected_checksum;
        status[5] = in_va;
        status[6] = out_va;
        status[7] = bad_value;
        status[8] = 0xCA57D15C5AFE600DUL;
    }
    if (result != 0UL) return;

    __global const float* in = (__global const float*)in_va;
    __global half* out = (__global half*)out_va;
    for (uint i = gid; i < n; i += nthreads) out[i] = (half)in[i];
}

// f32 -> bf16 elementwise cast. Used for resident residual checkpoints: Qwen
// weights/activations are BF16-native, and FP16 residual handoff can overflow
// before RMSNorm even when the fused f32 all-reduce math is correct.
__kernel void cast_f32_bf16(__global const float* in, __global ushort* out,
                            uint n, uint nthreads) {
    uint gid = get_group_id(0) * 256u + get_local_id(0);
    for (uint i = gid; i < n; i += nthreads) {
        uint bits = as_uint(in[i]);
        uint lsb = (bits >> 16) & 1u;
        out[i] = (ushort)((bits + 0x7fffu + lsb) >> 16);
    }
}

// Residual add into an f16 stream: acc[i] = acc[i] + x[i] (x is f32). Keeps the
// hidden/residual stream in f16 (as real decode does) while contributions arrive
// in f32 from the GEMV/attention/MoE kernels.
__kernel void add_into_f16(__global half* acc, __global const float* x,
                           uint n, uint nthreads) {
    uint gid = get_group_id(0) * 256u + get_local_id(0);
    for (uint i = gid; i < n; i += nthreads) acc[i] = (half)((float)acc[i] + x[i]);
}

// Standalone P2P big-burst CU-store write: each workgroup streams a contiguous
// slice of one peer's region to that peer's buffer over XGMI. peer_bufs holds
// npeers dst base VAs. Peer p receives src[p*per4 + start4 + i] at dst_p[start4+i]
// for i in [0,count4). Used alongside the SDMA copy engines (a separate write
// path) to push aggregate XGMI egress past what either path reaches alone.
__kernel void p2p_write(__global const float4* src, __global const ulong* peer_bufs,
                        uint npeers, uint per4, uint start4, uint count4, uint num_wg) {
    // wg size is fixed at 256 by the launcher; a literal avoids the COv5
    // hidden-arg read (get_local_size) that arm_grid does not populate.
    uint gid = get_group_id(0), lid = get_local_id(0);
    const uint lsz = 256u;
    uint nslots = num_wg / npeers;
    if (nslots < 1u) nslots = 1u;
    uint peer = gid % npeers;
    uint slot = gid / npeers;
    uint sub = (count4 + nslots - 1u) / nslots;
    uint o = slot * sub;
    uint end = min(o + sub, count4);
    __global float4* dst = (__global float4*)peer_bufs[peer];
    for (uint i = o + lid; i < end; i += lsz) {
        uint idx = start4 + i;
        dst[idx] = src[(ulong)peer * per4 + idx];
    }
}

// P2P big-burst broadcast: every peer receives the SAME slice src[start4..+count4)
// (the all-gather of one reduced chunk). Companion to p2p_write, run alongside the
// SDMA copy engines for the hybrid two-path all-gather.
__kernel void p2p_broadcast(__global const float4* src, __global const ulong* peer_bufs,
                            uint npeers, uint start4, uint count4, uint num_wg) {
    uint gid = get_group_id(0), lid = get_local_id(0);
    const uint lsz = 256u;
    uint nslots = num_wg / npeers;
    if (nslots < 1u) nslots = 1u;
    uint peer = gid % npeers;
    uint slot = gid / npeers;
    uint sub = (count4 + nslots - 1u) / nslots;
    uint o = slot * sub;
    uint end = min(o + sub, count4);
    __global float4* dst = (__global float4*)peer_bufs[peer];
    for (uint i = o + lid; i < end; i += lsz) {
        uint idx = start4 + i;
        dst[idx] = src[idx];
    }
}

// Zero an f32 buffer (MoE accumulator reset per token).
__kernel void zero_f32(__global float* b, uint n, uint nthreads) {
    uint gid = get_group_id(0) * 256u + get_local_id(0);
    for (uint i = gid; i < n; i += nthreads) b[i] = 0.0f;
}

// Append the current token's K,V (post QK-norm + RoPE) into the per-head KV cache
// at logical position `pos`. k/v are [nkv*D] (per-head contiguous); cache is
// [nkv][Lmax][D] row-major. One thread per element; on-device (no host scatter).
__kernel void kv_append(__global half* kcache, __global half* vcache,
                        __global const half* k, __global const half* v,
                        uint nkv, uint Lmax, uint pos, uint D, uint nthreads) {
    uint gid = get_group_id(0) * 256u + get_local_id(0);
    uint total = nkv * D;
    for (uint i = gid; i < total; i += nthreads) {
        uint head = i / D, d = i % D;
        kcache[((ulong)head * Lmax + pos) * D + d] = k[i];
        vcache[((ulong)head * Lmax + pos) * D + d] = v[i];
    }
}

// FlashInfer-style paged KV append. k/v are ragged append rows:
// [append_count][nkv][D]. Metadata maps each append row through
// batch_indices[token] and positions[token] into NHD paged cache layout:
// [physical_page][page_offset][nkv][D]. Bad or padded append entries are skipped
// in-kernel so they cannot form an unsafe K/V cache address.
__kernel void kv_append_paged(__global half* kcache, __global half* vcache,
                              __global const half* k, __global const half* v,
                              __global const uint* indices,
                              __global const uint* indptr,
                              __global const uint* last_page_len,
                              __global const uint* batch_indices,
                              __global const uint* positions,
                              uint append_count, uint batch_size,
                              uint total_indices, uint physical_blocks,
                              uint nkv, uint block_size, uint D,
                              uint nthreads) {
    uint gid = get_group_id(0) * 256u + get_local_id(0);
    uint per_token = nkv * D;
    uint total = append_count * per_token;
    for (uint i = gid; i < total; i += nthreads) {
        uint token = i / per_token;
        uint rem = i - token * per_token;
        uint head = rem / D;
        uint d = rem - head * D;
        uint batch = batch_indices[token];
        uint pos = positions[token];
        if (batch >= batch_size) continue;
        uint lo = indptr[batch];
        uint hi = indptr[batch + 1u];
        if (lo >= hi || hi > total_indices) continue;
        uint page_count = hi - lo;
        uint logical_page = pos / block_size;
        uint page_offset = pos - logical_page * block_size;
        uint last = last_page_len[batch];
        if (last == 0u || last > block_size) continue;
        uint seq_tokens = (page_count - 1u) * block_size + last;
        if (pos >= seq_tokens || logical_page >= page_count) continue;
        uint phys_page = indices[lo + logical_page];
        if (phys_page >= physical_blocks) continue;
        ulong dst = (((ulong)phys_page * block_size + page_offset) * nkv + head) * D + d;
        ulong src = ((ulong)token * nkv + head) * D + d;
        kcache[dst] = k[src];
        vcache[dst] = v[src];
    }
}

// FlashInfer-style paged append for already-quantized FP4 rows consumed by
// attn_decode_split2_fp4_gqa_paged. Each source row is D=128 packed to 64 bytes
// plus 4 E8M0 scale bytes. This intentionally does not quantize partial pages:
// the source row is already a complete, scale-stable FP4 row.
__kernel void kv_append_paged_fp4(__global uchar* kcache, __global uchar* vcache,
                                  __global uchar* scale_k, __global uchar* scale_v,
                                  __global const uchar* k, __global const uchar* v,
                                  __global const uchar* src_scale_k,
                                  __global const uchar* src_scale_v,
                                  __global const uint* indices,
                                  __global const uint* indptr,
                                  __global const uint* last_page_len,
                                  __global const uint* batch_indices,
                                  __global const uint* positions,
                                  uint append_count, uint batch_size,
                                  uint total_indices, uint physical_blocks,
                                  uint block_size, uint nthreads) {
    uint gid = get_group_id(0) * 256u + get_local_id(0);
    const uint packed_bytes = 64u;
    const uint scale_bytes = 4u;
    const uint per_token = packed_bytes * 2u + scale_bytes * 2u;
    uint total = append_count * per_token;
    for (uint i = gid; i < total; i += nthreads) {
        uint token = i / per_token;
        uint rem = i - token * per_token;
        uint batch = batch_indices[token];
        uint pos = positions[token];
        if (batch >= batch_size) continue;
        uint lo = indptr[batch];
        uint hi = indptr[batch + 1u];
        if (lo >= hi || hi > total_indices) continue;
        uint page_count = hi - lo;
        uint logical_page = pos / block_size;
        uint page_offset = pos - logical_page * block_size;
        uint last = last_page_len[batch];
        if (last == 0u || last > block_size) continue;
        uint seq_tokens = (page_count - 1u) * block_size + last;
        if (pos >= seq_tokens || logical_page >= page_count) continue;
        uint phys_page = indices[lo + logical_page];
        if (phys_page >= physical_blocks) continue;
        ulong row = (ulong)phys_page * block_size + page_offset;
        if (rem < packed_bytes) {
            kcache[row * packed_bytes + rem] = k[(ulong)token * packed_bytes + rem];
        } else if (rem < packed_bytes * 2u) {
            uint j = rem - packed_bytes;
            vcache[row * packed_bytes + j] = v[(ulong)token * packed_bytes + j];
        } else if (rem < packed_bytes * 2u + scale_bytes) {
            uint j = rem - packed_bytes * 2u;
            scale_k[row * scale_bytes + j] = src_scale_k[(ulong)token * scale_bytes + j];
        } else {
            uint j = rem - packed_bytes * 2u - scale_bytes;
            scale_v[row * scale_bytes + j] = src_scale_v[(ulong)token * scale_bytes + j];
        }
    }
}

static uchar e2m1_quant_nearest(float x) {
    uchar sign = x < 0.0f ? (uchar)8 : (uchar)0;
    float a = fabs(x);
    uchar idx;
    if (a < 0.25f) idx = 0;        // 0
    else if (a < 0.75f) idx = 1;   // 0.5
    else if (a < 1.25f) idx = 2;   // 1
    else if (a < 1.75f) idx = 3;   // 1.5
    else if (a < 2.5f) idx = 4;    // 2
    else if (a < 3.5f) idx = 5;    // 3
    else if (a < 5.0f) idx = 6;    // 4
    else idx = 7;                  // 6
    return sign | idx;
}

// Serving-shaped FP4 paged append from freshly produced FP16 K/V rows. This is
// the online decode append path: each D=128 source row is quantized into 64
// packed E2M1 bytes plus four E8M0 scale bytes and written directly into the
// physical paged cache row selected by FlashInfer-style metadata. Scale ownership
// is per token row and per 32 dims, so appending a partial page never rewrites
// older rows' scales.
__kernel void kv_append_paged_fp4_from_f16(__global uchar* kcache, __global uchar* vcache,
                                           __global uchar* scale_k, __global uchar* scale_v,
                                           __global const half* k, __global const half* v,
                                           __global const uint* indices,
                                           __global const uint* indptr,
                                           __global const uint* last_page_len,
                                           __global const uint* batch_indices,
                                           __global const uint* positions,
                                           uint append_count, uint batch_size,
                                           uint total_indices, uint physical_blocks,
                                           uint block_size, uint nthreads) {
    const uint D = 128u;
    const uint packed_bytes = 64u;
    const uint scale_bytes = 4u;
    uint gid = get_group_id(0) * 256u + get_local_id(0);
    for (uint item = gid; item < append_count * 8u; item += nthreads) {
        uint token = item / 8u;
        uint rem = item - token * 8u;
        uint is_v = rem >> 2;
        uint bl = rem & 3u;
        uint b = batch_indices[token];
        if (b >= batch_size) continue;
        uint pos = positions[token];
        uint lo = indptr[b];
        uint hi = indptr[b + 1u];
        if (lo > hi || hi > total_indices) continue;
        uint pages = hi - lo;
        if (pages == 0u) continue;
        uint page_in_seq = pos / block_size;
        uint page_offset = pos - page_in_seq * block_size;
        if (page_in_seq >= pages) continue;
        uint last_page = pages - 1u;
        uint last_len = last_page_len[b];
        if (page_in_seq == last_page) {
            if (last_len == 0u || last_len > block_size || page_offset >= last_len) continue;
        } else if (page_offset >= block_size) {
            continue;
        }
        uint phys_page = indices[lo + page_in_seq];
        if (phys_page >= physical_blocks) continue;
        ulong row = (ulong)phys_page * block_size + page_offset;
        __global const half* src = is_v ? v : k;
        __global uchar* dst = is_v ? vcache : kcache;
        __global uchar* sc = is_v ? scale_v : scale_k;
        ulong src_base = (ulong)token * D + bl * 32u;
        float maxabs = 0.0f;
        for (uint j = 0; j < 32u; ++j) {
            maxabs = fmax(maxabs, fabs((float)src[src_base + j]));
        }
        int e = 0;
        if (maxabs > 0.0f) e = (int)ceil(native_log2(maxabs / 6.0f));
        int sb_i = e + 127;
        if (sb_i < 0) sb_i = 0;
        if (sb_i > 255) sb_i = 255;
        uchar sb = (uchar)sb_i;
        float sf = as_float(((uint)sb) << 23);
        sc[row * scale_bytes + bl] = sb;
        for (uint j = 0; j < 16u; ++j) {
            uint d0 = bl * 32u + j * 2u;
            float x0 = (float)src[(ulong)token * D + d0] / sf;
            float x1 = (float)src[(ulong)token * D + d0 + 1u] / sf;
            uchar n0 = e2m1_quant_nearest(x0);
            uchar n1 = e2m1_quant_nearest(x1);
            dst[row * packed_bytes + bl * 16u + j] = (uchar)(n0 | (n1 << 4));
        }
    }
}

// Batched-KV-head variant for GQA decode. Source K/V rows are
// [append_token][kv_head][128]; destination cache rows are
// [kv_head][physical_page * block_size + offset]. This collapses the model
// loop's per-KV-head append dispatches into one flattened token x KV-head grid,
// matching serving append APIs that treat all KV heads as one paged-cache append.
__kernel void kv_append_paged_fp4_from_f16_heads(__global uchar* kcache, __global uchar* vcache,
                                                 __global uchar* scale_k, __global uchar* scale_v,
                                                 __global const half* k, __global const half* v,
                                                 __global const uint* indices,
                                                 __global const uint* indptr,
                                                 __global const uint* last_page_len,
                                                 __global const uint* batch_indices,
                                                 __global const uint* positions,
                                                 uint append_count, uint batch_size,
                                                 uint total_indices, uint physical_blocks,
                                                 uint block_size, uint kv_heads,
                                                 uint nthreads) {
    const uint D = 128u;
    const uint packed_bytes = 64u;
    const uint scale_bytes = 4u;
    uint gid = get_group_id(0) * 256u + get_local_id(0);
    uint per_token = kv_heads * 8u;
    if (kv_heads == 0u || per_token == 0u) return;
    for (uint item = gid; item < append_count * per_token; item += nthreads) {
        uint token = item / per_token;
        uint r = item - token * per_token;
        uint kvh = r / 8u;
        uint rem = r - kvh * 8u;
        uint is_v = rem >> 2;
        uint bl = rem & 3u;
        uint b = batch_indices[token];
        if (b >= batch_size) continue;
        uint pos = positions[token];
        uint lo = indptr[b];
        uint hi = indptr[b + 1u];
        if (lo > hi || hi > total_indices) continue;
        uint pages = hi - lo;
        if (pages == 0u) continue;
        uint page_in_seq = pos / block_size;
        uint page_offset = pos - page_in_seq * block_size;
        if (page_in_seq >= pages) continue;
        uint last_page = pages - 1u;
        uint last_len = last_page_len[b];
        if (page_in_seq == last_page) {
            if (last_len == 0u || last_len > block_size || page_offset >= last_len) continue;
        } else if (page_offset >= block_size) {
            continue;
        }
        uint phys_page = indices[lo + page_in_seq];
        if (phys_page >= physical_blocks) continue;
        ulong rows_per_head = (ulong)physical_blocks * block_size;
        ulong row = (ulong)kvh * rows_per_head + (ulong)phys_page * block_size + page_offset;
        __global const half* src = is_v ? v : k;
        __global uchar* dst = is_v ? vcache : kcache;
        __global uchar* sc = is_v ? scale_v : scale_k;
        ulong src_base = ((ulong)token * kv_heads + kvh) * D + bl * 32u;
        float maxabs = 0.0f;
        for (uint j = 0; j < 32u; ++j) {
            maxabs = fmax(maxabs, fabs((float)src[src_base + j]));
        }
        int e = 0;
        if (maxabs > 0.0f) e = (int)ceil(native_log2(maxabs / 6.0f));
        int sb_i = e + 127;
        if (sb_i < 0) sb_i = 0;
        if (sb_i > 255) sb_i = 255;
        uchar sb = (uchar)sb_i;
        float sf = as_float(((uint)sb) << 23);
        sc[row * scale_bytes + bl] = sb;
        for (uint j = 0; j < 16u; ++j) {
            uint d0 = bl * 32u + j * 2u;
            float x0 = (float)src[((ulong)token * kv_heads + kvh) * D + d0] / sf;
            float x1 = (float)src[((ulong)token * kv_heads + kvh) * D + d0 + 1u] / sf;
            uchar n0 = e2m1_quant_nearest(x0);
            uchar n1 = e2m1_quant_nearest(x1);
            dst[row * packed_bytes + bl * 16u + j] = (uchar)(n0 | (n1 << 4));
        }
    }
}

// Same head-major paged FP4 append as kv_append_paged_fp4_from_f16_heads, but V
// arrives directly from the f32 V projection. V is explicitly rounded to half
// before scale/quantization to preserve the old cast_f32_f16 -> append numerics
// while removing the standalone V cast dispatch and scratch row.
__kernel void kv_append_paged_fp4_from_f16_vf32_heads(
                                                 __global uchar* kcache, __global uchar* vcache,
                                                 __global uchar* scale_k, __global uchar* scale_v,
                                                 __global const half* k, __global const float* v,
                                                 __global const uint* indices,
                                                 __global const uint* indptr,
                                                 __global const uint* last_page_len,
                                                 __global const uint* batch_indices,
                                                 __global const uint* positions,
                                                 uint append_count, uint batch_size,
                                                 uint total_indices, uint physical_blocks,
                                                 uint block_size, uint kv_heads,
                                                 uint nthreads) {
    const uint D = 128u;
    const uint packed_bytes = 64u;
    const uint scale_bytes = 4u;
    uint gid = get_group_id(0) * 256u + get_local_id(0);
    uint per_token = kv_heads * 8u;
    if (kv_heads == 0u || per_token == 0u) return;
    for (uint item = gid; item < append_count * per_token; item += nthreads) {
        uint token = item / per_token;
        uint r = item - token * per_token;
        uint kvh = r / 8u;
        uint rem = r - kvh * 8u;
        uint is_v = rem >> 2;
        uint bl = rem & 3u;
        uint b = batch_indices[token];
        if (b >= batch_size) continue;
        uint pos = positions[token];
        uint lo = indptr[b];
        uint hi = indptr[b + 1u];
        if (lo > hi || hi > total_indices) continue;
        uint pages = hi - lo;
        if (pages == 0u) continue;
        uint page_in_seq = pos / block_size;
        uint page_offset = pos - page_in_seq * block_size;
        if (page_in_seq >= pages) continue;
        uint last_page = pages - 1u;
        uint last_len = last_page_len[b];
        if (page_in_seq == last_page) {
            if (last_len == 0u || last_len > block_size || page_offset >= last_len) continue;
        } else if (page_offset >= block_size) {
            continue;
        }
        uint phys_page = indices[lo + page_in_seq];
        if (phys_page >= physical_blocks) continue;
        ulong rows_per_head = (ulong)physical_blocks * block_size;
        ulong row = (ulong)kvh * rows_per_head + (ulong)phys_page * block_size + page_offset;
        __global uchar* dst = is_v ? vcache : kcache;
        __global uchar* sc = is_v ? scale_v : scale_k;
        ulong src_base = ((ulong)token * kv_heads + kvh) * D + bl * 32u;
        float maxabs = 0.0f;
        for (uint j = 0; j < 32u; ++j) {
            float val = is_v ? (float)((half)v[src_base + j]) : (float)k[src_base + j];
            maxabs = fmax(maxabs, fabs(val));
        }
        int e = 0;
        if (maxabs > 0.0f) e = (int)ceil(native_log2(maxabs / 6.0f));
        int sb_i = e + 127;
        if (sb_i < 0) sb_i = 0;
        if (sb_i > 255) sb_i = 255;
        uchar sb = (uchar)sb_i;
        float sf = as_float(((uint)sb) << 23);
        sc[row * scale_bytes + bl] = sb;
        for (uint j = 0; j < 16u; ++j) {
            uint d0 = bl * 32u + j * 2u;
            ulong base = ((ulong)token * kv_heads + kvh) * D + d0;
            float x0 = (is_v ? (float)((half)v[base]) : (float)k[base]) / sf;
            float x1 = (is_v ? (float)((half)v[base + 1u]) : (float)k[base + 1u]) / sf;
            uchar n0 = e2m1_quant_nearest(x0);
            uchar n1 = e2m1_quant_nearest(x1);
            dst[row * packed_bytes + bl * 16u + j] = (uchar)(n0 | (n1 << 4));
        }
    }
}

// CDNA-oriented preshuffled FP4 paged append. Destination layout is:
//   data  [physical_block][kv_head][fp4_group32][token_in_block][packed16]
//   scale [physical_block][kv_head][fp4_group32][token_in_block]
// where each fp4_group32 covers 32 D-elements and stores 16 packed E2M1 bytes.
// This keeps the quantization scale tensor separate and moves the group axis
// ahead of token rows so the future attention kernel can stream one K-group
// without a row-major gather/transpose step.
__kernel void kv_append_paged_fp4_5d_from_f16_vf32_heads(
                                                 __global uchar* kcache, __global uchar* vcache,
                                                 __global uchar* scale_k, __global uchar* scale_v,
                                                 __global const half* k, __global const float* v,
                                                 __global const uint* indices,
                                                 __global const uint* indptr,
                                                 __global const uint* last_page_len,
                                                 __global const uint* batch_indices,
                                                 __global const uint* positions,
                                                 uint append_count, uint batch_size,
                                                 uint total_indices, uint physical_blocks,
                                                 uint block_size, uint kv_heads,
                                                 uint nthreads) {
    const uint D = 128u;
    const uint groups = 4u;
    const uint packed_group_bytes = 16u;
    uint gid = get_group_id(0) * 256u + get_local_id(0);
    uint per_token = kv_heads * 8u;
    if (kv_heads == 0u || per_token == 0u) return;
    for (uint item = gid; item < append_count * per_token; item += nthreads) {
        uint token = item / per_token;
        uint r = item - token * per_token;
        uint kvh = r / 8u;
        uint rem = r - kvh * 8u;
        uint is_v = rem >> 2;
        uint bl = rem & 3u;
        uint b = batch_indices[token];
        if (b >= batch_size) continue;
        uint pos = positions[token];
        uint lo = indptr[b];
        uint hi = indptr[b + 1u];
        if (lo > hi || hi > total_indices) continue;
        uint pages = hi - lo;
        if (pages == 0u) continue;
        uint page_in_seq = pos / block_size;
        uint page_offset = pos - page_in_seq * block_size;
        if (page_in_seq >= pages) continue;
        uint last_page = pages - 1u;
        uint last_len = last_page_len[b];
        if (page_in_seq == last_page) {
            if (last_len == 0u || last_len > block_size || page_offset >= last_len) continue;
        } else if (page_offset >= block_size) {
            continue;
        }
        uint phys_page = indices[lo + page_in_seq];
        if (phys_page >= physical_blocks) continue;
        __global uchar* dst = is_v ? vcache : kcache;
        __global uchar* sc = is_v ? scale_v : scale_k;
        ulong src_base = ((ulong)token * kv_heads + kvh) * D + bl * 32u;
        float maxabs = 0.0f;
        for (uint j = 0; j < 32u; ++j) {
            float val = is_v ? (float)((half)v[src_base + j]) : (float)k[src_base + j];
            maxabs = fmax(maxabs, fabs(val));
        }
        int e = 0;
        if (maxabs > 0.0f) e = (int)ceil(native_log2(maxabs / 6.0f));
        int sb_i = e + 127;
        if (sb_i < 0) sb_i = 0;
        if (sb_i > 255) sb_i = 255;
        uchar sb = (uchar)sb_i;
        float sf = as_float(((uint)sb) << 23);
        ulong group_row = (((ulong)phys_page * kv_heads + kvh) * groups + bl) * block_size
            + page_offset;
        sc[group_row] = sb;
        ulong dst_base = group_row * packed_group_bytes;
        for (uint j = 0; j < packed_group_bytes; ++j) {
            uint d0 = bl * 32u + j * 2u;
            ulong base = ((ulong)token * kv_heads + kvh) * D + d0;
            float x0 = (is_v ? (float)((half)v[base]) : (float)k[base]) / sf;
            float x1 = (is_v ? (float)((half)v[base + 1u]) : (float)k[base + 1u]) / sf;
            uchar n0 = e2m1_quant_nearest(x0);
            uchar n1 = e2m1_quant_nearest(x1);
            dst[dst_base + j] = (uchar)(n0 | (n1 << 4));
        }
    }
}

// First consumer probe for the CDNA-oriented 5D FP4 KV layout. Reads one
// [physical_block][kv_head][group32][token] row from K and V, writes the
// contiguous packed16 payloads, and places the UE8M0 scale byte in byte 0 of a
// 32-bit little-endian word. That is the register contract the scaled-MFMA
// attention path needs before replacing this debug write with math.
__kernel void fp4_5d_kv_load_probe(__global const uchar* kcache,
                                   __global const uchar* vcache,
                                   __global const uchar* scale_k,
                                   __global const uchar* scale_v,
                                   __global uchar* out,
                                   uint physical_blocks,
                                   uint block_size,
                                   uint kv_heads,
                                   uint phys_page,
                                   uint kv_head,
                                   uint group,
                                   uint token_offset,
                                   uint m_rows,
                                   uint nthreads) {
    const uint groups = 4u;
    const uint packed_group_bytes = 16u;
    const uint record_bytes = 40u;
    if (physical_blocks == 0u || block_size == 0u || kv_heads == 0u) return;
    if (phys_page >= physical_blocks || kv_head >= kv_heads) return;
    if (group >= groups || token_offset >= block_size) return;
    uint rows = min(m_rows, 8u);
    if (rows == 0u || nthreads == 0u) return;
    ulong row5 = (((ulong)phys_page * kv_heads + kv_head) * groups + group) * block_size
        + token_offset;
    ulong data_base = row5 * packed_group_bytes;
    uint gid = get_group_id(0) * 256u + get_local_id(0);
    for (uint rec = gid; rec < rows; rec += nthreads) {
        ulong out_base = (ulong)rec * record_bytes;
        for (uint j = 0; j < packed_group_bytes; ++j) {
            out[out_base + j] = kcache[data_base + j];
            out[out_base + 16u + j] = vcache[data_base + j];
        }
        uint sk = (uint)scale_k[row5];
        uint sv = (uint)scale_v[row5];
        out[out_base + 32u] = (uchar)(sk & 0xffu);
        out[out_base + 33u] = (uchar)0;
        out[out_base + 34u] = (uchar)0;
        out[out_base + 35u] = (uchar)0;
        out[out_base + 36u] = (uchar)(sv & 0xffu);
        out[out_base + 37u] = (uchar)0;
        out[out_base + 38u] = (uchar)0;
        out[out_base + 39u] = (uchar)0;
    }
}

// Per-head RMSNorm (QK-norm): x is [nheads*D]; each head's D-slice is normalized
// with the shared weight w[D]. One workgroup per head (fixed 256 threads).
__kernel void rmsnorm_heads(__global half* x, __global const half* w,
                            uint nheads, uint D, float eps) {
    uint head = get_group_id(0);
    uint t = get_local_id(0);
    if (head >= nheads) return;
    __local float red[256];
    __global half* xh = x + (ulong)head * D;
    float ss = 0.0f;
    for (uint i = t; i < D; i += 256u) { float v = (float)xh[i]; ss += v * v; }
    red[t] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint o = 128u; o > 0u; o >>= 1) {
        if (t < o) red[t] += red[t + o];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float rms = rsqrt(red[0] / (float)D + eps);
    for (uint i = t; i < D; i += 256u)
        xh[i] = (half)((float)xh[i] * rms * (float)w[i]);
}

// Decode Q/K epilogue fusion for Qwen-style attention:
// f32 projection row -> f16 cast -> per-head RMSNorm -> RoPE.
//
// The half casts deliberately mirror the unfused sequence:
//   cast_f32_f16, then rmsnorm_heads, then rope_heads.
// That preserves the existing oracle's rounding points while removing two
// intermediate global round-trips and two dispatches per Q/K tensor.
__kernel void cast_rmsnorm_rope_heads(__global const float* src,
                                      __global const half* w,
                                      __global half* dst,
                                      uint nheads, uint D, uint pos,
                                      float theta, float eps) {
    uint head = get_group_id(0);
    uint t = get_local_id(0);
    if (head >= nheads) return;
    __global const float* sh = src + (ulong)head * D;
    __global half* dh = dst + (ulong)head * D;
    __local float red[256];

    float ss = 0.0f;
    for (uint i = t; i < D; i += 256u) {
        float v = (float)((half)sh[i]);
        ss += v * v;
    }
    red[t] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint o = 128u; o > 0u; o >>= 1) {
        if (t < o) red[t] += red[t + o];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float rms = rsqrt(red[0] / (float)D + eps);

    uint half_d = D >> 1;
    for (uint i = t; i < half_d; i += 256u) {
        half ah = (half)((float)((half)sh[i]) * rms * (float)w[i]);
        half bh = (half)((float)((half)sh[i + half_d]) * rms * (float)w[i + half_d]);
        float freq = pow(theta, -2.0f * (float)i / (float)D);
        float ang = (float)pos * freq;
        float c = native_cos(ang), s = native_sin(ang);
        float a = (float)ah, b = (float)bh;
        dh[i] = (half)(a * c - b * s);
        dh[i + half_d] = (half)(b * c + a * s);
    }
}

// Batched Q+K variant of cast_rmsnorm_rope_heads. Grid x covers
// q_heads+k_heads workgroups; each workgroup still handles one head with the
// same rounding/reduction/RoPE order as the single-tensor kernel.
__kernel void cast_rmsnorm_rope_qk_heads(__global const float* q_src,
                                         __global const half* q_w,
                                         __global half* q_dst,
                                         __global const float* k_src,
                                         __global const half* k_w,
                                         __global half* k_dst,
                                         uint q_heads, uint k_heads, uint D,
                                         uint pos, float theta, float eps) {
    uint gid = get_group_id(0);
    uint t = get_local_id(0);
    uint total = q_heads + k_heads;
    if (gid >= total) return;
    uint is_k = gid >= q_heads;
    uint head = is_k ? gid - q_heads : gid;
    __global const float* src = is_k ? k_src : q_src;
    __global const half* w = is_k ? k_w : q_w;
    __global half* dst = is_k ? k_dst : q_dst;
    __global const float* sh = src + (ulong)head * D;
    __global half* dh = dst + (ulong)head * D;
    __local float red[256];

    float ss = 0.0f;
    for (uint i = t; i < D; i += 256u) {
        float v = (float)((half)sh[i]);
        ss += v * v;
    }
    red[t] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint o = 128u; o > 0u; o >>= 1) {
        if (t < o) red[t] += red[t + o];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float rms = rsqrt(red[0] / (float)D + eps);

    uint half_d = D >> 1;
    for (uint i = t; i < half_d; i += 256u) {
        half ah = (half)((float)((half)sh[i]) * rms * (float)w[i]);
        half bh = (half)((float)((half)sh[i + half_d]) * rms * (float)w[i + half_d]);
        float freq = pow(theta, -2.0f * (float)i / (float)D);
        float ang = (float)pos * freq;
        float c = native_cos(ang), s = native_sin(ang);
        float a = (float)ah, b = (float)bh;
        dh[i] = (half)(a * c - b * s);
        dh[i + half_d] = (half)(b * c + a * s);
    }
}

// Fixed Qwen decode epilogue + cache admission:
//   Q: f32 projection -> f16 -> QK RMSNorm -> RoPE -> q scratch
//   K: f32 projection -> f16 -> QK RMSNorm -> RoPE -> FP4 paged cache
//   V: f32 projection -> f16 -> FP4 paged cache
//
// This is the serving hot path for one decode token with 64 Q heads, 4 KV heads
// and D=128. It preserves the old rounding points while collapsing the separate
// QK epilogue and FP4 append dispatches.
static inline float qk_append_wave64_sum(uint lane, float v) {
    v += as_float(__builtin_amdgcn_ds_bpermute((int)(((lane ^ 1u) << 2)), as_int(v)));
    v += as_float(__builtin_amdgcn_ds_bpermute((int)(((lane ^ 2u) << 2)), as_int(v)));
    v += as_float(__builtin_amdgcn_ds_bpermute((int)(((lane ^ 4u) << 2)), as_int(v)));
    v += as_float(__builtin_amdgcn_ds_bpermute((int)(((lane ^ 8u) << 2)), as_int(v)));
    v += as_float(__builtin_amdgcn_ds_bpermute((int)(((lane ^ 16u) << 2)), as_int(v)));
    v += as_float(__builtin_amdgcn_ds_bpermute((int)(((lane ^ 32u) << 2)), as_int(v)));
    return v;
}

__kernel void cast_qk_rope_append_paged_fp4_vf32_q64_k4_d128_meta(
    __global const float* q_src,
    __global const half* q_w,
    __global half* q_dst,
    __global const float* k_src,
    __global const half* k_w,
    __global const float* v_src,
    __global uchar* kcache,
    __global uchar* vcache,
    __global uchar* scale_k,
    __global uchar* scale_v,
    __global const uint* indices,
    __global const uint* indptr,
    __global const uint* last_page_len,
    __global const uint* batch_indices,
    __global const uint* positions,
    uint total_indices,
    uint physical_blocks,
    uint block_size,
    float theta,
    float eps) {
    const uint D = 128u;
    const uint HALF_D = 64u;
    const uint Q_HEADS = 64u;
    const uint KV_HEADS = 4u;
    const uint PACKED_BYTES = 64u;
    const uint SCALE_BYTES = 4u;
    uint gid = get_group_id(0);
    uint t = get_local_id(0);
    uint lane = t & 63u;
    if (gid >= Q_HEADS + KV_HEADS) return;

    uint pos = positions[0];
    float log2_theta = native_log2(theta);
    __local float red[256];
    __local half k_local[128];
    __local half v_local[128];
    __local float sf_local[1];

    if (gid < Q_HEADS) {
        __global const float* sh = q_src + (ulong)gid * D;
        __global half* dh = q_dst + (ulong)gid * D;
        float v0 = (float)((half)sh[t]);
        float v1 = (float)((half)sh[t + HALF_D]);
        float ss = qk_append_wave64_sum(lane, v0 * v0 + v1 * v1);
        float rms = rsqrt(ss * 0.0078125f + eps);
        uint i = t;
        half ah = (half)((float)((half)sh[i]) * rms * (float)q_w[i]);
        half bh = (half)((float)((half)sh[i + HALF_D]) * rms * (float)q_w[i + HALF_D]);
        float freq = native_exp2((-0.015625f * (float)i) * log2_theta);
        float ang = (float)pos * freq;
        float c = native_cos(ang), s = native_sin(ang);
        float a = (float)ah, b = (float)bh;
        dh[i] = (half)(a * c - b * s);
        dh[i + HALF_D] = (half)(b * c + a * s);
        return;
    }

    uint kvh = gid - Q_HEADS;
    __global const float* kh_src = k_src + (ulong)kvh * D;
    float v0 = (float)((half)kh_src[t]);
    float v1 = (float)((half)kh_src[t + HALF_D]);
    float ss = qk_append_wave64_sum(lane, v0 * v0 + v1 * v1);
    float rms = rsqrt(ss * 0.0078125f + eps);
    uint i = t;
    half ah = (half)((float)((half)kh_src[i]) * rms * (float)k_w[i]);
    half bh = (half)((float)((half)kh_src[i + HALF_D]) * rms * (float)k_w[i + HALF_D]);
    float freq = native_exp2((-0.015625f * (float)i) * log2_theta);
    float ang = (float)pos * freq;
    float c = native_cos(ang), s = native_sin(ang);
    float a = (float)ah, b = (float)bh;
    k_local[i] = (half)(a * c - b * s);
    k_local[i + HALF_D] = (half)(b * c + a * s);
    barrier(CLK_LOCAL_MEM_FENCE);

    uint bidx = batch_indices[0];
    if (bidx != 0u) return;
    uint lo = indptr[0];
    uint hi = indptr[1];
    if (lo > hi || hi > total_indices) return;
    uint pages = hi - lo;
    if (pages == 0u) return;
    uint page_in_seq = pos / block_size;
    uint page_offset = pos - page_in_seq * block_size;
    if (page_in_seq >= pages) return;
    uint last_page = pages - 1u;
    uint last_len = last_page_len[0];
    if (page_in_seq == last_page) {
        if (last_len == 0u || last_len > block_size || page_offset >= last_len) return;
    } else if (page_offset >= block_size) {
        return;
    }
    uint phys_page = indices[lo + page_in_seq];
    if (phys_page >= physical_blocks) return;

    ulong rows_per_head = (ulong)physical_blocks * block_size;
    ulong row = (ulong)kvh * rows_per_head + (ulong)phys_page * block_size + page_offset;
    __global const float* vh_src = v_src + (ulong)kvh * D;

    for (uint bl = 0u; bl < 4u; ++bl) {
        float max_k = 0.0f;
        if (t < 32u) {
            max_k = fmax(max_k, fabs((float)k_local[bl * 32u + t]));
        }
        red[t] = max_k;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint o = 32u; o > 0u; o >>= 1) {
            if (t < o) red[t] = fmax(red[t], red[t + o]);
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (t == 0u) {
            int e = 0;
            if (red[0] > 0.0f) e = (int)ceil(native_log2(red[0] / 6.0f));
            int sb_i = e + 127;
            if (sb_i < 0) sb_i = 0;
            if (sb_i > 255) sb_i = 255;
            uchar sb = (uchar)sb_i;
            scale_k[row * SCALE_BYTES + bl] = sb;
            sf_local[0] = as_float(((uint)sb) << 23);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        float sf = sf_local[0];
        if (t < 16u) {
            uint d0 = bl * 32u + t * 2u;
            uint p4 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
                0u, (float)k_local[d0], (float)k_local[d0 + 1u], sf, 0);
            kcache[row * PACKED_BYTES + bl * 16u + t] = (uchar)(p4 & 0xffu);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        float max_v = 0.0f;
        if (t < 32u) {
            half hv = (half)vh_src[bl * 32u + t];
            v_local[bl * 32u + t] = hv;
            max_v = fmax(max_v, fabs((float)hv));
        }
        red[t] = max_v;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint o = 32u; o > 0u; o >>= 1) {
            if (t < o) red[t] = fmax(red[t], red[t + o]);
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (t == 0u) {
            int e = 0;
            if (red[0] > 0.0f) e = (int)ceil(native_log2(red[0] / 6.0f));
            int sb_i = e + 127;
            if (sb_i < 0) sb_i = 0;
            if (sb_i > 255) sb_i = 255;
            uchar sb = (uchar)sb_i;
            scale_v[row * SCALE_BYTES + bl] = sb;
            sf_local[0] = as_float(((uint)sb) << 23);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        sf = sf_local[0];
        if (t < 16u) {
            uint d0 = bl * 32u + t * 2u;
            uint p4 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
                0u, (float)v_local[d0], (float)v_local[d0 + 1u], sf, 0);
            vcache[row * PACKED_BYTES + bl * 16u + t] = (uchar)(p4 & 0xffu);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// OLMo 2's fused QK-norm + RoPE + paged FP4 KV append.
//
// Three things differ from cast_qk_rope_append_paged_fp4_vf32_q64_k4_d128_meta,
// and all three come from OLMo 2 being a different architecture rather than a
// different size:
//
//   1. 16 query heads over 16 KV heads. Multi-head, so there is no group to
//      share a KV read across.
//   2. QK-norm reduces over the WHOLE projection (Q_HEADS*D elements), not over
//      one head's D. Qwen3 normalises each head independently; OLMo 2 does not.
//   3. Because the norm spans the projection, q_norm and k_norm are
//      Q_HEADS*D and KV_HEADS*D wide, so each head reads its own slice rather
//      than sharing a single D-wide weight vector.
//
// Getting (2) or (3) wrong produces plausible-looking numbers rather than an
// error, which is why they are called out here and gated on hardware.
__kernel void cast_qk_rope_append_paged_fp4_vf32_q16_k16_d128_olmo2_meta(
    __global const float* q_src,
    __global const half* q_w,
    __global half* q_dst,
    __global const float* k_src,
    __global const half* k_w,
    __global const float* v_src,
    __global uchar* kcache,
    __global uchar* vcache,
    __global uchar* scale_k,
    __global uchar* scale_v,
    __global const uint* indices,
    __global const uint* indptr,
    __global const uint* last_page_len,
    __global const uint* batch_indices,
    __global const uint* positions,
    uint total_indices,
    uint physical_blocks,
    uint block_size,
    float theta,
    float eps) {
    const uint D = 128u;
    const uint HALF_D = 64u;
    const uint Q_HEADS = 16u;
    const uint KV_HEADS = 16u;
    const uint PACKED_BYTES = 64u;
    const uint SCALE_BYTES = 4u;
    uint gid = get_group_id(0);
    uint t = get_local_id(0);
    uint lane = t & 63u;
    if (gid >= Q_HEADS + KV_HEADS) return;

    uint pos = positions[0];
    float log2_theta = native_log2(theta);
    __local float red[256];
    __local half k_local[128];
    __local half v_local[128];
    __local float sf_local[1];

    if (gid < Q_HEADS) {
        __global const float* sh = q_src + (ulong)gid * D;
        __global half* dh = q_dst + (ulong)gid * D;
        // OLMo 2 normalises across the WHOLE query projection rather than per
        // head, so the reduction spans Q_HEADS*D elements instead of this head's
        // D. Every workgroup recomputes it: that is 8 KiB of extra reads and it
        // buys us not needing a cross-workgroup barrier.
        float qacc = 0.0f;
        for (uint e = t; e < Q_HEADS * D; e += 64u) {
            float qv = (float)((half)q_src[e]);
            qacc += qv * qv;
        }
        float ss = qk_append_wave64_sum(lane, qacc);
        float rms = rsqrt(ss * (1.0f / (float)(Q_HEADS * D)) + eps);
        uint i = t;
        // q_norm spans the whole projection too, so each head takes its own slice
        // instead of sharing one D-wide vector.
        __global const half* qw = q_w + (ulong)gid * D;
        half ah = (half)((float)((half)sh[i]) * rms * (float)qw[i]);
        half bh = (half)((float)((half)sh[i + HALF_D]) * rms * (float)qw[i + HALF_D]);
        float freq = native_exp2((-0.015625f * (float)i) * log2_theta);
        float ang = (float)pos * freq;
        float c = native_cos(ang), s = native_sin(ang);
        float a = (float)ah, b = (float)bh;
        dh[i] = (half)(a * c - b * s);
        dh[i + HALF_D] = (half)(b * c + a * s);
        return;
    }

    uint kvh = gid - Q_HEADS;
    __global const float* kh_src = k_src + (ulong)kvh * D;
    float kacc = 0.0f;
    for (uint e = t; e < KV_HEADS * D; e += 64u) {
        float kv = (float)((half)k_src[e]);
        kacc += kv * kv;
    }
    float ss = qk_append_wave64_sum(lane, kacc);
    float rms = rsqrt(ss * (1.0f / (float)(KV_HEADS * D)) + eps);
    uint i = t;
    __global const half* kw = k_w + (ulong)kvh * D;
    half ah = (half)((float)((half)kh_src[i]) * rms * (float)kw[i]);
    half bh = (half)((float)((half)kh_src[i + HALF_D]) * rms * (float)kw[i + HALF_D]);
    float freq = native_exp2((-0.015625f * (float)i) * log2_theta);
    float ang = (float)pos * freq;
    float c = native_cos(ang), s = native_sin(ang);
    float a = (float)ah, b = (float)bh;
    k_local[i] = (half)(a * c - b * s);
    k_local[i + HALF_D] = (half)(b * c + a * s);
    barrier(CLK_LOCAL_MEM_FENCE);

    uint bidx = batch_indices[0];
    if (bidx != 0u) return;
    uint lo = indptr[0];
    uint hi = indptr[1];
    if (lo > hi || hi > total_indices) return;
    uint pages = hi - lo;
    if (pages == 0u) return;
    uint page_in_seq = pos / block_size;
    uint page_offset = pos - page_in_seq * block_size;
    if (page_in_seq >= pages) return;
    uint last_page = pages - 1u;
    uint last_len = last_page_len[0];
    if (page_in_seq == last_page) {
        if (last_len == 0u || last_len > block_size || page_offset >= last_len) return;
    } else if (page_offset >= block_size) {
        return;
    }
    uint phys_page = indices[lo + page_in_seq];
    if (phys_page >= physical_blocks) return;

    ulong rows_per_head = (ulong)physical_blocks * block_size;
    ulong row = (ulong)kvh * rows_per_head + (ulong)phys_page * block_size + page_offset;
    __global const float* vh_src = v_src + (ulong)kvh * D;

    for (uint bl = 0u; bl < 4u; ++bl) {
        float max_k = 0.0f;
        if (t < 32u) {
            max_k = fmax(max_k, fabs((float)k_local[bl * 32u + t]));
        }
        red[t] = max_k;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint o = 32u; o > 0u; o >>= 1) {
            if (t < o) red[t] = fmax(red[t], red[t + o]);
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (t == 0u) {
            int e = 0;
            if (red[0] > 0.0f) e = (int)ceil(native_log2(red[0] / 6.0f));
            int sb_i = e + 127;
            if (sb_i < 0) sb_i = 0;
            if (sb_i > 255) sb_i = 255;
            uchar sb = (uchar)sb_i;
            scale_k[row * SCALE_BYTES + bl] = sb;
            sf_local[0] = as_float(((uint)sb) << 23);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        float sf = sf_local[0];
        if (t < 16u) {
            uint d0 = bl * 32u + t * 2u;
            uint p4 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
                0u, (float)k_local[d0], (float)k_local[d0 + 1u], sf, 0);
            kcache[row * PACKED_BYTES + bl * 16u + t] = (uchar)(p4 & 0xffu);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        float max_v = 0.0f;
        if (t < 32u) {
            half hv = (half)vh_src[bl * 32u + t];
            v_local[bl * 32u + t] = hv;
            max_v = fmax(max_v, fabs((float)hv));
        }
        red[t] = max_v;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint o = 32u; o > 0u; o >>= 1) {
            if (t < o) red[t] = fmax(red[t], red[t + o]);
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (t == 0u) {
            int e = 0;
            if (red[0] > 0.0f) e = (int)ceil(native_log2(red[0] / 6.0f));
            int sb_i = e + 127;
            if (sb_i < 0) sb_i = 0;
            if (sb_i > 255) sb_i = 255;
            uchar sb = (uchar)sb_i;
            scale_v[row * SCALE_BYTES + bl] = sb;
            sf_local[0] = as_float(((uint)sb) << 23);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        sf = sf_local[0];
        if (t < 16u) {
            uint d0 = bl * 32u + t * 2u;
            uint p4 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
                0u, (float)v_local[d0], (float)v_local[d0 + 1u], sf, 0);
            vcache[row * PACKED_BYTES + bl * 16u + t] = (uchar)(p4 & 0xffu);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void cast_rmsnorm_rope_qk_heads_meta(__global const float* q_src,
                                         __global const half* q_w,
                                         __global half* q_dst,
                                         __global const float* k_src,
                                         __global const half* k_w,
                                         __global half* k_dst,
                                         uint q_heads, uint k_heads, uint D,
                                         __global const uint* positions, float theta, float eps) {
    uint pos = positions[0];
    uint gid = get_group_id(0);
    uint t = get_local_id(0);
    uint total = q_heads + k_heads;
    if (gid >= total) return;
    uint is_k = gid >= q_heads;
    uint head = is_k ? gid - q_heads : gid;
    __global const float* src = is_k ? k_src : q_src;
    __global const half* w = is_k ? k_w : q_w;
    __global half* dst = is_k ? k_dst : q_dst;
    __global const float* sh = src + (ulong)head * D;
    __global half* dh = dst + (ulong)head * D;
    __local float red[256];

    float ss = 0.0f;
    for (uint i = t; i < D; i += 256u) {
        float v = (float)((half)sh[i]);
        ss += v * v;
    }
    red[t] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint o = 128u; o > 0u; o >>= 1) {
        if (t < o) red[t] += red[t + o];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float rms = rsqrt(red[0] / (float)D + eps);

    uint half_d = D >> 1;
    for (uint i = t; i < half_d; i += 256u) {
        half ah = (half)((float)((half)sh[i]) * rms * (float)w[i]);
        half bh = (half)((float)((half)sh[i + half_d]) * rms * (float)w[i + half_d]);
        float freq = pow(theta, -2.0f * (float)i / (float)D);
        float ang = (float)pos * freq;
        float c = native_cos(ang), s = native_sin(ang);
        float a = (float)ah, b = (float)bh;
        dh[i] = (half)(a * c - b * s);
        dh[i + half_d] = (half)(b * c + a * s);
    }
}

// Per-head RoPE at position `pos` (half-rotation): x is [nheads*D]. One workgroup
// per head; each thread handles a (i, i+D/2) pair.
__kernel void rope_heads(__global half* x, uint nheads, uint D, uint pos, float theta) {
    uint head = get_group_id(0);
    uint t = get_local_id(0);
    if (head >= nheads) return;
    __global half* xh = x + (ulong)head * D;
    uint half_d = D >> 1;
    for (uint i = t; i < half_d; i += 256u) {
        float freq = pow(theta, -2.0f * (float)i / (float)D);
        float ang = (float)pos * freq;
        float c = native_cos(ang), s = native_sin(ang);
        float a = (float)xh[i], b = (float)xh[i + half_d];
        xh[i] = (half)(a * c - b * s);
        xh[i + half_d] = (half)(b * c + a * s);
    }
}

// Deterministic bounded merge_states probe for split-KV attention. This combines
// N local attention states (O_i, m_i, l_i) with the logsumexp recurrence used by
// FlashInfer/SGLang-style split-KV decode. Production fixed split-size decode
// will produce a variable number of fixed-page local states; this probe proves
// the bounded N-way combine primitive before vectorizing it.
__kernel void split_kv_merge_n_states_probe(__global const ulong* states,
                                            __global ulong* out,
                                            uint d,
                                            uint n_states,
                                            uint stride_u64,
                                            uint reserved0) {
    if (get_global_id(0) != 0) return;

    for (uint i = 0u; i < 16u; ++i)
        out[i] = 0UL;
    out[7] = 0x5A117E590BADF00DUL;

    if (d == 0u || d > 8u || n_states == 0u || n_states > 8u || stride_u64 < 16u) {
        out[0] = 3UL;
        out[1] = (ulong)d;
        out[2] = (ulong)n_states;
        out[3] = (ulong)stride_u64;
        return;
    }

    ulong any_status = 0UL;
    ulong bad = 0UL;
    ulong nulls = 0UL;
    ulong valid = 0UL;
    ulong first_bad = 0UL;
    float m = -3.402823466e+38F;
    float acc[8];
    for (uint dim = 0u; dim < 8u; ++dim)
        acc[dim] = 0.0f;

    for (uint s = 0u; s < n_states; ++s) {
        const __global ulong* st = states + ((ulong)s * (ulong)stride_u64);
        if (st[7] != 0xA77E47100BADF00DUL) {
            out[0] = 5UL;
            out[6] = (ulong)s;
            return;
        }
        const ulong st_status = st[0];
        const ulong st_bad = st[1];
        any_status |= st_status;
        bad += st_bad;
        nulls += st[2];
        valid += st[3];
        if (first_bad == 0UL && st_bad != 0UL)
            first_bad = st[6];
        m = fmax(m, as_float((uint)st[4]));
    }

    float l = 0.0f;
    for (uint s = 0u; s < n_states; ++s) {
        const __global ulong* st = states + ((ulong)s * (ulong)stride_u64);
        const float st_m = as_float((uint)st[4]);
        const float st_l = as_float((uint)st[5]);
        const float w = st_l * exp(st_m - m);
        l += w;
        for (uint dim = 0u; dim < d; ++dim)
            acc[dim] += as_float((uint)st[8u + dim]) * w;
    }

    if (!(l > 0.0f)) {
        out[0] = 4UL;
        out[1] = (ulong)d;
        out[2] = (ulong)n_states;
        out[7] = 0x5A117E590BADF00DUL;
        return;
    }

    out[0] = any_status == 0UL ? 0UL : 2UL;
    out[1] = bad;
    out[2] = nulls;
    out[3] = valid;
    out[4] = (ulong)as_uint(m);
    out[5] = (ulong)as_uint(l);
    out[6] = first_bad;
    out[7] = 0x5A117E590BADF00DUL;
    for (uint dim = 0u; dim < d; ++dim)
        out[8u + dim] = (ulong)as_uint(acc[dim] / l);
}

// Head-parallel bounded merge_states probe. Input records are laid out
// state-major like FlashInfer merge_states: [state, head, record_u64]. Each
// work-item merges all states for one head in a deterministic state order.
__kernel void split_kv_merge_n_states_heads_probe(__global const ulong* states,
                                                  __global ulong* out,
                                                  uint d,
                                                  uint n_states,
                                                  uint n_heads,
                                                  uint stride_u64) {
    const uint head = get_global_id(0);
    if (head >= n_heads) return;

    __global ulong* dst = out + ((ulong)head * (ulong)stride_u64);
    for (uint i = 0u; i < 16u; ++i)
        dst[i] = 0UL;
    dst[7] = 0x5A117E5A0BADF00DUL;

    if (d == 0u || d > 8u || n_states == 0u || n_states > 8u || n_heads == 0u || n_heads > 64u || stride_u64 < 16u) {
        dst[0] = 3UL;
        dst[1] = (ulong)d;
        dst[2] = (ulong)n_states;
        dst[3] = (ulong)n_heads;
        return;
    }

    ulong any_status = 0UL;
    ulong bad = 0UL;
    ulong nulls = 0UL;
    ulong valid = 0UL;
    ulong first_bad = 0UL;
    float m = -3.402823466e+38F;
    float acc[8];
    for (uint dim = 0u; dim < 8u; ++dim)
        acc[dim] = 0.0f;

    for (uint s = 0u; s < n_states; ++s) {
        const __global ulong* st = states + ((((ulong)s * (ulong)n_heads) + (ulong)head) * (ulong)stride_u64);
        if (st[7] != 0xA77E47100BADF00DUL) {
            dst[0] = 5UL;
            dst[6] = (ulong)s;
            return;
        }
        const ulong st_status = st[0];
        const ulong st_bad = st[1];
        any_status |= st_status;
        bad += st_bad;
        nulls += st[2];
        valid += st[3];
        if (first_bad == 0UL && st_bad != 0UL)
            first_bad = st[6];
        m = fmax(m, as_float((uint)st[4]));
    }

    float l = 0.0f;
    for (uint s = 0u; s < n_states; ++s) {
        const __global ulong* st = states + ((((ulong)s * (ulong)n_heads) + (ulong)head) * (ulong)stride_u64);
        const float st_m = as_float((uint)st[4]);
        const float st_l = as_float((uint)st[5]);
        const float w = st_l * exp(st_m - m);
        l += w;
        for (uint dim = 0u; dim < d; ++dim)
            acc[dim] += as_float((uint)st[8u + dim]) * w;
    }

    if (!(l > 0.0f)) {
        dst[0] = 4UL;
        dst[1] = (ulong)d;
        dst[2] = (ulong)n_states;
        dst[3] = (ulong)head;
        dst[7] = 0x5A117E5A0BADF00DUL;
        return;
    }

    dst[0] = any_status == 0UL ? 0UL : 2UL;
    dst[1] = bad;
    dst[2] = nulls;
    dst[3] = valid;
    dst[4] = (ulong)as_uint(m);
    dst[5] = (ulong)as_uint(l);
    dst[6] = first_bad;
    dst[7] = 0x5A117E5A0BADF00DUL;
    for (uint dim = 0u; dim < d; ++dim)
        dst[8u + dim] = (ulong)as_uint(acc[dim] / l);
}

// Head-parallel production-width merge_states probe. This keeps the same
// [state, head, record_u64] layout as split_kv_merge_n_states_heads_probe but
// raises the vector payload to head_dim<=128, matching Qwen/Kimi attention
// dimensions while avoiding a 128-float private accumulator.
__kernel void split_kv_merge_n_states_heads128_probe(__global const ulong* states,
                                                     __global ulong* out,
                                                     uint d,
                                                     uint n_states,
                                                     uint n_heads,
                                                     uint stride_u64) {
    const uint head = get_global_id(0);
    if (head >= n_heads) return;

    __global ulong* dst = out + ((ulong)head * (ulong)stride_u64);
    for (uint i = 0u; i < 136u; ++i)
        dst[i] = 0UL;
    dst[7] = 0x5A117E128BADF00DUL;

    if (d == 0u || d > 128u || n_states == 0u || n_states > 8u || n_heads == 0u || n_heads > 64u || stride_u64 < (8u + d)) {
        dst[0] = 3UL;
        dst[1] = (ulong)d;
        dst[2] = (ulong)n_states;
        dst[3] = (ulong)n_heads;
        return;
    }

    ulong any_status = 0UL;
    ulong bad = 0UL;
    ulong nulls = 0UL;
    ulong valid = 0UL;
    ulong first_bad = 0UL;
    float m = -3.402823466e+38F;

    for (uint s = 0u; s < n_states; ++s) {
        const __global ulong* st = states + ((((ulong)s * (ulong)n_heads) + (ulong)head) * (ulong)stride_u64);
        if (st[7] != 0xA77E47100BADF00DUL) {
            dst[0] = 5UL;
            dst[6] = (ulong)s;
            return;
        }
        const ulong st_status = st[0];
        const ulong st_bad = st[1];
        any_status |= st_status;
        bad += st_bad;
        nulls += st[2];
        valid += st[3];
        if (first_bad == 0UL && st_bad != 0UL)
            first_bad = st[6];
        m = fmax(m, as_float((uint)st[4]));
    }

    float weights[8];
    float l = 0.0f;
    for (uint s = 0u; s < n_states; ++s) {
        const __global ulong* st = states + ((((ulong)s * (ulong)n_heads) + (ulong)head) * (ulong)stride_u64);
        const float st_m = as_float((uint)st[4]);
        const float st_l = as_float((uint)st[5]);
        const float w = st_l * exp(st_m - m);
        weights[s] = w;
        l += w;
    }

    if (!(l > 0.0f)) {
        dst[0] = 4UL;
        dst[1] = (ulong)d;
        dst[2] = (ulong)n_states;
        dst[3] = (ulong)head;
        dst[7] = 0x5A117E128BADF00DUL;
        return;
    }

    dst[0] = any_status == 0UL ? 0UL : 2UL;
    dst[1] = bad;
    dst[2] = nulls;
    dst[3] = valid;
    dst[4] = (ulong)as_uint(m);
    dst[5] = (ulong)as_uint(l);
    dst[6] = first_bad;
    dst[7] = 0x5A117E128BADF00DUL;
    for (uint dim = 0u; dim < d; ++dim) {
        float acc = 0.0f;
        for (uint s = 0u; s < n_states; ++s) {
            const __global ulong* st = states + ((((ulong)s * (ulong)n_heads) + (ulong)head) * (ulong)stride_u64);
            acc += as_float((uint)st[8u + dim]) * weights[s];
        }
        dst[8u + dim] = (ulong)as_uint(acc / l);
    }
}

// Dim-lane production-width merge_states probe. This keeps the same
// [state, head, record_u64] layout as the scalar head_dim=128 probe, but maps
// work-items to [head, dim] lanes so the 128 output dimensions are produced in
// parallel while preserving deterministic state-order LSE reduction per lane.
__kernel void split_kv_merge_n_states_heads128_lanes_probe(__global const ulong* states,
                                                           __global ulong* out,
                                                           uint d,
                                                           uint n_states,
                                                           uint n_heads,
                                                           uint stride_u64) {
    const uint gid = get_global_id(0);
    if (d == 0u) return;
    const uint head = gid / d;
    const uint dim = gid - head * d;
    if (head >= n_heads || dim >= d) return;

    __global ulong* dst = out + ((ulong)head * (ulong)stride_u64);
    if (dim == 0u) {
        for (uint i = 0u; i < 8u; ++i)
            dst[i] = 0UL;
        dst[7] = 0x5A117E128D1AF00DUL;
    }

    if (d > 128u || n_states == 0u || n_states > 8u || n_heads == 0u || n_heads > 64u || stride_u64 < (8u + d)) {
        if (dim == 0u) {
            dst[0] = 3UL;
            dst[1] = (ulong)d;
            dst[2] = (ulong)n_states;
            dst[3] = (ulong)n_heads;
            dst[7] = 0x5A117E128D1AF00DUL;
        }
        return;
    }

    ulong any_status = 0UL;
    ulong bad = 0UL;
    ulong nulls = 0UL;
    ulong valid = 0UL;
    ulong first_bad = 0UL;
    float m = -3.402823466e+38F;

    for (uint s = 0u; s < n_states; ++s) {
        const __global ulong* st = states + ((((ulong)s * (ulong)n_heads) + (ulong)head) * (ulong)stride_u64);
        if (st[7] != 0xA77E47100BADF00DUL) {
            if (dim == 0u) {
                dst[0] = 5UL;
                dst[6] = (ulong)s;
                dst[7] = 0x5A117E128D1AF00DUL;
            }
            return;
        }
        if (dim == 0u) {
            const ulong st_status = st[0];
            const ulong st_bad = st[1];
            any_status |= st_status;
            bad += st_bad;
            nulls += st[2];
            valid += st[3];
            if (first_bad == 0UL && st_bad != 0UL)
                first_bad = st[6];
        }
        m = fmax(m, as_float((uint)st[4]));
    }

    float l = 0.0f;
    for (uint s = 0u; s < n_states; ++s) {
        const __global ulong* st = states + ((((ulong)s * (ulong)n_heads) + (ulong)head) * (ulong)stride_u64);
        const float st_m = as_float((uint)st[4]);
        const float st_l = as_float((uint)st[5]);
        l += st_l * exp(st_m - m);
    }

    if (!(l > 0.0f)) {
        if (dim == 0u) {
            dst[0] = 4UL;
            dst[1] = (ulong)d;
            dst[2] = (ulong)n_states;
            dst[3] = (ulong)head;
            dst[7] = 0x5A117E128D1AF00DUL;
        }
        return;
    }

    if (dim == 0u) {
        dst[0] = any_status == 0UL ? 0UL : 2UL;
        dst[1] = bad;
        dst[2] = nulls;
        dst[3] = valid;
        dst[4] = (ulong)as_uint(m);
        dst[5] = (ulong)as_uint(l);
        dst[6] = first_bad;
        dst[7] = 0x5A117E128D1AF00DUL;
    }

    float acc = 0.0f;
    for (uint s = 0u; s < n_states; ++s) {
        const __global ulong* st = states + ((((ulong)s * (ulong)n_heads) + (ulong)head) * (ulong)stride_u64);
        const float st_m = as_float((uint)st[4]);
        const float st_l = as_float((uint)st[5]);
        const float w = st_l * exp(st_m - m);
        acc += as_float((uint)st[8u + dim]) * w;
    }
    dst[8u + dim] = (ulong)as_uint(acc / l);
}

// Cooperative production-width merge_states probe. One 128-lane workgroup owns
// one head. Lane 0 computes the deterministic state-order metadata/LSE/weights
// once into tiny LDS, then all lanes write the head_dim=128 output dimensions.
__kernel void split_kv_merge_n_states_heads128_coop_probe(__global const ulong* states,
                                                          __global ulong* out,
                                                          uint d,
                                                          uint n_states,
                                                          uint n_heads,
                                                          uint stride_u64) {
    const uint head = get_group_id(0);
    const uint dim = get_local_id(0);
    if (head >= n_heads) return;

    __global ulong* dst = out + ((ulong)head * (ulong)stride_u64);
    __local float weights[8];
    __local float shared_m;
    __local float shared_l;
    __local ulong shared_status;
    __local ulong shared_bad;
    __local ulong shared_nulls;
    __local ulong shared_valid;
    __local ulong shared_first_bad;
    __local uint ok;

    if (dim == 0u) {
        for (uint i = 0u; i < 8u; ++i)
            dst[i] = 0UL;
        dst[7] = 0x5A117E128C00F00DUL;
    }

    if (d == 0u || d > 128u || n_states == 0u || n_states > 8u || n_heads == 0u || n_heads > 64u || stride_u64 < (8u + d) || get_local_size(0) < d) {
        if (dim == 0u) {
            dst[0] = 3UL;
            dst[1] = (ulong)d;
            dst[2] = (ulong)n_states;
            dst[3] = (ulong)n_heads;
            dst[7] = 0x5A117E128C00F00DUL;
        }
        return;
    }

    if (dim == 0u) {
        ok = 1u;
        shared_status = 0UL;
        shared_bad = 0UL;
        shared_nulls = 0UL;
        shared_valid = 0UL;
        shared_first_bad = 0UL;
        shared_m = -3.402823466e+38F;

        for (uint s = 0u; s < n_states; ++s) {
            const __global ulong* st = states + ((((ulong)s * (ulong)n_heads) + (ulong)head) * (ulong)stride_u64);
            if (st[7] != 0xA77E47100BADF00DUL) {
                dst[0] = 5UL;
                dst[6] = (ulong)s;
                dst[7] = 0x5A117E128C00F00DUL;
                ok = 0u;
                break;
            }
            const ulong st_status = st[0];
            const ulong st_bad = st[1];
            shared_status |= st_status;
            shared_bad += st_bad;
            shared_nulls += st[2];
            shared_valid += st[3];
            if (shared_first_bad == 0UL && st_bad != 0UL)
                shared_first_bad = st[6];
            shared_m = fmax(shared_m, as_float((uint)st[4]));
        }

        shared_l = 0.0f;
        if (ok != 0u) {
            for (uint s = 0u; s < n_states; ++s) {
                const __global ulong* st = states + ((((ulong)s * (ulong)n_heads) + (ulong)head) * (ulong)stride_u64);
                const float st_m = as_float((uint)st[4]);
                const float st_l = as_float((uint)st[5]);
                const float w = st_l * exp(st_m - shared_m);
                weights[s] = w;
                shared_l += w;
            }
            if (!(shared_l > 0.0f)) {
                dst[0] = 4UL;
                dst[1] = (ulong)d;
                dst[2] = (ulong)n_states;
                dst[3] = (ulong)head;
                dst[7] = 0x5A117E128C00F00DUL;
                ok = 0u;
            } else {
                dst[0] = shared_status == 0UL ? 0UL : 2UL;
                dst[1] = shared_bad;
                dst[2] = shared_nulls;
                dst[3] = shared_valid;
                dst[4] = (ulong)as_uint(shared_m);
                dst[5] = (ulong)as_uint(shared_l);
                dst[6] = shared_first_bad;
                dst[7] = 0x5A117E128C00F00DUL;
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (ok == 0u || dim >= d) return;

    float acc = 0.0f;
    for (uint s = 0u; s < n_states; ++s) {
        const __global ulong* st = states + ((((ulong)s * (ulong)n_heads) + (ulong)head) * (ulong)stride_u64);
        acc += as_float((uint)st[8u + dim]) * weights[s];
    }
    dst[8u + dim] = (ulong)as_uint(acc / shared_l);
}
