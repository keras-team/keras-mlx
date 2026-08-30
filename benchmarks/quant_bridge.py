"""Can mx.quantized_matmul stand in for a keras Dense int4/int8 forward pass?

keras Dense computes x @ kernel with kernel [in, out]. mlx quantized_matmul
computes x @ w.T with groups along the last axis of w, so w = kernel.T gives
groups along the reduction axis, which is what sub-channel keras does too.
"""

import mlx.core as mx
import numpy as np

rng = np.random.default_rng(0)
BATCH, IN, OUT = 8, 256, 128

x_np = rng.standard_normal((BATCH, IN)).astype("float32")
k_np = rng.standard_normal((IN, OUT)).astype("float32") * 0.1

x = mx.array(x_np)
kernel = mx.array(k_np)
ref = np.asarray(mx.matmul(x, kernel))  # float reference


def rel_err(got):
    return float(np.abs(np.asarray(got) - ref).max() / np.abs(ref).max())


def keras_style_per_channel_int8():
    """What keras _int8_call does: int8 kernel, integer matmul, descale."""
    scale = np.abs(k_np).max(axis=0, keepdims=True) / 127.0
    kq = np.round(k_np / scale).clip(-127, 127).astype("int8")
    out = x_np.astype("float32") @ kq.astype("float32")
    return out * scale


def mlx_fused(bits, group_size):
    """What we would call behind a backend seam."""
    w = mx.array(np.ascontiguousarray(k_np.T))  # [out, in], groups along in
    wq, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    return mx.quantized_matmul(
        x,
        wq,
        scales,
        biases,
        transpose=True,
        group_size=group_size,
        bits=bits,
    )


print("relative max error vs the float matmul\n")
keras_err = rel_err(keras_style_per_channel_int8())
print("  keras int8 per channel      %.5f" % keras_err)
for bits in (8, 4):
    for gs in (32, 64, 128):
        print(
            "  mlx fused bits=%d group=%-4d %.5f"
            % (bits, gs, rel_err(mlx_fused(bits, gs)))
        )

# shapes actually stored, which is the memory argument
w = mx.array(np.ascontiguousarray(k_np.T))
for bits in (8, 4):
    wq, sc, bi = mx.quantize(w, group_size=64, bits=bits)
    stored = wq.nbytes + sc.nbytes + bi.nbytes
    print(
        "\n  bits=%d group=64: packed %s %s + scales %s + biases %s"
        % (bits, tuple(wq.shape), wq.dtype, tuple(sc.shape), tuple(bi.shape))
    )
    print(
        "    %d bytes vs %d as float32, %.2fx smaller"
        % (stored, k_np.nbytes, k_np.nbytes / stored)
    )
