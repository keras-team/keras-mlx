# benchmarks

Scripts that measure the mlx backend against the path keras would otherwise
take. Each one runs standalone and prints a table.

Run them from anywhere with keras and keras-mlx importable:

```
KERAS_BACKEND=mlx python benchmarks/quant_bridge.py
```

Accuracy comparisons run anywhere, including linux cpu wheels. Throughput
numbers need Metal, so run those on Apple Silicon.

## quant_bridge.py

Checks whether `mx.quantized_matmul` can stand in for a keras `Dense` int8 or
int4 forward pass, and how the error compares. keras computes `x @ kernel` with
kernel `[in, out]`. mlx computes `x @ w.T` with groups along the last axis of
`w`, so `w = kernel.T` puts the groups on the reduction axis, which is what
keras sub channel quantization already does.

Measured 2026-08-30, relative max error against the float matmul:

| path | error |
| --- | --- |
| keras int8 per channel, what keras does today | 0.00764 |
| mlx fused int8, group 32 | 0.00416 |
| mlx fused int4, group 32 | 0.06939 |

So mlx's grouped affine int8 is about 1.8x more accurate than the current keras
path, at 3.56x compression for int8 and 6.40x for int4 including scales and
biases.

This is not reachable from keras yet. There is no backend dispatched quantized
op, so `Dense` composes from `ops.matmul` and never reaches a fused kernel, so
this measures the gap rather than exercising a fast path.
