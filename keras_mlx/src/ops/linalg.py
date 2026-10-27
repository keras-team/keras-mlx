import mlx.core as mx
import numpy as np

from keras.src import tree
from keras.src.backend.common import result_type
from keras.src.backend.common import standardize_dtype
from keras_mlx.src.ops.core import convert_to_tensor
from keras_mlx.src.ops.core import lapack_float
from keras_mlx.src.ops.core import promote_to_float


def _det_2x2(a):
    return a[..., 0, 0] * a[..., 1, 1] - a[..., 0, 1] * a[..., 1, 0]


def _det_3x3(a):
    return (
        a[..., 0, 0] * a[..., 1, 1] * a[..., 2, 2]
        + a[..., 0, 1] * a[..., 1, 2] * a[..., 2, 0]
        + a[..., 0, 2] * a[..., 1, 0] * a[..., 2, 1]
        - a[..., 0, 2] * a[..., 1, 1] * a[..., 2, 0]
        - a[..., 0, 0] * a[..., 1, 2] * a[..., 2, 1]
        - a[..., 0, 1] * a[..., 1, 0] * a[..., 2, 2]
    )


@mx.custom_function
def _eigvals_det(a):
    # The eigenvalue product handles singular matrices naturally.
    with mx.stream(mx.cpu):
        w = mx.linalg.eigvals(a)
    return mx.real(mx.prod(w, axis=-1))


@_eigvals_det.vjp
def _eigvals_det_vjp(primals, cotangent, output):
    # eigvals has no vjp. The gradient is the transposed adjugate, built from
    # the svd so it stays finite for singular matrices.
    with mx.stream(mx.cpu):
        u, s, vt = mx.linalg.svd(primals)
    # The singular values drop the sign, which is det(u @ vt).
    sign = _eigvals_det(mx.matmul(u, vt))
    others = mx.cumprod(s, axis=-1, inclusive=False) * mx.cumprod(
        s, axis=-1, reverse=True, inclusive=False
    )
    adjugate_t = mx.matmul(u * others[..., None, :], vt)
    return (cotangent * sign)[..., None, None] * adjugate_t


def det(a):
    a = convert_to_tensor(a)
    # Promote integer and bool inputs to float, matching jax and so the 2x2
    # and 3x3 fast paths do not return an integer determinant.
    a = promote_to_float(a)
    a_shape = a.shape
    if len(a_shape) >= 2 and a_shape[-1] == 2 and a_shape[-2] == 2:
        return _det_2x2(a)
    elif len(a_shape) >= 2 and a_shape[-1] == 3 and a_shape[-2] == 3:
        return _det_3x3(a)
    a, target = lapack_float(a)
    return _eigvals_det(a).astype(target)


def eig(a):
    # Ints and bools are not accepted by the mlx kernels, so promote like jax.
    a = promote_to_float(a)
    with mx.stream(mx.cpu):
        # This op is not yet supported on the GPU.
        return mx.linalg.eig(a)


def eigh(a):
    # Ints and bools are not accepted by the mlx kernels, so promote like jax.
    a = promote_to_float(a)
    with mx.stream(mx.cpu):
        return mx.linalg.eigh(a)


def lu_factor(a):
    # Ints and bools are not accepted by the mlx kernels, so promote like jax.
    a = promote_to_float(a)
    with mx.stream(mx.cpu):
        # This op is not yet supported on the GPU.
        lu, pivots = mx.linalg.lu_factor(a)
    return lu, pivots.astype(mx.int32)


def solve(a, b):
    # Promote both operands together, as jax does for mixed inputs.
    a, b = promote_to_float(a, b)
    vector_rhs = b.ndim == a.ndim - 1
    if vector_rhs:
        b = mx.expand_dims(b, axis=-1)
    with mx.stream(mx.cpu):
        # This op is not yet supported on the GPU.
        x = mx.linalg.solve(a, b)
    return mx.squeeze(x, axis=-1) if vector_rhs else x


def solve_triangular(a, b, lower=False):
    # Promote both operands together, as jax does for mixed inputs.
    a, b = promote_to_float(a, b)
    upper = not lower
    with mx.stream(mx.cpu):
        # This op is not yet supported on the GPU.
        if b.ndim == a.ndim - 1:
            b = mx.expand_dims(b, axis=-1)
            return mx.squeeze(
                mx.linalg.solve_triangular(a, b, upper=upper), axis=-1
            )
        return mx.linalg.solve_triangular(a, b, upper=upper)


def qr(x, mode="reduced"):
    if mode not in ("reduced", "complete"):
        raise ValueError(
            "`mode` argument value not supported. "
            "Expected one of {'reduced', 'complete'}. "
            f"Received: mode={mode}"
        )
    x = convert_to_tensor(x)
    # Ints and bools are not accepted by the mlx kernels, so promote like jax.
    x = promote_to_float(x)
    with mx.stream(mx.cpu):
        if mode == "reduced":
            return mx.linalg.qr(x)
        # mlx only computes the reduced form, so build the full basis by
        # factoring x with an identity appended. The extra columns fill out
        # the orthonormal complement, and R is then Q transpose times x.
        rows = x.shape[-2]
        eye = mx.broadcast_to(
            mx.eye(rows, dtype=x.dtype), x.shape[:-1] + (rows,)
        )
        q, _ = mx.linalg.qr(mx.concatenate([x, eye], axis=-1))
        r = mx.matmul(mx.swapaxes(q, -1, -2), x)
        return q, r


def svd(x, full_matrices=True, compute_uv=True):
    # Ints and bools are not accepted by the mlx kernels, so promote like jax.
    x = promote_to_float(x)
    with mx.stream(mx.cpu):
        if not compute_uv:
            return mx.linalg.svd(x, compute_uv=False)
        u, s, vt = mx.linalg.svd(x)
        if not full_matrices:
            n = min(x.shape[-2:])
            return u[..., :n], s, vt[..., :n, :]
        # mlx returns full matrices by default
        return u, s, vt


def cholesky(a, upper=False):
    # Ints and bools are not accepted by the mlx kernels, so promote like jax.
    a = promote_to_float(a)
    with mx.stream(mx.cpu):
        out = mx.linalg.cholesky(a, upper=upper)
    # Raise for a non positive definite input like numpy and jax. mlx returns a
    # negative diagonal rather than nan, so require > 0, which also catches nan.
    diagonal = mx.diagonal(out, axis1=-2, axis2=-1)
    try:
        positive = bool(mx.all(diagonal > 0))
    except ValueError:
        # Values cannot be read while tracing.
        positive = True
    if not positive:
        raise ValueError(
            "Cholesky decomposition failed. The input might not be a valid "
            "positive definite matrix."
        )
    return out


def norm(x, ord=None, axis=None, keepdims=False):
    dtype = standardize_dtype(x.dtype)
    if "int" in dtype or dtype == "bool":
        dtype = result_type(x.dtype, "float32")
    x = convert_to_tensor(x, dtype=dtype)
    x, result_dtype = lapack_float(x)
    # Singular value matrix norms are svd-based, which is CPU-only in mlx.
    if isinstance(axis, (tuple, list)):
        is_matrix_norm = True
    else:
        is_matrix_norm = axis is None and x.ndim == 2
    try:
        if is_matrix_norm and ord in (2, -2, "nuc"):
            with mx.stream(mx.cpu):
                output = mx.linalg.norm(
                    x, ord=ord, axis=axis, keepdims=keepdims
                )
        else:
            output = mx.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
    except IndexError as e:
        # Match the other backends, which raise a ValueError for an invalid
        # axis instead of mlx's IndexError.
        raise ValueError(str(e)) from e
    return output.astype(result_dtype)


def inv(a):
    # Ints and bools are not accepted by the mlx kernels, so promote like jax.
    a = promote_to_float(a)
    with mx.stream(mx.cpu):
        return mx.linalg.inv(a)


def lstsq(a, b, rcond=None):
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            "Incompatible shapes: a and b must have the same number of rows."
        )
    # Promote both operands together, as jax does for mixed inputs.
    a, b = promote_to_float(a, b)
    b_orig_ndim = b.ndim
    if b.ndim == 1:
        b = mx.expand_dims(b, axis=-1)
    elif b.ndim > 2:
        raise ValueError("b must be 1D or 2D.")

    m, n = a.shape
    dtype = a.dtype

    eps = np.finfo(standardize_dtype(dtype)).eps
    if rcond is None:
        rcond = eps * max(m, n)
    else:
        rcond = mx.where(rcond < 0, eps, rcond)
    u, s, vt = svd(a, full_matrices=False)

    # A zero matrix has s[0] == 0, which would keep every zero singular value.
    mask = (s > 0) & (s >= mx.array(rcond, dtype=s.dtype) * s[0])
    safe_s = mx.array(mx.where(mask, s, 1), dtype=dtype)
    s_inv = mx.where(mask, 1 / safe_s, 0)
    s_inv = mx.expand_dims(s_inv, axis=1)
    u_t_b = mx.matmul(mx.transpose(mx.conj(u)), b)
    x = mx.matmul(mx.transpose(mx.conj(vt)), s_inv * u_t_b)

    if b_orig_ndim == 1:
        x = mx.squeeze(x, axis=-1)

    return x


def _int_matmul(x, y):
    # mlx matmul rejects ints and bools, so multiply and sum in place. The
    # astype is needed because mx.sum widens small ints and bool.
    prod = mx.expand_dims(x, -1) * mx.expand_dims(y, -3)
    return mx.sum(prod, axis=-2).astype(x.dtype)


def matrix_power(a, n):
    a = convert_to_tensor(a)
    # 1. A zero power is the identity, broadcast over any batch dimensions.
    if n == 0:
        eye = mx.eye(a.shape[-1], dtype=a.dtype)
        return mx.broadcast_to(eye, a.shape)
    # 2. A negative power is the inverse raised to the absolute power.
    if n < 0:
        a = inv(a)
        n = -n
    # 3. Compute by binary exponentiation, matching numpy's repeated squaring.
    matmul = mx.matmul if mx.issubdtype(a.dtype, mx.inexact) else _int_matmul
    result = None
    while n > 0:
        if n % 2 == 1:
            result = a if result is None else matmul(result, a)
        n //= 2
        if n > 0:
            a = matmul(a, a)
    return result


def pinv(x, rcond=None):
    x = convert_to_tensor(x)
    # Ints and bools are not accepted by the mlx kernels, so promote like jax.
    x, target = lapack_float(promote_to_float(x))
    if rcond is None:
        with mx.stream(mx.cpu):
            return mx.linalg.pinv(x).astype(target)
    # mlx pinv has no rcond parameter, build the pseudo inverse from the
    # svd with the numpy cutoff rule.
    u, s, vt = svd(x, full_matrices=False)
    cutoff = rcond * mx.max(s, axis=-1, keepdims=True)
    mask = s > cutoff
    s_inv = mx.where(mask, 1 / mx.where(mask, s, 1), 0)
    v_scaled = mx.swapaxes(vt, -1, -2) * mx.expand_dims(s_inv, -2)
    return (v_scaled @ mx.swapaxes(u, -1, -2)).astype(target)


def matrix_rank(x, tol=None):
    x = convert_to_tensor(x)
    if x.ndim < 2:
        raise ValueError(
            "Expected input to have rank >= 2. "
            f"Received input with shape {x.shape}."
        )
    # Promote integer and bool inputs to float, matching jax.
    x = promote_to_float(x)
    std = standardize_dtype(x.dtype)
    x, _ = lapack_float(x)
    s = svd(x, compute_uv=False)
    if tol is None:
        # eps of the INPUT dtype, not the compute one. A float32 eps would put
        # the cutoff for a widened half input far below the data's noise.
        eps = np.finfo(std).eps
        tol = mx.max(s, axis=-1, keepdims=True) * max(x.shape[-2:]) * eps
    return mx.sum(s > tol, axis=-1).astype(mx.int32)


def cholesky_inverse(a, upper=False):
    a = convert_to_tensor(a)
    # Ints and bools are not accepted by the mlx kernels, so promote like jax.
    a = promote_to_float(a)
    identity = mx.eye(a.shape[-1], dtype=a.dtype)
    inv_chol = solve_triangular(a, identity, lower=not upper)
    inv_chol_t = mx.swapaxes(inv_chol, -1, -2)
    if upper:
        return mx.matmul(inv_chol, inv_chol_t)
    return mx.matmul(inv_chol_t, inv_chol)


def jvp(fun, primals, tangents, has_aux=False):
    primals = list(primals)
    tangents = list(tangents)

    # mx.jvp has no has_aux, so strip the aux output before differentiating and
    # record the output structure from the single call mx.jvp makes internally.
    structure_holder = [None]
    if has_aux:
        aux_holder = [None]

        def target(*args):
            out, aux = fun(*args)
            aux_holder[0] = aux
            structure_holder[0] = out
            return out
    else:

        def target(*args):
            out = fun(*args)
            structure_holder[0] = out
            return out

    primals_out, tangents_out = mx.jvp(target, primals, tangents)

    structure = structure_holder[0]
    primals_out = tree.pack_sequence_as(structure, primals_out)
    tangents_out = tree.pack_sequence_as(structure, tangents_out)

    if has_aux:
        return primals_out, tangents_out, aux_holder[0]
    return primals_out, tangents_out
