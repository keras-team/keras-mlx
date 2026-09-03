import math

import mlx.core as mx

from keras.src.backend.config import floatx
from keras.src.random.seed_generator import SeedGenerator  # noqa: F401
from keras.src.random.seed_generator import draw_seed
from keras.src.random.seed_generator import make_default_seed  # noqa: F401
from keras_mlx.src.ops.core import convert_to_tensor
from keras_mlx.src.ops.core import to_mlx_dtype
from keras_mlx.src.ops.numpy import _nan_scalar

# Rejection rounds for the Marsaglia and Tsang gamma sampler. Acceptance
# is worst at alpha = 1, where it is about 0.952 per round, so 24 rounds
# leave a rejection probability near 1e-31.
GAMMA_ROUNDS = 24


def mlx_draw_seed(seed):
    if isinstance(seed, mx.array):
        return seed
    else:
        return draw_seed(seed)


def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    dtype = to_mlx_dtype(dtype)
    seed = mlx_draw_seed(seed)
    # float64 sampling is not supported on the GPU stream.
    stream = mx.cpu if dtype == mx.float64 else mx.default_device()
    with mx.stream(stream):
        return mx.random.normal(
            shape=shape, loc=mean, scale=stddev, dtype=dtype, key=seed
        )


def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    dtype = to_mlx_dtype(dtype)
    seed = mlx_draw_seed(seed)
    return mx.random.uniform(
        low=minval, high=maxval, shape=shape, dtype=dtype, key=seed
    )


def categorical(logits, num_samples, dtype="int32", seed=None):
    logits = convert_to_tensor(logits)
    seed = mlx_draw_seed(seed)
    output = mx.random.categorical(logits, num_samples=num_samples, key=seed)
    return output.astype(to_mlx_dtype(dtype))


def randint(shape, minval, maxval, dtype="int32", seed=None):
    seed = mlx_draw_seed(seed)
    dtype = to_mlx_dtype(dtype)

    return mx.random.randint(
        low=minval, high=maxval, shape=shape, dtype=dtype, key=seed
    )


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    dtype = to_mlx_dtype(dtype)
    seed = mlx_draw_seed(seed)
    sample = mx.random.truncated_normal(
        lower=-2.0, upper=2.0, shape=shape, dtype=dtype, key=seed
    )
    return sample * stddev + mean


def _get_concrete_noise_shape(inputs, noise_shape):
    if noise_shape is None:
        return inputs.shape

    concrete_inputs_shape = inputs.shape
    concrete_noise_shape = []
    for i, value in enumerate(noise_shape):
        concrete_noise_shape.append(
            concrete_inputs_shape[i] if value is None else value
        )
    return concrete_noise_shape


def dropout(inputs, rate, noise_shape=None, seed=None):
    if rate == 1.0:
        return mx.zeros_like(inputs)
    if rate == 0.0:
        return inputs
    seed = mlx_draw_seed(seed)
    keep_prob = 1.0 - rate
    # The `noise_shape` may contain `None` so we need to convert it
    # into a concrete shape before passing it on to mlx.
    noise_shape = _get_concrete_noise_shape(inputs, noise_shape)
    mask = mx.random.bernoulli(p=keep_prob, shape=noise_shape, key=seed)
    mask = mx.broadcast_to(mask, inputs.shape)
    return mx.where(mask, inputs / keep_prob, mx.zeros_like(inputs))


def shuffle(x, axis=0, seed=None):
    seed = mlx_draw_seed(seed)
    x = convert_to_tensor(x)
    return mx.random.permutation(x, axis=axis, key=seed)


def gamma(shape, alpha, dtype=None, seed=None):
    # Ref: jax.random.gamma
    # Ref: Marsaglia and Tsang method for generating gamma variables
    # Algorithm description can be found here:
    # https://en.wikipedia.org/wiki/Gamma_distribution#Random_variate_generation
    if isinstance(shape, int):
        shape = (shape,)

    dtype = to_mlx_dtype(dtype or floatx())

    # The sampler below is elementwise, so broadcast alpha over the lanes and
    # an array alpha costs no more than a scalar one. Flatten a single
    # element first, its rank may exceed the output rank.
    alpha = mx.array(alpha).astype(mx.float32)
    if alpha.size == 1:
        alpha = alpha.reshape(())
    alpha = mx.broadcast_to(alpha, shape)
    # Split the boost key up front. The loop can stop early, so a key taken
    # after it would depend on how many rounds ran.
    key, boost_key = mx.random.split(mlx_draw_seed(seed), 2)

    # Gamma(alpha) = Gamma(alpha + 1) * U ** (1 / alpha) below 1, so sample
    # the boosted shape and correct after. Selected with `where` rather than
    # branched on, so alpha may be traced.
    below_one = alpha < 1.0
    boosted = mx.where(below_one, alpha + 1.0, alpha)

    d = boosted - 1.0 / 3.0
    c = 1.0 / mx.sqrt(9.0 * d)
    done = mx.zeros(shape, dtype=mx.bool_)
    results = mx.zeros(shape, dtype=mx.float32)
    # A fixed round count keeps the rejection loop a static graph instead of
    # a data dependent `while`, which cannot be traced. See GAMMA_ROUNDS.
    for _ in range(GAMMA_ROUNDS):
        key, key_x, key_u = mx.random.split(key, 3)

        x = mx.random.normal(key=key_x, shape=shape)
        u = mx.random.uniform(key=key_u, shape=shape)

        # A non positive v_ would make log(v) undefined, so mask those lanes
        # out and keep a harmless value in them.
        v_ = 1.0 + c * x
        usable = mx.logical_and(v_ > 0.0, mx.logical_not(done))
        v = mx.where(usable, v_, 1.0) ** 3

        # log(u) < 0.5 * x^2 + d * (1 - v + log(v))
        accept = mx.logical_and(
            usable, mx.log(u) < 0.5 * x * x + d * (1.0 - v + mx.log(v))
        )
        results = mx.where(accept, d * v, results)
        done = mx.logical_or(done, accept)

        # Stop once every lane has a sample. Reading `done` frees the
        # round's intermediates, which is what keeps a large `shape` off the
        # Metal buffer limit. While tracing the read is illegal, so all
        # GAMMA_ROUNDS run instead.
        try:
            if bool(mx.all(done)):
                break
        except ValueError:
            pass

    # d sits near the centre of the boosted distribution, a harmless stand in
    # for the vanishingly rare lanes that never accepted.
    results = mx.where(done, results, d)

    # The correction is a no op once alpha reaches 1, so skip the extra draw
    # when alpha is concrete. While tracing the value cannot be read, so it
    # is built and selected with `where` instead.
    try:
        needs_correction = bool(mx.any(below_one))
    except ValueError:
        needs_correction = True
    if not needs_correction:
        return results.astype(dtype)

    u = mx.random.uniform(key=boost_key, shape=shape)
    correction = mx.where(below_one, u ** (1.0 / alpha), 1.0)
    # A non positive alpha has no gamma distribution. jax, torch and
    # tensorflow return nan there rather than raising, so match them.
    results = mx.where(
        alpha > 0.0, results * correction, _nan_scalar(results.dtype)
    )
    return results.astype(dtype)


def beta(shape, alpha, beta, dtype=None, seed=None):
    # beta distribution using Gamma(alpha) / (Gamma(alpha) + Gamma(beta))
    dtype = to_mlx_dtype(dtype or floatx())

    if isinstance(shape, int):
        shape = (shape,)

    # No range check, reading the values is illegal while tracing and jax,
    # torch and tensorflow do not validate either.
    alpha_arr = mx.array(alpha, dtype=mx.float32)
    beta_arr = mx.array(beta, dtype=mx.float32)

    key = mlx_draw_seed(seed)
    key_x, key_y = mx.random.split(key, 2)
    x = gamma(shape, alpha_arr, dtype=dtype, seed=key_x)
    y = gamma(shape, beta_arr, dtype=dtype, seed=key_y)
    # Both draws can underflow to zero for a small alpha, and 0/0 is nan,
    # which is outside the support. jax and numpy return zero there.
    total = x + y
    nonzero = total > 0
    ratio = x / mx.where(nonzero, total, 1)
    return mx.where(nonzero, ratio, 0).astype(dtype)


def binomial(shape, counts, probabilities, dtype=None, seed=None):
    # Binomial(n, p) distribution by summing n Bernoulli(p) samples
    dtype = to_mlx_dtype(dtype or floatx())
    key = mlx_draw_seed(seed)

    if isinstance(shape, int):
        shape = (shape,)

    # counts will be handled as ints below
    counts_arr = mx.array(counts, dtype=mx.float32)
    probs_arr = mx.array(probabilities, dtype=mx.float32)

    if mx.any(counts_arr < 0.0):
        raise ValueError(
            "Invalid value for argument `counts`. All counts "
            f"must be >= 0, received counts={counts}"
        )
    if mx.any(probs_arr < 0.0) or mx.any(probs_arr > 1.0):
        raise ValueError(
            "Invalid value for argument `probabilities`. "
            "All probabilities must be in [0, 1], received "
            f"probabilities={probabilities}"
        )

    # Fast path for the common scalar case: draw n Bernoulli samples per
    # output element in one vectorized call instead of a Python loop.
    if counts_arr.size == 1 and probs_arr.size == 1:
        n = int(counts_arr.item())
        if n == 0:
            return mx.zeros(shape, dtype=dtype)
        draws = mx.random.bernoulli(
            p=float(probs_arr.item()), shape=(n,) + tuple(shape), key=key
        )
        return mx.sum(draws, axis=0).astype(dtype)

    # broadcast counts and probs to `shape``
    zeros_for_bcast = mx.zeros(shape=shape, dtype=mx.float32)
    counts_bcast = counts_arr + zeros_for_bcast
    probs_bcast = probs_arr + zeros_for_bcast

    flat_size = math.prod(shape)

    counts_flat = counts_bcast.reshape((flat_size,))
    probs_flat = probs_bcast.reshape((flat_size,))
    out_flat = mx.zeros((flat_size,), dtype=dtype)

    # for each element in flattened arrays
    # draw a single Binomial(n_i, p_i) sample by summing n_i Bernoulli draws
    carry_key = key
    for i in range(flat_size):
        n_i = counts_flat[i].astype(mx.int32).item()
        p_i = probs_flat[i].item()

        if n_i == 0:
            out_flat[i] = 0
            continue

        carry_key, subkey = mx.random.split(carry_key)
        bernoulli_samples = mx.random.bernoulli(key=subkey, shape=(n_i,), p=p_i)
        binomial_val = mx.sum(bernoulli_samples, axis=0)
        out_flat[i] = binomial_val

    out = out_flat.reshape(shape)
    return out.astype(dtype)
