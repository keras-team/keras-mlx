import contextlib

import mlx.core as mx

from keras.src import tree
from keras.src.backend.common import stateless_scope
from keras_mlx.src.ops.core import convert_to_tensor
from keras_mlx.src.ops.core import reverse_sequence
from keras_mlx.src.ops.core import scan
from keras_mlx.src.ops.core import unstack


def rnn(
    step_function,
    inputs,
    initial_states,
    go_backwards=False,
    mask=None,
    constants=None,
    unroll=False,
    input_length=None,
    time_major=False,
    zero_output_for_mask=False,
    return_all_outputs=True,
):
    def swap_batch_timestep(input_t):
        # Swap the batch and timestep dim for the incoming tensor.
        axes = list(range(len(input_t.shape)))
        axes[0], axes[1] = 1, 0
        return mx.transpose(input_t, axes)

    if not time_major:
        inputs = tree.map_structure(swap_batch_timestep, inputs)

    flattened_inputs = tree.flatten(inputs)
    time_steps = flattened_inputs[0].shape[0]

    if mask is not None:
        if mask.dtype != mx.bool_:
            mask = mask.astype(mx.bool_)
        if len(mask.shape) == 2:
            mask = mx.expand_dims(mask, axis=-1)
        if not time_major:
            mask = swap_batch_timestep(mask)

    if constants is None:
        constants = []

    def _expand_mask(mask_t, input_t, fixed_dim=1):
        if tree.is_nested(mask_t):
            raise ValueError(
                f"mask_t is expected to be tensor, but got {mask_t}"
            )
        if tree.is_nested(input_t):
            raise ValueError(
                f"input_t is expected to be tensor, but got {input_t}"
            )
        rank_diff = len(input_t.shape) - len(mask_t.shape)
        for _ in range(rank_diff):
            mask_t = mx.expand_dims(mask_t, axis=-1)
        multiples = [1] * fixed_dim + list(input_t.shape[fixed_dim:])
        return mx.tile(mask_t, multiples)

    if unroll:
        if not time_steps:
            raise ValueError("Unrolling requires a fixed number of timesteps.")
        states = tuple(initial_states)
        successive_states = []
        successive_outputs = []

        # Process the input tensors. The input tensor need to be split on the
        # time_step dim, and reverse if go_backwards is True. In the case of
        # nested input, the input is flattened and then transformed
        # individually.  The result of this will be a tuple of lists, each of
        # the item in tuple is list of the tensor with shape (batch, feature)
        def _process_single_input_t(input_t):
            input_t = unstack(input_t)  # unstack for time_step dim
            if go_backwards:
                input_t.reverse()
            return input_t

        if tree.is_nested(inputs):
            processed_input = tree.map_structure(
                _process_single_input_t, inputs
            )
        else:
            processed_input = (_process_single_input_t(inputs),)

        def _get_input_tensor(time):
            inp = [t_[time] for t_ in processed_input]
            return tree.pack_sequence_as(inputs, inp)

        if mask is not None:
            mask_list = unstack(mask)
            if go_backwards:
                mask_list.reverse()

            for i in range(time_steps):
                inp = _get_input_tensor(i)
                mask_t = mask_list[i]
                output, new_states = step_function(
                    inp, tuple(states) + tuple(constants)
                )
                tiled_mask_t = _expand_mask(mask_t, output)

                if not successive_outputs:
                    prev_output = mx.zeros_like(output)
                else:
                    prev_output = successive_outputs[-1]

                output = mx.where(tiled_mask_t, output, prev_output)

                flat_states = tree.flatten(states)
                flat_new_states = tree.flatten(new_states)
                tiled_mask_t = tuple(
                    _expand_mask(mask_t, s) for s in flat_states
                )
                flat_final_states = tuple(
                    mx.where(m, s, ps)
                    for m, s, ps in zip(
                        tiled_mask_t, flat_new_states, flat_states
                    )
                )
                states = tree.pack_sequence_as(states, flat_final_states)

                if return_all_outputs:
                    successive_outputs.append(output)
                    successive_states.append(states)
                else:
                    successive_outputs = [output]
                    successive_states = [states]
            last_output = successive_outputs[-1]
            new_states = successive_states[-1]
            outputs = mx.stack(successive_outputs)

        else:  # mask is None
            for i in range(time_steps):
                inp = _get_input_tensor(i)
                output, states = step_function(
                    inp, tuple(states) + tuple(constants)
                )
                if return_all_outputs:
                    successive_outputs.append(output)
                    successive_states.append(states)
                else:
                    successive_outputs = [output]
                    successive_states = [states]
            last_output = successive_outputs[-1]
            new_states = successive_states[-1]
            outputs = mx.stack(successive_outputs)

    else:  # Unroll == False
        if mask is not None:

            def _step(states, current_input):
                current_input, current_mask = current_input
                is_masked = mx.all(
                    mx.logical_not(current_mask), axis=-1, keepdims=True
                )

                output_t, new_states = step_function(current_input, states)

                if zero_output_for_mask:
                    masked_outs = mx.where(
                        is_masked, mx.zeros_like(output_t), output_t
                    )
                else:
                    # Assume the first state is the previous output.
                    output_tm1 = states[0]
                    if tree.is_nested(output_tm1):
                        # Stacked RNN case: assume first state of last cell.
                        output_tm1 = states[-1][0]
                    masked_outs = mx.where(is_masked, output_tm1, output_t)

                new_states = tree.map_structure(
                    lambda s, ns: mx.where(is_masked, s, ns),
                    states,
                    new_states,
                )
                return (new_states, masked_outs)

            scan_xs = (inputs, mask)

        else:

            def _step(states, current_input):
                output_t, new_states = step_function(current_input, states)
                return new_states, output_t

            scan_xs = inputs
        # Run scan inside a stateless scope so a variable update by
        # step_function during the loop does not leak into the model state.
        if stateless_scope.in_stateless_scope():
            # Reuse the existing parent stateless scope.
            scope = contextlib.nullcontext()
        else:
            scope = stateless_scope.StatelessScope()
        with scope:
            new_states, outputs = scan(
                f=_step,
                init=initial_states,
                xs=scan_xs,
                reverse=go_backwards,
            )

        if go_backwards:
            outputs = reverse_sequence(outputs)

        last_output = outputs[-1]

    if not time_major:
        outputs = tree.map_structure(swap_batch_timestep, outputs)

    return last_output, outputs, new_states


def bidirectional_lstm(*args, **kwargs):
    raise NotImplementedError


def bidirectional_gru(*args, **kwargs):
    raise NotImplementedError


def cudnn_ok(*args, **kwargs):
    return False


def lstm(
    inputs,
    initial_state_h,
    initial_state_c,
    mask,
    kernel,
    recurrent_kernel,
    bias,
    activation,
    recurrent_activation,
    return_sequences=False,
    go_backwards=False,
    unroll=False,
):
    # Masking needs zero_output_for_mask, which only the generic loop has.
    if mask is not None:
        raise NotImplementedError

    kernel = convert_to_tensor(kernel)
    recurrent_kernel = convert_to_tensor(recurrent_kernel)
    inputs = convert_to_tensor(inputs, dtype=kernel.dtype)
    h = convert_to_tensor(initial_state_h, dtype=kernel.dtype)
    c = convert_to_tensor(initial_state_c, dtype=kernel.dtype)
    inputs = mx.swapaxes(inputs, 0, 1)
    if go_backwards:
        inputs = inputs[::-1]

    # Project every timestep through the kernel in a single matmul, so the
    # loop only carries the recurrent one.
    x = mx.matmul(inputs, kernel)
    if bias is not None:
        x = x + convert_to_tensor(bias)

    outputs = []
    for x_t in x:
        z = x_t + mx.matmul(h, recurrent_kernel)
        z_i, z_f, z_c, z_o = mx.split(z, 4, axis=-1)
        i = recurrent_activation(z_i)
        f = recurrent_activation(z_f)
        c = f * c + i * activation(z_c)
        o = recurrent_activation(z_o)
        h = o * activation(c)
        outputs.append(h)

    if return_sequences:
        outputs = mx.stack(outputs, axis=1)
    else:
        outputs = mx.expand_dims(h, axis=1)
    return h, outputs, [h, c]


def gru(
    inputs,
    initial_state,
    mask,
    kernel,
    recurrent_kernel,
    bias,
    activation,
    recurrent_activation,
    return_sequences=False,
    go_backwards=False,
    unroll=False,
    reset_after=True,
):
    # Masking needs zero_output_for_mask, which only the generic loop has,
    # and reset_after=False splits the recurrent kernel per gate.
    if mask is not None or not reset_after:
        raise NotImplementedError

    kernel = convert_to_tensor(kernel)
    recurrent_kernel = convert_to_tensor(recurrent_kernel)
    inputs = convert_to_tensor(inputs, dtype=kernel.dtype)
    h = convert_to_tensor(initial_state, dtype=kernel.dtype)
    inputs = mx.swapaxes(inputs, 0, 1)
    if go_backwards:
        inputs = inputs[::-1]

    x = mx.matmul(inputs, kernel)
    recurrent_bias = None
    if bias is not None:
        bias = convert_to_tensor(bias)
        x = x + bias[0]
        recurrent_bias = bias[1]

    outputs = []
    for x_t in x:
        inner = mx.matmul(h, recurrent_kernel)
        if recurrent_bias is not None:
            inner = inner + recurrent_bias
        x_z, x_r, x_h = mx.split(x_t, 3, axis=-1)
        h_z, h_r, h_h = mx.split(inner, 3, axis=-1)
        z = recurrent_activation(x_z + h_z)
        r = recurrent_activation(x_r + h_r)
        hh = activation(x_h + r * h_h)
        h = z * h + (1 - z) * hh
        outputs.append(h)

    if return_sequences:
        outputs = mx.stack(outputs, axis=1)
    else:
        outputs = mx.expand_dims(h, axis=1)
    return h, outputs, [h]
