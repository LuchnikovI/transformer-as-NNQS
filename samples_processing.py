import tensorflow as tf


@tf.function
def _flip(samples, h):
    """Returns samples "passed" through a Hamiltonian
    
    Args:
        samples: int valued tensor of shape (batch_size, num_of_spins)
        h: list that contains two tensors, first tensor is int valued tensor
            of shape (num_of_local_terms, num_of_spins), it shows placement of
            sigma matrices in a local term, second tensor is a float tensor
            of shape (num_of_local_terms,) that shows amplitudes of local
            terms

    Returns:
        int valued tensor of shape (batch_size, num_of_local_terms, num_of_spins)
        """

    flip_flag = tf.logical_and(h[0] > 0, h[0] < 3)
    return tf.cast(tf.math.logical_xor((samples[..., tf.newaxis, :] == 1),
                                       flip_flag), dtype=samples.dtype)


@tf.function
def _ampl(samples, h):
    """Returns amplitudes after "passing" them through a Hamiltonian
    
    Args:
        samples: int valued tensor of shape (batch_size, num_of_spins)
        h: list that contains two tensors, first tensor is int valued tensor
            of shape (num_of_local_terms, num_of_spins), it shows placement of
            sigma matrices in a local term, second tensor is float tensor
            of shape (num_of_local_terms,) that shows amplitudes of local
            terms

    Returns:
        complex valued tensor of shape (batch_size, num_of_local_terms)
        """
    
    samples_shape = tf.shape(samples)
    batch_size, num_of_spins = samples_shape[0], samples_shape[1]
    num_of_local_terms = tf.shape(h[0])[0]

    Id = tf.constant([[1, 1]], dtype=tf.complex64)
    x = Id
    y = tf.constant([[-1j, 1j]], dtype=tf.complex64)
    z = tf.constant([[1, -1]], dtype=tf.complex64)
    T = tf.concat([Id, x, y, z], axis=0)

    ext_samples = tf.tile(samples[:, tf.newaxis, :, tf.newaxis],
                          (1, num_of_local_terms, 1, 1))
    ext_h = tf.tile(h[0][tf.newaxis, :, :, tf.newaxis], (batch_size, 1, 1, 1))
    inds = tf.reshape(tf.concat([ext_h, ext_samples], axis=-1), (-1, 2))
    a = tf.cast(h[1], dtype=tf.complex64)
    return a * tf.reduce_prod(tf.reshape(tf.gather_nd(T, inds),
                                         (batch_size,
                                          num_of_local_terms,
                                          num_of_spins)), axis=-1)


@tf.function
def _hamiltonian(samples, h):
    """Returns matrix elements of a hamiltonian and samples,
    after "passing" samples throung a hamiltonian

    Args:
        samples: int valued tensor of shape (batch_size, num_of_spins)
        h: list that contains two tensors, first tensor is int valued tensor
            of shape (num_of_local_terms, num_of_spins), it shows placement of
            sigma matrices in a local term, second tensor is float tensor
            of shape (num_of_local_terms,) that shows amplitudes of local
            terms

    Returns:
        two tensors, first tensor is int valued tensor of shape
        (batch_size, num_of_local_terms, num_of_spins), second tensor
        is complex valued tensor of shape (batch_size, num_of_local_terms)"""

    return _flip(samples, h), _ampl(samples, h)


@tf.function
def _tf_ravel_multi_index(multi_index, dims):
    """Inverce to tf.unravel_index"""

    strides = tf.math.cumprod(dims, exclusive=True, reverse=True)
    return tf.reduce_sum(multi_index * strides, axis=-1)


@tf.function
def _tf_unravel_index(index, dims):
    """Adapted version of tf.unravel_index"""

    index_shape = tf.shape(index)
    multi_index = tf.transpose(tf.unravel_index(tf.reshape(index, (-1,)), dims))
    multi_index_shape = tf.shape(multi_index)
    new_shape = tf.concat([index_shape, multi_index_shape[-1:]], axis=0)
    multi_index = tf.reshape(multi_index, new_shape)
    return multi_index

@tf.function
def hamiltonian(samples, h, separation):
    """Returns matrix elements of hamiltonian and samples,
    after "passing" samples throung a hamiltonian

    Args:
        samples: int valued tensor of shape (batch_size, num_of_spins)
        h: list that contains two tensors, first tensor is int valued tensor
            of shape (num_of_local_terms, num_of_spins), it shows placement of
            sigma matrices in a local term, second tensor is float tensor
            of shape (num_of_local_terms,) that shows amplitudes of local
            terms
        separation: list with two int numbers showing how we split a system
            into subsystems, e.g. (10, 4) says that we split a system
            consisting of 40 spins into 10 subsystems with 4 spins in each
            subsystem

    Returns:
        two tensors, first tensor is int valued tensor of shape
        (batch_size, num_of_local_terms, num_of_spins), second tensor
        is complex valued tensor of shape (batch_size, num_of_local_terms)"""

    shape = tf.shape(samples)[:-1]
    n = separation[0] * separation[1]  # total number of spins
    unravel_shape = tf.concat([shape, tf.constant([n])], axis=0)
    unravel_samples = _tf_unravel_index(samples,
                                        tf.constant(separation[1] * [2]))
    unravel_samples = tf.reshape(unravel_samples, unravel_shape)
    flipped_samples, ampls = _hamiltonian(unravel_samples, h)
    ravel_samples = _tf_ravel_multi_index(tf.reshape(flipped_samples,
    tf.concat([tf.shape(flipped_samples)[:-1], tf.constant(separation)], axis=0)),
    tf.constant(separation[1] * [2]))
    return ravel_samples, ampls
