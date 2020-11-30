import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_steps, max_dims, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        p = tf.range(max_steps, dtype=dtype)
        i = tf.range(int(max_dims / 2), dtype=dtype)
        phi = p[tf.newaxis] / 10000 ** (2 * i[:, tf.newaxis] / max_dims)
        pos_enc = tf.concat([tf.math.sin(phi), tf.math.cos(phi)], axis=1)
        self.pos_enc = tf.transpose(tf.reshape(pos_enc, (-1, max_steps)))
    def call(self, inputs):
        shape = tf.shape(inputs)
        return inputs + self.pos_enc[:shape[-2], :shape[-1]]

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, number_of_heads, key_size, **kwargs):
        super().__init__(**kwargs)
        self.number_of_heads = number_of_heads
        self.key_size = key_size

    def build(self, input_shape):
        self.keys = self.add_weight("keys",
                                    shape=(input_shape[-1],
                                           self.key_size,
                                           self.number_of_heads))
        self.querys = self.add_weight("querys",
                                      shape=(input_shape[-1],
                                             self.key_size,
                                             self.number_of_heads))
        self.values = self.add_weight("values",
                                      shape=(input_shape[-1],
                                             self.key_size,
                                             self.number_of_heads))
        self.kernel = self.add_weight("kernel",
                                      shape=(self.number_of_heads * self.key_size,
                                             self.key_size))

    def call(self, input):
        batch_size = tf.shape(input)[-3]
        seq_length = tf.shape(input)[-2]
        # k, q, v have shape=(seq_lenght, key_size, number_of_heads)
        k = tf.tensordot(input, self.keys, axes=1)
        q = tf.tensordot(input, self.querys, axes=1)
        v = tf.tensordot(input, self.values, axes=1)
        # qk^T and weights tensors have
        # shape=(seq_lenght, seq_lenght, number_of_heads)
        qkt = tf.einsum('...ijk,...ljk->...ilk', q, k)
        qkt = qkt / tf.math.sqrt(tf.cast(self.key_size, dtype=tf.float32))
        mask = tf.linalg.band_part(tf.ones((seq_length, seq_length), dtype=tf.float32), 0, -1)
        mask = mask - tf.eye(seq_length, dtype=tf.float32)
        mask = mask[..., tf.newaxis]
        weights = tf.nn.softmax(qkt - 1e8 * mask, axis=-2)
        # heads tensor has shape=(seq_lenght, key_size, number_of_heads)
        heads = tf.einsum('...ijk,...jlk->...ilk', weights, v)
        # attenstion tensor has shape=(seq_lenght, key_size)
        heads = tf.reshape(heads, (batch_size, seq_length, self.key_size*self.number_of_heads))
        return tf.tensordot(heads, self.kernel, axes=1)

def attention_model(local_dim, emb_dim, number_of_heads,
                    number_of_layers, max_steps):
    """Returns attention based autoregressive model.

    Args:
        local_dim: int value, local Hilbert space dim
        emb_dim: int value embedding dim
        number_of_heads: int value, number of attention heads
        number_of_layers: int value, number of layers with attention
        max_steps: int value, maximal number of subsystems

    Returns:
        keras model, attention based autoregressive model
    """
    inp = tf.keras.Input((None,))
    emb = tf.keras.layers.Embedding(local_dim, emb_dim)(inp)
    h = PositionalEncoding(max_steps, emb_dim)(emb)
    for _ in range(number_of_layers):
        h = MultiHeadAttention(number_of_heads, emb_dim)(h) + h
        h = tf.keras.layers.Conv1D(emb_dim, 1, activation='elu')(h) + h
    out = tf.keras.layers.Conv1D(2 * local_dim, 1, activation=None)(h)
    return tf.keras.Model(inputs=inp, outputs=out)
