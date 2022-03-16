# -*- coding: utf-8 -*-
"""
@author: NelsonRCM
"""

import tensorflow as tf


class EmbeddingLayer(tf.keras.layers.Layer):
    """
    Embedding Layer: generates a learned embedding to every token with a fixed size

    Args:
    - voc_size [int]: number of unique tokens
    - d_model [int]: embedding dimension
    - dropout_rate [float]: % of dropout
    - positional_enc [boolean]: positional encoding option: if true adds a positional embedding
    to the output of the embedding layer

    """

    def __init__(self, voc_size, d_model, dropout_rate, positional_enc=True, **kwargs):
        super(EmbeddingLayer, self).__init__(**kwargs)

        self.voc_size = voc_size
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.positional_enc = positional_enc

    def build(self, input_shape):
        self.emb_layer = tf.keras.layers.Embedding(self.voc_size, self.d_model)
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)

    def position_embedding(self, max_len):
        """
        Positional Embedding: Adds info about the position of each token using sin and cosine functions

        Args:
        - max_len [int]: number of input tokens (length)

        Shape:
        - Outputs:
        - pos_enc: (L,E) where L is the input sequence length, E is the embedding dimension

        """

        angle = tf.range(self.d_model, dtype=tf.float32)
        angle = 10000 ** (2 * (angle / self.d_model))

        angle = tf.expand_dims(tf.range(max_len, dtype=tf.float32), 1) / angle

        # for i in range(angle.shape[0]):
        #     for j in range(angle.shape[1]):
        #         if j % 2 == 0 :
        #             angle = tf.tensor_scatter_nd_update(angle,[[i,j]],[tf.math.sin(angle[i,j])])
        #         else :
        #             angle = tf.tensor_scatter_nd_update(angle,[[i,j]],[tf.math.cos(angle[i,j])])

        # return tf.cast(angle,dtype=tf.float32)

        values = tf.stack([tf.math.sin(angle[:, 0::2]), tf.math.cos(angle[:, 1::2])], axis=2)

        pos_enc = tf.reshape(values, shape=[tf.shape(values)[0], -1])

        return tf.cast(pos_enc, dtype=tf.float32)

    def call(self, sequences):
        """

        Args:
        - sequences: input sequences

        Shape:
        - Inputs:
        - sequences: (B,L) where B is the batch size, L is the sequence length
        - Outputs:
        - output: (B,L,E) where B is the batch size, L is the input sequence length,
                        E is the embedding dimension

        """

        max_len = sequences.shape[1]

        output = self.emb_layer(sequences) * tf.sqrt(tf.cast(self.d_model, dtype=tf.float32))

        if self.positional_enc:  # Add Positional Info
            output = output + self.position_embedding(max_len)
            output = self.dropout_layer(output)

        return output

    def get_config(self):
        config = super(EmbeddingLayer, self).get_config()
        config.update({
            'voc_size': self.voc_size,
            'd_model': self.d_model,
            'dropout_rate': self.dropout_rate,
            'positional_enc': self.positional_enc})

        return config
