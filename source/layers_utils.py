# -*- coding: utf-8 -*-
"""
@author: NelsonRCM
"""

import tensorflow as tf


class PosWiseFF(tf.keras.layers.Layer):
    """
    Feed-Forward Network (FFN): Position-Wise (Dense layers applied to the last dimension)
    - The first dense layer initially projects the last dimension of the input to
    a higher dimension with a certain expansion ratio
    - The second dense layer projects it back to the initial last dimension

    Args:
    - d_model [int]: embedding dimension
    - d_ff [int]: number of hidden neurons for the first dense layer (expansion ratio)
    - atv_fun: dense layers activation function
    - dropout_rate [float]: % of dropout

    """

    def __init__(self, d_model, d_ff, atv_fun, dropout_rate, **kwargs):
        super(PosWiseFF, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.atv_fun = atv_fun
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.dense_1 = tf.keras.layers.Dense(units=self.d_ff, activation=self.atv_fun)
        self.dense_2 = tf.keras.layers.Dense(units=self.d_model, activation=self.atv_fun)
        self.dropout_layer_1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dropout_layer_2 = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, x):
        """

        Args:
        - x: attention outputs

        Shape:
        - Inputs:
        - x: (B,L,E) where B is the batch size, L is the sequence length, E is the embedding dimension
        - Outputs:
        - x: (B,L,E) where B is the batch size, L is the input sequence length, E is the embedding dimension

        """

        x = self.dense_1(x)
        x = self.dropout_layer_1(x)
        x = self.dense_2(x)
        x = self.dropout_layer_2(x)

        return x

    def get_config(self):
        config = super(PosWiseFF, self).get_config()
        config.update({
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'atv_fun': self.atv_fun,
            'dropout_rate': self.dropout_rate})
        return config


class attn_pad_mask(tf.keras.layers.Layer):
    """
    Attention Padding Mask Layer: Creates the Padding mask for the attention weights

    """

    def __init__(self, **kwargs):
        super(attn_pad_mask, self).__init__(**kwargs)

        self.lambda_layer = tf.keras.layers.Lambda(lambda x: tf.cast(tf.equal(x, 0), dtype=tf.float32))
        self.reshape_layer = tf.keras.layers.Reshape([1, 1, -1])

    def call(self, x):
        """

        Args:
        - x: input sequences

        Shape:
        - Inputs:
        - x: (B,L) where B is the batch size, L is the sequence length
        - Outputs:
        - x: (B,1,1,L) where B is the batch size, L is the input sequence length

        """

        x = self.lambda_layer(self.reshape_layer(x))

        return x


def add_reg_token(x, voc_size):
    """
    Rp and Rs Tokens Function: adds the Rp or the Rs token to the input sequences

    Args:
    - x: inputs sequences
    - voc_size [int]: number of unique tokens

    Shape:
    - Inputs:
    - x: (B,L) where B is the batch size, L is the sequence length
    - Outputs:
    - x: (B,1+L) where B is the batch size, L is the input sequence length

    """

    reg_token = tf.convert_to_tensor(voc_size + 1, dtype=tf.int32)
    broadcast_shape = tf.where([True, False], tf.shape(x), [0, 1])
    reg_token = tf.broadcast_to(reg_token, broadcast_shape)

    return tf.concat([reg_token, x], axis=1)
