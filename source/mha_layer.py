# -*- coding: utf-8 -*-
"""
@author: NelsonRCM
"""

import tensorflow as tf


class ScaledDotProductAttention(tf.keras.layers.Layer):
    """
    Scaled Dot-Product Attention Layer

    Args:
    - dropout_rate [float]: % of dropout

    """

    def __init__(self, dropout_rate, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)

        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs, mask=None):
        """

        Args:
        - inputs: [Query, Key, Value]
        - mask: attention weights mask

        Shape:
        - Inputs:
        - Query: (B,H,L_q,D): where B is the batch size, H is the number of heads,
                        L_q is the query sequence length, D is the head dimension
        - Key: (B,H,L_k,D): where B is the batch size, H is the number of heads,
                        L_k is the key sequence length, D is the head dimension
        - Value: (B,H,L_v,D): where B is the batch size, H is the number of heads,
                        L_v is the value sequence length, D is the head dimension
        - mask: (B,1,1,L_k): where B is the batch size, L_k is the key sequence length

        - Outputs:
        - attention_output: (B,H,L_q,D): where B is the batch size, H is the number of heads,
                        L_q is the query sequence length, D is the head dimension

        - attention_weights: (B,H,L_q,L_k): where B is the batch size, H is the number of heads,
                        L_q is the query sequence length, L_k is the key sequence length

        """

        query, key, value = inputs

        dim_k = tf.cast(key.shape[-1], tf.float32)
        scale = 1 / tf.sqrt(dim_k)

        matmul_q_transp_k = tf.matmul(query, key, transpose_b=True)

        scaled_attention_scores = matmul_q_transp_k * scale

        if mask is not None:
            scaled_attention_scores += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_scores, axis=-1)
        attention_weights = self.dropout_layer(attention_weights)

        attention_output = tf.matmul(attention_weights, value)

        return attention_output, attention_weights

    def get_config(self):
        config = super(ScaledDotProductAttention, self).get_config()
        config.update({
            'dropout_rate': self.dropout_rate})
        return config


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-Head Attention Layer

    Args:
    - d_model [int]: embedding dimension
    - num_heads [int]: number of heads of attention
    - dropout_rate [float]: % of dropout

    """


    def __init__(self, d_model, num_heads, dropout_rate, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        assert d_model % num_heads == 0

    def build(self, input_shape):
        self.attention = ScaledDotProductAttention(self.dropout_rate)

        self.query_dense = tf.keras.layers.Dense(units=self.d_model)
        self.key_dense = tf.keras.layers.Dense(units=self.d_model)
        self.value_dense = tf.keras.layers.Dense(units=self.d_model)

        self.reshape = tf.keras.layers.Reshape((-1, self.num_heads, self.d_model // self.num_heads))
        self.transpose = tf.keras.layers.Permute((2, 1, 3))

        self.transpose_attn_output = tf.keras.layers.Permute((2, 1, 3))
        self.reshape_attn_output = tf.keras.layers.Reshape((-1, self.d_model))

        self.out = tf.keras.layers.Dense(units=self.d_model)

        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs, mask=None):
        """

        Args:
        - inputs: [Query, Key, Value]
        - mask: attention weights mask

        Shape:
        - Inputs:
        - Query: (B,L_q,E): where B is the batch size, L_q is the query sequence length,
                        E is the embedding dimension
        - Key: (B,L_k,E): where B is the batch size, L_k is the key sequence length,
                        E is the embedding dimension
        - Value: (B,L_v,E): where B is the batch size, L_q is the value sequence length,
                        E is the embedding dimension
        - mask: (B,1,1,L_k): where B is the batch size, L_k is the key sequence length

        - Outputs:
        - mh_attention_output: (B,L_q,E): where B is the batch size, L_q is the query sequence length,
                        E is the embedding dimension

        - attention_weights: (B,H,L_q,L_k): where B is the batch size, H is the number of heads,
                        L_q is the query sequence length, L_k is the key sequence length

        """

        query = inputs[0]
        key = inputs[1]
        value = inputs[2]

        query = self.query_dense(query)  # [batch_size, seq_len, d_model]
        key = self.key_dense(key)  # [batch_size, seq_len, d_model]
        value = self.value_dense(value)  # [batch_size, seq_len, d_model]

        query = self.transpose(self.reshape(query))
        key = self.transpose(self.reshape(key))
        value = self.transpose(self.reshape(value))

        attention_output, attention_weights = self.attention([query, key, value], mask=mask)
        # # [batch_size, num_heads, seq_len_q, head_dim], # [batch_size, num_heads, seq_len_q, seq_len_k]

        attention_output = self.transpose_attn_output(attention_output)  # [batch_size, seq_len_q, num_heads, head_dim]
        attention_output = self.reshape_attn_output(attention_output)  # [batch_size, seq_len_q, d_model]

        mh_attention_output = self.dropout_layer(self.out(attention_output))  # [batch_size, seq_len_q, d_model]

        return mh_attention_output, attention_weights

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate})
        return config
