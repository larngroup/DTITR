# -*- coding: utf-8 -*-
"""
@author: NelsonRCM
"""

import tensorflow as tf


def linear_proj_matrix(dim_k):
    """
    Dense layer with a certain projection value

    Args:
    - dim_k [int]: projection value

    """

    dense_proj_layer = tf.keras.layers.Dense(dim_k)
    return dense_proj_layer


class LinearAttentionHead(tf.keras.layers.Layer):
    """
    Linear Attention Single Head Layer

    Args:
    - E: Keys projection Matrix
    - F: Values projection matrix
    - dropout_rate [float]: % of dropout

    """
    def __init__(self, dropout, E_proj, F_proj):
        super(LinearAttentionHead, self).__init__()
        self.E = E_proj
        self.F = F_proj
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.transpose = tf.keras.layers.Permute((2, 1))

    def call(self, inputs, mask=None):
        """

        Args:
        - inputs: [Query, Key, Value]
        - mask: attention weights mask

        Shape:
        - Inputs:
        - Query: (B,L_q,D): where B is the batch size, L_q is the query sequence length,
                        D is the head dimension
        - Key: (B,L_k,D): where B is the batch size, L_k is the key sequence length,
                        D is the head dimension
        - Value: (B,L_v,D): where B is the batch size, L_v is the value sequence length,
                        D is the head dimension
        - mask: (B,1,1,L_k): where B is the batch size, L_k is the input sequence length

        - Outputs:
        - attention_output: (B,L_q,E): where B is the batch size, L_q is the query sequence length,
                        E is the embedding dimension

        - attention_weights: (B,L_q,L_k): where B is the batch size,
                        L_q is the query sequence length, L_k is the key sequence length

        """
        Q = inputs[0]
        K = inputs[1]
        V = inputs[2]

        if mask is not None:
            mask = tf.expand_dims(tf.squeeze(mask, axis=(1, 2)), axis=-1)
            K = tf.where(mask == True, 0.0, K)
            V = tf.where(mask == True, 0.0, V)

        dim_k = tf.cast(K.shape[-1], tf.float32)
        scale = 1 / tf.sqrt(dim_k)

        K = self.transpose(K)
        K = self.E(K)

        V = self.transpose(V)
        V = self.F(V)
        V = self.transpose(V)
        Q = tf.matmul(Q, K)

        scaled_attention_scores = Q * scale

        attention_weights = tf.nn.softmax(scaled_attention_scores, axis=-1)
        attention_weights = self.dropout(attention_weights)

        attention_output = tf.matmul(attention_weights, V)

        return attention_output, attention_weights

    def get_config(self):
        config = super(LinearAttentionHead, self).get_config()
        config.update({
            'E_proj': self.E_proj,
            'F_proj': self.F_proj})
        return config


class LMHAttention(tf.keras.layers.Layer):
    """
    Linear Multi-Head Attention Layer

    Args:
    - d_model [int]: embedding dimension
    - num_heads [int]: number of heads of attention
    - dropout_rate [float]: % of dropout
    - parameter_sharing: parameter sharing option
    - dim_k [int]: projection dimension
    - E: Keys projection Matrix
    - F: Values projection matrix

    """
    def __init__(self, d_model, num_heads, dropout_rate, parameter_sharing, dim_k, E_proj, F_proj, **kwargs):
        super(LMHAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.parameter_sharing = parameter_sharing
        self.dim_k = dim_k
        self.E_proj = E_proj
        self.F_proj = F_proj

    def build(self, input_shape):
        self.heads = []
        self.to_q = []
        self.to_k = []
        self.to_v = []

        if self.parameter_sharing != "layerwise":
            self.E_proj = linear_proj_matrix(self.dim_k)
            self.F_proj = linear_proj_matrix(
                self.dim_k) if self.parameter_sharing == "none" or self.parameter_sharing == "headwise" else self.E_proj

        for _ in range(self.num_heads):
            if self.parameter_sharing == "none":
                self.E_proj = linear_proj_matrix(self.dim_k)
                self.F_proj = linear_proj_matrix(self.dim_k)
            attn = LinearAttentionHead(self.dropout_rate, self.E_proj, self.F_proj)
            self.heads.append(attn)
            self.to_q.append(tf.keras.layers.Dense(units=self.d_model / self.num_heads))
            self.to_k.append(tf.keras.layers.Dense(units=self.d_model / self.num_heads))
            self.to_v.append(tf.keras.layers.Dense(units=self.d_model / self.num_heads))

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
        - Value: (B,L_v,E): where B is the batch size, L_v is the value sequence length,
                        E is the embedding dimension
        - mask: (B,1,1,L_k): where B is the batch size, L_k is the key sequence length

        - Outputs:
        - mha_out: (B,L_q,E): where B is the batch size, L_q is the query sequence length,
                        E is the embedding dimension

        - mha_weights: (B,H,L_q,L_k): where B is the batch size, H is the number of heads,
                        L_q is the query sequence length, L_k is the key sequence length

        """


        query = inputs[0]
        key = inputs[1]
        value = inputs[2]

        head_outputs = []
        head_weights = []

        for index, head in enumerate(self.heads):
            Q = self.to_q[index](query)
            K = self.to_k[index](key)
            V = self.to_v[index](value)
            head_outputs.append(head([Q, K, V], mask=mask))

        mha_out = tf.concat([i[0] for i in head_outputs], axis=-1)
        mha_weights = tf.concat([tf.expand_dims(i[1], axis=1) for i in head_outputs], axis=1)
        mha_out = self.dropout_layer(self.out(mha_out))

        return mha_out, mha_weights

    def get_config(self):
        config = super(LMHAttention, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
            'parameter_sharing': self.parameter_sharing,
            'dim_k': self.dim_k,
            'E_proj': self.E_proj,
            'F_proj': self.F_proj})
        return config
