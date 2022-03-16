# -*- coding: utf-8 -*-
"""
@author: NelsonRCM
"""

from mha_layer import *
from layers_utils import *
from lmha_layer import *


class CrossAttnLayer(tf.keras.layers.Layer):
    """
    Cross-Attention Encoder Layer

    Args:
    - d_model [int]: embedding dimension
    - cross_num_heads [int]: number of heads for the cross-attention mha
    - x1_num_heads [int]: number of heads for the self-attention mha for the input 1
    - x2_num_heads [int]: number of heads for the self-attention mha for the input 2
    - x1_d_ff [int]: number of hidden neurons for the first dense layer of the FFN for the input 1
    - x2_d_ff [int]: number of hidden neurons for the first dense layer of the FFN for the input 2
    - atv_fun: dense layers activation function
    - dropout_rate [float]: % of dropout
    - x1_dim_k [int]: Linear MHA projection dimension for the input 1
    - x1_parameter_sharing [str]: Linear MHA parameter sharing option for the input 1
    - x1_full_attention [boolean]: True - original O(n2) attention, False - Linear Attention (input 1)
    - x2_dim_k [int]: Linear MHA projection dimension for the input 2
    - x2_parameter_sharing [str]: Linear MHA parameter sharing option for the input 2
    - x2_full_attention [boolean]: True - original O(n2) attention, False - Linear Attention (input 2)

    """

    def __init__(self, d_model, cross_num_heads, x1_num_heads, x2_num_heads,
                 x1_d_ff, x2_d_ff, atv_fun, dropout_rate, x1_dim_k,
                 x1_parameter_sharing, x1_full_attention,
                 x2_dim_k, x2_parameter_sharing, x2_full_attention, **kwargs):
        super(CrossAttnLayer, self).__init__(**kwargs)

        self.d_model = d_model
        self.cross_num_heads = cross_num_heads
        self.x1_num_heads = x1_num_heads
        self.x2_num_heads = x2_num_heads
        self.x1_d_ff = x1_d_ff
        self.x2_d_ff = x2_d_ff
        self.atv_fun = atv_fun
        self.dropout_rate = dropout_rate
        self.x1_dim_k = x1_dim_k
        self.x1_parameter_sharing = x1_parameter_sharing
        self.x1_full_attention = x1_full_attention
        self.x2_dim_k = x2_dim_k
        self.x2_parameter_sharing = x2_parameter_sharing
        self.x2_full_attention = x2_full_attention

    def build(self, input_shape):
        self.mha_layer_1 = MultiHeadAttention(self.d_model, self.cross_num_heads, self.dropout_rate,
                                              name='x12_cross_attn')

        self.mha_layer_2 = MultiHeadAttention(self.d_model, self.cross_num_heads, self.dropout_rate,
                                              name='x21_cross_attn')

        self.ln_1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='x12_norm')

        self.ln_2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='x21_norm')

        self.ln_3 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='sub1_x1_cross_norm')

        self.ln_4 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='sub1_x2_cross_norm')

        self.ln_5 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='sub2_x1_cross_norm')

        self.ln_6 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name='sub2_x2_cross_norm')

        if self.x1_full_attention:
            self.mha_layer_3 = MultiHeadAttention(self.d_model, self.x1_num_heads, self.dropout_rate,
                                                  name='x1_cross_fmhattn')

        else:
            E_proj_x1 = linear_proj_matrix(self.x1_dim_k)

            self.mha_layer_3 = LMHAttention(self.d_model, self.x1_num_heads, self.dropout_rate,
                                            self.x1_parameter_sharing,
                                            self.x1_dim_k,
                                            E_proj_x1, E_proj_x1, name='x1_cross_lmhattn')

        if self.x2_full_attention:
            self.mha_layer_4 = MultiHeadAttention(self.d_model, self.x2_num_heads, self.dropout_rate,
                                                  name='x2_cross_fmhattn')

        else:
            E_proj_x2 = linear_proj_matrix(self.x2_dim_k)

            self.mha_layer_4 = LMHAttention(self.d_model, self.x2_num_heads, self.dropout_rate,
                                            self.x2_parameter_sharing,
                                            self.x2_dim_k,
                                            E_proj_x2, E_proj_x2, name='x2_cross_lmhattn')

        self.poswiseff_layer_1 = PosWiseFF(self.d_model, self.x1_d_ff, self.atv_fun, self.dropout_rate
                                           , name='pos_wise_ff_x1_cross')

        self.poswiseff_layer_2 = PosWiseFF(self.d_model, self.x2_d_ff, self.atv_fun, self.dropout_rate,
                                           name='pos_wise_ff_x2_cross')

    def rearrange_qkv(self, input1, input2):
        """
        Function to extract the first tokens (Rp or Rs) and the remaining tokens (L-1) for the input 1 and input 2

        Args:
        - input1 : input 1 sequences
        - input2: input 2 sequences

        Shape:
        - Inputs:
        - input1 : (B,L_1,E_1) where B is the batch size, L_1 is the sequence length for the input 1,
                                E_1 is the embedding dimension for the input 1
        - input2 : (B,L_2,E_2) where B is the batch size, L_2 is the sequence length for the input 2,
                                E_2 is the embedding dimension for the input 2
        - Outputs:
        - input1_pred_token : (B,1,E_1) where B is the batch size, L_1 is the sequence length for the input 1,
                                E_1 is the embedding dimension for the input 1
        - input1_tokens : (B,L_1-1,E_1) where B is the batch size, L_1 is the sequence length for the input 1,
                                E_1 is the embedding dimension for the input 1
        - input2_pred_token: (B,L_2,E_2) where B is the batch size, L_2 is the sequence length for the input 2,
                                E_2 is the embedding dimension for the input 2
        - input2_tokens: (B,L_2-1,E_2) where B is the batch size, L_2 is the sequence length for the input 2,
                                E_2 is the embedding dimension for the input 2

        """

        input1_pred_token = tf.expand_dims(tf.gather(input1, 0, axis=1), axis=1)
        input1_tokens = tf.gather(input1, tf.range(1, input1.shape[1]), axis=1)

        input2_pred_token = tf.expand_dims(tf.gather(input2, 0, axis=1), axis=1)
        input2_tokens = tf.gather(input2, tf.range(1, input2.shape[1]), axis=1)

        return input1_pred_token, input1_tokens, input2_pred_token, input2_tokens

    def call(self, inputs, mask_x12=None, mask_x21=None):
        """

        Args:
        - inputs: [input1,input2]: input 1 sequence and input 2 sequences
        - mask_x12: attention weights mask for the input 2
        . mask_x21: attention weights mask for the input 1

        Shape:
        - Inputs:
        - inputs1 : (B,L_1,E_1) where B is the batch size, L_1 is the sequence length for the input 1,
                                E_1 is the embedding dimension for the input 1
        - inputs2 : (B,L_2,E_2) where B is the batch size, L_2 is the sequence length for the input 2,
                                E_2 is the embedding dimension for the input 2

        - mask_x12: (B,1,1,L_2): where B is the batch size, L_2 is the sequence length for the input 2
        _ mask_x21: (B,1,1,L_1): where B is the batch size, L_1 is the sequence length for the input 1

        - Outputs:
        - x1_cross: (B,L_1,E_1) where B is the batch size, L_1 is the sequence length for the input 1,
                                E_1 is the embedding dimension for the input 1
        - x2_cross: (B,L_2,E_2) where B is the batch size, L_2 is the sequence length for the input 2,
                                E_2 is the embedding dimension for the input 2
        - attn_x12_w: (B,H,1,L_2): where B is the batch size, H is the number of heads,
                        L_2 is the input 2 sequence length.
        - attn_x21_w: (B,H,1,L_1): where B is the batch size, H is the number of heads,
                        L_1 is the input 1 sequence length.
        - attn_x1_w: (B,H,1,L_1): where B is the batch size, H is the number of heads,
                        L_1 is the input 1 sequence length.
        - attn_x2_w: (B,H,1,L_2): where B is the batch size, H is the number of heads,
                        L_2 is the input 2 sequence length.


        """

        x1_p_t, x1_t, x2_p_t, x2_t = self.rearrange_qkv(inputs[0], inputs[1])

        x12_qkv = tf.concat([x1_p_t, x2_t], axis=1)

        x21_qkv = tf.concat([x2_p_t, x1_t], axis=1)

        attn_x12_out, attn_x12_w = self.mha_layer_1([tf.expand_dims(tf.gather(x12_qkv, 0, axis=1), axis=1),
                                                     x12_qkv, x12_qkv], mask=mask_x12)

        attn_x21_out, attn_x21_w = self.mha_layer_2([tf.expand_dims(tf.gather(x21_qkv, 0, axis=1), axis=1),
                                                     x21_qkv, x21_qkv], mask=mask_x21)

        x1_p_t_cross = self.ln_1(x1_p_t + attn_x12_out)
        x2_p_t_cross = self.ln_2(x2_p_t + attn_x21_out)

        x1_cross = tf.concat([x1_p_t_cross, x1_t], axis=1)
        x2_cross = tf.concat([x2_p_t_cross, x2_t], axis=1)

        if self.x1_full_attention:
            attn_x1_out, attn_x1_w = self.mha_layer_3([x1_cross, x1_cross, x1_cross], mask=mask_x21)

        else:
            attn_x1_out, attn_x1_w = self.mha_layer_3([x1_cross, x1_cross, x1_cross], mask=mask_x21)

        if self.x2_full_attention:
            attn_x2_out, attn_x2_w = self.mha_layer_4([x2_cross, x2_cross, x2_cross], mask=mask_x12)

        else:
            attn_x2_out, attn_x2_w = self.mha_layer_4([x2_cross, x2_cross, x2_cross], mask=mask_x12)

        x1_cross = self.ln_3(x1_cross + attn_x1_out)
        x2_cross = self.ln_4(x2_cross + attn_x2_out)

        x1_cross_posff_out = self.poswiseff_layer_1(x1_cross)
        x2_cross_posff_out = self.poswiseff_layer_2(x2_cross)

        x1_cross = self.ln_5(x1_cross + x1_cross_posff_out)
        x2_cross = self.ln_6(x2_cross + x2_cross_posff_out)

        return [x1_cross, x2_cross], attn_x12_w, attn_x21_w, attn_x1_w, attn_x2_w


def get_config(self):
    config = super(CrossAttnLayer, self).get_config()
    config.update({
        'd_model': self.d_model,
        'cross_num_heads': self.cross_num_heads,
        'x1_num_heads': self.x1_num_heads,
        'x2_num_heads': self.x2_num_heads,
        'x1_d_ff': self.x1_d_ff,
        'x2_d_ff': self.x2_d_ff,
        'atv_fun': self.atv_fun,
        'dropout_rate': self.dropout_rate,
        'x1_dim_k': self.x1_dim_k,
        'x1_parameter_sharing': self.x1_parameter_sharing,
        'x1_full_attention': self.x1_full_attention,
        'x2_dim_k': self.x2_dim_k,
        'x2_parameter_sharing': self.x2_parameter_sharing,
        'x2_full_attention': self.x2_full_attention})

    return config


class CrossAttnBlock(tf.keras.Model):
    """
    Cross-Attention Transformer-Encoder

    Args:
    - d_model [int]: embedding dimension
    - num_layers [int]: number of cross-attention transformer-encoder layers
    - cross_num_heads [int]: number of heads for the cross-attention mha
    - x1_num_heads [int]: number of heads for the self-attention mha for the input 1
    - x2_num_heads [int]: number of heads for the self-attention mha for the input 2
    - x1_d_ff [int]: number of hidden neurons for the first dense layer of the FFN for the input 1
    - x2_d_ff [int]: number of hidden neurons for the first dense layer of the FFN for the input 2
    - atv_fun: dense layers activation function
    - dropout_rate [float]: % of dropout
    - x1_dim_k [int]: Linear MHA projection dimension for the input 1
    - x1_parameter_sharing [str]: Linear MHA parameter sharing option for the input 1
    - x1_full_attention [boolean]: True - original O(n2) attention, False - Linear Attention (input 1)
    - x2_dim_k [int]: Linear MHA projection dimension for the input 2
    - x2_parameter_sharing [str]: Linear MHA parameter sharing option for the input 2
    - x2_full_attention [boolean]: True - original O(n2) attention, False - Linear Attention (input 2)
    - return_intermediate [boolean]: if true returns the intermediate results

    """
    def __init__(self, d_model, num_layers, cross_num_heads, x1_num_heads, x2_num_heads,
                 x1_d_ff, x2_d_ff, atv_fun, dropout_rate, x1_dim_k,
                 x1_parameter_sharing, x1_full_attention,
                 x2_dim_k, x2_parameter_sharing, x2_full_attention,
                 return_intermediate=False, **kwargs):

        super(CrossAttnBlock, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers
        self.cross_num_heads = cross_num_heads
        self.x1_num_heads = x1_num_heads
        self.x2_num_heads = x2_num_heads
        self.x1_d_ff = x1_d_ff
        self.x2_d_ff = x2_d_ff
        self.atv_fun = atv_fun
        self.dropout_rate = dropout_rate
        self.x1_dim_k = x1_dim_k
        self.x1_parameter_sharing = x1_parameter_sharing
        self.x1_full_attention = x1_full_attention
        self.x2_dim_k = x2_dim_k
        self.x2_parameter_sharing = x2_parameter_sharing
        self.x2_full_attention = x2_full_attention
        self.return_intermediate = return_intermediate

    def build(self, input_shape):
        self.cross_attn_layers = [CrossAttnLayer(self.d_model, self.cross_num_heads, self.x1_num_heads,
                                                 self.x2_num_heads,
                                                 self.x1_d_ff, self.x2_d_ff, self.atv_fun,
                                                 self.dropout_rate, self.x1_dim_k,
                                                 self.x1_parameter_sharing,
                                                 self.x1_full_attention, self.x2_dim_k,
                                                 self.x2_parameter_sharing,
                                                 self.x2_full_attention, name='layer_cross_attn%d' % i)
                                  for i in range(self.num_layers)]

    def call(self, inputs, mask_12=None, mask_21=None):
        """

        Args:
        - inputs: [input1,input2]: input 1 sequences and input 2 sequences
        - mask_x12: attention weights mask for the input 2
        . mask_x21: attention weights mask for the input 1

        Shape:
        - Inputs:
        - inputs1 : (B,L_1,E_1) where B is the batch size, L_1 is the sequence length for the input 1,
                                E_1 is the embedding dimension for the input 1
        - inputs2 : (B,L_2,E_2) where B is the batch size, L_2 is the sequence length for the input 2,
                                E_2 is the embedding dimension for the input 2

        - mask_x12: (B,1,1,L_2): where B is the batch size, L_2 is the sequence length for the input 2
        _ mask_x21: (B,1,1,L_1): where B is the batch size, L_1 is the sequence length for the input 1

        - Outputs:
        - x: [(B,L_1,E_1),(B,L_1,E_1)] where B is the batch size, L_1 is the sequence length for the input 1,
                                        E_1 is the embedding dimension for the input 1,
                                        L_2 is the sequence length for the input 2,
                                        E_2 is the embedding dimension for the input 2
        - attention_weights: dictionary with the attentions weights [(B,H,1,L_2),(B,H,1,L_1),(B,H,1,L_1),(B,H,1,L_2)]
                                of each cross-attention encoder layer

        """

        x = inputs
        intermediate = []
        attention_weights = {}

        for layer in self.cross_attn_layers:
            x, x12_attn_w, x21_attn_w, x1_cross_attn_w, x2_cross_attn_w = layer(x, mask_12, mask_21)

            if self.return_intermediate:
                intermediate.append(x)

            attention_weights['attn_weights_layer{}'.format(self.cross_attn_layers.index(layer) + 1)] = [x12_attn_w,
                                                                                                         x21_attn_w,
                                                                                                         x1_cross_attn_w,
                                                                                                         x2_cross_attn_w]

        if self.return_intermediate:
            return tf.stack(intermediate, axis=0), attention_weights

        return x, attention_weights

    def get_config(self):
        config = super(CrossAttnBlock, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_layers': self.num_layers,
            'cross_num_heads': self.cross_num_heads,
            'x1_num_heads': self.x1_num_heads,
            'x2_num_heads': self.x2_num_heads,
            'x1_d_ff': self.x1_d_ff,
            'x2_d_ff': self.x2_d_ff,
            'atv_fun': self.atv_fun,
            'dropout_rate': self.dropout_rate,
            'x1_dim_k': self.x1_dim_k,
            'x1_parameter_sharing': self.x1_parameter_sharing,
            'x1_full_attention': self.x1_full_attention,
            'x2_dim_k': self.x2_dim_k,
            'x2_parameter_sharing': self.x2_parameter_sharing,
            'x2_full_attention': self.x2_full_attention,
            'return_intermediate': self.return_intermediate,
            'cross_attn_layers': self.cross_attn_layers})

        return config
