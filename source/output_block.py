# -*- coding: utf-8 -*-
"""
@author: NelsonRCM
"""

import tensorflow as tf


class OutputMLP(tf.keras.Model):
    """
    Fully-Connected Feed-Forward Network (FCNN)

    Args:
    - mlp_depth [int]: number of dense layers for the FCNN
    - mlp_units [list of ints]: number of hidden neurons for each one of the dense layers
    - atv_fun: dense layers activation function
    - out_atv_fun: final dense layer activation function
    - dropout_rate [float]: % of dropout


    """
    def __init__(self, mlp_depth, mlp_units, atv_fun, out_atv_fun, dropout_rate, **kwargs):
        super(OutputMLP, self).__init__(**kwargs)

        self.mlp_depth = mlp_depth
        self.mlp_units = mlp_units
        self.atv_fun = atv_fun
        self.out_atv_fun = out_atv_fun
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.concatenate_layer = tf.keras.layers.Concatenate(name='concatenate')
        self.output_dense = tf.keras.layers.Dense(units=1, activation=self.out_atv_fun, name='out_final')

        self.mlp_head = tf.keras.Sequential(name='out_dense')
        for i in range(self.mlp_depth):
            self.mlp_head.add(tf.keras.layers.Dense(self.mlp_units[i], activation=self.atv_fun))
            self.mlp_head.add(tf.keras.layers.Dropout(self.dropout_rate))

    def call(self, inputs):
        """

        Args:
        - inputs: [input1,input2]: input 1 sequences and input 2 sequences

        Shape:
        - Inputs:
        - inputs1 : (B,L_1,E_1) where B is the batch size, L_1 is the sequence length for the input 1,
                                E_1 is the embedding dimension for the input 1
        - inputs2 : (B,L_2,E_2) where B is the batch size, L_2 is the sequence length for the input 2,
                                E_2 is the embedding dimension for the input 2

        - Outputs:
        - out: (B,1): where B is the batch size

        """

        prot_input = tf.gather(inputs[0], 0, axis=1)
        smiles_input = tf.gather(inputs[1], 0, axis=1)

        concat_input = self.concatenate_layer([prot_input, smiles_input])

        concat_input = self.mlp_head(concat_input)

        out = self.output_dense(concat_input)

        return out

    def get_config(self):
        config = super(OutputMLP, self).get_config()
        config.update({
            'mlp_depth': self.mlp_depth,
            'mlp_units': self.mlp_units,
            'atv_fun': self.atv_fun,
            'out_atv_fun': self.out_atv_fun,
            'dropout_rate': self.dropout_rate,
            'concatenate_layer': self.concatenate_layer,
            'output_dense': self.output_dense,
            'mlp_head': self.mlp_head})
        return config
