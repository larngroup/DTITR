# -*- coding: utf-8 -*-
"""
@author: NelsonRCM
"""
import tensorflow as tf
import pandas as pd
import numpy as np
import glob
import json
import periodictable as pt
import re
from itertools import chain
from operator import itemgetter
from subword_nmt.apply_bpe import BPE
import codecs



class dataset_builder():
    def __init__(self, data_path, **kwargs):
        super(dataset_builder, self).__init__(**kwargs)
        self.data_path = data_path

    def get_data(self):
        dataset = pd.read_csv(self.data_path['data'], sep=',', memory_map=True)
        prot_dictionary = json.load(open(self.data_path['prot_dic']))
        smiles_dictionary = json.load(open(self.data_path['smiles_dic']))
        clusters = []
        bpe_codes_prot = ''
        bpe_codes_map_prot = ''
        bpe_codes_smiles = ''
        bpe_codes_map_smiles = ''

        for i in self.data_path['clusters']:
            if 'test' in i:
                clusters.append(('test', pd.read_csv(i, header=None)))
            else:
                clusters.append(('train', pd.read_csv(i, header=None)))

        if self.data_path['prot_bpe'] != '':
            bpe_codes_prot = codecs.open(self.data_path['prot_bpe'][0])
            bpe_codes_map_prot = pd.read_csv(self.data_path['prot_bpe'][1])

        if self.data_path['smiles_bpe'] != '':
            bpe_codes_smiles = codecs.open(self.data_path['smiles_bpe'][0])
            bpe_codes_map_smiles = pd.read_csv(self.data_path['smiles_bpe'][1])

        return (dataset, prot_dictionary, smiles_dictionary, clusters, bpe_codes_prot, bpe_codes_map_prot,
                bpe_codes_smiles, bpe_codes_map_smiles)

    def data_conversion(self, data, dictionary, max_len):
        keys = list(i for i in dictionary.keys() if len(i) > 1)

        if len(keys) == 0:
            data = pd.DataFrame([list(i) for i in data])

        else:
            char_list = []
            for i in data:
                positions = []
                for j in keys:
                    positions.extend([(k.start(), k.end() - k.start()) for k in re.finditer(j, i)])

                positions = sorted(positions, key=itemgetter(0))

                if len(positions) == 0:
                    char_list.append(list(i))

                else:
                    new_list = []
                    j = 0
                    positions_start = [k[0] for k in positions]
                    positions_len = [k[1] for k in positions]

                    while j < len(i):
                        if j in positions_start:
                            new_list.append(str(i[j] + i[j + positions_len[positions_start.index(j)] - 1]))
                            j = j + positions_len[positions_start.index(j)]
                        else:
                            new_list.append(i[j])
                            j = j + 1
                    char_list.append(new_list)

            data = pd.DataFrame(char_list)

        data.replace(dictionary, inplace=True)

        data = data.fillna(0)
        if len(data.iloc[0, :]) == max_len:
            return data
        else:
            zeros_array = np.zeros(shape=(len(data.iloc[:, 0]), max_len - len(data.iloc[0, :])))
            data = pd.concat((data, pd.DataFrame(zeros_array)), axis=1)
            return data

    def encoding_bpe(self, data, codes, codes_map, max_len):
        bpe = BPE(codes, merges=-1, separator='')
        idx2word = codes_map['index'].values
        words2idx = dict(zip(idx2word, range(0, len(idx2word))))

        vectors = []

        for i in data:
            t1 = bpe.process_line(i).split()  # split
            try:
                i1 = np.asarray([words2idx[j] + 1 for j in t1])  # index
            except:
                i1 = np.array([0])

            l = len(i1)

            if l < max_len:
                k = np.pad(i1, (0, max_len - l), 'constant', constant_values=0)
            else:
                k = i1[:max_len]
            vectors.append(k[None, :])

        return tf.cast(tf.concat(vectors, axis=0), dtype=tf.int32)

    def transform_dataset(self, prot_bpe_enc_opt, smiles_bpe_enc_opt,
                          protein_column, smiles_column,
                          kd_column, bpe_prot_max_len, prot_max_len, bpe_smiles_max_leb, smiles_max_len):

        if prot_bpe_enc_opt == True:

            protein_data = self.encoding_bpe(self.get_data()[0][protein_column], self.get_data()[4],
                                             self.get_data()[5], bpe_prot_max_len)

        else:

            protein_data = tf.convert_to_tensor(self.data_conversion(self.get_data()[0][protein_column],
                                                                     self.get_data()[1], prot_max_len).astype('int32'),
                                                dtype=tf.int32)

        if smiles_bpe_enc_opt == True:

            smiles_data = self.encoding_bpe(self.get_data()[0][smiles_column], self.get_data()[6],
                                            self.get_data()[7], bpe_smiles_max_leb)

        else:

            smiles_data = tf.convert_to_tensor(self.data_conversion(self.get_data()[0][smiles_column],
                                                                    self.get_data()[2], smiles_max_len).astype('int32'),
                                               dtype=tf.int32)

        kd_values = self.get_data()[0][kd_column].astype('float32')

        return protein_data, smiles_data, kd_values
