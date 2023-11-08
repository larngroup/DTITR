# -*- coding: utf-8 -*-
"""
@author: NelsonRCM
"""
import argparse
import os


def argparser():
    """
    Argument Parser Function

    Outputs:
    - FLAGS: arguments object

    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--mpath',
        type=str,
        help='specifies the specific path of the model.pb'
    )

    parser.add_argument(
        '--huggingSave',
        type=str,
        help='Saves model to hugging face repo'
    )

    parser.add_argument(
        '--option',
        type=str,
        help='Train, Validation or Evaluation')

    parser.add_argument(
        '--data_path',
        type=dict,
        default={},
        help='Data Path')

    parser.add_argument(
        '--pos_enc_option',
        type=bool,
        default=True,
        help='Position encoding option')

    parser.add_argument(
        '--bpe_option',
        type=list,
        default=[True, False],
        help='BPE encoding option [Protein,SMILES]')

    parser.add_argument(
        '--protein_len',
        type=int,
        default=1400,
        help='Protein Sequences Max Length')

    parser.add_argument(
        '--protein_bpe_len',
        type=int,
        default=556,
        help='Protein Sequences BPE Max Length')

    parser.add_argument(
        '--protein_dict_len',
        type=int,
        default=20,
        help='Protein AA Dictionary Length')

    parser.add_argument(
        '--protein_dict_bpe_len',
        type=int,
        default=16693,
        help='Protein BPE Dictionary Length')

    parser.add_argument(
        '--smiles_len',
        type=int,
        default=72,
        help='SMILES Strings Max Length')

    parser.add_argument(
        '--smiles_bpe_len',
        type=int,
        default=15,
        help='SMILES Sequences BPE Max Length')

    parser.add_argument(
        '--smiles_dict_len',
        type=int,
        default=26,
        help='SMILES Char Dictionary Length')

    parser.add_argument(
        '--smiles_dict_bpe_len',
        type=int,
        default=23532,
        help='SMILES BPE Dictionary Length')

    parser.add_argument(
        '--output_atv_fun',
        type=str,
        default='linear',
        help='Output Dense Layer Activation Function')

    parser.add_argument(
        '--dense_atv_fun',
        type=str,
        nargs='+',
        help='Dense Layer Activation Function')

    parser.add_argument(
        '--return_intermediate',
        type=bool,
        default=False,
        help='Return Intermediate Values (Enc,Cross)')

    parser.add_argument(
        '--loss_function',
        type=str,
        default='mean_squared_error',
        help='Loss Function')

    parser.add_argument(
        '--prot_transformer_depth',
        type=int,
        nargs='+',
        help='Protein Transformer Encoder Depth')

    parser.add_argument(
        '--smiles_transformer_depth',
        type=int,
        nargs='+',
        help='SMILES Transformer Encoder Depth')

    parser.add_argument(
        '--cross_block_depth',
        type=int,
        nargs='+',
        help='Cross Attention Block Depth')

    parser.add_argument(
        '--d_model',
        type=int,
        nargs='+',
        help='Emb Size')

    parser.add_argument(
        '--prot_transformer_heads',
        type=int,
        nargs='+',
        help='Protein Transformer Encoder Heads')

    parser.add_argument(
        '--smiles_transformer_heads',
        type=int,
        nargs='+',
        help='SMILES Transformer Encoder Heads')

    parser.add_argument(
        '--cross_block_heads',
        type=int,
        nargs='+',
        help='Cross Attention Block')

    parser.add_argument(
        '--prot_parameter_sharing',
        type=str,
        nargs='+',
        help='Protein Linear MHA Parameter Sharing Option: "layerwise", "none", "headwise" ')

    parser.add_argument(
        '--prot_dim_k',
        type=int,
        nargs='+',
        help='Protein Linear MHA Projection Dimension (Dim K)')

    parser.add_argument(
        '--prot_full_attn',
        type=bool,
        default=True,
        help='Protein Full Attention Option')

    parser.add_argument(
        '--smiles_parameter_sharing',
        type=str,
        default='none',
        help='SMILES Linear MHA Parameter Sharing Option: "layerwise", "none", "headwise" ')

    parser.add_argument(
        '--smiles_dim_k',
        type=int,
        default=0,
        help='SMILES Linear MHA Projection Dimension (Dim K)')

    parser.add_argument(
        '--smiles_full_attn',
        type=bool,
        default=True,
        help='SMILES Full Attention Option')

    parser.add_argument(
        '--prot_ff_dim',
        type=int,
        nargs='+',
        help='Protein PosWiseFF Dim')

    parser.add_argument(
        '--smiles_ff_dim',
        type=int,
        nargs='+',
        help='SMILES PosWiseFF Dim')

    parser.add_argument(
        '--dropout_rate',
        type=float,
        nargs='+',
        help='Dropout Rate')

    parser.add_argument(
        '--out_mlp_depth',
        type=int,
        nargs='+',
        help='Output MLP Block Depth')

    parser.add_argument(
        '--out_mlp_hdim',
        type=int,
        nargs='+',
        action='append',
        help='Output MLP Block Hidden Neurons')

    parser.add_argument(
        '--optimizer_fn',
        type=str,
        nargs='+',
        action='append',
        help='Optimizer Function Parameters')

    parser.add_argument(
        '--batch_dim',
        type=int,
        nargs='+',
        help='Batch Dim')

    parser.add_argument(
        '--num_epochs',
        type=int,
        nargs='+',
        help='Number of Epochs')

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='',
        help='Directory for checkpoint weights'
    )

    parser.add_argument(
        '--log_dir',
        type=str,
        default='',
        help='Directory for log data.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS


def logging(msg, FLAGS):
    """
    Logging function to update the log file

    Args:
    - msg [str]: info to add to the log file
    - FLAGS: arguments object

    """

    fpath = os.path.join(FLAGS.log_dir, "log.txt")

    with open(fpath, "a") as fw:
        fw.write("%s\n" % msg)
    print("------------------------//------------------------")
    print(msg)
    print("------------------------//------------------------")
