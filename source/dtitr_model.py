# -*- coding: utf-8 -*-
"""
@author: NelsonRCM
"""

from transformer_encoder import *
from cross_attention_transformer_encoder import *
from embedding_layer import *
from layers_utils import *
from output_block import *
from dataset_builder_util import *
import itertools
import tensorflow_addons as tfa
from argument_parser import *
import gc
from plot_utils import *
from utils import *
from huggingface_hub import HfApi
from datetime import datetime


def build_dtitr_model(FLAGS, prot_trans_depth, smiles_trans_depth, cross_attn_depth,
                      prot_trans_heads, smiles_trans_heads, cross_attn_heads,
                      prot_parameter_sharing, prot_dim_k,
                      prot_d_ff, smiles_d_ff, d_model, dropout_rate, dense_atv_fun,
                      out_mlp_depth, out_mlp_units, optimizer_fn):
    """
    Function to build the DTITR Model

    Args:
    - FLAGS: arguments object
    - prot_trans_depth [int]: number of protein transformer-encoders
    - smiles_trans_depth [int]: number of SMILES transformer-encoders
    - cross_attn_depth [int]: number of cross-attention transformer-encoders
    - prot_trans_heads [int]: number of heads for the protein self-attention mha
    - smiles_trans_heads [int]: number of heads for the smiles self-attention mha
    - cross_attn_heads [int]: number of heads for the cross-attention mha
    - prot_parameter_sharing [str]: protein parameter sharing option in the case of Linear MHA
    - prot_dim_k [int]: protein Linear MHA projection dimension
    - prot_d_ff [int]: hidden numbers for the first dense layer of the FFN in the case of the proteins
    - smiles_d_ff [int]: hidden numbers for the first dense layer of the FFN in the case of the smiles
    - d_model [int]: embedding dim
    - dropout_rate [float]: % of dropout
    - dense_atv_fun: dense layers activation function
    - out_mlp_depth [int]: FCNN number of layers
    - out_mlp_units [list of ints]: hidden neurons for each one of the dense layers of the FCNN
    - optimizer_fn: optimizer function

    Outputs:
    - dtitr_model

    """

    if FLAGS.bpe_option[0]:
        prot_input = tf.keras.Input(shape=(FLAGS.protein_bpe_len + 1,), dtype=tf.int32, name='protein_input')
        prot_mask = attn_pad_mask()(prot_input)
        encode_prot = EmbeddingLayer(FLAGS.protein_dict_bpe_len + 2, d_model,  # FLAGS.protein_bpe_len+1,
                                     dropout_rate, FLAGS.pos_enc_option)(prot_input)

    else:
        prot_input = tf.keras.Input(shape=(FLAGS.protein_len + 1,), dtype=tf.int32, name='protein_input')
        prot_mask = attn_pad_mask()(prot_input)
        encode_prot = EmbeddingLayer(FLAGS.protein_dict_len + 2, d_model,  # FLAGS.protein_len+1,
                                     dropout_rate, FLAGS.pos_enc_option)(prot_input)

    encode_prot, _ = Encoder(d_model, prot_trans_depth, prot_trans_heads, prot_d_ff, dense_atv_fun,
                             dropout_rate, prot_dim_k, prot_parameter_sharing,
                             FLAGS.prot_full_attn,
                             FLAGS.return_intermediate, name='encoder_prot')(encode_prot, prot_mask)

    if FLAGS.bpe_option[1]:
        smiles_input = tf.keras.Input(shape=(FLAGS.smiles_bpe_len + 1,), dtype=tf.int32, name='smiles_input')
        smiles_mask = attn_pad_mask()(smiles_input)
        encode_smiles = EmbeddingLayer(FLAGS.smiles_dict_bpe_len + 2, d_model,  # FLAGS.smiles_bpe_len+1,
                                       dropout_rate, FLAGS.pos_enc_option)(smiles_input)
    else:
        smiles_input = tf.keras.Input(shape=(FLAGS.smiles_len + 1,), dtype=tf.int32, name='smiles_input')
        smiles_mask = attn_pad_mask()(smiles_input)
        encode_smiles = EmbeddingLayer(FLAGS.smiles_dict_len + 2, d_model,  # FLAGS.smiles_len+1,
                                       dropout_rate, FLAGS.pos_enc_option)(smiles_input)

    encode_smiles, _ = Encoder(d_model, smiles_trans_depth, smiles_trans_heads, smiles_d_ff, dense_atv_fun,
                               dropout_rate, FLAGS.smiles_dim_k, FLAGS.smiles_parameter_sharing,
                               FLAGS.smiles_full_attn, FLAGS.return_intermediate,
                               name='encoder_smiles')(encode_smiles, smiles_mask)

    cross_prot_smiles, _ = CrossAttnBlock(d_model, cross_attn_depth, cross_attn_heads, prot_trans_heads,
                                          smiles_trans_heads, prot_d_ff, smiles_d_ff, dense_atv_fun,
                                          dropout_rate, prot_dim_k, prot_parameter_sharing,
                                          FLAGS.prot_full_attn, FLAGS.smiles_dim_k,
                                          FLAGS.smiles_parameter_sharing, FLAGS.smiles_full_attn,
                                          FLAGS.return_intermediate,
                                          name='cross_attn_block')([encode_prot,
                                                                    encode_smiles],
                                                                   smiles_mask,
                                                                   prot_mask)

    out = OutputMLP(out_mlp_depth, out_mlp_units, dense_atv_fun,
                    FLAGS.output_atv_fun, dropout_rate, name='output_block')(cross_prot_smiles)


    dtitr_model = tf.keras.Model(inputs=[prot_input, smiles_input], outputs=out, name='dtitr')

    dtitr_model.compile(optimizer=optimizer_fn, loss=FLAGS.loss_function,
                        metrics=[tf.keras.metrics.RootMeanSquaredError(), c_index])

    # tf.keras.utils.plot_model(dtitr_model, to_file='./dtitr.png', dpi=600)

    return dtitr_model


def chemogenomic_folds_grid_search(FLAGS, data, folds, model_function):
    """
    Grid Search function

    Args:
    - FLAGS: arguments object
    - data: [protein data, smiles data, kd values]
    - folds: [fold_1,fold_2,...,fold_n] fold_n: indices for the fold x
    - model_function: function that creates the model

    """

    epochs_set = FLAGS.num_epochs
    batch_set = FLAGS.batch_dim
    p_enc_depth = FLAGS.prot_transformer_depth
    s_enc_depth = FLAGS.smiles_transformer_depth
    cross_depth = FLAGS.cross_block_depth
    p_enc_heads = FLAGS.prot_transformer_heads
    s_enc_heads = FLAGS.smiles_transformer_heads
    cross_heads = FLAGS.cross_block_heads
    prot_param_sharing_set = FLAGS.prot_parameter_sharing
    prot_dim_k_set = FLAGS.prot_dim_k
    prot_dff = FLAGS.prot_ff_dim
    smiles_dff = FLAGS.smiles_ff_dim
    d_model_set = FLAGS.d_model
    drop_rate_set = FLAGS.dropout_rate
    dense_act_set = FLAGS.dense_atv_fun
    out_block_depth = FLAGS.out_mlp_depth
    out_block_units_set = FLAGS.out_mlp_hdim
    opt_set = FLAGS.optimizer_fn

    logging("--------------------Grid Search-------------------", FLAGS)

    for params in itertools.product(epochs_set, batch_set, p_enc_depth, s_enc_depth, cross_depth, p_enc_heads,
                                    s_enc_heads, cross_heads, prot_param_sharing_set, prot_dim_k_set,
                                    prot_dff, smiles_dff, d_model_set, drop_rate_set,
                                    dense_act_set, out_block_depth, out_block_units_set, opt_set):

        p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18 = params

        results = []

        if p18[0] == 'radam':
            p18 = tfa.optimizers.RectifiedAdam(learning_rate=float(p18[1]), beta_1=float(p18[2]),
                                               beta_2=float(p18[3]), epsilon=float(p18[4]),
                                               weight_decay=float(p18[5]))
        elif p18[0] == 'adam':
            p18 = tf.keras.optimizers.Adam(learning_rate=float(p18[1]), beta_1=float(p18[2]),
                                           beta_2=float(p18[3]), epsilon=float(p18[4]))

        elif p18[0] == 'adamw':
            p18 = tfa.optimizers.AdamW(learning_rate=float(p18[1]), beta_1=float(p18[2]),
                                       beta_2=float(p18[3]), epsilon=float(p18[4]),
                                       weight_decay=float(p18[5]))

        for fold_idx in range(len(folds)):
            index_train = list(itertools.chain.from_iterable([folds[i] for i in range(len(folds)) if i != fold_idx]))

            index_val = folds[fold_idx]

            data_train = [tf.gather(i, index_train) for i in data]

            data_val = [tf.gather(i, index_val) for i in data]

            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  min_delta=0.001, patience=50, mode='min',
                                                  restore_best_weights=True)

            # mc = tf.keras.callbacks.ModelCheckpoint(filepath = FLAGS.checkpoint_path+'/'+str(fold_idx)+'/',
            #                                         monitor = 'val_root_mean_squared_error',
            #                                save_best_only=True, save_weights_only = True, mode = 'min')

            grid_model = model_function(FLAGS, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18)

            grid_model.fit(x=[data_train[0], data_train[1]], y=data_train[2],
                           batch_size=p2, epochs=p1,
                           verbose=2, callbacks=[es],
                           validation_data=([data_val[0], data_val[1]], data_val[2]))

            mse, rmse, ci = grid_model.evaluate([data_val[0], data_val[1]], data_val[2])
            results.append((mse, rmse, ci))

            logging(("Epochs = %d,  Batch = %d, P Enc Depth = %d, S Enc Depth = %d, Cross Depth = %d, P Heads = %d, " +
                     "S Heads = %d, Cross Heads = %d, Prot P Sharing = %s, Prot Dim K = %d, " +
                     "P DFF = %d, S DFF = %d, D Model = %d, DropR = %0.2f, " +
                     "Dense AF = %s, Out Depth = %d, Out Units = %s, Optimizer = %s, " +
                     "Fold = %d, MSE = %0.3f, RMSE = %0.3f, CI = %0.3f") %
                    (p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14,
                     p15, p16, p17, p18.get_config(), fold_idx, mse, rmse, ci), FLAGS)

            del data_train
            del data_val
            del grid_model
            gc.collect()

        logging("Mean Folds - " + (" MSE = %0.3f, RMSE = %0.3f, CI = %0.3f" % (np.mean(results, axis=0)[0],
                                                                               np.mean(results, axis=0)[1],
                                                                               np.mean(results, axis=0)[2])), FLAGS)


def run_grid_search(FLAGS):
    """
    Run Grid Search function

    Args:
    - FLAGS: arguments object

    """

    model_function = build_dtitr_model
    protein_data, smiles_data, kd_values = dataset_builder(FLAGS.data_path).transform_dataset(FLAGS.bpe_option[0],
                                                                                              FLAGS.bpe_option[1],
                                                                                              'Sequence',
                                                                                              'SMILES',
                                                                                              'Kd',
                                                                                              FLAGS.protein_bpe_len,
                                                                                              FLAGS.protein_len,
                                                                                              FLAGS.smiles_bpe_len,
                                                                                              FLAGS.smiles_len)
    if FLAGS.bpe_option[0]:
        protein_data = add_reg_token(protein_data, FLAGS.protein_dict_bpe_len)
    else:
        protein_data = add_reg_token(protein_data, FLAGS.protein_dict_len)

    if FLAGS.bpe_option[1]:
        smiles_data = add_reg_token(smiles_data, FLAGS.smiles_dict_bpe_len)
    else:
        smiles_data = add_reg_token(smiles_data, FLAGS.smiles_dict_len)

        # kd_values = tf.expand_dims(kd_values,axis=1)

    _, _, _, clusters, _, _, _, _ = dataset_builder(FLAGS.data_path).get_data()

    clusters = [list(clusters[i][1].iloc[:, 0]) for i in range(len(clusters)) if clusters[i][0] != 'test']

    chemogenomic_folds_grid_search(FLAGS, [protein_data, smiles_data, kd_values], clusters, model_function)


def run_train_model(FLAGS):
    """
    Run Train function

    Args:
    - FLAGS: arguments object

    """

    protein_data, smiles_data, kd_values = dataset_builder(FLAGS.data_path).transform_dataset(FLAGS.bpe_option[0],
                                                                                              FLAGS.bpe_option[1],
                                                                                              'Sequence',
                                                                                              'SMILES',
                                                                                              'Kd',
                                                                                              FLAGS.protein_bpe_len,
                                                                                              FLAGS.protein_len,
                                                                                              FLAGS.smiles_bpe_len,
                                                                                              FLAGS.smiles_len)

    if FLAGS.bpe_option[0] == True:
        protein_data = add_reg_token(protein_data, FLAGS.protein_dict_bpe_len)
    else:
        protein_data = add_reg_token(protein_data, FLAGS.protein_dict_len)

    if FLAGS.bpe_option[1] == True:
        smiles_data = add_reg_token(smiles_data, FLAGS.smiles_dict_bpe_len)
    else:
        smiles_data = add_reg_token(smiles_data, FLAGS.smiles_dict_len)

        # kd_values = tf.expand_dims(kd_values,axis=1)

    _, _, _, clusters, _, _, _, _ = dataset_builder(FLAGS.data_path).get_data()

    train_idx = pd.concat([i.iloc[:, 0] for t, i in clusters if t == 'train'])
    test_idx = [i for t, i in clusters if t == 'test'][0].iloc[:, 0]

    prot_train = tf.gather(protein_data, train_idx)
    prot_test = tf.gather(protein_data, test_idx)

    smiles_train = tf.gather(smiles_data, train_idx)
    smiles_test = tf.gather(smiles_data, test_idx)

    kd_train = tf.gather(kd_values, train_idx)
    kd_test = tf.gather(kd_values, test_idx)

    FLAGS.optimizer_fn = FLAGS.optimizer_fn[0]

    if FLAGS.optimizer_fn[0] == 'radam':
        optimizer_fun = tfa.optimizers.RectifiedAdam(learning_rate=float(FLAGS.optimizer_fn[1]),
                                                     beta_1=float(FLAGS.optimizer_fn[2]),
                                                     beta_2=float(FLAGS.optimizer_fn[3]),
                                                     epsilon=float(FLAGS.optimizer_fn[4]),
                                                     weight_decay=float(FLAGS.optimizer_fn[5]))
    elif FLAGS.optimizer_fn[0] == 'adam':
        optimizer_fun = tf.keras.optimizers.Adam(learning_rate=float(FLAGS.optimizer_fn[1]),
                                                 beta_1=float(FLAGS.optimizer_fn[2]),
                                                 beta_2=float(FLAGS.optimizer_fn[3]),
                                                 epsilon=float(FLAGS.optimizer_fn[4]))

    elif FLAGS.optimizer_fn[0] == 'adamw':
        optimizer_fun = tfa.optimizers.AdamW(learning_rate=float(FLAGS.optimizer_fn[1]),
                                             beta_1=float(FLAGS.optimizer_fn[2]),
                                             beta_2=float(FLAGS.optimizer_fn[3]), epsilon=float(FLAGS.optimizer_fn[4]),
                                             weight_decay=float(FLAGS.optimizer_fn[5]))

    dtitr_model = build_dtitr_model(FLAGS, FLAGS.prot_transformer_depth[0], FLAGS.smiles_transformer_depth[0],
                                    FLAGS.cross_block_depth[0],
                                    FLAGS.prot_transformer_heads[0], FLAGS.smiles_transformer_heads[0],
                                    FLAGS.cross_block_heads[0],
                                    FLAGS.prot_parameter_sharing[0], FLAGS.prot_dim_k[0],
                                    FLAGS.prot_ff_dim[0], FLAGS.smiles_ff_dim[0], FLAGS.d_model[0],
                                    FLAGS.dropout_rate[0], FLAGS.dense_atv_fun[0],
                                    FLAGS.out_mlp_depth[0], FLAGS.out_mlp_hdim[0], optimizer_fun)

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                          min_delta=0.001, patience=60, mode='min',
                                          restore_best_weights=True)

    mc = tf.keras.callbacks.ModelCheckpoint(filepath=FLAGS.checkpoint_path + '/' + 'dtitr_model_v2' + '/',
                                            monitor='val_loss',
                                            save_best_only=True, save_weights_only=True, mode='min')

    dtitr_model.fit(x=[prot_train, smiles_train], y=kd_train,
                    batch_size=FLAGS.batch_dim[0], epochs=FLAGS.num_epochs[0],
                    verbose=2, callbacks=[es, mc],
                    validation_data=([prot_test, smiles_test], kd_test))

    mse, rmse, ci = dtitr_model.evaluate([prot_test, smiles_test], kd_test)

    dtitr_model.save('dtitr_model.h5')
    api = HfApi()
    api.upload_file(
        path_or_fileobj= os.path.join(os.getcwd(), 'dtitr_model.h5'),  
        path_in_repo=f'DTITR-{datetime.now()}',
        repo_id="DLSAutumn2023/DTITR_Recreation"
    )


    logging("Test Fold - " + (" MSE = %0.3f, RMSE = %0.3f, CI = %0.3f" % (mse, rmse, ci)), FLAGS)


def run_evaluation_model(FLAGS):
    """
    Run Evaluation function

    Args:
    - FLAGS: arguments object

    """

    protein_data, smiles_data, kd_values = dataset_builder(FLAGS.data_path).transform_dataset(FLAGS.bpe_option[0],
                                                                                              FLAGS.bpe_option[1],
                                                                                              'Sequence',
                                                                                              'SMILES',
                                                                                              'Kd',
                                                                                              FLAGS.protein_bpe_len,
                                                                                              FLAGS.protein_len,
                                                                                              FLAGS.smiles_bpe_len,
                                                                                              FLAGS.smiles_len)

    if FLAGS.bpe_option[0] == True:
        protein_data = add_reg_token(protein_data, FLAGS.protein_dict_bpe_len)
    else:
        protein_data = add_reg_token(protein_data, FLAGS.protein_dict_len)

    if FLAGS.bpe_option[1] == True:
        smiles_data = add_reg_token(smiles_data, FLAGS.smiles_dict_bpe_len)
    else:
        smiles_data = add_reg_token(smiles_data, FLAGS.smiles_dict_len)

        # kd_values = tf.expand_dims(kd_values,axis=1)

    _, _, _, clusters, _, _, _, _ = dataset_builder(FLAGS.data_path).get_data()

    test_idx = [i for t, i in clusters if t == 'test'][0].iloc[:, 0]

    prot_test = tf.gather(protein_data, test_idx)

    smiles_test = tf.gather(smiles_data, test_idx)

    kd_test = tf.gather(kd_values, test_idx)

    optimizer_fun = tfa.optimizers.RectifiedAdam(learning_rate=1e-04, beta_1=0.9,
                                                 beta_2=0.999, epsilon=1e-08,
                                                 weight_decay=1e-05)

    dtitr_model = build_dtitr_model(FLAGS, 3, 3, 1, 4, 4, 4, '', '', 512, 512, 128, 0.1, 'gelu', 3, [512, 512, 512],
                                    optimizer_fun)

    dtitr_model.load_weights('../model/dtitr_model/')

    metrics = inference_metrics(dtitr_model, [prot_test, smiles_test, kd_test])

    logging(metrics, FLAGS)
    pred_scatter_plot(kd_test, dtitr_model.predict([prot_test, smiles_test])[:, 0],
                      'Davis Dataset: Predictions vs True Values', 'True Values', 'Predictions',
                      False, '')


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    FLAGS = argparser()
    FLAGS.log_dir = os.getcwd() + '/logs/' + time.strftime("%d_%m_%y_%H_%M", time.gmtime()) + "/"
    FLAGS.checkpoint_path = os.getcwd() + '/checkpoints/' + time.strftime("%d_%m_%y_%H_%M", time.gmtime()) + "/"
    FLAGS.data_path = {'data': '../data/davis/dataset/davis_dataset_processed.csv',
                       'prot_dic': '../dictionary/davis_prot_dictionary.txt',
                       'smiles_dic': '../dictionary/davis_smiles_dictionary.txt',
                       'clusters': glob.glob('../data/davis/clusters/*'),
                       'prot_bpe': ['../dictionary/protein_codes_uniprot.txt',
                                    '../dictionary/subword_units_map_uniprot.csv'],
                       'smiles_bpe': ['../dictionary/drug_codes_chembl.txt',
                                      '../dictionary/subword_units_map_chembl.csv']}

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    if not os.path.exists(FLAGS.checkpoint_path):
        os.makedirs(FLAGS.checkpoint_path)

    logging(str(FLAGS), FLAGS)

    if FLAGS.option == 'Train':
        run_train_model(FLAGS)

    if FLAGS.option == 'Validation':
        run_grid_search(FLAGS)

    if FLAGS.option == 'Evaluation':
        run_evaluation_model(FLAGS)
