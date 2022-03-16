# DTITR: End-to-End Drut-Target Binding Affinity Prediction with Transformer
<p align="justify"> We propose an end-to-end Transformer-based and DTI-inspired architecture (DTITR) for predicting the logarithmic-transformed quantitative dissociation constant (pKd) of DTI pairs, where self-attention layers are exploited to learn the short and long-term biological and chemical context dependencies between the sequential and structural units of the protein sequences and compound SMILES strings, respectively, and cross-attention layers to exchange information and learn the pharmacological context associated with the interaction space. The architecture makes use of two parallel Transformer-Encoders to compute a contextual embedding of the protein sequences and SMILES strings, and a Cross-Attention Transformer-Encoder block to model the interaction, where the resulting aggregated representation hidden states are concatenated and used as input for a Fully-Connected Feed-Forward Network.</p>


## DTITR Architecture
<p align="center"><img src="/figures/dtitr_arch.png" width="70%" height="70%"/></p>

## Davis Kinase Binding Affinity
### Dataset
- **davis_original_dataset:** original dataset
- **davis_dataset_processed:** dataset processed : prot sequences + rdkit SMILES strings + pkd values
- **deep_features_dataset:** CNN deep representations: protein + SMILES deep representations
### Clusters
- **test_cluster:** independent test set indices
- **train_cluster_X:** train indices 
### Similarity
- **protein_sw_score:** protein Smith-Waterman similarity scores
- **protein_sw_score_norm:** protein Smith-Waterman similarity normalized scores
- **smiles_ecfp6_tanimoto_sim:** SMILES Morgan radius 3 similarity scores

## Dictionaries
- **davis_prot_dictionary**: AA char-integer dictionary
- **davis_smiles_dictionary**: SMILES char-integer dictionary
- **protein_codes_uniprot/subword_units_map_uniprot**: Protein Subwords Dictionary
- **drug_codes_chembl/subword_units_map_chembl**: SMILES Subwords Dictionary

## Requirements:
- Python 3.9.6
- Tensorflow 2.6.0
- Numpy 
- Pandas
- Scikit-learn
- Itertools
- Matplotlib
- Seaborn
- Glob
- Json
- periodictable
- subword_nmt

## Usage:
### Training
```
python dtitr_model.py --option Train --num_epochs 500 --batch_dim 32 --prot_transformer_depth 3 --smiles_transformer_depth 3 --cross_block_depth 1 --prot_transformer_heads 4 --smiles_transformer_heads 4 --cross_block_heads 4 --prot_parameter_sharing '' --prot_dim_k 0 --prot_ff_dim 512 --smiles_ff_dim 512 --d_model 128 --dropout_rate 0.1 --dense_atv_fun gelu --out_mlp_depth 3 --out_mlp_hdim 512 512 512 --optimizer_fn radam 1e-04 0.9 0.999 1e-08 1e-05
```
### Validation
```
python dtitr_model.py --option Validation --num_epochs 500 --batch_dim 32 --prot_transformer_depth 2 3 4 --smiles_transformer_depth 2 3 4 --cross_block_depth 1 2 3 4 --prot_transformer_heads 4 --smiles_transformer_heads 4 --cross_block_heads 4 --prot_parameter_sharing '' --prot_dim_k 0 --prot_ff_dim 512 --smiles_ff_dim 512 --d_model 128 --dropout_rate 0.1 --dense_atv_fun gelu --out_mlp_depth 3 --out_mlp_hdim 512 512 512 --optimizer_fn radam 1e-04 0.9 0.999 1e-08 1e-05
```

### Evaluation
```
python dtitr_model.py --option Evaluation
```
