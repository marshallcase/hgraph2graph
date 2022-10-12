# Hierarchical Generation and Prediction of Molecular Graphs using Structural Motifs

Original work is at https://arxiv.org/pdf/2002.03230.pdf

## Installation
First install the dependencies via conda (ensure that you generate vocab and preprocess with the same python environment as you train and sample!):
 * PyTorch >= 1.0.0
 * networkx
 * RDKit >= 2019.03
 * numpy
 * Python >= 3.6

And then run `pip install .`. Additional dependency for property-guided finetuning:
 * Chemprop >= 1.2.0


## Data Format
* For graph generation, each line of a training file is a SMILES string of a molecule
* For graph translation, each line of a training file is a pair of molecules (molA, molB) that are similar to each other but molB has better chemical properties. Please see `data/qed/train_pairs.txt`. The test file is a list of molecules to be optimized. Please see `data/qed/test.txt`.

## Molecule generation pretraining procedure
We can train a molecular language model on a large corpus of unlabeled molecules. We have uploaded a model checkpoint pre-trained on ChEMBL dataset in `ckpt/chembl-pretrained/model.ckpt`. If you wish to train your own language model, please follow the steps below:

1. Extract substructure vocabulary from a given set of molecules:
```
python get_vocab.py --ncpu 16 < data/chembl/all.txt > vocab.txt
```

2. Preprocess training data:
```
python preprocess.py --train data/chembl/all.txt --vocab data/chembl/all.txt --ncpu 16 --mode single
mkdir train_processed
mv tensor* train_processed/
```

3. Train graph generation model
```
mkdir ckpt/chembl-pretrained
python train_generator.py --train train_processed/ --vocab data/chembl/vocab.txt --save_dir ckpt/chembl-pretrained
```

4. Sample molecules from a model checkpoint
```
python generate.py --vocab data/chembl/vocab.txt --model ckpt/chembl-pretrained/model.ckpt --nsamples 1000
```


