# Coherence Modeling with TDA

## Setup

```py
conda env create -f environment.yml
conda activate tda-modeling-env
```

## Usage

Generating TDA features for a dataset:

```py
python feature_gen.py --cuda 0 --data_name clinton_train --input_dir GCDC_Dataset/ --output_dir gcdc_tda_features --batch_size 100
```

Train/test MLP using generated TDA features:

```py
python predict_tda.py --input_dir GCDC_dataset/ --feat_dir gcdc_tda_features/  --domain clinton
```

Generate a processed version of the WSJ dataset using `preproc_wsj.py`, (sentence and para jumbling).

## Data

- **GCDC** - Refer [GCDC-Corpus](https://github.com/aylai/GCDC-corpus) for the source
- **WSJ** - Available in [Penn Treebank](https://catalog.ldc.upenn.edu/LDC99T42). This repo uses a processed version from the [Transformer Models for Text Coherence Assessment](https://github.com/tushar117/Transformer-Models-for-Text-Coherence-Assessment) repo.

## Acknowledgements

We thank the authors of [Artificial Text Detection via Examining the Topology of Attention Maps (EMNLP 2021)](https://github.com/danchern97/tda4atd) for publishing their code. 