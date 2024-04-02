# Coherence Modeling with TDA

[**Beyond Words: A Topological Exploration of Coherence in Text Documents**](https://openreview.net/forum?id=QJxVhljAyS)

The Second Tiny Papers Track at ICLR 2024

## Setup

```py
conda env create -f environment.yml
conda activate tda-modeling-env
```

## Usage

Computing TDA features for a dataset:

```py
python feature_gen.py --cuda 0 --data_name clinton_train --input_dir GCDC_Dataset/ --output_dir gcdc_tda_features --batch_size 100
```

Train/test MLP using generated TDA features:

```py
python predict_tda.py --input_dir GCDC_Dataset/ --feat_dir gcdc_tda_features/  --domain clinton
```

## Data

- **GCDC** - Refer [GCDC-Corpus](https://github.com/aylai/GCDC-corpus) for the source

## Acknowledgements

We thank the authors of [Artificial Text Detection via Examining the Topology of Attention Maps (EMNLP 2021)](https://github.com/danchern97/tda4atd) for publishing their code. 

## Citing

If you find this code useful for your research, please cite the following paper:

```bibtex
@inproceedings{jain2024beyond,
  title={Beyond Words: A Topological Exploration of Coherence in Text Documents},
  author={Jain, Samyak and Singhal, Rishi and Krishna, Sriram and Singla, Yaman K and Shah, Rajiv Ratn},
  booktitle={The Second Tiny Papers Track at ICLR 2024}
}
```
