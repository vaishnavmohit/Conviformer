# Conviformer : Convolutionally guided Vision Transformers

This repository contains PyTorch code for Conviformer with ConViT base. But this is generic approach and can be applied with any vision transformer. 

For details see the [Conviformer paper](https://arxiv.org/abs/2208.08900) by Mohit Vaishnav, Thomas Fel, Ivan Felipe Rodrıguez, Thomas Serre.

If you use this code for a paper please cite:

```
@article{vaishnav2022conviformers,
  title={Conviformers: Convolutionally guided Vision Transformer},
  author={Vaishnav, Mohit and Fel, Thomas and Rodr{\i}guez, Ivan Felipe and Serre, Thomas},
  journal={arXiv preprint arXiv:2208.08900},
  year={2022}
}
```

# Usage

Install PyTorch 1.7.0+ and torchvision 0.8.1+ and [pytorch-image-models 0.3.2](https://github.com/rwightman/pytorch-image-models):

```
conda install -c pytorch pytorch torchvision
pip install timm==0.3.2
```

## Data preparation

Dataset is downloaded from `Kaggle Herbarium Challenge` page. We have evaluated `Conviformer` on [Herbarium 2021](https://www.kaggle.com/competitions/herbarium-2021-fgvc8) and [Herbarium 2022](https://kaggle.com/competitions/herbarium-2022-fgvc9/) dataset both of them are referred to as `Herbarium` and `Herbarium22` respectively in the code. 

## Evaluation
To evaluate Conviformer-Base on test set, run:
```
./eval.sh model_name input_size data_set checkpoint_path batch_size

```


## Training
To train Conviformer-Base on Herbarium 202x on a single node with ```n_gpus``` gpus and ```batch_size``` batch size for 300 epochs run:

```
./train_herbarium_patch batch_size n_gpus 
```

Similarly, for baseline reproducibility you may follow the above steps with 
```
./train_herbarium_base batch_size n_gpus 
``` 

# License
The majority of this repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.