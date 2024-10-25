# ReSet: A Residual Set-Transformer for Melanoma Detection

![framework](https://github.com/JulesCollenne/ReSeT/assets/43369571/f6f40a31-09b0-4bf8-84c7-0d61358d169b)

Welcome to the official repository for the paper "ReSet: A Residual Set-Transformer approach to tackle the ugly-duckling sign in melanoma detection".

## Abstract
The dermatological concept of the Ugly-Duckling Sign (UDS) emphasizes the importance of comparing skin lesions within the same patient for enhanced diagnostic accuracy in melanoma detection, stating that atypical lesions are more likely to be cancers. However this concept is still underutilized in research, as most work on melanoma detection rely on classification ConvNets which lack the capacity to compare images together. Addressing this research gap, we introduce ReSeT (Residual Set-Transformer), a framework designed to compare skin lesions within patients during prediction. ReSeT comprises an encoder that takes individual images as input to generate embeddings, and a Set-Transformer with a residual prediction layer that compares these embeddings while predicting. We demonstrate that our architecture ReSeT significantly enhances performance compared to ConvNets and we highlighting the necessity of residual connections in the context of multi-output Transformers. We also observe that self-supervised encoders are able to generate embeddings of comparable quality to those of supervised models, showing their robustness and impact on image comparison tasks.

## Repository Structure
The code will generate missing folders if they do not exist.
```
/
|- features/              # Extracted features directory
|- models/                # Models checkpoints
|- data_isic.py           # Code for loading ISIC 2020 per patient
|- reset.py               # Implementation of ReSeT
|- train_reset.py         # Source code to train ReSeT
|- test_reset.py          # Source code of test ReSeT
|- LICENSE
|- README.md
|- requirements.txt
```

## Example Usage

You can instantiate the model like so:

```
from reset import ReSeT
model = ReSeT()
```

If you want to use ISIC 2020 dataset, you can use the DataLoader class:
```
train_gen = DataLoaderISIC(
        f"features/{model_name}_train.csv",
        "GroundTruth.csv",
        batch_size=args.batch_size,
        n_vect=n_vect)
```
You need to make sure that you have the original GroundTruth.csv of ISIC 2020, along with your extracted features (e.g in features/XXX_train.csv)

## Getting Started

To get started with this repository, clone it and install the required dependencies:

```
git clone https://github.com/JulesCollenne/ReSeT.git
cd ReSeT
pip install -r requirements.txt
```

##  Citation
Soon!

[//]: # (If you find our work useful in your research, please consider citing:)

[//]: # ()
[//]: # (```)

[//]: # ()
[//]: # (@article{collenne2024,)

[//]: # ()
[//]: # (  title={ReSet: A Residual Set-Transformer approach to tackle the ugly-duckling sign in melanoma detection},)

[//]: # ()
[//]: # (  author={Collenne, Jules and Iguernaissi, Rabah and Dubuisson, Severine and Merad, Djamal},)

[//]: # ()
[//]: # (  journal={},)

[//]: # ()
[//]: # (  year={2024})

[//]: # ()
[//]: # (})

[//]: # ()
[//]: # (```)

[//]: # (And the SetTransformer paper:)

[//]: # (```)

[//]: # (@InProceedings{lee2019set,)

[//]: # (    title={Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks},)

[//]: # (    author={Lee, Juho and Lee, Yoonho and Kim, Jungtaek and Kosiorek, Adam and Choi, Seungjin and Teh, Yee Whye},)

[//]: # (    booktitle={Proceedings of the 36th International Conference on Machine Learning},)

[//]: # (    pages={3744--3753},)

[//]: # (    year={2019})

[//]: # (})

[//]: # (```)
