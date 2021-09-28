Tested Vision Transformer(ViT)

Pytorch version Reference : https://github.com/FrancescoSaverioZuppichini/ViT

Tested on CIFAR10 dataset without validation set for convenience

Trained with AutoAugment & CutMix & MixUp

Without pretraining, ViT did not perform as well as EfficientNet in both accuracy & training time w.r.t. small dataset like CIFAR10, probably due to lack of inductive bias.
