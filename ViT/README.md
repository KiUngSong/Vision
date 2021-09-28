Tested Vision Transformer(ViT)

Pytorch version Reference : https://github.com/FrancescoSaverioZuppichini/ViT

Tested on CIFAR10 dataset without validation set for convenience

When trained on CIFAR10 without pretraining, ViT did not perform as well as EfficientNet in both accuracy & training time, probably due to lack of inductive bias.

Trained with AutoAugment & CutMix & MixUp
