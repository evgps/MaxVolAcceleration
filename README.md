# MaxVolAcceleration

For running experiments, please, use experiments.ipynb and comments how to specify configs.

# Data structure (tree, only dirs) - example for vgg19 model on cifar10 dataset:

    config.data_root (datasets are downloads automatically, don't care)
    ├── cifar10
    ├── cifar100
    ├── svhn
    └── stl10

    config.storage_path/pretrained/
    └── cifar10
        └── vgg19
            └──<<!!download_initial_model_to_there!!!>>

    config.storage_path/interlayers/
    └── cifar10
        └── vgg19
            └── svd

    config.storage_path/compressed/
    └── cifar10
        └── vgg19
            └── finetuned
                ├── [None, None, None, None, None, None, None, None, None, None, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
                ...........................>>>>...................
                └── [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    config.storage_path/results/
    ├── cifar10
    │   └── vgg19
    │       └── svd_imgs
    └── images
