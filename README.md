# CS570 SRGAN-MobileNet

## Table of Contents

- [Environment](#environment)
- [Data](#data)
- [Model Training](#model-training)
  - [Pretraining](#pretraining-generator-only)
  - [Adversarial Training](#adversarial-training)
- [Inference](#inference)

## Environment

- python=3.9
- [PyTorch]((https://pytorch.org/))=1.12.1
- TorchVision=0.13.1
- OmegaConf
- Tensorboard (optional)



## Data

Training Dataset: [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

```bash
├── data/
    ├── DIV2K_TRAIN_HR/
    └── DIV2K_VALID_HR/
```



## Model Training

```bash
python train.py -h
```

### Pretraining (Generator only)

Training Script Format Example

```bash
python train.py \
  --config ./configs/pretrain.yaml \
  --exp_dir ./result/pretrain/ \
  --name <NAME> \
  --model_config ./configs/model/<MODEL>
```

Result

```bash
├── result/pretrain
    └── <NAME>/
        ├── config.yaml
        ├── output.log
        ├── save/         # Model Checkpoint
        ├── image_log/    # Training Log (LR,SR,HR)
        └── tensorboard/  # Tensorboard Log ( `tensorboard --logdir ...` )
```

### Adversarial Training (Generator, Discriminator)

TODO



## Inference

TODO
