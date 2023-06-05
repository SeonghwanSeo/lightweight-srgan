# HOw to LIghtweight SRGAN

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
- LPIPS



## Data

Training Dataset: [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

```bash
├── data/
    ├── DIV2K_train_HR/
    └── DIV2K_valid_HR/
```



## Model Training

```bash
python script/pretrain.py -h
python script/train.py -h
```

### Pretraining (Generator only)

Training Script Format Example

```bash
python pretrain.py \
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

Training Script Format Example

```bash
python train.py \
  --exp_dir ./result/gan/ \
  --pretrained_generator ./result/pretrain/<NAME>/save/best.tar \
  --name <NAME> \
  --model_config ./configs/model/<MODEL>
```

Result

```bash
├── result/gan
    └── <NAME>/
        ├── config.yaml
        ├── output.log
        ├── save/         # Model Checkpoint
        ├── image_log/    # Training Log (LR,SR,HR)
        └── tensorboard/  # Tensorboard Log ( `tensorboard --logdir ...` )
```

## Inference

```bash
python test.py ./weights/mobilenetv3.tar
```
