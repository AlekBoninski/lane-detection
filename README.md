# Lane Detection Master Thesis Work
This repository contains my master thesis work at the Sofia University's Faculty for Mathematics and Informatics.

## Train network
In this project I implemented a CNN with an architecture inspired by [ERFNet](https://www.researchgate.net/publication/320293291_ERFNet_Efficient_Residual_Factorized_ConvNet_for_Real-Time_Semantic_Segmentation).

To train the network you will need to download the TuSimple dataset found [here](https://www.kaggle.com/datasets/manideep1108/tusimple?resource=download).

Training is initiated by running the `./src/train.py` file with the following args:
```
python train.py \
  --model erfnet \
  --resume-training \
  --epochs 100 \
  --ckpt-dir ./checkpoints \
  --ckpt-save-interval 5 \
  --cuda \
  --height 720 \
  --dataset-root /data/TUSimple \
  --pretrained-encoder ./encoder_weights.pth
```

### `--model`
- **Description**: Model architecture to use for training
- **Type**: `str`
- **Choices**: encoder, encoder-low-dilation, erfnet, erfnet-low-dilation
- **Default**: `None`
- **Usage**: 
  ```bash
  --model [encoder|encoder-low-dilation|erfnet|erfnet-low-dilation]
  ```

### `--resume-training`
- **Description**: Flag indicating if the training should resume from a saved checkpoint.
- **Default**: `False`
- **Usage**: 
  ```bash
  --resume-training
  ```

### `--epochs`
- **Description**: Epochs to train for.
- **Type**: `int`
- **Default**: `150`
- **Usage**: 
  ```bash
  --epochs 150
  ```

### `--ckpt-dir`
- **Description**: Directory where to save the checkpoints.
- **Type**: `str`
- **Default**: `None`
- **Usage**: 
  ```bash
  --ckpt-dir ./checkpoints
  ```

### `--ckpt-latest-name`
- **Description**: Name of the latest checkpoint (the model will resume training from here).
- **Type**: `str`
- **Default**: `latest.pth.tar`
- **Usage**: 
  ```bash
  --ckpt-latest-name latest.pth.tar
  ```

### `--ckpt-best-name`
- **Description**: Name of the checkpoint that has achieved best results.
- **Type**: `str`
- **Default**: `best.pth.tar`
- **Usage**: 
  ```bash
  --ckpt-best-name best.pth.tar
  ```

### `--ckpt-interval-name`
- **Description**: Template for naming interval checkpoints during training. The placeholder {} will be replaced with the epoch number.
- **Type**: `str`
- **Default**: `checkpoint-{:04}.pth.tar`
- **Usage**: 
  ```bash
  --ckpt-interval-name checkpoint-{:04}.pth.tar
  ```

### `--ckpt-save-interval`
- **Description**: Interval (in epochs) at which to save intermediate checkpoints.
- **Type**: `int`
- **Default**: `10`
- **Usage**: 
  ```bash
  --ckpt-save-interval 10
  ```

### `--loss-log-step-interval`
- **Description**: Interval (in steps) at which to log the loss during training.
- **Type**: `int`
- **Default**: `50`
- **Usage**: 
  ```bash
  --loss-log-step-interval 50
  ```

### `--cuda`
- **Description**: Enables training on GPU (CUDA). This flag is used to toggle between GPU and CPU.
- **Default**: `True`
- **Usage**: 
  ```bash
  --cuda
  ```

### `--height`
- **Description**: Specifies the input height for the model.
- **Type**: `int`
- **Default**: `512`
- **Usage**: 
  ```bash
  --height 512
  ```

### `--dataset-root`
- **Description**: Path to the root directory of the dataset.
- **Type**: `str`
- **Default**: `../archive/TUSimple`
- **Usage**: 
  ```bash
  --dataset-root ../archive/TUSimple
  ```

### `--pretrained-encoder`
- **Description**: Path to the pretrained encoder weights.
- **Type**: `str`
- **Default**: `None`
- **Usage**: 
  ```bash
  --pretrained-encoder ../checkpoints/encoder/latest.pth.tar
  ```
  
