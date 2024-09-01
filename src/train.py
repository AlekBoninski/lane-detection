import json
import os
import random
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, InterpolationMode, ToPILImage

from src.dataset.transform import EncoderResize, ImageAndMaskToTensor, ERFNetResize, MaskRelabel, RandomAugment
from src.models import Encoder, ERFNet
from src.dataset import TuSimple
from src.models.encoder import EncoderLowDilation
from src.models.erfnet import ERFNetLowDilation
from src.visualize import ERFNetVisualizer
from src.train import CrossEntropyLoss2d, ModelTrainer
from src.train.trainer import CheckpointConfig


def get_model(args, num_classes):
    model_name = args.model

    if model_name == "encoder":
        return Encoder(num_classes, predict=True)
    if model_name == "erfnet":
        pretrained_encoder = args.pretrained_encoder

        if pretrained_encoder:
            encoder = torch.nn.DataParallel(Encoder(num_classes))
            encoder.load_state_dict(torch.load(pretrained_encoder).get("model_state_dict"))
        else:
            encoder = None

        return ERFNet(num_classes, encoder=encoder)
    if model_name == "erfnet-low-dilation":
        pretrained_encoder = args.pretrained_encoder

        if pretrained_encoder:
            encoder = torch.nn.DataParallel(EncoderLowDilation(num_classes))
            encoder.load_state_dict(torch.load(pretrained_encoder).get("model_state_dict"))
        else:
            encoder = None

        return ERFNetLowDilation(num_classes, encoder=encoder)
    raise ValueError(f"Invalid model: {model_name}")


def get_transformers(args):
    model_name = args.model

    if model_name == "encoder":
        return [
            EncoderResize(),
            RandomAugment(),
            ImageAndMaskToTensor(),
            MaskRelabel(),
        ]
    if model_name == "erfnet":
        return [
            ERFNetResize(),
            RandomAugment(),
            ImageAndMaskToTensor(),
            MaskRelabel(),
        ]
    if model_name == "erfnet-low-dilation":
        return [
            ERFNetResize(),
            RandomAugment(),
            ImageAndMaskToTensor(),
            MaskRelabel(),
        ]
    raise ValueError(f"Invalid model: {model_name}")


NUM_CLASSES = 2


def main(args):
    model = get_model(args, NUM_CLASSES)

    # Datasets
    dataset_root = args.dataset_root
    train_root = os.path.join(dataset_root, "train_set")
    train_labels = os.path.join(dataset_root, "train_set", "label_data_0313.json")
    eval_root = os.path.join(dataset_root, "train_set")
    eval_labels = os.path.join(dataset_root, "train_set", "label_data_0531.json")

    transformers = get_transformers(args)
    tusimple_train = TuSimple(train_root, train_labels, transformers)
    dataset_train = DataLoader(tusimple_train, num_workers=4, batch_size=6, shuffle=True)

    tusimple_eval = TuSimple(eval_root, eval_labels, transformers)
    dataset_eval = DataLoader(tusimple_eval, num_workers=4, batch_size=6, shuffle=True)

    criterion = CrossEntropyLoss2d()

    checkpoint_config = CheckpointConfig(
        save_interval=args.ckpt_save_interval,
        checkpoint_dir=args.ckpt_dir,
        latest_checkpoint_name=args.ckpt_latest_name,
        best_checkpoint_name=args.ckpt_best_name,
        interval_checkpoint_name=args.ckpt_interval_name,
    )

    trainer = ModelTrainer(
        model,
        dataset_train,
        dataset_eval,
        criterion,
        checkpoint_config,
        resume_training=args.resume_training,
        cuda=args.cuda,
        loss_log_steps_interval=args.loss_log_step_interval,
        num_classes=NUM_CLASSES,
    )

    trainer.train(args.epochs)


if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument("--model", type=str, choices=["encoder", "erfnet", "erfnet-low-dilation"])
    argparser.add_argument("--resume-training", action="store_true", default=False)
    argparser.add_argument("--epochs", type=int, default=150)

    argparser.add_argument("--ckpt-dir", type=str)
    argparser.add_argument("--ckpt-latest-name", type=str, default="latest.pth.tar")
    argparser.add_argument("--ckpt-best-name", type=str, default="best.pth.tar")
    argparser.add_argument("--ckpt-interval-name", type=str, default="checkpoint-{:04}.pth.tar")
    argparser.add_argument("--ckpt-save-interval", type=int, default=10)

    argparser.add_argument("--loss-log-step-interval", type=int, default=50)

    argparser.add_argument("--cuda", action="store_true", default=True)
    argparser.add_argument("--height", type=int, default=512)
    argparser.add_argument("--dataset-root", type=str, default="E:\\FMI\\Thesis\\archive\\TUSimple")

    argparser.add_argument("--pretrained-encoder", type=str)

    main(argparser.parse_args())
