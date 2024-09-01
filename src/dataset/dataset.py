import torch
import os
import json

from torch.utils.data import Dataset
from PIL import Image, ImageDraw

from src.dataset.transform import ImageAndMaskToTensor


class TuSimple(Dataset):
    def __init__(self, root_dir, labels_json, transformers=None):
        self.annotations = [json.loads(line) for line in open(labels_json)]

        self.root_dir = root_dir
        self.labels_json = labels_json
        self.transformers = transformers if transformers else [ImageAndMaskToTensor()]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir, self.annotations[index]["raw_file"])
        lanes = self.annotations[index]["lanes"]
        h_samples = self.annotations[index]["h_samples"]

        image = Image.open(image_path)
        mask = self.__build_mask(image, lanes, h_samples)

        for transformer in self.transformers:
            image, mask = transformer(image, mask)

        return image, mask

    def __build_mask(self, image, lanes, h_samples):
        mask = Image.new("L", (image.width, image.height), 0)
        draw = ImageDraw.Draw(mask)

        for lane in lanes:
            if len(lane) == 0:
                continue
            lane_line = [(lane[i], h_samples[i]) for i in range(len(lane)) if lane[i] != -2]
            draw.line(lane_line, fill=255, width=5)

        return mask


class TuSimpleV2(Dataset):
    def __init__(self, root_dir, labels_json, transformers=None):
        self.annotations = [json.loads(line) for line in open(labels_json)]

        self.root_dir = root_dir
        self.labels_json = labels_json
        self.transformers = transformers if transformers else [ImageAndMaskToTensor()]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir, self.annotations[index]["raw_file"])
        lanes = self.annotations[index]["lanes"]
        h_samples = self.annotations[index]["h_samples"]

        image = Image.open(image_path)
        mask = self.__build_mask(image, lanes, h_samples)

        for transformer in self.transformers:
            image, mask = transformer(image, mask)

        return image, mask

    def __build_mask(self, image, lanes, h_samples):
        mask = Image.new("L", (image.width, image.height), 0)
        draw = ImageDraw.Draw(mask)

        for lane in lanes:
            if len(lane) == 0:
                continue
            lane_line = [(lane[i], h_samples[i]) for i in range(len(lane)) if lane[i] != -2]
            draw.line(lane_line, fill=255, width=10)

        return mask
