import torch
from PIL import Image
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

from src.dataset import TuSimple
from src.dataset.transform import ERFNetResize, ImageAndMaskToTensor
from src.models import ERFNet
from src.models.erfnet import ERFNetLowDilation

NUM_CLASSES = 2
TEST_SAMPLES = 3


class ERFNetVisualizer:
    def __init__(self, dataset_root, labels_file, erfnet_checkpoint, model_class, cuda=True):
        self.dataset_root = dataset_root
        self.labels_file = labels_file
        self.erfnet_checkpoint = erfnet_checkpoint
        self.cuda = cuda

        self.model = self.__init_model(model_class)

        transformers = [
            ERFNetResize(),
            ImageAndMaskToTensor(),
        ]
        tusimple_test = TuSimple(dataset_root, labels_file, transformers)

        self.dataset = DataLoader(tusimple_test, num_workers=4, batch_size=1, shuffle=True)

    def __init_model(self, model_class):
        state_dict = torch.load(self.erfnet_checkpoint).get("model_state_dict")

        # Transform the keys to not begin with "module." because
        # for some reason this breaks loading the model
        new_state_dict = {}
        for key in state_dict.keys():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = state_dict[key]

        model = model_class(NUM_CLASSES)
        model.load_state_dict(new_state_dict)
        if self.cuda:
            model = torch.nn.DataParallel(model).cuda()

        return model

    def visualize_random_images(self, samples=TEST_SAMPLES):
        for step, (images, labels) in enumerate(self.dataset, 1):
            if self.cuda:
                images = images.cuda()
                labels = labels.cuda()

            with torch.no_grad():
                output = self.model(images)

            in_image = images.squeeze(0)
            in_image = ToPILImage()(in_image)

            label_image = labels.squeeze(0)
            label_image[label_image == 1] = 255
            label_image = label_image.to(torch.uint8)
            label_image = ToPILImage()(label_image)

            out_image = output.squeeze(0)
            out_image = F.softmax(out_image, dim=0)
            out_image = out_image[1]
            out_image = ToPILImage()(out_image)

            base_label = Image.new("RGBA", label_image.size, (255, 0, 0, 255))
            stacked_label = Image.composite(base_label, in_image, label_image)

            base_out = Image.new("RGBA", out_image.size, (255, 0, 0, 255))
            stacked_out = Image.composite(base_out, in_image, out_image)

            total_w = in_image.size[0] + stacked_label.size[0] + stacked_out.size[0]
            total_h = max(in_image.size[1], stacked_label.size[1], stacked_out.size[1])
            side_by_side = Image.new("RGB", (total_w, total_h))
            side_by_side.paste(in_image, (0, 0))
            side_by_side.paste(stacked_label, (stacked_label.size[0], 0))
            side_by_side.paste(stacked_out, (stacked_label.size[0] + stacked_out.size[0], 0))

            side_by_side.show()

            if step == samples:
                break


def test_random_images():
    visualizer = ERFNetVisualizer(
        "E:\\FMI\\Thesis\\archive\\TUSimple",
        "E:\\FMI\\Thesis\\archive\\TUSimple\\test_label_new.json",
        "../checkpoints_low_dilation_0_150/erfnet/best.pth.tar",
        # ERFNet,
        ERFNetLowDilation,
    )

    # visualizer = ERFNetVisualizer(
    #     "E:\\FMI\\Thesis\\archive\\TUSimple",
    #     "E:\\FMI\\Thesis\\archive\\TUSimple\\test_label_new.json",
    #     "../checkpoints_test/erfnet/best.pth.tar",
    # )

    visualizer.visualize_random_images()


if __name__ == "__main__":
    test_random_images()
