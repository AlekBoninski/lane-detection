import torch
from torch.utils.data import DataLoader

from src.accuracy_eval import AccuracyEval, PixelDistanceEval
from src.dataset import TuSimple
from src.dataset.transform import ERFNetResize, ImageAndMaskToTensor
from src.models.erfnet import ERFNetLowDilation, ERFNet

NUM_CLASSES = 2


class ERFNetTester:
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

        self.dataset = DataLoader(tusimple_test, num_workers=4, batch_size=6, shuffle=True)

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

    def test(self):
        accuracy_eval = AccuracyEval()
        pixel_accuracy_eval = PixelDistanceEval()
        for step, (images, labels) in enumerate(self.dataset):
            if self.cuda:
                images = images.cuda()
                labels = labels.cuda()

            with torch.no_grad():
                output = self.model(images)

            accuracy_eval.add_batch(output, labels)
            pixel_accuracy_eval.add_batch(output, labels)

            if step % 50 == 0:
                avg_iou, avg_props = accuracy_eval()
                avg_distance, avg_inverted_distance = pixel_accuracy_eval()
                print(f"Step: {step}, IoU: {avg_iou}, Proportions: {avg_props}, Pixel distance: {avg_distance}, Inverted pixel distance: {avg_inverted_distance}")

        avg_iou, avg_props = accuracy_eval()
        avg_distance, avg_inverted_distance = pixel_accuracy_eval()
        print(f"Testing done. IoU: {avg_iou}. Props: {avg_props}. Pixel distance: {avg_distance}. Inverted pixel distance: {avg_inverted_distance}")


def test_erfnet_efficiency():
    tester = ERFNetTester(
        "E:\\FMI\\Thesis\\archive\\TUSimple",
        "E:\\FMI\\Thesis\\archive\\TUSimple\\test_label_new.json",
        # "../checkpoints_0_100/erfnet/checkpoint-0100.pth.tar",
        "../checkpoints_80_20/erfnet/checkpoint-0020.pth.tar",
        # "../checkpoints_low_dilation_0_150/erfnet/checkpoint-0100.pth.tar",
        # "../checkpoints_low_dilation_80_20/erfnet/checkpoint-0020.pth.tar",
        ERFNet,
        # ERFNetLowDilation
    )

    tester.test()


if __name__ == "__main__":
    test_erfnet_efficiency()
