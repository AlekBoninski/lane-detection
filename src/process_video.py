import os
import time

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage, Resize, InterpolationMode, ToTensor

from src.models import ERFNet
from src.models.erfnet import ERFNetLowDilation

path = "../test_videos"
video_file = "IMG_6504.MOV"

def __get_trained_model(checkpoint="../checkpoints_low_dilation_0_150/erfnet/checkpoint-0100.pth.tar"):
    state_dict = torch.load(checkpoint).get("model_state_dict")
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = state_dict[key]

    model = ERFNetLowDilation(2)
    model.load_state_dict(new_state_dict)
    model = torch.nn.DataParallel(model).cuda()

    return model

# checkpoint = "../checkpoints_150_130/erfnet/checkpoint-0130.pth.tar"
#
# state_dict = torch.load(checkpoint).get("model_state_dict")
# new_state_dict = {}
# for key in state_dict.keys():
#
#     new_key = key.replace("module.", "")
#     new_state_dict[new_key] = state_dict[key]
#
# model = ERFNet(2)
# model.load_state_dict(new_state_dict)
# model = torch.nn.DataParallel(model).cuda()

model = __get_trained_model()

transformers = [
    Resize((512, 1024), InterpolationMode.BILINEAR),
    ToTensor(),
]


def crop_video(crop_top=700, crop_bottom=1200):
    video_path = os.path.join(path, video_file)
    video = cv2.VideoCapture(video_path)

    # fourcc = int(video.get(cv2.CAP_PROP_FOURCC))
    fourcc = cv2.VideoWriter.fourcc("m", "p", "4", "v")
    fps = video.get(cv2.CAP_PROP_FPS)
    og_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    height = min(int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), (crop_top + (og_height - crop_bottom)))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    out = os.path.join(path, f"{video_file.split('.')[0]}_resized.{video_file.split('.')[1]}")
    out_with_lanes = os.path.join(path, f"{video_file.split('.')[0]}_labelled.{video_file.split('.')[1]}")

    writer = cv2.VideoWriter(out, fourcc, fps, (width, height))
    writer_lanes = cv2.VideoWriter(out_with_lanes, fourcc, fps, (1024, 512))

    process_times = []

    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break

        cropped_frame = frame[:crop_bottom, :, :]
        cropped_frame = cropped_frame[crop_top:, :, :]
        print(f"No lanes shape: {cropped_frame.shape}")

        writer.write(cropped_frame)

        model_input = cropped_frame
        model_input = cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB)
        model_input = Image.fromarray(model_input)

        for transformer in transformers:
            model_input = transformer(model_input)

        frame_pil = ToPILImage()(model_input)

        model_input = model_input.unsqueeze(0)
        model_input = model_input.cuda()

        start = time.time()
        with torch.no_grad():
            output = model(model_input)
        end = time.time()
        process_times.append(end - start)

        output = output.squeeze(0)
        output = F.softmax(output, dim=0)
        output = ToPILImage()(output)

        base = Image.new("RGBA", output.size, (255, 0, 0, 255))
        stacked = Image.composite(base, frame_pil, output)

        stacked_opencv = stacked
        stacked_opencv = np.array(stacked_opencv)
        stacked_opencv = cv2.cvtColor(stacked_opencv, cv2.COLOR_RGB2BGR)
        print(f"Lanes: shape: {stacked_opencv.shape}")

        writer_lanes.write(stacked_opencv)

    video.release()
    writer.release()
    writer_lanes.release()

    avg_processing_time = sum(process_times)/len(process_times)
    print(f"AVG PROCESS: {avg_processing_time}. IN FPS: {fps}")


if __name__ == "__main__":
    crop_video()

# preprocessed_video_path = "../_test_videos_old/processed.mov"
# preprocessed_video = cv2.VideoCapture(preprocessed_video_path)
#
# _, frame = preprocessed_video.read()
# preprocessed_video.release()
#
# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# frame = Image.fromarray(frame)
# frame.show()
#
# model_input = frame
#
# for transformer in transformers:
#     model_input = transformer(model_input)
#
# model_input = model_input.unsqueeze(0)
# model_input = model_input.cuda()
#
# with torch.no_grad():
#     output = model(model_input)
#
# output = output.squeeze(0)
# output = F.softmax(output, dim=0)
# output = ToPILImage()(output)
#
# output.show()
#
# base = Image.new("RGBA", output.size, (0, 255, 0, 255))
# stacked = Image.composite(base, frame, output)
#
# stacked_opencv = stacked
# stacked_opencv = np.array(stacked_opencv)
# stacked_opencv = cv2.cvtColor(stacked_opencv, cv2.COLOR_RGB2BGR)
#
# cv2.imshow("image", stacked_opencv)
# cv2.waitKey(0)
# print(stacked_opencv.shape)
