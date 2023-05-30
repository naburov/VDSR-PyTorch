import os

import cv2
import numpy as np
import torch
from natsort import natsorted

import config
import imgproc
from model import VDSR
from plotting_utils import plot_minmax_normalized, plot_image
import torch.nn.functional as nnf


def load_high_res_image(hr_image_path):
    hr_image = cv2.imread(hr_image_path).astype(np.float32) / 255.0
    hr_image_height, hr_image_width = hr_image.shape[:2]
    hr_image_height_remainder = hr_image_height % 12
    hr_image_width_remainder = hr_image_width % 12
    hr_image = hr_image[:hr_image_height - hr_image_height_remainder, :hr_image_width - hr_image_width_remainder, ...]
    return hr_image


def load_low_res_image(lr_image_path):
    lr_image = cv2.imread(lr_image_path).astype(np.float32) / 255.0
    return lr_image


def psnr(true_hr_image, pred_hr_image):
    return 10. * torch.log10(1. / torch.mean((true_hr_image - pred_hr_image) ** 2))


def enhance_image(lr_image_tensor: torch.Tensor, hr_image_tensor: torch.Tensor, model, alpha):
    return alpha * calc_step_image(lr_image_tensor, hr_image_tensor, model)


def calc_step_image(lr_image_tensor: torch.Tensor, hr_image_tensor: torch.Tensor, model):
    # model should be frozen here
    lr_image_tensor.requires_grad = True
    sr_image = model(lr_image_tensor).clamp_(0, 1.0)
    loss = -psnr(hr_image_tensor, sr_image)
    loss.backward()
    # get max along channel axis
    step = lr_image_tensor.grad[0]
    print(step.size())

    # return grads
    return step, loss


lr_image_path = r'/Users/burovnikita/PycharmProjects/SREnhancement/VDSR-PyTorch/data/DIV2K_train_LR_wild/0001x4w1.png'
hr_image_path = r'/Users/burovnikita/PycharmProjects/SREnhancement/VDSR-PyTorch/data/DIV2K_train_HR/0001.png'


def main() -> None:
    model = VDSR().to(config.device)
    print("Build VDSR model successfully.")

    # Load the super-resolution model weights
    checkpoint = torch.load(config.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Load VDSR model weights `{os.path.abspath(config.model_path)}` successfully.")

    model.eval()

    hr_image = load_high_res_image(hr_image_path)

    # Make low-resolution image
    lr_image = imgproc.imresize(hr_image, 1 / config.upscale_factor)
    h, w = lr_image.shape[:2]
    lr_image = imgproc.imresize(lr_image, config.upscale_factor)

    # Convert BGR image to YCbCr image
    lr_ycbcr_image = imgproc.bgr2ycbcr(lr_image, use_y_channel=False)
    hr_ycbcr_image = imgproc.bgr2ycbcr(hr_image, use_y_channel=False)

    # Split YCbCr image data
    lr_y_image, lr_cb_image, lr_cr_image = cv2.split(lr_ycbcr_image)
    hr_y_image, hr_cb_image, hr_cr_image = cv2.split(hr_ycbcr_image)

    # Convert Y image data convert to Y tensor data
    lr_y_tensor = imgproc.image2tensor(lr_y_image, range_norm=False, half=False).to(config.device).unsqueeze_(0)
    hr_y_tensor = imgproc.image2tensor(hr_y_image, range_norm=False, half=False).to(config.device).unsqueeze_(0)
    print(lr_y_tensor.shape, hr_y_tensor.shape)
    s, loss = calc_step_image(lr_y_tensor, hr_y_tensor, model)
    print(loss)

    s = s.cpu().detach().unsqueeze_(0)
    change = nnf.interpolate(s, size=(h, w), mode='bicubic', align_corners=False)
    print(change.size())
    lr_image = imgproc.imresize(hr_image, 1 / config.upscale_factor)
    lr_ycbcr_image = imgproc.bgr2ycbcr(lr_image, use_y_channel=False)
    lr_y_image, lr_cb_image, lr_cr_image = cv2.split(lr_ycbcr_image)
    lr_y_image = lr_y_image + change
    lr_y_tensor = imgproc.image2tensor(lr_y_image, range_norm=False, half=False).to(config.device).unsqueeze_(0)

    # plot_minmax_normalized(s.cpu())
    sr_image = model(lr_y_tensor).clamp_(0, 1.0)
    loss = psnr(hr_y_tensor, sr_image)
    print(loss)
    plot_image(sr_image.cpu().detach())


if __name__ == '__main__':
    main()
