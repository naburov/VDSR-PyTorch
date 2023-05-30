import os

import cv2
import glob
import numpy as np
import torch

import config
import imgproc
from model import VDSR


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


def save_sr_image(hr_cb_image, hr_cr_image, sr_y_tensor, path):
    sr_y_tensor = sr_y_tensor.cpu().detach()
    sr_y_image = imgproc.tensor2image(sr_y_tensor, range_norm=False, half=False)
    sr_y_image = sr_y_image.astype(np.float32) / 255.0
    sr_ycbcr_image = cv2.merge([sr_y_image, hr_cb_image, hr_cr_image])
    sr_image = imgproc.ycbcr2bgr(sr_ycbcr_image)
    cv2.imwrite(path, sr_image * 255.0)


def calc_step_image(lr_image_tensor: torch.Tensor, hr_image_tensor: torch.Tensor, model):
    # model should be frozen here
    lr_image_tensor.requires_grad = True
    sr_image = model(lr_image_tensor).clamp_(0, 1.0)
    loss = -psnr(hr_image_tensor, sr_image)
    loss.backward()
    # get max along channel axis
    step = lr_image_tensor.grad[0]
    # print(step.size())

    # return grads
    return step, loss, sr_image


def main() -> None:
    model = VDSR().to(config.device)
    print("Build VDSR model successfully.")

    # Load the super-resolution model weights
    checkpoint = torch.load(config.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Load VDSR model weights `{os.path.abspath(config.model_path)}` successfully.")

    model.eval()

    hr_image_filenames = glob.glob(os.path.join(config.hr_dir, '*g'))
    for hr_image_path in hr_image_filenames[:config.n_images]:
        print('Processing {0}'.format(hr_image_path))
        try:
            hr_image = load_high_res_image(hr_image_path)

            # Make low-resolution image
            lr_image = imgproc.imresize(hr_image, 1 / config.upscale_factor)
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

            s, loss, sr_y_tensor = calc_step_image(lr_y_tensor, hr_y_tensor, model)
            loss = loss.cpu().detach().numpy()
            print('Vanilla loss: ', -loss)
            save_name = os.path.join(
                config.sr_dir,
                'vanilla_psnr_{0}'.format(-loss) + '_' + hr_image_path.split(os.sep)[-1]
            )
            save_sr_image(hr_cb_image, hr_cr_image, sr_y_tensor, save_name)
            with torch.no_grad():
                sr_y_tensor = model(lr_y_tensor - 0.5 * s).clamp_(0, 1.0)

            loss = psnr(hr_y_tensor, sr_y_tensor)
            loss = loss.cpu().detach().numpy()
            save_name = os.path.join(
                config.sr_dir,
                'enhanced_psnr_{0}'.format(loss) + '_' + hr_image_path.split(os.sep)[-1]
            )
            save_sr_image(hr_cb_image, hr_cr_image, sr_y_tensor, save_name)
            print('Enhanced loss: ', loss)
        except Exception as e:
            # as I run this on mac I have some memory issues
            print(e)


if __name__ == '__main__':
    main()
