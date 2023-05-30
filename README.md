# VDSR-PyTorch

### Overview

This repo forked from [VDSR PyTorch reimplementation](https://github.com/Lornatang/VDSR-PyTorch)

Idea for improving of VDSR perfomance on the given image is inspired by saliency maps for conv nets.
What if we change the input image for SRNet if we have access to the high-resolution in order to improve some criteria.

Main file is `ehnanced_validate.py` (changed version of `validate.py`).

LR - low-resolution, HR - high-resolution, SR - super-resolution

The algorithm is the following:

1) Freeze SR model;
2) Run SRNet on the low-resolution image and get SR image;
3) Compute PSNR between SR image and ground true HR image. Compute gradients of negative PSNR with respect to the
   input image.
4) Update input LR image via gradient descent.
5) Run SR again (optional).

Download files for VDSR from:

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

To run it is necessaty to change `config.py` file.

1) set `mode` to `valid`
2) change `if mode == "valid":` section accordingly
3) also in line 30: `upscale_factor` change to the magnification you need to enlarge.
4) change `device` to `cpu` or `cuda:0`

Experiments were done with DIV2K dataset. Some results are given in the /data/Results folder.
In test dataset there were ~35 images in total.

|           | Vanilla VDSR | With Enhancement |   
|-----------|--------------|------------------|
| Sum PSNR  | 1103.4       | 1108.1           |   
| Mean PSNR | 29.82        | 29.95            |  
|           |              |                  |   

And also in the silde with results.
Unfortunately, even though average PSNR have risen, difference is practically impossible to spot with the human eye,
which is might be seen from `comparision.pdf`

#### Accurate Image Super-Resolution Using Very Deep Convolutional Networks

[[Paper]](https://arxiv.org/pdf/1511.04587) [[Author's implements(MATLAB)]](https://cv.snu.ac.kr/research/VDSR/VDSR_code.zip)

```
@inproceedings{vedaldi15matconvnet,
  author    = {A. Vedaldi and K. Lenc},
  title     = {MatConvNet -- Convolutional Neural Networks for MATLAB},
  booktitle = {Proceeding of the {ACM} Int. Conf. on Multimedia},
  year      = {2015},
}
```
