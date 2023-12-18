from scipy.fftpack import dct, idct
import os
import shutil
import numpy as np
import paddle
from paddle.io import Dataset
from paddle.vision.datasets import DatasetFolder, ImageFolder
from paddle.vision.transforms import Compose, Resize, Transpose, Normalize
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pywt
import paddle.vision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
mean,std = ([0.5, 0.5,0.5],[0.5, 0.5,0.5])
mean = list(map(lambda x:x*255,mean))
std = list(map(lambda x:x*255,std))
dirpath="work/64.png"
img_PIL = Image.open(dirpath)
img_PIL = np.array(img_PIL)
print("img_PIL:",img_PIL.shape)
data=np.array(img_PIL).astype(np.float32)
data=data.reshape(64,64,3)
four=np.array(data)
print(data)
target1=[1]

dirpath2="work/blend.png"
img_PIL2 = Image.open(dirpath2)
img_PIL2 = np.array(img_PIL2)

blend=np.array(img_PIL2).astype(np.float32)
#cv2.normalize(data,data,-1,1,cv2.NORM_MINMAX)
blend=blend.reshape(32,32,3)
blend = cv2.resize(blend, (64,64))
blend = np.array(blend).reshape(64,64,3).astype(np.float32)

data1 = cv2.resize(data, (18,18))
mask11 = np.array(data1).reshape(3,18,18).astype(np.float32)
data2 = cv2.resize(data, (64,64))
mask2 = np.array(data2).reshape(3,64,64).astype(np.float32)


def dct2 (block):
    return dct(dct(block.T, norm = 'ortho').T, norm = 'ortho')
def idct2(block):
    return idct(idct(block.T, norm = 'ortho').T, norm = 'ortho')
def setwaterMark(Img):
    c = pywt.wavedec2(Img,'db2',level=3)
    [cl,(cH3,cV3,cD3),(cH2,cV2,cD2),(cH1,cV1,cD1)] = c
    d = pywt.wavedec2(mask11, 'db2', level=1)
    [ca1, (ch1, cv1, cd1)] = d
    e = pywt.wavedec2(mask2, 'db2', level=2)
    [cb1, (chb1, cvb1, cdb1),(chb2, cvb2, cdb2)] = e
    a1 = 0
    a2 = 0.6
    a3 = 0.6
    a4 = 0.6
    cl = cl + ca1 * a1
    cH3 = cH3 + ch1 * a2
    cV3 =  cv1 * a3
    cD3 =  cd1 * a4
    cH2 = cH2+ chb1 * a2
    cV2 =  cvb1 * a3
    cD2 = cdb1 * a4
    newImg = pywt.waverec2([cl,(cH3,cV3,cD3),(cH2,cV2,cD2),(cH1,cV1,cD1)],'db2')
    newImg = np.array(newImg,np.float32)
    return newImg
def setwaterMark2(Img):
    c = pywt.wavedec2(Img,'db2',level=3)
    [cl,(cH3,cV3,cD3),(cH2,cV2,cD2),(cH1,cV1,cD1)] = c
    d = pywt.wavedec2(mask11, 'db2', level=1)
    [ca1, (ch1, cv1, cd1)] = d
    e = pywt.wavedec2(mask2, 'db2', level=2)
    [cb1, (chb1, cvb1, cdb1),(chb2, cvb2, cdb2)] = e
    a1 = 0
    a2 = 0.8
    a3 = 0.8
    a4 = 0.8

    cl = cl + ca1 * a1
    cH3 = cH3 + ch1 * a2
    cV3 =  cv1 * a3
    cD3 = cd1 * a4

    cH2 = cH2*(1-a2) + chb1 * a2
    cV2 = cvb1 * a3
    cD2 =  cdb1 * a4

    newImg = pywt.waverec2([cl,(cH3,cV3,cD3),(cH2,cV2,cD2),(cH1,cV1,cD1)],'db2')
    newImg = np.array(newImg,np.float32)
    return newImg

import paddle.nn.functional as F
import paddle
ins = paddle.rand([1, 2,4, 4]) * 2 - 1
ins = ins / paddle.mean(paddle.abs(ins))

noise_grid = (
            F.upsample(ins, size=[64,64], mode="BICUBIC", align_corners=True)
            # .permute(0, 2, 3, 1)
        )
noise_grid=noise_grid.transpose([0,2,3,1])
array1d = paddle.linspace(-1, 1, num=64)
x, y = paddle.meshgrid(array1d, array1d)
identity_grid = paddle.stack([y, x], 2)
identity_grid=paddle.reshape(identity_grid, [1,64,64,2])
grid_temps = (identity_grid + 0.5 * noise_grid / 64) * 1.0
grid_temps = paddle.clip(grid_temps, -1, 1)
def WaNet(inputs):
    inputs=paddle.to_tensor(inputs)
    inputs_bd = F.grid_sample(inputs, grid_temps, align_corners=True)
    inputs_bd = paddle.squeeze(inputs_bd, axis=0)

    # inputs_bd=inputs_bd.transpose([2,1,0])
    inputs_bd=paddle.reshape(inputs_bd, [64,64,3])
    return inputs_bd.numpy()

def FIBA(img_,target_img,beta,ratio):
    img_=np.asarray(img_)
    target_img=np.asarray(target_img)
    #  get the amplitude and phase spectrum of trigger image
    fft_trg_cp = np.fft.fft2(target_img, axes=(-3, -2))
    amp_target, pha_target = np.abs(fft_trg_cp), np.angle(fft_trg_cp)
    amp_target_shift = np.fft.fftshift(amp_target, axes=(-3, -2))
    #  get the amplitude and phase spectrum of source image
    fft_source_cp = np.fft.fft2(img_, axes=(-3, -2))
    amp_source, pha_source = np.abs(fft_source_cp), np.angle(fft_source_cp)
    amp_source_shift = np.fft.fftshift(amp_source, axes=(-3, -2))
    # swap the amplitude part of local image with target amplitude spectrum
    c, h, w = img_.shape
    b = (np.floor(np.amin((h, w)) * beta)).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)

    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    amp_source_shift[:, h1:h2, w1:w2] = amp_source_shift[:, h1:h2, w1:w2] * (1 - ratio) + (amp_target_shift[:,h1:h2, w1:w2]) * ratio
    # IFFT
    amp_source_shift = np.fft.ifftshift(amp_source_shift, axes=(-3, -2))

    # get transformed image via inverse fft
    fft_local_ = amp_source_shift * np.exp(1j * pha_source)
    local_in_trg = np.fft.ifft2(fft_local_, axes=(-3, -2))
    local_in_trg = np.real(local_in_trg)

    return local_in_trg

def Blend(m2):
    return m2*0.6+blend*0.4
def BadNets(m2):
    m2[55:63, 55:63, ] = 0
    return m2

def DUBA(data):
    m = np.array(data).reshape(3, 64, 64).astype(np.float32)
    m2 = np.array(data).reshape(64, 64, 3).astype(np.float32)
    new=setwaterMark(m)
    img_np = new.reshape(64,64,3).astype(np.float32)
    fft_trg_cp = np.fft.fft2(img_np, axes=(-3, -2))
    amp_target, pha_target = np.abs(fft_trg_cp), np.angle(fft_trg_cp)
    amp_target_shift = np.fft.fftshift(amp_target, axes=(-3, -2)).astype(np.float32)

    fft_source_cp = np.fft.fft2(m2, axes=(-3, -2))
    amp_source, pha_source = np.abs(fft_source_cp), np.angle(fft_source_cp)
    amp_source_shift = np.fft.fftshift(amp_source, axes=(-3, -2))
    amp_source_shift = amp_source_shift
    # IFFT
    amp_source_shift = np.fft.ifftshift(amp_source_shift, axes=(-3, -2))
    # get transformed image via inverse fft
    fft_local_ = amp_source_shift * np.exp(1j * pha_target)
    local_in_trg = np.fft.ifft2(fft_local_, axes=(-3, -2))
    local_in_trg = np.real(local_in_trg).reshape(64,64,3).astype(np.float32)
    local_in_trg[m2>=190]=m2[m2>=190]
    local_in_trg[m2<=60]=m2[m2<=60]
    img_np = T.Transpose()(local_in_trg.astype("float32"))
    img_np = T.Normalize(mean, std)(img_np)
    return img_np

def DUBA2(data):
    m = np.array(data).reshape(3, 64, 64).astype(np.float32)
    m2 = np.array(data).reshape(64, 64, 3).astype(np.float32)
    new=setwaterMark2(m)
    img_np = new.reshape(64,64,3).astype(np.float32)

    fft_trg_cp = np.fft.fft2(img_np, axes=(-3, -2))
    amp_target, pha_target = np.abs(fft_trg_cp), np.angle(fft_trg_cp)
    amp_target_shift = np.fft.fftshift(amp_target, axes=(-3, -2)).astype(np.float32)

    fft_source_cp = np.fft.fft2(m2, axes=(-3, -2))
    amp_source, pha_source = np.abs(fft_source_cp), np.angle(fft_source_cp)
    amp_source_shift = np.fft.fftshift(amp_source, axes=(-3, -2))
    amp_source_shift = amp_source_shift
    # IFFT
    amp_source_shift = np.fft.ifftshift(amp_source_shift, axes=(-3, -2))
    # get transformed image via inverse fft
    fft_local_ = amp_source_shift * np.exp(1j * pha_target)
    local_in_trg = np.fft.ifft2(fft_local_, axes=(-3, -2))
    local_in_trg = np.real(local_in_trg).reshape(64,64,3).astype(np.float32)
    local_in_trg[m2>=230]=m2[m2>=230]
    local_in_trg[m2<=20]=m2[m2<=20]
    img_np = T.Transpose()(local_in_trg.astype("float32"))
    img_np = T.Normalize(mean, std)(img_np)
    return img_np


