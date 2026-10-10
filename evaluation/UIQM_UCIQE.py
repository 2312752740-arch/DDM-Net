### UCIQE: https://ieeexplore.ieee.org/abstract/document/7300447
### UIQM https://ieeexplore.ieee.org/abstract/document/7305804
import numpy as np
import math
import os
import argparse

from scipy import ndimage
import cv2
from PIL import Image
from tqdm import tqdm


# ========== UIQM helpers ==========

def _uiconm(x, window_size):
    k1 = x.shape[1]/window_size
    k2 = x.shape[0]/window_size
    w = -1./(k1*k2)
    blocksize_x = window_size
    blocksize_y = window_size
    x = x[0:int(blocksize_y*k2), 0:int(blocksize_x*k1)]
    alpha = 1
    val = 0
    k1 = int(k1)
    k2 = int(k2)
    for l in range(k1):
        for k in range(k2):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1), :]
            max_ = np.max(block)
            min_ = np.min(block)
            top = max_-min_
            bot = max_+min_
            if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0:
                val += 0.0
            else:
                val += alpha*math.pow((top/bot),alpha) * math.log(top/bot)
    return w*val


def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    x = sorted(x)
    K = len(x)
    T_a_L = math.ceil(alpha_L*K)
    T_a_R = math.floor(alpha_R*K)
    weight = (1/(K-T_a_L-T_a_R))
    s = int(T_a_L+1)
    e = int(K-T_a_R)
    val = sum(x[s:e])
    return weight*val


def s_a(x, mu):
    val = 0
    for pixel in x:
        val += math.pow((pixel-mu), 2)
    return val/len(x)


def _uicm(x):
    R = x[:,:,0].flatten()
    G = x[:,:,1].flatten()
    B = x[:,:,2].flatten()
    RG = R-G
    YB = ((R+G)/2)-B
    mu_a_RG = mu_a(RG)
    mu_a_YB = mu_a(YB)
    s_a_RG = s_a(RG, mu_a_RG)
    s_a_YB = s_a(YB, mu_a_YB)
    l = math.sqrt( (math.pow(mu_a_RG,2)+math.pow(mu_a_YB,2)) )
    r = math.sqrt(s_a_RG+s_a_YB)
    return (-0.0268*l)+(0.1586*r)


def sobel(x):
    dx = ndimage.sobel(x, 0)
    dy = ndimage.sobel(x, 1)
    mag = np.hypot(dx, dy)
    if np.max(mag) == 0:
        return np.zeros_like(mag)
    mag *= 255.0 / np.max(mag)
    return mag


def eme1(x, window_size):
    k1 = x.shape[1]/window_size
    k2 = x.shape[0]/window_size
    w = 2./(k1*k2)
    blocksize_x = window_size
    blocksize_y = window_size
    x = x[0:int(blocksize_y*k2), 0:int(blocksize_x*k1)]
    val = 0
    k1 = int(k1)
    k2 = int(k2)
    for l in range(k1):
        for k in range(k2):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1)]
            max_ = np.max(block)
            min_ = np.min(block)
            if min_ == 0.0: val += 0
            elif max_ == 0.0: val += 0
            else: val += math.log(max_/min_)
    return w*val


def _uism(x):
    R = x[:,:,0]
    G = x[:,:,1]
    B = x[:,:,2]
    Rs = sobel(R)
    Gs = sobel(G)
    Bs = sobel(B)
    R_edge_map = np.multiply(Rs, R)
    G_edge_map = np.multiply(Gs, G)
    B_edge_map = np.multiply(Bs, B)
    r_eme = eme1(R_edge_map, 10)
    g_eme = eme1(G_edge_map, 10)
    b_eme = eme1(B_edge_map, 10)
    return 0.299*r_eme + 0.587*g_eme + 0.144*b_eme


def getUIQM1(x):
    x = np.asarray(x).astype(np.float32)
    c1 = 0.0282
    c2 = 0.2953
    c3 = 3.5753
    uicm   = _uicm(x)
    uism   = _uism(x)
    uiconm = _uiconm(x, 10)
    return (c1*uicm) + (c2*uism) + (c3*uiconm)


# ========== UCIQE helper ==========

def getUCIQE2(loc):
    img_bgr = cv2.imread(loc)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    coe_metric = [0.4680, 0.2745, 0.2576]
    img_lum = img_lab[..., 0]/255
    img_a   = img_lab[..., 1]/255
    img_b   = img_lab[..., 2]/255
    img_chr = np.sqrt(np.square(img_a)+np.square(img_b))
    img_sat = img_chr/np.sqrt(np.square(img_chr)+np.square(img_lum))
    aver_sat = np.mean(img_sat)
    aver_chr = np.mean(img_chr)
    var_chr  = np.sqrt(np.mean(abs(1-np.square(aver_chr/img_chr))))
    nbins = 256 if img_lum.dtype == np.uint8 else 65536
    hist, bins = np.histogram(img_lum, nbins)
    cdf = np.cumsum(hist)/np.sum(hist)
    ilow  = np.where(cdf > 0.0100)
    ihigh = np.where(cdf >= 0.9900)
    tol = [(ilow[0][0]-1)/(nbins-1), (ihigh[0][0]-1)/(nbins-1)]
    con_lum = tol[1]-tol[0]
    return coe_metric[0]*var_chr + coe_metric[1]*con_lum + coe_metric[2]*aver_sat


# ========== Single-directory evaluation ==========

def calculate_path(path):
    all_images = [f for f in os.listdir(path)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    sum_uiqm, sum_uciqe = 0.0, 0.0
    num = 0

    for item in tqdm(all_images, desc="Evaluating"):
        impath = os.path.join(path, item)
        try:
            imgRGB = Image.open(impath).convert("RGB")
            uiqm  = getUIQM1(np.array(imgRGB))
            uciqe = getUCIQE2(impath)
            sum_uiqm  += uiqm
            sum_uciqe += uciqe
            num += 1
        except Exception as e:
            print(f"Error {impath}: {e}")

    if num == 0:
        print("No images found!")
        return 0, 0

    return sum_uiqm / num, sum_uciqe / num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    args = parser.parse_args()

    muiqm, muciqe = calculate_path(args.input_dir)
    print(f"UIQM: {muiqm:.4f} | UCIQE: {muciqe:.4f}")