import numpy as np
import cv2 as cv
import random

import dct_wat
from dct_wat import *

def hor_cutoff(wm, cutoff_size):
    for i in range(cutoff_size):
        wm[:,i] = 0
    return wm
def cutout(wm, cs):
    rows, cols = wm.shape
    rows = int(rows / 2)
    cols = int(cols / 2)
    half_cs = int(cs / 2)
    wm[rows - half_cs : rows + half_cs, cols - half_cs : cols + half_cs] = 0
    return wm

def average_wm(wms):
    filtered = []
    for i in wms:
        print(np.mean(i))
        if np.mean(i) > 5.0:
            filtered.append(i.astype(float))
    sums = sum(filtered)
    avg = sums / len(filtered)
    return avg

def gaus_noise(wm, mu, sigma):

    noise = np.random.normal(mu, sigma, np.shape(wm))
    wm = wm + noise
    return wm

def jpeg_compress(wm, cm_rate):

    encode_param = [int(cv.IMWRITE_JPEG_QUALITY), cm_rate]
    result, encimg = cv.imencode('.jpg', wm, encode_param)
    decimg = cv.imdecode(encimg, 1)
    return cv.cvtColor(decimg, cv.COLOR_RGB2GRAY)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    orig = cv.imread('data/skek.png')

    wm = cv.imread('data/logo.png')
    img = cv.cvtColor(orig, cv.COLOR_BGR2GRAY)
    w_mark = cv.cvtColor(wm, cv.COLOR_RGB2GRAY)

    #bin_wmark = cv.adaptiveThreshold(w_mark, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    bin_wmark = cv.threshold(w_mark, 200, 255, cv.THRESH_BINARY)[1]

    key = 10

    watermarked = dct_wat.dct_encode(img, bin_wmark, key, [64, 64])
    # Testing attacks
    # Cut off
    #watermarked = hor_cutoff(watermarked, 200)

    # Cutout
    watermarked = cutout(watermarked, 500)

    # Noise
    #watermarked = gaus_noise(watermarked.astype(float), 0, 6).clip(0, 255)

    #Salt and pepper

    #JPEG
    #watermarked = jpeg_compress(watermarked, 55)

    dec = dct_wat.bin_extract(watermarked, key, [64, 64])

    avg_dec = average_wm(dec)
    print("len: ", len(dec))
    res_img = abs(np.float32(watermarked) - np.float32(img))
    print(np.max(res_img), np.min(res_img))
    cv.imshow('orig', img)
    cv.imshow('watermarked', np.uint8(watermarked))
    cv.imshow('diff', np.uint8(res_img))
    cv.imshow('avg', np.uint8(avg_dec))
    cv.imshow('dec1', np.uint8(dec[0]))
    cv.imshow('dec2', np.uint8(dec[1]))
    cv.imshow('dec3', np.uint8(dec[2]))
    cv.imshow('dec4', np.uint8(dec[3]))
    cv.imshow('dec5', np.uint8(dec[4]))
    cv.imshow('dec6', np.uint8(dec[5]))
    cv.imshow('dec7', np.uint8(dec[6]))
    cv.imshow('dec8', np.uint8(dec[7]))
    cv.waitKey(0)
    cv.destroyAllWindows()
