import numpy as np
import cv2 as cv
import random


def resize_wm(orig, wm, bsize):
    sz = orig.shape
    min_size = min(sz)
    res = int(min_size / bsize)
    return cv.resize(wm, [res, res])


def single_embeding(orig, wm, alpha, bsize):
    # Resizes the watermark to fit orig image
    res_wm = resize_wm(orig, wm, bsize)
    # Flattens the watermark
    rav_wm = res_wm.ravel()
    # Variable to store final img
    embed_img = np.zeros(orig.shape, dtype=np.float32)

    wm_idx = 0
    # Iterate over each block in original image and embed watermark pixel
    for i in range(0, orig.shape[0], bsize):
        for j in range(0, orig.shape[1], bsize):
            # Extract one block and convert it with DCT
            block = orig[i:i + bsize, j:j + bsize]
            dct_block = cv.dct(np.float32(block))
            # Embed watermark pixel to predetermined location
            # Alpha is scaling factor
            dct_block[0][0] = dct_block[0][0] + alpha * np.float32(rav_wm[wm_idx])
            wm_idx += 1
            # Do invers DCT to convert it back to normal image pixels
            idct_block = cv.idct(dct_block)
            # Store it in final image
            embed_img[i:i + bsize, j:j + bsize] = idct_block
    return embed_img


def random_embeding(orig, wm, key, alpha, bsize):
    # Resizes the watermark to fit orig image
    res_wm = resize_wm(orig, wm, bsize)
    # Flattens the watermark
    rav_wm = res_wm.ravel()
    # Variable to store final img
    embed_img = np.zeros(orig.shape, dtype=np.float32)
    random.seed(key)
    wm_idx = 0
    # Iterate over each block in original image and embed watermark pixel
    for i in range(0, orig.shape[0], bsize):
        for j in range(0, orig.shape[1], bsize):
            # Extract one block and convert it with DCT
            block = orig[i:i + bsize, j:j + bsize]
            dct_block = cv.dct(np.float32(block))
            # Embed watermark pixel to random location
            # Alpha is scaling factor
            idx_x = random.randint(0, bsize-1)
            idx_y = random.randint(0, bsize-1)
            dct_block[idx_x][idx_y] = dct_block[idx_x][idx_y] + alpha * np.float32(rav_wm[wm_idx])
            wm_idx += 1
            # Do invers DCT to convert it back to normal image pixels
            idct_block = cv.idct(dct_block)
            # Store it in final image
            embed_img[i:i + bsize, j:j + bsize] = idct_block
    return embed_img


def window_embeding(orig, wm, alpha, bsize):
    # Resize watermark to size of image
    res_wm = cv.resize(wm, orig.shape)
    # Iterate over each block in original image and embed watermark block

    # Variable to store final img
    embed_img = np.zeros(orig.shape, dtype=np.float32)
    for i in range(0, orig.shape[0], bsize):
        for j in range(0, orig.shape[1], bsize):
            # Extract one block and convert it with DCT
            block = orig[i:i + bsize, j:j + bsize]
            dct_block = cv.dct(np.float32(block))

            wm_block = res_wm[i:i + bsize, j:j + bsize]

            dct_block = dct_block + alpha * wm_block

            idct_block = cv.idct(dct_block)

            # Store it in final image
            embed_img[i:i + bsize, j:j + bsize] = idct_block
    return embed_img


def dct_watermark(orig, wm, embeding='SINGLE', alpha=0.01, par1=1):
    _block_size = 8
    embed_img = np.zeros(orig.shape)

    if embeding == 'SINGLE':
        embed_img = single_embeding(orig, wm, alpha, _block_size)
    elif embeding == 'RANDOM':
        embed_img = random_embeding(orig, wm, par1, alpha, _block_size)
    elif embeding == 'WINDOW':
        embed_img = window_embeding(orig, wm, alpha, _block_size)
    else:
        print("Invalid embeding method")

    embed_img = np.uint8(embed_img)
    return embed_img


def cal_wm_shape(img, bsize):
    sz = img.shape
    min_size = min(sz)
    res = int(min_size / bsize)
    return res, res


def dct_decode(wm, orig, embeding='SINGLE', alpha=0.1, param1=1):
    _block_size = 8
    wm_shape = cal_wm_shape(orig, _block_size)
    watermark = np.zeros(wm_shape[0] * wm_shape[1])
    wm_idx = 0

    if embeding == 'RANDOM':
        random.seed(param1)

    for i in range(0, wm.shape[0], _block_size):
        for j in range(0, wm.shape[1], _block_size):

            orig_block = orig[i:i + _block_size, j:j + _block_size]
            wm_block = wm[i:i + _block_size, j:j + _block_size]

            dct_orig = cv.dct(np.float32(orig_block))
            dct_wm = cv.dct(np.float32(wm_block))

            wm_val = 0
            if embeding == 'SINGLE':
                wm_val = np.float32(dct_wm[0][0] - dct_orig[0][0]) / alpha
            elif embeding == 'RANDOM':
                ind_x = random.randint(0, _block_size-1)
                ind_y = random.randint(0, _block_size-1)
                wm_val = np.float32(dct_wm[ind_x][ind_y] - dct_orig[ind_x][ind_y]) / alpha

            watermark[wm_idx] = wm_val
            wm_idx += 1
    watermark = watermark.clip(0, 255).reshape(wm_shape)
    return watermark

def assign_to_block(block, part_wm):
    # Iterate over 8 pixels of wm and store it in dct coefficient
    # The modified coefficient are:
    # ┌                  ┐
    # │  - - - 7 - - - - │
    # │  - - 5 6 - - - - │
    # │  - 3 4 - - - - - │
    # │  1 2 - - - - - - │
    # │  0 - - - - - - - │
    # │  - - - - - - - - │
    # │  - - - - - - - - │
    # │  - - - - - - - - │
    # └                  ┘
    #
    for i in range(0,len(part_wm)):
        delta = 8
        # Calculate indexes
        x_idx = 4 - int((i + 1) / 2)
        y_idx = int(i / 2)
        # If wm value is white
        if part_wm[i] == 255:
            # Store it as odd number
            block[x_idx][y_idx] = (round(block[x_idx][y_idx] / delta) | 1) * delta
        else:
            # If value is black store it as even
            quant = round(block[x_idx][y_idx] / delta)
            quant = quant >> 1
            quant = quant << 1
            block[x_idx][y_idx] = quant * delta
    return block


def bin_watermark(orig, wm, key):
    _block_size = 8
    # Resize the watermark so it fits
    res_wm = cv.resize(wm, [64, 64])
    # Output img
    embed_img = np.zeros(orig.shape, dtype=np.float32)
    # Shuffle the watermark to mitigate spatial correlation

    rav_wm = res_wm.ravel()
    order = list(range(len(rav_wm)))
    random.seed(key)
    random.shuffle(order)
    print(len(order))
    rav_wm = [rav_wm[i] for i in order]
    print(rav_wm)
    #############################################################
    #cv.imshow('scrambled', rav_wm.reshape([64, 64]))
    #############################################################
    # Counter to help with indexing
    counter = 0
    # Store several watermarks to fit the whole image
    for i in range(0, orig.shape[0], _block_size):
        for j in range(0, orig.shape[1], _block_size):

            # Extract one block from image
            block = orig[i:i + _block_size, j:j + _block_size]
            # Do dct
            dct_block = cv.dct(np.float32(block))
            # Embed watermark into image
            wm_block = assign_to_block(dct_block, rav_wm[counter:counter + 8])
            counter += 8
            # Inverse dct to recreate img with watermark
            idct_block = cv.idct(wm_block)
            # Store the block in appropriate position
            embed_img[i:i + _block_size, j:j + _block_size] = idct_block

            # Reset the indexing
            if counter >= len(rav_wm):
                counter = 0
    # Convert to uint
    return np.uint8(embed_img)

def extract_from_block(block):
    delta = 8
    wm_part = []
    for i in range(0,8):
        x_idx = 4 - int((i + 1) / 2)
        y_idx = int(i / 2)
        if round(block[x_idx][y_idx] / delta) % 2 == 1:
            wm_part.append(255)
        else:
            wm_part.append(0)
    return wm_part


def inverse_reordering(decoded, order):
    l_out = [0] * len(decoded)
    for i, j in enumerate(order):
        l_out[j] = decoded[i]
    return np.array(l_out)


def bin_extract(orig, key):
    _block_size = 8
    # Recreate the shuffle order for backward reshuffling
    order = list(range(4096))
    random.seed(key)
    random.shuffle(order)
    print(len(order))
    # Return values
    wms = []
    decoded = np.zeros(64*64)
    counter = 0
    for i in range(0, orig.shape[0], _block_size):
        for j in range(0, orig.shape[1], _block_size):

            # Extract one block![](data/fit.jpg)
            block = orig[i:i + _block_size, j:j + _block_size]
            # Do dct
            dct_block = cv.dct(np.float32(block))
            # Extract part of watermark from block
            wm_block = extract_from_block(dct_block)
            # Save to tmp array
            decoded[counter:counter + 8] = wm_block
            # Increment
            counter += 8
            # When we decoded one watermark save it and search for more
            if counter >= 4096:
                counter = 0
                reordered = inverse_reordering(decoded, order)
                # For rounding error clip the values and then reshape to wm shape
                b = reordered.clip(0, 255).reshape([64, 64])
                # Append to all wms
                wms.append(np.uint8(b))
                decoded = np.zeros(64 * 64)
    return wms


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    orig = cv.imread('data/lena.png')
    wm = cv.imread('data/logo.png')
    img = cv.cvtColor(orig, cv.COLOR_BGR2GRAY)
    w_mark = cv.cvtColor(wm, cv.COLOR_RGB2GRAY)

    #bin_wmark = cv.adaptiveThreshold(w_mark, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    bin_wmark = cv.threshold(w_mark, 200, 255, cv.THRESH_BINARY)[1]

    key = 10
    print(int(1/2))
    #watermarked = dct_watermark(img, w_mark, 'RANDOM', alpha=0.1, par1=key)
    watermarked = bin_watermark(img, bin_wmark, key)
    # Testing attacks
    # Scale attack

    watermarked = cv.resize(watermarked,[512, 512])
    watermarked = cv.resize(watermarked,[512, 512])
    # Cutout
    #watermarked[300:, 300:] = 0


    dec = bin_extract(watermarked, key)

    res_img = abs(np.float32(watermarked) - np.float32(img))
    print(np.max(res_img), np.min(res_img))
    cv.imshow('orig', img)
    cv.imshow('watermarked', watermarked)
    cv.imshow('diff', np.uint8(res_img))
    cv.imshow('decod1', np.uint8(dec[0]))
    cv.imshow('decod2', np.uint8(dec[1]))
    cv.imshow('decod3', np.uint8(dec[2]))
    cv.imshow('decod4', np.uint8(dec[3]))
    cv.imshow('decod5', np.uint8(dec[4]))
    cv.imshow('decod6', np.uint8(dec[5]))

    cv.waitKey(0)
    cv.destroyAllWindows()
